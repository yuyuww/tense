# ------------------------------------------------------------------
# [FILE 1] stats/pivots.py
# long df -> (필터/조건 반영) 교차표(tab/pivot) 생성
# ------------------------------------------------------------------
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union, Literal

import numpy as np
import pandas as pd

from stats.filtering import apply_filters, FilterValue, _topn_values


CountMode = Literal["auto", "rows", "weight"]


def make_context_outcome_tab(
    df: pd.DataFrame,
    *,
    context_col: str,
    outcome_col: str,
    outcome_transform: Optional[Callable[[Any], Any]] = None,
    # Counting mode
    count_mode: CountMode = "auto",
    weight_col: str = "ID",
    # Filters
    filters: Optional[Dict[str, Any]] = None,
    # Limit contexts
    context_top_n: Optional[int] = None,
    context_min_count: Optional[float] = None,
    # Limit outcome levels
    outcome_top_n: Optional[int] = None,
    outcome_min_count: Optional[float] = None,
    dropna: bool = True,
) -> Tuple[pd.DataFrame, List[Any], List[Any], str]:
    """
    long df -> contingency table (RxC): context × outcome_level

    Returns
    -------
    tab : pd.DataFrame
        index=context, columns=outcome_level, values=counts
    context_values : list
    outcome_values : list
    resolved_count_mode : str ("rows" or "weight")
    """
    if count_mode == "auto":
        resolved = "weight" if weight_col else "rows"
    else:
        resolved = count_mode

    needed = {context_col, outcome_col}
    if resolved == "weight":
        needed.add(weight_col)

    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    f = df.copy()
    if filters:
        f = apply_filters(f, filters=filters)

    # normalize outcome if needed
    out_col = "_OUTCOME_VAL"
    if outcome_transform is not None:
        f[out_col] = f[outcome_col].map(outcome_transform)
    else:
        f[out_col] = f[outcome_col]

    # choose contexts/outcomes by frequency consistent with count_mode
    context_values = _topn_values(
        f, context_col,
        mode=resolved, weight_col=weight_col,
        top_n=context_top_n, min_count=context_min_count,
        dropna=dropna
    )

    outcome_values = _topn_values(
        f, out_col,
        mode=resolved, weight_col=weight_col,
        top_n=outcome_top_n, min_count=outcome_min_count,
        dropna=dropna
    )

    f = f[f[context_col].isin(context_values) & f[out_col].isin(outcome_values)].copy()

    # build contingency table: context × outcome
    if resolved == "weight":
        tab = (
            f.groupby([context_col, out_col], dropna=False)[weight_col]
             .sum()
             .unstack(out_col, fill_value=0.0)
        )
    else:
        tab = (
            f.groupby([context_col, out_col], dropna=False)
             .size()
             .unstack(out_col, fill_value=0.0)
        )

    # ensure all selected contexts/outcomes exist (missing combos -> 0)
    tab = tab.reindex(index=context_values, columns=outcome_values, fill_value=0.0)

    return tab, context_values, outcome_values, resolved


def make_unit_context_binary_pivot(
    df: pd.DataFrame,
    *,
    unit_col: str,
    context_col: str,
    outcome_col: str,
    positive_fn: Optional[Callable[[Any], bool]] = None,
    # Counting mode
    count_mode: CountMode = "auto",
    weight_col: str = "ID",
    # Filters
    filters: Optional[Dict[str, Any]] = None,
    # Limit units/contexts
    unit_top_n: Optional[int] = None,
    unit_min_count: Optional[float] = None,
    context_top_n: Optional[int] = None,
    context_min_count: Optional[float] = None,
    dropna: bool = True,
) -> Tuple[pd.DataFrame, List[Any], List[Any], str]:
    """
    long df -> pivot:
      index=(unit, context), columns={True, False}, values=counts

    Returns
    -------
    pivot : pd.DataFrame
    unit_values : list
    context_values : list
    resolved_count_mode : str ("rows" or "weight")
    """
    if count_mode == "auto":
        resolved = "weight" if weight_col else "rows"
    else:
        resolved = count_mode

    needed = {unit_col, context_col, outcome_col}
    if resolved == "weight":
        needed.add(weight_col)

    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    f = df.copy()
    if filters:
        f = apply_filters(f, filters=filters)

    # select units/contexts using the SAME counting mode
    unit_values = _topn_values(
        f, unit_col,
        mode=resolved, weight_col=weight_col,
        top_n=unit_top_n, min_count=unit_min_count,
        dropna=dropna
    )
    context_values = _topn_values(
        f, context_col,
        mode=resolved, weight_col=weight_col,
        top_n=context_top_n, min_count=context_min_count,
        dropna=dropna
    )

    f = f[f[unit_col].isin(unit_values) & f[context_col].isin(context_values)].copy()

    # binary outcome
    bin_col = "_OUTCOME_ON"
    if positive_fn is None:
        if f[outcome_col].dropna().isin([True, False]).all():
            f[bin_col] = f[outcome_col]
        else:
            raise ValueError("positive_fn is required when outcome is not boolean.")
    else:
        f[bin_col] = f[outcome_col].apply(positive_fn)
        
    # group -> unstack
    if resolved == "weight":
        pivot = (
            f.groupby([unit_col, context_col, bin_col], dropna=False)[weight_col]
             .sum()
             .unstack(bin_col, fill_value=0.0)
        )
    else:
        pivot = (
            f.groupby([unit_col, context_col, bin_col], dropna=False)
             .size()
             .unstack(bin_col, fill_value=0.0)
        )

    # ensure both columns exist
    if True not in pivot.columns:
        pivot[True] = 0.0
    if False not in pivot.columns:
        pivot[False] = 0.0

    # ensure full unit×context grid exists
    idx = pd.MultiIndex.from_product([unit_values, context_values], names=[unit_col, context_col])
    pivot = pivot.reindex(idx, fill_value=0.0)

    return pivot, unit_values, context_values, resolved
