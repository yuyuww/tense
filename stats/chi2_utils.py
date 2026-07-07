# ------------------------------------------------------------------
# [FILE 3] stats/chi2_utils.py
# pivots.py의 파일로부터 계산된 교차표(pivot)로부터 카이제곱/크래머V 계산
# ------------------------------------------------------------------
'''
- chi2_overall_from_tab:
  전체 RxC 분할표에 대한 전역 카이제곱 + Cramér's V (요약용)

- chi2_by_unit_from_pivot:
  unit별 2xK(긍/부정 x 맥락) 카이제곱 + Cramér's V (랭킹/비교용)

- _chi2_and_cramers_v:
  실제 통계 계산 엔진 (공유)
'''

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def _drop_empty_rows_cols(table: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop empty rows/cols from RxC."""
    if table.size == 0:
        return table, np.array([], dtype=bool), np.array([], dtype=bool)
    row_sum = table.sum(axis=1)
    col_sum = table.sum(axis=0)
    keep_r = row_sum > 0
    keep_c = col_sum > 0
    table_valid = table[np.ix_(keep_r, keep_c)]
    return table_valid, keep_r, keep_c


def _drop_empty_cols(table: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Drop empty columns from a table (for 2xK)."""
    if table.size == 0:
        return table, np.array([], dtype=bool)
    col_sum = table.sum(axis=0)
    keep_c = col_sum > 0
    return table[:, keep_c], keep_c

#  실제 통계 계산 엔진 (공유)
def _chi2_and_cramers_v(table_valid: np.ndarray) -> Tuple[float, float, int, float]:
    """
    chi2 + Cramér's V for general RxC:
      V = sqrt(chi2 / (n * min(r-1, c-1)))
    (2xK에서는 자동으로 sqrt(chi2/n)로 단순화됨)
    """
    chi2, p, dfree, _ = stats.chi2_contingency(table_valid)
    n = float(table_valid.sum())
    r, c = table_valid.shape
    denom = n * min(r - 1, c - 1)
    v = math.sqrt(float(chi2) / denom) if denom > 0 else np.nan
    return float(chi2), float(p), int(dfree), float(v)

#  전체 RxC 분할표에 대한 전역 카이제곱 + Cramér's V (요약용)
def chi2_overall_from_tab(
    tab: pd.DataFrame,
    *,
    count_mode: str = "",
    drop_empty: bool = True,
) -> pd.DataFrame:
    """
    Overall independence test on an RxC contingency table (1-row DF).
    """
    table = tab.values.astype(float)
    if drop_empty:
        table_valid, _, _ = _drop_empty_rows_cols(table)
    else:
        table_valid = table

    r = int(table_valid.shape[0])
    c = int(table_valid.shape[1])
    n_total = float(table_valid.sum()) if table_valid.size else 0.0

    if r < 2 or c < 2 or n_total <= 0:
        return pd.DataFrame([{
            "n_context_used": r,
            "n_outcome_used": c,
            "N_total": n_total,
            "chi2": np.nan,
            "df": np.nan,
            "p_value": np.nan,
            "cramers_v": np.nan,
            "count_mode": count_mode,
            "note": "skip: table too small or empty",
        }])

    try:
        chi2, p, dfree, v = _chi2_and_cramers_v(table_valid)
        return pd.DataFrame([{
            "n_context_used": r,
            "n_outcome_used": c,
            "N_total": n_total,
            "chi2": chi2,
            "df": dfree,
            "p_value": p,
            "cramers_v": v,
            "count_mode": count_mode,
            "note": "",
        }])
    except ValueError as e:
        return pd.DataFrame([{
            "n_context_used": r,
            "n_outcome_used": c,
            "N_total": n_total,
            "chi2": np.nan,
            "df": np.nan,
            "p_value": np.nan,
            "cramers_v": np.nan,
            "count_mode": count_mode,
            "note": f"skip: {str(e)}",
        }])


def chi2_by_unit_from_pivot(
    pivot: pd.DataFrame,
    *,
    unit_col_name: str = "unit",
    pos_key: Any = True,
    neg_key: Any = False,
    count_mode: str = "",
    drop_empty_contexts: bool = True,
) -> pd.DataFrame:
    """
    For each unit u: build 2xK table (pos/neg × contexts) and run chi-square.

    Returns a DF with many rows (one per unit), sorted by p then V.
    """
    if pos_key not in pivot.columns or neg_key not in pivot.columns:
        raise ValueError(f"pivot must have columns {pos_key} and {neg_key}.")

    unit_values = pivot.index.get_level_values(0).unique().tolist()
    context_values = pivot.index.get_level_values(1).unique().tolist()

    tests: List[Dict[str, Any]] = []
    for u in unit_values:
        sub = pivot.loc[u].reindex(context_values)
        pos = sub[pos_key].fillna(0).astype(float).values
        neg = sub[neg_key].fillna(0).astype(float).values
        table = np.vstack([pos, neg])  # 2 x K

        if drop_empty_contexts:
            table_valid, _ = _drop_empty_cols(table)
        else:
            table_valid = table

        n_context_used = int(table_valid.shape[1])
        pos_total = float(table_valid[0].sum()) if table_valid.size else 0.0
        neg_total = float(table_valid[1].sum()) if table_valid.size else 0.0
        unit_total = pos_total + neg_total

        # Preconditions: contexts>=2, total>0, both rows>0
        if n_context_used < 2 or unit_total == 0 or pos_total == 0 or neg_total == 0:
            tests.append({
                unit_col_name: u,
                "n_context_used": n_context_used,
                "unit_total": unit_total,
                "pos_total": pos_total,
                "neg_total": neg_total,
                "pos_rate": (pos_total / unit_total) if unit_total > 0 else np.nan,
                "chi2": np.nan,
                "df": np.nan,
                "p_value": np.nan,
                "cramers_v": np.nan,
                "count_mode": count_mode,
                "note": "skip: n_context_used<2 or unit_total==0 or pos_total==0 or neg_total==0",
            })
            continue

        try:
            chi2, p, dfree, v = _chi2_and_cramers_v(table_valid)
            tests.append({
                unit_col_name: u,
                "n_context_used": n_context_used,
                "unit_total": unit_total,
                "pos_total": pos_total,
                "neg_total": neg_total,
                "pos_rate": (pos_total / unit_total) if unit_total > 0 else np.nan,
                "chi2": chi2,
                "df": dfree,
                "p_value": p,
                "cramers_v": v,
                "count_mode": count_mode,
                "note": "",
            })
        except ValueError as e:
            tests.append({
                unit_col_name: u,
                "n_context_used": n_context_used,
                "unit_total": unit_total,
                "pos_total": pos_total,
                "neg_total": neg_total,
                "pos_rate": (pos_total / unit_total) if unit_total > 0 else np.nan,
                "chi2": np.nan,
                "df": np.nan,
                "p_value": np.nan,
                "cramers_v": np.nan,
                "count_mode": count_mode,
                "note": f"skip: {str(e)}",
            })

    chi2_df = pd.DataFrame(tests)
    if not chi2_df.empty and "p_value" in chi2_df.columns:
        chi2_df = chi2_df.sort_values(
            by=["p_value", "cramers_v", unit_col_name],
            na_position="last",
            ascending=[True, False, True],
        )
    return chi2_df

