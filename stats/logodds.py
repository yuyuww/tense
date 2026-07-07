# ------------------------------------------------------------------
# [FILE 2] stats/logodds.py
# pivots.py의 파일로부터 계산된 교차표(pivot)로부터 로그오즈비 계산
# ------------------------------------------------------------------
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Literal

import numpy as np
import pandas as pd
from scipy import stats


def log_odds_2x2(
    a: float, b: float, c: float, d: float,
    *,
    alpha: float = 0.5,
    add_alpha_if_zero: bool = True
) -> Tuple[float, float, float, float, float]:
    """
    Returns (odds_ratio, log_or, se, z, p_value)
    If any cell is 0 and add_alpha_if_zero=True, adds alpha to ALL four cells.
    """
    if add_alpha_if_zero and (a == 0 or b == 0 or c == 0 or d == 0):
        a, b, c, d = a + alpha, b + alpha, c + alpha, d + alpha

    odds_ratio = (a * d) / (b * c)
    log_or = math.log(odds_ratio)
    se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    z = log_or / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(odds_ratio), float(log_or), float(se), float(z), float(p)


def default_binary_parser(x: Any) -> bool:
    """Robust binary parsing. Customize if needed."""
    if pd.isna(x):
        return False
    if isinstance(x, (int, np.integer)):
        return x != 0
    if isinstance(x, (float, np.floating)):
        return x != 0.0
    s = str(x).strip().lower()
    return s in {"1", "true", "t", "y", "yes", "on"}


def logodds_context_by_outcome_from_tab(
    tab: pd.DataFrame,
    *,
    context_col_name: str = "context",#tab에서 context로 지정된 행의 이름을 작성해야 함. context는 예시임.
    outcome_col_name: str = "outcome_level",#tab의 컬럼 이름이 outcome_level이므로 이 이름을 기본값으로 설정함.
    alpha: float = 0.5,
    add_alpha_if_zero: bool = True,
    count_mode: str = "",
) -> pd.DataFrame:
    """
    tab (context × outcome_level) -> one-vs-rest log-odds for each (g, o)

    For each context g and outcome level o:
      a = tab[g,o]
      b = sum(tab[g,*]) - a
      c = sum(tab[*,o]) - a
      d = grand_total - a - b - c
    """
    context_values = list(tab.index)
    outcome_values = list(tab.columns)

    context_totals = tab.sum(axis=1)
    outcome_totals = tab.sum(axis=0)
    grand_total = float(tab.values.sum())

    rows: List[Dict[str, Any]] = []
    for g in context_values:
        g_total = float(context_totals.loc[g])
        for o in outcome_values:
            a = float(tab.loc[g, o])
            b = float(g_total - a)
            c = float(outcome_totals.loc[o] - a)
            d = float(grand_total - a - b - c)

            odds_ratio, log_or, se, z, p = log_odds_2x2(
                a, b, c, d,
                alpha=alpha, add_alpha_if_zero=add_alpha_if_zero
            )

            rows.append({
                context_col_name: g,
                outcome_col_name: o,

                "a_in_context_and_level": a,
                "b_in_context_not_level": b,
                "c_not_context_in_level": c,
                "d_not_context_not_level": d,

                "odds_ratio": odds_ratio,
                "log_odds": log_or,
                "SE": se,
                "z": z,
                "p_value": p,

                "context_total": g_total,
                "outcome_total": float(outcome_totals.loc[o]),
                "grand_total": grand_total,

                "count_mode": count_mode,
                "alpha_added": (alpha if (add_alpha_if_zero and (a == 0 or b == 0 or c == 0 or d == 0)) else 0.0),
            })

    return pd.DataFrame(rows)


def logodds_unit_by_context_from_pivot(
    pivot: pd.DataFrame,
    *,
    unit_col_name: str = "unit",
    context_col_name: str = "context",
    alpha: float = 0.5,
    add_alpha_if_zero: bool = True,
    count_mode: str = "",
) -> pd.DataFrame:
    """
    pivot index=(unit, context), columns={True, False} -> log-odds for each (unit u, context g)
    where compare context g vs other contexts within the same unit u.

    a = pos in g
    b = neg in g
    c = pos in other contexts (same unit)
    d = neg in other contexts (same unit)
    """
    if True not in pivot.columns or False not in pivot.columns:
        raise ValueError("pivot must have columns {True, False}.")

    unit_values = pivot.index.get_level_values(0).unique().tolist()
    context_values = pivot.index.get_level_values(1).unique().tolist()

    rows: List[Dict[str, Any]] = []
    for u in unit_values:
        sub = pivot.loc[u].reindex(context_values)

        u_pos_total = float(sub[True].sum())
        u_neg_total = float(sub[False].sum())

        for g in context_values:
            a = float(sub.loc[g, True])
            b = float(sub.loc[g, False])
            c = float(u_pos_total - a)
            d = float(u_neg_total - b)

            odds_ratio, log_or, se, z, p = log_odds_2x2(
                a, b, c, d,
                alpha=alpha, add_alpha_if_zero=add_alpha_if_zero
            )

            rows.append({
                unit_col_name: u,
                context_col_name: g,

                "a_pos_in_context": a,
                "b_neg_in_context": b,
                "c_pos_other_contexts": c,
                "d_neg_other_contexts": d,

                "odds_ratio": odds_ratio,
                "log_odds": log_or,
                "SE": se,
                "z": z,
                "p_value": p,

                "unit_pos_total": u_pos_total,
                "unit_neg_total": u_neg_total,
                "unit_total": u_pos_total + u_neg_total,

                "count_mode": count_mode,
                "alpha_added": (alpha if (add_alpha_if_zero and (a == 0 or b == 0 or c == 0 or d == 0)) else 0.0),
            })

    return pd.DataFrame(rows)
