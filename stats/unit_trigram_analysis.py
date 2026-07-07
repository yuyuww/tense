"""Unit trigram analysis with smoothed odds-ratio columns.

Usage example:

    from stats.unit_trigram_analysis import (
        analyze_trigram_baseline,
        analyze_unit_sequence_effect,
        analyze_unit_trigram_against_baseline,
    )

    baseline = analyze_trigram_baseline(df, baseline_col="문서범주")
    result = analyze_unit_trigram_against_baseline(
        df,
        baseline_col="문서범주",
        unit_col="V_form",
        baseline_df=baseline,
        smoothing=0.5,
    )

The original ratio and delta columns are kept. Each probability-ratio column
gets adjacent odds-ratio and log-odds-ratio columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stats.analyze_unit_trigram_ver1 import (
    analyze_sequence_effect_baseline,
    analyze_trigram_baseline,
)
from stats.analyze_unit_trigram_ver1 import (
    analyze_unit_sequence_effect as _analyze_unit_sequence_effect_base,
)
from stats.analyze_unit_trigram_ver1 import (
    analyze_unit_trigram_against_baseline as _analyze_unit_trigram_against_baseline_base,
)


def smoothed_odds(p, smoothing: float = 0.5):
    """Compute odds for probability values without infinities at 0 or 1."""
    if smoothing < 0:
        raise ValueError("smoothing must be >= 0")
    return (p + smoothing) / ((1 - p) + smoothing)


def smoothed_odds_ratio(obs, exp, smoothing: float = 0.5):
    """Compute smoothed odds ratio for observed and expected probabilities."""
    return smoothed_odds(obs, smoothing) / smoothed_odds(exp, smoothing)


def log_smoothed_odds_ratio(obs, exp, smoothing: float = 0.5):
    """Compute log smoothed odds ratio."""
    return np.log(smoothed_odds_ratio(obs, exp, smoothing))


def _insert_or_assign_after(df: pd.DataFrame, name: str, values, after: str) -> None:
    if name in df.columns:
        df[name] = values
        return
    loc = df.columns.get_loc(after) + 1 if after in df.columns else len(df.columns)
    df.insert(loc, name, values)


def add_probability_effect_columns(
    df: pd.DataFrame,
    *,
    obs_col: str,
    exp_col: str,
    ratio_col: str,
    smoothing: float = 0.5,
) -> None:
    """Add odds-ratio and log-odds-ratio columns next to a ratio column."""
    if obs_col not in df.columns or exp_col not in df.columns:
        return
    odds_col = ratio_col.replace("_ratio_", "_odds_ratio_")
    log_odds_col = ratio_col.replace("_ratio_", "_log_odds_ratio_")
    odds = smoothed_odds_ratio(df[obs_col], df[exp_col], smoothing)
    _insert_or_assign_after(df, odds_col, odds, ratio_col)
    _insert_or_assign_after(df, log_odds_col, np.log(odds), odds_col)


def add_all_probability_effect_columns(
    df: pd.DataFrame,
    *,
    suffix: str,
    smoothing: float = 0.5,
) -> pd.DataFrame:
    """Add odds-ratio/log-odds-ratio columns for known analysis outputs."""
    out = df.copy()
    comparisons = [
        ("unit_P_T", "E_unit_P_T", f"unit_P_T_ratio_{suffix}"),
        ("unit_P_F", "E_unit_P_F", f"unit_P_F_ratio_{suffix}"),
        ("stay_Obs_rate", "stay_E_rate", f"stay_ratio_{suffix}"),
        ("switch_Obs_rate", "switch_E_rate", f"switch_ratio_{suffix}"),
        ("switch_return_Obs_rate", "switch_return_E_rate", f"switch_return_ratio_{suffix}"),
        ("switch_extension_Obs_rate", "switch_extension_E_rate", f"switch_extension_ratio_{suffix}"),
        ("stay_stay_Obs_rate", "stay_stay_E_rate", f"stay_stay_ratio_{suffix}"),
        ("stay_switch_Obs_rate", "stay_switch_E_rate", f"stay_switch_ratio_{suffix}"),
        ("return_after_TF_Obs_rate", "return_after_TF_E_rate", f"return_after_TF_ratio_{suffix}"),
        ("return_after_FT_Obs_rate", "return_after_FT_E_rate", f"return_after_FT_ratio_{suffix}"),
        ("extension_after_TF_Obs_rate", "extension_after_TF_E_rate", f"extension_after_TF_ratio_{suffix}"),
        ("extension_after_FT_Obs_rate", "extension_after_FT_E_rate", f"extension_after_FT_ratio_{suffix}"),
    ]

    for col in ["TT", "TF", "FT", "FF"]:
        comparisons.append((f"Obs_{col}_rate", f"E_{col}_rate", f"{col}_ratio_{suffix}"))

    for col in ["TTT", "TTF", "TFT", "TFF", "FTT", "FTF", "FFT", "FFF"]:
        comparisons.append((f"Obs_{col}_rate", f"E_{col}_rate", f"{col}_ratio_{suffix}"))

    for pair in ["TT", "TF", "FT", "FF"]:
        comparisons.extend(
            [
                (
                    f"P_next_T_given_{pair}",
                    f"baseline_P_next_T_given_{pair}",
                    f"next_T_after_{pair}_ratio_{suffix}",
                ),
                (
                    f"P_next_F_given_{pair}",
                    f"baseline_P_next_F_given_{pair}",
                    f"next_F_after_{pair}_ratio_{suffix}",
                ),
            ]
        )

    for obs_col, exp_col, ratio_col in comparisons:
        add_probability_effect_columns(
            out,
            obs_col=obs_col,
            exp_col=exp_col,
            ratio_col=ratio_col,
            smoothing=smoothing,
        )
    return out


def analyze_unit_sequence_effect(
    df: pd.DataFrame,
    *,
    baseline_col: str,
    unit_col: str,
    baseline_df=None,
    state_col: str = "sentence_f_EP_T",
    prev_state_col: str = "prev_sentence_f_EP_T",
    next_state_col: str = "next_sentence_f_EP_T",
    has_prev_col: str = "has_prev_sentence",
    has_next_col: str = "has_next_sentence",
    count_col: str = "count",
    min_unit_n: int = 0,
    min_bigram_n: int = 0,
    min_trigram_n: int = 0,
    smoothing: float = 0.5,
) -> pd.DataFrame:
    """Run unit sequence analysis and add smoothed odds/log-odds columns."""
    result = _analyze_unit_sequence_effect_base(
        df,
        baseline_col=baseline_col,
        unit_col=unit_col,
        baseline_df=baseline_df,
        state_col=state_col,
        prev_state_col=prev_state_col,
        next_state_col=next_state_col,
        has_prev_col=has_prev_col,
        has_next_col=has_next_col,
        count_col=count_col,
        min_unit_n=min_unit_n,
        min_bigram_n=min_bigram_n,
        min_trigram_n=min_trigram_n,
    )
    return add_all_probability_effect_columns(
        result,
        suffix="vs_expected",
        smoothing=smoothing,
    )


def analyze_unit_trigram_against_baseline(
    df: pd.DataFrame,
    *,
    baseline_col: str,
    unit_col: str,
    baseline_df: pd.DataFrame | None = None,
    state_col: str = "sentence_f_EP_T",
    prev_state_col: str = "prev_sentence_f_EP_T",
    next_state_col: str = "next_sentence_f_EP_T",
    has_prev_col: str = "has_prev_sentence",
    has_next_col: str = "has_next_sentence",
    count_col: str = "count",
    min_unit_n: int = 0,
    min_bigram_n: int = 0,
    min_trigram_n: int = 0,
    smoothing: float = 0.5,
) -> pd.DataFrame:
    """Run baseline comparison and add smoothed odds/log-odds columns."""
    result = _analyze_unit_trigram_against_baseline_base(
        df,
        baseline_col=baseline_col,
        unit_col=unit_col,
        baseline_df=baseline_df,
        state_col=state_col,
        prev_state_col=prev_state_col,
        next_state_col=next_state_col,
        has_prev_col=has_prev_col,
        has_next_col=has_next_col,
        count_col=count_col,
        min_unit_n=min_unit_n,
        min_bigram_n=min_bigram_n,
        min_trigram_n=min_trigram_n,
    )
    return add_all_probability_effect_columns(
        result,
        suffix="vs_baseline",
        smoothing=smoothing,
    )
