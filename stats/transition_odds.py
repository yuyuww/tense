"""Smoothed odds-ratio helpers for transition-analysis result dataframes."""

from __future__ import annotations

import numpy as np
import pandas as pd


def smoothed_odds(p, smoothing: float = 0.5):
    if smoothing < 0:
        raise ValueError("smoothing must be >= 0")
    return (p + smoothing) / ((1 - p) + smoothing)


def smoothed_odds_ratio(obs, exp, smoothing: float = 0.5):
    return smoothed_odds(obs, smoothing) / smoothed_odds(exp, smoothing)


def _first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    return next((col for col in candidates if col in df.columns), None)


def _effect_column_names(ratio_col: str) -> tuple[str, str]:
    if "_ratio_" in ratio_col:
        odds_col = ratio_col.replace("_ratio_", "_odds_ratio_")
        log_col = ratio_col.replace("_ratio_", "_log_odds_ratio_")
    elif ratio_col.endswith("_ratio"):
        odds_col = ratio_col[: -len("_ratio")] + "_odds_ratio"
        log_col = ratio_col[: -len("_ratio")] + "_log_odds_ratio"
    else:
        odds_col = ratio_col + "_odds_ratio"
        log_col = ratio_col + "_log_odds_ratio"
    return odds_col, log_col


def _infer_probability_columns(df: pd.DataFrame, ratio_col: str) -> tuple[str, str] | None:
    if ratio_col.endswith("_base_ratio"):
        stem = ratio_col[: -len("_base_ratio")]
        obs_col = _first_existing(df, [f"Obs_{stem}_pair_rate", f"Obs_{stem}_rate", f"{stem}_Obs_rate"])
        exp_col = _first_existing(df, [f"E_{stem}_base_pair_rate", f"E_{stem}_base_rate", f"{stem}_E_base_rate"])
        return (obs_col, exp_col) if obs_col and exp_col else None

    if ratio_col.endswith("_markov_ratio"):
        stem = ratio_col[: -len("_markov_ratio")]
        obs_col = _first_existing(df, [f"Obs_{stem}_rate", f"{stem}_Obs_rate"])
        exp_col = _first_existing(df, [f"E_{stem}_markov_rate", f"{stem}_E_markov_rate"])
        return (obs_col, exp_col) if obs_col and exp_col else None

    suffix_specs = [
        ("_ratio_vs_base", [("{stem}_Obs_rate",), ("Obs_{stem}_rate",)], [("{stem}_E_base_rate",), ("E_{stem}_base_rate",)]),
        ("_ratio_vs_markov", [("{stem}_Obs_rate",), ("Obs_{stem}_rate",)], [("{stem}_E_markov_rate",), ("E_{stem}_markov_rate",)]),
        ("_ratio_vs_expected", [("{stem}_Obs_rate",), ("Obs_{stem}_rate",), ("{stem}",)], [("{stem}_E_rate",), ("E_{stem}_rate",)]),
        ("_ratio_vs_baseline", [("{stem}_Obs_rate",), ("Obs_{stem}_rate",), ("{stem}",)], [("{stem}_E_rate",), ("E_{stem}_rate",), ("E_{stem}",)]),
    ]
    for suffix, obs_templates, exp_templates in suffix_specs:
        if ratio_col.endswith(suffix):
            stem = ratio_col[: -len(suffix)]
            obs_candidates = [tpl[0].format(stem=stem) for tpl in obs_templates]
            exp_candidates = [tpl[0].format(stem=stem) for tpl in exp_templates]
            obs_col = _first_existing(df, obs_candidates)
            exp_col = _first_existing(df, exp_candidates)
            return (obs_col, exp_col) if obs_col and exp_col else None

    return None


def _insert_or_assign_after(df: pd.DataFrame, name: str, values, after: str) -> None:
    if name in df.columns:
        df[name] = values
        return
    loc = df.columns.get_loc(after) + 1 if after in df.columns else len(df.columns)
    df.insert(loc, name, values)


def add_transition_odds_columns(
    df: pd.DataFrame,
    *,
    smoothing: float = 0.5,
    ratio_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Add odds-ratio/log-odds-ratio columns without changing existing results."""
    out = df.copy()
    if ratio_cols is None:
        ratio_cols = [
            col
            for col in out.columns
            if "ratio" in col and "odds_ratio" not in col and "log_odds_ratio" not in col
        ]

    for ratio_col in ratio_cols:
        inferred = _infer_probability_columns(out, ratio_col)
        if inferred is None:
            continue
        obs_col, exp_col = inferred
        odds_col, log_col = _effect_column_names(ratio_col)
        odds = smoothed_odds_ratio(out[obs_col], out[exp_col], smoothing)
        _insert_or_assign_after(out, odds_col, odds, ratio_col)
        _insert_or_assign_after(out, log_col, np.log(odds), odds_col)
    return out
