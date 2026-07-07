"""Transition-chain analysis functions extracted from notebook 4.5.1.

The original probability, ratio, and delta formulas are intentionally kept as-is.
The smoothing argument is used only when adding odds-ratio/log-odds-ratio columns.
"""

from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd

from stats.transition_odds import add_transition_odds_columns, smoothed_odds_ratio


# 다양한 형태의 T/F 값을 bool 또는 None으로 정규화
from typing import Any, Optional, Tuple

def _normalize_binary_state(x: Any) -> Optional[bool]:
    if pd.isna(x):
        return None

    if isinstance(x, (bool, np.bool_)):
        return bool(x)

    if isinstance(x, (int, np.integer, float, np.floating)):
        if x == 1:
            return True
        if x == 0:
            return False

    s = str(x).strip().upper()
    true_set = {"1", "T", "TRUE", "Y", "YES"}
    false_set = {"0", "F", "FALSE", "N", "NO"}

    if s in true_set:
        return True
    if s in false_set:
        return False

    raise ValueError(f"Could not normalize binary state: {x!r}")


# 3연쇄 분석함수: 기본 출현률 대비 2연쇄 &, 3연쇄 기본 기대값 & 3연쇄 1차 Markov기대값

#-----------------------
#-----------------------
# Main function3: analyze_trigram_from_weighted_df
#
# 주요 수식
#
# 1) 기본 T/F 출현률
#    P_T = n_T / n_sentences
#    P_F = n_F / n_sentences
#
# 2) 2연쇄 기본 기대값
#    E(TT) = count(prev=T) * P_T
#    E(TF) = count(prev=T) * P_F
#    E(FT) = count(prev=F) * P_T
#    E(FF) = count(prev=F) * P_F
#
#    pair_delta = 실제 2연쇄 출현률 - 기본 기대 출현률
#    pair_ratio = 실제 2연쇄 빈도 / 기본 기대 빈도
#
# 3) 3연쇄 기본 기대값
#    E(ABC)_base = count(AB) * P(C)
#
# 4) 3연쇄 1차 Markov 기대값
#    E(ABC)_markov = count(AB) * P(C | B)
#
# 5) 묶음 비교
#    유지 후 유지: TTT + FFF
#    유지 후 전환: TTF + FFT
#    전환 후 유지/연장: TFF + FTT
#    전환 후 전환/복귀: TFT + FTF
#-----------------------

def analyze_trigram_from_weighted_df(
    df,
    *,
    unit_col="docu_id",
    state_col="sentence_f_EP_T",
    prev_state_col="prev_sentence_f_EP_T",
    next_state_col="next_sentence_f_EP_T",
    has_prev_col="has_prev_sentence",
    has_next_col="has_next_sentence",
    count_col="count",
    smoothing: float = 0.5,
):
    work = df.copy()

    work["_prev"] = work[prev_state_col].map(_normalize_binary_state)
    work["_curr"] = work[state_col].map(_normalize_binary_state)
    work["_next"] = work[next_state_col].map(_normalize_binary_state)
    work["_has_prev"] = work[has_prev_col].map(_normalize_binary_state)
    work["_has_next"] = work[has_next_col].map(_normalize_binary_state)

    def safe_div(num, den):
        return num / den.replace(0, np.nan)

    # -------------------------------------------------
    # 1. 전체 현재문장 기준 T/F 비율
    # -------------------------------------------------
    all_units = pd.Index(sorted(work[unit_col].dropna().unique()), name=unit_col)
    out = pd.DataFrame(index=all_units)

    curr_counts = (
        work.loc[
            work[unit_col].notna()
            & work["_curr"].notna()
        ]
        .groupby([unit_col, "_curr"], observed=True)[count_col]
        .sum()
        .reset_index()
    )

    curr_pivot = curr_counts.pivot_table(
        index=unit_col,
        columns="_curr",
        values=count_col,
        fill_value=0,
    )

    out["n_T"] = (
        curr_pivot[True].reindex(out.index, fill_value=0).astype(float)
        if True in curr_pivot.columns else 0.0
    )
    out["n_F"] = (
        curr_pivot[False].reindex(out.index, fill_value=0).astype(float)
        if False in curr_pivot.columns else 0.0
    )

    out["n_sentences"] = out["n_T"] + out["n_F"]
    out["P_T"] = safe_div(out["n_T"], out["n_sentences"])
    out["P_F"] = safe_div(out["n_F"], out["n_sentences"])

    # -------------------------------------------------
    # 2. 2연쇄 prev → curr 계산
    # -------------------------------------------------
    pair_work = work.loc[
        (work["_has_prev"] == True)
        & work["_prev"].notna()
        & work["_curr"].notna()
        & work[unit_col].notna()
    ].copy()

    pair_g = (
        pair_work
        .groupby([unit_col, "_prev", "_curr"], observed=True)[count_col]
        .sum()
        .reset_index()
    )

    pair_pivot = pair_g.pivot_table(
        index=unit_col,
        columns=["_prev", "_curr"],
        values=count_col,
        fill_value=0,
    )

    def zero_series():
        return pd.Series(0.0, index=out.index)

    def get_pair(prev, curr):
        key = (prev, curr)
        if key in pair_pivot.columns:
            return pair_pivot[key].reindex(out.index, fill_value=0).astype(float)
        return zero_series()

    out["TT"] = get_pair(True, True)
    out["TF"] = get_pair(True, False)
    out["FT"] = get_pair(False, True)
    out["FF"] = get_pair(False, False)

    pair_cols = ["TT", "TF", "FT", "FF"]

    out["n_pairs"] = out[pair_cols].sum(axis=1)

    out["n_prev_T_pair"] = out["TT"] + out["TF"]
    out["n_prev_F_pair"] = out["FT"] + out["FF"]

    out["P_T_given_T_pair"] = safe_div(out["TT"], out["TT"] + out["TF"])
    out["P_F_given_T_pair"] = safe_div(out["TF"], out["TT"] + out["TF"])
    out["P_T_given_F_pair"] = safe_div(out["FT"], out["FT"] + out["FF"])
    out["P_F_given_F_pair"] = safe_div(out["FF"], out["FT"] + out["FF"])

    pair_cond_baselines = {
        "P_T_given_T_pair": "P_T",
        "P_F_given_T_pair": "P_F",
        "P_T_given_F_pair": "P_T",
        "P_F_given_F_pair": "P_F",
    }
    pair_cond_effect_cols = []
    for prob_col, baseline_col in pair_cond_baselines.items():
        delta_col = f"{prob_col}_delta_vs_{baseline_col}"
        ratio_col = f"{prob_col}_ratio_vs_{baseline_col}"
        odds_col = f"{prob_col}_odds_ratio_vs_{baseline_col}"
        log_odds_col = f"{prob_col}_log_odds_ratio_vs_{baseline_col}"

        out[delta_col] = out[prob_col] - out[baseline_col]
        out[ratio_col] = out[prob_col] / out[baseline_col].replace(0, np.nan)
        out[odds_col] = smoothed_odds_ratio(
            out[prob_col],
            out[baseline_col],
            smoothing=smoothing,
        )
        out[log_odds_col] = np.log(out[odds_col])
        pair_cond_effect_cols.extend([delta_col, ratio_col, odds_col, log_odds_col])

    # -------------------------------------------------
    # 2-1. 2연쇄 기본 T/F 출현률 기대값
    # 공식: E(AB) = count(A) * P(B)
    # -------------------------------------------------
    out["E_TT_base"] = out["n_prev_T_pair"] * out["P_T"]
    out["E_TF_base"] = out["n_prev_T_pair"] * out["P_F"]
    out["E_FT_base"] = out["n_prev_F_pair"] * out["P_T"]
    out["E_FF_base"] = out["n_prev_F_pair"] * out["P_F"]

    for col in pair_cols:
        out[f"Obs_{col}_pair_rate"] = safe_div(out[col], out["n_pairs"])
        out[f"E_{col}_base_pair_rate"] = safe_div(out[f"E_{col}_base"], out["n_pairs"])

        out[f"{col}_base_delta"] = (
            out[f"Obs_{col}_pair_rate"] - out[f"E_{col}_base_pair_rate"]
        )

        out[f"{col}_base_ratio"] = (
            out[col] / out[f"E_{col}_base"].replace(0, np.nan)
        )

    # 2연쇄 유지/전환 묶음
    pair_sets = {
        "pair_stay": ["TT", "FF"],
        "pair_switch": ["TF", "FT"],
    }

    for name, cols in pair_sets.items():
        out[f"{name}_Obs_rate"] = out[[f"Obs_{c}_pair_rate" for c in cols]].sum(axis=1)
        out[f"{name}_E_base_rate"] = out[[f"E_{c}_base_pair_rate" for c in cols]].sum(axis=1)
        out[f"{name}_delta_vs_base"] = out[f"{name}_Obs_rate"] - out[f"{name}_E_base_rate"]
        out[f"{name}_ratio_vs_base"] = (
            out[f"{name}_Obs_rate"] / out[f"{name}_E_base_rate"].replace(0, np.nan)
        )

    # -------------------------------------------------
    # 3. 3연쇄 prev → curr → next 계산
    # -------------------------------------------------
    tri_work = work.loc[
        (work["_has_prev"] == True)
        & (work["_has_next"] == True)
        & work["_prev"].notna()
        & work["_curr"].notna()
        & work["_next"].notna()
        & work[unit_col].notna()
    ].copy()

    tri_g = (
        tri_work
        .groupby([unit_col, "_prev", "_curr", "_next"], observed=True)[count_col]
        .sum()
        .reset_index()
    )

    tri_pivot = tri_g.pivot_table(
        index=unit_col,
        columns=["_prev", "_curr", "_next"],
        values=count_col,
        fill_value=0,
    )

    def get_tri(prev, curr, next_):
        key = (prev, curr, next_)
        if key in tri_pivot.columns:
            return tri_pivot[key].reindex(out.index, fill_value=0).astype(float)
        return zero_series()

    out["TTT"] = get_tri(True, True, True)
    out["TTF"] = get_tri(True, True, False)
    out["TFT"] = get_tri(True, False, True)
    out["TFF"] = get_tri(True, False, False)
    out["FTT"] = get_tri(False, True, True)
    out["FTF"] = get_tri(False, True, False)
    out["FFT"] = get_tri(False, False, True)
    out["FFF"] = get_tri(False, False, False)

    trigram_cols = ["TTT", "TTF", "TFT", "TFF", "FTT", "FTF", "FFT", "FFF"]
    out["n_trigrams"] = out[trigram_cols].sum(axis=1)

    # -------------------------------------------------
    # 4. 실제 3연쇄 조건부 확률
    # -------------------------------------------------
    for col in pair_cols:
        # 예: TT 뒤에 T/F가 올 확률
        out[f"n_prefix_{col}_tri"] = out[f"{col}T"] + out[f"{col}F"]

        out[f"P_T_given_{col}"] = safe_div(
            out[f"{col}T"],
            out[f"{col}T"] + out[f"{col}F"]
        )
        out[f"P_F_given_{col}"] = safe_div(
            out[f"{col}F"],
            out[f"{col}T"] + out[f"{col}F"]
        )

    # -------------------------------------------------
    # 5. 3연쇄 기본 T/F 출현률 기대값
    # 공식: E(ABC)_base = count(AB) * P(C)
    # -------------------------------------------------
    for prefix in pair_cols:
        out[f"E_{prefix}T_base"] = out[f"n_prefix_{prefix}_tri"] * out["P_T"]
        out[f"E_{prefix}F_base"] = out[f"n_prefix_{prefix}_tri"] * out["P_F"]

    for col in trigram_cols:
        out[f"Obs_{col}_rate"] = safe_div(out[col], out["n_trigrams"])

        out[f"E_{col}_base_rate"] = safe_div(
            out[f"E_{col}_base"],
            out["n_trigrams"]
        )

        out[f"{col}_base_delta"] = (
            out[f"Obs_{col}_rate"] - out[f"E_{col}_base_rate"]
        )

        out[f"{col}_base_ratio"] = (
            out[col] / out[f"E_{col}_base"].replace(0, np.nan)
        )

    # -------------------------------------------------
    # 6. 3연쇄 1차 Markov 기대값
    # 공식: E(ABC)_markov = count(AB) * P(C | B)
    # -------------------------------------------------
    out["E_TTT_markov"] = out["n_prefix_TT_tri"] * out["P_T_given_T_pair"]
    out["E_TTF_markov"] = out["n_prefix_TT_tri"] * out["P_F_given_T_pair"]

    out["E_TFT_markov"] = out["n_prefix_TF_tri"] * out["P_T_given_F_pair"]
    out["E_TFF_markov"] = out["n_prefix_TF_tri"] * out["P_F_given_F_pair"]

    out["E_FTT_markov"] = out["n_prefix_FT_tri"] * out["P_T_given_T_pair"]
    out["E_FTF_markov"] = out["n_prefix_FT_tri"] * out["P_F_given_T_pair"]

    out["E_FFT_markov"] = out["n_prefix_FF_tri"] * out["P_T_given_F_pair"]
    out["E_FFF_markov"] = out["n_prefix_FF_tri"] * out["P_F_given_F_pair"]

    for col in trigram_cols:
        out[f"E_{col}_markov_rate"] = safe_div(
            out[f"E_{col}_markov"],
            out["n_trigrams"]
        )

        out[f"{col}_markov_delta"] = (
            out[f"Obs_{col}_rate"] - out[f"E_{col}_markov_rate"]
        )

        out[f"{col}_markov_ratio"] = (
            out[col] / out[f"E_{col}_markov"].replace(0, np.nan)
        )

    # -------------------------------------------------
    # 7. 전체 T/F 비율 대비 조건부 비율차
    # -------------------------------------------------
    trigram_cond_effect_cols = []
    for col in pair_cols:
        out[f"Diff_T_after_{col}"] = out[f"P_T_given_{col}"] - out["P_T"]
        out[f"Diff_F_after_{col}"] = out[f"P_F_given_{col}"] - out["P_F"]
        out[f"Diff_T_after_{col}_ratio_vs_P_T"] = (
            out[f"P_T_given_{col}"] / out["P_T"].replace(0, np.nan)
        )
        out[f"Diff_F_after_{col}_ratio_vs_P_F"] = (
            out[f"P_F_given_{col}"] / out["P_F"].replace(0, np.nan)
        )
        out[f"Diff_T_after_{col}_odds_ratio_vs_P_T"] = smoothed_odds_ratio(
            out[f"P_T_given_{col}"],
            out["P_T"],
            smoothing=smoothing,
        )
        out[f"Diff_F_after_{col}_odds_ratio_vs_P_F"] = smoothed_odds_ratio(
            out[f"P_F_given_{col}"],
            out["P_F"],
            smoothing=smoothing,
        )
        out[f"Diff_T_after_{col}_log_odds_ratio_vs_P_T"] = np.log(
            out[f"Diff_T_after_{col}_odds_ratio_vs_P_T"]
        )
        out[f"Diff_F_after_{col}_log_odds_ratio_vs_P_F"] = np.log(
            out[f"Diff_F_after_{col}_odds_ratio_vs_P_F"]
        )
        trigram_cond_effect_cols.extend([
            f"Diff_T_after_{col}_ratio_vs_P_T",
            f"Diff_T_after_{col}_odds_ratio_vs_P_T",
            f"Diff_T_after_{col}_log_odds_ratio_vs_P_T",
            f"Diff_F_after_{col}_ratio_vs_P_F",
            f"Diff_F_after_{col}_odds_ratio_vs_P_F",
            f"Diff_F_after_{col}_log_odds_ratio_vs_P_F",
        ])

    # -------------------------------------------------
    # 8. 유지 후 유지/전환, 전환 후 유지/전환 묶음 비교
    # -------------------------------------------------
    trigram_sets = {
        # 유지 뒤
        "stay_stay": ["TTT", "FFF"],              # 유지 후 유지
        "stay_switch": ["TTF", "FFT"],            # 유지 후 전환

        # 전환 뒤
        "switch_extension": ["TFF", "FTT"],       # 전환 후 유지/연장
        "switch_return": ["TFT", "FTF"],          # 전환 후 전환/복귀
    }

    for name, cols in trigram_sets.items():
        out[f"{name}_Obs_rate"] = out[[f"Obs_{c}_rate" for c in cols]].sum(axis=1)

        # 기본 T/F 출현률 기대값 기준
        out[f"{name}_E_base_rate"] = out[[f"E_{c}_base_rate" for c in cols]].sum(axis=1)
        out[f"{name}_delta_vs_base"] = (
            out[f"{name}_Obs_rate"] - out[f"{name}_E_base_rate"]
        )
        out[f"{name}_ratio_vs_base"] = (
            out[f"{name}_Obs_rate"] / out[f"{name}_E_base_rate"].replace(0, np.nan)
        )

        # 1차 Markov 기대값 기준
        out[f"{name}_E_markov_rate"] = out[[f"E_{c}_markov_rate" for c in cols]].sum(axis=1)
        out[f"{name}_delta_vs_markov"] = (
            out[f"{name}_Obs_rate"] - out[f"{name}_E_markov_rate"]
        )
        out[f"{name}_ratio_vs_markov"] = (
            out[f"{name}_Obs_rate"] / out[f"{name}_E_markov_rate"].replace(0, np.nan)
        )

    # -------------------------------------------------
    # 9. 보기 편한 컬럼 순서 정리
    # -------------------------------------------------
    front_cols = [
        unit_col,
        "n_T", "n_F", "n_sentences", "P_T", "P_F",
        "n_pairs", "n_trigrams",
    ]

    pair_count_cols = [
        "TT", "TF", "FT", "FF",
        "n_prev_T_pair", "n_prev_F_pair",
        "P_T_given_T_pair", "P_F_given_T_pair",
        "P_T_given_F_pair", "P_F_given_F_pair",
        *pair_cond_effect_cols,
    ]

    pair_base_cols = []
    for col in pair_cols:
        pair_base_cols += [
            f"E_{col}_base",
            f"Obs_{col}_pair_rate",
            f"E_{col}_base_pair_rate",
            f"{col}_base_delta",
            f"{col}_base_ratio",
        ]

    pair_summary_cols = []
    for name in ["pair_stay", "pair_switch"]:
        pair_summary_cols += [
            f"{name}_Obs_rate",
            f"{name}_E_base_rate",
            f"{name}_delta_vs_base",
            f"{name}_ratio_vs_base",
        ]

    trigram_count_cols = trigram_cols.copy()

    trigram_cond_cols = []
    for col in pair_cols:
        trigram_cond_cols += [
            f"n_prefix_{col}_tri",
            f"P_T_given_{col}",
            f"P_F_given_{col}",
            f"Diff_T_after_{col}",
            f"Diff_F_after_{col}",
        ]
        trigram_cond_cols += [
            effect_col
            for effect_col in trigram_cond_effect_cols
            if effect_col.startswith(f"Diff_T_after_{col}_")
            or effect_col.startswith(f"Diff_F_after_{col}_")
        ]

    trigram_base_cols = []
    for col in trigram_cols:
        trigram_base_cols += [
            f"E_{col}_base",
            f"Obs_{col}_rate",
            f"E_{col}_base_rate",
            f"{col}_base_delta",
            f"{col}_base_ratio",
        ]

    trigram_markov_cols = []
    for col in trigram_cols:
        trigram_markov_cols += [
            f"E_{col}_markov",
            f"E_{col}_markov_rate",
            f"{col}_markov_delta",
            f"{col}_markov_ratio",
        ]

    trigram_summary_cols = []
    for name in ["stay_stay", "stay_switch", "switch_extension", "switch_return"]:
        trigram_summary_cols += [
            f"{name}_Obs_rate",
            f"{name}_E_base_rate",
            f"{name}_delta_vs_base",
            f"{name}_ratio_vs_base",
            f"{name}_E_markov_rate",
            f"{name}_delta_vs_markov",
            f"{name}_ratio_vs_markov",
        ]

    result = out.reset_index()

    ordered_cols = (
        front_cols
        + pair_count_cols
        + pair_base_cols
        + pair_summary_cols
        + trigram_count_cols
        + trigram_cond_cols
        + trigram_base_cols
        + trigram_markov_cols
        + trigram_summary_cols
    )

    # 실제 존재하는 컬럼만, 중복 없이 정렬
    seen = set()
    ordered_cols = [
        c for c in ordered_cols
        if c in result.columns and not (c in seen or seen.add(c))
    ]

    rest_cols = [c for c in result.columns if c not in ordered_cols]

    result = result[ordered_cols + rest_cols]
    return add_transition_odds_columns(result, smoothing=smoothing)
