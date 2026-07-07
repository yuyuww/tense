"""Legacy unit-level trigram analysis functions extracted from 4.5.1.

These functions were previously bundled in stats.transition_analysis. They are kept
here for 4.5.2/unit workflows. Original probability, ratio, and delta formulas are
left as-is; smoothing is only used when odds-ratio/log-odds-ratio columns are added.
"""

from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd

from stats.transition_odds import add_transition_odds_columns


#unit_effect unit별 1연쇄 / 2연쇄 / 3연쇄 분석 함수
# =========================================================
# 2. unit별 1연쇄 / 2연쇄 / 3연쇄 분석 함수
# =========================================================
def _normalize_binary_state(x):
    if pd.isna(x):
        return np.nan

    if x in [True, "True", "T", "t", 1, "1"]:
        return True

    if x in [False, "False", "F", "f", 0, "0"]:
        return False

    return np.nan


def _rate(num, den):
    return num / den.replace(0, np.nan)


def _ratio(obs, exp):
    return obs / exp.replace(0, np.nan)


# =========================================================
# 1. baseline 계산 함수
# =========================================================

def analyze_chain_baseline_for_unit_effect(
    df: pd.DataFrame,
    *,
    baseline_col: str,
    state_col: str = "sentence_f_EP_T",
    prev_state_col: str = "prev_sentence_f_EP_T",
    next_state_col: str = "next_sentence_f_EP_T",
    has_prev_col: str = "has_prev_sentence",
    has_next_col: str = "has_next_sentence",
    count_col: str = "count",
) -> pd.DataFrame:
    """
    baseline_col별 기본 T/F 비율을 계산한다.

    반환값:
    - baseline_P_T / baseline_P_F
      : 현재문장 기준 전체 T/F 비율

    - baseline_prev_P_T / baseline_prev_P_F
      : 2연쇄 분석에서 사용할 이전문장 T/F 비율

    - baseline_next_P_T / baseline_next_P_F
      : 3연쇄 분석에서 사용할 다음문장 기본 T/F 비율
    """

    work = df.copy()

    if count_col not in work.columns:
        work[count_col] = 1

    work["_prev"] = work[prev_state_col].map(_normalize_binary_state)
    work["_curr"] = work[state_col].map(_normalize_binary_state)
    work["_next"] = work[next_state_col].map(_normalize_binary_state)
    work["_has_prev"] = work[has_prev_col].map(_normalize_binary_state)
    work["_has_next"] = work[has_next_col].map(_normalize_binary_state)

    # -------------------------------------------------
    # 1) 현재문장 기준 baseline P_T
    # -------------------------------------------------
    curr_work = work[
        work[baseline_col].notna()
        & work["_curr"].notna()
    ]

    curr_counts = (
        curr_work
        .groupby([baseline_col, "_curr"], observed=True)[count_col]
        .sum()
        .unstack("_curr", fill_value=0)
    )

    out = pd.DataFrame(index=curr_counts.index)

    out["baseline_n_T"] = curr_counts[True] if True in curr_counts.columns else 0.0
    out["baseline_n_F"] = curr_counts[False] if False in curr_counts.columns else 0.0
    out["baseline_n"] = out["baseline_n_T"] + out["baseline_n_F"]

    out["baseline_P_T"] = _rate(out["baseline_n_T"], out["baseline_n"])
    out["baseline_P_F"] = _rate(out["baseline_n_F"], out["baseline_n"])

    # -------------------------------------------------
    # 2) 2연쇄 분석용 prev T/F 비율
    # -------------------------------------------------
    prev_work = work[
        (work["_has_prev"] == True)
        & work[baseline_col].notna()
        & work["_prev"].notna()
        & work["_curr"].notna()
    ]

    prev_counts = (
        prev_work
        .groupby([baseline_col, "_prev"], observed=True)[count_col]
        .sum()
        .unstack("_prev", fill_value=0)
    )

    out = out.reindex(out.index.union(prev_counts.index))

    out["baseline_prev_n_T"] = (
        prev_counts[True].reindex(out.index, fill_value=0).astype(float)
        if True in prev_counts.columns else 0.0
    )
    out["baseline_prev_n_F"] = (
        prev_counts[False].reindex(out.index, fill_value=0).astype(float)
        if False in prev_counts.columns else 0.0
    )

    out["baseline_prev_n"] = out["baseline_prev_n_T"] + out["baseline_prev_n_F"]
    out["baseline_prev_P_T"] = _rate(out["baseline_prev_n_T"], out["baseline_prev_n"])
    out["baseline_prev_P_F"] = _rate(out["baseline_prev_n_F"], out["baseline_prev_n"])

    # -------------------------------------------------
    # 3) 3연쇄 분석용 next T/F 비율
    #    실제 3연쇄와 같은 조건: has_prev=True, has_next=True
    # -------------------------------------------------
    next_work = work[
        (work["_has_prev"] == True)
        & (work["_has_next"] == True)
        & work[baseline_col].notna()
        & work["_prev"].notna()
        & work["_curr"].notna()
        & work["_next"].notna()
    ]

    next_counts = (
        next_work
        .groupby([baseline_col, "_next"], observed=True)[count_col]
        .sum()
        .unstack("_next", fill_value=0)
    )

    out = out.reindex(out.index.union(next_counts.index))

    out["baseline_next_n_T"] = (
        next_counts[True].reindex(out.index, fill_value=0).astype(float)
        if True in next_counts.columns else 0.0
    )
    out["baseline_next_n_F"] = (
        next_counts[False].reindex(out.index, fill_value=0).astype(float)
        if False in next_counts.columns else 0.0
    )

    out["baseline_next_n"] = out["baseline_next_n_T"] + out["baseline_next_n_F"]
    out["baseline_next_P_T"] = _rate(out["baseline_next_n_T"], out["baseline_next_n"])
    out["baseline_next_P_F"] = _rate(out["baseline_next_n_F"], out["baseline_next_n"])

    return out.reset_index()

# Main function4: analyze_unit_chain_effect : unit별 시제 연쇄 효과 분석 함수
def analyze_unit_chain_effect(
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
    """
    unit별 시제 연쇄 효과 분석.

    1연쇄:
    - unit_P_T vs baseline_P_T

    2연쇄:
    - 기대값 = unit_prev_P_T/F × baseline_P_T/F
    - 해당 unit이 놓인 이전문장 분포를 고정했을 때, 현재문장이 baseline T/F 비율대로 나오는가

    3연쇄:
    - 기대값 = 실제 Obs_TT/TF/FT/FF_rate × baseline_next_P_T/F
    - 실제 2연쇄 분포를 고정한 뒤 next가 기본 T/F 비율대로 가는지 봄.
    """

    work = df.copy()

    if count_col not in work.columns:
        work[count_col] = 1

    work["_prev"] = work[prev_state_col].map(_normalize_binary_state)
    work["_curr"] = work[state_col].map(_normalize_binary_state)
    work["_next"] = work[next_state_col].map(_normalize_binary_state)
    work["_has_prev"] = work[has_prev_col].map(_normalize_binary_state)
    work["_has_next"] = work[has_next_col].map(_normalize_binary_state)

    if baseline_df is None:
        baseline_df = analyze_chain_baseline_for_unit_effect(
            df,
            baseline_col=baseline_col,
            state_col=state_col,
            prev_state_col=prev_state_col,
            next_state_col=next_state_col,
            has_prev_col=has_prev_col,
            has_next_col=has_next_col,
            count_col=count_col,
        )

    # =====================================================
    # 1) unit별 현재문장 P_T
    # =====================================================
    curr_work = work[
        work[baseline_col].notna()
        & work[unit_col].notna()
        & work["_curr"].notna()
    ]

    curr_counts = (
        curr_work
        .groupby([baseline_col, unit_col, "_curr"], observed=True)[count_col]
        .sum()
        .unstack("_curr", fill_value=0)
    )

    out = pd.DataFrame(index=curr_counts.index)

    out["unit_n_T"] = curr_counts[True] if True in curr_counts.columns else 0.0
    out["unit_n_F"] = curr_counts[False] if False in curr_counts.columns else 0.0
    out["unit_n"] = out["unit_n_T"] + out["unit_n_F"]

    out["unit_P_T"] = _rate(out["unit_n_T"], out["unit_n"])
    out["unit_P_F"] = _rate(out["unit_n_F"], out["unit_n"])

    # =====================================================
    # 2) unit별 2연쇄 실제값
    # =====================================================
    bi_work = work[
        (work["_has_prev"] == True)
        & work[baseline_col].notna()
        & work[unit_col].notna()
        & work["_prev"].notna()
        & work["_curr"].notna()
    ]

    bi_counts = (
        bi_work
        .groupby([baseline_col, unit_col, "_prev", "_curr"], observed=True)[count_col]
        .sum()
        .unstack(["_prev", "_curr"], fill_value=0)
    )

    out = out.reindex(out.index.union(bi_counts.index))

    def get_bi(prev, curr):
        key = (prev, curr)
        if key in bi_counts.columns:
            return bi_counts[key].reindex(out.index, fill_value=0).astype(float)
        return pd.Series(0.0, index=out.index)

    out["TT"] = get_bi(True, True)
    out["TF"] = get_bi(True, False)
    out["FT"] = get_bi(False, True)
    out["FF"] = get_bi(False, False)

    bigram_cols = ["TT", "TF", "FT", "FF"]
    out["n_bigrams"] = out[bigram_cols].sum(axis=1)

    for col in bigram_cols:
        out[f"Obs_{col}_rate"] = _rate(out[col], out["n_bigrams"])

    # 조건부 2연쇄 확률
    out["P_curr_T_given_prev_T"] = _rate(out["TT"], out["TT"] + out["TF"])
    out["P_curr_F_given_prev_T"] = _rate(out["TF"], out["TT"] + out["TF"])

    out["P_curr_T_given_prev_F"] = _rate(out["FT"], out["FT"] + out["FF"])
    out["P_curr_F_given_prev_F"] = _rate(out["FF"], out["FT"] + out["FF"])

    # =====================================================
    # 3) unit별 3연쇄 실제값
    # =====================================================
    tri_work = work[
        (work["_has_prev"] == True)
        & (work["_has_next"] == True)
        & work[baseline_col].notna()
        & work[unit_col].notna()
        & work["_prev"].notna()
        & work["_curr"].notna()
        & work["_next"].notna()
    ]

    tri_counts = (
        tri_work
        .groupby([baseline_col, unit_col, "_prev", "_curr", "_next"], observed=True)[count_col]
        .sum()
        .unstack(["_prev", "_curr", "_next"], fill_value=0)
    )

    out = out.reindex(out.index.union(tri_counts.index))

    def get_tri(prev, curr, next_):
        key = (prev, curr, next_)
        if key in tri_counts.columns:
            return tri_counts[key].reindex(out.index, fill_value=0).astype(float)
        return pd.Series(0.0, index=out.index)

    tri_cols = ["TTT", "TTF", "TFT", "TFF", "FTT", "FTF", "FFT", "FFF"]

    out["TTT"] = get_tri(True, True, True)
    out["TTF"] = get_tri(True, True, False)

    out["TFT"] = get_tri(True, False, True)
    out["TFF"] = get_tri(True, False, False)

    out["FTT"] = get_tri(False, True, True)
    out["FTF"] = get_tri(False, True, False)

    out["FFT"] = get_tri(False, False, True)
    out["FFF"] = get_tri(False, False, False)

    out["n_trigrams"] = out[tri_cols].sum(axis=1)

    for col in tri_cols:
        out[f"Obs_{col}_rate"] = _rate(out[col], out["n_trigrams"])

    # 3연쇄에서 사용할 실제 2연쇄 분포
    # 주의: 2연쇄 전체가 아니라, 3연쇄 가능한 표본 안에서의 TT/TF/FT/FF 비율
    out["tri_TT"] = out["TTT"] + out["TTF"]
    out["tri_TF"] = out["TFT"] + out["TFF"]
    out["tri_FT"] = out["FTT"] + out["FTF"]
    out["tri_FF"] = out["FFT"] + out["FFF"]

    out["Obs_tri_TT_rate"] = _rate(out["tri_TT"], out["n_trigrams"])
    out["Obs_tri_TF_rate"] = _rate(out["tri_TF"], out["n_trigrams"])
    out["Obs_tri_FT_rate"] = _rate(out["tri_FT"], out["n_trigrams"])
    out["Obs_tri_FF_rate"] = _rate(out["tri_FF"], out["n_trigrams"])

    # 조건부 3연쇄 확률
    out["P_next_T_given_TT"] = _rate(out["TTT"], out["TTT"] + out["TTF"])
    out["P_next_F_given_TT"] = _rate(out["TTF"], out["TTT"] + out["TTF"])

    out["P_next_T_given_TF"] = _rate(out["TFT"], out["TFT"] + out["TFF"])
    out["P_next_F_given_TF"] = _rate(out["TFF"], out["TFT"] + out["TFF"])

    out["P_next_T_given_FT"] = _rate(out["FTT"], out["FTT"] + out["FTF"])
    out["P_next_F_given_FT"] = _rate(out["FTF"], out["FTT"] + out["FTF"])

    out["P_next_T_given_FF"] = _rate(out["FFT"], out["FFT"] + out["FFF"])
    out["P_next_F_given_FF"] = _rate(out["FFF"], out["FFT"] + out["FFF"])

    # =====================================================
    # 4) baseline 병합
    # =====================================================
    out = out.reset_index()

    out = out.merge(
        baseline_df,
        on=baseline_col,
        how="left",
    )

    # =====================================================
    # 5) 1연쇄 비교
    # =====================================================
    out["E_unit_P_T"] = out["baseline_P_T"]
    out["E_unit_P_F"] = out["baseline_P_F"]

    out["unit_P_T_delta_vs_baseline"] = out["unit_P_T"] - out["E_unit_P_T"]
    out["unit_P_F_delta_vs_baseline"] = out["unit_P_F"] - out["E_unit_P_F"]

    out["unit_P_T_ratio_vs_baseline"] = _ratio(out["unit_P_T"], out["E_unit_P_T"])
    out["unit_P_F_ratio_vs_baseline"] = _ratio(out["unit_P_F"], out["E_unit_P_F"])

    # =====================================================
    # 6) 2연쇄 기대값
    #
    # 기대값 =
    # unit의 실제 prev T/F 비율 × baseline의 현재 T/F 비율
    #
    # 의미:
    # 이 unit이 실제로 놓인 이전문장 T/F 환경은 인정한다.
    # 그 환경에서 현재문장이 baseline T/F 비율대로 나온다면
    # TT, TF, FT, FF가 어느 정도여야 하는지를 계산한다.
    # =====================================================

    # ---------------------------------
    # 6-1. unit의 실제 prev T/F 비율
    # ---------------------------------
    out["unit_prev_T_n"] = out["TT"] + out["TF"]
    out["unit_prev_F_n"] = out["FT"] + out["FF"]
    out["unit_prev_n"] = out["unit_prev_T_n"] + out["unit_prev_F_n"]

    out["unit_prev_P_T"] = _rate(out["unit_prev_T_n"], out["unit_prev_n"])
    out["unit_prev_P_F"] = _rate(out["unit_prev_F_n"], out["unit_prev_n"])

    # ---------------------------------
    # 6-2. 기대 2연쇄 비율
    # unit 실제 prev 분포 × baseline 현재 T/F 비율
    # ---------------------------------
    out["E_TT_rate"] = out["unit_prev_P_T"] * out["baseline_P_T"]
    out["E_TF_rate"] = out["unit_prev_P_T"] * out["baseline_P_F"]

    out["E_FT_rate"] = out["unit_prev_P_F"] * out["baseline_P_T"]
    out["E_FF_rate"] = out["unit_prev_P_F"] * out["baseline_P_F"]

    # 기대 빈도도 같이 만들어 두면 나중에 확인하기 좋음
    out["E_TT_n"] = out["n_bigrams"] * out["E_TT_rate"]
    out["E_TF_n"] = out["n_bigrams"] * out["E_TF_rate"]
    out["E_FT_n"] = out["n_bigrams"] * out["E_FT_rate"]
    out["E_FF_n"] = out["n_bigrams"] * out["E_FF_rate"]

    # ---------------------------------
    # 6-3. 실제값 - 기대값 / 실제값 ÷ 기대값
    # ---------------------------------
    for col in bigram_cols:
        out[f"{col}_delta_vs_expected"] = (
            out[f"Obs_{col}_rate"] - out[f"E_{col}_rate"]
        )

        out[f"{col}_ratio_vs_expected"] = _ratio(
            out[f"Obs_{col}_rate"],
            out[f"E_{col}_rate"]
        )

    # =====================================================
    # 6-4. 2연쇄 해석용 묶음
    # =====================================================

    # 유지: TT + FF
    out["stay_Obs_rate"] = out["Obs_TT_rate"] + out["Obs_FF_rate"]
    out["stay_E_rate"] = out["E_TT_rate"] + out["E_FF_rate"]
    out["stay_delta_vs_expected"] = (
        out["stay_Obs_rate"] - out["stay_E_rate"]
    )
    out["stay_ratio_vs_expected"] = _ratio(
        out["stay_Obs_rate"],
        out["stay_E_rate"]
    )

    # 전환: TF + FT
    out["switch_Obs_rate"] = out["Obs_TF_rate"] + out["Obs_FT_rate"]
    out["switch_E_rate"] = out["E_TF_rate"] + out["E_FT_rate"]
    out["switch_delta_vs_expected"] = (
        out["switch_Obs_rate"] - out["switch_E_rate"]
    )
    out["switch_ratio_vs_expected"] = _ratio(
        out["switch_Obs_rate"],
        out["switch_E_rate"]
    )

    # =====================================================
    # 6-5. prev가 curr에 미치는 방향별 영향
    #
    # 기준은 unit_P_T가 아니라 baseline_P_T
    # 즉, prev=T/F 조건에서 현재문장의 T 비율이
    # 기본 T 비율보다 얼마나 달라지는지를 본다.
    # =====================================================

    out["prev_T_to_curr_T_delta_vs_baseline_P_T"] = (
        out["P_curr_T_given_prev_T"] - out["baseline_P_T"]
    )

    out["prev_F_to_curr_T_delta_vs_baseline_P_T"] = (
        out["P_curr_T_given_prev_F"] - out["baseline_P_T"]
    )

    out["prev_T_to_curr_F_delta_vs_baseline_P_F"] = (
        out["P_curr_F_given_prev_T"] - out["baseline_P_F"]
    )

    out["prev_F_to_curr_F_delta_vs_baseline_P_F"] = (
        out["P_curr_F_given_prev_F"] - out["baseline_P_F"]
    )

    # prev=T일 때와 prev=F일 때 현재문장 T 비율이 얼마나 갈라지는가
    out["prev_chain_gap_on_curr_T"] = (
        out["P_curr_T_given_prev_T"] - out["P_curr_T_given_prev_F"]
    )

    # prev=T일 때와 prev=F일 때 현재문장 F 비율이 얼마나 갈라지는가
    out["prev_chain_gap_on_curr_F"] = (
        out["P_curr_F_given_prev_T"] - out["P_curr_F_given_prev_F"]
    )

    # =====================================================
    # 7) 3연쇄 기대값
    #
    # 기대값 =
    # 실제 TT/TF/FT/FF 비율 × baseline의 next T/F 비율
    # =====================================================

    out["E_TTT_rate"] = out["Obs_tri_TT_rate"] * out["baseline_next_P_T"]
    out["E_TTF_rate"] = out["Obs_tri_TT_rate"] * out["baseline_next_P_F"]

    out["E_TFT_rate"] = out["Obs_tri_TF_rate"] * out["baseline_next_P_T"]
    out["E_TFF_rate"] = out["Obs_tri_TF_rate"] * out["baseline_next_P_F"]

    out["E_FTT_rate"] = out["Obs_tri_FT_rate"] * out["baseline_next_P_T"]
    out["E_FTF_rate"] = out["Obs_tri_FT_rate"] * out["baseline_next_P_F"]

    out["E_FFT_rate"] = out["Obs_tri_FF_rate"] * out["baseline_next_P_T"]
    out["E_FFF_rate"] = out["Obs_tri_FF_rate"] * out["baseline_next_P_F"]

    for col in tri_cols:
        out[f"{col}_delta_vs_expected"] = out[f"Obs_{col}_rate"] - out[f"E_{col}_rate"]
        out[f"{col}_ratio_vs_expected"] = _ratio(out[f"Obs_{col}_rate"], out[f"E_{col}_rate"])

    # 3연쇄 조건부 비교
    # 각 TT/TF/FT/FF 뒤의 next=T/F가 baseline_next_P_T/F보다 얼마나 다른가
    out["next_T_after_TT_delta_vs_baseline"] = out["P_next_T_given_TT"] - out["baseline_next_P_T"]
    out["next_T_after_TF_delta_vs_baseline"] = out["P_next_T_given_TF"] - out["baseline_next_P_T"]
    out["next_T_after_FT_delta_vs_baseline"] = out["P_next_T_given_FT"] - out["baseline_next_P_T"]
    out["next_T_after_FF_delta_vs_baseline"] = out["P_next_T_given_FF"] - out["baseline_next_P_T"]

    # =====================================================
    # 8) 복귀/지속/유지/전환 묶음 지표
    # =====================================================

    # 전환 후 복귀: TFT, FTF
    out["switch_return_Obs_rate"] = out["Obs_TFT_rate"] + out["Obs_FTF_rate"]
    out["switch_return_E_rate"] = out["E_TFT_rate"] + out["E_FTF_rate"]
    out["switch_return_delta_vs_expected"] = (
        out["switch_return_Obs_rate"] - out["switch_return_E_rate"]
    )
    out["switch_return_ratio_vs_expected"] = _ratio(
        out["switch_return_Obs_rate"],
        out["switch_return_E_rate"]
    )

    # 전환 후 지속: TFF, FTT
    out["switch_extension_Obs_rate"] = out["Obs_TFF_rate"] + out["Obs_FTT_rate"]
    out["switch_extension_E_rate"] = out["E_TFF_rate"] + out["E_FTT_rate"]
    out["switch_extension_delta_vs_expected"] = (
        out["switch_extension_Obs_rate"] - out["switch_extension_E_rate"]
    )
    out["switch_extension_ratio_vs_expected"] = _ratio(
        out["switch_extension_Obs_rate"],
        out["switch_extension_E_rate"]
    )

    # 유지 후 유지: TTT, FFF
    out["stay_stay_Obs_rate"] = out["Obs_TTT_rate"] + out["Obs_FFF_rate"]
    out["stay_stay_E_rate"] = out["E_TTT_rate"] + out["E_FFF_rate"]
    out["stay_stay_delta_vs_expected"] = (
        out["stay_stay_Obs_rate"] - out["stay_stay_E_rate"]
    )
    out["stay_stay_ratio_vs_expected"] = _ratio(
        out["stay_stay_Obs_rate"],
        out["stay_stay_E_rate"]
    )

    # 유지 후 전환: TTF, FFT
    out["stay_switch_Obs_rate"] = out["Obs_TTF_rate"] + out["Obs_FFT_rate"]
    out["stay_switch_E_rate"] = out["E_TTF_rate"] + out["E_FFT_rate"]
    out["stay_switch_delta_vs_expected"] = (
        out["stay_switch_Obs_rate"] - out["stay_switch_E_rate"]
    )
    out["stay_switch_ratio_vs_expected"] = _ratio(
        out["stay_switch_Obs_rate"],
        out["stay_switch_E_rate"]
    )

    # 방향별 복귀율
    # TF 뒤에 T가 오면 원래 T로 복귀
    out["return_after_TF_Obs_rate"] = out["P_next_T_given_TF"]
    out["return_after_TF_E_rate"] = out["baseline_next_P_T"]
    out["return_after_TF_delta_vs_expected"] = (
        out["return_after_TF_Obs_rate"] - out["return_after_TF_E_rate"]
    )

    # FT 뒤에 F가 오면 원래 F로 복귀
    out["return_after_FT_Obs_rate"] = out["P_next_F_given_FT"]
    out["return_after_FT_E_rate"] = out["baseline_next_P_F"]
    out["return_after_FT_delta_vs_expected"] = (
        out["return_after_FT_Obs_rate"] - out["return_after_FT_E_rate"]
    )

    # 방향별 지속율
    out["extension_after_TF_Obs_rate"] = out["P_next_F_given_TF"]
    out["extension_after_TF_E_rate"] = out["baseline_next_P_F"]
    out["extension_after_TF_delta_vs_expected"] = (
        out["extension_after_TF_Obs_rate"] - out["extension_after_TF_E_rate"]
    )

    out["extension_after_FT_Obs_rate"] = out["P_next_T_given_FT"]
    out["extension_after_FT_E_rate"] = out["baseline_next_P_T"]
    out["extension_after_FT_delta_vs_expected"] = (
        out["extension_after_FT_Obs_rate"] - out["extension_after_FT_E_rate"]
    )

    # =====================================================
    # 9) 최소 빈도 필터
    # =====================================================
    if min_unit_n > 0:
        out = out[out["unit_n"] >= min_unit_n]

    if min_bigram_n > 0:
        out = out[out["n_bigrams"] >= min_bigram_n]

    if min_trigram_n > 0:
        out = out[out["n_trigrams"] >= min_trigram_n]

    return add_transition_odds_columns(out, smoothing=smoothing)


# 3연쇄 unit별 연쇄효과 분석 함수
# =========================================================
# 0. 공통 유틸 함수
# =========================================================

def _normalize_binary_state(x):
    """
    T/F, True/False, 1/0 등을 True/False로 정규화한다.
    """
    if pd.isna(x):
        return np.nan

    if x in [True, "True", "TRUE", "T", "t", 1, "1"]:
        return True

    if x in [False, "False", "FALSE", "F", "f", 0, "0"]:
        return False

    return np.nan


def _rate(num, den):
    """
    안전한 나눗셈.
    분모가 0이면 NaN.
    """
    return num / den.replace(0, np.nan)


def _ratio(obs, exp):
    """
    실제값 / 기대값.
    기대값이 0이면 NaN.
    """
    return obs / exp.replace(0, np.nan)


# =========================================================
# 1. baseline 계산 함수
# =========================================================

def analyze_sequence_effect_baseline(
    df: pd.DataFrame,
    *,
    baseline_col: str,
    state_col: str = "sentence_f_EP_T",
    prev_state_col: str = "prev_sentence_f_EP_T",
    next_state_col: str = "next_sentence_f_EP_T",
    has_prev_col: str = "has_prev_sentence",
    has_next_col: str = "has_next_sentence",
    count_col: str = "count",
) -> pd.DataFrame:
    """
    연쇄효과 분석에 필요한 baseline 값을 계산한다.

    =====================================================
    이 함수에서 구하는 baseline
    =====================================================

    1. 현재문장 기준 기본 T/F 비율

        baseline_P_T = baseline_n_T / baseline_n
        baseline_P_F = baseline_n_F / baseline_n

    2. 3연쇄 분석용 다음문장 기본 T/F 비율

        baseline_next_P_T = baseline_next_n_T / baseline_next_n
        baseline_next_P_F = baseline_next_n_F / baseline_next_n

        =====================================================
    """

    work = df.copy()

    if count_col not in work.columns:
        work[count_col] = 1

    work["_prev"] = work[prev_state_col].map(_normalize_binary_state)
    work["_curr"] = work[state_col].map(_normalize_binary_state)
    work["_next"] = work[next_state_col].map(_normalize_binary_state)
    work["_has_prev"] = work[has_prev_col].map(_normalize_binary_state)
    work["_has_next"] = work[has_next_col].map(_normalize_binary_state)

    # -------------------------------------------------
    # 1) 현재문장 기준 baseline P_T / P_F
    # -------------------------------------------------
    curr_work = work[
        work[baseline_col].notna()
        & work["_curr"].notna()
    ]

    curr_counts = (
        curr_work
        .groupby([baseline_col, "_curr"], observed=True)[count_col]
        .sum()
        .unstack("_curr", fill_value=0)
    )

    out = pd.DataFrame(index=curr_counts.index)

    out["baseline_n_T"] = (
        curr_counts[True].astype(float)
        if True in curr_counts.columns else 0.0
    )

    out["baseline_n_F"] = (
        curr_counts[False].astype(float)
        if False in curr_counts.columns else 0.0
    )

    out["baseline_n"] = out["baseline_n_T"] + out["baseline_n_F"]

    out["baseline_P_T"] = _rate(out["baseline_n_T"], out["baseline_n"])
    out["baseline_P_F"] = _rate(out["baseline_n_F"], out["baseline_n"])

    # -------------------------------------------------
    # 2) 3연쇄 분석용 baseline next P_T / P_F
    #
    # 실제 3연쇄와 같은 조건:
    # has_prev=True, has_next=True인 행에서 next의 기본 T/F 비율
    # -------------------------------------------------
    tri_base_work = work[
        (work["_has_prev"] == True)
        & (work["_has_next"] == True)
        & work[baseline_col].notna()
        & work["_prev"].notna()
        & work["_curr"].notna()
        & work["_next"].notna()
    ]

    next_counts = (
        tri_base_work
        .groupby([baseline_col, "_next"], observed=True)[count_col]
        .sum()
        .unstack("_next", fill_value=0)
    )

    out = out.reindex(out.index.union(next_counts.index))

    out["baseline_next_n_T"] = (
        next_counts[True].reindex(out.index, fill_value=0).astype(float)
        if True in next_counts.columns else 0.0
    )

    out["baseline_next_n_F"] = (
        next_counts[False].reindex(out.index, fill_value=0).astype(float)
        if False in next_counts.columns else 0.0
    )

    out["baseline_next_n"] = (
        out["baseline_next_n_T"] + out["baseline_next_n_F"]
    )

    out["baseline_next_P_T"] = _rate(
        out["baseline_next_n_T"],
        out["baseline_next_n"]
    )

    out["baseline_next_P_F"] = _rate(
        out["baseline_next_n_F"],
        out["baseline_next_n"]
    )

    return out.reset_index()


# =========================================================
# 2. unit별 연쇄효과 분석 함수
# =========================================================

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
) -> pd.DataFrame:
    """
    unit별 1연쇄·2연쇄·3연쇄 분석 함수

    목적:
        이 함수는 특정 unit이 나타난 문맥에서
        이전문장-현재문장-다음문장의 T/F 연쇄가
        기본 T/F 출현률 기대값보다 얼마나 벗어나는지를 본다.

        즉, 핵심은 unit의 고유한 T 출현률이 아니라 prev → curr, prev-curr → next의 연쇄효과이다.

    기호:
        T = 문장 끝에 '-었-'이 있음
        F = 문장 끝에 '-었-'이 없음

        prev = 이전문장
        curr = 현재문장
        next = 다음문장

        baseline = baseline_col별 전체 기준값
        unit = 동사, 어미, 보조용언 등 분석 대상

    =====================================================
    1. 1연쇄: unit 자체의 T/F 출현률
    =====================================================

    실제값:
        unit_P_T = unit_n_T / unit_n
        unit_P_F = unit_n_F / unit_n

    기대값:
        E_unit_P_T = baseline_P_T
        E_unit_P_F = baseline_P_F

    비교:
        unit_P_T_delta = unit_P_T - baseline_P_T
        unit_P_T_ratio = unit_P_T / baseline_P_T

    해석:
        이 unit이 baseline보다 '-었-'을 많이/적게 쓰는지 본다.
        단, 이것은 연쇄효과 자체가 아니라 unit의 기본 T/F 성향이다.

    =====================================================
    2. 2연쇄: prev가 curr에 미치는 연쇄효과
    =====================================================

    실제 2연쇄:
        TT = prev=T, curr=T
        TF = prev=T, curr=F
        FT = prev=F, curr=T
        FF = prev=F, curr=F

    unit이 실제로 놓인 prev 환경:
        unit_prev_P_T = (TT + TF) / n_bigrams
        unit_prev_P_F = (FT + FF) / n_bigrams

    기대값:
        E_TT = unit_prev_P_T × unit_curr_P_T
        E_TF = unit_prev_P_T × unit_curr_P_F

        E_FT = unit_prev_P_F × unit_curr_P_T
        E_FF = unit_prev_P_F × unit_curr_P_F

    해석:
        이 unit이 실제로 어떤 이전문장 T/F 환경에 놓였는지는 고정한다.
        그 상태에서 현재문장이 unit T/F 비율대로 나오지 않고,
        이전문장 상태에 끌리는지를 본다.

    =====================================================
    3. 3연쇄: prev-curr 연쇄가 next에 미치는 효과
    =====================================================

    실제 3연쇄:
        TTT, TTF, TFT, TFF, FTT, FTF, FFT, FFF

    3연쇄 안에서 실제 prev-curr 분포:
        Obs_tri_TT = (TTT + TTF) / n_trigrams
        Obs_tri_TF = (TFT + TFF) / n_trigrams
        Obs_tri_FT = (FTT + FTF) / n_trigrams
        Obs_tri_FF = (FFT + FFF) / n_trigrams

    기대값:
        E_TTT = Obs_tri_TT × baseline_next_P_T
        E_TTF = Obs_tri_TT × baseline_next_P_F

        E_TFT = Obs_tri_TF × baseline_next_P_T
        E_TFF = Obs_tri_TF × baseline_next_P_F

        E_FTT = Obs_tri_FT × baseline_next_P_T
        E_FTF = Obs_tri_FT × baseline_next_P_F

        E_FFT = Obs_tri_FF × baseline_next_P_T
        E_FFF = Obs_tri_FF × baseline_next_P_F

    해석:
        실제 prev-curr 연쇄 분포는 고정한다.
        그 뒤 next가 baseline T/F 비율대로 나오지 않고,
        앞의 TT, TF, FT, FF 연쇄에 끌리는지를 본다.

    =====================================================
    4. 묶음 지표
    =====================================================

    2연쇄:
        stay   = TT + FF
        switch = TF + FT

    3연쇄:
        switch_return    = TFT + FTF
            전환 후 원래 상태로 복귀

        switch_extension = TFF + FTT
            전환된 상태가 다음문장까지 지속

        stay_stay        = TTT + FFF
            유지된 상태가 다음문장까지 유지

        stay_switch      = TTF + FFT
            유지되던 상태가 다음문장에서 전환

    방향별 복귀:
        return_after_TF = TFT / (TFT + TFF)
        return_after_FT = FTF / (FTT + FTF)

    방향별 지속:
        extension_after_TF = TFF / (TFT + TFF)
        extension_after_FT = FTT / (FTT + FTF)
    """

    work = df.copy()

    if count_col not in work.columns:
        work[count_col] = 1

    work["_prev"] = work[prev_state_col].map(_normalize_binary_state)
    work["_curr"] = work[state_col].map(_normalize_binary_state)
    work["_next"] = work[next_state_col].map(_normalize_binary_state)
    work["_has_prev"] = work[has_prev_col].map(_normalize_binary_state)
    work["_has_next"] = work[has_next_col].map(_normalize_binary_state)

    if baseline_df is None:
        baseline_df = analyze_sequence_effect_baseline(
            df,
            baseline_col=baseline_col,
            state_col=state_col,
            prev_state_col=prev_state_col,
            next_state_col=next_state_col,
            has_prev_col=has_prev_col,
            has_next_col=has_next_col,
            count_col=count_col,
        )

    # =====================================================
    # 1) unit별 현재문장 P_T / P_F
    # =====================================================

    curr_work = work[
        work[baseline_col].notna()
        & work[unit_col].notna()
        & work["_curr"].notna()
    ]

    curr_counts = (
        curr_work
        .groupby([baseline_col, unit_col, "_curr"], observed=True)[count_col]
        .sum()
        .unstack("_curr", fill_value=0)
    )

    out = pd.DataFrame(index=curr_counts.index)

    out["unit_n_T"] = (
        curr_counts[True].astype(float)
        if True in curr_counts.columns else 0.0
    )

    out["unit_n_F"] = (
        curr_counts[False].astype(float)
        if False in curr_counts.columns else 0.0
    )

    out["unit_n"] = out["unit_n_T"] + out["unit_n_F"]

    out["unit_P_T"] = _rate(out["unit_n_T"], out["unit_n"])
    out["unit_P_F"] = _rate(out["unit_n_F"], out["unit_n"])

    # =====================================================
    # 2) unit별 prev-curr 2연쇄 실제값
    # =====================================================

    bi_work = work[
        (work["_has_prev"] == True)
        & work[baseline_col].notna()
        & work[unit_col].notna()
        & work["_prev"].notna()
        & work["_curr"].notna()
    ]

    bi_counts = (
        bi_work
        .groupby(
            [baseline_col, unit_col, "_prev", "_curr"],
            observed=True
        )[count_col]
        .sum()
        .unstack(["_prev", "_curr"], fill_value=0)
    )

    out = out.reindex(out.index.union(bi_counts.index))

    def get_bi(prev, curr):
        key = (prev, curr)
        if key in bi_counts.columns:
            return bi_counts[key].reindex(out.index, fill_value=0).astype(float)
        return pd.Series(0.0, index=out.index)

    out["TT"] = get_bi(True, True)
    out["TF"] = get_bi(True, False)
    out["FT"] = get_bi(False, True)
    out["FF"] = get_bi(False, False)

    bigram_cols = ["TT", "TF", "FT", "FF"]

    out["n_bigrams"] = out[bigram_cols].sum(axis=1)

    for col in bigram_cols:
        out[f"Obs_{col}_rate"] = _rate(out[col], out["n_bigrams"])

    # unit이 실제로 놓인 이전문장 T/F 환경
    out["unit_prev_T_n"] = out["TT"] + out["TF"]
    out["unit_prev_F_n"] = out["FT"] + out["FF"]
    out["unit_prev_n"] = out["unit_prev_T_n"] + out["unit_prev_F_n"]

    out["unit_prev_P_T"] = _rate(out["unit_prev_T_n"], out["unit_prev_n"])
    out["unit_prev_P_F"] = _rate(out["unit_prev_F_n"], out["unit_prev_n"])

    # 조건부 실제값
    out["P_curr_T_given_prev_T"] = _rate(out["TT"], out["TT"] + out["TF"])
    out["P_curr_F_given_prev_T"] = _rate(out["TF"], out["TT"] + out["TF"])

    out["P_curr_T_given_prev_F"] = _rate(out["FT"], out["FT"] + out["FF"])
    out["P_curr_F_given_prev_F"] = _rate(out["FF"], out["FT"] + out["FF"])

    # =====================================================
    # 3) unit별 prev-curr-next 3연쇄 실제값
    # =====================================================

    tri_work = work[
        (work["_has_prev"] == True)
        & (work["_has_next"] == True)
        & work[baseline_col].notna()
        & work[unit_col].notna()
        & work["_prev"].notna()
        & work["_curr"].notna()
        & work["_next"].notna()
    ]

    tri_counts = (
        tri_work
        .groupby(
            [baseline_col, unit_col, "_prev", "_curr", "_next"],
            observed=True
        )[count_col]
        .sum()
        .unstack(["_prev", "_curr", "_next"], fill_value=0)
    )

    out = out.reindex(out.index.union(tri_counts.index))

    def get_tri(prev, curr, next_):
        key = (prev, curr, next_)
        if key in tri_counts.columns:
            return tri_counts[key].reindex(out.index, fill_value=0).astype(float)
        return pd.Series(0.0, index=out.index)

    tri_cols = ["TTT", "TTF", "TFT", "TFF", "FTT", "FTF", "FFT", "FFF"]

    out["TTT"] = get_tri(True, True, True)
    out["TTF"] = get_tri(True, True, False)

    out["TFT"] = get_tri(True, False, True)
    out["TFF"] = get_tri(True, False, False)

    out["FTT"] = get_tri(False, True, True)
    out["FTF"] = get_tri(False, True, False)

    out["FFT"] = get_tri(False, False, True)
    out["FFF"] = get_tri(False, False, False)

    out["n_trigrams"] = out[tri_cols].sum(axis=1)

    for col in tri_cols:
        out[f"Obs_{col}_rate"] = _rate(out[col], out["n_trigrams"])

    # 3연쇄 내부의 실제 prev-curr 분포
    out["tri_TT"] = out["TTT"] + out["TTF"]
    out["tri_TF"] = out["TFT"] + out["TFF"]
    out["tri_FT"] = out["FTT"] + out["FTF"]
    out["tri_FF"] = out["FFT"] + out["FFF"]

    out["Obs_tri_TT_rate"] = _rate(out["tri_TT"], out["n_trigrams"])
    out["Obs_tri_TF_rate"] = _rate(out["tri_TF"], out["n_trigrams"])
    out["Obs_tri_FT_rate"] = _rate(out["tri_FT"], out["n_trigrams"])
    out["Obs_tri_FF_rate"] = _rate(out["tri_FF"], out["n_trigrams"])

    # 조건부 실제 next 확률
    out["P_next_T_given_TT"] = _rate(out["TTT"], out["TTT"] + out["TTF"])
    out["P_next_F_given_TT"] = _rate(out["TTF"], out["TTT"] + out["TTF"])

    out["P_next_T_given_TF"] = _rate(out["TFT"], out["TFT"] + out["TFF"])
    out["P_next_F_given_TF"] = _rate(out["TFF"], out["TFT"] + out["TFF"])

    out["P_next_T_given_FT"] = _rate(out["FTT"], out["FTT"] + out["FTF"])
    out["P_next_F_given_FT"] = _rate(out["FTF"], out["FTT"] + out["FTF"])

    out["P_next_T_given_FF"] = _rate(out["FFT"], out["FFT"] + out["FFF"])
    out["P_next_F_given_FF"] = _rate(out["FFF"], out["FFT"] + out["FFF"])

    # =====================================================
    # 4) baseline 병합
    # =====================================================

    out = out.reset_index()

    out = out.merge(
        baseline_df,
        on=baseline_col,
        how="left",
    )

    # =====================================================
    # 5) 1연쇄 비교
    # =====================================================

    out["E_unit_P_T"] = out["baseline_P_T"]
    out["E_unit_P_F"] = out["baseline_P_F"]

    out["unit_P_T_delta_vs_baseline"] = (
        out["unit_P_T"] - out["E_unit_P_T"]
    )

    out["unit_P_F_delta_vs_baseline"] = (
        out["unit_P_F"] - out["E_unit_P_F"]
    )

    out["unit_P_T_ratio_vs_baseline"] = _ratio(
        out["unit_P_T"],
        out["E_unit_P_T"]
    )

    out["unit_P_F_ratio_vs_baseline"] = _ratio(
        out["unit_P_F"],
        out["E_unit_P_F"]
    )

    # =====================================================
    # 6) 2연쇄 기대값
    #
    # 기대값 =
    # unit의 실제 prev T/F 비율 × unit의 실제 curr T/F 비율
    #
    # 의미:
    # 이 unit 안에서 prev와 curr가 서로 독립이라면
    # TT, TF, FT, FF가 어느 정도 나와야 하는지를 계산한다.
    #
    # 따라서 이 값과 실제값의 차이는
    # unit 자체의 T/F 출현률을 통제한 뒤에도
    # prev → curr 연쇄효과가 남는지를 보여준다.
    # =====================================================

    # ---------------------------------
    # 6-1. 2연쇄 표본 안에서의 unit curr T/F 비율
    # ---------------------------------
    out["unit_curr_T_n_in_bigrams"] = out["TT"] + out["FT"]
    out["unit_curr_F_n_in_bigrams"] = out["TF"] + out["FF"]
    out["unit_curr_n_in_bigrams"] = (
        out["unit_curr_T_n_in_bigrams"]
        + out["unit_curr_F_n_in_bigrams"]
    )

    out["unit_curr_P_T_in_bigrams"] = _rate(
        out["unit_curr_T_n_in_bigrams"],
        out["unit_curr_n_in_bigrams"]
    )

    out["unit_curr_P_F_in_bigrams"] = _rate(
        out["unit_curr_F_n_in_bigrams"],
        out["unit_curr_n_in_bigrams"]
    )

    # ---------------------------------
    # 6-2. 기대 2연쇄 비율
    # unit prev 분포 × unit curr 분포
    # ---------------------------------
    out["E_TT_rate"] = (
        out["unit_prev_P_T"]
        * out["unit_curr_P_T_in_bigrams"]
    )

    out["E_TF_rate"] = (
        out["unit_prev_P_T"]
        * out["unit_curr_P_F_in_bigrams"]
    )

    out["E_FT_rate"] = (
        out["unit_prev_P_F"]
        * out["unit_curr_P_T_in_bigrams"]
    )

    out["E_FF_rate"] = (
        out["unit_prev_P_F"]
        * out["unit_curr_P_F_in_bigrams"]
    )

    # ---------------------------------
    # 6-3. 실제값 - 기대값 / 실제값 ÷ 기대값
    # ---------------------------------
    for col in bigram_cols:
        out[f"{col}_delta_vs_expected"] = (
            out[f"Obs_{col}_rate"] - out[f"E_{col}_rate"]
        )

        out[f"{col}_ratio_vs_expected"] = _ratio(
            out[f"Obs_{col}_rate"],
            out[f"E_{col}_rate"]
        )

        out[f"E_{col}_n"] = (
            out["n_bigrams"] * out[f"E_{col}_rate"]
        )

    # =====================================================
    # 6-4. 2연쇄 해석용 묶음
    # =====================================================

    out["stay_Obs_rate"] = (
        out["Obs_TT_rate"] + out["Obs_FF_rate"]
    )

    out["stay_E_rate"] = (
        out["E_TT_rate"] + out["E_FF_rate"]
    )

    out["stay_delta_vs_expected"] = (
        out["stay_Obs_rate"] - out["stay_E_rate"]
    )

    out["stay_ratio_vs_expected"] = _ratio(
        out["stay_Obs_rate"],
        out["stay_E_rate"]
    )

    out["switch_Obs_rate"] = (
        out["Obs_TF_rate"] + out["Obs_FT_rate"]
    )

    out["switch_E_rate"] = (
        out["E_TF_rate"] + out["E_FT_rate"]
    )

    out["switch_delta_vs_expected"] = (
        out["switch_Obs_rate"] - out["switch_E_rate"]
    )

    out["switch_ratio_vs_expected"] = _ratio(
        out["switch_Obs_rate"],
        out["switch_E_rate"]
    )

    # =====================================================
    # 6-5. 방향별 prev → curr 연쇄효과
    #
    # 기준은 baseline_P_T가 아니라
    # unit 내부 curr T/F 비율이다.
    # =====================================================

    out["prev_T_to_curr_T_delta_vs_unit_curr_P_T"] = (
        out["P_curr_T_given_prev_T"]
        - out["unit_curr_P_T_in_bigrams"]
    )

    out["prev_F_to_curr_T_delta_vs_unit_curr_P_T"] = (
        out["P_curr_T_given_prev_F"]
        - out["unit_curr_P_T_in_bigrams"]
    )

    out["prev_T_to_curr_F_delta_vs_unit_curr_P_F"] = (
        out["P_curr_F_given_prev_T"]
        - out["unit_curr_P_F_in_bigrams"]
    )

    out["prev_F_to_curr_F_delta_vs_unit_curr_P_F"] = (
        out["P_curr_F_given_prev_F"]
        - out["unit_curr_P_F_in_bigrams"]
    )

    out["prev_chain_gap_on_curr_T"] = (
        out["P_curr_T_given_prev_T"]
        - out["P_curr_T_given_prev_F"]
    )

    out["prev_chain_gap_on_curr_F"] = (
        out["P_curr_F_given_prev_T"]
        - out["P_curr_F_given_prev_F"]
    )

    # =====================================================
    # 7) 3연쇄 기대값
    #
    # 기대값 =
    # 실제 prev-curr 분포 × baseline next T/F 비율
    # =====================================================

    out["E_TTT_rate"] = out["Obs_tri_TT_rate"] * out["baseline_next_P_T"]
    out["E_TTF_rate"] = out["Obs_tri_TT_rate"] * out["baseline_next_P_F"]

    out["E_TFT_rate"] = out["Obs_tri_TF_rate"] * out["baseline_next_P_T"]
    out["E_TFF_rate"] = out["Obs_tri_TF_rate"] * out["baseline_next_P_F"]

    out["E_FTT_rate"] = out["Obs_tri_FT_rate"] * out["baseline_next_P_T"]
    out["E_FTF_rate"] = out["Obs_tri_FT_rate"] * out["baseline_next_P_F"]

    out["E_FFT_rate"] = out["Obs_tri_FF_rate"] * out["baseline_next_P_T"]
    out["E_FFF_rate"] = out["Obs_tri_FF_rate"] * out["baseline_next_P_F"]

    for col in tri_cols:
        out[f"{col}_delta_vs_expected"] = (
            out[f"Obs_{col}_rate"] - out[f"E_{col}_rate"]
        )

        out[f"{col}_ratio_vs_expected"] = _ratio(
            out[f"Obs_{col}_rate"],
            out[f"E_{col}_rate"]
        )

        out[f"E_{col}_n"] = out["n_trigrams"] * out[f"E_{col}_rate"]

    # 3연쇄 방향별 next 효과
    out["next_T_after_TT_delta_vs_baseline_next_P_T"] = (
        out["P_next_T_given_TT"] - out["baseline_next_P_T"]
    )

    out["next_T_after_TF_delta_vs_baseline_next_P_T"] = (
        out["P_next_T_given_TF"] - out["baseline_next_P_T"]
    )

    out["next_T_after_FT_delta_vs_baseline_next_P_T"] = (
        out["P_next_T_given_FT"] - out["baseline_next_P_T"]
    )

    out["next_T_after_FF_delta_vs_baseline_next_P_T"] = (
        out["P_next_T_given_FF"] - out["baseline_next_P_T"]
    )

    out["next_F_after_TT_delta_vs_baseline_next_P_F"] = (
        out["P_next_F_given_TT"] - out["baseline_next_P_F"]
    )

    out["next_F_after_TF_delta_vs_baseline_next_P_F"] = (
        out["P_next_F_given_TF"] - out["baseline_next_P_F"]
    )

    out["next_F_after_FT_delta_vs_baseline_next_P_F"] = (
        out["P_next_F_given_FT"] - out["baseline_next_P_F"]
    )

    out["next_F_after_FF_delta_vs_baseline_next_P_F"] = (
        out["P_next_F_given_FF"] - out["baseline_next_P_F"]
    )

    # =====================================================
    # 8) 3연쇄 묶음 지표
    # =====================================================

    # 전환 후 복귀: TFT + FTF
    out["switch_return_Obs_rate"] = (
        out["Obs_TFT_rate"] + out["Obs_FTF_rate"]
    )

    out["switch_return_E_rate"] = (
        out["E_TFT_rate"] + out["E_FTF_rate"]
    )

    out["switch_return_delta_vs_expected"] = (
        out["switch_return_Obs_rate"]
        - out["switch_return_E_rate"]
    )

    out["switch_return_ratio_vs_expected"] = _ratio(
        out["switch_return_Obs_rate"],
        out["switch_return_E_rate"]
    )

    # 전환 후 지속: TFF + FTT
    out["switch_extension_Obs_rate"] = (
        out["Obs_TFF_rate"] + out["Obs_FTT_rate"]
    )

    out["switch_extension_E_rate"] = (
        out["E_TFF_rate"] + out["E_FTT_rate"]
    )

    out["switch_extension_delta_vs_expected"] = (
        out["switch_extension_Obs_rate"]
        - out["switch_extension_E_rate"]
    )

    out["switch_extension_ratio_vs_expected"] = _ratio(
        out["switch_extension_Obs_rate"],
        out["switch_extension_E_rate"]
    )

    # 유지 후 유지: TTT + FFF
    out["stay_stay_Obs_rate"] = (
        out["Obs_TTT_rate"] + out["Obs_FFF_rate"]
    )

    out["stay_stay_E_rate"] = (
        out["E_TTT_rate"] + out["E_FFF_rate"]
    )

    out["stay_stay_delta_vs_expected"] = (
        out["stay_stay_Obs_rate"]
        - out["stay_stay_E_rate"]
    )

    out["stay_stay_ratio_vs_expected"] = _ratio(
        out["stay_stay_Obs_rate"],
        out["stay_stay_E_rate"]
    )

    # 유지 후 전환: TTF + FFT
    out["stay_switch_Obs_rate"] = (
        out["Obs_TTF_rate"] + out["Obs_FFT_rate"]
    )

    out["stay_switch_E_rate"] = (
        out["E_TTF_rate"] + out["E_FFT_rate"]
    )

    out["stay_switch_delta_vs_expected"] = (
        out["stay_switch_Obs_rate"]
        - out["stay_switch_E_rate"]
    )

    out["stay_switch_ratio_vs_expected"] = _ratio(
        out["stay_switch_Obs_rate"],
        out["stay_switch_E_rate"]
    )

    # 방향별 복귀율
    out["return_after_TF_Obs_rate"] = out["P_next_T_given_TF"]
    out["return_after_TF_E_rate"] = out["baseline_next_P_T"]
    out["return_after_TF_delta_vs_expected"] = (
        out["return_after_TF_Obs_rate"]
        - out["return_after_TF_E_rate"]
    )
    out["return_after_TF_ratio_vs_expected"] = _ratio(
        out["return_after_TF_Obs_rate"],
        out["return_after_TF_E_rate"]
    )

    out["return_after_FT_Obs_rate"] = out["P_next_F_given_FT"]
    out["return_after_FT_E_rate"] = out["baseline_next_P_F"]
    out["return_after_FT_delta_vs_expected"] = (
        out["return_after_FT_Obs_rate"]
        - out["return_after_FT_E_rate"]
    )
    out["return_after_FT_ratio_vs_expected"] = _ratio(
        out["return_after_FT_Obs_rate"],
        out["return_after_FT_E_rate"]
    )

    # 방향별 지속율
    out["extension_after_TF_Obs_rate"] = out["P_next_F_given_TF"]
    out["extension_after_TF_E_rate"] = out["baseline_next_P_F"]
    out["extension_after_TF_delta_vs_expected"] = (
        out["extension_after_TF_Obs_rate"]
        - out["extension_after_TF_E_rate"]
    )
    out["extension_after_TF_ratio_vs_expected"] = _ratio(
        out["extension_after_TF_Obs_rate"],
        out["extension_after_TF_E_rate"]
    )

    out["extension_after_FT_Obs_rate"] = out["P_next_T_given_FT"]
    out["extension_after_FT_E_rate"] = out["baseline_next_P_T"]
    out["extension_after_FT_delta_vs_expected"] = (
        out["extension_after_FT_Obs_rate"]
        - out["extension_after_FT_E_rate"]
    )
    out["extension_after_FT_ratio_vs_expected"] = _ratio(
        out["extension_after_FT_Obs_rate"],
        out["extension_after_FT_E_rate"]
    )

    # =====================================================
    # 9) 최소 빈도 필터
    # =====================================================

    if min_unit_n > 0:
        out = out[out["unit_n"] >= min_unit_n]

    if min_bigram_n > 0:
        out = out[out["n_bigrams"] >= min_bigram_n]

    if min_trigram_n > 0:
        out = out[out["n_trigrams"] >= min_trigram_n]

    return out


#3연쇄 분석 함수에서 baseline_col 기준 기대값과 비교하는 함수, 동사/어미/보조용언용3


def _normalize_binary_state(x):
    if pd.isna(x):
        return np.nan
    if x in [True, "True", "T", "t", 1, "1"]:
        return True
    if x in [False, "False", "F", "f", 0, "0"]:
        return False
    return np.nan


def _safe_div(num, den):
    return num / den.replace(0, np.nan)


# =========================================================
# 1. baseline 계산 함수
# =========================================================

def analyze_trigram_baseline(
    df: pd.DataFrame,
    *,
    baseline_col: str,
    state_col: str = "sentence_f_EP_T",
    prev_state_col: str = "prev_sentence_f_EP_T",
    next_state_col: str = "next_sentence_f_EP_T",
    has_prev_col: str = "has_prev_sentence",
    has_next_col: str = "has_next_sentence",
    count_col: str = "count",
) -> pd.DataFrame:

    work = df.copy()

    work["_prev"] = work[prev_state_col].map(_normalize_binary_state)
    work["_curr"] = work[state_col].map(_normalize_binary_state)
    work["_next"] = work[next_state_col].map(_normalize_binary_state)
    work["_has_prev"] = work[has_prev_col].map(_normalize_binary_state)
    work["_has_next"] = work[has_next_col].map(_normalize_binary_state)

    # -----------------------------
    # 1) baseline P_T / P_F
    # -----------------------------
    curr = (
        work[
            work[baseline_col].notna()
            & work["_curr"].notna()
        ]
        .groupby([baseline_col, "_curr"], observed=True)[count_col]
        .sum()
        .unstack(fill_value=0)
    )

    out = pd.DataFrame(index=curr.index)

    out["baseline_n_T"] = curr[True] if True in curr.columns else 0
    out["baseline_n_F"] = curr[False] if False in curr.columns else 0
    out["baseline_n"] = out["baseline_n_T"] + out["baseline_n_F"]

    out["baseline_P_T"] = out["baseline_n_T"] / out["baseline_n"].replace(0, np.nan)
    out["baseline_P_F"] = out["baseline_n_F"] / out["baseline_n"].replace(0, np.nan)

    # -----------------------------
    # 2) baseline prev -> curr 전이확률
    # -----------------------------
    bi_work = work[
        (work["_has_prev"] == True)
        & work[baseline_col].notna()
        & work["_prev"].notna()
        & work["_curr"].notna()
    ]

    bi = (
        bi_work
        .groupby([baseline_col, "_prev", "_curr"], observed=True)[count_col]
        .sum()
        .unstack(["_prev", "_curr"], fill_value=0)
    )

    out = out.reindex(out.index.union(bi.index))

    def get_bi(prev, curr):
        key = (prev, curr)
        if key in bi.columns:
            return bi[key].reindex(out.index, fill_value=0).astype(float)
        return pd.Series(0.0, index=out.index)

    out["baseline_TT"] = get_bi(True, True)
    out["baseline_TF"] = get_bi(True, False)
    out["baseline_FT"] = get_bi(False, True)
    out["baseline_FF"] = get_bi(False, False)

    out["baseline_prev_T_n"] = out["baseline_TT"] + out["baseline_TF"]
    out["baseline_prev_F_n"] = out["baseline_FT"] + out["baseline_FF"]

    out["baseline_P_T_given_prev_T"] = out["baseline_TT"] / out["baseline_prev_T_n"].replace(0, np.nan)
    out["baseline_P_F_given_prev_T"] = out["baseline_TF"] / out["baseline_prev_T_n"].replace(0, np.nan)

    out["baseline_P_T_given_prev_F"] = out["baseline_FT"] / out["baseline_prev_F_n"].replace(0, np.nan)
    out["baseline_P_F_given_prev_F"] = out["baseline_FF"] / out["baseline_prev_F_n"].replace(0, np.nan)

    # -----------------------------
    # 3) baseline curr -> next 전이확률
    # -----------------------------
    next_work = work[
        (work["_has_next"] == True)
        & work[baseline_col].notna()
        & work["_curr"].notna()
        & work["_next"].notna()
    ]

    next_bi = (
        next_work
        .groupby([baseline_col, "_curr", "_next"], observed=True)[count_col]
        .sum()
        .unstack(["_curr", "_next"], fill_value=0)
    )

    out = out.reindex(out.index.union(next_bi.index))

    def get_next(curr, next_):
        key = (curr, next_)
        if key in next_bi.columns:
            return next_bi[key].reindex(out.index, fill_value=0).astype(float)
        return pd.Series(0.0, index=out.index)

    out["baseline_curr_T_next_T"] = get_next(True, True)
    out["baseline_curr_T_next_F"] = get_next(True, False)
    out["baseline_curr_F_next_T"] = get_next(False, True)
    out["baseline_curr_F_next_F"] = get_next(False, False)

    out["baseline_curr_T_n"] = out["baseline_curr_T_next_T"] + out["baseline_curr_T_next_F"]
    out["baseline_curr_F_n"] = out["baseline_curr_F_next_T"] + out["baseline_curr_F_next_F"]

    out["baseline_P_T_given_curr_T"] = out["baseline_curr_T_next_T"] / out["baseline_curr_T_n"].replace(0, np.nan)
    out["baseline_P_F_given_curr_T"] = out["baseline_curr_T_next_F"] / out["baseline_curr_T_n"].replace(0, np.nan)

    out["baseline_P_T_given_curr_F"] = out["baseline_curr_F_next_T"] / out["baseline_curr_F_n"].replace(0, np.nan)
    out["baseline_P_F_given_curr_F"] = out["baseline_curr_F_next_F"] / out["baseline_curr_F_n"].replace(0, np.nan)

    return out.reset_index()

# =========================================================
# 2. unit별 3연쇄 분석 함수
# =========================================================

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
    smoothing: float = 0.5,
) -> pd.DataFrame:

    work = df.copy()

    work["_prev"] = work[prev_state_col].map(_normalize_binary_state)
    work["_curr"] = work[state_col].map(_normalize_binary_state)
    work["_next"] = work[next_state_col].map(_normalize_binary_state)
    work["_has_prev"] = work[has_prev_col].map(_normalize_binary_state)
    work["_has_next"] = work[has_next_col].map(_normalize_binary_state)

    if baseline_df is None:
        baseline_df = analyze_trigram_baseline(
            df,
            baseline_col=baseline_col,
            state_col=state_col,
            prev_state_col=prev_state_col,
            next_state_col=next_state_col,
            has_prev_col=has_prev_col,
            has_next_col=has_next_col,
            count_col=count_col,
        )

    # =====================================================
    # 1) unit별 현재문장 P_T
    # =====================================================
    curr_work = work[
        work[baseline_col].notna()
        & work[unit_col].notna()
        & work["_curr"].notna()
    ]

    curr = (
        curr_work
        .groupby([baseline_col, unit_col, "_curr"], observed=True)[count_col]
        .sum()
        .unstack("_curr", fill_value=0)
    )

    out = pd.DataFrame(index=curr.index)

    out["unit_n_T"] = curr[True] if True in curr.columns else 0
    out["unit_n_F"] = curr[False] if False in curr.columns else 0
    out["unit_n"] = out["unit_n_T"] + out["unit_n_F"]

    out["unit_P_T"] = out["unit_n_T"] / out["unit_n"].replace(0, np.nan)
    out["unit_P_F"] = out["unit_n_F"] / out["unit_n"].replace(0, np.nan)

    # =====================================================
    # 2) unit별 prev-curr 2연쇄
    # =====================================================
    bi_work = work[
        (work["_has_prev"] == True)
        & work[baseline_col].notna()
        & work[unit_col].notna()
        & work["_prev"].notna()
        & work["_curr"].notna()
    ]

    bi = (
        bi_work
        .groupby([baseline_col, unit_col, "_prev", "_curr"], observed=True)[count_col]
        .sum()
        .unstack(["_prev", "_curr"], fill_value=0)
    )

    out = out.reindex(out.index.union(bi.index))

    def get_bi(prev, curr):
        key = (prev, curr)
        if key in bi.columns:
            return bi[key].reindex(out.index, fill_value=0).astype(float)
        return pd.Series(0.0, index=out.index)

    out["TT"] = get_bi(True, True)
    out["TF"] = get_bi(True, False)
    out["FT"] = get_bi(False, True)
    out["FF"] = get_bi(False, False)

    out["n_bigrams"] = out[["TT", "TF", "FT", "FF"]].sum(axis=1)

    for col in ["TT", "TF", "FT", "FF"]:
        out[f"Obs_{col}_rate"] = out[col] / out["n_bigrams"].replace(0, np.nan)

    out["unit_prev_T_n"] = out["TT"] + out["TF"]
    out["unit_prev_F_n"] = out["FT"] + out["FF"]

    out["unit_prev_P_T"] = out["unit_prev_T_n"] / out["n_bigrams"].replace(0, np.nan)
    out["unit_prev_P_F"] = out["unit_prev_F_n"] / out["n_bigrams"].replace(0, np.nan)

    # =====================================================
    # 3) unit별 prev-curr-next 3연쇄
    # =====================================================
    tri_work = work[
        (work["_has_prev"] == True)
        & (work["_has_next"] == True)
        & work[baseline_col].notna()
        & work[unit_col].notna()
        & work["_prev"].notna()
        & work["_curr"].notna()
        & work["_next"].notna()
    ]

    tri = (
        tri_work
        .groupby([baseline_col, unit_col, "_prev", "_curr", "_next"], observed=True)[count_col]
        .sum()
        .unstack(["_prev", "_curr", "_next"], fill_value=0)
    )

    out = out.reindex(out.index.union(tri.index))

    def get_tri(prev, curr, next_):
        key = (prev, curr, next_)
        if key in tri.columns:
            return tri[key].reindex(out.index, fill_value=0).astype(float)
        return pd.Series(0.0, index=out.index)

    tri_cols = ["TTT", "TTF", "TFT", "TFF", "FTT", "FTF", "FFT", "FFF"]

    out["TTT"] = get_tri(True, True, True)
    out["TTF"] = get_tri(True, True, False)
    out["TFT"] = get_tri(True, False, True)
    out["TFF"] = get_tri(True, False, False)
    out["FTT"] = get_tri(False, True, True)
    out["FTF"] = get_tri(False, True, False)
    out["FFT"] = get_tri(False, False, True)
    out["FFF"] = get_tri(False, False, False)

    out["n_trigrams"] = out[tri_cols].sum(axis=1)

    for col in tri_cols:
        out[f"Obs_{col}_rate"] = out[col] / out["n_trigrams"].replace(0, np.nan)

    # =====================================================
    # 4) baseline 붙이기
    # =====================================================
    out = out.reset_index()

    out = out.merge(
        baseline_df,
        on=baseline_col,
        how="left",
    )

    # =====================================================
    # 5) 1연쇄 기대값 비교
    # =====================================================
    out["E_unit_P_T"] = out["baseline_P_T"]
    out["E_unit_P_F"] = out["baseline_P_F"]

    out["unit_P_T_delta_vs_baseline"] = out["unit_P_T"] - out["E_unit_P_T"]
    out["unit_P_F_delta_vs_baseline"] = out["unit_P_F"] - out["E_unit_P_F"]

    out["unit_P_T_ratio_vs_baseline"] = out["unit_P_T"] / out["E_unit_P_T"].replace(0, np.nan)
    out["unit_P_F_ratio_vs_baseline"] = out["unit_P_F"] / out["E_unit_P_F"].replace(0, np.nan)

    # =====================================================
    # 6) 2연쇄 기대값
    # 이전문장의 T/F 비율은 unit 실제값 사용
    # curr 선택 확률은 baseline 조건부 확률 사용
    # =====================================================

    out["E_TT_rate"] = out["unit_prev_P_T"] * out["baseline_P_T_given_prev_T"]
    out["E_TF_rate"] = out["unit_prev_P_T"] * out["baseline_P_F_given_prev_T"]

    out["E_FT_rate"] = out["unit_prev_P_F"] * out["baseline_P_T_given_prev_F"]
    out["E_FF_rate"] = out["unit_prev_P_F"] * out["baseline_P_F_given_prev_F"]

    for col in ["TT", "TF", "FT", "FF"]:
        out[f"{col}_delta_vs_baseline"] = out[f"Obs_{col}_rate"] - out[f"E_{col}_rate"]
        out[f"{col}_ratio_vs_baseline"] = out[f"Obs_{col}_rate"] / out[f"E_{col}_rate"].replace(0, np.nan)

    # =====================================================
    # 7) 3연쇄 기대값
    # unit의 실제 TT/TF/FT/FF 비율을 보존하고
    # curr -> next 확률은 baseline 조건부 확률 사용
    # =====================================================

    out["E_TTT_rate"] = out["Obs_TT_rate"] * out["baseline_P_T_given_curr_T"]
    out["E_TTF_rate"] = out["Obs_TT_rate"] * out["baseline_P_F_given_curr_T"]

    out["E_TFT_rate"] = out["Obs_TF_rate"] * out["baseline_P_T_given_curr_F"]
    out["E_TFF_rate"] = out["Obs_TF_rate"] * out["baseline_P_F_given_curr_F"]

    out["E_FTT_rate"] = out["Obs_FT_rate"] * out["baseline_P_T_given_curr_T"]
    out["E_FTF_rate"] = out["Obs_FT_rate"] * out["baseline_P_F_given_curr_T"]

    out["E_FFT_rate"] = out["Obs_FF_rate"] * out["baseline_P_T_given_curr_F"]
    out["E_FFF_rate"] = out["Obs_FF_rate"] * out["baseline_P_F_given_curr_F"]

    for col in tri_cols:
        out[f"{col}_delta_vs_baseline"] = out[f"Obs_{col}_rate"] - out[f"E_{col}_rate"]
        out[f"{col}_ratio_vs_baseline"] = out[f"Obs_{col}_rate"] / out[f"E_{col}_rate"].replace(0, np.nan)

    # =====================================================
    # 8) 해석용 묶음 지표
    # =====================================================

    out["switch_return_Obs_rate"] = out["Obs_TFT_rate"] + out["Obs_FTF_rate"]
    out["switch_return_E_rate"] = out["E_TFT_rate"] + out["E_FTF_rate"]
    out["switch_return_delta_vs_baseline"] = out["switch_return_Obs_rate"] - out["switch_return_E_rate"]
    out["switch_return_ratio_vs_baseline"] = out["switch_return_Obs_rate"] / out["switch_return_E_rate"].replace(0, np.nan)

    out["switch_extension_Obs_rate"] = out["Obs_TFF_rate"] + out["Obs_FTT_rate"]
    out["switch_extension_E_rate"] = out["E_TFF_rate"] + out["E_FTT_rate"]
    out["switch_extension_delta_vs_baseline"] = out["switch_extension_Obs_rate"] - out["switch_extension_E_rate"]
    out["switch_extension_ratio_vs_baseline"] = out["switch_extension_Obs_rate"] / out["switch_extension_E_rate"].replace(0, np.nan)

    out["stay_stay_Obs_rate"] = out["Obs_TTT_rate"] + out["Obs_FFF_rate"]
    out["stay_stay_E_rate"] = out["E_TTT_rate"] + out["E_FFF_rate"]
    out["stay_stay_delta_vs_baseline"] = out["stay_stay_Obs_rate"] - out["stay_stay_E_rate"]
    out["stay_stay_ratio_vs_baseline"] = out["stay_stay_Obs_rate"] / out["stay_stay_E_rate"].replace(0, np.nan)

    out["stay_switch_Obs_rate"] = out["Obs_TTF_rate"] + out["Obs_FFT_rate"]
    out["stay_switch_E_rate"] = out["E_TTF_rate"] + out["E_FFT_rate"]
    out["stay_switch_delta_vs_baseline"] = out["stay_switch_Obs_rate"] - out["stay_switch_E_rate"]
    out["stay_switch_ratio_vs_baseline"] = out["stay_switch_Obs_rate"] / out["stay_switch_E_rate"].replace(0, np.nan)

    # 방향별 복귀율
    out["return_after_TF_Obs_rate"] = out["TFT"] / (out["TFT"] + out["TFF"]).replace(0, np.nan)
    out["return_after_TF_E_rate"] = out["baseline_P_T_given_curr_F"]
    out["return_after_TF_delta_vs_baseline"] = out["return_after_TF_Obs_rate"] - out["return_after_TF_E_rate"]

    out["return_after_FT_Obs_rate"] = out["FTF"] / (out["FTT"] + out["FTF"]).replace(0, np.nan)
    out["return_after_FT_E_rate"] = out["baseline_P_F_given_curr_T"]
    out["return_after_FT_delta_vs_baseline"] = out["return_after_FT_Obs_rate"] - out["return_after_FT_E_rate"]

    return add_transition_odds_columns(out, smoothing=smoothing)
