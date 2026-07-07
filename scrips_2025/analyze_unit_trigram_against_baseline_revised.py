# baseline 대비 unit 1연쇄 / 2연쇄 / 3연쇄 이탈 분석 함수
# ---------------------------------------------------------
# 목적:
#   baseline_col별 기본 전이패턴을 먼저 구한 뒤,
#   각 unit_col이 그 baseline 패턴에서 얼마나 벗어나는지 계산한다.
#
# 핵심 기준:
#   1연쇄 baseline: P(curr=T/F)
#   2연쇄 baseline: P(curr=T/F | prev=T/F)
#   3연쇄 baseline: P(next=T/F | prev-curr=TT/TF/FT/FF)
#
# T = '-었-' 있음
# F = '-었-' 없음

import numpy as np
import pandas as pd


def _normalize_binary_state(x):
    """T/F, True/False, 1/0 값을 True/False로 통일한다."""
    if pd.isna(x):
        return np.nan
    if x in [True, "True", "TRUE", "T", "t", 1, "1"]:
        return True
    if x in [False, "False", "FALSE", "F", "f", 0, "0"]:
        return False
    return np.nan


def _rate(num, den):
    """0으로 나누는 경우 NaN을 반환하는 비율 계산."""
    if hasattr(den, "replace"):
        return num / den.replace(0, np.nan)
    return num / (np.nan if den == 0 else den)


def _ratio(obs, exp):
    """기대값이 0인 경우 NaN을 반환하는 관찰값/기대값 비율."""
    if hasattr(exp, "replace"):
        return obs / exp.replace(0, np.nan)
    return obs / (np.nan if exp == 0 else exp)


def _series_from_table(table: pd.DataFrame, key, index) -> pd.Series:
    """unstack 결과에서 특정 column key를 꺼내고, 없으면 0 Series를 반환한다."""
    if key in table.columns:
        return table[key].reindex(index, fill_value=0).astype(float)
    return pd.Series(0.0, index=index)


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
    """
    baseline_col별 1연쇄, 2연쇄, 3연쇄 baseline을 계산한다.

    반환되는 주요 baseline:

    1) 1연쇄
        baseline_P_T
        baseline_P_F

    2) 2연쇄: 직전 상태 이후 현재 상태 확률
        baseline_P_curr_T_given_prev_T
        baseline_P_curr_F_given_prev_T
        baseline_P_curr_T_given_prev_F
        baseline_P_curr_F_given_prev_F

    3) 3연쇄: 이전 두 상태 이후 다음 상태 확률
        baseline_P_next_T_given_TT
        baseline_P_next_F_given_TT
        baseline_P_next_T_given_TF
        baseline_P_next_F_given_TF
        baseline_P_next_T_given_FT
        baseline_P_next_F_given_FT
        baseline_P_next_T_given_FF
        baseline_P_next_F_given_FF
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
    # 1) 1연쇄 baseline: 현재문장 기준 P_T / P_F
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
        if True in curr_counts.columns
        else pd.Series(0.0, index=out.index)
    )
    out["baseline_n_F"] = (
        curr_counts[False].astype(float)
        if False in curr_counts.columns
        else pd.Series(0.0, index=out.index)
    )
    out["baseline_n"] = out["baseline_n_T"] + out["baseline_n_F"]

    out["baseline_P_T"] = _rate(out["baseline_n_T"], out["baseline_n"])
    out["baseline_P_F"] = _rate(out["baseline_n_F"], out["baseline_n"])

    # -------------------------------------------------
    # 2) 2연쇄 baseline: P(curr | prev)
    # -------------------------------------------------
    bi_work = work[
        (work["_has_prev"] == True)
        & work[baseline_col].notna()
        & work["_prev"].notna()
        & work["_curr"].notna()
    ]

    bi_counts = (
        bi_work
        .groupby([baseline_col, "_prev", "_curr"], observed=True)[count_col]
        .sum()
        .unstack(["_prev", "_curr"], fill_value=0)
    )

    out = out.reindex(out.index.union(bi_counts.index))

    def get_bi(prev, curr):
        return _series_from_table(bi_counts, (prev, curr), out.index)

    out["baseline_TT"] = get_bi(True, True)
    out["baseline_TF"] = get_bi(True, False)
    out["baseline_FT"] = get_bi(False, True)
    out["baseline_FF"] = get_bi(False, False)

    out["baseline_n_bigrams"] = (
        out["baseline_TT"] + out["baseline_TF"]
        + out["baseline_FT"] + out["baseline_FF"]
    )

    out["baseline_prev_T_n"] = out["baseline_TT"] + out["baseline_TF"]
    out["baseline_prev_F_n"] = out["baseline_FT"] + out["baseline_FF"]

    out["baseline_P_curr_T_given_prev_T"] = _rate(
        out["baseline_TT"], out["baseline_prev_T_n"]
    )
    out["baseline_P_curr_F_given_prev_T"] = _rate(
        out["baseline_TF"], out["baseline_prev_T_n"]
    )
    out["baseline_P_curr_T_given_prev_F"] = _rate(
        out["baseline_FT"], out["baseline_prev_F_n"]
    )
    out["baseline_P_curr_F_given_prev_F"] = _rate(
        out["baseline_FF"], out["baseline_prev_F_n"]
    )

    # -------------------------------------------------
    # 3) 3연쇄 baseline: P(next | prev, curr)
    #    실제 3연쇄 분석과 같은 조건만 사용한다.
    # -------------------------------------------------
    tri_work = work[
        (work["_has_prev"] == True)
        & (work["_has_next"] == True)
        & work[baseline_col].notna()
        & work["_prev"].notna()
        & work["_curr"].notna()
        & work["_next"].notna()
    ]

    tri_counts = (
        tri_work
        .groupby([baseline_col, "_prev", "_curr", "_next"], observed=True)[count_col]
        .sum()
        .unstack(["_prev", "_curr", "_next"], fill_value=0)
    )

    out = out.reindex(out.index.union(tri_counts.index))

    def get_tri(prev, curr, next_):
        return _series_from_table(tri_counts, (prev, curr, next_), out.index)

    out["baseline_TTT"] = get_tri(True, True, True)
    out["baseline_TTF"] = get_tri(True, True, False)
    out["baseline_TFT"] = get_tri(True, False, True)
    out["baseline_TFF"] = get_tri(True, False, False)
    out["baseline_FTT"] = get_tri(False, True, True)
    out["baseline_FTF"] = get_tri(False, True, False)
    out["baseline_FFT"] = get_tri(False, False, True)
    out["baseline_FFF"] = get_tri(False, False, False)

    out["baseline_n_trigrams"] = (
        out["baseline_TTT"] + out["baseline_TTF"]
        + out["baseline_TFT"] + out["baseline_TFF"]
        + out["baseline_FTT"] + out["baseline_FTF"]
        + out["baseline_FFT"] + out["baseline_FFF"]
    )

    out["baseline_tri_TT_n"] = out["baseline_TTT"] + out["baseline_TTF"]
    out["baseline_tri_TF_n"] = out["baseline_TFT"] + out["baseline_TFF"]
    out["baseline_tri_FT_n"] = out["baseline_FTT"] + out["baseline_FTF"]
    out["baseline_tri_FF_n"] = out["baseline_FFT"] + out["baseline_FFF"]

    out["baseline_P_next_T_given_TT"] = _rate(
        out["baseline_TTT"], out["baseline_tri_TT_n"]
    )
    out["baseline_P_next_F_given_TT"] = _rate(
        out["baseline_TTF"], out["baseline_tri_TT_n"]
    )
    out["baseline_P_next_T_given_TF"] = _rate(
        out["baseline_TFT"], out["baseline_tri_TF_n"]
    )
    out["baseline_P_next_F_given_TF"] = _rate(
        out["baseline_TFF"], out["baseline_tri_TF_n"]
    )
    out["baseline_P_next_T_given_FT"] = _rate(
        out["baseline_FTT"], out["baseline_tri_FT_n"]
    )
    out["baseline_P_next_F_given_FT"] = _rate(
        out["baseline_FTF"], out["baseline_tri_FT_n"]
    )
    out["baseline_P_next_T_given_FF"] = _rate(
        out["baseline_FFT"], out["baseline_tri_FF_n"]
    )
    out["baseline_P_next_F_given_FF"] = _rate(
        out["baseline_FFF"], out["baseline_tri_FF_n"]
    )

    return out.reset_index()


# =========================================================
# 2. unit별 baseline 대비 이탈 분석 함수
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
    min_unit_n: int = 0,
    min_bigram_n: int = 0,
    min_trigram_n: int = 0,
) -> pd.DataFrame:
    """
    unit별 1연쇄 / 2연쇄 / 3연쇄가 baseline 전이패턴에서
    얼마나 벗어나는지 계산한다.

    ------------------------------------------------------
    1연쇄 기대값
    ------------------------------------------------------
        E_unit_P_T = baseline_P_T
        E_unit_P_F = baseline_P_F

    ------------------------------------------------------
    2연쇄 기대값
    ------------------------------------------------------
    unit의 실제 prev 분포는 보존하고,
    curr 선택은 baseline의 P(curr | prev)를 따른다고 가정한다.

        E_TT = unit_prev_P_T × baseline_P_curr_T_given_prev_T
        E_TF = unit_prev_P_T × baseline_P_curr_F_given_prev_T
        E_FT = unit_prev_P_F × baseline_P_curr_T_given_prev_F
        E_FF = unit_prev_P_F × baseline_P_curr_F_given_prev_F

    ------------------------------------------------------
    3연쇄 기대값
    ------------------------------------------------------
    unit의 실제 prev-curr 분포는 보존하고,
    next 선택은 baseline의 P(next | prev, curr)를 따른다고 가정한다.

        E_TTT = Obs_tri_TT × baseline_P_next_T_given_TT
        E_TTF = Obs_tri_TT × baseline_P_next_F_given_TT
        E_TFT = Obs_tri_TF × baseline_P_next_T_given_TF
        E_TFF = Obs_tri_TF × baseline_P_next_F_given_TF
        E_FTT = Obs_tri_FT × baseline_P_next_T_given_FT
        E_FTF = Obs_tri_FT × baseline_P_next_F_given_FT
        E_FFT = Obs_tri_FF × baseline_P_next_T_given_FF
        E_FFF = Obs_tri_FF × baseline_P_next_F_given_FF
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
    # 1) unit별 현재문장 T/F 분포
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
        if True in curr_counts.columns
        else pd.Series(0.0, index=out.index)
    )
    out["unit_n_F"] = (
        curr_counts[False].astype(float)
        if False in curr_counts.columns
        else pd.Series(0.0, index=out.index)
    )
    out["unit_n"] = out["unit_n_T"] + out["unit_n_F"]

    out["unit_P_T"] = _rate(out["unit_n_T"], out["unit_n"])
    out["unit_P_F"] = _rate(out["unit_n_F"], out["unit_n"])

    # =====================================================
    # 2) unit별 2연쇄 실제값: prev-curr
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
        return _series_from_table(bi_counts, (prev, curr), out.index)

    out["TT"] = get_bi(True, True)
    out["TF"] = get_bi(True, False)
    out["FT"] = get_bi(False, True)
    out["FF"] = get_bi(False, False)

    bigram_cols = ["TT", "TF", "FT", "FF"]
    out["n_bigrams"] = out[bigram_cols].sum(axis=1)

    for col in bigram_cols:
        out[f"Obs_{col}_rate"] = _rate(out[col], out["n_bigrams"])

    out["unit_prev_T_n"] = out["TT"] + out["TF"]
    out["unit_prev_F_n"] = out["FT"] + out["FF"]
    out["unit_prev_n"] = out["unit_prev_T_n"] + out["unit_prev_F_n"]

    out["unit_prev_P_T"] = _rate(out["unit_prev_T_n"], out["unit_prev_n"])
    out["unit_prev_P_F"] = _rate(out["unit_prev_F_n"], out["unit_prev_n"])

    out["P_curr_T_given_prev_T"] = _rate(out["TT"], out["TT"] + out["TF"])
    out["P_curr_F_given_prev_T"] = _rate(out["TF"], out["TT"] + out["TF"])
    out["P_curr_T_given_prev_F"] = _rate(out["FT"], out["FT"] + out["FF"])
    out["P_curr_F_given_prev_F"] = _rate(out["FF"], out["FT"] + out["FF"])

    # =====================================================
    # 3) unit별 3연쇄 실제값: prev-curr-next
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
        return _series_from_table(tri_counts, (prev, curr, next_), out.index)

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

    # 3연쇄 가능한 표본 안에서의 실제 prev-curr 분포
    out["tri_TT"] = out["TTT"] + out["TTF"]
    out["tri_TF"] = out["TFT"] + out["TFF"]
    out["tri_FT"] = out["FTT"] + out["FTF"]
    out["tri_FF"] = out["FFT"] + out["FFF"]

    out["Obs_tri_TT_rate"] = _rate(out["tri_TT"], out["n_trigrams"])
    out["Obs_tri_TF_rate"] = _rate(out["tri_TF"], out["n_trigrams"])
    out["Obs_tri_FT_rate"] = _rate(out["tri_FT"], out["n_trigrams"])
    out["Obs_tri_FF_rate"] = _rate(out["tri_FF"], out["n_trigrams"])

    # unit 내부 조건부 next 확률
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
    # 5) 1연쇄: unit T/F 비율이 baseline에서 얼마나 벗어나는가
    # =====================================================
    out["E_unit_P_T"] = out["baseline_P_T"]
    out["E_unit_P_F"] = out["baseline_P_F"]

    out["unit_P_T_delta_vs_baseline"] = out["unit_P_T"] - out["E_unit_P_T"]
    out["unit_P_F_delta_vs_baseline"] = out["unit_P_F"] - out["E_unit_P_F"]

    out["unit_P_T_ratio_vs_baseline"] = _ratio(out["unit_P_T"], out["E_unit_P_T"])
    out["unit_P_F_ratio_vs_baseline"] = _ratio(out["unit_P_F"], out["E_unit_P_F"])

    # =====================================================
    # 6) 2연쇄 기대값: unit prev 분포 × baseline P(curr | prev)
    # =====================================================
    out["E_TT_rate"] = out["unit_prev_P_T"] * out["baseline_P_curr_T_given_prev_T"]
    out["E_TF_rate"] = out["unit_prev_P_T"] * out["baseline_P_curr_F_given_prev_T"]
    out["E_FT_rate"] = out["unit_prev_P_F"] * out["baseline_P_curr_T_given_prev_F"]
    out["E_FF_rate"] = out["unit_prev_P_F"] * out["baseline_P_curr_F_given_prev_F"]

    for col in bigram_cols:
        out[f"E_{col}_n"] = out["n_bigrams"] * out[f"E_{col}_rate"]
        out[f"{col}_delta_vs_baseline"] = out[f"Obs_{col}_rate"] - out[f"E_{col}_rate"]
        out[f"{col}_ratio_vs_baseline"] = _ratio(out[f"Obs_{col}_rate"], out[f"E_{col}_rate"])

    # 2연쇄 묶음 지표
    out["stay_Obs_rate"] = out["Obs_TT_rate"] + out["Obs_FF_rate"]
    out["stay_E_rate"] = out["E_TT_rate"] + out["E_FF_rate"]
    out["stay_delta_vs_baseline"] = out["stay_Obs_rate"] - out["stay_E_rate"]
    out["stay_ratio_vs_baseline"] = _ratio(out["stay_Obs_rate"], out["stay_E_rate"])

    out["switch_Obs_rate"] = out["Obs_TF_rate"] + out["Obs_FT_rate"]
    out["switch_E_rate"] = out["E_TF_rate"] + out["E_FT_rate"]
    out["switch_delta_vs_baseline"] = out["switch_Obs_rate"] - out["switch_E_rate"]
    out["switch_ratio_vs_baseline"] = _ratio(out["switch_Obs_rate"], out["switch_E_rate"])

    # 2연쇄 방향별 이탈
    out["prev_T_to_curr_T_delta_vs_baseline"] = (
        out["P_curr_T_given_prev_T"] - out["baseline_P_curr_T_given_prev_T"]
    )
    out["prev_T_to_curr_F_delta_vs_baseline"] = (
        out["P_curr_F_given_prev_T"] - out["baseline_P_curr_F_given_prev_T"]
    )
    out["prev_F_to_curr_T_delta_vs_baseline"] = (
        out["P_curr_T_given_prev_F"] - out["baseline_P_curr_T_given_prev_F"]
    )
    out["prev_F_to_curr_F_delta_vs_baseline"] = (
        out["P_curr_F_given_prev_F"] - out["baseline_P_curr_F_given_prev_F"]
    )

    out["prev_chain_gap_on_curr_T"] = (
        out["P_curr_T_given_prev_T"] - out["P_curr_T_given_prev_F"]
    )
    out["baseline_prev_chain_gap_on_curr_T"] = (
        out["baseline_P_curr_T_given_prev_T"]
        - out["baseline_P_curr_T_given_prev_F"]
    )
    out["prev_chain_gap_on_curr_T_delta_vs_baseline"] = (
        out["prev_chain_gap_on_curr_T"]
        - out["baseline_prev_chain_gap_on_curr_T"]
    )

    # =====================================================
    # 7) 3연쇄 기대값: unit prev-curr 분포 × baseline P(next | prev, curr)
    # =====================================================
    out["E_TTT_rate"] = out["Obs_tri_TT_rate"] * out["baseline_P_next_T_given_TT"]
    out["E_TTF_rate"] = out["Obs_tri_TT_rate"] * out["baseline_P_next_F_given_TT"]

    out["E_TFT_rate"] = out["Obs_tri_TF_rate"] * out["baseline_P_next_T_given_TF"]
    out["E_TFF_rate"] = out["Obs_tri_TF_rate"] * out["baseline_P_next_F_given_TF"]

    out["E_FTT_rate"] = out["Obs_tri_FT_rate"] * out["baseline_P_next_T_given_FT"]
    out["E_FTF_rate"] = out["Obs_tri_FT_rate"] * out["baseline_P_next_F_given_FT"]

    out["E_FFT_rate"] = out["Obs_tri_FF_rate"] * out["baseline_P_next_T_given_FF"]
    out["E_FFF_rate"] = out["Obs_tri_FF_rate"] * out["baseline_P_next_F_given_FF"]

    for col in tri_cols:
        out[f"E_{col}_n"] = out["n_trigrams"] * out[f"E_{col}_rate"]
        out[f"{col}_delta_vs_baseline"] = out[f"Obs_{col}_rate"] - out[f"E_{col}_rate"]
        out[f"{col}_ratio_vs_baseline"] = _ratio(out[f"Obs_{col}_rate"], out[f"E_{col}_rate"])

    # 3연쇄 방향별 이탈
    pair_labels = ["TT", "TF", "FT", "FF"]
    for pair in pair_labels:
        out[f"next_T_after_{pair}_delta_vs_baseline"] = (
            out[f"P_next_T_given_{pair}"]
            - out[f"baseline_P_next_T_given_{pair}"]
        )
        out[f"next_F_after_{pair}_delta_vs_baseline"] = (
            out[f"P_next_F_given_{pair}"]
            - out[f"baseline_P_next_F_given_{pair}"]
        )
        out[f"next_T_after_{pair}_ratio_vs_baseline"] = _ratio(
            out[f"P_next_T_given_{pair}"],
            out[f"baseline_P_next_T_given_{pair}"]
        )
        out[f"next_F_after_{pair}_ratio_vs_baseline"] = _ratio(
            out[f"P_next_F_given_{pair}"],
            out[f"baseline_P_next_F_given_{pair}"]
        )

    # =====================================================
    # 8) 3연쇄 묶음 지표
    # =====================================================
    # 전환 후 복귀: TFT + FTF
    out["switch_return_Obs_rate"] = out["Obs_TFT_rate"] + out["Obs_FTF_rate"]
    out["switch_return_E_rate"] = out["E_TFT_rate"] + out["E_FTF_rate"]
    out["switch_return_delta_vs_baseline"] = (
        out["switch_return_Obs_rate"] - out["switch_return_E_rate"]
    )
    out["switch_return_ratio_vs_baseline"] = _ratio(
        out["switch_return_Obs_rate"], out["switch_return_E_rate"]
    )

    # 전환 후 지속: TFF + FTT
    out["switch_extension_Obs_rate"] = out["Obs_TFF_rate"] + out["Obs_FTT_rate"]
    out["switch_extension_E_rate"] = out["E_TFF_rate"] + out["E_FTT_rate"]
    out["switch_extension_delta_vs_baseline"] = (
        out["switch_extension_Obs_rate"] - out["switch_extension_E_rate"]
    )
    out["switch_extension_ratio_vs_baseline"] = _ratio(
        out["switch_extension_Obs_rate"], out["switch_extension_E_rate"]
    )

    # 유지 후 유지: TTT + FFF
    out["stay_stay_Obs_rate"] = out["Obs_TTT_rate"] + out["Obs_FFF_rate"]
    out["stay_stay_E_rate"] = out["E_TTT_rate"] + out["E_FFF_rate"]
    out["stay_stay_delta_vs_baseline"] = (
        out["stay_stay_Obs_rate"] - out["stay_stay_E_rate"]
    )
    out["stay_stay_ratio_vs_baseline"] = _ratio(
        out["stay_stay_Obs_rate"], out["stay_stay_E_rate"]
    )

    # 유지 후 전환: TTF + FFT
    out["stay_switch_Obs_rate"] = out["Obs_TTF_rate"] + out["Obs_FFT_rate"]
    out["stay_switch_E_rate"] = out["E_TTF_rate"] + out["E_FFT_rate"]
    out["stay_switch_delta_vs_baseline"] = (
        out["stay_switch_Obs_rate"] - out["stay_switch_E_rate"]
    )
    out["stay_switch_ratio_vs_baseline"] = _ratio(
        out["stay_switch_Obs_rate"], out["stay_switch_E_rate"]
    )

    # 방향별 복귀율 / 지속율
    out["return_after_TF_Obs_rate"] = out["P_next_T_given_TF"]
    out["return_after_TF_E_rate"] = out["baseline_P_next_T_given_TF"]
    out["return_after_TF_delta_vs_baseline"] = (
        out["return_after_TF_Obs_rate"] - out["return_after_TF_E_rate"]
    )
    out["return_after_TF_ratio_vs_baseline"] = _ratio(
        out["return_after_TF_Obs_rate"], out["return_after_TF_E_rate"]
    )

    out["return_after_FT_Obs_rate"] = out["P_next_F_given_FT"]
    out["return_after_FT_E_rate"] = out["baseline_P_next_F_given_FT"]
    out["return_after_FT_delta_vs_baseline"] = (
        out["return_after_FT_Obs_rate"] - out["return_after_FT_E_rate"]
    )
    out["return_after_FT_ratio_vs_baseline"] = _ratio(
        out["return_after_FT_Obs_rate"], out["return_after_FT_E_rate"]
    )

    out["extension_after_TF_Obs_rate"] = out["P_next_F_given_TF"]
    out["extension_after_TF_E_rate"] = out["baseline_P_next_F_given_TF"]
    out["extension_after_TF_delta_vs_baseline"] = (
        out["extension_after_TF_Obs_rate"] - out["extension_after_TF_E_rate"]
    )
    out["extension_after_TF_ratio_vs_baseline"] = _ratio(
        out["extension_after_TF_Obs_rate"], out["extension_after_TF_E_rate"]
    )

    out["extension_after_FT_Obs_rate"] = out["P_next_T_given_FT"]
    out["extension_after_FT_E_rate"] = out["baseline_P_next_T_given_FT"]
    out["extension_after_FT_delta_vs_baseline"] = (
        out["extension_after_FT_Obs_rate"] - out["extension_after_FT_E_rate"]
    )
    out["extension_after_FT_ratio_vs_baseline"] = _ratio(
        out["extension_after_FT_Obs_rate"], out["extension_after_FT_E_rate"]
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

    return out.reset_index(drop=True)
