from __future__ import annotations
from typing import List, Dict, Optional
import pandas as pd
import numpy as np

#================

DEFAULT_VLABEL_MAP: Dict[str, str] = {
    "VV": "VV", "VA": "VA", "VCN": "VA",  "VCP": "VP",
    "XSV": "VV", "XSA": "VA", "VX": "VX",
}

def v_affix_attach(
    df: pd.DataFrame,
    *,
    v_form_col: str = "V_form",
    n_form_col: str = "N_form",
    v_label_col: str = "V_label",
    v_form0_col: str = "V_form_0",
    v_label0_col: str = "V_label_0",
    conditions_prefix: tuple[str, ...] = ("XSV", "XSA"),
    vlabel_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    - V_form_0 생성
      * 기본: V_form
      * XSV/XSA: N_form을 '+' 기준으로 분리 → 마지막 요소만 사용
    - V_label_0 생성 (map 기반)
    - 저장하지 않고 DataFrame 반환
    """

    if vlabel_map is None:
        vlabel_map = DEFAULT_VLABEL_MAP

    required = {v_form_col, n_form_col, v_label_col}
    missing = required - set(df.columns)

    if missing:
        raise KeyError(f"Missing required columns: {missing}")    
    else:
        dfx = df.copy()

        # --- 1. V_form_0 기본값 ---
        dfx[v_form0_col] = dfx[v_form_col].astype("string").fillna("")

        # --- 2. XSV / XSA 처리 ---
        mask = dfx[v_label_col].isin(conditions_prefix)

        if mask.any():
            # N_form: '+' 있으면 뒤만, 없으면 전체
            n_processed = (
                dfx.loc[mask, n_form_col]
                .astype("string")
                .fillna("")
                .str.split("+")
                .str[-1]
            )

            v_part = (
                dfx.loc[mask, v_form_col]
                .astype("string")
                .fillna("")
            )

            dfx.loc[mask, v_form0_col] = n_processed + v_part

        # --- 3. V_label_0 ---
        dfx[v_label0_col] = dfx[v_label_col].map(vlabel_map)

    return dfx





def add_v_no_by_merge_and_fallback(
    df: pd.DataFrame,
    v_no_csv_path: str,
    *,
    label0_col: str = "V_label_0",
    form0_col: str = "V_form_0",
    vno_col: str = "V_No",
    encoding: str = "utf-8-sig", #보통 Windows 'ansi' = cp949인 경우가 많음
) -> pd.DataFrame:
    """
    1) V_No 테이블(000.V_No.csv)을 읽어서
    2) (V_label_0, V_form_0) 기준 left merge로 V_No 붙이고
    3) 매칭 실패(V_No NaN)면 규칙으로 채움:
       - V_label_0 비어있음/NaN -> -1
       - V_label_0 == "VV" -> 8999
       - V_label_0 == "VA" -> 3999
       - V_label_0 startswith "VC" -> 1009
       - V_label_0 == "VX" -> 999
       - 그외 라벨 -> 9999
    """

    required = {label0_col, form0_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"[add_v_no_by_merge_and_fallback] missing columns: {missing}")

    # V_No 테이블 로드
    df_v_no = pd.read_csv(v_no_csv_path, encoding=encoding, low_memory=False)

    required_ref = {label0_col, form0_col, vno_col}
    missing_ref = required_ref - set(df_v_no.columns)
    if missing_ref:
        raise KeyError(f"[add_v_no_by_merge_and_fallback] V_No file missing columns: {missing_ref}")

    # 필요한 열만
    df_v_no_selected = df_v_no[[label0_col, form0_col, vno_col]].drop_duplicates(
        subset=[label0_col, form0_col], keep="last"
    )

    # 원본 복사 후 merge
    out = df.copy()

    # 기존 V_No가 있으면 일단 drop하고 다시 붙이는 게 안전(중복 컬럼 방지)
    if vno_col in out.columns:
        out = out.drop(columns=[vno_col])

    out = out.merge(df_v_no_selected, on=[label0_col, form0_col], how="left")

    # ---- 매칭 실패 fallback 규칙 적용 ----
    miss = out[vno_col].isna()
    if miss.any():
        lab = out.loc[miss, label0_col]

        # 기본값 9999
        fill = np.full(miss.sum(), 9999, dtype="int64")

        # V_label_0 비어있는 경우 -> -1 (최우선)
        is_empty = lab.isna() | (lab.astype("string").str.len() == 0)
        fill[is_empty.to_numpy()] = -1

        # 나머지(비어있지 않은 것)만 규칙 적용
        lab2 = lab.astype("string")

        vv = (lab2 == "VV") & (~is_empty)
        va = (lab2 == "VA") & (~is_empty)
        vc = lab2.str.startswith("VC", na=False) & (~is_empty)
        vx = (lab2 == "VX") & (~is_empty)

        fill[vv.to_numpy()] = 8999
        fill[va.to_numpy()] = 3999
        fill[vc.to_numpy()] = 1009
        fill[vx.to_numpy()] = 999

        out.loc[miss, vno_col] = fill

    # 정수형으로 (NaN 없이 채웠으니 가능)
    out[vno_col] = pd.to_numeric(out[vno_col], errors="coerce").fillna(-1).astype("Int64")

    return out
