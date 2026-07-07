# en_no_fix.py
# 작성일 2026.01.11
# EN_label, EN_form을 이용하여, label파일과 대조해서 EN_No가 빈 곳을 채워주는 프로그램. 

# ⚠️ WARNING
# 이 스크립트는 label 파일의 번호 컬럼(no_col) 값을
# 원본 DataFrame(df)의 동일 컬럼에 덮어써서 보정한다.
#
# 이때,
# - label 파일의 번호 컬럼은 보통 pandas nullable integer(Int64)이고
# - df[no_col]은 기존 NaN 처리로 인해 float64인 경우가 많다.
#
# dtype이 일치하지 않은 상태에서 값을 대입하면
# pandas FutureWarning (incompatible dtype assignment)이 발생하며,
# 향후 버전에서는 오류로 바뀔 수 있다.
#
# 따라서 본 처리를 수행하기 전에
# df[no_col]을 반드시 Int64 dtype으로 명시적 변환해야 한다.
#
# 예:
#   df[no_col] = pd.to_numeric(df[no_col], errors="coerce").astype("Int64")

from __future__ import annotations

from pathlib import Path
import re
import pandas as pd
import numpy as np
import warnings

DEFAULT_BY_EN_LABEL = {"EC": 999, "EF": 1999, "ETM": 2999, "ETN": 3999}


# ===========================================
def _is_missing(s: pd.Series) -> pd.Series:
    """Missing if NA, empty string, or 'NULL' (case-insensitive)."""
    if pd.api.types.is_string_dtype(s) or s.dtype == object:
        ss = s.astype("string")
        return ss.isna() | (ss.str.strip() == "") | (ss.str.upper() == "NULL")
    return s.isna()


def floor_int64(series: pd.Series) -> pd.Series:
    """Convert to numeric, floor decimals, return nullable Int64."""
    num = pd.to_numeric(series, errors="coerce")
    flo = np.floor(num)
    return pd.Series(flo, index=series.index).astype("Int64")


def load_label_map(label_csv: str | Path, main_prefix: str = "EN") -> pd.DataFrame:
    """
    Load mapping: (EN_label, EN_form) -> EN_No (base integer)
    label_csv must contain columns: EN_label, EN_form, EN_No
    """
    #0. 컬럼 확인
    form_col = f"{main_prefix}_form"
    label_col = f"{main_prefix}_label"
    no_col = f"{main_prefix}_No"
        
    label_csv = Path(label_csv)
    lab = pd.read_csv(label_csv, low_memory=False)
    missing = {form_col, label_col, no_col} - set(lab.columns) #없는 컬럼이 있는지 확인

    if missing:
        print(f"{label_csv}에 다음의 컬럼이 없습니다.: {missing} ")
        raise ValueError(f"{label_csv}에 다음 컬럼이 없습니다: {missing}")

    # label_df 준비 (필요 컬럼만, EN_No는 정수화)
    lab = lab[[label_col, form_col, no_col]].copy()
    lab[label_col] = lab[label_col].astype("string")
    lab[form_col]  = lab[form_col].astype("string")
    lab[no_col]    = np.floor(pd.to_numeric(lab[no_col], errors="coerce")).astype("Int64")

    return lab

# ===========================================
def fix_en_number_with_merge(df: pd.DataFrame, 
                         label_csv: str | Path, 
                         main_prefix: str = "EN",
                         overwrite: bool = False) -> pd.DataFrame:
    df = df.copy()

    #0.0 컬럼 확인
    form_col = f"{main_prefix}_form"
    label_col = f"{main_prefix}_label"
    no_col = f"{main_prefix}_No"

    missing = {form_col, label_col} - set(df.columns) #없는 컬럼이 있는지 확인
    if missing:
        print(f"없는 컬럼: {missing} ")
        raise ValueError(f"DataFrame에 다음 컬럼이 없습니다: {missing}")
    
    if no_col not in df.columns: # no컬럼이 없는 경우 생성.
        df[no_col] = pd.Series(pd.NA, index=df.index, dtype="Int64")
        print(f"{no_col}을 생성했습니다.")

    if df[no_col].dtype != "Int64":
        warnings.warn(
            f"[dtype mismatch warning] Column '{no_col}' is {df[no_col].dtype}, "
            f"but label values are expected to be Int64. "
            f"Cast df['{no_col}'] to Int64 before assignment to avoid FutureWarning.",
            FutureWarning,
            stacklevel=2,
        )

    #0.1 target 지정
    target = (
        _is_missing(df[no_col])
        if not overwrite #덮어쓰기 금지
        else pd.Series(True, index=df.index) #덮어씀.
    )

    # 1) label 없는 행 no_col: -1   
    df.loc[target & _is_missing(df[label_col]), no_col] = -1

    # 2) "_form 있는데 _No 없는 행"만 골라서 merge 
    mask = target & (~_is_missing(df[form_col]))
    print(f"{mask.sum()}행에 en_no가 없습니다. 번호 부여를 시작합니다.")

    if mask.any():
        sub = df.loc[mask, [label_col, form_col]].copy()
        lab = load_label_map(label_csv, main_prefix) #label.csv읽어오기
        
        merged = sub.merge(lab, on=[label_col, form_col], how="left", suffixes=("", "_lab"))

        # 2-1) 매칭 성공한 것만 채움 (덮어쓰기 X라서 mask 영역만)
        df.loc[mask, no_col] = merged[no_col].values

        # 2-2) 그래도 NA면 기본값(EC/EF/ETM/ETN) 채우고 label이 이상한 경우 9999
        still =  _is_missing(df[no_col]) & (~_is_missing(df[form_col])) #여전히 매칭 안 된 것.
        #Series만듦.
        defaults = (df.loc[still, label_col].map(DEFAULT_BY_EN_LABEL)
                                            .fillna(9999).astype("Int64"))
        defaults = defaults.reindex(df.index)
        mask2 = still & defaults.notna()
        df.loc[mask2, no_col] = defaults[defaults.notna()].values

    return df

# ===========================================
# Make EN_No_sub.
def make_en_no_sub(
    df: pd.DataFrame,
    *,
    form_col: str = "EN_form",
    out_col: str = "EN_No_sub",
) -> pd.DataFrame:
    """
    EN_No_sub rules:
      +1  if contains 'ㄴ'
      +2  if contains '는'
      +3  if contains '은'
      +10 if contains any of {'ㄹ','을','리','러','려','래'} (once)
      +100 if contains any of {'더','던','드','든'} (once)
    """
    if form_col not in df.columns:
        return

    s = df[form_col].astype("string").fillna("")
    sub = pd.Series(0, index=df.index, dtype="int64")

    sub += s.str.contains("ㄴ", na=False).astype("int64") * 1
    sub += s.str.contains("는", na=False).astype("int64") * 2
    sub += s.str.contains("은", na=False).astype("int64") * 3

    re10 = re.compile(r"(ㄹ|을|리|러|려|래)")
    sub += s.str.contains(re10, na=False).astype("int64") * 10

    re100 = re.compile(r"(더|던|드|든)")
    sub += s.str.contains(re100, na=False).astype("int64") * 100

    df[out_col] = sub

    return df