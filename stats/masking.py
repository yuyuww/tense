# bareun/stats/masking.py
# 
from __future__ import annotations

from typing import Optional, Sequence
import numpy as np
import pandas as pd

def mask_feature(
    df: pd.DataFrame,
    feature_col: str,
    *,
    # 1) p-value 마스킹
    p_col: str = "p_value",
    p_max: Optional[float] = None,  # 예: 0.05

    # 2) 결합 카운트(희귀 결합) 마스킹
    count_col: str = "a_in_context_and_level",
    count_min: Optional[float] = None,  # 예: 5

    # 3) 효과크기 최소값 마스킹(|feature| 기준)
    abs_min: Optional[float] = None,    # 예: 0.1

    # 4) 지정한 컬럼의 결측치NaN 마스킹
    require_notna: Optional[Sequence[str]] = None,

    # 5) 마스킹된 값 대체(기본 0=중립)
    fill_value: float = 0.0,
) -> pd.Series:
    """
    df[feature_col]을 마스킹해서 Series로 반환.
    조건을 만족하지 못한 행은 fill_value(기본 0)로 바뀜.
    """
    if feature_col not in df.columns:
        raise KeyError(f"Missing feature column: {feature_col}")

    x = df[feature_col].astype(float).copy()
    keep = pd.Series(True, index=df.index)

    if require_notna:
        missing = [c for c in require_notna if c not in df.columns]
        if missing:
            raise KeyError(f"Missing columns in require_notna: {missing}")
        for c in require_notna:
            keep &= df[c].notna()

    if p_max is not None:
        if p_col not in df.columns:
            raise KeyError(f"p_max is set but missing p_col: {p_col}")
        keep &= (df[p_col].astype(float) <= float(p_max))

    if count_min is not None:
        if count_col not in df.columns:
            raise KeyError(f"count_min is set but missing count_col: {count_col}")
        keep &= (df[count_col].astype(float) >= float(count_min))

    if abs_min is not None:
        keep &= (np.abs(x) >= float(abs_min))

    return pd.Series(np.where(keep.values, x.values, float(fill_value)), index=df.index)
