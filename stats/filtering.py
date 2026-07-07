# bareun/stats/filtering.py

# 필터링 기능 제공
# 사용법 예시:
''' 
from stats.filtering import apply_filters, FilterValue, has_value, _topn_values

filters: Dict[str, FilterValue] = {
    "category": ["강의", "낭독"],
    "outcome_total": lambda s: s >= 500,
    "outcome_total": lambda s: (s >= 20) & (~s.isin(list([4999, 2999]))),
    }

filtered_df = apply_filters(df, filters)
'''

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Union, Literal
import numpy as np
import pandas as pd

FilterValue = Union[Any, Callable[[pd.Series], Union[pd.Series, np.ndarray]]]

#----------------------------
def apply_filters(df: pd.DataFrame, filters=None) -> pd.DataFrame:
    if not filters:
        return df

    mask = pd.Series(True, index=df.index)

    for col, rule in filters.items():
        if col not in df.columns:
            raise KeyError(f"Missing filter column: {col}")

        s = df[col]

        # 🔥 boolean 계열 자동 처리
        if isinstance(rule, bool):
            s_bool = _to_bool_series(s)
            mask &= (s_bool == rule)

        elif callable(rule):
            m = rule(s)
            mask &= pd.Series(m, index=df.index)

        elif isinstance(rule, (list, tuple, set)):
            mask &= s.isin(rule)

        else:
            mask &= (s == rule)

    return df.loc[mask].copy()


#----------------------------
def only_korean(regex: str = r"[가-힣ㄱ-ㅎㅏ-ㅣ]+", allow_na: bool = False) -> Callable[[pd.Series], pd.Series]:
    """
    Return a filter function that keeps rows whose values fully match the regex.
    Default: only Hangul syllables (가-힣ㄱ-ㅎㅏ-ㅣ)+
    """
    def _fn(s: pd.Series) -> pd.Series:
        m = s.astype(str).str.fullmatch(regex, na=False)
        if allow_na:
            return s.isna() | m
        return m
    return _fn

#----------------------------
def has_value(s: pd.Series) -> pd.Series:
    """
    NaN / 빈문자열 / 'NULL' 제외한 값 여부 판단
    → Boolean mask (pd.Series[bool]) 반환
    """
    return (
        s.notna()
        & s.astype(str).str.strip().ne("")
        & s.astype(str).str.upper().ne("NULL")
    )

#----------------------------
def _topn_values(
    df: pd.DataFrame,
    col: str,
    *,
    mode: Literal["rows", "weight"] = "weight",
    weight_col: str = "ID",
    top_n: Optional[int] = None,
    min_count: Optional[float] = None,
    dropna: bool = True
) -> list:
    """
    Return the most frequent values of `df[col]` as a Python list.

    빈도(frequency)를 어떻게 정의하느냐에 따라 두 가지 모드를 지원한다.

    Parameters
    ----------
    df : pd.DataFrame
        입력 데이터프레임.
    col : str
        '상위값(top N)'을 뽑을 대상 컬럼명. 예: "V_form", "category" 등.

    mode : {"rows", "weight"}, default "weight"
        빈도 계산 방식.
        - "rows": 단순 행 개수(출현 횟수)로 빈도 계산
                 => frequency(value) = number of rows where df[col] == value
        - "weight": 가중치 컬럼의 합으로 빈도 계산
                   => frequency(value) = sum(df[weight_col]) over rows where df[col] == value
        ※ 말뭉치에서 "행=토큰" 구조가 아닐 때(예: ID가 어절수/문자수/가중치 역할) 유용.

    weight_col : str, default "ID"
        mode="weight"에서 사용할 가중치 컬럼명.
        예: "ID"가 실제로는 어절 수, 길이, 가중치 등을 의미할 수 있음.
        이 컬럼이 df에 없으면 ValueError 발생.

    top_n : Optional[int], default None
        상위 N개만 반환.
        - None이면 '상위 제한 없음' (필터만 적용한 전체를 반환)

    min_count : Optional[float], default None
        최소 빈도(행 개수 또는 가중치 합) 기준.
        - None이면 최소 기준 없음
        - 예: min_count=10 이면 빈도 10 이상인 값들만 남김.

    dropna : bool, default True
        NaN(결측치) 처리.
        - True: col이 NaN인 행은 계산에서 제외(기본).
        - False: NaN도 하나의 범주로 포함하여 계산.

    Returns
    -------
    list
        빈도 기준으로 내림차순 정렬된 '값(value)'들의 리스트.
        ※ 빈도 숫자는 반환하지 않고, 값 목록만 반환한다.
        예: ["신문", "소설", "보도자료"]

    Notes
    -----
    - 반환값은 list[str]처럼 보이지만, 실제로는 df[col]의 dtype에 따라
      숫자/문자/카테고리 등 어떤 타입이든 될 수 있다.
    - 빈도 자체도 필요하면 vc(Series)를 반환하도록 변형하는 게 좋다.
    """

    # ------------------------------------------------------------
    # 1) NaN 처리: dropna=True이면 col이 NaN인 행은 아예 제외
    # ------------------------------------------------------------
    if dropna:
        # subset=[col]: col만 보고 NaN 행 제거
        # copy(): SettingWithCopyWarning 회피 및 안전한 편집을 위해 복사
        d = df.dropna(subset=[col]).copy()
    else:
        # NaN도 포함해서 빈도를 계산하고 싶으면 그대로 사용
        d = df.copy()

    # ------------------------------------------------------------
    # 2) 빈도 계산 (mode에 따라 다름)
    # ------------------------------------------------------------
    if mode == "rows":
        # (A) 행 개수 기반 빈도:
        # value_counts()는 각 값의 등장 횟수를 세어 Series로 반환
        #
        # dropna 파라미터는 "NaN을 세어줄지" 결정:
        # - dropna=True  -> NaN 제외
        # - dropna=False -> NaN 포함
        #
        # 여기서는 'dropna' 옵션과 일관되게 처리하려고
        # dropna=not dropna 를 사용:
        # - 사용자가 dropna=True면 => value_counts(dropna=False?) 가 아니라
        #   value_counts(dropna=not True)=value_counts(dropna=False) ??? 라서 헷갈릴 수 있음.
        #   하지만 위에서 이미 NaN을 제거했기 때문에( d = dropna(...) ),
        #   실제 결과에는 영향이 거의 없음.
        # - 사용자가 dropna=False면 => value_counts(dropna=True)로 NaN 제외가 되는데
        #   이건 의도와 반대라서, 보통은 아래를 value_counts(dropna=dropna)로 두는 게 직관적임.
        #
        # 다만 현재 코드는 groupby와 동일한 형태를 맞추려고 dropna=not dropna를 사용한 것으로 보임.
        vc = d[col].value_counts(dropna=not dropna)

    elif mode == "weight":
        # (B) 가중치 합 기반 빈도:
        # groupby(col)로 묶은 뒤 weight_col의 합을 구함.
        if weight_col not in d.columns:
            raise ValueError(f"weight_col '{weight_col}' not in dataframe.")

        # groupby(..., dropna=...)에서 dropna=False면 NaN도 그룹으로 포함 가능(판다스 버전에 따라)
        # sort_values(ascending=False): 가장 큰 가중치 합이 먼저 오도록 정렬
        vc = (
            d.groupby(col, dropna=not dropna)[weight_col]
             .sum()
             .sort_values(ascending=False)
        )

    else:
        # mode 오타/잘못된 값 방지
        raise ValueError("mode must be 'rows' or 'weight'.")

    # ------------------------------------------------------------
    # 3) (선택) 최소 빈도 기준 필터링
    # ------------------------------------------------------------
    if min_count is not None:
        # float로 강제 변환해 안전하게 비교
        vc = vc[vc >= float(min_count)]

    # ------------------------------------------------------------
    # 4) (선택) 상위 N개만 자르기
    # ------------------------------------------------------------
    if top_n is not None:
        vc = vc.iloc[: int(top_n)]

    # ------------------------------------------------------------
    # 5) 최종 반환: '빈도값'이 아니라 '값 목록'만 반환
    # ------------------------------------------------------------
    return vc.index.tolist()

#   ----------------------------
def _to_bool_series(s: pd.Series) -> pd.Series:
    """Robust TRUE/FALSE -> boolean (handles bool, 'TRUE'/'FALSE', 'True'/'False', 1/0)."""
    if pd.api.types.is_bool_dtype(s) or str(s.dtype) == "boolean":
        return s.astype("boolean")
    x = s.astype(str).str.strip().str.upper()
    return x.map({
        "TRUE": True, "FALSE": False,
        "1": True, "0": False,
        "T": True, "F": False,
        "YES": True, "NO": False,
        "Y": True, "N": False,
        "ON": True, "OFF": False,
        "O": True, "X": False
    }).astype("boolean")



# ----------------------------
# 자동 dtype 변환 함수, object -> boolean / numeric
def auto_cast_dtypes(
    df: pd.DataFrame,
    bool_threshold: float = 0.95,
    numeric_threshold: float = 0.95,
    verbose: bool = True
) -> pd.DataFrame:
    """
    object dtype 컬럼을 자동 감지하여 boolean 또는 numeric으로 변환.
    
    Parameters
    ----------
    bool_threshold : boolean으로 간주할 최소 비율
    numeric_threshold : numeric으로 간주할 최소 비율
    """

    df = df.copy()

    for col in df.columns:
        s = df[col]

        # 이미 numeric or boolean이면 skip
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            continue

        if s.dtype != "object":
            continue

        non_na = s.dropna()

        if len(non_na) == 0:
            continue

        # 🔎 Boolean 감지
        upper_vals = non_na.astype(str).str.strip().str.upper()
        bool_like = upper_vals.isin(
            ["TRUE", "FALSE", "1", "0", "T", "F", "YES", "NO"]
        )

        bool_ratio = bool_like.mean()

        if bool_ratio >= bool_threshold:
            df[col] = _to_bool_series(s)
            if verbose:
                print(f"[Boolean] {col}  (ratio={bool_ratio:.2f})")
            continue

        # 🔎 Numeric 감지
        numeric_conv = pd.to_numeric(non_na, errors="coerce")
        numeric_ratio = numeric_conv.notna().mean()

        if numeric_ratio >= numeric_threshold:
            df[col] = pd.to_numeric(s, errors="coerce")
            if verbose:
                print(f"[Numeric] {col}  (ratio={numeric_ratio:.2f})")
            continue

        # 🔎 그대로 유지
        if verbose:
            print(f"[Keep as object] {col}")

    return df
