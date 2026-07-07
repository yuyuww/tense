"""Plot helpers extracted from notebook 4.5.1."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 5. 직전 두 상태에 따른 다음 T/F 확률 시각화 함수
# =========================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import beta
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Clopper-Pearson 신뢰구간 계산 함수
def _clopper_pearson_ci(k, n, alpha=0.05):
    """
    Clopper-Pearson binomial confidence interval.
    """
    if n == 0 or pd.isna(n):
        return np.nan, np.nan

    if k == 0:
        lo = 0.0
    else:
        lo = beta.ppf(alpha / 2, k, n - k + 1)

    if k == n:
        hi = 1.0
    else:
        hi = beta.ppf(1 - alpha / 2, k + 1, n - k)

    return lo, hi

# '앞선 두 상태에 따른 다음 T/F 확률'을 막대그래프로 그리는 함수
def plot_prior2_probability_from_trigram_result(
    tri_result,
    *,
    target="T",
    title=None,
    xlabel=None,
    ylabel=None,
    base_P_label = "기본 었 출현률",
    x_super_label = "Sequence of prior states",
    Markov_label = "1차 연쇄 기대값",
    show_base_line=True,
    show_markov_expected=True,
    figsize=(8, 4),
):
    """
    analyze_trigram_from_weighted_df() 결과를 이용해
    '앞선 두 상태에 따른 다음 T/F 확률'을 막대그래프로 그린다.

    target="T"이면:
        y = P(next=T | prior two states)

    target="F"이면:
        y = P(next=F | prior two states)

    막대 높이:
        실제 조건부 확률

    점선:
        전체 기본 T/F 출현률

    검은 짧은 가로선:
        1차 Markov 기대확률 P(next | 1-back)
    """

    target = target.upper()
    if target not in ["T", "F"]:
        raise ValueError("target은 'T' 또는 'F'만 가능합니다.")

    tri = tri_result.copy()

    # 그림의 순서:
    # 0/2 T: FF
    # 1/2 T: TF, FT
    # 2/2 T: TT
    prefix_order = ["FF", "TF", "FT", "TT"]

    rows = []

    for prefix in prefix_order:
        k_col = f"{prefix}{target}"

        if target == "T":
            other_col = f"{prefix}F"
        else:
            other_col = f"{prefix}T"

        k = tri[k_col].sum()
        n = tri[k_col].sum() + tri[other_col].sum()

        prob = k / n if n > 0 else np.nan
        ci_low, ci_high = _clopper_pearson_ci(k, n)

        n_T_prior = prefix.count("T")
        panel = f"{n_T_prior}/2 T"

        one_back = prefix[1]

        # 1차 Markov 기대확률
        # prefix = AB일 때, Markov 기대는 P(target | B)
        # B가 T이면 P(target | T), B가 F이면 P(target | F)
        back = prefix[1]
        markov_num_col = f"{back}{target}"

        if target == "T":
            markov_den_cols = [f"{back}T", f"{back}F"]
        else:
            markov_den_cols = [f"{back}T", f"{back}F"]

        markov_num = tri[markov_num_col].sum()
        markov_den = tri[markov_den_cols[0]].sum() + tri[markov_den_cols[1]].sum()
        markov_prob = markov_num / markov_den if markov_den > 0 else np.nan

        rows.append({
            "prefix": prefix,
            "sequence_label": f"{prefix[0]}-{prefix[1]}→{target}",
            "panel": panel,
            "one_back": one_back,
            "k": k,
            "n": n,
            "prob": prob,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "markov_prob": markov_prob,
        })

    plot_df = pd.DataFrame(rows)

    # 기본 출현률 점선
    if target == "T":
        base_prob = tri["n_T"].sum() / tri["n_sentences"].sum()
        default_ylabel = "Probability of T"
    else:
        base_prob = tri["n_F"].sum() / tri["n_sentences"].sum()
        default_ylabel = "Probability of F"

    if ylabel is None:
        ylabel = default_ylabel

    panel_order = ["0/2 T", "1/2 T", "2/2 T"]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=figsize,
        sharey=True,
        gridspec_kw={"width_ratios": [1, 2, 1]}
    )

    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map = {
        "F": cycle[0],
        "T": cycle[1],
    }

    for ax, panel in zip(axes, panel_order):
        sub = plot_df.loc[plot_df["panel"] == panel].copy()

        x = np.arange(len(sub))
        y = sub["prob"].to_numpy()

        yerr = np.vstack([
            y - sub["ci_low"].to_numpy(),
            sub["ci_high"].to_numpy() - y
        ])

        colors = [color_map[v] for v in sub["one_back"]]

        ax.bar(x, y, color=colors)

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="none",
            ecolor="black",
            capsize=3,
            linewidth=1
        )

        if show_base_line:
            ax.axhline(
                base_prob,
                linestyle=(0, (4, 4)),
                color="black",
                linewidth=1
            )

        if show_markov_expected:
            ax.plot(
                x,
                sub["markov_prob"],
                linestyle="none",
                marker="x",
                markersize=6,
                markeredgewidth =1.2,
                color="black",
            )

        ax.set_title(panel)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["sequence_label"], rotation=45, ha="right")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

    axes[0].set_ylabel(ylabel)
    fig.supxlabel(x_super_label)

    if title is not None:
        fig.suptitle(title, y=1.05)

    legend_handles = [
        Patch(color=color_map["F"], label="직전 문장: 비었"),
        Patch(color=color_map["T"], label="직전 문장: 었"),
    ]

    if show_base_line:
        legend_handles.append(
            Line2D(
                [0], [0],
                color="black",
                linestyle=(0, (4, 4)),
                label=base_P_label
            )
        )

    if show_markov_expected:
        legend_handles.append(
            Line2D(
                [0], [0],
                color="black",
                marker="x",
                linestyle="none",
                markersize=6,
                markeredgewidth=1.2,
                label=Markov_label
            )
        )

    fig.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5)
    )

    fig.tight_layout()

    return fig, axes, plot_df


# 그래프: 직전 상태에 따른 다음 T/F 확률 비교 그래프

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import beta
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.font_manager as fm


def set_korean_font():
    font_candidates = [
        "Malgun Gothic",
        "AppleGothic",
        "NanumGothic",
        "Noto Sans CJK KR",
        "Noto Sans KR",
    ]

    installed_fonts = {f.name for f in fm.fontManager.ttflist}

    for font in font_candidates:
        if font in installed_fonts:
            plt.rcParams["font.family"] = font
            break

    plt.rcParams["axes.unicode_minus"] = False


def _clopper_pearson_ci(k, n, alpha=0.05):
    """
    Clopper-Pearson 이항비율 신뢰구간.
    """
    if n == 0 or pd.isna(n):
        return np.nan, np.nan

    if k == 0:
        lo = 0.0
    else:
        lo = beta.ppf(alpha / 2, k, n - k + 1)

    if k == n:
        hi = 1.0
    else:
        hi = beta.ppf(1 - alpha / 2, k + 1, n - k)

    return lo, hi


def plot_1prior_T_by_units_grouped(
    tri_result,
    *,
    unit_col,
    order=None,
    title="항목별 직전 문장 기준 었 출현 확률",
    xlabel="직전 문장의 시제",
    ylabel="다음 문장의 었 출현 확률",
    figsize=(8, 4),
    ylim=(0, 1),
    show_base=True,
    show_ci=True,
    ci_alpha=0.05,
    show_value=False,
):
    """
    analyze_trigram_from_weighted_df() 결과에서
    여러 항목의 1Prior T 확률을 함께 그린다.

    x축:
        F→T, T→T

    막대:
        항목별 확률

    색깔:
        항목별 자동 색상

    기준값:
        각 항목의 P_T를 검은 '_' 표시로 표시

    오차막대:
        Clopper-Pearson 이항비율 신뢰구간
        F→T: FT / (FT + FF)
        T→T: TT / (TT + TF)
    """

    set_korean_font()

    plot_df = tri_result.copy()

    needed_cols = [
        unit_col,
        "P_T",
        "P_T_given_F_pair",
        "P_T_given_T_pair",
        "FT", "FF", "TT", "TF",
    ]

    missing = [c for c in needed_cols if c not in plot_df.columns]
    if missing:
        raise ValueError(f"필요한 컬럼이 없습니다: {missing}")

    if order is not None:
        plot_df[unit_col] = pd.Categorical(
            plot_df[unit_col],
            categories=order,
            ordered=True
        )
        plot_df = plot_df.sort_values(unit_col)
    else:
        plot_df = plot_df.sort_values(unit_col)

    plot_df = plot_df.reset_index(drop=True)

    units = plot_df[unit_col].astype(str).tolist()
    n_units = len(units)

    x_groups = np.array([0, 1])
    group_labels = ["비었 뒤 었", "었 뒤 었"]

    values = {
        "F→T": plot_df["P_T_given_F_pair"].to_numpy(),
        "T→T": plot_df["P_T_given_T_pair"].to_numpy(),
    }

    base = plot_df["P_T"].to_numpy()

    total_width = 0.75
    bar_width = total_width / max(n_units, 1)

    offsets = (
        np.arange(n_units) - (n_units - 1) / 2
    ) * bar_width

    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(n_units)]

    fig, ax = plt.subplots(figsize=figsize)

    # -----------------------------
    # 막대 그리기
    # -----------------------------
    for i, unit in enumerate(units):
        s = plot_df.iloc[i]

        y = np.array([
            values["F→T"][i],
            values["T→T"][i],
        ], dtype=float)

        xpos = x_groups + offsets[i]

        ax.bar(
            xpos,
            y,
            width=bar_width * 0.9,
            color=colors[i],
            label=unit,
        )

        # -----------------------------
        # 신뢰구간
        # -----------------------------
        if show_ci:
            # F→T = FT / (FT + FF)
            lo_ft, hi_ft = _clopper_pearson_ci(
                s["FT"],
                s["FT"] + s["FF"],
                alpha=ci_alpha
            )

            # T→T = TT / (TT + TF)
            lo_tt, hi_tt = _clopper_pearson_ci(
                s["TT"],
                s["TT"] + s["TF"],
                alpha=ci_alpha
            )

            ci_low = np.array([lo_ft, lo_tt], dtype=float)
            ci_high = np.array([hi_ft, hi_tt], dtype=float)

            yerr = np.vstack([
                y - ci_low,
                ci_high - y
            ])

            # 부동소수점 오차로 음수가 되는 경우 방지
            yerr = np.where(yerr < 0, 0, yerr)

            ax.errorbar(
                xpos,
                y,
                yerr=yerr,
                fmt="none",
                ecolor="black",
                capsize=3,
                linewidth=0.9,
            )

        # 항목별 base P_T 표시
        if show_base:
            ax.plot(
                xpos,
                [base[i], base[i]],
                linestyle="none",
                marker="_",
                markersize=12,
                color="black",
            )

        if show_value:
            for x, v in zip(xpos, y):
                if not np.isnan(v):
                    ax.text(
                        x,
                        v + 0.015,
                        f"{v:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        rotation=90 if n_units > 5 else 0,
                    )

    # -----------------------------
    # 축 설정
    # -----------------------------
    ax.set_xticks(x_groups)
    ax.set_xticklabels(group_labels)

    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)

    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    # -----------------------------
    # 범례
    # -----------------------------
    unit_handles = [
        Patch(color=colors[i], label=units[i])
        for i in range(n_units)
    ]

    legend1 = ax.legend(
        handles=unit_handles,
        title=unit_col,
        frameon=False,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    ax.add_artist(legend1)

    marker_handles = []

    if show_base:
        marker_handles.append(
            Line2D(
                [0], [0],
                color="black",
                marker="_",
                linestyle="none",
                markersize=12,
                label="기본 ‘-었-’ 출현률",
            )
        )

    if show_ci:
        marker_handles.append(
            Line2D(
                [0], [0],
                color="black",
                linestyle="-",
                linewidth=0.9,
                label="95% 신뢰구간",
            )
        )

    if marker_handles:
        ax.legend(
            handles=marker_handles,
            frameon=False,
            bbox_to_anchor=(1.02, 0),
            loc="lower left",
        )

    fig.tight_layout()

    return fig, ax


# 직전 문장 이후 출현확률 그리기. : UNIT별

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import beta
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


def _clopper_pearson_ci(k, n, alpha=0.05):
    if n == 0 or pd.isna(n):
        return np.nan, np.nan

    if k == 0:
        lo = 0.0
    else:
        lo = beta.ppf(alpha / 2, k, n - k + 1)

    if k == n:
        hi = 1.0
    else:
        hi = beta.ppf(1 - alpha / 2, k + 1, n - k)

    return lo, hi


def plot_1prior_T_by_units(
    tri_result,
    *,
    unit_col,
    order=None,
    title="항목별 직전 문장 기준 ‘-었-’ 출현 확률",
    figsize=(10, 4),
    ylim=(0, 1),
    show_value=False,
    show_ci=True,
):
    """
    analyze_trigram_from_weighted_df() 결과에서
    여러 항목의 1Prior T 확률을 함께 그린다.

    막대:
    - F→T = P_T_given_F_pair = FT / (FT + FF)
    - T→T = P_T_given_T_pair = TT / (TT + TF)

    검은 짧은 가로선:
    - 해당 항목의 base P_T

    오차막대:
    - Clopper-Pearson binomial 95% confidence interval
    """

    plot_df = tri_result.copy()

    if order is not None:
        plot_df[unit_col] = pd.Categorical(
            plot_df[unit_col],
            categories=order,
            ordered=True
        )
        plot_df = plot_df.sort_values(unit_col)
    else:
        plot_df = plot_df.sort_values(unit_col)

    plot_df = plot_df.reset_index(drop=True)

    labels = plot_df[unit_col].astype(str).tolist()

    y_FT = plot_df["P_T_given_F_pair"].to_numpy()
    y_TT = plot_df["P_T_given_T_pair"].to_numpy()
    base = plot_df["P_T"].to_numpy()

    # -----------------------------
    # 신뢰구간 계산
    # -----------------------------
    ci_FT_low = []
    ci_FT_high = []
    ci_TT_low = []
    ci_TT_high = []

    for _, s in plot_df.iterrows():
        # F→T: FT / (FT + FF)
        k_ft = s["FT"]
        n_ft = s["FT"] + s["FF"]
        lo, hi = _clopper_pearson_ci(k_ft, n_ft)
        ci_FT_low.append(lo)
        ci_FT_high.append(hi)

        # T→T: TT / (TT + TF)
        k_tt = s["TT"]
        n_tt = s["TT"] + s["TF"]
        lo, hi = _clopper_pearson_ci(k_tt, n_tt)
        ci_TT_low.append(lo)
        ci_TT_high.append(hi)

    ci_FT_low = np.array(ci_FT_low, dtype=float)
    ci_FT_high = np.array(ci_FT_high, dtype=float)
    ci_TT_low = np.array(ci_TT_low, dtype=float)
    ci_TT_high = np.array(ci_TT_high, dtype=float)

    x = np.arange(len(plot_df))
    width = 0.36

    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map = {
        "F": cycle[0],
        "T": cycle[1],
    }

    fig, ax = plt.subplots(figsize=figsize)

    ax.bar(
        x - width / 2,
        y_FT,
        width,
        color=color_map["F"],
        label="F→T"
    )

    ax.bar(
        x + width / 2,
        y_TT,
        width,
        color=color_map["T"],
        label="T→T"
    )

    # -----------------------------
    # 신뢰구간 표시
    # -----------------------------
    if show_ci:
        yerr_FT = np.vstack([
            y_FT - ci_FT_low,
            ci_FT_high - y_FT
        ])

        yerr_TT = np.vstack([
            y_TT - ci_TT_low,
            ci_TT_high - y_TT
        ])

        ax.errorbar(
            x - width / 2,
            y_FT,
            yerr=yerr_FT,
            fmt="none",
            ecolor="black",
            capsize=3,
            linewidth=0.9
        )

        ax.errorbar(
            x + width / 2,
            y_TT,
            yerr=yerr_TT,
            fmt="none",
            ecolor="black",
            capsize=3,
            linewidth=0.9
        )

    # 항목별 base P_T
    ax.plot(
        x,
        base,
        linestyle="none",
        marker="_",
        markersize=18,
        color="black",
        label="기본 ‘-었-’ 출현률"
    )

    if show_value:
        for xi, v in zip(x - width / 2, y_FT):
            if not np.isnan(v):
                ax.text(
                    xi,
                    v + 0.015,
                    f"{v:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8
                )

        for xi, v in zip(x + width / 2, y_TT):
            if not np.isnan(v):
                ax.text(
                    xi,
                    v + 0.015,
                    f"{v:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8
                )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")

    ax.set_ylim(*ylim)
    ax.set_ylabel("다음 문장의 ‘-었-’ 출현 확률")
    ax.set_xlabel(unit_col)
    ax.set_title(title)

    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    legend_handles = [
        Patch(color=color_map["F"], label="비과거 뒤 ‘-었-’"),
        Patch(color=color_map["T"], label="‘-었-’ 뒤 ‘-었-’"),
        Line2D(
            [0], [0],
            color="black",
            marker="_",
            linestyle="none",
            markersize=14,
            label="기본 ‘-었-’ 출현률"
        )
    ]

    ax.legend(handles=legend_handles, frameon=False)

    fig.tight_layout()

    return fig, ax


# 이전 두 항목 이후 출현률 보이기

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import beta
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.font_manager as fm


def set_korean_font():
    font_candidates = [
        "Malgun Gothic",
        "AppleGothic",
        "NanumGothic",
        "Noto Sans CJK KR",
        "Noto Sans KR",
    ]

    installed_fonts = {f.name for f in fm.fontManager.ttflist}

    for font in font_candidates:
        if font in installed_fonts:
            plt.rcParams["font.family"] = font
            break

    plt.rcParams["axes.unicode_minus"] = False


def _clopper_pearson_ci(k, n, alpha=0.05):
    if n == 0 or pd.isna(n):
        return np.nan, np.nan

    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)

    return lo, hi


def plot_prior2_probability_by_units(
    tri_result,
    *,
    unit_col,
    target="T",
    order=None,
    title="항목별 직전 두 문장 기준 ‘-었-’ 출현 확률",
    ylabel=None,
    xlabel="직전 두 문장의 시제 배열",
    base_label = "기본 었 출현률",
    figsize=(12, 4),
    ylim=(0, 1),
    show_ci=True,
    show_base=True,
    show_markov_expected=True,
    show_value=False,
    label_style="korean",
):
    """
    analyze_trigram_from_weighted_df() 결과를 이용해
    유닛별 2Prior 그래프를 그린다.

    target="T":
        FF→T, TF→T, FT→T, TT→T

    target="F":
        FF→F, TF→F, FT→F, TT→F

    막대:
        각 유닛의 실제 조건부 확률
        P_T_given_FF, P_T_given_TF, P_T_given_FT, P_T_given_TT

    검은 '_':
        각 유닛의 base P_T 또는 base P_F

    검은 'x':
        각 유닛의 1차 Markov 기대확률
        P(target | 1-back)
    """

    set_korean_font()

    target = target.upper()
    if target not in ["T", "F"]:
        raise ValueError("target은 'T' 또는 'F'만 가능합니다.")

    plot_df = tri_result.copy()

    if order is not None:
        plot_df[unit_col] = pd.Categorical(
            plot_df[unit_col],
            categories=order,
            ordered=True
        )
        plot_df = plot_df.sort_values(unit_col)
    else:
        plot_df = plot_df.sort_values(unit_col)

    plot_df = plot_df.reset_index(drop=True)

    prefix_order = ["FF", "TF", "FT", "TT"]

    needed_cols = [unit_col]

    if target == "T":
        needed_cols += [
            "P_T",
            "P_T_given_FF",
            "P_T_given_TF",
            "P_T_given_FT",
            "P_T_given_TT",
            "P_T_given_F_pair",
            "P_T_given_T_pair",
        ]
    else:
        needed_cols += [
            "P_F",
            "P_F_given_FF",
            "P_F_given_TF",
            "P_F_given_FT",
            "P_F_given_TT",
            "P_F_given_F_pair",
            "P_F_given_T_pair",
        ]

    # CI 계산용 빈도 컬럼
    for prefix in prefix_order:
        needed_cols += [f"{prefix}T", f"{prefix}F"]

    missing = [c for c in needed_cols if c not in plot_df.columns]
    if missing:
        raise ValueError(f"필요한 컬럼이 없습니다: {missing}")

    units = plot_df[unit_col].astype(str).tolist()
    n_units = len(units)

    if ylabel is None:
        ylabel = f"다음 문장의 {target} 확률"

    # 색깔: 유닛별 자동 배정
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(n_units)]

    # 패널 구성
    panel_info = {
        "0/2 T": ["FF"],
        "1/2 T": ["TF", "FT"],
        "2/2 T": ["TT"],
    }

    fig, axes = plt.subplots(
        1,
        3,
        figsize=figsize,
        sharey=True,
        gridspec_kw={"width_ratios": [1, 2, 1]}
    )

    total_width = 0.78
    bar_width = total_width / max(n_units, 1)
    offsets = (np.arange(n_units) - (n_units - 1) / 2) * bar_width

    for ax, (panel_name, prefixes) in zip(axes, panel_info.items()):
        x_groups = np.arange(len(prefixes))

        for i, unit in enumerate(units):
            s = plot_df.iloc[i]

            y_values = []
            ci_lows = []
            ci_highs = []
            base_values = []
            markov_values = []

            for prefix in prefixes:
                # 실제 확률 컬럼
                prob_col = f"P_{target}_given_{prefix}"
                prob = s[prob_col]

                # CI
                k_col = f"{prefix}{target}"
                other = "F" if target == "T" else "T"
                other_col = f"{prefix}{other}"

                k = s[k_col]
                n = s[k_col] + s[other_col]
                ci_low, ci_high = _clopper_pearson_ci(k, n)

                # base P_T / P_F
                base_col = f"P_{target}"
                base_prob = s[base_col]

                # Markov 기대값: 마지막 상태만 기준
                one_back = prefix[1]
                markov_col = f"P_{target}_given_{one_back}_pair"
                markov_prob = s[markov_col]

                y_values.append(prob)
                ci_lows.append(ci_low)
                ci_highs.append(ci_high)
                base_values.append(base_prob)
                markov_values.append(markov_prob)

            xpos = x_groups + offsets[i]

            ax.bar(
                xpos,
                y_values,
                width=bar_width * 0.9,
                color=colors[i],
                label=unit,
            )

            if show_ci:
                y = np.array(y_values, dtype=float)
                ci_low_arr = np.array(ci_lows, dtype=float)
                ci_high_arr = np.array(ci_highs, dtype=float)

                yerr = np.vstack([
                    y - ci_low_arr,
                    ci_high_arr - y
                ])

                ax.errorbar(
                    xpos,
                    y,
                    yerr=yerr,
                    fmt="none",
                    ecolor="black",
                    capsize=2,
                    linewidth=0.8,
                )

            if show_base:
                ax.plot(
                    xpos,
                    base_values,
                    linestyle="none",
                    marker="_",
                    markersize=11,
                    color="black",
                )

            if show_markov_expected:
                ax.plot(
                    xpos,
                    markov_values,
                    linestyle="none",
                    marker="x",
                    markersize=5,
                    color="black",
                )

            if show_value:
                for x, v in zip(xpos, y_values):
                    if not pd.isna(v):
                        ax.text(
                            x,
                            v + 0.015,
                            f"{v:.2f}",
                            ha="center",
                            va="bottom",
                            fontsize=8,
                            rotation=90 if n_units > 5 else 0,
                        )

        # x축 라벨
        if label_style == "korean":
            if target == "T":
                label_map = {
                    "FF": "비과거-비과거 뒤",
                    "TF": "‘-었-’-비과거 뒤",
                    "FT": "비과거-‘-었-’ 뒤",
                    "TT": "‘-었-’-‘-었-’ 뒤",
                }
            else:
                label_map = {
                    "FF": "비과거-비과거 뒤",
                    "TF": "‘-었-’-비과거 뒤",
                    "FT": "비과거-‘-었-’ 뒤",
                    "TT": "‘-었-’-‘-었-’ 뒤",
                }

            xtick_labels = [label_map[p] for p in prefixes]

        else:
            xtick_labels = [f"{p}→{target}" for p in prefixes]

        ax.set_title(panel_name)
        ax.set_xticks(x_groups)
        ax.set_xticklabels(xtick_labels, rotation=35, ha="right")
        ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

    axes[0].set_ylabel(ylabel)
    fig.supxlabel(xlabel)

    if title is not None:
        fig.suptitle(title, y=1.05)

    # 범례
    unit_handles = [
        Patch(color=colors[i], label=units[i])
        for i in range(n_units)
    ]

    marker_handles = []

    if show_base:
        marker_handles.append(
            Line2D(
                [0], [0],
                color="black",
                marker="_",
                linestyle="none",
                markersize=12,
                label=base_label,
            )
        )

    if show_markov_expected:
        marker_handles.append(
            Line2D(
                [0], [0],
                color="black",
                marker="x",
                linestyle="none",
                markersize=6,
                label="1차 연쇄 기대값",
            )
        )

    legend1 = fig.legend(
        handles=unit_handles,
        title=unit_col,
        loc="center left",
        bbox_to_anchor=(1.01, 0.58),
        frameon=False,
    )

    if marker_handles:
        fig.legend(
            handles=marker_handles,
            loc="center left",
            bbox_to_anchor=(1.01, 0.18),
            frameon=False,
        )

    fig.tight_layout()

    return fig, axes


def plot_prior2_probability_by_units(
    tri_result,
    *,
    unit_col,
    target="T",
    order=None,
    title="항목별 직전 두 문장 기준 ‘-었-’ 출현 확률",
    ylabel=None,
    xlabel="직전 두 문장의 시제 배열",
    figsize=(12, 4),
    ylim=(0, 1),
    show_ci=True,
    show_base=False,              # 기본값을 False로 변경
    show_markov_expected=True,
    show_value=False,
    label_style="symbol",
):
    """
    analyze_trigram_from_weighted_df() 결과를 이용해
    유닛별 2Prior 그래프를 그린다.

    막대:
        각 유닛의 실제 조건부 확률

    회색 '_':
        각 유닛의 기본 P_T 또는 P_F

    검은 'x':
        각 유닛의 1차 연쇄 기대값
    """

    set_korean_font()

    target = target.upper()
    if target not in ["T", "F"]:
        raise ValueError("target은 'T' 또는 'F'만 가능합니다.")

    plot_df = tri_result.copy()

    if order is not None:
        plot_df[unit_col] = pd.Categorical(
            plot_df[unit_col],
            categories=order,
            ordered=True
        )
        plot_df = plot_df.sort_values(unit_col)
    else:
        plot_df = plot_df.sort_values(unit_col)

    plot_df = plot_df.reset_index(drop=True)

    prefix_order = ["FF", "TF", "FT", "TT"]

    needed_cols = [unit_col, f"P_{target}"]

    for prefix in prefix_order:
        needed_cols += [
            f"P_{target}_given_{prefix}",
            f"{prefix}T",
            f"{prefix}F",
        ]

    needed_cols += [
        f"P_{target}_given_F_pair",
        f"P_{target}_given_T_pair",
    ]

    missing = [c for c in needed_cols if c not in plot_df.columns]
    if missing:
        raise ValueError(f"필요한 컬럼이 없습니다: {missing}")

    units = plot_df[unit_col].astype(str).tolist()
    n_units = len(units)

    if ylabel is None:
        ylabel = f"다음 문장의 {'-었-' if target == 'T' else '비과거'} 출현 확률"

    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(n_units)]

    panel_info = {
        "0/2 T": ["FF"],
        "1/2 T": ["TF", "FT"],
        "2/2 T": ["TT"],
    }

    fig, axes = plt.subplots(
        1,
        3,
        figsize=figsize,
        sharey=True,
        gridspec_kw={"width_ratios": [1, 2, 1]},
        constrained_layout=True
    )

    total_width = 0.78
    bar_width = total_width / max(n_units, 1)
    offsets = (np.arange(n_units) - (n_units - 1) / 2) * bar_width

    for ax, (panel_name, prefixes) in zip(axes, panel_info.items()):
        x_groups = np.arange(len(prefixes))

        for i, unit in enumerate(units):
            s = plot_df.iloc[i]

            y_values = []
            ci_lows = []
            ci_highs = []
            base_values = []
            markov_values = []

            for prefix in prefixes:
                prob_col = f"P_{target}_given_{prefix}"
                prob = s[prob_col]

                k_col = f"{prefix}{target}"
                other = "F" if target == "T" else "T"
                other_col = f"{prefix}{other}"

                k = s[k_col]
                n = s[k_col] + s[other_col]
                ci_low, ci_high = _clopper_pearson_ci(k, n)

                base_prob = s[f"P_{target}"]

                one_back = prefix[1]
                markov_col = f"P_{target}_given_{one_back}_pair"
                markov_prob = s[markov_col]

                y_values.append(prob)
                ci_lows.append(ci_low)
                ci_highs.append(ci_high)
                base_values.append(base_prob)
                markov_values.append(markov_prob)

            xpos = x_groups + offsets[i]

            ax.bar(
                xpos,
                y_values,
                width=bar_width * 0.9,
                color=colors[i],
                label=unit,
            )

            if show_ci:
                y = np.array(y_values, dtype=float)
                ci_low_arr = np.array(ci_lows, dtype=float)
                ci_high_arr = np.array(ci_highs, dtype=float)

                yerr = np.vstack([
                    y - ci_low_arr,
                    ci_high_arr - y
                ])

                ax.errorbar(
                    xpos,
                    y,
                    yerr=yerr,
                    fmt="none",
                    ecolor="black",
                    capsize=2,
                    linewidth=0.8,
                )

            # 기본 출현률은 회색으로 약하게
            if show_base:
                ax.plot(
                    xpos,
                    base_values,
                    linestyle="none",
                    marker="_",
                    markersize=11,
                    color="gray",
                    alpha=0.7,
                )

            # 1차 연쇄 기대값은 검은 x
            if show_markov_expected:
                ax.plot(
                    xpos,
                    markov_values,
                    linestyle="none",
                    marker="x",
                    markersize=5,
                    color="black",
                )

            if show_value:
                for x, v in zip(xpos, y_values):
                    if not pd.isna(v):
                        ax.text(
                            x,
                            v + 0.015,
                            f"{v:.2f}",
                            ha="center",
                            va="bottom",
                            fontsize=8,
                            rotation=90 if n_units > 5 else 0,
                        )

        if label_style == "korean":
            label_map = {
                "FF": "비과거-비과거 뒤",
                "TF": "‘-었-’-비과거 뒤",
                "FT": "비과거-‘-었-’ 뒤",
                "TT": "‘-었-’-‘-었-’ 뒤",
            }
            xtick_labels = [label_map[p] for p in prefixes]
        else:
            xtick_labels = [f"{p}→{target}" for p in prefixes]

        ax.set_title(panel_name)
        ax.set_xticks(x_groups)
        ax.set_xticklabels(xtick_labels, rotation=35, ha="right")
        ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

    axes[0].set_ylabel(ylabel)
    fig.supxlabel(xlabel)

    if title is not None:
        fig.suptitle(title)

    unit_handles = [
        Patch(color=colors[i], label=units[i])
        for i in range(n_units)
    ]

    marker_handles = []

    if show_base:
        marker_handles.append(
            Line2D(
                [0], [0],
                color="gray",
                marker="_",
                linestyle="none",
                markersize=12,
                label=f"기본 출현률 P_{target}",
            )
        )

    if show_markov_expected:
        marker_handles.append(
            Line2D(
                [0], [0],
                color="black",
                marker="x",
                linestyle="none",
                markersize=6,
                label="1차 연쇄 기대값",
            )
        )

    legend1 = fig.legend(
        handles=unit_handles,
        title=unit_col,
        loc="center left",
        bbox_to_anchor=(1.01, 0.58),
        frameon=False,
    )

    if marker_handles:
        fig.legend(
            handles=marker_handles,
            loc="center left",
            bbox_to_anchor=(1.01, 0.18),
            frameon=False,
        )

    return fig, axes
