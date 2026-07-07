"""Flexible plotting helpers for transition-analysis result dataframes."""

from __future__ import annotations

import pandas as pd


def prepare_transition_plot_data(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    group_col: str | None = None,
    label_col: str | None = None,
    min_count_col: str | None = None,
    min_count: int | float | None = None,
) -> pd.DataFrame:
    cols = [x_col, y_col]
    for col in [group_col, label_col, min_count_col]:
        if col and col not in cols:
            cols.append(col)
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing plot column(s): {missing}")

    data = df.loc[:, cols].copy()
    data[x_col] = pd.to_numeric(data[x_col], errors="coerce")
    data[y_col] = pd.to_numeric(data[y_col], errors="coerce")
    data = data.dropna(subset=[x_col, y_col])
    if min_count_col and min_count is not None:
        data[min_count_col] = pd.to_numeric(data[min_count_col], errors="coerce")
        data = data[data[min_count_col] >= min_count]
    return data


def plot_transition_effect(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    group_col: str | None = None,
    label_col: str | None = None,
    min_count_col: str | None = None,
    min_count: int | float | None = None,
    ax=None,
    figsize: tuple[float, float] = (9, 6),
    point_size: float = 35,
    alpha: float = 0.75,
    hline: float | None = None,
    vline: float | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    annotate: bool = False,
):
    """Scatter any transition probability, ratio, odds-ratio, log-odds, or delta."""
    import matplotlib.pyplot as plt

    data = prepare_transition_plot_data(
        df,
        x_col=x_col,
        y_col=y_col,
        group_col=group_col,
        label_col=label_col,
        min_count_col=min_count_col,
        min_count=min_count,
    )
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if group_col:
        for group, part in data.groupby(group_col, observed=True, dropna=False):
            ax.scatter(part[x_col], part[y_col], s=point_size, alpha=alpha, label=str(group))
        ax.legend(title=group_col)
    else:
        ax.scatter(data[x_col], data[y_col], s=point_size, alpha=alpha)

    if annotate and label_col:
        for _, row in data.iterrows():
            ax.annotate(str(row[label_col]), (row[x_col], row[y_col]), fontsize=8)
    if hline is not None:
        ax.axhline(hline, color="black", linewidth=1, linestyle="--")
    if vline is not None:
        ax.axvline(vline, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel(xlabel or x_col)
    ax.set_ylabel(ylabel or y_col)
    if title:
        ax.set_title(title)
    return ax


def plot_transition_binned_line(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    group_col: str | None = None,
    bins: int | list[float] = 10,
    agg: str = "median",
    ax=None,
    figsize: tuple[float, float] = (9, 6),
    title: str | None = None,
):
    """Plot binned aggregate lines for transition metrics."""
    import matplotlib.pyplot as plt

    data = prepare_transition_plot_data(df, x_col=x_col, y_col=y_col, group_col=group_col)
    data["_bin"] = pd.cut(data[x_col], bins=bins)
    grouped = [group_col, "_bin"] if group_col else ["_bin"]
    summary = (
        data.groupby(grouped, observed=True)
        .agg(x_mid=(x_col, "mean"), y_value=(y_col, agg), n=(y_col, "size"))
        .reset_index()
    )

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    if group_col:
        for group, part in summary.groupby(group_col, observed=True):
            ax.plot(part["x_mid"], part["y_value"], marker="o", label=str(group))
        ax.legend(title=group_col)
    else:
        ax.plot(summary["x_mid"], summary["y_value"], marker="o")
    ax.set_xlabel(x_col)
    ax.set_ylabel(f"{agg}({y_col})")
    if title:
        ax.set_title(title)
    return ax
