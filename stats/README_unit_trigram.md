# Unit Trigram Analysis Usage

Use `stats.unit_trigram_analysis` for the current 3-chain unit analysis. It
keeps the existing ratio and delta columns, and adds smoothed odds-ratio and
log-odds-ratio columns next to the ratio columns.

```python
from stats.unit_trigram_analysis import (
    analyze_trigram_baseline,
    analyze_unit_sequence_effect,
    analyze_unit_trigram_against_baseline,
)

baseline_tri = analyze_trigram_baseline(
    df,
    baseline_col="문서범주",
)

result = analyze_unit_trigram_against_baseline(
    df,
    baseline_col="문서범주",
    unit_col="V_form",
    baseline_df=baseline_tri,
    smoothing=0.5,  # change this if 0/1 probability handling should be stronger/weaker
)
```

For plotting, keep analysis and visualization separate:

```python
from stats.unit_trigram_plots import plot_effect_scatter

ax = plot_effect_scatter(
    result,
    x_col="unit_P_T",
    y_col="switch_return_log_odds_ratio_vs_baseline",
    group_col="문서범주",
    hline=0,
    title="Switch-return log odds ratio",
)
```
