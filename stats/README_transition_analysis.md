# Transition Analysis Usage

`4.5.1` is now the general transition-analysis notebook. Unit-specific
workflows were separated into `4.5.2` and `stats.unit_trigram_analysis_old`.

The `smoothing` argument is used only when creating odds-ratio and
log-odds-ratio columns. Existing probability, ratio, and delta formulas are
left unchanged.

```python
from stats.transition_analysis import analyze_trigram_from_weighted_df

tri_df = analyze_trigram_from_weighted_df(
    df,
    state_col="sentence_f_EP_T",
    prev_state_col="prev_sentence_f_EP_T",
    next_state_col="next_sentence_f_EP_T",
    unit_col="docu_id",
    smoothing=0.5,
)
```

Unit-specific workflow:

```python
from stats.unit_trigram_analysis_old import (
    analyze_chain_baseline_for_unit_effect,
    analyze_unit_chain_effect,
    analyze_sequence_effect_baseline,
    analyze_unit_sequence_effect,
    analyze_trigram_baseline,
    analyze_unit_trigram_against_baseline,
)

unit_chain = analyze_unit_chain_effect(
    df,
    baseline_col="문서범주",
    unit_col="V_form",
    smoothing=0.5,
)
```

Plotting is separated into `stats.transition_plots`:

```python
from stats.transition_plots import plot_transition_effect

ax = plot_transition_effect(
    tri_df,
    x_col="P_T",
    y_col="switch_return_log_odds_ratio_vs_markov",
    group_col="문서범주",
    hline=0,
)
```
