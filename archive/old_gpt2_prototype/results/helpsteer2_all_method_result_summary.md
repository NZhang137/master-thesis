# HelpSteer2 All-Method Result Summary

This report summarizes the full HelpSteer2 all-method adapter-merge evaluation. All coefficient rows were evaluated on the same fixed prompt set inside the fixed Rewarded-Soups-style interpolation family.

The scores are lightweight proxy scores. They are not HelpSteer2 human labels, not reward-model scores, and should not be interpreted as final thesis evidence.

## Best Tested Setting by Preference

| Preference | Best method | Hyperparameter | Mean utility | Improvement over direct | Improvement over uniform |
| --- | --- | --- | ---: | ---: | ---: |
| balanced | C2 | `rho_1__tau_0p2` | 0.203964 | 0.049563 | 0.045137 |
| detailed_answer | C2 | `rho_0p1__tau_0p1` | 0.203445 | 0.045907 | 0.06036 |
| helpfulness_focused | M2 | `rho_0p1` | 0.361037 | 0.130107 | 0.170707 |
| quality_focused | M2 | `rho_0p1` | 0.222406 | 0.024459 | 0.043832 |

## Evaluation Files

- `results/helpsteer2_all_method_generations.csv`: raw generated responses.
- `results/helpsteer2_all_method_scores.csv`: generated responses with proxy scores and utility.
- `results/helpsteer2_all_method_result_summary.csv`: aggregate utility and method comparisons.
- `results/helpsteer2_all_method_result_summary.json`: machine-readable run metadata and best settings.

## Limitations

- Proxy scores are surface-level heuristics.
- Generated responses do not automatically have HelpSteer2 labels.
- The evaluation does not establish global Pareto-front improvement.
- Stronger reward-model or human evaluation remains future work.
