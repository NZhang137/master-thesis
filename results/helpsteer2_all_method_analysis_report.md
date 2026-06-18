# HelpSteer2 All-Method Analysis Report

This compact report summarizes the completed HelpSteer2 all-method merge evaluation with table-focused comparisons and chart outputs. The comparison is inside the fixed Rewarded-Soups-style interpolation family.

All quality values are proxy scores. They are not HelpSteer2 human labels, not external reward-model scores, and do not establish global Pareto-front improvement.

## Output Tables

- `results/helpsteer2_method_comparison_table.csv`: combined quality, distance-to-preference, and cost table by method.
- `results/helpsteer2_method_cost_summary.csv`: runtime and memory summary by method.
- `results/helpsteer2_method_performance_summary.csv`: proxy utility and preference-distance summary by method.

## Best Row Per Preference

| preference_name | method | hyperparameter_id | mean_utility | improvement_over_direct_preference | improvement_over_uniform | l1_distance_to_p | l2_distance_to_p | runtime_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| balanced | C2 | rho_1__tau_0p2 | 0.203964 | 0.049563 | 0.045137 | 0.063205 | 0.037208 | 0.03620119 |
| detailed_answer | C2 | rho_0p1__tau_0p1 | 0.203445 | 0.045907 | 0.060360 | 0.283397 | 0.159666 | 0.04784052 |
| helpfulness_focused | M2 | rho_0p1 | 0.361037 | 0.130107 | 0.170707 | 1.000000 | 0.559055 | 0.02081196 |
| quality_focused | M2 | rho_0p1 | 0.222406 | 0.024459 | 0.043832 | 0.660288 | 0.360353 | 0.02125615 |

## Method Comparison Table

| method | method_family | num_runs | mean_utility | std_utility | best_utility | mean_improvement_over_direct | mean_improvement_over_uniform | mean_l1_distance_to_p | mean_l2_distance_to_p | mean_runtime_seconds | mean_peak_memory_mb | mean_solver_iterations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P1 | PCGrad-inspired | 12 | 0.199467 | 0.035944 | 0.263294 | 0.014263 | 0.031763 | 0.000000 | 0.000000 | 0.00028640 | 0.005844 | 0.000000 |
| M2 | MGDA-inspired | 12 | 0.195386 | 0.062071 | 0.361037 | 0.010182 | 0.027682 | 0.188155 | 0.102521 | 0.01392225 | 0.016827 |  |
| P2 | PCGrad-inspired | 12 | 0.192874 | 0.048340 | 0.275246 | 0.007670 | 0.025170 | 0.000000 | 0.000000 | 0.00497863 | 0.016439 |  |
| C1 | CAGrad-inspired | 12 | 0.185840 | 0.029296 | 0.225417 | 0.000636 | 0.018136 | 0.465223 | 0.241785 | 0.03203260 | 0.020415 |  |
| C2 | CAGrad-inspired | 24 | 0.185679 | 0.031296 | 0.251827 | 0.000475 | 0.017975 | 0.227218 | 0.122170 | 0.04142040 | 0.016551 |  |
| M1 | MGDA-inspired | 12 | 0.185263 | 0.036472 | 0.263271 | 0.000059 | 0.017559 | 0.176781 | 0.087637 | 0.14359159 | 1.773107 |  |
| direct_preference | baseline | 4 | 0.185204 | 0.036366 | 0.230930 | 0.000000 | 0.017500 | 0.000000 | 0.000000 | 0.00009832 | 0.000884 | 0.000000 |
| uniform | baseline | 4 | 0.167704 | 0.020936 | 0.190330 | -0.017500 | 0.000000 | 0.300000 | 0.159779 | 0.00001830 | 0.000292 | 0.000000 |

## Computational Cost Summary

| method | mean_runtime_seconds | total_runtime_seconds | mean_peak_memory_mb | mean_solver_iterations | num_runs |
| --- | --- | --- | --- | --- | --- |
| uniform | 0.00001830 | 0.00007320 | 0.000292 | 0.000000 | 4 |
| direct_preference | 0.00009832 | 0.00039328 | 0.000884 | 0.000000 | 4 |
| P1 | 0.00028640 | 0.00343680 | 0.005844 | 0.000000 | 12 |
| P2 | 0.00497863 | 0.05974361 | 0.016439 |  | 12 |
| M2 | 0.01392225 | 0.16706699 | 0.016827 |  | 12 |
| C1 | 0.03203260 | 0.38439123 | 0.020415 |  | 12 |
| C2 | 0.04142040 | 0.99408961 | 0.016551 |  | 24 |
| M1 | 0.14359159 | 1.72309914 | 1.773107 |  | 12 |

## Performance Summary

| method | mean_utility | std_utility | best_utility | mean_improvement_over_direct | mean_improvement_over_uniform | mean_l1_distance_to_p | mean_l2_distance_to_p | num_runs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P1 | 0.199467 | 0.035944 | 0.263294 | 0.014263 | 0.031763 | 0.000000 | 0.000000 | 12 |
| M2 | 0.195386 | 0.062071 | 0.361037 | 0.010182 | 0.027682 | 0.188155 | 0.102521 | 12 |
| P2 | 0.192874 | 0.048340 | 0.275246 | 0.007670 | 0.025170 | 0.000000 | 0.000000 | 12 |
| C1 | 0.185840 | 0.029296 | 0.225417 | 0.000636 | 0.018136 | 0.465223 | 0.241785 | 12 |
| C2 | 0.185679 | 0.031296 | 0.251827 | 0.000475 | 0.017975 | 0.227218 | 0.122170 | 24 |
| M1 | 0.185263 | 0.036472 | 0.263271 | 0.000059 | 0.017559 | 0.176781 | 0.087637 | 12 |
| direct_preference | 0.185204 | 0.036366 | 0.230930 | 0.000000 | 0.017500 | 0.000000 | 0.000000 | 4 |
| uniform | 0.167704 | 0.020936 | 0.190330 | -0.017500 | 0.000000 | 0.300000 | 0.159779 | 4 |

## Chart Files

- `results/plots/helpsteer2_method_mean_utility_bar.png`: mean proxy utility by method.
- `results/plots/helpsteer2_method_runtime_bar.png`: mean coefficient-computation runtime by method.
- `results/plots/helpsteer2_method_utility_vs_cost_bar.png`: normalized mean utility and normalized mean runtime by method.

## Interpretation

- `P1` has the highest average proxy utility across all tested settings in this aggregate method view.
- The best individual settings per preference are split between `C2` and `M2`, so the promising method family depends on the preference vector.
- `P1` is the cheapest non-baseline method in the coefficient-computation cost table.
- The cheapest methods overall are uniform, direct_preference, P1, but direct and uniform are baselines rather than correction methods.
- Stronger proxy utility can come with larger movement away from the original preference vector `p`; this should be treated as a utility-vs-preference-faithfulness tradeoff.
- Another targeted hyperparameter round is most useful around the currently promising `C2` and `M2` settings.

## Limitations

- Proxy scores are deterministic heuristics and may not reflect real HelpSteer2 human preferences.
- GPT-2 is a small prototype model and can produce low-quality generations.
- The fixed prompt set improves reproducibility but is still limited.
- Runtime numbers measure coefficient computation, not the full generation cost.
- The results compare choices inside one fixed interpolation family and do not show global Pareto-front improvement.
