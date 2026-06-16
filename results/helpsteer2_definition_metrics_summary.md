# HelpSteer2 Definition-Style Evaluation Metrics

This file summarizes the metrics corresponding to the evaluation definition used in the thesis draft. The existing HelpSteer2 proxy scores are treated as normalized rewards $\tilde r_i$.

The scores are lightweight heuristic proxy scores. They are not HelpSteer2 human labels and not reward-model scores.

## Best Utility by Preference

| Preference | Best method | Setting | $U_p$ | $\Delta U_p$ vs direct | Gap to $\lambda_{\mathrm{best}}$ |
| --- | --- | --- | ---: | ---: | ---: |
| Balanced | C1 | rho=0.1 | 0.572662 | +0.139868 | 0.104578 |
| Detailed Answer | C1 | rho=0.1 | 0.569239 | +0.236272 | 0.115774 |
| Helpfulness Focused | C1 | rho=0.1 | 0.655081 | +0.040320 | 0.122800 |
| Quality Focused | C1 | rho=0.1 | 0.536672 | +0.203474 | 0.172221 |

## Best Normalized Tchebychev Score by Preference

Smaller $T_p^{\mathrm{norm}}$ values are better. A positive $\Delta T_p^{\mathrm{norm}}$ means the method reduces the worst preference-weighted objective shortfall relative to $\lambda=p$.

| Preference | Best method | Setting | $T_p^{\mathrm{norm}}$ | $\Delta T_p^{\mathrm{norm}}$ vs direct |
| --- | --- | --- | ---: | ---: |
| Balanced | C1 | rho=0.1 | 0.000000 | +0.200000 |
| Detailed Answer | C1 | rho=0.1 | 0.000000 | +0.250000 |
| Helpfulness Focused | C1 | rho=0.1 | 0.027025 | +0.072149 |
| Quality Focused | C1 | rho=0.1 | 0.000000 | +0.300000 |

## Notes

- `avg_reward` is the arithmetic mean over the five objective proxy rewards.
- `preference_weighted_utility` is $U_p=\sum_i p_i\tilde r_i$.
- `delta_utility_over_direct` compares against the `direct_preference` row for the same preference vector.
- `utility_gap_to_lambda_best` uses the fixed finite-sweep reference when available.
- `r_geometric_distance_to_p` is $(\lambda-p)^TR(\lambda-p)$ when the relationship matrix is available.
- `z_best_*` and `z_worst_*` are the per-objective benchmark values used to normalize the Tchebychev shortfalls.
- Tchebychev best and worst values are computed over the rows in the input comparison table.
