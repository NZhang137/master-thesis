# HelpSteer2 M1/C1 Result Summary

## Comparison

The comparison evaluates uniform coefficients, direct preference (\(\lambda=p\)), M1 relationship-softmax correction, and the C1 CAGrad-inspired one-shot mapping. For M1 and C1, the table selects the hyperparameter setting with the highest `utility_for_preference` for each preference vector.

## Best Method by Preference

| Preference | Best method | Setting | Proxy utility | Improvement over direct | Improvement over uniform |
| --- | --- | --- | ---: | ---: | ---: |
| Balanced | C1 | rho=0.1 | 0.572662 | +0.139868 | +0.139868 |
| Quality Focused | C1 | rho=0.1 | 0.536672 | +0.203474 | +0.127731 |
| Detailed Answer | C1 | rho=0.1 | 0.569239 | +0.236272 | +0.135256 |
| Helpfulness Focused | C1 | rho=0.1 | 0.655081 | +0.040320 | +0.158163 |

## M1 and C1 Relative to Direct Preference

- **Balanced:** best M1 (tau=2) changes proxy utility by +0.018528; best C1 (rho=0.1) changes it by +0.139868 relative to direct preference.
- **Quality Focused:** best M1 (tau=2) changes proxy utility by +0.039566; best C1 (rho=0.1) changes it by +0.203474 relative to direct preference.
- **Detailed Answer:** best M1 (tau=0.5) changes proxy utility by +0.000000; best C1 (rho=0.1) changes it by +0.236272 relative to direct preference.
- **Helpfulness Focused:** best M1 (tau=0.5) changes proxy utility by -0.003049; best C1 (rho=0.1) changes it by +0.040320 relative to direct preference.

## Distance from the Original Preference

The selected M1 settings remain close to the original preference vectors (L1 distance 0.0093-0.0308; L2 distance 0.0046-0.0150). The selected C1 settings move farther (L1 distance 0.4089-1.0089; L2 distance 0.2013-0.5516). This describes coefficient displacement only; it is not evidence of general model quality.

## Limitations

- These are lightweight heuristic proxy scores.
- They are not HelpSteer2 human labels.
- They are not reward-model scores.
- Generated responses do not automatically have HelpSteer2 labels.
- The comparison uses a small prompt set and a single recorded run.
- Hyperparameters are selected on the same proxy evaluation used for reporting.
- Any finite-sweep reference should be described as \(\lambda_{\mathrm{best}}\), not as an oracle.
- The results do not establish improvement of the global Pareto front.
