# HelpSteer2 Experiment Result Report

## Experiment Setup

This report summarizes the current HelpSteer2 proxy-evaluation experiment for preference-aware coefficient correction in Rewarded-Soups-style model merging.

- Base model: GPT-2.
- Adapter type: LoRA adapters loaded and merged with PEFT.
- Objectives: helpfulness, correctness, coherence, complexity, and verbosity.
- Relationship matrix: `results/helpsteer2_relationship_matrix.csv`, computed from flattened LoRA-adapter parameter geometry using cosine similarity.
- Coefficient sources: `results/helpsteer2_all_method_coefficients.csv` with direct preference, uniform, M1, M2, C1, C2, P1, and P2.
- Evaluation prompts: fixed prompt set with 32 prompts.
- Decoding and generation outputs: raw generations and proxy-scored generations saved under `results/`.

The completed run contains 92 coefficient rows, 2944 generated responses, 2944 scored responses, and 92 aggregate summary rows.

## Evaluation Framing

This experiment compares coefficient choices inside the same fixed Rewarded-Soups-style interpolation family. It does not claim global Pareto-front improvement. The reported scores are proxy scores; they are not final human preference scores and not external reward-model scores. If `lambda_best` is mentioned, it means the best found point under the finite tested evaluation budget rather than an oracle optimum.

## Relationship Matrix Summary

| Statistic | Value |
| --- | --- |
| Off-diagonal minimum | 0.6806 |
| Off-diagonal mean | 0.7446 |
| Off-diagonal maximum | 0.8864 |

All off-diagonal cosine similarities are positive in this run. The highest similarities occur among helpfulness, correctness, and coherence, while complexity and verbosity are still positively related but somewhat less aligned with the first three attributes. This matrix is a proxy from adapter geometry and should be validated against stronger evaluation signals.

## Main Results

### Best Tested Setting Per Preference

| Preference | Best method | Family | Hyperparameter | Mean utility | Delta vs direct | Delta vs uniform | L1 to p | L2 to p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| balanced | C2 | CAGrad-inspired | `rho_1__tau_0p2` | 0.2040 | 0.0496 | 0.0451 | 0.0632 | 0.0372 |
| detailed_answer | C2 | CAGrad-inspired | `rho_0p1__tau_0p1` | 0.2034 | 0.0459 | 0.0604 | 0.2834 | 0.1597 |
| helpfulness_focused | M2 | MGDA-inspired | `rho_0p1` | 0.3610 | 0.1301 | 0.1707 | 1.0000 | 0.5591 |
| quality_focused | M2 | MGDA-inspired | `rho_0p1` | 0.2224 | 0.0245 | 0.0438 | 0.6603 | 0.3604 |

The best tested settings are split between CAGrad-inspired C2 and MGDA-inspired M2. C2 is best for the balanced and detailed-answer preferences, while M2 with `rho_0p1` is best for the helpfulness-focused and quality-focused preferences under the current proxy utility.

### Mean Utility By Method

| Method | Mean utility across tested settings |
| --- | --- |
| P1 | 0.1995 |
| M2 | 0.1954 |
| P2 | 0.1929 |
| C1 | 0.1858 |
| C2 | 0.1857 |
| M1 | 0.1853 |
| direct_preference | 0.1852 |
| uniform | 0.1677 |

Averaged across all tested hyperparameter settings and preferences, P1 has the highest mean utility, followed by M2 and P2. This method-level average should be interpreted cautiously because it averages across several hyperparameter settings rather than selecting the best setting per preference.

### Best Setting Per Method Family

| Preference | Family | Best method | Hyperparameter | Mean utility | L1 to p |
| --- | --- | --- | --- | --- | --- |
| balanced | CAGrad-inspired | C2 | `rho_1__tau_0p2` | 0.2040 | 0.0632 |
| balanced | MGDA-inspired | M2 | `rho_0p1` | 0.1842 | 0.0510 |
| balanced | PCGrad-inspired | P1 | `beta_1` | 0.1842 | 0.0000 |
| balanced | baseline | uniform | `default` | 0.1588 | 0.0000 |
| detailed_answer | CAGrad-inspired | C2 | `rho_0p1__tau_0p1` | 0.2034 | 0.2834 |
| detailed_answer | MGDA-inspired | M2 | `rho_0p1` | 0.1821 | 0.2416 |
| detailed_answer | PCGrad-inspired | P2 | `eps_1em08__rho_1` | 0.1888 | 0.0000 |
| detailed_answer | baseline | direct_preference | `default` | 0.1575 | 0.0000 |
| helpfulness_focused | CAGrad-inspired | C2 | `rho_0p1__tau_0p2` | 0.2518 | 0.4118 |
| helpfulness_focused | MGDA-inspired | M2 | `rho_0p1` | 0.3610 | 1.0000 |
| helpfulness_focused | PCGrad-inspired | P2 | `eps_1em08__rho_0p1` | 0.2752 | 0.0000 |
| helpfulness_focused | baseline | direct_preference | `default` | 0.2309 | 0.0000 |
| quality_focused | CAGrad-inspired | C1 | `c_1__eps_1em08` | 0.2221 | 0.6609 |
| quality_focused | MGDA-inspired | M2 | `rho_0p1` | 0.2224 | 0.6603 |
| quality_focused | PCGrad-inspired | P1 | `beta_1` | 0.2188 | 0.0000 |
| quality_focused | baseline | direct_preference | `default` | 0.1979 | 0.0000 |

The CAGrad-inspired family is strongest for balanced and detailed-answer preferences. The MGDA-inspired family is strongest for helpfulness-focused and quality-focused preferences. PCGrad-inspired methods are competitive in some settings and often stay close to the original preference vector, but they are not the top utility winner for the four tested preferences.

## Distance To The Original Preference Vector

The utility gains sometimes come with substantial movement away from the original preference vector `p`. For the balanced preference, the winning C2 setting moves only modestly from `p` (L1 distance 0.0632). For the detailed-answer preference, the winning C2 setting moves more noticeably (L1 distance 0.2834). For the helpfulness-focused and quality-focused preferences, the winning M2 settings move far from `p` (L1 distances 1.0000 and 0.6603). This should be framed as a utility-vs-preference-faithfulness tradeoff, not as a universally better coefficient choice.

## Computational Cost Observations

| Method | Mean runtime (s) | Max runtime (s) | Mean peak memory (MB) |
| --- | --- | --- | --- |
| uniform | 0.000018 | 0.000024 | 0.0003 |
| direct_preference | 0.000098 | 0.000152 | 0.0009 |
| P1 | 0.000286 | 0.000431 | 0.0058 |
| P2 | 0.004979 | 0.005286 | 0.0164 |
| M2 | 0.013922 | 0.021256 | 0.0168 |
| C1 | 0.032033 | 0.041237 | 0.0204 |
| C2 | 0.041420 | 0.063176 | 0.0166 |
| M1 | 0.143592 | 1.517320 | 1.7731 |

Direct preference and uniform are essentially cost-free baselines. P1 is the cheapest proposed method in this run. P2 and M2 are also inexpensive. C1 and C2 have modest optimization cost. M1 is slower on average here, mainly due to one higher-runtime setting, but all coefficient-computation costs are small compared with model generation and adapter evaluation.

## Interpretation

- C2 looks promising for balanced or detailed-response preferences where a CAGrad-style soft-min objective can improve proxy utility with moderate movement from `p`.
- M2 looks promising for preference vectors where the proxy utility favors stronger reweighting, especially helpfulness-focused and quality-focused settings.
- P1 and P2 are attractive as cheap and comparatively preference-faithful variants, but they are weaker than the best C2/M2 settings in this run.
- The current results motivate a targeted hyperparameter round around C2 and M2, especially near the winning `rho` and `tau` settings.
- Any thesis claim should distinguish proxy-utility improvement from faithful preservation of the user preference vector.

## Limitations

- Proxy scores are deterministic heuristics, not HelpSteer2 human labels.
- Proxy scores are not external reward-model scores.
- GPT-2 is a small prototype model with limited generation quality.
- The fixed prompt set is useful for reproducibility but still small.
- Generated text quality may affect proxy scores in ways unrelated to true helpfulness, correctness, coherence, complexity, or verbosity.
- The relationship matrix is derived from LoRA-adapter geometry and remains a proxy for task-vector relationships.
- The experiment compares points inside a fixed interpolation family and does not establish global Pareto-front improvement.

## Next Step

The recommended next evaluation layer is ArmoRM or another stronger reward model / external evaluator. In parallel, a targeted hyperparameter refinement around the current promising C2 and M2 settings is reasonable, especially if the next evaluation layer confirms the same trend. The main report should continue to separate proxy-score observations from final thesis evidence.
