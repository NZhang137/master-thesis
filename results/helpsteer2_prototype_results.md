# HelpSteer2 Prototype Results

## Prototype Status

This experiment is a lightweight GPT-2 + LoRA prototype for preference-aware
coefficient correction in Rewarded-Soups-style model merging. It uses five
attributes from HelpSteer2:

- helpfulness
- correctness
- coherence
- complexity
- verbosity

The prototype tests the complete research workflow at small scale. Its proxy
scores and generated responses are suitable for checking the implementation,
but they are not final evidence about alignment quality.

## Pipeline Summary

The implemented HelpSteer2 pipeline is:

1. Train one objective-specific LoRA adapter per attribute from the same GPT-2
   base model.
2. Merge the adapters with fixed coefficient vectors.
3. Generate responses and assign lightweight heuristic proxy scores.
4. Compute a relationship matrix \(R\) from cosine similarities between
   flattened LoRA adapter parameters.
5. Compute direct-preference, M1, and C1 coefficient vectors.
6. Merge and compare the resulting M1/C1 models using the shared prompt and
   proxy-evaluation setup.

The final comparison stage has been implemented in code, but its output files
are not present in the repository at the time of this summary.

## Result File Status

| File | Status | Contents |
| --- | --- | --- |
| `results/helpsteer2_adapter_merge_generations.csv` | Present | 36 responses from nine fixed coefficient vectors evaluated on four prompts. |
| `results/helpsteer2_adapter_merge_scored_generations.csv` | Present | The 36 fixed-sweep responses with five heuristic attribute proxies, response length, and an empty-response flag. |
| `results/helpsteer2_lambda_sweep_summary.csv` | Present | Aggregate proxy scores and preference-weighted utilities for the nine fixed candidates. |
| `results/helpsteer2_relationship_matrix.csv` | Present | The \(5 \times 5\) cosine-similarity matrix for the five LoRA adapters. |
| `results/helpsteer2_relationship_matrix_metadata.json` | Present | Adapter paths, vector dimensions, similarity definition, and geometry caveats. |
| `results/helpsteer2_m1_c1_coefficients.csv` | Present | 28 direct-preference, M1, and C1 coefficient settings for four preference vectors. |
| `results/helpsteer2_m1_c1_coefficients_metadata.json` | Present | Objective order, preferences, hyperparameters, optimizer settings, and PSD handling. |
| `results/helpsteer2_m1_c1_merge_generations.csv` | Missing | Expected generated responses for uniform, direct-preference, M1, and C1 merges. |
| `results/helpsteer2_m1_c1_scored_generations.csv` | Missing | Expected proxy scores for the M1/C1 comparison responses. |
| `results/helpsteer2_m1_c1_comparison.csv` | Missing | Expected aggregate utilities and method comparisons. |
| `results/helpsteer2_m1_c1_comparison_metadata.json` | Missing | Expected metadata for the response-level M1/C1 comparison. |

The missing files can be produced after the local adapters are available by
running:

```bash
python scripts/evaluate_helpsteer2_m1_c1_merges.py
```

## Relationship Matrix

The measured relationship matrix is:

| Adapter | Helpfulness | Correctness | Coherence | Complexity | Verbosity |
| --- | ---: | ---: | ---: | ---: | ---: |
| Helpfulness | 1.0000 | 0.9306 | 0.9015 | 0.8568 | 0.8396 |
| Correctness | 0.9306 | 1.0000 | 0.9372 | 0.8467 | 0.8275 |
| Coherence | 0.9015 | 0.9372 | 1.0000 | 0.8132 | 0.8136 |
| Complexity | 0.8568 | 0.8467 | 0.8132 | 1.0000 | 0.9080 |
| Verbosity | 0.8396 | 0.8275 | 0.8136 | 0.9080 | 1.0000 |

All off-diagonal similarities are positive and relatively high. They range
from 0.8132 to 0.9372, with a mean of 0.8675. The strongest pair is
correctness/coherence (0.9372), followed by helpfulness/correctness (0.9306).
Complexity/verbosity is also strongly aligned (0.9080). The weakest pairs are
coherence/complexity (0.8132) and coherence/verbosity (0.8136).

This pattern suggests that the five learned adapter parameter vectors occupy a
closely related region of the prototype's LoRA parameter space. It does not by
itself establish that the objectives are behaviorally equivalent: cosine
similarity between flattened adapter parameters is only a proxy for specialist
relationships.

## Coefficient Mappings

The coefficient table evaluates four user preference vectors and three
settings for each correction method.

### Direct Preference

The direct baseline uses

\[
\boldsymbol{\lambda} = \mathbf{p}.
\]

It preserves the user preference vector exactly.

### M1: Relationship-Softmax

M1 computes

\[
\mathbf{s} = R\mathbf{p},
\qquad
\lambda_i =
\frac{p_i\exp(\tau s_i)}
{\sum_k p_k\exp(\tau s_k)}.
\]

The parameter \(\tau\) controls the correction strength. Larger values place
more emphasis on the relationship scores. In the current coefficient table,
M1 remains close to the original preferences: its L1 distance ranges from
0.0052 to 0.0622 and its L2 distance from 0.0026 to 0.0358. The displacement
increases consistently as \(\tau\) increases from 0.5 to 2.0.

### C1: CAGrad-Inspired One-Shot Mapping

C1 solves a simplex-constrained, one-shot optimization that balances the worst
relationship score against a quadratic penalty for moving away from
\(\mathbf{p}\). The parameter \(\rho\) controls that penalty: larger values
keep the result closer to the original preference vector.

The current C1 coefficients move farther than M1. Across all reported settings,
their L1 distances range from 0.0732 to 1.0089 and their L2 distances from
0.0420 to 0.5516. The largest changes occur at \(\rho=0.1\); at \(\rho=10\),
the L1 distance is approximately 0.0732 for all four preferences. These are
coefficient-space observations, not response-quality results.

## M1/C1 Comparison

The response-level comparison file
`results/helpsteer2_m1_c1_comparison.csv` is currently missing. Therefore, this
summary cannot yet identify:

- the best method for each preference vector by `utility_for_preference`;
- whether M1 improves on direct preference in generated-response utility;
- whether C1 improves on direct preference in generated-response utility; or
- whether those comparisons vary across preference vectors.

The available coefficient table only shows how the methods transform
\(\mathbf{p}\). It shows that M1 makes small, temperature-dependent
corrections, while C1 can make substantially larger changes when its proximity
penalty is weak. Running the merge-comparison script is required before making
claims about which mapping performs better under the prototype proxy scores.

## Fixed Lambda Sweep

The fixed sweep evaluates nine candidate coefficient vectors. Under the
lightweight proxy utilities, the best tested candidate
\(\lambda_{\mathrm{best}}\) is:

| Preference | \(\lambda_{\mathrm{best}}\) | Candidate | Utility |
| --- | --- | --- | ---: |
| Balanced | [0, 1, 0, 0, 0] | One-hot correctness | 0.677240 |
| Quality focused | [0, 1, 0, 0, 0] | One-hot correctness | 0.708893 |
| Detailed answer | [0, 1, 0, 0, 0] | One-hot correctness | 0.685013 |
| Helpfulness focused | [1, 0, 0, 0, 0] | One-hot helpfulness | 0.777881 |

Here, \(\lambda_{\mathrm{best}}\) means the best of the nine tested fixed
candidates, not a global optimum. The concentration on one-hot candidates may
reflect the adapters, the small four-prompt sample, the heuristic scoring
rules, or generation randomness. It should not be interpreted as a general
conclusion about multi-objective merging.

## Limitations

- The proxy scores are lightweight heuristic scores.
- They are not HelpSteer2 human labels.
- They are not reward-model scores.
- Generated responses do not automatically inherit HelpSteer2 labels.
- Each fixed candidate is evaluated on only four prompts.
- GPT-2 is a small prototype model with limited instruction-following ability.
- The adapters were trained with supervised prototype procedures rather than
  a final alignment method.
- LoRA adapter geometry is only a proxy for full task-vector geometry.
- Only one recorded run is available; random-seed variability is not measured.
- The response-level M1/C1 comparison artifacts are currently missing.
- The method does not claim improvement of the global Pareto front.

## Next Steps

1. Run the M1/C1 merge evaluation and add the missing comparison artifacts.
2. Replace or supplement heuristic proxies with stronger reward models or
   external evaluators.
3. Increase the number and diversity of evaluation prompts.
4. Repeat training and generation with multiple random seeds.
5. Compare each mapping against \(\lambda_{\mathrm{best}}\) from a clearly
   defined finite sweep.
6. Optionally scale the prototype to TinyLlama or another stronger small model.
7. Optionally add M2, P1, P2, C2, or IC1 in later experiments.
