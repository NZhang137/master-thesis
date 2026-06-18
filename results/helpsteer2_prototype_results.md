# HelpSteer2 Prototype Results

## Prototype Status

This experiment is a lightweight GPT-2 + LoRA prototype for
preference-aware coefficient correction in Rewarded-Soups-style model
merging. It uses five HelpSteer2 attributes:

- helpfulness
- correctness
- coherence
- complexity
- verbosity

The prototype implements the pipeline

$$
\delta_i \rightarrow R \rightarrow \lambda=f(p,R)
\rightarrow \theta(\lambda)
$$

for a small multi-objective setting. Its purpose is to test the infrastructure,
coefficient mappings, and evaluation workflow before moving to stronger models
and evaluators.

## Pipeline Summary

The implemented HelpSteer2 workflow is:

1. Train one objective-specific LoRA adapter per attribute from the same GPT-2
   base model.
2. Merge the adapters using fixed coefficient vectors.
3. Generate responses and compute lightweight heuristic proxy scores.
4. Compute a relationship matrix $R$ from cosine similarities between
   flattened LoRA adapter parameters.
5. Compute direct-preference and the current M1, M2, C1, C2, P1, and P2
   coefficient vectors.
6. Merge the adapters using those coefficients and compare their generated
   responses.
7. Select the best tested hyperparameter for each method and preference
   according to `utility_for_preference`.

## Result Artifacts

The main result files are:

| File | Contents |
| --- | --- |
| `results/helpsteer2_adapter_merge_generations.csv` | Responses generated for the initial fixed coefficient candidates. |
| `results/helpsteer2_adapter_merge_scored_generations.csv` | Fixed-candidate responses with heuristic attribute proxy scores. |
| `results/helpsteer2_lambda_sweep_summary.csv` | Aggregate proxy scores and preference-weighted utilities for the finite sweep. |
| `results/helpsteer2_relationship_matrix.csv` | The $5 \times 5$ cosine-similarity matrix for the five LoRA adapters. |
| `results/helpsteer2_relationship_matrix_metadata.json` | Adapter paths, vector representation, and relationship-matrix metadata. |
| `results/helpsteer2_all_method_coefficients.csv` | Current direct-preference, uniform, M1, M2, C1, C2, P1, and P2 coefficients for the active preference vectors. |
| `results/helpsteer2_method_costs.csv` | Runtime, lightweight peak-memory, solver-status, and output-summary diagnostics for each coefficient computation. |
| `results/helpsteer2_all_method_coefficients_metadata.json` | Method definitions, hyperparameters, objective order, and simplex-validation metadata for the all-method coefficient table. |
| `archive/old_results/helpsteer2_m1_c1_legacy_softmax/` | Archived earlier M1/C1 comparison outputs from the relationship-softmax prototype. |

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
from 0.8132 to 0.9372, with a mean of approximately 0.8675. The strongest
pair is correctness/coherence, followed by helpfulness/correctness.
Complexity/verbosity is also strongly aligned.

These values indicate similarity between flattened LoRA parameter vectors.
They do not establish behavioral equivalence between objectives. In this
prototype, adapter geometry is a proxy for specialist relationships and
requires further empirical validation.

## Coefficient Mappings

### Direct Preference

The direct baseline preserves the user preference vector:

$$
\lambda=p.
$$

### M1: MGDA-Inspired One-Shot Mapping

M1 computes

$$
\lambda_{\mathrm{M1}}
:=
\argmin_{\lambda\in\Delta_m}
\lambda^\top R\lambda+\rho\|\lambda-p\|_2^2.
$$

The parameter $\rho$ controls how strongly the mapping stays near the original
preference vector $p$.

### C1: CAGrad-Inspired One-Shot Mapping

C1 computes a trust-region CAGrad-inspired coefficient vector:

$$
\lambda_{\mathrm{C1}}
:=
\argmax_{\lambda\in\Delta_m}\min_i(R\lambda)_i
$$

subject to

$$
(\lambda-p)^\top R(\lambda-p)\le c^2\max(p^\top Rp,\varepsilon).
$$

The parameter $c$ controls the trust-region size around $p$.

## Archived M1/C1 Proxy Comparison

The earlier M1/C1 proxy comparison has been moved to
`archive/old_results/helpsteer2_m1_c1_legacy_softmax/`. It used the earlier
relationship-softmax M1 prototype and should be treated as historical
development output rather than current thesis evidence.

The current active coefficient result is
`results/helpsteer2_all_method_coefficients.csv`, which contains the
direct-preference and uniform baselines plus M1, M2, C1, C2, P1, and P2 under
the thesis-aligned definitions.

## Fixed Lambda Sweep

The finite fixed sweep evaluates nine coefficient candidates. Its best tested
candidate, denoted $\lambda_{\mathrm{best}}$, is:

| Preference | $\lambda_{\mathrm{best}}$ | Candidate | Utility |
| --- | --- | --- | ---: |
| Balanced | [0, 1, 0, 0, 0] | One-hot correctness | 0.677240 |
| Quality focused | [0, 1, 0, 0, 0] | One-hot correctness | 0.708893 |
| Detailed answer | [0, 1, 0, 0, 0] | One-hot correctness | 0.685013 |
| Helpfulness focused | [1, 0, 0, 0, 0] | One-hot helpfulness | 0.777881 |

Here, $\lambda_{\mathrm{best}}$ means only the best of the nine tested fixed
candidates, not a global optimum. The dominance of one-hot candidates may
reflect the adapters, heuristic scoring rules, small prompt sample, or
generation variability.

## Interpretation

The current evidence supports a narrower observation: the coefficient
computation pipeline now produces validated simplex coefficients for the
direct-preference and uniform baselines plus M1, M2, C1, C2, P1, and P2 from
the HelpSteer2 relationship matrix. The next empirical step is to regenerate
merge outputs and proxy metrics for these thesis-aligned definitions.

## Definition-Style Evaluation Metrics

The earlier definition-style metric outputs were derived from the archived
M1/C1 proxy comparison and are stored under
`archive/old_results/helpsteer2_m1_c1_legacy_softmax/`. They should be
regenerated after the thesis-aligned all-method merge evaluation is run.

## Limitations

- The scores are lightweight heuristic proxy scores.
- They are not HelpSteer2 human labels.
- They are not reward-model scores.
- Generated responses do not automatically have HelpSteer2 labels.
- The comparison uses only four prompts per coefficient setting.
- The current results represent a single recorded run without multiple random
  seeds.
- Hyperparameters are selected using the same proxy evaluation reported in the
  comparison.
- GPT-2 is a small prototype model with limited instruction-following ability.
- The adapters use supervised prototype training rather than a final alignment
  method.
- Flattened LoRA adapter geometry is only a proxy for full task-vector
  geometry.
- The finite sweep covers only nine coefficient vectors.
- The results do not establish global Pareto-front improvement.

## Next Steps

1. Replace or supplement heuristic proxies with stronger reward models or
   external evaluators.
2. Increase the number and diversity of evaluation prompts.
3. Repeat training and generation with multiple random seeds.
4. Separate hyperparameter selection from final evaluation.
5. Compare methods against a more systematic finite
   $\lambda_{\mathrm{best}}$ sweep.
6. Test whether the utility/preference-faithfulness trade-off persists with
   TinyLlama or another stronger small model.
7. Optionally evaluate additional mappings such as M2, P1, P2, C2, or IC1.

The archived earlier M1/C1 comparison is retained at
`archive/old_results/helpsteer2_m1_c1_legacy_softmax/`.
