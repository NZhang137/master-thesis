# Prototype Results

## Prototype Status

This repository currently contains a lightweight GPT-2 + LoRA prototype for
preference-aware coefficient correction. The first two-objective setup uses
the `helpful-base` and `harmless-base` subsets of Anthropic HH-RLHF to train
separate helpful and harmless adapters from the same GPT-2 base model.

The adapters are trained with supervised language modeling on `chosen`
responses. This is not full RLHF or PPO training, and the `rejected` responses
are not used yet. Evaluation currently relies on deterministic heuristic proxy
scores to test the end-to-end pipeline; these scores are not final
reward-model evaluation.

## Pipeline Summary

The implemented prototype performs the following steps:

1. Train separate helpful and harmless LoRA adapters.
2. Merge the adapters with a fixed grid of coefficient vectors
   \(\lambda\).
3. Generate responses for a small shared prompt set.
4. Compute heuristic helpfulness and harmlessness proxy scores.
5. Compute a relationship matrix \(R\) from flattened LoRA adapter geometry.
6. Compute corrected coefficients with the M1 mapping
   \(\lambda=f(p,R)\).
7. Compare M1 with uniform coefficients, direct preference coefficients, and
   the best result found in the fixed-grid sweep.

## Result Files

- `results/adapter_merge_generations.csv`: 15 generated responses covering
  five fixed coefficient pairs and three prompts.
- `results/adapter_merge_scored_generations.csv`: the same 15 responses with
  heuristic helpfulness, harmlessness, response-length, and empty-response
  fields.
- `results/lambda_sweep_summary.csv`: aggregate proxy scores and
  preference-weighted utilities for the five fixed coefficient pairs.
- `results/relationship_matrix.csv`: the labeled \(2\times2\) cosine
  relationship matrix for the helpful and harmless adapters.
- `results/relationship_matrix_metadata.json`: adapter paths, representation,
  vector lengths, similarity type, and caveats for the relationship matrix.
- `results/m1_coefficients.csv`: M1 coefficients for three preference vectors
  and four values of \(\tau\), together with relationship scores and distances
  from the original preferences.
- `results/m1_baseline_comparison.csv`: aggregate proxy scores and utilities
  for uniform, direct-preference, and M1 coefficients, with the best fixed-grid
  utility included as a reference.
- `results/m1_baseline_generations.csv`: response-level generations and proxy
  scores underlying the baseline comparison.

## Relationship Matrix \(R\)

The current cosine relationship matrix is:

| Adapter | Helpful | Harmless |
|---|---:|---:|
| Helpful | 1.000000 | 0.930816 |
| Harmless | 0.930816 | 1.000000 |

The diagonal values are 1.0, as expected when each nonzero adapter vector is
compared with itself using cosine similarity. The off-diagonal value of
approximately 0.931 indicates high positive geometric similarity between the
flattened helpful and harmless LoRA parameters in this prototype.

This value is only a parameter-geometry proxy for objective or specialist
relationships. It does not by itself establish behavioral compatibility,
conflict, redundancy, or transfer, and it must be empirically validated.

## M1 Coefficient Mapping

M1 applies the relationship-softmax correction

$$
\mathrm{scores}=Rp,
$$

$$
\lambda_i =
\frac{p_i\exp(\tau\,\mathrm{scores}_i)}
{\sum_k p_k\exp(\tau\,\mathrm{scores}_k)}.
$$

Here, \(p\) is the user preference vector, \(R\) is the adapter relationship
matrix, \(\lambda\) is the corrected merge coefficient vector, and \(\tau\)
controls the correction strength. Setting \(\tau=0\) recovers
\(\lambda=p\).

With the current symmetric matrix and \(\tau=1\):

| Preference \(p\) | M1 coefficient \(\lambda\) | L1 distance from \(p\) |
|---|---|---:|
| \([0.5, 0.5]\) | \([0.5, 0.5]\) | 0.000000 |
| \([0.8, 0.2]\) | \([0.806559, 0.193441]\) | 0.013118 |
| \([0.2, 0.8]\) | \([0.193441, 0.806559]\) | 0.013118 |

The current relationship matrix therefore leaves the balanced preference
unchanged and slightly strengthens the larger component of each unbalanced
preference.

## Baseline Comparison

The comparison uses the same three prompts and heuristic proxy scoring for all
newly generated candidates. The fixed-grid column is the best stored utility
from the earlier five-point lambda sweep.

| Preference | Uniform | Direct preference | M1 (\(\tau=1\)) | Best fixed grid |
|---|---:|---:|---:|---:|
| \([0.5, 0.5]\) | 0.708333 | 0.708333 | 0.708333 | 0.708333 |
| \([0.8, 0.2]\) | 0.743333 | 0.755000 | 0.755000 | 0.743333 |
| \([0.2, 0.8]\) | 0.673333 | 0.665556 | 0.665556 | 0.673333 |

In this prototype run, M1 and direct preference received identical heuristic
utilities for all three preferences, despite the small M1 coefficient changes
for the unbalanced cases. For \(p=[0.8,0.2]\), both methods scored about
0.0117 above the best stored fixed-grid utility. For \(p=[0.2,0.8]\), both
scored about 0.0078 below the best stored fixed-grid utility. For the balanced
preference, all methods use or effectively recover \([0.5,0.5]\) and have the
same utility.

Using heuristic proxy scores, this run does not show an advantage of M1 over
the direct-preference baseline. The result is preliminary and may reflect the
small model, small prompt set, coarse proxy metrics, stochastic generation,
and the high similarity between the two adapter vectors. It should not be
interpreted as general evidence for or against the method.

## Limitations

- GPT-2 is used only as a small infrastructure prototype.
- Adapter training is supervised training on `chosen` responses, not full
  RLHF or PPO.
- The original `rejected` responses are not used.
- The helpfulness and harmlessness scores are heuristic proxies, not learned
  reward-model scores.
- The relationship matrix derived from LoRA parameter geometry is a proxy that
  requires behavioral validation.
- Only two objectives and three evaluation prompts are tested.
- The fixed lambda sweep is coarse and does not represent a continuous oracle.
- These results are preliminary and should not be treated as final thesis
  evidence or as evidence of global Pareto-front improvement.

## Next Steps

- Replace the heuristic proxies with appropriate reward models.
- Extend the experiments to more than two objectives.
- Test TinyLlama or another stronger small language model.
- Evaluate datasets such as UltraFeedback or HelpSteer.
- Improve M1 or compare it with additional one-shot relationship-aware
  mappings inspired by MGDA and CAGrad while retaining the fixed interpolation
  family.
- Increase the prompt set, repeat generation across seeds, and report
  uncertainty.
- Prepare the next meeting presentation.
