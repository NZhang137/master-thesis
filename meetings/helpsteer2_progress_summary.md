# HelpSteer2 Progress Summary

## 1. Current Thesis Focus

The thesis studies preference-aware coefficient correction within a fixed
Rewarded-Soups-style model-merging family:

$$
\delta_i \rightarrow R \rightarrow \lambda=f(p,R)
\rightarrow \theta(\lambda).
$$

Here, $p$ represents a user's objective preferences, $R$ represents
relationships between objective-specific specialists, and $\lambda$
determines the final merge. The contribution is coefficient selection within
the fixed interpolation family rather than expansion of the global Pareto
front.

## 2. Current Pipeline

The current HelpSteer2 prototype implements:

1. objective-specific HelpSteer2 LoRA adapters;
2. a relationship matrix $R$ from adapter-weight geometry;
3. direct-preference and the current M1, M2, C1, C2, P1, and P2 coefficient mappings;
4. PEFT-based weighted adapter merging; and
5. response generation with preference-weighted proxy evaluation.

## 3. Implemented Prototype

- **Base model:** GPT-2
- **Parameter-efficient training:** LoRA with Hugging Face PEFT
- **HelpSteer2 attributes:** helpfulness, correctness, coherence, complexity,
  and verbosity
- **Specialists:** five objective-specific adapters trained from the same base
  model
- **Relationship matrix:** cosine similarities between flattened LoRA adapter
  parameters
- **M1:** MGDA-inspired one-shot coefficient mapping
- **M2:** preference-weighted alpha-MGDA variant
- **P1:** PCGrad-inspired reconstruction with deterministic strongest-conflict ordering
- **P2:** reverse-order PCGrad-inspired reconstruction variant
- **C1:** trust-region CAGrad-inspired mapping
- **C2:** soft-min CAGrad-inspired mapping

## 4. Main Result

The current all-method coefficient table has been computed and validated for
all four preference vectors. Every reported coefficient vector is
non-negative, has the correct dimension, and sums to one.

- M1 and M2 implement the MGDA-inspired definitions from the thesis draft.
- P1 implements the R-metric PCGrad reconstruction with strongest negative
  conflict ordering.
- P2 uses the same PCGrad reconstruction equations with reverse deterministic
  ordering; it should be treated as a named variant unless the thesis draft
  adds a separate formal P2 definition.
- C1 implements the trust-region CAGrad-inspired definition with radius
  parameter $c$.
- C2 implements the soft-min CAGrad-inspired variant.

The earlier M1/C1 proxy comparison should be regenerated before being used as
current thesis evidence, because the active M1 definition is now the
MGDA-inspired mapping rather than the earlier relationship-softmax prototype.

## 5. Limitations

- The scores are lightweight heuristic proxy scores.
- They are not HelpSteer2 human labels.
- They are not reward-model scores.
- Generated responses do not automatically have HelpSteer2 labels.
- The current comparison uses only four prompts per coefficient setting and
  one recorded run.
- GPT-2 is a small prototype model with limited instruction-following ability.
- LoRA adapter geometry is only a proxy for full task-vector geometry.
- Hyperparameters are selected on the same proxy evaluation used for
  reporting.
- No global Pareto-front improvement is claimed.

## 6. Suggested Next Steps

1. Increase the number and diversity of evaluation prompts.
2. Add repeated runs or random seeds.
3. Replace or supplement the heuristics with reward models or stronger
   external evaluators.
4. Compare mappings against a more systematic finite
   $\lambda_{\mathrm{best}}$ sweep.
5. Optionally scale the prototype to TinyLlama or another stronger small
   model.
6. Evaluate the current M1, M2, P1, P2, C1, and C2 coefficients on the fixed
   prompt set.


## Supporting Results

- [Full HelpSteer2 prototype report](../results/helpsteer2_prototype_results.md)
- [Current all-method coefficients](../results/helpsteer2_all_method_coefficients.csv)
