# HelpSteer2 Progress Summary

## 1. Current Thesis Focus

The thesis studies preference-aware coefficient correction within a fixed
Rewarded-Soups-style model-merging family:

\[
\delta_i \rightarrow R \rightarrow \lambda=f(p,R)
\rightarrow \theta(\lambda).
\]

Here, \(p\) represents a user's objective preferences, \(R\) represents
relationships between objective-specific specialists, and \(\lambda\)
determines the final merge. The contribution is coefficient selection within
the fixed interpolation family rather than expansion of the global Pareto
front.

## 2. Current Pipeline

The current HelpSteer2 prototype implements:

1. objective-specific HelpSteer2 LoRA adapters;
2. a relationship matrix \(R\) from adapter-weight geometry;
3. direct-preference, M1, and C1 coefficient mappings;
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
- **M1:** relationship-softmax correction
- **C1:** CAGrad-inspired one-shot simplex mapping with a preference-distance
  penalty

## 4. Main Result

For each preference and method, the current analysis selects the
hyperparameter setting with the highest `utility_for_preference`.

- C1 with \(\rho=0.1\) has the highest heuristic proxy utility among uniform,
  direct preference, M1, and C1 for all four tested preference vectors.
- In this run, the selected C1 setting improves over direct preference for all
  four preferences.
- M1 remains much closer to the original preference vector \(p\). Its selected
  L1 distances range from 0.0093 to 0.0308.
- C1 moves substantially farther from \(p\). Its selected L1 distances range
  from 0.4089 to 1.0089.

The current finding is therefore best framed as a
**proxy-utility versus preference-faithfulness trade-off**. C1 obtains higher
heuristic utility in this prototype run, while M1 preserves the stated
preference more closely. This does not establish that C1 is generally better.

The best finite-sweep candidate, denoted \(\lambda_{\mathrm{best}}\), still has
higher proxy utility than the selected C1 setting for each tested preference.
This is a reference to the best of nine tested candidates, not a global
optimum.

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
   \(\lambda_{\mathrm{best}}\) sweep.
5. Optionally scale the prototype to TinyLlama or another stronger small
   model.
6. Later, optionally evaluate M2, P1, P2, C2, or IC1.

## 7. Questions for the Supervisor

1. Is HelpSteer2 suitable as the main many-objective benchmark, or should it be
   complemented by another dataset?
2. Is the C1 formulation appropriate as a main thesis method despite the
   observed movement away from \(p\)?
3. Should the next priority be stronger evaluation, more repetitions, or
   scaling to a stronger base model?
4. Are M1 and C1 sufficient for the first thesis version, or should a
   PCGrad-inspired variant be included as an additional comparison?
5. How prominently should the current heuristic proxy evaluation be presented
   in the thesis before reward-model or external-evaluator results are
   available?

## Supporting Results

- [Full HelpSteer2 prototype report](../results/helpsteer2_prototype_results.md)
- [Concise M1/C1 result summary](../results/helpsteer2_m1_c1_result_summary.md)
