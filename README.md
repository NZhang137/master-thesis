# Master Thesis

**Working title:** Preference-Aware Coefficient Correction for Rewarded-Soups-Style Model Merging

## Goal

The main goal of this thesis is to study mappings of the form

$$
\boldsymbol{\lambda} = f(\mathbf{p}, \mathbf{R})
$$

where $\mathbf{p}$ is a user preference vector over objectives, $\mathbf{R}$ is a relationship matrix derived from task-vector or LoRA-adapter geometry, and $\boldsymbol{\lambda}$ are corrected merge coefficients for Rewarded-Soups-style model merging.

The thesis investigates whether merge coefficients can be selected more effectively than by the direct baseline

$$
\boldsymbol{\lambda} = \mathbf{p}.
$$

The focus is not on expanding the global Pareto front, but on improving coefficient selection within a fixed Rewarded-Soups-style interpolation family.

## Current Prototype

The current prototype is a small-scale Google Colab setup using:

- GPT-2 as a lightweight base model
- LoRA / PEFT for efficient adapter training
- Anthropic HH-RLHF as the first dataset
- separate helpfulness and harmlessness adapters as the first two-objective setup
- prototype Rewarded-Soups-style adapter merging with fixed $\boldsymbol{\lambda}$ values

## Planned Extensions

After the GPT-2 infrastructure prototype, the project may move to a more
realistic small LLM setup using TinyLlama and additional multi-objective
alignment datasets such as UltraFeedback. A parallel HelpSteer2 prototype path
is documented below.

## Prototype Adapter Training

`scripts/train_hh_rlhf_adapters.py` trains separate helpfulness and
harmlessness LoRA adapters from fresh GPT-2 base models. It uses the `chosen`
HH-RLHF responses for lightweight supervised language modeling. The
corresponding `rejected` responses are not used yet. This is not full RLHF or
PPO, and it is not the final preference-aware coefficient correction method
$\boldsymbol{\lambda}=f(\mathbf{p},\mathbf{R})$.

Generated adapters and checkpoints are intentionally ignored by git.

The default settings are `gpt2`, `train[:100]`, 2 epochs, learning rate
`1e-4`, maximum sequence length 512, and batch size 1. Batch size 1 is the
default for simplicity and stability on Colab GPUs.

Fast smoke test:

```bash
python scripts/train_hh_rlhf_adapters.py --split "train[:20]" --num_epochs 1
```

Small prototype run:

```bash
python scripts/train_hh_rlhf_adapters.py --split "train[:100]" --num_epochs 2
```

Verify the generated adapters:

```bash
python scripts/check_adapters.py
```

For Google Colab, clone or open the repository, enable a GPU runtime, change
into the repository root, install `torch`, `transformers`, `datasets`, and
`peft`, and run one of the commands above. The training script does not save
intermediate checkpoints. It saves only:

- `adapters/gpt2-helpful-adapter`
- `adapters/gpt2-harmless-adapter`

These generated adapters are ignored by git. Training uses only the `chosen`
responses as lightweight supervised language-modeling text. It is not full
RLHF or PPO, and it does not yet implement the final
$\boldsymbol{\lambda}=f(\mathbf{p},\mathbf{R})$ method.

## HelpSteer2 Prototype

HelpSteer2 is the next explicitly multi-objective dataset path after the
HH-RLHF helpful/harmless prototype. The parallel training script supports the
`helpfulness`, `correctness`, `coherence`, `complexity`, and `verbosity`
ratings from `nvidia/HelpSteer2`.

Run the default Colab-friendly prototype:

```bash
python scripts/train_helpsteer2_adapters.py --split "train[:100]" --num_epochs 1
python scripts/check_helpsteer2_adapters.py
```

For each attribute, the script selects examples rated at least 3 out of 4,
sorts them by that attribute score, and uses their prompt/response text for
supervised causal language modeling. Every adapter starts from a fresh GPT-2
base model with a new LoRA adapter. The default output folders are:

- `adapters/helpsteer2-gpt2-helpfulness-adapter`
- `adapters/helpsteer2-gpt2-correctness-adapter`
- `adapters/helpsteer2-gpt2-coherence-adapter`
- `adapters/helpsteer2-gpt2-complexity-adapter`
- `adapters/helpsteer2-gpt2-verbosity-adapter`

Generated adapters remain ignored by git. This is supervised prototype
training based on attribute-rated examples, not full RLHF or PPO and not
reward-model training. It adds a richer specialist-training path but does not
replace the thesis coefficient mapping
$\boldsymbol{\lambda}=f(\mathbf{p},\mathbf{R})$.

### HelpSteer2 Adapter Merging

After all five local HelpSteer2 adapters have been trained and checked, run:

```bash
python scripts/evaluate_helpsteer2_adapter_merges.py
```

The script evaluates nine fixed five-objective coefficient vectors on four
prompts and writes:

- `results/helpsteer2_adapter_merge_generations.csv`

The adapters must already exist under `adapters/`. Generated adapter and model
files remain ignored by git, while the small result CSV may be committed. This
step only tests many-objective PEFT LoRA adapter merging. It does not yet
compute a HelpSteer2 relationship matrix \(\mathbf{R}\), does not yet apply M1
to the five-objective setup, and does not replace the existing HH-RLHF
prototype.

## Prototype Adapter Merging

After training and checking both local adapters, run:

```bash
python scripts/evaluate_adapter_merges.py
```

The script expects these directories to exist locally:

- `adapters/gpt2-helpful-adapter`
- `adapters/gpt2-harmless-adapter`

It evaluates five helpful/harmless coefficient pairs and writes 15 generated
responses to `results/adapter_merge_generations.csv`. The adapter files remain
ignored by git, while the small CSV result is not ignored.

This step only verifies fixed Rewarded-Soups-style LoRA adapter interpolation.
It does not compute the relationship matrix $\mathbf{R}$ and does not yet
implement the final
$\boldsymbol{\lambda}=f(\mathbf{p},\mathbf{R})$ correction method.

## Prototype Lambda-Sweep Evaluation

After generating `results/adapter_merge_generations.csv`, run:

```bash
python scripts/evaluate_lambda_sweep.py
```

This produces:

- `results/adapter_merge_scored_generations.csv`
- `results/lambda_sweep_summary.csv`

The script applies simple deterministic helpfulness and harmlessness proxy
heuristics, then summarizes the fixed lambda grid for three example preference
vectors. These proxies are placeholders, not final reward-model scores and not
a full RLHF evaluation. This step does not compute $\mathbf{R}$ and does not
implement the final
$\boldsymbol{\lambda}=f(\mathbf{p},\mathbf{R})$ correction method.

## Prototype Relationship Matrix

After both local adapters have been trained, run:

```bash
python scripts/compute_relationship_matrix.py
```

The script expects the Helpful and Harmless adapters under `adapters/`, whose
generated weights remain ignored by git. It writes:

- `results/relationship_matrix.csv`
- `results/relationship_matrix_metadata.json`

The small CSV and JSON result files can be committed. The matrix uses cosine
similarity between flattened LoRA adapter parameters as a static proxy for
objective or specialist relationships. This geometry proxy must be empirically
validated. This step computes $\mathbf{R}$ only and does not yet implement
$\boldsymbol{\lambda}=f(\mathbf{p},\mathbf{R})$.

## M1 Relationship-Softmax Coefficients

After computing `results/relationship_matrix.csv`, run:

```bash
python scripts/compute_m1_coefficients.py
```

This writes `results/m1_coefficients.csv` for three example preference vectors
and four correction-strength values. M1 is the first implemented
$\boldsymbol{\lambda}=f(\mathbf{p},\mathbf{R})$ mapping in the prototype. It
uses the static relationship matrix computed from LoRA adapter geometry and
does not train or modify any model.

The method performs a direct one-shot coefficient correction inside the fixed
Rewarded-Soups-style interpolation family. It does not compute or use
coefficient-space gradients, and it makes no claim of global Pareto-front
improvement.

## M1 Baseline Comparison

After computing the relationship matrix and training both local adapters, run:

```bash
python scripts/compare_m1_to_baselines.py
```

The script compares M1 at `tau=1.0` against uniform coefficients and direct
preference coefficients for the three example preference vectors. If
`results/lambda_sweep_summary.csv` exists, it also reports the best fixed-grid
utility as a reference. A positive grid gap means that the newly evaluated
candidate has a higher heuristic utility than that stored grid benchmark.

The comparison writes:

- `results/m1_baseline_generations.csv`
- `results/m1_baseline_comparison.csv`

All candidates use the same small prompt set and the same deterministic
helpfulness and harmlessness proxy heuristics as the lambda-sweep evaluation.
This remains a lightweight prototype comparison, not final reward-model
evaluation. Learned reward-model evaluation is future work, and these proxy
results do not establish global Pareto-front improvement.

## Current Prototype Results

The current matrices, coefficient corrections, baseline comparisons,
limitations, and next steps are summarized in
[`results/prototype_results.md`](results/prototype_results.md). The supporting
small CSV and JSON result files are stored under `results/`. Generated LoRA
adapters and model-weight files remain local and are ignored by git.

## Repository Structure

- `thesis/`: Proposal & LaTeX thesis draft
- `notebooks/`: Colab notebooks and experiments
- `src/`: reusable Python code
- `scripts/`: runnable prototype training scripts
- `results/`: experiment outputs, tables, and plots
- `meetings/`: meeting slides

