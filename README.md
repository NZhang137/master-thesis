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

After the initial GPT-2 prototype, the project may move to a more realistic small LLM setup using TinyLlama and multi-objective alignment datasets such as UltraFeedback or HelpSteer.

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

## Repository Structure

- `thesis/`: Proposal & LaTeX thesis draft
- `notebooks/`: Colab notebooks and experiments
- `src/`: reusable Python code
- `scripts/`: runnable prototype training scripts
- `results/`: experiment outputs, tables, and plots
- `meetings/`: meeting slides

