# Master Thesis

**Working title:** Preference-Aware Coefficient Correction for Rewarded-Soups-Style Model Merging

## Goal

The thesis studies post-hoc coefficient selection for Rewarded-Soups-style
model merging. The main pipeline is:

$$
\delta_i \rightarrow R \rightarrow \lambda = f(p, R) \rightarrow \theta(\lambda)
$$

where \(p\) is a user preference vector over objectives, \(R\) is a
relationship matrix derived from task-vector or LoRA-adapter geometry, and
\(\lambda\) contains corrected merge coefficients inside a fixed interpolation
family.

The focus is better coefficient selection inside the fixed model-merging
family, not a claim of global Pareto-front improvement.

## Current Active Pipeline

The current active experiment is the GPT-2 + HelpSteer2 pipeline. The workflow
is:

1. Train HelpSteer2 GPT-2 LoRA adapters with long-run support.
2. Compute the relationship matrix \(R\) from adapter geometry.
3. Compute coefficient mappings M1, M2, C1, C2, P1, and P2.
4. Evaluate all methods on a fixed prompt set.
5. Report proxy scores carefully.
6. Later add stronger reward-model evaluation, for example with ArmoRM.

The currently implemented HelpSteer2 scripts cover long-running adapter
training, fixed adapter merging, proxy scoring, relationship-matrix
computation, M1/C1 coefficient computation, M1/C1 merge evaluation,
definition-style metrics, and result summaries. Additional mappings such as
M2, C2, P1, and P2 are part of the active method family and can be added beside
the existing M1/C1 code.

Proxy scores are lightweight deterministic evaluation aids. They are not
HelpSteer2 human labels, not reward-model scores, and should not be interpreted
as final thesis evidence.

## Active HelpSteer2 Commands

Train the five HelpSteer2 adapters:

```bash
python scripts/train_helpsteer2_adapters.py --split "train[:100]" --num_epochs 1
python scripts/check_helpsteer2_adapters.py
```

Run a longer Colab training job with live `eval_loss` monitoring:

```bash
python scripts/train_helpsteer2_adapters.py --split "train[:1000]" --num_epochs 100 --logging_steps 10 --eval_steps 100 --use_tensorboard
```

If Colab reports an incompatible `torchao` version, run:

```bash
pip install -U "torchao>=0.16.0"
```

Then restart the runtime before starting training.

Open TensorBoard in Colab:

```python
%load_ext tensorboard
%tensorboard --logdir results/tensorboard/helpsteer2
```

To stop cleanly after the current epoch, create the stop file from the
repository root while training is running:

```bash
touch STOP_TRAINING
```

The trainer writes:

- CSV logs under `results/training_logs/`
- TensorBoard event logs under `results/tensorboard/helpsteer2/`
- checkpoints under `checkpoints/helpsteer2/`
- final adapter folders under `adapters/`

Evaluate fixed HelpSteer2 adapter merges and proxy scores:

```bash
python scripts/evaluate_helpsteer2_adapter_merges.py
python scripts/evaluate_helpsteer2_lambda_sweep.py
```

Compute the HelpSteer2 relationship matrix:

```bash
python scripts/compute_helpsteer2_relationship_matrix.py
```

Compute and evaluate M1/C1 coefficients:

```bash
python scripts/compute_helpsteer2_m1_c1_coefficients.py
python scripts/evaluate_helpsteer2_m1_c1_merges.py
```

Create thesis-style summary tables, plots, and metrics:

```bash
python scripts/summarize_helpsteer2_m1_c1_results.py
python scripts/evaluate_helpsteer2_definition_metrics.py
```

## Current Result Reports

The main HelpSteer2 experiment reports are:

- `results/helpsteer2_prototype_results.md`
- `results/helpsteer2_m1_c1_result_summary.md`
- `results/helpsteer2_definition_metrics_summary.md`
- `meetings/helpsteer2_progress_summary.md`

Small CSV, JSON, Markdown, and plot outputs under `results/` are intended for
experiment documentation. Generated adapters, checkpoints, TensorBoard event
files, zip files, and model weights stay local and are ignored by git.

## Active Repository Structure

- `scripts/`: active runnable HelpSteer2 prototype scripts
- `src/`: reusable Python utilities
- `notebooks/`: active Colab runner notebooks
- `results/`: current HelpSteer2 result files, tables, summaries, and plots
- `thesis/`: LaTeX thesis material
- `meetings/`: supervisor-facing notes and progress summaries
- `archive/old_notebooks/`: historical notebooks
- `archive/old_scripts/`: historical scripts
- `archive/old_results/`: historical result files

If `data/evaluation_prompts/` or `configs/` are added later, they should hold
shared prompt sets and reusable experiment settings for the active pipeline.

## Archived Prototype Files

Older HH-RLHF and early GPT-2 notebooks, scripts, and result files are kept
under `archive/`. They are historical prototypes from the two-objective
helpful/harmless phase and are no longer the main active workflow.

The active experiment should use the HelpSteer2 scripts and notebooks in the
main `scripts/` and `notebooks/` folders.

## Git Safety

Generated artifacts must not be committed:

- `adapters/`
- `checkpoints/`
- `*.safetensors`
- `*.bin`
- `*.pt`
- `*.pth`
- `*.zip`
- `wandb/`
- `__pycache__/`
- `.ipynb_checkpoints/`

Small result files in `results/` may be committed when they document an
experiment run.
