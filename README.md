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
computation, all-method coefficient computation for M1, M2, C1, C2, P1, and
P2, narrowed M1/C1 merge evaluation, definition-style metrics, and result
summaries.

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

In Colab, the long-run notebook starts this command as a background process so
the kernel stays free for stop controls. If training is run as a normal
foreground notebook cell, widget buttons and later cells may not execute until
that training cell finishes.

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

Use control files from the repository root while training is running:

Finish the current epoch, save the current adapter and logs, then continue
with the next selected HelpSteer2 attribute:

```bash
touch STOP_CURRENT_ADAPTER
```

Finish the current epoch, save the current adapter and logs, then stop the
whole training run:

```bash
touch STOP_TRAINING
```

By default, `--max_steps` also ends only the current adapter and continues with
the next selected attribute. Add `--stop_all_on_max_steps` when `max_steps`
should stop the entire run.

The stop files are checked after each epoch. With large splits such as
`train[:1000]`, stopping can take time because the current epoch finishes
before the control file is handled.

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

Compute the current all-method coefficient table:

```bash
python scripts/compute_helpsteer2_all_method_coefficients.py
python scripts/validate_coefficient_methods.py
```

This writes:

- `results/helpsteer2_all_method_coefficients.csv`
- `results/helpsteer2_method_costs.csv`
- `results/helpsteer2_all_method_coefficients_metadata.json`

The table contains direct-preference and uniform baselines plus M1, M2, C1,
C2, P1, and P2 coefficients for the active HelpSteer2 preference vectors.
Each coefficient vector is validated to be non-negative and normalized on the
simplex.

The cost table records one row per method, preference vector, and
hyperparameter setting. `runtime_seconds` measures wall-clock coefficient
computation time, `peak_memory_mb` is a lightweight `tracemalloc` peak-memory
estimate, and `solver_iterations` is `0` for closed-form or direct mappings.
For optimizer-backed mappings where the current wrapper does not expose the
iteration count, this field is left empty.

Create thesis-style summary tables, plots, and metrics after regenerating the
current M1/C1 merge evaluation:

```bash
python scripts/summarize_helpsteer2_m1_c1_results.py
python scripts/evaluate_helpsteer2_definition_metrics.py
```

## Current Result Reports

The main HelpSteer2 experiment reports are:

- `results/helpsteer2_prototype_results.md`
- `meetings/helpsteer2_progress_summary.md`

Earlier M1/C1 proxy comparison outputs from the relationship-softmax prototype
are archived under:

- `archive/old_results/helpsteer2_m1_c1_legacy_softmax/`

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
