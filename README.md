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

The final experiment direction is **TinyLlama + HelpSteer2 + ArmoRM**:

1. Train five independent TinyLlama HelpSteer2 LoRA/QLoRA adapters.
2. Compute the TinyLlama adapter relationship matrix \(R\).
3. Compute M1, M2, C1, C2, P1, and P2 coefficients
   \(\lambda=f(p,R)\).
4. Merge the TinyLlama adapters and generate responses on fixed prompts.
5. Score generated responses with ArmoRM.
6. Analyze preference-weighted utility, distance to \(p\), and computational
   cost.

ArmoRM monitoring in the training script is evaluation only. Its scores are
not used as an optimization signal.

## Current Final Experiment

TinyLlama + HelpSteer2 + ArmoRM is the current final experiment direction.
The earlier GPT-2 outputs are prototype results retained in the archive. New
scripts should use
`configs/tinyllama_helpsteer2_armorm.yaml` as their central source for model
names, attribute order, preference vectors, and experiment paths wherever
practical.

Validate the central configuration with:

```bash
python scripts/validate_tinyllama_helpsteer2_config.py
```

## Inspecting HelpSteer2

Inspect the configured attributes, rating distributions, and deterministic
high-rated training-text selections before adapter training:

```bash
python scripts/inspect_helpsteer2_dataset.py --split "train[:1000]"
```

The command writes a compact, reproducible summary to
`results/helpsteer2_dataset_summary.json`.

The configured training threshold is rating `>= 3` for helpfulness,
correctness, coherence, and complexity. Verbosity uses rating `>= 2` because
ratings of 3 and 4 are less frequent for that attribute. An explicit
`min_rating` passed to the data utility still overrides the configured value.

## Active Commands

Run a one-attribute 4-bit smoke test:

```bash
python scripts/train_tinyllama_helpsteer2_adapters.py --attributes helpfulness --split "train[:20]" --num_epochs 1 --use_4bit --use_tensorboard
python scripts/check_tinyllama_helpsteer2_adapters.py --attributes helpfulness
```

Train all five specialists:

```bash
python scripts/train_tinyllama_helpsteer2_adapters.py --split "train[:1000]" --num_epochs 100 --learning_rate 1e-4 --max_length 512 --batch_size 1 --logging_steps 10 --eval_steps 100 --save_steps 500 --use_4bit --use_tensorboard
```

Compute the relationship matrix and coefficient grids after training:

```bash
python scripts/check_tinyllama_helpsteer2_adapters.py
python scripts/compute_helpsteer2_relationship_matrix.py
python scripts/compute_helpsteer2_all_method_coefficients.py
python scripts/validate_coefficient_methods.py
```

Validate the fixed prompt set:

```bash
python scripts/validate_helpsteer2_fixed_prompts.py
```

Create `STOP_CURRENT_ADAPTER` to finish the current epoch, save the current
specialist, and continue with the next attribute. Create `STOP_TRAINING` to
finish the current epoch, save the specialist, and stop the complete run.

TinyLlama training writes CSV logs to
`results/tinyllama_helpsteer2_training_logs/`, TensorBoard events to
`results/tensorboard/tinyllama_helpsteer2/`, and optional reward-monitoring
CSVs to `results/tinyllama_helpsteer2_reward_monitoring/`.

The dedicated TinyLlama merged-generation, ArmoRM scoring, and final analysis
scripts are the next implementation stage. Archived GPT-2 evaluation scripts
must not be used as if they produced TinyLlama results.

## Current Outputs

The active `results/` folder is reserved for the final TinyLlama experiment.
Historical GPT-2 result tables, generations, reports, and plots are under
`archive/old_gpt2_prototype/results/`.

## Active Repository Structure

- `configs/`: reusable settings for final experiments
- `scripts/`: active TinyLlama training, geometry, coefficient, and validation scripts
- `src/`: reusable TinyLlama, HelpSteer2, geometry, and coefficient utilities
- `notebooks/`: active Colab runner notebooks
- `data/evaluation_prompts/`: fixed evaluation and monitoring prompts
- `results/`: outputs from the final TinyLlama experiment
- `thesis/`: LaTeX thesis material
- `meetings/`: current supervisor notes
- `archive/`: historical GPT-2 and HH-RLHF prototypes

## Archived Prototype Files

The completed GPT-2 + HelpSteer2 experiment is stored under
`archive/old_gpt2_prototype/`. The earlier two-objective HH-RLHF experiment is
stored under `archive/old_hh_rlhf_prototype/`. Both are retained for history
and reproducibility but are not part of the active TinyLlama experiment. See
[`archive/README.md`](archive/README.md) for the archive index.

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
