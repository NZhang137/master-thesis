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

1. Train five independent TinyLlama HelpSteer2 LoRA adapters.
2. Compute the TinyLlama adapter relationship matrix \(R\).
3. Compute corrected merge coefficients \(\lambda=f(p,R)\) for the target
   method set M1, M2, P1, P2, C1, C2, and C3.
4. Merge the TinyLlama adapters and generate responses on fixed prompts.
5. Score generated responses with ArmoRM.
6. Analyze preference-weighted utility, distance to \(p\), and computational
   cost.

ArmoRM monitoring in the training script is evaluation only. Its scores are
not used as an optimization signal.

Optional ArmoRM monitoring evaluates the current adapter every 200 optimizer
steps on the same first 20 prompts from
`data/evaluation_prompts/helpsteer2_reward_monitor_prompts.jsonl`. Monitoring
generation is deterministic (`do_sample=False`) and runs without gradients.
For each adapter, the logged score is the matching ArmoRM HelpSteer objective:
`helpsteer-helpfulness`, `helpsteer-correctness`, `helpsteer-coherence`,
`helpsteer-complexity`, or `helpsteer-verbosity`. It is not the global
ArmoRM preference score.
The default maximum generation length is 256 new tokens and the currently
supported reward batch size is 1. It is temporarily disabled in the Colab
training commands. To enable it again, add:

```bash
--use_armorm_monitoring \
--reward_eval_steps 200 \
--reward_monitor_num_prompts 20 \
--reward_max_new_tokens 256 \
--reward_batch_size 1 \
--reward_csv_mode overwrite
```

`--reward_csv_mode overwrite` starts clean monitoring logs for each adapter
run. Use `append` only when adding rows to an intentionally retained run.

The active coefficient implementation currently covers M1 and M2
(MGDA-inspired), P1 and P2 (PCGrad-inspired), and C1 and C2
(CAGrad-inspired). C3 is defined in the thesis method set but is not yet wired
into the coefficient-generation and validation scripts.

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

The configured training thresholds are rating `>= 4` for helpfulness,
correctness, and coherence, and rating `>= 2` for complexity and verbosity.
An explicit `min_rating` passed to the data utility still
overrides the configured value. To keep the five specialist adapters balanced,
the central config caps training selection at the best `8434` examples per
attribute. Selection is ordered by descending rating and then original dataset
row index, so each adapter uses the highest-rated available examples first.
For example, helpfulness uses its 8434 rating-4 examples, while correctness
and coherence use the first 8434 rating-4 examples under this deterministic
ordering.

## Active Commands

Run a one-attribute LoRA smoke test:

```bash
python scripts/train_tinyllama_helpsteer2_adapters.py --attributes helpfulness --split "train[:20]" --num_epochs 1 --use_tensorboard
python scripts/check_tinyllama_helpsteer2_adapters.py --attributes helpfulness
```

Train one specialist per command. Start the five attributes manually in this
order: `helpfulness`, `correctness`, `coherence`, `complexity`, and `verbosity`.
The recommended Colab workflow is
[`notebooks/02_train_tinyllama_helpsteer2_adapters_colab.ipynb`](notebooks/02_train_tinyllama_helpsteer2_adapters_colab.ipynb),
which launches each adapter as a separate background process so its stop and
status cells remain usable.
For example, start helpfulness with:

```bash
python scripts/train_tinyllama_helpsteer2_adapters.py \
  --attributes helpfulness \
  --split "train" \
  --eval_split "validation" \
  --num_epochs 5 \
  --batch_size 8 \
  --max_length 1024 \
  --learning_rate 1e-4 \
  --lr_scheduler_type cosine_decay \
  --min_lr_ratio 0.1 \
  --warmup_ratio 0.06 \
  --weight_decay 0.01 \
  --logging_steps 10 \
  --eval_steps 100 \
  --save_steps 500 \
  --use_tensorboard
```

The LoRA configuration keeps the rank fixed at `r=8` and uses
`lora_alpha=16`, which keeps adapter capacity moderate. Regularization uses
`lora_dropout=0.1` together with AdamW `weight_decay=0.01`. The first 6% of
planned optimizer steps linearly warm the learning rate from near zero to
`1e-4`. After warmup, the notebook uses `cosine_decay` to smoothly decay the
rate toward the `1e-5` floor (`min_lr_ratio=0.1`). The script itself keeps
`lr_scheduler_type=constant` as the backward-compatible post-warmup default
when no scheduler option is passed.

Training texts are shuffled once per epoch with the deterministic seed
`67 + epoch_index`. The original selected-text list is not mutated, and the
evaluation-text order remains unchanged.

Training uses response-only SFT labels: prompt tokens up to and including the
`Assistant:` marker are masked with `-100`, so the loss is computed only on the
assistant response tokens. Padding tokens are also ignored. Loss curves from
response-only runs are therefore not directly comparable to older runs that
used the full prompt-response text as labels.

After that command finishes, run the same command again with
`--attributes correctness`, then `coherence`, `complexity`, and `verbosity`.
The script still accepts multiple values after `--attributes` for compatibility,
but one attribute per command is the recommended workflow.

Compute the relationship matrix and coefficient grids after training:

```bash
python scripts/check_tinyllama_helpsteer2_adapters.py
python scripts/compute_helpsteer2_relationship_matrix.py
python scripts/compute_helpsteer2_all_method_coefficients.py
python scripts/validate_coefficient_methods.py
```

These commands currently generate and validate M1, M2, P1, P2, C1, and C2.
C3 must be implemented in the active coefficient scripts before it is included
in final coefficient grids.

The relationship matrix uses cosine similarity between effective, PEFT-scaled
LoRA updates, `delta_W = scaling * (B @ A)`. Frobenius inner products are
computed layer by layer from the low-rank factors, so full update matrices do
not need to be materialized. Raw concatenated A/B factors are not used for R.

The effective-update geometry can be checked with:

```bash
python -m unittest tests.test_relationship_utils -v
```

Validate the fixed prompt set:

```bash
python scripts/validate_helpsteer2_fixed_prompts.py
```

## Exporting Training Curves

Export TensorBoard-style loss and learning-rate curves directly from the CSV
training logs without starting a TensorBoard server:

```bash
python scripts/export_tinyllama_training_curves.py --smoothing 0.5 --theme dark
```

PNG figures and corresponding curve CSV files are written to
`results/plots/tensorboard/`. Generated PNG files should only be committed when
they are intentionally selected for a thesis chapter, report, or meeting.
Individual plots show the raw curve as a light line and an EMA-smoothed curve
as a darker line. The exporter also supports `--raw_color`, `--smoothed_color`,
`--raw_alpha`, `--raw_linewidth`, `--smoothed_linewidth`, `--grid_alpha`, and
`--dpi` for reproducible visual adjustments.

Export diagnostic ArmoRM mean-reward curves from the lightweight summary CSVs
without loading TinyLlama or ArmoRM:

```bash
python scripts/export_tinyllama_reward_monitoring_curves.py --smoothing 0.5
```

The plots are written to
`results/tinyllama_helpsteer2_reward_monitoring/plots/`.

Create `STOP_CURRENT_ADAPTER` to request a graceful stop. The script detects
the file during training, continues only until the next `save_steps` boundary,
saves the numbered checkpoint and final current adapter, removes
`STOP_CURRENT_ADAPTER`, and stops the current script run. `STOP_TRAINING` has
the same stop-at-the-next-save-step behavior, but is not removed automatically.
Choose `save_steps` based on the maximum number of additional optimizer steps
you are willing to wait before a requested stop takes effect.

TinyLlama training writes CSV logs to
`results/tinyllama_helpsteer2_training_logs/`, TensorBoard events to
`results/tensorboard/tinyllama_helpsteer2/`, and optional reward-monitoring
CSVs to `results/tinyllama_helpsteer2_reward_monitoring/`. Per-prompt rows are
stored in `tinyllama_helpsteer2_<attribute>_reward_prompts.csv`; aggregate
mean, population standard deviation, minimum, and maximum rewards are stored
in `tinyllama_helpsteer2_<attribute>_reward_summary.csv`. TensorBoard uses the
backward-compatible scalar `<attribute>/armorm_mean_reward` for the selected
attribute-specific HelpSteer objective, an explicit scalar such as
`<attribute>/armorm_helpsteer_helpfulness`, and `<attribute>/learning_rate`.

The 20-prompt ArmoRM curve is diagnostic monitoring only. It can reveal trends
during training, but it is not a statistically reliable final evaluation.
Final evaluation still uses the complete fixed evaluation prompt set and the
thesis merge-evaluation pipeline. These generated
logs, exported plots, adapters, checkpoints, model files, and zip archives
should remain outside Git unless a small result artifact is selected
intentionally for documentation.

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
