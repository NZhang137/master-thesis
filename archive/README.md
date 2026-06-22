# Archive Index

The archive retains superseded prototype files for reproducibility and project
history. Archived code and outputs are not part of the active TinyLlama +
HelpSteer2 + ArmoRM experiment.

## GPT-2 HelpSteer2 Prototype

`old_gpt2_prototype/` contains the completed GPT-2 many-objective prototype:

- `scripts/`: GPT-2 training, PEFT merging, proxy scoring, and M1/C1 analysis
- `notebooks/`: Colab runners for the GPT-2 HelpSteer2 workflow
- `results/`: generations, heuristic proxy scores, coefficient tables,
  reports, metadata, and plots
- `src/`: GPT-2-specific model/merge helpers and legacy proxy utilities
- `meetings/`: supervisor notes describing the GPT-2 prototype stage

These files document the pipeline that preceded the final TinyLlama
experiment. Their scores are heuristic prototype scores, not ArmoRM results.

## HH-RLHF Prototype

`old_hh_rlhf_prototype/` contains the earlier two-objective helpful/harmless
prototype:

- `scripts/`: HH-RLHF adapter training, merging, sweep, and baseline scripts
- `notebooks/`: the original GPT-2 + HH-RLHF Colab workflow
- `results/`: the corresponding two-objective generations and summaries
- `src/`: HH-RLHF data and heuristic-scoring helpers

## Other Archive Folders

- `old_presentations/`: superseded presentation drafts, when present
- `misc/`: historical files that do not belong to a specific prototype
- `old_notebooks/`, `old_results/`, and `old_scripts/`: compatibility buckets
  retained from the earlier archive layout

No thesis files are archived. Generated adapters, checkpoints, model weights,
and zip backups remain excluded from version control rather than stored here.
