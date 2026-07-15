# v9 ArmoRM Circular Pipeline Archive

This directory freezes the superseded v9 circular ArmoRM pipeline pieces.

Superseded on 2026-07-15. Kept for reference only.

The old combined pipeline mixed PPO, geometry, reward collection, Wall-A, LMC,
and bundling in one line. Its numbers are contaminated per handoff v9 section 11:
the old run used PEFT `linear` adapter merging, which introduces cross terms,
and bf16 merge storage, which can swamp small endpoint differences.

The new training line is:

- `notebooks/08_train_tinyllama_helpsteer2_rs_ppo_colab.ipynb`
- `scripts/train_rs_ppo.py`

The five RS-PPO adapters are trained first. Relationship matrices are computed
later in Notebook 05.

Do not use these archived artifacts as the active training or binding-run
pipeline. Shared reusable modules remain in `src/` and are intentionally not
archived here.
