# v9 ArmoRM Circular Pipeline Archive

This directory freezes the v9 circular ArmoRM binding-run pipeline.

Superseded on 2026-07-15. Kept for reference only.

The numbers from this line are contaminated per handoff v9 section 11:
the old run used PEFT `linear` adapter merging, which introduces cross terms,
and bf16 merge storage, which can swamp small endpoint differences.

Do not use these artifacts as the active training or binding-run pipeline.
Shared reusable modules remain in `src/` and are intentionally not archived here.
