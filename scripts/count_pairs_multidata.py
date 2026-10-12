"""Count the NB16 DPO pairs of all 14 experts, as NB12 did (no training, no Llama download).

For every expert this prints, with exactly the rules of ``scripts/0_dpo_expert_multidata.py``:

* ``candidate_pairs``: response pairs of the same prompt whose values on the objective differ;
* ``eligible_prompts``: prompts with at least one such pair. With one pair per prompt (NB12
  rule) this is the largest possible N for the expert, and the smallest value within a
  dataset is the largest common N of that dataset.

HelpSteer1 uses the NB12 revision and must reproduce the NB12 counts (checked). The humor
count depends on the humor margin (default 0.5).

Colab (any runtime; a GPU only speeds up the humor classifier):

    !git clone https://github.com/NZhang137/master-thesis.git /content/master-thesis
    %cd /content/master-thesis
    !python scripts/count_pairs_multidata.py --out results/nb16_pair_counts.csv
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().with_name("0_dpo_expert_multidata.py")
_spec = importlib.util.spec_from_file_location("nb16_dpo_expert_multidata", _SCRIPT)
assert _spec is not None and _spec.loader is not None
multi = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(multi)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default="nb16_pair_counts.csv")
    parser.add_argument("--humor_margins", default="0.5", help="comma-separated, e.g. 0.2,0.5")
    parser.add_argument("--skip_humor", action="store_true")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    args = parser.parse_args()
    margins = [float(m) for m in args.humor_margins.split(",") if m.strip()]

    import pandas as pd

    rows_out = []
    for key in ("hh_rlhf", "pku_saferlhf", "helpsteer1", "ultrafeedback"):
        revision = multi.PINNED_REVISIONS.get(key)
        cache = {}
        for objective in multi.objectives(key):
            subset = multi.HH_SUBSET[objective] if key == "hh_rlhf" else None
            if subset not in cache:
                cache[subset] = multi.load_objective_rows(key, objective, revision)
            rows = cache[subset]
            if objective == "humor":
                if args.skip_humor:
                    continue
                import torch
                device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
                texts = multi.humor_texts(rows)
                print(f"[humor] scoring {len(texts):,} responses on {device} ...", flush=True)
                scores = multi.score_humor(texts, device=device, batch_size=256 if device == "cuda" else 64)
                for margin in margins:
                    pairs = multi.build_pairs(key, objective, rows, humor_scores=scores, humor_margin=margin)
                    rows_out.append({"dataset": key, "objective": f"humor (margin {margin})", "rows": len(rows),
                                     "candidate_pairs": len(pairs), "eligible_prompts": multi.eligible_prompts(pairs)})
                continue
            pairs = multi.build_pairs(key, objective, rows)
            row = {"dataset": key, "objective": objective, "rows": len(rows),
                   "candidate_pairs": len(pairs), "eligible_prompts": multi.eligible_prompts(pairs)}
            rows_out.append(row)
            print(f"[pairs] {key}/{objective}: {row['candidate_pairs']:,} pairs, "
                  f"{row['eligible_prompts']:,} eligible prompts", flush=True)
            if key == "helpsteer1":
                expected = (multi.NB12_CANDIDATES[objective], multi.NB12_ELIGIBLE_PROMPTS[objective])
                assert (row["candidate_pairs"], row["eligible_prompts"]) == expected, \
                    f"HelpSteer1 {objective} differs from NB12: {row} vs {expected}"
        cache.clear()

    table = pd.DataFrame(rows_out)
    table["max_common_N_of_dataset"] = table.groupby("dataset")["eligible_prompts"].transform("min")
    table.to_csv(args.out, index=False)
    with pd.option_context("display.width", 200):
        print("\n" + table.to_string(index=False))
    print("\nHelpSteer1 reproduces the NB12 counts (checked).")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
