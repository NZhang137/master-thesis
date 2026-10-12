"""Count the NB16 DPO candidate pairs of all 14 experts (no training, no Llama download).

For every expert this prints the number of preference pairs whose two responses have
different values on the expert's objective, using exactly the pair rules of
``scripts/0_dpo_expert_multidata.py``. For the HH-RLHF humor expert the count depends on the
humor margin, so it is reported for several margins.

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

MARGINS = (0.05, 0.1, 0.2, 0.3, 0.5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default="nb16_pair_counts.csv")
    parser.add_argument("--revision", default=None, help="dataset revision for all datasets (default: latest)")
    parser.add_argument("--skip_humor", action="store_true")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    args = parser.parse_args()

    import numpy as np
    import pandas as pd

    rows_out = []
    for key in ("hh_rlhf", "pku_saferlhf", "helpsteer1", "ultrafeedback"):
        cache = {}
        for objective in multi.objectives(key):
            subset = multi.HH_SUBSET[objective] if key == "hh_rlhf" else None
            if subset not in cache:
                cache[subset] = multi.load_objective_rows(key, objective, args.revision)
            rows = cache[subset]
            if objective == "humor":
                if args.skip_humor:
                    continue
                import torch
                device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
                texts = multi.humor_texts(rows)
                print(f"[humor] scoring {len(texts):,} responses on {device} ...", flush=True)
                scores = multi.score_humor(texts, device=device, batch_size=256 if device == "cuda" else 64)
                triples, _ = multi.parse_hh_rows(rows)
                diff = np.array([abs(scores[multi.text_sha256(a)] - scores[multi.text_sha256(b)]) for _, a, b in triples])
                for margin in MARGINS:
                    count = len(multi.build_pairs(key, objective, rows, humor_scores=scores, humor_margin=margin))
                    rows_out.append({"dataset": key, "objective": f"humor (margin {margin})", "rows": len(rows),
                                     "pairs_with_different_values": count})
                print(f"[humor] responses with P(HUMOR) > 0.5: {np.mean(np.array(list(scores.values())) > 0.5):.1%}; "
                      f"median |difference| within a pair: {np.median(diff):.3f}", flush=True)
                continue
            count = len(multi.build_pairs(key, objective, rows))
            rows_out.append({"dataset": key, "objective": objective, "rows": len(rows),
                             "pairs_with_different_values": count})
            print(f"[pairs] {key}/{objective}: {count:,}", flush=True)
        cache.clear()

    table = pd.DataFrame(rows_out)
    table.to_csv(args.out, index=False)
    with pd.option_context("display.width", 200):
        print("\n" + table.to_string(index=False))
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
