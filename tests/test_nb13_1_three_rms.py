from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "13.1_method_comparison_helpsteer2_dpo_three_rms_colab.ipynb"


def test_nb13_1_is_parseable_and_has_three_frozen_evaluators() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) == 40
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    first_code = "".join(notebook["cells"][2]["source"])
    assert first_code.index('drive.mount("/content/drive"') < first_code.index("repo_path")
    assert "RLHFlow/ArmoRM-Llama3-8B-v0.1" in source
    assert "LxzGordon/URM-LLaMa-3.1-8B" in source
    assert "nvidia/Llama2-13B-SteerLM-RM" in source
    assert "tritonclient[http]==2.60.0" in source
    assert "nvidia-pytriton" not in source
    assert "RUN_STEERLM = False" in source
    assert "if RUN_REWARD_COLLECTION and RUN_STEERLM" in source
    assert "set(RM_TENSORS) == REQUIRED_THIS_PASS" in source
    assert "provisional_steerlm_deferred" in source
    assert '"three_rm_protocol_complete": THREE_RM_COMPLETE' in source
    assert "3792c4d6bcd68c0917729b04f31168e953db2f9eb2a12059fa942fe1b256a9c4" in source
    assert "answer_cache_sha256" in source
    assert "raw_scores_pooled_across_reward_models\": False" in source
    assert "should_exclude_cert_from_phase_b" in source
    assert 'lam_df_all["method"] != "Cert"' in source
    assert '"Cert" not in set(final_df["method"])' in source
    assert '"Cert" not in set(stats_df["method"])' in source
    assert "p_holm_within_rm" in source and "p_holm_global" in source

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        assert cell.get("outputs", []) == []
        transformed = []
        for line in "".join(cell.get("source", [])).splitlines():
            stripped = line.lstrip()
            if stripped.startswith(("!", "%")):
                transformed.append(line[: len(line) - len(stripped)] + "pass")
            else:
                transformed.append(line)
        ast.parse("\n".join(transformed), filename=f"NB13.1 cell {index}")
