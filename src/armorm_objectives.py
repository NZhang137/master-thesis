"""Authoritative ArmoRM objective names and an EXTERNAL golden sample.

Why this file exists
--------------------
The project rule after the QRM incident was: never hard-code a reward-head index,
always read it from `config.id2label`. That rule cannot be applied to ArmoRM,
because ArmoRM does not ship an id2label. Its config.json literally contains

    "id2label":  {"0": "LABEL_0"},
    "label2id":  {"LABEL_0": 0},
    "num_objectives": 19,

i.e. the transformers default for a single-label head, while the 19 objectives are
only documented in the model card. `_name_to_index` was therefore standing on an
unverified assumption about an external artifact -- the exact bug class this project
keeps hitting. It failed loudly, which is the correct behaviour.

The replacement is NOT "hard-code 0..4 and hope". It is:

  1. take the objective ORDER from the model card (below, verbatim),
  2. assert the model agrees on the count (config.num_objectives == 19 and
     output.rewards.shape[-1] == 19),
  3. pin the mapping with a GOLDEN SAMPLE published by the model's authors --
     a number that does NOT come from our own code and therefore cannot be
     satisfied by a self-referential check,
  4. and still run the empirical degeneracy / axis-discriminance assertions
     (A3/A4) against the HelpSteer2 ground-truth labels.

Source: https://huggingface.co/RLHFlow/ArmoRM-Llama3-8B-v0.1 (model card, "Demo Code").
"""

from __future__ import annotations

# Verbatim from the model card's `attributes` list. Order IS the head order.
ARMORM_OBJECTIVES: tuple[str, ...] = (
    "helpsteer-helpfulness",            # 0
    "helpsteer-correctness",            # 1
    "helpsteer-coherence",              # 2
    "helpsteer-complexity",             # 3
    "helpsteer-verbosity",              # 4
    "ultrafeedback-overall_score",      # 5
    "ultrafeedback-instruction_following",
    "ultrafeedback-truthfulness",
    "ultrafeedback-honesty",
    "ultrafeedback-helpfulness",
    "beavertails-is_safe",
    "prometheus-score",
    "argilla-overall_quality",
    "argilla-judge_lm",
    "code-complexity",
    "code-style",
    "code-explanation",
    "code-instruction-following",
    "code-readability",                 # 18
)

N_ARMORM_OBJECTIVES = 19

# Our five HelpSteer2 axes -> the ArmoRM objective name that carries them.
ARMORM_HELPSTEER_OBJECTIVE_NAMES: dict[str, str] = {
    "helpfulness": "helpsteer-helpfulness",
    "correctness": "helpsteer-correctness",
    "coherence": "helpsteer-coherence",
    "complexity": "helpsteer-complexity",
    "verbosity": "helpsteer-verbosity",
}

# The model card states the raw HelpSteer scale is recovered by this affine map.
# It is strictly increasing, so it changes NO rank and no Spearman value; it exists
# only so the golden sample below can be compared against published numbers.
HELPSTEER_SCALE = 5.0
HELPSTEER_OFFSET = -0.5


def to_helpsteer_scale(raw_rewards):
    """Map the first five raw ArmoRM heads onto the original 0-4 HelpSteer scale."""
    return raw_rewards * HELPSTEER_SCALE + HELPSTEER_OFFSET


# --- EXTERNAL golden sample (model card, "Demo Code") -------------------------
# Published by the model's authors, not computed by us. This is the only anchor in
# the whole pipeline that a self-referential check cannot fake.
GOLDEN_SAMPLE = {
    "prompt": 'What are some synonyms for the word "beautiful"?',
    "response": (
        "Nicely, Beautifully, Handsome, Stunning, Wonderful, Gorgeous, "
        "Pretty, Stunning, Elegant"
    ),
    # multi_obj_rewards[0, :5] * 5 - 0.5, as printed in the model card
    "expected_helpsteer_rewards": (2.78125, 2.859375, 3.484375, 1.3847656, 1.296875),
    # the HelpSteer ground truth the card quotes for this example
    "helpsteer_ground_truth": (3, 3, 4, 2, 2),
    # generous: the card's numbers are bf16; we additionally run 4-bit NF4.
    "atol": 0.35,
    "source": "https://huggingface.co/RLHFlow/ArmoRM-Llama3-8B-v0.1",
}


def objective_index(name: str) -> int:
    """Return the head index of an ArmoRM objective, by NAME."""
    try:
        return ARMORM_OBJECTIVES.index(name)
    except ValueError as error:
        raise KeyError(
            f"{name!r} is not an ArmoRM objective. Known: {list(ARMORM_OBJECTIVES)}"
        ) from error


def helpsteer_head_indices(axes: list[str] | tuple[str, ...]) -> list[int]:
    """Resolve our five axes to ArmoRM head indices, by name -- never positionally."""
    return [objective_index(ARMORM_HELPSTEER_OBJECTIVE_NAMES[axis]) for axis in axes]


def assert_model_agrees(config, rewards_last_dim: int | None = None) -> None:
    """Assert the loaded model matches the objective list we are indexing into."""
    declared = getattr(config, "num_objectives", None)
    if declared is None:
        raise RuntimeError(
            "ArmoRM config has no num_objectives; this is not the expected model."
        )
    if int(declared) != N_ARMORM_OBJECTIVES:
        raise RuntimeError(
            f"ArmoRM declares num_objectives={declared}, but the objective list in "
            f"src/armorm_objectives.py has {N_ARMORM_OBJECTIVES} entries. The model "
            "changed; re-read the model card before touching any index."
        )
    if rewards_last_dim is not None and int(rewards_last_dim) != N_ARMORM_OBJECTIVES:
        raise RuntimeError(
            f"ArmoRM returned {rewards_last_dim} rewards, expected {N_ARMORM_OBJECTIVES}."
        )
