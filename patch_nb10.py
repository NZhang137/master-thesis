"""Patch NB10 into the single runnable Phase B notebook.

Apply to the PRISTINE `10_method_comparison_colab.ipynb`. Every edit is anchored
on cell content, never on an index, and asserts a unique match; a notebook that
differs from the reviewed version aborts the run instead of being half-patched.
Re-running is safe.

Edits
-----
  5a+   provenance guard: preference source, regime, adapter paths, R matrices
  5b    Phase-B preference subset, stated as a rule
  6b+   evaluation prompt file built in-notebook, excluding earlier prompt sets
  6c    merge points restricted to the Phase-B subset
  7-    binding record: sha256 over prompts, adapters, matrices, scorer, model
  7     pre-registration carries both preference counts and the binding record
  7     gate verifies the full binding record, not only the lambda table
  8     adapters resolved from the RS-PPO layout, regime asserted
  8     generation via apply_chat_template, identical to NB08
  8     ArmoRM 8-bit scorer wired; per-prompt rewards retained
  9c    bootstrap, Mean Rank, Selection Regret, Holm - BEFORE the export cell
  11    open-items section rewritten to the actual state

Usage:
    python patch_nb10.py --notebook notebooks/10_method_comparison_colab.ipynb
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


# =============================================================================
# 5a+  provenance guard
# =============================================================================

MD_PROVENANCE = '''### 5a+. Provenienz — Regime, Adapter, Matrizen

Der teuerste stille Fehler dieses Notebooks waere, ein Adapterregime zu mergen und mit der
Geometrie eines anderen zu korrigieren. Das Notebook liefe durch und produzierte bedeutungslose
Zahlen. Diese Zelle laesst das scheitern, bevor GPU-Zeit verbraucht wird.
'''

CODE_PROVENANCE = '''REGIME = "rs_ppo"          # "rs_ppo" oder "sft" — bestimmt Adapter UND Matrizen

# 8-bit spiegelt exakt den Belohnungspfad des PPO-Laufs (ppo_log: armorm_precision=8bit).
# Das ist die Voraussetzung des Obergrenzen-Arguments: der Pruefer muss derselbe sein wie
# das Trainingssignal. bf16 waere hoeherpraezise, aber dann eine bewusst abweichende
# Evaluation — dann hier "bfloat16" eintragen UND es in Kapitel 6 als Abweichung berichten.
ARMORM_PRECISION = "8bit"

# (1) Die Praeferenzen duerfen nur aus einer Quelle kommen. Config und src/preferences.py
#     sind zwei Kopien; weichen sie ab, misst das Notebook etwas anderes als der Rest
#     des Projekts.
from src.preferences import PREFERENCES as PREFERENCES_MODULE

_config_names, _module_names = set(PREFERENCES), set(PREFERENCES_MODULE)
if _config_names != _module_names:
    raise AssertionError(
        "Praeferenzen weichen ab.\\n"
        f"  nur in der Config:          {sorted(_config_names - _module_names)}\\n"
        f"  nur in src/preferences.py:  {sorted(_module_names - _config_names)}\\n"
        "Eine der beiden Kopien ist veraltet. Angleichen, bevor irgendetwas laeuft."
    )
for _name in sorted(_config_names):
    if not np.allclose(np.asarray(PREFERENCES[_name], float),
                       np.asarray(PREFERENCES_MODULE[_name], float), atol=1e-12):
        raise AssertionError(f"Praeferenz {_name!r} hat in Config und Modul verschiedene Werte.")
print(f"[OK] Praeferenzen konsistent: {len(_config_names)} Eintraege, {sorted(_config_names)}")

# (2) Adapter aufloesen. Die RS-PPO-Laeufe liegen unter den NB08-Pfaden, die SFT-Adapter
#     unter adapters/. Es wird NICHT geraten: passt kein Layout vollstaendig, bricht es ab.
ADAPTER_LAYOUTS = {
    "rs_ppo": [
        "results/rs_ppo_armorm_circular/rs_runs/ppo_{axis}/adapter",
        "results/rs_ppo_armorm_circular/rs_runs/ppo_{axis}",
    ],
    "sft": [
        str(config.get("adapter_dir", "adapters")) + "/tinyllama-helpsteer2-{axis}-adapter",
        "adapters/tinyllama-helpsteer2-{axis}-adapter",
    ],
}

def _resolve_adapters(regime):
    for pattern in ADAPTER_LAYOUTS[regime]:
        paths = {a: (PROJECT_ROOT / pattern.format(axis=a)).resolve() for a in ATTRIBUTES}
        if all((p / "adapter_config.json").is_file() for p in paths.values()):
            return pattern, paths
    tried = "\\n".join("  " + p for p in ADAPTER_LAYOUTS[regime])
    raise FileNotFoundError(
        f"Keine vollstaendige Adaptermenge fuer Regime {regime!r} gefunden. Geprueft:\\n{tried}\\n"
        "Pfad korrigieren, statt das Notebook auf ein anderes Regime ausweichen zu lassen."
    )

ADAPTER_PATTERN, ADAPTER_PATHS = _resolve_adapters(REGIME)
print(f"[OK] Adapter ({REGIME}): {ADAPTER_PATTERN}")
for _a, _p in ADAPTER_PATHS.items():
    print(f"       {_a:12s} {_p.relative_to(PROJECT_ROOT)}")

# (3) Matrizen regime-abhaengig aufloesen. Der Dateiname ist KEIN Nachweis: der
#     bisherige Pfad results/tinyllama_helpsteer2_R/ sagt nicht, aus welchem Lauf die
#     Matrix stammt. Deshalb wird sie hier explizit ueber das Regime gewaehlt und ihr
#     Hash in den Bindungsnachweis aufgenommen.
MATRIX_LAYOUTS = {
    "rs_ppo": [
        ("results/nb09_1_geometry_run1/R_cos.csv",
         "results/nb09_1_geometry_run1/R_gram.csv"),
        ("results/rs_ppo_armorm_circular/geometry/R_cos.csv",
         "results/rs_ppo_armorm_circular/geometry/R_gram.csv"),
        ("results/rs_ppo_armorm_circular/nb09_1_R_cos.csv",
         "results/rs_ppo_armorm_circular/nb09_1_R_gram.csv"),
    ],
    "sft": [
        ("results/tinyllama_helpsteer2_R/tinyllama_helpsteer2_R_cos.csv",
         "results/tinyllama_helpsteer2_R/tinyllama_helpsteer2_R_gram.csv"),
    ],
}

def _resolve_matrices(regime):
    for cos_rel, gram_rel in MATRIX_LAYOUTS[regime]:
        cos_path, gram_path = PROJECT_ROOT / cos_rel, PROJECT_ROOT / gram_rel
        if cos_path.is_file() and gram_path.is_file():
            return cos_path.resolve(), gram_path.resolve()
    tried = "\\n".join(f"  {c}\\n  {g}" for c, g in MATRIX_LAYOUTS[regime])
    raise FileNotFoundError(
        f"Keine R-Matrizen fuer Regime {regime!r} gefunden. Geprueft:\\n{tried}\\n"
        "Die aus NB09.1 berechneten RS-PPO-Matrizen dorthin kopieren, statt auf einen "
        "Pfad auszuweichen, dessen Regime der Name nicht ausweist."
    )

COSINE_MATRIX_CSV, GRAM_MATRIX_CSV = _resolve_matrices(REGIME)
R_cos = load_labeled_matrix_csv(COSINE_MATRIX_CSV, ATTRIBUTES)
R_gram = load_labeled_matrix_csv(GRAM_MATRIX_CSV, ATTRIBUTES)
R = R_cos if PRIMARY_MATRIX == "R_cos" else R_gram
_eig = np.linalg.eigvalsh(R)
assert _eig.min() > 0, f"R ist nicht positiv definit (min EW {_eig.min():.3e})."
print(f"[OK] Matrizen ({REGIME}): {COSINE_MATRIX_CSV.relative_to(PROJECT_ROOT)}")
print(f"       dominanter Anteil {_eig.max() / _eig.sum() * 100:.2f} % "
      f"(RS-PPO erwartet ~35.1 %, SFT ~46 %)")

eigenvalues = _eig
offdiag = R[np.triu_indices(m, 1)]
print(f"Attribute: {list(ATTRIBUTES)}")
print(f"Eigenwerte {PRIMARY_MATRIX}: {np.round(eigenvalues, 4)}")
print(f"Off-Diagonalen: {offdiag.min():.4f} .. {offdiag.max():.4f} "
      f"(Mittel {offdiag.mean():.4f})")

# Pi0(R 1) != 0 ist die Bedingung, unter der Fair dem Kollaps entkommt (Prop 26, v13.2).
# Bei perfekt aequikorreliertem R waere sie verletzt und Fair gaebe p zurueck.
escape = R @ np.ones(m)
escape = escape - escape.mean()
print(f"||Pi_0(R 1)|| = {np.linalg.norm(escape):.6f}  "
      f"({'Fair kann entkommen' if np.linalg.norm(escape) > 1e-9 else 'DEGENERIERT: Fair kollabiert'})")

display(pd.DataFrame(R, index=list(ATTRIBUTES), columns=list(ATTRIBUTES)).round(4))
'''


OLD_CFG_MATRIX = """R_cos  = load_labeled_matrix_csv(COSINE_MATRIX_CSV, ATTRIBUTES)
R_gram = load_labeled_matrix_csv(GRAM_MATRIX_CSV, ATTRIBUTES)
R      = R_cos if PRIMARY_MATRIX == "R_cos" else R_gram

eigenvalues = np.linalg.eigvalsh(R)
assert eigenvalues.min() > 0, f"R ist nicht positiv definit (min EW {eigenvalues.min():.3e})."

offdiag = R[np.triu_indices(m, 1)]
print(f"Attribute: {list(ATTRIBUTES)}")
print(f"Eigenwerte {PRIMARY_MATRIX}: {np.round(eigenvalues, 4)}")
print(f"Dominanter Anteil: {eigenvalues.max() / eigenvalues.sum() * 100:.2f} %")
print(f"Off-Diagonalen: {offdiag.min():.4f} .. {offdiag.max():.4f} (Mittel {offdiag.mean():.4f})")

# Pi0(R 1) != 0 ist die Bedingung, unter der Fair dem Kollaps entkommt (Prop 26, v13.2).
# Bei perfekt aequikorreliertem R waere sie verletzt und Fair gaebe p zurueck.
escape = R @ np.ones(m)
escape = escape - escape.mean()
print(f"||Pi_0(R 1)|| = {np.linalg.norm(escape):.6f}  "
      f"({'Fair kann entkommen' if np.linalg.norm(escape) > 1e-9 else 'DEGENERIERT: Fair kollabiert'})")

display(pd.DataFrame(R, index=list(ATTRIBUTES), columns=list(ATTRIBUTES)).round(4))
"""

NEW_CFG_MATRIX = """# Die Matrizen werden in Abschnitt 5a+ regime-abhaengig aufgeloest und geladen. Hier
# NICHT vorab laden: die Pfadkonstanten aus Abschnitt 2 weisen kein Regime aus.
R_cos = R_gram = R = None
eigenvalues = None
"""


# =============================================================================
# 5b  Phase-B preference subset
# =============================================================================

MD_PHASE_B = '''### 5b. Phase-B-Teilmenge

Phase A ist reward-frei und darf breit laufen. Phase B kostet GPU-Stunden und ArmoRM-Kontakt.
Die Teilmenge ist als REGEL formuliert, nicht als Liste, damit sie nicht nachtraeglich
waehlbar ist.
'''

CODE_PHASE_B = '''PHASE_B_RULE = "all named preferences from the experiment config; no Dirichlet draws"
PREF_SET_B = [(name, p) for name, p in PREF_SET if name in PREFERENCES]
PHASE_B_NAMES = {name for name, _ in PREF_SET_B}

assert len(PREF_SET_B) == len(PREFERENCES), (
    f"PREF_SET_B hat {len(PREF_SET_B)} Eintraege, PREFERENCES {len(PREFERENCES)}."
)
assert PHASE_B_NAMES, "Phase-B-Praeferenzmenge ist leer."
print(f"Phase A: |P| = {len(PREF_SET)}    Phase B: |P_B| = {len(PREF_SET_B)}")
print("Phase B:", sorted(PHASE_B_NAMES))
'''


# =============================================================================
# 6b+  build the evaluation prompt file
# =============================================================================

MD_PROMPTS = '''### 6b+. Evaluationsprompts

Die Prompts entstehen im Notebook, damit `n_prompts` in der Vorregistrierung nicht von einem
separat ausgefuehrten Skript abhaengt. Gezogen wird aus dem VALIDATION-Split — den TRAIN-Split
haben die RS-PPO-Adapter in ihren PPO-Schritten gesehen. Prompttexte aus frueheren
Promptdateien werden ausgeschlossen, und die Zusammenfassung berichtet, wie viele es waren.
'''

CODE_PROMPTS = '''from src.eval_prompts import build_eval_prompt_file, ensure_nb06_prompt_files

N_EVAL_PROMPTS = 80
EVAL_PROMPT_SEED = 137

# Fail-closed: diese Dateien MUESSEN gefunden werden, sonst bricht der Bau ab. Prompts zu
# ziehen, weil eine Ausschlussdatei nicht auffindbar war, ist genau der Fehler, den dieser
# Waechter verhindern soll. Fehlt eine laut Projektverlauf wirklich, bewusst
# ALLOW_MISSING_EXCLUSIONS = True setzen UND die Luecke berichten.
ALLOW_MISSING_EXCLUSIONS = False

# In einem frischen Clone fehlen die beiden historischen NB06-Dateien. Sie werden
# mit der eingefrorenen NB06.1-Regel (Validation-Split, Seed 137, 80+80)
# reproduziert; vorhandene Dateien werden geprueft und nie ueberschrieben.
NB06_PROMPT_SUMMARY = ensure_nb06_prompt_files(
    PROJECT_ROOT,
    dataset_name=str(config["dataset_name"]),
    split="validation",
    seed=EVAL_PROMPT_SEED,
    n_per_set=80,
)
print(json.dumps(NB06_PROMPT_SUMMARY, indent=2, ensure_ascii=False))

REWARD_PROMPT_PATH.parent.mkdir(parents=True, exist_ok=True)
PROMPT_SUMMARY = build_eval_prompt_file(
    REWARD_PROMPT_PATH,
    n=N_EVAL_PROMPTS,
    seed=EVAL_PROMPT_SEED,
    project_root=PROJECT_ROOT,
    allow_missing_exclusions=ALLOW_MISSING_EXCLUSIONS,
)
print(json.dumps(PROMPT_SUMMARY, indent=2, ensure_ascii=False))
if not PROMPT_SUMMARY.get("disjointness_verified", False):
    print("\\nHINWEIS: Disjunktheit zu frueheren Evaluationen ist NICHT nachgewiesen. "
          "So berichten, nicht anders.")
'''


# =============================================================================
# 2  preference count: 11 named, 75 search points
# =============================================================================

OLD_FLAG = """USE_FULL_77 = False

PREF_SET = [(name, np.asarray(vec, dtype=np.float64)) for name, vec in PREFERENCES.items()]

if USE_FULL_77:"""

NEW_FLAG = """# 11 benannte Praeferenzen + 64 Dirichlet-Punkte = 75. Die 77 aus NB09.1 sind
# historisch: sie enthielten quality_focused und detailed_answer, die entfernt wurden.
USE_FULL_SEARCH_SET = True

PREF_SET = [(name, np.asarray(vec, dtype=np.float64)) for name, vec in PREFERENCES.items()]

if USE_FULL_SEARCH_SET:"""

OLD_PREF_MD = """Die Config enthaelt bereits alle 13 vorregistrierten Praeferenzen, einschliesslich der fuenf
Vertices (`only_*`) und `balanced` als uniform. Genau diese 13 bilden in NB09.1 die
vorregistrierte Teilmenge der 77. Die vollen 77 lassen sich zuschalten, sprengen in Phase B
aber die Sitzung."""

NEW_PREF_MD = """Die Config enthaelt die **11** benannten Praeferenzen: fuenf Vertices (`only_*`), fuenf
dominante und `balanced`. Mit 64 Dirichlet-Punkten ergibt der volle Suchsatz **75** Punkte.

NB09.1 registrierte 13 Praeferenzen und 77 Punkte vor; `quality_focused` und
`detailed_answer` wurden seither entfernt und `uniform` in `balanced` umbenannt. Wo im
Notebook 13 oder 77 steht, ist das eine historische NB09.1-Angabe. Die Abweichung ist in der
Vorregistrierung als solche vermerkt und gehoert so berichtet.

`USE_FULL_SEARCH_SET` schaltet Phase A auf alle 75 Punkte; Phase B bleibt in jedem Fall bei
den 11 benannten."""


# =============================================================================
# 6c  restrict merge points
# =============================================================================

OLD_BUDGET = """for _, r in lam_df.iterrows():
    _register(r[p_cols].to_numpy(float), f"baseline:{r['p_name']}")
    if bool(r["usable"]) and bool(r["moved"]):
        _register(r[lam_cols].to_numpy(float), f"{r['method']}({r['params']}):{r['p_name']}")
"""

NEW_BUDGET = """# Nur die Phase-B-Teilmenge wird gemergt. Bei USE_FULL_SEARCH_SET = True bleibt Phase A
# vollstaendig (75 Punkte), waehrend Phase B bei den 11 benannten bezahlbar bleibt.
lam_df_B = lam_df[lam_df["p_name"].isin(PHASE_B_NAMES)]
assert len(lam_df_B) > 0, "Keine lambda-Zeile faellt in die Phase-B-Teilmenge."

for _, r in lam_df_B.iterrows():
    _register(r[p_cols].to_numpy(float), f"baseline:{r['p_name']}")
    if bool(r["usable"]) and bool(r["moved"]):
        _register(r[lam_cols].to_numpy(float), f"{r['method']}({r['params']}):{r['p_name']}")
"""


# =============================================================================
# 7-  binding record
# =============================================================================

MD_BINDING = '''### 7-. Bindungsnachweis

Der lambda-Hash bindet nur die Koeffiziententabelle — also ausgerechnet den Teil, der
reward-frei jederzeit reproduzierbar ist. Prompts, Adaptergewichte, R-Matrizen, der
Scorer-Quelltext und das Generierungsmodell gehen bisher nirgends ein. Ohne sie koennte man
nach dem Lauf die Prompts austauschen, und der Hash passte weiter.
'''

CODE_BINDING = '''def _sha256_file(path):
    """Return the SHA256 of one file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def _sha256_dir(path, patterns=("*.safetensors", "*.bin", "adapter_config.json")):
    """Return a stable SHA256 over the relevant files of one adapter directory."""
    digest = hashlib.sha256()
    for pattern in patterns:
        for file_path in sorted(Path(path).glob(pattern)):
            digest.update(file_path.name.encode())
            digest.update(_sha256_file(file_path).encode())
    return digest.hexdigest()

def _runtime_versions():
    """Record the numerical software path in the frozen binding."""
    import platform
    from importlib import metadata

    packages = ("torch", "transformers", "tokenizers", "peft", "accelerate",
                "bitsandbytes", "numpy", "pandas", "scipy")
    return {"python": platform.python_version(),
            **{name: metadata.version(name) for name in packages}}

def _model_revisions():
    """Best-effort exact revisions of the two external models."""
    revisions = {}
    for label, repo in (("base_model", str(config["base_model_name"])),
                        ("armorm", "RLHFlow/ArmoRM-Llama3-8B-v0.1")):
        try:
            from huggingface_hub import HfApi

            revisions[label] = HfApi().model_info(repo).sha
        except Exception as error:                      # offline or no hub access
            revisions[label] = f"unresolved: {type(error).__name__}"
    return revisions

BINDING = {
    "regime": REGIME,
    "adapter_pattern": ADAPTER_PATTERN,
    "adapters_sha256": {a: _sha256_dir(p) for a, p in ADAPTER_PATHS.items()},
    "matrix_cos_sha256": _sha256_file(COSINE_MATRIX_CSV),
    "matrix_gram_sha256": _sha256_file(GRAM_MATRIX_CSV),
    "prompts_sha256": _sha256_file(REWARD_PROMPT_PATH),
    "scorer_sha256": _sha256_file(PROJECT_ROOT / "src" / "armorm_scorer.py"),
    "portfolio_sha256": _sha256_file(PROJECT_ROOT / "src" / "coefficient_portfolio.py"),
    "base_model_name": str(config["base_model_name"]),
    "armorm_precision": ARMORM_PRECISION,
    "attribute_order": list(ATTRIBUTES),
    "generation": {"max_new_tokens": MAX_NEW_TOKENS,
                   "repetition_penalty": REPETITION_PENALTY,
                   "no_repeat_ngram_size": NO_REPEAT_NGRAM_SIZE,
                   "decoding": "greedy, do_sample=False, num_beams=1"},
    "grids": {"rho_avg": RHO_AVG, "cert_c": CERT_C, "cert_eps": CERT_EPS,
              "c_grid": list(C_GRID), "alpha_grid": list(ALPHA_GRID), "eps_grid": list(EPS_GRID)},
    "src_sha256": {name: _sha256_file(PROJECT_ROOT / "src" / name) for name in (
        "merge.py", "proxy_validation.py", "metrics.py", "lambda_utils.py",
        "armorm_objectives.py", "eval_prompts.py")},
    "model_revisions": _model_revisions(),
    "runtime_versions": _runtime_versions(),
}

# Ein einziger Hash ueber den gesamten Nachweis. Er wandert in den Reward-Cache, damit
# eine unterbrochene Sitzung nicht unter geaenderten Bedingungen fortgesetzt wird.
BINDING_SHA256 = hashlib.sha256(
    json.dumps(BINDING, sort_keys=True).encode()).hexdigest()
print(json.dumps(BINDING, indent=2))
print()
print(f"BINDING_SHA256 = {BINDING_SHA256}")
'''


# =============================================================================
# 7  pre-registration
# =============================================================================

OLD_ROLE = """    "armorm_role": "read-only evaluation; never used for selection, stopping or any training decision",
"""

NEW_ROLE = '''    "armorm_role": (
        "REGIME-DEPENDENT. In the RS-PPO regime ArmoRM is frozen during evaluation but "
        "was ALSO the PPO reward model, so this evaluation is circular and supports only "
        "upper-bound and diagnostic claims, not held-out proxy validity. The read-only "
        "firewall claim holds for the SFT regime only, and there only pending the AKUT #6 "
        "audit. Do not restate the firewall as a global property of the thesis."
    ),
'''

OLD_DECISION_RULE = '''    "decision_rule": (
        "Primary: sign and magnitude of Delta U_p = U_p(lambda) - U_p(p) per method, with a "
        "paired bootstrap CI over prompts. A method counts as improving only if the CI "
        "excludes zero. No threshold is lowered after seeing the numbers. Cert is reported as "
        "a certificate row with Delta U_p = 0 by construction. Infeasible MaxMin cells are "
        "reported as infeasible and are not counted as collapses."
    ),
'''

NEW_DECISION_RULE = '''    "decision_rule": (
        "Primary: sign and magnitude of Delta U_p = U_p(lambda) - U_p(p) per method. "
        "Confirmatory improvement requires Delta U_p > 0 and a Holm-adjusted p-value < 0.05 "
        "across all method/preference comparisons; harm is defined analogously for Delta U_p "
        "< 0. The unadjusted percentile bootstrap CI is descriptive only. No threshold is "
        "lowered after seeing the numbers. Cert is reported as a certificate row with Delta "
        "U_p = 0 by construction. Infeasible MaxMin cells are reported as infeasible and are "
        "not counted as collapses."
    ),
'''

OLD_PREREG = """    "n_preferences": len(PREF_SET),
    "preference_names": [n for n, _ in PREF_SET],
"""

NEW_PREREG = """    "n_preferences_phase_a": len(PREF_SET),
    "n_preferences_phase_b": len(PREF_SET_B),
    "phase_b_preference_rule": PHASE_B_RULE,
    "preference_names": [n for n, _ in PREF_SET],
    "phase_b_preference_names": [n for n, _ in PREF_SET_B],
    "preference_set_note": (
        "11 named preferences. Deviates from the 13 pre-registered in NB09.1: "
        "quality_focused and detailed_answer removed, uniform renamed to balanced. "
        "Report this deviation explicitly."
    ),
    "binding": BINDING,
    "binding_sha256": BINDING_SHA256,
    "prompt_provenance": PROMPT_SUMMARY,
    "reward_model": {
        "name": "RLHFlow/ArmoRM-Llama3-8B-v0.1",
        "precision": ARMORM_PRECISION,
        "batch_size": 1,
        "head_mapping": "src.armorm_objectives.helpsteer_head_indices (external golden sample)",
        "scoring_format": "apply_chat_template(user + assistant)",
    },
    "generation_format": "apply_chat_template(user, add_generation_prompt=True) - identical to NB08",
    "raw_scores_retained": True,
    "error_layer": "paired bootstrap over prompts, 10000 draws, alpha=0.05",
    "multiplicity": "Holm-Bonferroni over all (method, preference) comparisons",
"""


# =============================================================================
# 7  gate
# =============================================================================

OLD_GATE = """    frozen = json.loads(PREREG_JSON.read_text(encoding="utf-8"))
    GATE_OPEN = frozen["lambda_table_sha256"] == lambda_hash
    print("GATE OFFEN — Vorregistrierung passt zur lambda-Tabelle." if GATE_OPEN
          else "GATE ZU — Hash-Mismatch.")
"""

NEW_GATE = """    frozen = json.loads(PREREG_JSON.read_text(encoding="utf-8"))
    _diffs = []
    if frozen.get("lambda_table_sha256") != lambda_hash:
        _diffs.append("lambda_table_sha256")
    for _k, _v in BINDING.items():
        if frozen.get("binding", {}).get(_k) != _v:
            _diffs.append(f"binding.{_k}")
    GATE_OPEN = not _diffs
    if GATE_OPEN:
        print("GATE OFFEN — Vorregistrierung passt zu lambda-Tabelle UND Bindungsnachweis.")
    else:
        print("GATE ZU — Abweichung in: " + ", ".join(_diffs))
        print("  Etwas hat sich seit dem Einfrieren geaendert: Prompts, Adapter, Matrix, "
              "Scorer oder Gitter. Ursache klaeren, nicht neu einfrieren.")
"""


# =============================================================================
# 8  adapters
# =============================================================================

OLD_ADAPTERS = """    BASE_MODEL_NAME = str(config["base_model_name"])
    ADAPTER_ROOT = (PROJECT_ROOT / str(config["adapter_dir"])).resolve()
    adapter_paths = {
        a: ADAPTER_ROOT / f"tinyllama-helpsteer2-{a}-adapter" for a in ATTRIBUTES
    }
"""

NEW_ADAPTERS = """    BASE_MODEL_NAME = str(config["base_model_name"])
    # Aufgeloest und gegen das Regime geprueft in Abschnitt 5a+. Hier NICHT erneut aus der
    # Config raten: config["adapter_dir"] zeigt auf die SFT-Adapter.
    adapter_paths = dict(ADAPTER_PATHS)
    assert BINDING["adapters_sha256"] == {a: _sha256_dir(p) for a, p in adapter_paths.items()}, \\
        "Adaptergewichte haben sich seit dem Bindungsnachweis geaendert."
"""


# =============================================================================
# 8  generation
# =============================================================================

OLD_GEN = '''        text = f"Human: {prompt}\\n\\nAssistant: "   # Trainingsformat, exakt wie NB06.1
        encoded = generation_tokenizer(text, return_tensors="pt")
'''

NEW_GEN = '''        # NB08-Format. Die RS-PPO-Adapter wurden mit apply_chat_template und
        # add_generation_prompt=True angesprochen; ein anderes Format hiesse,
        # off-distribution zu evaluieren. add_special_tokens=False, weil das Template
        # das Praefix bereits enthaelt — Kontrolle unten in dieser Zelle.
        text = generation_tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = generation_tokenizer(text, return_tensors="pt", add_special_tokens=False)
'''

OLD_GEN_TAIL = '''    print("Merge- und Generierungsroutinen bereit.")'''

NEW_GEN_TAIL = '''    _probe = generation_tokenizer.apply_chat_template(
        [{"role": "user", "content": "probe"}], tokenize=False, add_generation_prompt=True)
    _bos = generation_tokenizer.bos_token_id
    _ids = generation_tokenizer(_probe, add_special_tokens=False)["input_ids"]
    assert _bos is None or _ids.count(_bos) <= 1, "Doppeltes BOS im Generierungsprompt."
    print("Generierungsformat (NB08-identisch):")
    print(repr(_probe))
    print("Merge- und Generierungsroutinen bereit.")'''


# =============================================================================
# 8  scorer
# =============================================================================

NEW_REWARD_CELL = '''if RUN_REWARD_COLLECTION:
    import logging
    import time
    from datetime import timedelta
    from zoneinfo import ZoneInfo
    from src.armorm_objectives import ARMORM_HELPSTEER_OBJECTIVE_NAMES
    from src.armorm_scorer import make_score_prompt_answer
    from src.proxy_validation import collect_reward_tensor
    from src.tinyllama_training_utils import load_reward_prompts
    _cell38_started = time.perf_counter()

    # bitsandbytes meldet dieselbe bekannte BF16->FP16-Konvertierung fuer viele
    # Schichten erneut. Nur diese Meldung filtern; andere Warnungen bleiben sichtbar.
    class _BnbCastMessageFilter(logging.Filter):
        def filter(self, record):
            return ("MatMul8bitLt: inputs will be cast from torch.bfloat16 "
                    "to float16 during quantization") not in record.getMessage()

    _bnb_logger = logging.getLogger("bitsandbytes.autograd._functions")
    if not any(getattr(f, "_nb10_bnb_cast_filter", False) for f in _bnb_logger.filters):
        _bnb_filter = _BnbCastMessageFilter()
        _bnb_filter._nb10_bnb_cast_filter = True
        _bnb_logger.addFilter(_bnb_filter)

    reward_prompts = load_reward_prompts(REWARD_PROMPT_PATH)
    assert len(reward_prompts) == n_prompts, "Promptzahl weicht von der Vorregistrierung ab."
    assert all(a in ARMORM_HELPSTEER_OBJECTIVE_NAMES for a in ATTRIBUTES), \\
        "ATTRIBUTES enthaelt eine Achse ohne verankerte ArmoRM-Kopfzuordnung."

    # 8-bit: identisch zum Belohnungspfad des PPO-Laufs. Der Golden Sample laeuft dadurch
    # bei der lockeren Toleranz 0.35 des Model Cards, weil int8 die Regressionskoepfe
    # leicht verschiebt.
    score_prompt_answer, scorer = make_score_prompt_answer(
        dtype="bfloat16", load_in_8bit=(ARMORM_PRECISION == "8bit"))
    assert scorer.describe()["precision"] == ("int8" if ARMORM_PRECISION == "8bit" else "bfloat16")
    scorer.assert_golden_sample()          # VOR dem ersten echten Scoring
    print("Scorer:", scorer.describe())
    print(f"[time] ArmoRM-Setup inklusive Golden-Test: "
          f"{time.perf_counter() - _cell38_started:.1f} s")

    def _format_duration(seconds):
        seconds = max(0, int(round(seconds)))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    # Zeitmessung nur fuer noch nicht gecachte Merge-Punkte. Die Zuordnung ueber
    # coefficient_key bleibt auch bei einem unterbrochenen Lauf korrekt.
    _cached_keys = set()
    if REWARD_CACHE.is_file():
        for _line in REWARD_CACHE.read_text(encoding="utf-8").splitlines():
            if not _line.strip():
                continue
            try:
                _record = json.loads(_line)
            except json.JSONDecodeError:
                continue
            if "key" in _record:
                _cached_keys.add(str(_record["key"]))
    _eval_index_by_key = {coefficient_key(lam): i + 1
                          for i, lam in enumerate(EVAL_POINTS)}
    _cached_eval_count = sum(key in _cached_keys for key in _eval_index_by_key)
    _missing_initial = n_unique - _cached_eval_count
    _progress = {"started": time.perf_counter(), "new_done": 0}
    _nominal_remaining = _missing_initial * n_prompts * 2.5
    print(f"[time] Zelle 38: {n_unique} Merge-Punkte x {n_prompts} Prompts; "
          f"{_cached_eval_count} aus Cache, {_missing_initial} noch offen.")
    print(f"[time] Erste Restzeitschaetzung bei 2.5 s je Promptpaar: "
          f"{_format_duration(_nominal_remaining)}.")

    def reward_of_lambda(lam):
        """Per-prompt ArmoRM rewards for one merge point, shape (n_prompts, m)."""
        _point_started = time.perf_counter()
        _key = coefficient_key(lam)
        _position = _eval_index_by_key[_key]
        _new_number = _progress["new_done"] + 1
        print(f"[time] Starte Merge-Punkt {_position}/{n_unique} "
              f"(offener Punkt {_new_number}/{_missing_initial}).")
        with merged_model(base_model, DELTAS, lam):
            answers = [generate_answer(record["prompt"]) for record in reward_prompts]
        scores = np.asarray([
            score_prompt_answer(record["prompt"], answer, ATTRIBUTES)
            for record, answer in zip(reward_prompts, answers)
        ], dtype=np.float64)
        assert scores.shape == (len(reward_prompts), m), scores.shape
        _progress["new_done"] += 1
        _point_elapsed = time.perf_counter() - _point_started
        _reward_elapsed = time.perf_counter() - _progress["started"]
        _cell_elapsed = time.perf_counter() - _cell38_started
        _average = _reward_elapsed / _progress["new_done"]
        _remaining = _missing_initial - _progress["new_done"]
        _eta_seconds = _average * _remaining
        _finish = (datetime.now(ZoneInfo("Europe/Berlin")) + timedelta(seconds=_eta_seconds)).strftime("%d.%m. %H:%M %Z")
        print(f"[time] Fertig {_progress['new_done']}/{_missing_initial} offen | "
              f"letzter Punkt {_format_duration(_point_elapsed)} | "
              f"Reward-Zeit {_format_duration(_reward_elapsed)} | "
              f"Zellzeit {_format_duration(_cell_elapsed)} | "
              f"Rest {_format_duration(_eta_seconds)} | erwartet {_finish}")
        return scores          # NICHT mitteln: der gepaarte Bootstrap braucht die Rohwerte

    REWARD_TENSOR = collect_reward_tensor(
        EVAL_POINTS, reward_of_lambda, REWARD_CACHE,
        num_prompts=n_prompts, binding_sha256=BINDING_SHA256)
    REWARD_MATRIX = REWARD_TENSOR.mean(axis=1)     # identisch zur frueheren Matrix
    np.save(RESULTS_DIR / f"nb10_{RUN_TAG}_reward_tensor.npy", REWARD_TENSOR)
    print(f"Reward-Tensor: {REWARD_TENSOR.shape}   Reward-Matrix: {REWARD_MATRIX.shape}")
'''


# =============================================================================
# 9c  statistics, before the export section
# =============================================================================

MD_STATS = '''### 9c. Fehlerschicht und Multiplizitaet

`Delta U_p > 0` allein ist kein Ergebnis. Der gepaarte Bootstrap laeuft ueber dieselben
Prompts fuer `lambda` und fuer `p`, sodass die Promptschwierigkeit herausfaellt. Bei rund
zweihundert Vergleichen erzeugt `alpha = 0.05` etwa zehn Zufallstreffer, deshalb Holm ueber
alle Paare (Methode, Praeferenz). Diese Zelle steht VOR dem Export, damit `stats.csv` im ZIP
landet.
'''

CODE_STATS = '''STATS_CSV = RESULTS_DIR / f"nb10_{RUN_TAG}_stats.csv"

if RUN_REWARD_COLLECTION:
    from src.lambda_utils import holm_adjust, lambda_key
    from src.metrics import mean_rank, paired_bootstrap_ci, selection_regret

    # Gleiche Schluesselfunktion wie die Deduplikation in Abschnitt 6c, sonst findet
    # der Lookup die Punkte nicht wieder.
    def _key(vec):
        return lambda_key(np.asarray(vec, dtype=float), decimals=LAMBDA_DEDUP_DECIMALS)

    point_index = {_key(pt): i for i, pt in enumerate(EVAL_POINTS)}
    stats_rows, utilities_by_method = [], {}

    for _, r in lam_df_B.iterrows():
        if not (bool(r["usable"]) and bool(r["moved"])):
            continue
        p_vec = r[p_cols].to_numpy(float)
        lam_vec = r[lam_cols].to_numpy(float)
        i_lam, i_p = point_index.get(_key(lam_vec)), point_index.get(_key(p_vec))
        if i_lam is None or i_p is None:
            raise KeyError(f"Merge-Punkt fehlt im Tensor: {r['p_name']}/{r['method']}")

        boot = paired_bootstrap_ci(REWARD_TENSOR[i_lam], REWARD_TENSOR[i_p], p_vec)
        label = f"{r['method']}({r['params']})" if r["params"] else str(r["method"])
        stats_rows.append({
            "p_name": r["p_name"], "method": r["method"], "params": r["params"], "label": label,
            "U_p_baseline": float(REWARD_TENSOR[i_p].mean(axis=0) @ p_vec),
            "U_p_lambda": float(REWARD_TENSOR[i_lam].mean(axis=0) @ p_vec),
            **{k: boot[k] for k in ("delta_u_p", "ci_low", "ci_high", "excludes_zero", "p_value")},
        })
        utilities_by_method.setdefault(label, {})[r["p_name"]] = stats_rows[-1]["U_p_lambda"]
        utilities_by_method.setdefault("Baseline (lambda=p)", {})[r["p_name"]] = \\
            stats_rows[-1]["U_p_baseline"]

    stats_df = pd.DataFrame(stats_rows)
    if len(stats_df):
        # Konfirmatorisch zaehlt AUSSCHLIESSLICH Holm. Das unadjustierte CI bleibt als
        # deskriptive Spalte erhalten, traegt aber keine Aussage: bei rund zweihundert
        # Vergleichen erzeugt alpha = 0.05 etwa zehn Zufallstreffer.
        stats_df["p_holm"] = holm_adjust(stats_df["p_value"].to_numpy(float))
        stats_df["holm_improves"] = (stats_df["delta_u_p"] > 0) & (stats_df["p_holm"] < 0.05)
        stats_df["holm_harms"] = (stats_df["delta_u_p"] < 0) & (stats_df["p_holm"] < 0.05)
        stats_df["significant"] = stats_df["holm_improves"] | stats_df["holm_harms"]

        common = set.intersection(*(set(v) for v in utilities_by_method.values()))
        if common:
            order = sorted(common)
            ranks = mean_rank({k: [v[n] for n in order] for k, v in utilities_by_method.items()})
            print(f"Mean Rank ueber {len(order)} Praeferenzen (1 = bester):")
            for name, value in sorted(ranks.items(), key=lambda kv: kv[1]):
                print(f"  {value:5.2f}  {name}")

        # Selection Regret gegen den besten Punkt der GESAMTEN evaluierten Menge, nicht
        # nur gegen die fuer diese Praeferenz erzeugten. Nur so ist der Referenzsatz fuer
        # alle Methoden identisch — und nur so deckt sich die Zahl mit der Formulierung
        # "bester evaluierter Punkt" in der Arbeit.
        for pname, p_vec in PREF_SET_B:
            search_utilities = (REWARD_MATRIX @ np.asarray(p_vec, dtype=float)).tolist()
            mask = stats_df["p_name"] == pname
            stats_df.loc[mask, "selection_regret"] = [
                selection_regret(u, search_utilities) for u in stats_df.loc[mask, "U_p_lambda"]]
            stats_df.loc[mask, "baseline_regret"] = selection_regret(
                float(stats_df.loc[mask, "U_p_baseline"].iloc[0]), search_utilities)

        stats_df.to_csv(STATS_CSV, index=False)
        print(f"\\n{len(stats_df)} Vergleiche, "
              f"{int(stats_df['excludes_zero'].sum())} mit CI ohne Null, "
              f"{int(stats_df['holm_improves'].sum())} nach Holm besser, "
              f"{int(stats_df['holm_harms'].sum())} nach Holm schlechter.")
        display(stats_df.sort_values("delta_u_p", ascending=False).round(5))
    else:
        print("Keine bewegenden lambda in der Phase-B-Teilmenge: unter Floor-Kollaps faellt "
              "jede Methode mit lambda = p auf die Baseline. Das ist das Zertifikat aus "
              "NB09.1, kein fehlendes Ergebnis.")
else:
    print("RUN_REWARD_COLLECTION ist False — Statistik uebersprungen.")
'''


# =============================================================================
# 10  export
# =============================================================================

OLD_EXPORT = """    for path in (LAMBDA_CSV, FINAL_CSV, ROBUST_CSV, REPORT_JSON, PREREG_JSON, REWARD_CACHE):"""

NEW_EXPORT = """    for path in (LAMBDA_CSV, FINAL_CSV, ROBUST_CSV, STATS_CSV, REPORT_JSON, PREREG_JSON,
                 REWARD_CACHE, REWARD_PROMPT_PATH):"""

OLD_REPORT = """    "n_preferences": len(PREF_SET),
    "n_lambda_rows": int(len(lam_df)),"""

NEW_REPORT = """    "n_preferences_phase_a": len(PREF_SET),
    "n_preferences_phase_b": len(PREF_SET_B),
    "regime": REGIME,
    "armorm_precision": ARMORM_PRECISION,
    "n_lambda_rows": int(len(lam_df)),"""


# =============================================================================
# 11  open items
# =============================================================================

NEW_OPEN = '''## 11. Was noch offen ist

**Vor dem Gate zu klaeren, nicht im Notebook loesbar**

* **AKUT #6 — SFT-Pipeline-Audit.** Entscheidet, ob RQ2 regime-beschraenkt gilt oder
  entfaellt. Zu pruefen ist, dass ArmoRM im SFT-Pfad ausschliesslich der finalen Evaluation
  diente: nicht der Checkpoint-Auswahl, nicht dem Early Stopping, nicht der
  Hyperparameterwahl, nicht der Datenfilterung, nicht der Kandidatenauswahl, nicht dem
  Equal-N-Schnitt.
* **Provenienz von `reward_matrix.npy`** — SFT- oder PPO-Bundle.
* **Abweichung der Praeferenzmenge.** NB09.1 registrierte 13 vor; hier laufen 11.
  `quality_focused` und `detailed_answer` sind entfernt, `uniform` heisst `balanced`. Das
  gehoert als Abweichung berichtet, nicht stillschweigend uebernommen.

**Vor dem Gate im Notebook zu entscheiden**

* **Budget.** Die Schaetzung in Abschnitt 6c rechnet mit 2.5 s je Punkt. An fuenf Punkten
  nachmessen, bevor eingefroren wird. Reicht die Zeit nicht, ist Fair(alpha=0) der
  schmerzfreieste Schnitt: das Ziel ist dort linear, `min_delta` trifft exakt `-eps`, und der
  Schritt schreibt `lambda` neu, statt es zu korrigieren. Kuerzen ist NACH dem Gate nicht
  mehr zulaessig.

**Erledigt**

* Scorer: `src/armorm_scorer.py`, Kopf-Aufloesung ueber `helpsteer_head_indices`, Golden
  Sample vor dem ersten echten Scoring, `batch_size=1`, ArmoRM in 8-bit wie im PPO-Lauf.
* Rohwerte je Prompt bleiben erhalten (`collect_reward_tensor`), Fehlerschicht ist der
  gepaarte Bootstrap mit Holm-Korrektur.
* `RHO_AVG` fest, Praeferenzmenge fest, Generierung im NB08-Format.
* Bindungsnachweis ueber Prompts, Adapter, Matrizen, Scorer und Basismodell; das Gate prueft
  ihn vollstaendig.

**Bleibt als Hygiene ausserhalb dieses Notebooks**

* `ARMORM_HELPSTEER_OBJECTIVES` in `tinyllama_training_utils.py` hartcodiert die Indizes 0-4;
  ebenso ein Dict in NB06.1 Zelle 27. Nur `armorm_objectives.py` ist per Golden Sample
  verankert, die anderen sollten darauf umgestellt werden.
* `src/coefficient_methods.py` ist ein drittes, totes Portfolio (Prae-v13-Nomenklatur) und
  genau das Muster, nach dem Gate G6 sucht.
'''


# =============================================================================
# machinery
# =============================================================================

def cell_source(cell: dict[str, Any]) -> str:
    """Return a notebook cell's source as one string."""
    return "".join(cell["source"])


def set_source(cell: dict[str, Any], text: str) -> None:
    """Write a string back into a cell, keeping the line-list format."""
    cell["source"] = text.splitlines(keepends=True)


def new_cell(kind: str, text: str) -> dict[str, Any]:
    """Build a fresh notebook cell."""
    cell: dict[str, Any] = {"cell_type": kind, "metadata": {},
                            "source": text.splitlines(keepends=True)}
    if kind == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def find_cell(cells: list, needle: str, label: str) -> int:
    """Return the index of the single cell containing `needle`."""
    hits = [i for i, c in enumerate(cells) if needle in cell_source(c)]
    if len(hits) != 1:
        raise SystemExit(f"ABORT: {label}: {len(hits)} candidate cells, expected exactly 1.")
    return hits[0]


def replace_in(cells: list, index: int, old: str, new: str, label: str, notes: list) -> None:
    """Replace `old` by `new` in one cell, asserting a unique match."""
    text = cell_source(cells[index])
    if new in text:
        notes.append(f"skip    cell {index}: {label}")
        return
    count = text.count(old)
    if count != 1:
        raise SystemExit(
            f"ABORT cell {index} ({label}): expected 1 occurrence, found {count}.\n"
            f"--- target ---\n{old}\n--- end ---\n"
            "The notebook differs from the reviewed version."
        )
    set_source(cells[index], text.replace(old, new))
    notes.append(f"patch   cell {index}: {label}")


def insert_pair(cells: list, index: int, md: str, code: str, marker: str,
                label: str, notes: list) -> None:
    """Insert a markdown/code pair once, keyed by a marker string."""
    if any(marker in cell_source(c) for c in cells):
        notes.append(f"skip    {label}")
        return
    cells.insert(index, new_cell("code", code))
    cells.insert(index, new_cell("markdown", md))
    notes.append(f"insert  cells {index}-{index + 1}: {label}")


def main() -> int:
    """Apply every edit and write the patched notebook."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", required=True)
    parser.add_argument("--out", default=None, help="Default: overwrite in place.")
    args = parser.parse_args()

    path = Path(args.notebook)
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    notes: list[str] = []

    # In-place edits first: they do not shift indices.
    replace_in(cells, find_cell(cells, "eval_points, origins = [], {}", "budget"),
               OLD_BUDGET, NEW_BUDGET, "merge points restricted to Phase B", notes)
    i_pref_code = find_cell(cells, "PREF_SET = [(name, np.asarray(vec", "preference set")
    replace_in(cells, i_pref_code, OLD_FLAG, NEW_FLAG, "USE_FULL_SEARCH_SET", notes)
    replace_in(cells, find_cell(cells, "### 5a. Praeferenzmenge", "preference markdown"),
               OLD_PREF_MD, NEW_PREF_MD, "preference counts 11/75", notes)
    i_pb_md = next((i for i, c in enumerate(cells)
                    if "## 8. Phase B" in cell_source(c)), None)
    if i_pb_md is not None and "collect_reward_matrix`" in cell_source(cells[i_pb_md]):
        replace_in(cells, i_pb_md, "collect_reward_matrix`", "collect_reward_tensor`",
                   "phase B markdown", notes)
    else:
        notes.append("skip    phase B markdown")
    i_prereg = find_cell(cells, "PREREG_CONFIRM =", "pre-registration")
    replace_in(cells, i_prereg, OLD_PREREG, NEW_PREREG, "pre-registration fields", notes)
    replace_in(cells, i_prereg, OLD_ROLE, NEW_ROLE, "circularity stated regime-dependently", notes)
    replace_in(cells, i_prereg, OLD_DECISION_RULE, NEW_DECISION_RULE,
               "Holm is the confirmatory decision rule", notes)
    replace_in(cells, find_cell(cells, "GATE_OPEN = False", "gate"),
               OLD_GATE, NEW_GATE, "gate verifies the binding record", notes)
    replace_in(cells, find_cell(cells, "BASE_MODEL_NAME = str(config", "adapter loading"),
               OLD_ADAPTERS, NEW_ADAPTERS, "adapters from the resolved regime", notes)
    i_gen = find_cell(cells, "def generate_answer(prompt: str) -> str:", "generation")
    replace_in(cells, i_gen, OLD_GEN, NEW_GEN, "chat-template generation", notes)
    replace_in(cells, i_gen, OLD_GEN_TAIL, NEW_GEN_TAIL, "generation format check", notes)
    i_export = find_cell(cells, "zip_path = RESULTS_DIR", "export")
    replace_in(cells, i_export, OLD_EXPORT, NEW_EXPORT, "export includes stats and prompts", notes)
    replace_in(cells, i_export, OLD_REPORT, NEW_REPORT, "report carries regime and precision", notes)

    i_reward = find_cell(cells, "def reward_of_lambda", "reward collection")
    if "make_score_prompt_answer" in cell_source(cells[i_reward]):
        notes.append(f"skip    cell {i_reward}: scorer already wired")
    else:
        if "# OFFEN" not in cell_source(cells[i_reward]):
            raise SystemExit(f"ABORT cell {i_reward}: expected the open scorer placeholder.")
        set_source(cells[i_reward], NEW_REWARD_CELL)
        notes.append(f"replace cell {i_reward}: scorer wired, per-prompt rewards retained")

    i_open = find_cell(cells, "## 11. Was noch offen ist", "open items")
    if "Bindungsnachweis ueber Prompts" in cell_source(cells[i_open]):
        notes.append(f"skip    cell {i_open}: open items already rewritten")
    else:
        set_source(cells[i_open], NEW_OPEN)
        notes.append(f"replace cell {i_open}: open items rewritten")

    # Insertions last, from the back, so earlier indices stay valid.
    insert_pair(cells, find_cell(cells, "## 10. Export", "export heading"),
                MD_STATS, CODE_STATS, "STATS_CSV = ", "statistics before export", notes)
    insert_pair(cells, find_cell(cells, "PREREG_CONFIRM =", "pre-registration"),
                MD_BINDING, CODE_BINDING, "BINDING = {", "binding record", notes)
    insert_pair(cells, find_cell(cells, "eval_points, origins = [], {}", "budget"),
                MD_PROMPTS, CODE_PROMPTS, "build_eval_prompt_file(", "prompt file", notes)
    insert_pair(cells, find_cell(cells, "PREF_SET = [(name, np.asarray(vec", "preference set") + 1,
                MD_PHASE_B, CODE_PHASE_B, 'PHASE_B_RULE = "', "Phase-B subset", notes)
    i_cfg = find_cell(cells, "PREFERENCES = validate_preference_vectors", "config")
    replace_in(cells, i_cfg, OLD_CFG_MATRIX, NEW_CFG_MATRIX, "matrix load deferred to 5a+", notes)
    insert_pair(cells, i_cfg + 1,
                MD_PROVENANCE, CODE_PROVENANCE, 'REGIME = "', "provenance guard", notes)

    out = Path(args.out) if args.out else path
    changed = any(note.startswith(("patch", "insert", "replace")) for note in notes)
    if changed or out.resolve() != path.resolve():
        out.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for note in notes:
        print(note)
    action = "Written" if changed or out.resolve() != path.resolve() else "Unchanged"
    print(f"\n{action}: {out}  ({len(cells)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
