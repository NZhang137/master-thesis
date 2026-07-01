import json
from pathlib import Path


NOTEBOOK_PATH = Path("notebooks/05_compute_tinyllama_helpsteer2_relationship_matrices.ipynb")


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip("\n").splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip("\n").splitlines(keepends=True),
    }


cells = [
    md(
        """
# Compute TinyLlama HelpSteer2 Relationship Matrices

This Colab notebook computes relationship matrices for the five trained TinyLlama HelpSteer2 LoRA adapters.

It should be run after adapter training and after checking the effective LoRA delta norms in notebook 04. It reads only adapter configs and LoRA weight files on CPU. TinyLlama, ArmoRM, tokenizer weights, and reward-model weights are not loaded.

Primary output:

- cosine relationship matrix `R_cos`, scale invariant

Variant output:

- Gram relationship matrix `R_gram`, scale sensitive

Both matrices are computed from effective LoRA task vectors, not from raw concatenated `A` and `B` factors:

`delta_W_l = (lora_alpha / r) * (B_l @ A_l)`

`G_ij = <delta_i, delta_j>_F`

`R_cos,ij = G_ij / (|delta_i| * |delta_j|)`

Generated CSV, JSON, PNG, and ZIP files should stay out of GitHub.
"""
    ),
    md(
        """
## 1. Clone or update the repository

Run this first in Colab. On a local checkout, it keeps the current folder.
"""
    ),
    code(
        r"""
import os
import subprocess
from pathlib import Path

REPO_URL = "https://github.com/NZhang137/master-thesis.git"
PROJECT_DIR = Path("/content/master-thesis")

if Path("/content").exists():
    if (PROJECT_DIR / ".git").is_dir():
        print(f"Updating repository at {PROJECT_DIR}")
        subprocess.run(["git", "-C", str(PROJECT_DIR), "pull", "--ff-only"], check=True)
    else:
        print(f"Cloning repository to {PROJECT_DIR}")
        subprocess.run(["git", "clone", REPO_URL, str(PROJECT_DIR)], check=True)
    os.chdir(PROJECT_DIR)
else:
    print("Not running in Colab; using the current local checkout.")

print(f"Working directory: {Path.cwd()}")
"""
    ),
    md(
        """
## 2. Check runtime

This notebook is CPU-only. A GPU is not required.
"""
    ),
    code(
        r"""
import platform
import shutil
import subprocess

print(f"Python runtime: {platform.python_version()}")
print(f"Platform: {platform.platform()}")
print("GPU required: no")

if shutil.which("nvidia-smi"):
    print("GPU detected by runtime, but this notebook will not use it:")
    subprocess.run(["nvidia-smi"], check=False)
else:
    print("No GPU detected. That is fine for this CPU-only analysis notebook.")
"""
    ),
    md(
        """
## 3. Install lightweight dependencies

Only adapter-weight loading, tables, and plots are needed.
"""
    ),
    code(
        r"""
import subprocess
import sys

subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "pandas==2.2.2",
        "matplotlib",
        "safetensors",
    ],
    check=True,
)
print("Installed lightweight analysis dependencies.")
"""
    ),
    md(
        """
## 4. Settings

Change these values if your adapters are stored in another folder or checkpoint subdirectory.
"""
    ),
    code(
        r"""
from __future__ import annotations

import json
import math
import os
import sys
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXPECTED_ATTRIBUTES = (
    "helpfulness",
    "correctness",
    "coherence",
    "complexity",
    "verbosity",
)

BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
EXPECTED_LORA_RANK = 8
EXPECTED_LORA_ALPHA = 16

# Adapter location examples:
#   ADAPTER_SUBDIR = ""              # adapter weights directly in each adapter directory
#   ADAPTER_SUBDIR = "best-eval-loss"
#   ADAPTER_SUBDIR = "final"
ADAPTER_ROOT = Path("adapters")
ADAPTER_DIR_TEMPLATE = "tinyllama-helpsteer2-{attribute}-adapter"
ADAPTER_SUBDIR = ""

# Upload a zipped adapter folder in Colab when adapters are not already present.
RUN_ADAPTER_ZIP_UPLOAD = True
ADAPTER_ZIP_EXTRACT_ROOT = Path(".")

RESULTS_DIR = Path("results/tinyllama_helpsteer2_geometry")
PLOTS_DIR = RESULTS_DIR / "plots"

COSINE_OUTPUT_CSV = RESULTS_DIR / "tinyllama_helpsteer2_R_cos.csv"
GRAM_OUTPUT_CSV = RESULTS_DIR / "tinyllama_helpsteer2_R_gram.csv"
METADATA_OUTPUT_JSON = RESULTS_DIR / "tinyllama_helpsteer2_relationship_matrices_metadata.json"
COSINE_PLOT_PNG = PLOTS_DIR / "tinyllama_helpsteer2_R_cos.png"
GRAM_PLOT_PNG = PLOTS_DIR / "tinyllama_helpsteer2_R_gram.png"
RELATIONSHIP_OUTPUT_ZIP = Path("tinyllama_helpsteer2_relationship_matrices_outputs.zip")

# Optional compatibility output for older scripts that expect this exact path.
WRITE_COMPATIBILITY_COSINE_CSV = False
COMPATIBILITY_COSINE_OUTPUT_CSV = Path("results/tinyllama_helpsteer2_relationship_matrix.csv")

EPS = 1e-12


def find_project_root(start: Path | None = None) -> tuple[Path, bool]:
    '''Find the repository root, or fall back to the current working directory.'''
    start = (start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "src").is_dir() and (candidate / "notebooks").is_dir():
            return candidate, True
    return start, False


PROJECT_ROOT, REPO_ROOT_FOUND = find_project_root()
if PROJECT_ROOT != Path.cwd().resolve():
    os.chdir(PROJECT_ROOT)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ADAPTER_ROOT = (PROJECT_ROOT / ADAPTER_ROOT).resolve()
ADAPTER_ZIP_EXTRACT_ROOT = (PROJECT_ROOT / ADAPTER_ZIP_EXTRACT_ROOT).resolve()
RESULTS_DIR = (PROJECT_ROOT / RESULTS_DIR).resolve()
PLOTS_DIR = (PROJECT_ROOT / PLOTS_DIR).resolve()
COSINE_OUTPUT_CSV = (PROJECT_ROOT / COSINE_OUTPUT_CSV).resolve()
GRAM_OUTPUT_CSV = (PROJECT_ROOT / GRAM_OUTPUT_CSV).resolve()
METADATA_OUTPUT_JSON = (PROJECT_ROOT / METADATA_OUTPUT_JSON).resolve()
COSINE_PLOT_PNG = (PROJECT_ROOT / COSINE_PLOT_PNG).resolve()
GRAM_PLOT_PNG = (PROJECT_ROOT / GRAM_PLOT_PNG).resolve()
RELATIONSHIP_OUTPUT_ZIP = (PROJECT_ROOT / RELATIONSHIP_OUTPUT_ZIP).resolve()
COMPATIBILITY_COSINE_OUTPUT_CSV = (PROJECT_ROOT / COMPATIBILITY_COSINE_OUTPUT_CSV).resolve()

print(f"Project root: {PROJECT_ROOT}")
print(f"Repo utilities available: {REPO_ROOT_FOUND}")
print(f"Adapter root: {ADAPTER_ROOT}")
print(f"Adapter subdir: {ADAPTER_SUBDIR!r}")
print(f"Cosine CSV: {COSINE_OUTPUT_CSV}")
print(f"Gram CSV: {GRAM_OUTPUT_CSV}")
"""
    ),
    md(
        """
## 5. Optional: upload adapter zip

If the adapters are already present in `adapters/`, this can be skipped by setting `RUN_ADAPTER_ZIP_UPLOAD = False`.
"""
    ),
    code(
        r"""
def safe_extract_zip(zip_file: zipfile.ZipFile, extract_root: Path) -> None:
    '''Extract a zip file while preventing paths from escaping extract_root.'''
    extract_root = extract_root.resolve()
    for member in zip_file.infolist():
        target_path = (extract_root / member.filename).resolve()
        if not str(target_path).startswith(str(extract_root)):
            raise ValueError(f"Unsafe zip member path: {member.filename}")
    zip_file.extractall(extract_root)


if RUN_ADAPTER_ZIP_UPLOAD:
    try:
        from google.colab import files
    except ImportError as error:
        raise RuntimeError("Adapter zip upload is only available in Google Colab.") from error

    uploaded_files = files.upload()
    if not uploaded_files:
        raise RuntimeError("No adapter zip was uploaded.")

    for uploaded_name in uploaded_files:
        uploaded_path = Path(uploaded_name)
        if uploaded_path.suffix.lower() != ".zip":
            raise ValueError(f"Expected a .zip adapter archive, got: {uploaded_path}")
        print(f"Extracting {uploaded_path} to {ADAPTER_ZIP_EXTRACT_ROOT}")
        with zipfile.ZipFile(uploaded_path) as zip_file:
            safe_extract_zip(zip_file, ADAPTER_ZIP_EXTRACT_ROOT)
else:
    print("Adapter zip upload skipped.")
"""
    ),
    md(
        """
## 6. Show important files

This prints the paths the notebook will use before any computation starts.
"""
    ),
    code(
        r"""
print(f"Current working directory: {Path.cwd()}")
print(f"Project root: {PROJECT_ROOT}")
print(f"Repo utilities available: {REPO_ROOT_FOUND}")
print(f"Adapter root: {ADAPTER_ROOT}")
print(f"Adapter subdir: {ADAPTER_SUBDIR!r}")
print(f"Cosine CSV: {COSINE_OUTPUT_CSV}")
print(f"Gram CSV: {GRAM_OUTPUT_CSV}")
print(f"Metadata JSON: {METADATA_OUTPUT_JSON}")
print(f"Cosine plot: {COSINE_PLOT_PNG}")
print(f"Gram plot: {GRAM_PLOT_PNG}")

important_paths = [
    PROJECT_ROOT / "src" / "effective_lora_geometry.py",
    PROJECT_ROOT / "src" / "relationship_utils.py",
    ADAPTER_ROOT,
    RESULTS_DIR,
    PLOTS_DIR,
]
for important_path in important_paths:
    print(f"{'FOUND' if important_path.exists() else 'MISSING'}: {important_path}")

print("\nAdapter candidates:")
for attribute in EXPECTED_ATTRIBUTES:
    adapter_dir = ADAPTER_ROOT / ADAPTER_DIR_TEMPLATE.format(attribute=attribute)
    adapter_path = adapter_dir if ADAPTER_SUBDIR in {"", ".", None} else adapter_dir / str(ADAPTER_SUBDIR)
    config_path = adapter_path / "adapter_config.json"
    safetensors_path = adapter_path / "adapter_model.safetensors"
    bin_path = adapter_path / "adapter_model.bin"
    weight_status = "safetensors" if safetensors_path.is_file() else "bin" if bin_path.is_file() else "missing"
    print(f"  {attribute:11s}: {adapter_path}")
    print(f"      config: {'yes' if config_path.is_file() else 'no'}")
    print(f"      weights: {weight_status}")
"""
    ),
    md(
        """
## 7. Discover adapters

The notebook expects exactly the five HelpSteer2 attributes.
"""
    ),
    code(
        r"""
def resolve_adapter_path(attribute: str) -> Path:
    '''Resolve one adapter path from the configurable adapter root and subdirectory.'''
    adapter_dir = ADAPTER_ROOT / ADAPTER_DIR_TEMPLATE.format(attribute=attribute)
    if ADAPTER_SUBDIR in {"", ".", None}:
        return adapter_dir
    return adapter_dir / str(ADAPTER_SUBDIR)


def discover_adapters() -> dict[str, Path]:
    '''Discover the five expected HelpSteer2 adapters and fail clearly if any are missing.'''
    discovered: dict[str, Path] = {}
    missing: list[tuple[str, Path]] = []

    print("Configured adapter paths:")
    for attribute in EXPECTED_ATTRIBUTES:
        path = resolve_adapter_path(attribute)
        status = "FOUND" if path.is_dir() else "MISSING"
        print(f"  {attribute:11s} {status:7s} {path}")
        if path.is_dir():
            discovered[attribute] = path
        else:
            missing.append((attribute, path))

    if missing:
        details = "\n".join(f"  - {attribute}: {path}" for attribute, path in missing)
        raise FileNotFoundError(
            "Missing adapter directories. Upload/extract adapters or adjust ADAPTER_ROOT/ADAPTER_SUBDIR:\n"
            + details
        )

    if set(discovered) != set(EXPECTED_ATTRIBUTES):
        raise RuntimeError(f"Expected attributes {EXPECTED_ATTRIBUTES}, got {tuple(discovered)}")
    if len(discovered) != len(EXPECTED_ATTRIBUTES):
        raise RuntimeError(f"Expected five adapters, found {len(discovered)}")
    return discovered


adapter_paths = discover_adapters()
"""
    ),
    md(
        """
## 8. Load LoRA geometry utilities

The computation uses effective LoRA updates from `src/effective_lora_geometry.py`.
"""
    ),
    code(
        r"""
from src.effective_lora_geometry import (
    effective_lora_inner_product,
    effective_lora_update_norm,
    effective_lora_update_numel,
    load_adapter_config,
    load_effective_lora_geometry,
    validate_compatible_geometries,
)

print("Using repository LoRA geometry utilities from src.effective_lora_geometry.")
print("Representation: effective LoRA updates, delta_W = scaling * (B @ A).")
print("TinyLlama and ArmoRM are not loaded.")
"""
    ),
    md(
        """
## 9. Load adapter geometries

Each adapter is converted into layer-wise effective LoRA factors on CPU.
"""
    ),
    code(
        r"""
def _single_or_mixed(values: set[object]) -> object:
    if len(values) == 1:
        return next(iter(values))
    return "mixed"


def summarize_geometry(attribute: str, adapter_path: Path, geometry: dict) -> dict[str, object]:
    config = load_adapter_config(adapter_path)
    ranks = {int(layer.lora_a.shape[0]) for layer in geometry.values()}
    scalings = {float(layer.scaling) for layer in geometry.values()}
    norm = float(effective_lora_update_norm(geometry, eps=EPS))
    squared_norm = float(effective_lora_inner_product(geometry, geometry))
    return {
        "attribute": attribute,
        "adapter_path": str(adapter_path),
        "num_lora_modules": len(geometry),
        "lora_rank": _single_or_mixed(ranks),
        "lora_alpha": config.get("lora_alpha"),
        "scaling": _single_or_mixed(scalings),
        "effective_update_numel": int(effective_lora_update_numel(geometry)),
        "delta_norm": norm,
        "delta_norm_squared": squared_norm,
    }


geometries = {}
summary_rows = []
for attribute in EXPECTED_ATTRIBUTES:
    adapter_path = adapter_paths[attribute]
    geometry = load_effective_lora_geometry(adapter_path)
    if not geometry:
        raise RuntimeError(f"Adapter has no LoRA A/B pairs: {adapter_path}")
    geometries[attribute] = geometry
    summary_rows.append(summarize_geometry(attribute, adapter_path, geometry))

validate_compatible_geometries(
    [geometries[attribute] for attribute in EXPECTED_ATTRIBUTES],
    EXPECTED_ATTRIBUTES,
)

summary_df = pd.DataFrame(summary_rows)
summary_df
"""
    ),
    md(
        """
## 10. Compute Gram and cosine matrices

`R_cos` is the primary relationship matrix because it is scale invariant. `R_gram` is saved as a scale-sensitive variant.
"""
    ),
    code(
        r"""
attributes = list(EXPECTED_ATTRIBUTES)
n = len(attributes)

gram = np.zeros((n, n), dtype=np.float64)
for i, attr_i in enumerate(attributes):
    for j, attr_j in enumerate(attributes[i:], start=i):
        value = float(effective_lora_inner_product(geometries[attr_i], geometries[attr_j]))
        gram[i, j] = value
        gram[j, i] = value

norms = np.sqrt(np.maximum(np.diag(gram), 0.0))
if np.any(norms < EPS):
    raise RuntimeError(f"Near-zero effective LoRA norm detected: {dict(zip(attributes, norms))}")

cosine = gram / np.outer(norms, norms)
cosine = np.clip(cosine, -1.0, 1.0)
np.fill_diagonal(cosine, 1.0)

cosine_df = pd.DataFrame(cosine, index=attributes, columns=attributes)
gram_df = pd.DataFrame(gram, index=attributes, columns=attributes)

print("Computed R_cos and R_gram from effective LoRA updates only.")
print(f"Cosine matrix shape: {cosine_df.shape}")
print(f"Gram matrix shape: {gram_df.shape}")
"""
    ),
    md(
        """
## 11. Show cosine matrix

This is the primary matrix for relationship-aware coefficient correction.
"""
    ),
    code('cosine_df.style.format("{:.6f}")\n'),
    md(
        """
## 12. Show Gram matrix

This variant keeps magnitude information and is therefore affected by adapter norm scale.
"""
    ),
    code('gram_df.style.format("{:.8f}")\n'),
    md(
        """
## 13. Interpret relationship matrices

Cosine values are scale invariant. Gram values are scale sensitive.
"""
    ),
    code(
        r"""
def print_matrix_interpretation() -> None:
    cosine_eigenvalues = np.linalg.eigvalsh(cosine)
    gram_eigenvalues = np.linalg.eigvalsh(gram)
    max_cosine_offdiag = max(
        abs(float(cosine[i, j]))
        for i in range(n)
        for j in range(n)
        if i != j
    )
    min_cosine_offdiag = min(
        float(cosine[i, j])
        for i in range(n)
        for j in range(n)
        if i != j
    )
    max_cosine_pair = max(
        ((attributes[i], attributes[j], float(cosine[i, j])) for i in range(n) for j in range(i + 1, n)),
        key=lambda item: item[2],
    )
    min_cosine_pair = min(
        ((attributes[i], attributes[j], float(cosine[i, j])) for i in range(n) for j in range(i + 1, n)),
        key=lambda item: item[2],
    )

    print("R_cos is primary because it normalizes away adapter norm scale.")
    print("R_gram is a useful variant when magnitude should remain visible.")
    print(f"Cosine eigenvalues: {[round(float(x), 8) for x in cosine_eigenvalues]}")
    print(f"Gram eigenvalues: {[round(float(x), 8) for x in gram_eigenvalues]}")
    print(f"Largest absolute off-diagonal cosine: {max_cosine_offdiag:.6f}")
    print(f"Most similar pair: {max_cosine_pair[0]} / {max_cosine_pair[1]} = {max_cosine_pair[2]:.6f}")
    print(f"Least aligned pair: {min_cosine_pair[0]} / {min_cosine_pair[1]} = {min_cosine_pair[2]:.6f}")
    if min_cosine_offdiag < -0.1:
        print("WARNING: At least one pair has a clearly negative cosine; inspect this as a potential conflict signal.")
    if np.min(cosine_eigenvalues) < -1e-8:
        print("WARNING: R_cos has a negative eigenvalue beyond numerical tolerance.")
    if np.min(gram_eigenvalues) < -1e-8:
        print("WARNING: R_gram has a negative eigenvalue beyond numerical tolerance.")


print_matrix_interpretation()
"""
    ),
    md(
        """
## 14. Plot relationship heatmaps

The plots use matplotlib only.
"""
    ),
    code(
        r"""
def plot_matrix(
    matrix_df: pd.DataFrame,
    title: str,
    output_path: Path,
    *,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
    image = ax.imshow(matrix_df.values, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(matrix_df.columns)))
    ax.set_yticks(range(len(matrix_df.index)))
    ax.set_xticklabels(matrix_df.columns, rotation=35, ha="right")
    ax.set_yticklabels(matrix_df.index)
    ax.set_title(title)
    for row in range(matrix_df.shape[0]):
        for column in range(matrix_df.shape[1]):
            value = float(matrix_df.iloc[row, column])
            color = "black" if abs(value) < 0.65 else "white"
            ax.text(column, row, f"{value:.3f}", ha="center", va="center", color=color, fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.show()
    print(f"Saved plot to {output_path}")


plot_matrix(
    cosine_df,
    "TinyLlama HelpSteer2 R_cos from Effective LoRA Updates",
    COSINE_PLOT_PNG,
    cmap="coolwarm",
    vmin=-1.0,
    vmax=1.0,
)
plot_matrix(
    gram_df,
    "TinyLlama HelpSteer2 R_gram from Effective LoRA Updates",
    GRAM_PLOT_PNG,
    cmap="viridis",
)
"""
    ),
    md(
        """
## 15. Save outputs

The generated files are analysis artifacts and should stay out of GitHub.
"""
    ),
    code(
        r"""
def write_labeled_matrix_csv(matrix_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_df.to_csv(output_path, index_label="adapter", float_format="%.10f")


write_labeled_matrix_csv(cosine_df, COSINE_OUTPUT_CSV)
write_labeled_matrix_csv(gram_df, GRAM_OUTPUT_CSV)

if WRITE_COMPATIBILITY_COSINE_CSV:
    write_labeled_matrix_csv(cosine_df, COMPATIBILITY_COSINE_OUTPUT_CSV)

metadata = {
    "base_model": BASE_MODEL_NAME,
    "attributes": attributes,
    "adapter_paths": {attribute: str(adapter_paths[attribute]) for attribute in attributes},
    "representation": "effective LoRA updates: delta_W = scaling * (B @ A)",
    "primary_matrix": "R_cos",
    "primary_matrix_reason": "scale invariant cosine relationship between effective LoRA task vectors",
    "variant_matrix": "R_gram",
    "variant_matrix_reason": "scale-sensitive Frobenius inner products between effective LoRA task vectors",
    "cosine_output_csv": str(COSINE_OUTPUT_CSV),
    "gram_output_csv": str(GRAM_OUTPUT_CSV),
    "cosine_plot_png": str(COSINE_PLOT_PNG),
    "gram_plot_png": str(GRAM_PLOT_PNG),
    "compatibility_cosine_output_csv": str(COMPATIBILITY_COSINE_OUTPUT_CSV) if WRITE_COMPATIBILITY_COSINE_CSV else None,
    "summary_rows": summary_df.to_dict(orient="records"),
    "cosine_eigenvalues": [float(value) for value in np.linalg.eigvalsh(cosine)],
    "gram_eigenvalues": [float(value) for value in np.linalg.eigvalsh(gram)],
    "note": "TinyLlama, ArmoRM, tokenizer weights, and reward-model weights are not loaded.",
}
METADATA_OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
with METADATA_OUTPUT_JSON.open("w", encoding="utf-8") as output_file:
    json.dump(metadata, output_file, indent=2)
    output_file.write("\n")

print(f"Saved R_cos to {COSINE_OUTPUT_CSV}")
print(f"Saved R_gram to {GRAM_OUTPUT_CSV}")
print(f"Saved metadata to {METADATA_OUTPUT_JSON}")
if WRITE_COMPATIBILITY_COSINE_CSV:
    print(f"Saved compatibility R_cos to {COMPATIBILITY_COSINE_OUTPUT_CSV}")
else:
    print("Compatibility R_cos copy skipped. Set WRITE_COMPATIBILITY_COSINE_CSV = True if an older script needs it.")
"""
    ),
    md(
        """
## 16. Validate outputs

These checks catch missing adapters, empty LoRA geometry, invalid matrix values, and missing saved files.
"""
    ),
    code(
        r"""
assert tuple(attributes) == EXPECTED_ATTRIBUTES, "Unexpected attribute order."
assert set(adapter_paths) == set(EXPECTED_ATTRIBUTES), "All five expected attributes must be present."
assert (summary_df["num_lora_modules"] > 0).all(), "Each adapter must expose at least one LoRA A/B pair."
assert np.isfinite(summary_df["delta_norm"]).all(), "All delta norms must be finite."
assert (summary_df["delta_norm"] > 0).all(), "All delta norms must be positive."
assert cosine_df.shape == (5, 5), "R_cos must be 5x5."
assert gram_df.shape == (5, 5), "R_gram must be 5x5."
assert np.isfinite(cosine).all(), "R_cos must contain only finite values."
assert np.isfinite(gram).all(), "R_gram must contain only finite values."
assert np.allclose(cosine, cosine.T, atol=1e-10), "R_cos must be symmetric."
assert np.allclose(gram, gram.T, atol=1e-8), "R_gram must be symmetric."
assert np.allclose(np.diag(cosine), 1.0, atol=1e-10), "R_cos diagonal must be 1."
assert (np.diag(gram) > 0).all(), "R_gram diagonal must be positive."
assert (cosine >= -1.0000001).all() and (cosine <= 1.0000001).all(), "R_cos values must be in [-1, 1]."
assert COSINE_OUTPUT_CSV.is_file(), f"Missing saved CSV: {COSINE_OUTPUT_CSV}"
assert GRAM_OUTPUT_CSV.is_file(), f"Missing saved CSV: {GRAM_OUTPUT_CSV}"
assert METADATA_OUTPUT_JSON.is_file(), f"Missing saved metadata: {METADATA_OUTPUT_JSON}"
assert COSINE_PLOT_PNG.is_file(), f"Missing saved plot: {COSINE_PLOT_PNG}"
assert GRAM_PLOT_PNG.is_file(), f"Missing saved plot: {GRAM_PLOT_PNG}"
print("Validation passed.")
"""
    ),
    md(
        """
## 17. Create a zip archive for download

This archive contains the generated matrices, metadata, and plots. Keep the zip file out of GitHub.
"""
    ),
    code(
        r"""
files_to_zip = [
    COSINE_OUTPUT_CSV,
    GRAM_OUTPUT_CSV,
    METADATA_OUTPUT_JSON,
    COSINE_PLOT_PNG,
    GRAM_PLOT_PNG,
]
if WRITE_COMPATIBILITY_COSINE_CSV:
    files_to_zip.append(COMPATIBILITY_COSINE_OUTPUT_CSV)

with zipfile.ZipFile(RELATIONSHIP_OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
    for file_path in files_to_zip:
        if not file_path.is_file():
            raise FileNotFoundError(f"Cannot add missing file to zip: {file_path}")
        zip_file.write(file_path, arcname=file_path.relative_to(PROJECT_ROOT))

print(f"Created {RELATIONSHIP_OUTPUT_ZIP}")
"""
    ),
    md(
        """
## 18. Download the zip archive

In Colab this opens the browser download. Locally it prints the path.
"""
    ),
    code(
        r"""
try:
    from google.colab import files

    files.download(str(RELATIONSHIP_OUTPUT_ZIP))
except ImportError:
    print(f"Download manually from: {RELATIONSHIP_OUTPUT_ZIP}")
"""
    ),
    md(
        """
## 19. Git safety check

Generated relationship matrices and plots are analysis outputs. They should normally stay out of commits.
"""
    ),
    code(
        r"""
import subprocess

paths_to_check = [
    COSINE_OUTPUT_CSV,
    GRAM_OUTPUT_CSV,
    METADATA_OUTPUT_JSON,
    COSINE_PLOT_PNG,
    GRAM_PLOT_PNG,
    RELATIONSHIP_OUTPUT_ZIP,
]
if WRITE_COMPATIBILITY_COSINE_CSV:
    paths_to_check.append(COMPATIBILITY_COSINE_OUTPUT_CSV)

print("Git ignore status for generated outputs:")
for path in paths_to_check:
    relative_path = path.relative_to(PROJECT_ROOT)
    result = subprocess.run(
        ["git", "check-ignore", str(relative_path)],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode == 0:
        print(f"  ignored:     {relative_path}")
    else:
        print(f"  NOT ignored: {relative_path}")

print("\nCurrent generated-output status:")
subprocess.run(
    ["git", "status", "--short", "--", *[str(path.relative_to(PROJECT_ROOT)) for path in paths_to_check]],
    cwd=PROJECT_ROOT,
    check=False,
)
print("\nIf a generated file appears as untracked or modified, do not commit it unless you intentionally want the artifact in GitHub.")
"""
    ),
    md(
        """
## 20. Next step: use R in coefficient computation

Use `R_cos` as the primary relationship matrix for Rewarded-Soups-style coefficient correction because it compares directions independent of adapter norm scale.

Use `R_gram` only as a scale-sensitive variant or diagnostic.

If an older script expects `results/tinyllama_helpsteer2_relationship_matrix.csv`, rerun the save cell with `WRITE_COMPATIBILITY_COSINE_CSV = True`, or pass the `R_cos` CSV path explicitly.
"""
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": [], "toc_visible": True},
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Wrote {NOTEBOOK_PATH} with {len(cells)} cells.")
