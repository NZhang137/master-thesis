"""Export TinyLlama training-log metrics as reproducible PNG and CSV curves."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = "results/tinyllama_helpsteer2_training_logs"
DEFAULT_OUTPUT_DIR = "results/plots/tensorboard"
METRICS = ("train_loss", "eval_loss", "learning_rate")
COMBINED_METRICS = ("train_loss", "eval_loss")
ATTRIBUTE_ORDER = (
    "helpfulness",
    "correctness",
    "coherence",
    "complexity",
    "verbosity",
)


def resolve_project_path(value: str) -> Path:
    """Resolve a command-line path relative to the repository root."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    """Parse input folder, output folder, and image resolution."""
    parser = argparse.ArgumentParser(
        description="Export TinyLlama CSV training logs as PNG and CSV curves."
    )
    parser.add_argument(
        "--log_dir",
        "--log-dir",
        dest="log_dir",
        default=DEFAULT_LOG_DIR,
    )
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        dest="output_dir",
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def attribute_from_filename(path: Path) -> str:
    """Derive an attribute name from the standard training-log filename."""
    name = path.stem
    prefix = "tinyllama_helpsteer2_"
    suffix = "_training_log"
    if name.startswith(prefix):
        name = name[len(prefix) :]
    if name.endswith(suffix):
        name = name[: -len(suffix)]
    return name.strip()


def safe_slug(value: str) -> str:
    """Convert one attribute label to a safe deterministic filename slug."""
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip()).strip("_")
    if not slug:
        raise ValueError("Attribute names must contain filename-safe characters.")
    return slug.lower()


def parse_number(value: object, *, label: str, path: Path) -> float | None:
    """Parse an optional finite numeric CSV value."""
    if value is None or not str(value).strip():
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid {label} value in {path}: {value!r}") from error
    if not math.isfinite(number):
        raise ValueError(f"Non-finite {label} value in {path}: {value!r}")
    return number


def load_log_file(path: Path) -> tuple[str, dict[str, list[tuple[float, float]]]]:
    """Load available metric points from one CSV training log."""
    with path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        columns = reader.fieldnames or []
        if "global_step" not in columns:
            raise ValueError(f"Training log is missing global_step: {path}")
        available_metrics = [metric for metric in METRICS if metric in columns]
        if not available_metrics:
            return attribute_from_filename(path), {}

        fallback_attribute = attribute_from_filename(path)
        attributes: set[str] = set()
        points: dict[str, list[tuple[float, float]]] = {
            metric: [] for metric in available_metrics
        }
        for row in reader:
            step = parse_number(row.get("global_step"), label="global_step", path=path)
            if step is None:
                continue
            row_attribute = str(row.get("attribute", "")).strip()
            if row_attribute:
                attributes.add(row_attribute)
            for metric in available_metrics:
                value = parse_number(row.get(metric), label=metric, path=path)
                if value is not None:
                    points[metric].append((step, value))

    if len(attributes) > 1:
        raise ValueError(f"Training log contains multiple attributes: {path}")
    attribute = next(iter(attributes), fallback_attribute)
    if not attribute:
        raise ValueError(f"Could not determine the attribute for {path}")
    return attribute, points


def merge_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Sort points and keep the latest value for duplicate global steps."""
    by_step: dict[float, float] = {}
    for step, value in points:
        by_step[step] = value
    return sorted(by_step.items())


def write_metric_csv(
    path: Path,
    metric: str,
    series: dict[str, list[tuple[float, float]]],
) -> None:
    """Write one or more attribute curves in long-form CSV format."""
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["attribute", "global_step", metric])
        for attribute in ordered_attributes(series):
            for step, value in series[attribute]:
                writer.writerow([attribute, f"{step:g}", f"{value:.12g}"])


def ordered_attributes(series: dict[str, object]) -> list[str]:
    """Order known HelpSteer2 attributes first and extras alphabetically."""
    known = [attribute for attribute in ATTRIBUTE_ORDER if attribute in series]
    extras = sorted(set(series).difference(known))
    return [*known, *extras]


def metric_label(metric: str) -> str:
    """Return a readable axis label for one CSV metric."""
    return metric.replace("_", " ").title()


def save_individual_plot(
    *,
    pyplot,
    attribute: str,
    metric: str,
    points: list[tuple[float, float]],
    output_path: Path,
    dpi: int,
) -> None:
    """Save one attribute/metric curve."""
    steps, values = zip(*points)
    figure, axis = pyplot.subplots(figsize=(7.2, 4.4))
    axis.plot(steps, values, linewidth=1.6, marker="o", markersize=2.5)
    axis.set_title(f"TinyLlama HelpSteer2: {attribute} {metric_label(metric)}")
    axis.set_xlabel("Global Step")
    axis.set_ylabel(metric_label(metric))
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    pyplot.close(figure)


def save_combined_plot(
    *,
    pyplot,
    metric: str,
    series: dict[str, list[tuple[float, float]]],
    output_path: Path,
    dpi: int,
) -> None:
    """Save one comparison curve across all available attributes."""
    figure, axis = pyplot.subplots(figsize=(8.2, 5.0))
    for attribute in ordered_attributes(series):
        steps, values = zip(*series[attribute])
        axis.plot(steps, values, linewidth=1.5, label=attribute)
    axis.set_title(f"TinyLlama HelpSteer2: All Attributes {metric_label(metric)}")
    axis.set_xlabel("Global Step")
    axis.set_ylabel(metric_label(metric))
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    pyplot.close(figure)


def main() -> None:
    """Load every training log and export all available curves."""
    args = parse_args()
    if args.dpi < 1:
        raise ValueError("dpi must be a positive integer.")
    log_dir = resolve_project_path(args.log_dir)
    output_dir = resolve_project_path(args.output_dir)
    if not log_dir.is_dir():
        raise FileNotFoundError(f"Training log directory not found: {log_dir}")
    log_paths = sorted(log_dir.glob("*.csv"))
    if not log_paths:
        raise FileNotFoundError(f"No CSV training logs found in {log_dir}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot
    except ImportError as error:
        raise ImportError(
            "Training-curve export requires matplotlib. Install it with "
            "`pip install matplotlib`."
        ) from error

    collected: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for log_path in log_paths:
        attribute, metrics = load_log_file(log_path)
        if not metrics:
            print(f"[SKIP] No supported metric columns in {log_path.name}")
            continue
        for metric, points in metrics.items():
            collected[attribute][metric].extend(points)

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    combined: dict[str, dict[str, list[tuple[float, float]]]] = {
        metric: {} for metric in COMBINED_METRICS
    }

    for attribute in ordered_attributes(collected):
        attribute_slug = safe_slug(attribute)
        for metric in METRICS:
            points = merge_points(collected[attribute].get(metric, []))
            if not points:
                print(f"[SKIP] {attribute}: no {metric} values")
                continue
            png_path = output_dir / f"{attribute_slug}_{metric}.png"
            csv_path = output_dir / f"{attribute_slug}_{metric}.csv"
            save_individual_plot(
                pyplot=pyplot,
                attribute=attribute,
                metric=metric,
                points=points,
                output_path=png_path,
                dpi=args.dpi,
            )
            write_metric_csv(csv_path, metric, {attribute: points})
            saved.extend((png_path, csv_path))
            if metric in combined:
                combined[metric][attribute] = points

    for metric in COMBINED_METRICS:
        series = combined[metric]
        if not series:
            print(f"[SKIP] Combined plot: no {metric} values")
            continue
        stem = f"tinyllama_all_attributes_{metric}"
        png_path = output_dir / f"{stem}.png"
        csv_path = output_dir / f"{stem}.csv"
        save_combined_plot(
            pyplot=pyplot,
            metric=metric,
            series=series,
            output_path=png_path,
            dpi=args.dpi,
        )
        write_metric_csv(csv_path, metric, series)
        saved.extend((png_path, csv_path))

    if not saved:
        raise ValueError("No supported metric values were found in the CSV logs.")
    print(f"\nExported {len(saved)} files to {output_dir}:")
    for path in saved:
        print(f"  - {path.name}")


if __name__ == "__main__":
    main()
