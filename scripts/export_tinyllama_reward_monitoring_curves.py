"""Export diagnostic ArmoRM reward-monitoring curves from summary CSV files."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = "results/tinyllama_helpsteer2_reward_monitoring"
DEFAULT_OUTPUT_DIR = "results/tinyllama_helpsteer2_reward_monitoring/plots"
ATTRIBUTE_ORDER = (
    "helpfulness",
    "correctness",
    "coherence",
    "complexity",
    "verbosity",
)


def resolve_project_path(value: str) -> Path:
    """Resolve a path relative to the repository root."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    """Parse lightweight reward-curve export settings."""
    parser = argparse.ArgumentParser(
        description="Export ArmoRM mean-reward monitoring curves from CSV logs."
    )
    parser.add_argument("--log_dir", "--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--output_dir", "--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--smoothing", type=float, default=0.5)
    parser.add_argument("--theme", choices=("dark", "light"), default="dark")
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def load_summary(path: Path) -> tuple[str, list[tuple[float, float]]]:
    """Load one attribute's global-step and mean-reward series."""
    with path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        required = {"attribute", "global_step", "mean_reward"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
        attributes: set[str] = set()
        by_step: dict[float, float] = {}
        for row in reader:
            attribute = str(row["attribute"]).strip()
            if attribute:
                attributes.add(attribute)
            try:
                step = float(row["global_step"])
                reward = float(row["mean_reward"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid numeric value in {path}: {row}") from error
            if not math.isfinite(step) or not math.isfinite(reward):
                raise ValueError(f"Non-finite numeric value in {path}: {row}")
            by_step[step] = reward
    if len(attributes) != 1:
        raise ValueError(f"Expected exactly one attribute in {path}: {attributes}")
    if not by_step:
        raise ValueError(f"No reward rows found in {path}")
    return next(iter(attributes)), sorted(by_step.items())


def smooth(values: list[float], smoothing: float) -> list[float]:
    """Return an exponential moving average with the first value as its start."""
    if not values or smoothing == 0:
        return list(values)
    result = [values[0]]
    for value in values[1:]:
        result.append(smoothing * result[-1] + (1 - smoothing) * value)
    return result


def ordered_attributes(series: dict[str, object]) -> list[str]:
    """Return the fixed HelpSteer2 order followed by any extra attributes."""
    known = [attribute for attribute in ATTRIBUTE_ORDER if attribute in series]
    return [*known, *sorted(set(series).difference(known))]


def configure_axis(figure, axis, theme: str) -> None:
    """Apply a TensorBoard-like plot background and labels."""
    background = "#303030" if theme == "dark" else "#ffffff"
    figure.patch.set_facecolor(background)
    axis.set_facecolor(background)
    axis.set_xlabel("Global Step")
    axis.set_ylabel("Mean ArmoRM Reward")
    axis.grid(True, alpha=0.3)


def save_plots(
    pyplot,
    series: dict[str, list[tuple[float, float]]],
    output_dir: Path,
    smoothing: float,
    theme: str,
    dpi: int,
) -> list[Path]:
    """Save individual and combined raw/smoothed reward curves."""
    saved: list[Path] = []
    for attribute in ordered_attributes(series):
        steps, rewards = zip(*series[attribute])
        figure, axis = pyplot.subplots(figsize=(7.2, 4.4))
        configure_axis(figure, axis, theme)
        axis.plot(steps, rewards, color="#65717f", alpha=0.55, label="raw")
        axis.plot(
            steps,
            smooth(list(rewards), smoothing),
            color="#ffffff" if theme == "dark" else "#111111",
            linewidth=2.0,
            label=f"smoothed ({smoothing:g})",
        )
        axis.set_title(f"{attribute}/armorm_mean_reward", loc="left")
        axis.legend()
        figure.tight_layout()
        path = output_dir / f"{attribute}_armorm_mean_reward.png"
        figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=figure.get_facecolor())
        pyplot.close(figure)
        saved.append(path)

    figure, axis = pyplot.subplots(figsize=(8.2, 5.0))
    configure_axis(figure, axis, theme)
    for attribute in ordered_attributes(series):
        steps, rewards = zip(*series[attribute])
        (raw_line,) = axis.plot(steps, rewards, alpha=0.3, label="_nolegend_")
        axis.plot(
            steps,
            smooth(list(rewards), smoothing),
            color=raw_line.get_color(),
            linewidth=2.0,
            label=attribute,
        )
    axis.set_title("All attributes: ArmoRM mean reward", loc="left")
    axis.legend()
    figure.tight_layout()
    combined_path = output_dir / "tinyllama_all_attributes_armorm_mean_reward.png"
    figure.savefig(
        combined_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
    )
    pyplot.close(figure)
    saved.append(combined_path)
    return saved


def main() -> None:
    """Load summary CSVs and export plots without loading either model."""
    args = parse_args()
    if not 0 <= args.smoothing < 1:
        raise ValueError("smoothing must be at least 0 and less than 1.")
    if args.dpi < 1:
        raise ValueError("dpi must be positive.")
    log_dir = resolve_project_path(args.log_dir)
    output_dir = resolve_project_path(args.output_dir)
    paths = sorted(log_dir.glob("tinyllama_helpsteer2_*_reward_summary.csv"))
    if not paths:
        raise FileNotFoundError(f"No reward summary CSV files found in {log_dir}")

    series: dict[str, list[tuple[float, float]]] = {}
    for path in paths:
        attribute, points = load_summary(path)
        if attribute in series:
            raise ValueError(f"Multiple summary files found for {attribute!r}.")
        series[attribute] = points

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot
    except ImportError as error:
        raise ImportError(
            "Reward-curve export requires matplotlib: pip install matplotlib"
        ) from error
    pyplot.style.use("dark_background" if args.theme == "dark" else "default")
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = save_plots(
        pyplot,
        series,
        output_dir,
        args.smoothing,
        args.theme,
        args.dpi,
    )
    print(f"Exported {len(saved)} ArmoRM reward plots to {output_dir}")
    for path in saved:
        print(f"  - {path.name}")


if __name__ == "__main__":
    sys.exit(main())
