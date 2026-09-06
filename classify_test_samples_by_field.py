"""Classify test patches using a fixed HMI strong-field criterion.

Dependencies: Python >= 3.10, NumPy >= 1.23, pandas >= 1.5,
Matplotlib >= 3.6. Random seed: 42.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from constants import h_shape_lr, w_shape_lr
from utils_load_data import load_output_sample


RANDOM_SEED = 42  # No stochastic operation is used.
STRONG_FIELD_THRESHOLD_G = 100.0
AR_P99_THRESHOLD_G = 600.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify AR-like patches using HMI Q99.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("ltew/test_20260416"),
        help="Experiment directory containing test_pred.",
    )
    parser.add_argument("--dataset-type", default="test")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to <run-dir>/<dataset>_sample_classification.csv.",
    )
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def natural_key(path: Path) -> tuple[int, str]:
    match = re.search(r"sample_(\d+)", path.name)
    return (int(match.group(1)) if match else 10**12, path.name)


def extract_features(npz_path: Path) -> dict[str, float | int | str]:
    inp, _, _, sample_idx = load_output_sample(npz_path)
    field = np.asarray(inp[0], dtype=np.float64)
    valid = field[np.isfinite(field)]
    if valid.size == 0:
        raise ValueError(f"No finite HMI pixels in {npz_path.name}")

    abs_field = np.abs(valid)
    positive_fraction = np.mean(valid >= STRONG_FIELD_THRESHOLD_G)
    negative_fraction = np.mean(valid <= -STRONG_FIELD_THRESHOLD_G)

    dist_field = np.asarray(inp[1], dtype=np.float64)
    distance_arcsec = dist_field[h_shape_lr // 2, w_shape_lr // 2]
    theta_deg = np.degrees(np.arcsin(np.clip(distance_arcsec / 960.0, -1.0, 1.0)))

    return {
        "prediction_file": npz_path.name,
        "sample_idx": int(sample_idx),
        "theta_deg": float(theta_deg),
        "mean_B_G": float(np.mean(valid)),
        "mean_abs_B_G": float(np.mean(abs_field)),
        "std_B_G": float(np.std(valid)),
        "p95_abs_B_G": float(np.percentile(abs_field, 95)),
        "p99_abs_B_G": float(np.percentile(abs_field, 99)),
        "max_abs_B_G": float(np.max(abs_field)),
        "strong_fraction_100G": float(np.mean(abs_field >= STRONG_FIELD_THRESHOLD_G)),
        "strong_fraction_300G": float(np.mean(abs_field >= 300.0)),
        "strong_pixel_count_600G": int(np.sum(abs_field >= AR_P99_THRESHOLD_G)),
        "strong_fraction_600G": float(np.mean(abs_field >= AR_P99_THRESHOLD_G)),
        "positive_fraction_100G": float(positive_fraction),
        "negative_fraction_100G": float(negative_fraction),
        "bipolar_100G": bool(positive_fraction > 0 and negative_fraction > 0),
    }


def classify_samples(table: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    table = table.copy()
    table["is_ar"] = table["p99_abs_B_G"] >= AR_P99_THRESHOLD_G
    table["region_type"] = np.where(table["is_ar"], "AR", "non-AR")

    metadata = {
        "method": "fixed upper-tail threshold on the HMI magnetic-field distribution",
        "random_seed": RANDOM_SEED,
        "stochastic_operation_used": False,
        "input": "HMI field only; SP ground truth and model prediction are not used",
        "criterion": "AR if p99(abs(B_HMI)) >= 600 G; non-AR otherwise",
        "percentile": 99.0,
        "ar_p99_threshold_G": AR_P99_THRESHOLD_G,
        "interpretation": "approximately 1% of patch pixels have abs(B_HMI) >= 600 G",
        "label_scope": "AR-like magnetic-activity class, not NOAA/HARP membership",
        "num_samples": int(len(table)),
        "num_ar": int(table["is_ar"].sum()),
        "num_non_ar": int((~table["is_ar"]).sum()),
    }
    return table, metadata


def plot_classification(table: pd.DataFrame, output_path: Path, show: bool) -> None:
    colors = {"AR": "crimson", "non-AR": "steelblue"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for label in ("non-AR", "AR"):
        subset = table[table["region_type"] == label]
        axes[0].hist(
            subset["p99_abs_B_G"], bins=40, histtype="step", linewidth=2,
            color=colors[label], label=f"{label} ({len(subset)})"
        )
        axes[1].scatter(
            subset["strong_fraction_100G"], subset["mean_abs_B_G"],
            s=8, alpha=0.45, color=colors[label], label=label
        )
        axes[2].hist(
            subset["theta_deg"], bins=np.arange(0, 91, 5), histtype="step",
            linewidth=2, color=colors[label], label=label
        )

    axes[0].set(xlabel=r"$|B_{\rm HMI}|_{99}$ (G)", ylabel="Sample count")
    axes[1].set(xlabel=r"Fraction($|B_{\rm HMI}|\geq100$ G)", ylabel=r"Mean $|B_{\rm HMI}|$ (G)")
    axes[2].set(xlabel="Heliocentric angle (deg)", ylabel="Sample count", xlim=(0, 90))
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle(r"AR-like classification: $Q_{99}(|B_{HMI}|) \geq 600$ G")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = parse_args()
    prediction_dir = args.run_dir / f"{args.dataset_type}_pred"
    files = sorted(prediction_dir.glob("*.npz"), key=natural_key)
    if not files:
        raise FileNotFoundError(f"No NPZ predictions found in {prediction_dir}")

    records = []
    for index, path in enumerate(files, start=1):
        records.append(extract_features(path))
        if index % 200 == 0 or index == len(files):
            print(f"Extracted magnetic features: {index}/{len(files)}")

    table, metadata = classify_samples(pd.DataFrame(records))
    table = table.sort_values("sample_idx").reset_index(drop=True)

    output_path = args.output or args.run_dir / f"{args.dataset_type}_sample_classification.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False, encoding="utf-8-sig")

    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    figure_path = output_path.with_name(f"{output_path.stem}_distribution.png")
    plot_classification(table, figure_path, args.show)

    print(f"AR samples: {metadata['num_ar']}")
    print(f"non-AR samples: {metadata['num_non_ar']}")
    print(f"Saved classification: {output_path}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Saved diagnostic figure: {figure_path}")


if __name__ == "__main__":
    main()
