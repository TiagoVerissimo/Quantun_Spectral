from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from summarize_section7_results import (
    ArtifactValidationError,
    EXPECTED_NS,
    RESULTS_PATH,
    load_rows,
    sha256_path,
)


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "results"


def _group_rows() -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in load_rows(RESULTS_PATH):
        grouped[int(row["N"])].append(row)
    if tuple(sorted(grouped)) != EXPECTED_NS:
        raise ArtifactValidationError(
            f"plots require exactly N={EXPECTED_NS}, found {tuple(sorted(grouped))}"
        )
    return dict(grouped)


def _median_iqr(values: np.ndarray) -> tuple[float, float, float]:
    median = float(np.median(values))
    lower = median - float(np.percentile(values, 25))
    upper = float(np.percentile(values, 75)) - median
    return median, lower, upper


def plot_efficiency(
    grouped: dict[int, list[dict[str, Any]]], csv_sha256: str
) -> Path:
    medians_adaptive = []
    lower_adaptive = []
    upper_adaptive = []
    medians_uniform = []
    lower_uniform = []
    upper_uniform = []

    for N in EXPECTED_NS:
        rows = grouped[N]
        adaptive = np.asarray(
            [int(row["solves_hybrid"]) for row in rows], dtype=float
        )
        uniform = np.asarray(
            [int(row["solves_uniform"]) for row in rows], dtype=float
        )
        median, lower, upper = _median_iqr(adaptive)
        medians_adaptive.append(median)
        lower_adaptive.append(lower)
        upper_adaptive.append(upper)
        median, lower, upper = _median_iqr(uniform)
        medians_uniform.append(median)
        lower_uniform.append(lower)
        upper_uniform.append(upper)

    positions = np.arange(len(EXPECTED_NS), dtype=float)
    width = 0.35
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(
        positions,
        medians_adaptive,
        yerr=np.asarray([lower_adaptive, upper_adaptive]),
        capsize=5,
        color="#3a86c8",
        width=width,
        edgecolor="grey",
        label="Adaptive continuation sweep",
    )
    axis.bar(
        positions + width,
        medians_uniform,
        yerr=np.asarray([lower_uniform, upper_uniform]),
        capsize=5,
        color="#f25c54",
        width=width,
        edgecolor="grey",
        label="Fixed-grid workload reference",
    )
    axis.set_xlabel("Effective qubits ($N$)", fontweight="bold", fontsize=11)
    axis.set_ylabel("Median sparse eigensolve calls", fontweight="bold", fontsize=11)
    axis.set_xticks(positions + width / 2.0, [str(N) for N in EXPECTED_NS])
    axis.set_title(
        "Solve-count workload: adaptive versus fixed grid",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    axis.grid(True, linestyle=":", alpha=0.6)
    axis.legend(fontsize=10)
    figure.tight_layout()
    output = OUTPUT_DIR / "efficiency_comparison.png"
    figure.savefig(
        output,
        dpi=160,
        metadata={
            "Artifact": output.name,
            "Section7CSV-SHA256": csv_sha256,
        },
    )
    plt.close(figure)
    return output


def plot_certificate_coverage(
    grouped: dict[int, list[dict[str, Any]]],
    csv_sha256: str,
) -> Path:
    weyl_means = []
    floor_means = []
    for N in EXPECTED_NS:
        rows = grouped[N]
        weyl_means.append(
            float(np.mean([float(row["frac_weyl"]) for row in rows]))
        )
        floor_means.append(
            float(np.mean([float(row["frac_floor"]) for row in rows]))
        )

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        EXPECTED_NS,
        weyl_means,
        "o-",
        color="#9ea1a5",
        linewidth=2,
        label="Weyl endpoint envelope",
    )
    axis.plot(
        EXPECTED_NS,
        floor_means,
        "d-",
        color="#3a86c8",
        linewidth=2,
        label="PSD floor (oracle inputs)",
    )
    axis.set_xlabel("Effective qubits ($N$)", fontweight="bold", fontsize=11)
    axis.set_ylabel(
        "Sampled fraction with positive conditional bound",
        fontweight="bold",
        fontsize=11,
    )
    axis.set_xticks(EXPECTED_NS)
    axis.set_ylim(0.0, max(0.35, 1.1 * max(weyl_means + floor_means)))
    axis.set_title(
        "Endpoint-bound sampled coverage",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    axis.grid(True, linestyle=":", alpha=0.6)
    axis.legend(fontsize=9, loc="upper right")
    figure.tight_layout()
    output = OUTPUT_DIR / "certificate_coverage.png"
    figure.savefig(
        output,
        dpi=160,
        metadata={
            "Artifact": output.name,
            "Section7CSV-SHA256": csv_sha256,
        },
    )
    plt.close(figure)
    return output


def main() -> int:
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        grouped = _group_rows()
        csv_sha256 = sha256_path(RESULTS_PATH)
        outputs = [
            plot_efficiency(grouped, csv_sha256),
            plot_certificate_coverage(grouped, csv_sha256),
        ]
    except ArtifactValidationError as exc:
        print(f"plot validation failed: {exc}", file=sys.stderr)
        return 1

    rendered = ", ".join(str(path.relative_to(PROJECT_ROOT)) for path in outputs)
    print(f"Generated {rendered} from the schema-v4 Section 7 CSV.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
