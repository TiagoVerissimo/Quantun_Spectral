from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from generate_updated_comparison import (  # noqa: E402
    load_archived_instance,
    sampled_ritz_gap_curve,
)
from main import PathOperator, certified_sweep  # noqa: E402
from summarize_section7_results import (  # noqa: E402
    ArtifactValidationError,
    RESULTS_PATH,
    sha256_path,
)


OUTPUT_DIR = PROJECT_ROOT / "results"


def generate_grid_density_plot(
    N: int,
    seed: int = 0,
    delta_target: float = 0.25,
) -> Path:
    if N not in (10, 12):
        raise ValueError("the archived grid-density figures are limited to N=10,12")
    if delta_target <= 0.0 or not np.isfinite(delta_target):
        raise ValueError("delta_target must be finite and positive")

    print(f"Running grid-density analysis for N={N}, seed={seed}...")
    instance = load_archived_instance(N=N, seed=seed)
    path_operator = PathOperator(N, instance.costs)

    records, _, _ = certified_sweep(
        path_operator,
        instance.K_gap_cert,
    )
    s_adaptive = np.asarray([float(record[0]) for record in records])
    adaptive_ritz_gaps = np.asarray(
        [float(record[2] - record[1]) for record in records]
    )

    if instance.K_gap_cert == 0.0:
        s_uniform = np.asarray([0.0])
    else:
        n_uniform = int(
            np.ceil(instance.K_gap_cert / float(delta_target))
        ) + 1
        s_uniform = np.linspace(0.0, 1.0, n_uniform)

    s_grid = np.linspace(0.0, 1.0, 201)
    sampled_gaps = sampled_ritz_gap_curve(path_operator, s_grid)

    colors = {
        "reference": "#1e1e24",
        "adaptive": "#3a86c8",
        "uniform": "#f25c54",
    }
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(
        s_grid,
        sampled_gaps,
        color=colors["reference"],
        linewidth=2.5,
        label="Sampled Ritz gap",
    )
    axis.scatter(
        s_adaptive,
        adaptive_ritz_gaps,
        color=colors["adaptive"],
        s=45,
        zorder=5,
        label=f"Adaptive solve points ({len(s_adaptive)} eigensolves)",
    )

    maximum_gap = float(np.max(sampled_gaps))
    rug_height = max(maximum_gap * 0.05, 1e-3)
    rug_baseline = -rug_height
    axis.vlines(
        s_adaptive,
        rug_baseline,
        rug_baseline + rug_height,
        colors=colors["adaptive"],
        linewidth=1.5,
        alpha=0.8,
        label="Adaptive solve density (rug)",
    )
    axis.vlines(
        s_uniform,
        rug_baseline - rug_height,
        rug_baseline,
        colors=colors["uniform"],
        linewidth=1.0,
        alpha=0.6,
        label=f"Fixed-grid density (rug: {len(s_uniform)} calls)",
    )
    axis.axhline(0.0, color="gray", linewidth=0.8)
    axis.set_xlabel("Interpolation parameter $s$", fontweight="bold", fontsize=11)
    axis.set_ylabel("Reduced-sector Ritz gap", fontweight="bold", fontsize=11)
    axis.set_title(
        f"Solve density versus sampled Ritz gap ($N={N}$, seed {seed})",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    axis.grid(True, linestyle=":", alpha=0.4)
    axis.set_ylim(rug_baseline - 1.2 * rug_height, maximum_gap * 1.1)
    axis.legend(fontsize=9, loc="upper right")
    figure.tight_layout()

    output = (
        OUTPUT_DIR / "grid_density_vs_gap.png"
        if N == 10
        else OUTPUT_DIR / f"grid_density_vs_gap_N{N}.png"
    )
    figure.savefig(
        output,
        dpi=160,
        metadata={
            "Artifact": output.name,
            "Section7CSV-SHA256": sha256_path(RESULTS_PATH),
            "GraphRecord-SHA256": str(
                instance.row["graph_record_sha256"]
            ),
        },
    )
    plt.close(figure)
    print(f"Saved {output.relative_to(PROJECT_ROOT)}.")
    return output


def main() -> int:
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        for N in (10, 12):
            generate_grid_density_plot(N, seed=0)
    except (ArtifactValidationError, OSError, ValueError) as exc:
        print(f"grid-density plot generation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
