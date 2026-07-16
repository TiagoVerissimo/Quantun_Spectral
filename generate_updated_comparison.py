import os

import matplotlib.pyplot as plt
import numpy as np

from main import er_graph, pinned_cost_vector
from maxcut_gap_benchmark import (
    adaptive_weyl,
    driver_matrix,
    exact_gap_curve,
    weyl_envelope,
)


def generate_graphs():
    outdir = "results"
    os.makedirs(outdir, exist_ok=True)

    N = 10
    p = 0.5
    seed = 0
    rng = np.random.default_rng(seed)
    n_v = N + 1
    while True:
        edges = er_graph(n_v, p, rng)
        cost = pinned_cost_vector(N, edges)
        two_lowest = np.partition(cost, 1)[:2]
        if two_lowest[1] > two_lowest[0]:
            break

    driver = driver_matrix(n_v, sector=True)
    s_grid = np.linspace(0.0, 1.0, 201)

    print("Computing numerical reference gap curve...")
    e0, e1 = exact_gap_curve(s_grid, driver, cost)
    gap = e1 - e0

    endpoint_anchors = [
        (0.0, 2.0),
        (1.0, float(np.sort(cost)[1] - np.sort(cost)[0])),
    ]
    total_weight = sum(edge[2] for edge in edges)
    norm_bound = float(total_weight + n_v)

    print("Computing Weyl endpoint envelope...")
    endpoint_bound = weyl_envelope(s_grid, endpoint_anchors, norm_bound)
    endpoint_coverage = np.mean(endpoint_bound > 0.0) * 100.0

    plt.figure(figsize=(8, 5))
    plt.plot(s_grid, gap, "k-", lw=2.5, label=r"Numerical Reference Gap $\Delta(s)$")
    plt.plot(
        s_grid,
        endpoint_bound,
        color="#3a86c8",
        lw=1.8,
        label=f"Weyl Bound (Endpoints) [Positive: {endpoint_coverage:.1f}% of path]",
    )
    plt.axhline(0.0, color="#9ea1a5", lw=0.8)
    plt.xlabel("Annealing Parameter (s)")
    plt.ylabel("Spectral Gap")
    plt.ylim(-0.2, 1.25 * gap.max())
    plt.title(f"Weyl Bound from Endpoint Data (N={N}, Seed={seed})")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "comparison_updated_N10_no_anchors.png"), dpi=160)
    plt.close()

    budgets = [5, 15, 25]
    bounds = []
    anchor_counts = []
    print("Computing adaptive Weyl envelopes...")
    for budget in budgets:
        bound, anchors, _, _, _ = adaptive_weyl(
            s_grid, endpoint_anchors, norm_bound, driver, cost, max_anchors=budget
        )
        bounds.append(bound)
        anchor_counts.append(len(anchors))

    fig, axes = plt.subplots(3, 1, figsize=(10, 14), sharex=True)
    for axis, budget, count, bound in zip(axes, budgets, anchor_counts, bounds):
        axis.plot(s_grid, gap, color="#1e1e24", lw=2.5, label=r"Numerical Reference Gap $\Delta(s)$")
        axis.plot(
            s_grid,
            bound,
            color="#3a86c8",
            lw=1.8,
            label=f"Adaptive Weyl ({count} eigensolves + exact endpoint)",
        )
        axis.axhline(0.0, color="#9ea1a5", lw=0.8)
        axis.set_title(f"Adaptive Weyl eigensolve budget: {budget}", fontsize=11, fontweight="bold")
        axis.set_ylabel("Gap / Bound Value", fontsize=10)
        axis.grid(True, linestyle=":", alpha=0.5)
        axis.legend(loc="upper right", fontsize=9)
        axis.set_ylim(-0.2, gap.max() * 1.15)

    axes[-1].set_xlabel("Annealing Parameter (s)", fontsize=11)
    fig.suptitle(f"Weyl Bounds with Increasing Anchor Budgets: N={N}, Seed={seed}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    plt.savefig(os.path.join(outdir, "comparison_updated_N10_anchors.png"), dpi=160)
    plt.close()

    print("Generated Weyl-only comparison figures.")


if __name__ == "__main__":
    generate_graphs()
