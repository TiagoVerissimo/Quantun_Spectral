import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add current workspace directory to sys.path
sys.path.append(os.path.abspath("."))

from main import (
    PathOperator, er_graph, pinned_cost_vector, path_lipschitz, certified_sweep, lowest_two
)

os.makedirs("results", exist_ok=True)

def generate_grid_density_plot(N, seed=0, p=0.5, delta_target=0.25):
    print(f"Running grid density analysis for N={N}, seed={seed}...")
    rng = np.random.default_rng(seed)
    n = N + 1
    
    # Ensure a nondegenerate cost vector (matching the driver resampling loop)
    while True:
        edges = er_graph(n, p, rng)
        c = pinned_cost_vector(N, edges)
        two = np.partition(c, 1)[:2]
        if two[1] > two[0]:
            break
            
    pathop = PathOperator(N, c)
    L = path_lipschitz(pathop)
    
    # 1. Run certified continuation sweep to get the hybrid anchors
    records, windows = certified_sweep(pathop, L, delta_target)
    s_hybrid = np.array([r[0] for r in records])
    
    # 2. Get uniform grid anchors of equal rigor
    h_min = delta_target / (2 * L)
    n_uniform = int(np.ceil(1.0 / h_min)) + 1
    s_uniform = np.linspace(0.0, 1.0, n_uniform)
    
    # 3. Compute true spectral gap along s
    sgrid = np.linspace(0.0, 1.0, 201)
    true_gaps = []
    for s in sgrid:
        vals, _, _ = lowest_two(pathop.H(s), tol=1e-8)
        true_gaps.append(vals[1] - vals[0])
    true_gaps = np.array(true_gaps)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    colors = {
        'true': '#1e1e24',      # Dark Charcoal
        'hybrid': '#3a86c8',    # Soft Blue
        'uniform': '#f25c54',   # Coral Red
        'grid': '#9ea1a5'
    }
    
    # Plot true gap curve
    plt.plot(sgrid, true_gaps, color=colors['true'], lw=2.5, label=r"True Spectral Gap $\Delta(s)$")
    
    # Find the true gap values at the hybrid anchor points to plot them on the curve
    gap_hybrid = []
    for s in s_hybrid:
        vals, _, _ = lowest_two(pathop.H(s), tol=1e-8)
        gap_hybrid.append(vals[1] - vals[0])
    gap_hybrid = np.array(gap_hybrid)
    
    # Plot Hybrid anchor points on the gap curve
    plt.scatter(s_hybrid, gap_hybrid, color=colors['hybrid'], s=45, zorder=5, 
                label=f"Hybrid Solve Points (Alg 1: {len(s_hybrid)} solves)")
    
    # Add "Rug Plots" (vertical tick marks) at the bottom to show density of solves
    rug_height = max(true_gaps) * 0.05
    plt.vlines(s_hybrid, -0.05, -0.05 + rug_height, colors=colors['hybrid'], lw=1.5, alpha=0.8,
               label="Hybrid Solve Density (Rug)")
    plt.vlines(s_uniform, -0.05 - rug_height, -0.05, colors=colors['uniform'], lw=1.0, alpha=0.6,
               label=f"Uniform Solve Density (Rug: {len(s_uniform)} solves)")
               
    # Formatting
    plt.axhline(0.0, color='gray', lw=0.8)
    plt.xlabel("Annealing Parameter (s)", fontweight="bold", fontsize=11)
    plt.ylabel("Spectral Gap", fontweight="bold", fontsize=11)
    plt.title(f"Grid Solve Density vs. True Spectral Gap (N={N}, Instance Seed={seed})", fontsize=13, fontweight="bold", pad=15)
    plt.grid(True, linestyle=":", alpha=0.4)
    plt.ylim(-0.05 - 1.2 * rug_height, max(true_gaps) * 1.1)
    plt.legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    
    if N == 10:
        out_path = "results/grid_density_vs_gap.png"
    else:
        out_path = f"results/grid_density_vs_gap_N{N}.png"
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"Saved {out_path} successfully.")

if __name__ == "__main__":
    for N in [10, 12, 14]:
        generate_grid_density_plot(N, seed=0)
