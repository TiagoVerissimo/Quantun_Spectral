import os
import sys
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

from main import er_graph, pinned_cost_vector
from maxcut_gap_benchmark import (
    driver_matrix, exact_gap_curve, weyl_lipschitz_constant,
    lowest_eigs, path_hamiltonian, adaptive_weyl, weyl_envelope
)
from horn_anchors_test import horn_anchored_bound

def generate_graphs():
    outdir = "results"
    os.makedirs(outdir, exist_ok=True)
    
    # 1. Generate the same instance as plot_grid_vs_gap.py
    N = 10
    p = 0.5
    seed = 0
    rng = np.random.default_rng(seed)
    n = N + 1
    while True:
        edges = er_graph(n, p, rng)
        c = pinned_cost_vector(N, edges)
        two = np.partition(c, 1)[:2]
        if two[1] > two[0]:
            break
            
    DP = c
    HI = driver_matrix(N, sector=False)
    
    s_grid = np.linspace(0.0, 1.0, 201)
    
    # Exact gap
    print("Computing exact gap curve...")
    E0, E1 = exact_gap_curve(s_grid, HI, DP)
    gap = E1 - E0
    
    # Endpoints
    print("Computing endpoints and Lipschitz constant...")
    gap_s0 = float(np.sort(np.linalg.eigvalsh(HI.toarray()))[1])
    gap_s1 = float(np.sort(DP)[1] - np.sort(DP)[0])
    endpoint_anchors = [(0.0, gap_s0), (1.0, gap_s1)]
    
    # L
    L = weyl_lipschitz_constant(HI, DP, exact=True)
    
    # --- PART 1: Without Anchors (Endpoints only) ---
    print("Computing bounds without anchors...")
    weyl_no_anchors = weyl_envelope(s_grid, endpoint_anchors, L)
    horn_no_anchors = horn_anchored_bound(s_grid, [0.0, 1.0], HI, DP)
    
    weyl_percent = np.mean(weyl_no_anchors > 0) * 100
    horn_percent = np.mean(horn_no_anchors > 0) * 100
    
    print(f"Percentage of area certified > 0 without anchors:")
    print(f"  Weyl: {weyl_percent:.2f}%")
    print(f"  Horn: {horn_percent:.2f}%")
    
    # Plot Part 1
    print("Generating part 1 plot...")
    plt.figure(figsize=(8, 5))
    plt.plot(s_grid, gap, 'k-', lw=2.5, label=r"Exact Gap $\Delta(s)$")
    plt.plot(s_grid, weyl_no_anchors, color='#3a86c8', linestyle="-", lw=1.8, label=f"Weyl Bound (Endpoints) [Certified Positive: {weyl_percent:.1f}% of path]")
    plt.plot(s_grid, horn_no_anchors, color='#f25c54', linestyle="--", lw=1.8, label=f"Horn Bound (Endpoints) [Certified Positive: {horn_percent:.1f}% of path]")
    plt.axhline(0.0, color="#9ea1a5", lw=0.8, linestyle="-")
    plt.xlabel("Annealing Parameter (s)")
    plt.ylabel("Spectral Gap")
    plt.ylim(-0.2, 1.25 * gap.max())  # Prevent stretching down to -25
    plt.title(f"Bounds Without Anchors (N={N}, Seed={seed})")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "comparison_updated_N10_no_anchors.png"), dpi=160)
    plt.close()
    
    # --- PART 2: Three trials of increasing number of anchors ---
    def oracle(s):
        w, _ = lowest_eigs(path_hamiltonian(s, HI, DP), k=6)
        return float(w[1] - w[0])
        
    num_anchors_list = [5, 15, 25]
    weyl_bounds = []
    print("Computing Weyl bounds with anchors...")
    for num_anc in num_anchors_list:
        weyl_b, _, _, _ = adaptive_weyl(s_grid, endpoint_anchors, L, oracle, max_anchors=num_anc)
        weyl_bounds.append(weyl_b)
        
    horn_bounds = []
    print("Computing Horn bounds with anchors...")
    for num_anc in num_anchors_list:
        horn_b = horn_anchored_bound(s_grid, np.linspace(0, 1, num_anc), HI, DP)
        horn_bounds.append(horn_b)
        
    # Plot Part 2
    print("Generating part 2 plot...")
    fig, axes = plt.subplots(3, 1, figsize=(10, 14), sharex=True)
    
    colors = {
        'exact': '#1e1e24',      
        'weyl': '#3a86c8',       
        'horn': '#4895ef'        
    }
    
    for idx, (num_anc, w_b, h_b) in enumerate(zip(num_anchors_list, weyl_bounds, horn_bounds)):
        ax = axes[idx]
        ax.plot(s_grid, gap, color=colors['exact'], lw=2.5, label=r"Exact Gap $\Delta(s)$")
        ax.plot(s_grid, w_b, color=colors['weyl'], linestyle="-", lw=1.8, label=f"Adaptive Weyl ({num_anc} anchors)")
        ax.plot(s_grid, h_b, color=colors['horn'], linestyle="--", lw=1.8, label=f"Anchored Horn ({num_anc} anchors)")
        
        ax.axhline(0.0, color="#9ea1a5", lw=0.8, linestyle="-")
        ax.set_title(f"Configuration: {num_anc} Anchors", fontsize=11, fontweight='bold', color='#2b2d42')
        ax.set_ylabel("Gap / Bound Value", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="upper right", fontsize=9)
        
        ymin = -0.2  # Prevent stretching down to highly negative values
        ymax = gap.max() * 1.15
        ax.set_ylim(ymin, ymax)
        
    axes[2].set_xlabel("Annealing Parameter (s)", fontsize=11)
    fig.suptitle(f"Method Comparison with Increasing Anchors: N={N}, Seed={seed}", 
                 fontsize=14, fontweight='bold', color='#2b2d42')
    fig.tight_layout()
    
    plt.savefig(os.path.join(outdir, "comparison_updated_N10_anchors.png"), dpi=160)
    plt.close()
    
    print("Done generating updated comparison graphs.")

if __name__ == "__main__":
    generate_graphs()
