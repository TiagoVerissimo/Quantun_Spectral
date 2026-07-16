# Bounding Spectral Gaps

This repository contains the code and data for the computational study in *A Numerical Realization of Matrix-Analysis Certificates for Adiabatic Spectral Gaps*.

The code implements Weyl--Lipschitz continuation and positive-semidefinite energy-floor bounds for spectral gaps of adiabatic quantum computing paths, using weighted Max-Cut instances on random graphs.

## Repository structure

- **`main.py`**: Primary ensemble driver. It generates the 60 archived weighted Erdős--Rényi instances, runs Algorithm 1 and the uniform workload reference, and writes `section7_results.csv`. With validation enabled, it also writes a self-identifying conditional Level-2 JSON diagnostic record.
- **`maxcut_gap_benchmark.py`**: Standalone exploratory benchmark and shared Hamiltonian library. It implements Weyl--Lipschitz and PSD-floor bounds, alongside Krylov and two-level comparators. Its CLI writes `results/results.csv`; it does not generate the archived Section 7 CSV.
- **`generate_updated_comparison.py`**: Generates the Weyl-only endpoint and increasing-anchor comparison figures for the selected `N=10`, seed-0 instance.
- **`plot_new_results.py`**: Reads `section7_results.csv` and generates summary plots, including the adaptive-versus-uniform solve-count comparison.
- **`section7_results.csv`**: Archived numerical benchmarking metrics across the 60 instances.
- **`results/certificate_log_N10_seed0.json`**: Self-identifying conditional Level-2 record for the selected densely checked run, including the full instance, hashes, Ritz values, residuals, solver policy, unresolved set, and software/repository metadata.

## Reproduction

Install the pinned dependencies:

```bash
pip install -r requirements.txt
```

Generate the figures used in the manuscript:

```bash
python generate_updated_comparison.py
python plot_new_results.py
python scratch/plot_grid_vs_gap.py
```

Regenerate the archived Section 7 ensemble:

```bash
python main.py
```

This executes 20 seeds for each effective size \(N\in\{10,12,14\}\) and can take several hours on a laptop.

Run the standalone exploratory benchmark:

```bash
python maxcut_gap_benchmark.py -h
```

The numerical results are conditional Level-2 realizations based on floating-point sparse eigensolves and residual diagnostics; they are not verified Level-3 indexed eigenvalue enclosures.
