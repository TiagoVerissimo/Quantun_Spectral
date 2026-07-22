# Bounding Spectral Gaps

This repository accompanies *Shift-Invariant Spectral-Gap Continuation from Local Bounds*. It contains the implementation, numerical data, figures, and verification tools used to study spectral-gap bounds for adiabatic quantum-computing Hamiltonians.

The computational experiments apply an adaptive continuation method to symmetry-reduced weighted Max-Cut instances. The method estimates local spectral gaps with sparse eigensolvers and propagates those estimates over intervals of the interpolation parameter using a shift-invariant Weyl bound.

The archived study contains 60 instances at effective problem sizes `N=10`, `N=12`, and `N=14`, with 20 seeded graphs at each size. This README explains how to verify the archived artifacts, reproduce the numerical experiment, regenerate every computational table and figure in `main.tex`, and compile the manuscript.

## Archived cohort

The primary Level 2 archive contains exactly **60 instances**:

- effective qubit sizes `N=10`, `N=12`, and `N=14`, corresponding to `n_v=11`, `n_v=13`, and `n_v=15` physical vertices;
- seeds `0` through `19` once at each size;
- connected weighted Erdős--Rényi graphs with `p=0.5` and a pinned endpoint optimum that `verify_artifacts.py` checks exactly for nondegeneracy.

## Scope of the numerical results

The aggregate CSV and sparse-eigensolver records are conditional numerical results. Ritz residuals are recorded, but the calculations do not independently verify eigenvalue indexing or use enclosure arithmetic. The selected `N=10`, seed-0 dense-grid comparison is likewise a diagnostic consistency check rather than a continuous-path certificate.

The experiments evaluate the finite-size behavior, coverage, and sparse-eigensolve workload of the continuation method. They do not establish asymptotic gap scaling, adiabatic runtime bounds, or computational speedup.

## Numerical and artifact conventions

Archived binary64 edge weights are interpreted as exact dyadic rationals. If `W_exact` is their exact rational sum, then `W_upper` is the least binary64 value satisfying `W_upper >= W_exact`, and

\[
K_{\mathrm{gap,cert}}
=\operatorname{up}_{64}\!\left(
\operatorname{exact}_{64}(W_{\mathrm{upper}})+\Lambda_I
\right),
\qquad
\Lambda_I=2\left\lfloor\frac{n_v}{2}\right\rfloor.
\]

Here `up_64` denotes least upward binary64 rounding. The compatibility column `W` aliases `W_upper`; `W_model` remains the ordinary rounded Python sum used for model diagnostics. These conventions make the archived weight sums, propagation constants, and cross-file checks reproducible without changing the conditional status of the sparse-eigensolver bounds.

## Repository structure

- `main.py`: contains the shared Level 2 experiment implementation and regenerates the original 40-row `N=10,12` base cohort.
- `extend_level2_n14.py`: retains the archived base rows, evaluates seeds 0--19 at `N=14` through the same implementation, and atomically writes the 60-row aggregate CSV, graph archive, and run manifest.
- `section7_results.csv`: result-schema-4 aggregate metrics for the 60 retained instances, including exact rational and upward-rounding metadata.
- `summarize_section7_results.py`: strictly validates the cohort and exact rounding chain, then writes deterministic JSON and TeX summaries with weight, spectral-diameter, solve-count, and global conditional-bound statistics.
- `plot_new_results.py`: generates the aggregate solve-count and endpoint-coverage figures. It does not generate a complexity-scaling plot.
- `generate_updated_comparison.py`: generates the two `N=10`, seed-0 Weyl-envelope figures from the archived graph record.
- `scratch/plot_grid_vs_gap.py`: generates the `N=10`, `N=12`, and `N=14` seed-0 solve-density figures from archived graph records.
- `test_main.py`: exercises the analytic gap-width bound and continuation edge cases, including upward rounding, floor probes, invalid parameters, and zero-diameter paths.
- `verify_artifacts.py`: cross-checks the CSV, run manifest, graph metadata and hashes, summaries, selected Level 2 interval algebra, and required figures; it also exactly enumerates every pinned endpoint configuration to prove nondegeneracy for all 60 graph-schema-2 records.
- `maxcut_gap_benchmark.py`: a separate exploratory benchmark and shared Hamiltonian library; it does not generate the Section 7 archive.

## Reference environment

The recorded reference run used Python 3.12.10. Install the pinned Python dependencies from the repository root:

```bash
python --version
python -m pip install -r requirements.txt
```

A TeX distribution with `latexmk` is required only for the manuscript build and is not installed by `requirements.txt`.

Sparse ARPACK solves can differ slightly across SciPy, BLAS, CPU, and thread configurations. Graph and artifact identities are checked with exact SHA-256 values; endpoint nondegeneracy is checked with exact dyadic arithmetic and no floating tolerance; Level 2 floating-point algebra is checked with explicit numerical tolerances. Wall-clock timings are platform-, BLAS-, load-, process-count-, and scheduler-dependent and should not be expected to reproduce exactly.

## License and release practice

Copyright (c) 2026 Tiago Verissimo. This repository is distributed under the [MIT License](LICENSE). The license permits reuse subject to its notice and disclaimer; users remain responsible for checking compatibility with any third-party dependency or venue policy.

For a citable public release, create a Git tag and GitHub Release for the exact reviewed commit, then archive that release with a DOI service such as Zenodo. Record the release tag, commit SHA-256, and DOI in the manuscript's code-and-data availability statement. The current artifact records already contain SHA-256 links for the graph and run payloads.

## Reproducing the results in `main.tex`

The mathematical propositions and proofs in `main.tex` are analytic. The reproducibility pipeline below covers every computational table, figure, and quantitative observation in the manuscript's Computational Results section. Run all commands from the repository root.

### Route A: replay the archived results

Use this route to verify the evidence shipped with the repository without repeating the expensive sparse-eigensolver experiment:

```bash
python -m unittest -v test_main.py
python summarize_section7_results.py --check
python verify_artifacts.py
latexmk -pdf main.tex
```

The unit suite exercises the propagation constant, directed upward rounding, continuation edge cases, floor probing, and the zero-diameter branch. `summarize_section7_results.py --check` reconstructs the deterministic JSON and TeX summaries in memory and requires byte-for-byte agreement with the archived files. `verify_artifacts.py` then checks the complete 60-row chain:

- result-schema-4 rows for `N=10,12,14` and seeds 0--19;
- the run manifest, individual graph records, canonical graph archive, source hashes, and cross-file SHA-256 links;
- exact interpretation of every binary64 edge weight, directed rounding of the propagation bound, and exact endpoint nondegeneracy by enumeration of all pinned configurations;
- the selected `N=10`, seed-0 continuation log and its interval algebra;
- deterministic summary files and provenance metadata embedded in all required figures.

The checker is read-only and returns a nonzero status for an incomplete cohort, stale derived artifact, inconsistent hash, invalid rounding record, endpoint degeneracy, or missing figure. `latexmk` imports `results/section7_summary.tex` into `main.tex` and produces `main.pdf`.

### Route B: recompute the complete 60-instance experiment

This route replaces the archived numerical outputs. It performs many sparse ARPACK solves and can take several hours:

```bash
python main.py
python extend_level2_n14.py --processes 5
python summarize_section7_results.py
python plot_new_results.py
python generate_updated_comparison.py
python scratch/plot_grid_vs_gap.py
python verify_artifacts.py
latexmk -pdf main.tex
```

The order matters:

1. `main.py` regenerates the fixed 40-instance base cohort for `N=10,12`, seeds 0--19. It initially replaces `section7_results.csv`, `results/graph_instances.jsonl`, and `results/section7_run_manifest.json` with the 40-row base cohort. The driver uses five worker processes.
2. `extend_level2_n14.py` retains those 40 rows, computes the 20 `N=14` instances, and atomically replaces the aggregate CSV, graph archive, and manifest with the final 60-row cohort. Set `--processes` to a smaller positive number if memory is limited; this changes concurrency, not the requested instances.
3. `summarize_section7_results.py` validates the final cohort and generates `results/section7_summary.json` and `results/section7_summary.tex`.
4. The three plotting commands regenerate every manuscript figure from the validated CSV and archived graph records.
5. `verify_artifacts.py` checks the complete regenerated dependency chain before `latexmk` builds the paper.

Do not run the summary, plotting, or verification steps between steps 1 and 2: the intermediate archive intentionally contains only 40 rows, while the manuscript and validators require all 60.

### Manuscript artifact map

The generated table rows all come from `results/section7_summary.tex`, which is created solely from `section7_results.csv`:

| Manuscript result | Generated macro | Source fields |
| --- | --- | --- |
| Table `tab:gap_diameter` | `\SectionSevenGapDiameterRows` | estimated spectral diameter and `K_gap_cert` |
| Table `tab:bound_comparison` | `\SectionSevenBoundComparisonRows` | previous, uncentered, and shift-invariant propagation constants |
| Table `tab:endpoint_coverage` | `\SectionSevenCoverageRows` | 201-point endpoint-bound coverage fractions |
| Table `tab:anchor_stats` | `\SectionSevenSolveCountRows` | uniform-grid and adaptive sparse-eigensolve counts |
| Table `tab:global-conditional` | `\SectionSevenGlobalConditionalRows` | per-run global conditional lower bounds |
| Table `tab:unresolved_stats` | `\SectionSevenUnresolvedRows` | merged unresolved-window counts and widths |

The scripts and output files for the manuscript figures are:

| Command | Manuscript outputs |
| --- | --- |
| `python generate_updated_comparison.py` | `results/comparison_updated_N10_no_anchors.png`, `results/comparison_updated_N10_anchors.png` |
| `python plot_new_results.py` | `results/efficiency_comparison.png` |
| `python scratch/plot_grid_vs_gap.py` | `results/grid_density_vs_gap.png`, `results/grid_density_vs_gap_N12.png`, `results/grid_density_vs_gap_N14.png` |

`plot_new_results.py` also writes `results/certificate_coverage.png`. This is a checked archive diagnostic, although it is not currently included as a figure in `main.tex`. All plotting scripts use a headless Matplotlib backend, resolve paths relative to the repository root, and embed the source CSV hash in the PNG metadata. Instance-specific plots also embed the corresponding graph-record hash.

The explicit workload observations in the manuscript—including the `N=12` seed-9 and seed-13 call counts, the single width-`0.001` unresolved window, the `N=14` seed-7 maximum, ensemble means and medians, and the resolved-run counts—are computed from the same 60 CSV rows by `summarize_section7_results.py`. The integral-cost illustration (`56.52` estimated calls versus 57 observed calls) comes from the selected `N=10`, seed-0 row and its `results/conditional_log_N10_seed0.json` record. `verify_artifacts.py` cross-checks that selected record against the CSV and graph data.

### Numerical reproducibility expectations

Graph generation is fixed by the archived seeds and graph schema, but sparse eigensolver results and wall-clock measurements can vary slightly with SciPy, BLAS, CPU, process scheduling, and warm-start behavior. Use the pinned Python versions above for the closest reproduction. The manuscript treats these outputs as conditional floating-point results; wall times are recorded diagnostics and are not expected to match exactly. The artifact checker uses exact comparisons for identities, rational metadata, and endpoint enumeration, and explicit tolerances for the documented floating-point algebra.

To remove LaTeX auxiliary files without touching numerical artifacts, run:

```bash
latexmk -c main.tex
```

## Exploratory benchmark

The standalone exploratory driver has a separate output contract:

```bash
python maxcut_gap_benchmark.py -h
```

Its outputs are not checked as part of the archived Section 7 artifact chain.
