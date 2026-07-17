# Bounding Spectral Gaps

This repository contains the code and archived artifacts for the computational study in *Shift-Invariant Gap Continuation and Selected Replayable Level-3 Anchors*.

The Section 7 benchmark applies shift-invariant Weyl continuation to symmetry-reduced weighted Max-Cut Hamiltonians. Archived binary64 edge weights are interpreted as exact dyadic rationals. If `W_exact` is their exact rational sum, then `W_upper` is the least binary64 value satisfying `W_upper >= W_exact`, and

\[
K_{\mathrm{gap,cert}}
=\operatorname{up}_{64}\!\left(
\operatorname{exact}_{64}(W_{\mathrm{upper}})+\Lambda_I
\right),
\qquad
\Lambda_I=2\left\lfloor\frac{n_v}{2}\right\rfloor,
\]

where `up_64` denotes least upward binary64 rounding. The compatibility column `W` aliases `W_upper`; `W_model` remains the ordinary rounded Python sum used for model diagnostics.

## Archived cohort and artifact levels

The primary archive contains exactly **40 instances**:

- effective qubit sizes `N=10` and `N=12`, corresponding to `n_v=11` and `n_v=13` physical vertices;
- seeds `0` through `19` once at each size;
- connected weighted Erdős--Rényi graphs with `p=0.5` and a pinned endpoint optimum that `verify_artifacts.py` checks exactly for nondegeneracy.

The aggregate CSV and ordinary sparse-eigensolver records are **Level 2 conditional numerical artifacts**. Ritz residuals are recorded, but eigenvalue indexing and outward rounding are not independently verified. The selected `N=10`, seed-0 dense-grid comparison is also a Level 2 consistency check; dense floating-point sampling does not make it Level 3.

The optional `results/level3/manifest.json` bundle covers exactly four representative exact-dyadic anchors: two for `N=10`, seed 0, and two for `N=12`, seed 0. A passing anchor is **Level 3 only for the exact-dyadic Hamiltonian and parameter recorded in its specification**. It does not upgrade either continuous sweep or the other aggregate rows. When the manifest is present, `verify_artifacts.py` checks every exact hash and invokes `verify_level3_anchors.py` in read-only mode for the exact rational certificate obligations.

## Repository structure

- `main.py`: generates the 40 result-schema-4 rows, graph-schema-2 records, `results/graph_instances.jsonl`, `results/section7_run_manifest.json`, and the selected Level 2 conditional log.
- `section7_results.csv`: result-schema-4 aggregate metrics for the 40 retained instances, including exact rational and upward-rounding metadata.
- `summarize_section7_results.py`: strictly validates the cohort and exact rounding chain, then writes deterministic JSON and TeX summaries with weight, spectral-diameter, solve-count, and global conditional-bound statistics.
- `plot_new_results.py`: generates the aggregate solve-count and endpoint-coverage figures. It does not generate a complexity-scaling plot.
- `generate_updated_comparison.py`: generates the two `N=10`, seed-0 Weyl-envelope figures from the archived graph record.
- `scratch/plot_grid_vs_gap.py`: generates the `N=10` and `N=12` seed-0 solve-density figures from archived graph records.
- `test_main.py`: exercises the analytic gap-width bound and continuation edge cases, including upward rounding, floor probes, invalid parameters, and zero-diameter paths.
- `verify_artifacts.py`: cross-checks the CSV, run manifest, graph metadata and hashes, summaries, selected Level 2 interval algebra, required figures, and any optional Level 3 manifest; it also exactly enumerates every pinned endpoint configuration to prove nondegeneracy for all 40 graph-schema-2 records.
- `generate_level3_anchors.py` and `verify_level3_anchors.py`: generate untrusted witnesses and independently verify the four selected exact-dyadic Level 3 anchor certificates; they do not certify a continuous sweep.
- `maxcut_gap_benchmark.py`: a separate exploratory benchmark and shared Hamiltonian library; it does not generate the Section 7 archive.

## Reference environment

The recorded reference run used Python 3.12.10. Install the pinned Python dependencies from the repository root:

```bash
python --version
python -m pip install -r requirements.txt
```

A TeX distribution with `latexmk` is required only for the manuscript build and is not installed by `requirements.txt`.

Sparse ARPACK solves can differ slightly across SciPy, BLAS, CPU, and thread configurations. Graph and artifact identities are checked with exact SHA-256 values; endpoint nondegeneracy is checked with exact dyadic arithmetic and no floating tolerance; Level 2 floating-point algebra is checked with explicit numerical tolerances. Wall-clock timings are platform-, BLAS-, load-, process-count-, and scheduler-dependent and should not be expected to reproduce exactly.

## Verify the archived artifacts

Run the focused unit suite first:

```bash
python -m unittest -v test_main.py
```

The artifact checker is read-only:

```bash
python verify_artifacts.py
```

It fails on an incomplete cohort, a result schema other than version 4, a graph schema other than version 2, invalid exact rational/upward-rounding metadata, stale summaries, graph/hash mismatches, inconsistent selected-log algebra, or missing manuscript figures. For every graph-schema-2 record, it interprets `edge_weight_hex` as exact dyadic fractions, enumerates all `2^N` pinned representatives, and requires a unique minimum with a strictly positive exact gap to the second ordered uncut cost. No floating tolerance is used for this endpoint-nondegeneracy check. If no Level 3 manifest is present, it reports that the archive remains Level 2. The selected exact verifier can also be run directly without rewriting its result JSON:

```bash
python verify_level3_anchors.py --no-write-results
```

To check only that the deterministic summaries match the CSV without rewriting them:

```bash
python summarize_section7_results.py --check
```

## Regenerate summaries and figures from the archived CSV

These commands do not rerun the 40-instance ensemble:

```bash
python summarize_section7_results.py
python plot_new_results.py
python generate_updated_comparison.py
python scratch/plot_grid_vs_gap.py
python verify_artifacts.py
```

They write:

- `results/section7_summary.json`;
- `results/section7_summary.tex`;
- `results/efficiency_comparison.png` and `results/certificate_coverage.png`;
- `results/comparison_updated_N10_no_anchors.png` and `results/comparison_updated_N10_anchors.png`;
- `results/grid_density_vs_gap.png` and `results/grid_density_vs_gap_N12.png`.

The plotting scripts resolve paths relative to the project root and use a headless Matplotlib backend. Instance-specific figures load and hash-check archived graph records rather than resampling graphs.

## Full numerical reproduction

From the repository root, run the commands in this order:

```bash
python main.py
python summarize_section7_results.py
python plot_new_results.py
python generate_updated_comparison.py
python scratch/plot_grid_vs_gap.py
python generate_level3_anchors.py
python verify_artifacts.py
```

`main.py` executes 20 seeds for each of `N=10` and `N=12`. The full run performs many sparse eigensolves and can take several hours. It atomically rewrites `section7_results.csv`, the graph archive, and the run manifest only after all 40 rows are available.

## Build the manuscript

After summary and figure generation, build the manuscript from the repository root:

```bash
latexmk -pdf main.tex
```

To remove LaTeX auxiliary files without touching numerical artifacts:

```bash
latexmk -c main.tex
```

## Exploratory benchmark

The standalone exploratory driver has a separate output contract:

```bash
python maxcut_gap_benchmark.py -h
```

Its outputs are not checked as part of the archived Section 7 artifact chain.
