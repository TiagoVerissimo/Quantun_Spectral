#!/usr/bin/env python3
"""Generate untrusted witnesses for four representative Level-3 anchors.

The script consumes the authoritative schema-2 seed-0 graph records written
by the full benchmark, uses SciPy only to generate candidate vectors, writes
binary64 ``.npy`` witnesses plus JSON specs/manifest under ``results/level3/``,
and invokes the independent exact verifier by default. It never creates or
overwrites primary graph records.

Passing results are Level-3 anchor certificates for the exact-dyadic
Hamiltonians reconstructed from the archived binary64 values. The existing
continuous sweeps remain Level 2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import scipy
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

import verify_level3_anchors as exact_verifier


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
LEVEL3_DIR = RESULTS_DIR / "level3"
DEFAULT_MANIFEST = LEVEL3_DIR / "manifest.json"

SOLVER_TOLERANCE = 1.0e-13  # Untrusted witness-generation setting only.
SOLVER_MAXITER = 50_000
SOLVER_NCV = 80


# ----------------------------- generic helpers ------------------------------


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")
    temporary.replace(path)


def _project_relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")


# ----------------------- authoritative graph records ------------------------


def _pinned_cost_vector_float(N: int, edges: Sequence[tuple[int, int, float]]) -> np.ndarray:
    states = np.arange(1 << N, dtype=np.int64)
    costs = np.zeros(1 << N, dtype=np.float64)
    for u, v, weight in edges:
        if u == 0:
            costs += weight * (((states >> (v - 1)) & 1) == 0)
        else:
            costs += weight * (
                ((states >> (u - 1)) & 1) == ((states >> (v - 1)) & 1)
            )
    return costs


def _require_primary_graph_record(
    N: int, seed: int
) -> tuple[Path, exact_verifier.ExactGraph]:
    path = RESULTS_DIR / f"graph_N{N}_seed{seed}.json"
    instruction = (
        f"Run `python main.py` to regenerate the authoritative schema-2 benchmark "
        f"record {path.name}, then rerun this generator."
    )
    if not path.is_file():
        raise RuntimeError(f"Missing primary graph record: {path}. {instruction}")
    try:
        graph = exact_verifier.load_exact_graph_record(path, expected_N=N, expected_seed=seed)
    except exact_verifier.VerificationError as exc:
        raise RuntimeError(
            f"Primary graph record is not a valid authoritative schema-2 record: {path}: "
            f"{exc}. {instruction}"
        ) from exc
    print(
        f"Using authoritative graph record: {_project_relative(path)} "
        f"(record {graph.graph_record_sha256}, payload {graph.edge_payload_sha256})"
    )
    return path, graph


def _graph_float_edges(graph: exact_verifier.ExactGraph) -> list[tuple[int, int, float]]:
    return [(u, v, float(weight)) for u, v, weight in graph.edges]


# ------------------------- sparse witness matrices --------------------------


def _build_z_matrix(
    N: int, s: float, edges: Sequence[tuple[int, int, float]]
) -> Any:
    dimension = 1 << N
    states = np.arange(dimension, dtype=np.int64)
    costs = _pinned_cost_vector_float(N, edges)
    diagonal = (1.0 - s) * ((N + 1) / 2.0) + s * costs

    row_blocks: list[np.ndarray] = []
    column_blocks: list[np.ndarray] = []
    data_blocks: list[np.ndarray] = []
    coefficient = -0.5 * (1.0 - s)
    for bit in range(N):
        row_blocks.append(states ^ (1 << bit))
        column_blocks.append(states)
        data_blocks.append(np.full(dimension, coefficient, dtype=np.float64))
    row_blocks.append(states ^ (dimension - 1))
    column_blocks.append(states)
    data_blocks.append(np.full(dimension, coefficient, dtype=np.float64))

    off_diagonal = sp.coo_matrix(
        (
            np.concatenate(data_blocks),
            (np.concatenate(row_blocks), np.concatenate(column_blocks)),
        ),
        shape=(dimension, dimension),
    ).tocsr()
    return (off_diagonal + sp.diags(diagonal, format="csr")).tocsr()


def _build_x_matrices(
    N: int, s: float, edges: Sequence[tuple[int, int, float]]
) -> tuple[Any, Any]:
    dimension = 1 << N
    states = np.arange(dimension, dtype=np.int64)
    driver_energy = np.fromiter(
        (state.bit_count() + (state.bit_count() & 1) for state in range(dimension)),
        dtype=np.float64,
        count=dimension,
    )
    total_weight = sum(weight for _, _, weight in edges)
    diagonal = (1.0 - s) * driver_energy + s * total_weight / 2.0

    row_blocks: list[np.ndarray] = []
    column_blocks: list[np.ndarray] = []
    data_blocks: list[np.ndarray] = []
    for u, v, weight in edges:
        if u == 0:
            mask = 1 << (v - 1)
        else:
            mask = (1 << (u - 1)) ^ (1 << (v - 1))
        row_blocks.append(states ^ mask)
        column_blocks.append(states)
        data_blocks.append(np.full(dimension, s * weight / 2.0, dtype=np.float64))

    positive_off_diagonal = sp.coo_matrix(
        (
            np.concatenate(data_blocks),
            (np.concatenate(row_blocks), np.concatenate(column_blocks)),
        ),
        shape=(dimension, dimension),
    ).tocsr()
    diagonal_matrix = sp.diags(diagonal, format="csr")
    actual = (diagonal_matrix + positive_off_diagonal).tocsr()
    comparison = (diagonal_matrix - positive_off_diagonal).tocsr()
    return actual, comparison


def _deterministic_start(length: int, salt: int, positive: bool) -> np.ndarray:
    indices = np.arange(length, dtype=np.int64)
    values = ((indices * (104729 + 2 * salt) + 8191 + salt) % 130363) + 1
    vector = values.astype(np.float64)
    if not positive:
        signs = np.where(((indices * 17 + salt) & 1) == 0, 1.0, -1.0)
        vector *= signs
    vector /= np.linalg.norm(vector)
    return vector


def _smallest_eigenpair(matrix: Any, v0: np.ndarray) -> tuple[float, np.ndarray]:
    ncv = min(int(matrix.shape[0]) - 1, SOLVER_NCV)
    values, vectors = eigsh(
        matrix,
        k=1,
        which="SA",
        v0=v0,
        tol=SOLVER_TOLERANCE,  # type: ignore[arg-type]
        maxiter=SOLVER_MAXITER,
        ncv=ncv,
    )
    value = float(values[0])
    vector = np.asarray(vectors[:, 0], dtype=np.float64)
    pivot = int(np.argmax(np.abs(vector)))
    if vector[pivot] < 0.0:
        vector = -vector
    return value, vector


def _principal_submatrix(matrix: Any, removed_coordinate: int) -> Any:
    keep = np.arange(int(matrix.shape[0])) != removed_coordinate
    return matrix[keep][:, keep].tocsr()


def _save_npy(path: Path, vector: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vector = np.asarray(vector, dtype="<f8")
    with path.open("wb") as handle:
        np.save(handle, vector, allow_pickle=False)


# ---------------------------- anchor generation -----------------------------


def _generate_anchor(
    anchor_id: str,
    configuration: dict[str, Any],
    graph_path: Path,
    graph: exact_verifier.ExactGraph,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.perf_counter()
    N = int(configuration["N"])
    seed = int(configuration["seed"])
    basis = str(configuration["basis"])
    s = float(configuration["s_decimal"])
    dimension = 1 << N

    edges = _graph_float_edges(graph)
    matrix_started = time.perf_counter()
    if basis == "Z":
        actual = _build_z_matrix(N, s, edges)
        comparison = actual
    elif basis == "X":
        actual, comparison = _build_x_matrices(N, s, edges)
    else:
        raise RuntimeError(f"unsupported basis: {basis}")
    matrix_seconds = time.perf_counter() - matrix_started

    ground_started = time.perf_counter()
    ground_value, ground = _smallest_eigenpair(
        actual,
        _deterministic_start(dimension, salt=N + (0 if basis == "Z" else 101), positive=False),
    )
    ground_seconds = time.perf_counter() - ground_started

    removed_coordinate = int(np.argmax(np.abs(ground))) if basis == "Z" else 0
    principal_matrix = _principal_submatrix(comparison, removed_coordinate)
    principal_started = time.perf_counter()
    principal_value, principal = _smallest_eigenpair(
        principal_matrix,
        _deterministic_start(dimension - 1, salt=N + 211, positive=True),
    )
    # Any positive vector is a valid candidate. Taking componentwise absolute
    # values removes the arbitrary numerical eigenvector sign without entering
    # the trusted proof; the exact verifier checks strict positivity and every
    # resulting inequality from scratch.
    principal = np.abs(principal)
    if not np.all(np.isfinite(principal)) or not np.all(principal > 0.0):
        raise RuntimeError(f"failed to generate a strictly positive principal witness: {anchor_id}")
    principal_seconds = time.perf_counter() - principal_started

    ground_path = LEVEL3_DIR / f"{anchor_id}.ground.npy"
    principal_path = LEVEL3_DIR / f"{anchor_id}.principal.npy"
    _save_npy(ground_path, ground)
    _save_npy(principal_path, principal)

    ground_residual = float(np.linalg.norm(actual @ ground - ground_value * ground))
    principal_residual = float(
        np.linalg.norm(principal_matrix @ principal - principal_value * principal)
    )

    spec_path = LEVEL3_DIR / f"{anchor_id}.spec.json"
    result_path = LEVEL3_DIR / f"{anchor_id}.result.json"
    spec = {
        "schema_version": 1,
        "record_type": "level3_anchor_spec",
        "anchor_id": anchor_id,
        "certificate_intent": "Level-3 indexed anchor certificate",
        "scope_statement": (
            "This spec can certify one exact-dyadic anchor. Continuous sweeps remain Level 2."
        ),
        "N_effective_qubits": N,
        "n_vertices": N + 1,
        "dimension": dimension,
        "seed": seed,
        "basis": basis,
        "s_hex": s.hex(),
        "s_decimal_input": configuration["s_decimal"],
        "matrix_semantics": (
            "exact reduced Hamiltonian reconstructed from authoritative schema-2 "
            "binary64 edge weights and exact binary64 dyadic s"
        ),
        "graph_schema_version": 2,
        "graph_record_type": "generated_weighted_graph_instance",
        "graph_path": _project_relative(graph_path),
        "graph_file_sha256": _sha256_file(graph_path),
        "graph_record_sha256": graph.graph_record_sha256,
        "edge_payload_sha256": graph.edge_payload_sha256,
        "removed_coordinate": removed_coordinate,
        "ground_witness": {
            "path": _project_relative(ground_path),
            "sha256": _sha256_file(ground_path),
            "dtype": "binary64",
            "shape": [dimension],
            "role": "untrusted Rayleigh-quotient candidate for the E0 upper bound",
        },
        "principal_witness": {
            "path": _project_relative(principal_path),
            "sha256": _sha256_file(principal_path),
            "dtype": "binary64",
            "shape": [dimension - 1],
            "role": "untrusted strictly-positive candidate for the comparison principal minor",
        },
        "required_exact_proof": {
            "E0_upper": "U0=q^T H q/(q^T q), evaluated exactly",
            "E1_lower": (
                "ell1=min_i (M^(p)x)_i/x_i; scaled diagonal dominance and the "
                "comparison inequality give lambda_min(H^(p))>=ell1; Cauchy "
                "interlacing gives E1(H)>=ell1"
            ),
            "gap": "ell1-U0 must be strictly positive as an exact rational",
            "acceptance_tolerances": "none",
        },
        "untrusted_scipy_diagnostics": {
            "solver": "scipy.sparse.linalg.eigsh",
            "which": "SA",
            "k": 1,
            "tol": SOLVER_TOLERANCE,
            "maxiter": SOLVER_MAXITER,
            "ncv": min(dimension - 1, SOLVER_NCV),
            "ground_candidate_value": ground_value,
            "ground_candidate_residual": ground_residual,
            "principal_candidate_value": principal_value,
            "principal_candidate_residual": principal_residual,
            "principal_candidate_min_component": float(np.min(principal)),
        },
        "artifact_paths": {
            "spec": _project_relative(spec_path),
            "result": _project_relative(result_path),
        },
        "generation_timings_seconds": {
            "matrix_assembly": matrix_seconds,
            "ground_witness": ground_seconds,
            "principal_witness": principal_seconds,
            "total": time.perf_counter() - started,
        },
    }
    _write_json(spec_path, spec)

    manifest_entry = {
        "anchor_id": anchor_id,
        "spec_path": _project_relative(spec_path),
        "spec_sha256": _sha256_file(spec_path),
        "result_path": _project_relative(result_path),
    }
    return manifest_entry, spec


def generate_all() -> Path:
    graph_inputs = {N: _require_primary_graph_record(N, seed=0) for N in (10, 12)}
    LEVEL3_DIR.mkdir(parents=True, exist_ok=True)

    manifest_entries: list[dict[str, Any]] = []
    generation_summary: list[dict[str, Any]] = []
    for anchor_id, configuration in exact_verifier.EXPECTED_ANCHORS.items():
        print(
            f"Generating {anchor_id}: N={configuration['N']}, "
            f"s={configuration['s_decimal']}, basis={configuration['basis']}"
        )
        graph_path, graph = graph_inputs[int(configuration["N"])]
        entry, spec = _generate_anchor(
            anchor_id,
            configuration,
            graph_path,
            graph,
        )
        manifest_entries.append(entry)
        generation_summary.append(
            {
                "anchor_id": anchor_id,
                "removed_coordinate": spec["removed_coordinate"],
                "generation_timings_seconds": spec["generation_timings_seconds"],
                "ground_witness_sha256": spec["ground_witness"]["sha256"],
                "principal_witness_sha256": spec["principal_witness"]["sha256"],
            }
        )

    manifest = {
        "schema_version": 1,
        "record_type": "level3_anchor_manifest",
        "certificate_scope": (
            "Exactly four representative Level-3 anchors for exact-dyadic Hamiltonians"
        ),
        "scope_statement": (
            "Passing anchors are Level 3 for exact-dyadic Hamiltonians. "
            "Continuous sweeps remain Level 2."
        ),
        "matrix_semantics": (
            "Authoritative schema-2 edge_list binary64 values, checked against parallel "
            "edge_weight_hex, and anchor s values are interpreted as exact dyadic rationals."
        ),
        "anchors": manifest_entries,
        "graph_records": [
            {
                "path": _project_relative(graph_inputs[N][0]),
                "graph_file_sha256": _sha256_file(graph_inputs[N][0]),
                "graph_record_sha256": graph_inputs[N][1].graph_record_sha256,
                "edge_payload_sha256": graph_inputs[N][1].edge_payload_sha256,
            }
            for N in (10, 12)
        ],
        "verification_summary_path": "results/level3/verification_manifest.json",
        "source_hashes": {
            "generate_level3_anchors.py": _sha256_file(Path(__file__).resolve()),
            "verify_level3_anchors.py": _sha256_file(
                PROJECT_ROOT / "verify_level3_anchors.py"
            ),
        },
        "software_for_untrusted_witness_generation": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "generation_summary": generation_summary,
    }
    _write_json(DEFAULT_MANIFEST, manifest)
    print(f"Wrote manifest: {_project_relative(DEFAULT_MANIFEST)}")
    return DEFAULT_MANIFEST


# ---------------------------------- CLI -------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic SciPy candidate witnesses and exact-verification specs "
            "for four representative N=10/N=12 Level-3 anchors. Exact verification runs "
            "by default after generation."
        )
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="do not generate anything; run the independent exact verifier on the manifest",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="generate witnesses/specs/manifest but do not run exact verification",
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"manifest used by --verify-only (default: {DEFAULT_MANIFEST})",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.verify_only:
        ok = exact_verifier.verify_manifest(args.manifest, write_results=True, verbose=True)
        return 0 if ok else 1
    if args.no_verify and args.verify_only:
        print("--no-verify cannot be combined with --verify-only", file=sys.stderr)
        return 2

    try:
        manifest_path = generate_all()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if args.no_verify:
        print(
            "Witness generation completed without verification. No Level-3 certificate "
            "is accepted until verify_level3_anchors.py passes."
        )
        return 0

    print("Running independent exact verifier...")
    ok = exact_verifier.verify_manifest(manifest_path, write_results=True, verbose=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
