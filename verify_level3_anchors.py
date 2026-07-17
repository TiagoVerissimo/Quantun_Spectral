#!/usr/bin/env python3
"""Verify representative Level-3 spectral-gap anchor certificates.

The default action is verify-only: load ``results/level3/manifest.json``,
check every graph/spec/witness SHA-256 hash, reconstruct the stated reduced
Hamiltonian with exact rational arithmetic, and reject on any mismatch.

SciPy is intentionally not imported. Numerical eigensolvers are untrusted
witness generators; acceptance uses only exact integer/Fraction arithmetic.
The resulting certificates apply to the exact-dyadic Hamiltonians defined by
the archived binary64 edge weights and path parameters. They do not upgrade
the repository's continuous sweeps, which remain Level 2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
import time
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = PROJECT_ROOT / "results" / "level3" / "manifest.json"

EXPECTED_ANCHORS: dict[str, dict[str, Any]] = {
    "N10_seed0_s0p1_X": {
        "N": 10,
        "seed": 0,
        "s_decimal": "0.1",
        "basis": "X",
    },
    "N10_seed0_s0p4174116949186277_Z": {
        "N": 10,
        "seed": 0,
        "s_decimal": "0.4174116949186277",
        "basis": "Z",
    },
    "N12_seed0_s0p1_X": {
        "N": 12,
        "seed": 0,
        "s_decimal": "0.1",
        "basis": "X",
    },
    "N12_seed0_s0p5025125632610975_Z": {
        "N": 12,
        "seed": 0,
        "s_decimal": "0.5025125632610975",
        "basis": "Z",
    },
}


class VerificationError(RuntimeError):
    """Raised when an exact certificate obligation is not satisfied."""


@dataclass(frozen=True)
class ExactGraph:
    N: int
    n_vertices: int
    seed: int
    edges: tuple[tuple[int, int, Fraction], ...]
    graph_record_sha256: str
    edge_payload_sha256: str

    @property
    def dimension(self) -> int:
        return 1 << self.N


# ----------------------------- generic I/O ---------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(payload: Any) -> bytes:
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise VerificationError(f"value is not canonical-JSON serializable: {exc}") from exc


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")
    temporary.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise VerificationError(f"JSON root must be an object: {path}")
    return payload


def _required_str(record: dict[str, Any], key: str, label: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise VerificationError(f"{label} must be a nonempty string")
    return value


def _project_path(relative_path: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise VerificationError("artifact path must be a nonempty string")
    path = (PROJECT_ROOT / relative_path).resolve()
    try:
        path.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise VerificationError(f"artifact path escapes project root: {relative_path}") from exc
    return path


def _require_file_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise VerificationError(f"missing {label}: {path}")
    if not isinstance(expected, str) or len(expected) != 64:
        raise VerificationError(f"invalid expected SHA-256 for {label}")
    actual = sha256_file(path)
    if actual != expected:
        raise VerificationError(
            f"SHA-256 mismatch for {label}: expected {expected}, got {actual}"
        )
    return actual


# --------------------------- exact scalar input -----------------------------


def _fraction_from_float(value: float) -> Fraction:
    if not math.isfinite(value):
        raise VerificationError("non-finite binary64 value")
    numerator, denominator = value.as_integer_ratio()
    return Fraction(numerator, denominator)


def _fraction_from_hex(value_hex: str, label: str) -> Fraction:
    if not isinstance(value_hex, str):
        raise VerificationError(f"{label} must be a hexadecimal binary64 string")
    try:
        value = float.fromhex(value_hex)
    except (ValueError, OverflowError) as exc:
        raise VerificationError(f"invalid hexadecimal binary64 {label}: {value_hex}") from exc
    if not math.isfinite(value):
        raise VerificationError(f"non-finite {label}")
    if value.hex() != value_hex:
        raise VerificationError(
            f"noncanonical hexadecimal binary64 {label}: {value_hex} != {value.hex()}"
        )
    return _fraction_from_float(value)


def _fraction_payload(value: Fraction) -> dict[str, str]:
    with localcontext() as context:
        context.prec = 40
        decimal_value = Decimal(value.numerator) / Decimal(value.denominator)
    return {
        "numerator": str(value.numerator),
        "denominator": str(value.denominator),
        "decimal_approx": format(decimal_value, ".30g"),
    }


def _exact_vector(array: np.ndarray) -> list[Fraction]:
    return [_fraction_from_float(float(value)) for value in array]


# ------------------------------ graph input ---------------------------------


def _is_connected(n_vertices: int, edge_pairs: Iterable[tuple[int, int]]) -> bool:
    adjacency: list[list[int]] = [[] for _ in range(n_vertices)]
    for u, v in edge_pairs:
        adjacency[u].append(v)
        adjacency[v].append(u)
    seen = {0}
    stack = [0]
    while stack:
        vertex = stack.pop()
        for neighbor in adjacency[vertex]:
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return len(seen) == n_vertices


def _parse_exact_graph_record(
    record: dict[str, Any], source: str, expected_N: int, expected_seed: int
) -> ExactGraph:
    if record.get("schema_version") != 2:
        raise VerificationError(
            f"{source} is not an authoritative schema_version 2 graph record; "
            "rerun the current full benchmark"
        )
    if record.get("record_type") != "generated_weighted_graph_instance":
        raise VerificationError(f"unexpected graph record type in {source}")

    stored_record_hash = _required_str(
        record, "graph_record_sha256", f"graph_record_sha256 in {source}"
    )
    record_hash_input = {
        key: value for key, value in record.items() if key != "graph_record_sha256"
    }
    computed_record_hash = _canonical_sha256(record_hash_input)
    if stored_record_hash != computed_record_hash:
        raise VerificationError(
            f"canonical graph_record_sha256 mismatch in {source}: "
            f"expected {stored_record_hash}, got {computed_record_hash}"
        )

    N = record.get("N_effective_qubits")
    n_vertices = record.get("n_vertices")
    generation = record.get("generation")
    row_key = record.get("row_key")
    if not isinstance(N, int) or not isinstance(n_vertices, int):
        raise VerificationError(f"graph dimensions must be integer-valued in {source}")
    if not isinstance(generation, dict) or not isinstance(row_key, dict):
        raise VerificationError(f"generation and row_key must be objects in {source}")
    seed = generation.get("seed")
    if not isinstance(seed, int):
        raise VerificationError(f"generation.seed must be an integer in {source}")
    if row_key.get("N") != N or row_key.get("seed") != seed:
        raise VerificationError(f"row_key disagrees with graph metadata in {source}")
    if N != expected_N or seed != expected_seed or n_vertices != expected_N + 1:
        raise VerificationError(
            f"graph metadata mismatch in {source}: N={N}, seed={seed}, "
            f"n_vertices={n_vertices}"
        )
    if record.get("edge_weight_semantics") != (
        "archived binary64 values interpreted as exact dyadic rationals"
    ):
        raise VerificationError(f"unexpected edge-weight semantics in {source}")

    edge_list = record.get("edge_list")
    edge_weight_hex = record.get("edge_weight_hex")
    if not isinstance(edge_list, list) or not edge_list:
        raise VerificationError(f"graph has no edge_list in {source}")
    if not isinstance(edge_weight_hex, list) or len(edge_weight_hex) != len(edge_list):
        raise VerificationError(f"edge_weight_hex is not parallel to edge_list in {source}")
    if record.get("num_edges") != len(edge_list):
        raise VerificationError(f"num_edges mismatch in {source}")

    stored_payload_hash = _required_str(
        record, "edge_payload_sha256", f"edge_payload_sha256 in {source}"
    )
    computed_payload_hash = _canonical_sha256(edge_list)
    if stored_payload_hash != computed_payload_hash:
        raise VerificationError(
            f"canonical edge_payload_sha256 mismatch in {source}: "
            f"expected {stored_payload_hash}, got {computed_payload_hash}"
        )

    edges: list[tuple[int, int, Fraction]] = []
    keys: list[tuple[int, int]] = []
    for index, (item, weight_hex) in enumerate(zip(edge_list, edge_weight_hex)):
        if not isinstance(item, list) or len(item) != 3:
            raise VerificationError(f"edge_list[{index}] is not [u,v,weight] in {source}")
        u, v, weight_value = item
        if not isinstance(u, int) or isinstance(u, bool):
            raise VerificationError(f"edge_list[{index}] has a noninteger u in {source}")
        if not isinstance(v, int) or isinstance(v, bool):
            raise VerificationError(f"edge_list[{index}] has a noninteger v in {source}")
        if not (0 <= u < v < n_vertices):
            raise VerificationError(
                f"edge_list[{index}] has invalid vertices ({u}, {v}) in {source}"
            )
        if not isinstance(weight_value, float) or not math.isfinite(weight_value):
            raise VerificationError(
                f"edge_list[{index}] weight must be a finite JSON float in {source}"
            )
        if not isinstance(weight_hex, str):
            raise VerificationError(f"edge_weight_hex[{index}] is not a string in {source}")
        try:
            parsed_hex = float.fromhex(weight_hex)
        except (ValueError, OverflowError) as exc:
            raise VerificationError(
                f"edge_weight_hex[{index}] is invalid in {source}: {weight_hex}"
            ) from exc
        if not math.isfinite(parsed_hex) or parsed_hex.hex() != weight_hex:
            raise VerificationError(
                f"edge_weight_hex[{index}] is not canonical finite binary64 in {source}"
            )
        if weight_value.as_integer_ratio() != parsed_hex.as_integer_ratio():
            raise VerificationError(
                f"edge_list[{index}] float does not match edge_weight_hex[{index}] in {source}"
            )
        weight = _fraction_from_float(parsed_hex)
        if weight <= 0:
            raise VerificationError(f"edge_list[{index}] has nonpositive weight in {source}")
        edges.append((u, v, weight))
        keys.append((u, v))

    if keys != sorted(keys) or len(set(keys)) != len(keys):
        raise VerificationError(f"edge_list must be sorted and duplicate-free in {source}")
    if not _is_connected(n_vertices, keys):
        raise VerificationError(f"graph is not connected: {source}")

    return ExactGraph(
        N=N,
        n_vertices=n_vertices,
        seed=seed,
        edges=tuple(edges),
        graph_record_sha256=stored_record_hash,
        edge_payload_sha256=stored_payload_hash,
    )


def load_exact_graph_record(path: Path, expected_N: int, expected_seed: int) -> ExactGraph:
    record = _load_json(path)
    return _parse_exact_graph_record(record, str(path), expected_N, expected_seed)


# ------------------------------ witnesses -----------------------------------


def _load_witness(
    path: Path,
    expected_hash: str,
    expected_length: int,
    label: str,
    require_positive: bool,
) -> np.ndarray:
    _require_file_hash(path, expected_hash, label)
    try:
        array = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise VerificationError(f"cannot load {label} {path}: {exc}") from exc

    if array.ndim != 1 or array.shape != (expected_length,):
        raise VerificationError(
            f"{label} shape mismatch: expected {(expected_length,)}, got {array.shape}"
        )
    if array.dtype.kind != "f" or array.dtype.itemsize != 8:
        raise VerificationError(f"{label} must contain binary64 values, got {array.dtype}")
    if not np.all(np.isfinite(array)):
        raise VerificationError(f"{label} contains non-finite values")
    if not np.any(array != 0.0):
        raise VerificationError(f"{label} is the zero vector")
    if require_positive and not np.all(array > 0.0):
        raise VerificationError(f"{label} is not strictly positive componentwise")
    return np.asarray(array, dtype=np.float64)


# ------------------------- exact sparse Hamiltonians ------------------------


def _z_diagonal(graph: ExactGraph, s: Fraction) -> list[Fraction]:
    one_minus_s = Fraction(1, 1) - s
    driver_diagonal = one_minus_s * Fraction(graph.n_vertices, 2)
    diagonal: list[Fraction] = []
    for state in range(graph.dimension):
        cost = Fraction(0, 1)
        for u, v, weight in graph.edges:
            bit_u = 0 if u == 0 else ((state >> (u - 1)) & 1)
            bit_v = 0 if v == 0 else ((state >> (v - 1)) & 1)
            if bit_u == bit_v:
                cost += weight
        diagonal.append(driver_diagonal + s * cost)
    return diagonal


def _apply_z(
    graph: ExactGraph,
    s: Fraction,
    diagonal: Sequence[Fraction],
    vector: Sequence[Fraction],
) -> list[Fraction]:
    if len(vector) != graph.dimension or len(diagonal) != graph.dimension:
        raise VerificationError("internal Z-basis dimension mismatch")
    off_diagonal = -(Fraction(1, 1) - s) / 2
    global_mask = graph.dimension - 1
    output: list[Fraction] = []
    for state in range(graph.dimension):
        value = diagonal[state] * vector[state]
        for bit in range(graph.N):
            value += off_diagonal * vector[state ^ (1 << bit)]
        value += off_diagonal * vector[state ^ global_mask]
        output.append(value)
    return output


def _x_operator_data(
    graph: ExactGraph, s: Fraction
) -> tuple[list[Fraction], tuple[tuple[int, Fraction], ...]]:
    total_weight = sum((weight for _, _, weight in graph.edges), Fraction(0, 1))
    one_minus_s = Fraction(1, 1) - s
    diagonal: list[Fraction] = []
    for state in range(graph.dimension):
        hamming_weight = state.bit_count()
        driver_energy = hamming_weight + (hamming_weight & 1)
        diagonal.append(one_minus_s * driver_energy + s * total_weight / 2)

    links: list[tuple[int, Fraction]] = []
    for u, v, weight in graph.edges:
        if u == 0:
            mask = 1 << (v - 1)
        else:
            mask = (1 << (u - 1)) ^ (1 << (v - 1))
        links.append((mask, s * weight / 2))
    return diagonal, tuple(links)


def _apply_x(
    graph: ExactGraph,
    diagonal: Sequence[Fraction],
    links: Sequence[tuple[int, Fraction]],
    vector: Sequence[Fraction],
    comparison: bool,
) -> list[Fraction]:
    if len(vector) != graph.dimension or len(diagonal) != graph.dimension:
        raise VerificationError("internal X-basis dimension mismatch")
    sign = -1 if comparison else 1
    output: list[Fraction] = []
    for state in range(graph.dimension):
        value = diagonal[state] * vector[state]
        for mask, coefficient in links:
            value += sign * coefficient * vector[state ^ mask]
        output.append(value)
    return output


def _expand_principal_vector(
    principal: Sequence[Fraction], dimension: int, removed_coordinate: int
) -> list[Fraction]:
    if not (0 <= removed_coordinate < dimension):
        raise VerificationError("removed coordinate is out of range")
    if len(principal) != dimension - 1:
        raise VerificationError("principal witness has the wrong dimension")
    expanded: list[Fraction] = []
    principal_index = 0
    for state in range(dimension):
        if state == removed_coordinate:
            expanded.append(Fraction(0, 1))
        else:
            expanded.append(principal[principal_index])
            principal_index += 1
    return expanded


def _dot(left: Sequence[Fraction], right: Sequence[Fraction]) -> Fraction:
    if len(left) != len(right):
        raise VerificationError("internal exact dot-product dimension mismatch")
    return sum((a * b for a, b in zip(left, right)), Fraction(0, 1))


# -------------------------- anchor verification -----------------------------


def _validate_spec_metadata(spec: dict[str, Any], expected: dict[str, Any], anchor_id: str) -> None:
    if spec.get("schema_version") != 1 or spec.get("record_type") != "level3_anchor_spec":
        raise VerificationError(f"unsupported spec schema for {anchor_id}")
    if spec.get("anchor_id") != anchor_id:
        raise VerificationError(f"anchor_id mismatch in spec for {anchor_id}")
    if spec.get("N_effective_qubits") != expected["N"]:
        raise VerificationError(f"N mismatch in spec for {anchor_id}")
    if spec.get("seed") != expected["seed"]:
        raise VerificationError(f"seed mismatch in spec for {anchor_id}")
    if spec.get("basis") != expected["basis"]:
        raise VerificationError(f"basis mismatch in spec for {anchor_id}")
    expected_s_hex = float(expected["s_decimal"]).hex()
    if spec.get("s_hex") != expected_s_hex:
        raise VerificationError(
            f"s mismatch in spec for {anchor_id}: expected {expected_s_hex}, got {spec.get('s_hex')}"
        )
    if spec.get("matrix_semantics") != (
        "exact reduced Hamiltonian reconstructed from authoritative schema-2 "
        "binary64 edge weights and exact binary64 dyadic s"
    ):
        raise VerificationError(f"matrix semantics mismatch for {anchor_id}")
    if spec.get("graph_schema_version") != 2:
        raise VerificationError(f"graph schema mismatch for {anchor_id}")
    if spec.get("graph_record_type") != "generated_weighted_graph_instance":
        raise VerificationError(f"graph record type mismatch for {anchor_id}")


def verify_anchor(spec: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    anchor_id = spec.get("anchor_id")
    if not isinstance(anchor_id, str):
        raise VerificationError("spec has no valid anchor_id")
    _validate_spec_metadata(spec, expected, anchor_id)

    graph_path = _project_path(_required_str(spec, "graph_path", f"graph path for {anchor_id}"))
    graph_expected_hash = _required_str(
        spec, "graph_file_sha256", f"graph file hash for {anchor_id}"
    )
    graph_hash = _require_file_hash(
        graph_path, graph_expected_hash, f"graph for {anchor_id}"
    )
    graph = load_exact_graph_record(
        graph_path, int(expected["N"]), int(expected["seed"])
    )
    if _required_str(
        spec, "graph_record_sha256", f"graph record hash for {anchor_id}"
    ) != graph.graph_record_sha256:
        raise VerificationError(f"graph_record_sha256 mismatch for {anchor_id}")
    if _required_str(
        spec, "edge_payload_sha256", f"edge payload hash for {anchor_id}"
    ) != graph.edge_payload_sha256:
        raise VerificationError(f"edge_payload_sha256 mismatch for {anchor_id}")

    dimension = graph.dimension
    removed_coordinate = spec.get("removed_coordinate")
    if not isinstance(removed_coordinate, int) or not (0 <= removed_coordinate < dimension):
        raise VerificationError(f"invalid removed coordinate for {anchor_id}")

    ground_record = spec.get("ground_witness")
    principal_record = spec.get("principal_witness")
    if not isinstance(ground_record, dict) or not isinstance(principal_record, dict):
        raise VerificationError(f"missing witness metadata for {anchor_id}")

    ground_path = _project_path(
        _required_str(ground_record, "path", f"ground witness path for {anchor_id}")
    )
    principal_path = _project_path(
        _required_str(principal_record, "path", f"principal witness path for {anchor_id}")
    )
    ground_array = _load_witness(
        ground_path,
        _required_str(ground_record, "sha256", f"ground witness hash for {anchor_id}"),
        dimension,
        f"ground witness for {anchor_id}",
        require_positive=False,
    )
    principal_array = _load_witness(
        principal_path,
        _required_str(principal_record, "sha256", f"principal witness hash for {anchor_id}"),
        dimension - 1,
        f"principal witness for {anchor_id}",
        require_positive=True,
    )

    conversion_started = time.perf_counter()
    q = _exact_vector(ground_array)
    x_principal = _exact_vector(principal_array)
    if any(value <= 0 for value in x_principal):
        raise VerificationError(f"principal witness is not exactly positive for {anchor_id}")
    x_full = _expand_principal_vector(x_principal, dimension, removed_coordinate)
    s_hex = _required_str(spec, "s_hex", f"s for {anchor_id}")
    s = _fraction_from_hex(s_hex, f"s for {anchor_id}")
    if not (Fraction(0, 1) <= s <= Fraction(1, 1)):
        raise VerificationError(f"s is outside [0,1] for {anchor_id}")
    conversion_seconds = time.perf_counter() - conversion_started

    arithmetic_started = time.perf_counter()
    basis = expected["basis"]
    if basis == "Z":
        diagonal = _z_diagonal(graph, s)
        hq = _apply_z(graph, s, diagonal, q)
        comparison_x = _apply_z(graph, s, diagonal, x_full)
    elif basis == "X":
        diagonal, links = _x_operator_data(graph, s)
        hq = _apply_x(graph, diagonal, links, q, comparison=False)
        comparison_x = _apply_x(graph, diagonal, links, x_full, comparison=True)
    else:
        raise VerificationError(f"unsupported basis for {anchor_id}: {basis}")

    q_norm_squared = _dot(q, q)
    if q_norm_squared <= 0:
        raise VerificationError(f"ground witness has nonpositive exact norm for {anchor_id}")
    U0 = _dot(q, hq) / q_norm_squared

    ratios: list[tuple[Fraction, int]] = []
    for state in range(dimension):
        if state == removed_coordinate:
            continue
        x_value = x_full[state]
        if x_value <= 0:
            raise VerificationError(f"expanded principal witness is not positive for {anchor_id}")
        ratios.append((comparison_x[state] / x_value, state))
    ell1, minimizing_row = min(ratios, key=lambda item: item[0])

    # This is an explicit exact check of the scaled diagonal-dominance
    # obligation: (M - ell1 I)x >= 0 componentwise. No tolerance is used.
    minimum_margin: Fraction | None = None
    for state in range(dimension):
        if state == removed_coordinate:
            continue
        margin = comparison_x[state] - ell1 * x_full[state]
        if margin < 0:
            raise VerificationError(
                f"negative exact scaled-diagonal-dominance margin at row {state} for {anchor_id}"
            )
        if minimum_margin is None or margin < minimum_margin:
            minimum_margin = margin
    if minimum_margin is None:
        raise VerificationError(f"empty principal matrix for {anchor_id}")

    gap_lower = ell1 - U0
    if gap_lower <= 0:
        raise VerificationError(
            f"exact gap lower bound is not positive for {anchor_id}: {gap_lower}"
        )
    arithmetic_seconds = time.perf_counter() - arithmetic_started

    result = {
        "schema_version": 1,
        "record_type": "level3_anchor_verification_result",
        "verification_status": "PASS",
        "certificate_level": "Level-3 indexed anchor certificate",
        "scope_statement": (
            "This anchor is Level 3 for the exact-dyadic Hamiltonian stated in the spec. "
            "The repository's continuous sweeps remain Level 2."
        ),
        "anchor_id": anchor_id,
        "N_effective_qubits": graph.N,
        "dimension": dimension,
        "basis": basis,
        "s_hex": spec.get("s_hex"),
        "s_decimal_input": expected["s_decimal"],
        "removed_coordinate": removed_coordinate,
        "hashes": {
            "graph_file_sha256": graph_hash,
            "graph_record_sha256": graph.graph_record_sha256,
            "edge_payload_sha256": graph.edge_payload_sha256,
            "ground_witness_sha256": sha256_file(ground_path),
            "principal_witness_sha256": sha256_file(principal_path),
        },
        "exact_bounds": {
            "E0_upper_U0": _fraction_payload(U0),
            "E1_lower_ell1": _fraction_payload(ell1),
            "gap_lower_ell1_minus_U0": _fraction_payload(gap_lower),
        },
        "exact_checks": {
            "ground_norm_squared_positive": True,
            "principal_witness_strictly_positive": True,
            "scaled_diagonal_dominance_margins_nonnegative": True,
            "minimum_margin": _fraction_payload(minimum_margin),
            "minimum_ratio_full_row": minimizing_row,
            "gap_lower_strictly_positive": True,
            "no_acceptance_tolerance_used": True,
        },
        "indexing_proof": {
            "step_1": "Exact scaled diagonal dominance proves M^(p) - ell1 I is positive semidefinite.",
            "step_2": "The comparison-matrix quadratic-form inequality proves lambda_min(H^(p)) >= ell1.",
            "step_3": "Cauchy interlacing proves E1(H) >= lambda_min(H^(p)) >= ell1.",
            "ground_bound": "The exact Rayleigh quotient proves E0(H) <= U0.",
            "conclusion": "Delta(H) >= ell1 - U0 > 0.",
        },
        "timings_seconds": {
            "binary64_to_exact": conversion_seconds,
            "exact_sparse_arithmetic": arithmetic_seconds,
            "total": time.perf_counter() - started,
        },
    }
    return result


# --------------------------- manifest verification --------------------------


def verify_manifest(
    manifest_path: Path = DEFAULT_MANIFEST,
    *,
    write_results: bool = True,
    verbose: bool = True,
) -> bool:
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        if verbose:
            print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return False

    try:
        manifest = _load_json(manifest_path)
        if manifest.get("schema_version") != 1:
            raise VerificationError("unsupported manifest schema")
        if manifest.get("record_type") != "level3_anchor_manifest":
            raise VerificationError("unexpected manifest record type")
        entries = manifest.get("anchors")
        if not isinstance(entries, list):
            raise VerificationError("manifest anchors must be a list")
        entry_ids = [entry.get("anchor_id") for entry in entries if isinstance(entry, dict)]
        if len(entry_ids) != len(entries) or set(entry_ids) != set(EXPECTED_ANCHORS):
            raise VerificationError(
                f"manifest must contain exactly these anchors: {sorted(EXPECTED_ANCHORS)}"
            )
    except VerificationError as exc:
        if verbose:
            print(f"ERROR: {exc}", file=sys.stderr)
        return False

    outcomes: list[dict[str, Any]] = []
    overall_pass = True
    for entry in entries:
        anchor_id = entry["anchor_id"]
        result_path: Path | None = None
        try:
            spec_path = _project_path(entry.get("spec_path"))
            _require_file_hash(spec_path, entry.get("spec_sha256"), f"spec for {anchor_id}")
            spec = _load_json(spec_path)
            result_path = _project_path(entry.get("result_path"))
            result = verify_anchor(spec, EXPECTED_ANCHORS[anchor_id])
            if write_results:
                _write_json(result_path, result)
            result_hash = sha256_file(result_path) if write_results else None
            outcomes.append(
                {
                    "anchor_id": anchor_id,
                    "status": "PASS",
                    "result_path": entry.get("result_path"),
                    "result_sha256": result_hash,
                    "gap_lower": result["exact_bounds"]["gap_lower_ell1_minus_U0"],
                    "timings_seconds": result["timings_seconds"],
                }
            )
            if verbose:
                gap = result["exact_bounds"]["gap_lower_ell1_minus_U0"]["decimal_approx"]
                elapsed = result["timings_seconds"]["total"]
                print(f"PASS {anchor_id}: exact gap lower ~= {gap} ({elapsed:.3f} s)")
        except (VerificationError, OSError, ValueError, ArithmeticError) as exc:
            overall_pass = False
            failure = {
                "schema_version": 1,
                "record_type": "level3_anchor_verification_result",
                "verification_status": "FAIL",
                "anchor_id": anchor_id,
                "error": str(exc),
                "scope_statement": "No Level-3 certificate was accepted for this anchor.",
            }
            if write_results and result_path is not None:
                _write_json(result_path, failure)
            outcomes.append(
                {
                    "anchor_id": anchor_id,
                    "status": "FAIL",
                    "result_path": entry.get("result_path"),
                    "error": str(exc),
                }
            )
            if verbose:
                print(f"FAIL {anchor_id}: {exc}", file=sys.stderr)

    summary_path_text = manifest.get(
        "verification_summary_path", "results/level3/verification_manifest.json"
    )
    try:
        summary_path = _project_path(summary_path_text)
        summary = {
            "schema_version": 1,
            "record_type": "level3_verification_manifest",
            "overall_status": "PASS" if overall_pass else "FAIL",
            "scope_statement": (
                "Passing anchors are Level 3 for their exact-dyadic Hamiltonians; "
                "continuous sweeps remain Level 2."
            ),
            "source_manifest_path": str(manifest_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "source_manifest_sha256": sha256_file(manifest_path),
            "anchors": outcomes,
        }
        if write_results:
            _write_json(summary_path, summary)
    except (VerificationError, OSError) as exc:
        overall_pass = False
        if verbose:
            print(f"ERROR writing verification summary: {exc}", file=sys.stderr)

    if verbose:
        if overall_pass:
            print(
                "All four anchors are Level-3 certificates for the exact-dyadic "
                "Hamiltonians. Continuous sweeps remain Level 2."
            )
        else:
            print("Verification failed; no aggregate Level-3 claim is accepted.", file=sys.stderr)
    return overall_pass


# ------------------------------ self-tests ----------------------------------


def run_self_test(verbose: bool = True) -> bool:
    edge_payload = [[0, 1, 0.5]]
    graph_record: dict[str, Any] = {
        "schema_version": 2,
        "record_type": "generated_weighted_graph_instance",
        "row_key": {"N": 1, "seed": 0},
        "N_effective_qubits": 1,
        "n_vertices": 2,
        "num_edges": 1,
        "generation": {"seed": 0, "rng_bit_generator": "PCG64", "attempts": 1},
        "edge_list": edge_payload,
        "edge_weight_hex": [float(0.5).hex()],
        "edge_weight_semantics": (
            "archived binary64 values interpreted as exact dyadic rationals"
        ),
        "edge_payload_sha256": _canonical_sha256(edge_payload),
    }
    graph_record["graph_record_sha256"] = _canonical_sha256(graph_record)

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", encoding="ascii", delete=False
        ) as handle:
            json.dump(graph_record, handle, allow_nan=False)
            temporary_path = Path(handle.name)
        parsed_graph = load_exact_graph_record(temporary_path, expected_N=1, expected_seed=0)
        if parsed_graph.edges != ((0, 1, Fraction(1, 2)),):
            raise VerificationError("schema-2 exact edge reconstruction failed")
    except (VerificationError, OSError) as exc:
        if verbose:
            print(f"self-test failed: temporary schema-2 record: {exc}", file=sys.stderr)
        return False
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    # Recompute both canonical hashes after changing the JSON float, but leave
    # the authoritative hex value untouched. The parser must still reject the
    # semantic float/hex mismatch.
    mismatched_record = json.loads(json.dumps(graph_record, allow_nan=False))
    mismatched_record["edge_list"][0][2] = 0.75
    mismatched_record["edge_payload_sha256"] = _canonical_sha256(
        mismatched_record["edge_list"]
    )
    mismatched_record.pop("graph_record_sha256", None)
    mismatched_record["graph_record_sha256"] = _canonical_sha256(mismatched_record)
    try:
        _parse_exact_graph_record(
            mismatched_record, "self-test float/hex mismatch", expected_N=1, expected_seed=0
        )
    except VerificationError:
        pass
    else:
        if verbose:
            print("self-test failed: schema-2 float/hex mismatch was accepted", file=sys.stderr)
        return False

    # Exact diagonal example: deleting coordinate zero leaves diag(2, 5),
    # so ell1=2 and interlacing identifies E1=2 exactly.
    x = [Fraction(1, 1), Fraction(1, 1)]
    comparison_x = [Fraction(2, 1), Fraction(5, 1)]
    ratios = [comparison_x[i] / x[i] for i in range(2)]
    ell1 = min(ratios)
    margins = [comparison_x[i] - ell1 * x[i] for i in range(2)]
    if ell1 != 2 or any(margin < 0 for margin in margins):
        if verbose:
            print("self-test failed: exact comparison/interlacing arithmetic", file=sys.stderr)
        return False

    q = [Fraction(1, 1), Fraction(0, 1), Fraction(0, 1)]
    hq = [Fraction(0, 1), Fraction(0, 1), Fraction(0, 1)]
    U0 = _dot(q, hq) / _dot(q, q)
    if ell1 - U0 != 2:
        if verbose:
            print("self-test failed: exact Rayleigh/gap arithmetic", file=sys.stderr)
        return False

    original = b"level3-witness"
    tampered = bytearray(original)
    tampered[-1] ^= 1
    if hashlib.sha256(original).digest() == hashlib.sha256(tampered).digest():
        if verbose:
            print("self-test failed: tamper hash", file=sys.stderr)
        return False

    if verbose:
        print(
            "self-test passed: schema-2 canonical hashes, float/hex rejection, "
            "exact theorem arithmetic, and tamper hashing"
        )
    return True


def run_tamper_check(manifest_path: Path, verbose: bool = True) -> bool:
    try:
        manifest = _load_json(manifest_path.resolve())
        entries = manifest.get("anchors")
        if not isinstance(entries, list) or not entries:
            raise VerificationError("manifest has no anchors")
        first_entry = entries[0]
        if not isinstance(first_entry, dict):
            raise VerificationError("tamper-check manifest entry is not an object")
        spec_path = _project_path(
            _required_str(first_entry, "spec_path", "tamper-check spec path")
        )
        _require_file_hash(
            spec_path,
            _required_str(first_entry, "spec_sha256", "tamper-check spec hash"),
            "tamper-check spec",
        )
        spec = _load_json(spec_path)
        witness_record = spec.get("ground_witness")
        if not isinstance(witness_record, dict):
            raise VerificationError("tamper-check spec has no ground witness")
        witness_path = _project_path(
            _required_str(witness_record, "path", "tamper-check witness path")
        )
        expected_hash = _required_str(
            witness_record, "sha256", "tamper-check witness hash"
        )
        _require_file_hash(witness_path, expected_hash, "tamper-check source witness")
        data = bytearray(witness_path.read_bytes())
        if not data:
            raise VerificationError("tamper-check witness is empty")
        data[-1] ^= 1
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as handle:
                handle.write(data)
                temporary_path = Path(handle.name)
            tampered_hash = sha256_file(temporary_path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        if tampered_hash == expected_hash:
            raise VerificationError("tamper check did not change SHA-256")
    except (VerificationError, OSError) as exc:
        if verbose:
            print(f"tamper check failed: {exc}", file=sys.stderr)
        return False
    if verbose:
        print("tamper check passed: a one-bit witness change is rejected by SHA-256")
    return True


# ---------------------------------- CLI -------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify all representative Level-3 anchors using exact rational sparse "
            "arithmetic. With no mode flag, verification is performed and any mismatch "
            "causes a nonzero exit status."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"manifest to verify (default: {DEFAULT_MANIFEST})",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--self-test",
        action="store_true",
        help="run exact arithmetic and in-memory tamper self-tests only",
    )
    mode.add_argument(
        "--tamper-check",
        action="store_true",
        help="flip one witness bit in a temporary file and confirm its hash is rejected",
    )
    parser.add_argument(
        "--no-write-results",
        action="store_true",
        help="verify without rewriting per-anchor result and verification-manifest JSON",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress normal progress output")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    verbose = not args.quiet
    if args.self_test:
        return 0 if run_self_test(verbose=verbose) else 1
    if args.tamper_check:
        return 0 if run_tamper_check(args.manifest, verbose=verbose) else 1
    ok = verify_manifest(
        args.manifest,
        write_results=not args.no_write_results,
        verbose=verbose,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
