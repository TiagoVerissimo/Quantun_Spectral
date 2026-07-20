#!/usr/bin/env python3
"""Generate full exact-dyadic Level-3 continuations for selected N=10 graphs.

The generator uses SciPy only to propose witnesses.  Every accepted anchor is
immediately checked by ``verify_level3_anchors.verify_anchor`` with exact
rational arithmetic.  The resulting manifest is independently replayed by
``verify_level3_continuations.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from fractions import Fraction
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import scipy

import generate_level3_anchors as witness_generator
import verify_level3_anchors as anchor_verifier


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR = RESULTS_DIR / "level3_continuations"
DEFAULT_SEEDS = (18, 4, 10)
OVERLAP_SAFETY_ULPS = 1
MAX_ANCHORS = 2_000


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
    for attempt in range(20):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.1 * (attempt + 1))


def _project_relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")


def _fraction(payload: dict[str, Any]) -> Fraction:
    return Fraction(int(payload["numerator"]), int(payload["denominator"]))


def _gap_certificate_from_graph(path: Path) -> Fraction:
    record = json.loads(path.read_text(encoding="utf-8"))
    k_gap = Fraction(
        int(record["K_gap_cert_numerator"]), int(record["K_gap_cert_denominator"])
    )
    w_exact = Fraction(int(record["W_exact_numerator"]), int(record["W_exact_denominator"]))
    if k_gap < w_exact + int(record["Lambda_I"]):
        raise RuntimeError(f"invalid K_gap_cert record: {path}")
    return k_gap


def _exact_problem_endpoint_gap(graph: anchor_verifier.ExactGraph) -> Fraction:
    costs: list[Fraction] = []
    for state in range(graph.dimension):
        cost = Fraction(0, 1)
        for u, v, weight in graph.edges:
            if u == 0:
                uncut = ((state >> (v - 1)) & 1) == 0
            else:
                uncut = ((state >> (u - 1)) & 1) == ((state >> (v - 1)) & 1)
            if uncut:
                cost += weight
        costs.append(cost)
    ordered = sorted(costs)
    if len(ordered) < 2 or ordered[1] <= ordered[0]:
        raise RuntimeError("problem endpoint is not exactly nondegenerate")
    return ordered[1] - ordered[0]


def _candidate_gap(
    N: int,
    s: float,
    edges: Sequence[tuple[int, int, float]],
    basis: str,
    salt: int,
) -> float:
    """Return an untrusted floating candidate used only to choose a basis."""
    dimension = 1 << N
    if basis == "Z":
        actual = witness_generator._build_z_matrix(N, s, edges)
        comparison = actual
    elif basis == "X":
        actual, comparison = witness_generator._build_x_matrices(N, s, edges)
    else:
        raise ValueError(f"unsupported basis: {basis}")
    ground_value, ground = witness_generator._smallest_eigenpair(
        actual, witness_generator._deterministic_start(dimension, salt, positive=False)
    )
    removed = int(np.argmax(np.abs(ground))) if basis == "Z" else 0
    principal = witness_generator._principal_submatrix(comparison, removed)
    principal_value, _ = witness_generator._smallest_eigenpair(
        principal,
        witness_generator._deterministic_start(dimension - 1, salt + 211, positive=True),
    )
    return float(principal_value - ground_value)


def _inside_covered_frontier(covered: Fraction) -> float:
    """Return a binary64 point strictly inside the exact covered interval."""
    candidate = float(covered)
    for _ in range(OVERLAP_SAFETY_ULPS + 8):
        if Fraction.from_float(candidate) < covered:
            return candidate
        candidate = float(np.nextafter(candidate, -np.inf))
    raise RuntimeError("could not select a binary64 anchor inside covered frontier")


def _anchor_entry_from_existing(
    spec_path: Path, result_path: Path, k_gap: Fraction, covered: Fraction
) -> tuple[dict[str, Any], Fraction]:
    """Recover an accepted anchor after an interrupted batch run.

    Final acceptance never relies on this recovery path: the independent
    continuation verifier reruns every exact anchor calculation.
    """
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("verification_status") != "PASS":
        raise RuntimeError(f"interrupted result is not PASS: {result_path}")
    delta = _fraction(result["exact_bounds"]["gap_lower_ell1_minus_U0"])
    s = Fraction.from_float(float.fromhex(spec["s_hex"]))
    raw_right = s + delta / k_gap
    if not s - delta / k_gap < covered or raw_right <= covered:
        raise RuntimeError(f"interrupted anchor does not extend strict coverage: {spec_path}")
    entry = {
        "anchor_id": spec["anchor_id"],
        "spec_path": _project_relative(spec_path),
        "spec_sha256": _sha256_file(spec_path),
        "result_path": _project_relative(result_path),
        "basis_selection": "recovered accepted exact anchor",
        "s_hex": spec["s_hex"],
        "s_decimal_input": spec["s_decimal_input"],
        "exact_gap_lower": result["exact_bounds"]["gap_lower_ell1_minus_U0"],
        "covered_right_after": {
            "numerator": str(raw_right.numerator),
            "denominator": str(raw_right.denominator),
            "decimal_approx": format(float(raw_right), ".30g"),
        },
    }
    return entry, raw_right


def _initial_record(seed: int, graph_path: Path, graph: anchor_verifier.ExactGraph, k_gap: Fraction) -> dict[str, Any]:
    return {
        "N_effective_qubits": 10,
        "seed": seed,
        "graph_path": _project_relative(graph_path),
        "graph_file_sha256": _sha256_file(graph_path),
        "graph_record_sha256": graph.graph_record_sha256,
        "edge_payload_sha256": graph.edge_payload_sha256,
        "K_gap_cert": {
            "numerator": str(k_gap.numerator),
            "denominator": str(k_gap.denominator),
            "decimal_approx": format(float(k_gap), ".30g"),
        },
        "initial_exact_anchor": {
            "s": "0",
            "gap_lower": "2",
            "justification": "The reduced even-sector driver has E0(0)=0 and E1(0)=2.",
        },
        "anchors": [],
    }


def _recover_or_initialize_seed(
    seed: int, graph_path: Path, graph: anchor_verifier.ExactGraph, k_gap: Fraction, output_dir: Path
) -> tuple[dict[str, Any], Fraction]:
    progress_path = output_dir / f"progress_N10_seed{seed}.json"
    if progress_path.is_file():
        record = json.loads(progress_path.read_text(encoding="utf-8"))
        if record.get("seed") != seed:
            raise RuntimeError(f"wrong seed in progress record: {progress_path}")
        anchors = record.get("anchors")
        if not isinstance(anchors, list):
            raise RuntimeError(f"invalid anchor list in progress record: {progress_path}")
        if anchors:
            covered = _fraction(anchors[-1]["covered_right_after"])
        else:
            covered = Fraction(2, 1) / k_gap
        return record, covered

    record = _initial_record(seed, graph_path, graph, k_gap)
    covered = Fraction(2, 1) / k_gap
    existing: list[tuple[int, Path, Path]] = []
    for result_path in output_dir.glob(f"N10_seed{seed}_continuation_*.result.json"):
        spec_path = result_path.with_name(result_path.name.replace(".result.json", ".spec.json"))
        if not spec_path.is_file():
            continue
        stem = result_path.name.split("_continuation_", 1)[1]
        index = int(stem.split("_", 1)[0])
        existing.append((index, spec_path, result_path))
    for _, spec_path, result_path in sorted(existing):
        entry, covered = _anchor_entry_from_existing(spec_path, result_path, k_gap, covered)
        record["anchors"].append(entry)
    _write_json(progress_path, record)
    return record, covered


def _generate_seed(seed: int, output_dir: Path, max_new_anchors: int | None) -> tuple[dict[str, Any], bool]:
    graph_path, graph = witness_generator._require_primary_graph_record(10, seed)
    k_gap = _gap_certificate_from_graph(graph_path)
    edges = witness_generator._graph_float_edges(graph)

    record, covered = _recover_or_initialize_seed(seed, graph_path, graph, k_gap, output_dir)
    endpoint_gap = _exact_problem_endpoint_gap(graph)
    anchor_entries = record["anchors"]
    progress_path = output_dir / f"progress_N10_seed{seed}.json"
    print(
        f"N=10 seed={seed}: {len(anchor_entries)} accepted anchors, "
        f"coverage through {float(covered):.12g}",
        flush=True,
    )

    initial_anchor_count = len(anchor_entries)
    for index in range(initial_anchor_count, MAX_ANCHORS):
        if covered > 1:
            break
        endpoint_left = Fraction(1, 1) - endpoint_gap / k_gap
        if endpoint_left < covered:
            record["final_exact_endpoint_anchor"] = {
                "s": "1",
                "gap_lower": {
                    "numerator": str(endpoint_gap.numerator),
                    "denominator": str(endpoint_gap.denominator),
                    "decimal_approx": format(float(endpoint_gap), ".30g"),
                },
                "justification": (
                    "Exact enumeration of the reduced diagonal problem Hamiltonian "
                    "gives a unique minimum and this exact second-cost gap."
                ),
            }
            covered = Fraction(1, 1) + endpoint_gap / k_gap
            break
        if max_new_anchors is not None and index - initial_anchor_count >= max_new_anchors:
            record["anchors"] = anchor_entries
            _write_json(progress_path, record)
            return record, False
        s = _inside_covered_frontier(covered)
        candidates: list[tuple[float, str]] = []
        # Once the exact continuation has selected the stoquastic Z-basis
        # certificate, it remains the useful certificate on these paths.  This
        # avoids an expensive, known-loose X-basis trial at every later anchor.
        basis_options = ("Z",) if anchor_entries and str(anchor_entries[-1]["anchor_id"]).endswith("_Z") else ("X", "Z")
        for basis in basis_options:
            try:
                estimate = _candidate_gap(10, s, edges, basis, salt=10 + 37 * index)
                if np.isfinite(estimate) and estimate > 0:
                    candidates.append((estimate, basis))
            except Exception as exc:  # Untrusted selection failure; exact check decides acceptance.
                print(f"  candidate {index} {basis} failed: {exc}", flush=True)
        if not candidates:
            raise RuntimeError(f"no positive floating witness candidate at s={s!r}")

        accepted: tuple[Fraction, str, dict[str, Any], dict[str, Any]] | None = None
        for _, basis in sorted(candidates, reverse=True):
            anchor_id = f"N10_seed{seed}_continuation_{index:04d}_{basis}"
            configuration = {"N": 10, "seed": seed, "s_decimal": repr(s), "basis": basis}
            try:
                entry, spec = witness_generator._generate_anchor(
                    anchor_id, configuration, graph_path, graph
                )
                spec["scope_statement"] = (
                    "This exact-dyadic anchor is one member of a separate Level-3 "
                    "continuation manifest.  Acceptance is determined only by the "
                    "independent continuation verifier."
                )
                spec_path = PROJECT_ROOT / entry["spec_path"]
                _write_json(spec_path, spec)
                entry["spec_sha256"] = _sha256_file(spec_path)
                result = anchor_verifier.verify_anchor(spec, configuration)
                delta = _fraction(result["exact_bounds"]["gap_lower_ell1_minus_U0"])
                if delta <= 0:
                    raise RuntimeError("nonpositive exact gap bound")
                accepted = (delta, basis, entry, result)
                break
            except Exception as exc:
                print(f"  exact anchor {index} {basis} rejected: {exc}", flush=True)
        if accepted is None:
            raise RuntimeError(f"no exact certificate accepted at s={s!r}")

        delta, basis, entry, result = accepted
        s_exact = Fraction.from_float(s)
        raw_right = s_exact + delta / k_gap
        if raw_right <= covered:
            raise RuntimeError("accepted anchor does not extend exact coverage")
        result_path = PROJECT_ROOT / entry["result_path"]
        witness_generator._write_json(result_path, result)
        entry.update(
            {
                "basis_selection": "largest positive untrusted principal-gap candidate",
                "s_hex": s.hex(),
                "s_decimal_input": repr(s),
                "exact_gap_lower": result["exact_bounds"]["gap_lower_ell1_minus_U0"],
                "covered_right_after": {
                    "numerator": str(raw_right.numerator),
                    "denominator": str(raw_right.denominator),
                    "decimal_approx": format(float(raw_right), ".30g"),
                },
            }
        )
        anchor_entries.append(entry)
        covered = raw_right
        record["anchors"] = anchor_entries
        _write_json(progress_path, record)
        print(
            f"  {index:04d} s={s:.12g} basis={basis} "
            f"delta={float(delta):.9g} cover={float(covered):.12g}",
            flush=True,
        )
    else:
        raise RuntimeError(f"exceeded {MAX_ANCHORS} anchors for seed {seed}")

    record["anchors"] = anchor_entries
    record["final_unclipped_coverage_right"] = {
            "numerator": str(covered.numerator),
            "denominator": str(covered.denominator),
            "decimal_approx": format(float(covered), ".30g"),
        }
    _write_json(progress_path, record)
    return record, True


def generate(
    seeds: Sequence[int], output_dir: Path, max_new_anchors: int | None, append: bool
) -> Path | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    existing_continuations: list[dict[str, Any]] = []
    if manifest_path.is_file():
        if not append:
            raise RuntimeError(
                f"refusing to overwrite completed manifest: {manifest_path}; use --append "
                "to add different seeds"
            )
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest.get("record_type") != "level3_continuation_manifest":
            raise RuntimeError(f"unsupported existing manifest: {manifest_path}")
        existing_continuations = existing_manifest.get("continuations")
        if not isinstance(existing_continuations, list):
            raise RuntimeError(f"existing manifest has no continuation list: {manifest_path}")
        existing_seeds = {record.get("seed") for record in existing_continuations if isinstance(record, dict)}
        duplicates = sorted(set(seeds) & {seed for seed in existing_seeds if isinstance(seed, int)})
        if duplicates:
            raise RuntimeError(f"continuation manifest already contains seeds: {duplicates}")
    original_dir = witness_generator.LEVEL3_DIR
    try:
        witness_generator.LEVEL3_DIR = output_dir.resolve()
        generated = [_generate_seed(seed, output_dir, max_new_anchors) for seed in seeds]
    finally:
        witness_generator.LEVEL3_DIR = original_dir

    incomplete = [record["seed"] for record, complete in generated if not complete]
    if incomplete:
        print(f"Saved resumable progress; incomplete seeds: {incomplete}")
        return None
    continuations = existing_continuations + [record for record, _ in generated]
    manifest = {
        "schema_version": 1,
        "record_type": "level3_continuation_manifest",
        "certificate_scope": (
            "Complete Level-3 gap continuations for the listed exact-dyadic N=10 paths."
        ),
        "scope_statement": (
            "Every non-endpoint anchor is independently rechecked in exact arithmetic; "
            "strictly overlapping Weyl intervals cover [0,1]."
        ),
        "continuations": continuations,
        "verification_summary_path": _project_relative(output_dir / "verification_manifest.json"),
        "source_hashes": {
            "generate_level3_continuations.py": _sha256_file(Path(__file__).resolve()),
            "verify_level3_continuations.py": _sha256_file(
                PROJECT_ROOT / "verify_level3_continuations.py"
            ),
            "verify_level3_anchors.py": _sha256_file(
                PROJECT_ROOT / "verify_level3_anchors.py"
            ),
        },
        "software_for_untrusted_witness_generation": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--max-new-anchors",
        type=int,
        default=None,
        help="generate at most this many new exact anchors per seed, then save progress and exit",
    )
    parser.add_argument("--append", action="store_true", help="append new seeds to an existing manifest")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest_path = generate(
        tuple(args.seeds), args.output.resolve(), args.max_new_anchors, args.append
    )
    if manifest_path is None:
        return 2
    print(f"Wrote continuation manifest: {_project_relative(manifest_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
