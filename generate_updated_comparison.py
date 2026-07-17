from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from main import PathOperator, certified_sweep, lowest_two, pinned_cost_vector
from summarize_section7_results import (
    ArtifactValidationError,
    EXPECTED_GRAPH_SCHEMA_VERSION,
    RESULTS_PATH,
    exact_binary64_sum,
    least_binary64_not_below,
    load_rows,
    sha256_path,
)


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "results"


@dataclass(frozen=True)
class ArchivedInstance:
    N: int
    seed: int
    edges: tuple[tuple[int, int, float], ...]
    costs: np.ndarray
    K_gap_cert: float
    row: dict[str, Any]
    graph_record: dict[str, Any]


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _project_path(relative_path: str) -> Path:
    path = (PROJECT_ROOT / relative_path).resolve()
    try:
        path.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ArtifactValidationError(
            f"artifact path escapes the project root: {relative_path}"
        ) from exc
    return path


def _verify_graph_record(record: dict[str, Any]) -> None:
    if record.get("schema_version") != EXPECTED_GRAPH_SCHEMA_VERSION:
        raise ArtifactValidationError(
            f"expected graph schema {EXPECTED_GRAPH_SCHEMA_VERSION}"
        )
    expected_record_hash = str(record.get("graph_record_sha256", ""))
    hash_input = {
        key: value
        for key, value in record.items()
        if key != "graph_record_sha256"
    }
    if _sha256(_canonical_json_bytes(hash_input)) != expected_record_hash:
        raise ArtifactValidationError("archived graph-record SHA-256 mismatch")
    edge_payload = record.get("edge_list")
    if not isinstance(edge_payload, list):
        raise ArtifactValidationError("archived graph record has no edge_list")
    if _sha256(_canonical_json_bytes(edge_payload)) != record.get(
        "edge_payload_sha256"
    ):
        raise ArtifactValidationError("archived edge-payload SHA-256 mismatch")


def _load_record_from_archive(
    archive_path: Path, N: int, seed: int
) -> dict[str, Any]:
    try:
        lines = archive_path.read_bytes().splitlines()
    except OSError as exc:
        raise ArtifactValidationError(
            f"cannot read graph archive {archive_path}: {exc}"
        ) from exc
    for line in lines:
        if not line.strip():
            continue
        record = json.loads(line.decode("ascii"))
        row_key = record.get("row_key", {})
        if row_key.get("N") == N and row_key.get("seed") == seed:
            return record
    raise ArtifactValidationError(
        f"graph archive has no record for N={N}, seed={seed}"
    )


def load_archived_instance(N: int, seed: int) -> ArchivedInstance:
    rows = load_rows(RESULTS_PATH)
    matches = [
        row
        for row in rows
        if int(row["N"]) == N and int(row["seed"]) == seed
    ]
    if len(matches) != 1:
        raise ArtifactValidationError(
            f"expected one CSV row for N={N}, seed={seed}, found {len(matches)}"
        )
    row = matches[0]
    archive_path = _project_path(str(row["graph_archive_path"]))
    if archive_path.is_file():
        record = _load_record_from_archive(archive_path, N, seed)
    else:
        record_path = _project_path(str(row["graph_record_path"]))
        if not record_path.is_file():
            raise ArtifactValidationError(
                f"missing graph archive and individual record for N={N}, seed={seed}"
            )
        record = json.loads(record_path.read_text(encoding="ascii"))

    _verify_graph_record(record)
    row_key = record.get("row_key", {})
    if row_key != {"N": N, "seed": seed}:
        raise ArtifactValidationError("graph row key does not match the CSV row")
    if record.get("graph_record_sha256") != row["graph_record_sha256"]:
        raise ArtifactValidationError("CSV and graph-record hashes disagree")
    if record.get("edge_payload_sha256") != row["edge_payload_sha256"]:
        raise ArtifactValidationError("CSV and edge-payload hashes disagree")

    try:
        edges = tuple(
            (int(edge[0]), int(edge[1]), float(edge[2]))
            for edge in record["edge_list"]
        )
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ArtifactValidationError("invalid edge_list in graph record") from exc
    if len(edges) != int(row["m"]):
        raise ArtifactValidationError("edge count differs between CSV and graph record")
    W_model = float(sum(weight for _, _, weight in edges))
    W_exact = exact_binary64_sum(weight for _, _, weight in edges)
    W_upper = least_binary64_not_below(W_exact)
    if W_model != float(row["W_model"]):
        raise ArtifactValidationError("nominal W_model differs from the CSV")
    if W_model.hex() != row["W_model_hex"]:
        raise ArtifactValidationError("W_model hexadecimal metadata disagrees")
    if W_upper != float(row["W_upper"]) or float(row["W"]) != W_upper:
        raise ArtifactValidationError("upward-rounded W_upper differs from the CSV")

    n_vertices = N + 1
    lambda_i = 2 * (n_vertices // 2)
    expected_K_gap = least_binary64_not_below(
        Fraction.from_float(W_upper) + lambda_i
    )
    K_gap_cert = float(row["K_gap_cert"])
    if K_gap_cert != expected_K_gap:
        raise ArtifactValidationError(
            "K_gap_cert is not the required upward-rounded bound"
        )

    costs = pinned_cost_vector(N, edges)
    return ArchivedInstance(
        N=N,
        seed=seed,
        edges=edges,
        costs=costs,
        K_gap_cert=K_gap_cert,
        row=row,
        graph_record=record,
    )


def lower_envelope(
    s_grid: np.ndarray,
    anchors: list[tuple[float, float]],
    K_gap_cert: float,
) -> np.ndarray:
    envelope = np.full_like(s_grid, -np.inf, dtype=float)
    for s_anchor, gap_lower_bound in anchors:
        envelope = np.maximum(
            envelope,
            gap_lower_bound - K_gap_cert * np.abs(s_grid - s_anchor),
        )
    return envelope


def sampled_ritz_gap_curve(
    path_operator: PathOperator, s_grid: np.ndarray
) -> np.ndarray:
    gaps = []
    warm_start = None
    for s in s_grid:
        values, vectors, _ = lowest_two(path_operator.H(float(s)), v0=warm_start)
        gaps.append(float(values[1] - values[0]))
        warm_start = vectors[:, 0]
    return np.asarray(gaps, dtype=float)


def generate_graphs() -> list[Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    instance = load_archived_instance(N=10, seed=0)
    figure_metadata = {
        "Section7CSV-SHA256": sha256_path(RESULTS_PATH),
        "GraphRecord-SHA256": str(instance.row["graph_record_sha256"]),
    }
    path_operator = PathOperator(instance.N, instance.costs)
    endpoint_gap = float(
        np.partition(instance.costs, 1)[1]
        - np.partition(instance.costs, 1)[0]
    )
    endpoint_anchors = [(0.0, 2.0), (1.0, endpoint_gap)]
    s_grid = np.linspace(0.0, 1.0, 201)

    print("Computing sampled floating-point reference gap curve...")
    reference_gap = sampled_ritz_gap_curve(path_operator, s_grid)
    endpoint_bound = lower_envelope(
        s_grid, endpoint_anchors, instance.K_gap_cert
    )

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(s_grid, reference_gap, "k-", linewidth=2.5, label="Sampled Ritz gap")
    axis.plot(
        s_grid,
        endpoint_bound,
        color="#3a86c8",
        linewidth=1.8,
        label="Weyl endpoint envelope",
    )
    axis.axhline(0.0, color="#9ea1a5", linewidth=0.8)
    axis.set_xlabel("Interpolation parameter $s$")
    axis.set_ylabel("Gap / lower envelope")
    axis.set_ylim(-0.2, 1.25 * float(reference_gap.max()))
    axis.set_title("Endpoint Weyl envelope ($N=10$, seed 0)")
    axis.legend()
    axis.grid(True, linestyle=":", alpha=0.5)
    figure.tight_layout()
    endpoint_output = OUTPUT_DIR / "comparison_updated_N10_no_anchors.png"
    figure.savefig(
        endpoint_output,
        dpi=160,
        metadata={"Artifact": endpoint_output.name, **figure_metadata},
    )
    plt.close(figure)

    budgets = (5, 15, 25)
    figure, axes = plt.subplots(3, 1, figsize=(10, 14), sharex=True)
    print("Computing budgeted adaptive Weyl envelopes...")
    for axis, budget in zip(axes, budgets):
        records, _, _ = certified_sweep(
            path_operator,
            instance.K_gap_cert,
            max_anchors=budget,
        )
        adaptive_anchors = [
            (float(s), float(gap_lower_bound))
            for s, _, _, gap_lower_bound in records
        ]
        bound = lower_envelope(
            s_grid,
            endpoint_anchors + adaptive_anchors,
            instance.K_gap_cert,
        )
        axis.plot(
            s_grid,
            reference_gap,
            color="#1e1e24",
            linewidth=2.5,
            label="Sampled Ritz gap",
        )
        axis.plot(
            s_grid,
            bound,
            color="#3a86c8",
            linewidth=1.8,
            label=(
                f"Weyl envelope ({len(records)} sparse eigensolves "
                "+ endpoint data)"
            ),
        )
        axis.axhline(0.0, color="#9ea1a5", linewidth=0.8)
        axis.set_title(
            f"Adaptive sparse-eigensolve budget: {budget}",
            fontsize=11,
            fontweight="bold",
        )
        axis.set_ylabel("Gap / lower envelope", fontsize=10)
        axis.grid(True, linestyle=":", alpha=0.5)
        axis.legend(loc="upper right", fontsize=9)
        axis.set_ylim(-0.2, float(reference_gap.max()) * 1.15)

    axes[-1].set_xlabel("Interpolation parameter $s$", fontsize=11)
    figure.suptitle(
        "Budgeted Weyl envelopes: $N=10$, seed 0",
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout()
    anchors_output = OUTPUT_DIR / "comparison_updated_N10_anchors.png"
    figure.savefig(
        anchors_output,
        dpi=160,
        metadata={"Artifact": anchors_output.name, **figure_metadata},
    )
    plt.close(figure)
    return [endpoint_output, anchors_output]


def main() -> int:
    try:
        outputs = generate_graphs()
    except (ArtifactValidationError, OSError, ValueError) as exc:
        print(f"comparison plot generation failed: {exc}", file=sys.stderr)
        return 1
    rendered = ", ".join(str(path.relative_to(PROJECT_ROOT)) for path in outputs)
    print(f"Generated {rendered} from the archived schema-v4 instance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
