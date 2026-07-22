from __future__ import annotations

import hashlib
import json
import math
import re
import struct
import sys
import zlib
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping

from summarize_section7_results import (
    ArtifactValidationError,
    EXPECTED_GRAPH_SCHEMA_VERSION,
    EXPECTED_NS,
    EXPECTED_ROW_COUNT,
    EXPECTED_SCHEMA_VERSION,
    EXPECTED_SEEDS,
    OUTPUT_PATH,
    PROJECT_ROOT,
    RESULTS_PATH,
    TEX_OUTPUT_PATH,
    exact_binary64_sum,
    expected_outputs,
    least_binary64_not_below,
    load_rows,
    sha256_path,
)


RESULTS_DIR = PROJECT_ROOT / "results"
RUN_MANIFEST_PATH = RESULTS_DIR / "section7_run_manifest.json"
_FLOAT_REL_TOL = 1e-11
_FLOAT_ABS_TOL = 1e-12
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_HEX_DYADIC_RE = re.compile(
    r"^(?P<sign>[+-]?)0[xX](?P<integer>[0-9a-fA-F]+)"
    r"(?:\.(?P<fraction>[0-9a-fA-F]*))?[pP](?P<exponent>[+-]?\d+)$"
)
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_REQUIRED_FIGURES = (
    "comparison_updated_N10_no_anchors.png",
    "comparison_updated_N10_anchors.png",
    "efficiency_comparison.png",
    "certificate_coverage.png",
    "grid_density_vs_gap.png",
    "grid_density_vs_gap_N12.png",
    "grid_density_vs_gap_N14.png",
)


class ArtifactVerificationError(ValueError):
    """Raised when archived artifacts fail an integrity check."""


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: object, label: str) -> str:
    digest = str(value)
    if not _SHA256_RE.fullmatch(digest):
        raise ArtifactVerificationError(
            f"{label} must be a lowercase SHA-256 digest"
        )
    return digest


def _close(left: float, right: float) -> bool:
    return math.isclose(
        left,
        right,
        rel_tol=_FLOAT_REL_TOL,
        abs_tol=_FLOAT_ABS_TOL,
    )


def _as_float(value: object, label: str) -> float:
    try:
        parsed = float(str(value))
    except (TypeError, ValueError) as exc:
        raise ArtifactVerificationError(f"{label} is not numeric") from exc
    if not math.isfinite(parsed):
        raise ArtifactVerificationError(f"{label} must be finite")
    return parsed


def _as_int(value: object, label: str) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError) as exc:
        raise ArtifactVerificationError(f"{label} is not an integer") from exc


def _require_close(left: object, right: object, label: str) -> None:
    left_float = _as_float(left, label)
    right_float = _as_float(right, label)
    if not _close(left_float, right_float):
        raise ArtifactVerificationError(
            f"{label} mismatch: {left_float!r} != {right_float!r}"
        )


def _project_path(relative_path: object, label: str) -> Path:
    text = str(relative_path).strip()
    if not text:
        raise ArtifactVerificationError(f"{label} must be nonempty")
    path = (PROJECT_ROOT / text).resolve()
    try:
        path.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ArtifactVerificationError(
            f"{label} escapes the project root: {text}"
        ) from exc
    return path


def _load_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="ascii"))
    except OSError as exc:
        raise ArtifactVerificationError(f"cannot read {label}: {exc}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactVerificationError(f"invalid JSON in {label}: {exc}") from exc


def _verify_run_manifest(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], Path]:
    manifest = _load_json(RUN_MANIFEST_PATH, "run manifest")
    if not isinstance(manifest, dict):
        raise ArtifactVerificationError("run manifest must contain a JSON object")
    expected_metadata = {
        "schema_version": 1,
        "result_schema_version": EXPECTED_SCHEMA_VERSION,
        "N_values": list(EXPECTED_NS),
        "seeds": list(EXPECTED_SEEDS),
        "rows": EXPECTED_ROW_COUNT,
    }
    for field, expected in expected_metadata.items():
        if manifest.get(field) != expected:
            raise ArtifactVerificationError(
                f"run manifest {field} mismatch: {manifest.get(field)!r} != {expected!r}"
            )

    csv_entry = manifest.get("csv")
    if not isinstance(csv_entry, dict):
        raise ArtifactVerificationError("run manifest has no csv object")
    csv_path = _project_path(csv_entry.get("path"), "manifest csv path")
    if csv_path != RESULTS_PATH.resolve():
        raise ArtifactVerificationError(
            "run manifest must reference section7_results.csv"
        )
    expected_csv_hash = _require_sha256(
        csv_entry.get("sha256"), "manifest CSV hash"
    )
    if sha256_path(csv_path) != expected_csv_hash:
        raise ArtifactVerificationError("section7_results.csv SHA-256 mismatch")

    archive_entry = manifest.get("graph_archive")
    if not isinstance(archive_entry, dict):
        raise ArtifactVerificationError("run manifest has no graph_archive object")
    if archive_entry.get("records") != EXPECTED_ROW_COUNT:
        raise ArtifactVerificationError(
            f"graph archive manifest must report {EXPECTED_ROW_COUNT} records"
        )
    archive_path = _project_path(
        archive_entry.get("path"), "manifest graph archive path"
    )
    archive_hash = _require_sha256(
        archive_entry.get("sha256"), "manifest graph archive hash"
    )
    if not archive_path.is_file():
        raise ArtifactVerificationError(
            f"missing graph archive: {archive_path.relative_to(PROJECT_ROOT)}"
        )
    if sha256_path(archive_path) != archive_hash:
        raise ArtifactVerificationError("graph archive SHA-256 mismatch")

    first = rows[0]
    if str(first["graph_archive_path"]) != str(archive_entry["path"]):
        raise ArtifactVerificationError("CSV and manifest graph archive paths disagree")
    if str(first["graph_archive_sha256"]) != archive_hash:
        raise ArtifactVerificationError("CSV and manifest graph archive hashes disagree")

    source_hashes = manifest.get("source_sha256")
    if not isinstance(source_hashes, dict):
        raise ArtifactVerificationError("run manifest has no source_sha256 object")
    source_files = {
        "main.py": PROJECT_ROOT / "main.py",
        "maxcut_gap_benchmark.py": PROJECT_ROOT / "maxcut_gap_benchmark.py",
        "extend_level2_n14.py": PROJECT_ROOT / "extend_level2_n14.py",
    }
    for name, path in source_files.items():
        expected = _require_sha256(
            source_hashes.get(name), f"manifest source hash for {name}"
        )
        if sha256_path(path) != expected:
            raise ArtifactVerificationError(
                f"source hash mismatch for {name}; artifacts were generated "
                "from different source bytes"
            )
    return manifest, archive_path


def _record_fraction(
    record: Mapping[str, Any],
    nested_key: str,
    flat_prefix: str | None,
    label: str,
) -> Fraction:
    nested = record.get(nested_key)
    if not isinstance(nested, dict):
        raise ArtifactVerificationError(f"{label}: missing rational metadata")
    numerator = _as_int(nested.get("numerator"), f"{label} numerator")
    denominator = _as_int(nested.get("denominator"), f"{label} denominator")
    if denominator <= 0:
        raise ArtifactVerificationError(f"{label}: denominator must be positive")
    value = Fraction(numerator, denominator)
    if value.numerator != numerator or value.denominator != denominator:
        raise ArtifactVerificationError(f"{label}: rational is not canonical")
    if flat_prefix is not None:
        if _as_int(record.get(f"{flat_prefix}_numerator"), label) != numerator:
            raise ArtifactVerificationError(f"{label}: flat numerator disagrees")
        if _as_int(record.get(f"{flat_prefix}_denominator"), label) != denominator:
            raise ArtifactVerificationError(f"{label}: flat denominator disagrees")
    return value


def _exact_dyadic_from_hex(value: object, label: str) -> Fraction:
    if not isinstance(value, str):
        raise ArtifactVerificationError(f"{label} must be a hexadecimal string")
    match = _HEX_DYADIC_RE.fullmatch(value)
    if match is None:
        raise ArtifactVerificationError(f"{label} is not a hexadecimal dyadic")

    fractional_digits = match.group("fraction") or ""
    significand = int(match.group("integer") + fractional_digits, 16)
    if match.group("sign") == "-":
        significand = -significand
    binary_exponent = int(match.group("exponent")) - 4 * len(fractional_digits)
    if binary_exponent >= 0:
        return Fraction(significand << binary_exponent, 1)
    return Fraction(significand, 1 << -binary_exponent)


def _verify_exact_endpoint_nondegeneracy(
    record: Mapping[str, Any],
    expected_key: tuple[int, int],
    edge_list: list[Any],
) -> Fraction:
    N = expected_key[0]
    weight_hex = record.get("edge_weight_hex")
    if not isinstance(weight_hex, list) or len(weight_hex) != len(edge_list):
        raise ArtifactVerificationError(
            f"graph {expected_key}: edge_weight_hex length mismatch"
        )

    exact_weights = [
        _exact_dyadic_from_hex(value, f"graph {expected_key} edge {index} weight")
        for index, value in enumerate(weight_hex)
    ]
    common_denominator = max(weight.denominator for weight in exact_weights)
    integer_edges = [
        (
            int(edge[0]),
            int(edge[1]),
            weight.numerator * (common_denominator // weight.denominator),
        )
        for edge, weight in zip(edge_list, exact_weights)
    ]

    first_cost: int | None = None
    second_cost: int | None = None
    for state in range(1 << N):
        cost = 0
        for u, v, weight in integer_edges:
            u_bit = 0 if u == 0 else (state >> (u - 1)) & 1
            v_bit = (state >> (v - 1)) & 1
            if u_bit == v_bit:
                cost += weight
        if first_cost is None or cost < first_cost:
            second_cost = first_cost
            first_cost = cost
        elif second_cost is None or cost < second_cost:
            second_cost = cost

    if first_cost is None or second_cost is None:
        raise ArtifactVerificationError(
            f"graph {expected_key}: endpoint cost ordering is incomplete"
        )
    if second_cost <= first_cost:
        ordered_cost = Fraction(first_cost, common_denominator)
        raise ArtifactVerificationError(
            f"graph {expected_key}: exact pinned endpoint minimum is not unique; "
            f"the first two ordered costs are both {ordered_cost}"
        )
    exact_gap = Fraction(second_cost - first_cost, common_denominator)
    if exact_gap <= 0:
        raise ArtifactVerificationError(
            f"graph {expected_key}: exact pinned endpoint gap is not positive"
        )
    return exact_gap


def _verify_graph_record_object(
    record: Mapping[str, Any],
    expected_key: tuple[int, int],
) -> None:
    N, seed = expected_key
    if N not in EXPECTED_NS or seed not in EXPECTED_SEEDS:
        raise ArtifactVerificationError(f"graph {expected_key}: unexpected cohort key")
    if record.get("schema_version") != EXPECTED_GRAPH_SCHEMA_VERSION:
        raise ArtifactVerificationError(
            f"graph {expected_key}: expected graph schema "
            f"{EXPECTED_GRAPH_SCHEMA_VERSION}"
        )
    if record.get("record_type") != "generated_weighted_graph_instance":
        raise ArtifactVerificationError(
            f"graph {expected_key}: unexpected record_type"
        )
    if record.get("row_key") != {"N": N, "seed": seed}:
        raise ArtifactVerificationError(
            f"graph {expected_key}: row_key mismatch"
        )
    if record.get("N_effective_qubits") != N:
        raise ArtifactVerificationError(
            f"graph {expected_key}: effective-qubit metadata mismatch"
        )
    n_vertices = N + 1
    if record.get("n_vertices") != n_vertices:
        raise ArtifactVerificationError(
            f"graph {expected_key}: vertex-count metadata mismatch"
        )

    graph_model = record.get("graph_model")
    expected_graph_model = {
        "name": "Erdos-Renyi",
        "p": 0.5,
        "p_hex": "0x1.0000000000000p-1",
        "edge_weight_distribution": "independent uniform [0.5, 1.5)",
        "acceptance_conditions": [
            "connected",
            "nondegenerate pinned optimum",
        ],
    }
    if graph_model != expected_graph_model:
        raise ArtifactVerificationError(
            f"graph {expected_key}: graph-model metadata mismatch"
        )
    generation = record.get("generation")
    if not isinstance(generation, dict):
        raise ArtifactVerificationError(
            f"graph {expected_key}: generation metadata is missing"
        )
    if generation.get("seed") != seed:
        raise ArtifactVerificationError(
            f"graph {expected_key}: generation seed mismatch"
        )
    attempts = generation.get("attempts")
    if (
        not isinstance(attempts, int)
        or isinstance(attempts, bool)
        or attempts <= 0
    ):
        raise ArtifactVerificationError(
            f"graph {expected_key}: invalid generation-attempt metadata"
        )
    rng_name = generation.get("rng_bit_generator")
    if not isinstance(rng_name, str) or not rng_name:
        raise ArtifactVerificationError(
            f"graph {expected_key}: invalid RNG metadata"
        )

    expected_record_hash = _require_sha256(
        record.get("graph_record_sha256"),
        f"graph {expected_key} record hash",
    )
    hash_input = {
        key: value
        for key, value in record.items()
        if key != "graph_record_sha256"
    }
    if _sha256_bytes(_canonical_json_bytes(hash_input)) != expected_record_hash:
        raise ArtifactVerificationError(
            f"graph {expected_key}: graph_record_sha256 mismatch"
        )

    edge_list = record.get("edge_list")
    if not isinstance(edge_list, list):
        raise ArtifactVerificationError(f"graph {expected_key}: edge_list is missing")
    edge_hash = _require_sha256(
        record.get("edge_payload_sha256"),
        f"graph {expected_key} edge hash",
    )
    if _sha256_bytes(_canonical_json_bytes(edge_list)) != edge_hash:
        raise ArtifactVerificationError(
            f"graph {expected_key}: edge_payload_sha256 mismatch"
        )
    if record.get("num_edges") != len(edge_list):
        raise ArtifactVerificationError(f"graph {expected_key}: edge count mismatch")

    weights = []
    edge_pairs: set[tuple[int, int]] = set()
    adjacency = [[] for _ in range(n_vertices)]
    for edge_index, edge in enumerate(edge_list):
        if not isinstance(edge, list) or len(edge) != 3:
            raise ArtifactVerificationError(
                f"graph {expected_key}: malformed edge {edge_index}"
            )
        u, v, weight = edge
        if (
            not isinstance(u, int)
            or isinstance(u, bool)
            or not isinstance(v, int)
            or isinstance(v, bool)
            or not 0 <= u < v < n_vertices
        ):
            raise ArtifactVerificationError(
                f"graph {expected_key}: malformed edge endpoints at {edge_index}"
            )
        if (u, v) in edge_pairs:
            raise ArtifactVerificationError(
                f"graph {expected_key}: duplicate edge endpoints at {edge_index}"
            )
        edge_pairs.add((u, v))
        adjacency[u].append(v)
        adjacency[v].append(u)
        parsed_weight = float(weight)
        if not math.isfinite(parsed_weight) or not 0.5 <= parsed_weight < 1.5:
            raise ArtifactVerificationError(
                f"graph {expected_key}: invalid edge weight at {edge_index}"
            )
        weights.append(parsed_weight)

    reached = {0}
    pending = [0]
    while pending:
        vertex = pending.pop()
        for neighbor in adjacency[vertex]:
            if neighbor not in reached:
                reached.add(neighbor)
                pending.append(neighbor)
    if len(reached) != n_vertices:
        raise ArtifactVerificationError(
            f"graph {expected_key}: graph is not connected as metadata claims"
        )

    expected_hex = [weight.hex() for weight in weights]
    if record.get("edge_weight_hex") != expected_hex:
        raise ArtifactVerificationError(
            f"graph {expected_key}: exact hexadecimal weights disagree"
        )
    if record.get("edge_weight_semantics") != (
        "archived binary64 values interpreted as exact dyadic rationals"
    ):
        raise ArtifactVerificationError(
            f"graph {expected_key}: exact edge-weight semantics mismatch"
        )
    _verify_exact_endpoint_nondegeneracy(record, expected_key, edge_list)

    W_model = float(sum(weights))
    W_exact = exact_binary64_sum(weights)
    W_upper = least_binary64_not_below(W_exact)
    if _as_float(record.get("edge_weights_sum"), "edge_weights_sum") != W_model:
        raise ArtifactVerificationError(
            f"graph {expected_key}: edge_weights_sum must alias W_model"
        )
    if record.get("edge_weights_sum_hex") != W_model.hex():
        raise ArtifactVerificationError(
            f"graph {expected_key}: nominal edge-weight sum hex disagrees"
        )
    if _as_float(record.get("W_model"), "W_model") != W_model:
        raise ArtifactVerificationError(f"graph {expected_key}: W_model disagrees")
    if record.get("W_model_hex") != W_model.hex():
        raise ArtifactVerificationError(f"graph {expected_key}: W_model hex disagrees")
    recorded_W_exact = _record_fraction(
        record, "W_exact", "W_exact", f"graph {expected_key} W_exact"
    )
    if recorded_W_exact != W_exact:
        raise ArtifactVerificationError(f"graph {expected_key}: W_exact disagrees")
    recorded_W_upper = _as_float(record.get("W_upper"), "W_upper")
    if recorded_W_upper != W_upper or record.get("W_upper_hex") != W_upper.hex():
        raise ArtifactVerificationError(
            f"graph {expected_key}: W_upper is not the least binary64 upper bound"
        )
    W_upper_exact = _record_fraction(
        record, "W_upper_exact", "W_upper", f"graph {expected_key} W_upper"
    )
    if W_upper_exact != Fraction.from_float(W_upper):
        raise ArtifactVerificationError(
            f"graph {expected_key}: W_upper rational metadata disagrees"
        )

    lambda_i = _as_int(record.get("Lambda_I"), "Lambda_I")
    if lambda_i != 2 * (n_vertices // 2):
        raise ArtifactVerificationError(
            f"graph {expected_key}: Lambda_I metadata mismatch"
        )
    K_gap_exact = _record_fraction(
        record, "K_gap_exact", "K_gap_exact", f"graph {expected_key} K_gap_exact"
    )
    if K_gap_exact != W_exact + lambda_i:
        raise ArtifactVerificationError(
            f"graph {expected_key}: K_gap_exact disagrees"
        )
    K_gap_from_upper = _record_fraction(
        record,
        "K_gap_from_W_upper_exact",
        None,
        f"graph {expected_key} exact W_upper+Lambda_I",
    )
    expected_K_input = W_upper_exact + lambda_i
    if K_gap_from_upper != expected_K_input:
        raise ArtifactVerificationError(
            f"graph {expected_key}: exact W_upper+Lambda_I disagrees"
        )
    K_gap_cert = least_binary64_not_below(expected_K_input)
    if _as_float(record.get("K_gap_cert"), "K_gap_cert") != K_gap_cert:
        raise ArtifactVerificationError(
            f"graph {expected_key}: K_gap_cert upward rounding disagrees"
        )
    if record.get("K_gap_cert_hex") != K_gap_cert.hex():
        raise ArtifactVerificationError(
            f"graph {expected_key}: K_gap_cert hex disagrees"
        )
    K_gap_cert_exact = _record_fraction(
        record,
        "K_gap_cert_exact",
        "K_gap_cert",
        f"graph {expected_key} K_gap_cert",
    )
    if K_gap_cert_exact != Fraction.from_float(K_gap_cert):
        raise ArtifactVerificationError(
            f"graph {expected_key}: K_gap_cert rational metadata disagrees"
        )


def _verify_graph_archive(
    rows: list[dict[str, Any]], archive_path: Path
) -> tuple[dict[tuple[int, int], dict[str, Any]], set[Path]]:
    try:
        archive_payload = archive_path.read_bytes()
    except OSError as exc:
        raise ArtifactVerificationError(f"cannot read graph archive: {exc}") from exc
    if not archive_payload.endswith(b"\n"):
        raise ArtifactVerificationError("graph archive must end with one LF")
    lines = archive_payload.splitlines()
    if len(lines) != EXPECTED_ROW_COUNT:
        raise ArtifactVerificationError(
            f"graph archive must contain {EXPECTED_ROW_COUNT} lines, "
            f"found {len(lines)}"
        )

    records: dict[tuple[int, int], dict[str, Any]] = {}
    canonical_payload_parts = []
    for line_number, line in enumerate(lines, start=1):
        try:
            record = json.loads(line.decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ArtifactVerificationError(
                f"invalid graph archive JSON on line {line_number}: {exc}"
            ) from exc
        if not isinstance(record, dict):
            raise ArtifactVerificationError(
                f"graph archive line {line_number} must be an object"
            )
        row_key = record.get("row_key", {})
        try:
            key = (int(row_key["N"]), int(row_key["seed"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactVerificationError(
                f"graph archive line {line_number} has no valid row_key"
            ) from exc
        if key in records:
            raise ArtifactVerificationError(f"duplicate graph archive key {key}")
        _verify_graph_record_object(record, key)
        records[key] = record
        canonical_payload_parts.append(_canonical_json_bytes(record) + b"\n")

    expected_keys = {
        (N, seed) for N in EXPECTED_NS for seed in EXPECTED_SEEDS
    }
    if set(records) != expected_keys:
        raise ArtifactVerificationError(
            f"graph archive cohort is not N={EXPECTED_NS}, seeds 0-19"
        )
    if archive_payload != b"".join(canonical_payload_parts):
        raise ArtifactVerificationError(
            "graph archive is not in canonical sorted-key JSONL form"
        )
    if list(records) != sorted(records):
        raise ArtifactVerificationError("graph archive records are not sorted by (N, seed)")

    row_by_key = {(int(row["N"]), int(row["seed"])): row for row in rows}
    for key, record in records.items():
        row = row_by_key[key]
        if row["graph_record_sha256"] != record["graph_record_sha256"]:
            raise ArtifactVerificationError(f"graph {key}: CSV record hash mismatch")
        if row["edge_payload_sha256"] != record["edge_payload_sha256"]:
            raise ArtifactVerificationError(f"graph {key}: CSV edge hash mismatch")
        if int(row["m"]) != int(record["num_edges"]):
            raise ArtifactVerificationError(f"graph {key}: CSV edge count mismatch")
        if int(row["n_vertices"]) != int(record["n_vertices"]):
            raise ArtifactVerificationError(f"graph {key}: CSV vertex count mismatch")
        if int(row["Lambda_I"]) != int(record["Lambda_I"]):
            raise ArtifactVerificationError(f"graph {key}: CSV Lambda_I mismatch")
        if not bool(row["connected"]):
            raise ArtifactVerificationError(f"graph {key}: CSV connectivity mismatch")
        if int(row["graph_schema_version"]) != EXPECTED_GRAPH_SCHEMA_VERSION:
            raise ArtifactVerificationError(f"graph {key}: CSV graph schema mismatch")
        graph_model = record["graph_model"]
        generation = record["generation"]
        if row["graph_model"] != graph_model["name"]:
            raise ArtifactVerificationError(f"graph {key}: CSV graph model mismatch")
        if float(row["graph_p"]) != float(graph_model["p"]):
            raise ArtifactVerificationError(f"graph {key}: CSV graph p mismatch")
        if int(row["graph_generation_seed"]) != int(generation["seed"]):
            raise ArtifactVerificationError(
                f"graph {key}: CSV generation seed mismatch"
            )
        for field in ("W_model", "W_upper", "K_gap_cert"):
            if float(row[field]) != float(record[field]):
                raise ArtifactVerificationError(
                    f"graph {key}: CSV and graph {field} disagree"
                )
        if float(row["W"]) != float(record["W_upper"]):
            raise ArtifactVerificationError(
                f"graph {key}: CSV W does not alias graph W_upper"
            )
        rational_links = (
            ("W_exact", "W_exact", "W_exact"),
            ("W_upper", "W_upper_exact", "W_upper"),
            ("K_gap_exact", "K_gap_exact", "K_gap_exact"),
            (
                "K_gap_from_W_upper_exact",
                "K_gap_from_W_upper_exact",
                None,
            ),
            ("K_gap_cert", "K_gap_cert_exact", "K_gap_cert"),
        )
        for row_prefix, nested_key, flat_prefix in rational_links:
            graph_fraction = _record_fraction(
                record, nested_key, flat_prefix, f"graph {key} {row_prefix}"
            )
            row_fraction = Fraction(
                int(row[f"{row_prefix}_numerator"]),
                int(row[f"{row_prefix}_denominator"]),
            )
            if graph_fraction != row_fraction:
                raise ArtifactVerificationError(
                    f"graph {key}: CSV and graph {row_prefix} disagree"
                )
        if int(row["graph_generation_attempts"]) != int(
            record["generation"]["attempts"]
        ):
            raise ArtifactVerificationError(
                f"graph {key}: generation-attempt count mismatch"
            )
        if row["rng_bit_generator"] != record["generation"]["rng_bit_generator"]:
            raise ArtifactVerificationError(f"graph {key}: RNG metadata mismatch")

        individual_path = _project_path(
            row["graph_record_path"], f"graph {key} record path"
        )
        individual = _load_json(individual_path, f"graph {key} record")
        if individual != record:
            raise ArtifactVerificationError(
                f"graph {key}: individual record differs from JSONL archive"
            )
    return records


def _verify_summary() -> None:
    _, expected_json, expected_tex = expected_outputs(RESULTS_PATH)
    for path, expected in (
        (OUTPUT_PATH, expected_json),
        (TEX_OUTPUT_PATH, expected_tex),
    ):
        if not path.is_file():
            raise ArtifactVerificationError(
                f"missing generated summary: {path.relative_to(PROJECT_ROOT)}"
            )
        if path.read_bytes() != expected:
            raise ArtifactVerificationError(
                f"stale generated summary: {path.relative_to(PROJECT_ROOT)}; "
                "run `python summarize_section7_results.py`"
            )


def _merged_windows(intervals: Iterable[Mapping[str, Any]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for interval in intervals:
        if bool(interval["is_conditionally_resolved"]):
            continue
        start = float(interval["s_start"])
        end = float(interval["s_end"])
        if merged and start <= merged[-1][1] + _FLOAT_ABS_TOL:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _verify_conditional_log(
    rows: list[dict[str, Any]],
    graph_records: Mapping[tuple[int, int], Mapping[str, Any]],
) -> None:
    selected = [row for row in rows if row["conditional_log_path"]]
    selected_keys = {(int(row["N"]), int(row["seed"])) for row in selected}
    if selected_keys != {(10, 0)} or len(selected) != 1:
        raise ArtifactVerificationError(
            "expected exactly the selected N=10, seed-0 Level-2 conditional log"
        )
    row = selected[0]
    key = (10, 0)
    path = _project_path(row["conditional_log_path"], "conditional log path")
    log = _load_json(path, "selected conditional log")
    if not isinstance(log, dict):
        raise ArtifactVerificationError("selected conditional log must be an object")
    if log.get("schema_version") != EXPECTED_SCHEMA_VERSION:
        raise ArtifactVerificationError(
            f"selected conditional log must use schema {EXPECTED_SCHEMA_VERSION}"
        )
    if log.get("record_level") != "Level-2 conditional numerical diagnostic":
        raise ArtifactVerificationError("selected log is not identified as Level-2")
    if log.get("N_effective_qubits") != 10 or log.get("graph_record_sha256") != row[
        "graph_record_sha256"
    ]:
        raise ArtifactVerificationError("selected log graph identity mismatch")
    if log.get("edge_payload_sha256") != row["edge_payload_sha256"]:
        raise ArtifactVerificationError("selected log edge hash mismatch")
    if graph_records[key]["graph_record_sha256"] != log["graph_record_sha256"]:
        raise ArtifactVerificationError("selected log and graph archive disagree")

    for field in ("W_model", "W_upper", "K_gap_cert"):
        if _as_float(log.get(field), f"conditional log {field}") != float(row[field]):
            raise ArtifactVerificationError(
                f"selected log and CSV {field} disagree"
            )
    for field in ("W_model", "W_upper", "K_gap_cert"):
        if log.get(f"{field}_hex") != float(row[field]).hex():
            raise ArtifactVerificationError(
                f"selected log {field} hexadecimal metadata disagrees"
            )
    rational_links = (
        ("W_exact", "W_exact", "W_exact"),
        ("W_upper", "W_upper_exact", "W_upper"),
        ("K_gap_exact", "K_gap_exact", "K_gap_exact"),
        (
            "K_gap_from_W_upper_exact",
            "K_gap_from_W_upper_exact",
            None,
        ),
        ("K_gap_cert", "K_gap_cert_exact", "K_gap_cert"),
    )
    for row_prefix, nested_key, flat_prefix in rational_links:
        log_fraction = _record_fraction(
            log, nested_key, flat_prefix, f"conditional log {row_prefix}"
        )
        row_fraction = Fraction(
            int(row[f"{row_prefix}_numerator"]),
            int(row[f"{row_prefix}_denominator"]),
        )
        if log_fraction != row_fraction:
            raise ArtifactVerificationError(
                f"selected log and CSV {row_prefix} disagree"
            )

    K_gap_cert = _as_float(log.get("K_gap_cert"), "conditional log K_gap_cert")
    eta = _as_float(log.get("eta"), "conditional log eta")
    h_floor = _as_float(log.get("h_floor"), "conditional log h_floor")
    if K_gap_cert != float(row["K_gap_cert"]):
        raise ArtifactVerificationError("conditional log K_gap_cert differs from CSV")
    _require_close(eta, row["eta"], "conditional log eta")
    _require_close(h_floor, row["h_floor"], "conditional log h_floor")
    if K_gap_cert < 0.0 or not 0.0 < eta < 1.0 or h_floor <= 0.0:
        raise ArtifactVerificationError("invalid continuation parameters in selected log")

    strategy = log.get("unresolved_probe_strategy")
    if not isinstance(strategy, dict):
        raise ArtifactVerificationError("selected log has no probe strategy")
    probe_initial = _as_float(strategy.get("initial"), "probe initial")
    probe_growth = _as_float(strategy.get("growth_factor"), "probe growth")
    probe_maximum = _as_float(strategy.get("maximum"), "probe maximum")
    if probe_initial <= 0.0 or probe_growth != 2.0 or probe_maximum <= 0.0:
        raise ArtifactVerificationError("invalid probe strategy in selected log")

    intervals = log.get("conditional_intervals")
    if not isinstance(intervals, list) or not intervals:
        raise ArtifactVerificationError("selected log has no conditional intervals")
    previous_end = 0.0
    floor_streak = 0
    for index, interval in enumerate(intervals):
        if not isinstance(interval, dict):
            raise ArtifactVerificationError(f"interval {index} is not an object")
        prefix = f"conditional interval {index}"
        start = _as_float(interval.get("s_start"), f"{prefix} s_start")
        end = _as_float(interval.get("s_end"), f"{prefix} s_end")
        delta_lo = _as_float(interval.get("delta_lo"), f"{prefix} delta_lo")
        theta0 = _as_float(interval.get("theta0"), f"{prefix} theta0")
        theta1 = _as_float(interval.get("theta1"), f"{prefix} theta1")
        r0 = _as_float(interval.get("r0"), f"{prefix} r0")
        r1 = _as_float(interval.get("r1"), f"{prefix} r1")
        h_actual = _as_float(interval.get("h_actual"), f"{prefix} h_actual")
        remaining = 1.0 - start
        for value, label in (
            (start, "s_start"),
            (end, "s_end"),
            (delta_lo, "delta_lo"),
            (theta0, "theta0"),
            (theta1, "theta1"),
            (r0, "r0"),
            (r1, "r1"),
            (h_actual, "h_actual"),
        ):
            if not math.isfinite(value):
                raise ArtifactVerificationError(f"{prefix} {label} is not finite")
        _require_close(start, previous_end, f"{prefix} continuity")
        if not 0.0 <= start < end <= 1.0:
            raise ArtifactVerificationError(f"{prefix} has invalid endpoints")
        _require_close(h_actual, end - start, f"{prefix} actual length")
        _require_close(interval.get("h_step"), h_actual, f"{prefix} h_step")
        _require_close(interval.get("K_gap_cert"), K_gap_cert, f"{prefix} K_gap")
        _require_close(
            delta_lo,
            max(0.0, (theta1 - r1) - theta0),
            f"{prefix} conditional anchor algebra",
        )
        residuals = interval.get("residuals")
        if not isinstance(residuals, list) or len(residuals) != 2:
            raise ArtifactVerificationError(f"{prefix} residual list is malformed")
        _require_close(residuals[0], r0, f"{prefix} r0 alias")
        _require_close(residuals[1], r1, f"{prefix} r1 alias")

        if K_gap_cert == 0.0:
            h_proposed = remaining
            h_floor_overridden = remaining
            h_probe_planned = remaining
            floor_override = False
            expected_mode = "constant_gap"
            floor_streak = 0
        else:
            h_proposed = eta * delta_lo / K_gap_cert
            floor_override = h_proposed < h_floor
            h_floor_overridden = h_floor if floor_override else h_proposed
            if floor_override:
                initial = max(h_floor, probe_initial)
                cap = max(h_floor, probe_maximum)
                h_probe_planned = min(
                    cap, initial * (probe_growth ** min(floor_streak, 30))
                )
                floor_streak += 1
                expected_mode = "floor_mark_and_probe"
            else:
                h_probe_planned = h_proposed
                floor_streak = 0
                expected_mode = "adaptive"

        _require_close(interval.get("h_proposed"), h_proposed, f"{prefix} proposal")
        _require_close(
            interval.get("h_floor_overridden"),
            h_floor_overridden,
            f"{prefix} floor-adjusted length",
        )
        _require_close(
            interval.get("h_probe_planned"),
            h_probe_planned,
            f"{prefix} planned probe",
        )
        if bool(interval.get("floor_override_applied")) != floor_override:
            raise ArtifactVerificationError(f"{prefix} floor-override flag mismatch")
        if interval.get("step_mode") != expected_mode:
            raise ArtifactVerificationError(f"{prefix} step_mode mismatch")
        endpoint_clip = h_probe_planned > remaining
        if bool(interval.get("endpoint_clip_applied")) != endpoint_clip:
            raise ArtifactVerificationError(f"{prefix} endpoint-clip flag mismatch")
        _require_close(
            end,
            1.0 if h_probe_planned >= remaining else start + h_probe_planned,
            f"{prefix} endpoint",
        )

        expected_bound = delta_lo - K_gap_cert * h_actual
        _require_close(interval.get("B_k"), expected_bound, f"{prefix} B_k")
        resolved = expected_bound > 0.0
        if bool(interval.get("is_conditionally_resolved")) != resolved:
            raise ArtifactVerificationError(f"{prefix} resolution flag mismatch")
        previous_end = end

    _require_close(previous_end, 1.0, "conditional interval coverage")
    if _as_int(log.get("num_anchor_points"), "conditional-log anchor count") != len(intervals):
        raise ArtifactVerificationError("conditional-log anchor count mismatch")
    if int(row["solves_hybrid"]) != len(intervals):
        raise ArtifactVerificationError("CSV and conditional-log solve counts disagree")

    expected_windows = _merged_windows(intervals)
    logged_windows = log.get("unresolved_windows")
    if not isinstance(logged_windows, list) or len(logged_windows) != len(
        expected_windows
    ):
        raise ArtifactVerificationError("conditional-log unresolved-window count mismatch")
    for index, (logged, expected) in enumerate(zip(logged_windows, expected_windows)):
        _require_close(logged.get("s_start"), expected[0], f"window {index} start")
        _require_close(logged.get("s_end"), expected[1], f"window {index} end")

    unresolved_width = sum(end - start for start, end in expected_windows)
    resolved_width = sum(
        _as_float(interval["h_actual"], "resolved interval length")
        for interval in intervals
        if bool(interval["is_conditionally_resolved"])
    )
    raw_global = min(
        _as_float(interval["B_k"], "interval bound") for interval in intervals
    )
    resolved_bounds = [
        _as_float(interval["B_k"], "resolved interval bound")
        for interval in intervals
        if bool(interval["is_conditionally_resolved"])
    ]
    _require_close(
        log.get("unresolved_fraction"),
        unresolved_width,
        "conditional-log unresolved fraction",
    )
    _require_close(
        log.get("conditionally_resolved_fraction"),
        resolved_width,
        "conditional-log resolved fraction",
    )
    _require_close(
        log.get("global_conditional_gap_lb_raw"),
        raw_global,
        "conditional-log global raw bound",
    )
    _require_close(
        log.get("global_conditional_gap_lb"),
        max(0.0, raw_global),
        "conditional-log global nonnegative bound",
    )
    if resolved_bounds:
        _require_close(
            log.get("resolved_interval_conditional_gap_lb_min"),
            min(resolved_bounds),
            "conditional-log minimum resolved bound",
        )
    _require_close(row["unresolved_frac"], unresolved_width, "CSV unresolved fraction")
    if int(row["num_windows"]) != len(expected_windows):
        raise ArtifactVerificationError("CSV unresolved-window count mismatch")

    repository = log.get("repository")
    if not isinstance(repository, dict):
        raise ArtifactVerificationError("conditional log has no repository metadata")
    source_hashes = repository.get("source_sha256")
    if not isinstance(source_hashes, dict):
        raise ArtifactVerificationError("conditional log has no source hashes")
    if source_hashes.get("main.py") != row["source_main_sha256"]:
        raise ArtifactVerificationError("conditional-log main.py hash mismatch")
    if source_hashes.get("maxcut_gap_benchmark.py") != row["source_driver_sha256"]:
        raise ArtifactVerificationError("conditional-log driver hash mismatch")


def _png_text_metadata(path: Path) -> dict[str, str]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ArtifactVerificationError(f"cannot read required figure {path}: {exc}") from exc
    if len(payload) < 24 or payload[:8] != _PNG_SIGNATURE or payload[12:16] != b"IHDR":
        raise ArtifactVerificationError(
            f"required figure is not a valid PNG: {path.relative_to(PROJECT_ROOT)}"
        )
    width, height = struct.unpack(">II", payload[16:24])
    if width == 0 or height == 0:
        raise ArtifactVerificationError(
            f"required figure has zero dimensions: {path.relative_to(PROJECT_ROOT)}"
        )

    metadata: dict[str, str] = {}
    offset = len(_PNG_SIGNATURE)
    saw_iend = False
    while offset + 12 <= len(payload):
        length = struct.unpack(">I", payload[offset : offset + 4])[0]
        chunk_type = payload[offset + 4 : offset + 8]
        data_start = offset + 8
        data_end = data_start + length
        crc_end = data_end + 4
        if crc_end > len(payload):
            raise ArtifactVerificationError(
                f"truncated PNG chunk in {path.relative_to(PROJECT_ROOT)}"
            )
        chunk_data = payload[data_start:data_end]
        expected_crc = struct.unpack(">I", payload[data_end:crc_end])[0]
        actual_crc = zlib.crc32(chunk_type)
        actual_crc = zlib.crc32(chunk_data, actual_crc) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            raise ArtifactVerificationError(
                f"PNG CRC mismatch in {path.relative_to(PROJECT_ROOT)}"
            )
        if chunk_type == b"tEXt" and b"\x00" in chunk_data:
            keyword, value = chunk_data.split(b"\x00", 1)
            metadata[keyword.decode("latin-1")] = value.decode("latin-1")
        elif chunk_type == b"iTXt" and b"\x00" in chunk_data:
            keyword, remainder = chunk_data.split(b"\x00", 1)
            if len(remainder) >= 2 and remainder[0] == 0:
                parts = remainder[2:].split(b"\x00", 2)
                if len(parts) == 3:
                    metadata[keyword.decode("latin-1")] = parts[2].decode("utf-8")
        if chunk_type == b"IEND":
            saw_iend = True
            if crc_end != len(payload):
                raise ArtifactVerificationError(
                    f"trailing bytes after PNG IEND in {path.relative_to(PROJECT_ROOT)}"
                )
            break
        offset = crc_end
    if not saw_iend:
        raise ArtifactVerificationError(
            f"PNG has no IEND chunk: {path.relative_to(PROJECT_ROOT)}"
        )
    return metadata


def _verify_required_figures(rows: list[dict[str, Any]]) -> None:
    csv_sha256 = sha256_path(RESULTS_PATH)
    graph_hashes = {
        (int(row["N"]), int(row["seed"])): str(row["graph_record_sha256"])
        for row in rows
    }
    graph_keys = {
        "comparison_updated_N10_no_anchors.png": (10, 0),
        "comparison_updated_N10_anchors.png": (10, 0),
        "grid_density_vs_gap.png": (10, 0),
        "grid_density_vs_gap_N12.png": (12, 0),
        "grid_density_vs_gap_N14.png": (14, 0),
    }
    for filename in _REQUIRED_FIGURES:
        metadata = _png_text_metadata(RESULTS_DIR / filename)
        if metadata.get("Artifact") != filename:
            raise ArtifactVerificationError(
                f"figure {filename} has no matching Artifact metadata"
            )
        if metadata.get("Section7CSV-SHA256") != csv_sha256:
            raise ArtifactVerificationError(
                f"figure {filename} was not generated from the current CSV"
            )
        graph_key = graph_keys.get(filename)
        if graph_key is not None and metadata.get("GraphRecord-SHA256") != graph_hashes[
            graph_key
        ]:
            raise ArtifactVerificationError(
                f"figure {filename} was not generated from graph {graph_key}"
            )


def verify() -> int:
    rows = load_rows(RESULTS_PATH)
    _, archive_path = _verify_run_manifest(rows)
    graph_records = _verify_graph_archive(rows, archive_path)
    _verify_summary()
    _verify_conditional_log(rows, graph_records)
    _verify_required_figures(rows)

    print("Verified schema-v4 CSV cohort and run manifest.")
    print(
        f"Verified all {EXPECTED_ROW_COUNT} graph-schema-2 records, metadata, exact dyadic "
        "rounding, and SHA-256 links."
    )
    print(
        "Verified exact endpoint nondegeneracy for all "
        f"{EXPECTED_ROW_COUNT} graph-schema-2 records."
    )
    print("Verified deterministic JSON/TeX summary regeneration.")
    print("Verified selected N=10, seed-0 Level-2 continuation algebra.")
    print(f"Verified {len(_REQUIRED_FIGURES)} required manuscript figures.")
    if (RESULTS_DIR / "runtime_scaling.png").exists():
        print(
            "Warning: legacy results/runtime_scaling.png is present but is not "
            "generated or treated as a manuscript artifact.",
            file=sys.stderr,
        )
    return 0


def main() -> int:
    try:
        return verify()
    except (ArtifactValidationError, ArtifactVerificationError) as exc:
        print(f"artifact verification failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
