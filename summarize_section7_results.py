from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_PATH = PROJECT_ROOT / "section7_results.csv"
OUTPUT_PATH = PROJECT_ROOT / "results" / "section7_summary.json"
TEX_OUTPUT_PATH = PROJECT_ROOT / "results" / "section7_summary.tex"
EXPECTED_SCHEMA_VERSION = 4
EXPECTED_GRAPH_SCHEMA_VERSION = 2
EXPECTED_NS = (10, 12)
EXPECTED_SEEDS = tuple(range(20))
EXPECTED_ROW_COUNT = len(EXPECTED_NS) * len(EXPECTED_SEEDS)
_FLOAT_REL_TOL = 1e-12
_FLOAT_ABS_TOL = 1e-12
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ArtifactValidationError(ValueError):
    """Raised when a Section 7 artifact violates the schema-v4 contract."""


_REQUIRED_COLUMNS = {
    "schema_version",
    "N",
    "n_vertices",
    "m",
    "seed",
    "connected",
    "W",
    "W_semantics",
    "W_model",
    "W_model_hex",
    "W_exact_numerator",
    "W_exact_denominator",
    "W_upper",
    "W_upper_hex",
    "W_upper_numerator",
    "W_upper_denominator",
    "Lambda_I",
    "K_gap_exact_numerator",
    "K_gap_exact_denominator",
    "K_gap_from_W_upper_exact_numerator",
    "K_gap_from_W_upper_exact_denominator",
    "K_gap_cert",
    "K_gap_cert_hex",
    "K_gap_cert_numerator",
    "K_gap_cert_denominator",
    "K_gap_cert_derivation",
    "D_norm_bound",
    "D_norm_exact_numerator",
    "D_norm_exact_denominator",
    "D_norm_bound_derivation",
    "D_lambda_min_est",
    "D_lambda_max_est",
    "D_spectral_diameter_est",
    "D_norm_est",
    "K_gap_cert_over_diameter_est",
    "diagnostic_matvecs",
    "E0P_model",
    "E1P_model",
    "delta_target",
    "eta",
    "h_floor",
    "unresolved_probe_initial",
    "unresolved_probe_max",
    "dmin_est",
    "s_star",
    "anchor_ritz_gap_min_est",
    "anchor_ritz_gap_s_min_est",
    "frac_weyl",
    "frac_floor",
    "frac_floor_poly",
    "weyl0_frac_pos",
    "psd_frac_pos",
    "solves_hybrid",
    "conditional_oracle_calls",
    "weylA_oracle_calls",
    "solves_uniform",
    "uniform_solves",
    "target_conditionally_resolved_frac",
    "positive_conditionally_resolved_frac",
    "unresolved_frac",
    "num_windows",
    "total_window_width",
    "conditional_interval_coverage",
    "global_conditional_gap_lb_raw",
    "global_conditional_gap_lb",
    "global_conditionally_resolved",
    "conditional_envelope_sampled_min_raw",
    "conditional_envelope_sampled_min_nonnegative",
    "uniform_sampled_target_frac",
    "uniform_sampled_positive_frac",
    "uniform_sampled_unresolved_frac",
    "uniform_target_frac",
    "uniform_positive_frac",
    "uniform_uncert_frac",
    "matvecs",
    "dense_consistency_checked",
    "validated",
    "wall",
    "t_weylA",
    "wall_validation",
    "conditional_log_path",
    "graph_schema_version",
    "graph_model",
    "graph_p",
    "graph_generation_seed",
    "graph_generation_attempts",
    "rng_bit_generator",
    "edge_payload_sha256",
    "graph_record_sha256",
    "graph_record_path",
    "source_main_sha256",
    "source_driver_sha256",
    "graph_archive_path",
    "graph_archive_sha256",
    "graph_archive_records",
}

_INT_FIELDS = {
    "schema_version",
    "N",
    "n_vertices",
    "m",
    "seed",
    "Lambda_I",
    "W_exact_numerator",
    "W_exact_denominator",
    "W_upper_numerator",
    "W_upper_denominator",
    "K_gap_exact_numerator",
    "K_gap_exact_denominator",
    "K_gap_from_W_upper_exact_numerator",
    "K_gap_from_W_upper_exact_denominator",
    "K_gap_cert_numerator",
    "K_gap_cert_denominator",
    "D_norm_exact_numerator",
    "D_norm_exact_denominator",
    "diagnostic_matvecs",
    "solves_hybrid",
    "conditional_oracle_calls",
    "weylA_oracle_calls",
    "solves_uniform",
    "uniform_solves",
    "num_windows",
    "matvecs",
    "graph_schema_version",
    "graph_generation_seed",
    "graph_generation_attempts",
    "graph_archive_records",
}

_FLOAT_FIELDS = {
    "W",
    "W_model",
    "W_upper",
    "K_gap_cert",
    "D_norm_bound",
    "D_lambda_min_est",
    "D_lambda_max_est",
    "D_spectral_diameter_est",
    "D_norm_est",
    "K_gap_cert_over_diameter_est",
    "E0P_model",
    "E1P_model",
    "delta_target",
    "eta",
    "h_floor",
    "unresolved_probe_initial",
    "unresolved_probe_max",
    "dmin_est",
    "s_star",
    "anchor_ritz_gap_min_est",
    "anchor_ritz_gap_s_min_est",
    "frac_weyl",
    "frac_floor",
    "frac_floor_poly",
    "weyl0_frac_pos",
    "psd_frac_pos",
    "target_conditionally_resolved_frac",
    "positive_conditionally_resolved_frac",
    "unresolved_frac",
    "total_window_width",
    "conditional_interval_coverage",
    "global_conditional_gap_lb_raw",
    "global_conditional_gap_lb",
    "conditional_envelope_sampled_min_raw",
    "conditional_envelope_sampled_min_nonnegative",
    "uniform_sampled_target_frac",
    "uniform_sampled_positive_frac",
    "uniform_sampled_unresolved_frac",
    "uniform_target_frac",
    "uniform_positive_frac",
    "uniform_uncert_frac",
    "wall",
    "t_weylA",
    "wall_validation",
    "graph_p",
}

_OPTIONAL_FLOAT_FIELDS = {
    "resolved_interval_conditional_gap_lb_min",
    "sampled_reference_gap_min_est",
    "sampled_reference_s_min_est",
    "sampled_conditional_lb_at_reference_min",
    "sampled_tightness_ratio_at_reference_min",
    "sampled_tightness_ratio_mean",
    "sampled_tightness_ratio_min",
    "sampled_tightness_ratio_max",
    "sampled_additive_slack_mean",
    "sampled_additive_slack_max",
    "global_lb_to_sampled_reference_min_ratio",
    "heuristic_integral_sampled_est",
    "N_heuristic",
}

_BOOL_FIELDS = {
    "connected",
    "global_conditionally_resolved",
    "dense_consistency_checked",
    "validated",
}

_HASH_FIELDS = {
    "edge_payload_sha256",
    "graph_record_sha256",
    "source_main_sha256",
    "source_driver_sha256",
    "graph_archive_sha256",
}

_FRACTION_FIELDS = {
    "frac_weyl",
    "frac_floor",
    "frac_floor_poly",
    "weyl0_frac_pos",
    "psd_frac_pos",
    "target_conditionally_resolved_frac",
    "positive_conditionally_resolved_frac",
    "unresolved_frac",
    "total_window_width",
    "conditional_interval_coverage",
    "uniform_sampled_target_frac",
    "uniform_sampled_positive_frac",
    "uniform_sampled_unresolved_frac",
    "uniform_target_frac",
    "uniform_positive_frac",
    "uniform_uncert_frac",
}


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_binary64_sum(values: Iterable[float]) -> Fraction:
    """Return the exact dyadic sum of finite binary64 values."""
    total = Fraction(0)
    for value in values:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ArtifactValidationError("binary64 values must be finite")
        total += Fraction.from_float(parsed)
    return total


def least_binary64_not_below(exact_value: Fraction) -> float:
    """Return the least finite binary64 value not below ``exact_value``."""
    exact_value = Fraction(exact_value)
    candidate = float(exact_value)
    if not math.isfinite(candidate):
        raise ArtifactValidationError("exact value is outside binary64 range")
    if Fraction.from_float(candidate) < exact_value:
        candidate = math.nextafter(candidate, math.inf)
    predecessor = math.nextafter(candidate, -math.inf)
    if Fraction.from_float(candidate) < exact_value:
        raise ArtifactValidationError("upward binary64 rounding failed")
    if math.isfinite(predecessor) and Fraction.from_float(predecessor) >= exact_value:
        raise ArtifactValidationError("binary64 upper bound is not minimal")
    return candidate


def rational_from_fields(
    row: Mapping[str, Any], prefix: str, label: str
) -> Fraction:
    numerator = int(row[f"{prefix}_numerator"])
    denominator = int(row[f"{prefix}_denominator"])
    if denominator <= 0:
        raise ArtifactValidationError(f"{label}: denominator must be positive")
    value = Fraction(numerator, denominator)
    if value.numerator != numerator or value.denominator != denominator:
        raise ArtifactValidationError(f"{label}: rational metadata is not canonical")
    return value


def _is_close(left: float, right: float) -> bool:
    return math.isclose(
        left,
        right,
        rel_tol=_FLOAT_REL_TOL,
        abs_tol=_FLOAT_ABS_TOL,
    )


def _parse_int(value: object, field: str, row_number: int) -> int:
    try:
        parsed = int(str(value))
    except (TypeError, ValueError) as exc:
        raise ArtifactValidationError(
            f"row {row_number}: {field} must be an integer"
        ) from exc
    return parsed


def _parse_float(value: object, field: str, row_number: int) -> float:
    try:
        parsed = float(str(value))
    except (TypeError, ValueError) as exc:
        raise ArtifactValidationError(
            f"row {row_number}: {field} must be numeric"
        ) from exc
    if not math.isfinite(parsed):
        raise ArtifactValidationError(
            f"row {row_number}: {field} must be finite"
        )
    return parsed


def _parse_bool(value: object, field: str, row_number: int) -> bool:
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ArtifactValidationError(
        f"row {row_number}: {field} must be True or False"
    )


def _require_columns(fieldnames: Iterable[str] | None) -> None:
    names = set(fieldnames or ())
    if "schema_version" not in names:
        raise ArtifactValidationError(
            "section7_results.csv is a legacy archive without schema_version; "
            "expected schema 4. Regenerate it with the current main.py"
        )
    missing = sorted(_REQUIRED_COLUMNS - names)
    if missing:
        raise ArtifactValidationError(
            "section7_results.csv is missing schema-v4 columns: "
            + ", ".join(missing)
        )


def _normalize_row(
    row: Mapping[str, Any], row_number: int
) -> dict[str, Any]:
    normalized: dict[str, Any] = dict(row)
    for field in _INT_FIELDS:
        normalized[field] = _parse_int(row.get(field), field, row_number)
    for field in _FLOAT_FIELDS:
        normalized[field] = _parse_float(row.get(field), field, row_number)
    for field in _BOOL_FIELDS:
        normalized[field] = _parse_bool(row.get(field), field, row_number)
    for field in _OPTIONAL_FLOAT_FIELDS:
        value = row.get(field)
        if value is None or str(value).strip() == "":
            normalized[field] = None
        else:
            normalized[field] = _parse_float(value, field, row_number)
    for field in _HASH_FIELDS:
        value = str(row.get(field, "")).strip()
        if not _SHA256_RE.fullmatch(value):
            raise ArtifactValidationError(
                f"row {row_number}: {field} must be a lowercase SHA-256 digest"
            )
        normalized[field] = value
    for field in (
        "W_semantics",
        "W_model_hex",
        "W_upper_hex",
        "K_gap_cert_hex",
        "K_gap_cert_derivation",
        "D_norm_bound_derivation",
        "graph_model",
        "rng_bit_generator",
        "graph_record_path",
        "graph_archive_path",
    ):
        value = str(row.get(field, "")).strip()
        if not value:
            raise ArtifactValidationError(
                f"row {row_number}: {field} must be nonempty"
            )
        normalized[field] = value
    conditional_log_path = str(row.get("conditional_log_path", "")).strip()
    normalized["conditional_log_path"] = conditional_log_path or None
    return normalized


def validate_rows(
    rows: Iterable[Mapping[str, Any]],
    fieldnames: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    raw_rows = list(rows)
    if fieldnames is None and raw_rows:
        fieldnames = raw_rows[0].keys()
    _require_columns(fieldnames)
    if len(raw_rows) != EXPECTED_ROW_COUNT:
        raise ArtifactValidationError(
            f"expected exactly {EXPECTED_ROW_COUNT} rows, found {len(raw_rows)}"
        )

    normalized_rows = [
        _normalize_row(row, index)
        for index, row in enumerate(raw_rows, start=2)
    ]

    versions = {int(row["schema_version"]) for row in normalized_rows}
    if versions != {EXPECTED_SCHEMA_VERSION}:
        found = ", ".join(str(version) for version in sorted(versions))
        raise ArtifactValidationError(
            f"expected result schema 4 on every row, found: {found}"
        )

    expected_keys = {
        (N, seed) for N in EXPECTED_NS for seed in EXPECTED_SEEDS
    }
    actual_keys = {
        (int(row["N"]), int(row["seed"])) for row in normalized_rows
    }
    if len(actual_keys) != len(normalized_rows):
        raise ArtifactValidationError("duplicate (N, seed) rows are not allowed")
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        raise ArtifactValidationError(
            f"cohort must be N=10/12 with seeds 0-19; "
            f"missing={missing}, unexpected={unexpected}"
        )

    archive_paths = set()
    archive_hashes = set()
    source_main_hashes = set()
    source_driver_hashes = set()
    graph_hashes = set()
    edge_hashes = set()

    for row in normalized_rows:
        key = (int(row["N"]), int(row["seed"]))
        label = f"N={key[0]}, seed={key[1]}"
        N = int(row["N"])
        n_vertices = int(row["n_vertices"])
        W_alias = float(row["W"])
        W_model = float(row["W_model"])
        W_upper = float(row["W_upper"])
        W_exact = rational_from_fields(row, "W_exact", label)
        W_upper_exact = rational_from_fields(row, "W_upper", label)
        lambda_i = int(row["Lambda_I"])
        K_gap_exact = rational_from_fields(row, "K_gap_exact", label)
        K_gap_from_W_upper_exact = rational_from_fields(
            row, "K_gap_from_W_upper_exact", label
        )
        K_gap_cert = float(row["K_gap_cert"])
        K_gap_cert_exact = rational_from_fields(row, "K_gap_cert", label)
        D_norm_exact = rational_from_fields(row, "D_norm_exact", label)
        diameter = float(row["D_spectral_diameter_est"])

        if not bool(row["connected"]):
            raise ArtifactValidationError(f"{label}: graph must be connected")
        if n_vertices != N + 1:
            raise ArtifactValidationError(f"{label}: n_vertices must equal N+1")
        if int(row["graph_generation_seed"]) != key[1]:
            raise ArtifactValidationError(
                f"{label}: graph_generation_seed does not match seed"
            )
        if int(row["graph_archive_records"]) != EXPECTED_ROW_COUNT:
            raise ArtifactValidationError(
                f"{label}: graph archive must contain 40 records"
            )
        if int(row["graph_schema_version"]) != EXPECTED_GRAPH_SCHEMA_VERSION:
            raise ArtifactValidationError(
                f"{label}: expected graph schema {EXPECTED_GRAPH_SCHEMA_VERSION}"
            )
        expected_lambda_i = 2 * (n_vertices // 2)
        if lambda_i != expected_lambda_i:
            raise ArtifactValidationError(
                f"{label}: Lambda_I must equal 2*floor(n_vertices/2)"
            )
        if W_model <= 0.0 or W_exact <= 0 or W_upper <= 0.0:
            raise ArtifactValidationError(
                f"{label}: weight sums must be positive"
            )
        if K_gap_cert <= 0.0 or diameter <= 0.0:
            raise ArtifactValidationError(
                f"{label}: K_gap_cert and diameter estimate must be positive"
            )
        if row["W_semantics"] != "compatibility alias of W_upper":
            raise ArtifactValidationError(f"{label}: unexpected W alias semantics")
        if W_alias != W_upper:
            raise ArtifactValidationError(f"{label}: W must exactly alias W_upper")
        if row["W_model_hex"] != W_model.hex():
            raise ArtifactValidationError(f"{label}: W_model hex encoding disagrees")
        if row["W_upper_hex"] != W_upper.hex():
            raise ArtifactValidationError(f"{label}: W_upper hex encoding disagrees")
        if W_upper_exact != Fraction.from_float(W_upper):
            raise ArtifactValidationError(f"{label}: W_upper rational metadata disagrees")
        if W_upper != least_binary64_not_below(W_exact):
            raise ArtifactValidationError(
                f"{label}: W_upper is not the least binary64 above W_exact"
            )
        if K_gap_exact != W_exact + lambda_i:
            raise ArtifactValidationError(f"{label}: K_gap_exact is inconsistent")
        expected_from_upper = W_upper_exact + lambda_i
        if K_gap_from_W_upper_exact != expected_from_upper:
            raise ArtifactValidationError(
                f"{label}: exact W_upper+Lambda_I metadata is inconsistent"
            )
        if row["K_gap_cert_hex"] != K_gap_cert.hex():
            raise ArtifactValidationError(f"{label}: K_gap_cert hex encoding disagrees")
        if K_gap_cert_exact != Fraction.from_float(K_gap_cert):
            raise ArtifactValidationError(
                f"{label}: K_gap_cert rational metadata disagrees"
            )
        if K_gap_cert != least_binary64_not_below(expected_from_upper):
            raise ArtifactValidationError(
                f"{label}: K_gap_cert is not the required upward rounding"
            )
        if K_gap_cert_exact < K_gap_exact:
            raise ArtifactValidationError(
                f"{label}: K_gap_cert does not upper-bound W_exact+Lambda_I"
            )
        if not _is_close(
            float(row["D_spectral_diameter_est"]),
            float(row["D_lambda_max_est"])
            - float(row["D_lambda_min_est"]),
        ):
            raise ArtifactValidationError(
                f"{label}: spectral-diameter diagnostic is inconsistent"
            )
        if not _is_close(
            float(row["D_norm_est"]),
            max(
                abs(float(row["D_lambda_min_est"])),
                abs(float(row["D_lambda_max_est"])),
            ),
        ):
            raise ArtifactValidationError(
                f"{label}: operator-norm diagnostic is inconsistent"
            )
        expected_D_norm_exact = max(W_exact, Fraction(lambda_i))
        if D_norm_exact != expected_D_norm_exact:
            raise ArtifactValidationError(
                f"{label}: exact D norm-bound metadata is inconsistent"
            )
        expected_D_norm_bound = max(W_upper, float(lambda_i))
        if float(row["D_norm_bound"]) != expected_D_norm_bound:
            raise ArtifactValidationError(
                f"{label}: D_norm_bound must exactly equal max(W_upper, Lambda_I)"
            )
        if Fraction.from_float(float(row["D_norm_bound"])) < D_norm_exact:
            raise ArtifactValidationError(
                f"{label}: D_norm_bound is not upward safe"
            )
        if not _is_close(
            float(row["K_gap_cert_over_diameter_est"]),
            K_gap_cert / diameter,
        ):
            raise ArtifactValidationError(
                f"{label}: K_gap_cert/diameter ratio is inconsistent"
            )
        if not _is_close(
            float(row["global_conditional_gap_lb"]),
            max(0.0, float(row["global_conditional_gap_lb_raw"])),
        ):
            raise ArtifactValidationError(
                f"{label}: nonnegative global conditional bound is inconsistent"
            )
        if not _is_close(
            float(row["total_window_width"]), float(row["unresolved_frac"])
        ):
            raise ArtifactValidationError(
                f"{label}: unresolved width aliases disagree"
            )
        if not _is_close(float(row["conditional_interval_coverage"]), 1.0):
            raise ArtifactValidationError(
                f"{label}: conditional intervals must cover [0,1]"
            )

        integer_aliases = (
            ("solves_hybrid", "conditional_oracle_calls"),
            ("solves_hybrid", "weylA_oracle_calls"),
            ("solves_uniform", "uniform_solves"),
        )
        for canonical, alias in integer_aliases:
            if int(row[canonical]) != int(row[alias]):
                raise ArtifactValidationError(
                    f"{label}: {canonical} and {alias} disagree"
                )
        float_aliases = (
            ("frac_weyl", "weyl0_frac_pos"),
            ("frac_floor", "psd_frac_pos"),
            ("uniform_sampled_target_frac", "uniform_target_frac"),
            ("uniform_sampled_positive_frac", "uniform_positive_frac"),
            ("uniform_sampled_unresolved_frac", "uniform_uncert_frac"),
            ("wall", "t_weylA"),
        )
        for canonical, alias in float_aliases:
            if not _is_close(float(row[canonical]), float(row[alias])):
                raise ArtifactValidationError(
                    f"{label}: {canonical} and {alias} disagree"
                )

        for field in _FRACTION_FIELDS:
            value = float(row[field])
            if value < -_FLOAT_ABS_TOL or value > 1.0 + _FLOAT_ABS_TOL:
                raise ArtifactValidationError(
                    f"{label}: {field} must lie in [0,1]"
                )
        if int(row["solves_hybrid"]) <= 0 or int(row["solves_uniform"]) <= 0:
            raise ArtifactValidationError(
                f"{label}: solve counts must be positive"
            )
        if int(row["num_windows"]) < 0:
            raise ArtifactValidationError(
                f"{label}: num_windows must be nonnegative"
            )

        graph_hashes.add(str(row["graph_record_sha256"]))
        edge_hashes.add(str(row["edge_payload_sha256"]))
        archive_paths.add(str(row["graph_archive_path"]))
        archive_hashes.add(str(row["graph_archive_sha256"]))
        source_main_hashes.add(str(row["source_main_sha256"]))
        source_driver_hashes.add(str(row["source_driver_sha256"]))

    if len(graph_hashes) != EXPECTED_ROW_COUNT:
        raise ArtifactValidationError("all 40 graph-record hashes must be unique")
    if len(edge_hashes) != EXPECTED_ROW_COUNT:
        raise ArtifactValidationError("all 40 edge-payload hashes must be unique")
    for label, values in (
        ("graph_archive_path", archive_paths),
        ("graph_archive_sha256", archive_hashes),
        ("source_main_sha256", source_main_hashes),
        ("source_driver_sha256", source_driver_hashes),
    ):
        if len(values) != 1:
            raise ArtifactValidationError(
                f"all rows must share one {label}; found {len(values)} values"
            )

    return sorted(
        normalized_rows, key=lambda row: (int(row["N"]), int(row["seed"]))
    )


def load_rows(path: Path = RESULTS_PATH) -> list[dict[str, Any]]:
    """Load and strictly validate the schema-v4 Section 7 CSV."""
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fieldnames = reader.fieldnames
    except OSError as exc:
        raise ArtifactValidationError(f"cannot read {path}: {exc}") from exc
    return validate_rows(rows, fieldnames)


def float_column(rows: Iterable[Mapping[str, Any]], name: str) -> np.ndarray:
    values = np.asarray([float(row[name]) for row in rows], dtype=float)
    if np.any(~np.isfinite(values)):
        raise ArtifactValidationError(f"summary column {name} is not finite")
    return values


def _size_summary(rows: list[dict[str, Any]], N: int) -> dict[str, Any]:
    group = [row for row in rows if int(row["N"]) == N]
    adaptive = float_column(group, "solves_hybrid")
    uniform = float_column(group, "solves_uniform")
    reductions = 100.0 * (uniform - adaptive) / uniform
    widths = float_column(group, "total_window_width")
    windows = float_column(group, "num_windows")
    W_model = float_column(group, "W_model")
    W_upper = float_column(group, "W_upper")
    W_exact_values = [
        rational_from_fields(row, "W_exact", f"N={N}, seed={row['seed']}")
        for row in group
    ]
    W_exact_mean = sum(W_exact_values, Fraction(0)) / len(W_exact_values)
    W_rounding_deltas = [
        Fraction.from_float(float(row["W_upper"])) - exact
        for row, exact in zip(group, W_exact_values)
    ]
    K_gap_cert = float_column(group, "K_gap_cert")
    K_gap_rounding_deltas = [
        Fraction.from_float(float(row["K_gap_cert"]))
        - rational_from_fields(
            row,
            "K_gap_from_W_upper_exact",
            f"N={N}, seed={row['seed']}",
        )
        for row in group
    ]
    diameter = float_column(group, "D_spectral_diameter_est")
    cert_ratios = float_column(group, "K_gap_cert_over_diameter_est")
    global_bounds = float_column(group, "global_conditional_gap_lb")
    global_raw_bounds = float_column(group, "global_conditional_gap_lb_raw")
    resolved_mask = np.asarray(
        [bool(row["global_conditionally_resolved"]) for row in group], dtype=bool
    )
    resolved_bounds = global_bounds[resolved_mask]
    wall = float_column(group, "wall")
    outliers = sorted(
        (
            {"seed": int(row["seed"]), "adaptive_calls": int(row["solves_hybrid"])}
            for row in group
        ),
        key=lambda item: (item["adaptive_calls"], item["seed"]),
        reverse=True,
    )[:2]

    return {
        "instances": len(group),
        "seeds": [int(row["seed"]) for row in group],
        "all_connected": all(bool(row["connected"]) for row in group),
        "weights": {
            "W_model_mean": float(np.mean(W_model)),
            "W_exact_mean": float(W_exact_mean),
            "W_exact_mean_rational": {
                "numerator": W_exact_mean.numerator,
                "denominator": W_exact_mean.denominator,
            },
            "W_upper_mean": float(np.mean(W_upper)),
            "W_upper_minus_W_exact_mean": float(
                sum(W_rounding_deltas, Fraction(0)) / len(W_rounding_deltas)
            ),
            "W_upper_minus_W_exact_max": float(max(W_rounding_deltas)),
            "upward_rounded_instances": int(
                sum(delta > 0 for delta in W_rounding_deltas)
            ),
        },
        "gap_lipschitz": {
            "K_gap_cert_mean": float(np.mean(K_gap_cert)),
            "K_gap_cert_population_sd": float(np.std(K_gap_cert)),
            "K_gap_cert_upward_rounding_mean": float(
                sum(K_gap_rounding_deltas, Fraction(0))
                / len(K_gap_rounding_deltas)
            ),
            "K_gap_cert_upward_rounding_max": float(max(K_gap_rounding_deltas)),
            "K_gap_cert_upward_rounded_instances": int(
                sum(delta > 0 for delta in K_gap_rounding_deltas)
            ),
            "spectral_diameter_estimate_mean": float(np.mean(diameter)),
            "spectral_diameter_estimate_population_sd": float(np.std(diameter)),
            "cert_over_diameter_estimate_mean": float(np.mean(cert_ratios)),
            "cert_over_diameter_estimate_min": float(np.min(cert_ratios)),
            "cert_over_diameter_estimate_max": float(np.max(cert_ratios)),
        },
        "solve_counts": {
            "uniform_median": float(np.median(uniform)),
            "uniform_mean": float(np.mean(uniform)),
            "uniform_population_sd": float(np.std(uniform)),
            "adaptive_median": float(np.median(adaptive)),
            "adaptive_mean": float(np.mean(adaptive)),
            "adaptive_population_sd": float(np.std(adaptive)),
            "relative_count_change_mean_percent": float(np.mean(reductions)),
            "relative_count_change_population_sd_percent": float(np.std(reductions)),
            "relative_count_change_min_percent": float(np.min(reductions)),
            "relative_count_change_median_percent": float(np.median(reductions)),
            "relative_count_change_max_percent": float(np.max(reductions)),
            "largest_adaptive_call_counts": outliers,
        },
        "global_conditional": {
            "resolved_instances": int(np.count_nonzero(resolved_mask)),
            "unresolved_instances": int(len(group) - np.count_nonzero(resolved_mask)),
            "lower_bound_mean": float(np.mean(global_bounds)),
            "lower_bound_min": float(np.min(global_bounds)),
            "lower_bound_max": float(np.max(global_bounds)),
            "raw_lower_bound_mean": float(np.mean(global_raw_bounds)),
            "resolved_lower_bound_mean": (
                float(np.mean(resolved_bounds)) if resolved_bounds.size else None
            ),
            "resolved_lower_bound_min": (
                float(np.min(resolved_bounds)) if resolved_bounds.size else None
            ),
            "resolved_lower_bound_max": (
                float(np.max(resolved_bounds)) if resolved_bounds.size else None
            ),
        },
        "unresolved": {
            "full_path_resolved": int(np.count_nonzero(windows == 0.0)),
            "globally_conditionally_resolved": int(np.count_nonzero(resolved_mask)),
            "mean_components": float(np.mean(windows)),
            "mean_width": float(np.mean(widths)),
            "maximum_width": float(np.max(widths)),
        },
        "endpoint_coverage": {
            "weyl_positive_fraction_mean": float(
                np.mean(float_column(group, "frac_weyl"))
            ),
            "psd_floor_positive_fraction_mean": float(
                np.mean(float_column(group, "frac_floor"))
            ),
            "psd_floor_poly_positive_fraction_mean": float(
                np.mean(float_column(group, "frac_floor_poly"))
            ),
        },
        "observed_timing_seconds": {
            "median": float(np.median(wall)),
            "mean": float(np.mean(wall)),
            "population_sd": float(np.std(wall)),
        },
    }


def summary(
    rows: Iterable[Mapping[str, Any]],
    *,
    csv_sha256: str | None = None,
    csv_path: str = "section7_results.csv",
) -> dict[str, Any]:
    """Build the deterministic aggregate report from validated rows."""
    validated = validate_rows(list(rows))
    if csv_sha256 is not None and not _SHA256_RE.fullmatch(csv_sha256):
        raise ArtifactValidationError("csv_sha256 must be a lowercase SHA-256 digest")

    first = validated[0]
    report: dict[str, Any] = {
        "_metadata": {
            "summary_schema_version": 2,
            "result_schema_version": EXPECTED_SCHEMA_VERSION,
            "graph_schema_version": EXPECTED_GRAPH_SCHEMA_VERSION,
            "csv_path": csv_path,
            "csv_sha256": csv_sha256,
            "rows": EXPECTED_ROW_COUNT,
            "N_values": list(EXPECTED_NS),
            "seeds": list(EXPECTED_SEEDS),
            "graph_archive_path": first["graph_archive_path"],
            "graph_archive_sha256": first["graph_archive_sha256"],
            "graph_archive_records": first["graph_archive_records"],
            "source_sha256": {
                "main.py": first["source_main_sha256"],
                "maxcut_gap_benchmark.py": first["source_driver_sha256"],
            },
            "rounding_policy": (
                "W_upper and K_gap_cert are the least binary64 upper bounds "
                "of their recorded exact dyadic rational inputs"
            ),
            "timing_note": (
                "Observed wall-clock timings are platform-, BLAS-, and "
                "scheduler-dependent"
            ),
        }
    }
    for N in EXPECTED_NS:
        report[str(N)] = _size_summary(validated, N)
    return report


def build_report(path: Path = RESULTS_PATH) -> dict[str, Any]:
    rows = load_rows(path)
    return summary(
        rows,
        csv_sha256=sha256_path(path),
        csv_path=path.name,
    )


def render_json(report: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _format_scientific(value: float) -> str:
    if value == 0.0:
        return "0"
    exponent = int(math.floor(math.log10(abs(value))))
    mantissa = value / (10.0**exponent)
    return rf"${mantissa:.3f}\times 10^{{{exponent}}}$"


def _macro_rows(lines: list[str]) -> str:
    body = "%\n".join(lines)
    return "{%\n" + body + "%\n}"


def render_tex(report: Mapping[str, Any]) -> bytes:
    metadata = report["_metadata"]
    weight_rows = []
    gap_rows = []
    global_rows = []
    solve_rows = []
    unresolved_rows = []
    coverage_rows = []
    for N in EXPECTED_NS:
        section = report[str(N)]
        weights = section["weights"]
        gap = section["gap_lipschitz"]
        global_conditional = section["global_conditional"]
        solves = section["solve_counts"]
        unresolved = section["unresolved"]
        coverage = section["endpoint_coverage"]
        weight_rows.append(
            f"{N} & {weights['W_model_mean']:.6f} & "
            f"{weights['W_exact_mean']:.6f} & "
            f"{weights['W_upper_mean']:.6f} & "
            f"{weights['upward_rounded_instances']}/20 \\\\"
        )
        gap_rows.append(
            f"{N} & {gap['spectral_diameter_estimate_mean']:.2f} & "
            f"{gap['K_gap_cert_mean']:.2f} & "
            f"{gap['cert_over_diameter_estimate_mean']:.2f} \\\\"
        )
        global_rows.append(
            f"{N} & {global_conditional['resolved_instances']}/20 & "
            f"{_format_scientific(global_conditional['lower_bound_mean'])} & "
            f"{_format_scientific(global_conditional['lower_bound_min'])} & "
            f"{_format_scientific(global_conditional['lower_bound_max'])} \\\\"
        )
        solve_rows.append(
            rf"{N} & ${solves['uniform_median']:.1f}\,/\,"
            f"{solves['uniform_mean']:.2f} \\pm "
            f"{solves['uniform_population_sd']:.2f}$ & "
            rf"${solves['adaptive_median']:.1f}\,/\,"
            f"{solves['adaptive_mean']:.2f} \\pm "
            f"{solves['adaptive_population_sd']:.2f}$ & "
            f"${solves['relative_count_change_mean_percent']:.2f}\\% "
            f"\\pm {solves['relative_count_change_population_sd_percent']:.2f}\\%$ & "
            f"${solves['relative_count_change_min_percent']:.2f}\\%$ & "
            f"${solves['relative_count_change_median_percent']:.2f}\\%$ & "
            f"${solves['relative_count_change_max_percent']:.2f}\\%$ \\\\"
        )
        unresolved_rows.append(
            f"{N} & ${unresolved['full_path_resolved']}/20$ & "
            f"{unresolved['mean_components']:.2f} & "
            f"{_format_scientific(unresolved['mean_width'])} & "
            f"{_format_scientific(unresolved['maximum_width'])} \\\\"
        )
        coverage_rows.append(
            f"{N} & {coverage['weyl_positive_fraction_mean']:.4f} & "
            f"{coverage['psd_floor_positive_fraction_mean']:.4f} & "
            f"{coverage['psd_floor_poly_positive_fraction_mean']:.4f} \\\\"
        )

    text = "\n".join(
        [
            "% Generated by summarize_section7_results.py; do not edit.",
            rf"\providecommand{{\SectionSevenResultSchema}}{{{metadata['result_schema_version']}}}",
            rf"\providecommand{{\SectionSevenInstanceCount}}{{{metadata['rows']}}}",
            rf"\providecommand{{\SectionSevenCsvSha}}{{\texttt{{{metadata['csv_sha256']}}}}}",
            rf"\providecommand{{\SectionSevenGraphArchiveSha}}{{\texttt{{{metadata['graph_archive_sha256']}}}}}",
            rf"\providecommand{{\SectionSevenWeightRows}}{_macro_rows(weight_rows)}",
            rf"\providecommand{{\SectionSevenGapDiameterRows}}{_macro_rows(gap_rows)}",
            rf"\providecommand{{\SectionSevenGlobalConditionalRows}}{_macro_rows(global_rows)}",
            rf"\providecommand{{\SectionSevenSolveCountRows}}{_macro_rows(solve_rows)}",
            rf"\providecommand{{\SectionSevenUnresolvedRows}}{_macro_rows(unresolved_rows)}",
            rf"\providecommand{{\SectionSevenCoverageRows}}{_macro_rows(coverage_rows)}",
            "",
        ]
    )
    return text.encode("ascii")


def expected_outputs(
    csv_path: Path = RESULTS_PATH,
) -> tuple[dict[str, Any], bytes, bytes]:
    report = build_report(csv_path)
    return report, render_json(report), render_tex(report)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _check_file(path: Path, expected: bytes) -> str | None:
    if not path.is_file():
        return f"missing generated file: {path.relative_to(PROJECT_ROOT)}"
    if path.read_bytes() != expected:
        return f"stale generated file: {path.relative_to(PROJECT_ROOT)}"
    return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and summarize the schema-v4 Section 7 archive."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify generated summary files without rewriting them",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report, json_payload, tex_payload = expected_outputs()
        if args.check:
            problems = [
                problem
                for problem in (
                    _check_file(OUTPUT_PATH, json_payload),
                    _check_file(TEX_OUTPUT_PATH, tex_payload),
                )
                if problem is not None
            ]
            if problems:
                raise ArtifactValidationError(
                    "; ".join(problems)
                    + ". Run `python summarize_section7_results.py`"
                )
            print("Section 7 summary JSON and TeX are current.")
            return 0

        _atomic_write(OUTPUT_PATH, json_payload)
        _atomic_write(TEX_OUTPUT_PATH, tex_payload)
        metadata = report["_metadata"]
        print(
            "Wrote results/section7_summary.json and "
            "results/section7_summary.tex from "
            f"{metadata['rows']} schema-v4 rows."
        )
        return 0
    except ArtifactValidationError as exc:
        print(f"summary validation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
