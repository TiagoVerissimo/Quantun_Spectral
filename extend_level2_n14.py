"""Extend the archived Section 7 Level-2 cohort with N=14 instances.

The existing N=10 and N=12 rows and graph records are retained byte-for-byte.
Only the 20 N=14 instances are evaluated, using ``main.run_instance_parallel``
with the same numerical parameters as the original cohort.  Aggregate outputs
are replaced atomically after every N=14 worker has completed successfully.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing
from pathlib import Path
from typing import Any

import main as benchmark


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_PATH = PROJECT_ROOT / "section7_results.csv"
RUN_MANIFEST_PATH = PROJECT_ROOT / "results" / "section7_run_manifest.json"
BASE_NS = (10, 12)
EXTENSION_N = 14
EXTENDED_NS = BASE_NS + (EXTENSION_N,)
EXPECTED_SEEDS = tuple(range(20))


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_archived_base_rows(path: Path = RESULTS_PATH) -> list[dict[str, Any]]:
    """Load exactly the archived N=10/12 rows without changing their values."""
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    expected_keys = {(N, seed) for N in BASE_NS for seed in EXPECTED_SEEDS}
    base_rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[int, int]] = set()
    for row in rows:
        key = (int(row["N"]), int(row["seed"]))
        if key[0] not in BASE_NS:
            continue
        if key in seen_keys:
            raise ValueError(f"duplicate archived base row {key}")
        seen_keys.add(key)
        row["schema_version"] = int(row["schema_version"])
        row["N"], row["seed"] = key
        base_rows.append(row)

    if seen_keys != expected_keys:
        missing = sorted(expected_keys - seen_keys)
        unexpected = sorted(seen_keys - expected_keys)
        raise ValueError(
            "the archived base cohort must contain N=10/12, seeds 0-19; "
            f"missing={missing}, unexpected={unexpected}"
        )

    expected_main_hash = benchmark._source_sha256("main.py")
    expected_driver_hash = benchmark._source_sha256("maxcut_gap_benchmark.py")
    for row in base_rows:
        key = (row["N"], row["seed"])
        if row["source_main_sha256"] != expected_main_hash:
            raise ValueError(f"archived base row {key} has a stale main.py hash")
        if row["source_driver_sha256"] != expected_driver_hash:
            raise ValueError(
                f"archived base row {key} has a stale maxcut driver hash"
            )
        graph_path = PROJECT_ROOT / row["graph_record_path"]
        if not graph_path.is_file():
            raise FileNotFoundError(f"missing archived graph record: {graph_path}")

    return sorted(base_rows, key=lambda row: (row["N"], row["seed"]))


def _record_extension_source_hash() -> None:
    with RUN_MANIFEST_PATH.open("r", encoding="ascii") as handle:
        manifest = json.load(handle)
    source_hashes = manifest.setdefault("source_sha256", {})
    source_hashes[Path(__file__).name] = _sha256_path(Path(__file__))
    benchmark._write_json(str(RUN_MANIFEST_PATH), manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "run the 20 N=14 Level-2 instances and extend the archived "
            "N=10/12 Section 7 cohort"
        )
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=5,
        help="number of multiprocessing workers (default: 5)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.processes <= 0:
        raise ValueError("--processes must be positive")

    base_rows = _load_archived_base_rows()
    print(
        f"Running N={EXTENSION_N} with {len(EXPECTED_SEEDS)} seeds "
        f"across {args.processes} workers...",
        flush=True,
    )
    with multiprocessing.Pool(processes=args.processes) as pool:
        extension_rows = pool.map(
            benchmark.run_instance_parallel,
            [(EXTENSION_N, seed) for seed in EXPECTED_SEEDS],
        )

    for row in sorted(extension_rows, key=lambda item: int(item["seed"])):
        print(
            f"N={row['N']} seed={row['seed']}: "
            f"adaptive={row['solves_hybrid']}, "
            f"uniform={row['solves_uniform']}, "
            f"resolved={row['global_conditionally_resolved']}, "
            f"wall={row['wall']:.2f}s",
            flush=True,
        )

    benchmark.BENCHMARK_NS = EXTENDED_NS
    archive, manifest_path, csv_sha256 = benchmark._write_benchmark_outputs(
        base_rows + extension_rows
    )
    _record_extension_source_hash()
    print(
        f"Wrote {archive['records']} graph records to {archive['path']} "
        f"(sha256={archive['sha256']})."
    )
    print(
        f"Wrote section7_results.csv (sha256={csv_sha256}) and "
        f"{manifest_path}."
    )
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
