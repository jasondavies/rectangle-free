#!/usr/bin/env python3
"""Validate and aggregate manifest-driven v3 exact GPU campaigns."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import tempfile
from typing import Any


class ValidationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Geometry:
    name: str
    quantity: str
    result_magic: str
    orbit_magic: bytes
    columns: int
    expected_records: int
    expected_labelled_weight: int
    expected_covered_weight: int
    expected_contribution: int | None = None
    transpose_quotient: bool = False


GEOMETRIES = {
    "6x10": Geometry(
        "6x10",
        "T_4(6,10)",
        "RECT6X10_PREFIX_RESULT",
        b"R6ORB01\0",
        10,
        502_732_239,
        635_593_043_085_854_200,
        1 << 60,
        134_801_843_107_132_031_823_174_944_563_200,
    ),
    "6x11": Geometry(
        "6x11",
        "T_4(6,11)",
        "RECT6X11_PREFIX_RESULT",
        b"R6W1101\0",
        11,
        3_294_410_345,
        40_503_202_364_427_236_102,
        1 << 66,
        76_380_896_192_602_995_200_411_451_026_841_600,
    ),
    "7x9": Geometry(
        "7x9",
        "T_4(7,9)",
        "RECT7X9_PACKED_RESULT",
        b"R7ORB09\0",
        9,
        3_608_247_685,
        1 << 62,
        1 << 63,
        2_504_755_357_815_289_286_302_895_662_387_200,
    ),
    "8x8": Geometry(
        "8x8",
        "T_4(8,8)",
        "RECT8X8_PREFIX_RESULT",
        b"R8SQT01\0",
        8,
        3_671_999_389,
        10_139_684_107_326_071_075,
        1 << 64,
        transpose_quotient=True,
    ),
}

WORK_ID = re.compile(r"[A-Za-z0-9_.-]+")
SHA256 = re.compile(r"[0-9a-f]{64}")
COMMON_INTEGER_FIELDS = (
    "records",
    "labelled_weight",
    "kernels",
    "covered_weight",
    "verified",
    "direct_comparisons",
    "contribution",
    "minimum_free_bytes",
)
PROVENANCE_FIELDS = (
    "solver_binary_sha256",
    "solver_configuration_sha256",
    "canonical_cache_sha256",
    "orbit_corpus_sha256",
)


@dataclass(frozen=True)
class WorkItem:
    id: str
    path: str
    start: int
    end: int
    filter_mod: int = 0
    filter_id: int = 0


@dataclass(frozen=True)
class OrbitHeader:
    path: Path
    records: int


def parse_nonnegative(text: str, context: str) -> int:
    if not text.isdecimal():
        raise ValidationError(f"invalid nonnegative integer for {context}: {text!r}")
    return int(text)


def read_manifest(path: Path) -> list[WorkItem]:
    items: list[WorkItem] = []
    ids: set[str] = set()
    with path.open() as source:
        for line_number, line in enumerate(source, 1):
            fields = line.split()
            if not fields or fields[0].startswith("#"):
                continue
            if len(fields) not in (4, 6):
                raise ValidationError(f"invalid manifest line {line_number}")
            identifier = fields[0]
            if not WORK_ID.fullmatch(identifier) or identifier in ids:
                raise ValidationError(
                    f"invalid or duplicate work id on line {line_number}"
                )
            ids.add(identifier)
            start = parse_nonnegative(fields[2], f"line {line_number} start")
            end = parse_nonnegative(fields[3], f"line {line_number} end")
            filter_mod = (
                parse_nonnegative(fields[4], f"line {line_number} filter modulus")
                if len(fields) == 6
                else 0
            )
            filter_id = (
                parse_nonnegative(fields[5], f"line {line_number} filter id")
                if len(fields) == 6
                else 0
            )
            if len(fields) == 6 and (
                filter_mod == 0 or filter_id >= filter_mod
            ):
                raise ValidationError(f"invalid manifest filter on line {line_number}")
            items.append(
                WorkItem(
                    identifier,
                    fields[1],
                    start,
                    end,
                    filter_mod,
                    filter_id,
                )
            )
    if not items:
        raise ValidationError("manifest is empty")
    return items


def result_path(directory: Path, item: WorkItem) -> Path:
    return directory / f"{item.id}.result"


def read_result(path: Path, item: WorkItem, geometry: Geometry) -> dict[str, Any]:
    lines = path.read_bytes().splitlines(keepends=True)
    expected_header = f"{geometry.result_magic} 3\n".encode()
    if not lines or lines[0] != expected_header:
        raise ValidationError(f"invalid result header: {path}")
    fields: dict[str, str] = {}
    payload = bytearray()
    claimed_checksum: str | None = None
    for line_number, raw_line in enumerate(lines[1:], 2):
        if not raw_line.endswith(b"\n"):
            raise ValidationError(f"unterminated result line {line_number}: {path}")
        try:
            words = raw_line[:-1].decode("ascii").split()
        except UnicodeDecodeError as error:
            raise ValidationError(f"non-ASCII result field: {path}") from error
        if len(words) != 2:
            raise ValidationError(f"invalid result field on line {line_number}: {path}")
        key, value = words
        if key == "result_payload_sha256":
            if claimed_checksum is not None or line_number != len(lines):
                raise ValidationError(f"misplaced or duplicate result checksum: {path}")
            claimed_checksum = value
            continue
        if key in fields:
            raise ValidationError(f"duplicate result field {key}: {path}")
        fields[key] = value
        payload.extend(raw_line)
    actual_checksum = hashlib.sha256(payload).hexdigest()
    if claimed_checksum != actual_checksum:
        raise ValidationError(f"result checksum mismatch: {path}")

    expected_identity = {
        "id": item.id,
        "path": item.path,
        "start": str(item.start),
        "end": str(item.end),
        "filter_mod": str(item.filter_mod),
        "filter_id": str(item.filter_id),
    }
    for key, expected in expected_identity.items():
        if fields.get(key) != expected:
            raise ValidationError(f"result identity mismatch for {key}: {path}")
    if fields.get("geometry") != geometry.name or fields.get(
        "token_plane_quotient"
    ) != "1":
        raise ValidationError(f"unsupported exact representation: {path}")
    if geometry.transpose_quotient and fields.get("transpose_quotient") != "1":
        raise ValidationError(f"missing mandatory transpose quotient: {path}")
    for key in PROVENANCE_FIELDS:
        if not SHA256.fullmatch(fields.get(key, "")):
            raise ValidationError(f"invalid provenance field {key}: {path}")
    for key in COMMON_INTEGER_FIELDS:
        if key not in fields:
            raise ValidationError(f"missing result field {key}: {path}")
        fields[key] = parse_nonnegative(fields[key], f"{path}:{key}")
    if fields["kernels"] != fields["records"]:
        raise ValidationError(f"record/kernel mismatch: {path}")
    for key, value in tuple(fields.items()):
        if key.endswith("_seconds"):
            try:
                converted = float(value)
            except ValueError as error:
                raise ValidationError(f"invalid timing field {key}: {path}") from error
            if converted < 0:
                raise ValidationError(f"negative timing field {key}: {path}")
            if not math.isfinite(converted):
                raise ValidationError(f"non-finite timing field {key}: {path}")
            fields[key] = converted
    for key in ("gpu_seconds", "total_seconds"):
        if not isinstance(fields.get(key), float):
            raise ValidationError(f"missing timing field {key}: {path}")
    fields["result_payload_sha256"] = actual_checksum
    return fields


def resolve_corpus_path(item: WorkItem, corpus_root: Path | None) -> Path:
    original = Path(item.path)
    candidates = [original]
    if corpus_root is not None:
        if not original.is_absolute():
            candidates.append(corpus_root / original)
        candidates.append(corpus_root / original.name)
    found = []
    for candidate in candidates:
        if candidate.is_file():
            resolved = candidate.resolve()
            if resolved not in found:
                found.append(resolved)
    if len(found) != 1:
        reason = "not found" if not found else "ambiguous"
        raise ValidationError(f"corpus path {reason} for {item.path!r}")
    return found[0]


def read_orbit_header(path: Path, geometry: Geometry) -> OrbitHeader:
    with path.open("rb") as source:
        raw = source.read(20)
    if len(raw) != 20:
        raise ValidationError(f"truncated orbit header: {path}")
    magic, columns, records = struct.unpack("<8sIQ", raw)
    if magic != geometry.orbit_magic or columns != geometry.columns:
        raise ValidationError(
            f"wrong orbit format for {geometry.name}: {path} "
            f"(magic={magic!r}, columns={columns})"
        )
    expected_size = 20 + 16 * records
    if path.stat().st_size != expected_size:
        raise ValidationError(f"orbit file size mismatch: {path}")
    return OrbitHeader(path, records)


def exact_coverage(
    items: list[WorkItem],
    geometry: Geometry,
    corpus_root: Path | None,
) -> tuple[dict[str, Any], dict[str, Path]]:
    resolved = {item.id: resolve_corpus_path(item, corpus_root) for item in items}
    headers = {
        path: read_orbit_header(path, geometry) for path in set(resolved.values())
    }
    groups: dict[Path, list[tuple[WorkItem, int]]] = {}
    for item in items:
        header = headers[resolved[item.id]]
        end = item.end or header.records
        if item.start > end or end > header.records:
            raise ValidationError(f"invalid range for {item.id}")
        if item.start == end:
            raise ValidationError(f"empty range for {item.id}")
        groups.setdefault(header.path, []).append((item, end))

    segments = 0
    filtered_segments = 0
    for path, work in groups.items():
        header = headers[path]
        boundaries = {0, header.records}
        for item, end in work:
            boundaries.add(item.start)
            boundaries.add(end)
        ordered = sorted(boundaries)
        for begin, end in zip(ordered, ordered[1:]):
            if begin == end:
                continue
            active = [item for item, stop in work if item.start <= begin and stop >= end]
            unfiltered = [item for item in active if item.filter_mod == 0]
            if len(unfiltered) == 1 and len(active) == 1:
                segments += 1
                continue
            if unfiltered or not active:
                raise ValidationError(
                    f"gap or overlap in {path} over [{begin},{end})"
                )
            moduli = {item.filter_mod for item in active}
            if len(moduli) != 1:
                raise ValidationError(
                    f"mixed filter moduli in {path} over [{begin},{end})"
                )
            modulus = next(iter(moduli))
            owner_ids = [item.filter_id for item in active]
            if len(owner_ids) != modulus or set(owner_ids) != set(range(modulus)):
                raise ValidationError(
                    f"incomplete or duplicate filter owners in {path} "
                    f"over [{begin},{end})"
                )
            segments += 1
            filtered_segments += 1
    return (
        {
            "exact": True,
            "files": len(groups),
            "records": sum(header.records for header in headers.values()),
            "segments": segments,
            "filtered_segments": filtered_segments,
        },
        resolved,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_input_hashes(
    results: dict[str, dict[str, Any]], resolved: dict[str, Path], workers: int
) -> int:
    paths = sorted({resolved[identifier] for identifier in results})
    if not paths:
        return 0
    with ThreadPoolExecutor(max_workers=min(workers, len(paths))) as pool:
        digests = dict(zip(paths, pool.map(sha256_file, paths)))
    for identifier, result in results.items():
        actual = digests[resolved[identifier]]
        if result["orbit_corpus_sha256"] != actual:
            raise ValidationError(f"orbit corpus SHA-256 mismatch for {identifier}")
    return len(paths)


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w") as output:
            json.dump(value, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def accessible_file(path: Path) -> bool:
    """Treat inaccessible remote manifest paths as absent on this host."""
    try:
        return path.is_file()
    except OSError:
        return False


def aggregate_campaign(
    geometry: Geometry,
    manifest: Path,
    results_dir: Path,
    corpus_root: Path | None = None,
    full: bool = False,
    verify_inputs: bool = False,
    hash_workers: int = 4,
    require_single_binary: bool = False,
    solver_binaries: tuple[Path, ...] = (),
    canonical_cache: Path | None = None,
) -> dict[str, Any]:
    items = read_manifest(manifest)
    expected_names = {f"{item.id}.result" for item in items}
    extras = sorted(
        path.name for path in results_dir.glob("*.result") if path.name not in expected_names
    )
    if extras:
        raise ValidationError("unmanifested result files: " + ", ".join(extras))
    results: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for item in items:
        path = result_path(results_dir, item)
        if not path.is_file():
            missing.append(item.id)
            continue
        results[item.id] = read_result(path, item, geometry)

    configurations = {
        result["solver_configuration_sha256"] for result in results.values()
    }
    canonical_caches = {
        result["canonical_cache_sha256"] for result in results.values()
    }
    binaries = {result["solver_binary_sha256"] for result in results.values()}
    if len(configurations) > 1:
        raise ValidationError("mixed solver configuration provenance")
    if len(canonical_caches) > 1:
        raise ValidationError("mixed canonical cache provenance")
    if require_single_binary and len(binaries) > 1:
        raise ValidationError("mixed solver binary provenance")
    supplied_binary_digests = {sha256_file(path) for path in solver_binaries}
    if results and supplied_binary_digests and supplied_binary_digests != binaries:
        raise ValidationError("supplied solver binaries do not match result provenance")
    canonical_cache_verified = False
    if canonical_cache is not None and results:
        supplied_cache_digest = sha256_file(canonical_cache)
        if canonical_caches != {supplied_cache_digest}:
            raise ValidationError(
                "supplied canonical cache does not match result provenance"
            )
        canonical_cache_verified = True
    for item in items:
        same_path_hashes = {
            results[other.id]["orbit_corpus_sha256"]
            for other in items
            if other.path == item.path and other.id in results
        }
        if len(same_path_hashes) > 1:
            raise ValidationError(f"mixed corpus provenance for {item.path}")

    should_check_coverage = full or verify_inputs or corpus_root is not None or all(
        accessible_file(Path(item.path)) for item in items
    )
    coverage: dict[str, Any] | None = None
    resolved: dict[str, Path] = {}
    if should_check_coverage:
        coverage, resolved = exact_coverage(items, geometry, corpus_root)
    verified_input_files = 0
    if verify_inputs:
        if not resolved:
            raise ValidationError("input verification requires accessible corpus files")
        verified_input_files = verify_input_hashes(results, resolved, hash_workers)

    totals = {
        key: sum(int(result[key]) for result in results.values())
        for key in (
            "records",
            "labelled_weight",
            "kernels",
            "covered_weight",
            "verified",
            "direct_comparisons",
            "contribution",
        )
    }
    timing_names = sorted(
        {
            key
            for result in results.values()
            for key in result
            if key.endswith("_seconds")
        }
    )
    timings = {
        key: sum(float(result.get(key, 0.0)) for result in results.values())
        for key in timing_names
    }
    gpu_seconds = timings.get("gpu_seconds", 0.0)
    solver_seconds = timings.get("total_seconds", 0.0)
    performance = {
        "gpu_hours": gpu_seconds / 3600,
        "solver_hours": solver_seconds / 3600,
        "weighted_comparisons_per_second": (
            totals["direct_comparisons"] / gpu_seconds if gpu_seconds else 0.0
        ),
    }

    full_failures: list[str] = []
    if full:
        if missing:
            full_failures.append("results are missing")
        if coverage is None or not coverage["exact"]:
            full_failures.append("exact work coverage was not established")
        wanted = {
            "records": geometry.expected_records,
            "kernels": geometry.expected_records,
            "labelled_weight": geometry.expected_labelled_weight,
            "covered_weight": geometry.expected_covered_weight,
        }
        for key, expected in wanted.items():
            if totals[key] != expected:
                full_failures.append(f"{key} {totals[key]} != {expected}")
        if coverage is not None and coverage["records"] != geometry.expected_records:
            full_failures.append(
                f"corpus records {coverage['records']} != {geometry.expected_records}"
            )
        if (
            geometry.expected_contribution is not None
            and totals["contribution"] != geometry.expected_contribution
        ):
            full_failures.append(
                f"known contribution mismatch: {totals['contribution']} != "
                f"{geometry.expected_contribution}"
            )

    report = {
        "schema_version": 1,
        "geometry": geometry.name,
        "quantity": geometry.quantity,
        "manifest": str(manifest),
        "results_directory": str(results_dir),
        "expected_items": len(items),
        "completed_items": len(results),
        "missing_items": missing,
        "complete": not missing and not full_failures,
        "full_check_requested": full,
        "full_check_failures": full_failures,
        "coverage": coverage,
        "verified_input_files": verified_input_files,
        "verified_solver_binaries": len(supplied_binary_digests) if results else 0,
        "canonical_cache_verified": canonical_cache_verified,
        "provenance": {
            "solver_configuration_sha256": sorted(configurations),
            "canonical_cache_sha256": sorted(canonical_caches),
            "solver_binary_sha256": sorted(binaries),
        },
        "totals": totals,
        "timings": timings,
        "performance": performance,
    }
    return report


def print_report(report: dict[str, Any]) -> None:
    totals = report["totals"]
    performance = report["performance"]
    print(
        f"CAMPAIGN geometry={report['geometry']} "
        f"completed={report['completed_items']}/{report['expected_items']} "
        f"missing={len(report['missing_items'])} "
        f"binary_variants={len(report['provenance']['solver_binary_sha256'])}"
    )
    coverage = report["coverage"]
    if coverage is None:
        print("COVERAGE unchecked=1")
    else:
        print(
            f"COVERAGE exact=1 files={coverage['files']} "
            f"records={coverage['records']} segments={coverage['segments']} "
            f"filtered_segments={coverage['filtered_segments']}"
        )
    print(
        f"TOTAL records={totals['records']} kernels={totals['kernels']} "
        f"labelled_weight={totals['labelled_weight']} "
        f"covered_weight={totals['covered_weight']} "
        f"comparisons={totals['direct_comparisons']} "
        f"contribution={totals['contribution']}"
    )
    print(
        f"TIMING gpu_hours={performance['gpu_hours']:.6f} "
        f"solver_hours={performance['solver_hours']:.6f} "
        f"weighted_comparisons_per_second="
        f"{performance['weighted_comparisons_per_second']:.3f}"
    )
    if report["missing_items"]:
        print("MISSING " + " ".join(report["missing_items"]))
    if report["full_check_requested"]:
        if report["full_check_failures"]:
            print("FULL_CHECK FAIL")
            for failure in report["full_check_failures"]:
                print(f"FAILURE {failure}")
        else:
            print("FULL_CHECK OK")
            print(f"{report['quantity']}={totals['contribution']} COMPLETE")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("geometry", choices=sorted(GEOMETRIES))
    parser.add_argument("manifest", type=Path)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument(
        "--corpus-root",
        type=Path,
        help="local corpus directory; remote manifest paths fall back to basenames",
    )
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--verify-input-sha256", action="store_true")
    parser.add_argument("--hash-workers", type=int, default=4)
    parser.add_argument("--require-single-binary", action="store_true")
    parser.add_argument(
        "--solver-binary",
        type=Path,
        action="append",
        default=[],
        help="verify provenance against one or more exact campaign executables",
    )
    parser.add_argument(
        "--canonical-cache",
        type=Path,
        help="verify provenance against the exact 7x5 cache or 8x4 seed file",
    )
    parser.add_argument("--write-json", type=Path)
    args = parser.parse_args()
    if args.hash_workers < 1:
        parser.error("--hash-workers must be positive")
    try:
        report = aggregate_campaign(
            GEOMETRIES[args.geometry],
            args.manifest,
            args.results_dir,
            args.corpus_root,
            args.full,
            args.verify_input_sha256,
            args.hash_workers,
            args.require_single_binary,
            tuple(args.solver_binary),
            args.canonical_cache,
        )
        print_report(report)
        if args.write_json is not None:
            atomic_write_json(args.write_json, report)
        if args.full and report["full_check_failures"]:
            raise SystemExit(1)
    except ValidationError as error:
        raise SystemExit(f"VALIDATION_ERROR {error}") from error


if __name__ == "__main__":
    main()
