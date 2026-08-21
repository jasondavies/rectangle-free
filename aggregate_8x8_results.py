#!/usr/bin/env python3
"""Archive-compatible aggregator for the pre-transpose 8x8 campaign.

New manifest-driven, transpose-quotient v3 campaigns must use
``aggregate_gpu_v3.py``. This script retains the historical provider status
and R8ORB01 corpus workflow so old partial results remain auditable.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import tempfile
from typing import Any


SHARDS = 1024
EXPECTED_RECORDS = 7_343_033_248
EXPECTED_MIDPOINT = 708_153_662
EXPECTED_SELF_COMPLEMENTARY = 435_808
EXPECTED_LABELLED_WEIGHT = 10_139_684_107_326_071_075
EXPECTED_COVERED_WEIGHT = 1 << 64
RESULT_FIELDS = (
    "records",
    "labelled_weight",
    "kernels",
    "covered_weight",
    "right_groups",
    "verified",
    "comparisons",
    "contribution",
)
TIMING_FIELDS = ("gpu", "total", "comparisons_per_second")
STATUS_PATTERN = re.compile(r"s(\d{4})\.status\.json$")


class ValidationError(RuntimeError):
    pass


def validate_expected(expected: dict[int, dict[str, int]], source: str) -> None:
    if set(expected) != set(range(SHARDS)):
        raise ValidationError(f"{source} does not cover shards 0 through 1023")
    totals = {
        name: sum(item[name] for item in expected.values())
        for name in next(iter(expected.values()))
    }
    wanted = {
        "records": EXPECTED_RECORDS,
        "midpoint": EXPECTED_MIDPOINT,
        "self_complementary": EXPECTED_SELF_COMPLEMENTARY,
        "labelled_weight": EXPECTED_LABELLED_WEIGHT,
        "covered_weight": EXPECTED_COVERED_WEIGHT,
    }
    for name, value in wanted.items():
        if totals[name] != value:
            raise ValidationError(f"{source} {name} total {totals[name]} != {value}")


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_fields(line: str, prefix: str, required: tuple[str, ...],
                 value_type: type[int] | type[float]) -> dict[str, int | float]:
    if not line.startswith(prefix + " "):
        raise ValidationError(f"expected {prefix} line, got {line!r}")
    fields: dict[str, int | float] = {}
    for word in line.split()[1:]:
        if "=" not in word:
            continue
        name, value = word.split("=", 1)
        try:
            fields[name] = value_type(value)
        except ValueError as error:
            raise ValidationError(f"invalid {prefix} field {word!r}") from error
    missing = sorted(set(required) - fields.keys())
    if missing:
        raise ValidationError(f"missing {prefix} fields: {', '.join(missing)}")
    return fields


def result_line(result: dict[str, Any]) -> str:
    return "RESULT " + " ".join(f"{name}={result[name]}" for name in RESULT_FIELDS)


def timing_line(timing: dict[str, Any]) -> str:
    return "TIMING " + " ".join(f"{name}={timing[name]}" for name in TIMING_FIELDS)


def checkpoint_expected(checkpoint: dict[str, Any]) -> dict[int, dict[str, int]]:
    checked = checkpoint.get("checked_shards", {})
    if len(checked) != SHARDS:
        raise ValidationError(
            f"checkpoint has {len(checked)} checked shards, expected {SHARDS}"
        )
    expected: dict[int, dict[str, int]] = {}
    for text_shard, item in checked.items():
        shard = int(text_shard)
        if item["shard"] != shard or item["shards"] != SHARDS:
            raise ValidationError(f"checkpoint checker identity mismatch for {shard}")
        expected[shard] = {
            name: int(item[name])
            for name in (
                "records",
                "midpoint",
                "self_complementary",
                "labelled_weight",
                "covered_weight",
            )
        }
    validate_expected(expected, "checkpoint")
    return expected


def expected_tsv(expected: dict[int, dict[str, int]]) -> bytes:
    lines = [
        "shard\trecords\tmidpoint\tself_complementary\tlabelled_weight\tcovered_weight"
    ]
    for shard in range(SHARDS):
        item = expected[shard]
        lines.append(
            "\t".join(str(value) for value in (
                shard,
                item["records"],
                item["midpoint"],
                item["self_complementary"],
                item["labelled_weight"],
                item["covered_weight"],
            ))
        )
    return ("\n".join(lines) + "\n").encode()


def load_expected(path: Path) -> dict[int, dict[str, int]]:
    lines = path.read_text().splitlines()
    header = (
        "shard\trecords\tmidpoint\tself_complementary\tlabelled_weight\tcovered_weight"
    )
    if not lines or lines[0] != header:
        raise ValidationError(f"invalid expected metadata header in {path}")
    names = header.split("\t")
    expected: dict[int, dict[str, int]] = {}
    for line in lines[1:]:
        values = line.split("\t")
        if len(values) != len(names):
            raise ValidationError(f"invalid expected metadata row {line!r}")
        item = dict(zip(names, map(int, values)))
        shard = item.pop("shard")
        if shard in expected:
            raise ValidationError(f"duplicate expected shard {shard}")
        expected[shard] = item
    validate_expected(expected, str(path))
    return expected


def export_modal(checkpoint: dict[str, Any], results_root: Path) -> int:
    destination = results_root / "modal"
    atomic_write(destination / "campaign-checkpoint.json", json_bytes(checkpoint))
    exported = 0
    for text_shard, item in sorted(
        checkpoint.get("solved", {}).items(), key=lambda pair: int(pair[0])
    ):
        shard = int(text_shard)
        result = {name: int(item["result"][name]) for name in RESULT_FIELDS}
        timing = {
            "gpu": float(item["timing"]["gpu_seconds"]),
            "total": float(item["timing"]["total_seconds"]),
            "comparisons_per_second": float(
                item["timing"]["comparisons_per_second"]
            ),
        }
        result_text = result_line(result)
        timing_text = timing_line(timing)
        log = (
            f"MODAL_CHECKPOINT_EXPORT shard={shard}\n"
            f"{result_text}\n{timing_text}\n"
        ).encode()
        checksum = hashlib.sha256(log).hexdigest()
        status = {
            "schema_version": 1,
            "shard": shard,
            "provider": "modal",
            "state": "complete",
            "source": "modal_checkpoint",
            "result_sha256": checksum,
            "result": result_text,
            "timing": timing_text,
        }
        stem = f"s{shard:04d}"
        atomic_write(destination / f"{stem}.log", log)
        atomic_write(destination / f"{stem}.status.json", json_bytes(status))
        exported += 1
    return exported


def read_solve_header(path: Path) -> tuple[bytes, int, int]:
    with path.open("rb") as input_file:
        header = input_file.read(20)
    if len(header) != 20:
        raise ValidationError(f"truncated solve header: {path}")
    return struct.unpack("<8sIQ", header)


def validate_status(status_path: Path, solve_root: Path,
                    expected: dict[int, dict[str, int]],
                    input_hashes: dict[int, str] | None) -> dict[str, Any]:
    match = STATUS_PATTERN.fullmatch(status_path.name)
    if not match:
        raise ValidationError(f"invalid status filename: {status_path}")
    filename_shard = int(match.group(1))
    status = json.loads(status_path.read_text())
    shard = int(status["shard"])
    if shard != filename_shard or not 0 <= shard < SHARDS:
        raise ValidationError(f"shard identity mismatch in {status_path}")
    if status.get("state") != "complete":
        raise ValidationError(f"non-complete status in {status_path}")
    provider = status.get("provider") or status_path.parent.name
    if provider != status_path.parent.name:
        raise ValidationError(f"provider/directory mismatch in {status_path}")
    stem = f"s{shard:04d}"
    candidates = [
        status_path.with_name(f"{stem}.log"),
        status_path.with_name(f"{stem}.result"),
    ]
    artifacts = [path for path in candidates if path.is_file()]
    if len(artifacts) != 1:
        raise ValidationError(
            f"shard {shard} has {len(artifacts)} result artifacts, expected one"
        )
    artifact_path = artifacts[0]
    artifact = artifact_path.read_bytes()
    checksum = hashlib.sha256(artifact).hexdigest()
    claimed_checksum = status.get("result_sha256")
    warnings: list[str] = []
    if claimed_checksum is None:
        warnings.append("status_missing_result_sha256")
    elif claimed_checksum != checksum:
        raise ValidationError(f"result SHA-256 mismatch for shard {shard}")
    artifact_lines = artifact.decode().splitlines()
    result_lines = [line for line in artifact_lines if line.startswith("RESULT ")]
    timing_lines = [line for line in artifact_lines if line.startswith("TIMING ")]
    if len(result_lines) != 1 or len(timing_lines) != 1:
        raise ValidationError(
            f"shard {shard} does not have exactly one RESULT and TIMING line"
        )
    result_text = status.get("result")
    if result_text is None:
        warnings.append("status_missing_result_line")
        result_text = result_lines[0]
    elif result_lines[0] != result_text:
        raise ValidationError(f"status/result line mismatch for shard {shard}")
    timing_text = status.get("timing")
    if timing_text is None:
        warnings.append("status_missing_timing_line")
        timing_text = timing_lines[0]
    elif timing_lines[0] != timing_text:
        raise ValidationError(f"status/timing line mismatch for shard {shard}")
    result = {
        name: int(value)
        for name, value in parse_fields(
            result_text, "RESULT", RESULT_FIELDS, int
        ).items()
    }
    timing = {
        name: float(value)
        for name, value in parse_fields(
            timing_text, "TIMING", TIMING_FIELDS, float
        ).items()
    }
    wanted = expected[shard]
    for name in ("records", "labelled_weight", "covered_weight"):
        if result[name] != wanted[name]:
            raise ValidationError(
                f"shard {shard} {name} {result[name]} != {wanted[name]}"
            )
    if result["kernels"] != wanted["records"] or result["verified"] != 4:
        raise ValidationError(f"shard {shard} kernel/validation mismatch")
    solve_path = solve_root / f"s{shard:04d}.orbits"
    magic, columns, records = read_solve_header(solve_path)
    if magic != b"R8ORB01\0" or columns != 8 or records != wanted["records"]:
        raise ValidationError(f"solve header mismatch for shard {shard}")
    expected_size = 20 + 16 * records
    if solve_path.stat().st_size != expected_size:
        raise ValidationError(f"solve file size mismatch for shard {shard}")
    input_checksum = status.get("input_sha256")
    input_checksum_verified: bool | None = None
    if input_checksum is not None and input_hashes is not None:
        if input_hashes[shard] != input_checksum:
            raise ValidationError(f"solve input SHA-256 mismatch for shard {shard}")
        input_checksum_verified = True
    source_status_path = None
    source_status_checksum = None
    if status.get("source_status"):
        source_status_name = status["source_status"]
        if Path(source_status_name).name != source_status_name:
            raise ValidationError(f"invalid source-status path for shard {shard}")
        source_status_path = status_path.parent / source_status_name
        source_status = json.loads(source_status_path.read_text())
        if int(source_status.get("shard", -1)) != shard:
            raise ValidationError(f"source-status identity mismatch for shard {shard}")
        if "exit_code" in source_status and int(source_status["exit_code"]) != 0:
            raise ValidationError(f"source-status failure for shard {shard}")
        source_log_checksum = source_status.get("log_sha256")
        if source_log_checksum is None:
            warnings.append("source_status_missing_log_sha256")
        elif source_log_checksum != checksum:
            raise ValidationError(f"source-status log mismatch for shard {shard}")
        for name in ("input_sha256", "solver_sha256"):
            if source_status.get(name) is not None and status.get(name) is not None:
                if source_status[name] != status[name]:
                    raise ValidationError(
                        f"source-status {name} mismatch for shard {shard}"
                    )
        source_status_checksum = file_sha256(source_status_path)
    return {
        "shard": shard,
        "provider": provider,
        "status": str(status_path),
        "status_sha256": file_sha256(status_path),
        "artifact": str(artifact_path),
        "sha256": checksum,
        "sha256_claimed": claimed_checksum is not None,
        "input_sha256": input_checksum,
        "input_sha256_verified": input_checksum_verified,
        "solver_sha256": status.get("solver_sha256"),
        "source_status": str(source_status_path) if source_status_path else None,
        "source_status_sha256": source_status_checksum,
        "warnings": warnings,
        "result": result,
        "timing": timing,
    }


def relative_to_results(path: str, results_root: Path) -> str:
    return str(Path(path).resolve().relative_to(results_root.resolve()))


def build_input_hashes(statuses: list[Path], solve_root: Path) -> dict[int, str]:
    claimed: set[int] = set()
    for status_path in statuses:
        status = json.loads(status_path.read_text())
        if status.get("input_sha256") is not None:
            match = STATUS_PATTERN.fullmatch(status_path.name)
            if match is None:
                raise ValidationError(f"invalid status filename: {status_path}")
            claimed.add(int(match.group(1)))

    def hash_shard(shard: int) -> tuple[int, str]:
        return shard, file_sha256(solve_root / f"s{shard:04d}.orbits")

    workers = min(8, len(claimed))
    if workers == 0:
        return {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(hash_shard, sorted(claimed)))


def performance_summary(entries: dict[str, Any]) -> dict[str, Any]:
    def new_accumulator() -> dict[str, int | float]:
        return {
            "result_count": 0,
            "comparisons": 0,
            "gpu_seconds": 0.0,
            "solver_seconds": 0.0,
        }

    def add(accumulator: dict[str, int | float], comparisons: int,
            source: dict[str, Any]) -> None:
        accumulator["result_count"] += 1
        accumulator["comparisons"] += comparisons
        accumulator["gpu_seconds"] += source["timing"]["gpu"]
        accumulator["solver_seconds"] += source["timing"]["total"]

    def finish(accumulator: dict[str, int | float]) -> dict[str, int | float]:
        gpu_seconds = float(accumulator["gpu_seconds"])
        solver_seconds = float(accumulator["solver_seconds"])
        result_count = int(accumulator["result_count"])
        result = dict(accumulator)
        result.update({
            "gpu_hours": gpu_seconds / 3600.0,
            "solver_hours": solver_seconds / 3600.0,
            "non_gpu_seconds": solver_seconds - gpu_seconds,
            "non_gpu_hours": (solver_seconds - gpu_seconds) / 3600.0,
            "mean_gpu_seconds": gpu_seconds / result_count if result_count else 0.0,
            "mean_solver_seconds": (
                solver_seconds / result_count if result_count else 0.0
            ),
            "weighted_comparisons_per_second": (
                int(accumulator["comparisons"]) / gpu_seconds
                if gpu_seconds else 0.0
            ),
        })
        return result

    all_sources = new_accumulator()
    deduplicated = new_accumulator()
    by_provider: dict[str, dict[str, int | float]] = {}
    for entry in entries.values():
        comparisons = int(entry["result"]["comparisons"])
        sources = entry["sources"]
        add(deduplicated, comparisons, sources[0])
        for source in sources:
            add(all_sources, comparisons, source)
            provider = source["provider"]
            if provider not in by_provider:
                by_provider[provider] = new_accumulator()
            add(by_provider[provider], comparisons, source)
    return {
        "scope": (
            "successful result artifacts only; sums concurrent compute and excludes "
            "failed attempts, provider startup, transfer, idle time, and storage"
        ),
        "deduplicated_source_policy": (
            "first source sorted by provider and status path for each unique shard"
        ),
        "all_sources": finish(all_sources),
        "deduplicated_shards": finish(deduplicated),
        "by_provider": {
            provider: finish(values)
            for provider, values in sorted(by_provider.items())
        },
    }


def build_manifest(results_root: Path, solve_root: Path,
                   expected: dict[int, dict[str, int]],
                   verify_input_sha256: bool = False) -> dict[str, Any]:
    by_shard: dict[int, list[dict[str, Any]]] = {}
    statuses = sorted(results_root.glob("*/s[0-9][0-9][0-9][0-9].status.json"))
    input_hashes = (
        build_input_hashes(statuses, solve_root) if verify_input_sha256 else None
    )
    for status_path in statuses:
        item = validate_status(status_path, solve_root, expected, input_hashes)
        by_shard.setdefault(item["shard"], []).append(item)
    entries: dict[str, Any] = {}
    for shard, sources in sorted(by_shard.items()):
        reference = sources[0]["result"]
        for source in sources[1:]:
            if source["result"] != reference:
                providers = ", ".join(item["provider"] for item in sources)
                raise ValidationError(
                    f"conflicting results for shard {shard}: {providers}"
                )
        entries[str(shard)] = {
            "result": reference,
            "sources": [
                {
                    "provider": source["provider"],
                    "status": relative_to_results(source["status"], results_root),
                    "status_sha256": source["status_sha256"],
                    "artifact": relative_to_results(
                        source["artifact"], results_root
                    ),
                    "sha256": source["sha256"],
                    "sha256_claimed": source["sha256_claimed"],
                    "input_sha256": source["input_sha256"],
                    "input_sha256_verified": source["input_sha256_verified"],
                    "solver_sha256": source["solver_sha256"],
                    "source_status": (
                        relative_to_results(source["source_status"], results_root)
                        if source["source_status"] else None
                    ),
                    "source_status_sha256": source["source_status_sha256"],
                    "warnings": source["warnings"],
                    "timing": source["timing"],
                }
                for source in sorted(
                    sources, key=lambda item: (item["provider"], item["status"])
                )
            ],
        }
    completed = sorted(by_shard)
    missing = sorted(set(range(SHARDS)) - set(completed))
    totals = {
        name: sum(entries[str(shard)]["result"][name] for shard in completed)
        for name in RESULT_FIELDS
    }
    complete = len(completed) == SHARDS
    if complete:
        wanted = {
            "records": EXPECTED_RECORDS,
            "kernels": EXPECTED_RECORDS,
            "labelled_weight": EXPECTED_LABELLED_WEIGHT,
            "covered_weight": EXPECTED_COVERED_WEIGHT,
            "verified": 4 * SHARDS,
        }
        for name, value in wanted.items():
            if totals[name] != value:
                raise ValidationError(f"complete {name} total {totals[name]} != {value}")
    provider_counts: dict[str, int] = {}
    manifest_warnings: list[dict[str, Any]] = []
    input_claims = 0
    input_verified = 0
    result_checksum_claims = 0
    source_count = 0
    for shard, sources in sorted(by_shard.items()):
        for source in sources:
            source_count += 1
            provider = source["provider"]
            provider_counts[provider] = provider_counts.get(provider, 0) + 1
            result_checksum_claims += int(source["sha256_claimed"])
            input_claims += int(source["input_sha256"] is not None)
            input_verified += int(source["input_sha256_verified"] is True)
            for warning in source["warnings"]:
                manifest_warnings.append({
                    "provider": provider,
                    "shard": shard,
                    "warning": warning,
                })
    return {
        "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "geometry": "8x8",
        "quantity": "T_4(8,8)",
        "expected_shards": SHARDS,
        "complete": complete,
        "source_count": source_count,
        "duplicate_source_count": source_count - len(completed),
        "provider_counts": provider_counts,
        "completed_count": len(completed),
        "completed_shards": completed,
        "missing_shards": missing,
        "validation": {
            "result_sha256_claims": result_checksum_claims,
            "input_sha256_claims": input_claims,
            "input_sha256_verified": input_verified,
            "full_input_verification_requested": verify_input_sha256,
        },
        "warnings": manifest_warnings,
        "performance": performance_summary(entries),
        "totals": totals,
        "entries": entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data_root", type=Path, nargs="?", default=Path("../rectangle-free-data-v2")
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path(".rect8x8-production-checkpoint.json")
    )
    parser.add_argument("--export-modal", action="store_true")
    parser.add_argument(
        "--verify-input-sha256", action="store_true",
        help="hash every solve input carrying a provider SHA-256 claim",
    )
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()

    results_root = args.data_root / "results"
    solve_root = args.data_root / "solve"
    expected_path = results_root / "expected.tsv"
    checkpoint = None
    if args.checkpoint.exists():
        checkpoint = json.loads(args.checkpoint.read_text())
        expected = checkpoint_expected(checkpoint)
        atomic_write(expected_path, expected_tsv(expected))
    else:
        expected = load_expected(expected_path)
    exported = 0
    if args.export_modal:
        if checkpoint is None:
            raise ValidationError("--export-modal requires a checkpoint")
        exported = export_modal(checkpoint, results_root)
    manifest = build_manifest(
        results_root, solve_root, expected, args.verify_input_sha256
    )
    if args.write_manifest:
        atomic_write(results_root / "manifest.json", json_bytes(manifest))
    totals = manifest["totals"]
    print(
        f"verified={manifest['completed_count']}/{SHARDS} "
        f"missing={len(manifest['missing_shards'])} exported_modal={exported} "
        f"sources={manifest['source_count']} warnings={len(manifest['warnings'])} "
        f"providers={json.dumps(manifest['provider_counts'], sort_keys=True, separators=(',', ':'))}"
    )
    print(
        f"partial_records={totals['records']} comparisons={totals['comparisons']} "
        f"contribution={totals['contribution']} "
        f"covered_weight={totals['covered_weight']}"
    )
    performance = manifest["performance"]
    print(
        f"source_gpu_hours={performance['all_sources']['gpu_hours']:.6f} "
        f"source_solver_hours={performance['all_sources']['solver_hours']:.6f} "
        f"unique_gpu_hours={performance['deduplicated_shards']['gpu_hours']:.6f} "
        f"weighted_comparisons_per_second="
        f"{performance['deduplicated_shards']['weighted_comparisons_per_second']:.6f}"
    )
    if manifest["complete"]:
        print(f"T_4(8,8)={totals['contribution']} COMPLETE")


if __name__ == "__main__":
    main()
