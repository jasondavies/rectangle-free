#!/usr/bin/env python3
"""Aggregate independently checkpointed packed 7x9 solver results."""

import argparse
import hashlib
from pathlib import Path


EXPECTED_RECORDS = 3_608_247_685
EXPECTED_KERNELS = 3_608_247_685
EXPECTED_LABELLED_WEIGHT = 1 << 62
EXPECTED_COVERED_WEIGHT = 1 << 63


def read_manifest(path: Path):
    items = []
    ids = set()
    with path.open() as source:
        for line_number, line in enumerate(source, 1):
            fields = line.split()
            if not fields or fields[0].startswith("#"):
                continue
            if len(fields) not in (4, 6):
                raise ValueError(f"invalid manifest line {line_number}")
            item = {
                "id": fields[0],
                "path": fields[1],
                "start": int(fields[2]),
                "end": int(fields[3]),
                "filter_mod": int(fields[4]) if len(fields) == 6 else 0,
                "filter_id": int(fields[5]) if len(fields) == 6 else 0,
            }
            if item["id"] in ids:
                raise ValueError(f"duplicate manifest id {item['id']}")
            ids.add(item["id"])
            items.append(item)
    if not items:
        raise ValueError("manifest is empty")
    return items


def read_result(path: Path, item):
    with path.open() as source:
        header = source.readline().rstrip("\n")
        if header not in ("RECT7X9_PACKED_RESULT 2", "RECT7X9_PACKED_RESULT 3"):
            raise ValueError(f"invalid result header: {path}")
        version = int(header.rsplit(maxsplit=1)[1])
        fields = {}
        payload = bytearray()
        payload_digest = None
        for line in source:
            key, value = line.split(maxsplit=1)
            if version == 3 and key == "result_payload_sha256":
                if payload_digest is not None:
                    raise ValueError(f"duplicate result checksum: {path}")
                payload_digest = value.strip()
                continue
            if version == 3:
                payload.extend(line.encode())
            if key in fields:
                raise ValueError(f"duplicate field {key}: {path}")
            fields[key] = value.strip()
    if version == 3 and (
        payload_digest is None
        or hashlib.sha256(payload).hexdigest() != payload_digest
    ):
        raise ValueError(f"result checksum mismatch: {path}")
    identity = ("id", "path", "start", "end", "filter_mod", "filter_id")
    for key in identity:
        if key not in fields or str(item[key]) != fields[key]:
            raise ValueError(f"result identity mismatch for {key}: {path}")
    integer_fields = (
        "records",
        "labelled_weight",
        "kernels",
        "covered_weight",
        "left_prefixes",
        "left_entries",
        "left_buckets",
        "right_prefixes",
        "right_batches",
        "maximum_right_entries",
        "maximum_right_buckets",
        "effective_right_entry_cap",
        "verified",
        "direct_comparisons",
        "contribution",
        "minimum_free_bytes",
    )
    timing_fields = (
        "load_seconds",
        "left_factory_seconds",
        "left_layout_seconds",
        "right_layout_seconds",
        "gpu_seconds",
        "validation_seconds",
        "total_seconds",
    )
    for key in integer_fields:
        fields[key] = int(fields[key])
    for key in timing_fields:
        fields[key] = float(fields[key])
    fields["format_version"] = version
    if version == 3:
        for key in (
            "geometry",
            "token_plane_quotient",
            "solver_binary_sha256",
            "solver_configuration_sha256",
            "canonical_cache_sha256",
            "orbit_corpus_sha256",
        ):
            if key not in fields:
                raise ValueError(f"missing provenance field {key}: {path}")
        if fields["geometry"] != "7x9" or fields["token_plane_quotient"] != "1":
            raise ValueError(f"unsupported exact representation: {path}")
    return fields


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument(
        "--full",
        action="store_true",
        help="require the complete 7x9 record/kernel/weight gates",
    )
    args = parser.parse_args()

    items = read_manifest(args.manifest)
    results = []
    missing = []
    for item in items:
        path = args.results_dir / f"{item['id']}.result"
        if not path.exists():
            missing.append(item["id"])
            continue
        results.append(read_result(path, item))

    summed = {}
    for key in (
        "records",
        "labelled_weight",
        "kernels",
        "covered_weight",
        "left_prefixes",
        "left_entries",
        "left_buckets",
        "right_prefixes",
        "right_batches",
        "verified",
        "direct_comparisons",
        "contribution",
    ):
        summed[key] = sum(result[key] for result in results)
    for key in (
        "load_seconds",
        "left_factory_seconds",
        "left_layout_seconds",
        "right_layout_seconds",
        "gpu_seconds",
        "validation_seconds",
        "total_seconds",
    ):
        summed[key] = sum(result[key] for result in results)
    minimum_free = min(
        (result["minimum_free_bytes"] for result in results), default=0
    )
    maximum_right_entries = max(
        (result["maximum_right_entries"] for result in results), default=0
    )
    maximum_right_buckets = max(
        (result["maximum_right_buckets"] for result in results), default=0
    )
    effective_right_caps = {
        result["effective_right_entry_cap"] for result in results
    }
    formats = {result["format_version"] for result in results}
    provenance_sets = {
        key: {result[key] for result in results if result["format_version"] == 3}
        for key in (
            "solver_binary_sha256",
            "solver_configuration_sha256",
            "canonical_cache_sha256",
        )
    }
    for key in ("solver_configuration_sha256", "canonical_cache_sha256"):
        if len(provenance_sets[key]) > 1:
            raise ValueError(f"mixed result provenance for {key}")

    print(
        f"AGGREGATE completed={len(results)} expected={len(items)} "
        f"missing={len(missing)} records={summed['records']} "
        f"labelled_weight={summed['labelled_weight']} kernels={summed['kernels']} "
        f"covered_weight={summed['covered_weight']} "
        f"left_prefixes={summed['left_prefixes']} "
        f"left_entries={summed['left_entries']} "
        f"left_buckets={summed['left_buckets']} "
        f"right_prefixes={summed['right_prefixes']} "
        f"right_batches={summed['right_batches']} "
        f"maximum_right_entries={maximum_right_entries} "
        f"maximum_right_buckets={maximum_right_buckets} "
        f"effective_right_entry_caps={','.join(map(str, sorted(effective_right_caps)))} "
        f"direct_comparisons={summed['direct_comparisons']} "
        f"contribution={summed['contribution']} minimum_free={minimum_free}"
    )
    print(
        f"PROVENANCE formats={','.join(map(str, sorted(formats)))} "
        + " ".join(
            f"{key}="
            f"{next(iter(values), 'legacy-v2') if len(values) <= 1 else str(len(values)) + '-variants'}"
            for key, values in provenance_sets.items()
        )
    )
    print(
        f"TIMING gpu_seconds={summed['gpu_seconds']:.6f} "
        f"solver_seconds={summed['total_seconds']:.6f} "
        f"left_factory_seconds={summed['left_factory_seconds']:.6f} "
        f"left_layout_seconds={summed['left_layout_seconds']:.6f} "
        f"right_layout_seconds={summed['right_layout_seconds']:.6f} "
        f"validation_seconds={summed['validation_seconds']:.6f}"
    )
    if missing:
        print("MISSING " + " ".join(missing))
    if args.full:
        if missing:
            raise SystemExit("FULL_CHECK FAIL: results are missing")
        gates = (
            summed["records"] == EXPECTED_RECORDS
            and summed["kernels"] == EXPECTED_KERNELS
            and summed["labelled_weight"] == EXPECTED_LABELLED_WEIGHT
            and summed["covered_weight"] == EXPECTED_COVERED_WEIGHT
        )
        print(
            f"FULL_CHECK expected_records={EXPECTED_RECORDS} "
            f"expected_kernels={EXPECTED_KERNELS} "
            f"expected_labelled_weight={EXPECTED_LABELLED_WEIGHT} "
            f"expected_covered_weight={EXPECTED_COVERED_WEIGHT} "
            f"{'OK' if gates else 'FAIL'}"
        )
        if not gates:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
