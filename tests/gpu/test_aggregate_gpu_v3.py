#!/usr/bin/env python3

import hashlib
from pathlib import Path
import struct
import tempfile
import unittest

from tools.aggregate_gpu_v3 import (
    GEOMETRIES,
    Geometry,
    ValidationError,
    WorkItem,
    aggregate_campaign,
    exact_coverage,
    read_manifest,
)


def write_orbits(path: Path, records: int = 4) -> None:
    path.write_bytes(
        struct.pack("<8sIQ", b"R8SQT01\0", 8, records)
        + b"\0" * (16 * records)
    )


def write_result(
    path: Path,
    item: WorkItem,
    records: int,
    configuration: str = "b" * 64,
) -> None:
    fields = [
        f"id {item.id}",
        f"path {item.path}",
        f"start {item.start}",
        f"end {item.end}",
        f"filter_mod {item.filter_mod}",
        f"filter_id {item.filter_id}",
        "geometry 8x8",
        "token_plane_quotient 1",
        f"solver_binary_sha256 {'a' * 64}",
        f"solver_configuration_sha256 {configuration}",
        f"canonical_cache_sha256 {'c' * 64}",
        f"orbit_corpus_sha256 {hashlib.sha256(Path(item.path).read_bytes()).hexdigest()}",
        "transpose_quotient 1",
        f"records {records}",
        f"labelled_weight {records}",
        f"kernels {records}",
        f"covered_weight {2 * records}",
        "verified 1",
        f"direct_comparisons {100 * records}",
        f"contribution {1000 * records}",
        "minimum_free_bytes 1024",
        "gpu_seconds 0.5",
        "total_seconds 1.0",
    ]
    payload = ("\n".join(fields) + "\n").encode()
    path.write_bytes(
        b"RECT8X8_PREFIX_RESULT 3\n"
        + payload
        + f"result_payload_sha256 {hashlib.sha256(payload).hexdigest()}\n".encode()
    )


class AggregateGpuV3Test(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.corpus = self.root / "s0000.orbits"
        self.results = self.root / "results"
        self.results.mkdir()
        write_orbits(self.corpus)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_manifest(self, items: list[WorkItem]) -> Path:
        path = self.root / "work.tsv"
        path.write_text(
            "".join(
                f"{item.id} {item.path} {item.start} {item.end}"
                + (
                    f" {item.filter_mod} {item.filter_id}"
                    if item.filter_mod
                    else ""
                )
                + "\n"
                for item in items
            )
        )
        return path

    def two_ranges(self) -> tuple[list[WorkItem], Path]:
        items = [
            WorkItem("a", str(self.corpus), 0, 2),
            WorkItem("b", str(self.corpus), 2, 0),
        ]
        return items, self.write_manifest(items)

    def test_valid_partial_campaign_and_exact_coverage(self) -> None:
        items, manifest = self.two_ranges()
        write_result(self.results / "a.result", items[0], 2)
        write_result(self.results / "b.result", items[1], 2)
        report = aggregate_campaign(
            GEOMETRIES["8x8"], manifest, self.results, verify_inputs=True
        )
        self.assertEqual(report["completed_items"], 2)
        self.assertEqual(report["totals"]["records"], 4)
        self.assertEqual(report["coverage"]["records"], 4)
        self.assertEqual(report["verified_input_files"], 1)

    def test_interrupted_campaign_becomes_full_exact_campaign(self) -> None:
        items, manifest = self.two_ranges()
        write_result(self.results / "a.result", items[0], 2)
        partial = aggregate_campaign(GEOMETRIES["8x8"], manifest, self.results)
        self.assertEqual(partial["missing_items"], ["b"])

        write_result(self.results / "b.result", items[1], 2)
        tiny = Geometry(
            "8x8",
            "test",
            "RECT8X8_PREFIX_RESULT",
            b"R8SQT01\0",
            8,
            4,
            4,
            8,
            4000,
            True,
        )
        complete = aggregate_campaign(tiny, manifest, self.results, full=True)
        self.assertFalse(complete["full_check_failures"])
        self.assertTrue(complete["complete"])

    def test_missing_result_is_reported(self) -> None:
        items, manifest = self.two_ranges()
        write_result(self.results / "a.result", items[0], 2)
        report = aggregate_campaign(GEOMETRIES["8x8"], manifest, self.results)
        self.assertEqual(report["missing_items"], ["b"])

    def test_checksum_corruption_is_rejected(self) -> None:
        items, manifest = self.two_ranges()
        write_result(self.results / "a.result", items[0], 2)
        with (self.results / "a.result").open("ab") as output:
            output.write(b"tampered 1\n")
        with self.assertRaisesRegex(ValidationError, "checksum"):
            aggregate_campaign(GEOMETRIES["8x8"], manifest, self.results)

    def test_mixed_configuration_is_rejected(self) -> None:
        items, manifest = self.two_ranges()
        write_result(self.results / "a.result", items[0], 2)
        write_result(self.results / "b.result", items[1], 2, "d" * 64)
        with self.assertRaisesRegex(ValidationError, "mixed solver configuration"):
            aggregate_campaign(GEOMETRIES["8x8"], manifest, self.results)

    def test_unmanifested_result_is_rejected(self) -> None:
        items, manifest = self.two_ranges()
        write_result(self.results / "a.result", items[0], 2)
        write_result(self.results / "extra.result", items[0], 2)
        with self.assertRaisesRegex(ValidationError, "unmanifested"):
            aggregate_campaign(GEOMETRIES["8x8"], manifest, self.results)

    def test_overlap_is_rejected(self) -> None:
        items = [
            WorkItem("a", str(self.corpus), 0, 3),
            WorkItem("b", str(self.corpus), 2, 0),
        ]
        with self.assertRaisesRegex(ValidationError, "gap or overlap"):
            exact_coverage(items, GEOMETRIES["8x8"], None)

    def test_empty_range_is_rejected(self) -> None:
        items = [WorkItem("a", str(self.corpus), 2, 2)]
        with self.assertRaisesRegex(ValidationError, "empty range"):
            exact_coverage(items, GEOMETRIES["8x8"], None)

    def test_complete_filter_partition_is_accepted(self) -> None:
        items = [
            WorkItem("a", str(self.corpus), 0, 0, 2, 0),
            WorkItem("b", str(self.corpus), 0, 0, 2, 1),
        ]
        coverage, _ = exact_coverage(items, GEOMETRIES["8x8"], None)
        self.assertEqual(coverage["filtered_segments"], 1)

    def test_zero_filter_owner_is_parsed(self) -> None:
        item = WorkItem("a", str(self.corpus), 0, 0, 2, 0)
        manifest = self.write_manifest([item])
        self.assertEqual(read_manifest(manifest), [item])

    def test_incomplete_filter_partition_is_rejected(self) -> None:
        items = [WorkItem("a", str(self.corpus), 0, 0, 2, 0)]
        with self.assertRaisesRegex(ValidationError, "filter owners"):
            exact_coverage(items, GEOMETRIES["8x8"], None)


if __name__ == "__main__":
    unittest.main()
