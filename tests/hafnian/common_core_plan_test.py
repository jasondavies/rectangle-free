#!/usr/bin/env python3
"""Small exact catalog/plan round trip and adversarial integrity checks."""
import hashlib
from pathlib import Path
import struct
import subprocess
import tempfile

root = Path(__file__).resolve().parents[2]
planner = root / "build/hafnian_common_core_plan"
census = root / "build/six_by_twenty_eight_defect_census"


def run(args, good=True):
    result = subprocess.run([str(x) for x in args], capture_output=True, text=True)
    if (result.returncode == 0) != good:
        raise AssertionError(result.stdout + result.stderr)
    return result.stdout


with tempfile.TemporaryDirectory(prefix="common-core-plan-") as tmp:
    base = Path(tmp)
    catalog, plan = base / "catalog", base / "plan"
    run([census, "--slack", 1, "--threads", 2, "--export-catalog", catalog])
    run([planner, "--catalog", catalog, "--output", plan])
    text = run([planner, "--catalog", catalog, "--verify", plan, "--all-maps"])
    assert "queries=29" in text and "exact_once=OK" in text
    run([planner, "--catalog", catalog, "--output", plan], good=False)  # exclusive create
    original = plan.read_bytes()
    bad = base / "bad-plan"
    bad.write_bytes(original[:-1])
    run([planner, "--catalog", catalog, "--verify", bad], good=False)
    # Header: magic + four u64 + catalog digest. Singleton records: three
    # u64 header fields plus one (id, removed) member. Resign the deliberately
    # duplicated ID: semantic checks, not the checksum, must reject it.
    payload = bytearray(original[:-64])
    first = 8 + 4 * 8 + 64
    payload[first + 40 + 24:first + 40 + 32] = payload[first + 24:first + 32]
    bad.write_bytes(payload + hashlib.sha256(payload).hexdigest().encode())
    run([planner, "--catalog", catalog, "--verify", bad], good=False)
    # A perfectly checksummed but incomplete catalog must also fail.
    raw = bytearray(catalog.read_bytes()[:-64])
    struct.pack_into("<Q", raw, 16, 28)
    bad_catalog = base / "bad-catalog"
    bad_catalog.write_bytes(raw + hashlib.sha256(raw).hexdigest().encode())
    run([planner, "--catalog", bad_catalog, "--verify", plan], good=False)
    catalog2, plan1, plan2 = base / "catalog2", base / "plan1", base / "plan2"
    run([census, "--slack", 2, "--threads", 2, "--export-catalog", catalog2])
    run([planner, "--catalog", catalog2, "--output", plan1, "--threads", 1])
    run([planner, "--catalog", catalog2, "--output", plan2, "--threads", 4])
    assert plan1.read_bytes() == plan2.read_bytes(), "ownership depends on thread count"
    assert "maps=all" in run([planner, "--catalog", catalog2, "--verify", plan2, "--all-maps"])
print("CORE_PLAN_TEST roundtrip=OK no_overwrite=OK truncation=OK duplicate_id=OK incomplete_catalog=OK parallel_determinism=OK")
