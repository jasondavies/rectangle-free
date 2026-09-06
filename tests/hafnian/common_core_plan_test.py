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
    # The explicit order-50 gate must still produce complete exact ownership.
    wide1, wide2 = base / "wide1", base / "wide2"
    for target, threads in ((wide1, 1), (wide2, 4)):
        run([planner, "--catalog", catalog2, "--output", target,
             "--group-max-order", 50, "--threads", threads])
        assert "maps=all" in run([planner, "--catalog", catalog2, "--verify", target, "--all-maps"])
    assert wide1.read_bytes() == wide2.read_bytes()
    run([planner, "--catalog", catalog2, "--output", base / "unsupported-order",
         "--group-max-order", 52], good=False)
    # Synthetic costs exercise ownership-preserving repair, not a runtime
    # estimate. All grouped shapes in the small complete catalog are covered.
    costs = base / "costs"
    with costs.open("w") as out:
        out.write("HCCOST01 hess=1 boundary=1 scratch=1\n")
        for n in (42, 44, 46, 48):
            for q in (5, 7, 9, 11):
                # Odd sizes interpolate; one-child later CRT images clamp
                # below the smallest measured-size endpoint.
                for g in range(2, 257, 2):
                    for p in range(4):
                        out.write(f"{n} {n-q+3} {q} {g} {p} {250000+3000*g} 1\n")
    repaired1, repaired2 = base / "repair1", base / "repair2"
    for target, threads in ((repaired1, 1), (repaired2, 4)):
        text = run([planner, "--catalog", catalog2, "--output", target,
                    "--replan", plan1, "--costs", costs, "--repair-anchors", 2,
                    "--threads", threads])
        summary = next(l for l in text.splitlines() if l.startswith("CORE_REPLAN_DONE "))
        fields = dict(x.split("=", 1) for x in summary.split()[1:])
        assert float(fields["model_after_h"]) <= float(fields["model_before_h"])
        assert "maps=all" in run([planner, "--catalog", catalog2, "--verify", target, "--all-maps"])
    assert repaired1.read_bytes() == repaired2.read_bytes(), "repair depends on producer scheduling"
    run([planner, "--catalog", catalog2, "--output", base / "lower-cap", "--replan", plan1,
         "--costs", costs, "--cap", 7], good=False)
    bad_costs = base / "bad-costs"
    bad_costs.write_text("HCCOST01 hess=0 boundary=1 scratch=1\n")
    run([planner, "--catalog", catalog2, "--output", base / "bad-repair", "--replan", plan1,
         "--costs", bad_costs], good=False)
print("CORE_PLAN_TEST roundtrip=OK no_overwrite=OK truncation=OK duplicate_id=OK incomplete_catalog=OK parallel_determinism=OK repair=OK repair_determinism=OK model_rejection=OK")
