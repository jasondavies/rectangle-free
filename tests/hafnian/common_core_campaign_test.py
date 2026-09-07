#!/usr/bin/env python3
"""Local workflow tests: durable restart, strict provenance, exact reduction.

No timing from these CPU tests is a GPU benchmark. Optional --real-plan runs
the real persistent arithmetic worker on a bounded 6x28 shared group twice.
"""
import argparse
import io
import json
import os
from pathlib import Path
import sqlite3
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import common_core_campaign as cc


def identity():
    return dict(format=cc.FORMAT, catalog="catalog", plan="plan", backend="cpu-reference",
                solver_binary="binary", controller="controller", configuration=cc.CONFIG,
                primes=list(cc.PRIMES), group_start=0, group_end=2)


class Workflow(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "worker.sqlite"
        self.meta = dict(domain=7, bounds=[65, 25], primes=[3, 1])

    def tearDown(self):
        self.directory.cleanup()

    def test_restart_tail_chunk_and_tamper(self):
        journal = cc.Journal(self.path, identity())
        journal.put_group(0, self.meta)
        journal.put_range(0, 0, 0, 3, self.meta, [12, 34], 1.25, 1.5)
        journal.close()
        journal = cc.Journal(self.path, identity())
        self.assertEqual(list(cc.gaps(journal.ranges(0, 0, self.meta), 7, 3)), [(3, 6), (6, 7)])
        journal.put_range(0, 0, 3, 6, self.meta, [56, 78], 1., 1.)
        journal.put_range(0, 0, 6, 7, self.meta, [90, 12], .3, .4)
        self.assertEqual(list(cc.gaps(journal.ranges(0, 0, self.meta), 7, 3)), [])
        with journal.db:
            journal.db.execute("UPDATE ranges SET residues=? WHERE begin=3", (struct.pack("<II", 0, 0),))
        with self.assertRaisesRegex(ValueError, "checksum"):
            list(journal.ranges(0, 0, self.meta))
        journal.close()

    def test_changed_provenance(self):
        cc.Journal(self.path, identity()).close()
        for field in ("catalog", "plan", "solver_binary", "controller", "configuration", "backend"):
            changed = identity()
            changed[field] = "different"
            with self.assertRaisesRegex(ValueError, "provenance"):
                cc.Journal(self.path, changed)

    def test_overlap_within_and_across_journals(self):
        journal = cc.Journal(self.path, identity())
        journal.put_group(0, self.meta)
        journal.put_range(0, 0, 0, 4, self.meta, [0, 0], 0., 0.)
        journal.put_range(0, 0, 3, 7, self.meta, [0, 0], 0., 0.)
        with self.assertRaisesRegex(ValueError, "overlapping"):
            list(journal.ranges(0, 0, self.meta))
        journal.close()
        with self.assertRaisesRegex(ValueError, "overlapping"):
            cc.reduce_group(self.meta, [[(0, 7, [0, 0]), (0, 7, [0, 0])], [], [], []],
                            [(1 << 60) | 7] * 2, 2)

    def test_exact_crt_normalization_and_missing_prime(self):
        # One real-shaped key: one defect of size3, hence three dummy vertices
        # at slack2. Normalize shared signs by domain*3!, not child sign count.
        key = (1 << 60) | 7
        meta = dict(domain=8, bounds=[65], primes=[3])
        count = (1 << 61) + 123456789
        records = [[] for _ in cc.PRIMES]
        for pi, prime in enumerate(cc.PRIMES[:3]):
            residue = count * 8 * 6 % prime
            records[pi] = [(0, 3, [91]), (3, 8, [(residue - 91) % prime])]
        self.assertEqual(cc.reduce_group(meta, records, [key], 2), [count])
        records[2] = []
        self.assertEqual(cc.reduce_group(meta, records, [key], 2), [None])
        # A CRT reconstruction inside the modulus but outside the bound fails.
        records = [[(0, 8, [((1 << 65) + 1) * 48 % p])] for p in cc.PRIMES[:3]] + [[]]
        with self.assertRaisesRegex(ValueError, "bound"):
            cc.reduce_group(meta, records, [key], 2)

    def test_claim_is_exclusive(self):
        with cc.claim(self.path):
            code = ("import fcntl,sys; f=open(sys.argv[1],'a'); "
                    "fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)")
            result = subprocess.run([sys.executable, "-c", code, str(self.path) + ".lock"], capture_output=True)
            self.assertNotEqual(result.returncode, 0)
        with cc.claim(self.path):
            pass

    def test_killed_transaction_does_not_publish(self):
        journal = cc.Journal(self.path, identity())
        journal.put_group(0, self.meta)
        journal.close()
        code = """
import os,sqlite3,sys
d=sqlite3.connect(sys.argv[1]);d.execute('BEGIN IMMEDIATE')
d.execute('INSERT INTO ranges VALUES(0,0,0,7,?,0,0,?)',(b'bad','bad'))
os._exit(19)
"""
        result = subprocess.run([sys.executable, "-c", code, str(self.path)])
        self.assertEqual(result.returncode, 19)
        journal = cc.Journal(self.path, identity())
        self.assertEqual(list(journal.ranges(0, 0, self.meta)), [])
        journal.close()

    def test_batch_visibility_cap_and_acknowledgement(self):
        journal = cc.Journal(self.path, identity())
        acknowledgements = []
        def ack(count):
            reader = cc.Journal(self.path)
            acknowledgements.append((count, reader.db.execute("SELECT count(*) FROM ranges").fetchone()[0]))
            reader.close()
        with journal.batch(2, 100, ack):
            journal.put_group(0, self.meta)
            journal.put_range(0, 0, 0, 3, self.meta, [12, 34], 1., 1.)
            self.assertEqual(len(list(journal.ranges(0, 0, self.meta))), 1)
            reader = cc.Journal(self.path)
            self.assertIsNone(reader.group(0))  # no premature publication
            reader.close()
            journal.put_group(1, self.meta)
            journal.put_range(1, 0, 0, 3, self.meta, [56, 78], 1., 1.)
            self.assertEqual(acknowledgements, [(2, 2)])
            journal.put_range(1, 0, 3, 7, self.meta, [90, 12], 1., 1.)
        self.assertEqual(acknowledgements, [(2, 2), (1, 3)])
        journal.close()

    def test_batch_age_and_exception_rollback(self):
        journal = cc.Journal(self.path, identity())
        with patch.object(cc.time, "monotonic", return_value=10.) as clock:
            with journal.batch(32, 1.):
                journal.put_group(0, self.meta)
                journal.put_range(0, 0, 0, 3, self.meta, [1, 2], 0., 0.)
                clock.return_value = 10.9
                journal.flush_if_due()
                self.assertEqual(journal.committed_ranges, 0)
                clock.return_value = 11.
                journal.flush_if_due()
                self.assertEqual(journal.committed_ranges, 1)
        with self.assertRaisesRegex(RuntimeError, "abort"):
            with journal.batch():
                journal.put_group(1, self.meta)
                journal.put_range(1, 0, 0, 3, self.meta, [1, 2], 0., 0.)
                raise RuntimeError("abort")
        self.assertIsNone(journal.group(1))
        self.assertEqual(len(list(journal.ranges(0, 0, self.meta))), 1)
        journal.close()

    def test_sigkill_batch_then_resume(self):
        code = """
import os,signal,sys
sys.path.insert(0,sys.argv[1]);import common_core_campaign as cc
import json
j=cc.Journal(sys.argv[2],json.loads(sys.argv[3]));m=json.loads(sys.argv[4])
with j.batch(1,100):
 j.put_group(0,m);j.put_range(0,0,0,3,m,[1,2],0.,0.)
with j.batch(32,100):
 j.put_group(1,m);j.put_range(1,0,0,3,m,[3,4],0.,0.)
 os.kill(os.getpid(),signal.SIGKILL)
"""
        child = subprocess.run([sys.executable, "-c", code, str(ROOT / "tools"), str(self.path),
                                json.dumps(identity()), json.dumps(self.meta)])
        self.assertEqual(child.returncode, -9)
        journal = cc.Journal(self.path, identity())
        self.assertIsNone(journal.group(1))
        self.assertEqual(list(cc.gaps(journal.ranges(0, 0, self.meta), 7, 4)), [(3, 7)])
        with journal.batch(2, 100):  # a different policy is safe on resume
            journal.put_range(0, 0, 3, 7, self.meta, [5, 6], 0., 0.)
            journal.put_group(1, self.meta)
            journal.put_range(1, 0, 0, 7, self.meta, [7, 8], 0., 0.)
        self.assertEqual(list(cc.gaps(journal.ranges(0, 0, self.meta), 7, 4)), [])
        self.assertEqual(list(cc.gaps(journal.ranges(1, 0, self.meta), 7, 4)), [])
        journal.close()

    def test_commit_failure_never_acknowledges(self):
        journal = cc.Journal(self.path, identity())
        acknowledgements = []
        connection = journal.db
        class FailedCommit:
            def __getattr__(self, name):
                return getattr(connection, name)
            def commit(self):
                raise sqlite3.OperationalError("injected commit failure")
        journal.db = FailedCommit()
        with self.assertRaisesRegex(sqlite3.OperationalError, "injected"):
            with journal.batch(1, 100, acknowledgements.append):
                journal.put_group(0, self.meta)
                journal.put_range(0, 0, 0, 7, self.meta, [1, 2], 0., 0.)
        self.assertEqual(acknowledgements, [])
        self.assertIsNone(journal.group(0))
        journal.close()

    def test_complete_reducer_includes_independent_tail(self):
        class Catalog:
            slack, count, digest = 2, 3, "catalog"
            keys = [4755944160245056391, 4755944160245056907, 0]
            def __getitem__(self, query):
                key = self.keys[query]
                return key, [5, 7, 11][query], cc.prime_count(cc.bound_power(key, 2))
        class Plan:
            digest = "plan"
            def groups(self):
                yield 0, 144258141817667969, 282031183369742, [(0, 518), (1, 1034)]
                yield 1, 0, 0, [(2, 0)]
        catalog, plan = Catalog(), Plan()
        answers = [42, 17, 9]
        journal = cc.Journal(self.path, identity())
        for gid, parent, boundary, members in plan.groups():
            meta = cc.expected_meta(catalog, parent, boundary, members)
            journal.put_group(gid, meta)
            for pi in range(max(meta["primes"])):
                values = []
                for (query, _), required in zip(members, meta["primes"]):
                    if required > pi:
                        key = catalog[query][0]
                        unmatched = 4 - ((key & cc.FULL).bit_count() - 2*(key >> 60))
                        values.append(answers[query]*meta["domain"]*cc.math.factorial(unmatched) % cc.PRIMES[pi])
                journal.put_range(gid, pi, 0, meta["domain"], meta, values, 0., 0.)
        journal.close()
        output = io.StringIO()
        with redirect_stdout(output):
            cc.reduce(argparse.Namespace(journals=[self.path], cpu_reference=True,
                      query_results=False, require_complete=True), catalog, plan, {})
        summary = json.loads(output.getvalue())
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["complete_queries"], 3)
        self.assertEqual(summary["pending_independent_queries"], 0)
        want = cc.math.factorial(28)*sum(catalog[q][1]*answers[q]*(1 << (28-(catalog[q][0]>>60))) for q in range(3))
        self.assertEqual(int(summary["partial_labelled_count"]), want)


def real_plan():
    catalog_path = ROOT / "build/common-core-6x28.catalog"
    plan_path = ROOT / "build/common-core-6x28-order50-481.plan"
    catalog = cc.Catalog(catalog_path)
    plan = cc.Plan(plan_path, catalog)
    try:
        audit = plan.audit()
        group, _, _, members = next(g for g in plan.groups() if len(g[3]) > 1)
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(group_start=group, group_end=group + 1,
                chunk_terms=2, checkpoint_terms=3, max_checkpoints=1,
                journal=Path(directory) / "real.sqlite", cpu_reference=True,
                worker=ROOT / "build/hafnian_common_worker_host")
            cc.run(args, catalog, plan, audit)
            cc.run(args, catalog, plan, audit)
            journal = cc.Journal(args.journal)
            meta = journal.group(group)
            rows = list(journal.ranges(group, 0, meta))
            assert [(a, b) for a, b, _ in rows] == [(0, 3), (3, 6)]
            # Fresh persistent process, same six signs in one call, all values
            # match the sum across the deliberately stopped/restarted process.
            _, parent, boundary, members = next(g for g in plan.groups() if g[0] == group)
            worker = cc.Worker(args.worker, True)
            try:
                worker.request(f"prepare {catalog.slack} {parent} {boundary} {len(members)} " +
                    " ".join(f"{catalog[q][0]} {removed}" for q, removed in members))
                response = worker.request("run 0 0 6 4")
                want = [(a + b) % cc.PRIMES[0] for a, b in zip(rows[0][2], rows[1][2])]
                assert list(map(int, response[3:])) == want
            finally:
                worker.close()
                journal.close()
            output = io.StringIO()
            with redirect_stdout(output):
                cc.reduce(argparse.Namespace(journals=[args.journal], cpu_reference=True,
                          query_results=False, require_complete=False), catalog, plan, audit)
            summary = json.loads(output.getvalue().splitlines()[-1])
            assert summary["status"] == "partial" and summary["complete_queries"] == 0
            assert summary["pending_independent_queries"] == audit["singletons"]
            tail_gid = next(gid for gid, _, _, children in plan.groups() if len(children) == 1)
            args.group_start, args.group_end = tail_gid, tail_gid + 1
            args.journal = Path(directory) / "tail.sqlite"
            cc.run(args, catalog, plan, audit)
            cc.run(args, catalog, plan, audit)
            tail_journal = cc.Journal(args.journal)
            tail_meta = tail_journal.group(tail_gid)
            assert [(a, b) for a, b, _ in tail_journal.ranges(tail_gid, 0, tail_meta)] == [(0, 3), (3, 6)]
            assert tail_meta["domain"] == 1 << 31  # root 6x28 has 64 vertices
            tail_journal.close()
        print("COMMON_CORE_CAMPAIGN real_6x28_group=OK process_restart=OK chunk_parity=OK incomplete_rejected=OK")
    finally:
        plan.close()
        catalog.close()


def archived_parity(journal_path, archive):
    # Compare normalized complete child residues against independently
    # calculated full-matrix production results, not another shared-core run.
    from reduce_six_by_twenty_eight_hafnian import read_result
    catalog = cc.Catalog(ROOT / "build/common-core-6x28.catalog")
    plan = cc.Plan(ROOT / "build/common-core-6x28-order50-481.plan", catalog)
    journal = cc.Journal(journal_path)
    try:
        identity = journal.identity
        assert identity["catalog"] == catalog.digest and identity["plan"] == plan.digest
        members = {}
        groups = []
        for gid, parent, boundary, children in plan.groups():
            if not identity["group_start"] <= gid < identity["group_end"]:
                continue
            meta = cc.expected_meta(catalog, parent, boundary, children)
            assert journal.group(gid) == meta
            records = [list(journal.ranges(gid, pi, meta)) for pi in range(4)]
            keys = [catalog[q][0] for q, _ in children]
            answers = cc.reduce_group(meta, records, keys, catalog.slack)
            assert None not in answers
            for key, answer in zip(keys, answers):
                members[key] = answer
            groups.append(gid)
        found = {}
        for path in archive.rglob("p2147483647-q*-b0.result"):
            fields = dict(line.split(" ", 1) for line in path.read_text().splitlines())
            key = int(fields["occupied_tokens"]) | (int(fields["defect_count"]) << 60)
            if key in members:
                found[key] = int(fields["query_id"])
                if len(found) == len(members):
                    break
        assert len(found) == len(members), "missing archived query metadata"
        checks = 0
        for key, query in found.items():
            required = cc.prime_count(cc.bound_power(key, catalog.slack))
            residues = []
            for prime in cc.PRIMES[:required]:
                path = next(archive.rglob(f"p{prime}-q{query:05d}-b0.result"))
                fields = read_result(path)
                assert int(fields["begin"]) == 0 and fields["end"] == fields["total_terms"]
                assert int(fields["occupied_tokens"]) | (int(fields["defect_count"]) << 60) == key
                divisor = int(fields["total_terms"]) * cc.math.factorial(int(fields["unmatched_tokens"]))
                residue = int(fields["partial_glynn_sum"]) * pow(divisor, -1, prime) % prime
                assert members[key] % prime == residue
                residues.append(residue)
                checks += 1
            assert cc.crt(residues)[0] == members[key]
        print(f"COMMON_CORE_CAMPAIGN groups={groups} complete_queries={len(members)} "
              f"archived_prime_residues={checks} exact_matching_counts=OK")
    finally:
        journal.close()
        plan.close()
        catalog.close()


if __name__ == "__main__":
    if "--real-plan" in sys.argv:
        real_plan()
    elif "--archived-parity" in sys.argv:
        archived_parity(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        unittest.main()
