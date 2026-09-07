#!/usr/bin/env python3
"""Checkpointed shared-core and independent residual-hafnian campaign.

One journal and persistent worker per GPU. Journals are ordinary SQLite files
using rollback journals and FULL synchronous transactions, not loose per-query
files. Stop the writer or use SQLite's backup API before copying a live journal.
Neither a partial group nor the shared-only total is a completed grid result.
"""
import argparse
from array import array
from contextlib import contextmanager
from functools import lru_cache
import fcntl
import hashlib
import heapq
import json
import math
import mmap
import os
from pathlib import Path
import sqlite3
import struct
import subprocess
import time

PRIMES = (2147483647, 2147483629, 2147483587, 2147483579)
FULL = (1 << 60) - 1
FORMAT = "common-core-campaign-v2"
CONFIG = "exp484-p11-o16-i1-l1-c1-w1-s1-h1-b1-a1-t128-tail1"
PAIR = [(i, j) for i in range(6) for j in range(i + 1, 6)]
NEIGHBOURS = [sum(1 << b for b in range(60) if a // 15 != b // 15 and
                 not (set(PAIR[a % 15]) & set(PAIR[b % 15]))) for a in range(60)]
FACT_POWERS = [(math.factorial(i) - 1).bit_length() for i in range(19)]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(encoded(value).encode()).hexdigest()


def file_digest(path):
    with open(path, "rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


class Artifact:
    def __init__(self, path, magic):
        self.path = Path(path).absolute()
        self.file = open(path, "rb")
        self.stamp = self._stamp(os.fstat(self.file.fileno()))
        self.data = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        require(len(self.data) >= 88 and self.data[:8] == magic, "unknown/truncated artifact")
        h = hashlib.sha256()
        for begin in range(0, len(self.data) - 64, 8 << 20):
            h.update(self.data[begin:min(begin + (8 << 20), len(self.data) - 64)])
        self.digest = h.hexdigest()
        require(self.data[-64:] == self.digest.encode(), "artifact checksum mismatch")
        self.assert_unchanged()

    @staticmethod
    def _stamp(stat):
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    def assert_unchanged(self):
        require(self._stamp(os.fstat(self.file.fileno())) == self.stamp and
                self._stamp(self.path.stat()) == self.stamp, "artifact changed since validation")

    def u64(self, offset):
        require(offset + 8 <= len(self.data) - 64, "truncated artifact record")
        return struct.unpack_from("<Q", self.data, offset)[0]

    def close(self):
        self.data.close()
        self.file.close()


class Catalog(Artifact):
    def __init__(self, path):
        super().__init__(path, b"HCCAT001")
        self.slack, self.count = self.u64(8), self.u64(16)
        require(self.slack in (1, 2, 3) and
                self.count == (0, 29, 36398, 45007139)[self.slack], "incomplete catalog census")
        require(len(self.data) == 24 + 17 * self.count + 64, "wrong catalog length")

    def __getitem__(self, query):
        require(0 <= query < self.count, "query ID outside catalog")
        return struct.unpack_from("<QQB", self.data, 24 + 17 * query)


class Plan(Artifact):
    def __init__(self, path, catalog):
        super().__init__(path, b"HCPLAN01")
        self.catalog = catalog
        require(self.u64(8) == catalog.slack and self.u64(32) == catalog.count and
                self.data[40:104] == catalog.digest.encode(), "plan/catalog identity mismatch")
        require(self.u64(16) in (7, 9, 11), "unsupported production plan cap")
        self._offsets = None
        self._audit = None

    def groups(self, start=0, end=None, *, _index=None):
        position, group = 104, 0
        if self._offsets is not None:
            self.assert_unchanged()
            require(0 <= start <= len(self._offsets) and
                    (end is None or start <= end <= len(self._offsets)), "invalid indexed group range")
            if start == len(self._offsets):
                return
            position, group = self._offsets[start], start
        else:
            require(start == 0 and end is None, "indexed access requires a completed audit")
        while True:
            if end is not None and group >= end:
                return
            record_start = position
            parent = self.u64(position)
            position += 8
            if parent == (1 << 64) - 1:
                require(self.u64(position) == group and position + 8 == len(self.data) - 64,
                        "invalid plan group count/trailer")
                return
            boundary, count = self.u64(position), self.u64(position + 8)
            position += 16
            require(1 <= count <= 165, "unsupported group size")
            members = []
            for _ in range(count):
                members.append((self.u64(position), self.u64(position + 8)))
                position += 16
            if _index is not None:
                _index.append(record_start)
            yield group, parent, boundary, members
            group += 1

    def audit(self):
        self.assert_unchanged()
        self.catalog.assert_unchanged()
        if self._audit is not None:
            return dict(self._audit)
        offsets = array('Q')
        seen = bytearray(self.catalog.count)
        queries = groups = singletons = coefficient = 0
        for gid, parent, boundary, members in self.groups(_index=offsets):
            groups = gid + 1
            if len(members) == 1:
                require(parent == boundary == members[0][1] == 0, "invalid independent group")
                singletons += 1
            else:
                require(parent != 0 and not ((parent | boundary) & ~FULL) and not (parent & boundary)
                        and boundary.bit_count() in (5, 7, 9, 11), "invalid group geometry")
            sector = None
            for query, removed in members:
                key, weight, images = self.catalog[query]
                d, used = key >> 60, (key & FULL).bit_count()
                require(not seen[query], "duplicate query ownership")
                seen[query] = 1
                require(weight > 0 and 1 <= images <= 4 and d <= 2 * self.catalog.slack
                        and 0 <= used - 2 * d <= 2 * self.catalog.slack, "invalid catalog entry")
                if len(members) > 1:
                    require(removed.bit_count() == 3 and not (removed & ~boundary) and
                            (parent | removed).bit_count() == used, "bad member embedding shape")
                    require(sector is None or sector == (d, used), "mixed defect sectors")
                    sector = (d, used)
                    core = 60 - (parent | boundary).bit_count() + 2 * self.catalog.slack - (used - 2 * d)
                    require(0 <= core <= 48 and core % 2 == 0 and
                            core + boundary.bit_count() <= 64, "unsupported core")
                queries += 1
                coefficient += weight
        require(queries == self.catalog.count, "incomplete query ownership")
        self.assert_unchanged()
        self.catalog.assert_unchanged()
        # Full canonical row-map equality is checked by the worker before each
        # group is evaluated. This preflight checks shape, integrity and coverage.
        self._audit = dict(groups=groups, queries=queries, singletons=singletons,
                    coefficient_sum=coefficient, plan_sha256=self.digest,
                    catalog_sha256=self.catalog.digest)
        self._offsets = offsets
        return dict(self._audit)


def bound_power(key, slack):
    remaining = FULL ^ (key & FULL)
    unmatched = 2 * slack - ((key & FULL).bit_count() - 2 * (key >> 60))
    numerator, scan = 0, remaining
    while scan:
        bit = scan & -scan
        scan ^= bit
        degree = (NEIGHBOURS[bit.bit_length() - 1] & remaining).bit_count()
        if degree:
            numerator += FACT_POWERS[degree] * (24504480 // (2 * degree))
    return ((numerator + 24504479) // 24504480 +
            (math.comb(remaining.bit_count(), unmatched) - 1).bit_length())


def prime_count(bound):
    product = 1
    for count, prime in enumerate(PRIMES, 1):
        product *= prime
        if product > 1 << bound:
            return count
    raise ValueError("insufficient certified CRT range")


@contextmanager
def claim(path):
    with open(str(path) + ".lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


class Journal:
    def __init__(self, path, identity=None):
        self._batch = None
        self.pending_ranges = 0
        self._pending_since = None
        self.committed_ranges = 0
        if identity is None:
            self.db = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
            self.db.execute("BEGIN")  # stable snapshot even during writer commits
        else:
            self.db = sqlite3.connect(path)
            self.db.execute("PRAGMA journal_mode=DELETE")
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS header (id INTEGER PRIMARY KEY CHECK(id=1), value TEXT, hash TEXT);
                CREATE TABLE IF NOT EXISTS groups (gid INTEGER PRIMARY KEY, value TEXT, hash TEXT);
                CREATE TABLE IF NOT EXISTS ranges (gid INTEGER, pi INTEGER, begin INTEGER, end INTEGER,
                    residues BLOB, seconds REAL, wall REAL, hash TEXT, PRIMARY KEY(gid,pi,begin));
            """)
            if self.db.execute("SELECT count(*) FROM header").fetchone()[0] == 0:
                with self.db:
                    self.db.execute("INSERT INTO header VALUES(1,?,?)", (encoded(identity), digest(identity)))
        require(self.db.execute("PRAGMA quick_check").fetchone()[0] == "ok", "damaged SQLite journal")
        value, self.hash = self.db.execute("SELECT value,hash FROM header WHERE id=1").fetchone()
        self.identity = json.loads(value)
        require(digest(self.identity) == self.hash, "journal header checksum mismatch")
        if identity is not None:
            require(self.identity == identity, "journal provenance/work range mismatch")
        require(0 <= self.identity["group_start"] < self.identity["group_end"], "invalid journal ownership")
        require(not self.db.execute("SELECT 1 FROM groups WHERE gid<? OR gid>=? LIMIT 1",
                    (self.identity["group_start"], self.identity["group_end"])).fetchone(),
                "group outside journal ownership")
        require(not self.db.execute("SELECT 1 FROM ranges r LEFT JOIN groups g ON r.gid=g.gid "
                    "WHERE g.gid IS NULL OR r.pi<0 OR r.pi>=4 LIMIT 1").fetchone(),
                "orphan result or invalid prime index")

    def group(self, gid):
        row = self.db.execute("SELECT value,hash FROM groups WHERE gid=?", (gid,)).fetchone()
        if row is None:
            return None
        value = self._decode_group(gid, row)
        require(not self.db.execute("SELECT 1 FROM ranges WHERE gid=? AND pi>=? LIMIT 1",
                                   (gid, max(value["primes"]))).fetchone(), "unrequested prime image")
        return value

    def _decode_group(self, gid, row):
        value, checksum = json.loads(row[0]), row[1]
        require(checksum == digest([self.hash, gid, value]), "group metadata checksum mismatch")
        return value

    def _decode_range(self, gid, pi, meta, row, last):
        begin, end, blob, seconds, wall, checksum = row
        active = sum(count > pi for count in meta["primes"])
        require(active and len(blob) == 4 * active, "result child count mismatch")
        values = list(struct.unpack("<" + "I" * active, blob))
        require(0 <= begin < end <= meta["domain"] and begin >= last, "overlapping/invalid sign ranges")
        require(all(0 <= x < PRIMES[pi] for x in values), "residue outside field")
        require(math.isfinite(seconds) and math.isfinite(wall) and seconds >= 0 and wall >= 0,
                "invalid elapsed time")
        require(checksum == digest([self.hash, gid, pi, begin, end, meta, values, seconds, wall]),
                "result payload checksum mismatch")
        return begin, end, values

    def ordered_groups(self):
        """Two ordered SQL cursors, bounded to one group's checked payloads.

        The read-only journal holds a stable transaction snapshot. No per-group
        SQL lookups; the same payload decoder is used by ordinary resume reads.
        """
        groups = self.db.execute("SELECT gid,value,hash FROM groups ORDER BY gid")
        ranges = iter(self.db.execute("SELECT gid,pi,begin,end,residues,seconds,wall,hash "
                                     "FROM ranges ORDER BY gid,pi,begin"))
        row = next(ranges, None)
        for gid, value, checksum in groups:
            meta = self._decode_group(gid, (value, checksum))
            records = [[] for _ in PRIMES]
            last = [0] * len(PRIMES)
            require(row is None or row[0] >= gid, "orphan result")
            while row is not None and row[0] == gid:
                pi = row[1]
                require(0 <= pi < max(meta["primes"]) and pi < len(PRIMES), "unrequested prime image")
                record = self._decode_range(gid, pi, meta, row[2:], last[pi])
                records[pi].append(record)
                last[pi] = record[1]
                row = next(ranges, None)
            yield gid, meta, records
        require(row is None, "orphan result")

    def put_group(self, gid, value):
        previous = self.group(gid)
        if previous is not None:
            require(previous == value, "prepared group changed on resume")
        else:
            self._insert("INSERT INTO groups VALUES(?,?,?)",
                         (gid, encoded(value), digest([self.hash, gid, value])))

    def ranges(self, gid, pi, meta):
        rows = self.db.execute("SELECT begin,end,residues,seconds,wall,hash FROM ranges "
                               "WHERE gid=? AND pi=? ORDER BY begin", (gid, pi))
        last = 0
        for row in rows:
            record = self._decode_range(gid, pi, meta, row, last)
            last = record[1]
            yield record

    def put_range(self, gid, pi, begin, end, meta, values, seconds, wall):
        checksum = digest([self.hash, gid, pi, begin, end, meta, values, seconds, wall])
        self._insert("INSERT INTO ranges VALUES(?,?,?,?,?,?,?,?)", (gid, pi, begin, end,
                     struct.pack("<" + "I" * len(values), *values), seconds, wall, checksum), is_range=True)

    def _insert(self, sql, parameters, is_range=False):
        if self._batch is None:
            with self.db:
                self.db.execute(sql, parameters)
            self.committed_ranges += int(is_range)
            return
        if self._pending_since is None:
            self._pending_since = time.monotonic()
        self.db.execute(sql, parameters)
        self.pending_ranges += int(is_range)
        if is_range:
            self.flush_if_due()

    def flush_if_due(self):
        if self._batch is not None and self._pending_since is not None:
            ranges, seconds, _ = self._batch
            if self.pending_ranges >= ranges or time.monotonic()-self._pending_since >= seconds:
                self.flush()

    def flush(self):
        if self._batch is not None and self.db.in_transaction:
            count = self.pending_ranges
            self.db.commit()  # acknowledgements must follow the FULL-sync commit
            self.pending_ranges = 0
            self._pending_since = None
            self.committed_ranges += count
            if self._batch[2] is not None:
                self._batch[2](count)

    @contextmanager
    def batch(self, max_ranges=32, max_seconds=1., on_commit=None):
        """Atomic group/range batches. Normal exit flushes; errors roll back.

        Time is checked between compute requests, so crash exposure is at most
        the interval plus one in-flight request (not a hard real-time deadline).
        Leave batching explicit for callers that require immediate durability.
        """
        require(self._batch is None and not self.db.in_transaction, "nested/read-only transaction batch")
        require(1 <= max_ranges <= 4096 and math.isfinite(max_seconds) and max_seconds > 0,
                "invalid commit batch limits")
        # Do not acquire an early EXCLUSIVE lock by spilling dirty pages while
        # the GPU is running. The range cap bounds the in-memory transaction.
        spill = self.db.execute("PRAGMA cache_spill").fetchone()[0]
        self.db.execute("PRAGMA cache_spill=OFF")
        self._batch = (max_ranges, max_seconds, on_commit)
        try:
            yield self
            self.flush()
        except BaseException:
            self.db.rollback()
            self.pending_ranges = 0
            self._pending_since = None
            raise
        finally:
            self._batch = None
            self.db.execute(f"PRAGMA cache_spill={int(spill)}")

    def close(self):
        self.db.close()


def gaps(rows, domain, limit):
    cursor = 0
    for begin, end, _ in [*rows, (domain, domain, None)]:
        require(begin >= cursor and end >= begin, "overlapping ranges")
        while cursor < begin:
            stop = min(begin, cursor + limit)
            yield cursor, stop
            cursor = stop
        cursor = end


class Worker:
    def __init__(self, binary, cpu_reference):
        self.binary_hash = file_digest(binary)
        self.process = subprocess.Popen([str(Path(binary).resolve())], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, text=True, bufsize=1)
        try:
            greeting = self.process.stdout.readline().split()
            require(greeting == ["HCCWORKER2", "cpu-reference" if cpu_reference else "cuda",
                                 CONFIG, self.binary_hash], "worker backend/build handshake mismatch")
        except BaseException:
            self.close()
            raise
        self.backend = greeting[1]

    def request(self, line):
        self.process.stdin.write(line + "\n")
        self.process.stdin.flush()
        response = self.process.stdout.readline().split()
        require(response, "worker failed; current checkpoint was not committed")
        return response

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
        self.process.wait()


def expected_meta(catalog, parent, boundary, members):
    keys = [catalog[q][0] for q, _ in members]
    bounds = [bound_power(key, catalog.slack) for key in keys]
    primes = [prime_count(b) for b in bounds]
    require(primes == [catalog[q][2] for q, _ in members], "uncertified catalog prime schedule")
    key = keys[0]
    unmatched = 2 * catalog.slack - ((key & FULL).bit_count() - 2 * (key >> 60))
    # A singleton uses the full independent sign domain. Shared children use
    # the smaller common-core domain; these must never be interchanged.
    core = (60 - (key & FULL).bit_count() + unmatched if len(members) == 1 else
            60 - (parent | boundary).bit_count() + unmatched)
    require(core % 2 == 0 and 0 <= core <= (66 if len(members) == 1 else 48), "invalid sign domain")
    return dict(domain=1 << max(0, core // 2 - 1), bounds=bounds, primes=primes)


def run(args, catalog, plan, audit, worker=None):
    end = args.group_end if args.group_end is not None else audit["groups"]
    require(0 <= args.group_start < end <= audit["groups"], "invalid group interval")
    require(0 < args.chunk_terms <= 1 << 20 and args.checkpoint_terms > 0 and args.max_checkpoints >= 0,
            "invalid chunk/checkpoint limit")
    commit_ranges = getattr(args, "commit_ranges", 32)
    commit_seconds = getattr(args, "commit_seconds", 1.)
    require(1 <= commit_ranges <= 4096 and math.isfinite(commit_seconds) and commit_seconds > 0,
            "invalid commit batch limits")
    args.journal.parent.mkdir(parents=True, exist_ok=True)
    own_worker = worker is None
    with claim(args.journal):
        if own_worker:
            worker = Worker(args.worker, args.cpu_reference)
        journal = None
        try:
            identity = dict(format=FORMAT, catalog=catalog.digest, plan=plan.digest,
                            solver_binary=worker.binary_hash, controller=file_digest(__file__),
                            backend=worker.backend, configuration=CONFIG, primes=PRIMES,
                            group_start=args.group_start, group_end=end)
            # JSON round trip fixes the tuple/list representation on resume.
            identity = json.loads(encoded(identity))
            journal = Journal(args.journal, identity)
            def acknowledged(count):
                print(encoded(dict(status="checkpoints_committed", ranges=count,
                                   committed_ranges=journal.committed_ranges)), flush=True)
            with journal.batch(commit_ranges, commit_seconds, acknowledged):
                complete = solve_ranges(args, catalog, plan, end, worker, journal)
            # A deliberate bounded stop and normal completion both flush before
            # reporting success. Exceptions retain only earlier committed batches.
            print(encoded(dict(status="assigned_ranges_complete" if complete else "checkpoint_limit_reached")))
            return complete
        finally:
            if own_worker:
                worker.close()
            if journal:
                journal.close()


def solve_ranges(args, catalog, plan, end, worker, journal):
    completed = 0
    for gid, parent, boundary, members in plan.groups(args.group_start, end):
        journal.flush_if_due()
        meta = expected_meta(catalog, parent, boundary, members)
        stored = journal.group(gid)
        require(stored is None or stored == meta, "stored group differs from plan")
        ready = False
        for pi in range(max(meta["primes"])):
            # Freeze existing rows only; generate future gaps lazily. This sees
            # our own uncommitted rows, so no pending range can be resubmitted.
            todo = gaps(list(journal.ranges(gid, pi, meta)), meta["domain"], args.checkpoint_terms)
            for begin, stop in todo:
                journal.flush_if_due()
                if not ready:
                    if len(members) == 1:
                        message = f"prepare_single {catalog.slack} {catalog[members[0][0]][0]}"
                    else:
                        message = f"prepare {catalog.slack} {parent} {boundary} {len(members)} "
                        message += " ".join(f"{catalog[q][0]} {removed}" for q, removed in members)
                    reply = worker.request(message)
                    expected = ["prepared", str(meta["domain"]), str(len(members))]
                    for bound, count in zip(meta["bounds"], meta["primes"]):
                        expected.extend((str(bound), str(count)))
                    require(reply == expected, "CPU/worker metadata mismatch")
                    journal.put_group(gid, meta)
                    ready = True
                journal.flush_if_due()
                started = time.monotonic()
                reply = worker.request(f"run {pi} {begin} {stop - begin} {args.chunk_terms}")
                active = sum(count > pi for count in meta["primes"])
                require(len(reply) == 3 + active and reply[0] == "result" and int(reply[2]) == active,
                        "malformed worker result")
                seconds, values = float(reply[1]), list(map(int, reply[3:]))
                require(math.isfinite(seconds) and seconds >= 0 and all(0 <= x < PRIMES[pi] for x in values),
                        "invalid worker result")
                journal.put_range(gid, pi, begin, stop, meta, values, seconds, time.monotonic() - started)
                completed += 1
                print(encoded(dict(checkpoint=completed, durable=not journal.db.in_transaction,
                                   group=gid, prime=PRIMES[pi], begin=begin, end=stop,
                                   compute_seconds=seconds)), flush=True)
                if args.max_checkpoints and completed >= args.max_checkpoints:
                    return False
    return True


CRT_MODULI = tuple(math.prod(PRIMES[:i]) for i in range(len(PRIMES) + 1))
CRT_INVERSES = tuple(pow(CRT_MODULI[i], -1, p) for i, p in enumerate(PRIMES))


@lru_cache(maxsize=4096)
def normalization_inverse(domain, unmatched, pi):
    return pow(domain * math.factorial(unmatched), -1, PRIMES[pi])


def crt(values):
    require(len(values) <= len(PRIMES), "too many CRT images")
    value, modulus = 0, 1
    for i, (residue, prime) in enumerate(zip(values, PRIMES)):
        value += modulus * ((residue - value) * CRT_INVERSES[i] % prime)
        modulus = CRT_MODULI[i + 1]
    return value, modulus


def reduce_group(meta, records, keys, slack):
    images = [[] for _ in keys]
    complete = [True] * len(keys)
    for pi in range(max(meta["primes"])):
        active = [j for j, count in enumerate(meta["primes"]) if count > pi]
        cursor, sums = 0, [0] * len(active)
        contiguous = True
        for begin, end, values in sorted(records[pi]):
            require(begin >= cursor, "overlapping ranges across journals")
            contiguous &= begin == cursor
            cursor = end
            sums = [(a + b) % PRIMES[pi] for a, b in zip(sums, values)]
        contiguous &= cursor == meta["domain"]
        for column, j in enumerate(active):
            complete[j] &= contiguous
            key = keys[j]
            unmatched = 2 * slack - ((key & FULL).bit_count() - 2 * (key >> 60))
            # Shared signs enumerate the common core, NOT the child's full
            # augmented matrix. Normalize by that exact core sign domain.
            images[j].append(sums[column] * normalization_inverse(meta["domain"], unmatched, pi) % PRIMES[pi])
    answers = []
    for j, image in enumerate(images):
        if not complete[j]:
            answers.append(None)
            continue
        value, modulus = crt(image)
        require(modulus > 1 << meta["bounds"][j] and value <= 1 << meta["bounds"][j],
                "CRT matching count exceeds certified bound")
        answers.append(value)
    return answers


def reduce(args, catalog, plan, audit):
    journals = []
    heap = []
    def advance(index, iterator):
        row = next(iterator, None)
        if row is not None:
            gid, meta, records = row
            heapq.heappush(heap, (gid, index, meta, records, iterator))
    try:
        baseline = None
        for path in args.journals:
            journal = Journal(path)
            journals.append(journal)
            identity = journal.identity
            require(identity["format"] == FORMAT and identity["plan"] == plan.digest and
                    identity["catalog"] == catalog.digest and identity["configuration"] == CONFIG and
                    identity["primes"] == list(PRIMES), "reducer provenance mismatch")
            require(identity["backend"] == ("cpu-reference" if args.cpu_reference else "cuda"),
                    "cannot mix CPU reference and GPU results")
            common = {k: v for k, v in identity.items() if k not in ("group_start", "group_end")}
            require(baseline is None or baseline == common, "mixed solver/controller provenance")
            baseline = common
            require("groups" not in audit or identity["group_end"] <= audit["groups"],
                    "journal ownership outside plan")
            advance(len(journals)-1, iter(journal.ordered_groups()))
        total = complete = missing = tail = 0
        for gid, parent, boundary, members in plan.groups():
            present = []
            require(not heap or heap[0][0] >= gid, "result group outside plan")
            while heap and heap[0][0] == gid:
                _, index, saved, checked, iterator = heapq.heappop(heap)
                present.append((saved, checked))
                advance(index, iterator)
            if not present:
                if len(members) == 1:
                    tail += 1
                else:
                    missing += len(members)
                continue
            meta = expected_meta(catalog, parent, boundary, members)
            records = [[] for _ in PRIMES]
            for saved, checked in present:
                require(saved == meta, "result metadata differs from catalog")
                for pi in range(max(meta["primes"])):
                    records[pi].extend(checked[pi])
            keys = [catalog[q][0] for q, _ in members]
            values = reduce_group(meta, records, keys, catalog.slack)
            for (query, _), key, value in zip(members, keys, values):
                if value is None:
                    if len(members) == 1:
                        tail += 1
                    else:
                        missing += 1
                else:
                    complete += 1
                    coefficient = catalog[query][1]
                    total += coefficient * (1 << (30 - catalog.slack - (key >> 60))) * value
                    if args.query_results:
                        print(encoded(dict(query_id=query, matching_count=str(value))))
        require(not heap, "result group outside plan")
        summary = dict(status="partial" if missing or tail else "complete", complete_queries=complete,
                       missing_shared_queries=missing, pending_independent_queries=tail,
                       partial_labelled_count=str(total * math.factorial(30 - catalog.slack)),
                       **audit)
        print(encoded(summary))
        if args.require_complete:
            require(not missing and not tail, "campaign is incomplete; no final grid result certified")
    finally:
        for journal in journals:
            journal.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("audit")
    commands.add_parser("tail", help="export independent query IDs/keys/coefficients/CRT schedule")
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--worker", type=Path, required=True)
    run_parser.add_argument("--journal", type=Path, required=True)
    run_parser.add_argument("--group-start", type=int, default=0)
    run_parser.add_argument("--group-end", type=int)
    run_parser.add_argument("--chunk-terms", type=int, default=32768)
    run_parser.add_argument("--checkpoint-terms", type=int, default=1 << 20)
    run_parser.add_argument("--max-checkpoints", type=int, default=0, help="bounded interruption/restart pilot")
    run_parser.add_argument("--commit-ranges", type=int, default=32, help="maximum completed ranges per durable transaction")
    run_parser.add_argument("--commit-seconds", type=float, default=1., help="flush age checked between worker requests")
    run_parser.add_argument("--cpu-reference", action="store_true", help="test backend; never reported as GPU work")
    reducer = commands.add_parser("reduce")
    reducer.add_argument("journals", type=Path, nargs="+")
    reducer.add_argument("--require-complete", action="store_true")
    reducer.add_argument("--query-results", action="store_true")
    reducer.add_argument("--cpu-reference", action="store_true")
    args = parser.parse_args()
    catalog = Catalog(args.catalog)
    plan = None
    try:
        plan = Plan(args.plan, catalog)
        audit = plan.audit()
        if args.command == "audit":
            print(encoded(audit))
        elif args.command == "run":
            run(args, catalog, plan, audit)
        elif args.command == "reduce":
            reduce(args, catalog, plan, audit)
        else:
            print(encoded(dict(format="common-core-independent-tail-v1", **audit)))
            for gid, _, _, members in plan.groups():
                if len(members) == 1:
                    query = members[0][0]
                    key, coefficient, count = catalog[query]
                    print(encoded(dict(group_id=gid, query_id=query, key=key,
                                       coefficient=coefficient, primes=PRIMES[:count])))
    finally:
        if plan:
            plan.close()
        catalog.close()


if __name__ == "__main__":
    main()
