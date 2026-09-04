#!/usr/bin/env python3
"""Independent CPU range gate and historical full-query comparison for 6x29."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import re
import subprocess

import reduce_six_by_twenty_nine_optimized as current
import reduce_six_by_twenty_nine_hafnian as historical


def gate(binary, cpu, directory, gpus, control=None):
    directory.mkdir(parents=True, exist_ok=True)
    tasks = []
    for query in current.CATALOG:
        for prime in current.PRIMES:
            for begin, end in ((0, 16), (12345, 12473), (query["terms"]-17, query["terms"])):
                path = directory/f"q{query['id']}-p{prime}-b{begin}.result"
                tasks.append((query["id"], prime, begin, end, path))

    def gpu_worker(index):
        batch = directory/f"gpu{index}.batch"
        assigned = tasks[index::len(gpus)]
        batch.write_text("".join(f"{q} {p} {b} {e} {path.resolve()}\n" for q,p,b,e,path in assigned))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus[index])
        with (directory/f"gpu{index}.log").open("w") as log:
            subprocess.run([str(binary), "--batch", str(batch), "--chunk-terms", "64"],
                           env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        list(pool.map(gpu_worker, range(len(gpus))))

    def check(task):
        query, prime, begin, end, path = task
        text = subprocess.check_output([str(cpu), "--query", str(query), "--prime", str(prime),
                   "--begin", str(begin), "--end", str(end), "--threads", "1"], text=True)
        match = re.search(r" residue=(\d+) ", text)
        fields = current.read_result(path)
        if not match or int(match[1]) != int(fields["partial_glynn_sum"]):
            raise ValueError(f"CPU/GPU mismatch: {path}\n{text}")
        return int(fields["gray_fallback_chunks"])
    with ThreadPoolExecutor(max_workers=8) as pool:
        failures = sum(pool.map(check, tasks))
    print(f"CPU_GPU_GATE queries=33 primes=3 ranges=297 fallback_chunks={failures} exact=OK", flush=True)
    if control:
        # Same Gray-index range, with the independent Hessenberg kernel as
        # control. Include every new monomer orbit and all original size classes.
        def benchmark(pair):
            index, query = pair
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus[index % len(gpus)])
            timings = []
            residues = []
            for name, executable in (("control", control), ("gray", binary)):
                path = directory/f"ab-{query}-{name}.result"
                with (directory/f"ab-{query}-{name}.log").open("w") as log:
                    subprocess.run([str(executable), "--run", "--query", str(query),
                        "--prime", str(current.PRIMES[0]), "--begin", "0", "--end", str(1<<20),
                        "--output", str(path)], stdout=log, stderr=subprocess.STDOUT, env=env, check=True)
                fields = dict(line.split(" ", 1) for line in path.read_text().splitlines())
                timings.append(float(fields["elapsed_seconds"]))
                residues.append(int(fields["partial_glynn_sum"]))
            if residues[0] != residues[1]:
                raise ValueError(f"A/B residue mismatch for {query}")
            return {"query": query, "independent_seconds": timings[0], "gray_seconds": timings[1],
                    "speedup": timings[0]/timings[1]}
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            results = list(pool.map(benchmark, enumerate([0,1,2,3,4,5,7,8])))
        (directory/"ab.json").write_text(json.dumps(results, indent=2)+"\n")
        print("AB_GATE " + json.dumps(results), flush=True)
        if sum(r["gray_seconds"] for r in results) > 1.2*sum(r["independent_seconds"] for r in results):
            raise ValueError("Gray A/B aggregate regressed by more than 20%; do not launch production")


def compare(directory, old_directory):
    paths = sorted(directory.glob("*.result"))
    reduced = current.reduce_results(paths)
    expected = 17358733447918084452169454975226757275803964484580835695001600000000
    if int(reduced["T_4(6,29)"]) != expected:
        raise ValueError("complete verification differs from known T_4(6,29)")
    old_jobs = {}
    for path in old_directory.glob("*.result"):
        fields = historical.read_result(path)
        key = int(fields["prime"]), int(fields["query_id"])
        if key[0] in current.PRIMES:
            old_jobs.setdefault(key, []).append(fields)
    counts = {int(k): int(v) for k,v in reduced["matching_counts"].items()}
    checks = 0
    for prime in current.PRIMES:
        for query in range(29):
            pieces = sorted(old_jobs.get((prime, query), []), key=lambda f:int(f["begin"]))
            cursor = residue = 0
            for fields in pieces:
                if int(fields["begin"]) != cursor:
                    raise ValueError("historical range gap/overlap")
                cursor = int(fields["end"])
                residue = (residue+int(fields["partial_glynn_sum"])) % prime
            if not pieces or cursor != int(pieces[0]["total_terms"]):
                raise ValueError(f"historical query {query}, prime {prime} is incomplete")
            matching = residue * pow(cursor, -1, prime) % prime
            matching = matching * pow(math.factorial(int(pieces[0]["unmatched_tokens"])), -1, prime) % prime
            value = (sum(item["coefficient"]*counts[item["id"]] for item in current.CATALOG[:5])
                     if query == 0 else counts[query+4])
            if value % prime != matching:
                raise ValueError(f"historical full-query mismatch {query}, {prime}")
            checks += 1
    reduced["historical_query_prime_checks"] = checks
    (directory.parent/"verification.json").write_text(json.dumps(reduced, indent=2)+"\n")
    print(f"HISTORICAL_GATE checks={checks} T_4(6,29)={expected} exact=OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--cpu", type=Path)
    parser.add_argument("--control", type=Path)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--historical", type=Path)
    args = parser.parse_args()
    if args.historical:
        compare(args.directory, args.historical)
    else:
        gate(args.binary.resolve(), args.cpu.resolve(), args.directory.resolve(), args.gpus.split(","),
             args.control.resolve() if args.control else None)
