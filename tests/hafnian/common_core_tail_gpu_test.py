#!/usr/bin/env python3
"""Two complete larger shared groups vs independent full-matrix CRT counts."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import common_core_campaign as cc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    args = parser.parse_args()
    env = json.loads(args.sample.read_text())
    payload = env["payload"]
    cc.require(cc.digest(payload) == env["sha256"], "sample checksum mismatch")
    worker = cc.Worker(args.worker, False)
    start = time.monotonic()
    def prepare(message):
        r = worker.request(message)
        cc.require(r[0] == "prepared", "bad metadata")
        return int(r[1]), [(int(r[i]), int(r[i+1])) for i in range(3,len(r),2)]
    def whole(domain, count, pi):
        sums = [0]*count
        for begin in range(0,domain,1<<20):
            r = worker.request(f"run {pi} {begin} {min(1<<20,domain-begin)} 32768")
            cc.require(r[0] == "result" and int(r[2]) == count and len(r) == count+3, "bad result")
            sums = [(a+int(b))%cc.PRIMES[pi] for a,b in zip(sums,r[3:])]
        return sums
    try:
        for n in (52,54):
            cells = [c for c in payload["bins"] if c["pool"] == 11 and c["core"]+8 == n]
            cell = min(cells, key=lambda c:c["schedule"][0])
            sample = cell["samples"][0]
            keys = [r[0] for r in sample["rows"]]
            message = f"prepare {payload['slack']} {sample['parent']} {sample['boundary']} {len(keys)} "
            message += " ".join(f"{key} {removed}" for key,(_,removed) in zip(keys,sample["members"]))
            domain, metadata = prepare(message)
            images = [[] for key in keys]
            for pi in range(max(p for _,p in metadata)):
                active = [i for i,(_,p) in enumerate(metadata) if p>pi]
                sums = whole(domain,len(active),pi)
                for i,value in zip(active,sums):
                    unmatched=2*payload["slack"]-((keys[i]&cc.FULL).bit_count()-2*(keys[i]>>60))
                    images[i].append(value*pow(domain*math.factorial(unmatched),-1,cc.PRIMES[pi])%cc.PRIMES[pi])
            for i,key in enumerate(keys):
                independent, bounds = prepare(f"prepare_single {payload['slack']} {key}")
                cc.require(bounds == [metadata[i]], "independent bound/prime mismatch")
                want=[]
                unmatched=2*payload["slack"]-((key&cc.FULL).bit_count()-2*(key>>60))
                for pi in range(bounds[0][1]):
                    value=whole(independent,1,pi)[0]
                    want.append(value*pow(independent*math.factorial(unmatched),-1,cc.PRIMES[pi])%cc.PRIMES[pi])
                cc.require(images[i] == want, "shared/full independent residue mismatch")
                value, modulus=cc.crt(want)
                cc.require(modulus>1<<bounds[0][0] and value<=1<<bounds[0][0], "CRT bound failure")
                print(cc.encoded(dict(order=n,key=key,matching_count=str(value),primes=len(want),
                                     exact="OK",seconds=time.monotonic()-start)),flush=True)
        print(cc.encoded(dict(status="complete",seconds=time.monotonic()-start)),flush=True)
    finally:
        worker.close()


if __name__ == "__main__":
    main()
