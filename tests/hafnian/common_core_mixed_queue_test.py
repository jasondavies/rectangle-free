#!/usr/bin/env python3
"""Bounded two-process CPU rehearsal: kill, snapshot restore, resume and check.

Requires a real catalog, plan, CPU-bound mixed manifest and host worker.
Only the first few sign checkpoints of queues 0/1 execute. Never a full solve
or GPU throughput measurement. Every restored residue is recomputed directly.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools'))
import common_core_campaign as cc
import common_core_mixed_campaign as mix
import common_core_sign_shards as ss


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('catalog','plan','manifest','worker','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    payload=mix.load(args.manifest)
    cc.require(payload['backend']=='cpu-reference','CPU-only rehearsal refuses GPU manifest')
    checksum=cc.digest(payload);children=[];logs=[]
    base=[sys.executable,str(ROOT/'tools/common_core_mixed_campaign.py')]
    inputs=['--catalog',str(args.catalog),'--plan',str(args.plan),'--manifest',str(args.manifest)]
    def launch(owner,root,bounded=False):
        command=base+['run',*inputs,'--worker',str(args.worker),'--worker-id',str(owner),
                      '--journal-dir',str(root),'--checkpoint-terms','64','--chunk-terms','32',
                      '--commit-ranges','1']
        if bounded:command+=['--max-checkpoints','1']
        log=open(args.output/f'queue-{owner}-{len(logs)}.log','w');logs.append(log)
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(child);return child
    def read(path,tid):
        j=cc.Journal(path)
        try:return list(mix.checked_groups(j,payload,tid))
        finally:j.close()
    try:
        live=args.output/'live';restored=args.output/'restored';paths=[]
        owners=(0,1)
        for owner in owners:
            tid=payload['workers'][owner]['task_ids'][0]
            cc.require(payload['tasks'][tid]['kind']=='signs','first task must be a bounded independent interval')
            paths.append(live/checksum/payload['tasks'][tid]['journal'])
        processes=[launch(owner,live) for owner in owners]
        saved=[]
        for owner,child,source in zip(owners,processes,paths):
            tid=payload['workers'][owner]['task_ids'][0];deadline=time.monotonic()+20
            while True:
                cc.require(child.poll() is None,'worker exited before interruption')
                cc.require(time.monotonic()<deadline,'no durable checkpoint within timeout')
                try:
                    records=read(source,tid) if source.exists() else []
                    ready=any(image for _,_,r in records for image in r)
                except Exception:ready=False  # initial schema transaction may not be visible
                if ready:break
                time.sleep(.01)
            snapshot=args.output/f'published-{owner}.sqlite'
            receipt=json.loads(subprocess.check_output(base+['snapshot',*inputs,'--source',str(source),
                                                         '--output',str(snapshot)],text=True,timeout=20))
            os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
            subprocess.run(base+['snapshot',*inputs,'--source',str(snapshot),'--verify-sha256',receipt['sha256']],
                           check=True,capture_output=True,timeout=20)
            target=restored/checksum/source.name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(snapshot,target);saved.append(read(target,tid))
        for owner in owners:cc.require(launch(owner,restored,True).wait(timeout=20)==0,'resume failed')
        final=[];checks=0
        with ss.artifacts(args.catalog,args.plan) as (catalog,plan,audit):
            mix.bind(payload,catalog,plan,audit);worker=cc.Worker(args.worker,True)
            try:
                for owner,source,before in zip(owners,paths,saved):
                    tid=payload['workers'][owner]['task_ids'][0];target=restored/checksum/source.name
                    after=read(target,tid);gid,meta,records=after[0]
                    old={(pi,a,b,tuple(v)) for pi,image in enumerate(before[0][2]) for a,b,v in image}
                    new={(pi,a,b,tuple(v)) for pi,image in enumerate(records) for a,b,v in image}
                    cc.require(old < new and len(new)==len(old)+1,'resume lost or duplicated committed work')
                    query,_=ss.resolve(catalog,plan,gid)
                    worker.request(f'prepare_single {catalog.slack} {catalog[query][0]}')
                    for pi,image in enumerate(records):
                        for a,b,values in image:
                            reply=worker.request(f'run {pi} {a} {b-a} 32')
                            cc.require(list(map(int,reply[3:]))==values,'restored residue differs from recomputation')
                            checks+=1
                    final.append(target)
            finally:worker.close();worker.process.stdin.close();worker.process.stdout.close()
        reduced=json.loads(subprocess.check_output(base+['reduce',*inputs,'--journals',*map(str,final)],text=True,timeout=20))
        cc.require(reduced['status']=='partial' and reduced['complete_queries']==0,'partial rehearsal claimed a grid result')
        (args.output/'summary.json').write_text(cc.encoded(dict(status='passed',queues=2,kill_restore='OK',
                    checkpoints_recomputed=checks,reduction='partial',scope='CPU functional rehearsal; not a full calculation'))+'\n')
        print((args.output/'summary.json').read_text(),end='')
    finally:
        for child in children:
            if child.poll() is None:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
        for log in logs:log.close()


if __name__=='__main__':main()
