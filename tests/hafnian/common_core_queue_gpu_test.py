#!/usr/bin/env python3
"""Bounded two-process queue/snapshot/crash rehearsal on real 6x28 groups.

Both processes may share one GPU: this tests functionality, NOT multi-GPU
throughput. The unselected plan remainder belongs to queue 2, NEVER launched.
"""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
import common_core_campaign as cc
import common_core_manifest as cm
from common_core_snapshot import snapshot


def fixture(catalog,plan,audit,worker,path):
    shared=[];single=[]
    for gid,parent,boundary,members in plan.groups():
        if len(members)>1:
            if len(shared)<2:shared.append(gid)
        else:single.append((cc.expected_meta(catalog,parent,boundary,members)['domain'],gid))
    chosen={g:0 for g in shared}
    chosen.update({g:1 for _,g in sorted(single)[:2]})
    cc.require(len(chosen)==4,'four pilot groups required')
    boundaries=sorted({0,audit['groups']}|{x for g in chosen for x in (g,g+1)})
    tasks=[];queues=[dict(id=i,task_ids=[],seconds=0.) for i in range(3)]
    for begin,end in zip(boundaries,boundaries[1:]):
        tid=len(tasks);owner=chosen.get(begin,2)
        task=dict(id=tid,group_start=begin,group_end=end,seconds=1.,worker=owner,journal=f'task-{tid:04d}.sqlite')
        tasks.append(task);queues[owner]['task_ids'].append(tid);queues[owner]['seconds']+=1
    payload=dict(format=cm.FORMAT,audit=audit,catalog_path=str(catalog.path),plan_path=str(plan.path),
                 controller_sha256=cc.file_digest(cc.__file__),worker_sha256=cc.file_digest(worker),
                 configuration=cc.CONFIG,primes=list(cc.PRIMES),tasks=tasks,workers=queues,
                 total_seconds=float(len(tasks)),purpose='bounded correctness rehearsal, queue 2 forbidden')
    cm.validate(payload)
    checksum=cc.digest(payload)
    path.write_text(cc.encoded(dict(payload=payload,sha256=checksum))+'\n')
    return payload,checksum


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker',type=Path,required=True);p.add_argument('--control',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    catalog=cc.Catalog(ROOT/'build/common-core-6x28.catalog')
    plan=cc.Plan(ROOT/'build/common-core-6x28-order50-481.plan',catalog)
    children=[];logs=[]
    started=time.monotonic()
    try:
        audit=plan.audit()
        manifest=args.output/'manifest.json'
        payload,checksum=fixture(catalog,plan,audit,args.worker,manifest)
        live=args.output/'live';restored=args.output/'restored';published=args.output/'published'
        def launch(owner,root,fine=False):
            cc.require(owner in (0,1),'remainder queue must never run')
            command=[sys.executable,str(ROOT/'tools/common_core_manifest.py'),'run',
                '--manifest',str(manifest),'--worker-id',str(owner),'--worker',str(args.worker),
                '--journal-dir',str(root),'--chunk-terms',str(1024 if fine else 32768),
                '--checkpoint-terms',str(4096 if fine else 1<<20)]
            log=open(args.output/f'queue-{owner}-{len(logs)}.log','w');logs.append(log)
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            children.append(child);return child
        first=launch(0,live,True);second=launch(1,live)
        tid=payload['workers'][0]['task_ids'][0]
        source=live/checksum/payload['tasks'][tid]['journal']
        saved=published/source.name
        while True:
            cc.require(time.monotonic()-started<60,'timeout waiting for first committed checkpoint')
            cc.require(first.poll() is None,'pilot finished before interruption could be exercised')
            if source.exists():
                try:
                    journal=cc.Journal(source)
                    try:count=journal.db.execute('SELECT count(*) FROM ranges').fetchone()[0]
                    finally:journal.close()
                except Exception:count=0  # first schema transaction may not yet be visible
                if count:
                    report=snapshot(source,saved)
                    cc.require(first.poll() is None,'pilot finished before stop')
                    os.killpg(first.pid,signal.SIGSTOP)
                    print(cc.encoded({**report,'status':'paused_after_live_snapshot'}),flush=True)
                    break
            time.sleep(.01)
        # Give an external poller time to pull the published CLOSED snapshot.
        time.sleep(8)
        os.killpg(first.pid,signal.SIGKILL);first.wait(timeout=10)
        restore=restored/checksum/source.name;restore.parent.mkdir(parents=True)
        shutil.copy2(saved,restore)  # fresh path; preserve the killed worker's files
        resumed=launch(0,restored)
        for child in (second,resumed):
            cc.require(child.wait(timeout=180)==0,'queue failed; inspect its log')
        final=[]
        for owner,root in ((0,restored),(1,live)):
            for task_id in payload['workers'][owner]['task_ids']:
                src=root/checksum/payload['tasks'][task_id]['journal']
                snapshot(src,published/src.name)
                final.append(published/src.name)
        # Resume completed queues: no new ranges or changed DB contents.
        for owner,root in ((0,restored),(1,live)):
            cc.require(launch(owner,root).wait(timeout=60)==0,'idempotent resume failed')
        after=[]
        for owner,root in ((0,restored),(1,live)):
            for task_id in payload['workers'][owner]['task_ids']:
                src=root/checksum/payload['tasks'][task_id]['journal']
                j=cc.Journal(src)
                try:
                    after.append([(g,m,r) for g,m,r in j.ordered_groups()])
                finally:j.close()
        for expected,path in zip(after,final):
            j=cc.Journal(path)
            try:cc.require(list(j.ordered_groups())==expected,'completed resume changed records')
            finally:j.close()
        # A separate baseline binary computes exactly the same four groups.
        control_paths=[]
        worker=cc.Worker(args.control,False)
        try:
            for owner in (0,1):
                for task_id in payload['workers'][owner]['task_ids']:
                    task=payload['tasks'][task_id];path=args.output/f'control-{task_id}.sqlite';control_paths.append(path)
                    options=argparse.Namespace(worker=args.control,cpu_reference=False,journal=path,
                        group_start=task['group_start'],group_end=task['group_end'],chunk_terms=32768,
                        checkpoint_terms=1<<20,max_checkpoints=0,commit_ranges=32,commit_seconds=1.)
                    cc.run(options,catalog,plan,audit,worker=worker)
        finally:worker.close()
        outputs=[]
        for paths in (final,control_paths):
            with contextlib.redirect_stdout(io.StringIO()) as result:
                cc.reduce(argparse.Namespace(journals=paths,cpu_reference=False,query_results=True,
                                            require_complete=False),catalog,plan,audit)
            outputs.append(result.getvalue())
        cc.require(outputs[0]==outputs[1],'recovered/candidate/control exact reduction mismatch')
        (args.output/'reduction.jsonl').write_text(outputs[0])
        print(cc.encoded(dict(status='complete',groups=4,complete_queries=json.loads(outputs[0].splitlines()[-1])['complete_queries'],
                             snapshot_restore='OK',crash_resume='OK',idempotent='OK',control_parity='OK',
                             seconds=time.monotonic()-started,scope='functional two-process test; not multi-GPU throughput')),flush=True)
    finally:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
        for log in logs:log.close()
        plan.close();catalog.close()


if __name__=='__main__':main()
