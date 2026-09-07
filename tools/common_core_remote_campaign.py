#!/usr/bin/env python3
"""Run an audited queue per visible GPU and publish verified closed snapshots.

No cloud credentials or provisioning. An external supervisor must pull and
verify receipts before deleting resources. Config and all data are private
campaign artifacts, not part of this module.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import common_core_campaign as cc
import common_core_manifest as cm
from common_core_snapshot import snapshot


def atomic_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w') as out:
        out.write(cc.encoded(value)+'\n');out.flush();os.fsync(out.fileno())
    os.replace(temporary,path)
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def complete_snapshot(path, task):
    journal=cc.Journal(path)
    try:
        cc.require(journal.identity['group_start']==task['group_start'] and
                   journal.identity['group_end']==task['group_end'], 'snapshot task ownership mismatch')
        wanted=task['group_start']
        for gid,meta,records in journal.ordered_groups():
            if gid!=wanted:return False
            wanted+=1
            for pi in range(max(meta['primes'])):
                end=0
                for begin,stop,_ in records[pi]:
                    if begin!=end:return False
                    end=stop
                if end!=meta['domain']:return False
        return wanted==task['group_end']
    finally:journal.close()


def run(config):
    manifest=Path(config['manifest']);binary=Path(config['worker'])
    payload,digest=cm.load_manifest(manifest)
    cc.require(cc.file_digest(binary)==payload['worker_sha256'],'wrong worker binary')
    root=Path(config['output']);root.mkdir(parents=True,exist_ok=True)
    visible=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()
    cc.require(len(visible)==len(payload['workers']),'one physical GPU per queue required')
    stop=threading.Event();publish_now=threading.Event();errors=[];published={}
    processes={};handles=[];retries={};progress={};offsets={}
    def launch(owner):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(owner),OMP_NUM_THREADS='2')
        log=open(root/f'queue-{owner}.log','a');handles.append(log)
        command=[sys.executable,'tools/common_core_manifest.py','run','--manifest',str(manifest),
                 '--worker-id',str(owner),'--worker',str(binary),'--journal-dir',str(root/'journals')]
        processes[owner]=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    def publish_all():
        for task in payload['tasks']:
            source=root/'journals'/digest/task['journal']
            if not source.exists():continue
            stamp=(source.stat().st_size,source.stat().st_mtime_ns)
            if published.get(task['id'])==stamp:continue
            destination=root/'published'/task['journal']
            report=snapshot(source,destination,backup_timeout=60.)
            report.update(task=task['id'],manifest=digest,complete=complete_snapshot(destination,task))
            atomic_json(destination.with_suffix('.json'),report)
            published[task['id']]=stamp
    def publisher():
        while not stop.is_set():
            try:publish_all()
            except Exception as e:errors.append(dict(time=time.time(),error=str(e)))
            publish_now.wait(config.get('snapshot_seconds',300));publish_now.clear()
    thread=threading.Thread(target=publisher,daemon=True)
    def progress_read(owner):
        path=root/f'queue-{owner}.log'
        stats=progress.setdefault(owner,dict(checkpoints=0,gpu_seconds=0.))
        with path.open() as log:
            log.seek(offsets.get(owner,0))
            while True:
                pos=log.tell();line=log.readline()
                if not line.endswith('\n'):log.seek(pos);break
                try:r=json.loads(line)
                except ValueError:continue
                if 'checkpoint' in r:
                    stats['checkpoints']+=1;stats['gpu_seconds']+=r['compute_seconds']
                    stats['group']=r['group'];stats['last_result']=time.time()
                if r.get('status')=='task_start':stats['task']=r['task']
            offsets[owner]=log.tell()
        return dict(stats,pid=processes[owner].pid,exit=processes[owner].poll(),restarts=retries.get(owner,0))
    try:
        for w in payload['workers']:launch(w['id'])
        thread.start();state='running'
        while True:
            statuses={i:progress_read(i) for i in processes}
            atomic_json(root/'status.json',dict(state=state,time=time.time(),manifest=digest,queues=statuses,
                                               deadline=config['deadline'],snapshot_errors=errors[-8:]))
            if (root/'STOP').exists() or time.time()>=config['deadline']:
                state='deadline_or_stop';break
            failed=[i for i,p in processes.items() if p.poll() not in (None,0)]
            if failed:
                for i in failed:
                    if retries.get(i,0)>=2:state='failed';break
                    retries[i]=retries.get(i,0)+1;launch(i)
                if state=='failed':break
            if all(p.poll()==0 for p in processes.values()):state='complete';break
            time.sleep(5)
        if state!='complete':
            for p in processes.values():
                if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        for p in processes.values():
            try:p.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        stop.set();publish_now.set();thread.join(timeout=600)
        cc.require(not thread.is_alive(),'publisher did not stop')
        publish_all()
        if state=='complete':
            for task in payload['tasks']:
                receipt=json.loads((root/'published'/task['journal']).with_suffix('.json').read_text())
                cc.require(receipt['complete'] and receipt['manifest']==digest,'incomplete final snapshot')
        atomic_json(root/'status.json',dict(state=state,time=time.time(),manifest=digest,
                        queues={i:progress_read(i) for i in processes},snapshot_errors=errors[-8:]))
    finally:
        stop.set();publish_now.set()
        for p in processes.values():
            if p.poll() is None:os.killpg(p.pid,signal.SIGKILL);p.wait()
        for log in handles:log.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    run(json.loads(args.config.read_text()))
