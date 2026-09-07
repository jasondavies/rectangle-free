#!/usr/bin/env python3
"""Measured mixed whole-group/sign-range queues with exact campaign reduction.

Opt-in format: does not modify old whole-group campaigns. One persistent
worker per queue, one durable journal per task. No cloud provisioning.
"""
import argparse
from array import array
from collections import Counter
import heapq
import json
import math
from pathlib import Path
import shlex

import common_core_campaign as cc
import common_core_manifest as cm
import common_core_sign_shards as ss
import common_core_snapshot_io as snapshot_io

FORMAT = 'common-core-mixed-campaign-v1'


def sources():
    return {name: cc.file_digest(path) for name, path in
            (('controller', cc.__file__), ('scheduler', cm.__file__),
             ('sign_runner', ss.__file__), ('snapshot_io', snapshot_io.__file__), ('mixed_runner', __file__))}


def integer(value):
    return type(value) is int


def make_tasks(costs, independent_domains, workers, target_shards):
    cc.require(integer(workers) and workers > 0 and integer(target_shards) and target_shards > 0 and costs,
               'invalid worker/shard counts')
    cc.require(all(math.isfinite(c) and c > 0 for c in costs), 'invalid group cost')
    target = math.fsum(costs) / target_shards
    tasks, start, pending = [], 0, 0.

    def emit(end):
        nonlocal start, pending
        if start < end:
            tasks.append(dict(kind='groups', group_start=start, group_end=end, seconds=pending))
        start, pending = end, 0.

    for gid, seconds in enumerate(costs):
        domain = independent_domains.get(gid)
        parts = min(workers, domain, math.ceil(seconds/target)) if domain else 1
        if parts > 1:
            emit(gid)
            for a, b in ss.split_domain(domain, parts):
                tasks.append(dict(kind='signs', group_start=gid, group_end=gid+1,
                                  begin=a, end=b, domain=domain, seconds=seconds*(b-a)/domain))
            start = gid+1
        else:
            if pending and pending + seconds > target:
                emit(gid)
            pending += seconds
    emit(len(costs))
    for i, task in enumerate(tasks):
        task['id'] = i
    return tasks


def validate(payload):
    cc.require(payload['format'] == FORMAT and payload['tasks'] and payload['workers'], 'invalid mixed manifest')
    cc.require(payload['backend'] in ('cuda', 'cpu-reference'), 'invalid backend')
    cursor, sign_cursor, domain = 0, 0, None
    tasks = payload['tasks']
    for i, t in enumerate(tasks):
        cc.require(integer(t['id']) and t['id'] == i and
                   integer(t['group_start']) and integer(t['group_end']) and
                   t['group_start'] == cursor < t['group_end'], 'gap/overlap in group ownership')
        cc.require(math.isfinite(t['seconds']) and t['seconds'] > 0 and
                   t['journal'] == f'task-{i:04d}.sqlite', 'invalid task cost/journal')
        if t['kind'] == 'groups':
            cc.require(sign_cursor == 0 and not any(k in t for k in ('begin','end','domain')),
                       'whole group overlaps sign tasks')
            cursor = t['group_end']
        else:
            cc.require(t['kind'] == 'signs' and t['group_end'] == cursor+1 and
                       integer(t['domain']) and 0 < t['domain'] <= 1 << 32 and
                       not t['domain'] & (t['domain']-1), 'invalid sign task domain')
            if sign_cursor == 0:
                domain = t['domain']
            cc.require(t['domain'] == domain and integer(t['begin']) and integer(t['end']) and
                       t['begin'] == sign_cursor < t['end'] <= domain, 'gap/overlap in sign ownership')
            sign_cursor = t['end']
            if sign_cursor == domain:
                sign_cursor = 0
                cursor += 1
    cc.require(sign_cursor == 0 and cursor == payload['audit']['groups'], 'incomplete manifest coverage')
    seen = set()
    for i, queue in enumerate(payload['workers']):
        cc.require(integer(queue['id']) and queue['id'] == i, 'invalid queue ID')
        for tid in queue['task_ids']:
            cc.require(integer(tid) and 0 <= tid < len(tasks) and tid not in seen and
                       integer(tasks[tid]['worker']) and tasks[tid]['worker'] == i, 'duplicate/invalid queue ownership')
            seen.add(tid)
        cc.require(math.isclose(queue['seconds'], math.fsum(tasks[t]['seconds'] for t in queue['task_ids']),
                                rel_tol=1e-12, abs_tol=1e-7), 'queue cost mismatch')
    cc.require(len(seen) == len(tasks), 'unassigned tasks')
    cc.require(math.isclose(payload['total_seconds'], math.fsum(t['seconds'] for t in tasks),
                            rel_tol=1e-12, abs_tol=1e-7), 'total cost mismatch')


def load(path):
    e = json.loads(path.read_text()); p = e['payload']
    cc.require(e['sha256'] == cc.digest(p), 'manifest checksum mismatch')
    validate(p)
    cc.require(p['sources'] == sources() and p['configuration'] == cc.CONFIG and p['primes'] == list(cc.PRIMES),
               'controller/configuration changed; regenerate manifest')
    return p


def bind(payload, catalog, plan, audit):
    cc.require(audit == payload['audit'], 'manifest artifact/audit mismatch')
    checked = set()
    for t in payload['tasks']:
        if t['kind'] == 'signs' and t['group_start'] not in checked:
            _, meta = ss.resolve(catalog, plan, t['group_start'])
            cc.require(meta['domain'] == t['domain'], 'sign domain differs from plan')
            checked.add(t['group_start'])


def build(args):
    with ss.artifacts(args.catalog, args.plan) as (catalog, plan, audit):
        bs, baseline = cm.load_projection(args.baseline_sample, args.baseline_projection, catalog)
        rs, replacement = cm.load_projection(args.replacement_sample, args.replacement_projection, catalog)
        cc.require(not bs.get('selected_orders') and rs['plan'] == plan.digest and
                   rs['total_plan_groups'] == audit['groups'] and
                   rs['coefficient_sum'] == bs['coefficient_sum'] == audit['coefficient_sum'],
                   'baseline/replacement coverage mismatch')
        # Conservative: retain old overhead per replaced child. Do not credit
        # an unmeasured storage/batched-transaction improvement.
        model, population = cm.cost_model(baseline, replacement, rs['selected_orders'], 1.)
        costs = array('d'); domains = {}; actual = Counter()
        for gid, parent, boundary, members in plan.groups():
            rows = [catalog[q] for q, _ in members]
            key = rows[0][0]; used = (key & cc.FULL).bit_count()
            unmatched = 2*catalog.slack - (used-2*(key>>60))
            core = 60-((key & cc.FULL) if len(members)==1 else parent|boundary).bit_count()+unmatched
            sig = (core, boundary.bit_count(), *(sum(r[2] > pi for r in rows) for pi in range(4)))
            cc.require(sig in model, f'unmeasured stratum {sig}')
            actual[sig] += 1; costs.append(model[sig])
            if len(members) == 1:
                domains[gid] = 1 << max(0, core//2-1)
        cc.require(actual == population and len(costs) == audit['groups'], 'measured/actual population mismatch')
        tasks = make_tasks(costs, domains, args.workers, args.target_shards)
        queues = cm.assign(tasks, args.workers)
        p = dict(format=FORMAT, sources=sources(), audit=audit, configuration=cc.CONFIG,
                 primes=list(cc.PRIMES), solver_binary=cc.file_digest(args.worker),
                 backend='cpu-reference' if args.cpu_reference else 'cuda',
                 measurements={name:cc.file_digest(getattr(args,name)) for name in
                               ('baseline_sample','baseline_projection','replacement_sample','replacement_projection')},
                 tasks=tasks, workers=queues, total_seconds=math.fsum(costs),
                 largest_group_seconds=max(costs), target_shards=args.target_shards,
                 assumptions='Sampled RTX PRO 6000 costs; old overhead retained per replaced child; no storage speedup credited. Excludes repeated process/task setup, audit, transfer, final reduction, interruptions and multi-GPU contention.')
        validate(p); bind(p,catalog,plan,audit)
        with args.output.open('x') as f:
            f.write(cc.encoded(dict(payload=p,sha256=cc.digest(p)))+'\n')
        print(cc.encoded(dict(status='manifest_created', manifest=cc.digest(p), tasks=len(tasks),
                              sign_tasks=sum(t['kind']=='signs' for t in tasks),
                              total_hours=p['total_seconds']/3600,
                              queue_hours=[q['seconds']/3600 for q in queues])))


def identity(payload, tid):
    cc.require(integer(tid) and 0 <= tid < len(payload['tasks']), 'task outside manifest')
    t = payload['tasks'][tid]
    return dict(format=FORMAT, manifest=cc.digest(payload), task=t,
                group_start=t['group_start'], group_end=t['group_end'],
                backend=payload['backend'], solver_binary=payload['solver_binary'])


def checked_groups(journal, payload, tid):
    cc.require(journal.identity == identity(payload,tid), 'journal provenance/task mismatch')
    t = payload['tasks'][tid]
    for gid,meta,records in journal.ordered_groups():
        cc.require(t['group_start'] <= gid < t['group_end'], 'result outside task group ownership')
        if t['kind'] == 'signs':
            cc.require(meta['domain'] == t['domain'] and len(meta['primes']) == 1, 'invalid range task metadata')
            for image in records:
                for a,b,_ in image:
                    cc.require(t['begin'] <= a < b <= t['end'], 'result outside sign task ownership')
        yield gid,meta,records


def check_snapshot(path, payload, catalog, plan):
    j=cc.Journal(path)
    try:
        groups=ranges=0
        for gid,saved,records in checked_groups(j,payload,j.identity['task']['id']):
            _,parent,boundary,members=next(plan.groups(gid,gid+1))
            cc.require(saved==cc.expected_meta(catalog,parent,boundary,members),'snapshot metadata differs from plan')
            groups+=1;ranges+=sum(len(image) for image in records)
        return dict(identity=j.identity,groups=groups,ranges=ranges)
    finally:j.close()


def execute_task(args, payload, tid, catalog, plan, worker, journal):
    t = payload['tasks'][tid]
    if t['kind'] == 'groups':
        # Reuse the maintained whole-group execution and prime scheduling.
        cc.require(journal.identity == identity(payload,tid), 'journal provenance/task mismatch')
        options = argparse.Namespace(**vars(args)); options.group_start=t['group_start']
        return cc.solve_ranges(options,catalog,plan,t['group_end'],worker,journal)
    query,meta = ss.resolve(catalog,plan,t['group_start'])
    records = [[] for _ in cc.PRIMES]
    for gid,saved,checked in checked_groups(journal,payload,tid):
        cc.require(saved == meta, 'stored metadata differs from plan')
        records = checked
    return ss.solve_interval(t['group_start'],query,meta,t['begin'],t['end'],records,
                             catalog,worker,journal,args.chunk_terms,args.checkpoint_terms,args.max_checkpoints)


def run_queue(args, payload, catalog, plan, audit):
    cc.require(integer(args.worker_id) and 0 <= args.worker_id < len(payload['workers']), 'invalid worker ID')
    cc.require(0 < args.chunk_terms <= 1 << 20 and args.checkpoint_terms > 0 and args.max_checkpoints >= 0 and
               1 <= args.commit_ranges <= 4096 and math.isfinite(args.commit_seconds) and args.commit_seconds > 0,
               'invalid execution/checkpoint settings')
    cc.require(cc.file_digest(args.worker) == payload['solver_binary'], 'worker binary changed')
    worker = cc.Worker(args.worker,payload['backend']=='cpu-reference')
    try:
        for tid in payload['workers'][args.worker_id]['task_ids']:
            cc.require(plan.audit() == audit and sources() == payload['sources'] and
                       cc.file_digest(args.worker) == payload['solver_binary'], 'queue inputs changed')
            path = args.journal_dir/cc.digest(payload)/payload['tasks'][tid]['journal']
            path.parent.mkdir(parents=True,exist_ok=True)
            with cc.claim(path):
                journal = cc.Journal(path,identity(payload,tid))
                try:
                    def acknowledged(count):
                        print(cc.encoded(dict(status='checkpoints_committed',task=tid,ranges=count)),flush=True)
                    print(cc.encoded(dict(status='task_start',task=tid)),flush=True)
                    with journal.batch(args.commit_ranges,args.commit_seconds,acknowledged):
                        done = execute_task(args,payload,tid,catalog,plan,worker,journal)
                    print(cc.encoded(dict(status='task_complete' if done else 'checkpoint_limit_reached',task=tid)),flush=True)
                    if not done:
                        return
                finally:
                    journal.close()
        print(cc.encoded(dict(status='queue_complete',worker=args.worker_id)),flush=True)
    finally:
        worker.close(); worker.process.stdin.close(); worker.process.stdout.close()


def reduce_campaign(args, payload, catalog, plan, audit):
    journals, heap, seen = [], [], set()
    def advance(index, iterator):
        row = next(iterator,None)
        if row is not None:
            gid,meta,records = row
            heapq.heappush(heap,(gid,index,meta,records,iterator))
    try:
        for path in args.journals:
            j = cc.Journal(path); journals.append(j)
            tid = j.identity['task']['id']
            cc.require(tid not in seen,'duplicate task journal; choose one snapshot per task')
            seen.add(tid)
            advance(len(journals)-1,iter(checked_groups(j,payload,tid)))
        total = complete = missing = tail = 0
        for gid,parent,boundary,members in plan.groups():
            cc.require(not heap or heap[0][0] >= gid,'result group outside plan')
            meta = cc.expected_meta(catalog,parent,boundary,members)
            records = [[] for _ in cc.PRIMES]
            while heap and heap[0][0] == gid:
                _,index,saved,checked,iterator = heapq.heappop(heap)
                cc.require(saved == meta,'result metadata differs from catalog')
                for pi,image in enumerate(checked):
                    records[pi].extend(image)
                advance(index,iterator)
            keys = [catalog[q][0] for q,_ in members]
            values = cc.reduce_group(meta,records,keys,catalog.slack)
            for (qid,_),key,value in zip(members,keys,values):
                if value is None:
                    if len(members)==1: tail += 1
                    else: missing += 1
                else:
                    complete += 1
                    # Outer coefficient and column multiplicity once per QUERY,
                    # never once per task, sign interval or prime image.
                    total += catalog[qid][1]*(1 << (30-catalog.slack-(key>>60)))*value
                    if args.query_results:
                        print(cc.encoded(dict(query_id=qid,matching_count=str(value))))
        cc.require(not heap,'result group outside plan')
        summary = dict(status='partial' if missing or tail else 'complete',manifest=cc.digest(payload),
                       complete_queries=complete,missing_shared_queries=missing,pending_independent_queries=tail,
                       partial_labelled_count=str(total*math.factorial(30-catalog.slack)),**audit)
        print(cc.encoded(summary))
        cc.require(not args.require_complete or not (missing or tail),'campaign incomplete; no final grid result certified')
        return summary
    finally:
        for j in journals:j.close()


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    for command in ('build','verify','commands','run','reduce','snapshot'):
        a=sub.add_parser(command)
        a.add_argument('--catalog',type=Path,required=True);a.add_argument('--plan',type=Path,required=True)
        if command != 'build':a.add_argument('--manifest',type=Path,required=True)
        if command in ('build','verify','commands','run'):a.add_argument('--worker',type=Path,required=True)
        if command=='build':
            for name in ('baseline-sample','baseline-projection','replacement-sample','replacement-projection','output'):
                a.add_argument('--'+name,type=Path,required=True)
            a.add_argument('--workers',type=int,default=8);a.add_argument('--target-shards',type=int,default=64)
            a.add_argument('--cpu-reference',action='store_true',help='separate functional-test identity, not GPU timings')
        if command in ('commands','run'):
            a.add_argument('--worker-id',type=int,required=True);a.add_argument('--journal-dir',type=Path,required=True)
        if command=='run':
            a.add_argument('--chunk-terms',type=int,default=32768);a.add_argument('--checkpoint-terms',type=int,default=1<<20)
            a.add_argument('--commit-ranges',type=int,default=32);a.add_argument('--commit-seconds',type=float,default=1.)
            a.add_argument('--max-checkpoints',type=int,default=0)
        if command=='reduce':
            a.add_argument('--journals',type=Path,nargs='+',required=True)
            a.add_argument('--require-complete',action='store_true');a.add_argument('--query-results',action='store_true')
        if command=='snapshot':
            a.add_argument('--source',type=Path,required=True)
            mode=a.add_mutually_exclusive_group(required=True)
            mode.add_argument('--output',type=Path)
            mode.add_argument('--verify-sha256')
    args=p.parse_args()
    if args.command=='build':build(args);return
    payload=load(args.manifest)
    with ss.artifacts(args.catalog,args.plan) as (catalog,plan,audit):
        bind(payload,catalog,plan,audit)
        if args.command in ('verify','commands'):
            cc.require(cc.file_digest(args.worker)==payload['solver_binary'],'worker binary changed')
        if args.command=='verify':print(cc.encoded(dict(status='manifest_verified',manifest=cc.digest(payload))))
        elif args.command=='commands':
            cc.require(0 <= args.worker_id < len(payload['workers']),'invalid worker ID')
            print('#!/bin/bash\nset -euo pipefail')
            print(shlex.join(['python3','tools/common_core_mixed_campaign.py','run','--manifest',str(args.manifest),
                '--catalog',str(args.catalog),'--plan',str(args.plan),'--worker',str(args.worker),
                '--worker-id',str(args.worker_id),'--journal-dir',str(args.journal_dir)]))
        elif args.command=='run':run_queue(args,payload,catalog,plan,audit)
        elif args.command=='snapshot':
            check=lambda path:check_snapshot(path,payload,catalog,plan)
            if args.output:
                print(cc.encoded(snapshot_io.snapshot(args.source,args.output,check)))
            else:
                cc.require(cc.file_digest(args.source)==args.verify_sha256,'downloaded snapshot checksum mismatch')
                report=check(args.source)
                cc.require(cc.file_digest(args.source)==args.verify_sha256,'snapshot changed during verification')
                print(cc.encoded(dict(status='download_verified',sha256=args.verify_sha256,
                                      groups=report['groups'],ranges=report['ranges'])))
        else:reduce_campaign(args,payload,catalog,plan,audit)


if __name__=='__main__':main()
