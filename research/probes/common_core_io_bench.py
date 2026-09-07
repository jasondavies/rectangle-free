#!/usr/bin/env python3
"""Local synthetic reducer replay and real checksummed-plan seek benchmark.

No synthetic residue is production work. The deliberately non-hash artifact
identities cannot bind to any real catalog or plan accepted by the controller.
"""
import argparse
import contextlib
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
import common_core_campaign as cc


def module(path):
    spec=importlib.util.spec_from_file_location('baseline_io',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def replay(args):
    args.output.mkdir(parents=True,exist_ok=False)
    old=module(args.baseline)
    class Catalog:
        slack,digest=2,'synthetic-replay-not-a-catalog-digest'
        count=2*args.groups
        keys=[4755944160245056391,4755944160245056907]
        images=[cc.prime_count(cc.bound_power(key,2)) for key in keys]
        def __getitem__(self,q):
            key=self.keys[q%2]
            return key,1,self.images[q%2]
    class Plan:
        digest='synthetic-replay-not-a-plan-digest'
        def groups(self):
            for gid in range(args.groups):
                yield gid,144258141817667969,282031183369742,[(2*gid,518),(2*gid+1,1034)]
    c,p=Catalog(),Plan()
    meta=cc.expected_meta(c,144258141817667969,282031183369742,[(0,518),(1,1034)])
    paths=[]
    for i in range(args.journals):
        start,end=args.groups*i//args.journals,args.groups*(i+1)//args.journals
        if start==end:continue
        path=args.output/f'replay-{i}.sqlite';paths.append(path)
        identity=dict(format=cc.FORMAT,catalog=c.digest,plan=p.digest,configuration=cc.CONFIG,
                      primes=list(cc.PRIMES),backend='cpu-reference',controller='replay-only',
                      solver_binary='not-a-solver',group_start=start,group_end=end)
        j=cc.Journal(path,identity)
        try:
            with j.batch(4096,1000):
                for gid in range(start,end):
                    j.put_group(gid,meta)
                    for pi in range(max(meta['primes'])):
                        values=[]
                        for column,key in enumerate(c.keys):
                            if meta['primes'][column]>pi:
                                unmatched=4-((key&cc.FULL).bit_count()-2*(key>>60))
                                values.append([42,17][column]*meta['domain']*math.factorial(unmatched)%cc.PRIMES[pi])
                        j.put_range(gid,pi,0,meta['domain'],meta,values,0.,0.)
        finally:j.close()
    options=argparse.Namespace(journals=paths,cpu_reference=True,query_results=False,require_complete=True)
    expected=None
    for name,implementation in [('old',old),('new',cc),('new',cc),('old',old)]:
        started=time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()) as output:
            implementation.reduce(options,c,p,dict(groups=args.groups))
        seconds=time.perf_counter()-started
        result=json.loads(output.getvalue())
        cc.require(expected is None or result==expected,'A/B result mismatch')
        expected=result
        print(cc.encoded(dict(kind='reducer',version=name,groups=args.groups,journals=len(paths),
                              seconds=seconds,summary_sha256=cc.digest(result),parity='OK')),flush=True)


def seek(args):
    envelope=json.loads(args.manifest.read_text());payload=envelope['payload']
    cc.require(cc.digest(payload)==envelope['sha256'],'manifest checksum')
    started=time.perf_counter();c=cc.Catalog(args.catalog);p=cc.Plan(args.plan,c)
    try:
        audit=p.audit();first=time.perf_counter()-started
        cc.require(audit==payload['audit'],'audit mismatch')
        started=time.perf_counter()
        for _ in payload['tasks']:cc.require(p.audit()==audit,'cached audit mismatch')
        reuse=time.perf_counter()-started
        started=time.perf_counter();selected={}
        for task in payload['tasks']:
            for gid in (task['group_start'],task['group_end']-1):
                selected[gid]=next(p.groups(gid,gid+1))
        direct=time.perf_counter()-started
        started=time.perf_counter();found={}
        for group in p.groups():
            if group[0] in selected:found[group[0]]=group
        scan=time.perf_counter()-started
        cc.require(found==selected,'index differs from sequential parsing')
        print(cc.encoded(dict(kind='seek',groups=audit['groups'],index_bytes=len(p._offsets)*p._offsets.itemsize,
                              initial_audit_seconds=first,cached_audits=len(payload['tasks']),
                              cached_audit_seconds=reuse,selected_groups=len(selected),
                              indexed_seconds=direct,one_full_scan_seconds=scan,parity='OK')),flush=True)
    finally:p.close();c.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    q=s.add_parser('replay');q.add_argument('--baseline',type=Path,required=True)
    q.add_argument('--output',type=Path,required=True);q.add_argument('--groups',type=int,default=20000)
    q.add_argument('--journals',type=int,default=64)
    q=s.add_parser('seek')
    for n in ('catalog','plan','manifest'):q.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    if a.command=='replay':
        cc.require(a.groups>0 and a.journals>0,'invalid replay sizes');replay(a)
    else:seek(a)


if __name__=='__main__':main()
