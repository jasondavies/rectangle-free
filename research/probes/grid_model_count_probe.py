#!/usr/bin/env python3
"""Bounded exact #SAT gate for labelled four-colour rectangle-free grids.

Research only. CNF encodings are parsimonious: every colouring has exactly
one satisfying assignment. Optional first-cell anchoring restores a factor
of four, not 24. No spatial symmetry breaking or auxiliary variables.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]


def encode(rows, columns, encoding='bits', anchor=False):
    if not (1 <= rows <= 9 and 1 <= columns <= 9):
        raise ValueError('this bounded probe supports dimensions 1..9')
    if encoding not in ('bits', 'onehot'):
        raise ValueError('unknown encoding')
    width = 2 if encoding == 'bits' else 4
    def var(r, c, b):
        return width * (r * columns + c) + b + 1
    clauses = []
    if encoding == 'onehot':
        for r in range(rows):
            for c in range(columns):
                clauses.append(tuple(var(r,c,b) for b in range(4)))
                clauses.extend((-var(r,c,a),-var(r,c,b)) for a,b in itertools.combinations(range(4),2))
    for r,s in itertools.combinations(range(rows),2):
        for c,d in itertools.combinations(range(columns),2):
            cells = ((r,c),(r,d),(s,c),(s,d))
            for colour in range(4):
                if encoding == 'bits':
                    clauses.append(tuple((-1 if colour >> b & 1 else 1)*var(i,j,b)
                                         for i,j in cells for b in range(2)))
                else:
                    clauses.append(tuple(-var(i,j,colour) for i,j in cells))
    if anchor:
        clauses.extend([(-var(0,0,0),),(-var(0,0,1),)] if encoding == 'bits' else [(var(0,0,0),)])
    return width*rows*columns, clauses, 4 if anchor else 1


def dimacs(rows, columns, encoding='bits', anchor=False):
    variables,clauses,factor=encode(rows,columns,encoding,anchor)
    return ('c rectangle-free four-colour count; restore_factor '+str(factor)+'\n'+
            f'p cnf {variables} {len(clauses)}\n'+
            ''.join(' '.join(map(str,c))+' 0\n' for c in clauses))


def parse_count(text, solver):
    # Some counters print a partial accumulator on timeout: never accept it.
    if re.search(r'^\s*(?:c o )?(?:TIMEOUT|ABORTED)\b',text,re.I|re.M):
        return None
    pattern = r'^# solutions\s*\n(\d+)\s*$' if solver == 'original' else r'^c s exact arb int (\d+)\s*$'
    matches = re.findall(pattern,text,re.M)
    if len(matches)!=1:
        return None
    return int(matches[0])


def bounded_run(command, cwd, log_path, seconds, memory_gib):
    def limits():
        cap=int(memory_gib*(1<<30))
        resource.setrlimit(resource.RLIMIT_AS,(cap,cap))
        resource.setrlimit(resource.RLIMIT_CPU,(math.ceil(seconds)+2,math.ceil(seconds)+3))
        resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    started=time.monotonic();timed_out=False
    with log_path.open('x') as log:
        child=subprocess.Popen(command,cwd=cwd,stdout=log,stderr=subprocess.STDOUT,
                               start_new_session=True,preexec_fn=limits)
        try:
            while True:
                pid,status,usage=os.wait4(child.pid,os.WNOHANG)
                if pid:break
                if time.monotonic()-started>=seconds:
                    timed_out=True
                    os.killpg(child.pid,signal.SIGKILL)
                    pid,status,usage=os.wait4(child.pid,0);break
                time.sleep(.02)
            child.returncode=os.waitstatus_to_exitcode(status)
        finally:
            if child.returncode is None:
                os.killpg(child.pid,signal.SIGKILL);child.wait()
    return dict(exit_code=child.returncode,timed_out=timed_out,wall_seconds=time.monotonic()-started,
                cpu_seconds=usage.ru_utime+usage.ru_stime,peak_rss_kib=usage.ru_maxrss)


def benchmark(args):
    binary=args.binary.resolve()
    if not binary.is_file():raise ValueError('solver binary missing')
    if args.seconds<=0 or args.memory_gib<=0 or not 0<args.cache_mib<args.memory_gib*1024:
        raise ValueError('invalid resource limits')
    args.output.mkdir(parents=True,exist_ok=False)
    with (ROOT/'results.txt').open() as f:known=list(csv.reader(f))
    results=[]
    for shape in args.shapes:
        rows,columns=map(int,shape.split('x'))
        case=args.output/shape;case.mkdir();case=case.resolve()
        text=dimacs(rows,columns,args.encoding,args.anchor)
        source=case/'grid.cnf';source.write_text(text)
        variables,clauses,factor=encode(rows,columns,args.encoding,args.anchor)
        command=[str(binary),'-cs',str(args.cache_mib)]
        if args.solver=='td':
            command+=['-decot',str(args.td_seconds),'-decow','100','-tmpdir',str(case)]
        command.append(str(source))
        # TD expects flow_cutter_pace17 beside its executable. Original sharpSAT
        # writes data.out in its cwd, so give each control run a private cwd.
        cwd=binary.parent if args.solver=='td' else case
        report=bounded_run(command,cwd,case/'solver.log',args.seconds,args.memory_gib)
        log=(case/'solver.log').read_text()
        count=parse_count(log,args.solver) if report['exit_code']==0 and not report['timed_out'] else None
        expected=known[rows-1][columns-1]
        labelled=None if count is None else factor*count
        with binary.open('rb') as source_binary:
            binary_digest=hashlib.file_digest(source_binary,'sha256').hexdigest()
        report.update(shape=shape,solver=args.solver,encoding=args.encoding,anchor=args.anchor,
                      restore_factor=factor,variables=variables,clauses=len(clauses),
                      labelled_count=None if labelled is None else str(labelled),
                      expected_count=expected or None,
                      status='timeout' if report['timed_out'] else ('complete' if count is not None else 'solver_failure'),
                      seconds_limit=args.seconds,memory_limit_gib=args.memory_gib,command=command,
                      binary_sha256=binary_digest,
                      cnf_sha256=hashlib.sha256(text.encode()).hexdigest())
        if labelled is not None and expected and labelled!=int(expected):report['status']='WRONG_COUNT'
        for name,pattern in [('decisions',r'^(?:c o )?decisions\s+(\d+)'),('conflicts',r'^(?:c o )?conflicts\s+(\d+)'),
                             ('treewidth',r'^c o (?:Treewidth|treewidth|tw|width)\s+(\d+)')]:
            values=re.findall(pattern,log,re.M)
            if values:report[name]=int(values[-1])
        (case/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        results.append(report)
        (args.output/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps({k:report[k] for k in ('shape','solver','encoding','anchor','status','wall_seconds','peak_rss_kib','labelled_count')}),flush=True)
        if report['status'] in ('WRONG_COUNT','solver_failure'):
            raise RuntimeError(f'{shape}: {report["status"]}; inspect {case}/solver.log')


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    emit=sub.add_parser('cnf');emit.add_argument('rows',type=int);emit.add_argument('columns',type=int)
    bench=sub.add_parser('bench');bench.add_argument('--binary',type=Path,required=True)
    bench.add_argument('--solver',choices=('original','td'),required=True)
    bench.add_argument('--shapes',nargs='+',default=['5x5','6x6','7x7'])
    bench.add_argument('--seconds',type=float,default=60);bench.add_argument('--memory-gib',type=float,default=4)
    bench.add_argument('--cache-mib',type=int,default=1024);bench.add_argument('--td-seconds',type=float,default=1)
    bench.add_argument('--output',type=Path,required=True)
    for cmd in (emit,bench):
        cmd.add_argument('--encoding',choices=('bits','onehot'),default='bits')
        cmd.add_argument('--anchor',action='store_true')
    args=p.parse_args()
    if args.command=='cnf':print(dimacs(args.rows,args.columns,args.encoding,args.anchor),end='')
    else:benchmark(args)


if __name__=='__main__':main()
