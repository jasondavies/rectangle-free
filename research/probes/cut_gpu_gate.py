#!/usr/bin/env python3
"""Matched production-solver A/B for research cuts; never a campaign corpus."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import selectors
import shutil
import statistics
import struct
import subprocess
import time


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def assemble(left, right):
    return sum((((left >> (4*r)) & 15) | (((right >> (4*r)) & 15) << 4)) << (8*r)
               for r in range(8))


def write_orbits(path, rows):
    with path.open('xb') as f:
        f.write(struct.pack('<8sIQ', b'R8SQT01\0', 8, len(rows)))
        for key, weight in rows:
            if key.bit_count() > 32 or not 0 < weight < 2**64:
                raise ValueError('invalid minority-side probe record')
            f.write(struct.pack('<QQ', key, weight))


def prepare(source, output, binary):
    import cut_batch_gate as batch
    import reuse_budget_cut as cut
    output.mkdir(parents=True, exist_ok=False)
    (output/'inputs').mkdir()
    seed = set()
    config = dict(kind='research-only matched cuts, not a canonical orbit corpus',
                  rounds=4, batch_edges=4096, panels=[])
    for i, panel in enumerate(sorted(source.glob('owner-*'))):
        layouts, records = cut.load(panel/'cost.warm.jsonl')
        batch.validate_records(records, panel/'records.tsv')
        selected = json.loads((panel/'result.json').read_text())
        if selected['input_sha256'] != digest(panel/'records.tsv'):
            raise ValueError('selection is bound to different input records')
        choices = selected['adaptive']['choices']
        if len(choices) != len(records):
            raise ValueError('selection length mismatch')
        baseline, adaptive = [], []
        for row, slot in zip(records, choices):
            if not isinstance(slot, int) or not 0 <= slot < len(row['choices']):
                raise ValueError('invalid selection index')
            a = row['choices'][slot]
            key, weight = int(row['key']), int(row['weight'])
            new = assemble(a['left'], a['right'])
            if new.bit_count() != key.bit_count():
                raise ValueError('outer complement factor changed')
            baseline.append((key, weight))
            adaptive.append((new, weight))
            seed.update((key, new))
        for name, data in [('baseline', baseline), ('adaptive', adaptive)]:
            write_orbits(output/'inputs'/f'p{i:02d}-{name}.orbits', data)
        config['panels'].append(dict(id=i, records=len(records),
                                      input_sha256=digest(panel/'records.tsv'),
                                      selection_sha256=digest(panel/'result.json'),
                                      labelled_weight=sum(w for _, w in baseline),
                                      covered_weight=sum(w*(2 if k.bit_count()<32 else 1) for k,w in baseline)))
    if len(config['panels']) != 8:
        raise ValueError('this gate requires the eight Experiment 511 panels')
    # Same superset seed in both variants. Its weights are irrelevant to the
    # canonical factory, and it is never treated as a solve work item.
    write_orbits(output/'inputs/seed.orbits', [(k, 1) for k in sorted(seed)])
    with (output/'check.tsv').open('x') as f:
        for p in config['panels']:
            for variant in ('baseline', 'adaptive'):
                f.write(f"check-p{p['id']:02d}-{variant} inputs/p{p['id']:02d}-{variant}.orbits 0 0\n")
    with (output/'work.tsv').open('x') as f:
        for variant in ('baseline', 'adaptive'):
            f.write(f'warmup-{variant} inputs/p00-{variant}.orbits 0 0\n')
        for rep in range(config['rounds']):
            order = ('baseline', 'adaptive') if rep % 2 == 0 else ('adaptive', 'baseline')
            for p in config['panels']:
                for variant in order:
                    f.write(f"r{rep}-p{p['id']:02d}-{variant} inputs/p{p['id']:02d}-{variant}.orbits 0 0\n")
    shutil.copy2(binary, output/'solver')
    shutil.copy2(Path(__file__), output/'gate.py')
    config['files'] = {str(p.relative_to(output)):digest(p) for p in sorted(output.rglob('*')) if p.is_file()}
    (output/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    print(json.dumps(dict(records=sum(p['records'] for p in config['panels']),
                          seed_records=len(seed), payload_bytes=sum(p.stat().st_size for p in output.rglob('*') if p.is_file()))))


def parse_result(path):
    lines = path.read_text().splitlines(keepends=True)
    if not lines or lines[0] != 'RECT8X8_PREFIX_RESULT 3\n' or not lines[-1].startswith('result_payload_sha256 '):
        raise ValueError('invalid v3 result framing')
    if hashlib.sha256(''.join(lines[1:-1]).encode()).hexdigest() != lines[-1].split()[1]:
        raise ValueError('result payload checksum mismatch')
    fields = {}
    for line in lines:
        key, *rest = line.rstrip('\n').split(maxsplit=1)
        if key in fields:
            raise ValueError('duplicate result field')
        fields[key] = rest[0] if rest else ''
    return fields


def summarize(root):
    config = json.loads((root/'config.json').read_text())
    for file, expected in config['files'].items():
        if digest(root/file) != expected:
            raise ValueError('payload hash mismatch: '+file)
    panels = []
    measures = ['total_seconds', 'gpu_seconds', 'left_layout_seconds',
                'right_layout_seconds', 'canonical_resolve_seconds', 'load_seconds']
    checks = 0
    for p in config['panels']:
        i = p['id']
        expected = None
        variants = {}
        for variant in ('baseline', 'adaptive'):
            rows = []
            for prefix, directory in [('check', 'check-results'), *[(f'r{r}', 'results') for r in range(config['rounds'])]]:
                row = parse_result(root/directory/f'{prefix}-p{i:02d}-{variant}.result')
                expected_path = f'inputs/p{i:02d}-{variant}.orbits'
                if (row['id'] != f'{prefix}-p{i:02d}-{variant}' or row['path'] != expected_path or
                    row['solver_binary_sha256'] != config['files']['solver'] or
                    row['canonical_cache_sha256'] != config['files']['inputs/seed.orbits'] or
                    row['orbit_corpus_sha256'] != config['files'][expected_path] or
                    row['geometry'] != '8x8' or row['token_plane_quotient'] != '1'):
                    raise ValueError('result provenance mismatch')
                if int(row['records']) != p['records'] or int(row['kernels']) != p['records']:
                    raise ValueError('GPU record coverage mismatch')
                if int(row['labelled_weight']) != p['labelled_weight'] or int(row['covered_weight']) != p['covered_weight']:
                    raise ValueError('GPU coefficient coverage mismatch')
                value = int(row['contribution'])
                if expected is not None and value != expected:
                    raise ValueError('baseline/adaptive or repeat contribution mismatch')
                expected = value
                if prefix == 'check':
                    if int(row['verified']) != 2:
                        raise ValueError('CPU join checks missing')
                    checks += 2
                else:
                    if float(row['cache_factory_seconds']) or float(row['canonical_upload_seconds']) or float(row['validation_seconds']):
                        raise ValueError('timed item contains cold initialization or validation')
                    rows.append(row)
            variants[variant] = dict(samples=rows, median={k:statistics.median(float(r[k]) for r in rows) for k in measures})
        panels.append(dict(panel=i, contribution=str(expected), **variants))
    totals = {v:{k:sum(p[v]['median'][k] for p in panels) for k in measures} for v in ('baseline','adaptive')}
    result = dict(exact=True, cpu_join_checks=checks, panels=panels, sums_of_panel_medians=totals,
                  recurring_time_reduction_percent=100*(1-totals['adaptive']['total_seconds']/totals['baseline']['total_seconds']))
    return result


def run(root):
    os.chdir(root)
    if Path('results').exists() or Path('check-results').exists():
        raise ValueError('fresh benchmark output required; do not mix interrupted timings')
    config = json.loads(Path('config.json').read_text())
    for file, expected in config['files'].items():
        if digest(Path(file)) != expected:
            raise ValueError('payload hash mismatch')
    env = dict(os.environ, OMP_NUM_THREADS='16')
    # Verify separately, so scalar reference work never pollutes timed samples.
    for manifest, directory, verify, limit in [('check.tsv','check-results',2,300), ('work.tsv','results',0,900)]:
        start = time.monotonic()
        intervals = []
        last = start
        child = subprocess.Popen(['stdbuf','-oL','./solver','inputs/seed.orbits',manifest,directory,str(config['batch_edges']),str(verify)],
                                 env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        poller = selectors.DefaultSelector()
        poller.register(child.stdout, selectors.EVENT_READ)
        pending = b''
        try:
            with open(directory+'.log','xb') as log:
                while poller.get_map():
                    if time.monotonic()-start > limit:
                        raise TimeoutError('bounded solver run timed out')
                    for key, _ in poller.select(1):
                        chunk = os.read(key.fd,65536)
                        if not chunk:
                            poller.unregister(key.fileobj)
                            continue
                        log.write(chunk); log.flush()
                        pending += chunk
                        while b'\n' in pending:
                            line,pending = pending.split(b'\n',1)
                            if line.startswith(b'CHECKPOINT id='):
                                now = time.monotonic()
                                ident = line.split()[1].split(b'=',1)[1].decode()
                                intervals.append(dict(id=ident, seconds=now-last))
                                last = now
                if child.wait(timeout=5):
                    raise RuntimeError('solver failed; see '+directory+'.log')
        finally:
            if child.poll() is None:
                child.kill(); child.wait(timeout=10)
            child.stdout.close(); poller.close()
        # These external consecutive-checkpoint intervals include publication
        # and the preceding item's left release; the current release belongs
        # to the next interval. Report alongside solver phase timings, not as
        # perfectly isolated per-item latency. Alternating order limits bias.
        Path(directory+'-intervals.json').write_text(json.dumps(intervals,indent=2)+'\n')
        Path(directory+'-wall.json').write_text(json.dumps(dict(seconds=time.monotonic()-start))+'\n')
    result = summarize(Path('.'))
    Path('summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'panels'}),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('source',type=Path); p.add_argument('output',type=Path); p.add_argument('binary',type=Path)
    p = sub.add_parser('run'); p.add_argument('root',type=Path)
    p = sub.add_parser('summarize'); p.add_argument('root',type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.source,args.output,args.binary)
    elif args.command == 'run':
        run(args.root.resolve())
    else:
        print(json.dumps(summarize(args.root),indent=2))
