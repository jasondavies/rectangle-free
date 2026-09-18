#!/usr/bin/env python3
"""Bounded local structural comparison; no GPU timing or campaign rewrite."""
import argparse
import json
from pathlib import Path
import subprocess

from cut_gpu_gate import digest


def summarize(path):
    totals, joins, maximum, completion = {}, [], 0, None
    with path.open() as source:
        for line in source:
            row = json.loads(line)
            if completion is not None:
                raise ValueError('data after completion')
            if row['type'] == 'complete':
                completion = row
                continue
            if row['type'] != 'join' or (row['record'], row['side']) != divmod(len(joins), 2):
                raise ValueError('unexpected join order/type')
            joins.append(row['tiles'])
            maximum = max(maximum, row['max_bucket_tiles'])
            for key, value in row.items():
                if key in ('type', 'record', 'side', 'max_bucket_tiles'):
                    continue
                if type(value) is not int or value < 0:
                    raise ValueError('invalid feature counter')
                totals[key] = totals.get(key, 0) + value
    if not completion or not joins or len(joins) != 2*completion['records']:
        raise ValueError('incomplete join census')
    ordered = sorted(joins)
    quantiles = {str(q):ordered[min(len(ordered)-1, int(q*(len(ordered)-1)))]
                 for q in (0.5, 0.95, 0.99, 1.0)}
    return dict(totals=totals, join_tile_quantiles=quantiles,
                top_one_percent_tile_share=sum(ordered[-max(1,(len(ordered)+99)//100):])/max(1,sum(ordered)),
                max_bucket_tiles=maximum,
                tile_occupancy=totals['useful_pairs']/max(1,128*totals['tiles']),
                completion=completion)


def run(payload, output, binary, panels, seconds):
    if not panels or len(set(panels)) != len(panels):
        raise ValueError('unique nonempty panel IDs required')
    original = json.loads((payload/'config.json').read_text())
    if not set(panels) <= {p['id'] for p in original['panels']}:
        raise ValueError('unknown panel')
    if not 0 < seconds <= 600:
        raise ValueError('time cap must be in (0,600]')
    output.mkdir(parents=True, exist_ok=False)
    binary = binary.resolve()
    config = dict(binary_sha256=digest(binary), inputs={}, panels=panels, seconds=seconds,
                  note='Exact structural counts, not a device profile; no orbit coefficient weighting.')
    summary = {}
    for panel in panels:
        variants = {}
        for variant in ('baseline', 'adaptive'):
            name = f'p{panel:02d}-{variant}'
            relative = f'inputs/{name}.orbits'
            source = payload/relative
            checksum = digest(source)
            if checksum != original['files'][relative]:
                raise ValueError('input hash differs from GPU benchmark')
            config['inputs'][relative] = checksum
            (output/'config.json').write_text(json.dumps(config, indent=2)+'\n')
            with (output/f'{name}.jsonl').open('xb') as out, (output/f'{name}.log').open('xb') as log:
                subprocess.run([str(binary), str(source), str(seconds)], stdout=out, stderr=log,
                               check=True, timeout=seconds+30)
            variants[variant] = summarize(output/f'{name}.jsonl')
            expected = next(p['records'] for p in original['panels'] if p['id']==panel)
            if variants[variant]['completion']['records'] != expected:
                raise ValueError('panel record count mismatch')
            print(json.dumps(dict(panel=panel, variant=variant, **variants[variant])), flush=True)
        summary[str(panel)] = variants
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('payload', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--binary', type=Path, default=Path('build/cut_join_features'))
    p.add_argument('--panels', type=int, nargs='+', default=[2,3])
    p.add_argument('--seconds', type=float, default=120)
    a=p.parse_args()
    run(a.payload,a.output,a.binary,a.panels,a.seconds)
