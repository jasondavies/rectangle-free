#!/usr/bin/env python3
"""Bounded, sequential CPU timing panel drawn from a completed parent census.

This is a purposive eight-family gate, not a campaign runtime estimator.
No remote resources are used. Keep incomplete/capped cases in the panel.
"""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import time


def select_panel(rows):
    panel = []
    for low, high in [(1, 7), (8, 19), (20, 79), (80, 256)]:
        group = sorted((r for r in rows if low <= r['fanout'] <= high),
                       key=lambda r: (r['tiles'], int(r['parent'])))
        if not group:
            raise ValueError(f'empty fanout band {low}..{high}')
        for quantile in [0.5, 0.9]:
            row = dict(group[min(len(group) - 1, int(len(group) * quantile))])
            row.update(fanout_band=[low, high], sampled_record_tile_quantile=quantile)
            panel.append(row)
    return panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('parents', type=Path)
    parser.add_argument('output', type=Path, help='new directory; never overwrite prior results')
    parser.add_argument('--binary', type=Path, default=Path('build/shared_column_family_census'))
    parser.add_argument('--seconds', type=float, default=30)
    args = parser.parse_args()
    if not 0 < args.seconds <= 300:
        parser.error('seconds must be in (0,300]')
    data = args.parents.read_bytes()
    records = [json.loads(line) for line in data.splitlines()]
    if not records or records[-1].get('type') != 'parent_complete':
        raise ValueError('parent census has no final completion marker')
    rows = [r for r in records if r.get('type') == 'parent_sample']
    if len(rows) != records[-1]['samples']:
        raise ValueError('parent census sample count mismatch')
    panel = select_panel(rows)
    binary = args.binary.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = dict(panel=panel, parent_sha256=hashlib.sha256(data).hexdigest(),
                    binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                    seconds_cap_per_method_side=args.seconds, work_cap=100000000000,
                    cache_cap=100000, address_space_cap_gib=8)
    (args.output / 'selection.json').write_text(json.dumps(metadata, indent=2) + '\n')
    resource.setrlimit(resource.RLIMIT_AS, (8 << 30, 8 << 30))
    for index, row in enumerate(panel):
        command = [str(binary), 'family', row['parent'], '100000',
                   '100000000000', str(args.seconds)]
        print(f"starting {index + 1}/{len(panel)} parent={row['parent']} fanout={row['fanout']}", flush=True)
        started = time.monotonic()
        with (args.output / f'family{index:02d}.jsonl').open('x') as out, \
             (args.output / f'family{index:02d}.stderr').open('x') as err:
            result = subprocess.run(command, stdout=out, stderr=err,
                                    timeout=8 * args.seconds + 120)
        logs = [json.loads(line) for line in
                (args.output / f'family{index:02d}.jsonl').read_text().splitlines()]
        if result.returncode or not logs or logs[-1].get('type') != 'family_complete':
            raise RuntimeError(f'family {index} failed; preserve its partial log, do not aggregate it')
        if logs[-1]['fanout'] != row['fanout']:
            raise RuntimeError('fanout differs from source census')
        # Isolate the cost of changing the column split/row gauge. The C bridge
        # uses the corpus generator's real representative, not a nauty key.
        modeled = subprocess.run([str(binary), 'model-parent', row['parent']],
                                 capture_output=True, text=True, timeout=120, check=True)
        (args.output / f'model{index:02d}.jsonl').write_text(modeled.stdout)
        models = [json.loads(line) for line in modeled.stdout.splitlines()]
        if not models or models[-1].get('type') != 'parent_model_complete':
            raise RuntimeError('missing parent model completion marker')
        sampled = next(m for m in models[:-1] if m['production_key'] == row['key'])
        if sampled['production_tiles'] != row['tiles']:
            raise RuntimeError('original sampled production cost changed')
        print(f"finished {index + 1}: {time.monotonic()-started:.3f}s "
              f"capped={logs[-1]['capped_method_sides']} "
              f"checked={logs[-1]['checked_answers']}", flush=True)
    (args.output / 'complete.json').write_text(json.dumps({'families': len(panel)}) + '\n')


if __name__ == '__main__':
    main()
