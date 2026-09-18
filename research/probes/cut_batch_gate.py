#!/usr/bin/env python3
"""Bounded CPU-only adaptive-cut gate; models output buffers, NOT total VRAM."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import struct
import sys
import time

import reuse_budget_cut as cut

ROOT = Path(__file__).resolve().parents[2]


def batch_plan(layouts, records, choices, byte_cap, edge_cap=4096):
    """Production-like right ordering: a right group is never split.

    All chosen left layouts stay resident for this owner. Only one right
    output batch is resident. Canonical sources, scratch, high-water arenas,
    descriptors/results and CUDA reserves are NOT included in byte_cap.
    Different owners call this independently, never sharing right builds.
    """
    if byte_cap <= 0 or edge_cap <= 0 or len(records) != len(choices):
        raise ValueError('invalid budget or choice count')
    groups = {}
    for i, (r, choice) in enumerate(zip(records, choices)):
        if not 0 <= choice < len(r['choices']):
            raise ValueError('invalid choice')
        c = r['choices'][choice]
        if c['right'] not in layouts or c['left'] not in layouts:
            raise ValueError('missing layout')
        groups.setdefault(c['right'], []).append(i)
    batches = []
    current = dict(bytes=0, entries=0, layouts=0, records=0, tiles=0)
    for right, indices in sorted(groups.items()):
        layout = layouts[right]
        if layout['bytes'] > byte_cap or len(indices) > edge_cap:
            raise ValueError('one right group exceeds batch budget')
        if current['records'] and (current['bytes'] + layout['bytes'] > byte_cap
                                  or current['records'] + len(indices) > edge_cap):
            batches.append(current)
            current = dict(bytes=0, entries=0, layouts=0, records=0, tiles=0)
        current['bytes'] += layout['bytes']
        current['entries'] += layout['entries']
        current['layouts'] += 1
        current['records'] += len(indices)
        current['tiles'] += sum(records[i]['choices'][choices[i]]['tiles'] for i in indices)
    if current['records']:
        batches.append(current)
    return dict(output_byte_cap=byte_cap, edge_cap=edge_cap, batches=batches,
                batch_count=len(batches),
                right_build_bytes=sum(b['bytes'] for b in batches),
                right_build_entries=sum(b['entries'] for b in batches),
                right_build_layouts=sum(b['layouts'] for b in batches),
                peak_right_output_bytes=max((b['bytes'] for b in batches), default=0),
                tiles=sum(b['tiles'] for b in batches),
                records=sum(b['records'] for b in batches))


def half(key, columns):
    result = 0
    for r in range(8):
        bits = [((key >> ((7-r)*8+c)) & 1) for c in range(8) if columns >> c & 1]
        result = (result << 4) | sum(b << i for i, b in enumerate(bits))
    return result


def validate_records(records, input_path):
    """Independent bit extraction; checks identities/weights, not join answers."""
    original = [tuple(map(int, line.split())) for line in input_path.read_text().splitlines()]
    if len(original) != len(records):
        raise ValueError('record count changed')
    for row, (source, index, key, weight) in zip(records, original):
        if (row['source'], row['index'], int(row['key']), int(row['weight'])) != (source, index, key, weight):
            raise ValueError('record identity or coefficient changed')
        transposed = sum(((key >> ((7-r)*8+c)) & 1) << ((7-c)*8+r)
                         for r in range(8) for c in range(8))
        for c in row['choices']:
            if c['columns'] not in range(256) or c['columns'].bit_count() != 4:
                raise ValueError('invalid cut')
            k = transposed if c['transpose'] else key
            pair = (half(k, c['columns']), half(k, c['columns'] ^ 255))
            if c['reverse']:
                pair = pair[::-1]
            if pair != (c['left'], c['right']):
                raise ValueError('cut changed underlying mask')
        b = row['choices'][0]
        if (b['columns'], b['transpose'], b['reverse']) != (15, 0, 0):
            raise ValueError('baseline changed')


def analyze(cold_path, warm_path, input_path, selection_path, caps):
    start = time.monotonic()
    layouts, records = cut.load(warm_path)
    cold_layouts, cold_records = cut.load(cold_path)
    if (layouts, records) != (cold_layouts, cold_records):
        raise ValueError('cold/warm exact metadata mismatch')
    del cold_layouts, cold_records
    validate_records(records, input_path)
    check_seconds = time.monotonic() - start
    baseline = cut.statistics(layouts, records, [0]*len(records))
    start = time.monotonic()
    selected = cut.optimize(layouts, records, Fraction(2), Fraction(1000000))
    selector_seconds = time.monotonic() - start
    start = time.monotonic()
    plans = []
    for cap in caps:
        b = batch_plan(layouts, records, [0]*len(records), cap)
        a = batch_plan(layouts, records, selected['choices'], cap)
        if b['tiles'] != baseline['tiles'] or a['tiles'] != selected['tiles']:
            raise ValueError('batching changed tile sum')
        if b['records'] != len(records) or a['records'] != len(records):
            raise ValueError('batching changed coverage')
        plans.append(dict(baseline=b, adaptive=a))
    batch_seconds = time.monotonic() - start
    selection = json.loads(selection_path.read_text())
    groups = []
    for group in selection['groups']:
        indices = [i for i, r in enumerate(records) if r['choices'][0]['left'] == group['left']]
        if len(indices) != group['records']:
            raise ValueError('group incomplete')
        b = sum(records[i]['choices'][0]['tiles'] for i in indices)
        a = sum(records[i]['choices'][selected['choices'][i]]['tiles'] for i in indices)
        groups.append(dict(group, baseline_tiles=b, adaptive_tiles=a))
    return dict(records=len(records), input_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
                baseline=baseline, adaptive=selected, plans=plans, groups=groups,
                selector_seconds=selector_seconds, batch_seconds=batch_seconds,
                validation_seconds=check_seconds, exact_metadata_check=True)


def command(args, output, log):
    start = time.monotonic()
    with output.open('x') as out, log.open('x') as err:
        subprocess.run(['/usr/bin/time', '-v', *map(str, args)], stdout=out, stderr=err, check=True)
    return time.monotonic()-start


def check_reference(panel):
    """Check the largest baseline-cost record in each group using the old scorer.

    Run separately after performance measurement to avoid CPU contention.
    This verifies metadata/tile costs, not a new complete colouring count.
    """
    layouts, records = cut.load(panel/'cost.warm.jsonl')
    selected = {}
    for row in records:
        left = row['choices'][0]['left']
        if left not in selected or row['choices'][0]['tiles'] > selected[left]['choices'][0]['tiles']:
            selected[left] = row
    selected_ids = {(r['source'], r['index']) for r in selected.values()}
    for source, dest in [('records.tsv', 'reference-input.tsv'),
                         ('shortlist/shortlist.tsv', 'reference-shortlist.tsv')]:
        with (panel/dest).open('x') as out:
            for line in (panel/source).read_text().splitlines():
                if tuple(map(int, line.split()[:2])) in selected_ids:
                    out.write(line+'\n')
    seconds = command([ROOT/'build/reuse_budget_cut_census', panel/'reference-input.tsv',
                       'all', panel/'reference-shortlist.tsv'],
                      panel/'reference.jsonl', panel/'reference.log')
    ref_layouts, ref_records = cut.load(panel/'reference.jsonl')
    expected = [r for r in records if (r['source'], r['index']) in selected_ids]
    if expected != ref_records or any(layouts[k] != v for k, v in ref_layouts.items()):
        raise ValueError('independent Cartesian reference mismatch')
    result = dict(exact=True, records=len(ref_records),
                  candidates=sum(len(r['choices']) for r in ref_records), seconds=seconds)
    (panel/'reference-check.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def population_census(paths, bands):
    """Describe sampling coverage; do not treat group samples as work-unbiased."""
    import numpy as np
    out = []
    for path in paths:
        with path.open('rb') as f:
            magic, width, count = struct.unpack('<8sIQ', f.read(20))
            f.seek(0)
            digest = hashlib.file_digest(f, 'sha256').hexdigest()
        if magic not in (b'R8SQT01\0', b'R8ORB01\0') or width != 8 or path.stat().st_size != 20+16*count:
            raise ValueError('invalid eight-row corpus')
        data = np.memmap(path, mode='r', offset=20, shape=(count,), dtype=[('key', '<u8'), ('weight', '<u8')])
        left = np.zeros(count, dtype=np.uint32)
        for row in range(8):
            left = (left << 4) | ((data['key'] >> ((7-row)*8)) & 15).astype(np.uint32)
        _, counts = np.unique(left, return_counts=True)
        spans = []
        for low, high in bands:
            mask = (counts >= low) & (counts <= high)
            spans.append(dict(band=[low, high], groups=int(mask.sum()), records=int(counts[mask].sum())))
        out.append(dict(path=str(path.resolve()), sha256=digest, records=count, groups=len(counts),
                        maximum_fanout=int(counts.max()), bands=spans))
    return out


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    # Fix sampling before seeing any costs. Each owner has 4 complete groups;
    # the largest possible panel is 13,564 records, below the exporter cap.
    bands = [(64, 255), (256, 1023), (1024, 4095), (4096, 8191)]
    config = dict(seed=args.seed, bands=bands, files=list(map(str, args.files)),
                  support_cache=str(args.cache), output_caps_mib=[64, 256, 1024],
                  note='Independent owner budgets; output-only batching, not a VRAM or runtime prediction.')
    (args.output/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    summaries = []
    cache = args.cache
    for i, path in enumerate(args.files):
        panel = args.output/f'owner-{i:02d}'
        t = time.monotonic()
        cut.sample([path], panel, args.seed+i, 1, bands)
        times = dict(sample=time.monotonic()-t)
        times['counts'] = command([ROOT/'build/cut_support_counts', panel/'records.tsv',
                                   panel/'cache.tsv', cache], panel/'counts.jsonl', panel/'counts.log')
        cache = panel/'cache.tsv'
        times['shortlist'] = command([sys.executable, ROOT/'research/probes/cut_shortlist.py',
                                      panel/'counts.jsonl', panel/'shortlist', '--mode', 'reuse', '--cuts', '4'],
                                     panel/'shortlist.summary', panel/'shortlist.log')
        times['score_cold_and_warm'] = command([ROOT/'build/cut_histogram_census', panel/'records.tsv',
                                               panel/'shortlist/shortlist.tsv', panel/'cost', 'demand'],
                                              panel/'cost.summary', panel/'cost.log')
        result = analyze(panel/'cost.cold.jsonl', panel/'cost.warm.jsonl', panel/'records.tsv',
                         panel/'selection.json', [m*1024**2 for m in config['output_caps_mib']])
        result['wall_seconds'] = times
        for name in ['cold', 'warm']:
            with (panel/f'cost.{name}.jsonl').open() as f:
                for line in f:
                    last = line
            result[name+'_scoring'] = json.loads(last)
        (panel/'result.json').write_text(json.dumps(result, indent=2)+'\n')
        concise = {k:v for k,v in result.items() if k not in ('plans', 'adaptive')}
        concise['adaptive'] = {k:v for k,v in result['adaptive'].items() if k != 'choices'}
        concise['plans'] = [{s:{k:v for k,v in p[s].items() if k != 'batches'}
                            for s in ('baseline', 'adaptive')} for p in result['plans']]
        summaries.append(concise)
        (args.output/'summary.json').write_text(json.dumps(summaries, indent=2)+'\n')
        print(json.dumps(dict(owner=i, records=result['records'], baseline_tiles=result['baseline']['tiles'],
                              adaptive_tiles=result['adaptive']['tiles'],
                              warm_seconds=result['warm_scoring']['seconds'])), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    p.add_argument('files', nargs='+', type=Path)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--seed', type=int, default=511)
    a = p.parse_args()
    if len({f.resolve() for f in a.files}) != len(a.files):
        p.error('duplicate owner file')
    run(a)
