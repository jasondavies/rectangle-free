#!/usr/bin/env python3
"""Support-only shortlist gate; exact tile metadata is used only AFTER ranking."""
import argparse
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path
import time

import reuse_budget_cut as cut


def load_counts(path):
    supports, records, final = {}, [], None
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if final is not None:
            raise ValueError('data after completion')
        if row['type'] == 'support':
            if row['key'] in supports or min(row['selected'], row['complement']) < 0:
                raise ValueError('invalid support')
            supports[row['key']] = row['selected'] + row['complement']
        elif row['type'] == 'record':
            if len(row['choices']) != 140:
                raise ValueError('expected all 70 cuts in both directions')
            records.append(row)
        elif row['type'] == 'complete':
            final = row
        else:
            raise ValueError('unknown count row')
    if not final or final['records'] != len(records) or final['supports'] != len(supports):
        raise ValueError('incomplete count census')
    if len({(r['source'], r['index']) for r in records}) != len(records):
        raise ValueError('duplicate record')
    for r in records:
        for c in r['choices']:
            if c['left'] not in supports or c['right'] not in supports or c['product'] < 0:
                raise ValueError('invalid support reference')
    return supports, records


def shortlist(supports, records, count, mode):
    """Always baseline plus count other cuts, each in both directions.

    reuse ranks predicted support-product saving divided by amortized NEW
    support entries. Frequencies count records, not repeated candidate slots.
    mixed reserves one choice for raw product, filling the rest by reuse.
    No exact prefix, class, byte, or tile data enters this function.
    """
    if count not in (2, 4) or mode not in ('product', 'reuse', 'mixed'):
        raise ValueError('unsupported shortlist configuration')
    sides = ('left', 'right')
    opened = {s: {r['choices'][0][s] for r in records} for s in sides}
    freq = {s: Counter(k for r in records for k in {c[s] for c in r['choices']}) for s in sides}
    total = {s: max(1, sum(supports[k] for k in opened[s])) for s in sides}
    answer = []
    for r in records:
        choices = r['choices']
        raw = sorted(range(1, 70), key=lambda i: (choices[2*i]['product'], i))
        def reuse_score(i):
            charge = min(sum(0 if c[s] in opened[s] else
                             supports[c[s]] / freq[s][c[s]] / total[s]
                             for s in sides) for c in choices[2*i:2*i+2])
            gain = max(0, choices[0]['product'] - choices[2*i]['product'])
            return (-gain / max(charge, 1e-12), choices[2*i]['product'], i)
        reuse = sorted(range(1, 70), key=reuse_score)
        order = raw if mode == 'product' else reuse
        if mode == 'mixed':
            order = raw[:1] + [i for i in reuse if i != raw[0]]
        slots = [0] + sorted(s for i in order[:count] for s in (2*i, 2*i+1))
        answer.append(slots)
    return answer


def evaluate(supports, count_records, exact_path, slots):
    layouts, records = cut.load(exact_path)
    if len(records) != len(count_records):
        raise ValueError('record count mismatch')
    for key, n in supports.items():
        if key not in layouts or layouts[key]['entries'] != n:
            raise ValueError('support count differs from exact labelled layout')
    restricted = []
    for r, cheap, keep in zip(records, count_records, slots):
        if (r['source'], r['index']) != (cheap['source'], cheap['index']):
            raise ValueError('record identity mismatch')
        for c, d in zip(r['choices'], cheap['choices']):
            if (c['left'], c['right']) != (d['left'], d['right']):
                raise ValueError('candidate ordering mismatch')
        if len(r['choices']) != 140:
            raise ValueError('evaluation requires complete all-cut control')
        restricted.append(dict(r, choices=[r['choices'][i] for i in keep]))
    start = time.monotonic()
    result = cut.optimize(layouts, restricted, Fraction(2), Fraction(1000000))
    result['selector_seconds'] = time.monotonic() - start
    result['baseline_tiles'] = sum(r['choices'][0]['tiles'] for r in records)
    result['saving_fraction'] = 1 - result['tiles'] / result['baseline_tiles']
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('counts', type=Path)
    p.add_argument('output', type=Path, help='new output directory')
    p.add_argument('--mode', choices=['product', 'reuse', 'mixed'], required=True)
    p.add_argument('--cuts', type=int, choices=[2, 4], required=True)
    p.add_argument('--exact', type=Path, help='optional all-cut control for retrospective evaluation')
    a = p.parse_args()
    supports, records = load_counts(a.counts)
    start = time.monotonic()
    slots = shortlist(supports, records, a.cuts, a.mode)
    report = dict(mode=a.mode, cuts=a.cuts, records=len(records),
                  ranking_seconds=time.monotonic()-start, scored_slots=1+2*a.cuts)
    a.output.mkdir(parents=True, exist_ok=False)
    with (a.output/'shortlist.tsv').open('x') as f:
        for r, keep in zip(records, slots):
            f.write('\t'.join(map(str, [r['source'], r['index'], *keep]))+'\n')
    if a.exact:
        report['evaluation'] = evaluate(supports, records, a.exact, slots)
    (a.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'evaluation'}))
    if 'evaluation' in report:
        print(json.dumps({k: v for k, v in report['evaluation'].items() if k != 'choices'}))


if __name__ == '__main__':
    main()
