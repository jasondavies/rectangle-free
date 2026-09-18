#!/usr/bin/env python3
"""CPU-only, reuse-budgeted column-cut research gate. No corpus is rewritten."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import struct


def sample(paths, output, seed, per_band, bands=None):
    import numpy as np
    rng = np.random.default_rng(seed)
    output.mkdir(parents=True, exist_ok=False)
    bands = bands or [(8, 31), (64, 255), (256, 1023)]
    if any(low < 1 or high < low for low, high in bands):
        raise ValueError('invalid fanout band')
    metadata = dict(seed=seed, per_band=per_band, bands=bands, files=[], groups=[])
    records = []
    for source, path in enumerate(paths):
        with path.open('rb') as f:
            header = f.read(20)
        if len(header) != 20:
            raise ValueError('truncated header')
        magic, width, count = struct.unpack('<8sIQ', header)
        if magic not in (b'R8SQT01\0', b'R8ORB01\0') or width != 8 or path.stat().st_size != 20 + 16 * count:
            raise ValueError('invalid eight-row corpus')
        data = np.memmap(path, mode='r', offset=20, shape=(count,),
                         dtype=[('key', '<u8'), ('weight', '<u8')])
        left = np.zeros(count, dtype=np.uint32)
        for row in range(8):
            left = (left << 4) | ((data['key'] >> ((7-row)*8)) & 15).astype(np.uint32)
        keys, counts = np.unique(left, return_counts=True)
        chosen = []
        for low, high in bands:
            eligible = np.flatnonzero((counts >= low) & (counts <= high))
            if len(eligible) < per_band:
                raise ValueError('not enough complete left groups in band')
            for i in sorted(rng.choice(eligible, per_band, replace=False)):
                chosen.append(int(keys[i]))
                metadata['groups'].append(dict(source=source, left=int(keys[i]),
                                               records=int(counts[i]), band=[low, high],
                                               eligible_groups=len(eligible)))
        indices = np.flatnonzero(np.isin(left, chosen))
        for i in indices:
            records.append((source, int(i), int(data['key'][i]), int(data['weight'][i])))
        metadata['files'].append(dict(path=str(path.resolve()), records=count,
                                     left_groups=len(keys), selected_records=len(indices)))
    if not records or len(records) > 16384:
        raise ValueError('bounded panel must contain 1..16384 records')
    payload = ''.join('\t'.join(map(str, r)) + '\n' for r in records)
    metadata['input_sha256'] = hashlib.sha256(payload.encode()).hexdigest()
    (output / 'records.tsv').write_text(payload)
    (output / 'selection.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(dict(records=len(records), groups=len(metadata['groups']))))


def load(path):
    layouts, records, final = {}, [], None
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if final is not None:
            raise ValueError('data after completion marker')
        if row['type'] == 'layout':
            if row['key'] in layouts:
                raise ValueError('duplicate layout')
            layouts[row['key']] = row
        elif row['type'] == 'record':
            if not row['choices'] or any(c['tiles'] < 0 for c in row['choices']):
                raise ValueError('invalid choices')
            records.append(row)
        elif row['type'] == 'complete':
            final = row
        else:
            raise ValueError('unknown row type')
    if not final or final['records'] != len(records) or final['layouts'] != len(layouts):
        raise ValueError('incomplete candidate census')
    ids = {(r['source'], r['index']) for r in records}
    if len(ids) != len(records):
        raise ValueError('duplicate corpus records')
    for r in records:
        for c in r['choices']:
            if c['left'] not in layouts or c['right'] not in layouts:
                raise ValueError('unresolved layout')
    return layouts, records


def statistics(layouts, records, choices):
    result = dict(tiles=0, changed_records=0)
    for side in ['left', 'right']:
        keys = {r['choices'][c][side] for r, c in zip(records, choices)}
        result[side + '_keys'] = len(keys)
        result[side + '_bytes'] = sum(layouts[k]['bytes'] for k in keys)
        result[side + '_entries'] = sum(layouts[k]['entries'] for k in keys)
    for r, c in zip(records, choices):
        result['tiles'] += r['choices'][c]['tiles']
        result['changed_records'] += c != 0
    return result


def optimize(layouts, records, growth, identity_growth=None):
    """Greedy left-facility bundles, charging each opened layout only once.

    Keep baseline facilities reserved, even if a later move stops using one.
    Thus the budget is conservative. This is a feasible heuristic, not an
    exact optimum or a proof that other bounded selectors cannot do better.
    """
    choices = [0] * len(records)
    baseline = statistics(layouts, records, choices)
    opened = {s: {r['choices'][0][s] for r in records} for s in ['left', 'right']}
    used = {s: sum(layouts[k]['bytes'] for k in opened[s]) for s in opened}
    cap_bytes = {s: int(baseline[s+'_bytes'] * growth) for s in opened}
    identity_growth = growth if identity_growth is None else identity_growth
    cap_keys = {s: int(baseline[s+'_keys'] * identity_growth) for s in opened}
    groups = {}
    for i, r in enumerate(records):
        for j, c in enumerate(r['choices']):
            groups.setdefault(c['left'], []).append((i, j))
    rounds = 0
    while True:
        best = None
        for left, edges in sorted(groups.items()):
            delta_left = 0 if left in opened['left'] else layouts[left]['bytes']
            if used['left'] + delta_left > cap_bytes['left'] or len(opened['left']) + (left not in opened['left']) > cap_keys['left']:
                continue
            options = []
            for i, j in edges:
                c = records[i]['choices'][j]
                gain = records[i]['choices'][choices[i]]['tiles'] - c['tiles']
                if gain <= 0:
                    continue
                right_cost = 0 if c['right'] in opened['right'] else layouts[c['right']]['bytes']
                options.append((gain / max(1, right_cost), gain, i, j))
            options.sort(reverse=True)
            new_right, moved, proposal = set(), set(), []
            right_bytes = gain = 0
            for _, saving, i, j in options:
                if i in moved:
                    continue
                right = records[i]['choices'][j]['right']
                extra = right not in opened['right'] and right not in new_right
                cost = layouts[right]['bytes'] if extra else 0
                if used['right'] + right_bytes + cost > cap_bytes['right'] or len(opened['right']) + len(new_right) + extra > cap_keys['right']:
                    continue
                if extra:
                    new_right.add(right)
                right_bytes += cost
                gain += saving
                moved.add(i)
                proposal.append((i, j))
            if not gain:
                continue
            charge = (delta_left / max(1, baseline['left_bytes']) +
                      right_bytes / max(1, baseline['right_bytes']) +
                      (left not in opened['left']) / baseline['left_keys'] +
                      len(new_right) / baseline['right_keys'])
            score = gain / max(1e-12, charge)
            if best is None or (score, gain) > best[:2]:
                best = (score, gain, left, new_right, delta_left, right_bytes, proposal)
        if best is None:
            break
        _, gain, left, right, dl, dr, proposal = best
        for i, j in proposal:
            choices[i] = j
        opened['left'].add(left)
        opened['right'].update(right)
        used['left'] += dl
        used['right'] += dr
        rounds += 1
    result = statistics(layouts, records, choices)
    assert result['tiles'] <= baseline['tiles']
    for s in opened:
        assert sum(layouts[k]['bytes'] for k in opened[s]) == used[s] <= cap_bytes[s]
        assert result[s+'_bytes'] <= used[s]
        assert len(opened[s]) <= cap_keys[s]
        result[s+'_reserved_bytes'] = used[s]
        result[s+'_reserved_keys'] = len(opened[s])
    result.update(rounds=rounds, growth=str(growth), identity_growth=str(identity_growth), choices=choices)
    return result


def savings_upper_bound(layouts, records, growth, allow_retirement=False):
    """Optimistic gain: right layouts free, overlap double-counted, fractional bytes.

    The retirement relaxation even gives original left layouts for free and
    permits up to the entire final left budget in additional identities.
    Thus it also bounds strategies which can close unused baseline layouts.
    """
    base = statistics(layouts, records, [0] * len(records))
    original = {r['choices'][0]['left'] for r in records}
    free = [min(c['tiles'] for c in r['choices'] if c['left'] in original) for r in records]
    free_gain = sum(r['choices'][0]['tiles'] - f for r, f in zip(records, free))
    gains = {}
    for r, initial in zip(records, free):
        local = {}
        for c in r['choices']:
            if c['left'] not in original:
                local[c['left']] = max(local.get(c['left'], 0), initial - c['tiles'])
        for key, gain in local.items():
            gains[key] = gains.get(key, 0) + gain
    keys = int(base['left_keys'] * growth) - (0 if allow_retirement else base['left_keys'])
    byte_budget = int(base['left_bytes'] * growth) - (0 if allow_retirement else base['left_bytes'])
    count_bound = sum(sorted(gains.values(), reverse=True)[:keys])
    byte_bound = Fraction(0)
    for key in sorted(gains, key=lambda k: Fraction(gains[k], layouts[k]['bytes']), reverse=True):
        used = min(byte_budget, layouts[key]['bytes'])
        byte_bound += Fraction(used * gains[key], layouts[key]['bytes'])
        byte_budget -= used
        if not byte_budget:
            break
    unrestricted = sum(r['choices'][0]['tiles'] - min(c['tiles'] for c in r['choices']) for r in records)
    # Ceiling preserves an upper bound when the fractional knapsack is nonintegral.
    fractional_ceiling = (byte_bound.numerator + byte_bound.denominator - 1) // byte_bound.denominator
    return min(unrestricted, free_gain + min(count_bound, fractional_ceiling))


def solve(path, output, memory_only=False):
    layouts, records = load(path)
    baseline = statistics(layouts, records, [0] * len(records))
    oracle = [min(range(len(r['choices'])), key=lambda j: r['choices'][j]['tiles']) for r in records]
    result = dict(records=len(records), input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  baseline=baseline, unconstrained=statistics(layouts, records, oracle), budgets=[])
    for growth in map(Fraction, ['1', '1.25', '1.5', '2']):
        candidate = optimize(layouts, records, growth, Fraction(1000000) if memory_only else None)
        if not memory_only:
            candidate['saved_tiles_upper_bound_reserved'] = savings_upper_bound(layouts, records, growth)
            candidate['saved_tiles_upper_bound_retirement'] = savings_upper_bound(layouts, records, growth, True)
        result['budgets'].append(candidate)
        print(json.dumps({k: v for k, v in candidate.items() if k != 'choices'}), flush=True)
    with output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    s = sub.add_parser('sample')
    s.add_argument('output', type=Path)
    s.add_argument('paths', nargs='+', type=Path)
    s.add_argument('--seed', type=int, default=505)
    s.add_argument('--per-band', type=int, choices=range(1, 5), default=1)
    s.add_argument('--band', type=int, nargs=2, action='append', metavar=('LOW', 'HIGH'),
                   help='override default bands; every selected group stays complete')
    s = sub.add_parser('solve')
    s.add_argument('input', type=Path)
    s.add_argument('output', type=Path)
    s.add_argument('--memory-only', action='store_true', help='diagnostic: relax identity caps, retain byte caps')
    a = p.parse_args()
    if a.command == 'sample':
        if len(a.paths) > 4 or len(set(p.resolve() for p in a.paths)) != len(a.paths):
            p.error('one to four distinct input shards required')
        sample(a.paths, a.output, a.seed, a.per_band, a.band)
    else:
        solve(a.input, a.output, a.memory_only)


if __name__ == '__main__':
    main()
