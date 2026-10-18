r"""The conditional flip rate of the boundary improvements, measured.

Ten of the 50x25 improvements are exactly one hour, the smallest nonzero value
an integer makespan admits, so each sits one unit from "unchanged". The
disclosure sentence says those are sensitive to the solver's run-to-run
randomness; this measures HOW sensitive, by re-solving exactly those instances
under the identical protocol and counting how many flip to unchanged (or move).

REPORTING RULE (agreed 2026-08-20, before the data existed): what this yields is
the flip rate CONDITIONAL on being at minimum resolution. It may be quoted only
about that set ("of the k improvements at one hour, j flipped on a second run"),
never as a fraction of all improvements: the unconditional denominator would mix
in the robust solves and understate the fragility this measures.

The first control arm (rep2, regime N) taught the sampling lesson the hard way:
N's solves never improved, an unimproved solve returns its deterministic warm
start, and five reruns came back identical, measuring a quantity with no
randomness in it. The boundary solves are the opposite stratum: search succeeded
but barely, which is where racing workers can land either side.

Usage:
  python scripts/p61_boundary_flip.py           # solve missing, then report
  python scripts/p61_boundary_flip.py --report  # report only, never solve
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ARGV = sys.argv[1:]
sys.argv = sys.argv[:1]

SRC = 'or_solution/FAMILY/50x25_fairwarm1800s'
OUT = 'or_solution/FAMILY/50x25_fairwarm1800s_repB'
BUDGET, WORKERS = 1800, 8

# ---- the boundary set: improvements of exactly one hour -------------------
targets = []
for f in sorted(glob.glob(f'{SRC}/*/*.json')):
    d = json.load(open(f))
    o, w = d['objective'], d['warm_makespan']
    if o is not None and 0 < w - o <= 1.0 + 1e-9:
        targets.append((f.split('/')[-2], d['instance'], w, o))
print(f'boundary set: {len(targets)} solves at the one-hour minimum')
for r, i, w, o in targets:
    print(f'  {r:4s} {i}  {w:.0f} -> {o:.0f}')

def _solve(task):
    """One 1800 s re-solve on cores 0-15, annotated with the timing evidence
    (cpu_share, affinity, load average) that the validity rule reads."""
    stem, desc, out_path = task
    os.sched_setaffinity(0, list(range(16)))
    os.environ['OMP_NUM_THREADS'] = '1'
    import scripts.p12_cpsat_fairwarm as fw
    import scripts.p44_cpsat_scale as p44
    fw.WARM_CACHE = p44._paths('50x25')[2]
    fw.solve_one((stem, desc, out_path, BUDGET, WORKERS, True))
    rec = json.load(open(out_path))
    rec['cpu_share'] = round(rec['cpu_per_wall'] / WORKERS, 3)
    rec['affinity'] = sorted(os.sched_getaffinity(0))
    rec['loadavg'] = [round(x, 2) for x in os.getloadavg()]
    rec['hint'] = 'complete'
    with open(out_path + '.tmp', 'w') as f:
        json.dump(rec, f)
    os.replace(out_path + '.tmp', out_path)
    return out_path, rec['objective'], rec['cpu_share']


if '--report' not in _ARGV and __name__ == '__main__':
    import multiprocessing as mp
    import scripts.p44_cpsat_scale as p44
    data, base, cache = p44._paths('50x25')
    todo = []
    for r, inst, _, _ in targets:
        out_dir = f'{OUT}/{r}'
        os.makedirs(out_dir, exist_ok=True)
        out_path = f'{out_dir}/{inst}.json'
        if os.path.exists(out_path):
            print(f'skip {r}/{inst} (done)')
            continue
        todo.append((os.path.join(base, inst), f'{data}/desc/{r}/{inst}.desc.json', out_path))
    # two solves at a time on cores 0-15, as in the 1800 s campaign itself
    with mp.Pool(2, maxtasksperchild=1) as pool:
        for out_path, obj, share in pool.imap_unordered(_solve, todo):
            print(f'solved {out_path}: {obj} (cpu_share {share})', flush=True)

# ---- the conditional flip count -------------------------------------------
flips, moved, same, missing = 0, 0, 0, 0
for r, inst, w, o in targets:
    p = f'{OUT}/{r}/{inst}.json'
    if not os.path.exists(p):
        missing += 1
        continue
    d = json.load(open(p))
    o2 = d['objective'] if d['objective'] is not None else d['warm_makespan']
    o2 = min(o2, d['warm_makespan'])
    if o2 >= w - 1e-9:
        flips += 1          # improvement vanished: search returned the hint
    elif abs(o2 - o) > 1e-9:
        moved += 1          # still improved, by a different amount
    else:
        same += 1
print(f'\nCONDITIONAL on the {len(targets)}-solve boundary set: '
      f'{flips} flipped to unchanged, {moved} moved, {same} identical'
      + (f', {missing} not yet re-solved' if missing else ''))

# ---- the macros the supplement's disclosure sentence prints ----------------
# Emitted only from a complete replication, and only while every moved solve
# moved DOWN (improved further): the prose says "improved by a further j to
# k h", and a moved-up solve would make that sentence false, so the guard
# below forces the prose to be revisited rather than silently mislabeled.
if not missing:
    deltas = []
    for r, inst, w, o in targets:
        d = json.load(open(f'{OUT}/{r}/{inst}.json'))
        o2 = d['objective'] if d['objective'] is not None else d['warm_makespan']
        o2 = min(o2, d['warm_makespan'])
        if abs(o2 - o) > 1e-9 and o2 < w - 1e-9:
            deltas.append(o - o2)
    assert all(x > 0 for x in deltas), (
        'a moved solve moved UP; rewrite the disclosure sentence before emitting')
    with open('paper/macros_flip.tex', 'w') as fh:
        fh.write(
            '% GENERATED by scripts/p61_boundary_flip.py -- do not edit.\n'
            '% Re-solve of the one-hour boundary improvements at 50x25 under\n'
            '% the identical 1800 s protocol (or_solution/.../50x25_fairwarm1800s_repB).\n'
            f'\\newcommand{{\\FlipN}}{{{len(targets)}}}\n'
            f'\\newcommand{{\\FlipFlipped}}{{{flips}}}\n'
            f'\\newcommand{{\\FlipSame}}{{{same}}}\n'
            f'\\newcommand{{\\FlipMoved}}{{{moved}}}\n'
            f'\\newcommand{{\\FlipMovedLoH}}{{{min(deltas):.0f}}}\n'
            f'\\newcommand{{\\FlipMovedHiH}}{{{max(deltas):.0f}}}\n')
    print('wrote paper/macros_flip.tex')
