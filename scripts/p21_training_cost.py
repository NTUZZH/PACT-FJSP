"""Training cost: the number the paper's motivation rests on and never stated.

The whole premise is that retraining a policy every time the shop's timing rules
change is expensive. The manuscript costed the specialist CATALOGUE in megabytes,
which is not a cost anyone cares about (the supplementary table itself concedes
it is about a megabyte), and never once stated the wall-clock cost of training and
revalidating a specialist. Without that number the motivation is an assertion: if
a specialist trains in an afternoon on hardware the plant already owns, an
industrial reader shrugs and retrains.

Read from the training banners, which record their own wall-clock. Emits
macros_cost.tex.

THE ESTIMATOR IS THE MINIMUM OVER IDENTICAL RERUNS, and it has to be.

These runs were taken on a shared GPU that other users saturate (the queue guard
in run_family_queue.sh exists because of it), so a training banner is not a
measurement of compute -- it is compute PLUS however much of the card someone
else was holding. The signature is unmistakable once looked for: spec-S ran in
10.13 h and spec-S-s303 in 3.55 h, with byte-identical configurations, the same
1000 PPO updates, 1044 identical log lines, and the same converged quality
(best rel 0.7968 vs 0.7997). Identical work cannot honestly cost 2.9x more.

This mattered, because the previous estimator was the MEDIAN, and it was applied
per regime. Two regimes (S and LS) happened to land two of their three runs in a
contended window, so their medians were the contended values -- 10.13 h and
10.35 h, together 53% of a 39 h catalogue total. The joint's three runs were all
clean, so the denominator was not inflated at all. The published ratio was
therefore a contended numerator over a quiet denominator: 6.3x, where the
like-for-like number is 3.9x. The paper leans on that sentence for its entire
value proposition, and p19_integrity.py already refuses to let a CONTENDED
wall-clock become \\PolicySec. The same discipline belongs here.

Contention can only ever ADD wall-clock, never remove it, so across reruns of one
identical configuration the minimum is the least-contended observation and the
best available estimate of the true cost. Taking it on BOTH sides -- catalogue and
joint alike -- is what keeps the ratio honest; the old failure was not the median
as such, it was applying an estimator that absorbed contention to one side of a
ratio and not the other.

Usage: python scripts/p21_training_cost.py
"""
import glob
import os
import re
import statistics

LOGS = 'train_log'
SPEC_RE = re.compile(r'FAMILY_spec-([A-Z]+)(?:-s(\d+))?\.log$')
# Hyphen only, anchored: a log named with an underscore before its seed
# (FAMILY_joint-v1_s301.log) is not one of the three runs the paper reports, and
# a looser pattern would swallow it into the pool and bias the minimum.
# PROTAGONIST SWAP (2026-08-11): the deployed artifact is the blind joint, so
# its training cost is the honest denominator of the catalogue ratio.
JOINT_RE = re.compile(r'FAMILY_(jointblind-v1)(?:-s(\d+))?\.log$')
HOURS = re.compile(r'done in ([\d.]+) h')

# A rerun this much slower than its own fastest twin is contention, not compute.
CONTENTION_X = 1.5

# The catalogue a per-situation deployment must hold: one specialist per
# constraint-bearing regime. N is the base FJSP and needs no specialist.
CATALOGUE = ['L', 'S', 'W', 'LS', 'LW', 'SW', 'LSW']


def hours(path):
    m = HOURS.search(open(path, errors='replace').read())
    return float(m.group(1)) if m else None


spec, joint = {}, []
for p in glob.glob(os.path.join(LOGS, 'FAMILY_*.log')):
    h = hours(p)
    if h is None:
        continue
    b = os.path.basename(p)
    m = SPEC_RE.search(b)
    if m:
        spec.setdefault(m.group(1), []).append(h)
        continue
    if JOINT_RE.search(b):
        joint.append(h)

if not spec or not joint:
    raise SystemExit('no completed training banners found in train_log/ '
                     '(looking for "done in X h")')

# One specialist = one retraining event: what a plant pays when its rules change.
# Per regime, the cost is the cleanest run of that regime; the catalogue is the
# sum of those, and the joint is its own cleanest run. Same estimator both sides.
spec_clean = {r: min(v) for r, v in spec.items()}
spec_med = statistics.median(spec_clean.values())
cat = sum(spec_clean[r] for r in CATALOGUE if r in spec_clean)
joint_h = min(joint)
# The worst spread across reruns of ONE configuration. This is the evidence that
# the banners are contended, and the supplement quotes it, so it is a macro:
# a measurement typed by hand is a measurement nothing can recompute.
contend_x = max(max(v) / min(v) for v in spec.values() if len(v) > 1)

print(f'specialists measured : {sum(len(v) for v in spec.values())} runs over '
      f'{len(spec)} regimes')
print('  regime   runs (h)                        clean   discarded as contended')
dropped = 0
for r in sorted(spec, key=lambda r: (len(r), r)):
    runs = sorted(spec[r])
    c = spec_clean[r]
    bad = [h for h in runs if h > c * CONTENTION_X]
    dropped += len(bad)
    print(f'  {r:6}   {str([round(h, 2) for h in runs]):32}  {c:5.2f}   '
          f'{[round(h, 2) for h in bad] if bad else "-"}')
print(f'  -> {dropped} of {sum(len(v) for v in spec.values())} specialist runs '
      f'are >{CONTENTION_X}x their own fastest twin: contention, not compute')
print(f'  per specialist     : median of the clean runs {spec_med:.1f} h')
print(f'  {len(CATALOGUE)}-regime catalogue : {cat:.1f} h -> {cat:.0f}')
print(f'joint policy         : {joint_h:.2f} h '
      f'(cleanest of {len(joint)}: {sorted(round(h, 2) for h in joint)})')
print(f'  catalogue / joint  : {cat / joint_h:.1f}x')

if dropped:
    print(f'\nNOTE: the discarded runs are retained in train_log/ and are real '
          f'wall-clocks;\n      they are not the COST of the work, they are the '
          f'cost plus another user.')

out = [
    '% auto-generated by scripts/p21_training_cost.py -- do not hand-edit',
    '% Wall-clock on a SHARED GPU: the estimator is the minimum over identical',
    '% reruns, applied to catalogue and joint alike, because contention only ever',
    '% adds time. A per-regime median put two contended runs (S, LS) into the',
    '% numerator while the joint denominator stayed clean, and read 6.3x.',
    f'\\newcommand{{\\SpecHours}}{{{spec_med:.1f}}}'
    f'  % GPU-hours to train ONE specialist (median over the {len(spec_clean)}'
    f' per-regime clean runs)',
    f'\\newcommand{{\\JointHours}}{{{joint_h:.1f}}}'
    '  % GPU-hours to train the one joint policy',
    f'\\newcommand{{\\CatalogueHours}}{{{cat:.0f}}}'
    f'  % GPU-hours for the {len(CATALOGUE)}-specialist catalogue',
    f'\\newcommand{{\\CatalogueVsJointX}}{{{cat / joint_h:.1f}}}'
    '  % catalogue / joint, in training hours',
    f'\\newcommand{{\\TrainContendX}}{{{contend_x:.1f}}}'
    '  % worst wall-clock spread across reruns of ONE identical config',
]
with open('paper/macros_cost.tex', 'w') as f:
    f.write('\n'.join(out) + '\n')
print('\nwrote paper/macros_cost.tex')
