"""P29: does the policy survive constraint parameters it never trained on?

The benchmark is one factory drawn from one distribution, and the constraint
magnitudes never leave the range the policy trained on. The paper's industrial claims
are therefore only as wide as that distribution, unless the policy is taken outside it
and what happens is reported. The shifts worth running are the ones a real floor would
show: asymmetric setup matrices, unseen setup severity, and higher outage density.

This runs two of them. Nothing is retrained. The published policies are evaluated,
unchanged, on descriptors whose SETUP and WINDOW parameters are drawn from ranges
they never saw:

  ood-setup   cross-class changeovers drawn from U[100%, 200%] of the instance's
              mean processing time instead of U[50%, 100%], and the matrix is NO
              LONGER SYMMETRIC: S[a,b] and S[b,a] are drawn independently. Twice
              the severity, and a structure the generator never produced. Bites on
              S, LS, SW and LSW.

  ood-window  per-machine outage density drawn from U[20%, 35%] of the horizon
              instead of U[5%, 15%]. Between two and three times as much downtime
              as anything in training. Bites on W, LW, SW and LSW.

The base instances, the routes, the lags and the descriptor seeds are unchanged, so
each shifted instance is PAIRED with its in-distribution twin: the same shop, the
same order book, the same rng stream, and only the constraint distribution moved.

WHAT MAKES THIS A FAIR TEST. The shift is out of distribution for the joint policy
AND for the specialist AND for the dispatching rules, and all three are measured on
it. A dispatching rule has no distribution: it cannot be surprised, so SPT is the
control. The question the paper can then answer is not "does the policy get worse"
(everything gets worse when the shop gets harder) but "does the LEARNED policy lose
more than the rule it beats, and does it lose more than the specialist it claims
parity with". Those are the two claims a shift could actually break.

THE GUARD THAT MAKES ANY OF THIS TRUSTWORTHY. draw_setup_matrix and draw_windows
now take keyword arguments, and a reordered rng draw would silently rewrite every
descriptor the paper's results rest on. So before this script generates anything,
it REGENERATES THE PUBLISHED DESCRIPTORS with the default arguments and asserts they
come back byte-identical to the ones on disk. If they do not, it refuses to run.

Usage:
  python scripts/p29_ood.py --build          # descriptors only (with the guard)
  python scripts/p29_ood.py --build --eval   # and the evaluations
"""
import glob
import json
import os
import sys

_ARGV = sys.argv[1:]
sys.argv = sys.argv[:1]

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from constraint_family.descriptor import (REGIMES_ALL,  # noqa: E402
                                          make_descriptor)
from ppvc_instance_generator import load_instance  # noqa: E402

BASE = 'data/PPVC/10x25+ppvc-mixed'
REF = 'data/FAMILY/10x25'
DESC_SEED0 = 20260706

ARMS = {
    # Twice the changeover severity, and asymmetric, which the generator has
    # never produced. Only the four setup-bearing regimes change.
    '10x25_oodsetup': dict(cross=(1.00, 2.00), symmetric=False),
    # Two to three times the downtime. Only the four window-bearing regimes change.
    '10x25_oodwindow': dict(density_range=(0.20, 0.35)),
}


def guard():
    """Refuse to run if the DEFAULT arguments no longer reproduce the benchmark.

    The keyword arguments added to draw_setup_matrix change how many times the rng
    is drawn from when they are non-default. If they ever changed the default path,
    every descriptor in the paper would move and no test would notice. This is that
    test.
    """
    checked = mismatched = 0
    for regime in REGIMES_ALL:
        for p in sorted(glob.glob(f'{REF}/desc/{regime}/*.desc.json')):
            on_disk = json.load(open(p))
            jl, pt, meta = load_instance(on_disk['base_instance'])
            d = make_descriptor(regime, jl, pt, meta, pt.shape[1],
                                seed=on_disk['desc_seed']).to_json()
            for k in ('active', 'lag', 'op_class', 'setup', 'windows', 'horizon'):
                if json.dumps(d[k]) != json.dumps(on_disk[k]):
                    print(f'  MISMATCH {regime}/{os.path.basename(p)}: {k}')
                    mismatched += 1
                    break
            checked += 1
    print(f'guard: regenerated {checked} published descriptors with the default '
          f'arguments; {checked - mismatched} byte-identical')
    # A guard that verifies nothing reports the same "0 mismatched" as a guard that
    # verifies everything. If the glob misses (a moved directory, a renamed regime),
    # checked stays at zero, nothing is compared, and the one test standing between
    # a library edit and every number in the paper passes vacuously. So the count is
    # asserted, not printed and trusted.
    expect = len(REGIMES_ALL) * 100
    if checked != expect:
        raise SystemExit(
            f'REFUSING TO RUN: the guard compared {checked} descriptors, not '
            f'{expect}. It cannot certify that the published benchmark is unmoved '
            f'by looking at {checked} of it. Check that {REF}/desc/<regime>/ is '
            f'populated before trusting anything downstream.')
    if mismatched:
        raise SystemExit('REFUSING TO RUN: the default draw no longer reproduces '
                         'the published benchmark, so the library change moved the '
                         'paper out from under itself.')


def build(out_dir, shift):
    stems = sorted(glob.glob(os.path.join(BASE, 'instance_*.fjs')))
    stems = [s[:-len('.fjs')] for s in stems]
    assert stems, f'no instances under {BASE}'
    os.makedirs(out_dir, exist_ok=True)
    for idx, stem in enumerate(stems):
        jl, pt, meta = load_instance(stem)
        M = pt.shape[1]
        name = os.path.basename(stem)
        for regime in REGIMES_ALL:
            desc = make_descriptor(regime, jl, pt, meta, M,
                                   seed=DESC_SEED0 + idx, shift=shift)
            d = desc.to_json()
            d['base_instance'] = os.path.relpath(stem)
            d['regime'] = regime
            d['desc_seed'] = DESC_SEED0 + idx
            d['ood_shift'] = {k: list(v) if isinstance(v, tuple) else v
                              for k, v in shift.items()}
            rdir = os.path.join(out_dir, 'desc', regime)
            os.makedirs(rdir, exist_ok=True)
            json.dump(d, open(os.path.join(rdir, f'{name}.desc.json'), 'w'))
    json.dump({'base_dir': BASE, 'n_instances': len(stems),
               'regimes': list(REGIMES_ALL), 'desc_seed0': DESC_SEED0,
               'ood_shift': {k: list(v) if isinstance(v, tuple) else v
                             for k, v in shift.items()},
               'note': 'OUT OF DISTRIBUTION. Same base instances, same descriptor '
                       'seeds, constraint parameters drawn from ranges no policy '
                       'in this paper trained on.'},
              open(os.path.join(out_dir, 'dataset_meta.json'), 'w'))
    print(f'built {len(stems)} x {len(REGIMES_ALL)} descriptors -> {out_dir}')


def report(out_dir, shift):
    """How far out of distribution did we actually go? State it in numbers."""
    ref_s, ood_s, ref_w, ood_w = [], [], [], []
    for regime in ('S', 'W'):
        for p in sorted(glob.glob(f'{REF}/desc/{regime}/*.desc.json'))[:100]:
            d = json.load(open(p))
            if regime == 'S':
                ref_s += [v for row in d['setup'] for v in row]
            else:
                ref_w.append(sum(e - s for m in d['windows'] for s, e in m)
                             / (25 * d['horizon']))
        for p in sorted(glob.glob(f'{out_dir}/desc/{regime}/*.desc.json'))[:100]:
            d = json.load(open(p))
            if regime == 'S':
                ood_s += [v for row in d['setup'] for v in row]
            else:
                ood_w.append(sum(e - s for m in d['windows'] for s, e in m)
                             / (25 * d['horizon']))
    if 'cross' in shift:
        print(f'  setup entries : published mean {np.mean(ref_s):.2f} h -> '
              f'shifted {np.mean(ood_s):.2f} h '
              f'({np.mean(ood_s) / np.mean(ref_s):.2f}x)')
        A = np.array(json.load(open(f'{out_dir}/desc/S/instance_000.desc.json'
                                    ))['setup'])[:-1]
        print(f'  symmetric?    : published yes, shifted '
              f'{"yes" if np.allclose(A, A.T) else "NO"}')
    if 'density_range' in shift:
        print(f'  outage density: published mean {np.mean(ref_w):.3f} -> '
              f'shifted {np.mean(ood_w):.3f} '
              f'({np.mean(ood_w) / np.mean(ref_w):.2f}x of the horizon)')


if __name__ == '__main__':
    guard()
    print()
    for out, shift in ARMS.items():
        build(f'data/FAMILY/{out}', shift)
        report(f'data/FAMILY/{out}', shift)
        print()
    if '--eval' not in _ARGV:
        print('descriptors only. Re-run with --eval, or evaluate directly:')
        for out in ARMS:
            print(f'  python eval_family.py --family_model 10x25+family+joint-v1 '
                  f'--regimes N L S W LS LW SW LSW --methods greedy SPT '
                  f'--data data/FAMILY/{out}')
