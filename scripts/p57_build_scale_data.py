r"""Build the 50 and 80-module benchmarks, so the solver can be given a
literature-standard budget on a shop that is not small.

WHY. Every CP-SAT budget the paper reports is measured at 10, 20 or 30 modules,
and the longest is \CpsatCap s. Comparable studies routinely give an exact
solver half an hour per instance, so the open question is what happens
at that budget on a shop where the model is large. The two existing answers do
not cover it: at 10 modules the solver is stronger past 30 s (the paper says so),
and at 20 and 30 it overtakes on no regime, but only within 60 s.

These two sizes extend the ladder to 5x and 8x the training size. Nothing else
about them is new: same generator, same factory, same mix, same descriptor
streams, so the instances are drawn exactly as the published ones were and the
only variable is how many modules are scheduled together.

SEEDS. Each published set owns a disjoint block: 10x25 from 50000 (100
instances), 20x25 from 80000 (50), 30x25 from 90000 (20). These take 100000 and
110000, so no instance here can coincide with one the policy or a specialist
was trained, validated or tested on, and the blocks stay disjoint even if a set
is later extended by an order of magnitude.

DETERMINISM. ppvc_instance_generator is a pure function of (n_modules, mix,
factory, seed), so rerunning this script rewrites byte-identical instances. It
refuses rather than overwrites a directory whose recorded seed0, count or
module count differs from what is asked for, because silently regenerating a
benchmark under an existing name is how an evaluation stops meaning what its
name says.

Usage: python scripts/p57_build_scale_data.py [--sizes 50,80] [--n 20]
                                              [--factory default|tight]
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_ARGV = sys.argv[1:]
sys.argv = sys.argv[:1]


def _flag(name, default):
    return _ARGV[_ARGV.index(name) + 1] if name in _ARGV else default


from ppvc_instance_generator import (DEFAULT_FACTORY, TIGHT_FACTORY,
                                     ppvc_instance_generator, save_instance)

SIZES = [int(x) for x in _flag('--sizes', '50,80').split(',')]
N_INST = int(_flag('--n', '20'))
MIX = 'mixed'
# Which factory the modules compete for. 'default' is the published 25-station
# layout; 'tight' is the generator's own capacity-halved variant, 15 stations,
# which keeps two stations on six of the nine types and so keeps the routing
# choice the policy is trained to make. The 9-station SMALL_FACTORY is NOT
# offered: with one station per type there is no machine to choose and the
# problem stops being a FLEXIBLE job shop, which is a different problem class
# rather than a harder instance of this one.
FACTORY = {'default': DEFAULT_FACTORY, 'tight': TIGHT_FACTORY}[
    _flag('--factory', 'default')]
NMCH = sum(FACTORY.values())
# One disjoint seed block per (size, factory), continuing the published
# protocol. Contention cells get their own blocks so a tight-factory instance
# can never share a stream with the same-size default-factory one.
SEED0 = {(50, 25): 100000, (80, 25): 110000,
         (50, 15): 120000, (80, 15): 130000}
PURPOSE = ('{n}-module scale-up test set (eval-only): the solver at a '
           'literature-standard budget on a large shop')


def build(n_modules, n_inst):
    seed0 = SEED0.get((n_modules, NMCH))
    if seed0 is None:
        sys.exit(f'REFUSING: no seed block reserved for {n_modules} modules on '
                 f'{NMCH} stations. Add one to SEED0 that cannot overlap an '
                 f'existing set, rather than reusing another cell\'s stream.')
    out = f'data/PPVC/{n_modules}x{NMCH}+ppvc-mixed'
    meta_path = os.path.join(out, 'dataset_meta.json')
    if os.path.exists(meta_path):
        old = json.load(open(meta_path))
        for key, want in (('seed0', seed0), ('n_instances', n_inst),
                          ('n_modules', n_modules), ('ppvc_mix', MIX)):
            if old.get(key) != want:
                sys.exit(f'REFUSING: {out} already exists with {key}='
                         f'{old.get(key)!r}, this run wants {want!r}. Rebuilding '
                         f'it would change what every result file under that '
                         f'name refers to. Move the old set aside first.')
        print(f'{out}: already built and identical ({n_inst} instances) -- '
              f'nothing to do')
        return out
    os.makedirs(out, exist_ok=True)
    n_ops = None
    for i in range(n_inst):
        jl, pt, meta = ppvc_instance_generator(
            n_modules=n_modules, class_mix=MIX,
            station_counts=FACTORY, seed=seed0 + i)
        save_instance(os.path.join(out, f'instance_{i:03d}'), jl, pt, meta)
        n_ops = pt.shape[0]
    with open(meta_path, 'w') as f:
        json.dump({'seed0': seed0, 'n_instances': n_inst,
                   'n_modules': n_modules, 'ppvc_mix': MIX,
                   'ppvc_factory': _flag('--factory', 'default'),
                   'n_stations': NMCH,
                   'purpose': PURPOSE.format(n=n_modules)}, f, indent=1)
    print(f'{out}: {n_inst} instances, {n_ops} operations each, '
          f'seeds {seed0}..{seed0 + n_inst - 1}')
    return out


for n in SIZES:
    base = build(n, N_INST)
    print(f'  next: python scripts/p1_build_family_data.py --base {base} '
          f'--out data/FAMILY/{n}x{NMCH}')
