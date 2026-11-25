r"""Build the 50 and 80-module benchmarks, so the solver can be given a
literature-standard budget on a shop that is not small, and the 27-module
per-floor batch.

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

PER-FLOOR BATCH. One floor of the anchor project holds about 27 modules
(\ModulesPerFloor in manuscript/macros.tex). The set 27x25 (50 instances on the
default factory) takes the block from 140000. The blocks of instance seeds in
use when it was reserved (instance i of a set uses seed0 + i):
  data/PPVC 5x9 70000-70049; 10x15 60000-60099; 10x25 50000-50099;
  10x25 rc_project 90000-90049; 10x25 steel_project 95000-95049;
  20x25 80000-80049; 30x25 90000-90019; 40x25 86000-86019 (no meta file;
  identified by regenerating instance_000 and instance_001);
  50x25 100000-100029 (n30 set included); 80x25 110000-110029;
  50x15 120000-120019; 80x15 130000-130019;
  training (train_family.py) the validation instances 100-119 and their
  descriptor seeds 100000-100019, and per training instance a seed drawn from
  [0, 2^31 - 1) by np.random under seed_train (a stream, not a block).
Descriptor seeds (20260706 + index, p1 and p29) and the perturbation streams
of p87, p101 to p103 and the public sets are separate streams.

DETERMINISM. ppvc_instance_generator is a pure function of (n_modules, mix,
factory, seed). An existing directory is never rewritten: when its recorded
seed0, count, module count and mix are the ones asked for, every instance is
regenerated in memory and compared with the files on disk, and the run reports
"already built and identical" or refuses. It refuses rather than overwrites a
directory whose recorded seed0, count or module count differs from what is
asked for, because silently regenerating a benchmark under an existing name is
how an evaluation stops meaning what its name says.

Usage: python scripts/p57_build_scale_data.py [--sizes 50,80] [--n 20]
                                              [--factory default|tight]
       python scripts/p57_build_scale_data.py --sizes 27 --n 50
--n defaults to 50 for 27 modules and to 20 otherwise.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_ARGV = sys.argv[1:]
sys.argv = sys.argv[:1]


def _flag(name, default):
    return _ARGV[_ARGV.index(name) + 1] if name in _ARGV else default


import numpy as np  # noqa: E402

from ppvc_instance_generator import (DEFAULT_FACTORY, TIGHT_FACTORY,  # noqa: E402
                                     load_instance, ppvc_instance_generator,
                                     save_instance)

SIZES = [int(x) for x in _flag('--sizes', '50,80').split(',')]
# instances per set: --n when given, otherwise N_DEFAULT of the size, otherwise 20
N_DEFAULT = {27: 50}
N_FLAG = _flag('--n', None)
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
         (50, 15): 120000, (80, 15): 130000,
         # the per-floor batch on the default factory (module docstring)
         (27, 25): 140000}
PURPOSE = ('{n}-module scale-up test set (eval-only): the solver at a '
           'literature-standard budget on a large shop')
# a set whose purpose is not the scale-up of the solver comparison
PURPOSE_OF = {(27, 25): ('27-module per-floor batch test set (eval-only): the policy '
                         'against the specialists and the dispatching rules at the '
                         'load of one floor of the anchor project')}


def n_instances(n_modules):
    return int(N_FLAG) if N_FLAG is not None else N_DEFAULT.get(n_modules, 20)


def same_instance(a, b):
    """Two (job_length, op_pt, meta) triples hold the same instance."""
    (ja, pa, ma), (jb, pb, mb) = a, b
    if not (np.array_equal(ja, jb) and np.array_equal(pa, pb)) or set(ma) != set(mb):
        return False
    for k in ma:
        x, y = ma[k], mb[k]
        if isinstance(x, dict) or isinstance(y, dict):
            if dict(x) != dict(y):
                return False
        elif isinstance(x, list) or isinstance(y, list):
            if list(x) != list(y):
                return False
        elif not np.array_equal(np.asarray(x), np.asarray(y)):
            return False
    return True


def verify(out, seed0, n_modules, n_inst):
    """Regenerate every instance of an existing set in memory and compare it with the
    files on disk; refuses on the first difference. Nothing is written."""
    for i in range(n_inst):
        stem = os.path.join(out, f'instance_{i:03d}')
        if not os.path.exists(stem + '.fjs') or not os.path.exists(stem + '.meta.json'):
            sys.exit(f'REFUSING: {out} records {n_inst} instances but {stem} is missing')
        jl, pt, meta = ppvc_instance_generator(
            n_modules=n_modules, class_mix=MIX, station_counts=FACTORY, seed=seed0 + i)
        on_disk = load_instance(stem)
        # meta as save_instance stores it (arrays as lists) and load_instance reads it back
        gen = {k: (np.array(v.tolist()) if isinstance(v, np.ndarray) else v)
               for k, v in meta.items()}
        if not same_instance((jl, pt, gen), on_disk):
            sys.exit(f'REFUSING: {stem} differs from the instance the generator draws '
                     f'with seed {seed0 + i}. The set on disk is not this set; nothing '
                     f'was written.')
    extra = [f for f in os.listdir(out) if f.startswith('instance_')
             and int(f[len('instance_'):].split('.')[0]) >= n_inst]
    if extra:
        sys.exit(f'REFUSING: {out} holds instance files beyond the {n_inst} it records '
                 f'(first {sorted(extra)[0]})')


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
        verify(out, seed0, n_modules, n_inst)
        print(f'{out}: already built and identical ({n_inst} instances) -- '
              f'nothing to do')
        return out
    if os.path.isdir(out) and os.listdir(out):
        sys.exit(f'REFUSING: {out} holds files but no dataset_meta.json (an interrupted '
                 f'build?). Inspect and remove it by hand; nothing was written.')
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
                   'purpose': PURPOSE_OF.get((n_modules, NMCH),
                                             PURPOSE.format(n=n_modules))}, f, indent=1)
    print(f'{out}: {n_inst} instances, {n_ops} operations each, '
          f'seeds {seed0}..{seed0 + n_inst - 1}')
    return out


for n in SIZES:
    base = build(n, n_instances(n))
    print(f'  next: python scripts/p1_build_family_data.py --base {base} '
          f'--out data/FAMILY/{n}x{NMCH}')
