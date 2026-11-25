r"""P129: single-shell project mixes at the production batches, evaluation only.

Subcommands build, plan, evaluate, infer and check. CPU only, exact mask.

LAUNCH NOTE (the four questions, answered before the run)

1. Purpose. The training instances mix reinforced-concrete and steel modules, and so do the
   test sets behind the production claim of the paper (3 to 4% ahead of the best dispatching
   rule at 50 and 80 modules). A real project uses one shell system. At the training size the
   sensitivity study (scripts/p87_sensitivity.py, levels mix_rc and mix_steel) found the
   policy +0.69% [+0.09, +1.28] against the specialists with reinforced-concrete modules
   alone and not distinguishable from the best of four rules on six combinations, and ahead
   of the rules on all eight with steel modules alone. This study measures the same
   comparisons at 50 and 80 modules on single-shell sets, beside the mixed sets and beside
   the training size, by one procedure. The numbers land in the sensitivity table of the
   supplement and in the discussion of the production claim.
2. Expected result. The steel sets are expected to behave as at 10 modules (ahead of the
   rules); for the reinforced-concrete sets the question is whether the lead over the rules
   that the mixed sets show at 50 and 80 modules survives. Either outcome is reported; a lead
   that vanishes bounds the production claim to mixed projects.
3. Contamination. Every new file goes under --data_root (the new base sets and their
   descriptors, default test_results/FAMILY/mix_p129/data) and --root (the results, default
   test_results/FAMILY/mix_p129), a folder no existing script reads
   (tests/test_p129_project_mix.py checks every pattern over test_results/FAMILY). The stored
   sets are read only; build refuses to replace a file whose content differs. Rollouts are
   deterministic, so results are not timing measurements and may share cores; the recorded
   times are planning figures.
4. Data accuracy. build refuses unless (i) the generator path regenerates the stored
   single-shell sets data/PPVC/10x25+ppvc-rc_project and -steel_project from their stored
   settings file by file, (ii) the descriptor function reproduces the stored descriptors of
   data/FAMILY/50x25n30 and 80x25n30 (first two instances, eight combinations) byte for byte,
   (iii) the new seed blocks collide with no block in use and (iv) every new descriptor passes
   checks C1, C2, C3 and C5. infer refuses unless every 10-module unit that p87 stored equals
   this script's unit on every instance, and unless the rows of the default level and of the
   10-module levels equal the stored records (below).

LEVELS AND SIZES (10x25, 50x25n30, 80x25n30: 10, 50 and 80 modules on the default factory of
25 stations; combinations N, L, S, W, LS, LW, SW, LSW)
  default    the mixed test sets of the paper, READ and not evaluated: 10x25 (100 instances)
             from arm_registry.res_dir('10x25'); 50x25n30 and 80x25n30 (30 each) from
             arm_registry.res_dir(<set>) and the specialists from <set>_xmask_spec
             (scripts/run_production_specialists_eval.sh, read by
             scripts/p128_production_specialists.py spec_reader); seven rules from the same
             folders.
  mix_rc     modules of the two reinforced-concrete route classes alone (preset rc_project):
             at 10 modules the level mix_rc of p87 (data/PPVC/10x25+ppvc-rc_project, 50
             instances, p87's descriptors under its root); at 50 and 80 modules new sets of
             30 instances built here.
  mix_steel  the same with the steel route classes (preset steel_project).

NEW SETS (build)
  base instance i  ppvc_instance_generator(n_modules=n, class_mix=<preset>,
                   station_counts=DEFAULT_FACTORY, seed=seed0 + i), save_instance, the calls
                   that regenerate the stored single-shell sets (control (i)); the script that
                   wrote those sets is not in the repository, and their dataset_meta.json
                   records seed0, the count, the module count and the preset only.
  seed0            SEED0 below: 150000 (RC, 50 modules), 155000 (steel, 50), 160000 (RC, 80),
                   165000 (steel, 80); blocks of 30 that collide with no seed block of a folder
                   under data/PPVC (every dataset_meta.json) nor with KNOWN_BLOCKS.
  descriptor i     scripts/p86_extend_scale_sets.py descriptor_text (make_descriptor with the
                   default arguments and seed 20260706 + i, the stored form), the function that
                   wrote data/FAMILY/50x25n30 and 80x25n30 (control (ii)).
  folders          <data_root>/PPVC/<n>x25+ppvc-<preset>/ and
                   <data_root>/FAMILY/<n>x25_<level>/desc/<R>/ with manifest.json.

ARMS (exact-mask names of scripts/arm_registry.py)
  pact    released policy xmask-v1, training seeds --seeds (default 301-305)
  spec    specialist of the combination, xmask-solo-<R>, the same seeds
  rules   FIFO, MOR, SPT, MWKR (scripts/p101_mask_controls.py rule_rollout) and ECT, SSPT, LLF
          (scripts/p102_scale_sensitivity.py aware_rollout), each with np.random.seed(50)
          after the data are set
  Units run through scripts/p103_five_classes.py rollout (policy runs loaded as eval_family.py
  loads them, parameter count asserted) and every schedule passes p103's schedule_check (check
  C6 and the makespan against the last completion) at four classes; a violation stops the run.

OUTCOMES (infer)
  Per size and level: the policy against the specialist per combination and pooled over the
  eight, verdicts at the margins 0.5% and 1%; the policy against the best of the four
  classical rules and against the best of seven, combined verdict with Holm over the eight
  combinations of a level and size; pooled rows; verdict counts. d = 100 (a - b) / b, seed s
  against seed s; scripts/p103_five_classes.py policy_block, that is
  scripts/p102_scale_sensitivity.py component (stats_family.hier_boot, B = 20000, a fresh
  default_rng(11) per quantity; with one training run stats_family.instance_boot, fresh
  default_rng(7)) and superiority (p89_public_family.superiority_verdict).
  Controls of infer (each refuses the run):
    p87       every 10-module unit p87 stored (policy, specialist and the four classical rules
              of the seeds, levels mix_rc and mix_steel, eight combinations) equals this
              script's unit on every instance;
    default   at 50 and 80 modules the rows equal the record of
              scripts/p128_production_specialists.py (the same estimators and the same Holm
              family of eight): policy against specialist per combination and pooled (mean,
              interval ends, t interval ends, both verdicts), policy against the best
              classical rule per combination (mean, interval ends, the one-sided p-values with
              and without Holm, verdict) and pooled (mean, interval ends, t interval ends); at
              10 modules the pooled specialist row and the rule rows equal the record of
              scripts/p88_family_inference.py (its per-combination specialist rows draw from
              one stream in p27's order and are not compared);
    sens      at 10 modules the rows of mix_rc and mix_steel equal the sensitivity record of
              p87 (results/sensitivity_inference.json), keys as for default.
  Outputs: results/project_mix.json and .md, manuscript/macros_projectmix.tex (prefix Pmx), in
  place for a full run with every arm, otherwise under --out. The records of p88 and p128 are
  read under PACT_OUT (the chain's out/exact).

USAGE (repository root; CPU only)
  python scripts/p129_project_mix.py build
  python scripts/p129_project_mix.py plan     --parts K
  python scripts/p129_project_mix.py evaluate --part i --parts K        (each part on its core)
  PACT_OUT=out/exact python scripts/p129_project_mix.py infer
  python scripts/p129_project_mix.py check                              (read only)
  Selection: --levels mix_rc,mix_steel --sizes 10x25,50x25n30,80x25n30 --combos ... --arms
  pact,spec,rules --rules FIFO,... --seeds 301-305; smoke: --root <folder> --n 2, infer
  --partial --out <folder>. evaluate waits while the hold file (--hold, default
  train_log/EVAL_HOLD) exists, before each unit.
"""
import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(ROOT_DIR, 'scripts')
sys.path.insert(0, ROOT_DIR)
if SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)

import arm_registry as REG  # noqa: E402  (removes --pact-* options from sys.argv)

_ARGV = sys.argv[1:]
# params parses argv at import time and EnvState reads the device then: CPU only
sys.argv = sys.argv[:1] + ['--device', 'cpu']

import numpy as np  # noqa: E402

import p101_mask_controls as P101  # noqa: E402
import p102_scale_sensitivity as P102  # noqa: E402
import p103_five_classes as P103  # noqa: E402
import p87_sensitivity as P87  # noqa: E402

# ---------------------------------------------------------------------------
# design constants
# ---------------------------------------------------------------------------

LEVELS = ('default', 'mix_rc', 'mix_steel')
MIX_LEVELS = ('mix_rc', 'mix_steel')
MIX = {'mix_rc': 'rc_project', 'mix_steel': 'steel_project'}
SHELL = {'mix_rc': 'reinforced-concrete', 'mix_steel': 'steel'}
SIZES = ('10x25', '50x25n30', '80x25n30')
NEW_SIZES = ('50x25n30', '80x25n30')
MODULES = {'10x25': 10, '50x25n30': 50, '80x25n30': 80}
N_NEW = 30
N_STORED_MIX = 50
SEED0 = {('mix_rc', '50x25n30'): 150000, ('mix_steel', '50x25n30'): 155000,
         ('mix_rc', '80x25n30'): 160000, ('mix_steel', '80x25n30'): 165000}
DESC_SEED0 = 20260706
# seed blocks in use that no dataset_meta.json under data/PPVC records
# (scripts/p57_build_scale_data.py, SEEDS)
KNOWN_BLOCKS = {'data/PPVC/40x25+ppvc-mixed (no dataset_meta.json)': (86000, 20),
                'validation instances of train_family.py (seed_train_vali_datagen)': (100, 20)}
R8 = P101.R8
RULES4 = P101.RULES
AWARE = P102.AWARE
RULES7 = RULES4 + AWARE
POLICY_ARMS = ('pact', 'spec')
ARM_CHOICES = POLICY_ARMS + ('rules',)
SEEDS = (301, 302, 303, 304, 305)
DEFAULT_ROOT = 'test_results/FAMILY/mix_p129'
DEFAULT_DATA = DEFAULT_ROOT + '/data'
OUT_FILES = {'json': 'results/project_mix.json', 'md': 'results/project_mix.md',
             'macros': 'manuscript/macros_projectmix.tex'}
GENERATOR = 'scripts/p129_project_mix.py infer'
PREFIX = 'Pmx'
B = 20000
MARGINS = P102.MARGINS
MW = P102.MW
COMPARISONS = P103.COMPARISONS
LEVEL_WORDS = {'default': 'Mixed', 'mix_rc': 'Concrete', 'mix_steel': 'Steel'}
SIZE_WORDS = {'10x25': 'Ten', '50x25n30': 'Fifty', '80x25n30': 'Eighty'}
HOLD = 'train_log/EVAL_HOLD'
# planning cost, CPU seconds per instance, each including the check C6. Policy at 50 and 80
# modules: about 4.5 and 10 minutes per combination of 30 instances (the exact campaign's
# rollouts; the stored units of p103 took 8.9 and 19.3 s per instance on average). Rules: the
# largest per-instance means of the stored units of p103 (1.69 and 3.56 s). 10 modules: the
# stored mix units of p87 (policy 1.2 s on average, 1.5 s at most; rules 0.27 and 0.40 s).
COST_S = {'10x25': {'rule': 0.4, 'policy': 1.3},
          '50x25n30': {'rule': 1.7, 'policy': 9.0},
          '80x25n30': {'rule': 3.6, 'policy': 20.0}}
SPEC_KEYS = ('mean', 'lo', 'hi', 't_lo', 't_hi', 'v0.5', 'v1.0')
RULE_KEYS = ('mean', 'lo', 'hi', 'p_t', 'p_boot', 'p_t_greater', 'p_boot_greater', 'p_t_holm',
             'p_boot_holm', 'p_t_greater_holm', 'p_boot_greater_holm', 'verdict')
POOL_KEYS = ('mean', 'lo', 'hi', 't_lo', 't_hi')
REPRO_TOL = 1e-9

Refused = P101.Refused
refuse = P101.refuse
write_json_atomic = P101.write_json_atomic
jsonable = P101.jsonable
f2, u1 = P101.f2, P101.u1


def inst(i):
    return f'instance_{i:03d}'


def rel(path):
    """A path relative to the repository root when it lies inside, else absolute."""
    a = os.path.abspath(os.path.join(ROOT_DIR, path))
    r = os.path.relpath(a, ROOT_DIR)
    return a if r.startswith(os.pardir) else r


# ---------------------------------------------------------------------------
# folders and cells
# ---------------------------------------------------------------------------

def base_dir(data_root, level, size):
    """Base instances of a new set."""
    return os.path.join(data_root, 'PPVC', f'{MODULES[size]}x25+ppvc-{MIX[level]}')


def desc_dir(data_root, level, size):
    """Descriptors (desc/<R>/) and manifest of a new set."""
    return os.path.join(data_root, 'FAMILY', f'{MODULES[size]}x25_{level}')


def manifest_path(data_root, level, size):
    return os.path.join(desc_dir(data_root, level, size), 'manifest.json')


def result_path(root, level, size, combo, label):
    return os.path.join(root, 'results', 'exact', level, size, combo, f'{label}.npz')


def is_new(level, size):
    return level in MIX_LEVELS and size in NEW_SIZES


def level_names(level, size, data_root=DEFAULT_DATA, n=None):
    """Instance names of a cell, in the order of every array of it."""
    if level == 'default':
        return P101.names_of(size, n)
    if size == '10x25':
        return P87.level_names(level, n)
    d = os.path.join(desc_dir(data_root, level, size), 'desc', 'N')
    names = sorted(os.path.basename(p)[:-len('.desc.json')]
                   for p in glob.glob(os.path.join(d, '*.desc.json')))
    if len(names) != N_NEW:
        refuse(f'{d} holds {len(names)} descriptors, the study needs {N_NEW} (run build first)')
    return names[:n] if n else names


def load_cell(level, size, combo, data_root=DEFAULT_DATA, n=None):
    """[(name, jl, pt, meta, desc)] of one cell, read only."""
    if level == 'default':
        return P101.load_cell(size, combo, n)
    if size == '10x25':
        return P87.load_cell(P87.DEFAULT_ROOT, level, combo, n)
    from constraint_family.descriptor import ConstraintDescriptor
    from ppvc_instance_generator import load_instance
    out = []
    for name in level_names(level, size, data_root, n):
        p = os.path.join(desc_dir(data_root, level, size), 'desc', combo, f'{name}.desc.json')
        with open(p) as f:
            rec = json.load(f)
        jl, pt, meta = load_instance(os.path.join(ROOT_DIR, rec['base_instance']))
        out.append((name, jl, pt, meta,
                    ConstraintDescriptor.from_json(rec, setup_convention='attached')))
    return out


# ---------------------------------------------------------------------------
# build: the controls of the generator path and the new sets
# ---------------------------------------------------------------------------

def _p86():
    """scripts/p86_extend_scale_sets.py, imported with sys.argv kept (it strips argv)."""
    saved = sys.argv[:]
    try:
        import p86_extend_scale_sets as P86
    finally:
        sys.argv = saved
    return P86


def seed_blocks():
    """{folder or use: (seed0, count)} of every seed block in use."""
    out = {}
    for m in sorted(glob.glob(os.path.join(ROOT_DIR, 'data', 'PPVC', '*', 'dataset_meta.json'))):
        with open(m) as f:
            d = json.load(f)
        out[os.path.relpath(os.path.dirname(m), ROOT_DIR)] = (int(d['seed0']),
                                                             int(d['n_instances']))
    out.update(KNOWN_BLOCKS)
    return out


def _overlap(a, b):
    return a[0] < b[0] + b[1] and b[0] < a[0] + a[1]


def check_seed_blocks():
    """The four new blocks against every block in use and against each other."""
    used = seed_blocks()
    new = {f'{lv} {sz}': (s0, N_NEW) for (lv, sz), s0 in SEED0.items()}
    bad = [f'{k} [{a[0]}, {a[0] + a[1]}) and {u} [{b[0]}, {b[0] + b[1]})'
           for k, a in new.items() for u, b in used.items() if _overlap(a, b)]
    keys = list(new)
    bad += [f'{x} and {y}' for i, x in enumerate(keys) for y in keys[i + 1:]
            if _overlap(new[x], new[y])]
    if bad:
        refuse('new seed blocks collide: ' + '; '.join(bad))
    return dict(used={k: list(v) for k, v in used.items()},
                new={k: list(v) for k, v in new.items()})


def _files(folder):
    return sorted(os.path.relpath(os.path.join(d, f), folder)
                  for d, _, fs in os.walk(folder) for f in fs)


def regenerate_stored(scratch):
    """Control (i): the stored 10-module single-shell sets regenerated from their stored
    settings into scratch, compared file by file. Returns the record; refuses on a
    difference."""
    from ppvc_instance_generator import DEFAULT_FACTORY, ppvc_instance_generator, save_instance
    out = {}
    for level in MIX_LEVELS:
        stored = os.path.join(ROOT_DIR, P87.LEVELS[level]['base'])
        with open(os.path.join(stored, 'dataset_meta.json')) as f:
            meta = json.load(f)
        if meta.get('ppvc_mix') != MIX[level] or int(meta['n_modules']) != 10:
            refuse(f'{stored}/dataset_meta.json records {meta}, not a 10-module {MIX[level]} set')
        dst = os.path.join(scratch, os.path.basename(stored))
        os.makedirs(dst)
        for i in range(int(meta['n_instances'])):
            jl, pt, m = ppvc_instance_generator(n_modules=int(meta['n_modules']),
                                                class_mix=meta['ppvc_mix'],
                                                station_counts=DEFAULT_FACTORY,
                                                seed=int(meta['seed0']) + i)
            save_instance(os.path.join(dst, inst(i)), jl, pt, m)
        want = [f for f in _files(stored) if f != 'dataset_meta.json']
        got = _files(dst)
        if got != want:
            refuse(f'regenerated {len(got)} files, {stored} holds {len(want)} instance files')
        diff = [f for f in want if open(os.path.join(stored, f), 'rb').read()
                != open(os.path.join(dst, f), 'rb').read()]
        if diff:
            refuse(f'control (i): {len(diff)} of {len(want)} regenerated files of {stored} differ '
                   f'(first {diff[0]}); Part B stops')
        out[level] = dict(stored=rel(stored), settings=meta, files_compared=len(want),
                          identical=True,
                          call=f'ppvc_instance_generator(n_modules={meta["n_modules"]}, '
                               f'class_mix={meta["ppvc_mix"]!r}, station_counts=DEFAULT_FACTORY, '
                               f'seed={meta["seed0"]} + i); save_instance')
    return out


def descriptor_control(P86, sizes=NEW_SIZES, k=2):
    """Control (ii): descriptor_text of p86 on the stored base set of each production size,
    first k instances, eight combinations, against the stored descriptor files."""
    n = 0
    for size in sizes:
        spec = P101.SIZES[size]
        for i in range(k):
            for r in R8:
                p = os.path.join(ROOT_DIR, spec['data'], 'desc', r, f'{inst(i)}.desc.json')
                with open(p) as f:
                    stored = f.read()
                label = json.loads(stored)['base_instance']
                txt = P86.descriptor_text(os.path.join(spec['base'], inst(i)), label, i, r)
                if txt != stored:
                    refuse(f'control (ii): p86 descriptor_text does not reproduce {rel(p)}')
                n += 1
    return dict(function='scripts/p86_extend_scale_sets.py descriptor_text', sizes=list(sizes),
                instances=k, files_compared=n, identical=True)


def build_base(data_root, level, size):
    """The base set of a new level and size: generated into a staging folder, then moved into
    place, or compared file by file with the folder in place."""
    from ppvc_instance_generator import DEFAULT_FACTORY, ppvc_instance_generator, save_instance
    target = base_dir(data_root, level, size)
    seed0, n = SEED0[(level, size)], MODULES[size]
    stage = target + '.__build'
    if os.path.exists(stage):
        shutil.rmtree(stage)
    os.makedirs(stage)
    for i in range(N_NEW):
        jl, pt, m = ppvc_instance_generator(n_modules=n, class_mix=MIX[level],
                                            station_counts=DEFAULT_FACTORY, seed=seed0 + i)
        save_instance(os.path.join(stage, inst(i)), jl, pt, m)
    meta = {'seed0': seed0, 'n_instances': N_NEW, 'n_modules': n, 'ppvc_mix': MIX[level],
            'ppvc_factory': 'default', 'n_stations': int(sum(DEFAULT_FACTORY.values())),
            'purpose': f'{n}-module single-shell {SHELL[level]} project test set (eval-only) of '
                       f'scripts/p129_project_mix.py; instance i uses generator seed seed0 + i',
            'generator': 'scripts/p129_project_mix.py'}
    with open(os.path.join(stage, 'dataset_meta.json'), 'w') as f:
        f.write(json.dumps(meta, indent=1))
    if os.path.exists(target):
        a, b = _files(stage), _files(target)
        diff = [f for f in a if f not in b or open(os.path.join(stage, f), 'rb').read()
                != open(os.path.join(target, f), 'rb').read()]
        if a != b or diff:
            refuse(f'{target} exists with other content ({len(diff)} files differ); use another '
                   f'--data_root')
        shutil.rmtree(stage)
        state = 'in place and identical'
    else:
        os.makedirs(os.path.dirname(target), exist_ok=True)
        os.rename(stage, target)
        state = 'written'
    return target, state


def build_level(P86, data_root, level, size):
    """Descriptors of a new set, each checked by C1 to C5, and the manifest."""
    from constraint_family import checks
    from constraint_family.descriptor import ConstraintDescriptor
    from ppvc_instance_generator import ROUTING_CLASSES, load_instance
    base, state = build_base(data_root, level, size)
    digest = hashlib.sha256()
    for f in _files(base):
        digest.update(f.encode())
        digest.update(open(os.path.join(base, f), 'rb').read())
    c4 = {r: 0 for r in R8}
    classes, names = set(), []
    for i in range(N_NEW):
        stem = os.path.join(base, inst(i))
        jl, pt, meta = load_instance(stem)
        classes |= {ROUTING_CLASSES[int(c)] for c in meta['routing_class']}
        names.append(inst(i))
        for r in R8:
            txt = P86.descriptor_text(stem, rel(stem), i, r)
            P102.write_once(os.path.join(desc_dir(data_root, level, size), 'desc', r,
                                         f'{inst(i)}.desc.json'), txt)
            digest.update(txt.encode())
            d = ConstraintDescriptor.from_json(json.loads(txt), setup_convention='attached')
            rep = {x.id: x for x in checks.run_all(d, pt, False, job_length=jl)}
            bad = [c for c in ('C1', 'C2', 'C3', 'C5') if rep[c].status == checks.FAIL]
            if bad:
                refuse(f'{level} {size} {r} {inst(i)} fails {bad}: '
                       f'{[rep[c].messages[:1] for c in bad]}')
            c4[r] += rep['C4'].status == checks.WARN
    want = {'RC' if level == 'mix_rc' else 'Steel'}
    shells = {'RC' if c.startswith('RC') else 'Steel' for c in classes}
    if shells != want:
        refuse(f'{level} {size}: route classes {sorted(classes)} hold other shells than {want}')
    man = {'level': level, 'size': size, 'modules': MODULES[size], 'preset': MIX[level],
           'base': rel(base), 'base_state': state, 'seed0': SEED0[(level, size)],
           'n_instances': N_NEW, 'instances': names, 'combinations': list(R8),
           'desc_seed0': DESC_SEED0,
           'descriptor_function': 'scripts/p86_extend_scale_sets.py descriptor_text',
           'checks': 'C1, C2, C3 and C5 pass on every descriptor (constraint_family.checks.run_all)',
           'c4_warn_share': {r: v / N_NEW for r, v in c4.items()},
           'route_classes_present': sorted(classes), 'content_sha256': digest.hexdigest()}
    mp = manifest_path(data_root, level, size)
    if os.path.exists(mp):
        with open(mp) as f:
            if json.load(f).get('content_sha256') != man['content_sha256']:
                refuse(f'{mp} records different content')
    write_json_atomic(mp, man)
    return man


def build(args):
    P86 = _p86()
    os.chdir(ROOT_DIR)
    blocks = check_seed_blocks()
    print('[p129 build] seed blocks in use: ' + '; '.join(
        f'{k} {v[0]}-{v[0] + v[1] - 1}' for k, v in blocks['used'].items()), flush=True)
    print('[p129 build] new seed blocks: ' + '; '.join(
        f'{k} {v[0]}-{v[0] + v[1] - 1}' for k, v in blocks['new'].items()), flush=True)
    scratch = tempfile.mkdtemp(prefix='p129_regen_')
    try:
        regen = regenerate_stored(scratch)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    for lv, x in regen.items():
        print(f'[p129 build] control (i) {lv}: {x["stored"]} regenerated by {x["call"]}: '
              f'{x["files_compared"]} files identical', flush=True)
    desc = descriptor_control(P86)
    print(f'[p129 build] control (ii): {desc["function"]} reproduces {desc["files_compared"]} '
          f'stored descriptors of {", ".join(desc["sizes"])} (instances 000 and 001, eight '
          f'combinations) byte for byte', flush=True)
    for level in args.levels:
        for size in args.sizes:
            if not is_new(level, size):
                continue
            t0 = time.perf_counter()
            man = build_level(P86, args.data_root, level, size)
            print(f'[p129 build] {level} {size}: {man["base"]} ({man["base_state"]}), seeds '
                  f'{man["seed0"]}-{man["seed0"] + N_NEW - 1}, {N_NEW} instances x 8 combinations, '
                  f'route classes {", ".join(man["route_classes_present"])}, C1-C5 pass, C4 '
                  f'warnings ' + ', '.join(f'{r} {v:.2f}' for r, v in man['c4_warn_share'].items())
                  + f' ({time.perf_counter() - t0:.0f} s)', flush=True)
    return dict(blocks=blocks, regenerate=regen, descriptors=desc)


# ---------------------------------------------------------------------------
# units, evaluate, plan
# ---------------------------------------------------------------------------

def all_units(levels, sizes, combos, arms, seeds, rules=RULES7):
    """(level, size, combo, arm, seed, label) of every unit to evaluate (mix levels only)."""
    out = []
    for lv in levels:
        if lv not in MIX_LEVELS:
            continue
        for sz in sizes:
            for c in combos:
                out += [u for u in P103.units_for(lv, sz, c, arms, seeds)
                        if u[3] != 'rule' or u[5] in rules]
    return out


def unit_n(u, n=None):
    return n or (N_STORED_MIX if u[1] == '10x25' else N_NEW)


def unit_cost(u, n=None):
    return COST_S[u[1]]['rule' if u[3] == 'rule' else 'policy'] * unit_n(u, n)


def assign_parts(units, parts, n=None):
    order = sorted(range(len(units)), key=lambda i: (-unit_cost(units[i], n), i))
    loads, out = [0.0] * parts, [[] for _ in range(parts)]
    for i in order:
        p = int(np.argmin(loads))
        out[p].append(i)
        loads[p] += unit_cost(units[i], n)
    return [sorted(x) for x in out], loads


def unit_done(root, u):
    p = result_path(root, *u[:3], u[5])
    return os.path.exists(p) and os.path.exists(p[:-4] + '.json')


def evaluate_unit(root, unit, cell, runner, force):
    """One unit on one cell: rollouts, check C6 of every schedule, npz and JSON record."""
    level, size, combo, arm, seed, label = unit
    path = result_path(root, level, size, combo, label)
    names = [c[0] for c in cell]
    if os.path.exists(path) and os.path.exists(path[:-4] + '.json') and not force:
        with open(path[:-4] + '.json') as f:
            old = json.load(f)
        if old.get('instances') != names:
            refuse(f'{path} covers other instances; use --force or another --root')
        if arm != 'rule':
            ckpt = os.path.join(ROOT_DIR, 'trained_network', 'FAMILY',
                                f'{label[len("greedy+"):]}.pth')
            if old.get('policy', {}).get('checkpoint_sha256') != P101.sha256_file(ckpt):
                refuse(f'{path} was written by another checkpoint of the same run; use --force '
                       f'or another --root')
        return 'skip', 0.0
    P103.assert_class_count(P103.BASE_K)
    t0, c0 = time.perf_counter(), time.process_time()
    n, N = len(cell), cell[0][2].shape[0]
    arr = {k: np.zeros(n) for k in ('makespan', 'lb0', 'util', 'ms_over_lb0', 'setup_h')}
    arr['amch'] = np.zeros((n, N), dtype=np.int16)
    for i, (name, jl, pt, meta, desc) in enumerate(cell):
        r = P103.rollout(arm, label, jl, pt, meta, desc, runner)
        v = P103.schedule_check(jl, pt, desc, r)
        if v:
            refuse(f'{label} on {level}/{size}/{combo}/{name} is infeasible: {v[:2]}')
        u, ml, _ = P101.shop_load(pt, r['amch'], r['makespan'], r['lb0'])
        sh, _ = P103.changeover_hours(desc, pt, r['amch'], r['op_ct'])
        for k, val in (('makespan', r['makespan']), ('lb0', r['lb0']), ('util', u),
                       ('ms_over_lb0', ml), ('setup_h', sh)):
            arr[k][i] = val
        arr['amch'][i] = r['amch']
    P101.save_npz_atomic(path, **arr)
    rr = P101.run_record({'unit': list(unit)}, names, t0, c0)
    rr.update({'level': level, 'size': size, 'regime': combo, 'arm': arm, 'seed': seed,
               'label': label, 'mask': P101.MASK, 'environment': 'exact', 'validated': n,
               'violations': 0,
               'validator': 'checks.check_schedule_validation (C6) and the makespan '
                            '(scripts/p103_five_classes.py schedule_check)'})
    if runner is not None and arm != 'rule':
        rr['policy'] = runner.load(label[len('greedy+'):])[1]
    write_json_atomic(path[:-4] + '.json', rr)
    return 'done', rr['cpu_s']


def evaluate(args):
    if 'default' in args.levels:
        refuse('level default is read from the stored arrays, not evaluated (--levels '
               'mix_rc,mix_steel)')
    units = all_units(args.levels, args.sizes, args.combos, args.arms, args.seeds, args.rules)
    if args.part is not None:
        parts, _ = assign_parts(units, args.parts, args.n)
        units = [units[i] for i in parts[args.part]]
    runner = P101.PolicyRunner() if any(a in POLICY_ARMS for a in args.arms) else None
    cache = {}
    t_all = time.perf_counter()
    print(f'[p129 eval] {len(units)} units ({sum(unit_done(args.root, u) for u in units)} '
          f'finished); results under {args.root}', flush=True)
    for k, u in enumerate(units):
        if os.path.exists(args.hold):
            print(f'[p129 eval] hold file {args.hold} present; waiting', flush=True)
            while os.path.exists(args.hold):
                time.sleep(60)
        key = u[:3]
        if key not in cache:
            cache.clear()
            cache[key] = load_cell(*key, data_root=args.data_root, n=args.n)
        st, c = evaluate_unit(args.root, u, cache[key], runner, args.force)
        print(f'[p129 eval {k + 1}/{len(units)}] {u[0]:9s} {u[1]:9s} {u[2]:4s} {u[5]:40s} '
              f'{st} cpu={c:7.1f}s elapsed={time.perf_counter() - t_all:7.0f}s', flush=True)


def plan(args):
    py = sys.executable
    units = all_units(args.levels, args.sizes, args.combos, args.arms, args.seeds, args.rules)
    parts, loads = assign_parts(units, args.parts, args.n)
    built = {f'{lv} {sz}': os.path.exists(manifest_path(args.data_root, lv, sz))
             for lv in args.levels if lv in MIX_LEVELS for sz in args.sizes if is_new(lv, sz)}
    done = [unit_done(args.root, u) for u in units]
    cost = sum(unit_cost(u, args.n) for u in units)
    left = sum(unit_cost(u, args.n) for u, d in zip(units, done) if not d)
    print(f'[p129 plan] {len(units)} result files ({sum(done)} finished), estimated '
          f'{cost / 3600:.2f} core-hours of CPU time in all and {left / 3600:.2f} left (COST_S); '
          f'{args.parts} parts of {min(loads) / 3600:.2f} to {max(loads) / 3600:.2f} h; new sets '
          f'built: ' + ', '.join(f'{k} {"yes" if v else "NO"}' for k, v in built.items()))
    extra = ''
    if list(args.seeds) != list(SEEDS):
        extra += f' --seeds {",".join(map(str, args.seeds))}'
    for opt, val, full in (('--levels', args.levels, MIX_LEVELS), ('--sizes', args.sizes, SIZES),
                           ('--combos', args.combos, R8), ('--arms', args.arms, ARM_CHOICES),
                           ('--rules', args.rules, RULES7)):
        if list(val) != list(full):
            extra += f' {opt} {",".join(val)}'
    if args.root != DEFAULT_ROOT:
        extra += f' --root {args.root}'
    if args.data_root != DEFAULT_DATA:
        extra += f' --data_root {args.data_root}'
    for i in range(args.parts):
        mine = [units[j] for j in parts[i]]
        rest = sum(unit_cost(u, args.n) for u in mine if not unit_done(args.root, u))
        print(f'# part {i}: {len(mine)} units, {loads[i] / 3600:.2f} core-hours '
              f'({rest / 3600:.2f} left)')
        print(f'CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c <CORE_{i}> '
              f'{py} -B scripts/p129_project_mix.py evaluate --part {i} --parts {args.parts}'
              f'{extra}')
    print(f'[p129 plan] total {cost / 3600:.2f} core-hours')
    return dict(core_h=cost / 3600, parts=[x / 3600 for x in loads], units=len(units))


# ---------------------------------------------------------------------------
# reading results and the controls of infer
# ---------------------------------------------------------------------------

def read_unit(root, level, size, combo, label, names, tally):
    p = result_path(root, level, size, combo, label)
    if not (os.path.exists(p) and os.path.exists(p[:-4] + '.json')):
        refuse(f'result {level}/{size}/{combo}/{label} is missing ({p})')
    with open(p[:-4] + '.json') as f:
        rec = json.load(f)
    want = {'level': level, 'size': size, 'regime': combo, 'label': label}
    bad = {k: rec.get(k) for k, v in want.items() if rec.get(k) != v}
    if bad or rec.get('instances') != names:
        refuse(f'{p}: record {bad or "instance list"} does not match the study')
    if rec.get('violations') != 0 or rec.get('validated') != len(names):
        refuse(f'{p}: {rec.get("violations")} violations over {rec.get("validated")} validated '
               f'schedules, the study needs 0 over {len(names)}')
    tally['validated'] += rec['validated']
    with np.load(p) as z:
        return np.asarray(z['makespan'], dtype=float)


def _spec_default(size, combo, seed):
    if size == '10x25':
        return REG.result_path('specialist', combo, seed, size, combo)
    import p128_production_specialists as P128
    return P128.spec_path(size, combo, seed)


def default_paths(size, combo, seeds):
    """{('rule', R) or (arm, seed): array path} of the default level."""
    out = {('rule', q): f'{REG.rule_dir(size)}/Result_{q}_{combo}.npy' for q in RULES7}
    for s in seeds:
        out[('pact', s)] = REG.result_path('released', None, s, size, combo)
        out[('spec', s)] = _spec_default(size, combo, s)
    return out


def default_arrays(size, combo, seeds, n):
    """(rules {R: [n]}, {arm: [k, n]}) of the default level, from the stored arrays."""
    arr = {}
    for key, p in default_paths(size, combo, seeds).items():
        if not os.path.exists(p):
            refuse(f'the default-level array {p} is missing')
        a = REG.read_path(p)
        if len(a) != P101.SIZES[size]['n']:
            refuse(f'{p} holds {len(a)} rows, the test set has {P101.SIZES[size]["n"]}')
        arr[key] = a[:n]
    rules = {q: arr[('rule', q)] for q in RULES7}
    pol = {a: np.stack([arr[(a, s)] for s in seeds]) for a in POLICY_ARMS}
    return rules, pol


def p87_units(seeds):
    """[(unit, p87 result path)] of every 10-module unit p87 stored: the policy, the specialist
    and the four classical rules of these seeds, levels mix_rc and mix_steel."""
    out = []
    for level in MIX_LEVELS:
        for combo in R8:
            for u in all_units([level], ['10x25'], [combo], ARM_CHOICES, seeds, RULES4):
                p = P87.result_path(P87.DEFAULT_ROOT, 'exact', level, combo, u[5])
                if os.path.exists(p):
                    out.append((u, p))
    return out


def compare_p87(root, seeds, n=None, need_all=False):
    """Every 10-module unit p87 stored against this script's unit in root.
    {(arm word, level, combo): [equal makespans, compared]} and the differences; with
    need_all a p87 unit without its counterpart is a difference."""
    counts, diffs, absent = {}, [], []
    for u, p in p87_units(seeds):
        level, size, combo, arm, seed, label = u
        mine = result_path(root, level, size, combo, label)
        if not os.path.exists(mine):
            absent.append(mine)
            continue
        with np.load(p) as z:
            theirs = np.asarray(z['makespan'], dtype=float)
        with open(p[:-4] + '.json') as f:
            names = json.load(f).get('instances')
        with open(mine[:-4] + '.json') as f:
            my_names = json.load(f).get('instances')
        with np.load(mine) as z:
            a = np.asarray(z['makespan'], dtype=float)
        k = min(len(a), len(theirs)) if n else len(theirs)
        if (not n and (my_names != names or len(a) != len(theirs))) or \
                my_names[:k] != names[:k]:
            diffs.append(f'{mine}: instance list differs from {p}')
            continue
        eq = int((a[:k] == theirs[:k]).sum())
        word = 'rule ' + label if arm == 'rule' else ('policy' if arm == 'pact' else 'specialist')
        c = counts.setdefault((word, level, combo), [0, 0])
        c[0] += eq
        c[1] += k
        if eq != k:
            diffs.append(f'{label} {level}/{combo}: {k - eq} of {k} makespans differ from {p}')
    if need_all and absent:
        diffs += [f'{x}: missing (p87 stored this unit)' for x in absent[:5]]
    return counts, diffs, absent


def _cmp_row(mine, theirs, keys, where, bad):
    n = 0
    for k in keys:
        a, b = mine.get(k), theirs.get(k)
        if isinstance(b, str) or isinstance(a, str) or a is None or b is None:
            ok = a == b
        else:
            a, b = float(a), float(b)
            ok = (np.isnan(a) and np.isnan(b)) or abs(a - b) <= REPRO_TOL
        if not ok:
            bad.append(f'{where} {k}: {mine.get(k)!r} here, {theirs.get(k)!r} in the record')
        n += 1
    return n


def read_json(path, generator, seeds, boot=None, extra=None):
    if not os.path.exists(path):
        refuse(f'the record {path} does not exist')
    with open(path) as f:
        rec = json.load(f)
    if rec.get('generator') != generator:
        refuse(f'{path} was written by {rec.get("generator")!r}, not {generator}')
    if [int(x) for x in rec.get('seeds') or []] != list(seeds):
        refuse(f'{path} was written with seeds {rec.get("seeds")}, this run uses {list(seeds)}')
    if boot is not None and int(rec.get('B', rec.get('boot', 0)) or 0) != int(boot):
        refuse(f'{path} was computed with B = {rec.get("B", rec.get("boot"))}, this run with {boot}')
    for k, v in (extra or {}).items():
        if rec.get(k) != v:
            refuse(f'{path}: {k} is {rec.get(k)!r}, the comparison needs {v!r}')
    return rec


def record_paths():
    """The records the rows are compared with: p88's and p128's under PACT_OUT (as the chain
    sets it, out/exact; arm_registry.out refuses an unset PACT_OUT outside the default
    setting), p87's sensitivity record at the repository root."""
    import p128_production_specialists as P128
    return {'family': REG.out(P128.RECORD), 'prodspec': REG.out(P128.OUT_FILES['json']),
            'sens': P87.infer_outputs(ROOT_DIR)['json']}


def control_records(res, seeds, boot, paths=None):
    """The default rows against the records of p88 (10 modules) and p128 (50 and 80), the
    10-module mix rows against the sensitivity record. Returns the counts; refuses on a
    difference."""
    import p128_production_specialists as P128
    paths = paths or record_paths()
    out, bad = {}, []
    S = res['sizes']
    if 'default' in res['levels']:
        if '10x25' in S:
            rec = read_json(paths['family'], 'scripts/p88_family_inference.py', seeds, boot)
            P = S['10x25']['policy']['default']
            n = _cmp_row(P['pooled']['pol_spec'], rec['parity']['pooled']['All'], SPEC_KEYS,
                         '10x25 default pooled pol_spec', bad)
            for r in R8:
                n += _cmp_row(P['rows']['pol_rule'][r], rec['rule_pool'][r], RULE_KEYS,
                              f'10x25 default {r} pol_rule', bad)
            out['10x25 default'] = dict(record=paths['family'], values=n)
        sizes = [s for s in NEW_SIZES if s in S]
        if sizes:
            rec = read_json(paths['prodspec'], P128.GENERATOR, seeds, boot, {'dry': False})
            for size in sizes:
                P, X = S[size]['policy']['default'], rec['sizes'][size]
                n = 0
                for r in R8:
                    n += _cmp_row(P['rows']['pol_spec'][r], X['parity'][r], SPEC_KEYS,
                                  f'{size} default {r} pol_spec', bad)
                    n += _cmp_row(P['rows']['pol_rule'][r], X['rule_pool'][r], RULE_KEYS,
                                  f'{size} default {r} pol_rule', bad)
                n += _cmp_row(P['pooled']['pol_spec'], X['parity_pooled']['All'], SPEC_KEYS,
                              f'{size} default pooled pol_spec', bad)
                n += _cmp_row(P['pooled']['pol_rule'], X['rule_pool_pooled'], POOL_KEYS,
                              f'{size} default pooled pol_rule', bad)
                out[f'{size} default'] = dict(record=paths['prodspec'], values=n)
    mixes = [lv for lv in MIX_LEVELS if lv in res['levels']]
    if '10x25' in S and mixes:
        rec = read_json(paths['sens'], 'scripts/p87_sensitivity.py infer', seeds, boot,
                        {'partial': False, 'mask': 'exact'})
        for lv in mixes:
            P, X = S['10x25']['policy'][lv], rec['levels'][lv]
            n = 0
            for r in R8:
                n += _cmp_row(P['rows']['pol_spec'][r], X['rows']['pol_spec'][r], SPEC_KEYS,
                              f'10x25 {lv} {r} pol_spec', bad)
                n += _cmp_row(P['rows']['pol_rule'][r], X['rows']['pol_rule'][r], RULE_KEYS,
                              f'10x25 {lv} {r} pol_rule', bad)
            n += _cmp_row(P['pooled']['pol_spec'], X['pools']['pool']['pol_spec'], SPEC_KEYS,
                          f'10x25 {lv} pooled pol_spec', bad)
            n += _cmp_row(P['pooled']['pol_rule'], X['pools']['pool']['pol_rule'], POOL_KEYS,
                          f'10x25 {lv} pooled pol_rule', bad)
            out[f'10x25 {lv}'] = dict(record=paths['sens'], values=n)
    if bad:
        refuse(f'{len(bad)} values differ from the stored records (first: {bad[0]})')
    return out


# ---------------------------------------------------------------------------
# infer
# ---------------------------------------------------------------------------

def infer(args):
    """Every outcome of the module doc. Returns the result dictionary."""
    partial = bool(args.partial)
    arms = list(args.arms)
    full_set = (list(args.levels) == list(LEVELS) and list(args.sizes) == list(SIZES)
                and list(args.combos) == list(R8) and sorted(arms) == sorted(ARM_CHOICES)
                and list(args.rules) == list(RULES7) and not args.n)
    if args.n and not partial:
        refuse('--n (a subset of the instances) needs --partial')
    if not partial and not full_set:
        refuse('every level, size, combination, arm and rule is required; a subset needs '
               '--partial')
    if sorted(arms) != sorted(ARM_CHOICES) or list(args.rules) != list(RULES7):
        refuse('the comparisons need the arms pact, spec and rules and the seven rules')
    if args.out is None:
        if partial:
            refuse('--partial writes only to --out <folder>')
        out_root = ROOT_DIR
    else:
        out_root = os.path.abspath(args.out)
        P101.guard_out(out_root)
    paths = {k: os.path.join(out_root, v) for k, v in OUT_FILES.items()}
    for p in paths.values():
        if os.path.exists(p):
            os.remove(p)
    if len(args.seeds) < 2 and not partial:
        refuse('at least two training seeds are required (the unit of the hierarchical '
               'bootstrap)')
    rpaths = None if partial else record_paths()       # an unset PACT_OUT refuses here
    st = P102._stats()
    res = {'generator': GENERATOR, 'partial': partial, 'arms': arms, 'seeds': list(args.seeds),
           'mask': P101.MASK, 'B': args.boot, 'levels': list(args.levels),
           'convention': 'd = 100 (first - second) / second, seed s against seed s, per base '
                         'instance; negative means the first-named method is shorter',
           'best_rule': 'per-instance minimum of ' + ', '.join(RULES4),
           'best_rule7': 'per-instance minimum of ' + ', '.join(RULES7),
           'holm_family': 'the eight combinations of a level and size',
           'sizes': {}, 'controls': {}}
    if '10x25' in args.sizes and any(lv in MIX_LEVELS for lv in args.levels):
        counts, diffs, absent = compare_p87(args.root, args.seeds, args.n, need_all=not partial)
        if diffs:
            refuse(f'{len(diffs)} 10-module units differ from p87 (first: {diffs[0]})')
        res['controls']['p87'] = {
            'units': sum(1 for _ in p87_units(args.seeds)) - len(absent),
            'makespans_equal': sum(v[0] for v in counts.values()),
            'makespans_compared': sum(v[1] for v in counts.values()),
            'per_arm_combination': {f'{k[0]}|{k[1]}|{k[2]}': v for k, v in counts.items()}}
    for size in args.sizes:
        S = {'modules': MODULES[size], 'levels': {}, 'validated': 0}
        gaps = {lv: {} for lv in args.levels}
        tally = {'validated': 0}
        for lv in args.levels:
            names = level_names(lv, size, args.data_root, args.n)
            Lv = {'n': len(names), 'combos': {}}
            if is_new(lv, size):
                mp = manifest_path(args.data_root, lv, size)
                if not os.path.exists(mp):
                    refuse(f'{mp} is missing; run the build phase first')
                with open(mp) as f:
                    man = json.load(f)
                Lv['manifest'] = {k: v for k, v in man.items() if k != 'instances'}
            for combo in args.combos:
                if lv == 'default':
                    ms, P = default_arrays(size, combo, args.seeds, len(names))
                    src = 'stored arrays'
                else:
                    ms = {r: read_unit(args.root, lv, size, combo, r, names, tally)
                          for r in RULES7}
                    P = {a: np.stack([read_unit(args.root, lv, size, combo,
                                                'greedy+' + P103.model_name(a, combo, s),
                                                names, tally) for s in args.seeds])
                         for a in POLICY_ARMS}
                    src = 'this script'
                b4 = np.min(np.stack([ms[r] for r in RULES4]), axis=0)
                b7 = np.min(np.stack([ms[r] for r in RULES7]), axis=0)
                gaps[lv][combo] = {
                    'pol_spec': 100.0 * (P['pact'] - P['spec']) / P['spec'],
                    'pol_rule': 100.0 * (P['pact'] - b4[None]) / b4[None],
                    'pol_rule7': 100.0 * (P['pact'] - b7[None]) / b7[None]}
                Lv['combos'][combo] = dict(
                    rule_ms={r: float(ms[r].mean()) for r in RULES7},
                    best_rule_ms=float(b4.mean()), best_rule7_ms=float(b7.mean()),
                    policy_ms={a: float(P[a].mean()) for a in POLICY_ARMS}, source=src)
            S['levels'][lv] = Lv
        S['policy'] = P103.policy_block(st, gaps, list(args.levels), list(args.combos), args.boot)
        S['validated'] = tally['validated']
        res['sizes'][size] = S
    if partial:
        res['controls']['records'] = 'not compared (partial run)'
    else:
        res['controls']['records'] = control_records(res, args.seeds, args.boot, rpaths)
    lines = macro_lines(res)
    names_m = check_macro_names(lines, paths)
    from macro_io import emit
    emit(paths['macros'], lines)
    write_json_atomic(paths['json'], jsonable(res))
    os.makedirs(os.path.dirname(paths['md']), exist_ok=True)
    with open(paths['md'], 'w') as f:
        f.write('\n'.join(markdown(res)) + '\n')
    for k in ('json', 'md'):
        print('wrote', paths[k])
    print(f'{len(names_m)} macros' + ('; PARTIAL' if partial else ''))
    return res


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------

def _tex(v):
    import stats_family as SF
    return SF.TEX.get(v, v)


SPEC_WORDS = {'equivalent': 'Eq', 'non-inferior only': 'Ni', 'inferior': 'Inf',
              'inconclusive': 'Inc'}
RULE_WORDS = {'ahead': 'Ahead', 'behind': 'Behind', 'not distinguishable': 'Nd'}


def macro_lines(res):
    mark = 'PARTIAL ' if res['partial'] else ''
    seeds = ','.join(map(str, res['seeds']))
    L = [f'% Generated by {GENERATOR} -- do not hand-edit',
         '% Single-shell project mixes against the mixed test sets, exact mask, 10, 50 and 80 '
         'modules on 25 stations. Levels: Mixed (the test sets of the paper), Concrete '
         '(reinforced-concrete modules alone), Steel (steel modules alone).',
         f'% Gaps in %: d = 100 (policy - other) / other, the denominator the specialist, the '
         f'per-instance best of FIFO, MOR, SPT and MWKR, or the best of these and ECT, SSPT, '
         f'LLF; negative means the policy is shorter; training seed s against seed s over seeds '
         f'{seeds}; B = {res["B"]}; combined verdicts with Holm over the eight combinations of a '
         f'level and size.']
    if res['partial']:
        L.insert(1, '% PARTIAL: subset of levels, sizes, combinations or instances; not for the '
                    'manuscript.')
    for size, S in res['sizes'].items():
        w = SIZE_WORDS[size]
        P = S['policy']
        for lv, Lv in S['levels'].items():
            lw = LEVEL_WORDS[lv]
            where = f'{mark}{size}, level {lv} ({Lv["n"]} instances, seeds {seeds})'
            L.append(f'\\newcommand{{\\{PREFIX}N{lw}{w}}}{{{Lv["n"]}}}  % {where}: instances')
            for r, C in Lv['combos'].items():
                L.append(f'\\newcommand{{\\{PREFIX}MsPolicy{lw}{w}{r}}}{{{u1(C["policy_ms"]["pact"])}}}'
                         f'\\newcommand{{\\{PREFIX}MsRule{lw}{w}{r}}}{{{u1(C["best_rule_ms"])}}}'
                         f'\\newcommand{{\\{PREFIX}MsRuleSeven{lw}{w}{r}}}{{{u1(C["best_rule7_ms"])}}}'
                         f'  % {where}, {r}: mean makespan (h) of the policy, of the best '
                         f'classical rule and of the best of seven rules')
            for key, (first, second, cw) in COMPARISONS.items():
                for r in list(Lv['combos']) + ['Pool']:
                    c = P[lv]['pooled'][key] if r == 'Pool' else P[lv]['rows'][key][r]
                    s = (f'\\newcommand{{\\{PREFIX}{cw}{lw}{w}{r}}}{{{f2(c["mean"])}}}'
                         f'\\newcommand{{\\{PREFIX}{cw}Lo{lw}{w}{r}}}{{{f2(c["lo"])}}}'
                         f'\\newcommand{{\\{PREFIX}{cw}Hi{lw}{w}{r}}}{{{f2(c["hi"])}}}')
                    if key == 'pol_spec':
                        s += ''.join(f'\\newcommand{{\\{PREFIX}{cw}V{MW[mg]}{lw}{w}{r}}}'
                                     f'{{{_tex(c[f"v{mg}"])}}}' for mg in MARGINS)
                    elif r != 'Pool':
                        s += f'\\newcommand{{\\{PREFIX}{cw}Verdict{lw}{w}{r}}}{{{c["verdict"]}}}'
                    s += f'\\newcommand{{\\{PREFIX}{cw}Dir{lw}{w}{r}}}{{{_tex(c["direction"])}}}'
                    L.append(s + f'  % {where}, {r}: {first} vs {second}, {c["scheme"]} 95% '
                                 f'interval over {c["k"]} runs and {c["n"]} instances'
                             + (f', {c["c"]} combinations' if c['c'] > 1 else ''))
            cnt = P[lv]['counts']
            s = ''.join(f'\\newcommand{{\\{PREFIX}PolSpecV{MW[mg]}{SPEC_WORDS[v]}{lw}{w}}}'
                        f'{{{cnt[f"pol_spec|{mg}"][v]}}}' for mg in MARGINS for v in SPEC_WORDS)
            for key in ('pol_rule', 'pol_rule7'):
                s += ''.join(f'\\newcommand{{\\{PREFIX}{COMPARISONS[key][2]}{RULE_WORDS[v]}{lw}{w}}}'
                             f'{{{cnt[key][v]}}}' for v in RULE_WORDS)
            L.append(s + f'  % {where}: combinations per verdict, of {len(Lv["combos"])}')
    return L


def defined_macros(own_paths):
    rx = re.compile(r'\\(?:newcommand|renewcommand|providecommand)\*?\s*\{?\s*\\([A-Za-z@]+)'
                    r'|\\def\s*\\([A-Za-z@]+)')
    own = {os.path.abspath(p) for p in own_paths}
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT_DIR, 'manuscript', '**', '*.tex'), recursive=True)):
        if os.path.abspath(f) in own:
            continue
        text = '\n'.join(re.sub(r'(?<!\\)%.*', '', x)
                         for x in open(f, errors='replace').read().splitlines())
        for a, b in rx.findall(text):
            out.setdefault(a or b, os.path.relpath(f, ROOT_DIR))
    return out


def check_macro_names(lines, out_paths):
    names = re.findall(r'\\newcommand\{\\([^}]*)\}', '\n'.join(lines))
    bad = [x for x in names if not re.fullmatch(PREFIX + r'[A-Za-z]+', x)]
    if bad:
        refuse(f'macro names that are not {PREFIX} plus letters: {bad[:5]}')
    if len(names) != len(set(names)):
        refuse(f'duplicate macro names: {sorted({x for x in names if names.count(x) > 1})[:5]}')
    have = defined_macros([out_paths['macros'], os.path.join(ROOT_DIR, OUT_FILES['macros'])])
    clash = sorted(x for x in names if x in have)
    if clash:
        refuse('macro names already defined: ' + ', '.join(f'\\{x} ({have[x]})' for x in clash[:5]))
    taken = sorted(x for x in have if x.startswith(PREFIX))
    if taken:
        refuse(f'the prefix {PREFIX} is used by \\{taken[0]} ({have[taken[0]]})')
    return names


def markdown(res):
    L = [f'# Single-shell project mixes ({GENERATOR})', '']
    if res['partial']:
        L += ['PARTIAL: a subset of levels, sizes, combinations or instances. Not for the '
              'manuscript.', '']
    L += [f'Exact mask; training seeds {",".join(map(str, res["seeds"]))}; B = {res["B"]}. '
          'Levels: default (the mixed test sets), mix_rc (reinforced-concrete modules alone), '
          f'mix_steel (steel modules alone). {res["convention"]}. Best rule: {res["best_rule"]}; '
          f'best of seven: {res["best_rule7"]}. Holm over {res["holm_family"]}.', '']
    c = res['controls']
    if 'p87' in c:
        L += [f'Control p87: {c["p87"]["makespans_equal"]} of {c["p87"]["makespans_compared"]} '
              f'makespans of {c["p87"]["units"]} stored 10-module units equal.', '']
    if isinstance(c.get('records'), dict):
        L += ['Records reproduced: ' + '; '.join(f'{k} {v["values"]} values ({v["record"]})'
                                                for k, v in c['records'].items()) + '.', '']

    def g(x):
        return f'{f2(x["mean"])} [{f2(x["lo"])}, {f2(x["hi"])}]'
    for size, S in res['sizes'].items():
        lvls = list(S['levels'])
        P = S['policy']
        L += [f'## {size} ({S["modules"]} modules; '
              + ', '.join(f'{lv} {S["levels"][lv]["n"]} instances' for lv in lvls) + ')', '',
              '| combination | ' + ' | '.join(f'{lv}: policy - specialist | {lv}: policy - best '
                                              f'rule | {lv}: policy - best of seven'
                                              for lv in lvls) + ' |',
              '|' + '---|' * (1 + 3 * len(lvls))]
        combos = list(S['levels'][lvls[0]]['combos'])
        for r in combos + ['pooled']:
            cells = []
            for lv in lvls:
                for key in COMPARISONS:
                    x = P[lv]['pooled'][key] if r == 'pooled' else P[lv]['rows'][key][r]
                    tail = (f' {x["v0.5"]} / {x["v1.0"]}' if key == 'pol_spec'
                            else (f' {x["verdict"]}' if 'verdict' in x
                                  else f' direction {x["direction"]}'))
                    cells.append(g(x) + tail)
            L.append(f'| {r} | ' + ' | '.join(cells) + ' |')
        L += ['', '| level | ' + ' | '.join(combos) + ' |', '|' + '---|' * (1 + len(combos))]
        for lv in lvls:
            L.append(f'| {lv}: mean makespan policy / best rule / best of seven (h) | '
                     + ' | '.join(f'{u1(C["policy_ms"]["pact"])} / {u1(C["best_rule_ms"])} / '
                                  f'{u1(C["best_rule7_ms"])}'
                                  for C in S['levels'][lv]['combos'].values()) + ' |')
        L.append('')
    return L


# ---------------------------------------------------------------------------
# check (read only)
# ---------------------------------------------------------------------------

def unit_state(root, data_root, seeds):
    """(present, missing, units) over every unit of the full design."""
    units = all_units(MIX_LEVELS, SIZES, R8, ARM_CHOICES, seeds)
    have = [u for u in units if unit_done(root, u)]
    return have, [u for u in units if u not in have], units


def check(args):
    have, miss, units = unit_state(args.root, args.data_root, args.seeds)
    ok = not miss
    for lv in MIX_LEVELS:
        for sz in NEW_SIZES:
            mp = manifest_path(args.data_root, lv, sz)
            there = os.path.exists(mp)
            ok &= there
            print(f'p129: new set {lv} {sz}: ' + (f'built ({mp})' if there else 'NOT BUILT'))
    print(f'p129: {len(have)} of {len(units)} result files of the design under {args.root}')
    for lv in MIX_LEVELS:
        for sz in SIZES:
            for arm in ('pact', 'spec', 'rule'):
                k = sum(1 for u in units if u[:2] == (lv, sz) and u[3] == arm)
                h = sum(1 for u in have if u[:2] == (lv, sz) and u[3] == arm)
                print(f'p129:   {lv:9s} {sz:9s} {arm:5s} {h:3d} of {k}')
    counts, diffs, absent = compare_p87(args.root, args.seeds, args.n)
    for (word, lv, combo), (eq, n) in sorted(counts.items(), key=lambda x: (x[0][1], R8.index(
            x[0][2]), x[0][0])):
        print(f'p129: p87 control {lv} 10x25 {combo:4s} {word:30s} {eq} of {n} makespans equal')
    print(f'p129: p87 control: {sum(v[0] for v in counts.values())} of '
          f'{sum(v[1] for v in counts.values())} makespans equal over {len(counts)} units; '
          f'{len(absent)} stored p87 units not yet evaluated here')
    for d in diffs[:10]:
        print(f'p129: DIFFERENCE {d}')
    ok &= not diffs
    return 0 if ok else 1


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('phase', choices=['build', 'plan', 'evaluate', 'infer', 'check'])
    ap.add_argument('--root', default=DEFAULT_ROOT, help='results folder')
    ap.add_argument('--data_root', default=DEFAULT_DATA, help='folder of the new sets')
    ap.add_argument('--levels', default=None)
    ap.add_argument('--sizes', default=','.join(SIZES))
    ap.add_argument('--combos', default=','.join(R8))
    ap.add_argument('--arms', default=','.join(ARM_CHOICES))
    ap.add_argument('--rules', default=','.join(RULES7))
    ap.add_argument('--seeds', default='301-305')
    ap.add_argument('--n', type=int, default=0, help='first n instances (smoke only)')
    ap.add_argument('--part', type=int, default=None)
    ap.add_argument('--parts', type=int, default=1)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--partial', action='store_true')
    ap.add_argument('--out', default=None)
    ap.add_argument('--boot', type=int, default=B)
    ap.add_argument('--hold', default=HOLD, help='evaluate waits while this file exists')
    args = ap.parse_args(_ARGV)
    P101.check_mask_options()

    def canon(text, allowed, what):
        got = P101.parse_list(text, allowed, what)
        return [x for x in allowed if x in got]
    lv_default = ','.join(LEVELS if args.phase == 'infer' else MIX_LEVELS)
    args.levels = canon(args.levels or lv_default, LEVELS, 'levels')
    args.sizes = canon(args.sizes, SIZES, 'sizes')
    args.combos = canon(args.combos, R8, 'combinations')
    args.arms = canon(args.arms, ARM_CHOICES, 'arms')
    args.rules = canon(args.rules, RULES7, 'rules')
    args.seeds = list(REG.parse_seeds(args.seeds))
    for s in args.seeds:
        if s not in REG.PLANNED_SEEDS['exact']:
            sys.exit(f'REFUSING: seed {s} is not planned in the exact environment')
    args.n = args.n or None
    if args.parts < 1 or (args.part is not None and not 0 <= args.part < args.parts):
        sys.exit(f'--part {args.part} outside 0..{args.parts - 1}')
    os.chdir(ROOT_DIR)
    try:
        r = {'build': build, 'plan': plan, 'evaluate': evaluate, 'infer': infer,
             'check': check}[args.phase](args)
    except Refused as e:
        sys.exit(f'*** REFUSING ***\n{e}')
    if args.phase == 'check':
        sys.exit(r)


if __name__ == '__main__':
    main()
