r"""Extend the 50- and 80-module production-size test sets from 20 to 30
instances per regime, as the exact continuation of their generator sequence.

The 20-instance sets are frozen, because other studies reuse them:

  data/PPVC/50x25+ppvc-mixed   data/FAMILY/50x25
  data/PPVC/80x25+ppvc-mixed   data/FAMILY/80x25

This script only reads them. It writes into the scratch folder and into four
new folders:

  data/PPVC/<n>x25+ppvc-mixed-n30   base instances instance_000 to instance_029
  data/FAMILY/<n>x25n30             descriptors desc/<regime>/instance_000 to 029

Instances 000 to 019 of the new folders are byte copies of the frozen files.
Instances 020 to 029 are drawn with the seeds that continue the frozen
sequences, through the same calls that built the frozen sets:

  base instance i  scripts/p57_build_scale_data.py
                   ppvc_instance_generator(n_modules=n, class_mix='mixed',
                                           station_counts=DEFAULT_FACTORY,
                                           seed=seed0 + i)
                   save_instance(<folder>/instance_<i>, ...)
                   seed0 = 100000 (50 modules), 110000 (80 modules)
  descriptor i     scripts/p1_build_family_data.py
                   make_descriptor(regime, *load_instance(<folder>/instance_<i>),
                                   n_machines, seed=DESC_SEED0 + i)
                   DESC_SEED0 = 20260706

Subcommands
  reproduce  Regenerate instances 000 to 029 of each size into the scratch
             folder. Instances 000 to 019 (base files, descriptors of all eight
             regimes, both dataset_meta.json files) are compared byte for byte
             with the frozen files. The descriptors are also rebuilt with the
             unmodified build() of scripts/p1_build_family_data.py on the
             30-instance scratch base folder and compared field by field.
             Exit status 1 on any difference.
  build      Run `reproduce`; on success, write the two n30 folders per size.
             Refuses to write anywhere except the n30 folders and the scratch
             folder, and refuses to replace an existing file whose bytes differ
             unless --force is given (an identical existing file is left alone).
             The written folders are then re-read and verified.
  check      Read-only checks of the n30 folders: configuration checks C1 to C7
             (constraint_family/checks.run_all) on every descriptor, dispatching
             rules rolled out as in eval_family.py with each schedule re-validated
             by C6, and per-instance summary statistics. Writes nothing except
             the JSON record named by --out.
  table      Print, from a JSON record of `check` over instances 000 to 029, the
             range of each summary statistic over 000-019 and over 020-029.
  warmstart  Warm-start caches for the CP-SAT campaigns. Reproduces the stored
             best-of-four-rules caches or_solution/FAMILY/warmstart_cache_<n>x25
             (instances 000-019, built by p44 --phase cache through
             p12_cpsat_fairwarm._warm_start) in the scratch folder with the
             unmodified p12 procedure, and compares every field. Only when all
             reproduce: builds 020-029 with the same procedure, validates every
             schedule with cpsat_arms.validate_all, and writes
             or_solution/FAMILY/warmstart_cache_<n>x25n30 (000-019 byte copies of
             the stored files), with the overwrite guard of `build`. --resume
             reuses the archives of an interrupted run in the scratch folder.
  wstable    Print the warm-start makespans per regime of the 30-instance caches.
  wsxmask    After `warmstart`: the same procedure for instances 000-029 under
             the exact non-delay mask (PACT_FIRST_MASK=1 and PACT_MASK_TOL=1 in
             this process, checked through env.mask_setting in every rule
             rollout), validated with cpsat_arms.validate_all, written to
             or_solution/FAMILY/warmstart_cache_xmask_<n>x25n30 with a
             dataset_meta.json recording the mask settings (and one for the
             legacy-mask cache).
  wsxtable   Print legacy against exact-mask warm starts per regime.
  wsmasked   The exact-mask caches with the masked rule set
             (common_utils.masked_rule_action): masked FIFO, MOR and MWKR rolled
             out with p12's environment construction and seeding, SPT taken from
             the `wsxmask` build (the same rule; identical schedules checked on
             instances 000 and 029), the best of four with p12's tie rule,
             cpsat_arms.validate_all on every schedule; replaces the files of
             or_solution/FAMILY/warmstart_cache_xmask_<n>x25n30 and adds
             manifest.json and a dataset_meta.json with rule_set masked.
             --wait SECONDS lets it wait for `wsxmask` to finish a size.
  wsmtable   Print the masked-rule exact-mask caches against the legacy ones.
  verify     Read-only: the frozen folders hold the expected files, and the n30
             folders equal the frozen files (000-019) and a fresh regeneration
             (020-029; descriptors also field by field, convention left out).

Usage (from the repository root, CPU only)
  python scripts/p86_extend_scale_sets.py reproduce [--sizes 50,80] [--scratch DIR]
  python scripts/p86_extend_scale_sets.py build     [--sizes 50,80] [--scratch DIR] [--force]
  python scripts/p86_extend_scale_sets.py check     [--sizes 50,80] [--rules SPT]
                                                    [--indices 0-29] [--out FILE]
  python scripts/p86_extend_scale_sets.py table     --from FILE
  python scripts/p86_extend_scale_sets.py warmstart [--sizes 50,80] [--scratch DIR]
                                                    [--force] [--resume]
  python scripts/p86_extend_scale_sets.py wstable   [--sizes 50,80]
  python scripts/p86_extend_scale_sets.py wsxmask   [--sizes 50,80] [--scratch DIR]
                                                    [--force] [--resume]
  python scripts/p86_extend_scale_sets.py wsxtable  [--sizes 50,80]
  python scripts/p86_extend_scale_sets.py wsmasked  [--sizes 50,80] [--scratch DIR]
                                                    [--resume] [--wait SECONDS]
  python scripts/p86_extend_scale_sets.py wsmtable  [--sizes 50,80]
  python scripts/p86_extend_scale_sets.py verify    [--sizes 50,80]
"""
import glob
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)          # descriptor records store base_instance relative to the root

# params.py (imported through data_utils) parses sys.argv at import time;
# strip this script's flags first, as p1_build_family_data.py does.
_ARGV = sys.argv[1:]
sys.argv = sys.argv[:1]
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np  # noqa: E402

from params import configs  # noqa: E402

# CPU before the environment module fixes EnvState.device (as eval_ga_family.py
# does); the dispatching-rule dynamics do not depend on the device.
configs.device = 'cpu'

from constraint_family.descriptor import (REGIMES_ALL,  # noqa: E402
                                          ConstraintDescriptor, make_descriptor)
from ppvc_instance_generator import (DEFAULT_FACTORY, ROUTING_CLASSES,  # noqa: E402
                                     load_instance, ppvc_instance_generator,
                                     save_instance)

# ---------------------------------------------------------------------------
# Protocol of the frozen sets
# ---------------------------------------------------------------------------

N_FROZEN = 20
N_TOTAL = 30
SEED0 = {50: 100000, 80: 110000}   # scripts/p57_build_scale_data.py, SEED0[(n, 25)]
DESC_SEED0 = 20260706              # scripts/p1_build_family_data.py
MIX = 'mixed'
FACTORY_NAME = 'default'
NMCH = sum(DEFAULT_FACTORY.values())   # 25 stations
SCRATCH_MARK = '.p86_scratch'


def frozen_base(n):
    return f'data/PPVC/{n}x{NMCH}+ppvc-mixed'


def frozen_family(n):
    return f'data/FAMILY/{n}x{NMCH}'


def new_base(n):
    return f'data/PPVC/{n}x{NMCH}+ppvc-mixed-n{N_TOTAL}'


def new_family(n):
    return f'data/FAMILY/{n}x{NMCH}n{N_TOTAL}'


def stored_cache(n):
    """Best-of-four-rules warm starts of the 20-instance set (p44 --phase cache)."""
    return f'or_solution/FAMILY/warmstart_cache_{n}x{NMCH}'


def new_cache(n):
    """The same for the 30-instance set; the name p44._paths and
    p85_cpsat_arms.hint_path derive from the size tag <n>x25n30."""
    return f'or_solution/FAMILY/warmstart_cache_{n}x{NMCH}n{N_TOTAL}'


def new_xcache(n):
    """Warm starts of the 30-instance set under the exact non-delay mask
    (first_mask='exact', mask_ties='exact')."""
    return f'or_solution/FAMILY/warmstart_cache_xmask_{n}x{NMCH}n{N_TOTAL}'


FROZEN_DIRS = tuple(d for n in sorted(SEED0)
                    for d in (frozen_base(n), frozen_family(n), stored_cache(n)))
NEW_DIRS = tuple(d for n in sorted(SEED0)
                 for d in (new_base(n), new_family(n), new_cache(n), new_xcache(n)))
# Folders this script must never write into, besides the frozen sets.
NO_WRITE = ('train_log', 'trained_network', 'test_results')


def inst(i):
    return f'instance_{i:03d}'


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

_KNOWN_FLAGS = {'--sizes': True, '--scratch': True, '--force': False,
                '--rules': True, '--indices': True, '--out': True, '--from': True,
                '--resume': False, '--wait': True}
COMMANDS = ('reproduce', 'build', 'verify', 'check', 'table', 'warmstart', 'wstable',
            'wsxmask', 'wsxtable', 'wsmasked', 'wsmtable')


def parse_args(argv):
    if not argv or argv[0] not in COMMANDS:
        sys.exit(__doc__)
    cmd, opts, i = argv[0], {}, 1
    while i < len(argv):
        a = argv[i]
        if a not in _KNOWN_FLAGS:
            sys.exit(f'unknown argument {a!r}')
        if _KNOWN_FLAGS[a]:
            if i + 1 >= len(argv):
                sys.exit(f'{a} needs a value')
            opts[a] = argv[i + 1]
            i += 2
        else:
            opts[a] = True
            i += 1
    sizes = [int(x) for x in opts.get('--sizes', '50,80').split(',')]
    for n in sizes:
        if n not in SEED0:
            sys.exit(f'no frozen set for {n} modules; known sizes {sorted(SEED0)}')
    lo, hi = (int(x) for x in opts.get('--indices', f'0-{N_TOTAL - 1}').split('-'))
    return dict(cmd=cmd, sizes=sizes,
                scratch=opts.get('--scratch', 'out/p86_scratch'),
                force=bool(opts.get('--force', False)),
                resume=bool(opts.get('--resume', False)),
                wait=int(opts.get('--wait', '0')),
                rules=opts.get('--rules', 'SPT').split(','),
                indices=list(range(lo, hi + 1)),
                out=opts.get('--out'), src=opts.get('--from'))


# ---------------------------------------------------------------------------
# Write guard
# ---------------------------------------------------------------------------

def _inside(path, folder):
    p, f = os.path.realpath(path), os.path.realpath(folder)
    return p == f or p.startswith(f + os.sep)


def check_scratch_root(scratch):
    rel = os.path.relpath(os.path.realpath(scratch), ROOT)
    top = rel.split(os.sep)[0]
    if rel.startswith('..') or rel == '.' or top == 'data' or top in NO_WRITE \
            or top.startswith('paper'):
        sys.exit(f'REFUSING scratch folder {scratch!r}: it must lie inside the '
                 f'repository and outside data/, {", ".join(NO_WRITE)} and paper*/')


def assert_writable(path, scratch):
    """Only the n30 folders and the scratch folder may be written."""
    for d in FROZEN_DIRS:
        if _inside(path, d):
            raise SystemExit(f'REFUSING to write {path}: inside the read-only folder {d}')
    if any(_inside(path, d) for d in NEW_DIRS) or _inside(path, scratch):
        return
    raise SystemExit(f'REFUSING to write {path}: outside the n30 folders and the '
                     f'scratch folder')


def reset_scratch(sub, scratch):
    """Recreate a scratch subfolder. Only a folder this script created (it
    carries the marker file) is ever removed."""
    assert_writable(sub, scratch)
    if os.path.exists(sub):
        if not os.path.exists(os.path.join(sub, SCRATCH_MARK)):
            sys.exit(f'REFUSING to clear {sub}: it was not created by this script '
                     f'(no {SCRATCH_MARK} marker)')
        shutil.rmtree(sub)
    os.makedirs(sub)
    with open(os.path.join(sub, SCRATCH_MARK), 'w') as f:
        f.write('scratch output of scripts/p86_extend_scale_sets.py\n')


# ---------------------------------------------------------------------------
# Frozen-set inventory
# ---------------------------------------------------------------------------

def frozen_inventory(n):
    """Every file of the two frozen folders of size n, as relative paths."""
    files = [os.path.join(frozen_base(n), 'dataset_meta.json')]
    for i in range(N_FROZEN):
        files += [os.path.join(frozen_base(n), inst(i) + ext)
                  for ext in ('.fjs', '.meta.json')]
    files.append(os.path.join(frozen_family(n), 'dataset_meta.json'))
    for r in REGIMES_ALL:
        files += [os.path.join(frozen_family(n), 'desc', r, inst(i) + '.desc.json')
                  for i in range(N_FROZEN)]
    return files


def validate_frozen(n):
    """The frozen folders hold exactly the expected files and metadata."""
    on_disk = sorted(os.path.relpath(p) for d in (frozen_base(n), frozen_family(n))
                     for p in glob.glob(os.path.join(d, '**', '*'), recursive=True)
                     if os.path.isfile(p))
    want = sorted(frozen_inventory(n))
    if on_disk != want:
        extra = sorted(set(on_disk) - set(want))
        missing = sorted(set(want) - set(on_disk))
        sys.exit(f'frozen folders of {n} modules do not hold the expected files: '
                 f'extra {extra[:5]}, missing {missing[:5]}')
    with open(os.path.join(frozen_base(n), 'dataset_meta.json')) as f:
        bm = json.load(f)
    with open(os.path.join(frozen_family(n), 'dataset_meta.json')) as f:
        fm = json.load(f)
    expect_b = {'seed0': SEED0[n], 'n_instances': N_FROZEN, 'n_modules': n,
                'ppvc_mix': MIX, 'ppvc_factory': FACTORY_NAME}
    expect_f = {'base_dir': frozen_base(n), 'n_instances': N_FROZEN,
                'regimes': list(REGIMES_ALL), 'desc_seed0': DESC_SEED0}
    for meta, expect, path in ((bm, expect_b, frozen_base(n)),
                               (fm, expect_f, frozen_family(n))):
        for k, v in expect.items():
            if meta.get(k) != v:
                sys.exit(f'{path}/dataset_meta.json: {k} = {meta.get(k)!r}, '
                         f'expected {v!r}')
    return bm, fm


# ---------------------------------------------------------------------------
# Generation (mirrors p57 and p1 exactly)
# ---------------------------------------------------------------------------

def write_base_instance(stem, n, i):
    jl, pt, meta = ppvc_instance_generator(n_modules=n, class_mix=MIX,
                                           station_counts=DEFAULT_FACTORY,
                                           seed=SEED0[n] + i)
    save_instance(stem, jl, pt, meta)


STORED_CONVENTION = 'attached'


def load_stored_descriptor(rec):
    """Descriptor of a stored benchmark record. The generator draws attached
    setups only, and the stored records omit the convention field, so the
    loader states it (check C2 rejects an unstated convention)."""
    return ConstraintDescriptor.from_json(rec, setup_convention=STORED_CONVENTION)


def descriptor_record(stem, label, i, regime):
    """The descriptor record as p1_build_family_data.build builds it:
    to_json() of make_descriptor on the base instance read back from disk,
    then base_instance, regime and desc_seed."""
    jl, pt, meta = load_instance(stem)
    desc = make_descriptor(regime, jl, pt, meta, pt.shape[1], seed=DESC_SEED0 + i)
    d = desc.to_json()
    d['base_instance'] = label
    d['regime'] = regime
    d['desc_seed'] = DESC_SEED0 + i
    return d


def stored_form(rec):
    """A record in the form of the stored benchmark files. They were written
    before to_json() carried the setup convention, so they omit the field
    where it states the attached convention."""
    d = dict(rec)
    if d.get('setup_convention') == STORED_CONVENTION:
        del d['setup_convention']
    return d


def records_equal(a, b):
    """Field-by-field equality of two descriptor records (dicts, JSON text or
    bytes), with the setup-convention field left out where it states the
    attached convention."""
    def as_dict(x):
        return json.loads(x) if isinstance(x, (str, bytes)) else x
    return stored_form(as_dict(a)) == stored_form(as_dict(b))


def descriptor_text(stem, label, i, regime):
    """The descriptor file in the stored form, dumped without indentation as
    p1_build_family_data.build dumps it."""
    return json.dumps(stored_form(descriptor_record(stem, label, i, regime)))


def base_meta_text(n, n_inst, frozen_format):
    """dataset_meta.json of a base folder. frozen_format reproduces the key set
    of the frozen 20-instance files (written before p57 added n_stations)."""
    d = {'seed0': SEED0[n], 'n_instances': n_inst, 'n_modules': n,
         'ppvc_mix': MIX, 'ppvc_factory': FACTORY_NAME}
    if frozen_format:
        d['purpose'] = (f'{n}-module scale-up test set (eval-only): the solver at '
                        f'a literature-standard budget on a large shop')
    else:
        d['n_stations'] = NMCH
        d['purpose'] = (f'{n}-module production-size test set, {n_inst} instances '
                        f'(eval-only): instance_000 to instance_{N_FROZEN - 1:03d} '
                        f'are byte-identical to the 20-instance set '
                        f'{frozen_base(n)}; instance_{N_FROZEN:03d} to '
                        f'instance_{n_inst - 1:03d} continue its seed sequence '
                        f'(instance i uses seed seed0 + i)')
        d['extends'] = frozen_base(n)
        d['generator'] = 'scripts/p86_extend_scale_sets.py'
    return json.dumps(d, indent=1)


def family_meta_text(n, base_dir, n_inst, convention, frozen_format):
    d = {'base_dir': base_dir, 'n_instances': n_inst,
         'regimes': list(REGIMES_ALL), 'desc_seed0': DESC_SEED0,
         'convention': convention}
    if not frozen_format:
        d['extends'] = frozen_family(n)
        d['purpose'] = (f'instance_000 to instance_{N_FROZEN - 1:03d} are byte '
                        f'copies of {frozen_family(n)}, so their base_instance '
                        f'field names {frozen_base(n)}, whose files are '
                        f'byte-identical to the same instances of base_dir; '
                        f'instance_{N_FROZEN:03d} to instance_{n_inst - 1:03d} '
                        f'use descriptor seed desc_seed0 + index and name base_dir')
        d['generator'] = 'scripts/p86_extend_scale_sets.py'
    return json.dumps(d, indent=1)


def _read(path):
    with open(path, 'rb') as f:
        return f.read()


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _first_diff(a, b):
    k = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), min(len(a), len(b)))
    return k, a[max(0, k - 40):k + 40], b[max(0, k - 40):k + 40]


def _load_p1():
    spec = importlib.util.spec_from_file_location(
        'p1_build_family_data', os.path.join(ROOT, 'scripts', 'p1_build_family_data.py'))
    mod = importlib.util.module_from_spec(spec)
    saved = sys.argv
    spec.loader.exec_module(mod)
    sys.argv = saved
    return mod


def reproduce(n, scratch):
    """Regenerate instances 0..N_TOTAL-1 of size n into the scratch folder and
    compare the first N_FROZEN with the frozen files. Returns a summary dict;
    `ok` is False on any difference."""
    bm, fm = validate_frozen(n)
    root = os.path.join(scratch, f'{n}x{NMCH}')
    reset_scratch(root, scratch)
    sbase = os.path.join(root, 'base')
    sfam = os.path.join(root, 'family')
    os.makedirs(sbase)
    t0 = time.time()
    full = {}            # regenerated records as to_json() writes them today
    for i in range(N_TOTAL):
        stem = os.path.join(sbase, inst(i))
        assert_writable(stem, scratch)
        write_base_instance(stem, n, i)
        label = os.path.join(frozen_base(n) if i < N_FROZEN else new_base(n), inst(i))
        for r in REGIMES_ALL:
            p = os.path.join(sfam, 'desc', r, inst(i) + '.desc.json')
            os.makedirs(os.path.dirname(p), exist_ok=True)
            full[(r, i)] = descriptor_record(stem, label, i, r)
            with open(p, 'w') as f:
                f.write(json.dumps(stored_form(full[(r, i)])))
    with open(os.path.join(sbase, 'dataset_meta.json'), 'w') as f:
        f.write(base_meta_text(n, N_FROZEN, frozen_format=True))
    with open(os.path.join(sfam, 'dataset_meta.json'), 'w') as f:
        f.write(family_meta_text(n, frozen_base(n), N_FROZEN, fm['convention'],
                                 frozen_format=True))
    gen_s = time.time() - t0

    # Byte comparison of every frozen file with its regenerated counterpart
    # (descriptors in the stored form), and field-by-field comparison of every
    # frozen descriptor with the record to_json() writes today, the
    # setup-convention field left out.
    mism, n_cmp = [], 0
    field_mism, n_field, with_conv = [], 0, 0
    for fz in frozen_inventory(n):
        rel_b = os.path.relpath(fz, frozen_base(n))
        rel_f = os.path.relpath(fz, frozen_family(n))
        sc = (os.path.join(sbase, rel_b) if _inside(fz, frozen_base(n))
              else os.path.join(sfam, rel_f))
        a, b = _read(fz), _read(sc)
        n_cmp += 1
        if fz.endswith('.desc.json'):
            r, i = fz.split(os.sep)[-2], int(os.path.basename(fz)[9:12])
            n_field += 1
            with_conv += 'setup_convention' in full[(r, i)]
            if not records_equal(a, full[(r, i)]):
                field_mism.append(fz)
        if a != b:
            k, ca, cb = _first_diff(a, b)
            entry = {'file': fz, 'frozen_sha256': _sha(a), 'regenerated_sha256': _sha(b),
                     'first_diff_byte': k, 'frozen_context': ca.decode(errors='replace'),
                     'regenerated_context': cb.decode(errors='replace')}
            try:   # say whether the parsed content differs too
                entry['parsed_equal'] = (json.loads(a) == json.loads(b))
            except ValueError:
                entry['parsed_equal'] = None
            mism.append(entry)

    # cross-check: the unmodified p1 build() on the 30-instance scratch base
    p1 = _load_p1()
    p1_out = os.path.join(root, 'family_p1')
    assert_writable(p1_out, scratch)
    p1.build(sbase, p1_out)
    p1_mism = []
    for r in REGIMES_ALL:
        for i in range(N_TOTAL):
            with open(os.path.join(sfam, 'desc', r, inst(i) + '.desc.json')) as f:
                mine = json.load(f)
            with open(os.path.join(p1_out, 'desc', r, inst(i) + '.desc.json')) as f:
                theirs = json.load(f)
            lab_ok = theirs.pop('base_instance') == os.path.relpath(
                os.path.join(sbase, inst(i)))
            mine.pop('base_instance')
            if not records_equal(mine, theirs) or not lab_ok:
                p1_mism.append(f'{r}/{inst(i)}')
    with open(os.path.join(p1_out, 'dataset_meta.json')) as f:
        p1_meta = json.load(f)
    p1_count_ok = (p1_meta['n_instances'] == N_TOTAL
                   and p1_meta['desc_seed0'] == DESC_SEED0)

    ok = not mism and not field_mism and not p1_mism and p1_count_ok
    print(f'[{n}x{NMCH}] regenerated {N_TOTAL} base instances and '
          f'{N_TOTAL * len(REGIMES_ALL)} descriptors in {gen_s:.1f} s; '
          f'{n_cmp} frozen files compared byte for byte (descriptors in the '
          f'stored form): {n_cmp - len(mism)} identical, {len(mism)} different; '
          f'{n_field} frozen descriptors compared field by field with today\'s '
          f'to_json() record ({with_conv} of which carry setup_convention, left '
          f'out): {n_field - len(field_mism)} equal, {len(field_mism)} different; '
          f'p1 build() cross-check on {N_TOTAL * len(REGIMES_ALL)} descriptors: '
          f'{len(p1_mism)} different', flush=True)
    for m in mism[:10]:
        print('   DIFF', json.dumps(m), flush=True)
    for m in field_mism[:10]:
        print('   FIELD DIFF', m, flush=True)
    return {'size': f'{n}x{NMCH}', 'ok': ok, 'files_compared': n_cmp,
            'byte_identical': n_cmp - len(mism), 'differences': mism,
            'field_compared': n_field, 'field_differences': field_mism,
            'regenerated_with_convention_field': with_conv,
            'p1_crosscheck_differences': p1_mism, 'p1_meta_ok': p1_count_ok}


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def plan_build(n, scratch):
    """List of (destination, bytes, source) for every file of the n30 folders."""
    _, fm = validate_frozen(n)
    sbase = os.path.join(scratch, f'{n}x{NMCH}', 'base')
    sfam = os.path.join(scratch, f'{n}x{NMCH}', 'family')
    plan = []
    for i in range(N_TOTAL):
        src_b = (os.path.join(frozen_base(n), inst(i)) if i < N_FROZEN
                 else os.path.join(sbase, inst(i)))
        for ext in ('.fjs', '.meta.json'):
            plan.append((os.path.join(new_base(n), inst(i) + ext),
                         _read(src_b + ext), src_b + ext))
        for r in REGIMES_ALL:
            src_d = (os.path.join(frozen_family(n), 'desc', r, inst(i) + '.desc.json')
                     if i < N_FROZEN else
                     os.path.join(sfam, 'desc', r, inst(i) + '.desc.json'))
            plan.append((os.path.join(new_family(n), 'desc', r, inst(i) + '.desc.json'),
                         _read(src_d), src_d))
    plan.append((os.path.join(new_base(n), 'dataset_meta.json'),
                 base_meta_text(n, N_TOTAL, frozen_format=False).encode(), 'generated'))
    plan.append((os.path.join(new_family(n), 'dataset_meta.json'),
                 family_meta_text(n, new_base(n), N_TOTAL, fm['convention'],
                                  frozen_format=False).encode(), 'generated'))
    return plan


def execute_plan(plan, scratch, force):
    """Check every destination first, then write; nothing is written when any
    destination is refused."""
    to_write, same = [], 0
    for dst, data, _ in plan:
        assert_writable(dst, scratch)
        if os.path.exists(dst):
            if _read(dst) == data:
                same += 1
                continue
            if not force:
                raise SystemExit(f'REFUSING to overwrite {dst}: it exists with '
                                 f'different content (pass --force to replace it)')
        to_write.append((dst, data))
    for dst, data in to_write:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, 'wb') as f:
            f.write(data)
    return len(to_write), same


def verify_built(n, scratch):
    """Re-read the n30 folders: exact file set, 000-019 equal to the frozen
    files, 020-029 equal to a fresh regeneration in memory."""
    want = set()
    for i in range(N_TOTAL):
        want |= {os.path.join(new_base(n), inst(i) + e) for e in ('.fjs', '.meta.json')}
        want |= {os.path.join(new_family(n), 'desc', r, inst(i) + '.desc.json')
                 for r in REGIMES_ALL}
    want |= {os.path.join(new_base(n), 'dataset_meta.json'),
             os.path.join(new_family(n), 'dataset_meta.json')}
    have = {os.path.relpath(p) for d in (new_base(n), new_family(n))
            for p in glob.glob(os.path.join(d, '**', '*'), recursive=True)
            if os.path.isfile(p)}
    assert have == want, (sorted(have - want)[:5], sorted(want - have)[:5])
    bad = []
    for i in range(N_FROZEN):
        for e in ('.fjs', '.meta.json'):
            if _read(os.path.join(new_base(n), inst(i) + e)) != \
                    _read(os.path.join(frozen_base(n), inst(i) + e)):
                bad.append(f'base {inst(i)}{e}')
        for r in REGIMES_ALL:
            rel = os.path.join('desc', r, inst(i) + '.desc.json')
            if _read(os.path.join(new_family(n), rel)) != \
                    _read(os.path.join(frozen_family(n), rel)):
                bad.append(rel)
    # 020-029: regenerate in memory and compare with what was written
    from data_utils import matrix_to_text
    for i in range(N_FROZEN, N_TOTAL):
        jl, pt, meta = ppvc_instance_generator(n_modules=n, class_mix=MIX,
                                               station_counts=DEFAULT_FACTORY,
                                               seed=SEED0[n] + i)
        flex = float((pt != 0).sum(axis=1).mean())
        fjs = ''.join(line + '\n' for line in matrix_to_text(jl, pt, round(flex, 2)))
        js = json.dumps({k: (v.tolist() if isinstance(v, np.ndarray) else v)
                         for k, v in meta.items()}, indent=1)
        stem = os.path.join(new_base(n), inst(i))
        if _read(stem + '.fjs') != fjs.encode():
            bad.append(f'base {inst(i)}.fjs')
        if _read(stem + '.meta.json') != js.encode():
            bad.append(f'base {inst(i)}.meta.json')
        for r in REGIMES_ALL:
            rel = os.path.join('desc', r, inst(i) + '.desc.json')
            stored = _read(os.path.join(new_family(n), rel))
            if stored != descriptor_text(stem, stem, i, r).encode():
                bad.append(rel)
            # field by field with today's to_json() record, convention left out
            if not records_equal(stored, descriptor_record(stem, stem, i, r)):
                bad.append(rel + ' (fields)')
    print(f'[{n}x{NMCH}] verified {len(want)} files in {new_base(n)} and '
          f'{new_family(n)}: {len(bad)} problems', flush=True)
    return bad


# ---------------------------------------------------------------------------
# Check (read-only)
# ---------------------------------------------------------------------------

def rollout(rule, jl, pt, meta, desc):
    """One dispatching-rule rollout exactly as eval_family.py runs the PDR
    methods (published non-delay mask, seed configs.seed_test per instance)."""
    from constraint_family.fjsp_env_family import FJSPEnvFamily, resolve_delta_mode
    from common_utils import heuristic_select_action
    env = FJSPEnvFamily(jl.shape[0], pt.shape[1], family_mode=True,
                        blind_channels=(), mask_mode='nondelay', pair_extra='none')
    dm = resolve_delta_mode(configs)
    env.delta_mode = dm
    env.reward_delta_env = (dm != 'zero')
    env.set_initial_data([jl], [pt], descriptor_list=[desc],
                         op_type_list=[meta['op_type']],
                         mch_type_list=[meta['mch_type']])
    amch = np.full(pt.shape[0], -1, dtype=int)
    np.random.seed(configs.seed_test)
    while not env.done().all():
        a = heuristic_select_action(rule, env)
        job = a // pt.shape[1]
        amch[env.candidate[0, job]] = a % pt.shape[1]
        env.step(np.array([a]))
    return float(env.current_makespan[0]), amch, np.asarray(env.true_op_ct[0], float)


def instance_stats(jl, pt, meta, descs):
    """Summary statistics of one base instance and its descriptors."""
    cls = np.bincount(np.asarray(meta['routing_class'], int), minlength=len(ROUTING_CLASSES))
    lag = np.asarray(meta['time_lag'], float)
    s = {'mean_pt_h': float(pt[pt > 0].mean()),
         'total_min_pt_h': float(np.where(pt > 0, pt, np.inf).min(axis=1).sum()),
         'n_lag_ops': int((lag > 0).sum()),
         'total_lag_h': float(lag.sum()),
         'horizon_h': int(descs['LSW'].horizon)}
    for c, name in enumerate(ROUTING_CLASSES):
        s[f'n_{name}'] = int(cls[c])
    S = np.asarray(descs['S'].setup, float)
    for r in ('LS', 'SW', 'LSW'):
        assert np.array_equal(np.asarray(descs[r].setup, float), S), r
    off = S[:4][~np.eye(4, dtype=bool)]
    s.update(setup_min_h=float(S.min()), setup_max_h=float(S.max()),
             setup_cross_mean_h=float(off.mean()))
    W = descs['W'].windows
    for r in ('LW', 'SW', 'LSW'):
        assert descs[r].windows == W, r
    durs = [e - st for w in W for (st, e) in w]
    s.update(n_windows=len(durs), window_total_h=float(sum(durs)))
    return s


def check(sizes, rules, indices, out):
    from constraint_family import checks
    report = {'rules': rules, 'indices': indices, 'sizes': {}}
    for n in sizes:
        fam, base = new_family(n), new_base(n)
        with open(os.path.join(fam, 'dataset_meta.json')) as f:
            fm = json.load(f)
        assert fm['base_dir'] == base and fm['n_instances'] == N_TOTAL
        per = {}
        status_counts = {}
        t0 = time.time()
        for i in indices:
            name = inst(i)
            jl, pt, meta = load_instance(os.path.join(base, name))
            descs, recs = {}, {}
            for r in REGIMES_ALL:
                with open(os.path.join(fam, 'desc', r, name + '.desc.json')) as f:
                    recs[r] = json.load(f)
                descs[r] = load_stored_descriptor(recs[r])
                # the base_instance field must name a file equal to the base used
                lab = recs[r]['base_instance']
                for e in ('.fjs', '.meta.json'):
                    assert _read(lab + e) == _read(os.path.join(base, name) + e), (lab, e)
                assert recs[r]['desc_seed'] == DESC_SEED0 + i
            row = instance_stats(jl, pt, meta, descs)
            row['checks'] = {}
            for r in REGIMES_ALL:
                if rules == ['none']:
                    # configuration checks only; C6 is deferred (MANUAL)
                    rep = checks.run_all(descs[r], pt, False, job_length=jl)
                    st = {c.id: c.status for c in rep}
                    row['checks'][f'{r}/none'] = st
                    key = tuple(sorted(st.items()))
                    status_counts[key] = status_counts.get(key, 0) + 1
                    if any(st[c] == checks.FAIL for c in st) or st['C4'] != checks.PASS:
                        print(f'   {n}x{NMCH} {name} {r}: {checks.format_report(rep)}',
                              flush=True)
                    continue
                ms_rule = {}
                for rule in rules:
                    ms, amch, ct = rollout(rule, jl, pt, meta, descs[r])
                    rep = checks.run_all(descs[r], pt, False, job_length=jl,
                                         schedule={'op_mch': amch, 'op_ct': ct})
                    st = {c.id: c.status for c in rep}
                    c6 = next(c for c in rep if c.id == 'C6')
                    assert abs(c6.details['makespan'] - ms) < 1e-6
                    ms_rule[rule] = ms
                    row['checks'][f'{r}/{rule}'] = st
                    key = tuple(sorted(st.items()))
                    status_counts[key] = status_counts.get(key, 0) + 1
                    if st['C6'] != checks.PASS or st['C1'] != checks.PASS:
                        print(f'   {n}x{NMCH} {name} {r} {rule}: '
                              f'{checks.format_report(rep)}', flush=True)
                for rule, ms in ms_rule.items():
                    row[f'{rule}_{r}'] = ms
            per[name] = row
            print(f'[{n}x{NMCH}] {name} done ({(time.time() - t0) / 60:.1f} min)',
                  flush=True)
        # agreement with the published rule results on the frozen prefix (read only)
        published = {}
        for rule in (r_ for r_ in rules if r_ != 'none'):
            for r in REGIMES_ALL:
                p = f'test_results/FAMILY/{n}x{NMCH}/Result_{rule}_{r}.npy'
                if not os.path.exists(p):
                    continue
                pub = np.load(p)
                pub = pub[:, 0] if pub.ndim > 1 else pub
                idx = [i for i in indices if i < min(N_FROZEN, len(pub))]
                diffs = [abs(per[inst(i)][f'{rule}_{r}'] - float(pub[i])) for i in idx]
                published[f'{rule}_{r}'] = {'n': len(idx),
                                            'max_abs_diff': max(diffs) if diffs else None}
        report['sizes'][f'{n}x{NMCH}'] = {
            'per_instance': per,
            'status_combinations': [{'statuses': dict(k), 'count': v}
                                    for k, v in sorted(status_counts.items())],
            'published_rule_agreement': published}
    if out:
        for d in FROZEN_DIRS + NO_WRITE + ('data',):
            if _inside(out, d):
                sys.exit(f'REFUSING to write {out} inside {d}')
        with open(out, 'w') as f:
            json.dump(report, f, indent=1)
        print(f'wrote {out}', flush=True)
    return report


TABLE_ROWS = (
    ('mean_pt_h', 'Mean compatible processing time (h)', 2),
    ('n_RC-wet', 'Modules on route RC-wet', 0),
    ('n_RC-dry', 'Modules on route RC-dry', 0),
    ('n_Steel-wet', 'Modules on route Steel-wet', 0),
    ('n_Steel-dry', 'Modules on route Steel-dry', 0),
    ('n_lag_ops', 'Operations with a time lag', 0),
    ('total_lag_h', 'Total time lag (h)', 0),
    ('horizon_h', 'Job-chain horizon H (h)', 0),
    ('setup_min_h', 'Smallest setup entry (h)', 0),
    ('setup_max_h', 'Largest setup entry (h)', 0),
    ('setup_cross_mean_h', 'Mean cross-class setup (h)', 2),
    ('n_windows', 'Outage windows (count)', 0),
    ('window_total_h', 'Total outage length (h)', 0),
) + tuple((f'SPT_{r}', f'SPT makespan, regime {r} (h)', 0) for r in REGIMES_ALL)


def table(src):
    """Markdown table: range of each statistic over instances 000-019 and over
    020-029, and how many of 020-029 fall outside the 000-019 range. Reads the
    JSON record written by `check`."""
    with open(src) as f:
        rep = json.load(f)
    lines = []
    for size, blk in rep['sizes'].items():
        per = blk['per_instance']
        old = [per[inst(i)] for i in range(N_FROZEN) if inst(i) in per]
        new = [per[inst(i)] for i in range(N_FROZEN, N_TOTAL) if inst(i) in per]
        if len(old) != N_FROZEN or len(new) != N_TOTAL - N_FROZEN:
            sys.exit(f'{src}: {size} needs instances 000-{N_TOTAL - 1:03d}')
        lines += [f'\n{size} (instances 000-019: n = {len(old)}; 020-029: n = {len(new)})\n',
                  '| Statistic | 000-019 min to max (mean) | 020-029 min to max (mean) '
                  '| 020-029 outside the 000-019 range |',
                  '|---|---|---|---|']
        for key, label, dig in TABLE_ROWS:
            if key not in old[0]:
                continue
            a = np.array([r[key] for r in old], float)
            b = np.array([r[key] for r in new], float)
            out = [f'{v:.{dig}f}' for v in b if v < a.min() or v > a.max()]
            lines.append(f'| {label} | {a.min():.{dig}f} to {a.max():.{dig}f} '
                         f'({a.mean():.{dig}f}) | {b.min():.{dig}f} to {b.max():.{dig}f} '
                         f'({b.mean():.{dig}f}) | {len(out)}'
                         f'{" (" + ", ".join(out) + ")" if out else ""} |')
    print('\n'.join(lines))


# ---------------------------------------------------------------------------
# Warm-start caches (best of four dispatching rules, scripts/p12_cpsat_fairwarm.py)
# ---------------------------------------------------------------------------

WS_KEYS = ('amch', 'ct', 'ms', 'rule', 'rule_names', 'rule_ms')


def _npz_members(path):
    """{member name: raw .npy bytes} of an .npz archive. Two archives written
    at different times differ in their zip timestamps; the members hold the
    arrays with dtype and shape, so equal members mean equal content."""
    import zipfile
    with zipfile.ZipFile(path) as z:
        return {i.filename: z.read(i.filename) for i in z.infolist()}


def _ws_compare(stored, regen):
    """Field-by-field comparison of two warm-start archives."""
    a = np.load(stored, allow_pickle=True)
    b = np.load(regen, allow_pickle=True)
    diff = []
    if sorted(a.files) != sorted(b.files):
        diff.append(f'keys {sorted(a.files)} vs {sorted(b.files)}')
    for k in WS_KEYS:
        if k not in a.files or k not in b.files:
            continue
        x, y = a[k], b[k]
        if x.dtype != y.dtype or x.shape != y.shape or not np.array_equal(x, y):
            if k == 'ct' and x.shape == y.shape:
                diff.append(f'ct: {int((x != y).sum())} entries differ, max '
                            f'{float(np.abs(x.astype(float) - y.astype(float)).max()):g}')
            elif k == 'amch' and x.shape == y.shape:
                diff.append(f'amch: {int((x != y).sum())} entries differ')
            else:
                diff.append(f'{k}: {x.tolist()} vs {y.tolist()}')
    members_equal = _npz_members(stored) == _npz_members(regen)
    return diff, members_equal


def _fairwarm():
    """scripts/p12_cpsat_fairwarm.py as a module (it strips sys.argv itself)."""
    sys.path.insert(0, os.path.join(ROOT, 'scripts'))
    import p12_cpsat_fairwarm as fw
    return fw


def _validate_stored_cache(n):
    want = sorted(os.path.join(stored_cache(n), r, inst(i) + '.npz')
                  for r in REGIMES_ALL for i in range(N_FROZEN))
    have = sorted(os.path.relpath(p) for p in
                  glob.glob(os.path.join(stored_cache(n), '**', '*'), recursive=True)
                  if os.path.isfile(p))
    if have != want:
        sys.exit(f'{stored_cache(n)} does not hold exactly 8 x {N_FROZEN} warm '
                 f'starts: extra {sorted(set(have) - set(want))[:5]}, missing '
                 f'{sorted(set(want) - set(have))[:5]}')


def warmstart(sizes, scratch, force, resume):
    """Reproduce the stored warm-start caches for 000-019 with the unmodified
    p12._warm_start (the procedure p44 --phase cache calls), and only if every
    archive reproduces, build 020-029 with the same procedure, validate every
    schedule with cpsat_arms.validate_all and write the 30-instance caches.

    The rule schedules are additionally re-checked with C6 under the current
    validators: p12's validate_schedule is wrapped so that it returns exactly
    what it returns unwrapped, and the C6 status of every candidate schedule
    is recorded in the scratch folder."""
    from constraint_family import checks
    from constraint_family import cpsat_arms as A
    fw = _fairwarm()
    root = os.path.join(scratch, 'warmstart')
    log_path = os.path.join(root, 'rule_schedules.jsonl')
    if resume and os.path.exists(os.path.join(root, SCRATCH_MARK)):
        print(f'resuming in {root}', flush=True)
    else:
        reset_scratch(root, scratch)

    original = fw.validate_schedule
    current = {}

    def recording_validate(jl, pt, desc, st, ct, amch, *args, **kw):
        v = original(jl, pt, desc, st, ct, amch, *args, **kw)
        c6 = checks.check_schedule_validation(desc, jl, pt, amch, ct, op_st=st)
        k = current['calls']
        current['calls'] += 1
        rec = dict(current['key'], rule=fw.RULES[k], ms=float(np.max(ct)),
                   reference_violations=len(v), c6=c6.status,
                   c6_first=(c6.details['violations'][:1] or [None])[0])
        with open(log_path, 'a') as f:
            f.write(json.dumps(rec) + '\n')
        return v

    fw.validate_schedule = recording_validate

    def one(n, i, regime, base_dir, fam_dir, tag):
        fw.WARM_CACHE = os.path.join(root, tag)
        assert_writable(fw.WARM_CACHE, scratch)
        name = inst(i)
        stem = os.path.join(base_dir, name)
        jl, pt, _ = load_instance(stem)
        with open(os.path.join(fam_dir, 'desc', regime, name + '.desc.json')) as f:
            desc = load_stored_descriptor(json.load(f))
        current.update(calls=0, key={'size': tag, 'regime': regime, 'instance': name})
        fw._warm_start(stem, jl, pt, desc, regime, name)
        return os.path.join(fw.WARM_CACHE, regime, name + '.npz')

    # 1. reproduction of 000-019, both sizes, before anything is built
    table_rows, failed = [], False
    for n in sizes:
        validate_frozen(n)
        _validate_stored_cache(n)
        t0 = time.time()
        same = members = 0
        diffs = []
        for i in range(N_FROZEN):
            for r in REGIMES_ALL:
                p = one(n, i, r, frozen_base(n), frozen_family(n), f'{n}x{NMCH}')
                d, mem = _ws_compare(os.path.join(stored_cache(n), r, inst(i) + '.npz'), p)
                same += not d
                members += mem
                if d:
                    diffs.append(f'{r}/{inst(i)}: ' + '; '.join(d))
            print(f'[{n}x{NMCH}] warm starts {inst(i)} reproduced '
                  f'({(time.time() - t0) / 60:.1f} min)', flush=True)
        total = N_FROZEN * len(REGIMES_ALL)
        table_rows.append((f'{n}x{NMCH}', total, same, total - same, members, diffs))
        print(f'[{n}x{NMCH}] {total} stored warm starts compared: {same} identical '
              f'in every field, {total - same} different; archive members '
              f'byte-identical: {members}', flush=True)
        for d in diffs[:20]:
            print('   DIFF', d, flush=True)
        failed |= bool(diffs)
    if failed:
        print('REPRODUCTION FAILED: nothing was written to the 30-instance caches',
              flush=True)
        sys.exit(1)

    # 2. 020-029 with the same procedure, validated, then the caches written
    for n in sizes:
        tag = f'{n}x{NMCH}n{N_TOTAL}'
        t0 = time.time()
        built = {}
        for i in range(N_FROZEN, N_TOTAL):
            for r in REGIMES_ALL:
                built[(r, i)] = one(n, i, r, new_base(n), new_family(n), tag)
            print(f'[{tag}] warm starts {inst(i)} built ({(time.time() - t0) / 60:.1f} min)',
                  flush=True)
        bad = []
        for i in range(N_TOTAL):
            for r in REGIMES_ALL:
                src = (os.path.join(stored_cache(n), r, inst(i) + '.npz') if i < N_FROZEN
                       else built[(r, i)])
                case = A.load_case(tag, r, inst(i))
                (amch, st, ct), _ = A.load_hint_npz(src, case)
                v = A.validate_all(case, amch, st, ct)
                if v:
                    bad.append(f'{r}/{inst(i)}: {v[0]}')
        print(f'[{tag}] validate_all on {N_TOTAL * len(REGIMES_ALL)} warm starts '
              f'(000-019 stored, 020-029 new): {len(bad)} invalid', flush=True)
        if any(int(b.split('instance_')[1][:3]) >= N_FROZEN for b in bad):
            print(f'VALIDATION FAILED for new warm starts: {bad[:10]}', flush=True)
            sys.exit(1)
        for b in bad:
            print('   stored warm start invalid under the current checks:', b, flush=True)
        plan = []
        for i in range(N_TOTAL):
            for r in REGIMES_ALL:
                src = (os.path.join(stored_cache(n), r, inst(i) + '.npz') if i < N_FROZEN
                       else built[(r, i)])
                plan.append((os.path.join(new_cache(n), r, inst(i) + '.npz'), _read(src), src))
        wrote, same = execute_plan(plan, scratch, force)
        for i in range(N_FROZEN):
            for r in REGIMES_ALL:
                assert _read(os.path.join(new_cache(n), r, inst(i) + '.npz')) == \
                    _read(os.path.join(stored_cache(n), r, inst(i) + '.npz'))
        print(f'[{tag}] {new_cache(n)}: wrote {wrote} files, {same} already present '
              f'and identical; 000-019 byte-identical to {stored_cache(n)}', flush=True)
    fw.validate_schedule = original
    return table_rows


WS_PROCEDURE = ('scripts/p12_cpsat_fairwarm.py _warm_start, unmodified: FIFO, MOR, SPT '
                'and MWKR rolled out through FJSPEnvFamily (numpy seed 50 per '
                'rollout), each schedule checked with '
                'constraint_family.reference_sim.validate_schedule, the shortest '
                'valid one kept')


def cache_meta_text(n, mask):
    d = {'set': new_family(n), 'base_dir': new_base(n), 'n_instances': N_TOTAL,
         'regimes': list(REGIMES_ALL), 'first_mask': mask, 'mask_ties': mask,
         'procedure': WS_PROCEDURE, 'generator': 'scripts/p86_extend_scale_sets.py'}
    if mask == 'legacy':
        d['mask_source'] = ('p12._rollout passes no mask option and PACT_FIRST_MASK, '
                            'PACT_MASK_TOL were unset in the builder process')
        d['purpose'] = (f'instance_000 to instance_{N_FROZEN - 1:03d} are byte '
                        f'copies of {stored_cache(n)}; instance_{N_FROZEN:03d} to '
                        f'instance_{N_TOTAL - 1:03d} are built by the same procedure')
    else:
        d['mask_source'] = ('PACT_FIRST_MASK=1 and PACT_MASK_TOL=1 in the builder '
                            'process; env.mask_setting read exact for both options '
                            'in every rule rollout')
        d['purpose'] = 'all instances built under the exact non-delay mask'
    return json.dumps(d, indent=1)


def warmstart_xmask(sizes, scratch, force, resume):
    """Warm starts of all 30 instances under the exact non-delay mask, built
    by the unmodified p12._warm_start with PACT_FIRST_MASK=1 and
    PACT_MASK_TOL=1 set in this process. p12._rollout creates the
    environment without mask arguments, so env.mask_setting reads the
    variables at call time, in set_initial_data, at every step and in the
    readiness test of common_utils.heuristic_select_action. That the setting
    is in force is checked in every rollout. Every schedule is validated with
    cpsat_arms.validate_all before anything is written. Also writes a
    dataset_meta.json with the mask settings into both 30-instance caches."""
    from constraint_family import checks
    from constraint_family import cpsat_arms as A
    from constraint_family.fjsp_env_family import MASK_ENV_VARS, mask_option_from_env
    import common_utils
    for n in sizes:
        want = [os.path.join(new_cache(n), r, inst(i) + '.npz')
                for r in REGIMES_ALL for i in range(N_TOTAL)]
        missing = [p for p in want if not os.path.exists(p)]
        if missing:
            sys.exit(f'build the legacy-mask caches first (warmstart): {missing[:3]}')
    for name, var in MASK_ENV_VARS.items():
        if mask_option_from_env(name) == 'legacy':
            sys.exit(f'{var} asks for the legacy mask in this process; unset it')
    os.environ['PACT_FIRST_MASK'] = '1'
    os.environ['PACT_MASK_TOL'] = '1'
    fw = _fairwarm()
    root = os.path.join(scratch, 'warmstart_xmask')
    log_path = os.path.join(root, 'rule_schedules.jsonl')
    if resume and os.path.exists(os.path.join(root, SCRATCH_MARK)):
        print(f'resuming in {root}', flush=True)
    else:
        reset_scratch(root, scratch)

    seen, last_env = [], {'env': None}
    original_rule = common_utils.heuristic_select_action

    def recording_rule(method, env):
        # first call of each rollout: the settings the environment is using
        if last_env['env'] is not env:
            last_env['env'] = env
            seen.append((env.mask_setting('first_mask'), env.mask_setting('mask_ties'),
                         bool(getattr(env, '_first_mask', False))))
        return original_rule(method, env)

    original_validate = fw.validate_schedule
    current = {}

    def recording_validate(jl, pt, desc, st, ct, amch, *args, **kw):
        v = original_validate(jl, pt, desc, st, ct, amch, *args, **kw)
        c6 = checks.check_schedule_validation(desc, jl, pt, amch, ct, op_st=st)
        k = current['calls']
        current['calls'] += 1
        rec = dict(current['key'], rule=fw.RULES[k], ms=float(np.max(ct)),
                   first_mask='exact', mask_ties='exact',
                   reference_violations=len(v), c6=c6.status,
                   c6_first=(c6.details['violations'][:1] or [None])[0])
        with open(log_path, 'a') as f:
            f.write(json.dumps(rec) + '\n')
        return v

    common_utils.heuristic_select_action = recording_rule
    fw.validate_schedule = recording_validate
    try:
        for n in sizes:
            tag = f'{n}x{NMCH}n{N_TOTAL}'
            t0 = time.time()
            built, bad = {}, []
            for i in range(N_TOTAL):
                for r in REGIMES_ALL:
                    fw.WARM_CACHE = os.path.join(root, tag)
                    assert_writable(fw.WARM_CACHE, scratch)
                    name = inst(i)
                    stem = os.path.join(new_base(n), name)
                    jl, pt, _ = load_instance(stem)
                    with open(os.path.join(new_family(n), 'desc', r,
                                           name + '.desc.json')) as f:
                        desc = load_stored_descriptor(json.load(f))
                    current.update(calls=0, key={'size': tag, 'regime': r,
                                                 'instance': name})
                    k0 = len(seen)
                    fw._warm_start(stem, jl, pt, desc, r, name)
                    got = seen[k0:]
                    if len(got) not in (0, len(fw.RULES)) or \
                            any(s != ('exact', 'exact', True) for s in got):
                        sys.exit(f'{tag} {r}/{name}: rollouts did not run under the '
                                 f'exact mask: {got}')
                    built[(r, i)] = os.path.join(fw.WARM_CACHE, r, name + '.npz')
                print(f'[{tag} exact mask] warm starts {inst(i)} built '
                      f'({(time.time() - t0) / 60:.1f} min)', flush=True)
            for (r, i), p in sorted(built.items()):
                case = A.load_case(tag, r, inst(i))
                (amch, st, ct), _ = A.load_hint_npz(p, case)
                v = A.validate_all(case, amch, st, ct)
                if v:
                    bad.append(f'{r}/{inst(i)}: {v[0]}')
            print(f'[{tag} exact mask] validate_all on {len(built)} warm starts: '
                  f'{len(bad)} invalid', flush=True)
            if bad:
                sys.exit(f'VALIDATION FAILED, nothing written: {bad[:10]}')
            plan = [(os.path.join(new_xcache(n), r, inst(i) + '.npz'), _read(p), p)
                    for (r, i), p in sorted(built.items())]
            plan.append((os.path.join(new_xcache(n), 'dataset_meta.json'),
                         cache_meta_text(n, 'exact').encode(), 'generated'))
            plan.append((os.path.join(new_cache(n), 'dataset_meta.json'),
                         cache_meta_text(n, 'legacy').encode(), 'generated'))
            wrote, same = execute_plan(plan, scratch, force)
            print(f'[{tag} exact mask] {new_xcache(n)}: wrote {wrote} files, {same} '
                  f'already present and identical', flush=True)
    finally:
        common_utils.heuristic_select_action = original_rule
        fw.validate_schedule = original_validate
    print(f'mask settings checked in {len(seen)} rule rollouts', flush=True)


def xmask_table(sizes):
    """Legacy against exact-mask warm starts per regime, same 30 instances."""
    lines = []
    for n in sizes:
        lines += [f'\n{n}x{NMCH}n{N_TOTAL} (30 instances per regime)\n',
                  '| Regime | Legacy mask, mean makespan | Exact mask, mean makespan '
                  '| Chosen rule differs | Exact longer | Equal | Exact shorter |',
                  '|---|---|---|---|---|---|---|']
        tot = np.zeros(4, int)
        for r in REGIMES_ALL:
            leg, ex, rd = [], [], 0
            for i in range(N_TOTAL):
                a = np.load(os.path.join(new_cache(n), r, inst(i) + '.npz'), allow_pickle=True)
                b = np.load(os.path.join(new_xcache(n), r, inst(i) + '.npz'), allow_pickle=True)
                leg.append(float(a['ms']))
                ex.append(float(b['ms']))
                rd += str(a['rule']) != str(b['rule'])
            leg, ex = np.array(leg), np.array(ex)
            c = np.array([rd, (ex > leg).sum(), (ex == leg).sum(), (ex < leg).sum()])
            tot += c
            lines.append(f'| {r} | {leg.mean():.1f} | {ex.mean():.1f} | {c[0]} | {c[1]} '
                         f'| {c[2]} | {c[3]} |')
        lines.append(f'| all | | | {tot[0]} | {tot[1]} | {tot[2]} | {tot[3]} |')
    print('\n'.join(lines))


RULE_ORDER = ('FIFO', 'MOR', 'SPT', 'MWKR')    # p12_cpsat_fairwarm.RULES


def masked_rollout(rule, jl, pt, desc):
    """p12._rollout with common_utils.masked_rule_action in place of
    heuristic_select_action, under the exact mask stated as constructor
    options: the same environment construction (no type lists), the same
    seeding (numpy seed 50 right before the rollout) and the same read-out."""
    from constraint_family.fjsp_env_family import FJSPEnvFamily
    from common_utils import masked_rule_action
    env = FJSPEnvFamily(jl.shape[0], pt.shape[1], family_mode=True,
                        first_mask='exact', mask_ties='exact')
    env.set_initial_data([jl], [pt], descriptor_list=[desc])
    if not (env.mask_setting('first_mask') == 'exact'
            and env.mask_setting('mask_ties') == 'exact' and env._first_mask):
        raise RuntimeError('environment is not under the exact mask')
    np.random.seed(50)
    amch = np.full(pt.shape[0], -1, dtype=int)
    while not env.done().all():
        a = masked_rule_action(rule, env)
        job, mch = a // pt.shape[1], a % pt.shape[1]
        amch[env.candidate[0, job]] = mch
        env.step(np.array([a]))
    ct = np.asarray(env.true_op_ct[0], dtype=float).copy()
    return amch, ct, float(np.max(ct))


def _eligible(jl, pt, desc, amch, ct):
    """p12's eligibility test (reference validator) and the C6 status."""
    from constraint_family.reference_sim import validate_schedule
    from constraint_family import checks
    st = np.array([ct[i] - pt[i, amch[i]] if amch[i] >= 0 else -1.0
                   for i in range(pt.shape[0])])
    v = validate_schedule(jl, pt, desc, st, ct, amch)
    c6 = checks.check_schedule_validation(desc, jl, pt, amch, ct, op_st=st)
    return not v, c6.status


def masked_meta_text(n):
    d = {'set': new_family(n), 'base_dir': new_base(n), 'n_instances': N_TOTAL,
         'regimes': list(REGIMES_ALL), 'rule_set': 'masked',
         'first_mask': 'exact', 'mask_ties': 'exact',
         'rules': list(RULE_ORDER),
         'procedure': ('best of FIFO, MOR, SPT and MWKR, each rolled out with '
                       'common_utils.masked_rule_action through FJSPEnvFamily '
                       "(first_mask='exact', mask_ties='exact'; numpy seed 50 "
                       'right before each rollout, as in '
                       'scripts/p12_cpsat_fairwarm.py _rollout) and checked with '
                       'constraint_family.reference_sim.validate_schedule; the '
                       'SPT schedules are those of the exact-mask build with '
                       'heuristic_select_action, whose SPT is the same rule as '
                       'masked SPT (identical schedules checked on instance_000 '
                       'and instance_029 of every combination)'),
         'tie_rule': ('the shortest valid makespan; equal makespans go to the '
                      'earlier rule in the order FIFO, MOR, SPT, MWKR (the rule '
                      'of p12_cpsat_fairwarm._warm_start)'),
         'files': ('<regime>/<instance>.npz with amch, ct, ms, rule, rule_names, '
                   'rule_ms (the format of p12_cpsat_fairwarm._warm_start); '
                   'manifest.json lists the chosen rule and the four makespans'),
         'generator': 'scripts/p86_extend_scale_sets.py wsmasked'}
    return json.dumps(d, indent=1)


def warmstart_masked(sizes, scratch, resume, wait_s):
    """Best-of-four warm starts under the exact mask with the masked rule set.

    Phase A rolls out masked FIFO, MOR and MWKR on every instance and
    combination (scratch, resumable with --resume). Phase B, per size, waits
    until the exact-mask build of `wsxmask` has written that size's cache,
    takes its SPT schedules (a masked SPT rollout where SPT was not the chosen
    rule there), checks on instance_000 and instance_029 that masked SPT gives
    the identical schedule, takes the best of four, validates every schedule
    with cpsat_arms.validate_all and replaces the cache with a manifest and a
    dataset_meta.json. Nothing of the classical FIFO, MOR and MWKR stays in
    the cache."""
    from constraint_family import cpsat_arms as A
    root = os.path.join(scratch, 'warmstart_masked')
    if resume and os.path.exists(os.path.join(root, SCRATCH_MARK)):
        print(f'resuming in {root}', flush=True)
    else:
        reset_scratch(root, scratch)
    log_path = os.path.join(root, 'rule_schedules.jsonl')

    def load_cell(n, r, i):
        stem = os.path.join(new_base(n), inst(i))
        jl, pt, _ = load_instance(stem)
        with open(os.path.join(new_family(n), 'desc', r, inst(i) + '.desc.json')) as f:
            desc = load_stored_descriptor(json.load(f))
        return jl, pt, desc

    def run_rule(n, r, i, rule, jl, pt, desc):
        tag = f'{n}x{NMCH}n{N_TOTAL}'
        p = os.path.join(root, tag, rule, r, inst(i) + '.npz')
        if os.path.exists(p):
            z = np.load(p)
            return z['amch'], z['ct'], float(z['ms']), bool(z['valid'])
        assert_writable(p, scratch)
        amch, ct, ms = masked_rollout(rule, jl, pt, desc)
        ok, c6 = _eligible(jl, pt, desc, amch, ct)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        np.savez(p, amch=amch, ct=ct, ms=ms, valid=ok)
        with open(log_path, 'a') as f:
            f.write(json.dumps({'size': tag, 'regime': r, 'instance': inst(i),
                                'rule': rule, 'rule_set': 'masked', 'ms': ms,
                                'reference_valid': ok, 'c6': c6}) + '\n')
        return amch, ct, ms, ok

    # phase A
    for n in sizes:
        t0 = time.time()
        for i in range(N_TOTAL):
            for r in REGIMES_ALL:
                jl, pt, desc = load_cell(n, r, i)
                for rule in ('FIFO', 'MOR', 'MWKR'):
                    run_rule(n, r, i, rule, jl, pt, desc)
            print(f'[{n}x{NMCH}n{N_TOTAL} masked] FIFO, MOR, MWKR {inst(i)} done '
                  f'({(time.time() - t0) / 60:.1f} min)', flush=True)

    # phase B
    for n in sizes:
        tag = f'{n}x{NMCH}n{N_TOTAL}'
        prev = os.path.join(scratch, 'warmstart_xmask', tag)
        t_wait = time.time()
        while True:
            have = len(glob.glob(os.path.join(new_xcache(n), '*', '*.npz')))
            meta_ok = os.path.exists(os.path.join(new_xcache(n), 'dataset_meta.json'))
            if have == N_TOTAL * len(REGIMES_ALL) and meta_ok:
                break
            if time.time() - t_wait > wait_s:
                sys.exit(f'{new_xcache(n)} is not complete ({have} files) after '
                         f'{wait_s} s; run again with --resume when wsxmask has ended')
            time.sleep(60)
        best_dir = os.path.join(root, tag, 'best')
        manifest, plan, bad, spt_same, spt_rerun = {}, [], [], 0, 0
        for r in REGIMES_ALL:
            for i in range(N_TOTAL):
                name = inst(i)
                jl, pt, desc = load_cell(n, r, i)
                kept = np.load(os.path.join(prev, r, name + '.npz'), allow_pickle=True)
                kept_ms = {str(k): float(v) for k, v in zip(kept['rule_names'], kept['rule_ms'])}
                if str(kept['rule']) == 'SPT':
                    spt = (np.asarray(kept['amch']), np.asarray(kept['ct'], float),
                           float(kept['ms']), True)
                else:
                    spt = run_rule(n, r, i, 'SPT', jl, pt, desc)
                    spt_rerun += 1
                if abs(spt[2] - kept_ms['SPT']) > 1e-9:
                    sys.exit(f'{tag} {r}/{name}: SPT makespan {spt[2]} differs from '
                             f'the exact-mask build ({kept_ms["SPT"]})')
                if i in (0, N_TOTAL - 1) and str(kept['rule']) == 'SPT':
                    a2, c2, m2 = masked_rollout('SPT', jl, pt, desc)
                    if not (np.array_equal(a2, spt[0]) and np.array_equal(c2, spt[1])):
                        sys.exit(f'{tag} {r}/{name}: masked SPT differs from the kept SPT')
                    spt_same += 1
                cand = {'SPT': spt}
                for rule in ('FIFO', 'MOR', 'MWKR'):
                    cand[rule] = run_rule(n, r, i, rule, jl, pt, desc)
                rule_ms = [cand[k][2] if cand[k][3] else float('inf') for k in RULE_ORDER]
                k_best = int(np.argmin(rule_ms))          # first minimum: p12's tie rule
                best = RULE_ORDER[k_best]
                if not np.isfinite(rule_ms[k_best]):
                    sys.exit(f'{tag} {r}/{name}: no valid masked rule schedule')
                p = os.path.join(best_dir, r, name + '.npz')
                assert_writable(p, scratch)
                os.makedirs(os.path.dirname(p), exist_ok=True)
                np.savez(p, amch=cand[best][0], ct=cand[best][1], ms=rule_ms[k_best],
                         rule=best, rule_names=list(RULE_ORDER), rule_ms=rule_ms)
                case = A.load_case(tag, r, name)
                (amch, st, ct), _ = A.load_hint_npz(p, case)
                v = A.validate_all(case, amch, st, ct)
                if v:
                    bad.append(f'{r}/{name}: {v[0]}')
                manifest[f'{r}/{name}'] = {'rule': best, 'ms': rule_ms[k_best],
                                           'rule_ms': dict(zip(RULE_ORDER, rule_ms))}
                plan.append((os.path.join(new_xcache(n), r, name + '.npz'), _read(p), p))
        print(f'[{tag} masked] validate_all on {len(plan)} warm starts: {len(bad)} '
              f'invalid; SPT kept from the exact-mask build on '
              f'{len(plan) - spt_rerun} cells, rolled out on {spt_rerun}; masked SPT '
              f'identical to the kept SPT on {spt_same} checked cells', flush=True)
        if bad:
            sys.exit(f'VALIDATION FAILED, nothing written: {bad[:10]}')
        plan.append((os.path.join(new_xcache(n), 'manifest.json'),
                     json.dumps(manifest, indent=1).encode(), 'generated'))
        plan.append((os.path.join(new_xcache(n), 'dataset_meta.json'),
                     masked_meta_text(n).encode(), 'generated'))
        # replaces the classical-rule files of the exact-mask build
        wrote, same = execute_plan(plan, scratch, force=True)
        files = sorted(os.path.relpath(q) for q in glob.glob(
            os.path.join(new_xcache(n), '**', '*'), recursive=True) if os.path.isfile(q))
        assert len(files) == N_TOTAL * len(REGIMES_ALL) + 2, len(files)
        for dst, data, _ in plan:
            assert _read(dst) == data, dst
        print(f'[{tag} masked] {new_xcache(n)}: wrote {wrote} files, {same} identical; '
              f'{len(files)} files in the folder', flush=True)


def masked_table(sizes):
    """Exact mask with the masked rule set against the legacy-mask caches, and
    the rule chosen, per combination."""
    lines = []
    for n in sizes:
        with open(os.path.join(new_xcache(n), 'manifest.json')) as f:
            man = json.load(f)
        lines += [f'\n{n}x{NMCH}n{N_TOTAL} (30 instances per combination)\n',
                  '| Combination | Legacy mask, mean | Exact mask, masked rules, mean '
                  '| Chosen rule differs | Exact longer | Equal | Exact shorter '
                  '| Rules chosen (exact) | Best of four below SPT | Mean gain over SPT |',
                  '|---|---|---|---|---|---|---|---|---|---|']
        tot = np.zeros(5)
        allc = {}
        for r in REGIMES_ALL:
            leg, ex, rd, gains, nbelow, chosen = [], [], 0, [], 0, {}
            for i in range(N_TOTAL):
                a = np.load(os.path.join(new_cache(n), r, inst(i) + '.npz'), allow_pickle=True)
                m = man[f'{r}/{inst(i)}']
                leg.append(float(a['ms']))
                ex.append(m['ms'])
                rd += str(a['rule']) != m['rule']
                chosen[m['rule']] = chosen.get(m['rule'], 0) + 1
                allc[m['rule']] = allc.get(m['rule'], 0) + 1
                s = m['rule_ms']['SPT']
                gains.append(100.0 * (s - m['ms']) / s)
                nbelow += m['ms'] < s
            leg, ex = np.array(leg), np.array(ex)
            c = np.array([rd, (ex > leg).sum(), (ex == leg).sum(), (ex < leg).sum(), nbelow])
            tot += c
            lines.append(f'| {r} | {leg.mean():.1f} | {ex.mean():.1f} | {c[0]} | {c[1]} '
                         f'| {c[2]} | {c[3]} | '
                         f'{", ".join(f"{k} {v}" for k, v in sorted(chosen.items()))} '
                         f'| {c[4]} | {np.mean(gains):.2f}% |')
        lines.append(f'| all | | | {int(tot[0])} | {int(tot[1])} | {int(tot[2])} '
                     f'| {int(tot[3])} | '
                     f'{", ".join(f"{k} {v}" for k, v in sorted(allc.items()))} '
                     f'| {int(tot[4])} | |')
    print('\n'.join(lines))


def warmstart_table(sizes):
    """Warm-start makespan per regime: 000-019 against 020-029, with the
    chosen rules."""
    lines = []
    for n in sizes:
        lines += [f'\n{n}x{NMCH}n{N_TOTAL}\n',
                  '| Regime | 000-019 mean (min to max) | 020-029 mean (min to max) '
                  '| Rules chosen, 000-019 | Rules chosen, 020-029 |',
                  '|---|---|---|---|---|']
        for r in REGIMES_ALL:
            cols = []
            for idx in (range(N_FROZEN), range(N_FROZEN, N_TOTAL)):
                zs = [np.load(os.path.join(new_cache(n), r, inst(i) + '.npz'),
                              allow_pickle=True) for i in idx]
                ms = np.array([float(z['ms']) for z in zs])
                rules = {}
                for z in zs:
                    rules[str(z['rule'])] = rules.get(str(z['rule']), 0) + 1
                cols.append((f'{ms.mean():.1f} ({ms.min():.0f} to {ms.max():.0f})',
                             ', '.join(f'{k} {v}' for k, v in sorted(rules.items()))))
            lines.append(f'| {r} | {cols[0][0]} | {cols[1][0]} | {cols[0][1]} '
                         f'| {cols[1][1]} |')
    print('\n'.join(lines))


# ---------------------------------------------------------------------------

def main():
    a = parse_args(_ARGV)
    if a['cmd'] in ('reproduce', 'build'):
        check_scratch_root(a['scratch'])
        os.makedirs(a['scratch'], exist_ok=True)
        results = [reproduce(n, a['scratch']) for n in a['sizes']]
        if not all(r['ok'] for r in results):
            print('REPRODUCTION FAILED: the generator does not reproduce the frozen '
                  'files; nothing was written to the n30 folders', flush=True)
            sys.exit(1)
        print('reproduction: every frozen file regenerated byte for byte', flush=True)
        if a['cmd'] == 'build':
            for n in a['sizes']:
                plan = plan_build(n, a['scratch'])
                wrote, same = execute_plan(plan, a['scratch'], a['force'])
                print(f'[{n}x{NMCH}] {new_base(n)}, {new_family(n)}: wrote {wrote} '
                      f'files, {same} already present and identical', flush=True)
                bad = verify_built(n, a['scratch'])
                if bad:
                    print(f'VERIFICATION FAILED: {bad[:10]}', flush=True)
                    sys.exit(1)
    elif a['cmd'] == 'check':
        check(a['sizes'], a['rules'], a['indices'], a['out'])
    elif a['cmd'] == 'warmstart':
        check_scratch_root(a['scratch'])
        os.makedirs(a['scratch'], exist_ok=True)
        warmstart(a['sizes'], a['scratch'], a['force'], a['resume'])
        warmstart_table(a['sizes'])
    elif a['cmd'] == 'wstable':
        warmstart_table(a['sizes'])
    elif a['cmd'] == 'wsxmask':
        check_scratch_root(a['scratch'])
        os.makedirs(a['scratch'], exist_ok=True)
        warmstart_xmask(a['sizes'], a['scratch'], a['force'], a['resume'])
        xmask_table(a['sizes'])
    elif a['cmd'] == 'wsxtable':
        xmask_table(a['sizes'])
    elif a['cmd'] == 'verify':
        bad = []
        for n in a['sizes']:
            validate_frozen(n)
            bad += verify_built(n, a['scratch'])
        if bad:
            print(f'VERIFICATION FAILED: {bad[:10]}', flush=True)
            sys.exit(1)
    elif a['cmd'] == 'wsmasked':
        check_scratch_root(a['scratch'])
        os.makedirs(a['scratch'], exist_ok=True)
        warmstart_masked(a['sizes'], a['scratch'], a['resume'], a['wait'])
        masked_table(a['sizes'])
    elif a['cmd'] == 'wsmtable':
        masked_table(a['sizes'])
    else:
        if not a['src']:
            sys.exit('table needs --from <JSON written by check>')
        table(a['src'])


if __name__ == '__main__':
    main()
