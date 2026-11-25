r"""P105: what the non-delay mask costs, measured by replaying reference schedules.

Subcommands run, infer and plan. Evaluation only, CPU only, exact mask. No solver runs and
nothing runs on the GPU.

The environment of the paper (constraint_family/fjsp_env_family.py) appends every dispatched
operation to its machine (list scheduling, no insertion into an earlier gap), and its
non-delay mask admits only the operation-machine pairs whose processing, after any setup,
starts at the decision instant, the earliest processing start over all candidate pairs. The
solver and the memetic algorithm of the comparison are not bound by this mask. A reference
schedule R is the best schedule a search method found for one instance and one combination,
with the machine m_R(o), the processing start s_R(o) and the completion c_R(o) of every
operation o. This script makes the environment follow R as closely as its mask allows. The
makespan it reaches belongs to a schedule the mask admits, so it is an upper bound on the
best makespan of that class, and its distance to R bounds from above what the restriction
costs by itself. Where R is proven optimal, the bound applies to the true cost.

LAUNCH NOTE (the four questions, answered before the run)

1. Purpose. How much makespan does the restriction to schedules built under the non-delay
   mask cost by itself, apart from the quality of any policy? The numbers land in the
   supplement (the section on the controls for the mask, beside the random control of
   scripts/p101_mask_controls.py) and in one sentence of the main text.
2. Expected result. Unknown in size. A small gap (about 1% or less) says that the class of
   schedules the mask admits contains schedules close to the references, so the distance of
   a policy or a rule from a search method is not caused by the mask. A large gap says that
   the simple replay does not bound the cost tightly; the paper then reports the bound as it
   is and names it an upper bound.
3. Contamination. Instances, descriptors, reference records and stored rule arrays are read
   only. Cases go to a new folder, test_results/FAMILY/refreplay_p105 (--root), one JSON
   and one npz file per case, the npz written first and the JSON last, each through a
   temporary file and a rename, so an interrupted run leaves no case that looks finished. A
   stored case of another design or of another reference record (sha256) is refused, never
   overwritten (unless --force). The rollouts are deterministic and run on one thread, so a
   case is not a timing measurement and the run may share cores; the recorded times are
   planning figures.
4. Data accuracy. Every reference schedule is recomputed from its stored arrays: its
   makespan must equal the objective of its record, every duration must equal the
   processing time of the assigned machine, and it must pass the validator of p101
   (reference_sim.validate_schedule). The environment of every case must reproduce the
   stored exact-mask SPT makespan of that instance and combination. Only references the
   owning campaign accepts count: public records that scripts/p89_public_family.py counts
   (Inputs.load_refs), factory records that timed_campaign_plan.cell_states marks valid
   (a pending or invalid record is treated as absent).

REFERENCE SETS
  pub10x5   public benchmark, SD2 test set at 10x5 (data/PUBLIC/family/pub_SD2_test_10x5+mix),
            all eight combinations, 100 instances; reference arm cpsat150, the records of
            scripts/p91_public_cpsat_refs.py (CP-SAT, 150 s, 2 workers; N, L, W, LW proven
            optimal, the other combinations best found):
            or_solution/PUBFAMILY/pub_SD2_test_10x5+mix_cpsat150s_w2/<R>/<instance>.json,
            fields assigned_mch, op_st, op_ct, objective, proved_optimal.
  10x25     factory benchmark, training size, instances 000-029 (the first 30 test
            instances); reference arm memetic_rule600.
  50x25n30, 80x25n30  production batches, instances 003-029; reference arms
            memetic_rule600 and cpsat_rule600.
  memetic_rule600  the timed campaign's arm memetic_rule@600 (eval_memetic_family.py,
            seeded with the rule schedules): or_solution/FAMILY/<size>_memetic600s/<R>/
            instance_<i>_seed1.json, field schedule = {op_mch, op_st, op_ct}, objective
            makespan.
  cpsat_rule600  the timed campaign's arm cpsat_rule@600 (scripts/p85_cpsat_arms.py with the
            configuration the pilot selects, timed_campaign_plan.select_pilot):
            test_results/FAMILY/<size>_cparms/evidence_rule600s/<configuration>/<R>/
            instance_<i>.json with the schedule in instance_<i>.schedule.npz (amch, st, ct),
            objective objective.

THE REPLAY (one instance, one combination, one reference R)
  The environment is P101.make_env, the construction of eval_family.py with the exact mask
  (first_mask and mask_ties exact, mask_mode nondelay); a public instance has no type
  lists, as in eval_family.py. At every decision the legal pairs are read from the
  environment's own mask (dynamic_pair_mask). For a legal pair (j, m) with operation o =
  candidate[j], its completion in the environment is its processing start (the decision
  instant, read from pair_free_time in hours) plus the processing time of o on m. One pair
  is chosen by a fixed order, the smallest key first:
    A  (s_R(o), [m != m_R(o)], completion, m, j): the reference start first, then the
       reference machine before another machine, then the earlier completion, then the lower
       machine index, then the lower job index;
    B  ([m != m_R(o)], s_R(o), completion, m, j): the reference machine first, then the
       reference start;
    C  (c_R(o), [m != m_R(o)], completion, m, j): order A with the reference completion in
       place of the reference start.
  [x] is 0 when x is false and 1 when it is true. The job index makes every order total.
  Each order gives one schedule, which must pass the validator of p101, or run refuses. The
  replay makespan of a case is the smallest of the three; each order's own value is stored.

THE CLASS TEST (without the environment)
  R is a schedule the environment can build when the environment, dispatching R's
  operations in the order of their processing starts, never meets a decision at which the
  next operation of R is not legal. The test walks R's operations in increasing s_R (ties
  by operation index). Before operation o, the state is the one R's earlier operations
  define: the ready time of each job (completion plus lag of its previous operation, or
  its release date), the free time of each machine (completion of its previous operation)
  and the routing class each machine processed last. From this state the earliest
  processing start of every candidate pair (the next operation of each unfinished job on
  each compatible machine) is computed by the environment's rule (the attached setup from
  the machine's last class, the block of setup and processing pushed past every outage it
  overlaps, start = block + setup). Operation o passes when (i) its earliest start on
  m_R(o) equals s_R(o) and (ii) no candidate pair starts earlier than s_R(o). R is in the
  class when every operation passes. A failure of (i) means R starts o later than its job,
  its machine and the outages allow ('later than its earliest start'); a failure of (ii)
  means another pair could start earlier, so the mask would not admit o there ('an
  earlier pair'). Both counts are stored, with the first failure.
  The test is exact (necessary and sufficient), given that its start rule is the
  environment's. Sufficient: when every operation passes, dispatching R's operations in
  this order is a sequence of legal decisions, because by (i) the environment starts o at
  s_R(o) and by (ii) s_R(o) is the decision instant. Necessary: let a sequence Q of legal
  decisions build R, fix a start value s, and let d be the first operation Q dispatches with
  s_R(d) >= s. Every operation x with s_R(x) < s is dispatched before d: otherwise the
  earliest such x has its job and machine predecessors dispatched (they start earlier), so
  its pair on m_R(x) starts at s_R(x) < s <= s_R(d) and d would not be legal. So when Q
  dispatches d, the dispatched set is {o : s_R(o) < s}, the state the test sees before the
  first operation of start s, and since d is legal there no pair starts before s_R(d) >= s,
  which is (ii); (i) holds because Q starts every operation at its reference start and that
  start depends only on the operation's job and machine predecessors in R. Dispatching an
  operation of start s on machine m makes every new pair on m start at its completion
  (later than s) or later, makes its job's next operation ready no earlier than that
  completion, and leaves every other pair unchanged, so the other operations of start s
  stay admitted. All times are integer, so the half-quantum tolerance of the exact mask
  plays no role.
  Cross-check: when R is in the class, order A reproduces R exactly (at every decision the
  smallest reference start among the legal pairs is that of the next operations of R, and
  the reference machine comes first), and when A reproduces R, R is in the class by
  definition. run refuses a case in which the two disagree.

RECORDED PER CASE
  the reference makespan (recomputed from the stored completions, checked against the
  record's objective); whether the reference is proven optimal; the three replay makespans
  and the smallest; per order the number of decisions at which the chosen pair is not the
  next operation of R on its reference machine (the pair's machine differs from m_R(o), or
  an operation that R places on m_R(o) before o is not yet dispatched); whether replay A
  reproduces R exactly (same machine and same processing start for every operation); the
  class test; the decision count (the number of operations); the makespans of FIFO, MOR, SPT
  and MWKR restricted to the legal pairs from the stored exact-mask arrays and their
  per-instance best; the SPT check.

OUTCOMES (infer), per set, reference arm and combination, and pooled over the combinations
  gap         100 (replay - reference) / reference per case; mean and 95% percentile
              bootstrap over instances (stats_family.instance_boot, 20,000 resamples,
              fresh default_rng(7)); pooled, a drawn instance brings all its combinations;
  shares      % of cases with replay <= reference, with gap <= 1 (within 1%), with gap > 5;
  buildable   % of references in the class (the class test; equal to the % that replay A
              reproduces exactly);
  left        mean over cases of 100 x (decisions of order A off the reference) / decisions;
  outside     mean over cases of the % of the operations of R that fail the class test, in
              total and by kind (the operation starts later than its earliest start on its
              machine; another pair could start earlier): the share of R's dispatches that
              the non-delay rule does not admit along R itself;
  rule gap    100 (best classical rule - reference) / reference, mean and interval;
  replay or rule  the same gap for the shorter of the replay and the best classical rule
              per case: the rule's schedule is also built under the mask, so this is the
              tightest upper bound of the two on the best makespan the mask admits;
  proven      for the public set, gap and shares again over the references proven optimal
              (per combination; pooled over the combinations whose references are all
              proven, N, L, W and LW).
  Outputs: results/reference_replay.json, results/reference_replay.md and
  manuscript/macros_refreplay.tex (macro prefix Rply), or the same paths under --out.

FOLDERS
  <root>/<set>/<arm>/<R>/<instance>.json, .npz   one case; npz: amch [3, N] and op_ct [3, N]
                                                 of the schedules of orders A, B and C

USAGE (repository root; CPU only)
  python scripts/p105_reference_replay.py plan  [--parts K]
  python scripts/p105_reference_replay.py run   [--sets ...] [--refs ...] [--combos ...]
                                                [--part i --parts K]
  python scripts/p105_reference_replay.py infer [--refs ...] [--require all|present]
  A selection of sets or reference arms is a design choice and may write the repository
  outputs; a subset of combinations or instances (--n) and --require present need
  --partial, which writes under --out only. --require all (the default) refuses when a
  planned reference or case is missing.
  Smoke: run --root <folder> --n 3 --combos N,S,LSW, then infer --partial --require present
  --root <folder> --out <folder> --n 3 --combos N,S,LSW.
"""
import argparse
import glob
import json
import os
import re
import sys
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(ROOT_DIR, 'scripts')
sys.path.insert(0, ROOT_DIR)
if SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)

import arm_registry as REG  # noqa: E402,F401  (removes --pact-* options from sys.argv)

_ARGV = sys.argv[1:]
# params parses argv at import time and EnvState reads the device then: CPU only
sys.argv = sys.argv[:1] + ['--device', 'cpu']

import numpy as np  # noqa: E402

import p101_mask_controls as P101  # noqa: E402
import stats_family as SF  # noqa: E402

# ---------------------------------------------------------------------------
# design constants
# ---------------------------------------------------------------------------

R8 = P101.R8
RULES = P101.RULES                      # FIFO, MOR, SPT, MWKR
ORDERS = ('A', 'B', 'C')
EPS = 1e-6
B = SF.B_DEFAULT                        # 20,000 bootstrap resamples
CI_SEED = 7
DEFAULT_ROOT = 'test_results/FAMILY/refreplay_p105'
OUT_FILES = {'json': 'results/reference_replay.json', 'md': 'results/reference_replay.md',
             'macros': 'manuscript/macros_refreplay.tex'}
GENERATOR = 'scripts/p105_reference_replay.py infer'
PREFIX = 'Rply'
VERSION = 1

PUB_SET = 'pub_SD2_test_10x5+mix'
SETS = {
    'pub10x5': dict(kind='public', set=PUB_SET, data=f'data/PUBLIC/family/{PUB_SET}',
                    stored=f'test_results/PUBFAMILY/{PUB_SET}_xmask', instances=None,
                    text='public benchmark, SD2 test set 10x5, 100 instances'),
    '10x25': dict(kind='factory', size='10x25', instances=tuple(range(0, 30)),
                  text='factory benchmark, 10 modules, instances 000-029'),
    '50x25n30': dict(kind='factory', size='50x25n30', instances=tuple(range(3, 30)),
                     text='factory benchmark, 50 modules, instances 003-029'),
    '80x25n30': dict(kind='factory', size='80x25n30', instances=tuple(range(3, 30)),
                     text='factory benchmark, 80 modules, instances 003-029'),
}
REFS = {
    'cpsat150': dict(sets=('pub10x5',), token=None,
                     text='CP-SAT, 150 s, 2 workers (scripts/p91_public_cpsat_refs.py)'),
    'memetic_rule600': dict(sets=('10x25', '50x25n30', '80x25n30'), token='memetic_rule@600',
                            text='memetic algorithm, 600 s, seeded with the rule schedules '
                                 '(timed campaign, memetic_rule@600)'),
    'cpsat_rule600': dict(sets=('50x25n30', '80x25n30'), token='cpsat_rule@600',
                          text='CP-SAT, 600 s, 8 workers, hinted with the rule schedule '
                               '(timed campaign, cpsat_rule@600)'),
}
PAIRS = [(s, r) for r in REFS for s in REFS[r]['sets']]
WORD = {('pub10x5', 'cpsat150'): 'Pub', ('10x25', 'memetic_rule600'): 'TenMem',
        ('50x25n30', 'memetic_rule600'): 'FiftyMem', ('80x25n30', 'memetic_rule600'): 'EightyMem',
        ('50x25n30', 'cpsat_rule600'): 'FiftyCp', ('80x25n30', 'cpsat_rule600'): 'EightyCp'}
# planning cost, process CPU seconds per case (three replays, the class test, the SPT check
# and the loading), from the smoke cases (2026-11-13, one shared core): the 800 public cases,
# and three instances of N, S and LSW per factory size (LSW the slowest)
COST_S = {'pub10x5': 0.1, '10x25': 0.7, '50x25n30': 5.5, '80x25n30': 11.5}
ORDER_TEXT = {
    'A': '(s_R(o), [m != m_R(o)], completion, m, j), smallest first',
    'B': '([m != m_R(o)], s_R(o), completion, m, j), smallest first',
    'C': '(c_R(o), [m != m_R(o)], completion, m, j), smallest first'}
DEPARTURE_TEXT = ('the chosen pair (o, m) is off the reference when m != m_R(o) or an operation '
                  'that R places on m_R(o) before o is not yet dispatched')
CLASS_TEXT = ('R in the class: in the order of R\'s processing starts, every operation starts '
              'at its earliest start on its reference machine given R\'s earlier operations, '
              'and no candidate pair starts earlier')


def stop(msg):
    """A precondition of plan or run failed."""
    raise SystemExit(f'REFUSING: {msg}')


Refused = P101.Refused
refuse = P101.refuse          # infer: raises Refused('refused: ...'), outputs removed


def rel(p):
    """A path as stored in a record: relative to the repository root."""
    return os.path.relpath(os.path.join(ROOT_DIR, p) if not os.path.isabs(p) else p, ROOT_DIR)


# ---------------------------------------------------------------------------
# instances and stored rule arrays
# ---------------------------------------------------------------------------

_PUB_NAMES = {}


def public_names():
    """Sorted descriptor names of the public set (the row order of its stored arrays);
    every combination holds the same names."""
    if 'n' not in _PUB_NAMES:
        d = os.path.join(ROOT_DIR, SETS['pub10x5']['data'], 'desc')
        names = None
        for r in R8:
            nm = sorted(os.path.basename(p)[:-len('.desc.json')]
                        for p in glob.glob(os.path.join(d, r, '*.desc.json')))
            if names is None:
                names = nm
            elif nm != names:
                stop(f'{d}/{r} holds other descriptor names than N')
        if not names or len(names) != 100:
            stop(f'{d}: {len(names or [])} descriptors, expected 100')
        _PUB_NAMES['n'] = names
    return _PUB_NAMES['n']


def all_names(set_key):
    """The sorted names of a set (row order of the stored rule arrays)."""
    spec = SETS[set_key]
    if spec['kind'] == 'public':
        return public_names()
    return P101.names_of(spec['size'])


def planned_names(set_key, n=None):
    """The planned instances of a set, in order, optionally the first n."""
    spec = SETS[set_key]
    names = all_names(set_key)
    if spec['instances'] is None:
        out = list(names)
    else:
        out = [f'instance_{i:03d}' for i in spec['instances']]
        miss = [x for x in out if x not in names]
        if miss:
            stop(f'{set_key}: planned instances {miss[:3]} have no descriptor')
    return out[:n] if n else out


def load_instance_case(set_key, combo, name):
    """(jl, pt, meta, desc) of one instance and combination, read only. A factory instance
    is read as P101.load_cell reads it (descriptor of data/FAMILY/<size>, base instance
    <base>/<name>); a public instance as eval_family.py reads it (no type lists)."""
    from constraint_family.descriptor import ConstraintDescriptor
    spec = SETS[set_key]
    if spec['kind'] == 'public':
        from constraint_family.public_rules import load_family_record
        dp = os.path.join(ROOT_DIR, spec['data'], 'desc', combo, f'{name}.desc.json')
        jl, pt, desc, rec = load_family_record(dp, root=ROOT_DIR)
        if rec.get('regime') != combo or os.path.basename(rec['base_instance']) != name:
            stop(f'{rel(dp)} names regime {rec.get("regime")} and base instance '
                 f'{rec.get("base_instance")}')
        return jl, pt, {'op_type': None, 'mch_type': None}, desc
    from ppvc_instance_generator import load_instance
    sz = P101.SIZES[spec['size']]
    with open(os.path.join(ROOT_DIR, sz['data'], 'desc', combo, f'{name}.desc.json')) as f:
        rec = json.load(f)
    jl, pt, meta = load_instance(os.path.join(ROOT_DIR, sz['base'], name))
    return jl, pt, meta, ConstraintDescriptor.from_json(rec, setup_convention='attached')


_RULE_CACHE = {}


def rule_arrays(set_key, combo):
    """{rule: makespans over all_names(set_key)} of the stored exact-mask rule arrays that
    exist (column 0)."""
    key = (set_key, combo)
    if key not in _RULE_CACHE:
        spec = SETS[set_key]
        folder = (spec['stored'] if spec['kind'] == 'public'
                  else os.path.join(P101.RESULTS_ROOT, f'{spec["size"]}_xmask'))
        n = len(all_names(set_key))
        out = {}
        for r in RULES:
            p = os.path.join(ROOT_DIR, folder, f'Result_{r}_{combo}.npy')
            if os.path.exists(p):
                a = np.load(p)
                a = (a[:, 0] if a.ndim > 1 else a).astype(float)
                if len(a) != n:
                    stop(f'{rel(p)} has {len(a)} rows, the set has {n} instances')
                out[r] = a
        _RULE_CACHE[key] = out
    return _RULE_CACHE[key]


# ---------------------------------------------------------------------------
# references: where they are, which ones count
# ---------------------------------------------------------------------------

_STATE = {}


def cpsat_configuration(given=None):
    """The CP-SAT configuration of the campaign's arm cpsat_rule: the pilot's choice
    (timed_campaign_plan.select_pilot, as scripts/p94_timed_inference.py); a given value must
    equal it. Returns (configuration or None, reason)."""
    import timed_campaign_plan as P
    d = P.build_parser().parse_args(['table'])
    res = P.select_pilot(P.Campaign(d).tr_root(), P.SIZES, [P.parse_cores(s) for s in d.slots],
                         d.bars)
    if res['refused']:
        return None, f'the pilot selection cannot be determined: {res["refused"][:1]}'
    if given and given != res['chosen']:
        stop(f'--cpsat_arm {given} differs from the pilot\'s choice {res["chosen"]}')
    return res['chosen'], ''


def factory_states(cpsat_arm):
    """{(ref, size, combo, idx): (state, reason, record path)} of the campaign's cells, by
    timed_campaign_plan.cell_states (states valid, pending, invalid, foreign, held,
    missing)."""
    import timed_campaign_plan as P
    out = {}
    for ref in ('memetic_rule600', 'cpsat_rule600'):
        if ref == 'cpsat_rule600' and cpsat_arm is None:
            continue
        sizes = [SETS[s]['size'] for s in REFS[ref]['sets']]
        a = P.build_parser().parse_args(['table', '--arms', REFS[ref]['token'], '--sizes',
                                         *sizes, '--cpsat-arm', cpsat_arm or 'ref_fullhint'])
        camp = P.Campaign(a)
        for seg in camp.segments:
            for _, (c, state, why) in P.cell_states(camp, seg).items():
                out[(ref, c.size, c.combo, c.idx)] = (state, why, camp.record(c))
    return out


def public_states():
    """{(combo, name): (state, reason, record path, proved)} of the public references, by
    scripts/p89_public_family.py Inputs.load_refs (a record counts when p91 marked it valid,
    record_validity still says so, and it belongs to the campaign)."""
    import p89_public_family as P89
    inputs = P89.Inputs(ROOT_DIR)
    inputs.check_records(PUB_SET)
    R = inputs.load_refs(PUB_SET)
    names = R['names']
    if names != public_names():
        stop('p89 orders the public instances differently from this script')
    why = {}
    for line in R['not_counted']:
        k, _, reason = line.partition(': ')
        why[k] = reason
    out = {}
    for r in R8:
        x = R['regimes'][r]
        for i, nm in enumerate(names):
            path = os.path.join(R['folder'], r, f'{nm}.json')
            if x['valid'][i]:
                out[(r, nm)] = ('valid', '', path, bool(x['proved'][i]))
            elif x['present'][i]:
                out[(r, nm)] = ('invalid', why.get(f'{r}/{nm}', 'not counted by p89'), path,
                                False)
            else:
                out[(r, nm)] = ('missing', '', path, False)
    return out


def reference_states(args):
    """Every state the run and infer need, computed once per process and selection."""
    want = set(args.refs)
    key = (tuple(sorted(want)), getattr(args, 'cpsat_arm', None))
    if key not in _STATE:
        cp, cp_why = (None, 'not requested')
        if 'cpsat_rule600' in want:
            cp, cp_why = cpsat_configuration(getattr(args, 'cpsat_arm', None))
        fac = factory_states(cp) if want & {'memetic_rule600', 'cpsat_rule600'} else {}
        pub = public_states() if 'cpsat150' in want else {}
        _STATE[key] = dict(factory=fac, public=pub, cpsat_arm=cp, cpsat_why=cp_why)
    return _STATE[key]


def ref_state(states, set_key, ref, combo, name):
    """(state, reason, record path, proved or None) of one planned reference."""
    if SETS[set_key]['kind'] == 'public':
        return states['public'].get((combo, name), ('missing', 'no record', None, False))
    if ref == 'cpsat_rule600' and states['cpsat_arm'] is None:
        return ('missing', states['cpsat_why'], None, None)
    st, why, path = states['factory'].get((ref, SETS[set_key]['size'], combo,
                                           int(name.split('_')[-1])),
                                          ('missing', 'not a cell of the campaign', None))
    return st, why, path, None


def read_reference(set_key, ref, path):
    """(amch, st, ct, objective, files {relative path: sha256}, extra) of one reference
    record, as integer arrays."""
    full = os.path.join(ROOT_DIR, path)
    with open(full) as f:
        rec = json.load(f)
    files = {rel(full): P101.sha256_file(full)}
    if ref == 'cpsat150':
        amch, st, ct = rec['assigned_mch'], rec['op_st'], rec['op_ct']
        obj = float(rec['objective'])
        extra = dict(status=rec.get('status'), proved=bool(rec.get('proved_optimal')))
    elif ref == 'memetic_rule600':
        s = rec['schedule']
        amch, st, ct = s['op_mch'], s['op_st'], s['op_ct']
        obj = float(rec['makespan'])
        if abs(float(rec.get('objective', obj)) - obj) > EPS:
            stop(f'{rel(full)}: makespan {obj} and objective {rec.get("objective")} differ')
        extra = dict(status=rec.get('status'), proved=False)
    else:
        sp = os.path.join(os.path.dirname(full), rec['schedule_file'])
        with np.load(sp) as z:
            amch, st, ct, ms = z['amch'], z['st'], z['ct'], float(z['ms'])
        files[rel(sp)] = P101.sha256_file(sp)
        obj = float(rec['objective'])
        if abs(ms - obj) > EPS:
            stop(f'{rel(sp)}: stored makespan {ms} differs from the objective {obj}')
        extra = dict(status=rec.get('status'), exactness=rec.get('exactness'),
                     proved=rec.get('status') == 'OPTIMAL' and rec.get('exactness') != 'heuristic')
    return (np.asarray(amch, dtype=int), np.asarray(st, dtype=int), np.asarray(ct, dtype=int),
            obj, files, extra)


def check_reference(jl, pt, desc, amch, st, ct, objective, where):
    """The reference makespan recomputed from the stored schedule. Refuses when an array
    has the wrong length, a machine is not eligible, a duration differs from the
    processing time, the last completion differs from the record's objective, or the
    schedule fails the validator of p101."""
    N = pt.shape[0]
    if not (amch.shape == st.shape == ct.shape == (N,)):
        stop(f'{where}: schedule arrays of shape {amch.shape}, {st.shape}, {ct.shape}, '
             f'the instance has {N} operations')
    if (amch < 0).any() or (amch >= pt.shape[1]).any() or (pt[np.arange(N), amch] <= 0).any():
        stop(f'{where}: an operation is assigned a machine that cannot process it')
    if not np.array_equal(ct - st, pt[np.arange(N), amch]):
        stop(f'{where}: a duration differs from the processing time of its machine')
    ms = float(ct.max())
    if abs(ms - float(objective)) > EPS:
        stop(f'{where}: the schedule ends at {ms}, the record\'s objective is {objective}')
    v = P101.validate(jl, pt, desc, amch, ct.astype(float), ms)
    if v:
        stop(f'{where}: the reference schedule is infeasible under the descriptor: {v[:2]}')
    return ms


# ---------------------------------------------------------------------------
# the replay
# ---------------------------------------------------------------------------

def choose(order, js, ms, ops, compl, ref):
    """Index of the pair the order picks among the legal pairs (js[k], ms[k]) with
    operations ops[k] and completions compl[k]; ref holds amch, st and ct of R."""
    off = (ms != ref['amch'][ops]).astype(int)
    if order == 'A':
        keys = (js, ms, compl, off, ref['st'][ops])
    elif order == 'B':
        keys = (js, ms, compl, ref['st'][ops], off)
    elif order == 'C':
        keys = (js, ms, compl, off, ref['ct'][ops])
    else:
        raise ValueError(f'unknown order {order}')
    return int(np.lexsort(keys)[0])


def machine_sequences(amch, st, M):
    """Operations of R per machine in the order of their processing starts."""
    order = np.lexsort((np.arange(len(st)), st))
    seq = [[] for _ in range(M)]
    for o in order:
        seq[int(amch[o])].append(int(o))
    return seq


def replay(jl, pt, meta, desc, ref, order):
    """One schedule of the environment under the exact mask, choosing at every decision
    the legal pair the order picks. Returns makespan, amch, op_ct, the decisions off the
    reference, whether the schedule equals R, and the decision count."""
    N, M = pt.shape
    env = P101.make_env(jl, pt, meta, [desc])
    seq = machine_sequences(ref['amch'], ref['st'], M)
    ptr = [0] * M
    done = np.zeros(N, dtype=bool)
    amch = np.full(N, -1, dtype=int)
    off, steps = 0, 0
    while not env.done().all():
        legal = ~np.asarray(env.dynamic_pair_mask[0], dtype=bool)
        js, ms = np.nonzero(legal)
        if js.size == 0:
            raise RuntimeError('the mask admits no pair')
        ops = env.candidate[0, js]
        start = np.rint(np.asarray(env.pair_free_time[0], dtype=float)[js, ms] / env._slope)
        compl = start + pt[ops, ms]
        k = choose(order, js, ms, ops, compl, ref)
        j, m, o = int(js[k]), int(ms[k]), int(ops[k])
        mr = int(ref['amch'][o])
        while ptr[mr] < len(seq[mr]) and done[seq[mr][ptr[mr]]]:
            ptr[mr] += 1
        if m != mr or seq[mr][ptr[mr]] != o:
            off += 1
        done[o] = True
        amch[o] = m
        env.step(np.array([j * M + m]))
        steps += 1
    op_ct = np.asarray(env.true_op_ct[0], dtype=float).copy()
    msp = float(env.current_makespan[0])
    op_st = op_ct - pt[np.arange(N), amch]
    exact = bool(np.array_equal(amch, ref['amch']) and np.allclose(op_st, ref['st'], atol=EPS))
    return dict(makespan=msp, amch=amch, op_ct=op_ct, off=off, exact=exact, decisions=steps)


def class_test(jl, pt, desc, amch, st, ct):
    """THE CLASS TEST of the module doc, from R and the instance only. Returns in_class,
    the counts of the two failures (later than its earliest start, an earlier pair) and the
    first failure."""
    from constraint_family.descriptor import IDLE_CLASS
    from constraint_family.reference_sim import setup_mult_of
    pt = np.asarray(pt, dtype=float)
    N, M = pt.shape
    jl = np.asarray(jl, dtype=int)
    J = len(jl)
    first = np.concatenate([[0], np.cumsum(jl)[:-1]]).astype(int)
    last = first + jl - 1
    job = np.repeat(np.arange(J), jl)
    lag = np.asarray(desc.lag, dtype=float) if desc.has_lag else np.zeros(N)
    cls = np.asarray(desc.op_class, dtype=int)
    S = (np.asarray(desc.setup, dtype=float) if desc.has_setup and desc.setup is not None
         else None)
    u = setup_mult_of(desc, M)
    ws = we = None
    if desc.has_window and desc.windows is not None:
        K = max([len(w) for w in desc.windows] + [1])
        ws, we = np.full((M, K), np.inf), np.full((M, K), np.inf)
        for m, spans in enumerate(desc.windows):
            for k, (a, b) in enumerate(spans):
                ws[m, k], we[m, k] = float(a), float(b)
    rel_ = getattr(desc, 'release', None)
    ready = np.zeros(J) if rel_ is None else np.asarray(rel_, dtype=float).copy()
    free = np.zeros(M)
    lastc = np.full(M, IDLE_CLASS, dtype=int)
    nxt = first.copy()
    later = earlier = 0
    first_fail = None
    st = np.asarray(st, dtype=float)
    ct = np.asarray(ct, dtype=float)
    for step, o in enumerate(np.lexsort((np.arange(N), st))):
        o = int(o)
        j, m = int(job[o]), int(amch[o])
        if nxt[j] != o:
            raise ValueError(f'operation {o} starts before its job predecessor')
        live = nxt <= last
        c = np.where(live, nxt, 0)
        p = pt[c]                                                   # [J, M]
        ok = (p > 0) & live[:, None]
        s = (S[lastc[None, :], cls[c][:, None]] if S is not None else np.zeros((J, M)))
        if u is not None:
            s = s * u[None, :]
        blk = np.maximum(ready[:, None], free[None, :])
        if ws is not None:
            for k in range(ws.shape[1]):
                a, b = ws[None, :, k], we[None, :, k]
                hit = (blk < b) & (blk + s + p > a)
                blk = np.where(hit, b, blk)
        start = blk + s
        best = float(start[ok].min())
        own = float(start[j, m])
        kind = None
        if own > st[o] + EPS:
            raise ValueError(f'operation {o} starts at {st[o]}, before its earliest start '
                             f'{own} under the environment\'s rule')
        if own < st[o] - EPS:
            later += 1
            kind = 'later than its earliest start'
        elif best < st[o] - EPS:
            earlier += 1
            kind = 'an earlier pair'
        if kind and first_fail is None:
            first_fail = dict(step=step, op=o, kind=kind, start=float(st[o]),
                              earliest_own=own, earliest_any=best)
        ready[j] = ct[o] + lag[o]
        free[m] = ct[o]
        lastc[m] = cls[o]
        nxt[j] += 1
    return dict(in_class=(later + earlier == 0), later=later, earlier_pair=earlier,
                first_fail=first_fail)


def replay_case(jl, pt, meta, desc, ref, where='case'):
    """The three replays and the class test of one reference ref = dict(amch, st, ct).
    Returns (record fields, schedules {order: (amch, op_ct)}). Refuses an infeasible replay
    and a disagreement between the class test and order A."""
    out, sched = {}, {}
    for od in ORDERS:
        r = replay(jl, pt, meta, desc, ref, od)
        v = P101.validate(jl, pt, desc, r['amch'], r['op_ct'], r['makespan'])
        if v:
            stop(f'{where}: the schedule of order {od} is infeasible: {v[:2]}')
        if r['decisions'] != pt.shape[0]:
            stop(f'{where}: order {od} took {r["decisions"]} decisions for {pt.shape[0]} '
                 f'operations')
        out[od] = dict(makespan=r['makespan'], off=int(r['off']), exact=r['exact'])
        sched[od] = (r['amch'], r['op_ct'])
    ct = class_test(jl, pt, desc, ref['amch'], ref['st'], ref['ct'])
    if ct['in_class'] != out['A']['exact']:
        stop(f'{where}: the class test says {ct["in_class"]}, order A reproduces R: '
             f'{out["A"]["exact"]}; the two computations disagree')
    best = min(out[od]['makespan'] for od in ORDERS)
    rec = dict(replay=out, replay_makespan=best,
               best_order=[od for od in ORDERS if out[od]['makespan'] == best][0],
               exact_A=out['A']['exact'], in_class=ct['in_class'],
               class_failures=dict(later=ct['later'], earlier_pair=ct['earlier_pair']),
               class_first_failure=ct['first_fail'], decisions=int(pt.shape[0]),
               off_A=out['A']['off'], off_share_A=out['A']['off'] / pt.shape[0])
    return rec, sched


# ---------------------------------------------------------------------------
# cases: design record, paths, state
# ---------------------------------------------------------------------------

def design(set_key, ref, combo):
    """The fields that define a case's content; a stored case with other fields is
    refused."""
    return dict(kind='reference replay', version=VERSION, set=set_key, ref=ref, combo=combo,
                mask=P101.MASK, mask_mode='nondelay', orders=ORDER_TEXT,
                departure=DEPARTURE_TEXT, class_test=CLASS_TEXT,
                validator='reference_sim.validate_schedule (P101.validate)')


def case_paths(root, set_key, ref, combo, name):
    stem = os.path.join(root, set_key, ref, combo, name)
    return stem + '.json', stem + '.npz'


def case_state(root, set_key, ref, combo, name, files):
    """'done' when the case holds this design and this reference record, 'absent' when
    missing, 'other' otherwise."""
    js, npz = case_paths(root, set_key, ref, combo, name)
    if not (os.path.exists(js) and os.path.exists(npz)):
        return 'absent', None
    with open(js) as f:
        rec = json.load(f)
    if rec.get('design') != P101.jsonable(design(set_key, ref, combo)):
        return 'other', rec
    if files is not None and rec.get('reference', {}).get('files') != files:
        return 'other', rec
    return 'done', rec


def run_case(root, set_key, ref, combo, name, path, proved, force):
    """One case: load, check the reference, replay, write. Returns ('done' | 'skip', cpu s)."""
    amch, st, ct, obj, files, extra = read_reference(set_key, ref, path)
    state, _ = case_state(root, set_key, ref, combo, name, files)
    if state == 'done' and not force:
        return 'skip', 0.0
    js, npz = case_paths(root, set_key, ref, combo, name)
    if state == 'other' and not force:
        stop(f'the case {set_key}/{ref}/{combo}/{name} under --root holds another design or '
             f'reference record; use --force or another --root')
    t0, c0 = time.perf_counter(), time.process_time()
    where = f'{set_key}/{ref}/{combo}/{name}'
    jl, pt, meta, desc = load_instance_case(set_key, combo, name)
    ms_ref = check_reference(jl, pt, desc, amch, st, ct, obj, where)
    ref_arr = dict(amch=amch, st=st, ct=ct)
    rec, sched = replay_case(jl, pt, meta, desc, ref_arr, where)
    row = all_names(set_key).index(name)
    rules = rule_arrays(set_key, combo)
    rule_ms = {r: float(a[row]) for r, a in rules.items()}
    spt = P101.rule_rollout('SPT', jl, pt, meta, desc)['makespan']
    if 'SPT' in rule_ms:
        if abs(spt - rule_ms['SPT']) > EPS:
            stop(f'{where}: the SPT rollout of this environment ends at {spt}, the stored '
                 f'array holds {rule_ms["SPT"]}; the environments diverge')
        spt_check = 'equal to the stored array'
    else:
        spt_check = 'no stored SPT array'
    have4 = all(r in rule_ms for r in RULES)
    rec.update(
        design=P101.jsonable(design(set_key, ref, combo)), set=set_key, ref=ref, combo=combo,
        instance=name, row=row, n_ops=int(pt.shape[0]), n_jobs=int(len(jl)),
        n_machines=int(pt.shape[1]),
        reference=dict(files=files, objective=obj, makespan=ms_ref,
                       proved_optimal=bool(proved if proved is not None else extra['proved']),
                       status=extra.get('status'), exactness=extra.get('exactness')),
        ref_makespan=ms_ref, rules=rule_ms,
        best_rule=(min(rule_ms[r] for r in RULES) if have4 else None),
        best_rule_name=(min(RULES, key=lambda r: (rule_ms[r], RULES.index(r))) if have4
                        else None),
        spt_check=spt_check)
    rec['cpu_s'] = time.process_time() - c0
    rec['wall_s'] = time.perf_counter() - t0
    rec['timing_note'] = 'planning figures only; the replays are deterministic'
    P101.save_npz_atomic(npz, amch=np.stack([sched[o][0] for o in ORDERS]).astype(np.int16),
                         op_ct=np.stack([sched[o][1] for o in ORDERS]))
    P101.write_json_atomic(js, P101.jsonable(rec))
    return 'done', rec['cpu_s']


# ---------------------------------------------------------------------------
# plan and run
# ---------------------------------------------------------------------------

def selected_pairs(args):
    return [(s, r) for (s, r) in PAIRS if s in args.sets and r in args.refs]


def case_list(args):
    """(set, ref, combo, name) of every planned case of the selection, in a fixed order."""
    return [(s, r, c, nm) for (s, r) in selected_pairs(args) for c in args.combos
            for nm in planned_names(s, args.n)]


def assign_parts(cases, parts):
    """Longest-processing-time assignment of the cases to parts."""
    order = sorted(range(len(cases)), key=lambda i: (-COST_S[cases[i][0]], i))
    loads, out = [0.0] * parts, [[] for _ in range(parts)]
    for i in order:
        p = int(np.argmin(loads))
        out[p].append(i)
        loads[p] += COST_S[cases[i][0]]
    return [sorted(x) for x in out], loads


def plan(args):
    states = reference_states(args)
    cases = case_list(args)
    parts, loads = assign_parts(cases, args.parts)
    summary = {}
    for (s, r, c, nm) in cases:
        st, why, path, _ = ref_state(states, s, r, c, nm)
        k = summary.setdefault((s, r), dict(planned=0, valid=0, done=0, other={}))
        k['planned'] += 1
        if st == 'valid':
            k['valid'] += 1
            files = None
            js, npz = case_paths(args.root, s, r, c, nm)
            if os.path.exists(js) and os.path.exists(npz):
                files = read_reference(s, r, path)[4]
            if case_state(args.root, s, r, c, nm, files)[0] == 'done':
                k['done'] += 1
        else:
            k['other'][st] = k['other'].get(st, 0) + 1
    total = sum(COST_S[x[0]] for x in cases)
    print(f'[p105 plan] {len(cases)} planned cases, at most {total / 3600:.2f} core-hours of '
          f'CPU time (COST_S per case, measured on smoke cases), {args.parts} parts of '
          f'{min(loads) / 3600:.2f} to {max(loads) / 3600:.2f} h; CP-SAT configuration of '
          f'cpsat_rule600: {states["cpsat_arm"] or states["cpsat_why"]}')
    for (s, r), k in summary.items():
        oth = ', '.join(f'{v} {st}' for st, v in sorted(k['other'].items())) or 'none'
        print(f'  {s:9s} {r:16s} planned {k["planned"]:4d}  valid references {k["valid"]:4d}  '
              f'cases done {k["done"]:4d}  other references: {oth}')
    py = sys.executable + ' -B'
    for i in range(args.parts):
        print(f'CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c '
              f'<CORE_{i}> {py} scripts/p105_reference_replay.py run --part {i} '
              f'--parts {args.parts}')
    return dict(cases=len(cases), core_h=total / 3600, part_h=[x / 3600 for x in loads],
                summary={f'{s}|{r}': k for (s, r), k in summary.items()})


def run(args):
    P101._cpu_torch()
    states = reference_states(args)
    cases = case_list(args)
    if args.part is not None:
        parts, _ = assign_parts(cases, args.parts)
        cases = [cases[i] for i in parts[args.part]]
    t_all = time.perf_counter()
    counts = {}
    for k, (s, r, c, nm) in enumerate(cases):
        st, why, path, proved = ref_state(states, s, r, c, nm)
        if st != 'valid':
            counts[st] = counts.get(st, 0) + 1
            continue
        res, cpu = run_case(args.root, s, r, c, nm, path, proved, args.force)
        counts[res] = counts.get(res, 0) + 1
        if res == 'done':
            print(f'[p105 run {k + 1}/{len(cases)}] {s:9s} {r:16s} {c:4s} {nm} '
                  f'cpu={cpu:6.1f}s elapsed={time.perf_counter() - t_all:7.0f}s', flush=True)
    print(f'[p105 run] {len(cases)} planned cases: '
          + ', '.join(f'{v} {k}' for k, v in sorted(counts.items())) + '; references not '
          'valid now (pending, invalid, missing) are left out', flush=True)
    return counts


# ---------------------------------------------------------------------------
# infer
# ---------------------------------------------------------------------------

def gap(a, b):
    return SF.rel_gap(a, b)


def boot(d):
    """Mean and 95% percentile bootstrap over instances; d [n] or [n, c] (an instance brings
    its c combinations)."""
    r = SF.instance_boot(np.asarray(d, dtype=float), B=B, seed=CI_SEED)
    return dict(mean=r['mean'], lo=r['lo'], hi=r['hi'], n=int(r['n']), c=int(r['c']))


def pct(x):
    x = np.asarray(x, dtype=bool)
    return 100.0 * float(x.mean())


def shares(rep, ref):
    """% of cases at or below the reference, within 1% of it (gap <= 1) and more than 5%
    above it (gap > 5)."""
    rep, ref = np.asarray(rep, dtype=float), np.asarray(ref, dtype=float)
    g = gap(rep, ref)
    return dict(at_or_below=pct(rep <= ref + EPS), within_one=pct(g <= 1.0 + 1e-9),
                above_five=pct(g > 5.0 + 1e-9), n=int(rep.size))


def cell_stats(ref, rep, rule=None, build=None, off=None, orders=None, late=None,
               early=None):
    """Outcomes of one cell (arrays over instances [n]) or of a pooled block ([n, c]).
    late and early hold, per case, the % of the reference's operations that fail the class
    test by each kind."""
    ref, rep = np.asarray(ref, dtype=float), np.asarray(rep, dtype=float)
    out = dict(n=int(ref.shape[0]), cases=int(ref.size), ref_ms=float(ref.mean()),
               replay_ms=float(rep.mean()), gap=boot(gap(rep, ref)), shares=shares(rep, ref))
    if build is not None:
        out['buildable'] = pct(build)
    if off is not None:
        out['left'] = 100.0 * float(np.mean(off))
    if late is not None and early is not None:
        late, early = np.asarray(late, dtype=float), np.asarray(early, dtype=float)
        out['outside'] = dict(later=float(late.mean()), earlier_pair=float(early.mean()),
                              total=float((late + early).mean()))
    if rule is not None:
        rule = np.asarray(rule, dtype=float)
        out['rule_ms'] = float(rule.mean())
        out['rule_gap'] = boot(gap(rule, ref))
        both = np.minimum(rep, rule)
        out['bound_gap'] = boot(gap(both, ref))
        out['bound_shares'] = shares(both, ref)
    if orders is not None:
        out['orders'] = {o: dict(ms=float(np.mean(x)), gap=float(np.mean(gap(x, ref))))
                         for o, x in orders.items()}
    return out


def read_cases(args, states, s, r, c, names):
    """{name: case record} of the cases that count, and the list of what is missing."""
    got, missing = {}, []
    for nm in names:
        st, why, path, proved = ref_state(states, s, r, c, nm)
        if st != 'valid':
            missing.append(f'{s}/{r}/{c}/{nm}: reference {st}' + (f' ({why})' if why else ''))
            continue
        files = read_reference(s, r, path)[4]
        cs, rec = case_state(args.root, s, r, c, nm, files)
        if cs != 'done':
            missing.append(f'{s}/{r}/{c}/{nm}: case {"not run" if cs == "absent" else cs}'
                           + (' (the reference record changed or the design differs)'
                              if cs == 'other' else ''))
            continue
        if rec.get('instance') != nm or rec.get('combo') != c:
            refuse(f'{s}/{r}/{c}/{nm}: the case file names {rec.get("combo")}/'
                   f'{rec.get("instance")}')
        if st == 'valid' and proved is not None:
            rec['reference']['proved_optimal'] = bool(proved)
        got[nm] = rec
    return got, missing


def arrays_of(recs, names):
    """Per-instance arrays of the cases recs over names."""
    R = [recs[n] for n in names]
    out = dict(ref=np.array([x['ref_makespan'] for x in R], float),
               rep=np.array([x['replay_makespan'] for x in R], float),
               build=np.array([bool(x['in_class']) for x in R]),
               off=np.array([x['off_share_A'] for x in R], float),
               proved=np.array([bool(x['reference']['proved_optimal']) for x in R]),
               orders={o: np.array([x['replay'][o]['makespan'] for x in R], float)
                       for o in ORDERS})
    br = [x['best_rule'] for x in R]
    out['rule'] = None if any(v is None for v in br) else np.array(br, float)
    nops = np.array([x['n_ops'] for x in R], float)
    out['late'] = 100.0 * np.array([x['class_failures']['later'] for x in R], float) / nops
    out['early'] = (100.0 * np.array([x['class_failures']['earlier_pair'] for x in R], float)
                    / nops)
    return out


def pair_stats(args, states, s, r):
    """Outcomes of one set and reference arm: per combination and pooled."""
    names = planned_names(s, args.n)
    P = dict(set=s, ref=r, word=WORD[(s, r)], planned=len(names), combos={}, missing=[],
             set_text=SETS[s]['text'], ref_text=REFS[r]['text'])
    per = {}
    for c in args.combos:
        recs, miss = read_cases(args, states, s, r, c, names)
        P['missing'] += miss
        per[c] = recs
        if not recs:
            P['combos'][c] = dict(n=0, planned=len(names))
            continue
        nm = [x for x in names if x in recs]
        A = arrays_of(recs, nm)
        C = cell_stats(A['ref'], A['rep'], A['rule'], A['build'], A['off'], A['orders'],
                       late=A['late'], early=A['early'])
        C['planned'] = len(names)
        C['best_order'] = {o: sum(recs[x]['best_order'] == o for x in nm) for o in ORDERS}
        C['class_failures'] = dict(later=int(sum(recs[x]['class_failures']['later'] > 0
                                                 for x in nm)),
                                   earlier_pair=int(sum(recs[x]['class_failures']['earlier_pair']
                                                        > 0 for x in nm)))
        if SETS[s]['kind'] == 'public':
            k = A['proved']
            C['proven_n'] = int(k.sum())
            if k.any():
                C['proven'] = dict(gap=boot(gap(A['rep'][k], A['ref'][k])),
                                   shares=shares(A['rep'][k], A['ref'][k]))
        P['combos'][c] = C
    # pooled: the combinations with cases, the instances present in all of them
    cs = [c for c in args.combos if per[c]]
    common = [x for x in names if cs and all(x in per[c] for c in cs)]
    if cs and common:
        As = {c: arrays_of(per[c], common) for c in cs}

        def stack(k):
            return np.stack([As[c][k] for c in cs], axis=1)
        rule = None if any(As[c]['rule'] is None for c in cs) else stack('rule')
        Q = cell_stats(stack('ref'), stack('rep'), rule, stack('build'), stack('off'),
                       {o: np.stack([As[c]['orders'][o] for c in cs], axis=1) for o in ORDERS},
                       late=stack('late'), early=stack('early'))
        Q['combos'] = cs
        if SETS[s]['kind'] == 'public':
            full = [c for c in cs if As[c]['proved'].all()]
            if full:
                ref_ = np.stack([As[c]['ref'] for c in full], axis=1)
                rep_ = np.stack([As[c]['rep'] for c in full], axis=1)
                Q['proven'] = dict(gap=boot(gap(rep_, ref_)), shares=shares(rep_, ref_),
                                   combos=full)
        P['pooled'] = Q
    vals = [C['gap']['mean'] for C in P['combos'].values() if C.get('n')]
    if vals:
        P['gap_min'], P['gap_max'] = float(min(vals)), float(max(vals))
        b = [C['buildable'] for C in P['combos'].values() if C.get('n')]
        P['build_min'], P['build_max'] = float(min(b)), float(max(b))
    return P


def infer_outputs(out_root):
    return {k: os.path.join(out_root, v) for k, v in OUT_FILES.items()}


def infer(args):
    """Every outcome of the module doc from the cases of --root. Returns the result
    dictionary."""
    partial = bool(args.partial)
    if args.out is None:
        if partial:
            refuse('--partial writes only to --out <folder>')
        out_root = ROOT_DIR
    else:
        out_root = os.path.abspath(args.out)
        P101.guard_out(out_root)
    paths = infer_outputs(out_root)
    for p in paths.values():                # a refusal leaves no stale output
        if os.path.exists(p):
            os.remove(p)
    if not partial:
        if args.n or args.combos != list(R8):
            refuse('every combination and every planned instance is required; a subset '
                   'needs --partial')
        if args.require != 'all':
            refuse('--require present needs --partial')
    states = reference_states(args)
    res = {'generator': GENERATOR, 'partial': partial, 'require': args.require,
           'mask': P101.MASK, 'mask_mode': 'nondelay', 'orders': ORDER_TEXT,
           'departure': DEPARTURE_TEXT, 'class_test': CLASS_TEXT,
           'cpsat_configuration': states['cpsat_arm'],
           'gap': '100 (x - reference) / reference per case; negative means x is shorter',
           'bootstrap': f'stats_family.instance_boot, {B} resamples, fresh '
                        f'default_rng({CI_SEED}); pooled, a drawn instance brings all its '
                        f'combinations',
           'shares': 'at or below: replay <= reference; within 1%: gap <= 1; above 5%: gap > 5',
           'left': 'mean over cases of 100 x (decisions of order A off the reference) / '
                   'decisions',
           'buildable': '% of references in the class (equal to the % that order A '
                        'reproduces exactly)',
           'outside': 'mean over cases of the % of the operations of a reference that fail '
                      'the class test: in total, because the operation starts later than its '
                      'earliest start on its machine, and because another pair could start '
                      'earlier',
           'rule': 'per-instance best of FIFO, MOR, SPT and MWKR restricted to the legal '
                   'pairs (stored exact-mask arrays)',
           'bound': 'per case the shorter of the replay and the best classical rule; both '
                    'schedules are built under the mask, so its gap to the reference is also '
                    'an upper bound on what the mask costs',
           'sets': {s: SETS[s]['text'] for s in args.sets},
           'refs': {r: REFS[r]['text'] for r in args.refs},
           'combos': list(args.combos), 'n': args.n, 'pairs': {}}
    missing = []
    for (s, r) in selected_pairs(args):
        P = pair_stats(args, states, s, r)
        missing += P['missing']
        res['pairs'][f'{s}|{r}'] = P
    if missing and args.require == 'all':
        refuse(f'{len(missing)} planned cases do not count (--require all), e.g. '
               + '; '.join(missing[:4]))
    res['complete'] = not missing
    res['missing_count'] = len(missing)
    res['missing_examples'] = missing[:20]
    if not any(C.get('n') for P in res['pairs'].values() for C in P['combos'].values()):
        refuse('no case counts; nothing to report')
    lines = macro_lines(res)
    names_m = re.findall(r'\\newcommand\{\\([^}]*)\}', '\n'.join(lines))
    res['macro_files_checked'] = check_macro_names(names_m, paths)
    from macro_io import emit
    emit(paths['macros'], lines)
    P101.write_json_atomic(paths['json'], P101.jsonable(res))
    os.makedirs(os.path.dirname(paths['md']), exist_ok=True)
    with open(paths['md'], 'w') as f:
        f.write('\n'.join(markdown(res)) + '\n')
    for k in ('json', 'md'):
        print('wrote', paths[k])
    print(f'{len(names_m)} macros' + ('; PARTIAL' if partial else '')
          + (f'; {len(missing)} planned cases do not count yet' if missing else ''))
    return res


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------

f2, u1 = P101.f2, P101.u1


def macro_lines(res):
    mark = 'PARTIAL ' if res['partial'] else ''
    if not res['complete']:
        mark += 'INCOMPLETE '
    L = [f'% Generated by {GENERATOR} -- do not hand-edit',
         '% Replay of reference schedules under the exact non-delay mask: the environment '
         'follows each reference as closely as its mask allows (three fixed orders, the '
         'shortest schedule kept), which bounds from above what the mask costs.',
         '% Gaps in %: 100 (x - reference) / reference per case, mean over instances, 95% '
         'bootstrap over instances (pooled: an instance brings its combinations); negative '
         'means x is shorter. Best rule: per-instance best of FIFO, MOR, SPT and MWKR '
         'restricted to the legal pairs.']
    if res['partial']:
        L.insert(1, '% PARTIAL: subset of combinations or instances; not for the manuscript.')
    if not res['complete']:
        L.insert(1, f'% INCOMPLETE: {res["missing_count"]} planned cases do not count yet.')
    L.append(f'\\newcommand{{\\{PREFIX}Boot}}{{{B}}}  % {mark}bootstrap resamples')
    for key, P in res['pairs'].items():
        w = P['word']
        what = f'{P["set"]}, reference {P["ref"]}'
        rows = [(c, C) for c, C in P['combos'].items() if C.get('n')]
        if 'pooled' in P:
            rows.append(('All', P['pooled']))
        for c, C in rows:
            t = (f'{what}, {"pooled over " + ",".join(C["combos"]) if c == "All" else c}')
            g, sh = C['gap'], C['shares']
            L.append(f'\\newcommand{{\\{PREFIX}N{w}{c}}}{{{C["n"]}}}'
                     f'\\newcommand{{\\{PREFIX}Cases{w}{c}}}{{{C["cases"]}}}'
                     f'  % {mark}{t}: instances, and cases (instances times combinations)')
            L.append(f'\\newcommand{{\\{PREFIX}MsRef{w}{c}}}{{{u1(C["ref_ms"])}}}'
                     f'\\newcommand{{\\{PREFIX}MsReplay{w}{c}}}{{{u1(C["replay_ms"])}}}'
                     f'  % {mark}{t}: mean makespan of the references and of the replays '
                     f'(shortest of orders A, B, C)')
            L.append(f'\\newcommand{{\\{PREFIX}Gap{w}{c}}}{{{f2(g["mean"])}}}'
                     f'\\newcommand{{\\{PREFIX}GapLo{w}{c}}}{{{f2(g["lo"])}}}'
                     f'\\newcommand{{\\{PREFIX}GapHi{w}{c}}}{{{f2(g["hi"])}}}'
                     f'  % {mark}{t}: gap of the replay to the reference, %, mean and 95% '
                     f'interval ([{g["lo"]:+.4f}, {g["hi"]:+.4f}])')
            L.append(f'\\newcommand{{\\{PREFIX}AtOrBelow{w}{c}}}{{{u1(sh["at_or_below"])}}}'
                     f'\\newcommand{{\\{PREFIX}WithinOne{w}{c}}}{{{u1(sh["within_one"])}}}'
                     f'\\newcommand{{\\{PREFIX}AboveFive{w}{c}}}{{{u1(sh["above_five"])}}}'
                     f'  % {mark}{t}: share (%) of the {sh["n"]} cases with the replay at or '
                     f'below the reference, within 1% of it, and more than 5% above it')
            L.append(f'\\newcommand{{\\{PREFIX}Build{w}{c}}}{{{u1(C["buildable"])}}}'
                     f'\\newcommand{{\\{PREFIX}Left{w}{c}}}{{{u1(C["left"])}}}'
                     f'  % {mark}{t}: share (%) of references the environment can build '
                     f'exactly, and mean share (%) of decisions of order A off the reference')
            if 'outside' in C:
                oc = C['outside']
                L.append(f'\\newcommand{{\\{PREFIX}Outside{w}{c}}}{{{u1(oc["total"])}}}'
                         f'\\newcommand{{\\{PREFIX}OutsideLater{w}{c}}}{{{u1(oc["later"])}}}'
                         f'\\newcommand{{\\{PREFIX}OutsideEarlier{w}{c}}}'
                         f'{{{u1(oc["earlier_pair"])}}}'
                         f'  % {mark}{t}: mean share (%) of the operations of a reference that '
                         f'the non-delay rule does not admit along the reference, in total, '
                         f'because the operation starts later than its earliest start on its '
                         f'machine, and because another pair could start earlier')
            if 'rule_gap' in C:
                rg = C['rule_gap']
                L.append(f'\\newcommand{{\\{PREFIX}RuleGap{w}{c}}}{{{f2(rg["mean"])}}}'
                         f'\\newcommand{{\\{PREFIX}RuleGapLo{w}{c}}}{{{f2(rg["lo"])}}}'
                         f'\\newcommand{{\\{PREFIX}RuleGapHi{w}{c}}}{{{f2(rg["hi"])}}}'
                         f'  % {mark}{t}: gap of the best classical rule to the reference, %, '
                         f'mean and 95% interval')
                bg, bs = C['bound_gap'], C['bound_shares']
                L.append(f'\\newcommand{{\\{PREFIX}BoundGap{w}{c}}}{{{f2(bg["mean"])}}}'
                         f'\\newcommand{{\\{PREFIX}BoundGapLo{w}{c}}}{{{f2(bg["lo"])}}}'
                         f'\\newcommand{{\\{PREFIX}BoundGapHi{w}{c}}}{{{f2(bg["hi"])}}}'
                         f'\\newcommand{{\\{PREFIX}BoundWithinOne{w}{c}}}{{{u1(bs["within_one"])}}}'
                         f'  % {mark}{t}: gap to the reference of the shorter of the replay and '
                         f'the best classical rule (both built under the mask, so also an upper '
                         f'bound on the best makespan the mask admits), %, mean and 95% '
                         f'interval, and share (%) of cases within 1% of the reference')
            if 'proven' in C:
                pg, ps = C['proven']['gap'], C['proven']['shares']
                n_p = C.get('proven_n', ps['n'])
                L.append(f'\\newcommand{{\\{PREFIX}OptN{w}{c}}}{{{n_p}}}'
                         f'\\newcommand{{\\{PREFIX}OptGap{w}{c}}}{{{f2(pg["mean"])}}}'
                         f'\\newcommand{{\\{PREFIX}OptGapLo{w}{c}}}{{{f2(pg["lo"])}}}'
                         f'\\newcommand{{\\{PREFIX}OptGapHi{w}{c}}}{{{f2(pg["hi"])}}}'
                         f'\\newcommand{{\\{PREFIX}OptAtOrBelow{w}{c}}}{{{u1(ps["at_or_below"])}}}'
                         f'\\newcommand{{\\{PREFIX}OptWithinOne{w}{c}}}{{{u1(ps["within_one"])}}}'
                         f'\\newcommand{{\\{PREFIX}OptAboveFive{w}{c}}}{{{u1(ps["above_five"])}}}'
                         f'  % {mark}{t}, references proven optimal only'
                         + (f' (combinations {",".join(C["proven"]["combos"])})'
                            if c == 'All' else '')
                         + ': number of these references, gap of the replay (%, mean and 95% '
                           'interval), shares (%) at or below, within 1%, more than 5% above')
        if 'gap_min' in P:
            L.append(f'\\newcommand{{\\{PREFIX}GapMin{w}}}{{{f2(P["gap_min"])}}}'
                     f'\\newcommand{{\\{PREFIX}GapMax{w}}}{{{f2(P["gap_max"])}}}'
                     f'\\newcommand{{\\{PREFIX}BuildMin{w}}}{{{u1(P["build_min"])}}}'
                     f'\\newcommand{{\\{PREFIX}BuildMax{w}}}{{{u1(P["build_max"])}}}'
                     f'  % {mark}{what}: smallest and largest mean gap of the replay (%) and '
                     f'share of buildable references (%) over the combinations')
    return L


def check_macro_names(names, out_paths):
    """Every name is the prefix plus letters, unique, and absent from every other macro
    file of the repository's paper folders (the check of p101)."""
    bad = [x for x in names if not re.fullmatch(PREFIX + r'[A-Za-z]+', x)]
    if bad:
        refuse(f'macro names that are not {PREFIX} plus letters: {bad[:5]}')
    if len(names) != len(set(names)):
        refuse(f'duplicate macro names: {sorted({x for x in names if names.count(x) > 1})[:5]}')
    own = {os.path.abspath(out_paths['macros']),
           os.path.abspath(os.path.join(ROOT_DIR, OUT_FILES['macros']))}
    files = sorted(set(glob.glob(os.path.join(ROOT_DIR, 'paper*', 'macros*.tex'))
                       + glob.glob(os.path.join(ROOT_DIR, 'paper*', '*', 'macros*.tex'))
                       + glob.glob(os.path.join(ROOT_DIR, 'manuscript', 'sections_new', '*.tex'))))
    rx = re.compile(r'\\(?:newcommand|renewcommand|providecommand|def)\s*\{?\\([A-Za-z]+)')
    for f in files:
        if os.path.abspath(f) in own:
            continue
        with open(f, errors='replace') as fh:
            clash = set(names) & set(rx.findall(fh.read()))
        if clash:
            refuse(f'macro name collision with {os.path.relpath(f, ROOT_DIR)}: '
                   f'{sorted(clash)[:5]}')
    return len(files)


def markdown(res):
    L = [f'# Replay of reference schedules under the non-delay mask ({GENERATOR})', '']
    if res['partial']:
        L += ['PARTIAL: a subset of combinations or instances. Not for the manuscript.', '']
    if not res['complete']:
        L += [f'INCOMPLETE: {res["missing_count"]} planned cases do not count yet (pending, '
              f'invalid or missing references, or cases not run), e.g.:', '']
        L += [f'- {x}' for x in res['missing_examples'][:8]] + ['']
    L += ['Exact mask (first_mask and mask_ties exact, non-delay). Orders: '
          + '; '.join(f'{o} {t}' for o, t in res['orders'].items())
          + '. The replay of a case is the shortest of the three. '
          f'Gap: {res["gap"]}. Interval: {res["bootstrap"]}. Shares: {res["shares"]}. '
          f'Buildable: {res["buildable"]}. Outside the rule: {res["outside"]}. '
          f'Off the reference: {res["left"]}; '
          f'{res["departure"]}. Rule: {res["rule"]}.', '']
    if res.get('cpsat_configuration'):
        L += [f'CP-SAT configuration of cpsat_rule600: {res["cpsat_configuration"]}.', '']

    def g(x):
        return f'{f2(x["mean"])} [{f2(x["lo"])}, {f2(x["hi"])}]'
    for key, P in res['pairs'].items():
        L += [f'## {P["set"]}: {P["set_text"]}; reference {P["ref"]}: {P["ref_text"]}', '',
              '| combination | cases / planned | reference | replay | gap (%) | A / B / C '
              'mean gap (%) | at or below (%) | within 1% (%) | above 5% (%) | buildable (%) '
              '| decisions off (%) | best rule gap (%) | replay or rule, gap (%) '
              '| reference operations outside the rule (%): total (later / earlier pair) |',
              '|' + '---|' * 14]
        rows = list(P['combos'].items()) + ([('pooled', P['pooled'])] if 'pooled' in P else [])
        for c, C in rows:
            if not C.get('n'):
                L.append(f'| {c} | 0 / {C.get("planned", P["planned"])} |' + ' -- |' * 12)
                continue
            sh = C['shares']
            od = ' / '.join(f2(C['orders'][o]['gap']) for o in ORDERS)
            lab = c if c != 'pooled' else f'pooled ({",".join(C["combos"])})'
            cnt = (f'{C["n"]} / {C["planned"]}' if c != 'pooled'
                   else f'{C["cases"]} ({C["n"]} instances)')
            L.append(f'| {lab} | {cnt} | {u1(C["ref_ms"])} | {u1(C["replay_ms"])} | '
                     f'{g(C["gap"])} | {od} | {u1(sh["at_or_below"])} | {u1(sh["within_one"])} '
                     f'| {u1(sh["above_five"])} | {u1(C["buildable"])} | {u1(C["left"])} | '
                     f'{g(C["rule_gap"]) if "rule_gap" in C else "--"} | '
                     f'{g(C["bound_gap"]) if "bound_gap" in C else "--"} | '
                     + (f'{u1(C["outside"]["total"])} ({u1(C["outside"]["later"])} / '
                        f'{u1(C["outside"]["earlier_pair"])}) |' if 'outside' in C else '-- |'))
        if SETS[P['set']]['kind'] == 'public':
            L += ['', 'References proven optimal only:', '',
                  '| combination | proven | gap (%) | at or below (%) | within 1% (%) | '
                  'above 5% (%) |', '|' + '---|' * 6]
            for c, C in rows:
                if 'proven' not in C:
                    continue
                pr, sh = C['proven'], C['proven']['shares']
                lab = c if c != 'pooled' else f'pooled ({",".join(pr["combos"])})'
                L.append(f'| {lab} | {C.get("proven_n", sh["n"])} | {g(pr["gap"])} | '
                         f'{u1(sh["at_or_below"])} | {u1(sh["within_one"])} | '
                         f'{u1(sh["above_five"])} |')
        L.append('')
    return L


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('phase', choices=['run', 'infer', 'plan'])
    ap.add_argument('--root', default=DEFAULT_ROOT, help='folder of the cases')
    ap.add_argument('--sets', default=','.join(SETS))
    ap.add_argument('--refs', default=','.join(REFS))
    ap.add_argument('--combos', default=','.join(R8))
    ap.add_argument('--n', type=int, default=0,
                    help='the first n planned instances of each set (smoke only)')
    ap.add_argument('--cpsat_arm', default=None,
                    help='CP-SAT configuration of cpsat_rule600; must equal the pilot\'s choice')
    ap.add_argument('--part', type=int, default=None)
    ap.add_argument('--parts', type=int, default=1)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--require', default='all', choices=['all', 'present'],
                    help='infer: all refuses when a planned reference or case is missing; '
                         'present reports what exists (with --partial)')
    ap.add_argument('--partial', action='store_true',
                    help='infer: a subset of combinations or instances, or --require present, '
                         'under --out')
    ap.add_argument('--out', default=None, help='infer: root of the outputs')
    args = ap.parse_args(_ARGV)
    P101.check_mask_options()
    args.sets = P101.parse_list(args.sets, SETS, 'sets')
    args.refs = P101.parse_list(args.refs, REFS, 'reference arms')
    args.combos = P101.parse_list(args.combos, R8, 'combinations')
    args.n = args.n or None
    if args.part is not None and not 0 <= args.part < args.parts:
        sys.exit(f'--part {args.part} outside 0..{args.parts - 1}')
    if not selected_pairs(args):
        sys.exit('no reference arm covers the selected sets')
    os.chdir(ROOT_DIR)
    {'run': run, 'infer': infer, 'plan': plan}[args.phase](args)


if __name__ == '__main__':
    main()
