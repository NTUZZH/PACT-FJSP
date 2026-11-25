"""The control arm with the active-schedule mask (xmask-active) in the exact environment.

  * registry: the arm is a secondary arm, laid out as xmask-pair, seeds 301-303, with the
    stored counterpart jointact, after the earlier arms;
  * mask: under --mask_mode active with --first_mask exact --mask_ties exact the legal
    set at the first decision is the active set computed on the true (hour) axis from the
    descriptor alone (processing start strictly earlier than the earliest completion
    among the compatible pairs); it contains the non-delay set and is larger than it on
    instances with setups or windows; reset() restores it; with the earlier first mask
    every compatible pair is legal at the first decision. At every later decision of
    random rollouts the legal set is the active set of an independent reference state;
  * evaluation: the plan gives the arm the 10x25 set of xmask-pair, and the queue's
    command of its job, run on one instance with a network built from the arm's
    configuration, builds every environment of the rollout with mask_mode active and the
    exact options; the files the analysis reads are the files the plan's jobs write; no
    glob of an analysis script matches the arm's files except the ones that take every
    file of a folder;
  * analysis (scripts/p50_actmask_analysis.py, exact environment) on synthetic result
    trees with a known answer: differences, intervals (stats_family.hier_boot and seed_t
    with the streams of p95), counts, pooled row, a further size, the refusals and the
    skip, macro names and comments, no absolute path in any output, nothing written
    outside the output root; the earlier-environment branch writes the stored files
    byte for byte.

Run from the repository root:  python -m pytest tests/test_p50_actmask_exact.py -q
CPU only. The tests that read stored data, results or configurations skip without them;
nothing is written in the repository.
"""
import builtins  # noqa: F401  (used by the evaluation wrapper text below)
import contextlib
import fnmatch
import glob
import importlib
import io
import json
import os
import re
import subprocess
import sys
import time

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

_argv, sys.argv = sys.argv, sys.argv[:1] + ['--device', 'cpu']
try:
    from params import configs  # noqa: E402
    import data_utils  # noqa: E402,F401
finally:
    sys.argv = _argv
configs.device = 'cpu'

import torch  # noqa: E402

import fjsp_env_same_op_nums  # noqa: E402
fjsp_env_same_op_nums.EnvState.device = torch.device('cpu')

from constraint_family.descriptor import ConstraintDescriptor, IDLE_CLASS, make_descriptor  # noqa: E402
from constraint_family.fjsp_env_family import FJSPEnvFamily  # noqa: E402
from constraint_family.reference_sim import push_past_outages_scalar  # noqa: E402
from ppvc_instance_generator import (DEFAULT_FACTORY, build_factory, load_instance,  # noqa: E402
                                     ppvc_instance_generator)

import arm_registry as REG  # noqa: E402
import stats_family as SF  # noqa: E402

PY = sys.executable
R8 = list(REG.COMBOS)
SIX = ','.join(REG.TRAIN_COMBOS)
ARM = 'xmask-active'
SEEDS3 = (301, 302, 303)
SEEDS5 = (301, 302, 303, 304, 305)
EXTRA = os.path.join(ROOT, 'scripts', 'run_exactmask_extra_queue.sh')
EVAL_QUEUE = os.path.join(ROOT, 'scripts', 'run_eval_newseeds.sh')
GROUP_ENV = ('EVAL_GROUPS', 'PUBLIC', 'PUBSENS', 'PUBEQ', 'PUBACT', 'LEGACY45', 'CROSS',
             'HGNN', 'HGNN_LR', 'HGNN_SOLO')
needs_results = pytest.mark.skipif(
    not os.path.isdir(os.path.join(ROOT, 'test_results/FAMILY/10x25')) or
    not os.path.isdir(os.path.join(ROOT, 'data/FAMILY/10x25')),
    reason='needs the stored results and the instance data')
EXACT = dict(first_mask='exact', mask_ties='exact')
N_M = len(build_factory(DEFAULT_FACTORY)[0])


def clean_env(**extra):
    env = {k: v for k, v in os.environ.items() if k not in GROUP_ENV and not k.startswith('PACT_')}
    env.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               CUDA_VISIBLE_DEVICES='')
    env.update(extra)
    return env


# ================================================================== registry
def test_registry_entry_of_the_arm():
    arms = list(REG.SECONDARY_ARMS)
    assert arms[-1] == ARM
    assert arms[:-1] == ['xmask-eq', 'xmask-wthird', 'xmask-sparse', 'xmask-dzero', 'xmask-dmax',
                         'xmask-film', 'xmask-dr', 'xmask-all8', 'xmask-pair']
    recipe, legacy = REG.SECONDARY_ARMS[ARM]
    assert legacy == 'jointact' and '--mask_mode active' in recipe
    assert REG.OTHER_ARMS['jointact'] == ((301, 302, 303), False)
    assert REG.SECONDARY_SEEDS == SEEDS3
    for s in SEEDS3:
        name = REG.secondary_model(ARM, s)
        assert name == f'{ARM}-s{s}'
        assert REG.secondary_paths(ARM, s) == (f'trained_network/FAMILY/10x25+family+{name}.pth',
                                                f'train_log/FAMILY_{name}.log')
        assert REG.secondary_paths(ARM, s) == tuple(
            p.replace('xmask-pair', ARM) for p in REG.secondary_paths('xmask-pair', s))
        assert REG.other_model('jointact', s) == f'jointact-s{s}'
    for bad in (300, 304, 305):
        with pytest.raises(ValueError):
            REG.secondary_model(ARM, bad)


# ================================================================== mask
def make(jl, pt, meta, desc, **kw):
    env = FJSPEnvFamily(jl.shape[0], pt.shape[1], family_mode=True, **kw)
    env.delta_mode = 'min'
    env.reward_delta_env = True
    env.set_initial_data([jl], [pt], descriptor_list=[desc], op_type_list=[meta['op_type']],
                         mch_type_list=[meta['mch_type']])
    return env


def legal(env):
    return ~env.dynamic_pair_mask[0]


class Reference:
    """The state of one schedule on the true (hour) axis, sharing nothing with the
    environment: job ready times, machine free times and last classes, updated by the
    transition of reference_sim.simulate (attached setup, push past outages, lag after the
    operation). table() gives the processing start of every pair, the non-delay set
    (start equal to the earliest start) and the active set (start strictly earlier than
    the earliest completion), both among the compatible pairs."""

    def __init__(self, jl, pt, desc):
        self.jl = np.asarray(jl, dtype=int)
        self.pt = np.asarray(pt, dtype=float)
        self.desc = desc
        self.J, self.M = len(self.jl), self.pt.shape[1]
        self.first = np.concatenate([[0], np.cumsum(self.jl)[:-1]]).astype(int)
        self.next_op = self.first.copy()
        self.job_ready = np.zeros(self.J)
        self.mch_free = np.zeros(self.M)
        self.mch_last = np.full(self.M, IDLE_CLASS, dtype=int)

    def start(self, j, k):
        op = self.next_op[j]
        if op >= self.first[j] + self.jl[j] or self.pt[op, k] <= 0:
            return np.inf
        d = self.desc
        s = float(d.setup[self.mch_last[k], d.op_class[op]]) if d.has_setup else 0.0
        b = max(self.job_ready[j], self.mch_free[k])
        if d.has_window and d.windows is not None:
            b = push_past_outages_scalar(b, s + self.pt[op, k], d.windows[k])
        return b + s

    def table(self):
        st = np.array([[self.start(j, k) for k in range(self.M)] for j in range(self.J)])
        ct = np.full_like(st, np.inf)
        fin = np.isfinite(st)
        for j, k in np.argwhere(fin):
            ct[j, k] = st[j, k] + self.pt[self.next_op[j], k]
        return st, fin & (st == st.min()), fin & (st < ct.min())

    def step(self, j, k):
        op = self.next_op[j]
        st = self.start(j, k)
        ct = st + self.pt[op, k]
        self.job_ready[j] = ct + (self.desc.lag[op] if self.desc.has_lag else 0.0)
        self.mch_free[k] = ct
        self.mch_last[k] = self.desc.op_class[op]
        self.next_op[j] += 1
        return st


def small_instances():
    """Generated factory instances of three and four modules under N, S, W and LSW."""
    out = []
    for n_mod, seed in ((3, 11), (4, 12), (3, 13), (4, 14)):
        jl, pt, meta = ppvc_instance_generator(n_modules=n_mod, seed=seed)
        for r in ('N', 'S', 'W', 'LSW'):
            out.append((f'{r}/{n_mod} modules/seed {seed}', jl, pt, meta,
                        make_descriptor(r, jl, pt, meta, N_M, seed=seed * 1000)))
    return out


def stored_instances():
    """The first three test instances at 10x25 under S, W and LSW, when stored."""
    out = []
    d = os.path.join(ROOT, 'data/FAMILY/10x25/desc')
    if not os.path.isdir(d):
        return out
    for r in ('S', 'W', 'LSW'):
        for idx in range(3):
            with open(os.path.join(d, r, f'instance_{idx:03d}.desc.json')) as f:
                rec = json.load(f)
            jl, pt, meta = load_instance(os.path.join(ROOT, rec['base_instance']))
            out.append((f'{r}/10x25/{idx}', jl, pt, meta,
                        ConstraintDescriptor.from_json(rec, setup_convention='attached')))
    return out


@pytest.fixture(autouse=False)
def no_mask_vars(monkeypatch):
    for v in ('PACT_FIRST_MASK', 'PACT_MASK_TOL'):
        monkeypatch.delenv(v, raising=False)


def test_first_decision_legal_set_is_the_active_set(no_mask_vars):
    wider_than_nondelay, earlier_wider = 0, 0
    cases = small_instances() + stored_instances()
    for label, jl, pt, meta, desc in cases:
        act = make(jl, pt, meta, desc, mask_mode='active', **EXACT)
        nd = make(jl, pt, meta, desc, mask_mode='nondelay', **EXACT)
        old = make(jl, pt, meta, desc, mask_mode='active', first_mask='legacy', mask_ties='exact')
        st, want_nd, want_act = Reference(jl, pt, desc).table()
        assert np.array_equal(legal(act), want_act), label
        assert np.array_equal(legal(nd), want_nd), label
        assert want_nd.any() and not (want_nd & ~want_act).any(), label     # non-delay inside
        assert np.array_equal(act.state.dynamic_pair_mask_tensor.numpy(), act.dynamic_pair_mask)
        assert abs(float(act.next_schedule_time[0]) / act._slope - st.min()) < 1e-6, label
        compat = ~act.candidate_process_relation[0]
        assert np.array_equal(legal(old), compat), label    # earlier first mask: all compatible
        wider_than_nondelay += int((want_act & ~want_nd).any())
        earlier_wider += int((compat & ~want_act).any())
        act.reset()
        assert np.array_equal(legal(act), want_act), label
        assert np.array_equal(act.state.dynamic_pair_mask_tensor.numpy(), act.dynamic_pair_mask)
    # the instances are sharp enough to tell the three sets apart
    assert wider_than_nondelay > 0 and earlier_wider > 0, (wider_than_nondelay, earlier_wider)


def test_every_decision_of_random_rollouts_uses_the_active_set(no_mask_vars):
    rng = np.random.default_rng(5)
    n_dec, later, idle_choice = 0, 0, 0
    for label, jl, pt, meta, desc in small_instances():
        env = make(jl, pt, meta, desc, mask_mode='active', **EXACT)
        ref = Reference(jl, pt, desc)
        M = pt.shape[1]
        step = 0
        while not env.done().all():
            st, want_nd, want_act = ref.table()
            assert np.array_equal(legal(env), want_act), (label, step)
            assert not (want_nd & ~want_act).any(), (label, step)
            a = int(rng.choice(np.flatnonzero(want_act.reshape(-1))))
            j, k = divmod(a, M)
            op = int(env.candidate[0, j])
            s_ref = ref.step(j, k)
            env.step(np.array([a]))
            # the environment schedules the pair where the reference does
            assert abs(float(env.true_op_ct[0, op] - pt[op, k]) - s_ref) < 1e-9, (label, step)
            idle_choice += int(s_ref > st.min() + 1e-9)
            later += int(step > 0)
            step += 1
            n_dec += 1
        assert step == int(np.sum(jl)), label
    # some dispatched pairs start after the decision instant: the mask is not non-delay
    assert later > 0 and idle_choice > 0, (n_dec, idle_choice)


# ================================================================== evaluation plan
@contextlib.contextmanager
def plan_module(check=True, **env):
    """The plan module reloaded under env (the group switches cleared first), with the
    completeness check run once; reloaded with the defaults afterwards."""
    keep = {k: os.environ.pop(k, None) for k in GROUP_ENV}
    cwd = os.getcwd()
    os.chdir(ROOT)
    import eval_newseeds_plan as P
    try:
        os.environ.update(env)
        P = importlib.reload(P)
        P.EXPECTED.clear()
        if check:
            with contextlib.redirect_stdout(io.StringIO()):
                P.check()
        yield P
    finally:
        for k in GROUP_ENV:
            os.environ.pop(k, None)
        for k, v in keep.items():
            if v is not None:
                os.environ[k] = v
        importlib.reload(P)
        os.chdir(cwd)


def build_cmd(*args):
    """The evaluation queue's build_cmd function run alone: the command it builds."""
    with open(EVAL_QUEUE) as f:
        fn = re.search(r'^build_cmd\(\) \{.*?^\}\n', f.read(), re.S | re.M).group(0)
    script = ('PY=/py; MASK_OPTS="--first_mask exact --mask_ties exact"\n' + fn
              + 'build_cmd "$@"; printf "%s\\n" "${CMD[@]}"\n')
    r = subprocess.run(['bash', '-c', script, 'build_cmd'] + [str(a) for a in args],
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    return r.stdout.split('\n')[:-1]


def job_cmd(j, root='/stage/out', **change):
    j = dict(j, **change)
    return build_cmd('7', j['device'], j['omp'], j['tt'], j['extra'], j['model'], j['methods'],
                     j['data'], j['base'], j['tag'], j['limit'], j['regimes'], j['token_mode'],
                     j['blind'], j['best_known'], j['sched'], 'exact', root, j['rule_set'])


def arm_files(seeds=SEEDS3):
    d = 'test_results/FAMILY/10x25_xmask'
    return ([f'{d}/Result_greedy+10x25+family+{ARM}-s{s}_{r}.npy' for s in seeds for r in R8]
            + [f'{d}/summary_10x25+family+{ARM}-s{s}_xmask.md' for s in seeds])


@needs_results
def test_plan_evaluates_the_arm_like_xmask_pair_under_its_own_configuration():
    with plan_module(EVAL_GROUPS='secondary') as P:
        js = [j for j in P.jobs() if f'+{ARM}-' in j['model']]
        pair = {j['model'][-4:]: j for j in P.jobs() if '+xmask-pair-' in j['model']}
        expected = [f for f in P.EXPECTED if f'+{ARM}-' in f]
        sets = dict(P.SECONDARY_EVAL)
    assert sets[ARM] == sets['xmask-pair'] == [('10x25', R8)]
    assert sorted(j['id'] for j in js) == [f'exact:{ARM}_10x25:{ARM}-s{s}' for s in SEEDS3]
    for j in js:
        s = int(j['model'][-3:])
        name = f'{ARM}-s{s}'
        assert j['model'] == REG.PREFIX + name and (j['ckpt'], j['log']) == REG.secondary_paths(ARM, s)
        assert (j['group'], j['env'], j['device'], j['omp'], j['tt'], j['methods'], j['tag'],
                j['folder'], j['n'], j['regimes'].split()) == \
            ('secondary', 'exact', 'cpu', 1, 1, 'greedy', '_xmask', '10x25_xmask', 100, R8)
        # the job of xmask-pair of the same seed with the run name changed
        pj = pair[j['model'][-4:]]
        for k in j:
            if k in ('id', 'campaign', 'model', 'ckpt', 'log', 'files'):
                continue
            assert j[k] == pj[k], k
        assert j['files'] == [f.replace('xmask-pair', ARM) for f in pj['files']]
        cmd = job_cmd(j)
        assert '--mask_mode' not in cmd and cmd[-4:] == ['--first_mask', 'exact', '--mask_ties', 'exact']
        assert cmd[cmd.index('--family_model') + 1] == REG.PREFIX + name
    assert sorted(f for j in js for f in j['files']) == sorted(arm_files()) == sorted(expected)


@needs_results
def test_analysis_reads_the_files_the_plan_writes(monkeypatch):
    monkeypatch.setenv('PACT_ENV', 'exact')
    monkeypatch.setenv('PACT_SEEDS', '301-305')
    with plan_module(check=False, EVAL_GROUPS='principal secondary') as P:
        planned = {f for j in P.jobs() for f in j['files']}
    import p50_actmask_analysis as A
    A.check_layout()
    sets = A.plan_sets()
    assert sets == [('10x25', R8)]
    read = set()
    for folder, combos in sets:
        arm, pol = A.runs_of(A.compared_seeds(), folder)
        for runs in (arm, pol):
            for run in runs.values():
                read |= {run.array(c) for c in combos} | {run.summary()}
    assert A.compared_seeds() == SEEDS3
    assert len(read) == 2 * 3 * 9 and read <= planned, sorted(read - planned)[:5]
    assert {f for f in read if f'+{ARM}-' in f} == set(arm_files())


# Globs that take every file of a folder on purpose: staleness of the generated files of
# paper/ (p9), the spread of the time column (p19), the rule folders of the rule
# comparison (rulecmp_plan), counts and docstrings.
GENERIC = {'Result_*.npy', 'Result_*_W.npy', 'Result_greedy*.npy', 'Result_greedy+*.npy',
           'summary_*.md'}


def test_no_named_glob_of_an_analysis_script_matches_the_arms_files():
    names = [os.path.basename(f) for f in arm_files()]
    rx = re.compile(r"(?:Result|summary)_[^'\"\s`]*\*[^'\"\s`]*")
    hits, n = {}, 0
    scripts = sorted(glob.glob(os.path.join(ROOT, 'scripts', '*.py'))
                     + glob.glob(os.path.join(ROOT, 'scripts', '*.sh')))
    for path in scripts:
        with open(path, errors='replace') as f:
            text = f.read()
        for m in rx.findall(text):
            n += 1
            pat = m.rstrip(',);:.')
            if '{' in pat:          # a pattern of one run or rule, filled in at run time
                continue
            for nm in names:
                if fnmatch.fnmatchcase(nm, pat):
                    hits.setdefault(pat, set()).add(os.path.basename(path))
    # the scan counts the globs of the scripts present (98 in the repository, fewer in the
    # release, which holds the shipped scripts only); the finder is checked on a known line
    assert scripts and n > 0
    assert rx.findall("glob('Result_greedy+*_N.npy')") == ['Result_greedy+*_N.npy']
    assert set(hits) <= GENERIC, {k: sorted(v) for k, v in hits.items() if k not in GENERIC}
    # the stored counterpart's names and the arm's do not match each other's globs
    assert not fnmatch.fnmatchcase(names[0], 'Result_greedy+10x25+family+jointact-*.npy')


# ------------------------------------------------------------------ the job's command, run
def train_args(tmp_path, regs, upd, seed, name, flags, *extra):
    """The arguments the extra queue's train_cmd passes to train_family.py."""
    with open(EXTRA) as f:
        text = f.read()
    fn = re.search(r'^train_cmd\(\) \{.*?^\}\n', text, re.S | re.M).group(0)
    exe = tmp_path / 'args.sh'
    exe.write_text('#!/bin/sh\nfor a in "$@"; do printf "%s\\n" "$a"; done\n')
    exe.chmod(0o755)
    script = fn + f'PY={exe} MASK_OPTS="--first_mask exact --mask_ties exact" DR_SHIFT=x train_cmd "$@"\n'
    r = subprocess.run(['bash', '-c', script, 'train_cmd', regs, str(upd), str(seed), name, flags]
                       + list(extra), capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    lines = r.stdout.split('\n')[:-1]
    assert lines[0] == 'train_family.py'
    return lines[1:]


# A network with the arm's resolved configuration (the edits of train_family.py applied:
# input widths, station count, the two mask options), saved with that configuration
# where eval_family.py looks for them. No training.
SAVE_NET = r'''
import json, os, sys
sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
root, out, args = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
sys.path.insert(0, root)
sys.argv = ['train_family.py'] + args
from params import configs
from ppvc_instance_generator import DEFAULT_FACTORY, build_factory
from constraint_family.fjsp_env_family import resolve_mask_option
import torch
torch.set_num_threads(1)
torch.set_default_dtype(torch.float32)
torch.set_default_device('cpu')
configs.fea_j_input_dim, configs.fea_m_input_dim = 12, 10
configs.n_m = len(build_factory(DEFAULT_FACTORY)[0])
configs.first_mask = resolve_mask_option('first_mask', configs.first_mask)
configs.mask_ties = resolve_mask_option('mask_ties', configs.mask_ties)
from model.PPO import PPO_initialize
torch.manual_seed(0)
ppo = PPO_initialize()
name = '10x25+family+' + configs.model_suffix
for d in ('trained_network/FAMILY', 'train_log/FAMILY'):
    os.makedirs(os.path.join(out, d), exist_ok=True)
torch.save(ppo.policy.state_dict(), os.path.join(out, 'trained_network/FAMILY', name + '.pth'))
with open(os.path.join(out, 'train_log/FAMILY', 'config_' + name + '.json'), 'w') as f:
    json.dump(vars(configs), f, indent=1, default=str)
print(sum(p.numel() for p in ppo.policy.parameters()))
'''

# eval_family.py run as __main__ with the given arguments; every FJSPEnvFamily it builds
# records its mask_mode, first_mask and mask_ties (the class is patched at the first
# import after its module has finished loading, before any environment exists)
EVAL_WRAP = r'''
import builtins, json, os, runpy, sys
sys.dont_write_bytecode = True
root, args = sys.argv[1], json.loads(sys.argv[2])
sys.path.insert(0, root)
seen = []
_imp = builtins.__import__


def _hook(name, *a, **k):
    m = _imp(name, *a, **k)
    mod = sys.modules.get('constraint_family.fjsp_env_family')
    if mod is not None and hasattr(mod, 'FJSPEnvFamily') and not getattr(mod, '_recorded', False):
        mod._recorded = True
        init = mod.FJSPEnvFamily.__init__

        def rec(self, *x, **kw):
            seen.append([kw.get('mask_mode', 'nondelay'), kw.get('first_mask'), kw.get('mask_ties')])
            init(self, *x, **kw)
        mod.FJSPEnvFamily.__init__ = rec
    return m


builtins.__import__ = _hook
sys.argv = ['eval_family.py'] + args
try:
    runpy.run_path(os.path.join(root, 'eval_family.py'), run_name='__main__')
finally:
    print('ENVS ' + json.dumps(seen))
'''


@needs_results
def test_job_command_evaluates_under_the_active_mask(tmp_path):
    """The plan's job of xmask-active-s301, built by the queue's build_cmd, run on the first
    instance of S with a network built from the arm's configuration (the full run's
    arguments of the extra queue): every environment of the rollout has mask_mode active
    and the exact options, which come from the run's configuration."""
    args = train_args(tmp_path, SIX, 2000, 301, f'{ARM}-s301', '--mask_mode active') + ['--device', 'cpu']
    work = tmp_path / 'work'
    r = subprocess.run([PY, '-B', '-c', SAVE_NET, ROOT, str(work), json.dumps(args)], cwd=str(tmp_path),
                       env=clean_env(), capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-3000:]
    assert int(r.stdout.split()[-1]) == 32786
    cfg = json.load(open(work / 'train_log/FAMILY' / f'config_10x25+family+{ARM}-s301.json'))
    assert (cfg['mask_mode'], cfg['first_mask'], cfg['mask_ties']) == ('active', 'exact', 'exact')
    with plan_module(check=False, EVAL_GROUPS='secondary') as P:
        j = next(x for x in P.jobs() if x['id'] == f'exact:{ARM}_10x25:{ARM}-s301')
    out = work / 'out'
    cmd = job_cmd(j, root=str(out), limit=1, regimes='S',
                  data=os.path.join(ROOT, j['data']), base=os.path.join(ROOT, j['base']))
    ev = cmd[cmd.index('eval_family.py') + 1:]
    assert '--mask_mode' not in ev
    r = subprocess.run([PY, '-B', '-c', EVAL_WRAP, ROOT, json.dumps(ev)], cwd=str(work),
                       env=clean_env(), capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, (r.stdout[-2000:], r.stderr[-3000:])
    seen = json.loads(next(x for x in r.stdout.splitlines() if x.startswith('ENVS '))[5:])
    assert seen and all(x == ['active', 'exact', 'exact'] for x in seen), seen
    folder = out / '10x25_xmask'
    a = np.load(folder / f'Result_greedy+10x25+family+{ARM}-s301_S.npy')
    assert a.shape == (1, 2) and a[0, 0] > 0
    assert 'mask: first_mask=exact mask_ties=exact' in open(
        folder / f'summary_10x25+family+{ARM}-s301_xmask.md').read()


# ================================================================== analysis
N_INST = 6
BOOT = '200'
T0 = time.time() - 3600                # checkpoints are older than every array
# known difference of the arm from the policy in percent, per combination; W is zero
EFFECT = {'N': 1.0, 'L': 0.5, 'S': 2.5, 'W': 0.0, 'LS': -0.75, 'LW': 1.5, 'SW': -0.25, 'LSW': 3.0}


def base_ms(seed, combo, folder):
    rng = np.random.default_rng(seed * 100 + R8.index(combo) + 17 * (folder != '10x25'))
    k = 10.0 if seed > 303 else 1.0    # a comparison across seeds would show
    return k * rng.integers(100, 160, N_INST).astype(float)


def write(root, rel, text):
    p = os.path.join(root, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'w') as f:
        f.write(text)
    return p


def save(root, rel, ms):
    p = os.path.join(root, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    np.save(p, np.stack([np.asarray(ms, float), np.full(len(ms), 0.5)], axis=1))


def cfg_of(seed, mask_mode):
    c = dict(seed_train=seed, max_updates=2000, num_envs=20, family_regimes=SIX,
             first_mask='exact', mask_ties='exact')
    if mask_mode:
        c['mask_mode'] = mask_mode
    return c


def write_run(root, run, cfg, A):
    ck = write(root, run.ckpt, 'checkpoint')
    os.utime(ck, (T0, T0))
    write(root, run.log, '[train_family] done in 1.39 h; best rel=0.876400\n')
    write(root, run.config, json.dumps(cfg))
    write(root, run.summary(), f'# FAMILY eval\n\n{A.SUMMARY_MASK}\n\n| regime | method |\n')


def policy_run(A, s, folder):
    return A.Run(REG.model('released', None, s, 'exact'), s, folder)


def build(root, monkeypatch, A, sets=(('10x25', R8),), noise=0.0):
    """An exact-environment tree: the policy on seeds 301-305 (seeds 304 and 305 ten times
    larger), the arm on 301-303 with makespans (1 + EFFECT/100) times the policy's, plus
    noise times a fixed normal draw per seed, instance and combination when given."""
    import eval_newseeds_plan as PLAN
    monkeypatch.setenv('PACT_OUT', os.path.join(root, 'out'))
    rng = np.random.default_rng(3)
    for folder, combos in sets:
        data = PLAN.FACTORY[folder][0]
        for r in R8:
            for i in range(N_INST):
                write(root, f'{data}/desc/{r}/instance_{i:03d}.desc.json',
                      json.dumps({'base_instance': f'data/PPVC/{folder}+ppvc-mixed/instance_{i:03d}',
                                  'regime': r}))
        for s in SEEDS5:
            pol = policy_run(A, s, folder)
            write_run(root, pol, cfg_of(s, 'nondelay' if s % 2 else None), A)
            for c in combos:
                save(root, pol.array(c), base_ms(s, c, folder))
        for s in SEEDS3:
            arm = A.Run(REG.secondary_model(ARM, s), s, folder)
            write_run(root, arm, cfg_of(s, 'active'), A)
            for c in combos:
                b = base_ms(s, c, folder)
                z = rng.standard_normal(N_INST)
                ms = b if (EFFECT[c] == 0 and noise == 0) else \
                    b * (1 + EFFECT[c] / 100) * (1 + noise * z / 100)
                save(root, arm.array(c), ms)


@pytest.fixture
def A(monkeypatch):
    for v in ('PACT_ENV', 'PACT_RESULTS', 'PACT_SEEDS', 'PACT_OUT'):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv('PACT_ENV', 'exact')
    monkeypatch.setenv('PACT_SEEDS', '301-305')
    import p50_actmask_analysis as mod
    return mod


def run_a(A, root, *extra):
    return A.main(['--root', root, '--boot', BOOT] + list(extra))


def outs(root):
    return [os.path.join(root, 'out', rel) for rel in
            ('manuscript/macros_actmask.tex', 'manuscript/tables/actmask_table.tex',
             'results/active_mask_arm.json', 'results/active_mask_arm.md')]


def macros(path):
    return dict(re.findall(r'\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}', open(path).read()))


def test_known_differences_counts_and_pooled_row(tmp_path, monkeypatch, A):
    root = str(tmp_path)
    build(root, monkeypatch, A)
    res = run_a(A, root)
    assert res['status'] == 'computed'
    m = macros(outs(root)[0])
    assert (m['ActWorseN'], m['ActBetterN'], m['ActCombN']) == ('5', '2', '8')
    assert (m['ActGapLo'], m['ActGapHi']) == ('-0.75', '3.00')
    assert (m['ActSDelta'], m['ActSLo'], m['ActSHi'], m['ActNDelta']) == ('+2.50',) * 3 + ('+1.00',)
    pooled = np.mean(list(EFFECT.values()))
    assert (m['ActPoolDelta'], m['ActPoolLo'], m['ActPoolHi']) == (A.f2(pooled),) * 3 == ('+0.94',) * 3
    assert (m['ActSeedN'], m['ActInstN'], m['ActBoot']) == ('3', str(N_INST), BOOT)
    j = json.load(open(outs(root)[2]))
    x = j['folders']['10x25']
    assert j['seeds'] == list(SEEDS3) and j['environment'] == 'exact' and j['boot'] == int(BOOT)
    assert x['counts']['longer_combos'] == ['N', 'L', 'S', 'LW', 'LSW']
    assert x['counts']['shorter_combos'] == ['LS', 'SW'] and x['counts']['neither'] == 1
    for r in R8:
        c = x['components'][r]
        assert abs(c['mean'] - EFFECT[r]) < 1e-9 and (c['k'], c['n']) == (3, N_INST)
        assert c['direction'] == ('behind' if EFFECT[r] > 0 else 'ahead' if EFFECT[r] < 0 else 'neither')
    assert x['components']['W'] == dict(x['components']['W'], mean=0.0, lo=0.0, hi=0.0)
    assert x['runs']['policy'] == [f'xmask-v1-s{s}' for s in SEEDS3]
    t = open(outs(root)[1]).read()
    assert '$S$ & $+2.50$ & $[+2.50,\\,+2.50]$ \\\\' in t and 'all eight & $+0.94$' in t
    assert '\\toprule' in t and '\\bottomrule' in t and 'multicolumn' not in t
    md = open(outs(root)[3]).read()
    assert '| LS |' in md and 'above zero (the arm longer) in 5 of 8' in md


def test_intervals_are_the_hierarchical_bootstrap_of_stats_family(tmp_path, monkeypatch, A):
    """With noise, every interval equals stats_family.hier_boot (a fresh default_rng(11),
    B replicates) and seed_t on the differences recomputed here from the arrays."""
    root = str(tmp_path)
    build(root, monkeypatch, A, noise=1.5)
    run_a(A, root)
    x = json.load(open(outs(root)[2]))['folders']['10x25']
    D = {}
    for r in R8:
        D[r] = np.stack([SF.rel_gap(
            np.load(os.path.join(root, A.Run(REG.secondary_model(ARM, s), s, '10x25').array(r)))[:, 0],
            np.load(os.path.join(root, policy_run(A, s, '10x25').array(r)))[:, 0]) for s in SEEDS3])
        hb, t = SF.hier_boot(D[r], B=int(BOOT), seed=11), SF.seed_t(D[r])
        c = x['components'][r]
        assert (c['mean'], c['lo'], c['hi']) == (hb['mean'], hb['lo'], hb['hi']), r
        assert (c['t_lo'], c['t_hi']) == (t['lo'], t['hi']) and c['scheme'] == 'hierarchical'
    hb = SF.hier_boot(np.stack([D[r] for r in R8], axis=2), B=int(BOOT), seed=11)
    p = x['pooled']
    assert (p['mean'], p['lo'], p['hi'], p['c']) == (hb['mean'], hb['lo'], hb['hi'], 8)
    assert x['counts']['longer'] == sum(x['components'][r]['lo'] > 0 for r in R8)
    assert x['counts']['shorter'] == sum(x['components'][r]['hi'] < 0 for r in R8)


def test_further_size_of_the_plan_is_reported(tmp_path, monkeypatch, A):
    import eval_newseeds_plan as PLAN
    sets = [('10x25', R8), ('20x25', ['N', 'S', 'LSW'])]
    monkeypatch.setitem(PLAN.SECONDARY_EVAL, ARM, sets)
    root = str(tmp_path)
    build(root, monkeypatch, A, sets=sets)
    run_a(A, root)
    m = macros(outs(root)[0])
    assert (m['ActTwentyWorseN'], m['ActTwentyBetterN'], m['ActTwentyInstN']) == ('3', '0', str(N_INST))
    assert (m['ActTwentyGapLo'], m['ActTwentyGapHi']) == ('1.00', '3.00')
    assert m['ActTwentyPoolDelta'] == A.f2(np.mean([1.0, 2.5, 3.0])) == '+2.17'
    assert m['ActWorseN'] == '5'                                  # the training size unchanged
    t = open(outs(root)[1]).read()
    assert '\\multicolumn{3}{l}{20 modules}' in t and 'all 3 & $+2.17$' in t
    monkeypatch.setitem(PLAN.SECONDARY_EVAL, ARM, [('10x25', R8), ('70x25', R8)])
    with pytest.raises(SystemExit, match='no macro word'):
        run_a(A, root)


DAMAGE = ['missing arm array', 'missing policy array', 'arm without the active mask',
          'policy with the active mask', 'other training budget', 'other seed in configuration',
          'wrong rows', 'older than checkpoint', 'summary without the exact mask',
          'log without completion']


@pytest.mark.parametrize('damage', DAMAGE)
def test_refuses_names_the_problem_and_leaves_no_output(tmp_path, monkeypatch, A, damage):
    root = str(tmp_path)
    build(root, monkeypatch, A)
    run_a(A, root)
    assert all(os.path.exists(p) for p in outs(root))
    arm = {s: A.Run(REG.secondary_model(ARM, s), s, '10x25') for s in SEEDS3}
    pol = {s: policy_run(A, s, '10x25') for s in SEEDS3}
    want = None
    if damage == 'missing arm array':
        os.remove(os.path.join(root, arm[302].array('LW')))
        want = 'missing ' + arm[302].array('LW')
    elif damage == 'missing policy array':
        os.remove(os.path.join(root, pol[303].array('N')))
        want = 'missing ' + pol[303].array('N')
    elif damage == 'arm without the active mask':
        write(root, arm[301].config, json.dumps(cfg_of(301, 'nondelay')))
        want = "mask_mode='nondelay', want 'active'"
    elif damage == 'policy with the active mask':
        write(root, pol[301].config, json.dumps(cfg_of(301, 'active')))
        want = 'want the non-delay mask'
    elif damage == 'other training budget':
        write(root, arm[303].config, json.dumps(dict(cfg_of(303, 'active'), max_updates=1000)))
        want = 'not the same training budget'
    elif damage == 'other seed in configuration':
        write(root, arm[302].config, json.dumps(cfg_of(301, 'active')))
        want = 'records seed 301, want 302'
    elif damage == 'wrong rows':
        np.save(os.path.join(root, arm[301].array('N')), np.ones((N_INST - 1, 2)) * 100)
        want = 'has shape'
    elif damage == 'older than checkpoint':
        os.utime(os.path.join(root, arm[301].array('S')), (T0 - 60, T0 - 60))
        want = 'older than its checkpoint'
    elif damage == 'summary without the exact mask':
        write(root, arm[301].summary(), '# FAMILY eval\n')
        want = 'does not record'
    else:
        write(root, arm[303].log, 'Ep 1\n')
        want = 'no completion line'
    with pytest.raises(SystemExit) as e:
        run_a(A, root)
    msg = str(e.value)
    assert msg.startswith('REFUSING') and want in msg, msg
    assert root not in msg
    assert not any(os.path.exists(p) for p in outs(root))


def test_no_array_of_the_arm_skips_and_require_all_refuses(tmp_path, monkeypatch, A):
    root = str(tmp_path)
    build(root, monkeypatch, A)
    run_a(A, root)
    for s in SEEDS3:
        for r in R8:
            os.remove(os.path.join(root, A.Run(REG.secondary_model(ARM, s), s, '10x25').array(r)))
    res = run_a(A, root)
    assert res['status'] == 'skipped' and len(res['missing']) == 24
    assert not any(os.path.exists(p) for p in outs(root))
    with pytest.raises(SystemExit) as e:
        run_a(A, root, '--require', 'all')
    msg = str(e.value)
    assert msg.startswith('REFUSING: --require all') and \
        f'test_results/FAMILY/10x25_xmask/Result_greedy+10x25+family+{ARM}-s301_N.npy' in msg
    assert not any(os.path.exists(p) for p in outs(root))


def test_require_all_computes_on_a_complete_tree_and_needs_an_output_root(tmp_path, monkeypatch, A):
    root = str(tmp_path)
    build(root, monkeypatch, A)
    assert run_a(A, root, '--require', 'all')['status'] == 'computed'
    monkeypatch.delenv('PACT_OUT')
    with pytest.raises(SystemExit, match='PACT_OUT'):
        run_a(A, root)


def snapshot(top, skip=None):
    out = {}
    for d, dirs, files in os.walk(top):
        if skip and os.path.abspath(d).startswith(os.path.abspath(skip)):
            continue
        for f in dirs + files:
            p = os.path.join(d, f)
            if skip and os.path.abspath(p).startswith(os.path.abspath(skip)):
                continue
            st = os.lstat(p)
            out[os.path.relpath(p, top)] = (st.st_size, st.st_mtime_ns)
    return out


REPO_OUTPUTS = ('manuscript/macros_actmask.tex', 'manuscript/tables/actmask_table.tex',
                'paper/macros_actmask.tex', 'paper/tables/actmask_table.tex',
                'results/active_mask_arm.json', 'results/active_mask_arm.md', 'out')


def repo_state():
    out = {}
    for rel in REPO_OUTPUTS:
        p = os.path.join(ROOT, rel)
        out[rel] = (os.stat(p).st_size, os.stat(p).st_mtime_ns) if os.path.exists(p) else None
    out['out/exact'] = snapshot(os.path.join(ROOT, 'out')) if os.path.isdir(os.path.join(ROOT, 'out')) else None
    return out


def test_outputs_hold_no_absolute_path_and_nothing_is_written_outside_the_output_root(
        tmp_path, monkeypatch, A):
    root = str(tmp_path)
    build(root, monkeypatch, A)
    before, repo_before = snapshot(root, skip=os.path.join(root, 'out')), repo_state()
    run_a(A, root)
    assert snapshot(root, skip=os.path.join(root, 'out')) == before
    assert repo_state() == repo_before
    written = sorted(os.path.relpath(p, os.path.join(root, 'out'))
                     for p in glob.glob(os.path.join(root, 'out', '**', '*'), recursive=True)
                     if os.path.isfile(p))
    assert written == sorted(['manuscript/macros_actmask.tex', 'manuscript/tables/actmask_table.tex',
                              'results/active_mask_arm.json', 'results/active_mask_arm.md'])
    for p in outs(root):
        text = open(p).read()
        assert root not in text and ROOT not in text and '/home/' not in text and '/tmp/' not in text, p
    j = json.load(open(outs(root)[2]))
    assert j['inputs'] and not any(os.path.isabs(f) for f in j['inputs'])
    assert len(j['inputs']) == 2 * 3 * (4 + 8)


LEGACY_NAMES = {'ActSGaAct', 'ActSGaBlind', 'ActWorseN', 'IdleLo', 'IdleHi', 'IdleN', 'BeyondLo',
                'BeyondHi', 'ActGapLo', 'ActGapHi', 'ActSDelta', 'ActSLo', 'ActSHi', 'ActNDelta'}


def test_macro_names_comments_and_collisions(tmp_path, monkeypatch, A):
    root = str(tmp_path)
    build(root, monkeypatch, A)
    run_a(A, root)
    lines = open(outs(root)[0]).read().splitlines()
    assert lines[0].startswith('% Generated by scripts/p50_actmask_analysis.py -- do not hand-edit')
    assert re.search(r'environment (\w+), training seeds ([\d,]+)', lines[0]).groups() == ('exact', '301,302,303')
    names = []
    for ln in lines[1:]:
        m = re.fullmatch(r'\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}  % (.+)', ln)
        assert m, ln                                       # one comment per macro line
        names.append(m.group(1))
    assert len(names) == len(set(names)) == 15 and all(re.fullmatch(r'Act[A-Za-z]+', n) for n in names)
    kept = LEGACY_NAMES & set(names)
    assert kept == {'ActWorseN', 'ActGapLo', 'ActGapHi', 'ActSDelta', 'ActSLo', 'ActSHi', 'ActNDelta'}
    assert not set(names) & {'IdleLo', 'IdleHi', 'IdleN', 'BeyondLo', 'BeyondHi'}   # no audit macro
    # the names of the earlier file are those its generator writes today
    stored = os.path.join(ROOT, 'paper', 'macros_actmask.tex')
    if os.path.exists(stored):
        assert set(macros(stored)) == LEGACY_NAMES
    # a file of the same quantities is no collision; any other file is
    write(root, 'paper/macros_actmask.tex', '\\newcommand{\\ActWorseN}{8}\n')
    run_a(A, root)
    write(root, 'manuscript/macros_other.tex', '\\newcommand{\\ActPoolDelta}{1}\n')
    with pytest.raises(SystemExit, match='collision'):
        run_a(A, root)
    assert not any(os.path.exists(p) for p in outs(root))


def test_new_names_are_free_in_the_repository(A):
    other, n_files = A.defined_elsewhere(ROOT, set())
    names = ['ActWorseN', 'ActBetterN', 'ActCombN', 'ActGapLo', 'ActGapHi', 'ActSDelta', 'ActSLo',
             'ActSHi', 'ActNDelta', 'ActPoolDelta', 'ActPoolLo', 'ActPoolHi', 'ActSeedN',
             'ActInstN', 'ActBoot']
    names += [f'Act{w}{k}' for w in A.SIZE_WORD.values()
              for k in ('WorseN', 'BetterN', 'GapLo', 'GapHi', 'PoolDelta', 'PoolLo', 'PoolHi', 'InstN')]
    assert n_files > 10
    assert not set(names) & set(other), {n: other[n] for n in set(names) & set(other)}


def test_earlier_environment_branch_writes_the_stored_files(tmp_path):
    need = ['paper/macros_actmask.tex', 'paper/tables/actmask_table.tex', 'results/idle_quant_cells.json']
    if not all(os.path.exists(os.path.join(ROOT, p)) for p in need) or \
            not os.path.exists(os.path.join(ROOT, 'test_results/FAMILY/10x25/Result_GA_N.npy')):
        pytest.skip('needs the stored files of the earlier environment')
    out = tmp_path / 'out'
    r = subprocess.run([PY, '-B', os.path.join(ROOT, 'scripts', 'p50_actmask_analysis.py')], cwd=ROOT,
                       env=clean_env(PACT_OUT=str(out)), capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    for rel in need[:2]:
        assert open(out / rel, 'rb').read() == open(os.path.join(ROOT, rel), 'rb').read(), rel
    assert sorted(str(p.relative_to(out)) for p in out.rglob('*') if p.is_file()) == sorted(need[:2])
