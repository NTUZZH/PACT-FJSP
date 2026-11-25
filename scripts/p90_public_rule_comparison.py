"""Matched comparison of the public-family policies with dispatching rules.

Two parts in one file.

rollout (evaluation, CPU, one thread)
    --mode sample   K sampled schedules per instance from a stored policy, drawn
                    exactly as eval_family.py --methods sampling draws them (K
                    copies of the instance in one batch, torch.manual_seed(
                    configs.seed_test) before each instance, Categorical sampling
                    at every decision). eval_family.py keeps only the shortest of
                    the K schedules; this mode keeps all K makespans, so that the
                    best of k < K schedules can be estimated from one run. Every
                    one of the K schedules is re-validated by
                    reference_sim.validate_schedule.
    --mode trace    one greedy policy rollout (or one masked-rule rollout) per
                    instance, with per-decision counts of the pairs, jobs and
                    machines the mask admits, and per-operation setup and
                    processing time of the realised schedule.
    THE MASK OF A SCHEDULE. A policy's schedules are built under its run's own
                    mask: mask_mode and pair_extra come from the run's training
                    configuration (as in eval_family.py; absent = the non-delay
                    mask); a command-line mask_mode that differs from the
                    configuration refuses. A rule is rolled out under --rule_mask
                    (default nondelay, the mask eval_family.py applies to every
                    rule). --expect_mask M refuses before any rollout when the
                    schedules would be built under another mask. Every .npz of the
                    sample and trace modes records the mask in its key mask_mode.
    --mode valirules
                    the four masked rules on the validation instances of one
                    trained run: the instances train_family.py builds from the
                    run's training configuration (train_log/FAMILY/config_<run>
                    .json; the first vali_size <= 20 instances, drawn from
                    seed_train_vali_datagen, or the first records of
                    pub_vali_dir), each rule rolled out once per instance with
                    numpy seeded by seed_test, as eval_family.py --rule_set
                    masked does on stored records. Before the rules, the
                    trainer's MWKR norm (classical MWKR, legacy mask, mean over
                    the validation instances) is recomputed per combination and
                    compared with the "[vali] MWKR norms:" line of every
                    training log given by --check_logs; a difference refuses.
                    Output: Result_<rule>_<R>.npy (makespan, seconds) per rule
                    and combination, summary_valirules.md in the table format
                    of eval_family.py, valirules.json with the norm check.
    Output: one .npz per (model or rule, combination) under --out.
    Usage (from the repository root):
      python scripts/p90_public_rule_comparison.py rollout --mode sample --K 100 \
          --family_model 10x5+pubfamily+pubx-v1-s301 \
          --data data/PUBLIC/family/pub_SD2_test_10x5+mix --base record \
          --regimes N S --out <dir> --torch_threads 1 \
          --device cpu --device_id "" --first_mask exact --mask_ties exact

analyze (no evaluation; reads arrays only)
    Per combination: the policy (greedy, best of k sampled, greedy plus k-1
    sampled) against the per-instance best of the four masked rules and against
    each single rule, with shares of instances shorter, equal and longer, and
    hierarchical bootstrap intervals (scripts/stats_family.hier_boot, B = 20000,
    numpy default_rng(11)).
      python scripts/p90_public_rule_comparison.py analyze \
          --stored test_results/PUBFAMILY/pub_SD2_test_10x5+mix_xmask \
          --sampled <dir> --out <dir>/analysis.json

The best of k schedules is estimated from K >= k stored samples by the unbiased
order-statistic estimator of E[min of k draws]: with the K makespans sorted
ascending, x_(1) <= ... <= x_(K), the weight of x_(i) is C(K - i, k - 1) / C(K, k).
The probability that the best of k is shorter than (equal to) a reference value r
is estimated the same way: P(min > r) = C(#{x > r}, k) / C(K, k).
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RULES = ('FIFO', 'MOR', 'SPT', 'MWKR')
R8 = ('N', 'L', 'S', 'W', 'LS', 'LW', 'SW', 'LSW')
ARCH_KEYS = ('use_lag_features', 'use_type_embedding', 'fea_j_input_dim',
             'fea_m_input_dim', 'type_emb_dim', 'n_op_types', 'n_mch_types',
             'n_j', 'n_m', 'n_op', 'num_heads_OAB', 'num_heads_MAB',
             'layer_fea_output_dim', 'num_mlp_layers_actor',
             'hidden_dim_actor', 'num_mlp_layers_critic', 'hidden_dim_critic',
             'dropout_prob', 'use_film', 'token_dim', 'film_hidden',
             'token_to_heads', 'reward_delta_env', 'delta_mode',
             'mask_mode', 'pair_extra')   # as eval_family.py
MASKS = ('nondelay', 'active')           # the masks of params.py --mask_mode


# ----------------------------------------------------------------- rollout
def _rollout_main(argv):
    ap = argparse.ArgumentParser(prog='p90 rollout')
    ap.add_argument('--mode', choices=('sample', 'trace', 'valirules'), required=True)
    ap.add_argument('--family_model', default='')
    ap.add_argument('--rule', default='', help='trace mode: a masked rule instead of the policy')
    ap.add_argument('--data', default='', help='stored records (sample and trace modes)')
    ap.add_argument('--base', default='record')
    ap.add_argument('--regimes', nargs='+', default=list(R8))
    ap.add_argument('--K', type=int, default=100)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', required=True)
    ap.add_argument('--torch_threads', type=int, default=1)
    ap.add_argument('--check_logs', nargs='*', default=[],
                    help='valirules mode: training logs whose MWKR norms must agree')
    ap.add_argument('--rule_mask', default='nondelay', choices=MASKS,
                    help='trace mode with --rule: the mask the rule is rolled out under '
                         '(default nondelay)')
    ap.add_argument('--expect_mask', default='', choices=('',) + MASKS,
                    help='refuse unless every schedule is built under this mask')
    args, rest = ap.parse_known_args(argv)
    if args.mode != 'valirules' and not args.data:
        ap.error('the following arguments are required: --data')
    if args.rule_mask != 'nondelay' and not (args.mode == 'trace' and args.rule):
        ap.error('--rule_mask applies to a rule rollout (--mode trace --rule); a policy is '
                 'rolled out under its run\'s own mask')
    # the remaining options (--device, --device_id, --first_mask, --mask_ties,
    # --seed_test) are params.py options, parsed by params at import
    sys.argv = [sys.argv[0]] + rest
    sys.path.insert(0, ROOT)
    os.chdir(ROOT)
    import torch
    torch.set_num_threads(args.torch_threads)
    try:
        torch.set_num_interop_threads(args.torch_threads)
    except RuntimeError:
        pass
    from params import configs
    from constraint_family.descriptor import ConstraintDescriptor
    from constraint_family.fjsp_env_family import FJSPEnvFamily, resolve_delta_mode
    from constraint_family.reference_sim import validate_schedule
    from constraint_family.public_adapter import load_base_instance
    from common_utils import greedy_select_action, sample_action, masked_rule_action
    from constraint_family.public_types import (refuse_mismatch, snapshot_source,
                                                type_lists)
    if args.mode == 'valirules':
        return _valirules(args, configs)

    use_policy = not args.rule

    def types_of(meta, desc, M, K):
        """Type lists of K copies, as eval_family.py builds them."""
        if meta['op_type'] is not None:
            if use_policy and configs.pub_type_source != 'none':
                sys.exit('--pub_type_source applies to public instances only')
            return [meta['op_type']] * K, [meta['mch_type']] * K
        o, m = type_lists(configs.pub_type_source if use_policy else 'none', [desc], M)
        return (None, None) if o is None else (o * K, m * K)
    ppo = None
    if use_policy:
        with open(f'./train_log/FAMILY/config_{args.family_model}.json') as f:
            snap = json.load(f)
        refuse_mismatch(snapshot_source(snap), configs.pub_type_source, args.family_model)
        for k in ARCH_KEYS:
            if k in snap:
                v = snap[k]
                if isinstance(v, str) and v in ('True', 'False'):
                    v = v == 'True'
                setattr(configs, k, v)
        # the run's own mask: its configuration, or the non-delay mask when the
        # configuration predates the key
        run_mask = snap.get('mask_mode') or 'nondelay'
        if getattr(configs, 'mask_mode', 'nondelay') != run_mask:
            sys.exit(f'REFUSING: {args.family_model} was trained under the {run_mask} mask; '
                     f'--mask_mode {configs.mask_mode} would roll it out under another one')
        os.environ['CUDA_VISIBLE_DEVICES'] = configs.device_id
        from model.PPO import PPO_initialize
        ppo = PPO_initialize()
        ppo.policy.load_state_dict(torch.load(
            f'./trained_network/FAMILY/{args.family_model}.pth',
            map_location=torch.device(configs.device)))
        ppo.policy.eval()
    fm, mt = configs.first_mask, configs.mask_ties
    if fm is None or mt is None:
        sys.exit('pass --first_mask and --mask_ties explicitly')
    mm = getattr(configs, 'mask_mode', 'nondelay') if use_policy else args.rule_mask
    pe = getattr(configs, 'pair_extra', 'none') if use_policy else 'none'
    name = args.family_model if use_policy else args.rule
    if args.expect_mask and mm != args.expect_mask:
        sys.exit(f'REFUSING: the schedules of {name} would be built under the {mm} mask, '
                 f'not the {args.expect_mask} mask (--expect_mask)')
    os.makedirs(args.out, exist_ok=True)

    def forward(st):
        with torch.no_grad():
            pi, _ = ppo.policy(fea_j=st.fea_j_tensor, op_mask=st.op_mask_tensor,
                               candidate=st.candidate_tensor, fea_m=st.fea_m_tensor,
                               mch_mask=st.mch_mask_tensor, comp_idx=st.comp_idx_tensor,
                               dynamic_pair_mask=st.dynamic_pair_mask_tensor,
                               fea_pairs=st.fea_pairs_tensor, op_type=st.op_type_tensor,
                               mch_type=st.mch_type_tensor, token=st.token_tensor)
        return pi

    for regime in args.regimes:
        paths = sorted(glob.glob(os.path.join(args.data, 'desc', regime, '*.desc.json')))
        if args.limit:
            paths = paths[:args.limit]
        recs = []
        for p in paths:
            with open(p) as f:
                rec = json.load(f)
            desc = ConstraintDescriptor.from_json(rec, setup_convention='attached')
            stem = (rec['base_instance'] if args.base == 'record'
                    else os.path.join(args.base, os.path.basename(p)[:-len('.desc.json')]))
            recs.append((os.path.basename(p)[:-len('.desc.json')], stem, desc))
        out = {'instances': np.array([r[0] for r in recs])}
        t_all = time.time()
        if args.mode == 'sample':
            K = args.K
            ms_all = np.zeros((len(recs), K))
            feas_all = np.zeros((len(recs), K), dtype=bool)
            secs = np.zeros(len(recs))
            for i, (nm, stem, desc) in enumerate(recs):
                jl, pt, meta = load_base_instance(stem)
                typed = meta['op_type'] is not None
                M = pt.shape[1]
                senv = FJSPEnvFamily(jl.shape[0], M, family_mode=True, mask_mode=mm,
                                     pair_extra=pe, first_mask=fm, mask_ties=mt)
                if senv.mask_mode != mm:
                    sys.exit(f'REFUSING: the environment applies the {senv.mask_mode} mask, want {mm}')
                otl, mtl = types_of(meta, desc, M, K)
                senv.set_initial_data([jl] * K, [pt] * K, descriptor_list=[desc] * K,
                                      op_type_list=otl, mch_type_list=mtl)
                senv._attach_token()
                sstate = senv.state
                torch.manual_seed(configs.seed_test)
                amch_b = np.full((K, pt.shape[0]), -1, dtype=int)
                t0 = time.time()
                while not senv.done().all():
                    pi = forward(sstate)
                    acts = sample_action(pi)[0].cpu().numpy()
                    cj, cm = acts // M, acts % M
                    ops = senv.candidate[np.arange(K), cj]
                    amch_b[np.arange(K), ops] = cm
                    sstate, _, done = senv.step(acts)
                    if done.all():
                        break
                secs[i] = time.time() - t0
                ms_all[i] = senv.current_makespan
                for k in range(K):
                    ct = senv.true_op_ct[k]
                    st = ct - pt[np.arange(pt.shape[0]), amch_b[k]]
                    feas_all[i, k] = not validate_schedule(jl, pt, desc, st, ct, amch_b[k])
            out.update(makespan=ms_all, feasible=feas_all, seconds=secs, K=K,
                       seed_test=configs.seed_test, mask_mode=mm)
            fn = f'sample{K}+{name}_{regime}.npz'
        else:
            n_rec = len(recs)
            res = dict(makespan=np.zeros(n_rec), seconds=np.zeros(n_rec))
            per_dec = []           # per instance: [n_ops, 3] legal pairs, jobs, machines
            per_op = []            # per instance: [n_ops, 4] machine, processing, setup, start
            for i, (nm, stem, desc) in enumerate(recs):
                jl, pt, meta = load_base_instance(stem)
                typed = meta['op_type'] is not None
                M = pt.shape[1]
                env = FJSPEnvFamily(jl.shape[0], M, family_mode=True, mask_mode=mm,
                                    pair_extra=pe, first_mask=fm, mask_ties=mt)
                if env.mask_mode != mm:
                    sys.exit(f'REFUSING: the environment applies the {env.mask_mode} mask, want {mm}')
                dm = resolve_delta_mode(configs)
                env.delta_mode = dm
                env.reward_delta_env = (dm != 'zero')
                otl, mtl = types_of(meta, desc, M, 1)
                state = env.set_initial_data([jl], [pt], descriptor_list=[desc],
                                             op_type_list=otl, mch_type_list=mtl)
                if use_policy:
                    env._attach_token()
                    state = env.state
                else:
                    np.random.seed(configs.seed_test)
                # ops columns: machine, processing time, setup charged, processing
                # start, class the machine held before (IDLE_CLASS = cold start),
                # class of the operation, job
                dec, ops_rec = [], np.zeros((pt.shape[0], 7))
                t0 = time.time()
                while not env.done().all():
                    legal = ~env.dynamic_pair_mask[0]
                    dec.append((int(legal.sum()), int(legal.any(axis=1).sum()),
                                int(legal.any(axis=0).sum())))
                    if use_policy:
                        a = int(greedy_select_action(forward(state)).cpu().numpy()[0])
                    else:
                        a = int(masked_rule_action(args.rule, env))
                    j, m = a // M, a % M
                    op = int(env.candidate[0, j])
                    src = int(env.mch_last_class[0, m])
                    s = (env.true_setup[0, src, env.op_class[0, op]] * env.c_bits[0, 1])
                    state, _, _ = env.step(np.array([a]))
                    ct = env.true_op_ct[0, op]
                    ops_rec[op] = (m, pt[op, m], s, ct - pt[op, m], src,
                                   env.op_class[0, op], j)
                res['seconds'][i] = time.time() - t0
                res['makespan'][i] = float(env.current_makespan[0])
                per_dec.append(np.array(dec))
                per_op.append(ops_rec)
            out.update(res, mask_mode=mm)
            out['decisions'] = np.array(per_dec, dtype=object)
            out['ops'] = np.array(per_op, dtype=object)
            fn = f'trace+{name}_{regime}.npz'
        np.savez(os.path.join(args.out, fn), **out)
        msg = f'[{regime:>3}] {fn} n={len(recs)} wall={time.time() - t_all:.1f}s'
        if args.mode == 'sample':
            msg += (f' best-of-K mean={out["makespan"].min(1).mean():.2f}'
                    f' feasible={int(out["feasible"].sum())}/{out["feasible"].size}')
        else:
            msg += f' mean={out["makespan"].mean():.2f}'
        print(msg, flush=True)


# ----------------------------------------------------------------- validation rules
def trainer_validation(snap, regimes, limit=0, root=ROOT):
    """The validation instances train_family.py builds for a run whose training
    configuration is `snap`: {regime: [(name, job_length, op_pt, meta, desc)]}
    and the settings read. The trainer keeps min(vali_size, 20) instances: SD2
    or factory draws seeded seed_train_vali_datagen + i (or the first records
    of pub_vali_dir), descriptors seeded seed_train_vali_datagen * 1000 + i.
    `limit` keeps the first instances of every combination."""
    from constraint_family.descriptor import make_descriptor
    from constraint_family.public_adapter import sd2_instance
    from constraint_family.public_rules import load_family_record, make_public_descriptor
    from ppvc_instance_generator import (ppvc_instance_generator, build_factory,
                                         DEFAULT_FACTORY, SMALL_FACTORY, TIGHT_FACTORY)
    kind = snap.get('family_base') or 'ppvc'
    n_j = int(snap['n_j'])
    size = min(int(snap['vali_size']), 20)
    vseed = int(snap['seed_train_vali_datagen'])
    shift = json.loads(snap['dr_shift']) if snap.get('dr_shift') else None
    stored = snap.get('pub_vali_dir') or ''
    if kind == 'sd2':
        n_m = int(snap['n_m'])
        if shift is not None:
            sys.exit('dr_shift applies to the factory descriptor draws only')
    elif kind == 'ppvc':
        factory = {'default': DEFAULT_FACTORY, 'small': SMALL_FACTORY,
                   'tight': TIGHT_FACTORY}[snap['ppvc_factory']]
        n_m = len(build_factory(factory)[0])
    else:
        sys.exit(f'unknown family_base {kind!r}')
    names = [f'vali_{i:03d}' for i in range(size)]
    base = []
    if kind == 'sd2' and stored:
        files = sorted(os.listdir(os.path.join(root, stored, 'desc', 'N')))[:size]
        names = [x[:-len('.desc.json')] for x in files]
        for x in files:
            jl, pt, _, _ = load_family_record(os.path.join(root, stored, 'desc', 'N', x))
            base.append((jl, pt, {'op_type': None, 'mch_type': None}))
    else:
        for i in range(size):
            if kind == 'sd2':
                jl, pt = sd2_instance(n_j, n_m, np.random.default_rng(vseed + i))
                meta = {'op_type': None, 'mch_type': None}
            else:
                jl, pt, meta = ppvc_instance_generator(
                    n_modules=n_j, class_mix=snap['ppvc_mix'], station_counts=factory,
                    seed=vseed + i)
            base.append((jl, pt, meta))
    out = {}
    for regime in regimes:
        if kind == 'sd2' and stored:
            descs = [load_family_record(os.path.join(root, stored, 'desc', regime,
                                                     nm + '.desc.json'))[2] for nm in names]
        elif kind == 'sd2':
            descs = [make_public_descriptor(regime, jl, pt, seed=vseed * 1000 + i)
                     for i, (jl, pt, _) in enumerate(base)]
        else:
            descs = [make_descriptor(regime, jl, pt, meta, n_m, seed=vseed * 1000 + i,
                                     shift=shift)
                     for i, (jl, pt, meta) in enumerate(base)]
        rows = [(names[i], base[i][0], base[i][1], base[i][2], descs[i]) for i in range(size)]
        out[regime] = rows[:limit] if limit else rows
    settings = dict(family_base=kind, n_j=n_j, n_m=n_m, vali_size=size, vali_seed=vseed,
                    dr_shift=shift, pub_vali_dir=stored, norm_seed=int(snap['seed_test']))
    return out, settings


def logged_norms(paths):
    """{regime: [(log, printed norm)]} from the '[vali] MWKR norms:' line of each log."""
    import re
    got = {}
    for p in paths:
        with open(p, errors='replace') as f:
            m = re.search(r'\[vali\] MWKR norms: ([^\n]*)', f.read())
        if not m:
            sys.exit(f'REFUSING: {p} has no "[vali] MWKR norms:" line')
        for r, v in re.findall(r'\b([A-Z]+)=([\d.]+)', m.group(1)):
            got.setdefault(r, []).append((p, v))
    return got


def _valirules(args, configs):
    """--mode valirules: the four masked rules on a run's validation instances."""
    from constraint_family.fjsp_env_family import FJSPEnvFamily, resolve_delta_mode
    from constraint_family.reference_sim import validate_schedule
    from constraint_family.public_types import type_lists
    from common_utils import heuristic_select_action, masked_rule_action
    if not args.family_model:
        sys.exit('--mode valirules needs --family_model (the run whose validation set is used)')
    if args.expect_mask not in ('', 'nondelay'):
        sys.exit(f'REFUSING: --mode valirules rolls the rules out under the non-delay mask, not '
                 f'the {args.expect_mask} mask')
    fm, mt = configs.first_mask, configs.mask_ties
    if fm is None or mt is None:
        sys.exit('pass --first_mask and --mask_ties explicitly')
    with open(f'./train_log/FAMILY/config_{args.family_model}.json') as f:
        snap = json.load(f)
    V, settings = trainer_validation(snap, args.regimes, args.limit)
    n_j, n_m = settings['n_j'], settings['n_m']
    os.makedirs(args.out, exist_ok=True)

    # the trainer's MWKR norm: classical MWKR, legacy mask, no type input
    logged = logged_norms(args.check_logs)
    norms, bad = {}, []
    for regime in args.regimes:
        ms = []
        for nm, jl, pt, meta, desc in V[regime]:
            henv = FJSPEnvFamily(n_j, n_m, family_mode=True, first_mask='legacy',
                                 mask_ties='legacy')
            henv.set_initial_data([jl], [pt], descriptor_list=[desc])
            np.random.seed(settings['norm_seed'])
            while not henv.done().all():
                a = heuristic_select_action('MWKR', henv)
                henv.step(np.array([a]))
            ms.append(float(henv.current_makespan[0]))
        v = float(np.mean(ms))
        seen = logged.get(regime, [])
        norms[regime] = dict(computed=v, printed=f'{v:.1f}',
                             logged=[dict(log=p, value=x) for p, x in seen])
        bad += [f'{regime}: recomputed {v:.1f}, {p} prints {x}' for p, x in seen
                if x != f'{v:.1f}']
    uncovered = [r for r in args.regimes if r not in logged]
    if args.limit:
        check = f'not checked (first {args.limit} instances only)'
    elif not args.check_logs:
        check = 'not checked (no training log given)'
    elif bad:
        sys.exit('REFUSING: the validation instances differ from the trainer\'s:\n  '
                 + '\n  '.join(bad))
    elif uncovered:
        sys.exit(f'REFUSING: no training log gives the MWKR norm of {uncovered}')
    else:
        check = 'agree'
    print(f'[valirules] {args.family_model}: {settings}; MWKR norm check: {check}', flush=True)

    rows_md = []
    for regime in args.regimes:
        rows = V[regime]
        for q in RULES:
            res = np.zeros((len(rows), 2))
            for i, (nm, jl, pt, meta, desc) in enumerate(rows):
                M = pt.shape[1]
                env = FJSPEnvFamily(jl.shape[0], M, family_mode=True, mask_mode='nondelay',
                                    pair_extra='none', first_mask=fm, mask_ties=mt)
                dm = resolve_delta_mode(configs)
                env.delta_mode = dm
                env.reward_delta_env = (dm != 'zero')
                if meta['op_type'] is not None:
                    otl, mtl = [meta['op_type']], [meta['mch_type']]
                else:
                    otl, mtl = type_lists('none', [desc], M)
                env.set_initial_data([jl], [pt], descriptor_list=[desc],
                                     op_type_list=otl, mch_type_list=mtl)
                amch = np.full(pt.shape[0], -1, dtype=int)
                t0 = time.time()
                np.random.seed(configs.seed_test)
                while not env.done().all():
                    a = int(masked_rule_action(q, env))
                    amch[env.candidate[0, a // M]] = a % M
                    env.step(np.array([a]))
                res[i] = (float(env.current_makespan[0]), time.time() - t0)
                ct = env.true_op_ct[0]
                st = ct - pt[np.arange(pt.shape[0]), amch]
                viol = validate_schedule(jl, pt, desc, st, ct, amch)
                if viol:
                    sys.exit(f'REFUSING: invalid {q} schedule on {nm} ({regime}): {viol[:3]}')
            np.save(os.path.join(args.out, f'Result_{q}_{regime}.npy'), res)
            rows_md.append((regime, q, res[:, 0].mean(), res[:, 0].std(), len(rows)))
            print(f'[{regime:>3}] {q:<6} validation mean={res[:, 0].mean():7.1f} '
                  f'n={len(rows)}', flush=True)
    with open(os.path.join(args.out, 'summary_valirules.md'), 'w') as f:
        f.write(f'# validation rules — {args.family_model}\n\n'
                f'mask: first_mask={fm} mask_ties={mt}\n\nrules: rule_set=masked\n\n'
                '| regime | method | mean | std | feasible | gap(cov) '
                '| gap to best known (N only) |\n|---|---|---|---|---|---|---|\n')
        for r, q, m, s, n in rows_md:
            f.write(f'| {r} | {q} | {m:.1f} | {s:.1f} | {n}/{n} | n/a | n/a |\n')
    with open(os.path.join(args.out, 'valirules.json'), 'w') as f:
        json.dump(dict(family_model=args.family_model, settings=settings,
                       names=[x[0] for x in V[args.regimes[0]]], n=len(V[args.regimes[0]]),
                       seed_test=configs.seed_test, first_mask=fm, mask_ties=mt,
                       limit=args.limit, check_logs=list(args.check_logs),
                       norm_check=check, norms=norms), f, indent=1)
    print(f'wrote {args.out}/Result_<rule>_<R>.npy, summary_valirules.md, valirules.json',
          flush=True)


# ----------------------------------------------------------------- analysis
def best_of_k(x, k):
    """Unbiased estimate of E[min of k draws] from the K draws in x (1-D)."""
    x = np.sort(np.asarray(x, float))
    K = len(x)
    if k > K:
        raise ValueError(f'k={k} > K={K}')
    w = np.array([math.comb(K - i, k - 1) for i in range(1, K + 1)], float) / math.comb(K, k)
    return float((w * x).sum())


def p_shorter_equal(x, k, r):
    """(P[min of k draws < r], P[min of k draws == r]) from the K draws in x."""
    x = np.asarray(x, float)
    K, ck = len(x), math.comb(len(x), k)
    p_ge = math.comb(int((x >= r).sum()), k) / ck
    p_gt = math.comb(int((x > r).sum()), k) / ck
    return 1.0 - p_ge, p_ge - p_gt


def load_ms(path):
    a = np.load(path)
    return a[:, 0] if a.ndim == 2 else a


def _analyze_main(argv):
    ap = argparse.ArgumentParser(prog='p90 analyze')
    ap.add_argument('--stored', required=True, help='folder with Result_<rule>_<R>.npy and greedy arrays')
    ap.add_argument('--sampled', default='', help='folder with sample<K>+<model>_<R>.npz')
    ap.add_argument('--prefix', default='10x5+pubfamily+')
    ap.add_argument('--arm', default='pubx-v1')
    ap.add_argument('--seeds', nargs='+', type=int, default=[301, 302, 303, 304, 305])
    ap.add_argument('--regimes', nargs='+', default=list(R8))
    ap.add_argument('--ks', nargs='+', type=int, default=[4, 16, 100])
    ap.add_argument('--vali', default='', help='folder of the validation-set rule arrays (rule chosen in advance)')
    ap.add_argument('--B', type=int, default=20000)
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    sys.path.insert(0, os.path.join(ROOT, 'scripts'))
    from stats_family import hier_boot

    def rel(a, b):
        return 100.0 * (a - b) / b

    def boot(d):
        r = hier_boot(d, B=args.B, seed=11)
        return dict(mean=r['mean'], lo=r['lo'], hi=r['hi'])

    report = dict(stored=args.stored, sampled=args.sampled, arm=args.arm, seeds=args.seeds,
                  stats='stats_family.hier_boot, B=%d, numpy default_rng(11), seeds x instances' % args.B,
                  combos={})
    for R in args.regimes:
        rules = np.stack([load_ms(f'{args.stored}/Result_{q}_{R}.npy') for q in RULES])  # [4, n]
        best4 = rules.min(0)
        g = np.stack([load_ms(f'{args.stored}/Result_greedy+{args.prefix}{args.arm}-s{s}_{R}.npy')
                      for s in args.seeds])                                          # [k, n]
        entry = dict(n=int(best4.size))
        # single-rule spread
        entry['rules_mean'] = {q: float(rules[i].mean()) for i, q in enumerate(RULES)}
        entry['best4_over_best_single_pct'] = float(rel(best4, rules[np.argmin(rules.mean(1))]).mean())
        entry['greedy_vs_best4'] = boot(rel(g, best4[None]))
        entry['greedy_vs_rule'] = {q: boot(rel(g, rules[i][None])) for i, q in enumerate(RULES)}
        entry['greedy_vs_best4_share'] = dict(
            shorter=float((g < best4).mean()), equal=float((g == best4).mean()),
            longer=float((g > best4).mean()))
        if args.vali:
            vr = np.array([load_ms(f'{args.vali}/Result_{q}_{R}.npy').mean() for q in RULES])
            q0 = RULES[int(np.argmin(vr))]
            entry['vali_rule'] = q0
            entry['greedy_vs_vali_rule'] = boot(rel(g, rules[RULES.index(q0)][None]))
        if args.sampled:
            S = []
            for s in args.seeds:
                fp = glob.glob(f'{args.sampled}/sample*+{args.prefix}{args.arm}-s{s}_{R}.npz')
                if not fp:
                    S = None
                    break
                z = np.load(fp[0], allow_pickle=True)
                if not z['feasible'].all():
                    raise SystemExit(f'infeasible sampled schedule in {fp[0]}')
                S.append(z['makespan'])
            if S is not None:
                S = np.stack(S)                                                  # [k, n, K]
                K = S.shape[2]
                entry['K'] = int(K)
                for kk in [k for k in args.ks if k <= K]:
                    bk = np.array([[best_of_k(S[a, i], kk) for i in range(S.shape[1])]
                                   for a in range(S.shape[0])])
                    sh = np.array([[p_shorter_equal(S[a, i], kk, best4[i]) for i in range(S.shape[1])]
                                   for a in range(S.shape[0])])
                    entry[f'best{kk}_vs_best4'] = boot(rel(bk, best4[None]))
                    entry[f'best{kk}_vs_best4_share'] = dict(
                        shorter=float(sh[..., 0].mean()), equal=float(sh[..., 1].mean()),
                        longer=float(1 - sh[..., 0].mean() - sh[..., 1].mean()))
                    entry[f'best{kk}_vs_greedy'] = boot(rel(bk, g))
                # greedy plus three sampled schedules: min over 3 of min(g, x)
                for kk in [3, 15]:
                    if kk > K:
                        continue
                    gm = np.array([[best_of_k(np.minimum(S[a, i], g[a, i]), kk)
                                    for i in range(S.shape[1])] for a in range(S.shape[0])])
                    sh = np.array([[p_shorter_equal(np.minimum(S[a, i], g[a, i]), kk, best4[i])
                                    for i in range(S.shape[1])] for a in range(S.shape[0])])
                    entry[f'greedy+{kk}_vs_best4'] = boot(rel(gm, best4[None]))
                    entry[f'greedy+{kk}_vs_best4_share'] = dict(
                        shorter=float(sh[..., 0].mean()), equal=float(sh[..., 1].mean()),
                        longer=float(1 - sh[..., 0].mean() - sh[..., 1].mean()))
        report['combos'][R] = entry
        print(R, json.dumps({k: v for k, v in entry.items() if k.endswith('best4')
                             or k.endswith('share')}, default=float), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(report, f, indent=1, default=float)
    print('wrote', args.out)


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in ('rollout', 'analyze'):
        sys.exit('usage: p90_public_rule_comparison.py {rollout|analyze} ...')
    cmd = sys.argv[1]
    argv = sys.argv[2:]
    sys.argv = [sys.argv[0]] + argv
    (_rollout_main if cmd == 'rollout' else _analyze_main)(argv)
