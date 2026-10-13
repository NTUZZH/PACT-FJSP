"""
Greedy evaluation of an HGNN family checkpoint on PACT's FAMILY benchmark.

For every size and regime the instances are read exactly as eval_family.py
reads them: descriptors data/FAMILY/<size>/desc/<regime>/*.desc.json (sorted),
base instances from the base_dir named in data/FAMILY/<size>/dataset_meta.json.
Decoding is greedy, one instance at a time (batch of one), under the reward and
mask conventions the checkpoint was trained with (read from its
train_log/FAMILY/config_<name>.json). Every schedule is re-validated by
constraint_family.reference_sim.validate_schedule and the run stops on the
first violation.

Default scope: 10x25, 20x25 and 30x25 (all instances) and 50x25 and 80x25
(the first 10 instances per regime, --limit_large).

Output, one file per size and regime, same layout as the Result_greedy+...
files of eval_family.py (float array [n_instances, 2]: makespan in hours,
wall-clock seconds of the rollout):
    test_results/FAMILY/<size>/Result_<tag>_<regime>.npy
with <tag> = the model name without its '<n_j>x<n_m>+family+' prefix, e.g.
Result_hgnn-certified-joint-s301_LS.npy (a training size other than 10x25 is
appended as '-train<n_j>x<n_m>'). Existing files are not overwritten without
--force; the check runs before any rollout.

Example:
    python backbones/hgnn/eval_family_hgnn.py --family_model 10x25+family+hgnn-certified-joint-s301 \
        --device cpu --threads 1
"""
import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ppvc_hgnn as P  # noqa: E402
import pact_bridge as B  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

REGIMES = ('N', 'L', 'S', 'W', 'LS', 'LW', 'SW', 'LSW')
LARGE = ('50x25', '80x25')


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description='Evaluate an HGNN family checkpoint')
    ap.add_argument('--family_model', type=str, required=True)
    ap.add_argument('--sizes', type=str, default='10x25,20x25,30x25,50x25,80x25')
    ap.add_argument('--regimes', type=str, default=','.join(REGIMES))
    ap.add_argument('--limit', type=int, default=0, help='instances per regime for the smaller sizes (0 = all)')
    ap.add_argument('--limit_large', type=int, default=10, help='instances per regime at 50x25 and 80x25')
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--threads', type=int, default=1, help='torch threads for the batch-of-one forward')
    ap.add_argument('--data_root', type=str, default=os.path.join(B.REPO_ROOT, 'data', 'FAMILY'))
    ap.add_argument('--model_root', type=str, default=B.REPO_ROOT,
                    help='root holding trained_network/FAMILY and train_log/FAMILY')
    ap.add_argument('--out_root', type=str, default=os.path.join(B.REPO_ROOT, 'test_results', 'FAMILY'))
    ap.add_argument('--force', action='store_true')
    return ap.parse_args(argv)


def result_tag(snap):
    name = snap['model_name']
    prefix = '{0}x{1}+family+'.format(snap['n_j'], snap['n_m'])
    assert name.startswith(prefix), name
    tag = name[len(prefix):]
    if (snap['n_j'], snap['n_m']) != (10, 25):
        tag += '-train{0}x{1}'.format(snap['n_j'], snap['n_m'])
    return tag


def main(argv=None):
    from env.fjsp_env_family import FJSPEnvFamilyHGNN
    import PPO_model
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(args.threads)
    except RuntimeError:
        pass
    device = torch.device(args.device)
    torch.set_default_dtype(torch.float32)
    torch.set_default_device(device)

    with open(os.path.join(args.model_root, 'train_log', 'FAMILY', 'config_{0}.json'.format(args.family_model))) as f:
        snap = json.load(f)
    assert snap.get('backbone') == 'hgnn', 'not an HGNN family checkpoint'
    reward, mask = snap['hgnn_reward'], snap['hgnn_mask']
    cfg = P.load_config()
    env_paras, model_paras = P.make_paras(cfg, device, 'aware', snap['lag_features'], 1, 1, 1)
    assert model_paras['in_size_ope'] == snap['model_paras']['in_size_ope']
    model = P.build_model(model_paras, cfg['train_paras'],
                          ckpt=os.path.join(args.model_root, 'trained_network', 'FAMILY',
                                            args.family_model + '.pth'))
    model.policy_old.eval()
    n_params = sum(p.numel() for p in model.policy.parameters())
    assert n_params == snap['n_params'], (n_params, snap['n_params'])
    tag = result_tag(snap)
    sizes = [s.strip() for s in args.sizes.split(',') if s.strip()]
    regimes = [r.strip().upper() for r in args.regimes.split(',') if r.strip()]

    # refuse to overwrite before any rollout
    plan = []
    for size in sizes:
        out_dir = os.path.join(args.out_root, size)
        for regime in regimes:
            out = os.path.join(out_dir, 'Result_{0}_{1}.npy'.format(tag, regime))
            if os.path.exists(out) and not args.force:
                sys.exit('{0} exists; pass --force to overwrite'.format(out))
            plan.append((size, regime, out))
    print('[eval_family_hgnn] model={0} params={1} reward={2} mask={3} tag={4} device={5} threads={6}'.format(
        args.family_model, n_params, reward, mask, tag, device, args.threads), flush=True)

    P_ = B.pact()
    for size, regime, out in plan:
        with open(os.path.join(args.data_root, size, 'dataset_meta.json')) as f:
            base_dir = os.path.join(B.REPO_ROOT, json.load(f)['base_dir'])
        paths = sorted(glob.glob(os.path.join(args.data_root, size, 'desc', regime, '*.desc.json')))
        limit = args.limit_large if size in LARGE else args.limit
        if limit:
            paths = paths[:limit]
        assert paths, 'no descriptors for {0} {1}'.format(size, regime)
        rows = []
        for dp in paths:
            name = os.path.basename(dp).replace('.desc.json', '')
            with open(dp) as f:
                desc = P_.D.ConstraintDescriptor.from_json(json.load(f))
            jl, pt, _ = P_.G.load_instance(os.path.join(base_dir, name))
            env = FJSPEnvFamilyHGNN([jl], [pt], [desc], env_paras, reward=reward, mask=mask, checks=True)
            memory = PPO_model.Memory()
            state = env.state
            done = False
            dones = env.done_batch
            t0 = time.time()
            while ~done:
                with torch.no_grad():
                    actions = model.policy_old.act(state, memory, dones, flag_sample=False, flag_train=False)
                state, _, dones = env.step(actions)
                done = dones.all()
            secs = time.time() - t0
            viol = env.validate_family()[0]
            assert not viol, '{0} {1} {2}: {3}'.format(size, regime, name, viol[:3])
            ms = float(env.realized_makespan()[0])
            assert abs(ms - float(env.makespan_batch[0])) < 1e-4
            rows.append([ms, secs])
        res = np.asarray(rows, dtype=float)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.save(out, res)
        print('[{0} {1:>3}] {2:<40} n={3} mean={4:7.1f} std={5:5.1f} feasible={3}/{3} {6:.2f}s/inst'.format(
            size, regime, tag, len(rows), res[:, 0].mean(), res[:, 0].std(), res[:, 1].mean()), flush=True)


if __name__ == '__main__':
    main()
