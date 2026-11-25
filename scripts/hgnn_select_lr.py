"""Learning rate of the HGNN runs under the exact mask, chosen on the validation score.

The tuning runs (scripts/run_hgnn_exact_queue.sh, PHASE=tune) train the HGNN port
with the PACT reward on the six training combinations, training seed 301, under the
exact mask, for 2000 iterations, at three learning rates:

  1e-4   10x25+family+hgnn-certified-joint-s301-xmask-lr1e-4
  2e-4   10x25+family+hgnn-certified-joint-s301-xmask          (the upstream value)
  4e-4   10x25+family+hgnn-certified-joint-s301-xmask-lr4e-4

The score is the trainer's own selection score: every 10 iterations the greedy
makespan on the 20 validation instances of each training combination, divided by
that combination's MWKR makespan, averaged over the combinations (lower is better);
the trainer keeps the checkpoint with the lowest score, and that minimum is the
run's score here. The chosen rate is the one with the lowest score. A tie goes to
the upstream value 2e-4, then to the lower rate.

Before choosing, each run must be finished (completion line in its log), record the
exact mask, the PACT reward, seed 301, 2000 iterations, its own learning rate and
204250 parameters in its configuration, and all three must share the same
validation norms, so that their scores are on one scale. The minimum of each
validation curve is cross-checked against the 'best rel=' of the completion line.

Usage:
    python scripts/hgnn_select_lr.py            report the three scores and the choice
    python scripts/hgnn_select_lr.py --quiet    print only the chosen rate (1e-4, 2e-4 or 4e-4)
"""
import argparse
import ast
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREFIX = '10x25+family+'
STEM = 'hgnn-certified-joint-s301-xmask'
DEFAULT_LR = '2e-4'
RATES = ('1e-4', '2e-4', '4e-4')
ITERATIONS = 2000
PARAMS = 204250


def run_name(lr):
    """Run name of the tuning run at learning rate lr ('1e-4', '2e-4' or '4e-4')."""
    return STEM + ('' if lr == DEFAULT_LR else '-lr' + lr)


def record(lr, root=ROOT):
    """(score, iteration of the kept checkpoint, validation norms) of one tuning run; exits on any check."""
    name = PREFIX + run_name(lr)
    cfg_p = os.path.join(root, 'train_log', 'FAMILY', 'config_{0}.json'.format(name))
    val_p = os.path.join(root, 'train_log', 'FAMILY', 'valiquality_{0}.txt'.format(name))
    log_p = os.path.join(root, 'train_log', 'FAMILY_{0}.log'.format(run_name(lr)))
    for p in (cfg_p, val_p, log_p):
        if not os.path.exists(p):
            sys.exit('REFUSING: {0} is missing (tuning run at lr {1} not finished)'.format(p, lr))
    with open(cfg_p) as f:
        cfg = json.load(f)
    want = {'hgnn_mask': 'exact', 'hgnn_reward': 'certified'}
    for k, v in want.items():
        if cfg.get(k) != v:
            sys.exit('REFUSING: {0} records {1}={2!r}, want {3!r}'.format(cfg_p, k, cfg.get(k), v))
    a = cfg['args']
    checks = [('seed', a.get('seed'), 301), ('max_iterations', a.get('max_iterations'), ITERATIONS),
              ('lr', a.get('lr'), float(lr)), ('train_paras.lr', cfg['train_paras'].get('lr'), float(lr)),
              ('n_params', cfg.get('n_params'), PARAMS),
              ('family_regimes', a.get('family_regimes'), 'N,L,S,W,LS,LW')]
    for k, got, exp in checks:
        if got != exp:
            sys.exit('REFUSING: {0} records {1}={2!r}, want {3!r}'.format(cfg_p, k, got, exp))
    with open(log_p, errors='replace') as f:
        txt = f.read()
    m = re.findall(r'done in [0-9.]+ h; best rel=([0-9.]+)', txt)
    if not m:
        sys.exit('REFUSING: {0} has no completion line'.format(log_p))
    with open(val_p) as f:
        curve = ast.literal_eval(f.read())
    if len(curve) != ITERATIONS // 10:
        sys.exit('REFUSING: {0} has {1} validation points, want {2}'.format(val_p, len(curve), ITERATIONS // 10))
    best = float(min(curve))
    if abs(best - float(m[-1])) > 5e-5:
        sys.exit('REFUSING: {0} reports best rel={1}, the curve minimum is {2:.6f}'.format(log_p, m[-1], best))
    it = 10 * (curve.index(min(curve)) + 1)
    return best, it, cfg['vali_norm']


def select(root=ROOT):
    """(chosen rate, {rate: (score, iteration)})."""
    recs = {lr: record(lr, root) for lr in RATES}
    norms = {lr: r[2] for lr, r in recs.items()}
    if any(n != norms[DEFAULT_LR] for n in norms.values()):
        sys.exit('REFUSING: the validation norms of the tuning runs differ: {0}'.format(norms))
    order = sorted(RATES, key=lambda lr: (recs[lr][0], lr != DEFAULT_LR, float(lr)))
    return order[0], {lr: recs[lr][:2] for lr in RATES}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--quiet', action='store_true', help='print only the chosen learning rate')
    ap.add_argument('--root', default=ROOT, help='project root holding train_log/')
    args = ap.parse_args(argv)
    chosen, scores = select(args.root)
    if args.quiet:
        print(chosen)
        return
    print('HGNN learning rate under the exact mask, PACT reward, seed 301, {0} iterations'.format(ITERATIONS))
    print('validation score = mean over N, L, S, W, LS, LW of (greedy makespan / MWKR makespan), lower is better')
    for lr in RATES:
        s, it = scores[lr]
        print('  lr {0}  best validation score {1:.4f} at iteration {2:4d}   {3}{4}'.format(
            lr, s, it, PREFIX, run_name(lr)))
    print('chosen learning rate: {0}'.format(chosen))


if __name__ == '__main__':
    main()
