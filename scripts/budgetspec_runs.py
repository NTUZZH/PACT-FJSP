r"""Names, paths and checks of the specialists that share the policy's training budget.

THE RUNS
  xshare-solo-<R>-s<seed>: the recipe of the specialist xmask-solo-<R>-s<seed> (one combination R,
  exact mask, 32786 parameters, scripts/run_exactmask_queue.sh) with --max_updates 250 in place
  of 1000. Eight combinations R (N, L, S, W, LS, LW, SW, LSW), training seeds 301 to 310 (80
  runs), the ten seeds of the principal comparison (seed_extension.ALL_SEEDS): the five seeds
  of the registry and the second five seeds 306 to 310. The eight runs of one seed train for
  8 x 250 = 2000 updates, the number of updates of the policy, with as many episodes per update
  (num_envs).
    training    scripts/run_budgetspec_queue.sh
    evaluation  scripts/run_budgetspec_eval.sh: one greedy pass on the test instances of R at
                10x25 with the call of the evaluation plan's principal job of
                xmask-solo-<R>-s<seed> (for a seed 306 to 310 the plan's job of
                xmask-solo-<R>-s305 with the run name exchanged, as scripts/seed_extension.py
                builds the jobs of the second five seeds), the run name exchanged and the
                combinations restricted to R; the arrays go to <results>/10x25_xshare, never
                to 10x25_xmask
    analysis    scripts/p127_budget_specialists.py
  Checkpoint, training log, configuration and validation record follow the helpers of
  xmask-solo-<R>-s<seed> with the run name exchanged: the registry's for the seeds 301 to 305,
  those of scripts/seed_extension.py for the seeds 306 to 310.

A RUN IS TRAINED when
  its checkpoint exists; its training log holds the header line of train_family.py with its own
  model name, the combination R and 250 updates, the line of the exact mask, 32786 parameters on
  the line '[model] params total=' and the completion line; its configuration records
  first_mask exact and mask_ties exact and equals the configuration of xmask-solo-<R>-s<seed>
  except model_suffix (its own name) and max_updates (250 against 1000); its validation record
  exists.
A RUN IS EVALUATED when its array <results>/10x25_xshare/Result_greedy+10x25+family+<name>_<R>.npy
  holds one row per test instance of R at 10x25 (the descriptors of data/FAMILY/10x25/desc/<R>).

THE PROBE (scripts/run_budgetspec_queue.sh, before any training)
  One update on the CPU with seed 301 per combination R: the probe's configuration records the
  exact mask, trains R, and equals the stored configuration of xmask-solo-<R>-s301 except
  model_suffix, seed_train, device and max_updates; its log holds 32786 parameters, the mask
  line, and the MWKR norms line of the log of xmask-solo-<R>-s301.

THE CODE is that of scripts/longspec_runs.py, the specialists trained for 2000 updates: this
file loads it as a module of its own and exchanges the constants of the variant (VARIANT
below), so every name, path, check and command of that file applies with 250 updates, the
prefix xshare- and the folder 10x25_xshare. Its exception Refused is the one of
longspec_runs, so a caller catches the refusals of both variants with one class. Importing
this file gives that module; running it runs that module's command line.

Command line (the environment is exact; PACT_ENV defaults to exact here):
  python scripts/budgetspec_runs.py list [--seeds S]       the runs and their files
  python scripts/budgetspec_runs.py state [--seeds S] [--root D]
  python scripts/budgetspec_runs.py probe <R> <suffix> <log> [--root D]
  python scripts/budgetspec_runs.py trained <name> [--root D]
  python scripts/budgetspec_runs.py evaljob <name>
  python scripts/budgetspec_runs.py verifyjob
  python scripts/budgetspec_runs.py rows <n> <file> ...
  python scripts/budgetspec_runs.py compare <a.npy> <b.npy>
--seeds takes '301', '301,302' or '301-310' (default 301-310).
"""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import longspec_runs as _BASE  # noqa: E402
import seed_extension as _SE  # noqa: E402

VARIANT = dict(TAG='xshare', STEM='xshare-solo-{R}', UPDATES=250, RESULT_FOLDER='10x25_xshare',
               SEEDS=tuple(_SE.ALL_SEEDS), DEFAULT_SEEDS=tuple(_SE.ALL_SEEDS))


def load(name='budgetspec_runs'):
    """A new module object from the code of scripts/longspec_runs.py with the constants of
    VARIANT and the exception class of longspec_runs."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, 'longspec_runs.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    for k, v in VARIANT.items():
        if not hasattr(m, k):
            raise ImportError(f'scripts/longspec_runs.py defines no {k}; the variant cannot be built')
        setattr(m, k, v)
    m.Refused = _BASE.Refused
    m.__doc__ = __doc__
    m.VARIANT = dict(VARIANT)
    m.load = load
    return m


if __name__ == '__main__':
    sys.exit(load().main())
_MODULE = load(__name__)
_MODULE.LOADER = sys.modules[__name__]          # this file's module, kept alive with load()
sys.modules[__name__] = _MODULE
