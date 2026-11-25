#!/bin/bash
# Evaluation of the specialists that share the policy's training budget
# (xshare-solo-<R>-s<seed>, 250 updates, scripts/run_budgetspec_queue.sh): one greedy pass on
# the test instances of the run's own combination R at the training size (10x25), in the
# environment of the paper.
#
# The call. A job is the evaluation plan's principal 10x25 job of xmask-solo-<R>-s<seed>
#   (scripts/eval_newseeds_plan.py, group principal, read through seed_extension.plan_jobs)
#   with the run name exchanged and the combinations restricted to R
#   (scripts/budgetspec_runs.py evaljob). For a seed 306 to 310 the plan has no job; the job is
#   the plan's job of xmask-solo-<R>-s305 with the run name exchanged to xmask-solo-<R>-s<seed>
#   (seed_extension.renamed_job, as scripts/seed_extension.py builds the jobs of the second five
#   seeds) and then to the run's own name. Its command is built by build_cmd of
#   scripts/run_eval_newseeds.sh, so options, data folders, device and threads are those of the
#   principal call: OMP and MKL threads 1, CUDA_VISIBLE_DEVICES empty, eval_family.py --methods
#   greedy --device cpu --torch_threads 1 --tag _xmask --first_mask exact --mask_ties exact,
#   pinned to one core. Only --out_root differs: the job writes into its own staging folder,
#   and its files then move to <results>/10x25_xshare.
#
# PRE-RUN RECORD.
# 1. PURPOSE. The arrays that scripts/p127_budget_specialists.py reads: each budget-sharing
#    specialist on its own combination, set against the policy and the 1000-update specialist
#    of the same seed, for the ten seeds 301 to 310 (80 runs). The numbers land in Section 6.1,
#    in the section on the cost of a rule
#    change, in the decision table and in a supplement table.
# 2. IDEAL RESULT. The pre-flight reproduces the stored array, every job finishes with one row
#    per test instance, and 'scripts/budgetspec_runs.py state --seeds <SEEDS>' reports every
#    run trained and evaluated. A failed job is named and left for a later invocation; the
#    analysis computes nothing until every input exists.
# 3. CONTAMINATION. The arrays go to <results>/10x25_xshare, never to 10x25_xmask. No glob of
#    the analysis reads that folder; the completeness check of the evaluation plan lists every
#    folder of the results root but keeps only files of stored runs, and xshare- names none.
#    A job runs into its own staging folder (xshare_<name>), and its files move into place only
#    when both exist and the array holds one row per test instance. A job whose files partly
#    exist in place, or whose array in place holds another number of rows, is refused; a
#    finished job is skipped. No wall-clock of this runner is a measurement.
# 4. DATA ACCURACY. Pre-flight, before the first job and in VERIFY=1: the stored specialist
#    xmask-solo-N-s301 evaluated on N with the same call into a temporary folder must give the
#    makespans of its stored array in 10x25_xmask within 1e-6. A run is evaluated only once its
#    training checks are met ('scripts/budgetspec_runs.py trained': completion line, 32786
#    parameters, exact mask, the configuration of xmask-solo-<R>-s<seed> except model_suffix
#    and max_updates); a run whose log holds no completion line yet is left for a later
#    invocation, and a finished run whose checks fail is refused.
#
# Options: CORE (the one core of every job and of the launcher, required outside LIST_ONLY),
#   SEEDS (default 301-310), ONLY, LIST_ONLY=1, VERIFY=1, HOLD (default train_log/EVAL_HOLD),
#   and STAGE_ROOT, LOGDIR (default train_log/budgetspec_eval), RESULTS for tests, with the
#   meaning stated in the header of scripts/run_longspec_eval.sh. Exit status: 0 when no job
#   failed and no run was refused, 1 when a job or the pre-flight failed or a run was refused,
#   2 when a setting refuses.
#
# The code is that of scripts/run_longspec_eval.sh: this file runs it with the variant xshare.
#
# Usage:
#   CORE=<free core> VERIFY=1 bash scripts/run_budgetspec_eval.sh
#   CORE=<free core> setsid nohup bash scripts/run_budgetspec_eval.sh \
#       > train_log/budgetspec_eval.log 2>&1 &
if [ $# -ne 0 ]; then
  echo "[budgetspec-eval] REFUSING: no argument is taken (options are environment variables), got '$*'"; exit 2
fi
exec bash "$(dirname "$0")/run_longspec_eval.sh" xshare
