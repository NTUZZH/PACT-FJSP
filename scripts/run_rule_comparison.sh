#!/bin/bash
# The arrays of the comparison with dispatching rules, for one benchmark family
# per call: the K sampled makespans per instance of every run (released policy,
# single-rule control, specialists; K = 32 unless K or RULECMP_K says otherwise,
# at least 16), the decision traces of the released policy and of
# the four masked rules, the rule schedules with tie-break seeds 51-53, and the
# rules on the validation instances. Jobs, folders, checks and the scheduler:
# scripts/rulecmp_plan.py; the analysis and the macros:
# scripts/p93_rule_comparison.py.
#
# PRE-RUN RECORD (the launch note in train_log/rulecmp_launch_notes.txt states it
# for each start, with the commit, the cores and the processes above 5% CPU).
#
# 1. PURPOSE. Forms (a), (b) and (c) of the comparison with the rules and the two
#    measured variables (share of decisions with one admitted pair, share of
#    machine capacity taken by setups), written by p93 into
#    manuscript/macros_rulecmp_<label>.tex.
# 2. IDEAL RESULT. Every job writes arrays that pass their checks. Before the
#    first job, a preflight on two instances of combination S must show the
#    greedy trace equal to the stored greedy makespans, the best of the K sampled
#    schedules equal to what eval_family.py --methods sampling keeps, and
#    eval_family.py with tie-break seed 50 equal to the stored rule arrays;
#    otherwise nothing starts.
# 3. CONTAMINATION. New files only, in the rulecmp_* subfolders of the family's
#    result folder, which no existing analysis reads; each output is written under
#    a temporary name and renamed after its check; an existing output that fails
#    its check moves to invalid/. The jobs are deterministic evaluations, not
#    timing measurements; the requested cores are still checked for other
#    processes so that no timed experiment shares them (refusal unless
#    ALLOW_BUSY=1).
# 4. DATA ACCURACY. Run names, seeds, checkpoints and result folders come from
#    scripts/arm_registry.py; the instance order is the sorted record order every
#    evaluation program walks, and every output is checked against it.
#
# Usage (from anywhere; the script changes to the repository root):
#   FAMILY=public CORES=16-23 DRY=1 scripts/run_rule_comparison.sh
#       the plan: jobs and their status, counts per kind, missing checkpoints and
#       stored arrays, estimated core-hours and wall clock; starts and writes nothing
#   FAMILY=public CORES=16-23 scripts/run_rule_comparison.sh
#       checks, preflight, then detaches (log train_log/rulecmp_<label>.log, pid
#       file train_log/rulecmp_<label>.pid, job logs train_log/rulecmp_<label>/;
#       the label is the family, or public_<tag> for another public test set)
#   FAMILY=public SET=pub_Brandimarte CORES=16-23 DRY=1 scripts/run_rule_comparison.sh
#       the same for another public test set (released policy only unless ARMS)
#   FAMILY=factory ...   the exact-mask names (PACT_ENV=exact is set here; seeds
#       PACT_SEEDS, default 301-305); refuses while a checkpoint is missing
#   FAMILY=public MASK=active CORES=16-23 DRY=1 scripts/run_rule_comparison.sh
#       the public family under the active mask (the runs of the family trained under it,
#       the rules rolled out under it by scripts/p109_public_active_rules.py; outputs in
#       the rulecmp_* subfolders of test_results/PUBFAMILY/rules_act/<set>_xmask, label
#       public_act or public_act_<tag>); RULES_ONLY=1 plans and runs the jobs of the rules
#       alone, which need no trained run
# Stopping:
#   touch train_log/RULECMP_HOLD    no new job; the running jobs finish and are kept
#   kill -TERM $(cat train_log/rulecmp_<label>.pid)
#       stops at once; the running jobs are killed and their temporary folders
#       removed, so they run again at the next start
# Options (environment):
#   FAMILY          public or factory (required)
#   CORES           cores for the jobs, one job per core, e.g. 16-23 or 16,17,20
#   K               sampled schedules per instance (default 32, the K_DEFAULT of
#                   scripts/rulecmp_plan.py; exported as RULECMP_K, which p93 reads too)
#   SET             public test set (default pub_SD2_test_10x5+mix, the headline set;
#                   the others of arm_registry.PUB_SETS write to the rulecmp_*
#                   subfolders of their own _xmask folder, label public_<tag>)
#   ARMS            arms to sample, a subset of "released control specialist"
#                   (default all three on the headline set, "released" elsewhere)
#   KINDS           subset of "sample trace ties vali" (default all four; a set
#                   other than the headline set has no vali job)
#   MASK            nondelay (default) or active: the mask of the runs and of every
#                   schedule of the public family
#   RULES_ONLY=1    only the jobs of the rules (rule traces, tie-break seeds,
#                   validation records)
#   CONTROL_CORE    core of the scheduler process (default the first of CORES; it
#                   sleeps between polls)
#   ALLOW_BUSY=1    start although other processes use the requested cores
#   ALL_COMMANDS=1  DRY: print every job's command
#   PY              the Python interpreter (default the fjsp environment)
set -u
SELF=$(readlink -f "$0")
cd "$(dirname "$SELF")/.."
PY=${PY:-$HOME/miniconda3/envs/fjsp/bin/python}
export PY
FAMILY=${FAMILY:-}
DRY=${DRY:-0}
CORES=${CORES:-}
KINDS=${KINDS:-"sample trace ties vali"}
if [ -n "${K:-}" ]; then
  if [ -n "${RULECMP_K:-}" ] && [ "$RULECMP_K" != "$K" ]; then
    echo "REFUSING: K=$K and RULECMP_K=$RULECMP_K differ"; exit 1
  fi
  export RULECMP_K=$K
fi
ALLOW_BUSY=${ALLOW_BUSY:-0}
case "$FAMILY" in
  public) ;;
  factory)
    if [ "${PACT_ENV:-exact}" != exact ]; then
      echo "REFUSING: the factory family is the exact-mask environment (PACT_ENV=$PACT_ENV)"; exit 1
    fi
    export PACT_ENV=exact
    export PACT_SEEDS=${PACT_SEEDS:-301-305} ;;
  *) echo "REFUSING: set FAMILY=public or FAMILY=factory"; exit 1 ;;
esac
SET=${SET:-}
ARMS=${ARMS:-}
MASK=${MASK:-nondelay}
RULES_ONLY=${RULES_ONLY:-0}
if [ -n "$SET" ] && [ "$FAMILY" != public ]; then echo "REFUSING: SET applies to FAMILY=public"; exit 1; fi
case "$MASK" in nondelay|active) ;; *) echo "REFUSING: MASK=$MASK, want nondelay or active"; exit 1 ;; esac
case "$RULES_ONLY" in 0|1) ;; *) echo "REFUSING: RULES_ONLY=$RULES_ONLY, want 0 or 1"; exit 1 ;; esac
HOLD=train_log/RULECMP_HOLD
PLAN=("$PY" scripts/rulecmp_plan.py)
ARGS=(--family "$FAMILY" --kinds $KINDS)
[ -n "$SET" ] && ARGS+=(--set "$SET")
[ -n "$ARMS" ] && ARGS+=(--arms $ARMS)
[ "$MASK" != nondelay ] && ARGS+=(--mask "$MASK")
[ "$RULES_ONLY" = 1 ] && ARGS+=(--rules-only)
LABEL=$("${PLAN[@]}" label "${ARGS[@]}") || { echo "$LABEL"; exit 1; }
LOG=train_log/rulecmp_$LABEL.log
PIDFILE=train_log/rulecmp_$LABEL.pid
[ -n "$CORES" ] && ARGS+=(--cores "$CORES")

if [ "$DRY" = 1 ]; then
  "${PLAN[@]}" plan "${ARGS[@]}" ${ALL_COMMANDS:+--all-commands}; rc=$?
  echo "DRY=1: nothing started, nothing written"
  exit $rc
fi

if [ -z "$CORES" ]; then echo "REFUSING: set CORES (for example CORES=16-23)"; exit 1; fi
CONTROL_CORE=${CONTROL_CORE:-$(echo "$CORES" | sed 's/[-,].*//')}
EXTRA=()
[ "$ALLOW_BUSY" = 1 ] && EXTRA+=(--allow-busy)

if [ "${RULECMP_DETACHED:-0}" != 1 ]; then
  mkdir -p train_log
  if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2> /dev/null; then
    echo "REFUSING: a run of this family is going (pid $(cat "$PIDFILE"))"; exit 1
  fi
  if [ -f "$HOLD" ]; then echo "REFUSING: the hold file $HOLD exists; remove it to start"; exit 1; fi
  "${PLAN[@]}" check "${ARGS[@]}" | tee -a "$LOG"
  if [ "${PIPESTATUS[0]}" -ne 0 ]; then echo "REFUSING to start (see above)"; exit 1; fi
  "${PLAN[@]}" preflight "${ARGS[@]}" ${EXTRA[@]+"${EXTRA[@]}"} 2>&1 | tee -a "$LOG"
  if [ "${PIPESTATUS[0]}" -ne 0 ]; then echo "REFUSING to start: the preflight failed"; exit 1; fi
  rm -f "$PIDFILE"
  RULECMP_DETACHED=1 setsid -f nohup "$SELF" >> "$LOG" 2>&1 < /dev/null
  for _ in 1 2 3 4 5 6 7 8 9 10; do [ -s "$PIDFILE" ] && break; sleep 1; done
  echo "started: pid $(cat "$PIDFILE" 2> /dev/null), log $LOG, launch notes train_log/rulecmp_launch_notes.txt"
  echo "stop after the running jobs: touch $HOLD; stop at once: kill -TERM \$(cat $PIDFILE)"
  exit 0
fi

# ---- detached: the scheduler replaces this shell (same pid), pinned to CONTROL_CORE
echo $$ > "$PIDFILE"
if ! taskset -pc "$CONTROL_CORE" $$ > /dev/null; then
  echo "REFUSING: could not pin the scheduler to core $CONTROL_CORE"; rm -f "$PIDFILE"; exit 1
fi
exec "${PLAN[@]}" run "${ARGS[@]}" --pidfile "$PIDFILE" ${EXTRA[@]+"${EXTRA[@]}"}
