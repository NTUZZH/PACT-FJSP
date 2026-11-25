#!/bin/bash
# Evaluation of the specialists trained for 2000 updates (xlong-solo-<R>-s<seed>,
# scripts/run_longspec_queue.sh): one greedy pass on the test instances of the run's own
# combination R at the training size (10x25), in the environment of the paper.
#
# The call. A job is the evaluation plan's principal 10x25 job of xmask-solo-<R>-s<seed>
#   (scripts/eval_newseeds_plan.py, group principal, read through seed_extension.plan_jobs)
#   with the run name exchanged and the combinations restricted to R
#   (scripts/longspec_runs.py evaljob). Its command is built by build_cmd of
#   scripts/run_eval_newseeds.sh, read from that file at start, so options, data folders,
#   device and threads are those of the principal call: OMP and MKL threads 1,
#   CUDA_VISIBLE_DEVICES empty, eval_family.py --methods greedy --data data/FAMILY/10x25
#   --base data/PPVC/10x25+ppvc-mixed --device cpu --device_id "" --torch_threads 1
#   --tag _xmask --first_mask exact --mask_ties exact, pinned to one core. Only --out_root
#   differs: the job writes into its own staging folder, and its files then move to
#   <results>/10x25_xlong. eval_family.py evaluates each combination in its own loop, with a
#   new environment per instance and no random draw in a greedy pass, so the array of R does
#   not depend on the other combinations of a call; the pre-flight checks this on N.
#
# PRE-RUN RECORD.
# 1. PURPOSE. The arrays that scripts/p117_long_specialists.py reads: each longer specialist on
#    its own combination, set against the 1000-update specialist and the policy of the same
#    seed. The numbers land in the supplement section on the validation score during
#    training (manuscript/supp_new/sn06b_convergence.tex) and in one clause of Section 5.3.
# 2. IDEAL RESULT. The pre-flight reproduces the stored array, every job finishes with one row
#    per test instance, and 'scripts/longspec_runs.py state --seeds <SEEDS>' reports every run
#    trained and evaluated.
# 3. CONTAMINATION. The arrays go to <results>/10x25_xlong, never to 10x25_xmask. No glob of
#    the analysis reads that folder; the completeness check of the evaluation plan lists every
#    folder of the results root but keeps only files of stored runs, and xlong- names none.
#    A job runs into its own staging folder, and its files move into place only when both
#    exist and the array holds one row per test instance. A job whose files partly exist in
#    place, or whose array in place holds another number of rows, is refused; a finished job
#    is skipped. No wall-clock of this runner is a measurement.
# 4. DATA ACCURACY. Pre-flight, before the first job and in VERIFY=1: the stored specialist
#    xmask-solo-N-s301 evaluated on N with the same call into a temporary folder must give
#    the makespans of its stored array in 10x25_xmask within 1e-6 (the largest difference is
#    printed). A run is evaluated only once its training checks are met
#    ('scripts/longspec_runs.py trained': completion line, 32786 parameters, exact mask, the
#    configuration of xmask-solo-<R>-s<seed> except model_suffix and max_updates). A run whose
#    log holds no completion line yet is left for a later invocation; a finished run whose
#    checks fail is refused.
#
# Options:
#   CORE=<core>      the one core of every job, and of the launcher (required outside
#                    LIST_ONLY)
#   SEEDS=<list>     training seeds (default 301; seeds 301 to 305, separated by spaces or
#                    commas, ranges a-b allowed)
#   ONLY=<regex>     only the runs whose name matches
#   LIST_ONLY=1      print the pre-flight command and one line per run: what this invocation
#                    would do with it (evaluate, skip, await training, refuse) and its command;
#                    nothing is run or written
#   VERIFY=1         run the pre-flight only and exit (0 when the arrays agree)
#   HOLD=<file>      while it exists no new job starts (default train_log/EVAL_HOLD)
#   STAGE_ROOT, LOGDIR, RESULTS (PACT_RESULTS follows it)   folders (tests only)
# Exit status: 0 when no job failed and no run was refused, 1 when a job or the pre-flight
#   failed or a run was refused, 2 when a setting refuses.
# Usage:
#   CORE=<free core> VERIFY=1 bash scripts/run_longspec_eval.sh
#   CORE=<free core> SEEDS=301 setsid nohup bash scripts/run_longspec_eval.sh \
#       > train_log/longspec_eval.log 2>&1 &
#
# Variant. Without an argument the runs above are evaluated. With the one argument xshare
#   (given by scripts/run_budgetspec_eval.sh, whose header holds its pre-run record) the runs
#   xshare-solo-<R>-s<seed> of 250 updates are evaluated with the same call and the same
#   pre-flight, with the helper scripts/budgetspec_runs.py (arrays in <results>/10x25_xshare),
#   staging folders xshare_<name> and xshare_verify_<pid>, LOGDIR train_log/budgetspec_eval and
#   SEEDS 301-310 by default (seeds 301 to 310; the job of a seed 306 to 310 is the plan's job
#   of xmask-solo-<R>-s305 with the run name exchanged, see scripts/longspec_runs.py). Any other
#   argument refuses.
set -u
cd "$(dirname "$0")/.."
PY=$HOME/miniconda3/envs/fjsp/bin/python
if [ $# -gt 1 ] || { [ $# -eq 1 ] && [ "$1" != xshare ]; }; then
  echo "[longspec-eval] REFUSING: the one optional argument is the variant xshare, got '$*'"; exit 2
fi
VARIANT=${1:-}
HELP="$PY -B scripts/longspec_runs.py"
SEEDS_DEFAULT=301
LOGDIR_DEFAULT=train_log/longspec_eval
STAGE=xlong                    # staging folders <STAGE_ROOT>/<STAGE>_<name> and <STAGE>_verify_<pid>
WHO=longspec-eval              # the word of the log lines
if [ "$VARIANT" = xshare ]; then
  HELP="$PY -B scripts/budgetspec_runs.py"
  SEEDS_DEFAULT=301-310
  LOGDIR_DEFAULT=train_log/budgetspec_eval
  STAGE=xshare
  WHO=budgetspec-eval
fi
CORE=${CORE:-}
SEEDS=${SEEDS:-$SEEDS_DEFAULT}
ONLY=${ONLY:-.}
LIST_ONLY=${LIST_ONLY:-0}
VERIFY=${VERIFY:-0}
HOLD=${HOLD:-train_log/EVAL_HOLD}
STAGE_ROOT=${STAGE_ROOT:-test_results/.eval_staging}
LOGDIR=${LOGDIR:-$LOGDIR_DEFAULT}
export RESULTS=${RESULTS:-test_results/FAMILY}
export PACT_RESULTS=${PACT_RESULTS:-$RESULTS}
export PACT_ENV=exact
unset PACT_FIRST_MASK PACT_MASK_TOL
say() { echo "[$WHO $(date '+%m-%d %H:%M:%S')] $*"; }

case "$LIST_ONLY" in 0|1) ;; *) say "REFUSING: LIST_ONLY must be 0 or 1, got '$LIST_ONLY'"; exit 2 ;; esac
case "$VERIFY" in 0|1) ;; *) say "REFUSING: VERIFY must be 0 or 1, got '$VERIFY'"; exit 2 ;; esac

# the command builder, the printer and the mask options of the plan's queue, from its file
QUEUE=scripts/run_eval_newseeds.sh
eval "$(sed -n '/^build_cmd() {/,/^}/p; /^show() {.*}$/p; /^MASK_OPTS=/p' "$QUEUE")"
if ! declare -F build_cmd > /dev/null || ! declare -F show > /dev/null || [ -z "${MASK_OPTS:-}" ]; then
  say "REFUSING: could not read build_cmd, show and MASK_OPTS from $QUEUE"; exit 2
fi

if [ "$LIST_ONLY" = 1 ]; then
  C=${CORE:-<core>}
else
  [[ "$CORE" =~ ^[0-9]+$ ]] || { say "REFUSING: CORE=<one core> is required, got '$CORE'"; exit 2; }
  # the launcher shell, and so every job and helper it starts, stays on CORE
  if ! taskset -pc "$CORE" $$ > /dev/null; then say "REFUSING: could not pin the launcher to CORE=$CORE"; exit 2; fi
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
  C=$CORE
fi

NAMES=$($HELP list --seeds "$SEEDS") || { say "$NAMES"; exit 2; }
NAMES=$(awk '{print $1}' <<< "$NAMES")

parse() {  # one job line -> the fields of 'eval_newseeds_plan.py jobs'
  IFS='|' read -r id group camp envn device omp tt extra model methods data base tag limit \
    regimes token blind bk sched oroot folder n ckpt log pre rset files <<< "$1"
}

command_of() {  # root -> CMD (build_cmd of the plan's queue, the job's own fields)
  build_cmd "$C" "$device" "$omp" "$tt" "$extra" "$model" "$methods" "$data" "$base" "$tag" \
    "$limit" "$regimes" "$token" "$blind" "$bk" - exact "$1" "$rset"
}

# ---- pre-flight: the stored xmask-solo-N-s301 on N with the job's call, into a temporary folder
PREFLIGHT_DONE=0
preflight() {
  local line root rc stored staged plog
  line=$($HELP verifyjob) || { say "PREFLIGHT FAILED: $line"; return 1; }
  parse "$line"
  if [ "$sched" != "-" ] || [ "$envn" != exact ]; then say "PREFLIGHT FAILED: not an exact job without schedule export"; return 1; fi
  root="$STAGE_ROOT/${STAGE}_verify_$$"
  command_of "$root"
  stored=$files
  staged="$root/$folder/$(basename "$stored")"
  if [ "$LIST_ONLY" = 1 ]; then
    echo "PREFLIGHT: $(show)"
    echo "  then: $HELP compare $staged $stored"
    return 0
  fi
  mkdir -p "$LOGDIR"
  plog="$LOGDIR/preflight_$$.log"
  rm -rf "$root"; mkdir -p "$root"
  say "PREFLIGHT: $(show)"
  "${CMD[@]}" > "$plog" 2>&1; rc=$?
  [ $rc -eq 0 ] && { $HELP compare "$staged" "$stored" >> "$plog" 2>&1; rc=$?; }
  say "PREFLIGHT $([ $rc -eq 0 ] && echo passed || echo FAILED): $(tail -1 "$plog")"
  rm -rf "$root"
  [ $rc -eq 0 ] && PREFLIGHT_DONE=1
  return $rc
}

if [ "$VERIFY" = 1 ]; then
  preflight; exit $?
fi
[ "$LIST_ONLY" = 1 ] && preflight

FAILED=0
RAN=0
for name in $NAMES; do
  [[ "$name" =~ $ONLY ]] || continue
  line=$($HELP evaljob "$name") || { say "REFUSING $name: $line"; FAILED=$((FAILED + 1)); continue; }
  parse "$line"
  if [ "$sched" != "-" ] || [ "$envn" != exact ]; then
    say "REFUSING $name: not an exact job without schedule export"; FAILED=$((FAILED + 1)); continue
  fi
  read -r array summary <<< "$files"
  stage="$STAGE_ROOT/${STAGE}_$name"
  command_of "$stage/out"
  have=0; for f in $files; do [ -e "$f" ] && have=$((have + 1)); done
  if [ $have -eq 2 ]; then
    if out=$($HELP rows "$n" "$array"); then
      [ "$LIST_ONLY" = 1 ] && echo "TASK $name: skip (evaluated, $n rows): $(show)"
      [ "$LIST_ONLY" = 1 ] || say "SKIP $name (evaluated: $array holds $n rows)"
      continue
    fi
    say "REFUSING $name: $out; inspect and remove the files by hand"; FAILED=$((FAILED + 1)); continue
  fi
  if [ $have -gt 0 ]; then
    say "REFUSING $name: $have of 2 files exist in place ($files); inspect and remove them by hand"
    FAILED=$((FAILED + 1)); continue
  fi
  if ! why=$($HELP trained "$name"); then
    if grep -q "done in" "$log" 2>/dev/null; then       # finished, but a training check fails
      [ "$LIST_ONLY" = 1 ] && echo "TASK $name: refuse ($why): $(show)"
      [ "$LIST_ONLY" = 1 ] || say "REFUSING $name: $why"
      FAILED=$((FAILED + 1))
    else
      [ "$LIST_ONLY" = 1 ] && echo "TASK $name: await training: $(show)"
      [ "$LIST_ONLY" = 1 ] || say "WAIT $name: not trained yet (no completion line in $log)"
    fi
    continue
  fi
  if [ "$LIST_ONLY" = 1 ]; then echo "TASK $name: evaluate: $(show)"; continue; fi
  if [ "$PREFLIGHT_DONE" = 0 ]; then
    preflight || { say "the pre-flight failed; no job started"; exit 1; }
    # the pre-flight parsed its own job and built its own command: read this job again
    parse "$line"
    read -r array summary <<< "$files"
    command_of "$stage/out"
  fi
  while [ -f "$HOLD" ]; do sleep 60; done
  mkdir -p "$LOGDIR"
  rm -rf "$stage"; mkdir -p "$stage/out"
  say "RUN $name: $(show)"
  "${CMD[@]}" > "$LOGDIR/$name.log" 2>&1; rc=$?
  ok=1
  if [ $rc -ne 0 ]; then ok=0; fi
  for f in $files; do
    [ $ok = 1 ] || break
    [ -e "$stage/out/$folder/$(basename "$f")" ] || { ok=0; say "FAILED $name: $stage/out/$folder/$(basename "$f") was not written"; }
  done
  if [ $ok = 1 ] && ! out=$($HELP rows "$n" "$stage/out/$folder/$(basename "$array")"); then
    ok=0; say "FAILED $name: $out"
  fi
  if [ $ok = 1 ]; then
    for f in $files; do
      mkdir -p "$(dirname "$f")"
      mv -n "$stage/out/$folder/$(basename "$f")" "$f"
      [ -e "$stage/out/$folder/$(basename "$f")" ] && { ok=0; say "FAILED $name: $f appeared meanwhile; the staged copy stays in $stage"; }
    done
  fi
  if [ $ok = 1 ]; then
    rm -rf "$stage"; RAN=$((RAN + 1)); say "done $name: $array"
  else
    FAILED=$((FAILED + 1)); say "FAILED $name (exit $rc); see $LOGDIR/$name.log and $stage"
  fi
done
[ "$LIST_ONLY" = 1 ] && exit 0
say "finished: $RAN evaluated, $FAILED failed or refused"
$HELP state --seeds "$SEEDS" | head -1
[ "$FAILED" -eq 0 ] || exit 1
exit 0
