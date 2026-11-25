#!/bin/bash
# Evaluation of the policies trained with other held-out pairs of combinations
# (xsplit-<split>-s<seed>, scripts/run_heldout_splits_queue.sh): one greedy pass on the
# test instances of all eight combinations at the training size (10x25), in the
# environment of the paper.
#
# The call. A job is the evaluation plan's principal 10x25 job of xmask-v1-s<seed>
#   (scripts/eval_newseeds_plan.py, group principal, read through seed_extension.plan_jobs)
#   with the run name exchanged (scripts/xsplit_runs.py evaljob). Its command is built by
#   build_cmd of scripts/run_eval_newseeds.sh, read from that file at start, so options,
#   data folders, device and threads are those of the principal call: OMP and MKL threads 1,
#   CUDA_VISIBLE_DEVICES empty, eval_family.py --methods greedy --data data/FAMILY/10x25
#   --base data/PPVC/10x25+ppvc-mixed --regimes N L S W LS LW SW LSW --device cpu
#   --device_id "" --torch_threads 1 --tag _xmask --first_mask exact --mask_ties exact,
#   pinned to one core. Only --out_root differs: the job writes into its own staging
#   folder, and its nine files (eight arrays and the summary) then move to
#   <results>/10x25_xsplit. About ten minutes per run on one core.
#
# PRE-RUN RECORD.
# 1. PURPOSE. The arrays that scripts/p122_heldout_splits.py reads: each split policy on its
#    two held-out combinations against the specialist of the same seed, on its six training
#    combinations, and against xmask-v1 of the same seed. The numbers land in one sentence and
#    one row group of the held-out results and in a supplement table.
# 2. IDEAL RESULT. The pre-flight reproduces the stored array, every job finishes with one row
#    per test instance in each of its eight arrays, and 'scripts/xsplit_runs.py state
#    --seeds <SEEDS>' reports every run trained and evaluated.
# 3. CONTAMINATION. The arrays go to <results>/10x25_xsplit, never to 10x25_xmask. No glob of
#    the analysis reads that folder; the completeness check of the evaluation plan lists every
#    folder of the results root but keeps only files of stored runs, and xsplit- names none.
#    A job runs into its own staging folder, and its files move into place only when all nine
#    exist and every array holds one row per test instance. A job whose files partly exist in
#    place, or whose arrays in place hold another number of rows, is refused; a finished job
#    is skipped. No wall-clock of this runner is a measurement.
# 4. DATA ACCURACY. Pre-flight, before the first job and in VERIFY=1: the stored policy
#    xmask-v1-s301 evaluated with the same call on LSW, first 10 instances, into a temporary
#    folder must give the makespans of the first 10 rows of its stored array in 10x25_xmask
#    within 1e-6 (the largest difference is printed). A run is evaluated only once its
#    training checks are met ('scripts/xsplit_runs.py trained': completion line, 32786
#    parameters, exact mask, the configuration of xmask-v1-s<seed> except model_suffix and
#    family_regimes). A run whose log holds no completion line yet is left for a later
#    invocation; a finished run whose checks fail is refused.
#
# Options:
#   CORE=<core>      the one core of every job, and of the launcher (required outside
#                    LIST_ONLY)
#   SEEDS=<list>     training seeds (default 301-303; seeds 301 to 303, separated by spaces or
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
#   CORE=<free core> VERIFY=1 scripts/run_heldout_splits_eval.sh
#   CORE=<free core> setsid nohup scripts/run_heldout_splits_eval.sh \
#       > train_log/xsplit_eval.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=$HOME/miniconda3/envs/fjsp/bin/python
CORE=${CORE:-}
SEEDS=${SEEDS:-301-303}
ONLY=${ONLY:-.}
LIST_ONLY=${LIST_ONLY:-0}
VERIFY=${VERIFY:-0}
HOLD=${HOLD:-train_log/EVAL_HOLD}
STAGE_ROOT=${STAGE_ROOT:-test_results/.eval_staging}
LOGDIR=${LOGDIR:-train_log/xsplit_eval}
export RESULTS=${RESULTS:-test_results/FAMILY}
export PACT_RESULTS=${PACT_RESULTS:-$RESULTS}
export PACT_ENV=exact
unset PACT_FIRST_MASK PACT_MASK_TOL
HELP="$PY -B scripts/xsplit_runs.py"
say() { echo "[xsplit-eval $(date '+%m-%d %H:%M:%S')] $*"; }

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

# ---- pre-flight: the stored xmask-v1-s301 on LSW, first 10 instances, with the job's call,
# into a temporary folder
PREFLIGHT_DONE=0
preflight() {
  local line root rc stored staged plog
  line=$($HELP verifyjob) || { say "PREFLIGHT FAILED: $line"; return 1; }
  parse "$line"
  if [ "$sched" != "-" ] || [ "$envn" != exact ]; then say "PREFLIGHT FAILED: not an exact job without schedule export"; return 1; fi
  root="$STAGE_ROOT/xsplit_verify_$$"
  command_of "$root"
  stored=$files
  staged="$root/$folder/$(basename "$stored")"
  if [ "$LIST_ONLY" = 1 ]; then
    echo "PREFLIGHT: $(show)"
    echo "  then: $HELP compare $staged $stored $limit"
    return 0
  fi
  mkdir -p "$LOGDIR"
  plog="$LOGDIR/preflight_$$.log"
  rm -rf "$root"; mkdir -p "$root"
  say "PREFLIGHT: $(show)"
  "${CMD[@]}" > "$plog" 2>&1; rc=$?
  [ $rc -eq 0 ] && { $HELP compare "$staged" "$stored" "$limit" >> "$plog" 2>&1; rc=$?; }
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
  read -r -a FILES <<< "$files"
  nfiles=${#FILES[@]}
  ARRAYS=("${FILES[@]:0:nfiles-1}")            # the eight arrays; the last file is the summary
  stage="$STAGE_ROOT/xsplit_$name"
  command_of "$stage/out"
  have=0; for f in "${FILES[@]}"; do [ -e "$f" ] && have=$((have + 1)); done
  if [ $have -eq "$nfiles" ]; then
    if out=$($HELP rows "$n" "${ARRAYS[@]}"); then
      [ "$LIST_ONLY" = 1 ] && echo "TASK $name: skip (evaluated, $n rows per array): $(show)"
      [ "$LIST_ONLY" = 1 ] || say "SKIP $name (evaluated: ${#ARRAYS[@]} arrays of $n rows)"
      continue
    fi
    say "REFUSING $name: $out; inspect and remove the files by hand"; FAILED=$((FAILED + 1)); continue
  fi
  if [ $have -gt 0 ]; then
    say "REFUSING $name: $have of $nfiles files exist in place ($files); inspect and remove them by hand"
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
    read -r -a FILES <<< "$files"
    nfiles=${#FILES[@]}
    ARRAYS=("${FILES[@]:0:nfiles-1}")
    command_of "$stage/out"
  fi
  while [ -f "$HOLD" ]; do sleep 60; done
  mkdir -p "$LOGDIR"
  rm -rf "$stage"; mkdir -p "$stage/out"
  say "RUN $name: $(show)"
  "${CMD[@]}" > "$LOGDIR/$name.log" 2>&1; rc=$?
  ok=1
  if [ $rc -ne 0 ]; then ok=0; fi
  for f in "${FILES[@]}"; do
    [ $ok = 1 ] || break
    [ -e "$stage/out/$folder/$(basename "$f")" ] || { ok=0; say "FAILED $name: $stage/out/$folder/$(basename "$f") was not written"; }
  done
  if [ $ok = 1 ]; then
    STAGED=(); for f in "${ARRAYS[@]}"; do STAGED+=("$stage/out/$folder/$(basename "$f")"); done
    if ! out=$($HELP rows "$n" "${STAGED[@]}"); then ok=0; say "FAILED $name: $out"; fi
  fi
  if [ $ok = 1 ]; then
    for f in "${FILES[@]}"; do
      mkdir -p "$(dirname "$f")"
      mv -n "$stage/out/$folder/$(basename "$f")" "$f"
      [ -e "$stage/out/$folder/$(basename "$f")" ] && { ok=0; say "FAILED $name: $f appeared meanwhile; the staged copy stays in $stage"; }
    done
  fi
  if [ $ok = 1 ]; then
    rm -rf "$stage"; RAN=$((RAN + 1)); say "done $name: ${#ARRAYS[@]} arrays in $(dirname "${ARRAYS[0]}")"
  else
    FAILED=$((FAILED + 1)); say "FAILED $name (exit $rc); see $LOGDIR/$name.log and $stage"
  fi
done
[ "$LIST_ONLY" = 1 ] && exit 0
say "finished: $RAN evaluated, $FAILED failed or refused"
$HELP state --seeds "$SEEDS" | head -1
[ "$FAILED" -eq 0 ] || exit 1
exit 0
