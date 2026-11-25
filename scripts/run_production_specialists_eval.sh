#!/bin/bash
# Evaluation of the specialists at the production batches in the environment of the paper
# (exact non-delay mask): each specialist xmask-solo-<R>-s<seed>, seeds 301 to 305, on its own
# combination R of the 30-instance sets 50x25n30 and 80x25n30 (80 jobs).
#
# Jobs: 'scripts/production_specialists_plan.py jobs <i/K>', each the evaluation plan's job of
# the policy xmask-v1-s<seed> on the same set (scripts/eval_newseeds_plan.py, group n30) with
# the model, the combination list (R alone), the schedule export (none) and the tag
# (_xmask_spec, so the folder <set>_xmask_spec) exchanged. The command of a job is built by
# build_cmd of scripts/run_eval_newseeds.sh, read from that file at start.
#
# PRE-RUN RECORD.
# 1. PURPOSE. The specialists' arrays at 50 and 80 modules, in
#    <results>/50x25n30_xmask_spec and 80x25n30_xmask_spec, that
#    scripts/p128_production_specialists.py reads: the policy against the specialist of each
#    combination at the production batches (macros Pbs), and the default level of
#    scripts/p129_project_mix.py.
# 2. IDEAL RESULT. The pre-flight reproduces the stored array, every job finishes with one row
#    per test instance and 'scripts/production_specialists_plan.py check' reports no missing
#    file. The expected result is that the policy stays near the specialists, as at 20, 27 and
#    30 modules; whatever comes is reported.
# 3. CONTAMINATION. New folders only (<set>_xmask_spec). The standard folders <set>_xmask are
#    not written: scripts/p120_lower_bound_reference.py reads every Result_*.npy there as a
#    feasible makespan. A job runs into its own staging folder and its files move into place
#    only when both exist and the array holds one row per test instance; a finished job is
#    skipped; a job whose files partly exist in place, or whose array in place holds another
#    number of rows, is refused. Rollouts are deterministic at one thread; no wall-clock of this
#    runner is a measurement.
# 4. DATA ACCURACY. Every job: CPU, OMP and MKL threads 1, --device cpu --device_id "",
#    --torch_threads 1, --first_mask exact --mask_ties exact, the data and base folders of the
#    plan's n30 job (data/FAMILY/<set>, data/PPVC/<n>x25+ppvc-mixed-n30). Pre-flight, before
#    the first job of a lane (unless a pre-flight of the same code passed before, marker
#    <LOGDIR>/preflight_passed) and in VERIFY=1: the job of xmask-solo-S-s301 on the first two
#    instances of 50x25n30 must reproduce, bit for bit, the default level of
#    scripts/p103_five_classes.py for that run (its result file of level default, 50x25n30,
#    S, greedy+10x25+family+xmask-solo-S-s301 under its root). A run is evaluated only once
#    it is trained ('scripts/production_specialists_plan.py trained <model>').
#
# Options:
#   LANE=<i>/<K>     run only the jobs whose index in the plan's order (longest first) is i
#                    modulo K (default 0/1); K lanes on K cores hold every job once
#   CORES=<core>     the one core of the lane: the launcher and every job (required outside
#                    LIST_ONLY)
#   ONLY=<regex>     only jobs whose id matches
#   LIST_ONLY=1      print the pre-flight, the plan's n30 job of the first job of the lane
#                    beside it, every job of the lane (finished ones as SKIP), the estimate and
#                    the completeness check; nothing is run or written
#   VERIFY=1         run the pre-flight only and exit (0 when the arrays agree); a pass writes
#                    the marker the lanes read
#   HOLD=<file>      while it exists no new job starts (default train_log/EVAL_HOLD)
#   STAGE_ROOT, LOCKDIR, LOGDIR, RESULTS (PACT_RESULTS follows it)   folders (tests only)
# Exit status: 0 when no job failed and none was refused, 1 when a job or the pre-flight failed
#   or a job was refused, 2 when a setting refuses.
# Usage:
#   LIST_ONLY=1 LANE=0/8 bash scripts/run_production_specialists_eval.sh
#   CORES=<free core> VERIFY=1 bash scripts/run_production_specialists_eval.sh
#   i=0; for c in <K free cores>; do LANE=$i/K CORES=$c setsid nohup bash \
#       scripts/run_production_specialists_eval.sh > train_log/prodspec_lane$i.log 2>&1 &
#       i=$((i + 1)); done
set -u
cd "$(dirname "$0")/.."
PY=$HOME/miniconda3/envs/fjsp/bin/python
LIST_ONLY=${LIST_ONLY:-0}
VERIFY=${VERIFY:-0}
LANE=${LANE:-0/1}
ONLY=${ONLY:-.}
CORES=${CORES:-}
HOLD=${HOLD:-train_log/EVAL_HOLD}
STAGE_ROOT=${STAGE_ROOT:-test_results/.eval_staging/prodspec}
LOCKDIR=${LOCKDIR:-train_log/production_specialists_eval_locks}
LOGDIR=${LOGDIR:-train_log/production_specialists_eval}
export RESULTS=${RESULTS:-test_results/FAMILY}
export PACT_RESULTS=${PACT_RESULTS:-$RESULTS}
export PACT_ENV=exact PACT_SEEDS=301-305
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
unset PACT_FIRST_MASK PACT_MASK_TOL EVAL_GROUPS EXACT_SEEDS
PP="$PY -B scripts/production_specialists_plan.py"
say() { echo "[prodspec-eval lane $LANE $(date '+%m-%d %H:%M:%S')] $*"; }

case "$LIST_ONLY" in 0|1) ;; *) say "REFUSING: LIST_ONLY must be 0 or 1, got '$LIST_ONLY'"; exit 2 ;; esac
case "$VERIFY" in 0|1) ;; *) say "REFUSING: VERIFY must be 0 or 1, got '$VERIFY'"; exit 2 ;; esac
if ! [[ "$LANE" =~ ^(0|[1-9][0-9]*)/[1-9][0-9]*$ ]] || [ "${LANE%/*}" -ge "${LANE#*/}" ]; then
  say "REFUSING: LANE must be <i>/<K> with 0 <= i < K, got '$LANE'"; exit 2
fi
lane_i=${LANE%/*}

# the command builder, the printer and the mask options of the plan's queue, from its file
QUEUE=scripts/run_eval_newseeds.sh
eval "$(sed -n '/^build_cmd() {/,/^}/p; /^show() {.*}$/p; /^MASK_OPTS=/p' "$QUEUE")"
if ! declare -F build_cmd > /dev/null || ! declare -F show > /dev/null || [ -z "${MASK_OPTS:-}" ]; then
  say "REFUSING: could not read build_cmd, show and MASK_OPTS from $QUEUE"; exit 2
fi
# the evaluation code a pre-flight vouches for (as in run_eval_newseeds.sh), with this plan
FINGERPRINT=$(cat eval_family.py params.py common_utils.py constraint_family/*.py model/*.py \
  scripts/production_specialists_plan.py scripts/eval_newseeds_plan.py "$QUEUE" 2>/dev/null \
  | sha256sum | cut -c1-16)
MARKER="$LOGDIR/preflight_passed"

if [ "$LIST_ONLY" = 1 ]; then
  CORE="<core>"
else
  [[ "$CORES" =~ ^[0-9]+$ ]] || { say "REFUSING: CORES=<one core> is required (the core of this lane), got '$CORES'"; exit 2; }
  CORE=$CORES
  # the launcher shell, and so every job, wait and helper it starts, stays on the core
  if ! taskset -pc "$CORE" $$ > /dev/null; then say "REFUSING: could not pin the launcher to CORES=$CORE"; exit 2; fi
fi

JOBFILE=$(mktemp "${TMPDIR:-/tmp}/prodspec_jobs.XXXXXX")
trap 'rm -f "$JOBFILE"' EXIT
$PP jobs "$LANE" > "$JOBFILE" || { say "REFUSING: scripts/production_specialists_plan.py jobs $LANE failed: $(tail -1 "$JOBFILE")"; exit 2; }
say "$(wc -l < "$JOBFILE") jobs in this lane; results in $RESULTS/<set>_xmask_spec"

# ---- pre-flight: the seed-301 specialist of S on two instances of 50x25n30 reproduces p103
preflight() {
  local out line stored id group camp envn device omp tt extra model methods data base tag limit regimes
  local token blind bk sched oroot folder n ckpt log pre rset files root rc
  out=$($PP preflight) || { say "PREFLIGHT FAILED: $out"; return 1; }
  line=$(sed -n 1p <<< "$out"); stored=$(sed -n 2p <<< "$out")
  IFS='|' read -r id group camp envn device omp tt extra model methods data base tag limit regimes token blind bk sched oroot folder n ckpt log pre rset files <<< "$line"
  root="$STAGE_ROOT/preflight_lane${lane_i}_$$"
  build_cmd "$CORE" "$device" "$omp" "$tt" "$extra" "$model" "$methods" "$data" "$base" \
    "$tag" "$limit" "$regimes" "$token" "$blind" "$bk" - exact "$root" "$rset"
  if [ "$LIST_ONLY" = 1 ]; then
    echo "PREFLIGHT: $(show)"
    echo "  then: $PP compare $root   (against $stored)"
    return 0
  fi
  mkdir -p "$LOGDIR"
  rm -rf "$root"; mkdir -p "$root"
  say "PREFLIGHT: $(show)"
  "${CMD[@]}" > "$LOGDIR/preflight_lane${lane_i}.log" 2>&1; rc=$?
  [ $rc -eq 0 ] && { $PP compare "$root" >> "$LOGDIR/preflight_lane${lane_i}.log" 2>&1; rc=$?; }
  say "PREFLIGHT $([ $rc -eq 0 ] && echo passed || echo FAILED): $(tail -1 "$LOGDIR/preflight_lane${lane_i}.log")"
  if [ $rc -eq 0 ]; then
    rm -rf "$root"
    echo "$FINGERPRINT $(date '+%Y-%m-%d %H:%M:%S') lane $LANE core $CORE" > "$MARKER"
  fi
  return $rc
}

# ---- the plan's n30 job of the policy beside the first job of the lane (LIST_ONLY)
show_template() {
  local first tline id group camp envn device omp tt extra model methods data base tag limit regimes
  local token blind bk sched oroot folder n ckpt log pre rset files
  first=$(head -1 "$JOBFILE" | cut -d'|' -f1)
  [ -n "$first" ] || return 0
  tline=$($PP template "$first") || { echo "TEMPLATE of $first: $tline"; return 0; }
  IFS='|' read -r id group camp envn device omp tt extra model methods data base tag limit regimes token blind bk sched oroot folder n ckpt log pre rset files <<< "$tline"
  build_cmd "$CORE" "$device" "$omp" "$tt" "$extra" "$model" "$methods" "$data" "$base" "$tag" \
    "$limit" "$regimes" "$token" "$blind" "$bk" "$sched" exact "$STAGE_ROOT/$first/out" "$rset"
  echo "TEMPLATE $id (the plan's n30 job of the policy, the comparator of $first): $(show)"
}

FAILED=/dev/null
RAN=0
one_pass() {
  local line id group camp envn device omp tt extra model methods data base tag limit regimes
  local token blind bk sched oroot folder n ckpt log pre rset files have total f
  local lk stage staged ok rc arrays
  while IFS= read -r line; do
    IFS='|' read -r id group camp envn device omp tt extra model methods data base tag limit regimes token blind bk sched oroot folder n ckpt log pre rset files <<< "$line"
    [[ "$id" =~ $ONLY ]] || continue
    for f in $files; do
      [[ "$f" == "$oroot/$folder/"* ]] || { say "REFUSING $id: $f lies outside $oroot/$folder"; echo "$id outside" >> "$FAILED"; continue 2; }
    done
    have=0; total=0
    for f in $files; do total=$((total + 1)); [ -e "$f" ] && have=$((have + 1)); done
    if [ $have -eq $total ]; then
      if $PP rows "$n" $files > /dev/null 2>&1; then
        [ "$LIST_ONLY" = 1 ] && echo "SKIP $id (finished)"; continue
      fi
      say "REFUSING $id: its files exist but an array does not hold $n rows ($($PP rows "$n" $files | head -1))"
      echo "$id length" >> "$FAILED"; continue
    fi
    if [ $have -gt 0 ]; then
      say "REFUSING $id: $have of $total files exist in place; inspect and remove them by hand"
      echo "$id partial" >> "$FAILED"; continue
    fi
    if [ "$sched" != "-" ] || [ "$envn" != exact ]; then
      say "REFUSING $id: not an exact job without schedule export"; echo "$id layout" >> "$FAILED"; continue
    fi
    if ! $PP trained "$model" > /dev/null 2>&1; then
      say "NOT TRAINED $id: $($PP trained "$model" | head -1)"; continue
    fi
    stage="$STAGE_ROOT/$id"
    build_cmd "$CORE" "$device" "$omp" "$tt" "$extra" "$model" "$methods" "$data" "$base" "$tag" \
      "$limit" "$regimes" "$token" "$blind" "$bk" - exact "$stage/out" "$rset"
    if [ "$LIST_ONLY" = 1 ]; then
      echo "JOB $id ($total files, $n instances x $(wc -w <<< "$regimes") combinations x $(wc -w <<< "$methods") methods): $(show)"
      continue
    fi
    lk="$LOCKDIR/$id"
    mkdir "$lk" 2> /dev/null || { say "SKIP $id: locked by another launcher ($(cat "$lk/pid" 2> /dev/null))"; continue; }
    echo "$BASHPID $CORE $(date +%s)" > "$lk/pid"
    have=0; for f in $files; do [ -e "$f" ] && have=$((have + 1)); done
    if [ $have -gt 0 ]; then rm -rf "$lk"; continue; fi     # finished meanwhile
    while [ -f "$HOLD" ]; do sleep 60; done
    rm -rf "$stage"; mkdir -p "$stage/out"
    say "RUN $id: $(show)"
    "${CMD[@]}" > "$LOGDIR/jobs/${id//[:\/]/_}.log" 2>&1; rc=$?
    ok=1; arrays=()
    if [ $rc -eq 0 ]; then
      for f in $files; do
        staged="$stage/out/${f#"$oroot"/}"
        [ -e "$staged" ] || { ok=0; say "FAILED $id: $staged was not written"; break; }
        [[ "$f" == *.npy ]] && arrays+=("$staged")
      done
    else ok=0; fi
    if [ $ok = 1 ] && ! $PP rows "$n" "${arrays[@]}" >> "$LOGDIR/jobs/${id//[:\/]/_}.log" 2>&1; then
      ok=0; say "FAILED $id: an array does not hold $n rows (see $LOGDIR/jobs/${id//[:\/]/_}.log)"
    fi
    if [ $ok = 1 ]; then
      for f in $files; do
        staged="$stage/out/${f#"$oroot"/}"
        mkdir -p "$(dirname "$f")"
        mv -n "$staged" "$f"
        [ -e "$staged" ] && { ok=0; say "FAILED $id: $f appeared meanwhile; the staged copy stays in $stage"; }
      done
    fi
    if [ $ok = 1 ]; then
      rm -rf "$stage"; RAN=$((RAN + 1)); say "done $id ($total files)"
    else
      echo "$id" >> "$FAILED"; say "FAILED $id (exit $rc); see $LOGDIR/jobs/${id//[:\/]/_}.log and $stage"
    fi
    rm -rf "$lk"
  done < "$JOBFILE"
}

if [ "$LIST_ONLY" = 1 ]; then
  preflight
  show_template
  one_pass
  $PP estimate 8 16
  $PP check
  exit 0
fi
if [ "$VERIFY" = 1 ]; then
  preflight; rc=$?
  exit $rc
fi
mkdir -p "$LOGDIR/jobs" "$LOCKDIR" "$STAGE_ROOT"
if [ -f "$MARKER" ] && [ "$(cut -d' ' -f1 "$MARKER")" = "$FINGERPRINT" ]; then
  say "pre-flight passed earlier for this code ($(cut -d' ' -f2- "$MARKER"))"
else
  preflight || { say "the pre-flight failed; no job started"; exit 1; }
fi
FAILED="$LOGDIR/failed_lane${lane_i}_$$.txt"
: > "$FAILED"
one_pass
say "lane finished: $RAN job(s) run"
$PP check
if [ -s "$FAILED" ]; then
  say "$(wc -l < "$FAILED") job(s) failed or were refused: $(tr '\n' ' ' < "$FAILED")"; exit 1
fi
rm -f "$FAILED"
exit 0
