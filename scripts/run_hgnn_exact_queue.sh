#!/bin/bash
# HGNN (Song et al. 2023, port in backbones/hgnn) retrained under the exact
# non-delay mask, with the per-combination training experience of the released
# policy and a light tuning of the learning rate.
#
# Purpose. The paper's environment is the exact mask (first decision
#   restricted to the non-delay rule, tied starts admitted). The stored HGNN
#   runs used the earlier mask (--hgnn_mask pact), 1000 iterations of 20
#   episodes (3,333 episodes per training combination, half the released
#   policy's 2000 updates of 20 episodes) and the upstream learning rate 2e-4
#   untuned. These runs give the comparison with a published learned scheduler
#   in the paper's environment: the same legal action set as the policy
#   (backbones/hgnn/tests/test_exact_mask_hgnn.py), 2000 iterations (6,667
#   episodes per combination, as the policy), and a learning rate chosen on the
#   validation score. Lands in the HGNN comparison of the main text and its
#   supplement table.
# Ideal result. The tuned HGNN under the exact mask is measured on the same
#   footing as the policy. If it closes the gap to the policy, the text reports
#   the new gap and drops any claim the gap no longer supports; if the gap
#   stays, the claim rests on a tuned rival with equal experience.
# Contamination. New names only: suffix -xmask after the seed (and -lr<rate>
#   for a rate other than 2e-4), so p78_hgnn_rival.py,
#   whose patterns end in -s30[0-9]_<R>.npy, -s301.log and -joint-s30[0-9].log,
#   do not read them. The script refuses to overwrite a finished run, refuses
#   to start over partial files of an unfinished one, and starts no
#   evaluation. One job on the GPU at a time: it waits while the hold file
#   exists, while either queue of the policy's exact-mask arms is alive, and
#   while any training process of this project runs on the GPU (the CPU lanes,
#   which pass --device cpu, do not count); with NO_GATE=1 it waits only while
#   the hold file exists.
#   PACT_FIRST_MASK and PACT_MASK_TOL are unset, so the validation norms are
#   the stored ones.
# Data. Same sampler (pact_bridge.sample_family_batch, identical to the
#   policy's trainer), validation set and recipe as the stored runs. Every arm
#   is built once on the CPU for one iteration and must show (i) 204250
#   parameters, (ii) a resolved configuration equal to the stored run of the
#   same arm except for the intended keys (mask, iterations, learning rate,
#   seed, suffix and name, device, out_root), each of which is checked against
#   its intended value, and (iii) the stored validation norms. The same check
#   runs on the configuration of every real run as soon as the trainer writes
#   it, and a mismatch stops the run. Reference runs:
#     PACT reward, six combinations     hgnn-certified-joint-s303
#     HGNN reward, six combinations     hgnn-published-joint-s303
#     PACT reward, one combination R    hgnn-certified-<R>-s301
#
# Phases (names <arm>-s<seed>-xmask[-lr<rate>], prefix 10x25+family+):
#   PHASE=tune             hgnn-certified-joint-s301-xmask-lr1e-4,
#                          hgnn-certified-joint-s301-xmask (2e-4),
#                          hgnn-certified-joint-s301-xmask-lr4e-4;
#                          2000 iterations each; then the selection
#                          (scripts/hgnn_select_lr.py) is printed
#   PHASE=panel LR=<rate>  hgnn-certified-joint-s302, -s303 and
#                          hgnn-published-joint-s301, -s302, -s303 at the
#                          chosen rate, 2000 iterations (seed 301 of the PACT
#                          reward is the tuning run at that rate)
#   PHASE=panel2 LR=<rate> hgnn-certified-joint-s304, -s305 at the chosen rate,
#                          2000 iterations: seeds 304 and 305 of the PACT
#                          reward, so that the comparison with the policy pairs
#                          the policy's five seeds 301-305. Same recipe,
#                          reference run and probes as the PACT-reward runs of
#                          PHASE=panel; like the panel it needs LR and the
#                          selection, not the finished panel.
#   PHASE=solo LR=<rate>   hgnn-certified-<R>-s301 for the eight combinations,
#                          1000 iterations each; runs last. The stored HGNN
#                          specialists used 1000 iterations against 1000 for
#                          the joint runs; the policy's specialists use 1000
#                          updates against 2000 for the joint run, and this
#                          phase keeps that ratio.
#   LR must be the rate the selection chose; LR_OVERRIDE=1 accepts another one
#   of 1e-4, 2e-4, 4e-4.
# GPU budget, from the stored logs (5.65 s per iteration including validation
#   for a six-combination run, 4.1 s for a one-combination run, GPU, October):
#   tune 3 x 3.1 h = 9.4 h; panel 5 x 3.1 h = 15.7 h; panel2 2 x 3.1 h = 6.3 h;
#   solo 8 x 1.1 h = 9.1 h.
#
# Gates shared with the other jobs on this machine:
#   train_log/GPU_HOLD   while this file exists, no new job starts (in every
#                        mode, NO_GATE=1 included).
#   DEADLINE=<epoch s>   a job starts only if its estimated duration ends
#                        before the deadline, and it is stopped at the deadline.
#   CORES=<list>         cores for the CPU side of the job (default 10-13). The
#                        launcher pins its own shell to them at start and stops
#                        if it cannot, so its waits and helper commands stay there.
#   THREADS=<n>          OpenMP/MKL threads (default 4, the size of CORES); the
#                        trainer's torch threads stay 1 (--threads 1). A lane on
#                        one core is started with CORES=<one core> THREADS=1.
#   LANE=<i>/<n>         run and probe only the tasks of the phase whose index in
#                        the phase's full task list is i modulo n (default 0/1).
#                        The index is taken before ONLY, so the n lanes are
#                        disjoint whatever ONLY is. With n > 1 the probes run
#                        under train_log/hgnn_xmask_probe_<i>/ (each probe keeps
#                        the run's own suffix and name, and its log is
#                        <probe root>/<name>.log).
#   NO_GATE=1            before a run, wait only for the hold file, not for the
#                        exact-mask queues or another training process on the GPU
#                        (default 0). The order of the phases holds in every
#                        mode: panel, panel2 and solo need LR and the selection
#                        of the tuning runs, and solo needs the finished panel.
#   ONLY=<regex>         run only the tasks whose name matches.
#   PROBE_ONLY=1         run the CPU probes of the phase and exit without
#                        training (PHASE=all probes every arm; LR defaults to
#                        2e-4 there; the runs of panel2 are the panel's
#                        PACT-reward arm with other seeds, and PHASE=panel2
#                        PROBE_ONLY=1 probes them).
#   PROBE_TIMEOUT=<s>    limit of one probe (default 1800).
#   LIST_ONLY=1          print the tasks this invocation would train, one line
#                        "TASK <name>" each, after LANE, ONLY, the finished runs
#                        and the runs refused for partial files, and exit 0
#                        before the probes; the order checks of the phase run as
#                        for a start and stop it the same way. LIST_ONLY=all
#                        lists every task of the lane and skips those checks.
#                        Neither writes a file or starts a training process, and
#                        neither looks at the hold file, the gates or DEADLINE.
#
# Usage: PHASE=tune setsid nohup scripts/run_hgnn_exact_queue.sh > train_log/hgnn_exact_queue.log 2>&1 &
# Lanes, one core each, one queue log per lane:
#   for i in 0 1 2; do PHASE=tune LANE=$i/3 CORES=$((4 + i)) THREADS=1 NO_GATE=1 setsid nohup \
#       scripts/run_hgnn_exact_queue.sh > train_log/hgnn_exact_queue_lane$i.log 2>&1 & done
set -u
cd "$(dirname "$0")/.."
PY=$HOME/miniconda3/envs/fjsp/bin/python
THREADS=${THREADS:-4}
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset PACT_FIRST_MASK PACT_MASK_TOL
CORES=${CORES:-10-13}
DEADLINE=${DEADLINE:-0}
ONLY=${ONLY:-.}
PROBE_ONLY=${PROBE_ONLY:-0}
PHASE=${PHASE:-tune}
LR=${LR:-}
LR_OVERRIDE=${LR_OVERRIDE:-0}
LANE=${LANE:-0/1}
NO_GATE=${NO_GATE:-0}
LIST_ONLY=${LIST_ONLY:-0}
HOLD=train_log/GPU_HOLD
WAIT_QUEUES="run_exactmask_queue.sh run_exactmask_extra_queue.sh"
PROBE_TIMEOUT=${PROBE_TIMEOUT:-1800}
TRAINER=backbones/hgnn/train_family_hgnn.py
SIX="N,L,S,W,LS,LW"
PARAMS=204250
LANE_SFX=""   # _<i> when the queue is split into lanes (probe root)
say() { echo "[hgnn-xmask${LANE_SFX:+ lane $LANE} $(date '+%m-%d %H:%M')] $*"; }

if ! [[ "$LANE" =~ ^(0|[1-9][0-9]*)/[1-9][0-9]*$ ]] || [ "${LANE%/*}" -ge "${LANE#*/}" ]; then
  say "ABORT: LANE must be <i>/<n> with 0 <= i < n, got '$LANE'"; exit 1
fi
lane_i=${LANE%/*}; lane_n=${LANE#*/}
if [ "$lane_n" -gt 1 ]; then LANE_SFX="_${lane_i}"; fi
PROBE_ROOT=train_log/hgnn_xmask_probe${LANE_SFX}
case "$NO_GATE" in 0|1) ;; *) say "ABORT: NO_GATE must be 0 or 1, got '$NO_GATE'"; exit 1 ;; esac
case "$LIST_ONLY" in 0|1|all) ;; *) say "ABORT: LIST_ONLY must be 0, 1 or all, got '$LIST_ONLY'"; exit 1 ;; esac
# The launcher shell itself, and so every probe, sleep and pgrep it starts,
# stays on CORES.
if ! taskset -pc "$CORES" $$ > /dev/null; then say "ABORT: could not pin the launcher to CORES=$CORES"; exit 1; fi
ncores=$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc)
if [ "$THREADS" -gt "$ncores" ]; then say "WARNING: THREADS=$THREADS threads on the $ncores core(s) of CORES=$CORES"; fi

lr_tag() {  # rate -> name part ('' for the upstream rate)
  case "$1" in
    2e-4) echo "" ;;
    1e-4|4e-4) echo "-lr$1" ;;
    *) return 1 ;;
  esac
}

case "$PHASE" in
  tune) ;;
  panel|panel2|solo)
    if [ -z "$LR" ]; then say "ABORT: PHASE=$PHASE needs LR=<rate> (scripts/hgnn_select_lr.py)"; exit 1; fi ;;
  all)
    if [ "$PROBE_ONLY" != "1" ]; then say "ABORT: PHASE=all is for PROBE_ONLY=1"; exit 1; fi
    LR=${LR:-2e-4} ;;
  *) say "ABORT: PHASE must be tune, panel, panel2 or solo (or all with PROBE_ONLY=1), got '$PHASE'"; exit 1 ;;
esac
if [ -n "$LR" ] && ! lr_tag "$LR" > /dev/null; then say "ABORT: LR=$LR is not one of 1e-4, 2e-4, 4e-4"; exit 1; fi

# task: name | reward | regimes | iterations | seed | rate | reference run | minutes alone on the GPU
TASKS=()
add_task() {  # reward regime-tag regimes iterations seed rate reference minutes
  local sfx; sfx="xmask$(lr_tag "$6")"
  TASKS+=("hgnn-$1-$2-s$5-${sfx}|$1|$3|$4|$5|$6|$7|$8")
}
if [ "$PHASE" = tune ] || [ "$PHASE" = all ]; then
  for r in 1e-4 2e-4 4e-4; do add_task certified joint "$SIX" 2000 301 "$r" hgnn-certified-joint-s303 190; done
fi
if [ "$PHASE" = panel ] || [ "$PHASE" = all ]; then
  for s in 302 303; do add_task certified joint "$SIX" 2000 "$s" "$LR" hgnn-certified-joint-s303 190; done
  for s in 301 302 303; do add_task published joint "$SIX" 2000 "$s" "$LR" hgnn-published-joint-s303 190; done
fi
if [ "$PHASE" = panel2 ]; then
  for s in 304 305; do add_task certified joint "$SIX" 2000 "$s" "$LR" hgnn-certified-joint-s303 190; done
fi
if [ "$PHASE" = solo ] || [ "$PHASE" = all ]; then
  for R in N L S W LS LW SW LSW; do add_task certified "$R" "$R" 1000 301 "$LR" "hgnn-certified-${R}-s301" 70; done
fi

# The one command builder of the probes and the runs: they differ only in the
# iteration count, the device and (probe only) the output root.
build_cmd() {  # reward regimes iterations seed rate suffix device -> CMD
  CMD=($PY $TRAINER --family_regimes "$2" --hgnn_reward "$1" --hgnn_mask exact
       --max_iterations "$3" --seed "$4" --lr "$5" --model_suffix "$6"
       --device "$7" --threads 1)
}

check_config() {  # config file, reference run, name, iterations, seed, rate, suffix, device, out_root
  CFG="$1" REF="$2" WANT_NAME="$3" WANT_ITERS="$4" WANT_SEED="$5" WANT_LR="$6" WANT_SUFFIX="$7" \
  WANT_DEVICE="$8" WANT_OUTROOT="$9" WANT_PARAMS="$PARAMS" $PY - << 'PYEOF'
import json, os
e = os.environ
ref_name = e['REF']
ref = json.load(open(f"train_log/FAMILY/config_10x25+family+{ref_name}.json"))
new = json.load(open(e['CFG']))

def flat(d, p=''):
    out = {}
    for k, v in d.items():
        q = f'{p}.{k}' if p else k
        if isinstance(v, dict):
            out.update(flat(v, q))
        else:
            out[q] = v
    return out

R, N = flat(ref), flat(new)
want = {'model_name': '10x25+family+' + e['WANT_NAME'],
        'hgnn_mask': 'exact', 'args.hgnn_mask': 'exact',
        'args.max_iterations': int(e['WANT_ITERS']), 'train_paras.max_iterations': int(e['WANT_ITERS']),
        'args.lr': float(e['WANT_LR']), 'train_paras.lr': float(e['WANT_LR']),
        'args.seed': int(e['WANT_SEED']), 'args.model_suffix': e['WANT_SUFFIX'],
        'args.device': e['WANT_DEVICE'], 'args.out_root': e['WANT_OUTROOT']}
for k, v in want.items():
    if N.get(k, '<absent>') != v:
        raise SystemExit(f"ABORT: {e['CFG']} records {k}={N.get(k, '<absent>')!r}, want {v!r}")
if R.get('hgnn_mask') != 'pact' or R.get('args.max_iterations') not in (1000,):
    raise SystemExit(f"ABORT: the reference {ref_name} is not a stored earlier-mask run of 1000 iterations")
for k in ('n_params',):
    if N.get(k) != int(e['WANT_PARAMS']) or R.get(k) != int(e['WANT_PARAMS']):
        raise SystemExit(f"ABORT: n_params {N.get(k)} (reference {R.get(k)}), want {e['WANT_PARAMS']}")
diffs = sorted(k for k in set(R) | set(N) if R.get(k, '<absent>') != N.get(k, '<absent>'))
bad = [k for k in diffs if k not in want]
if bad:
    raise SystemExit(f"ABORT: configuration drift against {ref_name}: "
                     + str({k: (R.get(k, '<absent>'), N.get(k, '<absent>')) for k in bad}))
print(f"[hgnn-xmask] {e['WANT_NAME']}: configuration equal to {ref_name} except {diffs}; "
      f"intended values checked; {N['n_params']} parameters; validation norms "
      + ' '.join(f'{k}={v}' for k, v in new['vali_norm'].items()))
PYEOF
}

probe() {  # one TASKS entry
  IFS='|' read -r name rew regs its seed rate ref est <<< "$1"
  local sfx=${name#*-s${seed}-} root="${PROBE_ROOT}/${name}"
  local plog="${root}.log"
  rm -rf "$root" "$plog"; mkdir -p "$root"
  # CPU only: no CUDA device is visible to the probe
  build_cmd "$rew" "$regs" 1 "$seed" "$rate" "$sfx" cpu
  timeout $PROBE_TIMEOUT taskset -c $CORES env CUDA_VISIBLE_DEVICES= "${CMD[@]}" --out_root "$root" > "$plog" 2>&1
  local rc=$? np
  np=$(sed -nE 's/.*\[model\] params total=([0-9]+).*/\1/p' "$plog" | head -1)
  if [ $rc -ne 0 ]; then say "ABORT: the probe of ${name} failed (exit $rc, see $plog)"; return 1; fi
  if [ "$np" != "$PARAMS" ]; then say "ABORT: ${name} has $np parameters, want $PARAMS (see $plog)"; return 1; fi
  if ! grep -q "mask=exact iterations=1 " "$plog" || ! grep -q "^Ep 1[[:space:]]" "$plog"; then
    say "ABORT: the probe of ${name} did not run one iteration under the exact mask (see $plog)"; return 1
  fi
  check_config "${root}/train_log/FAMILY/config_10x25+family+${name}.json" "$ref" "$name" 1 "$seed" "$rate" \
      "$sfx" cpu "$root" || return 1
  rm -rf "$root" "$plog"
  rmdir "$PROBE_ROOT" 2>/dev/null
  return 0
}

# One GPU job at a time: true while a training process of this project runs on
# the GPU, i.e. train_family.py, train_family_hgnn.py or train_resched_family.py
# without --device cpu. The CPU lanes (the public-instance SD2 runs and every
# probe pass --device cpu) do not count, and only Python processes count, so a
# shell whose command line merely names a trainer does not.
gpu_training_running() {
  pgrep -af "[t]rain_(resched_)?family(_hgnn)?\.py" | awk '
    $2 !~ /python[0-9.]*$/ { next }
    $3 ~ /train_(resched_)?family(_hgnn)?\.py$/ && $0 !~ /--device cpu/ { found = 1 }
    END { exit found ? 0 : 1 }'
}
# true while a named queue script runs (a shell executing it, not a command line
# that mentions it)
queue_alive() {
  local q
  for q in $WAIT_QUEUES; do
    pgrep -af "[${q:0:1}]${q:1}" | awk -v q="$q" '
      $2 ~ /(^|\/)(ba)?sh$/ && (substr($3, length($3) - length(q) + 1) == q) { found = 1 }
      END { exit found ? 0 : 1 }' && return 0
  done
  return 1
}

# true while a run must not start: the hold file in every mode; unless
# NO_GATE=1, also a live exact-mask queue of the policy or a training process
# on the GPU. Lanes of this script never wait for each other through
# queue_alive (WAIT_QUEUES does not name this script); with the default they
# wait for each other through gpu_training_running, one GPU job at a time.
must_wait() {
  [ -f "$HOLD" ] && return 0
  [ "$NO_GATE" = "1" ] && return 1
  queue_alive || gpu_training_running
}

files_of() {  # name -> checkpoint, configuration, validation curve, per-iteration log
  echo "trained_network/FAMILY/10x25+family+$1.pth train_log/FAMILY/config_10x25+family+$1.json" \
       "train_log/FAMILY/valiquality_10x25+family+$1.txt train_log/FAMILY/reward_10x25+family+$1.txt"
}
finished() {  # name -> true when its checkpoint exists and its log has the completion line
  [ -f "trained_network/FAMILY/10x25+family+$1.pth" ] && grep -q "done in" "train_log/FAMILY_$1.log" 2>/dev/null
}
partial_of() {  # name -> the files of an unfinished run that exist (empty if none)
  local f out=""
  for f in $(files_of "$1") "train_log/FAMILY_$1.log"; do [ -e "$f" ] && out="$out $f"; done
  echo "$out"
}

# The tasks of this lane: index i modulo n over the phase's full list, then ONLY.
MINE=()
idx=-1
for t in "${TASKS[@]}"; do
  idx=$(( idx + 1 ))
  [ $(( idx % lane_n )) -eq "$lane_i" ] || continue
  [[ "${t%%|*}" =~ $ONLY ]] || continue
  MINE+=("$t")
done

if { [ "$PHASE" = panel ] || [ "$PHASE" = panel2 ] || [ "$PHASE" = solo ]; } && [ "$PROBE_ONLY" != "1" ] && [ "$LIST_ONLY" != "all" ]; then
  sel=$($PY scripts/hgnn_select_lr.py --quiet 2>&1)
  if [ $? -ne 0 ]; then say "ABORT: the learning-rate selection is not available: $sel"; exit 1; fi
  if [ "$sel" != "$LR" ]; then
    if [ "$LR_OVERRIDE" = "1" ]; then say "LR=$LR differs from the selection ($sel); LR_OVERRIDE=1 accepts it"
    else say "ABORT: LR=$LR, but the selection chose $sel (LR_OVERRIDE=1 to accept another rate)"; exit 1; fi
  fi
fi
if [ "$PHASE" = solo ] && [ "$PROBE_ONLY" != "1" ] && [ "$LIST_ONLY" != "all" ]; then
  lt=$(lr_tag "$LR")
  for n in hgnn-certified-joint-s302 hgnn-certified-joint-s303 hgnn-published-joint-s301 \
           hgnn-published-joint-s302 hgnn-published-joint-s303; do
    if ! grep -q "done in" "train_log/FAMILY_${n}-xmask${lt}.log" 2>/dev/null; then
      say "ABORT: PHASE=solo runs last; ${n}-xmask${lt} of PHASE=panel is not finished"; exit 1
    fi
  done
fi

if [ "$LIST_ONLY" != "0" ]; then
  for t in "${MINE[@]}"; do
    name=${t%%|*}
    if [ "$LIST_ONLY" = "1" ]; then
      if finished "$name"; then say "SKIP ${name} (finished)" >&2; continue; fi
      partial=$(partial_of "$name")
      if [ -n "$partial" ]; then say "REFUSE ${name}: files of an unfinished run exist:${partial}" >&2; continue; fi
    fi
    echo "TASK ${name}"
  done
  exit 0
fi

say "phase ${PHASE}${LR:+, lr $LR}: probing ${#TASKS[@]} arms on the CPU (cores $CORES) ..."
for t in "${MINE[@]}"; do
  probe "$t" || exit 1
done
say "all arms verified"
if [ "$PROBE_ONLY" = "1" ]; then say "PROBE_ONLY=1: not starting the queue"; exit 0; fi
export CUDA_VISIBLE_DEVICES=0
say "starting the queue"

for t in "${MINE[@]}"; do
  IFS='|' read -r name rew regs its seed rate ref est <<< "$t"
  sfx=${name#*-s${seed}-}
  log="train_log/FAMILY_${name}.log"
  pth="trained_network/FAMILY/10x25+family+${name}.pth"
  cfg="train_log/FAMILY/config_10x25+family+${name}.json"
  if finished "$name"; then say "SKIP ${name} (finished)"; continue; fi
  partial=$(partial_of "$name")
  if [ -n "$partial" ]; then say "REFUSE ${name}: files of an unfinished run exist:${partial}"; continue; fi
  while must_wait; do sleep 120; done
  tmo=""
  if [ "$DEADLINE" -gt 0 ]; then
    now=$(date +%s); left=$(( DEADLINE - now ))
    if [ $left -lt $(( est * 60 )) ]; then say "STOP before ${name}: ${left}s to the deadline, the run needs about $(( est * 60 ))s"; break; fi
    tmo="timeout -s TERM ${left}"
  fi
  say "TRAIN ${name} (reward ${rew}, regimes ${regs}, ${its} iterations, seed ${seed}, lr ${rate}, exact mask)"
  build_cmd "$rew" "$regs" "$its" "$seed" "$rate" "$sfx" cuda
  $tmo taskset -c $CORES "${CMD[@]}" > "$log" 2>&1 &
  pid=$!
  # the trainer prints its banner once the resolved configuration is written,
  # before the first iteration
  ok=1 started=0
  for _ in $(seq 1 120); do
    grep -q "^\[train_family_hgnn\] model=" "$log" 2>/dev/null && { started=1; break; }
    kill -0 $pid 2>/dev/null || break
    sleep 5
  done
  if [ $started -eq 1 ]; then
    if ! check_config "$cfg" "$ref" "$name" "$its" "$seed" "$rate" "$sfx" cuda "$PWD"; then ok=0; fi
  elif kill -0 $pid 2>/dev/null; then
    say "no configuration from ${name} after 10 minutes"; ok=0
  fi
  if [ $ok -ne 1 ]; then kill -TERM $pid 2>/dev/null; fi
  wait $pid
  rc=$?
  np=$(sed -nE 's/.*\[model\] params total=([0-9]+).*/\1/p' "$log" | head -1)
  if [ $ok -ne 1 ] || [ $rc -ne 0 ] || ! grep -q "done in" "$log"; then
    say "FAILED ${name} (exit $rc); removing its partial files"
    rm -f $(files_of "$name")
    mv "$log" "${log}.failed" 2>/dev/null
    [ $ok -ne 1 ] && { say "configuration check failed; stopping"; exit 1; }
    [ $rc -eq 124 ] && { say "the deadline ended the run; stopping"; break; }
    continue
  fi
  if [ "$np" != "$PARAMS" ]; then say "ABORT ${name}: $np parameters"; exit 1; fi
  say "trained ${name}: $(grep 'done in' "$log" | tail -1 | sed -E 's#checkpoint /[^ ]*/(trained_network/)#checkpoint \1#')"
done
if [ "$PHASE" = tune ]; then
  say "learning-rate selection (scripts/hgnn_select_lr.py):"
  $PY scripts/hgnn_select_lr.py || say "selection not available yet (a tuning run is missing)"
fi
say "queue finished"
