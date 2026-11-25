#!/bin/bash
# Retraining of the three principal arms under the exact mask
# (--first_mask exact --mask_ties exact), training seeds 301 to 305.
#
# Purpose. The simulator's mask deviates from the stated non-delay rule in two
#   places: at the first decision of an episode it admits every compatible
#   pair, and its start-time comparison on the normalized axis drops some pairs
#   that tie the earliest start. Both were measured without retraining. These
#   runs train every principal arm under the corrected mask, so the parity
#   verdicts, the lead over the rule pool and the single-rule control can be
#   compared between the legacy mask and the exact mask with the training run
#   as the unit. Lands in the supplement's mask section (exact against legacy,
#   per regime) and, if a verdict moves, in Section 8.1.
# Ideal result. Every verdict keeps its direction under the exact mask, with
#   makespan changes of the size the evaluation-only checks measured. If a
#   verdict changes, the claim is restated for the exact mask, which is the
#   mask the paper defines.
# Contamination. New names only, prefix xmask- (no existing glob or regular
#   expression of the analysis scripts matches it); the script refuses to
#   overwrite a finished run. No evaluation is started here, so no result file
#   enters test_results/. Exact-mask evaluations belong in their own folders
#   (eval_family.py --tag _xmask, which eval_family.py requires for an exact
#   mask). PACT_FIRST_MASK and PACT_MASK_TOL are unset and the two options are
#   passed explicitly; each finished run must record both as exact.
# Data. Same sampler, validation set and recipe as the legacy seeds: each arm
#   is built once on the CPU and must show (i) 32786 parameters, (ii) a
#   resolved configuration equal to the arm's stored legacy seed-303
#   configuration except for name, seed, device, max_updates and the two mask
#   options, which must read exact, and (iii) the same MWKR validation norms as
#   the seed-303 log for every multi-regime arm and for N and L (the norm env
#   stays on the legacy mask by design, so validation scores of both masks share
#   one scale). The stored specialist logs of S, W, SW and LSW carry the norm of
#   the earlier rule implementation; for one regime the norm is a constant factor
#   of the selection score, so that difference is printed, not refused.
#
# The second five seeds, 306 to 310 (SEED_TIERS="306 307 308 309 310"; the runs, their
#   files and the rule are in scripts/seed_extension.py).
# Purpose. The extension rule: if, with five seeds, the interval of any single
#   combination of the principal comparison (policy against specialist at 10x25) is
#   inconclusive at the 1% margin, seeds 306 to 310 of the policy and the eight
#   specialists are trained and every verdict is recomputed with ten seeds, under the
#   same verdict rule. The numbers land in the table of the principal comparison
#   (scripts/p110_seed_extension.py). The queue refuses to train a second seed unless
#   'scripts/seed_extension.py triggered' exits 0 (the rule applies in the five-seed
#   record); the rule names no control, so the control has no task of a second seed.
# Ideal result. Every combination is conclusive at the 1% margin with ten seeds. A
#   combination that stays inconclusive is reported as inconclusive with ten seeds.
# Contamination. New names only (xmask-v1-s306..s310, xmask-solo-<R>-s306..s310); no
#   glob or regular expression of the analysis and export scripts takes them into an
#   analysis of the first five seeds (arm_registry refuses their seeds); the script
#   refuses to overwrite a finished run.
# Data. Same sampler, validation set and recipe as the first five seeds: the probes and
#   the checks below (32786 parameters, the exact mask recorded); in addition each
#   finished run of a second seed must have a configuration equal to the stored
#   configuration of the same arm at seed 301 except model_suffix and seed_train, its
#   own name and seed (scripts/seed_extension.py trained); a difference aborts the queue.
#
# Arms (one job on the GPU at a time, unless the queue runs as lanes, see
#   LANE below), names <arm>-s<seed>:
#   xmask-v1-s301..s305         the released recipe, six regimes, 2000 updates
#   xmask-ctl-s301..s305        the single-rule control, N,L,S,W, 2000 updates
#   xmask-solo-<R>-s301..s305   one specialist per regime R, 1000 updates
#   with SEED_TIERS naming the second five seeds: xmask-v1-s306..s310 and
#   xmask-solo-<R>-s306..s310 (45 runs, no control)
# Order: per tier of SEED_TIERS (default "301 302 303;304 305"), the seeds of the
#   tier for the released recipe, then for the control (seeds 301 to 305 only),
#   then the specialists of each seed of the tier.
#
#   SEED_TIERS=<tiers>   tiers separated by ';', seeds of a tier by spaces (default
#                        "301 302 303;304 305", the two tiers above). Seeds 301 to 305
#                        and the second five seeds 306 to 310 are accepted, each once.
#
# Gates shared with the other jobs on this machine:
#   HOLD_FILE=<file>     default train_log/GPU_HOLD: while this file exists, no new
#                        job starts (in every mode, NO_GATE=1 included).
#   DEADLINE=<epoch s>   a job starts only if its estimated duration ends
#                        before the deadline, and it is killed at the deadline.
#   CORES=<list>         cores for the CPU side of the job (default 10-13). The
#                        launcher pins its own shell to them at start and stops
#                        if it cannot, so its waits and helper commands stay there.
#   THREADS=<n>          OpenMP/MKL threads (default 4, the size of CORES). A lane
#                        on one core is started with CORES=<one core> THREADS=1.
#   LANE=<i>/<n>         run only the tasks whose index in the full task list
#                        (the order below) is i modulo n (default 0/1). The index
#                        is taken before ONLY, so the n lanes are disjoint whatever
#                        ONLY is. With n > 1 the probe's model suffix and probe log
#                        carry the lane number (_xmprobe_<i>, xmask_probe_<i>.log).
#   NO_GATE=1            start a run without waiting for other training jobs
#                        (default 0). This queue waits for no other job in either
#                        mode, only for the hold file, so its lanes share the GPU
#                        in both; the option is accepted for one launch line across
#                        the exact-mask queues.
#   ONLY=<regex>         run only the tasks whose name matches.
#   PROBE_ONLY=1         run the CPU probes and exit without training.
#   LIST_ONLY=1          print the tasks this invocation would train, one line
#                        "TASK <name>" each, after LANE, ONLY and the finished
#                        runs, and exit 0 before the probes; LIST_ONLY=all lists
#                        the finished runs too. Neither starts Python or writes a
#                        file; neither looks at the hold file or DEADLINE.
#
# Usage: setsid nohup scripts/run_exactmask_queue.sh > train_log/exactmask_queue.log 2>&1 &
# Four lanes, one core each, one queue log per lane:
#   for i in 0 1 2 3; do LANE=$i/4 CORES=$((4 + i)) THREADS=1 setsid nohup \
#       scripts/run_exactmask_queue.sh > train_log/exactmask_queue_lane$i.log 2>&1 & done
# The second five seeds, seven lanes on the cores 0, 1, 2, 4, 5, 6 and 7 (lane i on
#   the i-th of them), one queue log per lane; lane 0:
#   SEED_TIERS="306 307 308 309 310" LANE=0/7 CORES=0 THREADS=1 setsid nohup \
#       scripts/run_exactmask_queue.sh > train_log/exactmask_seedext_lane0.log 2>&1 &
#   all seven:
#   i=0; for c in 0 1 2 4 5 6 7; do SEED_TIERS="306 307 308 309 310" LANE=$i/7 CORES=$c \
#       THREADS=1 setsid nohup scripts/run_exactmask_queue.sh \
#       > train_log/exactmask_seedext_lane$i.log 2>&1 & i=$((i + 1)); done
set -u
cd "$(dirname "$0")/.."
PY=$HOME/miniconda3/envs/fjsp/bin/python
THREADS=${THREADS:-4}
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset PACT_FIRST_MASK PACT_MASK_TOL
CORES=${CORES:-10-13}
DEADLINE=${DEADLINE:-0}
ONLY=${ONLY:-.}
PROBE_ONLY=${PROBE_ONLY:-0}
LANE=${LANE:-0/1}
NO_GATE=${NO_GATE:-0}
LIST_ONLY=${LIST_ONLY:-0}
SEED_TIERS=${SEED_TIERS:-301 302 303;304 305}
HOLD=${HOLD_FILE:-train_log/GPU_HOLD}
MASK_OPTS="--first_mask exact --mask_ties exact"
LANE_SFX=""   # _<i> when the queue is split into lanes (probe file names)
say() { echo "[exactmask${LANE_SFX:+ lane $LANE} $(date '+%m-%d %H:%M')] $*"; }

if ! [[ "$LANE" =~ ^(0|[1-9][0-9]*)/[1-9][0-9]*$ ]] || [ "${LANE%/*}" -ge "${LANE#*/}" ]; then
  say "ABORT: LANE must be <i>/<n> with 0 <= i < n, got '$LANE'"; exit 1
fi
lane_i=${LANE%/*}; lane_n=${LANE#*/}
if [ "$lane_n" -gt 1 ]; then LANE_SFX="_${lane_i}"; fi
case "$NO_GATE" in 0|1) ;; *) say "ABORT: NO_GATE must be 0 or 1, got '$NO_GATE'"; exit 1 ;; esac
case "$LIST_ONLY" in 0|1|all) ;; *) say "ABORT: LIST_ONLY must be 0, 1 or all, got '$LIST_ONLY'"; exit 1 ;; esac
# SEED_TIERS: seeds 301 to 305, and the second five seeds 306 to 310, each at most once
if ! [[ "$SEED_TIERS" =~ ^[0-9\ \;]+$ ]]; then say "ABORT: SEED_TIERS must hold seeds, spaces and ';', got '$SEED_TIERS'"; exit 1; fi
IFS=';' read -ra TIER_LIST <<< "$SEED_TIERS"
SECOND=0      # 1 when a tier names a second seed (306 to 310)
seen=" "
for tier in "${TIER_LIST[@]}"; do
  [ -n "${tier// /}" ] || { say "ABORT: SEED_TIERS has an empty tier: '$SEED_TIERS'"; exit 1; }
  for S in $tier; do
    if ! [[ "$S" =~ ^[1-9][0-9]{0,5}$ ]]; then say "ABORT: seed '$S' in SEED_TIERS is not a seed number"; exit 1
    elif [ "$S" -ge 301 ] && [ "$S" -le 305 ]; then :
    elif [ "$S" -ge 306 ] && [ "$S" -le 310 ]; then SECOND=1
    else say "ABORT: seed $S in SEED_TIERS: seeds 301 to 305, or the second five seeds 306 to 310"; exit 1; fi
    case "$seen" in *" $S "*) say "ABORT: seed $S appears twice in SEED_TIERS"; exit 1 ;; esac
    seen="$seen$S "
  done
done
# The launcher shell itself, and so every probe, sleep and helper it starts,
# stays on CORES.
if ! taskset -pc "$CORES" $$ > /dev/null; then say "ABORT: could not pin the launcher to CORES=$CORES"; exit 1; fi
ncores=$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc)
if [ "$THREADS" -gt "$ncores" ]; then say "WARNING: THREADS=$THREADS threads on the $ncores core(s) of CORES=$CORES"; fi

probe() {  # $1 regimes  $2 updates  $3 reference run (legacy seed 303)
  local suf="_xmprobe${LANE_SFX}"
  local plog="train_log/xmask_probe${LANE_SFX}.log"
  # NVML device count only, so the CPU probe never initializes CUDA
  timeout 900 taskset -c $CORES env PYTORCH_NVML_BASED_CUDA_CHECK=1 \
      $PY train_family.py --data_source FAMILY --n_j 10 \
      --use_type_embedding True --validate_timestep 25 --use_film False \
      --family_regimes "$1" --max_updates 1 --device cpu $MASK_OPTS \
      --model_suffix "$suf" --seed_train 301 > "$plog" 2>&1
  local np
  np=$(sed -nE 's/.*\[model\] params total=([0-9]+).*/\1/p' "$plog" | head -1)
  if [ "$np" != "32786" ]; then say "ABORT: $np parameters for regimes $1, want 32786 (see $plog)"; return 1; fi
  if ! grep -q "^\[train_family\] MASK: first_mask=exact mask_ties=exact" "$plog"; then
    say "ABORT: the probe for regimes $1 did not train under the exact mask (see $plog)"; return 1
  fi
  REF="$3" WANT_UPD="$2" WANT_REG="$1" PLOG="$plog" SUF="$suf" $PY - << 'PYEOF' || return 1
import json, os, re
ALLOWED = {'model_suffix', 'seed_train', 'device', 'max_updates', 'first_mask', 'mask_ties'}
INERT_NEW = {'dr_shift': ('',), 'delta_mode': (None,), 'mask_mode': (None, 'nondelay'),
             'pair_extra': (None, 'none'), 'reward_mode': ('dense',),
             'family_base': ('ppvc',), 'pub_vali_dir': ('',), 'pub_type_source': ('none',)}
ref_name = os.environ['REF']
ref = json.load(open(f"train_log/FAMILY/config_10x25+family+{ref_name}.json"))
probe = json.load(open(f"train_log/FAMILY/config_10x25+family+{os.environ['SUF']}.json"))
assert ref['family_regimes'] == os.environ['WANT_REG'], (ref['family_regimes'], os.environ['WANT_REG'])
assert int(ref['max_updates']) == int(os.environ['WANT_UPD']), ref['max_updates']
assert int(ref['seed_train']) == 303, ref['seed_train']
for k in ('first_mask', 'mask_ties'):
    if probe.get(k) != 'exact':
        raise SystemExit(f'ABORT: the probe configuration records {k}={probe.get(k)!r}, want exact')
    if ref.get(k, 'legacy') != 'legacy':
        raise SystemExit(f'ABORT: the reference {ref_name} records {k}={ref.get(k)!r}, want legacy')
diffs = set()
for k in set(ref) | set(probe):
    rv, pv = ref.get(k, '<absent>'), probe.get(k, '<absent>')
    if rv == pv:
        continue
    if k in INERT_NEW and rv == '<absent>' and pv in INERT_NEW[k]:
        continue
    diffs.add(k)
bad = diffs - ALLOWED
if bad:
    raise SystemExit(f"ABORT: configuration drift against {ref_name}: "
                     + str({k: (ref.get(k), probe.get(k)) for k in sorted(bad)}))
# Same validation set and scale: the MWKR norms of the probe against the reference
# log. With several regimes the norms weight the selection score, so all must be
# equal. With one regime the norm is a constant factor of that score; N and L must
# still be equal (their MWKR value fingerprints the validation set), while the
# stored specialist logs of the setup and window regimes carry the norm of the
# rule implementation that preceded the mask-restricted machine choice, so a
# difference there is printed, not refused.
def norms(p):
    m = re.search(r'\[vali\] MWKR norms: ([^\n]*)', open(p, errors='replace').read())
    return None if m is None else dict(re.findall(r'\b([A-Z]+)=([\d.]+)', m.group(1)))
want, got = norms(f'train_log/FAMILY_{ref_name}.log'), norms(os.environ['PLOG'])
if not want or not got or set(want) != set(got):
    raise SystemExit(f"ABORT: MWKR norm lines of {ref_name} and the probe do not match: {want} vs {got}")
moved = {r: (want[r], got[r]) for r in want if want[r] != got[r]}
if moved and (len(want) > 1 or set(moved) & {'N', 'L'}):
    raise SystemExit(f"ABORT: MWKR norms differ from {ref_name} (stored, probe): {moved}")
note = (f"; single-regime norm (stored, probe) {moved}, a constant factor of the selection score"
        if moved else '; MWKR norms equal')
print(f"[exactmask] configuration against {ref_name} OK, differs only in {sorted(diffs)}"
      f"{note} ({' '.join(f'{k}={v}' for k, v in got.items())})")
PYEOF
  # the probe's own files only (the trainer writes no other .txt), so a probe
  # never removes the files of another lane's probe
  rm -f "trained_network/FAMILY/10x25+family+${suf}.pth" \
        "train_log/FAMILY/config_10x25+family+${suf}.json" \
        "train_log/FAMILY/valiquality_10x25+family+${suf}.txt" \
        "train_log/FAMILY/reward_10x25+family+${suf}.txt" "$plog" 2>/dev/null
  return 0
}

# name | regimes | updates | seed | estimated minutes with the GPU to itself
# (the control of seeds 301 to 305 only: the extension rule names no control)
TASKS=()
for tier in "${TIER_LIST[@]}"; do
  for S in $tier; do TASKS+=("xmask-v1-s${S}|N,L,S,W,LS,LW|2000|${S}|125"); done
  for S in $tier; do [ "$S" -le 305 ] && TASKS+=("xmask-ctl-s${S}|N,L,S,W|2000|${S}|125"); done
  for S in $tier; do
    for R in N L S W LS LW SW LSW; do TASKS+=("xmask-solo-${R}-s${S}|${R}|1000|${S}|65"); done
  done
done

finished() {  # name -> true when its checkpoint exists and its log has the completion line
  [ -f "trained_network/FAMILY/10x25+family+$1.pth" ] && grep -q "done in" "train_log/FAMILY_$1.log" 2>/dev/null
}

# The tasks of this lane: index i modulo n over the full list, then ONLY.
MINE=()
idx=-1
for t in "${TASKS[@]}"; do
  idx=$(( idx + 1 ))
  [ $(( idx % lane_n )) -eq "$lane_i" ] || continue
  [[ "${t%%|*}" =~ $ONLY ]] || continue
  MINE+=("$t")
done
if [ "$LIST_ONLY" != "0" ]; then
  for t in "${MINE[@]}"; do
    name=${t%%|*}
    if [ "$LIST_ONLY" = "1" ] && finished "$name"; then say "SKIP ${name} (finished)" >&2; continue; fi
    echo "TASK ${name}"
  done
  exit 0
fi

# the second five seeds are trained only while the extension rule applies in the
# five-seed record (exit 0 of scripts/seed_extension.py triggered)
if [ "$SECOND" = 1 ]; then
  if ! env PACT_ENV=exact PACT_SEEDS=301-305 PACT_OUT="${PACT_OUT:-out/exact}" \
      $PY -B scripts/seed_extension.py triggered; then
    say "ABORT: SEED_TIERS names the second five seeds and the extension rule does not apply (scripts/seed_extension.py triggered)"; exit 1
  fi
fi

say "verifying the three recipes under the exact mask on the CPU (cores $CORES) ..."
probe "N,L,S,W,LS,LW" 2000 "jointblind-v1-s303" || exit 1
probe "N,L,S,W" 2000 "jointblind-nlsw-s303" || exit 1
for R in N L S W LS LW SW LSW; do probe "$R" 1000 "spec-${R}-s303" || exit 1; done
say "all recipes verified"
if [ "$PROBE_ONLY" = "1" ]; then say "PROBE_ONLY=1: not starting the queue"; exit 0; fi
say "starting the queue"

for t in "${MINE[@]}"; do
  IFS='|' read -r name regs upd seed est <<< "$t"
  log="train_log/FAMILY_${name}.log"
  pth="trained_network/FAMILY/10x25+family+${name}.pth"
  cfg="train_log/FAMILY/config_10x25+family+${name}.json"
  if finished "$name"; then say "SKIP ${name} (finished)"; continue; fi
  while [ -f "$HOLD" ]; do sleep 120; done   # the only wait of this queue, NO_GATE=1 included
  tmo=""
  if [ "$DEADLINE" -gt 0 ]; then
    now=$(date +%s); left=$(( DEADLINE - now ))
    if [ $left -lt $(( est * 60 )) ]; then say "STOP before ${name}: ${left}s to the deadline, the run needs about $(( est * 60 ))s"; break; fi
    tmo="timeout -s TERM ${left}"
  fi
  say "TRAIN ${name} (regimes ${regs}, ${upd} updates, seed ${seed}, exact mask)"
  $tmo taskset -c $CORES $PY train_family.py --data_source FAMILY --n_j 10 \
      --use_type_embedding True --validate_timestep 25 --use_film False \
      --family_regimes "$regs" --max_updates "$upd" --seed_train "$seed" \
      $MASK_OPTS --model_suffix "$name" > "$log" 2>&1
  rc=$?
  np=$(sed -nE 's/.*\[model\] params total=([0-9]+).*/\1/p' "$log" | head -1)
  if [ $rc -ne 0 ] || ! grep -q "done in" "$log"; then
    say "FAILED ${name} (exit $rc); removing its partial files"
    rm -f "$pth" "$cfg"
    mv "$log" "${log}.failed" 2>/dev/null
    [ $rc -eq 124 ] && { say "the deadline ended the run; stopping"; break; }
    continue
  fi
  if [ "$np" != "32786" ]; then say "ABORT ${name}: $np parameters"; exit 1; fi
  if ! grep -q '"first_mask": "exact"' "$cfg" || ! grep -q '"mask_ties": "exact"' "$cfg"; then
    say "ABORT ${name}: its configuration does not record the exact mask"; exit 1
  fi
  # a second seed: the configuration of the same arm at seed 301 except name and seed
  if [ "$seed" -ge 306 ] && ! env PACT_ENV=exact PACT_SEEDS=301-305 \
      $PY -B scripts/seed_extension.py trained "$name"; then
    say "ABORT ${name}: not trained as the same arm at seed 301 (scripts/seed_extension.py trained)"; exit 1
  fi
  say "trained ${name}: $(grep 'done in' "$log" | tail -1)"
done
say "queue finished"
