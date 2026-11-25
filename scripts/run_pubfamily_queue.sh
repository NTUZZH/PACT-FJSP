#!/bin/bash
# Training queue of the public-instance family (SD2 distribution, 10 jobs on 5
# machines), on the CPU. Decided 2026-11-11.
#
# Purpose. A second benchmark family built on public flexible job-shop
#   instances with best-known values: does one policy trained on six rule
#   combinations match the specialists and keep its quality on the two held-out
#   combinations outside the factory benchmark? Lands in the public-instance
#   table of the results section (SD2 10x5 in distribution, SD2 20x10,
#   Brandimarte and Hurink as transfer) and in the supplement.
# Ideal result. Parity with the specialists inside the margin and no loss on SW
#   and LSW. If a verdict fails on this family, it is reported as the boundary
#   of the finding; either outcome enters the paper.
# Contamination. New names only: checkpoints 10x5+pubfamily+<name>, logs
#   train_log/PUBFAMILY_<name>.log (outside the FAMILY_*.log glob that the
#   training-cost and release scripts read). The script refuses to overwrite a
#   finished run and starts no evaluation. CPU only (CUDA hidden), so it never
#   touches the GPU queue; training wall-clock of this family is not reported.
# Data. Instances from public_adapter.sd2_instance (distribution checked
#   against the released SD2 files by tests/test_public_adapter.py); validation
#   on the first 20 stored records of data/PUBLIC/family/pub_SD2_vali_10x5+mix;
#   rules drawn by constraint_family/public_rules.py. Each arm is built once
#   and must show 29602 parameters and a resolved configuration equal to the
#   factory recipe of the same arm (stored seed-303 configuration) except for
#   the instance family, the machine count, the type embedding, the validation
#   folder, the device, the name, the seed and the two mask options.
#
# MASK=exact (default) trains under the stated non-delay rule
#   (--first_mask exact --mask_ties exact), names pubx-*; MASK=legacy trains in
#   the environment of the released factory policy, names publ-*.
# TYPES=none (default) runs the arms above unchanged (--use_type_embedding
#   False, 29602 parameters). TYPES=class (exact mask only) feeds the setup
#   class of each operation's job through the type embedding
#   (--use_type_embedding True --pub_type_source class, tables of 4 classes and
#   1 station type, constraint_family/public_types.py), names pubc-*, 32714
#   parameters; everything else as for pubx.
# Arms, names <prefix>-<arm>-s<seed>, seeds 301 to 305:
#   v1         six combinations N,L,S,W,LS,LW, 2000 updates
#   ctl        single-rule control N,L,S,W, 2000 updates
#   solo-<R>   one specialist per combination R, 1000 updates
# Order: seeds 301-303 (v1, ctl, specialists), then seeds 304-305.
#
# Equal-experience arm (ARMS=eq, decided 2026-11-12; MASK=exact TYPES=none only):
#   eq         the v1 recipe with 6000 updates (arm_registry.PUB_EQ_UPDATES), names
#              pubx-eq-s<seed>, seeds 301 to 305 in this order. The v1 policy sees
#              each of its six combinations for a third of the episodes a
#              specialist sees of its one; with 6000 updates it sees as many. The
#              factory family's arm for the same question is xmask-eq
#              (scripts/run_exactmask_extra_queue.sh), with the 6000 updates of the
#              stored jointblind-eq runs. Purpose: whether the regime-N shortfall
#              of v1 against the specialist (+2.12%, 95% interval 0.37 to 3.97 on
#              SD2 10x5) closes with equal experience; lands in the PubEq macros of
#              scripts/p89_public_family.py. Contamination: new names only, the
#              default task list is unchanged and never contains them.
#   The arm is built once (one update at seed 301, cores CORES) and must show
#   29602 parameters, the exact mask and a resolved configuration equal to the
#   stored configuration of the finished run pubx-v1-s301 except for the name
#   and max_updates; the five stored pubx-v1 configurations must be one recipe
#   (they differ only in the name and the seed), and the stored factory runs
#   jointblind-eq-s301..303 must record 6000 updates on the same six
#   combinations. After each run, its stored configuration must equal that of
#   pubx-v1 of the same seed except for the name and max_updates = 6000.
#
# Rule arms (ARMS=<word>, decided 2026-11-13; MASK=exact TYPES=none only): the v1
#   recipe (six combinations N,L,S,W,LS,LW, 2000 updates) trained under another
#   admissibility rule, --mask_mode <rule>, and nothing else changed. RULE_ARMS below
#   lists each rule with the word of its run names ("<rule>|<word>"); one more entry
#   adds the arm of a further rule. Names pubx-<word>-s<seed>, seeds 301 to 303 in this
#   order (arm_registry.PUB_ACT_SEEDS).
#   act        rule active, the Giffler-Thompson condition for active schedules: a pair
#              is admissible while its start is earlier than the earliest completion
#              over the compatible pairs, so a machine may wait for a later job. Names
#              pubx-act-s<seed> (arm_registry.PUB_ACT_*).
#   Purpose: on SD2 10x5 the four dispatching rules are about 6% shorter under the
#     active rule than under the non-delay rule of every other run here
#     (scripts/p107_mask_rule_variants.py); does the trained policy gain as much?
#     Lands in scripts/p108_public_active_arm.py (macros PubAct*), in the supplement's
#     section on the controls for the mask and in one sentence of the main text. The
#     factory family's arm for the same question is xmask-active
#     (scripts/run_exactmask_extra_queue.sh).
#   Expected result: unknown in sign. Shorter: the admissibility rule is a
#     configuration choice that the machine structure of a shop decides. Equal or
#     longer: the trained policy does not need the wider rule where the dispatching
#     rules gain from it. Both outcomes enter the paper.
#   Contamination: new names only; the default and eq task lists are unchanged and
#     never contain them, and p89 reads none of their files. eval_family.py restores
#     mask_mode from a run's stored configuration, so the evaluation (group pubact of
#     scripts/eval_newseeds_plan.py) rolls the policy out under its own rule.
#   Data: the instances, validation records and rules of pubx-v1; the comparator is
#     pubx-v1 of the same seed.
#   The arm is built once (one update at seed 301, cores CORES) and must show 29602
#   parameters, the exact first-decision and tie options, the rule in the trainer's
#   banner and in mask_mode, and a resolved configuration equal to the stored
#   configuration of the finished run pubx-v1-s301 except for the name, mask_mode and
#   the probe's one update; the stored pubx-v1-s301..303 configurations must be one
#   recipe and record the non-delay rule. After each run, its stored configuration
#   must equal that of pubx-v1 of the same seed except for the name and mask_mode.
#
# Family arms under a rule (ARMS=<word>fam and ARMS=<word>eq for a word of RULE_ARMS,
#   decided 2026-11-14; MASK=exact TYPES=none only): every recipe of this family trained
#   under the rule of the rule arm, --mask_mode <rule>, and nothing else changed. The run
#   name puts the rule word right after the recipe word of the non-delay name
#   (arm_registry.PUB_RULE_STEM, pub_rule_model); the policy keeps the name of the rule arm:
#     pubx-v1-s<seed>        ->  pubx-<word>-s<seed>
#     pubx-ctl-s<seed>       ->  pubx-ctl<word>-s<seed>
#     pubx-solo-<R>-s<seed>  ->  pubx-solo<word>-<R>-s<seed>
#     pubx-eq-s<seed>        ->  pubx-eq<word>-s<seed>
#   actfam     the default task list under the active rule: the policy pubx-act-s<seed>, the
#              single-rule control pubx-ctlact-s<seed> and the specialists
#              pubx-soloact-<R>-s<seed>, seeds 301 to 305 (arm_registry.PUB_ACTFAM_SEEDS),
#              in the default order: seeds 301-303 (policy, control, specialists), then
#              seeds 304-305. The policy runs of seeds 301-303 are the runs of ARMS=act: they
#              must be finished, LIST_ONLY=all lists them, and this arm never trains them, so
#              it trains 47 runs, the 27 of seeds 301-303 first.
#   acteq      the eq recipe (6000 updates) under the active rule, names pubx-eqact-s<seed>,
#              for the seeds whose non-delay run pubx-eq-s<seed> is finished, read from the
#              stored files when the task list is built (seeds 301 to 305 on 2026-11-14).
#   Purpose: on SD2 10x5 the policy trained and evaluated under the active rule is 9.67%
#     shorter than under the non-delay rule (scripts/p108_public_active_arm.py); do the
#     findings of this benchmark (one policy against one specialist per combination, the
#     single-rule control, the policy with equal experience, the dispatching rules and the
#     references) hold under that rule? Lands in the supplement's section on the public
#     benchmark and in its section on the controls for the mask.
#   Expected result: unknown. A policy within the margin of its specialists under the active
#     rule means the finding holds under both rules; otherwise the paper reports where it
#     does not.
#   Contamination: new names only. No glob or name pattern of p89, p108 or the evaluation
#     plan takes a control, specialist or eq run of these arms (pubx-act-* of p108 and
#     pubx-eq-* of p89 need the separator after the word; tests/test_pubactfam_arm.py). The
#     policy runs pubx-act-s304 and pubx-act-s305 carry the name of the rule arm, so the arm
#     check of p108 (seeds 301-303) counts their files as unexpected and p108 refuses once
#     one of them exists. The default, eq and act task lists are unchanged and never contain
#     these names. eval_family.py restores mask_mode from a run's stored configuration.
#   Data: the instances, validation records and rules of the non-delay runs; the comparator
#     of each run is its non-delay counterpart of the same seed (pubx-v1, pubx-ctl,
#     pubx-solo-<R>, pubx-eq).
#   Each recipe among the lane's tasks is built once (one update at seed 301, cores CORES)
#   and must show the parameter count of its counterpart's training log, the exact
#   first-decision and tie options, the rule in the trainer's banner and in mask_mode, and a
#   resolved configuration equal to the stored configuration of its counterpart at seed 301
#   except for the name, mask_mode and the probe's one update; the stored counterparts of
#   the recipe's seeds in the arm must be one recipe under the non-delay rule. After each
#   run, its log must state the rule and its stored configuration must equal that of its
#   counterpart of the same seed except for the name and mask_mode.
#
# Second five seeds of the family under the active rule (ARMS=actfam with SEED_TIERS naming
#   seeds 306 to 310, decided 2026-11-15; scripts/pub_seed_extension.py holds the runs, their
#   files and their checks). AN ADDITION AFTER THE FIVE-SEED RESULT WAS KNOWN: the five-seed
#   record of scripts/p89_public_family.py --family active stays the reading of record.
#   Runs: the policy pubx-act-s306..s310 and the specialists pubx-soloact-<R>-s306..s310, 45
#   runs, the recipes of the first five seeds with nothing changed but the seed and the name;
#   no single-rule control and no eq recipe above seed 305. Order of a tier: the policy runs of
#   its seeds, then the eight specialists of each seed, so LANE=<i>/<n> with n of at least 5
#   puts one long policy run on each of the lanes 0 to 4.
#   Purpose: the comparison of the policy with the specialist of each combination on
#     pub_SD2_test_10x5+mix with ten training seeds (scripts/p115_public_seed_extension.py),
#     reported beside the five-seed record.
#   Expected result: unknown; the ten-seed rows are reported whatever they show.
#   Contamination: new names only; the default task list and every other arm are unchanged;
#     the readers of the first five seeds take none of their files
#     (tests/test_public_seed_extension.py).
#   Data: the instances, validation records and rules of the first five seeds; CPU only, as for
#     the first five seeds.
#   Each recipe among the lane's tasks of a second seed is built once (one update at seed 301,
#   cores CORES) and must show the parameter count of the training log of its seed-301 run
#   <stem>-s301, the exact first-decision and tie options, the rule in the trainer's banner and
#   in mask_mode, and a resolved configuration equal to the stored configuration of <stem>-s301
#   except for the name, the seed and the probe's one update; the stored runs <stem>-s301..s305
#   must be finished and one recipe (scripts/pub_seed_extension.py probecheck). After each run
#   its log must state the rule and its stored configuration must equal that of <stem>-s301
#   except for the name and the seed (scripts/pub_seed_extension.py trained); a difference
#   aborts the queue.
#
# Controls:
#   train_log/PUB_HOLD   while this file exists, no new run starts (runs of seeds 306 to 310
#                        excepted: they wait for EXT_HOLD_FILE).
#   EXT_HOLD_FILE=<file> default train_log/PUB_SEEDEXT_HOLD: while this file exists, no new
#                        run of seeds 306 to 310 starts.
#   SEED_TIERS=<tiers>   ARMS=actfam only: tiers separated by ';', seeds of a tier by spaces
#                        (default "301 302 303;304 305", the two tiers of the family). Seeds
#                        301 to 305 and the second five seeds 306 to 310 are accepted, each
#                        once; a tier of second seeds lists no control.
#   CORES=<list>         cores of this lane (default 10-11).
#   THREADS=<n>          OpenMP/MKL/torch threads (default 2, the size of CORES).
#   LANE=<i>/<n>         run only the tasks whose index is i modulo n (default 0/1);
#                        the index is taken in the task list of the selected arms,
#                        before ONLY.
#   ONLY=<regex>         run only the tasks whose name matches.
#   PROBE_ONLY=1         verify the recipes and exit.
#   ARMS=eq              the equal-experience arm instead of the default arms
#                        (unset or empty: v1, ctl and the specialists).
#   ARMS=act             the active-rule arm (a word of RULE_ARMS) instead of the
#                        default arms.
#   ARMS=actfam          the family under the active rule (policy, control and
#                        specialists); ARMS=acteq its eq recipe under the active rule.
#   LIST_ONLY=1         print the tasks this invocation would train, one line
#                        "TASK <name>" each, after LANE, ONLY and the finished runs,
#                        and exit 0 before the probes; LIST_ONLY=all lists the
#                        finished runs too. Neither starts Python, writes a file or
#                        looks at the hold file.
#
# Usage: MASK=exact CORES=10-11 setsid nohup scripts/run_pubfamily_queue.sh \
#            > train_log/pubfamily_exact_lane0.log 2>&1 &
#   Equal-experience arm, one lane per seed on five disjoint core sets (here two cores
#   and two threads each; one core with THREADS=1 works the same way):
#   EQ_CORES=(10-11 12-13 14-15 16-17 18-19)
#   for i in 0 1 2 3 4; do c=${EQ_CORES[$i]}; ARMS=eq LANE=$i/5 CORES=$c THREADS=2 \
#       setsid nohup taskset -c $c scripts/run_pubfamily_queue.sh \
#       > train_log/pubfamily_eq_lane$i.log 2>&1 & done
#   Active-rule arm, one lane per seed on three disjoint core sets (two cores and two
#   threads each; one core with THREADS=1 works the same way):
#   ACT_CORES=(10-11 12-13 14-15)
#   for i in 0 1 2; do c=${ACT_CORES[$i]}; ARMS=act LANE=$i/3 CORES=$c THREADS=2 \
#       setsid nohup taskset -c $c scripts/run_pubfamily_queue.sh \
#       > train_log/pubfamily_act_lane$i.log 2>&1 & done
#   Family under the active rule, eight lanes on eight single cores with one thread each;
#   after them its eq recipe, one lane per seed:
#   FAM_CORES=(8 9 10 11 12 13 14 15)
#   for i in 0 1 2 3 4 5 6 7; do c=${FAM_CORES[$i]}; ARMS=actfam LANE=$i/8 CORES=$c THREADS=1 \
#       setsid nohup taskset -c $c scripts/run_pubfamily_queue.sh \
#       > train_log/pubfamily_actfam_lane$i.log 2>&1 & done
#   for i in 0 1 2 3 4; do c=${FAM_CORES[$i]}; ARMS=acteq LANE=$i/5 CORES=$c THREADS=1 \
#       setsid nohup taskset -c $c scripts/run_pubfamily_queue.sh \
#       > train_log/pubfamily_acteq_lane$i.log 2>&1 & done
#   Second five seeds of the family under the active rule, eight lanes on eight single cores
#   with one thread each (lanes 0 to 4 each take one policy run first):
#   EXT_CORES=(8 9 10 11 12 13 14 15)
#   for i in 0 1 2 3 4 5 6 7; do c=${EXT_CORES[$i]}; ARMS=actfam SEED_TIERS="306 307 308 309 310" \
#       LANE=$i/8 CORES=$c THREADS=1 setsid nohup taskset -c $c scripts/run_pubfamily_queue.sh \
#       > train_log/pubfamily_seedext_lane$i.log 2>&1 & done
set -u
cd "$(dirname "$0")/.."
PY=$HOME/miniconda3/envs/fjsp/bin/python
MASK=${MASK:-exact}
THREADS=${THREADS:-2}
CORES=${CORES:-10-11}
LANE=${LANE:-0/1}
ONLY=${ONLY:-.}
PROBE_ONLY=${PROBE_ONLY:-0}
ARMS=${ARMS:-}
LIST_ONLY=${LIST_ONLY:-0}
HOLD=train_log/PUB_HOLD
SEED_TIERS=${SEED_TIERS:-}
EXT_HOLD=${EXT_HOLD_FILE:-train_log/PUB_SEEDEXT_HOLD}
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS CUDA_VISIBLE_DEVICES=
unset PACT_FIRST_MASK PACT_MASK_TOL
case "$MASK" in
  exact)  PREFIX=pubx; MASK_OPTS="--first_mask exact --mask_ties exact" ;;
  legacy) PREFIX=publ; MASK_OPTS="--first_mask legacy --mask_ties legacy" ;;
  *) echo "MASK must be exact or legacy"; exit 1 ;;
esac
TYPES=${TYPES:-none}
case "$TYPES" in
  none)  TYPE_OPTS="--use_type_embedding False"; WANT_PARAMS=29602 ;;
  class) [ "$MASK" = exact ] || { echo "TYPES=class is defined for MASK=exact only"; exit 1; }
         PREFIX=pubc; TYPE_OPTS="--use_type_embedding True --pub_type_source class"; WANT_PARAMS=32714 ;;
  *) echo "TYPES must be none or class"; exit 1 ;;
esac
SIX="N,L,S,W,LS,LW"
EQ_UPDATES=6000      # arm_registry.PUB_EQ_UPDATES
# Rule arms: "<admissibility rule>|<word of the run names>" (arm_registry.PUB_RULE_WORD);
# ARMS=<word> selects one, and the rule is passed to the trainer as --mask_mode <rule>.
RULE_ARMS=("active|act")
RULE_SEEDS="301 302 303"     # arm_registry.PUB_ACT_SEEDS
RULE_UPDATES=2000            # the v1 recipe
RULE=""
RULE_WORDS=""
# Family arms under a rule: ARMS=<word>fam (policy, control and specialists) and
# ARMS=<word>eq (the eq recipe) for each word of RULE_ARMS; FAM is fam or eq for them and
# RWORD the rule word of the run names (arm_registry.PUB_RULE_STEM).
FAM=""
RWORD=""
FAM_WORDS=""
for e in "${RULE_ARMS[@]}"; do
  RULE_WORDS="$RULE_WORDS ${e#*|}"
  FAM_WORDS="$FAM_WORDS ${e#*|}fam ${e#*|}eq"
  [ "$ARMS" = "${e#*|}" ] && { RULE=${e%%|*}; RWORD=${e#*|}; }
  [ "$ARMS" = "${e#*|}fam" ] && { RULE=${e%%|*}; RWORD=${e#*|}; FAM=fam; }
  [ "$ARMS" = "${e#*|}eq" ] && { RULE=${e%%|*}; RWORD=${e#*|}; FAM=eq; }
done
case "$ARMS" in
  '') ;;
  eq) if [ "$MASK" != exact ] || [ "$TYPES" != none ]; then
        echo "ARMS=eq is defined for MASK=exact TYPES=none only"; exit 1
      fi ;;
  *) if [ -z "$RULE" ]; then
       echo "ARMS must be empty (the default arms), eq, a rule arm (${RULE_WORDS# }) or a family arm under a rule (${FAM_WORDS# }), got '$ARMS'"; exit 1
     fi
     if [ "$MASK" != exact ] || [ "$TYPES" != none ]; then
       echo "ARMS=$ARMS is defined for MASK=exact TYPES=none only"; exit 1
     fi ;;
esac
case "$LIST_ONLY" in 0|1|all) ;; *) echo "LIST_ONLY must be 0, 1 or all, got '$LIST_ONLY'"; exit 1 ;; esac
if ! [[ "$LANE" =~ ^(0|[1-9][0-9]*)/[1-9][0-9]*$ ]] || [ "${LANE%/*}" -ge "${LANE#*/}" ]; then
  echo "LANE must be <i>/<n> with 0 <= i < n, got '$LANE'"; exit 1
fi
# SEED_TIERS (ARMS=actfam only): seeds 301 to 305, and the second five seeds 306 to 310 of
# scripts/pub_seed_extension.py, each at most once; SECOND is 1 when a tier names a second seed.
if [ -n "$SEED_TIERS" ] && [ "$ARMS" != actfam ]; then
  echo "ABORT: SEED_TIERS is defined for ARMS=actfam only, got ARMS='$ARMS'"; exit 1
fi
FAM_TIERS=${SEED_TIERS:-301 302 303;304 305}
SECOND=0
if ! [[ "$FAM_TIERS" =~ ^[0-9\ \;]+$ ]]; then echo "ABORT: SEED_TIERS must hold seeds, spaces and ';', got '$FAM_TIERS'"; exit 1; fi
IFS=';' read -ra FAM_TIER_LIST <<< "$FAM_TIERS"
seen=" "
for tier in "${FAM_TIER_LIST[@]}"; do
  [ -n "${tier// /}" ] || { echo "ABORT: SEED_TIERS has an empty tier: '$FAM_TIERS'"; exit 1; }
  for S in $tier; do
    if ! [[ "$S" =~ ^[1-9][0-9]{0,5}$ ]]; then echo "ABORT: seed '$S' in SEED_TIERS is not a seed number"; exit 1
    elif [ "$S" -ge 301 ] && [ "$S" -le 305 ]; then :
    elif [ "$S" -ge 306 ] && [ "$S" -le 310 ]; then SECOND=1
    else echo "ABORT: seed $S in SEED_TIERS: seeds 301 to 305, or the second five seeds 306 to 310"; exit 1; fi
    case "$seen" in *" $S "*) echo "ABORT: seed $S appears twice in SEED_TIERS"; exit 1 ;; esac
    seen="$seen$S "
  done
done
VALI=data/PUBLIC/family/pub_SD2_vali_10x5+mix
COMMON="--data_source FAMILY --family_base sd2 --n_j 10 --n_m 5 $TYPE_OPTS \
  --validate_timestep 25 --use_film False --pub_vali_dir $VALI --device cpu $MASK_OPTS"
say() { echo "[pubfamily $MASK${ARMS:+ $ARMS} lane $LANE $(date '+%m-%d %H:%M')] $*"; }

# Arguments of train_family.py for one run, in the array TRAIN_ARGS; the rule of a rule
# arm or of a family arm under a rule adds --mask_mode <rule>, every other run trains under
# the default non-delay rule.
train_args() {  # $1 regimes  $2 updates  $3 seed  $4 name  $5 rule ('' for none)
  TRAIN_ARGS=($COMMON --family_regimes "$1" --max_updates "$2" --seed_train "$3" --model_suffix "$4")
  if [ -n "${5:-}" ]; then TRAIN_ARGS+=(--mask_mode "$5"); fi
  return 0
}

probe() {  # $1 regimes  $2 updates  $3 factory reference run (seed 303)
  local tsuf=""; [ "$TYPES" = none ] || tsuf="_${TYPES}"
  local suf="_pubprobe_${MASK}${tsuf}_${LANE%/*}"
  local plog="train_log/pubfamily_probe_${MASK}${tsuf}_${LANE%/*}.log"
  timeout 900 taskset -c $CORES $PY train_family.py $COMMON \
      --family_regimes "$1" --max_updates 1 --model_suffix "$suf" --seed_train 301 > "$plog" 2>&1
  local np
  np=$(sed -nE 's/.*\[model\] params total=([0-9]+).*/\1/p' "$plog" | head -1)
  if [ "$np" != "$WANT_PARAMS" ]; then say "ABORT: $np parameters for regimes $1, want $WANT_PARAMS (see $plog)"; return 1; fi
  REF="$3" WANT_UPD="$2" WANT_REG="$1" SUF="$suf" WANT_MASK="$MASK" VALI="$VALI" WANT_TYPES="$TYPES" \
      WANT_PARAMS="$WANT_PARAMS" $PY - << 'PYEOF' || return 1
import json, os
ALLOWED = {'model_suffix', 'seed_train', 'device', 'max_updates', 'family_base', 'n_m',
           'use_type_embedding', 'pub_vali_dir', 'first_mask', 'mask_ties'}
INERT_NEW = {'dr_shift': ('',), 'delta_mode': (None,), 'mask_mode': (None, 'nondelay'),
             'pair_extra': (None, 'none'), 'reward_mode': ('dense',),
             'pub_type_source': ('none',)}
if os.environ['WANT_TYPES'] == 'class':
    # the type tables of the class input differ from the factory's 5 and 9 rows
    ALLOWED |= {'pub_type_source', 'n_op_types', 'n_mch_types'}
ref_name = os.environ['REF']
ref = json.load(open(f"train_log/FAMILY/config_10x25+family+{ref_name}.json"))
probe = json.load(open(f"train_log/FAMILY/config_10x5+pubfamily+{os.environ['SUF']}.json"))
assert ref['family_regimes'] == os.environ['WANT_REG'], (ref['family_regimes'], os.environ['WANT_REG'])
assert probe['family_regimes'] == os.environ['WANT_REG'], probe['family_regimes']
assert int(ref['max_updates']) == int(os.environ['WANT_UPD']), ref['max_updates']
want = {'family_base': 'sd2', 'n_m': 5, 'use_type_embedding': False, 'device': 'cpu',
        'pub_vali_dir': os.environ['VALI'], 'first_mask': os.environ['WANT_MASK'],
        'mask_ties': os.environ['WANT_MASK']}
if os.environ['WANT_TYPES'] == 'class':
    want.update({'use_type_embedding': True, 'pub_type_source': 'class',
                 'n_op_types': 4, 'n_mch_types': 1})
for k, v in want.items():
    if probe.get(k) != v:
        raise SystemExit(f'ABORT: the probe records {k}={probe.get(k)!r}, want {v!r}')
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
print(f"[pubfamily] regimes {os.environ['WANT_REG']}: {os.environ['WANT_PARAMS']} parameters, configuration against "
      f"{ref_name} differs only in {sorted(diffs)}")
PYEOF
  rm -f "trained_network/FAMILY/10x5+pubfamily+${suf}.pth" \
        "train_log/FAMILY/config_10x5+pubfamily+${suf}.json" \
        train_log/FAMILY/*"${suf}"*.txt "$plog" 2>/dev/null
  return 0
}

# Equal-experience arm: a configuration file against the stored configuration of
# the finished run pubx-v1-s<seed>. Equal except for the name and max_updates,
# which must be the wanted value; keys the trainer added after the reference was
# written count as equal while they hold their inert default.
eq_check() {  # $1 configuration file  $2 seed  $3 wanted max_updates  $4 probe|run
  CFG="$1" SEED="$2" WANT_UPD="$3" KIND="$4" SIX="$SIX" EQ_UPDATES="$EQ_UPDATES" \
      WANT_PARAMS="$WANT_PARAMS" $PY - << 'PYEOF'
import json, os
ALLOWED = {'model_suffix', 'max_updates'}
INERT_NEW = {'dr_shift': ('',), 'delta_mode': (None,), 'mask_mode': (None, 'nondelay'),
             'pair_extra': (None, 'none'), 'reward_mode': ('dense',),
             'pub_type_source': ('none',)}
SEEDS = (301, 302, 303, 304, 305)
six, eq_upd = os.environ['SIX'], int(os.environ['EQ_UPDATES'])


def cfg_of(name, prefix='10x5+pubfamily+'):
    with open(f'train_log/FAMILY/config_{prefix}{name}.json') as f:
        return json.load(f)


def drift(ref, new):
    out = set()
    for k in set(ref) | set(new):
        rv, nv = ref.get(k, '<absent>'), new.get(k, '<absent>')
        if rv == nv or (k in INERT_NEW and rv == '<absent>' and nv in INERT_NEW[k]):
            continue
        out.add(k)
    return out


seed, want = int(os.environ['SEED']), int(os.environ['WANT_UPD'])
with open(os.environ['CFG']) as f:
    new = json.load(f)
ref_name = f'pubx-v1-s{seed}'
ref = cfg_of(ref_name)
for k, v in (('family_regimes', six), ('max_updates', 2000), ('seed_train', seed),
             ('first_mask', 'exact'), ('mask_ties', 'exact'), ('family_base', 'sd2'),
             ('use_type_embedding', False), ('device', 'cpu')):
    if ref.get(k) != v:
        raise SystemExit(f'ABORT: the reference {ref_name} records {k}={ref.get(k)!r}, want {v!r}')
if int(new.get('max_updates', -1)) != want:
    raise SystemExit(f"ABORT: {os.environ['CFG']} records max_updates={new.get('max_updates')!r}, want {want}")
bad = drift(ref, new) - ALLOWED
if bad:
    raise SystemExit(f'ABORT: configuration drift against {ref_name}: '
                     + str({k: (ref.get(k), new.get(k)) for k in sorted(bad)}))
if os.environ['KIND'] == 'probe':
    # the five references are one recipe, so the seed-301 probe stands for every seed
    base = cfg_of('pubx-v1-s301')
    for s in SEEDS[1:]:
        extra = drift(base, cfg_of(f'pubx-v1-s{s}')) - {'model_suffix', 'seed_train'}
        if extra:
            raise SystemExit(f'ABORT: pubx-v1-s{s} differs from pubx-v1-s301 in {sorted(extra)}')
    # the factory family's equal-experience runs: the same six combinations, the same budget
    for s in (301, 302, 303):
        f = cfg_of(f'jointblind-eq-s{s}', prefix='10x25+family+')
        if f.get('family_regimes') != six or int(f.get('max_updates', -1)) != eq_upd:
            raise SystemExit(f'ABORT: jointblind-eq-s{s} records {f.get("family_regimes")!r}, '
                             f'{f.get("max_updates")!r} updates; want {six!r}, {eq_upd}')
    print(f"[pubfamily] eq: {os.environ['WANT_PARAMS']} parameters, exact mask; configuration against {ref_name} "
          f'differs only in {sorted(drift(ref, new))}; pubx-v1-s301..305 are one recipe; '
          f'jointblind-eq-s301..303 record {eq_upd} updates on {six}')
else:
    print(f"[pubfamily] {new.get('model_suffix')}: configuration against {ref_name} differs only in "
          f'{sorted(drift(ref, new))} (max_updates {want})')
PYEOF
}

probe_eq() {  # one update of the equal-experience recipe at seed 301
  local suf="_pubprobe_eq_${LANE%/*}" plog="train_log/pubfamily_probe_eq_${LANE%/*}.log"
  local s np
  for s in 301 302 303 304 305; do
    if ! [ -f "trained_network/FAMILY/10x5+pubfamily+pubx-v1-s${s}.pth" ] \
        || ! grep -q "done in" "train_log/PUBFAMILY_pubx-v1-s${s}.log" 2>/dev/null; then
      say "ABORT: the reference run pubx-v1-s${s} is not finished"; return 1
    fi
  done
  timeout 900 taskset -c $CORES $PY train_family.py $COMMON \
      --family_regimes "$SIX" --max_updates 1 --model_suffix "$suf" --seed_train 301 > "$plog" 2>&1
  np=$(sed -nE 's/.*\[model\] params total=([0-9]+).*/\1/p' "$plog" | head -1)
  if [ "$np" != "$WANT_PARAMS" ]; then say "ABORT: $np parameters for the eq recipe, want $WANT_PARAMS (see $plog)"; return 1; fi
  if ! grep -q "^\[train_family\] MASK: first_mask=exact mask_ties=exact" "$plog"; then
    say "ABORT: the probe of the eq recipe did not train under the exact mask (see $plog)"; return 1
  fi
  eq_check "train_log/FAMILY/config_10x5+pubfamily+${suf}.json" 301 1 probe || return 1
  # the probe's own files only, so a probe never removes the files of another lane's probe
  rm -f "trained_network/FAMILY/10x5+pubfamily+${suf}.pth" \
        "train_log/FAMILY/config_10x5+pubfamily+${suf}.json" \
        "train_log/FAMILY/valiquality_10x5+pubfamily+${suf}.txt" \
        "train_log/FAMILY/reward_10x5+pubfamily+${suf}.txt" "$plog" 2>/dev/null
  return 0
}

# Rule arm: a configuration file against the stored configuration of the finished run
# pubx-v1-s<seed>. Equal except for the name and mask_mode, which must be the arm's rule
# (the reference records the non-delay rule); the probe also differs in its one update.
# Keys the trainer added after the reference was written count as equal while they hold
# their inert default.
rule_check() {  # $1 configuration file  $2 seed  $3 wanted max_updates  $4 probe|run
  CFG="$1" SEED="$2" WANT_UPD="$3" KIND="$4" SIX="$SIX" RULE="$RULE" WORD="$ARMS" \
      RULE_SEEDS="$RULE_SEEDS" RULE_UPDATES="$RULE_UPDATES" WANT_PARAMS="$WANT_PARAMS" $PY - << 'PYEOF'
import json, os
INERT_NEW = {'dr_shift': ('',), 'delta_mode': (None,), 'mask_mode': (None, 'nondelay'),
             'pair_extra': (None, 'none'), 'reward_mode': ('dense',),
             'pub_type_source': ('none',)}
six, rule, word = os.environ['SIX'], os.environ['RULE'], os.environ['WORD']
seeds = [int(s) for s in os.environ['RULE_SEEDS'].split()]
recipe_upd = int(os.environ['RULE_UPDATES'])
kind = os.environ['KIND']
ALLOWED = {'model_suffix', 'mask_mode'} | ({'max_updates'} if kind == 'probe' else set())


def cfg_of(name):
    with open(f'train_log/FAMILY/config_10x5+pubfamily+{name}.json') as f:
        return json.load(f)


def drift(ref, new):
    out = set()
    for k in set(ref) | set(new):
        rv, nv = ref.get(k, '<absent>'), new.get(k, '<absent>')
        if rv == nv or (k in INERT_NEW and rv == '<absent>' and nv in INERT_NEW[k]):
            continue
        out.add(k)
    return out


seed, want = int(os.environ['SEED']), int(os.environ['WANT_UPD'])
with open(os.environ['CFG']) as f:
    new = json.load(f)
ref_name = f'pubx-v1-s{seed}'
ref = cfg_of(ref_name)
for k, v in (('family_regimes', six), ('max_updates', recipe_upd), ('seed_train', seed),
             ('first_mask', 'exact'), ('mask_ties', 'exact'), ('family_base', 'sd2'),
             ('use_type_embedding', False), ('device', 'cpu')):
    if ref.get(k) != v:
        raise SystemExit(f'ABORT: the reference {ref_name} records {k}={ref.get(k)!r}, want {v!r}')
if ref.get('mask_mode', 'nondelay') != 'nondelay':
    raise SystemExit(f"ABORT: the reference {ref_name} records mask_mode={ref.get('mask_mode')!r}, "
                     f"want the non-delay rule")
if not rule or rule == 'nondelay':
    raise SystemExit(f'ABORT: the rule arm {word!r} names the rule {rule!r}, want a rule other than nondelay')
if new.get('mask_mode') != rule:
    raise SystemExit(f"ABORT: {os.environ['CFG']} records mask_mode={new.get('mask_mode')!r}, want {rule!r}")
if int(new.get('max_updates', -1)) != want:
    raise SystemExit(f"ABORT: {os.environ['CFG']} records max_updates={new.get('max_updates')!r}, want {want}")
bad = drift(ref, new) - ALLOWED
if bad:
    raise SystemExit(f'ABORT: configuration drift against {ref_name}: '
                     + str({k: (ref.get(k), new.get(k)) for k in sorted(bad)}))
if kind == 'probe':
    # the references of the arm's seeds are one recipe, so the seed-301 probe stands for every seed
    base = cfg_of(f'pubx-v1-s{seeds[0]}')
    for s in seeds[1:]:
        extra = drift(base, cfg_of(f'pubx-v1-s{s}')) - {'model_suffix', 'seed_train'}
        if extra:
            raise SystemExit(f'ABORT: pubx-v1-s{s} differs from pubx-v1-s{seeds[0]} in {sorted(extra)}')
    print(f"[pubfamily] {word}: {os.environ['WANT_PARAMS']} parameters, exact mask, mask_mode {rule}; "
          f'configuration against {ref_name} differs only in {sorted(drift(ref, new))}; '
          f'pubx-v1-s{seeds[0]}..{seeds[-1]} are one recipe under the non-delay rule')
else:
    print(f"[pubfamily] {new.get('model_suffix')}: configuration against {ref_name} differs only in "
          f'{sorted(drift(ref, new))} (mask_mode {rule})')
PYEOF
}

# One update of a probe with the arguments in TRAIN_ARGS (train_args), on the lane's cores.
probe_train() {  # $1 log of the probe
  timeout 900 taskset -c $CORES $PY train_family.py "${TRAIN_ARGS[@]}" > "$1" 2>&1
}

probe_rule() {  # one update of the rule arm's recipe at seed 301
  local suf="_pubprobe_${ARMS}_${LANE%/*}" plog="train_log/pubfamily_probe_${ARMS}_${LANE%/*}.log"
  local s np
  for s in $RULE_SEEDS; do
    if ! [ -f "trained_network/FAMILY/10x5+pubfamily+pubx-v1-s${s}.pth" ] \
        || ! grep -q "done in" "train_log/PUBFAMILY_pubx-v1-s${s}.log" 2>/dev/null; then
      say "ABORT: the reference run pubx-v1-s${s} is not finished"; return 1
    fi
  done
  train_args "$SIX" 1 301 "$suf" "$RULE"
  probe_train "$plog"
  np=$(sed -nE 's/.*\[model\] params total=([0-9]+).*/\1/p' "$plog" | head -1)
  if [ "$np" != "$WANT_PARAMS" ]; then say "ABORT: $np parameters for the $ARMS recipe, want $WANT_PARAMS (see $plog)"; return 1; fi
  if ! grep -q "^\[train_family\] MASK: first_mask=exact mask_ties=exact" "$plog"; then
    say "ABORT: the probe of the $ARMS recipe did not train under the exact mask (see $plog)"; return 1
  fi
  if ! grep -q "^\[train_family\] OPTIONS: mask_mode=${RULE} pair_extra=none" "$plog"; then
    say "ABORT: the probe of the $ARMS recipe did not train under the $RULE rule (see $plog)"; return 1
  fi
  rule_check "train_log/FAMILY/config_10x5+pubfamily+${suf}.json" 301 1 probe || return 1
  # the probe's own files only, so a probe never removes the files of another lane's probe
  rm -f "trained_network/FAMILY/10x5+pubfamily+${suf}.pth" \
        "train_log/FAMILY/config_10x5+pubfamily+${suf}.json" \
        "train_log/FAMILY/valiquality_10x5+pubfamily+${suf}.txt" \
        "train_log/FAMILY/reward_10x5+pubfamily+${suf}.txt" "$plog" 2>/dev/null
  return 0
}

# Family arm under a rule: a configuration file against the stored configuration of the
# run's non-delay counterpart (pubx-v1, pubx-ctl, pubx-solo-<R> or pubx-eq of the same seed;
# seed 301 for the probe). Equal except for the name and mask_mode, which must be the
# arm's rule (the counterpart records the non-delay rule); the probe also differs in its one
# update. Keys the trainer added after the counterpart was written count as equal while
# they hold their inert default. The probe also checks that the stored counterparts of the
# recipe's seeds in the arm are one recipe, so the seed-301 probe stands for every seed.
fam_check() {  # $1 configuration file  $2 its run name  $3 counterpart  $4 regimes
               # $5 updates of the recipe  $6 wanted max_updates  $7 probe|run
               # $8 seeds of the recipe in the arm (probe)  $9 parameters of the probe
  CFG="$1" NAME="$2" REF="$3" WANT_REG="$4" RECIPE_UPD="$5" WANT_UPD="$6" KIND="$7" \
      SEEDS="${8:-}" NP="${9:-}" RULE="$RULE" WORD="$RWORD" $PY - << 'PYEOF'
import json, os
INERT_NEW = {'dr_shift': ('',), 'delta_mode': (None,), 'mask_mode': (None, 'nondelay'),
             'pair_extra': (None, 'none'), 'reward_mode': ('dense',),
             'pub_type_source': ('none',)}
cfg_p, name, ref_name = os.environ['CFG'], os.environ['NAME'], os.environ['REF']
regs, recipe_upd = os.environ['WANT_REG'], int(os.environ['RECIPE_UPD'])
want, kind = int(os.environ['WANT_UPD']), os.environ['KIND']
rule, word = os.environ['RULE'], os.environ['WORD']
ALLOWED = {'model_suffix', 'mask_mode'} | ({'max_updates'} if kind == 'probe' else set())


def cfg_of(run):
    with open(f'train_log/FAMILY/config_10x5+pubfamily+{run}.json') as f:
        return json.load(f)


def drift(ref, new):
    out = set()
    for k in set(ref) | set(new):
        rv, nv = ref.get(k, '<absent>'), new.get(k, '<absent>')
        if rv == nv or (k in INERT_NEW and rv == '<absent>' and nv in INERT_NEW[k]):
            continue
        out.add(k)
    return out


stem, _, seed = ref_name.rpartition('-s')
seed = int(seed)
ref = cfg_of(ref_name)
for k, v in (('model_suffix', ref_name), ('family_regimes', regs), ('max_updates', recipe_upd),
             ('seed_train', seed), ('first_mask', 'exact'), ('mask_ties', 'exact'),
             ('family_base', 'sd2'), ('use_type_embedding', False), ('device', 'cpu')):
    if ref.get(k) != v:
        raise SystemExit(f'ABORT: the counterpart {ref_name} records {k}={ref.get(k)!r}, want {v!r}')
if ref.get('mask_mode', 'nondelay') != 'nondelay':
    raise SystemExit(f"ABORT: the counterpart {ref_name} records mask_mode={ref.get('mask_mode')!r}, "
                     f"want the non-delay rule")
if not rule or rule == 'nondelay':
    raise SystemExit(f'ABORT: the rule word {word!r} names the rule {rule!r}, want a rule other than nondelay')
with open(cfg_p) as f:
    new = json.load(f)
if new.get('model_suffix') != name:
    raise SystemExit(f"ABORT: {cfg_p} records model_suffix={new.get('model_suffix')!r}, want {name!r}")
if new.get('mask_mode') != rule:
    raise SystemExit(f"ABORT: {cfg_p} records mask_mode={new.get('mask_mode')!r}, want {rule!r}")
if int(new.get('max_updates', -1)) != want:
    raise SystemExit(f"ABORT: {cfg_p} records max_updates={new.get('max_updates')!r}, want {want}")
bad = drift(ref, new) - ALLOWED
if bad:
    raise SystemExit(f'ABORT: configuration drift against {ref_name}: '
                     + str({k: (ref.get(k), new.get(k)) for k in sorted(bad)}))
if kind == 'probe':
    seeds = [int(s) for s in os.environ['SEEDS'].split()]
    for s in seeds:
        if s != seed:
            extra = drift(ref, cfg_of(f'{stem}-s{s}')) - {'model_suffix', 'seed_train'}
            if extra:
                raise SystemExit(f'ABORT: {stem}-s{s} differs from {ref_name} in {sorted(extra)}')
    print(f"[pubfamily] {word} {stem}: {os.environ['NP']} parameters as {ref_name}, exact mask, "
          f'mask_mode {rule}; configuration against {ref_name} differs only in '
          f'{sorted(drift(ref, new))}; {stem}-s{"/".join(map(str, seeds))} are one recipe under '
          f'the non-delay rule')
else:
    print(f'[pubfamily] {name}: configuration against {ref_name} differs only in '
          f'{sorted(drift(ref, new))} (mask_mode {rule})')
PYEOF
}

params_of() {  # training log -> the parameter count the trainer printed
  sed -nE 's/.*\[model\] params total=([0-9]+).*/\1/p' "$1" 2>/dev/null | head -1
}

finished() {  # name -> true when its checkpoint exists and its log has the completion line
  [ -f "trained_network/FAMILY/10x5+pubfamily+$1.pth" ] && grep -q "done in" "train_log/PUBFAMILY_$1.log" 2>/dev/null
}

# Family arm under a rule: one update at seed 301 of the recipe whose non-delay runs are
# <stem>-s<seed>, checked against the counterpart <stem>-s301.
probe_fam() {  # $1 counterpart stem  $2 regimes  $3 updates of the recipe  $4 seeds of the recipe in the arm
  local stem=$1 regs=$2 upd=$3 seeds=$4
  local key=${stem#"${PREFIX}"-}
  local suf="_pubprobe_${ARMS}_${key}_${LANE%/*}" plog="train_log/pubfamily_probe_${ARMS}_${key}_${LANE%/*}.log"
  local s np ref_np
  for s in 301 $seeds; do
    if ! finished "${stem}-s${s}"; then say "ABORT: the counterpart ${stem}-s${s} is not finished"; return 1; fi
  done
  ref_np=$(params_of "train_log/PUBFAMILY_${stem}-s301.log")
  train_args "$regs" 1 301 "$suf" "$RULE"
  probe_train "$plog"
  np=$(params_of "$plog")
  if [ "$np" != "$WANT_PARAMS" ] || [ "$np" != "$ref_np" ]; then
    say "ABORT: $np parameters for the ${key} recipe under the $RULE rule, ${ref_np:-none} in ${stem}-s301, want $WANT_PARAMS (see $plog)"; return 1
  fi
  if ! grep -q "^\[train_family\] MASK: first_mask=exact mask_ties=exact" "$plog"; then
    say "ABORT: the probe of the ${key} recipe did not train under the exact mask (see $plog)"; return 1
  fi
  if ! grep -q "^\[train_family\] OPTIONS: mask_mode=${RULE} pair_extra=none" "$plog"; then
    say "ABORT: the probe of the ${key} recipe did not train under the $RULE rule (see $plog)"; return 1
  fi
  fam_check "train_log/FAMILY/config_10x5+pubfamily+${suf}.json" "$suf" "${stem}-s301" "$regs" "$upd" 1 \
      probe "$seeds" "$np" || return 1
  # the probe's own files only, so a probe never removes the files of another lane's probe
  rm -f "trained_network/FAMILY/10x5+pubfamily+${suf}.pth" \
        "train_log/FAMILY/config_10x5+pubfamily+${suf}.json" \
        "train_log/FAMILY/valiquality_10x5+pubfamily+${suf}.txt" \
        "train_log/FAMILY/reward_10x5+pubfamily+${suf}.txt" "$plog" 2>/dev/null
  return 0
}

# Second five seeds of the family under the rule (seeds 306 to 310, ARMS=actfam): one update at
# seed 301 of the recipe whose runs of the first five seeds are <stem>-s301 to <stem>-s305 (the
# runs under the rule, not their non-delay counterparts), checked against <stem>-s301 by
# scripts/pub_seed_extension.py probecheck.
probe_ext() {  # $1 stem of the runs under the rule  $2 regimes  $3 updates of the recipe
  local stem=$1 regs=$2 upd=$3
  local key=${stem#"${PREFIX}"-}
  local suf="_pubprobe_${ARMS}_${key}_${LANE%/*}" plog="train_log/pubfamily_probe_${ARMS}_${key}_${LANE%/*}.log"
  local s np ref_np
  for s in 301 302 303 304 305; do
    if ! finished "${stem}-s${s}"; then say "ABORT: the run ${stem}-s${s} of the first five seeds is not finished"; return 1; fi
  done
  ref_np=$(params_of "train_log/PUBFAMILY_${stem}-s301.log")
  train_args "$regs" 1 301 "$suf" "$RULE"
  probe_train "$plog"
  np=$(params_of "$plog")
  if [ "$np" != "$WANT_PARAMS" ] || [ "$np" != "$ref_np" ]; then
    say "ABORT: $np parameters for the ${key} recipe of the second five seeds, ${ref_np:-none} in ${stem}-s301, want $WANT_PARAMS (see $plog)"; return 1
  fi
  if ! grep -q "^\[train_family\] MASK: first_mask=exact mask_ties=exact" "$plog"; then
    say "ABORT: the probe of the ${key} recipe did not train under the exact mask (see $plog)"; return 1
  fi
  if ! grep -q "^\[train_family\] OPTIONS: mask_mode=${RULE} pair_extra=none" "$plog"; then
    say "ABORT: the probe of the ${key} recipe did not train under the $RULE rule (see $plog)"; return 1
  fi
  $PY -B scripts/pub_seed_extension.py probecheck "train_log/FAMILY/config_10x5+pubfamily+${suf}.json" \
      "$stem" "$regs" "$upd" || return 1
  # the probe's own files only, so a probe never removes the files of another lane's probe
  rm -f "trained_network/FAMILY/10x5+pubfamily+${suf}.pth" \
        "train_log/FAMILY/config_10x5+pubfamily+${suf}.json" \
        "train_log/FAMILY/valiquality_10x5+pubfamily+${suf}.txt" \
        "train_log/FAMILY/reward_10x5+pubfamily+${suf}.txt" "$plog" 2>/dev/null
  return 0
}

# Run name of a recipe of the family under the rule word RWORD (arm_registry.pub_rule_model).
rule_name() {  # $1 recipe: v1 | ctl | solo-<R> | eq   $2 seed
  case "$1" in
    v1)     echo "${PREFIX}-${RWORD}-s$2" ;;
    solo-*) echo "${PREFIX}-solo${RWORD}-${1#solo-}-s$2" ;;
    *)      echo "${PREFIX}-$1${RWORD}-s$2" ;;
  esac
}

# Runs of the rule arm that a family arm lists: finished, never trained by the family arm.
PROTECT=" "
protected() { [[ "$PROTECT" == *" $1 "* ]]; }

# name | regimes | updates | seed [| rule of a rule arm [| non-delay counterpart of a family arm]]
# (a run of the second five seeds names, in place of the counterpart, the run of the same recipe
# under the rule at seed 301)
TASKS=()
if [ "$ARMS" = eq ]; then
  for S in 301 302 303 304 305; do TASKS+=("${PREFIX}-eq-s${S}|${SIX}|${EQ_UPDATES}|${S}"); done
elif [ "$FAM" = fam ]; then
  for tier in "${FAM_TIER_LIST[@]}"; do
    for S in $tier; do
      if [ "$S" -ge 306 ]; then TASKS+=("$(rule_name v1 "$S")|${SIX}|2000|${S}|${RULE}|$(rule_name v1 301)")
      else TASKS+=("$(rule_name v1 "$S")|${SIX}|2000|${S}|${RULE}|${PREFIX}-v1-s${S}"); fi
    done
    for S in $tier; do
      [ "$S" -le 305 ] && TASKS+=("$(rule_name ctl "$S")|N,L,S,W|2000|${S}|${RULE}|${PREFIX}-ctl-s${S}")
    done
    for S in $tier; do
      for R in N L S W LS LW SW LSW; do
        if [ "$S" -ge 306 ]; then
          TASKS+=("$(rule_name "solo-$R" "$S")|${R}|1000|${S}|${RULE}|$(rule_name "solo-$R" 301)")
        else
          TASKS+=("$(rule_name "solo-$R" "$S")|${R}|1000|${S}|${RULE}|${PREFIX}-solo-${R}-s${S}")
        fi
      done
    done
  done
  for S in $RULE_SEEDS; do
    n=$(rule_name v1 "$S")
    if ! finished "$n"; then
      echo "ARMS=$ARMS: $n is a run of ARMS=$RWORD and is not finished; this arm never trains it"; exit 1
    fi
    PROTECT="${PROTECT}${n} "
  done
elif [ "$FAM" = eq ]; then
  for S in 301 302 303 304 305; do
    if finished "${PREFIX}-eq-s${S}"; then
      TASKS+=("$(rule_name eq "$S")|${SIX}|${EQ_UPDATES}|${S}|${RULE}|${PREFIX}-eq-s${S}")
    fi
  done
  if [ ${#TASKS[@]} -eq 0 ]; then echo "ARMS=$ARMS: no finished run ${PREFIX}-eq-s<seed>"; exit 1; fi
elif [ -n "$RULE" ]; then
  for S in $RULE_SEEDS; do TASKS+=("${PREFIX}-${ARMS}-s${S}|${SIX}|${RULE_UPDATES}|${S}|${RULE}"); done
else
  for tier in "301 302 303" "304 305"; do
    for S in $tier; do TASKS+=("${PREFIX}-v1-s${S}|N,L,S,W,LS,LW|2000|${S}"); done
    for S in $tier; do TASKS+=("${PREFIX}-ctl-s${S}|N,L,S,W|2000|${S}"); done
    for S in $tier; do
      for R in N L S W LS LW SW LSW; do TASKS+=("${PREFIX}-solo-${R}-s${S}|${R}|1000|${S}"); done
    done
  done
fi

# The tasks of this lane: index i modulo n over the task list, then ONLY.
lane_i=${LANE%/*}; lane_n=${LANE#*/}
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
    if [ "$LIST_ONLY" = "1" ] && finished "$name"; then
      say "SKIP ${name} (finished$(protected "$name" && echo "; a run of ARMS=${RWORD}, never trained by ARMS=${ARMS}"))" >&2
      continue
    fi
    echo "TASK ${name}"
  done
  exit 0
fi

# Family arm under a rule: each recipe among this lane's tasks (the runs of the rule arm
# left out) is probed once, with the seeds of the recipe in the arm's whole task list.
probe_fam_lane() {
  local t u name regs upd seed rule cpart stem s c seeds seen=" "
  for t in "${MINE[@]}"; do
    IFS='|' read -r name regs upd seed rule cpart <<< "$t"
    protected "$name" && continue
    stem=${cpart%-s*}
    [[ "$seen" == *" $stem "* ]] && continue
    seen="${seen}${stem} "
    seeds=""
    for u in "${TASKS[@]}"; do
      IFS='|' read -r _ _ _ s _ c <<< "$u"
      [ "${c%-s*}" = "$stem" ] && seeds="$seeds $s"
    done
    if [ "$seed" -ge 306 ]; then
      probe_ext "$stem" "$regs" "$upd" || return 1
    else
      probe_fam "$stem" "$regs" "$upd" "${seeds# }" || return 1
    fi
  done
  return 0
}

if [ "$SECOND" = 1 ]; then
  say "SEED_TIERS names the second five seeds 306 to 310 of the policy and the specialists under the ${RULE} rule: an addition after the five-seed result was known (scripts/pub_seed_extension.py); their runs wait for $EXT_HOLD"
fi
say "verifying the recipes on cores $CORES ..."
if [ "$ARMS" = eq ]; then
  probe_eq || exit 1
elif [ -n "$FAM" ]; then
  probe_fam_lane || exit 1
elif [ -n "$RULE" ]; then
  probe_rule || exit 1
else
  probe "N,L,S,W,LS,LW" 2000 "jointblind-v1-s303" || exit 1
  probe "N,L,S,W" 2000 "jointblind-nlsw-s303" || exit 1
  for R in N L S W LS LW SW LSW; do probe "$R" 1000 "spec-${R}-s303" || exit 1; done
fi
say "all recipes verified"
if [ "$PROBE_ONLY" = "1" ]; then say "PROBE_ONLY=1: not starting the queue"; exit 0; fi

for t in "${MINE[@]}"; do
  IFS='|' read -r name regs upd seed rule cpart <<< "$t"
  log="train_log/PUBFAMILY_${name}.log"
  pth="trained_network/FAMILY/10x5+pubfamily+${name}.pth"
  cfg="train_log/FAMILY/config_10x5+pubfamily+${name}.json"
  if protected "$name"; then say "SKIP ${name} (a run of ARMS=${RWORD}, never trained by ARMS=${ARMS})"; continue; fi
  if finished "$name"; then say "SKIP ${name} (finished)"; continue; fi
  if [ "$seed" -ge 306 ]; then
    while [ -f "$EXT_HOLD" ]; do sleep 60; done
    say "TRAIN ${name} (combinations ${regs}, ${upd} updates, seed ${seed}, rule ${rule}, the recipe of ${cpart}; second five seeds)"
  else
    while [ -f "$HOLD" ]; do sleep 60; done
    say "TRAIN ${name} (regimes ${regs}, ${upd} updates, seed ${seed}${rule:+, rule ${rule}}${cpart:+, counterpart ${cpart}})"
  fi
  train_args "$regs" "$upd" "$seed" "$name" "$rule"
  taskset -c $CORES $PY train_family.py "${TRAIN_ARGS[@]}" > "$log" 2>&1
  rc=$?
  np=$(sed -nE 's/.*\[model\] params total=([0-9]+).*/\1/p' "$log" | head -1)
  if [ $rc -ne 0 ] || ! grep -q "done in" "$log"; then
    say "FAILED ${name} (exit $rc); removing its partial files"
    rm -f "$pth" "$cfg"
    mv "$log" "${log}.failed" 2>/dev/null
    continue
  fi
  if [ "$np" != "$WANT_PARAMS" ]; then say "ABORT ${name}: $np parameters"; exit 1; fi
  if [ "$TYPES" = class ] && ! grep -q '"pub_type_source": "class"' "$cfg"; then
    say "ABORT ${name}: its configuration does not record pub_type_source class"; exit 1
  fi
  if ! grep -q "\"first_mask\": \"${MASK}\"" "$cfg" || ! grep -q "\"mask_ties\": \"${MASK}\"" "$cfg"; then
    say "ABORT ${name}: its configuration does not record the ${MASK} mask"; exit 1
  fi
  if [ "$ARMS" = eq ] && ! eq_check "$cfg" "$seed" "$upd" run; then
    say "ABORT ${name}: its configuration is not the pubx-v1-s${seed} recipe with ${upd} updates"; exit 1
  fi
  if [ -z "$FAM" ] && [ -n "$rule" ] && ! rule_check "$cfg" "$seed" "$upd" run; then
    say "ABORT ${name}: its configuration is not the pubx-v1-s${seed} recipe under the ${rule} rule"; exit 1
  fi
  if [ -n "$FAM" ]; then
    if [ "$np" != "$(params_of "train_log/PUBFAMILY_${cpart}.log")" ]; then
      say "ABORT ${name}: $np parameters, ${cpart} has $(params_of "train_log/PUBFAMILY_${cpart}.log")"; exit 1
    fi
    if ! grep -q "^\[train_family\] OPTIONS: mask_mode=${rule} pair_extra=none" "$log"; then
      say "ABORT ${name}: its log does not state the ${rule} rule"; exit 1
    fi
    if [ "$seed" -ge 306 ]; then
      # a second seed: the configuration of ${cpart} except the name and the seed
      if ! $PY -B scripts/pub_seed_extension.py trained "$name"; then
        say "ABORT ${name}: not trained as ${cpart} with its own name and seed (scripts/pub_seed_extension.py trained)"; exit 1
      fi
    elif ! fam_check "$cfg" "$name" "$cpart" "$regs" "$upd" "$upd" run; then
      say "ABORT ${name}: its configuration is not the ${cpart} recipe under the ${rule} rule"; exit 1
    fi
  fi
  say "trained ${name}: $(grep 'done in' "$log" | tail -1)"
done
say "queue finished"
