r"""P94: the timed campaign against the policy, at budgets from 1 s to 600 s.

The campaign of scripts/run_timed_campaign.sh (planner scripts/timed_campaign_plan.py)
runs four arms with a time budget: the memetic algorithm (one thread) and the CP-SAT
configuration the pilot selected (8 workers), each started from the schedule of the best
dispatching rule (memetic_rule, cpsat_rule) and from the policy's schedule
(memetic_policy, cpsat_policy). Each cell is one run whose incumbent is recorded over
time, so every shorter budget is read from the same run. scripts/p97_policy_anytime.py
measures the two start schedules on the campaign's cores: its policy arm rolls out the
policy's greedy schedule and then sampled schedules within a budget, and its rule arm
computes the four dispatching-rule schedules the rule-started arms start from. This
script turns the records into the numbers of the paper's third question (when is a
search method or a rule the better tool): macros, a JSON file, a Markdown summary and
the data of the time-budget figure. It changes no existing number and writes new files
only.

INPUTS
  Records, through the planner's own code (imported, not copied):
    timed_campaign_plan.Campaign       folders and expected fields of every cell
    timed_campaign_plan.cell_states    valid / pending / invalid / foreign / held / missing;
                                       timing_validity.classify_single_thread for the
                                       memetic algorithm, timing_validity.classify with the
                                       frozen bars for CP-SAT (a pending CP-SAT record is
                                       not evidence)
    timed_campaign_plan.record_rows    the incumbent at time t (memetic records through
                                       baselines/memetic/records.incumbent_at); the
                                       schedule handed to the search counts from time 0
  The campaign is described as designed, one --campaign per launch,
  "<sizes> = <arm tokens>" in the planner's token syntax name[@budget[:instances]]; the
  planner's options (--budget, --slots, --rule-cache, --policy-cache, --result-root,
  --workers, --memetic-seed, --cpsat-seed, --bars) pass through unchanged. The instance
  range of a token is the design of its arm: the arm is read on those instances, and the
  JSON (results.<size>.design), the Markdown header and the macros (...InstN) state the
  instances per size and arm. A token without a range plans the size's evidence instances
  (003-029 at the production batches, 000-029 at 10x25), as the planner does.
  Records of p97 (read on the request clock only, through p97's own code: folders,
  p97.classify and p97.classify_rules; under --result-root when it is given):
    policy arm  or_solution/FAMILY/<size>_polsample<B>s_b<batch>_s<seed>/<R>/instance_<i>.json,
                seed the selected training run. It joins the campaign of its sizes as one
                token of this script's own, policy_sample@<B>[:instances] (the planner
                never sees it), with --policy-batch, --policy-workers and
                --policy-base-seed (p97's --batch, --workers, --base_seed, with p97's
                defaults 1, 8 and 50). A record counts when p97.classify calls it valid
                and it carries the plan's fields: arm, cell, budget_s = B, budget clock,
                workers, batch, base seed, the selected run's model and training seed,
                the environment.
    rule arm    or_solution/FAMILY/<size>_rulestart/<R>/instance_<i>.json, no token (one
                folder per size). A record counts when p97.classify_rules calls it valid
                and it carries arm rule_start, the cell and p97's four workers.
  Arrays of the exact environment (scripts/arm_registry.py; PACT_ENV is exact when unset
  and refused otherwise, PACT_SEEDS is 301-305 when unset):
    policy  arm_registry.result_path('released', None, seed, <size>, R), i.e.
            test_results/FAMILY/<size>_xmask/Result_greedy+10x25+family+xmask-v1-s<seed>_<R>.npy
            (scripts/eval_newseeds_plan.py: group n30 at 50x25n30 and 80x25n30, which also
            exports or_solution/FAMILY/pactwarm_cache_xmask_s<seed>_<size>, the schedules the
            policy-started arms start from; group principal at 10x25)
    rules   arm_registry.rule_dir(<size>)/Result_<FIFO|MOR|SPT|MWKR>_<R>.npy (group rules)
    Each array is [n, 2], column 0 the makespan; row i is the i-th descriptor of
    data/FAMILY/<size>/desc/<R>/ in sorted order. The best rule is the per-instance
    minimum of the four, the schedule the rule-started arms start from.
  The training run: arm_registry.select_seed('released'), the rule p88 uses (lowest
  validation score of the kept checkpoint, ties to the lower seed). The CP-SAT
  configuration: timed_campaign_plan.select_pilot, the pilot's selection rule; a given
  --cpsat-arm must equal it.

CLOCK (--origin, written into every output)
  request (default)  time since the request for a schedule, with a cold start for every
                   method. Two offsets per cell (size, combination, instance): o_rule, the
                   rule record's t_all (all four rule schedules exist; from the start of
                   the rule cell's process, imports and the instance load included), and
                   o_pol, the t of the greedy schedule in the selected run's policy record
                   (from the start of the policy cell's process, imports, checkpoint load
                   and environment construction included). At time x
                     the dispatching rules have the best of the four from o_rule on;
                     the policy has its greedy schedule from o_pol on;
                     the policy, sampled schedules, has the best schedule so far of its
                     record at its own t (the greedy schedule included); after the
                     record's budget its last value is held, and since further sampling
                     could only shorten it, a held value favours the other method; every
                     output that uses a held value says so;
                     a search started from the rules has its incumbent at run time
                     x - o_rule on the start clock below (model build or loading
                     included), a search started from the policy at run time x - o_pol;
                     neither has a schedule before its offset. The memetic arm started
                     from the policy also takes the rule schedule as a seed, so its
                     offset is the later of o_pol and o_rule.
                   A time is a property of the code and the machine, not of the training
                   run: row All takes the offsets of the same rule records and of the
                   selected run's policy records for every training run.
  start            wall clock from the start of the run. Memetic algorithm: from the start
                   of the process that solves the instance (descriptor and instance
                   loading, the start schedules and their validation, the numba warm-up
                   come first; record field search_start_wall_s). CP-SAT model arms: from
                   the start of the arm, model building included (record field build_s);
                   the instance load and the hint validation before it are not timed by the
                   runner. CP-SAT neighbourhood search (lns): from its start, every
                   sub-model build inside the budget.
  search           wall clock from the start of the search: the memetic algorithm's timed
                   window, the CP-SAT solver call (after the model build), the
                   neighbourhood search's start.

QUANTITIES (per size, per combination, pooled over the eight and pooled per group below)
  Budgets: start and search clocks --budgets (default 1 10 60 300 600) at every size;
  request clock --budgets at the production sets (default 60 300 600) and
  --training-budgets at the training size 10x25 (default 10 60 300 600).
  Every difference is 100 (first - second) / second per instance, the first-named method
  minus the second in percent of the second, so a negative value means the first-named
  method has the shorter schedule. On the request clock a number at budget b needs a
  schedule of both compared methods on every cell at b; where one of them has none on
  some cell, the number is not written: the JSON and the Markdown say "no schedule yet"
  with the count of such cells, and the macro is absent.
  (i)   policy greedy schedule vs each rule-started arm at the budget (policy minus arm).
        Row Sel: the selected training run, instance-level cluster bootstrap (p88's
        procedure for one checkpoint). Row All: every training run, crossed hierarchical
        bootstrap over training runs and base instances.
  (ii)  each policy-started arm at the budget vs the policy schedule it started from (arm
        minus policy): what the search adds to the policy's schedule within the budget.
        The memetic arm starts from the rule schedule as well, so its time-zero value is
        min(rule, policy) minus policy; the CP-SAT arm's hint is the policy alone.
  (iii) each policy-started arm vs the rule-started arm of the same method at the same
        budget (instance-level bootstrap).
  (iv)  the best dispatching rule vs the policy (rule minus policy), the time-zero value of
        the rule-started arms; rows Sel and All. With (i): the percent of cells on which
        the first-named method is shorter, equal, longer.
  (v)   time at which a rule-started arm first matches the selected run, on a one-second
        grid from 0 to the largest budget: (a) the earliest time at which the arm's mean
        makespan over the instances is at or below the policy's mean, or "not within the
        budget"; (b) per instance the earliest time its incumbent is at or below the
        policy's makespan: median over the instances (an instance never reached counts as
        later than the budget) and the percent never reached. The JSON adds the same with
        the seed-mean policy.
  (vi)  each rule-started arm at the budget vs the policy (arm minus policy), rows Sel and
        All: comparison (i) with the methods named in the other order, which is the order
        of the time-budget figure and of results/timed_curves.csv. Interval and direction
        only; the verdict of this comparison is the verdict of (i). Macros for the pooled
        row.
  Request clock only, row Sel only (the sampled schedules exist for the selected run):
  (vii)  the policy, sampled schedules, at the budget vs the policy's greedy schedule
         (sampled minus greedy): what sampling adds within the budget, never positive.
  (viii) the policy, sampled schedules, at the budget vs each rule-started arm at the same
         time since the request (policy minus arm): the comparison at equal computing
         effort. Verdict at the stated budget by the Sel rule of (i).
  (ix)   time of the first schedule per method and size: for the rules t_all, for the
         policy t and t_resident of the greedy schedule; median, minimum and maximum over
         the cells.
  (v) on the request clock: (a) and (b) against the greedy schedule's makespan, a fixed
      target as on the other clocks (whether or not the policy's own schedule exists yet),
      and in addition against the curve of the policy, sampled schedules, at the same
      time; a matching time after the policy record's budget uses a held value.
  Every clock:
  (x)    size of the solver's model, from the valid records of CP-SAT started from the rule
         schedule (arm cpsat_rule, the cells read above), per size, per combination, Pool and
         per group: the median, minimum and maximum over the cells, and the number of cells, of
           (a) the variables of the initial model: the number on the first line that starts
               '#Variables:' after the line that starts 'Initial optimization model' in the
               record's solver_log_head (the first pieces of the solver's log, each split
               into its lines; digits grouped by '). A record without solver_log_head, or
               without such a line before the next model of the log, gives no count; it is
               counted (...VarsMissing) and listed, never estimated. (a) is a property of
               the model and the instance.
           (b) build_s, the seconds to build the model;
           (c) peak_rss_gb, the peak memory of the solving process in GB;
           (d) the time of the first incumbent: the time of the first row of the record's
               anytime list, in seconds on the record's own clock, which for a model
               configuration starts with the solver call (after the model build, the search
               clock above) and for the neighbourhood search with the arm.
         (b) and (d) are timings of valid records (timing_validity's rule); (c) is measured on
         the same runs. A record without a value of (b), (c) or (d) leaves that value out and
         is listed. The statistics are descriptive: no bootstrap, no verdict and no random
         number, so every other number of this script stays as it is.
  Directions (ahead, behind, neither) come from stats_family.direction on the unrounded
  interval: ahead when the interval lies below zero.
  Statistics: scripts/stats_family.py, B = --boot (20000), alpha 0.05; a fresh
  default_rng(11) per hierarchical quantity and a fresh default_rng(7) per instance-level
  quantity (the conventions of p88 and p89). A pooled row keeps the combinations of a
  base instance together (cluster by base instance).

GROUPS (two pooled entries beside Pool, for every quantity and clock that has Pool)
  NoSetup  the combinations without setups (sequence-dependent setup times): N, L, W, LW
  Setup    the combinations with setups: S, LS, SW, LSW
  A group entry is the pooled row of Pool computed on the group's combinations alone, those
  the campaign holds: the same estimator, the same clustering by base instance, its own
  fresh default_rng(7) or default_rng(11) (so no entry draws from another entry's
  generator), interval and direction only, no verdict. It lists the combinations that
  entered. As for Pool it needs two or more of them with numbers, so a group none of whose
  combinations is in the campaign (or only one) is absent, (ix) included. On the request
  clock a group has a number
  when both compared methods have a schedule on every cell of its own combinations (a
  missing schedule in the other group does not remove it), and its curve starts at the
  first grid time at which every cell of its combinations has a schedule. (ix) per group:
  results.<size>.ix.<NoSetup|Setup>, over the group's cells. Macro names put NoSetup or
  Setup in the place of Pool; the names of (ix) end in the group (TmdFiftyFirstPolMedianSetup).
  Macro comments and the Markdown name the groups "combinations without setups" and
  "combinations with setups".

DECISION TABLE (request clock only)
  The table of the third question has one row per size, pooled entry (Pool, NoSetup,
  Setup) and reported budget of the size, and six columns, each a difference from the
  policy's greedy schedule in percent of it (row Sel):
    rules           the best of the four dispatching rules, (iv): Tmd<Size>RuleVsPol<G>Sel
    policy_sample   the policy, sampled schedules, (vii): Tmd<Size>PolSmpVsPol<B><G>
    memetic_rule    the memetic algorithm started from the rule schedule, (vi):
                    Tmd<Size>MemRuleStartVsPol<B><G>Sel
    cpsat_rule      CP-SAT started from the rule schedule, (vi): Tmd<Size>CpRuleStartVsPol<B><G>Sel
    memetic_policy  the memetic algorithm started from the policy's schedule, (ii):
                    Tmd<Size>MemPolStartVsPol<B><G>
    cpsat_policy    CP-SAT started from the policy's schedule, (ii): Tmd<Size>CpPolStartVsPol<B><G>
  (ii) follows the rules of (vi) on the request clock: the search started from the policy has
  its incumbent at run time x - o_pol (the memetic algorithm, seeded with the rule schedule
  as well, at x - max(o_pol, o_rule)), the greedy schedule counts from o_pol on, and a number
  needs a schedule of both on every cell of the entry; the pooled entries are those of (vi).
  A column has a number in a row when its value macro exists at that budget. The rule
  column has one value per size and entry and no budget: it has a number in the rows of the
  budgets at which the rules have a schedule on every cell of the entry (o_rule <= budget,
  the cells with a rule record), and in no row before. For every row and column one macro,
  the name of the value macro with the budget word in the place of the other columns and
  Bold at the end (Tmd<Size>RuleVsPol<B><G>SelBold, Tmd<Size>PolSmpVsPol<B><G>Bold,
  Tmd<Size>MemRuleStartVsPol<B><G>SelBold, Tmd<Size>CpRuleStartVsPol<B><G>SelBold,
  Tmd<Size>MemPolStartVsPol<B><G>Bold, Tmd<Size>CpPolStartVsPol<B><G>Bold), expands to
  \bfseries when the column holds the lowest value of the row and to nothing otherwise.
  The values compared are the printed texts of the value macros (two decimals) of all six
  columns, so two columns that print the same lowest value are both bold; a column without a
  number is never bold, and a row without a number has no bold.
  Same instances in a row: each value is a mean over the cells its own comparison selects,
  (iv) the instances of the size with both arrays (no record needed), (vii), (vi) and (ii)
  the cells with the records of both compared methods; a pooled entry takes the instances
  present in every one of its combinations. In the full mode every cell has every record,
  so the columns of a row share one set when the methods share their design instances; the
  full mode refuses a row whose columns with a number are means over different instances or
  combinations, except a row that is by_design at a size with the table on the common
  instances (below). The partial mode prints the
  number of instances of each column in the Markdown line of the row and goes on; there a
  cell without a valid record of an arm is left out (it is not a cell without a schedule),
  so the column of an incomplete arm can hold a value over fewer cells than the others.
  JSON results.<size>.decision.<budget>.<entry>: the printed values (null without a number),
  the bold columns, per column with a number its instances (n_instances, instances) and
  combinations (combinations_used), and same_instances (one set for every column with a
  number); the Markdown has one line per row.
  The table on the common instances: when the design gives the methods of a size different
  instances (a token range, e.g. CP-SAT on 000-009 at 10x25 and the other methods on
  000-029), the rows of the table above are means over different instances by design, and
  a second table of the same six columns is computed on the instances common to every
  method of the size: the intersection of the size's instances and of the design instances
  of every arm and of the policy records (the same comparisons (iv), (vii), (vi) and (ii),
  the same estimator, each with its own fresh generator, on the cells of these instances
  only). Each row of the table above then states by_design: every column with a number is
  a mean over its design cells (the rules over the size's instances, the policy, sampled
  schedules over the instances of the policy records, a search over the instances of its
  arm and of the policy records, each on the combinations of the entry). Macro names put
  Common after the size word (TmdTenCommonRuleVsPol<G>Sel, TmdTenCommonPolSmpVsPol<B><G>,
  TmdTenCommonMemRuleStartVsPol<B><G>Sel, TmdTenCommonCpRuleStartVsPol<B><G>Sel,
  TmdTenCommonMemPolStartVsPol<B><G>, TmdTenCommonCpPolStartVsPol<B><G>, the value of the
  mean only), the bold markers likewise (...Bold, by the rule above), and
  Tmd<Size>CommonInstN is the number of the common instances. JSON
  results.<size>.decision_common: instances, rows (as decision), and the entries iv, vii,
  vi and ii of the pooled rows it is computed from; the Markdown adds the table. A size
  whose methods share their design instances (the production batches when every token
  gives the same range) has no second table.

VERDICTS (only at the stated budget, --primary, default 600 s; only for (i))
  The shorter budgets are descriptive: intervals and curves, no verdict. At the stated
  budget, for (i) per size and rule-started arm, the family is the eight combinations
  (full mode refuses --combos other than the eight), with words ahead, behind and not
  distinguishable (p89_public_family.VERDICT_WORDS; ahead = the policy is shorter):
  Row Sel  one-sided p-values from the draws of the instance-level interval
           (stats_family.instance_boot with return_draws, boot_p_one_sided), Holm over the
           eight for H1: d < 0 and, as its own family, for H1: d > 0. ahead = the 95%
           interval lies below zero and the Holm-adjusted p for H1: d < 0 is below alpha;
           behind mirrored; not distinguishable otherwise. There is no seed-level test for
           one run.
  Row All  p89_public_family.superiority_verdict, the combined verdict of the rule
           comparison: hierarchical interval, seed-level t over the per-run means
           (stats_family.seed_t_p) and the hierarchical bootstrap p from the draws of the
           interval, each kind with its own Holm family over the eight per direction.
  The interval of a verdict is the interval reported for the same row (same estimator,
  seed and draws). Macros ...Verdict per combination and, per size, arm and row, the
  counts ...VerdictAheadN, ...VerdictBehindN, ...VerdictNotDistinguishableN. The pooled
  row and comparisons (ii) to (iv) and (vii) keep interval and direction only.
  Comparison (viii) has the verdict of row Sel above (ahead = the policy, sampled
  schedules, is shorter), at the stated budget on the request clock.

OUTPUTS (under --out-root, default the repository root)
  manuscript/macros_timed.tex      one \newcommand per number, names Tmd<letters>, and the bold
                                  markers of the decision table; per size Tmd<Size>InstN (the
                                  instances read at the size, the union of its arms') and per
                                  arm Tmd<Size><Arm>InstN (Arm: MemRuleStart, CpRuleStart,
                                  MemPolStart, CpPolStart, PolSmp), the instances per
                                  combination of its design; a variable count of (x) is
                                  printed with grouped digits, 562{,}796, which sets the same
                                  in text and in math
  results/timed_inference.json    every number, the settings, record counts per cell, the
                                  records and arrays used
  results/timed_inference.md      tables for reading
  results/timed_curves.csv        tidy rows on a logarithmic time grid, arm minus policy in
                                  percent of the policy; column row: Sel (the selected run,
                                  instance-level interval, every arm) or All (every training
                                  run, hierarchical interval, the rule-started arms; a
                                  policy-started arm ran from the selected run's schedule
                                  only); column n_runs; column origin states the clock.
                                  Request clock: the reference is the greedy schedule's
                                  makespan; each arm's rows start at the first grid time at
                                  which every cell of the arm has a schedule; method
                                  policy_sample adds the policy, sampled schedules (sampled
                                  minus greedy, row Sel); added columns held (yes when the
                                  row uses a held value), policy_budget_s (the policy
                                  record's budget), policy_first_s and rule_first_s (median
                                  o_pol and median o_rule of the size, for the policy's
                                  marker in the figure). Column group: all on the rows per
                                  combination and Pool; no_setups and setups on the pooled
                                  rows of a group (combination NoSetup or Setup), written
                                  after the rows of every size, at the end of the file;
                                  policy_first_s and rule_first_s of a group row are the
                                  medians over the group's cells

TAGGED RECORD (--tag <letters>, full mode, start or search clock only)
  A full run whose JSON, Markdown and CSV are written beside the outputs above, under names
  that carry the tag, and that writes no macro file:
    results/timed_inference_<tag>.json, results/timed_inference_<tag>.md,
    results/timed_curves_<tag>.csv
  Every check of the full mode applies. The JSON carries the field tag and the Markdown
  title names it; the untagged outputs are neither read nor written, and a refusal removes
  the three tagged files only. The request clock is refused with a tag: its outputs hold the
  macro file of record and the decision table. A tag is letters only, so a tagged name never
  equals an untagged one.

CHECKS (any failure refuses; a refusal writes nothing and removes the four outputs)
  * every cell of the plan has a valid record; each arm covers its design instances on all
    eight combinations, and an arm planned by a token without an instance range covers the
    size's evidence instances (003-029 at the production batches, 000-029 at 10x25); each
    record reaches every analysed time;
  * descriptors: the same sorted names in every combination, instance_<i> in row i, and
    instance i is the same base instance in every combination (the cluster unit);
  * arrays exist, are [n_descriptors, 2] with finite positive makespans;
  * the start schedule of every record equals the arrays: the rule schedule the best of
    the four rule arrays, the policy schedule the selected run's array (the
    policy-started records carry the selected seed, a planner field);
  * the training-run selection and the pilot's selection succeed;
  * decision table (request clock): every row is a mean over one set of instances and
    combinations, or, at a size with the table on the common instances, a row of the first
    table is by_design; every row of the table on the common instances is one set;
  * request clock: every size of the plan has a policy_sample token whose instances hold
    every instance of the size's plan, and every cell of the plan has a valid rule record
    and a valid policy record; the best rule makespan of a rule record equals the best of the four
    rule arrays and its t_all is a positive time; the greedy makespan of a policy record
    equals the selected run's array, its one greedy row carries greedy_t and that
    makespan, and its best schedule so far at greedy_t is no longer; the policy records of
    a size share one budget.

PARTIAL MODE
  --partial --out <folder> computes what the records on disk allow and writes only the
  JSON, the Markdown and the CSV into the folder (outside the repository's paper, results
  and record folders), each marked partial. Missing records, arrays and selections are
  listed and skipped; wrong shapes and mismatched instances still refuse.
  --partial-macros (only with --partial) also writes macros_timed.tex into the folder: the
  lines the full mode would write, built from the cells that exist, after a header line
  that marks the file partial, so that a table can be test built from a partial run. Its
  names pass the full mode's check (the macro file of record excepted), and a refusal
  removes it.
  --rule-as-policy (only with --partial) puts the best-rule schedule in the place of the
  policy and marks every output as a stand-in; --record-policy-seed (only with it) names
  the seed in the folders of policy-started records written from a stand-in schedule.
  --cpsat-folder (only with --partial) reads the rule-started CP-SAT records from
  <size>_cparms/<name>/<arm>/ (the pilot's folder 'pilot'), for a reading check.
  --rule-as-policy reads the start and search clocks only (no policy record exists for a
  stand-in policy). On the request clock a cell without its rule or policy record is
  listed and left out of every comparison that needs it.

Usage (repository root, one thread):
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PACT_ENV=exact PACT_SEEDS=301-305 \
    python scripts/p94_timed_inference.py \
      --campaign "50x25n30 80x25n30 = memetic_rule@600:3-11 memetic_policy@600:3-11 cpsat_rule@600:3-11 cpsat_policy@600:3-11 policy_sample@120:3-11" \
      --campaign "10x25 = memetic_rule@600 cpsat_rule@600:0-9 policy_sample@60" [--primary 600]
  ... --origin start --campaign "50x25n30 80x25n30 = memetic_rule@600 ..." (no policy_sample)
  ... --partial --out <folder> [--rule-as-policy --origin start] [--cpsat-arm ref]
      [--cpsat-folder pilot] [--partial-macros]
  ... --origin start --budgets 60 300 600 --tag startnine \
      --campaign "50x25n30 80x25n30 = memetic_rule@600:3-11 cpsat_rule@600:3-11"
"""
import argparse
import csv
import glob
import json
import math
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

import arm_registry as REG  # noqa: E402
import stats_family as SF  # noqa: E402
import timed_campaign_plan as P  # noqa: E402
from baselines.memetic.records import incumbent_at  # noqa: E402
from macro_io import emit  # noqa: E402
from p89_public_family import VERDICT_WORDS, f2, superiority_verdict  # noqa: E402

# p97 imports p75_deploy_cost, which empties sys.argv and moves to the repository root at
# import. The working directory is put back, so relative paths keep their meaning. sys.argv
# stays emptied, because params.py parses it whenever it is first imported (p97 imports it
# on demand); main() reads this script's own command line from the copy kept here.
_ARGV_KEPT, _CWD_KEPT = list(sys.argv), os.getcwd()
import p97_policy_anytime as P97  # noqa: E402
os.chdir(_CWD_KEPT)

COMBOS = tuple(P.COMBOS)
POOL = 'Pool'
# The two groups pooled beside Pool: (JSON key and macro word, text of the Markdown and the
# macro comments, the combinations, the value of column group of the CSV).
GROUPS = (('NoSetup', 'combinations without setups', ('N', 'L', 'W', 'LW'), 'no_setups'),
          ('Setup', 'combinations with setups', ('S', 'LS', 'SW', 'LSW'), 'setups'))
CSV_GROUP_ALL = 'all'               # column group of the rows per combination and of Pool
RULES =('FIFO', 'MOR', 'SPT', 'MWKR')
BUDGETS = (1, 10, 60, 300, 600)
PRIMARY = 600                       # the stated budget: the only one with verdict words
COUNT_WORD = dict(zip(VERDICT_WORDS, ('AheadN', 'BehindN', 'NotDistinguishableN')))
VERDICT_TEXT = {
    'Sel': ('ahead: the instance-level 95% interval lies below zero and the Holm-adjusted '
            'one-sided bootstrap p for "the policy is shorter" (draws of that interval) is below '
            'alpha; behind: the interval lies above zero and the Holm-adjusted one-sided bootstrap '
            'p for "the policy is longer" (its own Holm family) is below alpha; not '
            'distinguishable otherwise. No seed-level test exists for one run.'),
    'All': ('ahead: the hierarchical 95% interval lies below zero and both Holm-adjusted '
            'one-sided p-values for "the policy is shorter" (seed-level t over the per-run means, '
            'hierarchical bootstrap) are below alpha; behind: the interval lies above zero and '
            'both Holm-adjusted one-sided p-values for "the policy is longer" (each kind with its '
            'own Holm family) are below alpha; not distinguishable otherwise '
            '(p89_public_family.superiority_verdict).'),
}
HIER_SEED, INST_SEED = 11, 7
SIZE_WORD = {'50x25n30': 'Fifty', '80x25n30': 'Eighty', P.TRAINING_SIZE: 'Ten'}
METHOD_WORD = {'memetic': 'Mem', 'cpsat': 'Cp'}
METHOD_TEXT = {'memetic': 'memetic algorithm', 'cpsat': 'CP-SAT'}
ARM_OF = {v: k for k, v in P.ARMS.items()}        # (method, start) -> arm
DEFAULT_CAMPAIGNS = (
    '50x25n30 80x25n30 = memetic_rule@600 memetic_policy@600 cpsat_rule@600 cpsat_policy@600',
    f'{P.TRAINING_SIZE} = memetic_rule@600')
# the request clock adds the policy arm of p97 to the campaign of every size
POLICY_ARM, RULE_ARM = P97.ARM, P97.RULE_ARM            # 'policy_sample', 'rule_start'
DEFAULT_POLICY_TOKEN = f'{POLICY_ARM}@120'
DEFAULT_CAMPAIGNS_REQUEST = tuple(f'{c} {DEFAULT_POLICY_TOKEN}' for c in DEFAULT_CAMPAIGNS)
REQUEST_BUDGETS = (60, 300, 600)            # production sets
TRAINING_BUDGETS = (10, 60, 300, 600)       # the training size 10x25
SAMPLE_WORD = 'PolSmp'                      # macro word of the policy, sampled schedules
SAMPLE_TEXT = 'the policy, sampled schedules'
# the count macros of the design, Tmd<Size><word>InstN: the macro word and the text of each arm
ARM_WORD = {'memetic_rule': 'MemRuleStart', 'cpsat_rule': 'CpRuleStart',
            'memetic_policy': 'MemPolStart', 'cpsat_policy': 'CpPolStart',
            POLICY_ARM: SAMPLE_WORD}
ARM_TEXT = {'memetic_rule': 'memetic algorithm started from the rule schedule',
            'cpsat_rule': 'CP-SAT started from the rule schedule',
            'memetic_policy': "memetic algorithm started from the policy's schedule",
            'cpsat_policy': "CP-SAT started from the policy's schedule",
            POLICY_ARM: f'{SAMPLE_TEXT} (records of the policy arm)'}
# the decision table on the instances common to every method of a size: the word after the
# size word in its macro names (TmdTenCommonRuleVsPolPoolSel)
COMMON_WORD = 'Common'
NO_SCHEDULE = 'no schedule yet'
EPS = 1e-9                                  # a schedule at time t exists when its time <= t + EPS
OUTPUTS = (('macros', 'manuscript/macros_timed.tex'), ('json', 'results/timed_inference.json'),
           ('md', 'results/timed_inference.md'), ('csv', 'results/timed_curves.csv'))
# the outputs of a run with --tag: the three record files under names that carry the tag
TAG_OUTPUTS = (('json', 'results/timed_inference_{tag}.json'),
               ('md', 'results/timed_inference_{tag}.md'),
               ('csv', 'results/timed_curves_{tag}.csv'))
TAG_PATTERN = re.compile(r'[A-Za-z]+')
RECORD_DIRS = ('test_results', 'or_solution', 'train_log')
ORIGIN_TEXT = {
    'request': ('time since the request for a schedule, a cold start for every method: the '
                'dispatching rules from t_all of the rule record (all four rule schedules '
                'exist), the policy\'s greedy schedule from its t in the selected run\'s policy '
                'record (process start, checkpoint load included), the policy, sampled '
                'schedules, at its own t (best so far; after the record\'s budget the last value '
                'is held, which favours the other method); a search started from the rules at '
                'run time t - t_all of the start clock, a search started from the policy at run '
                'time t - its greedy t; no schedule before the offset; every training run takes '
                'the offsets of the selected run\'s records'),
    'start': ('wall clock from the start of the run: memetic algorithm from the start of the '
              'process that solves the instance (instance loading, start schedules and numba '
              'warm-up included); CP-SAT model arms from the start of the arm, model building '
              'included (instance load and hint validation before it not timed by the runner); '
              'the neighbourhood-search arm from its start'),
    'search': ('wall clock from the start of the search: the memetic algorithm\'s timed window, '
               'the CP-SAT solver call after the model build, the neighbourhood-search arm\'s '
               'start'),
}
NOT_REACHED = 'not within the budget'
# The decision table (request clock): (JSON key of the column, its text in the Markdown and
# the macro comments). The value macro of each column is named in decision_names.
DECISION_COLUMNS = (('rules', 'best of the four dispatching rules'),
                    ('policy_sample', SAMPLE_TEXT),
                    ('memetic_rule', 'memetic algorithm started from the rule schedule'),
                    ('cpsat_rule', 'CP-SAT started from the rule schedule'),
                    ('memetic_policy', "memetic algorithm started from the policy's schedule"),
                    ('cpsat_policy', "CP-SAT started from the policy's schedule"))
BOLD = '\\bfseries'
DECISION_RULE = (
    'one row per size, pooled entry and reported budget on the request clock; values: the '
    'printed text of the value macro of each column (p89 f2 of the difference from the greedy '
    'schedule of the selected run, in percent of it), null without a number; the rule column '
    'has a number at the budgets at which the rules have a schedule on every cell of the entry '
    'with a rule record (rules_from_s, the last o_rule of these cells); the columns started '
    'from the policy\'s schedule are comparison (ii), on the request clock by the rules of '
    '(vi); bold: the column or columns with the lowest printed value of the row over all six '
    'columns, none in a row without a number; macro per row and column: the value macro\'s '
    'name with the budget word, ending in Bold, \\bfseries or empty; instances, n_instances '
    'and combinations_used: the cells each column with a number is a mean over; '
    'same_instances: one set of cells for every column with a number, which the full mode '
    'requires (it refuses otherwise), except at a size with the table on the common instances, '
    'where a row whose columns are each a mean over the cells of their design (by_design) '
    'is kept')
DECISION_COMMON_RULE = (
    'results.<size>.decision_common, at a size whose methods have different design instances: '
    'the decision table computed on the instances common to every method of the size (the '
    "size's instances that are design instances of every arm and of the policy records), the "
    'same comparisons (iv), (vii), (vi) and (ii) on the cells of these instances only, each '
    'statistic with its own fresh generator; rows as in decision, every row one set of cells '
    '(the full mode refuses otherwise); macros: Common after the size word, the value of the '
    'mean and the bold markers; iv, vii, vi and ii hold the pooled entries it is computed from')
DESIGN_RULE = (
    'results.<size>.design: the instances of each arm in the plan and, on the request clock, '
    'of the policy records (policy_sample); range_given: every token of the arm gives an '
    'instance range, which is then the design of the arm; an arm with a token without one '
    "plans the size's evidence instances and must cover them")
MODEL_RULE = (
    'valid records of CP-SAT started from the rule schedule, per size, combination and pooled '
    'entry (two or more combinations): median, minimum and maximum over the cells of the '
    'variables of the initial model (solver_log_head: the first line starting "#Variables:" '
    'after the line starting "Initial optimization model", before the next model of the log), '
    'build_s, peak_rss_gb and the time of the first row of anytime (the record\'s clock: from '
    'the solver call for a model configuration, from the start of the arm for the '
    'neighbourhood search); a record without a value leaves it out and is listed under '
    'missing; descriptive statistics, no random number')
# (x) the size of the solver's model, from the records of this arm
MODEL_ARM = 'cpsat_rule'
MODEL_START_LINE = 'Initial optimization model'
MODEL_VARIABLES = re.compile(r"#Variables:\s*(\d[\d']*)")


def count_text(x):
    """A count with grouped digits as LaTeX sets them in text and in math: 562{,}796."""
    return f'{x:,.0f}'.replace(',', '{,}')


# (JSON key, macro word, words of the macro comments and the Markdown, text of a value)
MODEL_FIELDS = (
    ('variables', 'Vars', 'variables of the initial model', count_text),
    ('build_s', 'Build', 'seconds to build the model', lambda x: f'{x:.2f}'),
    ('peak_rss_gb', 'Memory', 'peak memory of the solve in GB', lambda x: f'{x:.1f}'),
    ('first_incumbent_s', 'FirstIncumbent', "seconds until the first incumbent, on the "
     "record's clock (for a model configuration from the solver call, after the model build)",
     lambda x: f'{x:.1f}'))


class Refusal(Exception):
    """The inputs do not allow the analysis; args[0] is the list of reasons."""


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

_ONES = ('Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten',
         'Eleven', 'Twelve', 'Thirteen', 'Fourteen', 'Fifteen', 'Sixteen', 'Seventeen',
         'Eighteen', 'Nineteen')
_TENS = ('', '', 'Twenty', 'Thirty', 'Forty', 'Fifty', 'Sixty', 'Seventy', 'Eighty', 'Ninety')


def number_word(n):
    """0 <= n < 1,000,000 in capitalised words without spaces: 300 -> ThreeHundred."""
    n = int(n)
    if not 0 <= n < 1000000:
        raise ValueError(f'no word for {n}')
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + (_ONES[n % 10] if n % 10 else '')
    if n < 1000:
        return _ONES[n // 100] + 'Hundred' + (number_word(n % 100) if n % 100 else '')
    return number_word(n // 1000) + 'Thousand' + (number_word(n % 1000) if n % 1000 else '')


def pools(combos):
    """The pooled entries of the given combinations: (key, column indices), Pool over all of
    them first, then each group over its combinations among them. A group with fewer than
    two of them is left out, as a pooled row needs two combinations."""
    out = [(POOL, list(range(len(combos))))]
    for key, _, members, _ in GROUPS:
        cols = [j for j, r in enumerate(combos) if r in members]
        if len(cols) >= 2:
            out.append((key, cols))
    return out


def group_keys():
    return tuple(g[0] for g in GROUPS)


def entry_text(r):
    """The words of an entry in the Markdown and the macro comments: a combination or Pool
    as it is named, a group by its text."""
    return next((g[1] for g in GROUPS if g[0] == r), r)


def gap(a, b):
    """100 (a - b) / b, elementwise (first-named minus second, in percent of the second)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return 100.0 * (a - b) / b


def log_grid(horizon, extra=(), per_decade=12, start=0.1):
    """0, points from `start` to `horizon` evenly spaced in log time (three significant
    digits), the horizon and the `extra` times: the time axis of the figure."""
    k0 = int(round(math.log10(start) * per_decade))
    k1 = int(math.floor(math.log10(horizon) * per_decade + 1e-9))
    pts = {float(f'{10 ** (k / per_decade):.3g}') for k in range(k0, k1 + 1)}
    pts = {p for p in pts if p <= horizon}
    return sorted({0.0, float(horizon)} | pts | {float(x) for x in extra if x <= horizon})


def shares(d):
    """Percent of entries shorter / equal / longer (d < 0, = 0, > 0)."""
    d = np.asarray(d, dtype=float).ravel()
    return dict(shorter=float(np.mean(d < 0)) * 100, equal=float(np.mean(d == 0)) * 100,
                longer=float(np.mean(d > 0)) * 100)


def seed_shares(d):
    """shares per training run of d [k, ...], averaged over the runs, with min and max."""
    per = [shares(x) for x in np.asarray(d)]
    out = {key: float(np.mean([p[key] for p in per])) for key in per[0]}
    out.update({f'{key}_min': float(min(p[key] for p in per)) for key in per[0]})
    out.update({f'{key}_max': float(max(p[key] for p in per)) for key in per[0]})
    return out


def stat_instance(d, B, alpha):
    """One fixed policy (or none): cluster bootstrap over base instances, a drawn instance
    bringing all its combinations (d [n] or [n, c]); fresh default_rng(7)."""
    ib = SF.instance_boot(d, B=B, seed=INST_SEED, alpha=alpha)
    return dict(mean=ib['mean'], lo=ib['lo'], hi=ib['hi'], n=int(ib['n']), c=int(ib['c']),
                direction=SF.direction(ib['lo'], ib['hi']), scheme=ib['scheme'],
                rng=f'default_rng({INST_SEED})', **shares(d))


def stat_seeds(d, B, alpha):
    """Every training run: crossed hierarchical bootstrap over runs and base instances
    (d [k, n] or [k, n, c]); fresh default_rng(11)."""
    hb = SF.hier_boot(d, B=B, seed=HIER_SEED, alpha=alpha)
    return dict(mean=hb['mean'], lo=hb['lo'], hi=hb['hi'], k=int(hb['k']), n=int(hb['n']),
                c=int(hb['c']), direction=SF.direction(hb['lo'], hb['hi']), scheme=hb['scheme'],
                rng=f'default_rng({HIER_SEED})', **seed_shares(d))


def compare(first, second, combos, B, alpha, seeds=False, used=None):
    """Rows of d = 100 (first - second) / second per combination and pooled.

    first, second: [n, C] arrays (NaN where a value is missing); with seeds=True one of
    them is [k, n, C] (every training run) and the other [n, C] or [1, n, C]. A
    combination uses the instances where every value is present; the pooled row (two or
    more combinations) the instances present in every one of them, each drawn instance
    bringing all its combinations. Each group of GROUPS gets the pooled row of its own
    combinations by the same code, after Pool; every statistic starts its own fresh
    generator, so the groups leave the numbers of the other entries unchanged.
    used: a dict that, when given, receives for every entry returned the cells its
    statistic is computed over, (instance rows of the arrays, combinations); it changes no
    number."""
    d = gap(first, second)
    ok = np.isfinite(d) if d.ndim == 2 else np.isfinite(d).all(axis=0)       # [n, C]
    def st(x):
        return stat_seeds(x, B, alpha) if seeds else stat_instance(x, B, alpha)
    out = {}
    for j, r in enumerate(combos):
        m = ok[:, j]
        if m.any():
            out[r] = dict(st(d[m, j] if d.ndim == 2 else d[:, m, j]), combinations=[r])
            if used is not None:
                used[r] = (np.flatnonzero(m).tolist(), [r])
    for key, members in pools(combos):
        cols = [j for j in members if ok[:, j].any()]
        if len(cols) >= 2:
            rows = ok[:, cols].all(axis=1)
            if rows.any():
                dp = d[rows][:, cols] if d.ndim == 2 else d[:, rows][:, :, cols]
                out[key] = dict(st(dp), combinations=[combos[j] for j in cols])
                if used is not None:
                    used[key] = (np.flatnonzero(rows).tolist(), [combos[j] for j in cols])
    return out


def selected_verdict(lo, hi, p_ahead, p_behind, alpha):
    """Verdict of one selected run against an arm: 'ahead' when the instance-level interval
    lies below zero and the Holm-adjusted one-sided bootstrap p for H1: d < 0 is below
    alpha, 'behind' when the interval lies above zero and the Holm-adjusted p for H1: d > 0
    (its own Holm family) is below alpha, 'not distinguishable' otherwise. A NaN p-value
    never meets its condition."""
    if hi < 0 and p_ahead < alpha:
        return VERDICT_WORDS[0]
    if lo > 0 and p_behind < alpha:
        return VERDICT_WORDS[1]
    return VERDICT_WORDS[2]


def _p(x):
    """A seed-level p that is not defined (one run, or per-run means that do not vary)
    enters Holm as 1."""
    return 1.0 if np.isnan(x) else float(x)


def verdict_family(first, second, combos, B, alpha, seeds=False):
    """Verdict words of comparison (i), one family over the given combinations.

    first, second and the instances used per combination as in compare, so the interval of
    each combination is the one compare reports (the same estimator, seed and draws).
    Row Sel (seeds=False): one-sided p-values from the instance-level bootstrap draws
    (stats_family.instance_boot, boot_p_one_sided), Holm over the family for H1: d < 0 and,
    as its own family, for H1: d > 0; no seed-level test. Row All (seeds=True): the
    seed-level t over the per-run means (stats_family.seed_t_p) and the hierarchical
    bootstrap p from the draws of the interval, each kind with its own Holm family per
    direction, and p89_public_family.superiority_verdict. Returns None when a combination
    has no instance (an incomplete family has no verdict)."""
    d = gap(first, second)
    rows = {}
    for j, r in enumerate(combos):
        if seeds:
            m = np.isfinite(d[:, :, j]).all(axis=0)
            x = d[:, m, j]
        else:
            m = np.isfinite(d[:, j])
            x = d[m, j]
        if not m.any():
            return None
        if seeds:
            hb = SF.hier_boot(x, B=B, seed=HIER_SEED, alpha=alpha, return_draws=True)
            rows[r] = dict(mean=hb['mean'], lo=hb['lo'], hi=hb['hi'], k=int(hb['k']),
                           n=int(hb['n']), p_t_less=_p(SF.seed_t_p(x, 'less')),
                           p_t_greater=_p(SF.seed_t_p(x, 'greater')),
                           p_boot_less=SF.boot_p_one_sided(hb['draws'], 'less'),
                           p_boot_greater=SF.boot_p_one_sided(hb['draws'], 'greater'))
        else:
            ib = SF.instance_boot(x, B=B, seed=INST_SEED, alpha=alpha, return_draws=True)
            rows[r] = dict(mean=ib['mean'], lo=ib['lo'], hi=ib['hi'], n=int(ib['n']),
                           p_boot_less=SF.boot_p_one_sided(ib['draws'], 'less'),
                           p_boot_greater=SF.boot_p_one_sided(ib['draws'], 'greater'))
    for kind in (('p_t', 'p_boot') if seeds else ('p_boot',)):
        for side in ('less', 'greater'):
            col = f'{kind}_{side}'
            for r, adj in zip(combos, SF.holm([rows[r][col] for r in combos])):
                rows[r][col + '_holm'] = adj
    for x in rows.values():
        x['verdict'] = (superiority_verdict(x['lo'], x['hi'], x['p_t_less_holm'],
                                            x['p_boot_less_holm'], x['p_t_greater_holm'],
                                            x['p_boot_greater_holm'], alpha) if seeds else
                        selected_verdict(x['lo'], x['hi'], x['p_boot_less_holm'],
                                         x['p_boot_greater_holm'], alpha))
    return dict(rows=rows, family=list(combos),
                counts={w: sum(x['verdict'] == w for x in rows.values()) for w in VERDICT_WORDS},
                rule=VERDICT_TEXT['All' if seeds else 'Sel'],
                scheme=('crossed hierarchical bootstrap over training runs and base instances, '
                        f'fresh default_rng({HIER_SEED})' if seeds else
                        f'instance-level cluster bootstrap, fresh default_rng({INST_SEED})'))


def first_time(ok, grid):
    """Earliest grid time at which ok (bool over the grid) holds, or inf."""
    ok = np.asarray(ok, dtype=bool)
    return float(grid[int(np.argmax(ok))]) if ok.any() else math.inf


def match_one(traj, pol, grid):
    """traj [T, m] arm makespans of m cells on the grid (non-increasing in time), pol [m]
    the policy's makespans of the same cells. Returns the matching time of the mean, the
    median of the per-cell matching times (a cell never reached counts as later than the
    grid) and the percent of cells never reached."""
    t_mean = first_time(traj.mean(axis=1) <= pol.mean() + 1e-9, grid)
    per = np.array([first_time(traj[:, i] <= pol[i] + 1e-9, grid) for i in range(traj.shape[1])])
    med = float(np.median(per))
    return dict(mean_s=None if math.isinf(t_mean) else t_mean,
                median_s=None if math.isinf(med) else med,
                never_pct=float(np.mean(np.isinf(per))) * 100,
                n_cells=int(traj.shape[1]), policy_mean=float(pol.mean()),
                arm_mean_at_zero=float(traj[0].mean()), arm_mean_at_end=float(traj[-1].mean()),
                per_cell_s=[None if math.isinf(x) else x for x in per.tolist()])


def match_block(traj, pol, combos, grid):
    """(v) per combination, pooled and per group: traj [T, n, C] on the one-second grid,
    pol [n, C]."""
    ok = np.isfinite(traj).all(axis=0) & np.isfinite(pol)                     # [n, C]
    out = {}
    for j, r in enumerate(combos):
        m = ok[:, j]
        if m.any():
            out[r] = dict(match_one(traj[:, m, j], pol[m, j], grid), combinations=[r],
                          n=int(m.sum()))
    for key, members in pools(combos):
        cols = [j for j in members if ok[:, j].any()]
        if len(cols) >= 2:
            rows = ok[:, cols].all(axis=1)
            if rows.any():
                t = traj[:, rows][:, :, cols].reshape(len(grid), -1)
                p = pol[rows][:, cols].reshape(-1)
                out[key] = dict(match_one(t, p, grid), combinations=[combos[j] for j in cols],
                                n=int(rows.sum()))
    return out


# ---------------------------------------------------------------------------
# request clock: a number only where both methods have a schedule on every cell
# ---------------------------------------------------------------------------

def _cells_ok(x):
    """[n, C] where a value is present (every training run of a [k, n, C] array)."""
    ok = np.isfinite(x)
    return ok if ok.ndim == 2 else ok.all(axis=0)


def lacking(first, second, pres):
    """Cells of pres [n, C] (cells with both records) on which either method has no
    schedule yet."""
    return pres & ~(_cells_ok(first) & _cells_ok(second))


def compare_at(first, second, pres, combos, B, alpha, seeds=False, used=None):
    """compare on the cells of pres [n, C] (cells with both records). A combination with a
    cell of pres on which either method has no schedule yet gets no number, and then
    neither does the pooled row: their entries are {no_schedule_yet: cells without a
    schedule, n_cells: cells}. A group is a pooled row of its own combinations: it has no
    number when one of them lacks a schedule, and compare's number otherwise. Every other
    entry is compare's, with its estimator and seed. used: as in compare, for the entries
    with a number."""
    first, second = np.where(pres, first, np.nan), np.where(pres, second, np.nan)
    miss = lacking(first, second, pres)
    got = {}
    if not miss.any():
        out = compare(first, second, combos, B, alpha, seeds, used=got)
    else:
        keep = ~miss.any(axis=0)
        cut = first.copy()
        cut[..., ~keep] = np.nan
        rows = compare(cut, second, combos, B, alpha, seeds, used=got)
        out = {}
        for j, r in enumerate(combos):
            if not keep[j]:
                out[r] = dict(no_schedule_yet=int(miss[:, j].sum()),
                              n_cells=int(pres[:, j].sum()), combinations=[r])
            elif r in rows:
                out[r] = rows[r]
        for key, members in pools(combos):
            if keep[members].all():
                if key in rows:             # a group whose combinations all have a schedule
                    out[key] = rows[key]
            elif sum(bool(pres[:, j].any()) for j in members) >= 2:
                out[key] = dict(no_schedule_yet=int(miss[:, members].sum()),
                                n_cells=int(pres[:, members].sum()),
                                combinations=[combos[j] for j in members])
    if used is not None:
        used.update({k: v for k, v in got.items() if 'mean' in out.get(k, {})})
    return out


def verdict_at(first, second, pres, combos, B, alpha, seeds=False):
    """verdict_family on the cells of pres, or {no_schedule_yet, n_cells} when either
    method has no schedule yet on one of them (a family with a missing number has no
    verdict)."""
    first, second = np.where(pres, first, np.nan), np.where(pres, second, np.nan)
    miss = lacking(first, second, pres)
    if miss.any():
        return dict(no_schedule_yet=int(miss.sum()), n_cells=int(pres.sum()))
    return verdict_family(first, second, combos, B, alpha, seeds)


def mark_held(rows, held, after):
    """Every entry of a comparison of the policy, sampled schedules, states whether it uses
    the value held after the policy record's budget."""
    for x in rows.values():
        x.update(held=bool(held), held_after_s=after)
    return rows


def match_one_request(traj, target, grid, held_after=None):
    """match_one on the request clock: traj [T, m] the arm (NaN before its first schedule),
    target [T, m] the policy (the greedy makespan at every time, or the sampled curve). A
    time counts when both have a value there and the arm's is at or below the target's.
    Also the first grid time at which the arm has a schedule on every cell and its mean
    then. With held_after (the policy record's budget), flags for each result that depends
    on a time after it."""
    with np.errstate(invalid='ignore'):
        t_mean = first_time(traj.mean(axis=1) <= target.mean(axis=1) + 1e-9, grid)
        per = np.array([first_time(traj[:, i] <= target[:, i] + 1e-9, grid)
                        for i in range(traj.shape[1])])
    full = np.isfinite(traj).all(axis=1)
    k0 = int(np.argmax(full)) if full.any() else None
    med = float(np.median(per))
    out = dict(mean_s=None if math.isinf(t_mean) else t_mean,
               median_s=None if math.isinf(med) else med,
               never_pct=float(np.mean(np.isinf(per))) * 100, n_cells=int(traj.shape[1]),
               policy_mean=float(target[-1].mean()),
               first_full_s=None if k0 is None else float(grid[k0]),
               arm_mean_at_first=None if k0 is None else float(traj[k0].mean()),
               arm_mean_at_end=float(traj[-1].mean()),
               per_cell_s=[None if math.isinf(x) else x for x in per.tolist()])
    if held_after is not None:
        late = grid[-1] > held_after + EPS

        def held(v):
            return bool(late if v is None else v > held_after + EPS)
        out.update(held_after_s=held_after, mean_held=held(out['mean_s']),
                   median_held=held(out['median_s']), never_held=bool(late))
    return out


def match_block_request(traj, target, pres, combos, grid, held_after=None):
    """(v) on the request clock, per combination, pooled and per group, over the cells of
    pres (cells with the records of both methods): traj and target [T, n, C] on the
    one-second grid. A pooled row takes the instances present in every one of its
    combinations."""
    out = {}
    for j, r in enumerate(combos):
        m = pres[:, j]
        if m.any():
            out[r] = dict(match_one_request(traj[:, m, j], target[:, m, j], grid, held_after),
                          combinations=[r], n=int(m.sum()))
    for key, members in pools(combos):
        cols = [j for j in members if pres[:, j].any()]
        if len(cols) >= 2:
            rows = pres[:, cols].all(axis=1)
            if rows.any():
                t = traj[:, rows][:, :, cols].reshape(len(grid), -1)
                p = target[:, rows][:, :, cols].reshape(len(grid), -1)
                out[key] = dict(match_one_request(t, p, grid, held_after),
                                combinations=[combos[j] for j in cols], n=int(rows.sum()))
    return out


def first_stats(x, pres):
    """Median, minimum and maximum of x [n, C] over the cells of pres (seconds)."""
    v = np.asarray(x, dtype=float)[pres]
    if not v.size:
        return None
    return dict(median_s=float(np.median(v)), min_s=float(v.min()), max_s=float(v.max()),
                n_cells=int(v.size))


# ---------------------------------------------------------------------------
# the decision table (request clock)
# ---------------------------------------------------------------------------

def decision_names(sw, bw, key):
    """{column: (value macro, bold macro)} of one row: size word, budget word, entry. The
    value macros of the columns started from the policy's schedule are those of (ii)."""
    return dict(rules=(f'Tmd{sw}RuleVsPol{key}Sel', f'Tmd{sw}RuleVsPol{bw}{key}SelBold'),
                policy_sample=(f'Tmd{sw}{SAMPLE_WORD}VsPol{bw}{key}',
                               f'Tmd{sw}{SAMPLE_WORD}VsPol{bw}{key}Bold'),
                memetic_rule=(f'Tmd{sw}MemRuleStartVsPol{bw}{key}Sel',
                              f'Tmd{sw}MemRuleStartVsPol{bw}{key}SelBold'),
                cpsat_rule=(f'Tmd{sw}CpRuleStartVsPol{bw}{key}Sel',
                            f'Tmd{sw}CpRuleStartVsPol{bw}{key}SelBold'),
                memetic_policy=(f'Tmd{sw}MemPolStartVsPol{bw}{key}',
                                f'Tmd{sw}MemPolStartVsPol{bw}{key}Bold'),
                cpsat_policy=(f'Tmd{sw}CpPolStartVsPol{bw}{key}',
                              f'Tmd{sw}CpPolStartVsPol{bw}{key}Bold'))


def _printed(x):
    """The text the value macro of an entry prints (p89's f2), or None without a number."""
    return f2(x['mean']) if x is not None and 'mean' in x else None


def lowest(values):
    """The columns whose printed value is the lowest of the row (every one of a tie), or []
    when no column has a number (values: {column: printed text or None})."""
    nums = {k: float(v) for k, v in values.items() if v is not None}
    if not nums:
        return []
    low = min(nums.values())
    return [k for k in values if k in nums and nums[k] == low]


def decision_table(S, o_rule, combos, used=None):
    """The rows of the decision table of one size: {budget: {entry: row}} over the reported
    budgets and the pooled entries, each row with the printed values of the six columns
    (p89's f2 of the entries the macros print; None without a number) and the bold columns.
    The rule column takes part when the rules have a schedule on every cell of the entry
    with a rule record (o_rule <= budget); rules_from_s is the last of these times. The
    columns started from the policy's schedule are (ii), those started from the rules (vi).
    used: {(column, budget or None for the rules): {entry: (instance rows, combinations)}},
    the cells of each value as compare selected them; with it, every row also states the
    instances and combinations of each column with a number and whether they are one set
    (same_instances; None without used)."""
    iv = S['iv'].get('selected', {})
    inst = S.get('instances')

    def cells(col, b, key):
        """(sorted instances, combinations) of the value of a column, or None."""
        u = (used or {}).get((col, None if col == 'rules' else str(b)), {}).get(key)
        if u is None:
            return None
        ids = [inst[k] for k in u[0]] if inst is not None else list(u[0])
        return sorted(int(i) for i in ids), list(u[1])
    out = {}
    for b in S['budgets']:
        t = float(b)
        rows = {}
        for key, members in pools(combos):
            o = o_rule[:, members]
            o = o[np.isfinite(o)]
            first = float(o.max()) if o.size else None
            src = dict(rules=iv.get(key) if first is not None and first <= t + EPS else None,
                       policy_sample=S['vii'].get(str(b), {}).get(key))
            for m in ('memetic', 'cpsat'):
                src[ARM_OF[(m, 'rule')]] = \
                    S['vi'].get(m, {}).get(str(b), {}).get('selected', {}).get(key)
                src[ARM_OF[(m, 'policy')]] = S.get('ii', {}).get(m, {}).get(str(b), {}).get(key)
            values = {c: _printed(src[c]) for c, _ in DECISION_COLUMNS}
            row = dict(values=values, bold=lowest(values), rules_from_s=first,
                       combinations=[combos[j] for j in members])
            if used is not None:
                got = {c: (cells(c, b, key) if v is not None else None)
                       for c, v in values.items()}
                sets = {(tuple(g[0]), tuple(sorted(g[1]))) for g in got.values() if g is not None}
                row.update(n_instances={c: None if g is None else len(g[0])
                                        for c, g in got.items()},
                           instances={c: None if g is None else g[0] for c, g in got.items()},
                           combinations_used={c: None if g is None else g[1]
                                              for c, g in got.items()},
                           same_instances=(len(sets) <= 1 and all(
                               got[c] is not None for c, v in values.items() if v is not None)))
            rows[key] = row
        out[str(b)] = rows
    return out


def decision_mismatches(R):
    """One text per row of a decision table whose columns with a number are not means over
    one set of instances and combinations (empty when every row shares one set). At a size
    with the table on the common instances (decision_common), a row of the first table whose
    columns are means over their design cells (by_design) is no mismatch; every row of the
    table on the common instances must be one set."""
    bad = []
    for size, S in R.items():
        common = S.get('decision_common')
        tables = [('decision table', S.get('decision', {}), common is not None)]
        if common is not None:
            tables.append(('decision table on the common instances', common.get('rows', {}),
                           False))
        for what, table, design_ok in tables:
            for b, rows in table.items():
                for key, row in rows.items():
                    if row.get('same_instances') is not False:
                        continue
                    if design_ok and row.get('by_design') is True:
                        continue
                    have = '; '.join(
                        f'{c} ' + ('cells unknown' if row['instances'][c] is None else
                                   f'{row["n_instances"][c]} instances '
                                   f'({P.compress(row["instances"][c])}), combinations '
                                   + ','.join(row['combinations_used'][c]))
                        for c, _ in DECISION_COLUMNS if row['values'][c] is not None)
                    bad.append(f'{what}, {size}, {b} s, {entry_text(key)}: the columns with a '
                               f'number are not means over one set of instances ({have}); '
                               f'every column of a row must be a mean over the same instances'
                               + (', or over the instances of its design' if design_ok else ''))
    return bad


def common_instances(instances, design):
    """The instances common to every method of a size: those of the size (instances) that
    are design instances of every arm and of the policy records (design: {arm:
    {'instances': [...]}}, results.<size>.design)."""
    keep = set(instances)
    for d in design.values():
        keep &= set(d['instances'])
    return sorted(keep)


def column_design(instances, design):
    """{column: sorted instances} of the decision table by design: the rules over the size's
    instances, the policy, sampled schedules over those of the policy records, a search over
    those of its arm and of the policy records; a column whose arm is not in the design is
    absent."""
    inst = set(instances)
    pol = set(design[POLICY_ARM]['instances']) & inst if POLICY_ARM in design else inst
    out = {'rules': sorted(inst)}
    for col, _ in DECISION_COLUMNS[1:]:
        if col == POLICY_ARM:
            if POLICY_ARM in design:
                out[col] = sorted(pol)
        elif col in design:
            out[col] = sorted(set(design[col]['instances']) & pol)
    return out


def mark_by_design(rows, want):
    """by_design for every row of a decision table built with its cells (decision_table with
    used): True when every column with a number is a mean over its design instances (want,
    from column_design) and over the combinations of the row. Changes no value."""
    for brows in rows.values():
        for row in brows.values():
            if 'instances' not in row:
                continue
            row['by_design'] = all(
                c in want and row['instances'][c] == want[c]
                and sorted(row['combinations_used'][c]) == sorted(row['combinations'])
                for c, v in row['values'].items() if v is not None)
    return rows


# ---------------------------------------------------------------------------
# (x) the size of the solver's model
# ---------------------------------------------------------------------------

def initial_variables(rec):
    """(count, None) or (None, reason): the variables of the initial model, the number on the
    first line that starts '#Variables:' after the line that starts 'Initial optimization
    model' in solver_log_head and before the next model of the log. solver_log_head is a list
    of the pieces of the log (one piece can hold several lines) or one text."""
    head = rec.get('solver_log_head')
    if not head:
        return None, 'no solver_log_head'
    # the solver hands its log over in pieces, and one piece can hold several lines
    lines = [y for x in ([head] if isinstance(head, str) else head) for y in str(x).splitlines()]
    start = next((k for k, x in enumerate(lines) if x.strip().startswith(MODEL_START_LINE)), None)
    if start is None:
        return None, f'no line starting {MODEL_START_LINE!r} in solver_log_head'
    for x in lines[start + 1:]:
        s = x.strip()
        if s.startswith('#Variables:'):
            m = MODEL_VARIABLES.match(s)
            if m is None:
                return None, f'variables line not read: {s[:80]!r}'
            return int(m.group(1).replace("'", '')), None
        if 'optimization model' in s:
            break
    return None, f"no '#Variables:' line after the line starting {MODEL_START_LINE!r}"


def _number(x):
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    return float(x) if math.isfinite(x) else None


def model_fields(rec):
    """The values of (x) of one record; None where the record does not state one."""
    v, why = initial_variables(rec)
    first = (rec.get('anytime') or [None])[0]
    return dict(variables=v, variables_reason=why, build_s=_number(rec.get('build_s')),
                peak_rss_gb=_number(rec.get('peak_rss_gb')),
                first_incumbent_s=(_number(first[0]) if isinstance(first, (list, tuple)) and first
                                   else None))


def model_stats(values):
    """Median, minimum and maximum of the values (None without one)."""
    if not values:
        return None
    v = np.asarray(values, dtype=float)
    return dict(median=float(np.median(v)), min=float(v.min()), max=float(v.max()),
                n_cells=int(v.size))


def solver_model(cells, combos):
    """(x) of one size: per combination with a valid record, per pooled entry of two or more
    such combinations (Pool and each group), the statistics of every field over its cells,
    the number of cells and of cells without a variable count; and the records without a
    value, per field. cells: dicts with combo, idx, record and the fields of model_fields."""
    by = {r: [c for c in cells if c['combo'] == r] for r in combos}

    def entry(cs, members):
        x = dict(n_cells=len(cs), combinations=members)
        for key, *_ in MODEL_FIELDS:
            x[key] = model_stats([c[key] for c in cs if c[key] is not None])
        x['variables_missing'] = sum(c['variables'] is None for c in cs)
        return x
    entries = {r: entry(by[r], [r]) for r in combos if by[r]}
    for key, members in pools(combos):
        have = [combos[j] for j in members if by[combos[j]]]
        if len(have) >= 2:
            entries[key] = entry([c for r in have for c in by[r]], have)
    missing = {key: [dict(record=c['record'], combination=c['combo'], instance=c['idx'],
                          **({'reason': c['variables_reason']} if key == 'variables' else {}))
                     for c in cells if c[key] is None] for key, *_ in MODEL_FIELDS}
    return dict(arm=MODEL_ARM, n_cells=len(cells), entries=entries, missing=missing)


def read_record(rec, runner, grid, origin, offset=None):
    """{t: makespan} of one record on the grid, through the planner's reader
    (timed_campaign_plan.record_rows); times beyond the run's budget are absent. With an
    offset (the request clock), the value at grid time t is the start-clock value at run
    time t - offset, and times before the offset are absent."""
    if offset is None:
        rows = P.record_rows(rec, runner, times=tuple(grid), origin=origin)
        return {float(r['t']): (None if r['makespan'] is None else float(r['makespan']))
                for r in rows}
    run = {float(t): max(float(t) - offset, 0.0) for t in grid if float(t) >= offset - EPS}
    rows = P.record_rows(rec, runner, times=tuple(run.values()), origin='start')
    at = {float(r['t']): (None if r['makespan'] is None else float(r['makespan'])) for r in rows}
    return {t: at[x] for t, x in run.items() if x in at}


def start_makespans(rec, runner, start):
    """(rule start, policy start): the makespans a record states for its start schedules."""
    if runner == 'memetic':
        seeds = rec.get('seeds') or []
        rule = [s.get('makespan') for s in seeds if s.get('source') == 'warm_cache']
        pol = [s.get('makespan') for s in seeds if s.get('source') == 'seed_from']
        return (float(rule[0]) if rule else None), (float(pol[0]) if pol else None)
    w = rec.get('warm_makespan')
    w = None if w is None else float(w)
    return (w, None) if start == 'rule' else (None, w)


def rel(path):
    return P.rel(path)


def seconds_text(v):
    return NOT_REACHED if v is None else f'{v:g}'


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return None if not math.isfinite(x) else x
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


# ---------------------------------------------------------------------------
# selections (the same rules through the same code)
# ---------------------------------------------------------------------------

def select_policy_seed():
    """arm_registry.select_seed('released') on the active seeds: (seed, {seed: (score,
    update index)})."""
    return REG.select_seed('released')


def select_cpsat_arm(tr_root, slots, bars):
    """The pilot's choice, timed_campaign_plan.select_pilot on both production batches.
    Returns (arm or None, reasons)."""
    res = P.select_pilot(tr_root, P.SIZES, slots, bars)
    if res['refused']:
        return None, res['refused']
    return res['chosen'], []


# ---------------------------------------------------------------------------
# campaigns and records
# ---------------------------------------------------------------------------

class FolderCampaign(P.Campaign):
    """The planner's campaign; cpsat_folder replaces the CP-SAT output folder name
    (evidence_rule<B>s) by another one, e.g. the pilot's 'pilot'."""
    cpsat_folder = None

    def cpsat_out(self, seg):
        return self.cpsat_folder or P.Campaign.cpsat_out(self, seg)


def parse_campaign(spec):
    """'50x25n30 80x25n30 = memetic_rule@600 cpsat_rule@600:3-29' -> (sizes, tokens)."""
    left, sep, right = spec.partition('=')
    sizes = left.replace(',', ' ').split()
    tokens = right.split()
    if not sep or not sizes or not tokens:
        raise Refusal([f'campaign {spec!r}: want "<sizes> = <arm tokens>"'])
    return sizes, tokens


def make_campaign(sizes, tokens, a, policy_seed, cpsat_arm):
    """The planner's Campaign for one launch, with the planner's defaults for every
    option not given here."""
    argv = ['table', '--arms', ' '.join(tokens), '--sizes', *sizes, '--combos', *a.combos,
            '--cpsat-arm', cpsat_arm, '--max-invalid', str(a.max_invalid)]
    for opt, val in (('--budget', a.budget), ('--rule-cache', a.rule_cache),
                     ('--policy-cache', a.policy_cache), ('--result-root', a.result_root),
                     ('--workers', a.workers), ('--memetic-seed', a.memetic_seed),
                     ('--cpsat-seed', a.cpsat_seed), ('--bars', a.bars),
                     ('--policy-seed', policy_seed)):
        if val is not None and val != '':
            argv += [opt, str(val)]
    if a.slots:
        argv += ['--slots', *a.slots]
    camp = FolderCampaign(P.build_parser().parse_args(argv))
    camp.cpsat_folder = a.cpsat_folder
    return camp


def read_cells(camps, grid, origin, offsets=None):
    """Every cell of every campaign: its state by the planner's rule and, for a valid
    record, its reading on the grid. Returns (cells, counts per arm|size|combination,
    gaps). offsets (request clock): {(size, combination, instance): {'rule': o_rule,
    'policy': o_pol}}; a record is read at run time t - the offset of its start, and a
    cell without that offset is left unread (its rule or policy record is listed)."""
    cells, counts, gaps = {}, {}, []
    for camp in camps:
        for seg in camp.segments:
            for cid, (c, state, why) in P.cell_states(camp, seg).items():
                key = (seg.arm, c.size, c.combo, c.idx)
                if key in cells:
                    raise Refusal([f'{cid}: the cell appears in two campaigns'])
                path = camp.record(c)
                cnt = counts.setdefault(f'{seg.arm}|{c.size}|{c.combo}', {})
                cnt[state] = cnt.get(state, 0) + 1
                x = dict(arm=seg.arm, runner=seg.runner, start=seg.start, segment=seg.label,
                         size=c.size, combo=c.combo, idx=c.idx, state=state, reason=why,
                         record=rel(path))
                cells[key] = x
                if state != 'valid':
                    gaps.append(f'{cid}: {state}' + (f' ({why})' if why else '')
                                + f' [{rel(path)}]')
                    continue
                rec = json.load(open(path))
                off = None
                if offsets is not None:
                    o = offsets.get((c.size, c.combo, c.idx), {})
                    off = o.get(seg.start)
                    if seg.runner == 'memetic' and seg.start == 'policy':
                        # seeded with the rule schedule as well: both must exist
                        off = (None if off is None or o.get('rule') is None
                               else max(off, o['rule']))
                    x['offset_s'] = off
                try:
                    if offsets is None:
                        x['reading'] = read_record(rec, seg.runner, grid, origin)
                    else:
                        x['reading'] = ({} if off is None else
                                        read_record(rec, seg.runner, grid, origin, offset=off))
                except (ValueError, KeyError) as e:
                    raise Refusal([f'{rel(path)}: the reader cannot read this record ({e})'])
                x['rule_start'], x['policy_start'] = start_makespans(rec, seg.runner, seg.start)
                x.update(budget_s=float(rec['budget_s']),
                         final_makespan=float(rec['makespan'] if seg.runner == 'memetic'
                                              else rec['objective']),
                         search_start_wall_s=rec.get('search_start_wall_s'),
                         build_s=rec.get('build_s'), cpsat_arm=rec.get('arm'),
                         cpu_share=rec.get('cpu_share'))
                if seg.arm == MODEL_ARM:
                    x['model'] = model_fields(rec)          # (x), on every clock
                if offsets is not None and off is None:
                    continue
                lacking = [t for t in grid if x['reading'].get(t) is None
                           and (off is None or t >= off - EPS)]
                if lacking:
                    gaps.append(f'{cid}: no reading at {len(lacking)} of {len(grid)} analysed '
                                f'times (first {lacking[0]:g} s; budget {rec["budget_s"]:g} s, '
                                f'origin {origin}) [{rel(path)}]')
    return cells, counts, gaps


def split_policy_tokens(specs, origin):
    """(campaigns without the policy_sample tokens, {size: (budget, instances or None,
    token)}). The planner never sees a policy_sample token; it is read on the request
    clock only."""
    out, pol = [], {}
    for sizes, tokens in specs:
        keep = []
        for tok in tokens:
            name, _, rest = tok.partition('@')
            if name != POLICY_ARM:
                keep.append(tok)
                continue
            if origin != 'request':
                raise Refusal([f'{tok}: the sampled schedules are read on the request clock '
                               f'only (--origin request)'])
            b, _, inst = rest.partition(':')
            try:
                budget = float(b)
                insts = P.parse_instances(inst) if inst else None
            except ValueError:
                raise Refusal([f'{tok}: want {POLICY_ARM}@<budget seconds>[:instances]'])
            if not budget > 0:
                raise Refusal([f'{tok}: the budget must be positive'])
            for s in sizes:
                if s in pol:
                    raise Refusal([f'{tok}: a second {POLICY_ARM} token for {s} '
                                   f'({pol[s][2]})'])
                pol[s] = (budget, insts, tok)
        out.append((sizes, keep))
    return out, pol


def new_record_state(path, want, check, max_invalid):
    """(state, reason, record) of a p97 record, by p97's own rule (plan_state): foreign when
    a field of the plan differs, else valid or invalid by check (p97.classify or
    p97.classify_rules); without the file held (max_invalid moved aside) or missing."""
    if os.path.exists(path):
        rec = json.load(open(path))
        for k, v in want.items():
            have = rec.get(k)
            if isinstance(v, float) and isinstance(have, (int, float)):
                if abs(have - v) > 1e-9:
                    return 'foreign', f'{k}={have!r}, plan {v!r}', rec
            elif have != v:
                return 'foreign', f'{k}={have!r}, plan {v!r}', rec
        ok, why = check(rec)
        return ('valid' if ok else 'invalid'), why, rec
    if P.n_invalid(path) >= max_invalid:
        return 'held', f'{P.n_invalid(path)} invalid records in {rel(P.invalid_dir(path))}', None
    return 'missing', '', None


def greedy_row(rec):
    """(t, t_resident, makespan) of the one greedy row of a policy record's schedules,
    or a reason why there is none."""
    cols = rec.get('schedules_columns') or []
    try:
        k, t, tr, ms = (cols.index(c) for c in ('kind', 't', 't_resident', 'makespan'))
    except ValueError:
        return f'schedules_columns {cols}'
    rows = [r for r in rec.get('schedules') or [] if r[k] == 'greedy']
    if len(rows) != 1:
        return f'{len(rows)} greedy rows'
    return float(rows[0][t]), float(rows[0][tr]), float(rows[0][ms])


def request_records(a, plan, pol_tokens, sel, grid, root):
    """The rule record and the selected run's policy record of every cell of the plan
    (p97's folders under root, p97's validity, the plan's fields). Returns ({(size,
    combination, instance): {'rule': ..., 'policy': ...}} for the valid records, the
    records' list, counts per arm|size|combination, gaps, refusals)."""
    out, listing, counts, gaps, refusals = {}, [], {}, [], []
    model = (REG.PREFIX + REG.model('released', None, sel)) if sel is not None else None
    sizes = list(dict.fromkeys(s for s, _, _ in plan))
    for size in sizes:
        # a range in the token is the design of the policy records; a cell of the plan
        # outside it has no policy record and is listed below
        if size not in pol_tokens:
            gaps.append(f'{size}: no {POLICY_ARM} token in the campaign; the request clock needs '
                        f'the policy record of every cell')
    if sel is None:
        gaps.append(f'{POLICY_ARM}: no selected training run, so no policy record is read')
    for size, combo, idx in plan:
        name = P97.instance_name(idx)
        cell = out.setdefault((size, combo, idx), {})
        jobs = [(RULE_ARM, RULE_ARM, P97.rule_record_path(root, size, combo, idx),
                 dict(arm=RULE_ARM, size=size, combination=combo, instance=name,
                      workers=P97.RULE_WORKERS), P97.classify_rules)]
        tok = pol_tokens.get(size)
        if tok is not None and sel is not None:
            budget, insts, text = tok
            if insts is not None and idx not in insts:
                gaps.append(f'{POLICY_ARM}@{budget:g}/{size}/{combo}/{idx:03d}: not in the plan '
                            f'of {text}')
            else:
                jobs.append((POLICY_ARM, f'{POLICY_ARM}@{budget:g}',
                             P97.record_path(root, size, budget, a.policy_batch, f's{sel}', combo,
                                             idx),
                             dict(arm=POLICY_ARM, size=size, combination=combo, instance=name,
                                  budget_s=budget, budget_clock='wall', workers=a.policy_workers,
                                  batch=a.policy_batch, base_seed=a.policy_base_seed, model=model,
                                  training_seed=sel, environment=REG.env()),
                             P97.classify))
        for arm, label, path, want, check in jobs:
            cid = f'{label}/{size}/{combo}/{idx:03d}'
            state, why, rec = new_record_state(path, want, check, a.max_invalid)
            cnt = counts.setdefault(f'{arm}|{size}|{combo}', {})
            cnt[state] = cnt.get(state, 0) + 1
            if state != 'valid':
                gaps.append(f'{cid}: {state}' + (f' ({why})' if why else '') + f' [{rel(path)}]')
                continue
            if arm == RULE_ARM:
                t_all = rec.get('t_all')
                if not isinstance(t_all, (int, float)) or not 0 < t_all < math.inf:
                    refusals.append(f'{rel(path)}: t_all {t_all!r} is not a positive time')
                    continue
                x = dict(t_all=float(t_all), best_makespan=float(rec['best_makespan']),
                         best_rule=rec.get('best_rule'), record=rel(path))
                listing.append(dict(arm=arm, size=size, combination=combo, instance=idx,
                                    record=rel(path), t_all=x['t_all'],
                                    best_rule=x['best_rule'], best_makespan=x['best_makespan'],
                                    valid_reason=why))
            else:
                g = greedy_row(rec)
                gm = rec.get('greedy_makespan')
                if isinstance(g, str) or abs(g[0] - float(rec['greedy_t'])) > 1e-9 \
                        or abs(g[2] - float(gm)) > 1e-9:
                    refusals.append(f'{rel(path)}: the greedy row of the schedules '
                                    f'({g}) does not carry greedy_t {rec.get("greedy_t")} and '
                                    f'greedy_makespan {gm}')
                    continue
                at_g = incumbent_at(rec, g[0], 'wall', 'process')
                if at_g is None or at_g > float(gm) + 1e-9:
                    refusals.append(f'{rel(path)}: best schedule so far at the greedy schedule\'s '
                                    f'time {g[0]:g} s is {at_g}, above the greedy makespan {gm:g}')
                    continue
                x = dict(t=g[0], t_resident=g[1], greedy_makespan=float(gm),
                         budget_s=float(rec['budget_s']), record=rel(path),
                         reading={float(t): incumbent_at(rec, float(t), 'wall', 'process')
                                  for t in grid})
                listing.append(dict(arm=arm, size=size, combination=combo, instance=idx,
                                    record=rel(path), greedy_t=x['t'], greedy_t_resident=x['t_resident'],
                                    greedy_makespan=x['greedy_makespan'], budget_s=x['budget_s'],
                                    best_makespan=rec.get('makespan'),
                                    n_schedules=rec.get('n_schedules'), cpu_share=rec.get('cpu_share'),
                                    valid_reason=why))
            cell['rule' if arm == RULE_ARM else 'policy'] = x
    return out, listing, counts, gaps, refusals


# ---------------------------------------------------------------------------
# descriptors and arrays
# ---------------------------------------------------------------------------

def descriptor_names(data_root, size, combo):
    return sorted(os.path.basename(p)[:-len('.desc.json')]
                  for p in glob.glob(os.path.join(data_root, size, 'desc', combo, '*.desc.json')))


def check_descriptors(data_root, size, combos, instances):
    """Number of descriptors per combination; refuses unless every combination holds the
    same sorted names, instance_<i> is row i, and instance i is one base instance in every
    combination."""
    names = {r: descriptor_names(data_root, size, r) for r in combos}
    first = names[combos[0]]
    if not first:
        raise Refusal([f'no descriptors in {rel(os.path.join(data_root, size, "desc"))}'])
    bad = [f'{size}/{r}: descriptor names differ from {combos[0]}'
           for r in combos if names[r] != first]
    for i in instances:
        if i >= len(first) or first[i] != f'instance_{i:03d}':
            bad.append(f'{size}: row {i} is not instance_{i:03d}')
            continue
        bases = {json.load(open(os.path.join(data_root, size, 'desc', r,
                                              f'instance_{i:03d}.desc.json')))['base_instance']
                 for r in combos}
        if len(bases) != 1:
            bad.append(f'{size}: instance_{i:03d} is not one base instance across the '
                       f'combinations ({sorted(bases)[:3]})')
    if bad:
        raise Refusal(bad)
    return len(first)


def load_column(path, n_desc, refusals, gaps, what):
    """Makespan column of a result array, or None (missing: a gap; bad: a refusal)."""
    if not os.path.exists(path):
        gaps.append(f'missing {what}: {rel(path)}')
        return None
    a = np.load(path)
    if a.ndim != 2 or a.shape != (n_desc, 2):
        refusals.append(f'{what} {rel(path)} has shape {a.shape}, want ({n_desc}, 2)')
        return None
    ms = a[:, 0].astype(float)
    if not np.all(np.isfinite(ms)) or np.any(ms <= 0):
        refusals.append(f'{what} {rel(path)} holds a non-finite or non-positive makespan')
        return None
    return ms


def load_arrays(size, combos, instances, n_desc, seeds, stand_in, refusals, gaps):
    """rule [n, C] (best of the four rules), policy {seed: [n, C]} (complete seeds only)
    on the given instances, and the files read."""
    idx = np.asarray(instances, dtype=int)
    rule = np.full((len(idx), len(combos)), np.nan)
    files = []
    for j, r in enumerate(combos):
        cols = []
        for q in RULES:
            p = f'{REG.rule_dir(size)}/Result_{q}_{r}.npy'
            files.append(rel(p))
            cols.append(load_column(p, n_desc, refusals, gaps, f'rule array {q}'))
        if all(c is not None for c in cols):
            rule[:, j] = np.min(np.stack(cols), axis=0)[idx]
    if stand_in:
        return rule, {'stand-in': rule.copy()}, files
    pol = {}
    for s in seeds:
        arr = np.full((len(idx), len(combos)), np.nan)
        for j, r in enumerate(combos):
            p = REG.result_path('released', None, s, size, r)
            files.append(rel(p))
            col = load_column(p, n_desc, refusals, gaps, f'policy array of seed {s}')
            if col is not None:
                arr[:, j] = col[idx]
        if np.isfinite(arr).all():
            pol[s] = arr
    return rule, pol, files


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------

def collect(a, budgets, origin, specs, sel, cpsat_arm, stand_in, record_seed, partial, grid,
            size_budgets=None):
    """Records, arrays and every check, before any statistic. Raises Refusal; returns the
    data per size, the gaps (empty in full mode), the counts and the files read.
    size_budgets (request clock): {size: budgets}."""
    gaps, refusals = [], []
    policy_seed = record_seed if stand_in else sel
    request = origin == 'request'
    specs, pol_tokens = split_policy_tokens(specs, origin)
    # (size, arm) -> for each token of the arm, whether it gives an instance range (the
    # design of the arm); an arm with a token without one plans the evidence instances
    ranged = {}
    for sizes, tokens in specs:
        for tok in tokens:
            arm, _, rest = tok.partition('@')
            for s in sizes:
                ranged.setdefault((s, arm), []).append(bool(rest.partition(':')[2]))
    camps, size_spec = [], {}
    for sizes, tokens in specs:
        keep = []
        for tok in tokens:
            arm = tok.partition('@')[0]
            if arm not in P.ARMS:
                raise Refusal([f'unknown arm {arm} in {tok!r}'])
            method, start = P.ARMS[arm]
            if start == 'policy' and method == 'cpsat' and a.cpsat_folder:
                raise Refusal([f'{tok}: --cpsat-folder reads rule-started CP-SAT records only'])
            if start == 'policy' and policy_seed is None:
                gaps.append(f'{tok}: no policy seed for the folders of policy-started records '
                            f'(the selection is unavailable); not read')
                continue
            keep.append(tok)
        for s in sizes:
            if s in size_spec:
                raise Refusal([f'size {s} appears in two campaigns'])
            size_spec[s] = keep
        if keep:
            camps.append(make_campaign(sizes, keep, a, policy_seed, cpsat_arm or 'ref_fullhint'))
    new, new_listing, offsets = {}, [], None
    if request:
        plan = list(dict.fromkeys((c.size, c.combo, c.idx) for camp in camps
                                  for seg in camp.segments for c in camp.cells(seg)))
        new, new_listing, new_counts, new_gaps, new_refusals = request_records(
            a, plan, pol_tokens, sel, grid, a.result_root or ROOT)
        gaps += new_gaps
        refusals += new_refusals
        offsets = {k: {start: v[key][field] for start, key, field in
                       (('rule', 'rule', 't_all'), ('policy', 'policy', 't')) if key in v}
                   for k, v in new.items()}
    cells, counts, cell_gaps = read_cells(camps, grid, origin, offsets)
    gaps += cell_gaps
    if request:
        counts.update(new_counts)

    combos = list(a.combos)
    ci = {r: j for j, r in enumerate(combos)}
    if not partial and combos != list(COMBOS):
        gaps.append(f'combinations {combos}: the analysis needs all eight')
    seeds = ['stand-in'] if stand_in else list(REG.seeds())
    data, files, records = {}, [], []
    for size in size_spec:
        here = [c for c in cells.values() if c['size'] == size]
        inst = sorted({c['idx'] for c in here})
        if not inst:
            continue
        arms = sorted({c['arm'] for c in here}, key=list(P.ARMS).index)
        evid = P.parse_instances(P.DEFAULT_INSTANCES.get(size, P.OTHER_DEFAULT_INSTANCES))
        # the design: the instances of each arm in the plan, and of the policy records
        design = {}
        for arm in arms:
            have = sorted({c['idx'] for c in here if c['arm'] == arm})
            given = all(ranged.get((size, arm), [False]))
            design[arm] = dict(instances=have, range_given=given)
            if have != evid and not given:
                gaps.append(f'{size} {arm}: the plan covers instances {P.compress(have)}, the '
                            f'evidence instances are {P.compress(evid)}')
        if request and size in pol_tokens:
            pins = pol_tokens[size][1]
            design[POLICY_ARM] = dict(instances=list(evid if pins is None else pins),
                                      range_given=pins is not None)
        ii = {i: k for k, i in enumerate(inst)}
        n_desc = check_descriptors(a.data_root, size, combos, inst)
        rule, pol, fs = load_arrays(size, combos, inst, n_desc, seeds, stand_in, refusals, gaps)
        files += fs
        if not stand_in and sel is not None and sel not in pol:
            gaps.append(f'{size}: the policy arrays of the selected seed {sel} are incomplete')
        if not stand_in and len(pol) != len(seeds):
            gaps.append(f'{size}: policy arrays complete for seeds {sorted(pol) or "none"} '
                        f'of {seeds}')
        pol_sel = pol.get('stand-in' if stand_in else sel)
        A, A_present = {}, {}
        n, C = len(inst), len(combos)
        for c in here:
            if c['state'] != 'valid':
                continue
            i, r = c['idx'], c['combo']
            A.setdefault(c['arm'], np.full((len(grid), len(inst), len(combos)), np.nan))
            A[c['arm']][:, ii[i], ci[r]] = [np.nan if c['reading'].get(t) is None
                                            else c['reading'][t] for t in grid]
            records.append(dict(record=c['record'], arm=c['arm'], size=size, combination=r,
                                instance=i, budget_s=c['budget_s'],
                                search_start_wall_s=c['search_start_wall_s'],
                                build_s=c['build_s'], cpu_share=c['cpu_share'],
                                start_rule=c['rule_start'], start_policy=c['policy_start'],
                                final_makespan=c['final_makespan'],
                                at={f'{b:g}': c['reading'].get(float(b))
                                    for b in (size_budgets[size] if request else budgets)}))
            if request:
                records[-1]['offset_s'] = c.get('offset_s')
                A_present.setdefault(c['arm'], np.zeros((n, C), dtype=bool))
                A_present[c['arm']][ii[i], ci[r]] = c.get('offset_s') is not None
            want = rule[ii[i], ci[r]]
            if np.isfinite(want) and (c['start'] == 'rule' or c['runner'] == 'memetic'):
                if c['rule_start'] is None or abs(c['rule_start'] - want) > 1e-6:
                    refusals.append(f'{c["record"]}: rule start schedule {c["rule_start"]}, best '
                                    f'of the four rule arrays {want:g} (instance {i}, {r})')
            if c['start'] == 'policy' and pol_sel is not None and np.isfinite(pol_sel[ii[i], ci[r]]):
                want = pol_sel[ii[i], ci[r]]
                if c['policy_start'] is None or abs(c['policy_start'] - want) > 1e-6:
                    refusals.append(f'{c["record"]}: policy start schedule {c["policy_start"]}, '
                                    f'policy array {want:g} (instance {i}, {r})')
        all_pol = (np.stack([pol[s] for s in seeds if s in pol])
                   if (not stand_in and pol) else None)
        extra = {}
        if request:
            o_rule, o_pol, o_res = (np.full((n, C), np.nan) for _ in range(3))
            smp, smp_present, budgets_seen = np.full((len(grid), n, C), np.nan), \
                np.zeros((n, C), dtype=bool), set()
            for (s, r, i), v in new.items():
                if s != size or i not in ii or r not in ci:
                    continue
                k, j = ii[i], ci[r]
                if 'rule' in v:
                    o_rule[k, j] = v['rule']['t_all']
                    want = rule[k, j]
                    if np.isfinite(want) and abs(v['rule']['best_makespan'] - want) > 1e-6:
                        refusals.append(f'{v["rule"]["record"]}: best rule makespan '
                                        f'{v["rule"]["best_makespan"]:g}, best of the four rule '
                                        f'arrays {want:g} (instance {i}, {r})')
                if 'policy' in v:
                    p = v['policy']
                    o_pol[k, j], o_res[k, j] = p['t'], p['t_resident']
                    smp[:, k, j] = [np.nan if p['reading'][t] is None else p['reading'][t]
                                    for t in grid]
                    smp_present[k, j] = True
                    budgets_seen.add(p['budget_s'])
                    if pol_sel is not None and np.isfinite(pol_sel[k, j]) \
                            and abs(p['greedy_makespan'] - pol_sel[k, j]) > 1e-6:
                        refusals.append(f'{p["record"]}: greedy makespan {p["greedy_makespan"]:g}, '
                                        f'policy array of the selected run {pol_sel[k, j]:g} '
                                        f'(instance {i}, {r})')
            if len(budgets_seen) > 1:
                refusals.append(f'{size}: policy records with budgets {sorted(budgets_seen)}; '
                                'one budget per size')
            extra = dict(o_rule=o_rule, o_pol=o_pol, o_res=o_res, S=smp, S_present=smp_present,
                         policy_budget_s=(min(budgets_seen) if budgets_seen else None),
                         A_present=A_present, budgets=list(size_budgets[size]))
        data[size] = dict(instances=inst, arms=arms, n_descriptors=n_desc, A=A, rule=rule,
                          pol_sel=pol_sel, all_pol=all_pol,
                          seeds_with_arrays=[s for s in pol if s != 'stand-in'], design=design,
                          **extra)
        if MODEL_ARM in arms:
            # (x): the valid records of CP-SAT started from the rule schedule
            data[size]['model_cells'] = [
                dict(combo=c['combo'], idx=c['idx'], record=c['record'], **c['model'])
                for c in sorted(here, key=lambda c: (c['idx'], ci[c['combo']]))
                if c['arm'] == MODEL_ARM and c['state'] == 'valid']
    if refusals:
        raise Refusal(refusals)
    if gaps and not partial:
        raise Refusal(gaps)
    if request:
        records = dict(search=records, new=new_listing)
    return data, gaps, counts, sorted(set(files)), records, cells


def analyse(data, combos, budgets, B, alpha, grid, match_grid, curve_grid, policy_text,
            primary=PRIMARY):
    """The quantities (i) to (v), the verdicts of (i) at the stated budget `primary` (family:
    the eight combinations; none when the combinations are not the eight) and the curves of
    every size."""
    gi = {t: k for k, t in enumerate(grid)}
    eight = sorted(combos) == sorted(COMBOS)
    out = {}
    for size, D in data.items():
        A, rule, pol_sel, all_pol = D['A'], D['rule'], D['pol_sel'], D['all_pol']
        S = dict(instances=D['instances'], combinations=list(combos), arms=D['arms'],
                 arms_read=[x for x in P.ARMS if x in A], n_descriptors=D['n_descriptors'],
                 policy=policy_text if pol_sel is not None else None,
                 seeds_with_arrays=D['seeds_with_arrays'], i={}, ii={}, iii={}, iv={}, v={},
                 vi={}, verdict={})
        if D.get('design') is not None:
            S['design'] = D['design']           # the instances per arm, stated only
        methods = [m for m in METHOD_WORD if any(P.ARMS[x][0] == m for x in A)]
        for m in methods:
            ra, pa = ARM_OF[(m, 'rule')], ARM_OF[(m, 'policy')]
            for b in budgets:
                k = gi[float(b)]
                if ra in A and (pol_sel is not None or all_pol is not None):
                    blk = S['i'].setdefault(m, {}).setdefault(str(b), {})
                    if pol_sel is not None:
                        blk['selected'] = compare(pol_sel, A[ra][k], combos, B, alpha)
                    if all_pol is not None:
                        blk['all_seeds'] = compare(all_pol, A[ra][k], combos, B, alpha, seeds=True)
                    # (vi): the same pair in the order of the curves (arm minus policy), by
                    # the calls that write the curve rows, so both hold the same numbers.
                    rev = S['vi'].setdefault(m, {}).setdefault(str(b), {})
                    if pol_sel is not None:
                        rev['selected'] = compare(A[ra][k], pol_sel, combos, B, alpha, False)
                    if all_pol is not None:
                        rev['all_seeds'] = compare(A[ra][k], all_pol, combos, B, alpha, True)
                    if b == primary and eight:
                        vb = S['verdict'].setdefault(m, dict(budget_s=b))
                        if pol_sel is not None:
                            vb['selected'] = verdict_family(pol_sel, A[ra][k], combos, B, alpha)
                        if all_pol is not None:
                            vb['all_seeds'] = verdict_family(all_pol, A[ra][k], combos, B, alpha,
                                                             seeds=True)
                if pa in A and pol_sel is not None:
                    S['ii'].setdefault(m, {})[str(b)] = compare(A[pa][k], pol_sel, combos, B, alpha)
                if pa in A and ra in A:
                    S['iii'].setdefault(m, {})[str(b)] = compare(A[pa][k], A[ra][k], combos, B, alpha)
            if ra in A and pol_sel is not None:
                tr = A[ra][[gi[t] for t in match_grid]]
                S['v'][m] = dict(selected=match_block(tr, pol_sel, combos, match_grid))
                if all_pol is not None:
                    S['v'][m]['seed_mean'] = match_block(tr, all_pol.mean(axis=0), combos,
                                                         match_grid)
        if np.isfinite(rule).any():
            if pol_sel is not None:
                S['iv']['selected'] = compare(rule, pol_sel, combos, B, alpha)
            if all_pol is not None:
                S['iv']['all_seeds'] = compare(rule[None], all_pol, combos, B, alpha, seeds=True)
        curves, group_curves = [], []
        csv_group = {g[0]: g[3] for g in GROUPS}
        for arm in A:
            # Sel: the selected run, instance-level interval. All: every training run,
            # hierarchical interval, for the rule-started arms only (a policy-started arm ran
            # from the selected run's schedule and has no counterpart for the other runs).
            rows_of = ([('Sel', pol_sel, False)] if pol_sel is not None else []) + \
                      ([('All', all_pol, True)] if all_pol is not None and P.ARMS[arm][1] == 'rule'
                       else [])
            for rw, pol, seeds in rows_of:
                for t in curve_grid:
                    for r, row in compare(A[arm][gi[t]], pol, combos, B, alpha, seeds).items():
                        c = dict(row=rw, size=size, combination=r, method=arm,
                                 solver=P.ARMS[arm][0], start=P.ARMS[arm][1], t_s=t,
                                 mean_pct=row['mean'], lo_pct=row['lo'], hi_pct=row['hi'],
                                 n_instances=row['n'], n_combinations=row['c'],
                                 n_runs=row.get('k', 1))
                        if r in csv_group:      # a group's pooled row: its own rows of the CSV
                            group_curves.append(dict(c, group=csv_group[r]))
                        else:
                            curves.append(c)
        if D.get('model_cells') is not None:
            S['solver_model'] = solver_model(D['model_cells'], combos)
        S['curves'], S['group_curves'] = curves, group_curves
        out[size] = S
    return out


def analyse_request(data, combos, B, alpha, grid, match_grid, curve_grid, policy_text,
                    primary=PRIMARY):
    """The quantities on the request clock, per size at its own budgets: (i) to (iii) and
    (vi) at the time since the request (compare_at: a number only where both methods have a
    schedule on every cell), (iv) as on the other clocks (it compares the two start
    schedules and has no time), (v) against the greedy makespan and against the sampled
    curve, (vii) to (ix), the verdicts of (i) and (viii) at the stated budget and the
    curves (reference: the greedy makespan; each arm from the first grid time at which
    every cell of the arm has a schedule). Row All takes the offsets of the selected
    run's records."""
    gi = {t: k for k, t in enumerate(grid)}
    eight = sorted(combos) == sorted(COMBOS)
    mk = [gi[t] for t in match_grid]
    out = {}
    for size, D in data.items():
        A, Ap = D['A'], D['A_present']
        rule, pol_sel, all_pol = D['rule'], D['pol_sel'], D['all_pol']
        o_rule, o_pol, smp, sp = D['o_rule'], D['o_pol'], D['S'], D['S_present']
        sbud, budgets = D['policy_budget_s'], D['budgets']
        has_pol = np.isfinite(o_pol)
        pol_pres = has_pol & np.isfinite(pol_sel) if pol_sel is not None else has_pol & False

        def greedy_at(t, x):
            """The greedy schedule (pol_sel [n, C] or all_pol [k, n, C]) where it exists at t."""
            with np.errstate(invalid='ignore'):
                return np.where(o_pol <= t + EPS, x, np.nan)

        def held(t):
            return sbud is not None and float(t) > sbud + EPS

        S = dict(instances=D['instances'], combinations=list(combos), arms=D['arms'],
                 arms_read=[x for x in P.ARMS if x in A], n_descriptors=D['n_descriptors'],
                 policy=policy_text if pol_sel is not None else None,
                 seeds_with_arrays=D['seeds_with_arrays'], budgets=list(budgets),
                 policy_budget_s=sbud, policy_cells=int(sp.sum()), i={}, ii={}, iii={}, iv={},
                 v={}, vi={}, vii={}, viii={}, ix={}, verdict={}, verdict_viii={})
        if D.get('design') is not None:
            S['design'] = D['design']           # the instances per arm, stated only
        # the cells of every value of the decision table, as compare selects them:
        # {(column, budget or None): {entry: (instance rows, combinations)}}
        used = {}
        methods = [m for m in METHOD_WORD if any(P.ARMS[x][0] == m for x in A)]
        for m in methods:
            ra, pa = ARM_OF[(m, 'rule')], ARM_OF[(m, 'policy')]
            for b in budgets:
                t = float(b)
                k = gi[t]
                g_sel = greedy_at(t, pol_sel) if pol_sel is not None else None
                g_all = greedy_at(t, all_pol) if all_pol is not None else None
                if ra in A and (pol_sel is not None or all_pol is not None):
                    blk = S['i'].setdefault(m, {}).setdefault(str(b), {})
                    rev = S['vi'].setdefault(m, {}).setdefault(str(b), {})
                    if pol_sel is not None:
                        pres = Ap[ra] & pol_pres
                        blk['selected'] = compare_at(g_sel, A[ra][k], pres, combos, B, alpha)
                        rev['selected'] = compare_at(A[ra][k], g_sel, pres, combos, B, alpha,
                                                     used=used.setdefault((ra, str(b)), {}))
                    if all_pol is not None:
                        pres_all = Ap[ra] & has_pol
                        blk['all_seeds'] = compare_at(g_all, A[ra][k], pres_all, combos, B, alpha,
                                                      seeds=True)
                        rev['all_seeds'] = compare_at(A[ra][k], g_all, pres_all, combos, B, alpha,
                                                      seeds=True)
                    if b == primary and eight:
                        vb = S['verdict'].setdefault(m, dict(budget_s=b))
                        if pol_sel is not None:
                            vb['selected'] = verdict_at(g_sel, A[ra][k], pres, combos, B, alpha)
                        if all_pol is not None:
                            vb['all_seeds'] = verdict_at(g_all, A[ra][k], pres_all, combos, B,
                                                         alpha, seeds=True)
                if pa in A and pol_sel is not None:
                    S['ii'].setdefault(m, {})[str(b)] = compare_at(
                        A[pa][k], g_sel, Ap[pa] & pol_pres, combos, B, alpha,
                        used=used.setdefault((pa, str(b)), {}))
                if pa in A and ra in A:
                    S['iii'].setdefault(m, {})[str(b)] = compare_at(A[pa][k], A[ra][k],
                                                                    Ap[pa] & Ap[ra], combos, B,
                                                                    alpha)
                if ra in A and sp.any():
                    pres_s = sp & Ap[ra]
                    S['viii'].setdefault(m, {})[str(b)] = mark_held(
                        compare_at(smp[k], A[ra][k], pres_s, combos, B, alpha), held(t), sbud)
                    if b == primary and eight:
                        S['verdict_viii'][m] = dict(
                            budget_s=b, held=held(t), held_after_s=sbud,
                            selected=verdict_at(smp[k], A[ra][k], pres_s, combos, B, alpha))
            if ra in A and pol_sel is not None:
                tr = A[ra][mk]
                pres = Ap[ra] & pol_pres
                S['v'][m] = dict(selected=match_block_request(
                    tr, np.broadcast_to(pol_sel, tr.shape), pres, combos, match_grid))
                if all_pol is not None:
                    S['v'][m]['seed_mean'] = match_block_request(
                        tr, np.broadcast_to(all_pol.mean(axis=0), tr.shape), pres, combos,
                        match_grid)
                if sp.any():
                    S['v'][m]['sampled'] = match_block_request(tr, smp[mk], pres & sp, combos,
                                                               match_grid, held_after=sbud)
        if np.isfinite(rule).any():
            if pol_sel is not None:
                S['iv']['selected'] = compare(rule, pol_sel, combos, B, alpha,
                                              used=used.setdefault(('rules', None), {}))
            if all_pol is not None:
                S['iv']['all_seeds'] = compare(rule[None], all_pol, combos, B, alpha, seeds=True)
        if sp.any() and pol_sel is not None:
            for b in budgets:
                t = float(b)
                S['vii'][str(b)] = mark_held(compare_at(smp[gi[t]], greedy_at(t, pol_sel),
                                                        sp & pol_pres, combos, B, alpha,
                                                        used=used.setdefault(
                                                            ('policy_sample', str(b)), {})),
                                             held(t), sbud)
        S['ix'] = dict(rules=dict(t_all=first_stats(o_rule, np.isfinite(o_rule))),
                       policy=dict(t=first_stats(o_pol, has_pol),
                                   t_resident=first_stats(D['o_res'], np.isfinite(D['o_res']))))
        groups = [(key, members) for key, members in pools(combos) if key != POOL]
        for key, members in groups:
            # (ix) over the cells of the group's combinations
            g = np.zeros(o_rule.shape, dtype=bool)
            g[:, members] = True
            S['ix'][key] = dict(rules=dict(t_all=first_stats(o_rule, np.isfinite(o_rule) & g)),
                                policy=dict(t=first_stats(o_pol, has_pol & g),
                                            t_resident=first_stats(D['o_res'],
                                                                   np.isfinite(D['o_res']) & g)),
                                combinations=[combos[j] for j in members])
        # the decision table: values as the macros print them, and the bold columns
        S['decision'] = decision_table(S, o_rule, combos, used)
        # the same table on the instances common to every method of the size, when the design
        # gives the methods different instances: the comparisons of its columns on the cells
        # of those instances only, each statistic with its own fresh generator, so every other
        # number stays as it is
        design = D.get('design')
        want = column_design(D['instances'], design) if design is not None else {}
        common = common_instances(D['instances'], design) if design is not None else []
        if pol_sel is not None and any(v != common for v in want.values()):
            cr = np.isin(np.asarray(D['instances']), common)[:, None]          # [n, 1]
            Sc = dict(instances=D['instances'], budgets=list(budgets), iv={}, vii={}, vi={},
                      ii={})
            uc = {}
            if np.isfinite(rule).any():
                Sc['iv']['selected'] = compare(np.where(cr, rule, np.nan), pol_sel, combos, B,
                                               alpha, used=uc.setdefault(('rules', None), {}))
            for b in budgets:
                t = float(b)
                k, g_sel = gi[t], greedy_at(t, pol_sel)
                if sp.any():
                    Sc['vii'][str(b)] = mark_held(compare_at(
                        smp[k], g_sel, sp & pol_pres & cr, combos, B, alpha,
                        used=uc.setdefault(('policy_sample', str(b)), {})), held(t), sbud)
                for m in methods:
                    ra, pa = ARM_OF[(m, 'rule')], ARM_OF[(m, 'policy')]
                    if ra in A:
                        Sc['vi'].setdefault(m, {})[str(b)] = dict(selected=compare_at(
                            A[ra][k], g_sel, Ap[ra] & pol_pres & cr, combos, B, alpha,
                            used=uc.setdefault((ra, str(b)), {})))
                    if pa in A:
                        Sc['ii'].setdefault(m, {})[str(b)] = compare_at(
                            A[pa][k], g_sel, Ap[pa] & pol_pres & cr, combos, B, alpha,
                            used=uc.setdefault((pa, str(b)), {}))
            keep = {key for key, _ in pools(combos)}

            def pooled(x):
                return {key: v for key, v in x.items() if key in keep}
            S['decision_common'] = dict(
                instances=common, n_instances=len(common),
                rows=decision_table(Sc, np.where(cr, o_rule, np.nan), combos, uc),
                iv=dict(selected=pooled(Sc['iv'].get('selected', {}))),
                vii={b: pooled(x) for b, x in Sc['vii'].items()},
                vi={m: {b: dict(selected=pooled(x['selected'])) for b, x in blk.items()}
                    for m, blk in Sc['vi'].items()},
                ii={m: {b: pooled(x) for b, x in blk.items()} for m, blk in Sc['ii'].items()})
            # the rows of the first table: is each column a mean over its design cells
            mark_by_design(S['decision'], want)
        if D.get('model_cells') is not None:
            S['solver_model'] = solver_model(D['model_cells'], combos)
        curves, group_curves, starts, group_starts = [], [], {}, {}
        csv_group = {g[0]: g[3] for g in GROUPS}
        series = [(arm, A[arm], Ap[arm], P.ARMS[arm][0], P.ARMS[arm][1]) for arm in A]
        if sp.any():
            series.append((POLICY_ARM, smp, sp, 'policy', 'none'))

        def first_full(vals, pres):
            """The first grid time at which every cell of pres has a schedule, or None."""
            full = [t for t in curve_grid if pres.any() and np.isfinite(vals[gi[t]][pres]).all()]
            return full[0] if full else None

        for arm, vals, pres, solver, start in series:
            starts[arm] = first_full(vals, pres)
            # a group's rows start when every cell of its own combinations has a schedule
            g_start = {}
            for key, members in groups:
                gp = np.zeros_like(pres)
                gp[:, members] = pres[:, members]
                g_start[key] = first_full(vals, gp)
                group_starts.setdefault(key, {})[arm] = g_start[key]
            begin = [t for t in (starts[arm], *g_start.values()) if t is not None]
            if not begin:
                continue
            # Sel: the selected run, instance-level interval; All: every training run,
            # hierarchical interval, for the rule-started arms only (as on the other clocks)
            rows_of = ([('Sel', pol_sel, False)] if pol_sel is not None else []) + \
                      ([('All', all_pol, True)] if all_pol is not None and start == 'rule'
                       else [])
            for rw, pol, seeds in rows_of:
                for t in curve_grid:
                    if t < min(begin):
                        continue
                    v = np.where(pres, vals[gi[t]], np.nan)
                    for r, row in compare(v, pol, combos, B, alpha, seeds).items():
                        first = g_start.get(r) if r in csv_group else starts[arm]
                        if first is None or t < first:
                            continue
                        c = dict(row=rw, size=size, combination=r, method=arm, solver=solver,
                                 start=start, t_s=t, mean_pct=row['mean'], lo_pct=row['lo'],
                                 hi_pct=row['hi'], n_instances=row['n'],
                                 n_combinations=row['c'], n_runs=row.get('k', 1),
                                 held='yes' if arm == POLICY_ARM and held(t) else 'no')
                        if r in csv_group:      # a group's pooled row: its own rows of the CSV
                            group_curves.append(dict(c, group=csv_group[r]))
                        else:
                            curves.append(c)
        S['curve_start_s'] = starts
        if groups:
            S['group_curve_start_s'] = group_starts
        S['curves'], S['group_curves'] = curves, group_curves
        out[size] = S
    return out


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------

def macro_lines(R, info):
    """Header comments, then one \\newcommand per number, each with its comment."""
    L = ['% Generated by scripts/p94_timed_inference.py -- do not hand-edit',
         f'% {info["setting"]}; selected training run {info["selection_text"]}; CP-SAT '
         f'configuration {info["cpsat_arm"]}; B = {info["boot"]}, alpha = {info["alpha"]}; '
         f'instance-level intervals fresh default_rng({INST_SEED}) per quantity, hierarchical '
         f'intervals fresh default_rng({HIER_SEED}) per quantity.',
         f'% Clock origin {info["origin"]}: {ORIGIN_TEXT[info["origin"]]}.',
         '% Differences in %: first-named method minus second, in percent of the second '
         '(negative = first-named shorter); direction ahead = interval below zero, behind = '
         'above zero, neither otherwise (unrounded ends). Campaign: '
         + '; '.join(info['campaigns']) + '.',
         f'% Verdicts (names ending in Verdict, with the counts VerdictAheadN, VerdictBehindN, '
         f'VerdictNotDistinguishableN) only at the stated budget of {info["primary"]:g} s, for the '
         f'policy against each rule-started arm, one family of the eight combinations per size, '
         f'arm and row (alpha {info["alpha"]}). Row Sel, ' + VERDICT_TEXT['Sel'] + ' Row All, '
         + VERDICT_TEXT['All'] + ' Shorter budgets and the pooled rows carry interval and '
         'direction only.']
    if info.get('run', 'full') != 'full':
        L.insert(1, f'% Partial run ({info["run"]}, --partial-macros): built from the cells that '
                    f'exist, for a test build; not the macro file of record.')
    if info['origin'] == 'request':
        L.append(f'% Request clock: a number at a budget is written only when both compared '
                 f'methods have a schedule on every cell at that budget, otherwise its macro is '
                 f'absent. {SAMPLE_TEXT[0].upper() + SAMPLE_TEXT[1:]} ({SAMPLE_WORD}) holds its last '
                 f'value after the budget of its records (macro ...{SAMPLE_WORD}Budget); a macro '
                 f'that uses a held value says so in its comment. Verdicts of {SAMPLE_TEXT} '
                 f'against each rule-started arm by the rule of row Sel at the stated budget.')
    if GROUPS:
        L.append(f'% Pooled rows per group beside {POOL} (names with '
                 + ' or '.join(g[0] for g in GROUPS) + f' in the place of {POOL}): '
                 + '; '.join(f'{g[0]}, the {g[1]} ({", ".join(g[2])})' for g in GROUPS)
                 + f'. Each pools the instances present in every combination of its group, '
                 f'with the estimator and the generator rule of {POOL}, interval and direction '
                 f'only.')

    def nc(name, value, comment):
        L.append(f'\\newcommand{{\\{name}}}{{{value}}}  % {comment}')

    def rows(base, x, what, with_shares):
        if 'no_schedule_yet' in x:
            return                      # no schedule of one method on some cell: no number
        n = f'{x["n"]} instances x {x["c"]} combinations' + (f' x {x["k"]} runs' if 'k' in x else '')
        nc(base, f2(x['mean']), f'{what}: mean difference, {n}')
        nc(base + 'Lo', f2(x['lo']), f'{what}: lower end of the 95% interval ({x["scheme"]})')
        nc(base + 'Hi', f2(x['hi']), f'{what}: upper end of the 95% interval ({x["scheme"]})')
        nc(base + 'Dir', SF.TEX[x['direction']],
           f'{what}: direction of the interval [{x["lo"]:+.4f}, {x["hi"]:+.4f}]')
        if with_shares:
            for key, w in (('shorter', 'Shorter'), ('equal', 'Equal'), ('longer', 'Longer')):
                nc(base + w, f'{x[key]:.1f}',
                   f'{what}: percent of cells on which the first-named method is {key}'
                   + (' (mean over runs)' if 'k' in x else ''))

    sel = info['sel']
    nc('TmdBoot', info['boot'], 'bootstrap resamples')
    nc('TmdSeedN', info['k'], 'training runs of the policy')
    nc('TmdSelSeed', sel, f'selected training run (lowest validation score: {info["selection_text"]})')
    for size, S in R.items():
        sw = SIZE_WORD[size]
        nc(f'Tmd{sw}InstN', len(S['instances']),
           f'{size}: instances per combination read at this size, the union of the instances '
           f'of its arms ({P.compress(S["instances"])})')
        for arm, d in S.get('design', {}).items():
            nc(f'Tmd{sw}{ARM_WORD[arm]}InstN', len(d['instances']),
               f'{size}: instances per combination of the {ARM_TEXT[arm]} in the campaign '
               f'({P.compress(d["instances"])}, '
               + ('the range of its token' if d['range_given'] else
                  "the size's evidence instances, its token gives no range")
               + '); a count, the same on every clock')
        for m, blk in S['i'].items():
            for b, both in blk.items():
                for key, rw in (('selected', 'Sel'), ('all_seeds', 'All')):
                    who = f'selected run s{sel}' if key == 'selected' else 'every run'
                    for r, x in both.get(key, {}).items():
                        rows(f'Tmd{sw}PolVs{METHOD_WORD[m]}{number_word(int(b))}{r}{rw}', x,
                             f'{size}, {entry_text(r)}, {b} s: policy ({who}) minus the '
                             f'rule-started {METHOD_TEXT[m]}', key == 'selected')
        for m, vb in S['verdict'].items():
            b = vb['budget_s']
            for key, rw in (('selected', 'Sel'), ('all_seeds', 'All')):
                fam = vb.get(key)
                if fam is None or 'no_schedule_yet' in fam:
                    continue
                who = f'selected run s{sel}' if key == 'selected' else 'every run'
                base = f'Tmd{sw}PolVs{METHOD_WORD[m]}{number_word(int(b))}'
                for r, x in fam['rows'].items():
                    ps = (f'Holm p shorter {x["p_boot_less_holm"]:.2g}, Holm p longer '
                          f'{x["p_boot_greater_holm"]:.2g} (bootstrap)' if key == 'selected' else
                          f'Holm p shorter {x["p_t_less_holm"]:.2g} (t), {x["p_boot_less_holm"]:.2g} '
                          f'(bootstrap); Holm p longer {x["p_t_greater_holm"]:.2g} (t), '
                          f'{x["p_boot_greater_holm"]:.2g} (bootstrap)')
                    nc(f'{base}{r}{rw}Verdict', x['verdict'],
                       f'{size}, {r}, {b} s: verdict of the policy ({who}) against the '
                       f'rule-started {METHOD_TEXT[m]}; interval [{x["lo"]:+.2f}, {x["hi"]:+.2f}]; '
                       f'{ps}')
                for w in VERDICT_WORDS:
                    nc(f'{base}{rw}Verdict{COUNT_WORD[w]}', fam['counts'][w],
                       f'{size}, {b} s: combinations (of {len(fam["rows"])}) on which the policy '
                       f'({who}) is {w} against the rule-started {METHOD_TEXT[m]}')
        for m, blk in S['ii'].items():
            for b, rr in blk.items():
                for r, x in rr.items():
                    rows(f'Tmd{sw}{METHOD_WORD[m]}PolStartVsPol{number_word(int(b))}{r}', x,
                         f'{size}, {entry_text(r)}, {b} s: policy-started {METHOD_TEXT[m]} minus '
                         f'the policy schedule it started from (s{sel})', False)
        for m, blk in S['iii'].items():
            for b, rr in blk.items():
                for r, x in rr.items():
                    rows(f'Tmd{sw}{METHOD_WORD[m]}PolStartVsRuleStart{number_word(int(b))}{r}', x,
                         f'{size}, {entry_text(r)}, {b} s: policy-started minus rule-started '
                         f'{METHOD_TEXT[m]}', False)
        for key, rw in (('selected', 'Sel'), ('all_seeds', 'All')):
            who = f'selected run s{sel}' if key == 'selected' else 'every run'
            for r, x in S['iv'].get(key, {}).items():
                rows(f'Tmd{sw}RuleVsPol{r}{rw}', x,
                     f'{size}, {entry_text(r)}: best dispatching rule minus the policy ({who})',
                     True)
        for m, blk in S['vi'].items():
            for b, both in blk.items():
                for key, rw in (('selected', 'Sel'), ('all_seeds', 'All')):
                    who = f'selected run s{sel}' if key == 'selected' else 'every run'
                    for pk in (POOL,) + group_keys():       # the pooled rows only
                        x = both.get(key, {}).get(pk)
                        if x is not None:
                            rows(f'Tmd{sw}{METHOD_WORD[m]}RuleStartVsPol{number_word(int(b))}{pk}'
                                 f'{rw}', x, f'{size}, {entry_text(pk)}, {b} s: rule-started '
                                 f'{METHOD_TEXT[m]} minus the policy ({who}), the order of the '
                                 f'time-budget figure', False)
        for m, blk in S['v'].items():
            pre = f'Tmd{sw}{METHOD_WORD[m]}Match'
            for r, x in blk['selected'].items():
                nc(pre + 'Mean' + r, seconds_text(x['mean_s']),
                   f'{size}, {entry_text(r)}: seconds until the mean makespan of the rule-started '
                   f'{METHOD_TEXT[m]} is at or below the policy mean (s{sel}), one-second grid')
                nc(pre + 'Median' + r, seconds_text(x['median_s']),
                   f'{size}, {entry_text(r)}: median over {x["n_cells"]} cells of the seconds '
                   f'until the rule-started {METHOD_TEXT[m]} is at or below the policy (s{sel})')
                nc(pre + 'Never' + r, f'{x["never_pct"]:.1f}',
                   f'{size}, {entry_text(r)}: percent of cells on which the rule-started '
                   f'{METHOD_TEXT[m]} does not reach the policy within {info["horizon"]:g} s')
        if 'vii' in S:
            request_macros(S, size, sw, sel, info, nc, rows)
        if 'solver_model' in S:
            model_macros(S['solver_model'], size, sw, nc)
    return L


def held_note(x, key='held'):
    """Comment text of a value that uses the value held after the policy record's budget."""
    return (f' (uses the value of {SAMPLE_TEXT} held after its budget of {x["held_after_s"]:g} s)'
            if x.get(key) else '')


def request_macros(S, size, sw, sel, info, nc, rows):
    """Macros of the request clock for one size: the policy records' budget, (ix), (vii),
    (viii) with its verdicts, and (v) against the sampled curve."""
    if S['policy_budget_s'] is not None:
        nc(f'Tmd{sw}{SAMPLE_WORD}Budget', f'{S["policy_budget_s"]:g}',
           f'{size}: budget in seconds of the policy records ({SAMPLE_TEXT}); after it the last '
           f'value is held')
    # (ix) over every cell, then per group (the name ends in the group, after the statistic)
    for gk in ('',) + group_keys():
        ix = S['ix'] if not gk else S['ix'].get(gk, {})
        where = f'{size}, {entry_text(gk)}' if gk else size
        for grp, fld, word, what in (
                ('rules', 't_all', 'FirstRule', 'all four dispatching-rule schedules exist (t_all '
                 'of the rule record, from the start of its process)'),
                ('policy', 't', 'FirstPol', "the policy's greedy schedule exists (t of the policy "
                 'record, from the start of its process)'),
                ('policy', 't_resident', 'FirstPolResident', "the policy's greedy schedule "
                 'exists, from the moment its worker had the policy loaded (t_resident)')):
            st = ix.get(grp, {}).get(fld)
            if st is None:
                continue
            for key, w, text in (('median_s', 'Median', 'median'), ('min_s', 'Min', 'minimum'),
                                 ('max_s', 'Max', 'maximum')):
                nc(f'Tmd{sw}{word}{w}{gk}', f'{st[key]:.2f}',
                   f'{where}: {text} over {st["n_cells"]} cells of the seconds until {what}')
    for b, rr in S['vii'].items():
        for r, x in rr.items():
            rows(f'Tmd{sw}{SAMPLE_WORD}VsPol{number_word(int(b))}{r}', x,
                 f'{size}, {entry_text(r)}, {b} s since the request: {SAMPLE_TEXT} minus the '
                 f'greedy schedule (s{sel}){held_note(x)}', True)
    for m, blk in S['viii'].items():
        for b, rr in blk.items():
            for r, x in rr.items():
                rows(f'Tmd{sw}{SAMPLE_WORD}Vs{METHOD_WORD[m]}{number_word(int(b))}{r}Sel', x,
                     f'{size}, {entry_text(r)}, {b} s since the request: {SAMPLE_TEXT} (s{sel}) '
                     f'minus the rule-started {METHOD_TEXT[m]}{held_note(x)}', True)
    for m, vb in S['verdict_viii'].items():
        fam, b = vb['selected'], vb['budget_s']
        if fam is None or 'no_schedule_yet' in fam:
            continue
        base = f'Tmd{sw}{SAMPLE_WORD}Vs{METHOD_WORD[m]}{number_word(int(b))}'
        for r, x in fam['rows'].items():
            nc(f'{base}{r}SelVerdict', x['verdict'],
               f'{size}, {r}, {b} s since the request: verdict of {SAMPLE_TEXT} (s{sel}) against '
               f'the rule-started {METHOD_TEXT[m]}; interval [{x["lo"]:+.2f}, {x["hi"]:+.2f}]; '
               f'Holm p shorter {x["p_boot_less_holm"]:.2g}, Holm p longer '
               f'{x["p_boot_greater_holm"]:.2g} (bootstrap){held_note(vb)}')
        for w in VERDICT_WORDS:
            nc(f'{base}SelVerdict{COUNT_WORD[w]}', fam['counts'][w],
               f'{size}, {b} s since the request: combinations (of {len(fam["rows"])}) on which '
               f'{SAMPLE_TEXT} (s{sel}) is {w} against the rule-started {METHOD_TEXT[m]}'
               f'{held_note(vb)}')
    for m, blk in S['v'].items():
        if 'sampled' not in blk:
            continue
        pre = f'Tmd{sw}{METHOD_WORD[m]}MatchSmp'
        for r, x in blk['sampled'].items():
            nc(pre + 'Mean' + r, seconds_text(x['mean_s']),
               f'{size}, {entry_text(r)}: seconds since the request until the mean makespan of '
               f'the rule-started {METHOD_TEXT[m]} is at or below the mean of {SAMPLE_TEXT} at the '
               f'same time (s{sel}), one-second grid{held_note(x, "mean_held")}')
            nc(pre + 'Median' + r, seconds_text(x['median_s']),
               f'{size}, {entry_text(r)}: median over {x["n_cells"]} cells of the seconds since the '
               f'request until the rule-started {METHOD_TEXT[m]} is at or below {SAMPLE_TEXT} at '
               f'the same time (s{sel}){held_note(x, "median_held")}')
            nc(pre + 'Never' + r, f'{x["never_pct"]:.1f}',
               f'{size}, {entry_text(r)}: percent of cells on which the rule-started '
               f'{METHOD_TEXT[m]} does not reach {SAMPLE_TEXT} within {info["horizon"]:g} s'
               f'{held_note(x, "never_held")}')
    # the bold markers of the decision table, one per row and column
    for b, rows in S.get('decision', {}).items():
        bw = number_word(int(b))
        for key, row in rows.items():
            names = decision_names(sw, bw, key)
            shown = '; '.join(f'{c} {row["values"][c] or "no number"}' for c, _ in DECISION_COLUMNS)
            for col, text in DECISION_COLUMNS:
                nc(names[col][1], BOLD if col in row['bold'] else '',
                   f'{size}, {entry_text(key)}, {b} s since the request: bold marker of '
                   f'\\{names[col][0]} ({text}), \\bfseries when its printed value is the lowest '
                   f'of the row, empty otherwise (row: {shown})'
                   + ('' if row.get('by_design') is None or row.get('same_instances') is not False
                      else '; the columns of this row are means over different instances, '
                      + ('each over the instances of its design' if row['by_design'] else
                         'not each over the instances of its design')
                      + f'; the table on the common instances is Tmd{sw}{COMMON_WORD}...'))
    if S.get('decision_common') is not None:
        common_macros(S, size, sw, sel, nc)


def common_macros(S, size, sw, sel, nc):
    """Macros of the decision table on the instances common to every method of a size: the
    number of these instances, the value of each column with a number (the mean only, as the
    value macros of the first table print it) and the bold marker of every row and column.
    Names: Common after the size word, otherwise those of decision_names."""
    dc = S['decision_common']
    cw = sw + COMMON_WORD
    on = (f'on the {dc["n_instances"]} instances common to every method of the size '
          f'({P.compress(dc["instances"])})')
    nc(f'Tmd{cw}InstN', dc['n_instances'],
       f'{size}: instances per combination common to every method of the size '
       f'({P.compress(dc["instances"])}), the instances of the decision table on the common '
       f'instances; a count, the same on every clock')
    text = dict(DECISION_COLUMNS)
    keys = [key for key, _ in pools(S['combinations'])]

    def value(name, x, what):
        if x is not None and 'mean' in x:
            nc(name, f2(x['mean']),
               f'{what}, {on}: mean difference from the greedy schedule of the selected run '
               f's{sel}, in percent of it, {x["n"]} instances x {x["c"]} combinations'
               f'{held_note(x)}')
    for key in keys:
        value(decision_names(cw, '', key)['rules'][0], dc['iv'].get('selected', {}).get(key),
              f'{size}, {entry_text(key)}: {text["rules"]}, one value at every time since the '
              f'request (a row has it from the time all rule schedules of the entry exist)')
    for b in S['budgets']:
        bw = number_word(int(b))
        for key in keys:
            names = decision_names(cw, bw, key)
            src = {POLICY_ARM: dc['vii'].get(str(b), {}).get(key)}
            for m in METHOD_WORD:
                src[ARM_OF[(m, 'rule')]] = \
                    dc['vi'].get(m, {}).get(str(b), {}).get('selected', {}).get(key)
                src[ARM_OF[(m, 'policy')]] = dc['ii'].get(m, {}).get(str(b), {}).get(key)
            for col, _ in DECISION_COLUMNS[1:]:
                value(names[col][0], src[col],
                      f'{size}, {entry_text(key)}, {b} s since the request: {text[col]}')
    for b, rows in dc['rows'].items():
        bw = number_word(int(b))
        for key, row in rows.items():
            names = decision_names(cw, bw, key)
            shown = '; '.join(f'{c} {row["values"][c] or "no number"}' for c, _ in DECISION_COLUMNS)
            for col, t in DECISION_COLUMNS:
                nc(names[col][1], BOLD if col in row['bold'] else '',
                   f'{size}, {entry_text(key)}, {b} s since the request, {on}: bold marker of '
                   f'\\{names[col][0]} ({t}), \\bfseries when its printed value is the lowest of '
                   f'the row, empty otherwise (row: {shown})')


def model_macros(sm, size, sw, nc):
    """Macros of (x) for one size: per entry the number of cells, the median, minimum and
    maximum of every field with a value, and the cells without a variable count."""
    for r, x in sm['entries'].items():
        where = f'{size}, {entry_text(r)}'
        nc(f'Tmd{sw}CpModelCells{r}', x['n_cells'],
           f'{where}: valid records of CP-SAT started from the rule schedule')
        for key, word, what, text in MODEL_FIELDS:
            st = x[key]
            if st is None:
                continue                # no record of the entry states this value
            for k, w, name in (('median', 'Median', 'median'), ('min', 'Min', 'minimum'),
                               ('max', 'Max', 'maximum')):
                nc(f'Tmd{sw}Cp{word}{w}{r}', text(st[k]),
                   f'{where}: {name} over {st["n_cells"]} of {x["n_cells"]} cells of the {what} '
                   f'(CP-SAT started from the rule schedule)')
        nc(f'Tmd{sw}CpVarsMissing{r}', x['variables_missing'],
           f'{where}: cells whose record states no variable count of the initial model')


def macro_names(lines):
    return re.findall(r'\\newcommand\{\\([^}]*)\}', '\n'.join(lines))


def check_macro_names(names, out_mac):
    """Every name Tmd<letters>, unique, and unknown to the other macro files (the file of
    the same name under manuscript/ is this script's own output and is not compared)."""
    bad = [n for n in names if not re.fullmatch(r'Tmd[A-Za-z]+', n)]
    if bad:
        raise Refusal([f'macro names not of the form Tmd<letters>: {bad[:5]}'])
    if len(names) != len(set(names)):
        seen, dup = set(), set()
        for n in names:
            (dup if n in seen else seen).add(n)
        raise Refusal([f'duplicate macro names: {sorted(dup)[:5]}'])
    own = os.path.basename(out_mac)
    for f in glob.glob(os.path.join(ROOT, 'manuscript', 'macros*.tex')) + \
            glob.glob(os.path.join(ROOT, 'manuscript', 'sections_new', 'macros*.tex')) + \
            glob.glob(os.path.join(ROOT, 'paper', 'macros*.tex')):
        # the file of this script itself, at the output path or as the copy that an
        # earlier run left in the repository while this run writes under another root
        if os.path.basename(f) == own and os.path.basename(os.path.dirname(f)) == 'manuscript':
            continue
        clash = set(names) & set(re.findall(r'\\newcommand\{\\([A-Za-z]+)\}', open(f).read()))
        if clash:
            raise Refusal([f'macro names collide with {rel(f)}: {sorted(clash)[:5]}'])


def _cell(x):
    if 'no_schedule_yet' in x:
        return f'{NO_SCHEDULE} ({x["no_schedule_yet"]} of {x["n_cells"]} cells)'
    return (f'{x["mean"]:+.2f} [{x["lo"]:+.2f}, {x["hi"]:+.2f}] {x["direction"]}'
            + (' (held)' if x.get('held') else ''))


def markdown(R, info, res):
    tag = f', tag {info["tag"]}' if info.get('tag') else ''
    md = [f'# Timed campaign against the policy ({info["run"]}{tag})', '',
          f'- setting: {info["setting"]}; policy: {info["policy_text"]}; CP-SAT configuration: '
          f'{info["cpsat_arm"]}',
          f'- clock origin: {info["origin"]} ({ORIGIN_TEXT[info["origin"]]})',
          f'- budgets {", ".join(f"{b:g}" for b in info["budgets"])} s; verdict words at the '
          f'stated budget of {info["primary"]:g} s only; B = {info["boot"]}; '
          f'differences are first-named minus second in percent of the second; the interval '
          f'of a selected run is the instance-level cluster bootstrap, of every run the crossed '
          f'hierarchical bootstrap',
          f'- campaign: {"; ".join(info["campaigns"])}'] + \
        ([f'- instances per size and arm (the instance range of a token is the design of its '
          f'arm; a token without one plans the size\'s evidence instances): ' + '; '.join(
              f'{size} ' + ', '.join(f'{arm} {P.compress(d["instances"])} '
                                     f'({len(d["instances"])})' for arm, d in S['design'].items())
              for size, S in R.items() if S.get('design'))]
         if any(S.get('design') for S in R.values()) else []) + \
        [f'- cells {res["cells_total"]}, valid {res["cells_valid"]}'] + \
        ([f'- pooled rows beside {POOL}: '
          + '; '.join(f'the {g[1]} ({", ".join(g[2])})' for g in GROUPS)
          + f', each over the instances present in every combination of its group, with the '
          f'estimator of {POOL}'] if GROUPS else []) + ['']
    if res['gaps']:
        md += [f'## Not available ({len(res["gaps"])})', ''] + [f'- {g}' for g in res['gaps'][:200]]
        if len(res['gaps']) > 200:
            md.append(f'- ... and {len(res["gaps"]) - 200} more (JSON key gaps)')
        md.append('')
    if info['origin'] == 'request':
        md[5:5] = [f'- request clock: budgets per size ' + '; '.join(
            f'{s} {", ".join(f"{b:g}" for b in S["budgets"])} s' for s, S in R.items())
            + f'; a number needs a schedule of both compared methods on every cell, else '
            f'"{NO_SCHEDULE}" with the count of such cells; (held) marks a value of '
            f'{SAMPLE_TEXT} held after the budget of its records; valid records of the rule arm '
            f'{res["new_valid"][RULE_ARM]}, of the policy arm {res["new_valid"][POLICY_ARM]}']
    for size, S in R.items():
        bud = S.get('budgets', info['budgets'])
        heads = [f'{b:g} s' for b in bud]
        md += [f'## {size} (instances {P.compress(S["instances"])}; policy: {S["policy"]})', '']
        if not S['arms_read']:
            md += ['No valid record.', '']
        for title, blkname, key in (
                ('(i) policy minus rule-started arm, selected run', 'i', 'selected'),
                ('(i) policy minus rule-started arm, every run', 'i', 'all_seeds'),
                ('(ii) policy-started arm minus policy', 'ii', None),
                ('(iii) policy-started minus rule-started arm', 'iii', None)):
            for m, blk in S[blkname].items():
                tab = {b: (v.get(key, {}) if key else v) for b, v in blk.items()}
                if not any(tab.values()):
                    continue
                md += [f'### {title}: {METHOD_TEXT[m]}', '',
                       '| comb. | ' + ' | '.join(heads) + ' |', '|---|' + '---|' * len(heads)]
                for r in list(S['combinations']) + [POOL] + list(group_keys()):
                    if not any(r in tab.get(str(b), {}) for b in bud):
                        continue
                    md.append(f'| {entry_text(r)} | ' + ' | '.join(
                        _cell(tab[str(b)][r]) if r in tab.get(str(b), {}) else '-'
                        for b in bud) + ' |')
                md.append('')
        for m, vb in S['verdict'].items():
            md += [f'### (i) verdict at the stated budget of {vb["budget_s"]:g} s: policy against '
                   f'the rule-started {METHOD_TEXT[m]}', '']
            for key, rw in (('selected', 'Sel'), ('all_seeds', 'All')):
                if key not in vb:
                    continue
                fam = vb[key]
                if fam is None:
                    md += [f'Row {rw}: no verdict, a combination has no instance.', '']
                    continue
                if 'no_schedule_yet' in fam:
                    md += [f'Row {rw}: no verdict, {NO_SCHEDULE} on {fam["no_schedule_yet"]} of '
                           f'{fam["n_cells"]} cells.', '']
                    continue
                kinds = '(bootstrap)' if key == 'selected' else '(t / bootstrap)'
                md += [f'Row {rw} ({fam["scheme"]}). {fam["rule"]}', '',
                       f'| comb. | difference | 95% interval | Holm p shorter {kinds} | '
                       f'Holm p longer {kinds} | verdict |', '|---|---|---|---|---|---|']
                for r, x in fam['rows'].items():
                    ps = ([x['p_boot_less_holm']], [x['p_boot_greater_holm']]) if key == 'selected' \
                        else ([x['p_t_less_holm'], x['p_boot_less_holm']],
                              [x['p_t_greater_holm'], x['p_boot_greater_holm']])
                    md.append(f'| {r} | {x["mean"]:+.2f} | [{x["lo"]:+.2f}, {x["hi"]:+.2f}] | '
                              + ' | '.join(' / '.join(f'{p:.2g}' for p in side) for side in ps)
                              + f' | {x["verdict"]} |')
                md += ['', f'Row {rw}, combinations of {len(fam["rows"])}: '
                       + ', '.join(f'{w} {fam["counts"][w]}' for w in VERDICT_WORDS), '']
        for key in ('selected', 'all_seeds'):
            tab = S['iv'].get(key, {})
            if tab:
                md += [f'### (iv) best rule minus policy ({key})', '',
                       '| comb. | difference | shorter % | equal % | longer % | n |',
                       '|---|---|---|---|---|---|']
                md += [f'| {entry_text(r)} | {_cell(x)} | {x["shorter"]:.1f} | {x["equal"]:.1f} | '
                       f'{x["longer"]:.1f} | {x["n"]} |' for r, x in tab.items()]
                md.append('')
        if 'vii' in S:
            md += markdown_request(S, info, bud, heads)
        if 'decision' in S:
            md += markdown_decision(S)
        if S.get('decision_common') is not None:
            dc = S['decision_common']
            md += markdown_decision(
                S, dc['rows'],
                f'### Decision table on the instances common to every method of the size '
                f'({P.compress(dc["instances"])}, {dc["n_instances"]} instances): difference from '
                f'the greedy schedule in % (row Sel); bold = the lowest printed value of the row; '
                f'(n) = instances of the value')
        if 'solver_model' in S:
            md += markdown_model(S['solver_model'])
        for m, blk in (S['v'].items() if 'vii' not in S else ()):
            md += [f'### (v) rule-started {METHOD_TEXT[m]} reaches the policy (selected run)', '',
                   '| comb. | mean reached at s | median cell s | never % | arm mean at 0 | '
                   'arm mean at end | policy mean |', '|---|---|---|---|---|---|---|']
            for r, x in blk['selected'].items():
                md.append(f'| {entry_text(r)} | {seconds_text(x["mean_s"])} | '
                          f'{seconds_text(x["median_s"])} | '
                          f'{x["never_pct"]:.1f} | {x["arm_mean_at_zero"]:.2f} | '
                          f'{x["arm_mean_at_end"]:.2f} | {x["policy_mean"]:.2f} |')
            md.append('')
    md += ['## Record counts per cell', '', '| arm, size, combination | states |', '|---|---|']
    md += [f'| {k} | {", ".join(f"{s} {n}" for s, n in sorted(v.items()))} |'
           for k, v in sorted(res['counts'].items())]
    return '\n'.join(md) + '\n'


def markdown_request(S, info, bud, heads):
    """Markdown of the request clock for one size: (vii), (viii) and its verdicts, (v)
    against the greedy makespan and against the sampled curve, (ix) and the curve starts."""
    md = []
    sb = S['policy_budget_s']
    md += [f'### {SAMPLE_TEXT[0].upper() + SAMPLE_TEXT[1:]}: records of {S["policy_cells"]} cells, '
           f'budget {seconds_text(sb) if sb is not None else "-"} s (values after it are held)', '']
    for title, tab in ([(f'(vii) {SAMPLE_TEXT} minus the greedy schedule', S['vii'])]
                       + [(f'(viii) {SAMPLE_TEXT} minus the rule-started {METHOD_TEXT[m]}', blk)
                          for m, blk in S['viii'].items()]):
        if not tab:
            continue
        md += [f'### {title}', '', '| comb. | ' + ' | '.join(heads) + ' |',
               '|---|' + '---|' * len(heads)]
        for r in list(S['combinations']) + [POOL] + list(group_keys()):
            if not any(r in tab.get(str(b), {}) for b in bud):
                continue
            md.append(f'| {entry_text(r)} | ' + ' | '.join(
                _cell(tab[str(b)][r]) if r in tab.get(str(b), {}) else '-' for b in bud) + ' |')
        md.append('')
    for m, vb in S['verdict_viii'].items():
        fam = vb['selected']
        md += [f'### (viii) verdict at the stated budget of {vb["budget_s"]:g} s: {SAMPLE_TEXT} '
               f'against the rule-started {METHOD_TEXT[m]}'
               + (' (held)' if vb['held'] else ''), '']
        if fam is None:
            md += ['Row Sel: no verdict, a combination has no instance.', '']
            continue
        if 'no_schedule_yet' in fam:
            md += [f'Row Sel: no verdict, {NO_SCHEDULE} on {fam["no_schedule_yet"]} of '
                   f'{fam["n_cells"]} cells.', '']
            continue
        md += [f'Row Sel ({fam["scheme"]}). {fam["rule"]}', '',
               '| comb. | difference | 95% interval | Holm p shorter (bootstrap) | '
               'Holm p longer (bootstrap) | verdict |', '|---|---|---|---|---|---|']
        md += [f'| {r} | {x["mean"]:+.2f} | [{x["lo"]:+.2f}, {x["hi"]:+.2f}] | '
               f'{x["p_boot_less_holm"]:.2g} | {x["p_boot_greater_holm"]:.2g} | {x["verdict"]} |'
               for r, x in fam['rows'].items()]
        md += ['', f'Row Sel, combinations of {len(fam["rows"])}: '
               + ', '.join(f'{w} {fam["counts"][w]}' for w in VERDICT_WORDS), '']
    for m, blk in S['v'].items():
        for key, what in (('selected', 'the greedy makespan of the selected run'),
                          ('sampled', f'{SAMPLE_TEXT} at the same time')):
            if key not in blk:
                continue
            md += [f'### (v) rule-started {METHOD_TEXT[m]} reaches {what}', '',
                   '| comb. | mean reached at s | median cell s | never % | all cells from s | '
                   'arm mean then | arm mean at end | policy mean at end |',
                   '|---|---|---|---|---|---|---|---|']
            for r, x in blk[key].items():
                mean = seconds_text(x['mean_s']) + (' (held)' if x.get('mean_held') else '')
                med = seconds_text(x['median_s']) + (' (held)' if x.get('median_held') else '')
                never = f'{x["never_pct"]:.1f}' + (' (held)' if x.get('never_held') else '')
                then = ('-' if x['arm_mean_at_first'] is None
                        else f'{x["arm_mean_at_first"]:.2f}')
                md.append(f'| {entry_text(r)} | {mean} | {med} | {never} | '
                          f'{seconds_text(x["first_full_s"])} | {then} | '
                          f'{x["arm_mean_at_end"]:.2f} | {x["policy_mean"]:.2f} |')
            md.append('')
    md += ['### (ix) time of the first schedule (seconds since the request)', '',
           '| method | time | median | minimum | maximum | cells |', '|---|---|---|---|---|---|']
    for gk in ('',) + group_keys():
        ix = S['ix'] if not gk else S['ix'].get(gk)
        if ix is None:
            continue
        for grp, fld, what in (('rules', 't_all', 'dispatching rules (best of four)'),
                               ('policy', 't', 'policy, greedy schedule'),
                               ('policy', 't_resident', 'policy, greedy schedule, from loaded')):
            st = ix[grp][fld]
            md.append(f'| {what}' + (f', {entry_text(gk)}' if gk else '') + f' | {fld} | '
                      + (' | '.join(f'{st[k]:.2f}' for k in ('median_s', 'min_s', 'max_s'))
                         + f' | {st["n_cells"]}' if st else '- | - | - | 0') + ' |')
    md += ['', '### Curves: first grid time at which every cell of the method has a schedule', '',
           '| method | from s |', '|---|---|']
    md += [f'| {arm} | {seconds_text(t)} |' for arm, t in S['curve_start_s'].items()]
    for gk, st in S.get('group_curve_start_s', {}).items():
        md += [f'| {arm}, {entry_text(gk)} | {seconds_text(t)} |' for arm, t in st.items()]
    md.append('')
    return md


def markdown_decision(S, table=None, title=None):
    """A decision table of one size (the first table by default), one line per row: the
    printed values, bold in bold, each with the number of instances it is a mean over, and
    whether the columns with a number share one set of instances and combinations (in the
    first table at a size with the table on the common instances: 'no, by design' when each
    column is a mean over the instances of its design)."""
    md = [title or ('### Decision table: difference from the greedy schedule in % (row Sel); '
                    'bold = the lowest printed value of the row; (n) = instances of the value'), '',
          '| s since the request | entry | ' + ' | '.join(t for _, t in DECISION_COLUMNS)
          + ' | lowest | same instances |', '|---|---|' + '---|' * (len(DECISION_COLUMNS) + 2)]
    for b, rows in (S['decision'] if table is None else table).items():
        for key, row in rows.items():
            n = row.get('n_instances', {})
            cells = [('-' if row['values'][c] is None else
                      (f'**{row["values"][c]}**' if c in row['bold'] else row['values'][c])
                      + (f' ({n[c]})' if n.get(c) is not None else ''))
                     for c, _ in DECISION_COLUMNS]
            same = row.get('same_instances')
            md.append(f'| {b} | {entry_text(key)} | ' + ' | '.join(cells) + ' | '
                      + (', '.join(row['bold']) or 'no number') + ' | '
                      + ('-' if same is None else 'yes' if same else
                         'no, by design' if row.get('by_design') else 'no') + ' |')
    return md + ['']


def markdown_model(sm):
    """(x) of one size: the statistics per entry, then the records without a value."""
    md = [f'### Size of the solver\'s model: {sm["n_cells"]} valid records of CP-SAT started '
          'from the rule schedule; median [minimum, maximum] over the cells', '',
          '| entry | cells | ' + ' | '.join(what for _, _, what, _ in MODEL_FIELDS) + ' |',
          '|---|---|' + '---|' * len(MODEL_FIELDS)]
    for r, x in sm['entries'].items():
        cells = []
        for key, _, _, text in MODEL_FIELDS:
            st = x[key]
            cells.append('-' if st is None else
                         f'{text(st["median"])} [{text(st["min"])}, {text(st["max"])}]'
                         .replace('{,}', ',') + (f' ({st["n_cells"]} cells)'
                                                 if st['n_cells'] != x['n_cells'] else ''))
        md.append(f'| {entry_text(r)} | {x["n_cells"]} | ' + ' | '.join(cells) + ' |')
    md.append('')
    for key, _, what, _ in MODEL_FIELDS:
        for c in sm['missing'][key]:
            md.append(f'- no value of the {what}: {c["record"]}'
                      + (f' ({c["reason"]})' if 'reason' in c else ''))
    return md + ([''] if any(sm['missing'].values()) else [])


# group: all on the rows per combination and of Pool, a group's CSV value on its pooled rows
CSV_COLUMNS = ('run', 'row', 'origin', 'policy', 'size', 'group', 'combination', 'method',
               'solver', 'start', 't_s', 'mean_pct', 'lo_pct', 'hi_pct', 'n_instances',
               'n_combinations', 'n_runs')
# request clock: whether the row uses a held value of the policy, sampled schedules; the
# budget of its records; the median first-schedule times of the policy and the rules
CSV_COLUMNS_REQUEST = CSV_COLUMNS + ('held', 'policy_budget_s', 'policy_first_s',
                                     'rule_first_s')


def _seconds_cell(st):
    return '' if st is None else f'{st["median_s"]:.6f}'


def write_csv(path, R, info):
    """The rows per combination and of Pool (group all) of every size, in the order of the
    curves, then the pooled rows of the groups of every size. On the request clock the
    first-schedule times of a group row are the medians over the group's cells."""
    request = info['origin'] == 'request'
    group_key = {g[3]: g[0] for g in GROUPS}
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS_REQUEST if request else CSV_COLUMNS)
        w.writeheader()
        for part in ('curves', 'group_curves'):
            for S in R.values():
                for row in S.get(part, ()):
                    grp = row.get('group', CSV_GROUP_ALL)
                    per_size = {}
                    if request:
                        ix = S['ix'] if grp == CSV_GROUP_ALL else S['ix'][group_key[grp]]
                        sb = S['policy_budget_s']
                        per_size = dict(policy_budget_s='' if sb is None else f'{sb:g}',
                                        policy_first_s=_seconds_cell(ix['policy']['t']),
                                        rule_first_s=_seconds_cell(ix['rules']['t_all']))
                    w.writerow(dict(row, run=info['run'], origin=info['origin'], group=grp,
                                    policy=info['policy_text'] if row['row'] == 'Sel' else
                                    info['all_text'], t_s=f'{row["t_s"]:g}',
                                    **{k: f'{row[k]:.6f}' for k in ('mean_pct', 'lo_pct',
                                                                    'hi_pct')},
                                    **per_size))


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------

def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--campaign', action='append', default=None,
                    help='"<sizes> = <arm tokens>" as launched, once per launch, plus '
                         f'{POLICY_ARM}@<budget>[:instances] on the request clock; default: '
                         + ' | '.join(DEFAULT_CAMPAIGNS_REQUEST) + ' (request), '
                         + ' | '.join(DEFAULT_CAMPAIGNS) + ' (start, search)')
    ap.add_argument('--budgets', nargs='+', type=int, default=None,
                    help=f'reported budgets; default {" ".join(map(str, BUDGETS))} (start, '
                         f'search), {" ".join(map(str, REQUEST_BUDGETS))} at the production sets '
                         f'(request)')
    ap.add_argument('--training-budgets', nargs='+', type=int, default=None,
                    help=f'request clock: reported budgets at {P.TRAINING_SIZE}; default '
                         f'{" ".join(map(str, TRAINING_BUDGETS))}')
    ap.add_argument('--primary', type=int, default=PRIMARY,
                    help='the stated budget, one of --budgets: the only budget with verdict words')
    ap.add_argument('--origin', choices=('request', 'start', 'search'), default='request')
    ap.add_argument('--policy-batch', type=int, default=None,
                    help=f'request clock: schedules per sampled batch of the policy records '
                         f'(p97 --batch; default {P97.DEFAULT_BATCH})')
    ap.add_argument('--policy-workers', type=int, default=None,
                    help=f'request clock: workers of the policy records (p97 --workers; default '
                         f'{P97.DEFAULT_WORKERS})')
    ap.add_argument('--policy-base-seed', type=int, default=None,
                    help=f'request clock: base seed of the policy records (p97 --base_seed; '
                         f'default {P97.DEFAULT_BASE_SEED})')
    ap.add_argument('--boot', type=int, default=SF.B_DEFAULT)
    ap.add_argument('--alpha', type=float, default=0.05)
    ap.add_argument('--combos', nargs='+', default=list(COMBOS))
    ap.add_argument('--cpsat-arm', default=None,
                    help="the pilot's choice by default; full mode refuses another one")
    # passed to the planner as given; an option left out keeps the planner's default
    ap.add_argument('--budget', type=float, default=None, help='default budget of a token')
    ap.add_argument('--slots', nargs='+', default=None)
    ap.add_argument('--rule-cache', default=None)
    ap.add_argument('--policy-cache', default=None)
    ap.add_argument('--result-root', default='')
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--memetic-seed', type=int, default=None)
    ap.add_argument('--cpsat-seed', type=int, default=None)
    ap.add_argument('--bars', default=None)
    ap.add_argument('--max-invalid', type=int, default=2)
    ap.add_argument('--data-root', default='data/FAMILY')
    ap.add_argument('--out-root', default=ROOT, help='full mode: root of manuscript/ and results/')
    ap.add_argument('--partial', action='store_true')
    ap.add_argument('--out', default=None, help='partial mode: the output folder')
    ap.add_argument('--rule-as-policy', action='store_true',
                    help='partial mode only: the best-rule schedule stands in for the policy')
    ap.add_argument('--record-policy-seed', type=int, default=None,
                    help='with --rule-as-policy: the seed in the folders of policy-started records')
    ap.add_argument('--cpsat-folder', default=None,
                    help="partial mode only: the CP-SAT output folder to read, e.g. 'pilot'")
    ap.add_argument('--partial-macros', action='store_true',
                    help='partial mode only: also write macros_timed.tex into --out, built from '
                         'the cells that exist, for a test build of a table')
    ap.add_argument('--tag', default=None,
                    help='full mode, start or search clock: write results/timed_inference_<tag>'
                         '.{json,md} and results/timed_curves_<tag>.csv and no macro file '
                         '(letters only)')
    return ap


def _inside(path, folder):
    path, folder = os.path.realpath(path), os.path.realpath(folder)
    return path == folder or path.startswith(folder + os.sep)


def check_options(a):
    bad = []
    if a.partial and not a.out:
        bad.append('--partial needs --out <folder>')
    if a.out and not a.partial:
        bad.append('--out is the folder of partial mode; give --partial')
    for opt, val in (('--rule-as-policy', a.rule_as_policy), ('--cpsat-folder', a.cpsat_folder),
                     ('--partial-macros', a.partial_macros)):
        if val and not a.partial:
            bad.append(f'{opt} is accepted only with --partial')
    if a.record_policy_seed is not None and not a.rule_as_policy:
        bad.append('--record-policy-seed is accepted only with --rule-as-policy')
    if a.tag is not None:
        if not TAG_PATTERN.fullmatch(a.tag):
            bad.append(f'--tag {a.tag!r}: letters only')
        if a.partial:
            bad.append('--tag names a record of the full mode; partial mode writes into --out')
        if a.origin == 'request':
            bad.append('--tag reads the start and search clocks only: the request clock writes '
                       'the macro file of record (give --origin start or search)')
    if a.out:
        banned = [os.path.join(ROOT, x) for x in RECORD_DIRS + ('results',)] + \
                 [p for p in glob.glob(os.path.join(ROOT, 'paper*')) if os.path.isdir(p)]
        for d in banned:
            if _inside(a.out, d):
                bad.append(f'--out {a.out} lies inside {rel(d)}/')
    if a.origin != 'request':
        for opt, val in (('--training-budgets', a.training_budgets),
                         ('--policy-batch', a.policy_batch), ('--policy-workers', a.policy_workers),
                         ('--policy-base-seed', a.policy_base_seed)):
            if val is not None:
                bad.append(f'{opt} is read on the request clock only (--origin request)')
    elif a.rule_as_policy:
        bad.append('--rule-as-policy reads the start and search clocks only: a stand-in policy '
                   'has no policy record (give --origin start or search)')
    for opt, val in (('--policy-batch', a.policy_batch), ('--policy-workers', a.policy_workers)):
        if val is not None and val < 1:
            bad.append(f'{opt} must be at least 1')
    if a.policy_base_seed is not None and a.policy_base_seed < 0:
        bad.append('--policy-base-seed must not be negative')
    if a.budgets is None:
        a.budgets = list(REQUEST_BUDGETS if a.origin == 'request' else BUDGETS)
    if a.origin == 'request' and a.training_budgets is None:
        a.training_budgets = list(TRAINING_BUDGETS)
    lists = [('--budgets', a.budgets)] + ([('--training-budgets', a.training_budgets)]
                                          if a.origin == 'request' else [])
    for opt, vals in lists:
        if any(b <= 0 for b in vals):
            bad.append('budgets must be positive whole seconds')
        if a.primary not in vals:
            bad.append(f'--primary {a.primary} is not one of {opt} {vals}')
    if not a.partial and sorted(a.combos) != sorted(COMBOS):
        bad.append('full mode reads the eight combinations (the family of the verdicts); '
                   '--combos is for partial mode')
    return bad


def output_paths(a):
    if a.partial:
        return {k: os.path.join(a.out, os.path.basename(p)) for k, p in OUTPUTS
                if k != 'macros' or a.partial_macros}
    if a.tag:
        return {k: os.path.join(a.out_root, p.format(tag=a.tag)) for k, p in TAG_OUTPUTS}
    return {k: os.path.join(a.out_root, p) for k, p in OUTPUTS}


def remove_outputs(paths):
    for p in paths.values():
        if os.path.exists(p):
            os.remove(p)
            print(f'removed {rel(p)} (refusing to certify)')


def refuse(paths, reasons, partial):
    if not partial:
        remove_outputs(paths)
    elif 'macros' in paths:
        # --partial-macros: no macro file of an earlier partial run is left for a test build
        remove_outputs({'macros': paths['macros']})
    print(f'REFUSING: {len(reasons)} problem(s)', file=sys.stderr)
    for r in reasons[:60]:
        print(f'  {r}', file=sys.stderr)
    if len(reasons) > 60:
        print(f'  ... and {len(reasons) - 60} more', file=sys.stderr)
    return 1


def main(argv=None):
    a = build_parser().parse_args(_ARGV_KEPT[1:] if argv is None else argv)
    if a.out:
        a.out = os.path.abspath(a.out)
    a.out_root = os.path.abspath(a.out_root)
    bad = check_options(a)
    if bad:
        print('REFUSING: ' + '; '.join(bad), file=sys.stderr)
        return 2
    os.environ.setdefault('PACT_ENV', 'exact')
    os.environ.setdefault('PACT_SEEDS', '301-305')
    paths = output_paths(a)
    # the macro file: full mode without a tag, or partial mode with --partial-macros
    writes_macros = (not a.partial and not a.tag) or a.partial_macros
    os.chdir(ROOT)
    if REG.env() != 'exact':
        return refuse(paths, [f'PACT_ENV={REG.env()}: this analysis reads the exact environment'],
                      a.partial)
    request = a.origin == 'request'
    if request:
        for opt, default in (('policy_batch', P97.DEFAULT_BATCH),
                             ('policy_workers', P97.DEFAULT_WORKERS),
                             ('policy_base_seed', P97.DEFAULT_BASE_SEED)):
            if getattr(a, opt) is None:
                setattr(a, opt, default)
    else:
        budgets = sorted(set(a.budgets))
        horizon = max(budgets)
        match_grid = [float(t) for t in range(0, int(horizon) + 1)]
        curve_grid = log_grid(horizon, extra=budgets)
        grid = sorted(set(match_grid) | set(curve_grid))
    stand_in = a.rule_as_policy
    try:
        specs = [parse_campaign(s) for s in
                 (a.campaign or (DEFAULT_CAMPAIGNS_REQUEST if request else DEFAULT_CAMPAIGNS))]
        size_budgets = None
        if request:
            # reported budgets per size; the grids cover every size's budgets
            size_budgets = {s: sorted(set(a.training_budgets if s == P.TRAINING_SIZE
                                          else a.budgets)) for sizes, _ in specs for s in sizes}
            budgets = sorted(set().union(*size_budgets.values()))
            horizon = max(budgets)
            match_grid = [float(t) for t in range(0, int(horizon) + 1)]
            # the budgets of the policy's sampling runs are curve points as well, so that
            # the sampled curve reaches the end of its run before its value is held
            pol_ends = []
            for _, toks in specs:
                for t in toks:
                    name, _, rest = t.partition('@')
                    if name == POLICY_ARM and rest.partition(':')[0].replace('.', '', 1).isdigit():
                        pol_ends.append(float(rest.partition(':')[0]))
            curve_grid = log_grid(horizon, extra=budgets + pol_ends)
            grid = sorted(set(match_grid) | set(curve_grid))
        if writes_macros:
            unknown = [s for sizes, _ in specs for s in sizes if s not in SIZE_WORD]
            if unknown:
                raise Refusal([f'sizes {unknown} have no macro word (known: {sorted(SIZE_WORD)})'])
        pre = []
        sel, sel_rec = None, {}
        if not stand_in:
            try:
                sel, sel_rec = select_policy_seed()
            except (REG.MissingResult, SystemExit) as e:
                pre.append(f'training-run selection unavailable: {e}')
        uses_cpsat = any(P.ARMS.get(t.partition('@')[0], ('',))[0] == 'cpsat'
                         for _, toks in specs for t in toks)
        cpsat_arm = a.cpsat_arm
        if uses_cpsat:
            tr_root = (os.path.join(a.result_root, 'test_results', 'FAMILY') if a.result_root
                       else 'test_results/FAMILY')
            slots = [P.parse_cores(s) for s in (a.slots or P.build_parser().parse_args(['table']).slots)]
            bars = a.bars or P.build_parser().parse_args(['table']).bars
            chosen, why = select_cpsat_arm(tr_root, slots, bars)
            if chosen is None:
                pre.append(f"pilot selection unavailable ({len(why)} problem(s), first: "
                           f"{why[0] if why else '-'})")
                if a.partial and not cpsat_arm:
                    raise Refusal(['CP-SAT arms need --cpsat-arm while the pilot selection is '
                                   'unavailable'])
            elif cpsat_arm and cpsat_arm != chosen:
                pre.append(f"--cpsat-arm {cpsat_arm} differs from the pilot's choice {chosen}")
            else:
                cpsat_arm = chosen
        if pre and not a.partial:
            raise Refusal(pre)
        data, gaps, counts, files, records, cells = collect(
            a, budgets, a.origin, specs, sel, cpsat_arm, stand_in, a.record_policy_seed,
            a.partial, grid, size_budgets=size_budgets)
        policy_text = ('stand-in: best of the four dispatching rules' if stand_in else
                       (f'selected training run s{sel}' if sel is not None else 'unavailable'))
        new_records = []
        if request:
            records, new_records = records['search'], records['new']
            R = analyse_request(data, list(a.combos), a.boot, a.alpha, grid, match_grid,
                                curve_grid, policy_text, primary=a.primary)
        else:
            R = analyse(data, list(a.combos), budgets, a.boot, a.alpha, grid, match_grid,
                        curve_grid, policy_text, primary=a.primary)
        res = dict(gaps=pre + gaps, counts=counts, cells_total=len(cells),
                   cells_valid=sum(c['state'] == 'valid' for c in cells.values()))
        if request:
            res['new_valid'] = {arm: sum(x['arm'] == arm for x in new_records)
                                for arm in (RULE_ARM, POLICY_ARM)}
        run = ('partial, stand-in policy' if stand_in else 'partial') if a.partial else 'full'
        info = dict(run=run, origin=a.origin, budgets=budgets, primary=a.primary, boot=a.boot,
                    alpha=a.alpha, setting=REG.describe(), cpsat_arm=cpsat_arm or '-',
                    campaigns=[f'{" ".join(s)} = {" ".join(t)}' for s, t in specs],
                    sel=sel, k=len(REG.seeds()), horizon=horizon, policy_text=policy_text,
                    all_text=('every training run (' + ', '.join(f's{s}' for s in REG.seeds())
                              + ')'),
                    selection_text=(', '.join(f's{s} {v[0]:.5f}' for s, v in sel_rec.items())
                                    if sel_rec else 'unavailable'), tag=a.tag)
        if not a.partial:
            # every column of a row of the decision table is a mean over the same instances
            mismatch = decision_mismatches(R)
            if mismatch:
                raise Refusal(mismatch)
        lines = None
        if writes_macros:
            lines = macro_lines(R, info)
            # a partial macro file shares its names with the macro file of record
            check_macro_names(macro_names(lines), paths['macros'] if not a.partial else
                              os.path.join(ROOT, dict(OUTPUTS)['macros']))
    except Refusal as e:
        return refuse(paths, list(e.args[0]), a.partial)
    for p in paths.values():
        os.makedirs(os.path.dirname(p) or '.', exist_ok=True)
    if lines is not None:
        emit(paths['macros'], lines)
    out = dict(generator='scripts/p94_timed_inference.py', run=run, partial=a.partial,
               stand_in=stand_in, setting=info['setting'], seeds=list(REG.seeds()),
               origin=a.origin, origin_text=ORIGIN_TEXT[a.origin], budgets=budgets,
               primary_budget_s=a.primary,
               verdicts=dict(budget_s=a.primary, comparison='(i) policy against each '
                             'rule-started arm, per size', family=list(COMBOS),
                             sel=VERDICT_TEXT['Sel'], all=VERDICT_TEXT['All'],
                             other='shorter budgets, the pooled rows and comparisons (ii) to (iv) '
                                   'carry interval and direction only'),
               boot=a.boot, alpha=a.alpha,
               streams=dict(instance=f'fresh default_rng({INST_SEED}) per quantity',
                            hierarchical=f'fresh default_rng({HIER_SEED}) per quantity'),
               sign='first-named method minus second, in percent of the second '
                    '(negative = first-named shorter)',
               selection=dict(rule='lowest validation score of the kept checkpoint '
                                   '(arm_registry.select_seed)', seed=sel,
                              records={f's{s}': dict(score=v[0], index=v[1])
                                       for s, v in sel_rec.items()}),
               policy=policy_text, cpsat_arm=cpsat_arm, cpsat_folder=a.cpsat_folder,
               campaigns=info['campaigns'], combinations=list(a.combos),
               grid=dict(match=match_grid, curve=curve_grid), matching_horizon_s=horizon,
               gaps=res['gaps'], counts=counts, cells_total=res['cells_total'],
               cells_valid=res['cells_valid'], arrays=files, records=records,
               results={s: {k: v for k, v in S.items() if k not in ('curves', 'group_curves')}
                        for s, S in R.items()})
    if a.tag:
        out['tag'] = a.tag
    if request:
        out['verdicts'].update(
            other='shorter budgets, the pooled rows and comparisons (ii) to (iv) and (vii) carry '
                  'interval and direction only',
            viii=(f'(viii) {SAMPLE_TEXT} against each rule-started arm, per size, row Sel: '
                  + VERDICT_TEXT['Sel']))
        out.update(
            budgets_per_size=size_budgets,
            request_rule=('a number at a budget needs a schedule of both compared methods on '
                          f'every cell; otherwise its entry is {{"no_schedule_yet": cells, '
                          f'"n_cells": cells}} and no macro is written'),
            held_rule=(f'{SAMPLE_TEXT} holds its last value after the budget of its records '
                       '(results.<size>.policy_budget_s); an entry with held true uses that value'),
            policy_sample=dict(arm=POLICY_ARM, method=SAMPLE_TEXT, batch=a.policy_batch,
                               workers=a.policy_workers, base_seed=a.policy_base_seed,
                               training_seed=sel,
                               folders={s: rel(P97.folder(a.result_root or ROOT, s, d['policy_budget_s'],
                                                          a.policy_batch, f's{sel}'))
                                        for s, d in data.items()
                                        if d.get('policy_budget_s') is not None},
                               validity='p97_policy_anytime.classify'),
            rule_start=dict(arm=RULE_ARM, workers=P97.RULE_WORKERS,
                            folders={s: rel(P97.rule_folder(a.result_root or ROOT, s))
                                     for s in data},
                            validity='p97_policy_anytime.classify_rules'),
            offsets=('o_rule = t_all of the rule record; o_pol = t of the greedy schedule in the '
                     'policy record of the selected run, for every training run'),
            policy_records=[x for x in new_records if x['arm'] == POLICY_ARM],
            rule_records=[x for x in new_records if x['arm'] == RULE_ARM],
            new_records_valid=res['new_valid'])
    if GROUPS:
        out['groups'] = dict(
            rule=(f'beside {POOL}, the pooled row of each group over its combinations that the '
                  f'campaign holds: the estimator, the clustering by base instance and the '
                  f'generator rule of {POOL} (a fresh generator per entry), interval and direction '
                  f'only; an entry lists the combinations that entered and, as {POOL}, needs two '
                  f'or more of them with numbers (a group with fewer in the campaign is absent, '
                  f'here too); on the request clock a group has a number when '
                  f'both methods have a schedule on every cell of its combinations; '
                  f'results.<size>.ix.<group> holds (ix) over the group\'s cells'),
            entries={g[0]: dict(text=g[1], combinations=[r for r in a.combos if r in g[2]],
                                defined=list(g[2]), csv_group=g[3])
                     for g in GROUPS if sum(r in g[2] for r in a.combos) >= 2},
            csv=(f'column group: {CSV_GROUP_ALL} on the rows per combination and of {POOL}; the '
                 'value csv_group on the pooled rows of a group (combination = the group\'s key), '
                 'after the rows of every size'))
    if any(S.get('design') for S in R.values()):
        out['design_rule'] = DESIGN_RULE
    if request:
        out['decision_table'] = dict(columns=dict(DECISION_COLUMNS), rule=DECISION_RULE)
        if any(S.get('decision_common') is not None for S in R.values()):
            out['decision_table']['common'] = DECISION_COMMON_RULE
    if any('solver_model' in S for S in R.values()):
        out['solver_model'] = dict(arm=MODEL_ARM, rule=MODEL_RULE,
                                   fields={key: what for key, _, what, _ in MODEL_FIELDS})
    with open(paths['json'], 'w') as f:
        json.dump(jsonable(out), f, indent=1)
    with open(paths['md'], 'w') as f:
        f.write(markdown(R, info, res))
    write_csv(paths['csv'], R, info)
    print(f'{run}: {res["cells_valid"]} of {res["cells_total"]} cells valid; '
          f'{len(res["gaps"])} not available; wrote ' + ', '.join(rel(p) for p in paths.values()))
    return 0


if __name__ == '__main__':
    sys.exit(main())
