"""Result figures of the manuscript, drawn at the size they print.

fq1_specialists
    One policy, trained on six rule combinations, against the specialist of each
    of the eight combinations, in two benchmark families: (a) the factory
    benchmark (10 modules on 25 stations) and (b) public instances (SD2, 10 jobs
    on 5 machines). Per row: the paired makespan difference, policy minus
    specialist, in percent of the specialist's makespan, with its 95% interval;
    the last row pools the eight combinations. The 1% margin is a pale band, the
    0.5% margin two dashed lines, zero a solid line, and the verdict at the 1%
    margin is the marker (filled circle: equivalent; open square: equivalence not
    shown). Variant A draws the policy only. Variant B adds, on every row the data
    hold, a lighter interval for the single-rule control (trained on N, L, S and
    W) against the same specialist.

fq3_time_budget
    When a search method is the better tool, at the two production batches: (a) 50
    modules and (b) 80 modules on 25 stations. Per search arm, the makespan of the
    search method minus the makespan of the policy's schedule, in percent of the
    policy's makespan, pooled over the eight combinations, against time on a
    logarithmic axis, with the 95% interval as a pale band. Colour is the method
    (memetic algorithm, CP-SAT solver), line style the start schedule (solid: the
    best dispatching rule; dashed: the policy's schedule). Zero, a thin solid line,
    is the policy. The clock is the file's own (column origin of the CSV):
    request  time since the request for a schedule, the default of p94. Every curve
             starts at its first time in the file (the first time at which every
             instance of the arm has a schedule) and nothing is drawn before it. Zero
             is the policy's first (greedy) schedule, and a black diamond on it marks
             the median time of that schedule (policy_first_s). Variants A and B add the
             policy with sampled schedules (method policy_sample, row Sel: sampled
             minus greedy, so it starts at zero and falls), a black line with its 95%
             interval as a pale grey band, drawn up to the budget of its run; the rows
             after that budget (held = yes, the last value kept) are not drawn. Ticks
             at 10 s and the budgets the text reports, and at 3 s (or 2 s) when the
             earliest curve starts within a factor 1.7 above it; the axis runs from
             just below the earliest first schedule to 1.12 times the largest budget.
             The value axis ends below at the lowest band edge, rounded down to the
             step of its tick labels, so no empty band remains under the curves.
    start    wall clock from the start of each run, the earlier mode of p94: ticks at
             the budgets the text reports, from 1 s, or from 0.1 s when a curve moves
             or the policy's mark lies below 1 s, and further left when the mark would
             touch the axis; the diamond marks the policy's time for one schedule at
             that size (the record of --policy-time).
    Variant A draws the four arms with the intervals of the selected training run
    (row Sel). Variant B draws the two rule-started arms with the intervals over every
    training run (row All); its companion, B_supplement, draws the two policy-started
    arms (row Sel) in the same way, for the supplement, and on the request clock
    gains no sampled schedules. --fq3-group no_setups or setups draws the same figure
    from the pooled rows of the combinations without setups (N, L, W, LW) or with setups
    (S, LS, SW, LSW) in place of the eight; its file names end in the group's name
    (fq3_time_budget_A_setups.png), before _dev.
    --memetic-parallel (request clock) draws the memetic algorithm of row Sel as the best of
    eight independent runs on eight cores, the incumbent at each time the shortest of the
    eight: variant A takes both memetic curves and their bands (solid: the best of eight
    started from the best rule; dashed: from the policy's schedule) and the supplement
    companion its policy-started curve from the curves file of p124_memetic_parallel.py, in
    the colour, line styles and legend entry of the memetic curves; nothing else changes, and
    the value axis follows the data by the rule above. Variant B keeps the single run, since
    that file holds row Sel only. A preview of p124 (the first set alone) draws only the
    curves it holds and only the variants that show one of them, into files ending in
    _preview.
    --memetic-minute (with --memetic-parallel only) adds to variant A, at 60 s in each panel,
    the memetic algorithm with a one-minute budget read one minute after the request, from the
    record of p137_memetic_one_minute.py: two circles in the memetic colour, each over its 95%
    interval as a thin vertical bar of the same colour, above every curve (filled: the best of
    eight one-minute runs started from the best rule; open: started from the policy's first
    schedule). The legend gains a third row: under the methods, the one-minute budget (the
    filled and the open circle side by side), and the two line style keys carry the circle of
    their start (solid with a filled circle, dashed with an open one). The intervals join the
    values the value axis covers; variant B and the companion do not change.
    --memetic-five (with --memetic-minute only) adds to variant A, at 300 s in each panel, the
    memetic algorithm with a five-minute budget read five minutes after the request, from the
    record of p137_memetic_one_minute.py --budget 300: the same two circles over their 95%
    intervals, in the same colour, faces, size and z-order as at 60 s (filled: the best of eight
    five-minute runs started from the best rule; open: started from the policy's first
    schedule). The third legend row then names both budgets ("memetic algorithm, budget equal to
    the time"), with the same filled and open circle; nothing else in the legend changes. The
    intervals join the values the value axis covers; variant B and the companion do not change.

fq4_boundary
    Where the lead over dispatching rules ends. (a) The share of decisions with one
    legal pair against the setup share of machine capacity, one point per rule
    combination and benchmark (the policy's rollouts, mean over the training runs);
    the levels of the factory benchmark's sensitivity study that change the setups
    (setup scale, stations charged with setups) are smaller open squares. (b) The
    makespan difference, policy minus best dispatching rule, in percent of the rule's
    makespan, with its 95% interval as a vertical line, against the share of decisions
    with one legal pair, the y of panel (a). Variant A plots the best of four
    sampled schedules against the per-instance best of the four rule schedules (form
    sample4 of p93); variant B one greedy schedule against the same rule schedules
    (form greedy_best4), the form of the sensitivity study, whose setup levels then
    appear in panel (b) as well. Marker shape and colour are the benchmark (factory
    benchmark: square, public instances: circle). Each labelled point carries its
    combination code next to it, beside the point, in a column or in a row, public
    codes on the right first and factory codes on the left first; a code moves at
    most 0.6 text heights (in a row, 1.2 widths of the row's widest code) from its
    point, the coded point nearest to a code is its own (or one with the same code),
    two rows side by side a few points apart share one baseline, and the figure is
    refused when no such place is clear. --fq4-codes b keeps the codes to panel (b).

fs_convergence
    Whether training had stopped improving when it ended. x: the training updates, 0 to
    2,000; y: the validation score itself, the greedy validation makespan divided by the
    MWKR rule's on the same instances (a ratio, lower is better); in both panels a thin dashed
    vertical line at update 1,000, where the specialists' training ends. (a) The specialists
    trained for 2,000 updates, each combination one thin solid line in a light slate (the
    median over the training seeds at each validation), the mean of these lines as one bold
    line in the dark slate, and the mean of the same lines of the eight specialists trained
    for 1,000 updates as one bold line in the medium green, which ends at update 1,000; a
    legend of three entries ("one specialist, 2,000 updates", "mean of the eight, 2,000
    updates", "mean of the eight, 1,000 updates"; a partial record names its own count).
    (b) The policy and the single-rule control, each median over its range across the seeds
    as a pale band. Each panel's y axis covers its lines and bands from update 200 on; a
    value before that which lies higher leaves the axis at the top, and the run prints it.
    The panels share the y axis when the span of each panel's values covers at least two
    thirds of the span of both together, and otherwise each has its own tick labels. Each
    legend sits inside its panel in a thin box, in the first corner where no line or band
    enters it; while none is clear, the panel's axis is raised by one tick step at the top.

fq2_reuse
    A specialist reused under another rule combination, at the training size. x: the eight
    combinations under which the schedules are built; y: the makespan difference, specialist
    minus policy, in percent of the policy's makespan (positive: the specialist's schedule is
    longer), zero a solid line (the policy), the 1% margin a pale band. In each column, left
    of the tick, the specialist trained for that combination (filled circle) and the
    specialist trained for all three rules, LSW (open square), each with its 95% interval as
    a vertical line (in column LSW the two are one specialist, drawn once as the first); right
    of the tick, the means of the other specialists as small open circles, without intervals,
    set apart sideways so that none hides another (six per column, seven in column LSW). One
    legend below the axes in a thin box. Drawn only with --only fq2_reuse.

fq2_heldout
    Combinations held out of training, at the training size. x: the four combinations of two
    or three rules (LS, LW, SW, LSW); y: the makespan difference, model minus specialist, in
    percent of the specialist's makespan (positive: the model's schedules are longer), zero a
    solid line (the specialist of the combination), the 1% margin a pale band. In each column,
    side by side in one fixed order and evenly spaced, one marker per model with its 95%
    interval as a vertical line: the policy (trained on N, L, S, W, LS, LW), the two policies
    with other held-out pairs (LS and LSW, or LW and LSW), the policy trained on all eight
    combinations and the single-rule control (N, L, S, W). Shape tells the models apart; the
    four policies share the house blue, the control has the rose of fs_convergence. The
    marker is filled when the combination of the column is in the model's list of training
    combinations and open (white face) when it was held out of training; the lists come from
    the registry (policy, control), from the record of p122 checked against xsplit_runs
    (policies b and c) and from the record of p95 (the policy trained on all eight). One
    legend right of the axes in a thin box. Drawn only with --only fq2_heldout. Full mode:
    every model must be present and complete, or the figure is refused with the list of what
    is missing (fq2_heldout.pdf/.png). --heldout-partial draws the models whose records exist,
    prints the absent ones, and writes fq2_heldout_partial.pdf/.png only.

fq5_rule_change
    What a rule change costs, two panels side by side. (a) Training before the first use against
    the number of rule combinations served, 1 to 8: one specialist per combination is k times
    the mean training hours of the eight specialists (slate line, squares), with a pale band from
    the sum of the k smallest to the sum of the k largest hours, the spread over the choice of
    the k combinations (left out when it is not thicker than the line at print size); one policy
    is a flat blue line (circles) at its training hours, since the policy trained once serves
    every combination. (b) The steps at a rule change on a logarithmic time axis with decade
    ticks: one row each for the descriptor checks, the re-planning of one batch and the
    validation of the new schedule, one marker shape per batch size (10, 50 and 80 modules) in
    one neutral grey; a marker that would touch one placed before it in its row moves 8 pt above
    or below the row's line. One legend inside each panel in a thin box, where no mark enters
    it. Drawn only with --only fq5_rule_change; there is no partial mode
    (fq5_rule_change.pdf/.png).

fq4_limits
    Where the policy reaches its limits, two panels: (a) the shop sizes 10, 20, 27, 30, 50 and 80
    modules, each difference pooled over the eight combinations; (b) the 22 changed levels of the
    sensitivity study at the training size in the nine groups of its table, each difference
    pooled over the combinations the level changes. At each position two markers with their 95%
    intervals: the policy against the specialists (the specialists' slate, filled circles) and
    the policy against the best dispatching rule (green, open squares); the 1% margin a pale band,
    zero a solid line. Variant A (--fq4-limits-variant A, the default): one row with the y axis
    shared, the values of (b) as rotated tick labels and one group label per group below them
    (at most three lines), all first lines on one baseline and every further line on the
    baseline of its row; labels of three lines make the figure taller below the axes. Variant
    B: panel (b) a forest plot with one row per level, the group label and the value as row
    labels on its left. One legend below the panels in a
    thin box. Drawn only with --only fq4_limits. Full mode: all six sizes and all 22 levels, or
    the figure is refused with the list of what is missing (fq4_limits_<variant>.pdf/.png).
    --fq4-limits-partial leaves out the sizes whose record is absent, prints them, and writes
    fq4_limits_<variant>_partial.pdf/.png only.

Every plotted number is read from the inference outputs; nothing is recomputed
here.
  factory  results/family_inference.json, the five-seed record written by
           p88_family_inference.py and resolved through arm_registry.out() for the
           active setting (--pact-env, --pact-seeds, --pact-out), as
           make_family_figures.py resolves its inputs. Keys: parity.components[R],
           parity.pooled.All, control.components.Spec[R], control.pooled.Spec.All.
  seedext  results/seed_extension.json, the ten-seed record written by
           p110_seed_extension.py, resolved the same way (--seedext). Keys: ten[R],
           pool_ten.All, and five[R], pool_five.All and first_seeds, which must repeat
           the five-seed record. --factory-seeds picks the record of the policy rows of
           panel (a), the eight combinations and All eight: five, ten, or auto (ten when
           the ten-seed record exists). The single-rule control has five seeds, so the
           control rows of variant B come from the five-seed record in every mode. The
           run prints one line naming the record and the number of training seeds
           behind each panel.
  public   results/public_family_act.json, the family under the active mask, written
           by p89_public_family.py --family active (--public-family active, the
           default), or results/public_family_inference.json, the non-delay family
           (--public-family nondelay); test set pub_SD2_test_10x5+mix. Keys:
           family.key and family.prefix (absent in the non-delay record),
           sets[set].parity.components[R], sets[set].parity.pooled.All,
           sets[set].control.control_vs_specialist[R] (the file holds no pooled
           control-vs-specialist row over all eight).
  Fields: mean; lo and hi, the 95% interval of the crossed hierarchical bootstrap
  over training seeds and test instances; v1.0, the verdict at the 1% margin.
  Sign: 100 (a - b) / b with a the policy (or the control) and b the specialist,
  so a negative value means the shorter schedule.
  timed    results/timed_curves.csv and results/timed_inference.json, written
           together by p94_timed_inference.py at the repository root (--timed-curves,
           --timed-json). CSV keys: run, origin, row (Sel or All), size, method
           (memetic_rule, cpsat_rule, memetic_policy, cpsat_policy, and on the request
           clock policy_sample), combination Pool (or, with --fq3-group, the group's key
           NoSetup or Setup), group when the file has it (only the rows of the drawn group
           are read: all, no_setups or setups; a file without it is all), n_combinations,
           t_s, mean_pct,
           lo_pct and hi_pct (the 95% interval: instance-level for Sel, crossed
           hierarchical over training runs and base instances for All). Request clock,
           per size: held (yes on the rows of policy_sample after its budget),
           policy_budget_s (that budget), policy_first_s (the median time of the
           policy's first schedule, the diamond) and rule_first_s (the median time of
           the four rule schedules, printed only). JSON keys: run, origin, sign,
           budgets (start clock) or budgets_per_size[<size>] (request clock), and the
           results.<size> entries of the gates below.
  mpar     fq3 with --memetic-parallel: results/memetic_parallel.json of
           p124_memetic_parallel.py and the curves file its block decision_table.curves names
           (field file), taken from the same folder (results/memetic_parallel_curves.csv). CSV:
           the columns of the timed CSV with n_runs; row Sel, methods memetic_rule_best8 and
           memetic_policy_best8 (solver memetic, start rule or policy), the same groups and
           per-size times. JSON keys: generator, run (full or preview), dry, origin,
           selected_run, decision_table.clock, decision_table.curves (file, row, methods,
           n_runs, grid, start_s, group_start_s) and decision_table.sizes.<size>.rows.<budget>
           .<entry>.values.<memetic_rule|memetic_policy>, the printed values of the decision
           table on eight cores (two decimals, difference from the policy's first schedule in
           percent of it, the sign of the curves).
  mmin     fq3 with --memetic-minute: results/memetic_one_minute.json of
           p137_memetic_one_minute.py. Keys: generator, run (full), dry, origin, read_at_s,
           budget_s, selected_run, instances, boot, and per size
           results.<size>.comparisons.<m60_vs_pol|mp60_vs_pol>.<Pool, or the group's key>
           (fields mean, lo, hi, the 95% interval over base instances; n, the instances; c and
           combinations, the combinations pooled): the best of eight runs with a budget of 60 s
           started from the best rule (m60_vs_pol) or from the policy's first schedule
           (mp60_vs_pol), read 60 s after the request, minus the policy's first schedule, in
           percent of it (the sign of the curves).
  mfiv     fq3 with --memetic-five: results/memetic_five_minute.json of
           p137_memetic_one_minute.py --budget 300. The keys of mmin, with read_at_s and budget_s
           300 and per size results.<size>.comparisons.<m300_vs_pol|mp300_vs_pol>.<Pool, or the
           group's key>: the best of eight runs with a budget of 300 s from each start, read 300 s
           after the request, minus the policy's first schedule, in percent of it.
  policy   start clock only (the request clock reads policy_first_s of the CSV and
           ignores this option): the record of p75_deploy_cost.py, by default
           results/deploy_cost_cpu_1thread.json through arm_registry.out
           (--policy-time takes one or more records). Its entry sizes[<size>] gives
           the mark of that shop size: the median time for one schedule in the
           combination with the largest median (the quantity of the macros
           DeployLatencyFifty and DeployLatencyEighty), and only when the entry's
           field valid is true. The campaign's instance sets 50x25n30 and 80x25n30
           take the entries 50x25 and 80x25, the same shop sizes. A record with a
           top-level field size and worst_regime_median_s (the form of
           p16_policy_latency.py) is read as well. A size without a valid entry
           gets no mark, and the run says so.
  Sign: 100 (a - b) / b with a the search arm at the budget (or the policy's best
  sampled schedule) and b the policy's schedule (on the request clock its greedy
  schedule), the order of the CSV, so a negative value means the first-named
  schedule is the shorter. p94 stores this order for the rule-started arms as its
  comparison (vi), for the sampled schedules as (vii), and the other order (policy
  minus arm, in percent of the arm, with the verdicts) as its comparison (i); nothing
  here converts one into the other.
  rulecmp  results/rulecmp_public.json and results/rulecmp_factory.json, written by
           p93_rule_comparison.py --family public and --family factory (the factory
           file has the same keys; it lacks cross_check_p89 and vali.all_records, which
           the figure does not read). Keys: settings.family, settings.test_set (public:
           pub_SD2_test_10x5+mix), settings.partial; combos[R].traces.policy_one_pair
           .mean and combos[R].traces.policy_setup_share.mean (percent, mean over the
           training runs of the released policy's greedy rollouts);
           combos[R].arms.released.<sample4|greedy_best4> fields mean, lo, hi (the
           crossed hierarchical 95% interval) and verdict. A missing factory file
           leaves its points out.
  sensitivity  results/sensitivity_inference.csv of p87_sensitivity.py infer, with its
           JSON beside it (fields partial and measured_variables.computed). Rows with
           parameter "setup scale" or "stations charged with setups", comparison
           pol_rule and reached 1 (the combinations S, LS, SW and LSW). Columns: level,
           row (the combination), mean, lo and hi (one greedy schedule of the policy
           against the per-instance best of the four rules), pol_one_pair and
           pol_setup_share. A missing file leaves its points out.
  Sign of fq4: 100 (a - b) / b with a the policy and b the best dispatching rule, so
  a negative value means the policy's schedule is the shorter.
  convergence  results/convergence.json, written by p116_convergence.py and resolved
           through arm_registry.out() (--convergence), with the macro file of the same run,
           manuscript/macros_convergence.tex (--convergence-macros). Keys: generator, seeds,
           partial, runs (curve, updates, arm, combo, change_after_1000 of arm long), arms
           (summaries; arms.long with windows, combos, pooled, min_run, max_run, negative),
           curves.<arm> and curves.specialist.<R> (share, median, min, max, n), raw_curves.<arm>
           for the policy and the control and raw_curves.<arm>.combos.<R> for the specialists
           of 1,000 (specialist) and 2,000 updates (long) (update, median, min, max, n, seeds),
           with raw_curves.<arm>.mean (update, mean, combos, n).
  crosseval  results/cross_eval.json, written by p100_cross_eval.py and resolved through
           arm_registry.out() (--crosseval), with the macro file of the same run,
           manuscript/macros_crosseval.tex (--crosseval-macros). Keys: generator, environment,
           sign, seeds, n, boot, cells["<R>|<C>"] (the specialist trained for R run under C:
           mean, lo, hi, v1.0, direction). Sign: 100 (a - b) / b with a the specialist and b
           the policy, so a positive value means the specialist's schedule is the longer.
  heldout  fq2_heldout. The policy and the single-rule control: the records and the choice of
           fq1 (--factory, --factory-seeds, --seedext; parity.components[R] or ten[R], and
           control.components.Spec[R] of the five-seed record in every mode), read by the same
           loaders and gates, so both figures show the same numbers. Policies b and c:
           results/heldout_splits.json and manuscript/macros_heldout.tex of
           p122_heldout_splits.py through arm_registry.out (--heldout-splits,
           --heldout-splits-macros); keys generator, seeds, dry, definitions
           .against_specialist, split.<b|c>.train, .held and .against_specialist[R]. The
           policy trained on all eight: results/secondary_inference.json and
           manuscript/macros_secondary.tex of p95_secondary_inference.py, which writes them at
           the repository root (--secondary, --secondary-macros); keys generator, environment,
           check, sign, blocks.all8.status and blocks.all8.result (stem, folder, seeds,
           effort.combos, vs_specialist.components[R]). Fields mean, lo, hi (the crossed
           hierarchical 95% interval over training seeds and test instances), v1.0 and
           direction. Sign: 100 (a - b) / b with a the model and b the specialist of the
           combination, seed s against seed s (section A of p88_family_inference.py).
  rulechange  fq5_rule_change. results/rule_change_cost.json and
           manuscript/macros_rulechange_cost.tex of p99_rule_change_cost.py infer, through
           arm_registry.out (--rule-change, --rule-change-macros); keys generator, environment,
           missing, records.released.hours and records['specialist <R>'].hours (wall-clock
           training hours of one run each with the GPU reserved for it), and macros
           RccDescChecksMs<W> and RccScheduleCheckMs<W> (ms, the median over the test instances
           on one thread, at the sizes of p99, 10x25, 50x25n30 and 80x25n30). Re-planning: the
           record of p75_deploy_cost.py, results/deploy_cost_cpu_1thread.json through
           arm_registry.out (--deploy), entries sizes[10x25|50x25|80x25] (the shops of the
           campaign's sets 50x25n30 and 80x25n30, as for fq3), the median over the instances
           of the combination with the largest median (p75_deploy_cost.worst_regime, the
           quantity of DeployLatency<Word>), and the macro file p75 wrote from it,
           paper/macros_deploy.tex through arm_registry.out (--deploy-macros).
  limits   fq4_limits. Through arm_registry.out: the record of p88 and its macros (--factory,
           --factory-macros; parity.pooled.All at 10 modules, larger_shops.<20x25|30x25>
           .parity_pooled.All and .rule_pool_pooled), results/per_floor.json and
           manuscript/macros_perfloor.tex of p118_per_floor.py (--per-floor, --per-floor-macros;
           sizes.27x25, and sizes.20x25 and 30x25, which must equal the rows of p88), and
           results/production_specialists.json and manuscript/macros_prodspec.tex of
           p128_production_specialists.py (--prodspec, --prodspec-macros; sizes.50x25n30 and
           80x25n30). The JSON that p87_sensitivity.py infer writes beside --sensitivity, with
           --sensitivity-macros: levels.<key> (parameter, value, reached, and pools.pool or
           pools.reached with combos, pol_spec and pol_rule); the 10-module point against the
           best dispatching rule is the default level's pools.pool.pol_rule, since p88 stores no
           pooled row against the rules at the training size. Fields mean, lo, hi (the
           hierarchical 95% interval over the five training seeds and the test instances) and c
           (the combinations pooled). Sign: 100 (a - b) / b with a the policy, so a negative
           value means the policy's schedule is the shorter.

Gates (any failure stops the figure): every verdict read from a file must agree
with its own interval and the margin, so a misread key cannot reach the plot;
the plotted values are held against the macros the text prints, at their two
printed decimals (macros_family.tex for the factory when it exists; with ten seeds,
the policy rows against the record's own manuscript/macros_seedext.tex, SeedExt<R>
and SeedExtPoolAll, which must exist; manuscript/macros_pubfam_act.tex, prefix
PubFamActSdten, or manuscript/macros_public.tex, prefix PubSdten, for the public
family), and every policy row must have its value, Lo, Hi and VOne macros; a
ten-seed record that does not extend the five-seed record, or that comes from a dry
run of p110 without --allow-partial (then every file name ends in _dev), is
refused, and so is a public record of another family than --public-family; every
text of the figure must lie inside the canvas and no two texts may overlap; the PDF
may embed the Times face only.
Gates of fq3, in addition to the layout and font gates: a CSV whose run column says
anything but full (a partial run of p94, or one with a substituted policy) is
refused unless --allow-partial is given, and then every file name ends in _dev; the
JSON must come from the same run (run and origin) and state p94's sign; the clock
must be the request or the start origin; every pooled row must pool the eight
combinations (with --fq3-group, the group's four, and the JSON's groups.entries.<key>
must list them); no pooled mean may rise with time (an incumbent only improves, so a
rising curve is a misread file). With --fq3-group every JSON key named .Pool below, and
ix on the request clock, is read as the group's key (.NoSetup or .Setup; ix.<key>).
  Start clock: the plotted values must equal the JSON, to the CSV's six decimals, at
  every reported budget (budgets) for the policy-started arms
  (results.<size>.ii.<memetic|cpsat>.<budget>.Pool, arm minus policy) and the
  rule-started arms (results.<size>.vi.<memetic|cpsat>.<budget>.<selected|all_seeds>
  .Pool, arm minus policy; a partial file written before p94 stored vi is drawn
  without it), and at time zero for the rule-started arms (results.<size>.iv.selected
  .Pool for row Sel and iv.all_seeds.Pool for row All, the best rule minus the policy).
  Request clock: the request columns must exist; each size takes one value of
  policy_first_s, rule_first_s and policy_budget_s, equal to results.<size>.ix.policy
  .t.median_s, ix.rules.t_all.median_s and policy_budget_s of the JSON (a full run
  must hold all three); held is yes exactly on the policy_sample rows after its
  budget; no sampled value may lie above zero (sampled minus greedy); and at every
  budget of budgets_per_size[<size>] the plotted values must equal ii (policy-started
  arms), vi (rule-started arms, rows Sel and All) and, at the budgets inside its own
  budget, vii (results.<size>.vii.<budget>.Pool, the sampled schedules minus the
  greedy schedule). An entry holding no_schedule_yet is skipped, and a size of the CSV
  whose JSON lacks one of these keys or entries is refused. The rule-started curves
  have no time-zero row on this clock, so the time-zero gate does not apply.
  --memetic-parallel, after the gates above on the timed record: the timed record must be
  on the request clock; the record of p124 must name p124 as its generator and be a full
  run (a preview, and a record of a dry run, only with --allow-partial; the file names then
  end in _preview, or in _dev for a dry run), on the request clock (origin and
  decision_table.clock, and the CSV's column origin); its selected training run (selected_run,
  and the CSV's column policy) and curve grid (decision_table.curves.grid, and every time of
  the drawn rows) must be those of the timed record (selection.seed, the policy column of
  its rows Sel, grid.curve); per size the CSV must hold one value each of policy_first_s,
  rule_first_s and policy_budget_s, equal to the timed CSV's for the drawn group; every row
  must hold row Sel, n_runs 8, held no and solver memetic, and decision_table.curves row Sel
  and n_runs 8; the CSV's methods must be those of decision_table.curves.methods, both
  methods in a full run, each with its pooled rows at both sizes, its start schedule, the
  group's combinations, the start time of start_s (group_start_s for a group), no rise and
  every mean inside its interval; and its value at 60, 300 and 600 s must equal the value
  the decision table prints for its column at that size, budget and entry (Pool, or the
  group's key), to the printed digit. The run prints the count of these checks.
  --memetic-minute, only with --memetic-parallel: the record of p137 must name p137 as its
  generator and be a full run (a preview and a dry run are refused, with or without
  --allow-partial), on the request clock, read at 60 s with a budget of 60 s, with the
  selected training run, the number of instances and the number of bootstrap draws of the
  record of p124; every entry read must hold a finite mean inside its interval and pool the
  group's combinations over that number of instances. Each circle must be drawn at 60 s with
  the record's mean and its bar from the record's lower to upper end, inside the axes and clear
  of the other circle and of the policy's diamond. The run prints both values with their
  intervals per size.
  --memetic-five, only with --memetic-minute: the gates of --memetic-minute on the record of
  p137 --budget 300, read at 300 s with a budget of 300 s (a record read at another time, the
  one-minute record among them, is refused, and so are a preview and a dry run, with or without
  --allow-partial). Each circle must be drawn at 300 s with the record's mean and its bar from
  the record's lower to upper end, inside the axes and clear of every other circle and of the
  policy's diamond. The run prints both values with their intervals per size, read back from
  the record, which must still hold the values drawn.
Gates of fq4, in addition to the layout and font gates: a file of another family or
test set, a file without a key the figure reads, and a sensitivity CSV without
measured variables or without one of the 24 rows (six setup levels on four
combinations) are refused; a file of a partial run is refused unless --allow-partial
is given, and then every file name ends in _dev; every verdict must agree with its
interval (ahead only below zero, behind only above); the plotted values are held
against the macros at their printed decimals: macros_rulecmp_public.tex and
macros_rulecmp_factory.tex (Rcp and Rcf: Pol<form><R> with Lo, Hi and Verdict for
both forms at two decimals, OnePair<R> and SetupShare<R> at one) and
macros_sensitivity.tex (Sens<level>PolOnePair<R>, Sens<level>PolSetupShare<R>; the
setup levels' differences against the rule have no macro); a macro file that does
not exist is skipped with a printed line; no code may touch a marker, an interval,
the zero line or another text.
Gates of fs_convergence, in addition to the layout and font gates: a record of another
generator, a record without the specialists trained for 2,000 updates, and a record of p116
--partial are refused (a partial record is drawn for review only, through a keyword that the
command line does not set, into a folder outside the repository, and the file names then end
in _review), and so is a full record that lacks a curve group or a combination; every stored
curve group must equal the median, minimum and maximum over its runs of 100 (score / best - 1)
and every raw curve group the median, minimum and maximum over its runs of the score,
recomputed from the runs' curves, every run's curve must reach its own best, each mean of the
specialists' per-combination medians must be recomputed, and the change after update 1,000 of
the specialists trained for 2,000 updates must be recomputed per run, per combination and over
all runs; the macro file must exist and agree with the record (counts exactly, signed values at
their two printed decimals, shares of training at their printed digits, the combinations as
letters and as running text); no value after update 200 may leave the axis, and no line or
band may enter a legend's box.
Gates of fq2_reuse, in addition to the layout and font gates: a missing record, a record of
another generator, environment or sign, and a record without one of the 64 cells or one of
their fields are refused; every cell's mean must lie inside its interval, and its verdict at
the 1% margin and its direction must be those of stats_family.verdict and
stats_family.direction for that interval; the macro file must exist, its counts XevSeedN,
XevInstN and XevBoot must equal the record's, and every plotted value must equal the macro
that prints it at its two signed decimals (Xev<R>On<C> for a mean, Lo and Hi for the bounds of
the two intervals of a column); a plotted value without a macro is compared with the record
only and listed in the run's output; no two markers may overlap, no marker may touch an
interval other than its own, and every mark must lie inside the axes.
Gates of fq2_heldout, in addition to the layout and font gates: the policy and the control
pass the loaders and macro gates of fq1 (macros_family.tex, or macros_seedext.tex for the
policy with ten seeds); a record of p122 or p95 of another generator, environment or sign, a
record of the check mode of p95, a record without one of the keys or fields the figure reads,
a training list of a split policy other than that of xsplit_runs, and a record of a dry run of
p122 without --allow-partial (then every file name ends in _dev) are refused; every plotted
mean must lie inside its interval and its verdict at the 1% margin and its direction must be
those of stats_family.verdict and stats_family.direction; in full mode policies b and c must
have the seeds of xsplit_runs.SEEDS and the policy trained on all eight the seeds of
arm_registry.SECONDARY_SEEDS; the macro files of p122 and p95 must exist, their seed counts
(HspSeeds, SecSeedN) must equal the records', and every plotted value must equal the macro
that prints it at its two signed decimals (Hsp<B|C><R>, SecAllEightPar<R>, each with Lo, Hi
and VOne); a plotted value without a macro is compared with the record only and listed; no two
markers may overlap, no marker may touch an interval other than its own, and every mark must
lie inside the axes.
Gates of fq5_rule_change, in addition to the layout and font gates: a record of p99 of another
generator or environment, or one without the policy, one of the eight specialists or the checks
at one of the three sizes, is refused with the list of what is missing; a record of p75 that is
not of the paper's environment is refused with every reason (it must carry the field
environment exact at the top and at each size, the policy of that environment, one torch
thread, allow_busy false, valid true at the top and at each size with cpu_share at least p75's
SHARE_MIN, and the eight combinations timed; the record of an earlier environment has no field
environment and four threads); both macro files must exist; every macro the figure reads from
the p99 record must be the one its macro file prints, the plotted hours must equal
RccPolicyHours, RccSpecHours<R> and RccCatalogueHours at their two decimals, and
RccSpecMeasuredN must be 8; the header of the p75 macro file must name the record as its
source, the same model and one torch thread, and every re-planning time must equal
DeployLatency<Word> at its two decimals; the two panels share one height and one baseline;
where the lines of panel (a) cross, at least 30% of a policy marker stays visible under a
specialist marker; no two markers of panel (b) overlap; every marker lies inside its panel and
no mark enters a legend's box.
Gates of fq4_limits, in addition to the layout and font gates: a record of another generator,
of other training seeds than 301 to 305 or of another environment than the paper's is refused,
and so is a record of a dry or partial run of p87, p118 or p128 unless --allow-partial is given
(then every file name ends in _dev); the record of p87 is needed in every mode, and each of its
levels must set the parameter and value of the table and change the combinations of its group;
every pooled row must hold finite values, its mean inside its interval, and pool the
combinations it names; the pooled difference from the specialists at 10 modules must be the
same in the records of p88 and p87, and the rows at 20 and 30 modules of p88 and p118, and the
10-module row against the best dispatching rule of p87 and p128, must be equal; every plotted mean and interval end must equal the macro that prints it at its two
signed decimals (FamParPoolAll, SensDefaultPolRulePool, FamShop<Twenty|Thirty> and
Pfl<Twenty|Thirty> with ParPoolAll and RulePool, PflParPoolAll, PflRulePool,
Pbs<Fifty|Eighty> with ParPoolAll and RulePool, PbsTenRulePool, and
Sens<level>Pol<Spec|Rule><Pool|Reached>, each with Lo and Hi), and a missing macro file or
macro is refused; the copy of each of these macro files in manuscript/, the files the
manuscript inputs, must print the same text for every macro the figure reads whenever it is
another file (a copy not yet made is listed); no two marks may overlap and every mark lies inside its panel; neighbouring
tick and group labels keep 2 pt between them (the rows of variant B 1 pt), and in variant A
every group label stays below the tick labels.

Type: every text in TeX Gyre Termes; the names of the eight rule combinations (tick and row
labels, legend entries, the codes of fq4) in its italic through mathtext ($LSW$, comb_tex), as
the manuscript sets them in its text, equations and tables; the words around a name stay upright.

Size: the page has one column with a text width of 164.6 mm.
The figure frame of main.tex sets the graphic in a parbox 13 pt narrower than the
text width, so includegraphics at width=linewidth prints it 6.30 in wide. The
figure is drawn at exactly 6.30 in and saved untrimmed, so every size below is
the printed size.

Usage: python scripts/make_figures.py
       [--only fq1|fq3|fq4|fs_convergence|fq2_reuse|fq2_heldout|fq5_rule_change|fq4_limits]
       [--factory JSON]
       [--factory-seeds auto|five|ten] [--seedext JSON] [--seedext-macros TEX]
       [--public-family active|nondelay] [--public JSON] [--public-set NAME]
       [--public-macros TEX] [--factory-macros TEX] [--timed-curves CSV] [--timed-json JSON]
       [--policy-time JSON [JSON ...]] [--rulecmp-public JSON]
       [--rulecmp-factory JSON] [--rulecmp-public-macros TEX]
       [--rulecmp-factory-macros TEX] [--sensitivity CSV] [--sensitivity-macros TEX]
       [--fq3-group all|no_setups|setups] [--memetic-parallel JSON] [--memetic-minute JSON]
       [--memetic-five JSON] [--fq4-codes ab|b] [--convergence JSON]
       [--convergence-macros TEX] [--crosseval JSON] [--crosseval-macros TEX]
       [--heldout-splits JSON] [--heldout-splits-macros TEX] [--secondary JSON]
       [--secondary-macros TEX] [--heldout-partial]
       [--rule-change JSON] [--rule-change-macros TEX] [--deploy JSON] [--deploy-macros TEX]
       [--per-floor JSON] [--per-floor-macros TEX] [--prodspec JSON] [--prodspec-macros TEX]
       [--fq4-limits-variant A|B] [--fq4-limits-partial]
       [--allow-partial] [--out DIR]
       [--pact-env legacy|exact] [--pact-seeds 301-305] [--pact-out DIR]
Without --only, fq1 is drawn, fq3 as well when its CSV exists, fq4 when the
public rule comparison exists, and fs_convergence when the record of p116 exists;
fq2_reuse is drawn only with --only fq2_reuse, fq2_heldout only with --only fq2_heldout,
fq5_rule_change only with --only fq5_rule_change, fq4_limits only with --only fq4_limits.
"""
import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arm_registry as REG  # noqa: E402  (pops the --pact-* options from sys.argv)

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as _fm  # noqa: E402
from matplotlib import transforms as mtransforms  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from matplotlib.ticker import (FixedLocator, FuncFormatter, MaxNLocator,  # noqa: E402
                               NullLocator)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DEFAULT = os.path.join('manuscript', 'sections_new', 'figures')

# ------------------------------------------------------------------ house style
# The same set-up as make_family_figures.py: TeX Gyre Termes (the face newtxtext
# puts on the page), converted to TrueType once and kept in scripts/fonts/, and
# every mathtext class mapped onto it so no second family can be embedded.
_FONTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fonts')
for _f in glob.glob(os.path.join(_FONTDIR, 'TeXGyreTermes-*.ttf')):
    _fm.fontManager.addfont(_f)
_SERIF = 'TeX Gyre Termes'

FS_TICK, FS_LABEL, FS_TITLE = 8.0, 8.5, 9.0     # printed sizes in pt
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': [_SERIF, 'Liberation Serif', 'Times New Roman'],
    'mathtext.fontset': 'custom',
    'mathtext.rm': _SERIF, 'mathtext.it': f'{_SERIF}:italic',
    'mathtext.bf': f'{_SERIF}:bold', 'mathtext.sf': _SERIF,
    'mathtext.tt': _SERIF, 'mathtext.cal': f'{_SERIF}:italic',
    'font.size': FS_LABEL, 'axes.titlesize': FS_TITLE, 'axes.labelsize': FS_LABEL,
    'xtick.labelsize': FS_TICK, 'ytick.labelsize': FS_TICK,
    'legend.fontsize': FS_TICK,
    'text.color': 'black', 'axes.labelcolor': 'black',
    'xtick.color': 'black', 'ytick.color': 'black',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
    'ytick.major.width': 0.6, 'pdf.fonttype': 42,
})

# Palette of the accepted forest plots (make_family_figures.py, f23 and f9).
C_EQUIV = '#0072B2'          # equivalent at the 1% margin: filled circle
C_OPEN = '#7d8a97'           # equivalence not shown: open square
C_CONTROL = '#e3a1aa'        # single-rule control: pale rose, thin, lighter
BAND_FILL = '#e4eef6'        # the 1% margin
LINE_HALF = '#7fa3c0'        # the 0.5% margin, dashed
C_ZERO = '0.35'
C_GRID = '0.85'
MS_CIRCLE, MS_SQUARE = 4.6, 4.0

MARGIN, HALF = 1.0, 0.5
COMBOS = list(REG.COMBOS)                       # N, L, S, W, LS, LW, SW, LSW


def comb_tex(name):
    """The printed form of the name of a rule combination: math italic ($LSW$), the form of
    the manuscript's text, equations and tables, set in the italic of the figure's face. Every
    figure prints a combination name through this function; data keys, macro names and the
    printed lines keep the plain name."""
    if name not in COMBOS:
        raise ValueError(f'{name!r} is not a rule combination ({", ".join(COMBOS)})')
    return f'${name}$'


HELD = list(REG.HELD_OUT)                       # SW, LSW
POOL_LABEL = 'All eight'
POLICY_LABELS = COMBOS + [POOL_LABEL]           # rows whose macros must exist
MINUS = '\u2212'

# The training seeds behind the policy rows of panel (a) (--factory-seeds): five, the
# five-seed record of p88; ten, the ten-seed record of p110_seed_extension.py; auto,
# ten when that record exists and five otherwise.
FQ1_SEED_MODES = ('auto', 'five', 'ten')
SEEDEXT_PREFIX = 'SeedExt'
# The public family of p89_public_family.py (--family): record, macro file and macro
# prefix. The record states its family in family.key and family.prefix; a record
# without the key is the non-delay family.
FQ1_PUBLIC = {'active': ('results/public_family_act.json',
                         'manuscript/macros_pubfam_act.tex', 'PubFamAct'),
              'nondelay': ('results/public_family_inference.json',
                           'manuscript/macros_public.tex', 'Pub')}

FIG_W = 6.30                                    # in, the printed width (see above)


# ------------------------------------------------------------------ data
def _item(src, key):
    """mean, lo, hi and the 1% verdict of one comparison, as the file holds them."""
    for k in ('mean', 'lo', 'hi', 'v1.0'):
        if k not in src:
            raise SystemExit(f'fq1 REFUSING: {key} has no field {k!r}')
    lo, hi = float(src['lo']), float(src['hi'])
    eq = src['v1.0'] == 'equivalent'
    if eq != (-MARGIN < lo and hi < MARGIN):     # the verdict must match its interval
        raise SystemExit(f'fq1 REFUSING: {key} verdict {src["v1.0"]!r} disagrees with '
                         f'its interval [{lo:+.3f}, {hi:+.3f}] at the 1% margin')
    if not lo <= float(src['mean']) <= hi:
        raise SystemExit(f'fq1 REFUSING: {key} mean outside its interval')
    return dict(mean=float(src['mean']), lo=lo, hi=hi, verdict=src['v1.0'], equiv=eq,
                n=src.get('n'), key=key)


def load_factory(path):
    d = json.load(open(path))
    par, ctl = d['parity'], d.get('control') or {}
    out = dict(setting=d.get('setting', '?'), seeds=d.get('seeds'), missing=[],
               rows={}, ctl={}, pool=None, ctl_pool=None)
    for r in COMBOS:
        out['rows'][r] = _item(par['components'][r], f'factory parity.components.{r}')
        spec = ctl.get('components', {}).get('Spec', {})
        if r in spec:
            out['ctl'][r] = _item(spec[r], f'factory control.components.Spec.{r}')
        else:
            out['missing'].append(f'factory control vs specialist, {r}')
    if 'All' in par.get('pooled', {}):
        out['pool'] = _item(par['pooled']['All'], 'factory parity.pooled.All')
    else:
        out['missing'].append('factory pooled parity over all eight')
    cp = ctl.get('pooled', {}).get('Spec', {})
    if 'All' in cp:
        out['ctl_pool'] = _item(cp['All'], 'factory control.pooled.Spec.All')
    else:
        out['missing'].append('factory pooled control vs specialist over all eight')
    return out


def load_public(path, set_name):
    d = json.load(open(path))
    if set_name not in d.get('sets', {}):
        raise SystemExit(f'fq1 REFUSING: {path} has no test set {set_name}')
    s = d['sets'][set_name]
    par, ctl = s['parity'], s['control']
    fam = d.get('family') or {}
    out = dict(setting=f'{set_name}, {s.get("n")} instances, seeds '
                       f'{",".join(map(str, d.get("seeds", [])))}',
               seeds=d.get('seeds'), missing=[], rows={}, ctl={}, pool=None,
               ctl_pool=None, family=fam.get('key', 'nondelay'),
               prefix=fam.get('prefix', 'Pub'))
    for r in COMBOS:
        out['rows'][r] = _item(par['components'][r], f'public parity.components.{r}')
        cs = ctl.get('control_vs_specialist', {})
        if r in cs:
            out['ctl'][r] = _item(cs[r], f'public control.control_vs_specialist.{r}')
        else:
            out['missing'].append(f'public control vs specialist, {r}')
    if 'All' in par.get('pooled', {}):
        out['pool'] = _item(par['pooled']['All'], 'public parity.pooled.All')
    else:
        out['missing'].append('public pooled parity over all eight')
    cp = ctl.get('pooled_control_vs_specialist', {})
    if 'All' in cp:
        out['ctl_pool'] = _item(cp['All'], 'public control.pooled_control_vs_specialist.All')
    else:
        out['missing'].append('public pooled control vs specialist over all eight '
                              f'(the file pools only {", ".join(cp) or "nothing"})')
    return out


def load_seedext(path, F):
    """The policy rows of panel (a) from the ten-seed record of p110_seed_extension.py:
    ten[R] per combination and pool_ten.All, each checked as _item checks it. The record
    must extend the five-seed record F, whose control rows variant B draws: its
    first_seeds are F's seeds, and its five[R] and pool_five.All equal F's policy rows."""
    if not path or not os.path.exists(path):
        raise SystemExit(f'fq1 REFUSING: the ten-seed record {path} not found (written by '
                         'p110_seed_extension.py)')
    d = json.load(open(path))
    lack = [k for k in ('seeds', 'first_seeds', 'ten', 'pool_ten', 'five', 'pool_five')
            if k not in d]
    if lack:
        raise SystemExit(f'fq1 REFUSING: {path} has no key {lack}')
    if list(d['first_seeds']) != list(F['seeds'] or []):
        raise SystemExit(f'fq1 REFUSING: {path} extends the seeds {d["first_seeds"]}, the '
                         f'five-seed record holds {F["seeds"]}')
    pairs = [(r, d['five'].get(r), F['rows'][r]) for r in COMBOS]
    pairs.append((POOL_LABEL, d['pool_five'].get('All'), F['pool']))
    for label, five, row in pairs:
        if row is None:
            continue
        if five is None:
            raise SystemExit(f'fq1 REFUSING: {path} holds no five-seed row {label}')
        it = _item(five, f'ten-seed record, five-seed row {label}')
        if it['verdict'] != row['verdict'] or any(abs(it[k] - row[k]) > 1e-9
                                                  for k in ('mean', 'lo', 'hi')):
            raise SystemExit(f'fq1 REFUSING: the five-seed row {label} of {path} is not the '
                             'row of the five-seed record, so the ten-seed record does not '
                             'extend it')
    rows = {}
    for r in COMBOS:
        if r not in d['ten']:
            raise SystemExit(f'fq1 REFUSING: {path} has no ten-seed row {r}')
        rows[r] = _item(d['ten'][r], f'ten-seed ten.{r}')
    if 'All' not in d['pool_ten']:
        raise SystemExit(f'fq1 REFUSING: {path} has no pooled ten-seed row over all eight')
    return dict(seeds=list(d['seeds']), rows=rows,
                pool=_item(d['pool_ten']['All'], 'ten-seed pool_ten.All'),
                dry=bool(d.get('dry')), setting=d.get('setting', '?'))


def fq1_seed_mode(mode, seedext_json):
    """five or ten for --factory-seeds: auto is ten when the ten-seed record exists;
    ten refuses when it does not."""
    if mode not in FQ1_SEED_MODES:
        raise SystemExit(f'fq1 REFUSING: --factory-seeds {mode!r} is not one of '
                         f'{FQ1_SEED_MODES}')
    have = bool(seedext_json) and os.path.exists(seedext_json)
    if mode == 'auto':
        return 'ten' if have else 'five'
    if mode == 'ten' and not have:
        raise SystemExit(f'fq1 REFUSING: --factory-seeds ten, but the ten-seed record '
                         f'{seedext_json} not found')
    return mode


# ------------------------------------------------------------------ macro gates
def _macros(path):
    return dict(re.findall(r'\\newcommand\{\\(\w+)\}\{([^}]*)\}', open(path).read()))


def _gate(fam, mac, path, plan):
    """plan: list of (label, item, macro stem). The stem's value, Lo and Hi macros
    must equal the item at the two printed decimals, and its VOne macro must give
    the same verdict class. A row of the policy (POLICY_LABELS) must have all four
    macros; a control row without its value macro is reported as 'no macro'.
    Returns the table rows for the report."""
    rows, bad = [], []
    for label, it, stem in plan:
        if it is None:
            continue
        got = {k: mac.get(stem + k) for k in ('', 'Lo', 'Hi', 'VOne')}
        if label in POLICY_LABELS and None in got.values():
            bad.append(f'{fam} {label}: no macro '
                       + ', '.join(f'\\{stem}{k}' for k, v in got.items() if v is None))
            continue
        if got[''] is None:
            rows.append((fam, label, it, None, 'no macro'))
            continue
        ok = True
        for k, v in (('', it['mean']), ('Lo', it['lo']), ('Hi', it['hi'])):
            if got[k] is None or abs(v - float(got[k])) > 0.005 + 1e-9:
                ok = False
                bad.append(f'{fam} {label}: \\{stem}{k} = {got[k]}, file {v:+.4f}')
        if got['VOne'] is not None and (got['VOne'] == 'equivalent') != it['equiv']:
            ok = False
            bad.append(f'{fam} {label}: \\{stem}VOne = {got["VOne"]}, file {it["verdict"]}')
        rows.append((fam, label, it, got, 'match' if ok else 'MISMATCH'))
    if bad:
        raise SystemExit(f'fq1 REFUSING, {path} disagrees with the plotted file:\n  '
                         + '\n  '.join(bad))
    return rows


def gate_factory(F, path, policy=True):
    """The factory rows against the five-seed macros (FamPar, FamCtlSpec); with policy
    False the control rows only, since the policy rows then come from the ten-seed
    record (gate_seedext)."""
    if not path or not os.path.exists(path):
        print(f'  factory macro check skipped: {path} not found')
        return [('factory', lab, it, None, 'no macro file')
                for lab, it in _labelled(F, policy)]
    mac = _macros(path)
    plan = []
    if policy:
        plan += [(r, F['rows'][r], f'FamPar{r}') for r in COMBOS]
        plan += [(POOL_LABEL, F['pool'], 'FamParPoolAll')]
    plan += [(f'{r} control', F['ctl'].get(r), f'FamCtlSpec{r}') for r in COMBOS]
    plan += [(f'{POOL_LABEL} control', F['ctl_pool'], 'FamCtlSpecPoolAll')]
    return _gate('factory', mac, path, plan)


def gate_seedext(F, path):
    """The policy rows of the ten-seed record against the macro file of the same record
    (SeedExt<R> and SeedExtPoolAll, each with Lo, Hi and VOne); refuses when the file
    is missing."""
    if not path or not os.path.exists(path):
        raise SystemExit(f'fq1 REFUSING: the macro file of the ten-seed record, {path}, '
                         'not found (written by p110_seed_extension.py)')
    plan = [(r, F['rows'][r], f'{SEEDEXT_PREFIX}{r}') for r in COMBOS]
    plan += [(POOL_LABEL, F['pool'], f'{SEEDEXT_PREFIX}PoolAll')]
    return _gate('factory', _macros(path), path, plan)


def gate_public(P, path, set_name):
    """The public rows against the macros of the record's family, named
    <prefix>Sdten... (prefix Pub for the non-delay family, PubFamAct under the active
    mask)."""
    word = {'pub_SD2_test_10x5+mix': 'Sdten'}.get(set_name)
    if word is None or not path or not os.path.exists(path):
        print(f'  public macro check skipped: {path} or the set {set_name} has no macros')
        return [('public', lab, it, None, 'no macro file') for lab, it in _labelled(P)]
    mac = _macros(path)
    stem = P['prefix'] + word
    plan = [(r, P['rows'][r], f'{stem}Par{r}') for r in COMBOS]
    plan += [(POOL_LABEL, P['pool'], f'{stem}ParPoolAll')]
    plan += [(f'{r} control', P['ctl'].get(r), f'{stem}CtlSpec{r}') for r in COMBOS]
    plan += [(f'{POOL_LABEL} control', P['ctl_pool'], f'{stem}CtlSpecPoolAll')]
    return _gate('public', mac, path, plan)


def _labelled(D, policy=True):
    out = [(r, D['rows'][r]) for r in COMBOS] + [(POOL_LABEL, D['pool'])] if policy else []
    out += [(f'{r} control', D['ctl'].get(r)) for r in COMBOS]
    out += [(f'{POOL_LABEL} control', D['ctl_pool'])]
    return [(a, b) for a, b in out if b is not None]


# ------------------------------------------------------------------ drawing
def _xaxis(values):
    """Shared range and ticks: whole percent ticks, the range padded to the next
    half percent and never narrower than the 1% band plus half a percent."""
    lo = min(min(values), -MARGIN - 0.5)
    hi = max(max(values), MARGIN + 0.5)
    x0 = np.floor((lo - 0.15) * 2) / 2
    x1 = np.ceil((hi + 0.15) * 2) / 2
    ticks = np.arange(np.ceil(x0), np.floor(x1) + 1e-9, 1.0)
    return (x0, x1), ticks


def _fmt(t):
    return '0' if t == 0 else (f'{MINUS}{abs(t):g}' if t < 0 else f'{t:g}')


def _interval(ax, it, y, lw=1.4):
    c, mk, ms = (C_EQUIV, 'o', MS_CIRCLE) if it['equiv'] else (C_OPEN, 's', MS_SQUARE)
    ax.plot([it['lo'], it['hi']], [y, y], lw=lw, color=c, solid_capstyle='butt',
            zorder=4)
    if it['equiv']:
        ax.plot([it['mean']], [y], mk, ms=ms, color=c, mec='white', mew=0.5, zorder=5)
    else:
        ax.plot([it['mean']], [y], mk, ms=ms, mfc='white', mec=c, mew=1.0, zorder=5)


def _control(ax, it, y):
    ax.plot([it['lo'], it['hi']], [y, y], lw=1.0, color=C_CONTROL,
            solid_capstyle='butt', zorder=3)
    ax.plot([it['mean']], [y], '|', ms=5.0, mew=1.2, color=C_CONTROL, zorder=3)


def _panel(ax, D, ys, y_pool, xlim, ticks, with_control, d_pol, d_ctl):
    ax.axvspan(-MARGIN, MARGIN, color=BAND_FILL, lw=0, zorder=0)
    for t in ticks:
        if abs(t) not in (0.0, MARGIN):
            ax.axvline(t, color=C_GRID, lw=0.4, zorder=0.5)
    for xv in (-HALF, HALF):
        ax.axvline(xv, color=LINE_HALF, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.axvline(0, color=C_ZERO, lw=0.6, zorder=1)
    for r, y in zip(COMBOS, ys):
        _interval(ax, D['rows'][r], y + d_pol)
        if with_control and D['ctl'].get(r) is not None:
            _control(ax, D['ctl'][r], y + d_ctl)
    if D['pool'] is not None:
        _interval(ax, D['pool'], y_pool + d_pol)
    if with_control and D['ctl_pool'] is not None:
        _control(ax, D['ctl_pool'], y_pool + d_ctl)
    ax.set_xlim(*xlim)
    ax.set_xticks(ticks)
    ax.set_xticklabels([_fmt(t) for t in ticks])
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.tick_params(axis='x', length=3, pad=2)


def _check_layout(fig, axes, legend, tag='fq1'):
    """Every text inside the canvas, no two texts overlapping, the legend clear of
    both panels (their tick labels and axis labels included)."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    texts = [t for t in fig.findobj(Text)
             if t.get_visible() and t.get_text().strip() and t.figure is fig]
    boxes = []
    for t in texts:
        b = t.get_window_extent(rend)
        if b.x0 < -0.5 or b.y0 < -0.5 or b.x1 > W + 0.5 or b.y1 > H + 0.5:
            raise SystemExit(f'{tag} REFUSING: text {t.get_text()!r} leaves the canvas '
                             f'({b.x0:.0f}, {b.y0:.0f}, {b.x1:.0f}, {b.y1:.0f}; '
                             f'canvas {W:.0f} x {H:.0f})')
        boxes.append((t.get_text(), b))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][1], boxes[j][1]
            if a.x0 + 1 < b.x1 and b.x0 + 1 < a.x1 and a.y0 + 1 < b.y1 and b.y0 + 1 < a.y1:
                raise SystemExit(f'{tag} REFUSING: texts {boxes[i][0]!r} and '
                                 f'{boxes[j][0]!r} overlap')
    lb = legend.get_window_extent(rend)
    for ax in axes:
        tb = ax.get_tightbbox(rend)
        if lb.overlaps(tb):
            raise SystemExit(f'{tag} REFUSING: the legend touches a panel')
    return len(boxes)


def _shown(path):
    """A path as the run prints it: relative to the repository when inside it."""
    a = os.path.abspath(path)
    return os.path.relpath(a, ROOT) if a.startswith(ROOT + os.sep) else path


def _n_seeds(seeds):
    return '?' if not seeds else str(len(seeds))


def fq1_inputs(factory_json, public_json, public_set, factory_macros=None,
               public_macros=None, factory_seeds='auto', seedext_json=None,
               seedext_macros=None, public_family='active', allow_partial=False):
    """The rows the figure plots, after every gate: F for panel (a), P for panel (b),
    the gate table, the line that names the records, and dev (a ten-seed record of a
    dry run, drawn only with allow_partial). In mode ten, F's policy rows (the eight
    combinations and the pooled row) come from the ten-seed record and its control
    rows from the five-seed record; in mode five every row of F comes from the
    five-seed record."""
    if public_family not in FQ1_PUBLIC:
        raise SystemExit(f'fq1 REFUSING: public family {public_family!r} is not one of '
                         f'{sorted(FQ1_PUBLIC)}')
    mode = fq1_seed_mode(factory_seeds, seedext_json)
    F = load_factory(factory_json)
    P = load_public(public_json, public_set)
    want = FQ1_PUBLIC[public_family][2]
    if P['family'] != public_family or P['prefix'] != want:
        raise SystemExit(f'fq1 REFUSING: {public_json} is the {P["family"]} family (macro '
                         f'prefix {P["prefix"]}), not the {public_family} family ({want})')
    print(f'  factory: {factory_json} ({F["setting"]})')
    print(f'  public:  {public_json} ({P["setting"]})')
    dev = False
    if mode == 'ten':
        T = load_seedext(seedext_json, F)
        if T['dry']:
            if not allow_partial:
                raise SystemExit(f'fq1 REFUSING: {seedext_json} is the record of a dry run '
                                 '(synthetic arrays); it is drawn only with --allow-partial')
            dev = True
        five_seeds = F['seeds']
        F = dict(F, rows=T['rows'], pool=T['pool'], seeds=T['seeds'],
                 setting=T['setting'])
        table = gate_seedext(F, seedext_macros) + gate_factory(F, factory_macros,
                                                                policy=False)
        policy = (f'{_shown(seedext_json)} ({_n_seeds(T["seeds"])} training seeds'
                  + (', dry run' if dev else '') + ')')
        control = (f'the five-seed record {_shown(factory_json)} '
                   f'({_n_seeds(five_seeds)} training seeds)')
    else:
        table = gate_factory(F, factory_macros)
        policy = f'{_shown(factory_json)} ({_n_seeds(F["seeds"])} training seeds)'
        control = f'the same five-seed record ({_n_seeds(F["seeds"])} training seeds)'
    table += gate_public(P, public_macros, public_set)
    line = (f'  fq1 records: (a) factory benchmark (--factory-seeds {factory_seeds}: '
            f'{mode}), policy rows from {policy}, the control rows of variant B from '
            f'{control} in every mode; (b) public instances, {public_family} family, '
            f'policy and control rows from {_shown(public_json)} '
            f'({_n_seeds(P["seeds"])} training seeds)')
    return dict(F=F, P=P, table=table, line=line, mode=mode, dev=dev)


def fq1_specialists(factory_json, public_json, public_set, out_dir,
                    factory_macros=None, public_macros=None, factory_seeds='auto',
                    seedext_json=None, seedext_macros=None, public_family='active',
                    allow_partial=False):
    """Draw variants A and B of the figure, then print the plotted numbers against
    the macro files."""
    X = fq1_inputs(factory_json, public_json, public_set, factory_macros, public_macros,
                   factory_seeds, seedext_json, seedext_macros, public_family,
                   allow_partial)
    F, P, table = X['F'], X['P'], X['table']
    print(X['line'])
    suffix = '_dev' if X['dev'] else ''

    # Rows top to bottom; the pooled row after a gap of 0.7 row.
    ys = np.arange(len(COMBOS), dtype=float)
    y_pool = len(COMBOS) + 0.7

    os.makedirs(out_dir, exist_ok=True)
    written = []
    for variant in ('A', 'B'):
        with_ctl = variant == 'B'
        vals = []
        for D in (F, P):
            items = [D['rows'][r] for r in COMBOS] + [D['pool']]
            if with_ctl:
                items += list(D['ctl'].values()) + [D['ctl_pool']]
            vals += [v for it in items if it is not None for v in (it['lo'], it['hi'])]
        xlim, ticks = _xaxis(vals)
        d_pol, d_ctl = (-0.17, 0.17) if with_ctl else (0.0, 0.0)

        # Layout in inches (printed size), bottom to top.
        fig_h = 3.18 if not with_ctl else 3.45
        left, right_room, gap = 0.80, 0.62, 0.20
        pw = (FIG_W - left - right_room - gap) / 2
        leg_y, ax_y0 = 0.04, 0.74
        ax_h = fig_h - ax_y0 - 0.30
        fig = plt.figure(figsize=(FIG_W, fig_h), dpi=300)
        ax = fig.add_axes([left / FIG_W, ax_y0 / fig_h, pw / FIG_W, ax_h / fig_h])
        bx = fig.add_axes([(left + pw + gap) / FIG_W, ax_y0 / fig_h, pw / FIG_W,
                           ax_h / fig_h], sharey=ax)
        for a, D, title in ((ax, F, '(a) Factory benchmark'),
                            (bx, P, '(b) Public instances')):
            _panel(a, D, ys, y_pool, xlim, ticks, with_ctl, d_pol, d_ctl)
            a.set_title(title, fontsize=FS_TITLE, pad=7)
        ax.set_ylim(y_pool + 0.65, -0.65)
        ax.set_yticks(list(ys) + [y_pool])
        ax.set_yticklabels([comb_tex(r) for r in COMBOS] + [POOL_LABEL])
        ax.set_ylabel('rule combination', labelpad=6)
        plt.setp(bx.get_yticklabels(), visible=False)

        # One axis title for both panels, centred under the pair.
        xc = (left + pw + gap / 2) / FIG_W
        who = 'policy or control' if with_ctl else 'policy'
        fig.text(xc, (ax_y0 - 0.30) / fig_h,
                 f'makespan difference, {who} minus specialist (%)',
                 ha='center', va='center', fontsize=FS_LABEL)

        # The two held-out rows, marked once: a thin bracket right of panel (b).
        tr = mtransforms.blended_transform_factory(bx.transAxes, bx.transData)
        i0, i1 = COMBOS.index(HELD[0]), COMBOS.index(HELD[-1])
        if [COMBOS[i] for i in range(i0, i1 + 1)] != HELD:
            raise SystemExit('fq1 REFUSING: the held-out rows are not adjacent')
        bx0, bx1 = 1.0 + 0.06 / pw, 1.0 + 0.11 / pw
        y0, y1 = ys[i0] - 0.38, ys[i1] + 0.38
        bx.plot([bx0, bx1, bx1, bx0], [y0, y0, y1, y1], color='black', lw=0.6,
                transform=tr, clip_on=False, solid_capstyle='butt')
        bx.text(bx1 + 0.06 / pw, (ys[i0] + ys[i1]) / 2, 'held out', transform=tr,
                ha='left', va='center', fontsize=FS_TICK)

        # One legend, below the panels, in a thin box.
        hs = [Patch(facecolor=BAND_FILL, edgecolor='none', label='\u00b11% margin'),
              Line2D([0], [0], color=LINE_HALF, lw=0.7, ls=(0, (3, 2)),
                     label='\u00b10.5% margin'),
              Line2D([0], [0], color=C_EQUIV, lw=1.4, marker='o', ms=MS_CIRCLE,
                     mec='white', mew=0.5, label='equivalent'),
              Line2D([0], [0], color=C_OPEN, lw=1.4, marker='s', ms=MS_SQUARE,
                     mfc='white', mec=C_OPEN, mew=1.0, label='equivalence not shown')]
        if with_ctl:
            hs.append(Line2D([0], [0], color=C_CONTROL, lw=1.0, marker='|', ms=5.0,
                             mew=1.2, label='single-rule control'))
        leg = fig.legend(handles=hs, loc='lower center',
                         bbox_to_anchor=(xc, leg_y / fig_h), ncol=len(hs),
                         fontsize=FS_TICK, frameon=True, fancybox=False,
                         edgecolor='0.6', framealpha=1.0, handlelength=1.8,
                         handletextpad=0.5, columnspacing=1.4, borderpad=0.4)
        leg.get_frame().set_linewidth(0.5)

        n_text = _check_layout(fig, (ax, bx), leg)
        stem = os.path.join(out_dir, f'fq1_specialists_{variant}{suffix}')
        fig.savefig(stem + '.pdf')
        fig.savefig(stem + '.png', dpi=300)
        plt.close(fig)
        written += [stem + '.pdf', stem + '.png']
        print(f'  variant {variant}: {FIG_W:.2f} x {fig_h:.2f} in, x range '
              f'{xlim[0]:+g} to {xlim[1]:+g}, {n_text} texts checked, no overlap')

    # ---- the plotted numbers against the macros
    print('\n  family   row                mean     lo      hi     verdict(1%)          '
          'macro mean/lo/hi/verdict          check')
    for fam, label, it, got, status in table:
        m = ('-' if got is None else
             f'{got[""]}/{got["Lo"]}/{got["Hi"]}/{got["VOne"]}')
        print(f'  {fam:8s} {label:18s} {it["mean"]:+.2f}  {it["lo"]:+.2f}  '
              f'{it["hi"]:+.2f}  {it["verdict"]:20s} {m:33s} {status}')
    for D, fam in ((F, 'factory'), (P, 'public')):
        for m in D['missing']:
            print(f'  not in the {fam} file, left out: {m}')
    return written


# ------------------------------------------------------------------ fq3: time budget
FQ3_SIZES = ('50x25n30', '80x25n30')
FQ3_SIZE_TEXT = {'50x25n30': '50 modules', '80x25n30': '80 modules'}
FQ3_ARMS = {'memetic_rule': ('memetic', 'rule'), 'cpsat_rule': ('cpsat', 'rule'),
            'memetic_policy': ('memetic', 'policy'), 'cpsat_policy': ('cpsat', 'policy')}
# Method by colour: the house blue and a medium rose whose grey value differs from
# the blue's by 0.18 of relative luminance, so a greyscale print keeps them apart.
FQ3_METHOD = {'memetic': ('memetic algorithm', '#d07a87'),
              'cpsat': ('CP-SAT solver', C_EQUIV)}
FQ3_START = {'rule': ('started from the best rule', '-'),
             'policy': ('started from the policy’s schedule', (0, (4, 2)))}
FQ3_LW, FQ3_BAND_ALPHA = 1.3, 0.18
FQ3_MARK = dict(marker='D', ms=4.4, color='black', mec='white', mew=0.5)
FQ3_SIGN = 'first-named method minus second, in percent of the second'
FQ3_TOL = 1e-6                     # the CSV keeps six decimals
FQ3_NEEDED = ('run', 'row', 'origin', 'size', 'combination', 'method', 't_s', 'mean_pct',
              'lo_pct', 'hi_pct', 'n_combinations')
# The clock of the file (column origin): request, the time since the request for a
# schedule (p94's default), or start, the wall clock from the start of each run.
FQ3_ORIGINS = ('request', 'start')
FQ3_NEEDED_REQUEST = ('held', 'policy_budget_s', 'policy_first_s', 'rule_first_s')
# The policy with sampled schedules (request clock, row Sel): black, its interval a pale
# grey band; variants A and B draw it, B's companion does not.
FQ3_SAMPLE = 'policy_sample'
FQ3_SAMPLE_VARIANTS = ('A', 'B')
FQ3_SAMPLE_TEXT = 'policy, sampled schedules'
FQ3_FIRST_TEXT = 'policy, first schedule'
FQ3_STYLE_KEY = '0.45'           # request clock: the line style keys, so black is the policy
# Request clock: ticks at 10 s and the reported budgets, and at one of these on the left
# when the earliest curve starts at most a factor FQ3_LEFT_SPAN above it.
FQ3_REQUEST_TICKS = (10.0,)
FQ3_LEFT_TICKS, FQ3_LEFT_SPAN = (3.0, 2.0), 1.7
# (file name, [(arm, row of the CSV)]): A all four arms with the selected run's
# intervals; B the rule-started arms over every training run; B's companion the
# policy-started arms, for the supplement.
FQ3_VARIANTS = (
    ('A', [('memetic_rule', 'Sel'), ('cpsat_rule', 'Sel'),
           ('memetic_policy', 'Sel'), ('cpsat_policy', 'Sel')]),
    ('B', [('memetic_rule', 'All'), ('cpsat_rule', 'All')]),
    ('B_supplement', [('memetic_policy', 'Sel'), ('cpsat_policy', 'Sel')]),
)
# Column group of the CSV (the groups of p94_timed_inference.GROUPS): group -> (the value
# of column combination on its pooled rows, which is also the JSON key of its entries; its
# combinations). The rows per combination and of Pool are group all; a file without the
# column is group all throughout.
FQ3_GROUPS = {'all': ('Pool', tuple(COMBOS)),
              'no_setups': ('NoSetup', ('N', 'L', 'W', 'LW')),
              'setups': ('Setup', ('S', 'LS', 'SW', 'LSW'))}
# The memetic algorithm as the best of eight runs on eight cores (--memetic-parallel): the
# record of p124 and the curves file it names; per arm of row Sel, the method of that file
# (which is also the column of its decision table on eight cores under the arm's name).
FQ3_MPAR_GENERATOR = 'scripts/p124_memetic_parallel.py'
FQ3_MPAR_METHODS = {'memetic_rule': 'memetic_rule_best8',
                    'memetic_policy': 'memetic_policy_best8'}
FQ3_MPAR_RUNS = 8
FQ3_MPAR_BUDGETS = (60.0, 300.0, 600.0)
FQ3_MPAR_NEEDED = FQ3_NEEDED + FQ3_NEEDED_REQUEST + ('group', 'policy', 'solver', 'start',
                                                     'n_runs', 'held')
# The memetic algorithm with a one-minute budget, read one minute after the request
# (--memetic-minute): the record of p137; per start schedule, the comparison of that record
# (the best of eight one-minute runs minus the policy's first schedule, in percent of it) and
# the face of its circle at x = 60 s in the variants that draw it. The bar of a circle is its
# 95% interval, behind the circle; both lie above every curve.
FQ3_MMIN_GENERATOR = 'scripts/p137_memetic_one_minute.py'
FQ3_MMIN_AT = 60.0
FQ3_MMIN_BUDGET = 60.0
FQ3_MMIN_ENTRIES = (('rule', 'm60_vs_pol'), ('policy', 'mp60_vs_pol'))
FQ3_MMIN_TEXT = 'memetic algorithm, one-minute budget'
FQ3_MMIN_VARIANTS = ('A',)
FQ3_MMIN_MARK = dict(marker='o', ms=4.4, mew=0.9)
FQ3_MMIN_BAR_LW = 0.8
FQ3_MMIN_Z = 6.5                 # the bars; the circles one above
FQ3_MMIN_FACE = {'rule': None, 'policy': 'white'}     # None: filled in the line's colour
FQ3_MMIN_ROWS = 3                # legend rows of a variant that draws the circles
# The legend column whose third row names the one-minute budget: the column of the start
# schedules, whose keys carry the same circles (under the methods the legend is wider than
# the two panels).
FQ3_MMIN_LEG_COL = 1
FQ3_LEG_ROW = 0.166              # in, the height of one more legend row
# The memetic algorithm with a five-minute budget, read five minutes after the request
# (--memetic-five, beside --memetic-minute): the record of p137 --budget 300, per start schedule
# the comparison of that record, drawn at x = 300 s as the circles of the one-minute budget (same
# colour, faces, size, bars and z-order). The legend row of the one-minute budget then names both
# budgets: each circle's budget equals its time since the request.
FQ3_MFIV_AT = 300.0
FQ3_MFIV_BUDGET = 300.0
FQ3_MFIV_ENTRIES = (('rule', 'm300_vs_pol'), ('policy', 'mp300_vs_pol'))
FQ3_MFIV_TEXT = 'memetic algorithm, budget equal to the time'
FQ3_BUDGET_WORD = {FQ3_MMIN_AT: 'one-minute', FQ3_MFIV_AT: 'five-minute'}


def _fq3_refuse(msg):
    raise SystemExit(f'fq3 REFUSING: {msg}')


def _fq3_seconds(v):
    return None if v == '' else float(v)


def load_timed(csv_path, json_path, allow_partial, group='all'):
    """The pooled curves of the CSV, keyed (row, size, method), and the JSON of the
    same run, after the run, origin, sign, pooling and no-rise gates; on the request
    clock also the per-size times of the CSV (policy_first_s, rule_first_s,
    policy_budget_s) and the held flags of the sampled schedules. Only the rows of
    `group` (column group; all when the file has no such column) are read: the pooled
    rows of all are those of Pool over the eight combinations, the pooled rows of a group
    those of its key (FQ3_GROUPS), whose JSON entry must list the same combinations."""
    if group not in FQ3_GROUPS:
        _fq3_refuse(f'group {group!r} is not one of {sorted(FQ3_GROUPS)}')
    pool, members = FQ3_GROUPS[group]
    for p in (csv_path, json_path):
        if not os.path.exists(p):
            _fq3_refuse(f'{p} not found (written by p94_timed_inference.py)')
    with open(csv_path, newline='') as f:
        rd = csv.DictReader(f)
        lack = [c for c in FQ3_NEEDED if c not in (rd.fieldnames or [])]
        if lack:
            _fq3_refuse(f'{csv_path} has no column {lack}')
        rows = list(rd)
        fields = list(rd.fieldnames)
    if 'group' in fields:
        rows = [r for r in rows if r['group'] == group]
        if not rows:
            _fq3_refuse(f'{csv_path} holds no row of group {group}')
    elif group != 'all':
        _fq3_refuse(f'{csv_path} has no column group, so it holds no rows of group {group}')
    with open(json_path) as f:
        J = json.load(f)
    if group != 'all':
        entry = (J.get('groups') or {}).get('entries', {}).get(pool)
        if not isinstance(entry, dict) or list(entry.get('combinations') or []) != list(members):
            _fq3_refuse(f'the JSON entry groups.entries.{pool} is '
                        f'{"absent" if entry is None else entry.get("combinations")}, not the '
                        f'combinations {list(members)} of group {group}')
    runs = sorted({r['run'] for r in rows})
    partial = runs != ['full']
    if partial and not allow_partial:
        _fq3_refuse(f'{csv_path} is not a full run (run column {runs}); a partial run, or '
                    'one with a substituted policy, is drawn only with --allow-partial')
    if runs != [J.get('run')]:
        _fq3_refuse(f'the CSV (run {runs}) and the JSON (run {J.get("run")!r}) are not '
                    'one output of p94')
    origins = sorted({r['origin'] for r in rows})
    if origins != [J.get('origin')]:
        _fq3_refuse(f'the CSV (origin {origins}) and the JSON (origin '
                    f'{J.get("origin")!r}) are not one output of p94')
    if len(origins) != 1 or origins[0] not in FQ3_ORIGINS:
        _fq3_refuse(f'clock origin {origins}; the figure draws the request origin (time '
                    'since the request for a schedule) or the start origin (wall clock from '
                    'the start of each run)')
    request = origins == ['request']
    if request:
        lack = [c for c in FQ3_NEEDED_REQUEST if c not in fields]
        if lack:
            _fq3_refuse(f'{csv_path} is on the request clock but has no column {lack}')
    if not str(J.get('sign', '')).startswith(FQ3_SIGN):
        _fq3_refuse(f'the JSON states the sign {J.get("sign")!r}, not {FQ3_SIGN!r}')
    sizes = {}
    if request:
        # one value per size of the times the figure and the gate read
        for size in sorted({r['size'] for r in rows}):
            got = {}
            for col in ('policy_first_s', 'rule_first_s', 'policy_budget_s'):
                vals = sorted({r[col] for r in rows if r['size'] == size})
                if len(vals) != 1:
                    _fq3_refuse(f'{size}: the column {col} holds {vals}, not one value')
                got[col] = _fq3_seconds(vals[0])
            sizes[size] = got
    names = set(FQ3_ARMS) | ({FQ3_SAMPLE} if request else set())
    grouped, notes = {}, []
    for r in rows:
        if r['combination'] == pool and r['method'] in names:
            grouped.setdefault((r['row'], r['size'], r['method']), []).append(r)
    curves = {}
    for key, rr in grouped.items():
        nc = sorted({int(r['n_combinations']) for r in rr})
        if nc != [len(members)]:
            msg = f'{key}: the pooled row pools {nc} combinations, not {len(members)}'
            if not partial:
                _fq3_refuse(msg)
            notes.append(msg)
        rr = sorted(rr, key=lambda r: float(r['t_s']))
        t = np.array([float(r['t_s']) for r in rr])
        if len(np.unique(t)) != len(t):
            _fq3_refuse(f'{key}: a time appears twice')
        c = dict(t=t, **{k: np.array([float(r[f'{k}_pct']) for r in rr])
                         for k in ('mean', 'lo', 'hi')})
        rise = np.flatnonzero(np.diff(c['mean']) > FQ3_TOL)
        if rise.size:
            i = int(rise[0])
            msg = (f'{key}: the pooled mean rises from {c["mean"][i]:+.6f} at {t[i]:g} s to '
                   f'{c["mean"][i + 1]:+.6f} at {t[i + 1]:g} s, but an incumbent only '
                   'improves (misread file or wrong sign)')
            if not partial:
                _fq3_refuse(msg)
            notes.append(msg)
        if ((c['lo'] > c['mean'] + FQ3_TOL) | (c['mean'] > c['hi'] + FQ3_TOL)).any():
            _fq3_refuse(f'{key}: a mean lies outside its interval')
        if request:
            flags = {r['held'] for r in rr}
            if not flags <= {'yes', 'no'}:
                _fq3_refuse(f'{key}: the column held holds {sorted(flags)}, not yes or no')
            c['held'] = np.array([r['held'] == 'yes' for r in rr])
            if key[2] == FQ3_SAMPLE:
                if key[0] != 'Sel':
                    _fq3_refuse(f'{key}: the sampled schedules exist for row Sel only')
                bud = sizes[key[1]]['policy_budget_s']
                after = t > bud + 1e-9 if bud is not None else np.zeros(len(t), bool)
                if (c['held'] != after).any():
                    _fq3_refuse(f'{key}: held is not yes exactly on the rows after the '
                                f'budget of the sampled schedules ({bud} s)')
                up = np.flatnonzero(c['mean'] > FQ3_TOL)
                if up.size:
                    msg = (f'{key}: the sampled schedules lie {c["mean"][up[0]]:+.6f} above the '
                           f'greedy schedule at {t[up[0]]:g} s, but the best sampled schedule '
                           'includes the greedy one (misread file or wrong sign)')
                    if not partial:
                        _fq3_refuse(msg)
                    notes.append(msg)
            elif c['held'].any():
                _fq3_refuse(f'{key}: a search arm has a row with held = yes')
        curves[key] = c
    budgets = [float(b) for b in J.get('budgets', [])]
    per = J.get('budgets_per_size')
    shown = [s for s in FQ3_SIZES if s in sizes]
    if request and isinstance(per, dict) and shown and all(s in per for s in shown):
        budgets = sorted({float(b) for s in shown for b in per[s]})
    return dict(curves=curves, json=J, run=runs[0], partial=partial, origin=origins[0],
                budgets=budgets, notes=notes, sizes=sizes, group=group, pool=pool)


def _fq3_at(c, t):
    k = np.flatnonzero(np.abs(c['t'] - t) < 1e-9)
    return None if k.size == 0 else tuple(float(c[a][k[0]]) for a in ('mean', 'lo', 'hi'))


def gate_timed(T):
    """The plotted values against the JSON, where the JSON holds them in the CSV's
    order: policy-started arms at every reported budget (comparison ii, arm minus
    policy), rule-started arms at time zero (comparison iv, best rule minus policy)
    and at every reported budget (comparison vi, arm minus policy). A full run must
    hold comparison vi; a partial file written before p94 stored it is drawn
    without that part of the gate. The entries read are those of the curves' group (Pool,
    or the group's key). On the request clock: _gate_timed_request."""
    if T['origin'] == 'request':
        return _gate_timed_request(T)
    R = T['json'].get('results', {})
    pk = T['pool']
    plan = []
    for size in FQ3_SIZES:
        S = R.get(size, {})
        if 'vi' not in S and not T['partial']:
            _fq3_refuse(f'results.{size}.vi is not in the JSON, so the rule-started arms '
                        'cannot be held against it at the reported budgets')
        for m in ('memetic', 'cpsat'):
            for b in T['budgets']:
                plan.append((('Sel', size, f'{m}_policy'), b,
                             S.get('ii', {}).get(m, {}).get(f'{b:g}', {}).get(pk),
                             f'results.{size}.ii.{m}.{b:g}.{pk}'))
            for key, row in (('selected', 'Sel'), ('all_seeds', 'All')):
                plan.append(((row, size, f'{m}_rule'), 0.0,
                             S.get('iv', {}).get(key, {}).get(pk),
                             f'results.{size}.iv.{key}.{pk}'))
                for b in T['budgets'] if 'vi' in S else ():
                    plan.append(((row, size, f'{m}_rule'), b,
                                 S['vi'].get(m, {}).get(f'{b:g}', {}).get(key, {}).get(pk),
                                 f'results.{size}.vi.{m}.{b:g}.{key}.{pk}'))
    table, bad = [], []
    for ck, t, src, where in plan:
        c = T['curves'].get(ck)
        if c is None:
            continue                    # not in the CSV: a variant that needs it refuses
        got = _fq3_at(c, t)
        if got is None or src is None:
            bad.append(f'{ck} at {t:g} s: ' + ('no CSV row' if got is None else
                                              f'{where} not in the JSON'))
            continue
        ref = tuple(float(src[k]) for k in ('mean', 'lo', 'hi'))
        ok = max(abs(g - r) for g, r in zip(got, ref)) <= FQ3_TOL
        if not ok:
            bad.append(f'{ck} at {t:g} s: CSV {got}, {where} {ref}')
        table.append((ck, t, got, where, 'match' if ok else 'MISMATCH'))
    if bad:
        _fq3_refuse('the CSV disagrees with the JSON:\n  ' + '\n  '.join(bad))
    return table


def _fq3_dig(d, path):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def _gate_timed_request(T):
    """Request clock, per production size the CSV holds: its times (policy_first_s,
    rule_first_s, policy_budget_s) against results.<size>.ix and policy_budget_s, and
    the plotted values at every budget of budgets_per_size[<size>] against ii
    (policy-started arms), vi (rule-started arms, rows Sel and All) and, inside the
    budget of the sampled schedules, vii. An entry holding no_schedule_yet is skipped;
    a missing key or entry is refused. The entries read are those of the curves' group
    (Pool and ix itself, or the group's key in both). Returns (curve key or time, budget,
    CSV values, JSON key, status) per check."""
    J = T['json']
    per, R = J.get('budgets_per_size'), J.get('results')
    pk = T['pool']
    ix = ('ix',) if T['group'] == 'all' else ('ix', pk)
    table, bad, plan = [], [], []
    for size in FQ3_SIZES:
        if not any(k[1] == size for k in T['curves']):
            continue                    # not in the CSV: a variant that needs it refuses
        if not isinstance(per, dict) or size not in per:
            _fq3_refuse(f'budgets_per_size.{size} is not in the JSON')
        S = R.get(size) if isinstance(R, dict) else None
        if not isinstance(S, dict):
            _fq3_refuse(f'results.{size} is not in the JSON')
        info = T['sizes'][size]
        for col, path in (('policy_first_s', ix + ('policy', 't', 'median_s')),
                          ('rule_first_s', ix + ('rules', 't_all', 'median_s')),
                          ('policy_budget_s', ('policy_budget_s',))):
            v, src = info[col], _fq3_dig(S, path)
            where = f'results.{size}.' + '.'.join(path)
            if v is None and src is None and T['partial']:
                table.append((('time', size, col), None, None, where, 'absent, partial run'))
                continue
            if v is None or src is None:
                bad.append(f'{size} {col}: CSV {v}, {where} {src}')
                continue
            ok = abs(v - float(src)) <= FQ3_TOL * max(1.0, abs(float(src)))
            if not ok:
                bad.append(f'{size} {col}: CSV {v}, {where} {src}')
            table.append((('time', size, col), None, (v,), where, 'match' if ok else 'MISMATCH'))
        bud = info['policy_budget_s']
        for b in sorted(float(x) for x in per[size]):
            bk = f'{b:g}'
            for m in ('memetic', 'cpsat'):
                plan.append((S, size, ('Sel', size, f'{m}_policy'), b, ('ii', m, bk, pk)))
                for key, row in (('selected', 'Sel'), ('all_seeds', 'All')):
                    plan.append((S, size, (row, size, f'{m}_rule'), b,
                                 ('vi', m, bk, key, pk)))
            if bud is not None and b <= bud + 1e-9:
                plan.append((S, size, ('Sel', size, FQ3_SAMPLE), b, ('vii', bk, pk)))
    for S, size, ck, b, path in plan:
        c = T['curves'].get(ck)
        if c is None:
            continue                    # not in the CSV: a variant that needs it refuses
        src, where = _fq3_dig(S, path), f'results.{size}.' + '.'.join(path)
        if not isinstance(src, dict):
            bad.append(f'{ck} at {b:g} s: {where} not in the JSON')
            continue
        if 'no_schedule_yet' in src:
            table.append((ck, b, None, where, 'no schedule yet'))
            continue
        if any(k not in src for k in ('mean', 'lo', 'hi')):
            bad.append(f'{ck} at {b:g} s: {where} has no mean, lo and hi')
            continue
        got = _fq3_at(c, b)
        if got is None:
            bad.append(f'{ck} at {b:g} s: no CSV row, but {where} holds a value')
            continue
        ref = tuple(float(src[k]) for k in ('mean', 'lo', 'hi'))
        ok = max(abs(g - r) for g, r in zip(got, ref)) <= FQ3_TOL
        if not ok:
            bad.append(f'{ck} at {b:g} s: CSV {got}, {where} {ref}')
        table.append((ck, b, got, where, 'match' if ok else 'MISMATCH'))
    if bad:
        _fq3_refuse('the CSV disagrees with the JSON:\n  ' + '\n  '.join(bad))
    return table


def load_memetic_parallel(json_path, T, timed_csv, allow_partial):
    """The best-of-eight curves of the record of p124 (json_path) and of the curves file its
    block decision_table.curves names, taken from the same folder, after the gates against the
    timed record T (load_timed of timed_csv, request clock): generator, run (a preview or a dry
    run only with allow_partial), clock, selected training run, curve grid, the per-size times,
    n_runs, both methods at both sizes (the methods the record lists, in a preview), the curve
    gates of load_timed, and the values at 60, 300 and 600 s against the decision table on
    eight cores at its printed digits. Only the rows of T's group are drawn. Returns curves
    keyed (Sel, size, arm), the arms they replace, the flags preview and dry, the CSV path
    and the decision-table checks (curve key, budget, value, printed value, status)."""
    J94 = T['json']
    if T['origin'] != 'request':
        _fq3_refuse(f'--memetic-parallel draws the best of eight on the request clock; the '
                    f'timed record is on the {T["origin"]} clock')
    if not os.path.exists(json_path):
        _fq3_refuse(f'{json_path} not found (written by p124_memetic_parallel.py)')
    with open(json_path) as f:
        J = json.load(f)
    if J.get('generator') != FQ3_MPAR_GENERATOR:
        _fq3_refuse(f'{json_path} names the generator {J.get("generator")!r}, not '
                    f'{FQ3_MPAR_GENERATOR}')
    run, dry = J.get('run'), bool(J.get('dry'))
    if run not in ('full', 'preview'):
        _fq3_refuse(f'{json_path} is not a full run of p124 (run {run!r}); its best of eight '
                    'takes cells with fewer than eight valid runs')
    preview = run == 'preview'
    if preview and not allow_partial:
        _fq3_refuse(f'{json_path} is a preview of p124 (the first set alone, not for the '
                    'paper); it is drawn only with --allow-partial, and its files end in _preview')
    if dry and not allow_partial:
        _fq3_refuse(f'{json_path} is a dry run of p124 (synthetic records); it is drawn only '
                    'with --allow-partial, and its files end in _dev')
    D = J.get('decision_table')
    C = D.get('curves') if isinstance(D, dict) else None
    if not isinstance(C, dict) or not isinstance(D.get('sizes'), dict):
        _fq3_refuse(f'{json_path} has no block decision_table with curves and sizes')
    clocks = [J.get('origin'), D.get('clock')]
    if clocks != ['request', 'request']:
        _fq3_refuse(f'{json_path}: origin {clocks[0]!r} and decision_table.clock {clocks[1]!r}, '
                    'not the request clock of the timed record')
    sel94 = (J94.get('selection') or {}).get('seed')
    if J.get('selected_run') != sel94:
        _fq3_refuse(f'{json_path} reads the selected training run {J.get("selected_run")!r}, '
                    f'the timed record s{sel94}')
    grid = [float(x) for x in C.get('grid') or []]
    grid94 = [float(x) for x in (J94.get('grid') or {}).get('curve') or []]
    if not grid94 or len(grid) != len(grid94) or \
            max(abs(a - b) for a, b in zip(grid, grid94)) > 1e-9:
        _fq3_refuse(f'{json_path}: decision_table.curves.grid ({len(grid)} times) is not the '
                    f'curve grid of the timed record (grid.curve, {len(grid94)} times)')
    if C.get('row') != 'Sel' or C.get('n_runs') != FQ3_MPAR_RUNS:
        _fq3_refuse(f'{json_path}: decision_table.curves has row {C.get("row")!r} and n_runs '
                    f'{C.get("n_runs")!r}, not row Sel and {FQ3_MPAR_RUNS} runs')
    methods = list(C.get('methods') or [])
    arm_of = {m: a for a, m in FQ3_MPAR_METHODS.items()}
    want = sorted(FQ3_MPAR_METHODS.values())
    if not methods or not set(methods) <= set(want) or (not preview and sorted(methods) != want):
        _fq3_refuse(f'{json_path}: decision_table.curves.methods {methods}, not '
                    f'{" and ".join(want)}' + (' (a preview: one or both)' if preview else ''))
    csv_path = os.path.join(os.path.dirname(os.path.abspath(json_path)),
                            os.path.basename(str(C.get('file') or '')))
    if not str(C.get('file') or '') or not os.path.exists(csv_path):
        _fq3_refuse(f'{csv_path}: the curves file that {json_path} names '
                    f'({C.get("file")!r}) is not beside it')
    with open(csv_path, newline='') as f:
        rd = csv.DictReader(f)
        lack = [c for c in FQ3_MPAR_NEEDED if c not in (rd.fieldnames or [])]
        if lack:
            _fq3_refuse(f'{csv_path} has no column {lack}')
        rows = list(rd)
    # every row of the file: one run of p124 on the request clock, row Sel, eight runs
    col_values = {c: sorted({r[c] for r in rows}) for c in
                  ('run', 'origin', 'row', 'n_runs', 'held', 'solver', 'method')}
    runs_ok = (col_values['run'] == ['full'] if not preview else
               len(col_values['run']) == 1 and col_values['run'][0].startswith('preview'))
    for col, ok, text in (
            ('run', runs_ok, 'the run of the JSON (' + ('a preview' if preview else 'full') + ')'),
            ('origin', col_values['origin'] == ['request'], 'request'),
            ('row', col_values['row'] == ['Sel'], 'Sel'),
            ('n_runs', col_values['n_runs'] == [str(FQ3_MPAR_RUNS)],
             f'{FQ3_MPAR_RUNS} on every row'),
            ('held', col_values['held'] == ['no'], 'no on every row (a search has no held value)'),
            ('solver', col_values['solver'] == ['memetic'], 'memetic'),
            ('method', col_values['method'] == sorted(methods),
             f'the methods of decision_table.curves.methods {methods}')):
        if not ok:
            _fq3_refuse(f'{csv_path}: the column {col} holds {col_values[col]}, not {text}')
    with open(timed_csv, newline='') as f:
        pol94 = sorted({r.get('policy') for r in csv.DictReader(f) if r.get('row') == 'Sel'})
    pol = sorted({r['policy'] for r in rows})
    if pol != pol94:
        _fq3_refuse(f'{csv_path}: the column policy holds {pol}, the rows Sel of the timed CSV '
                    f'{pol94}')
    pool, members = FQ3_GROUPS[T['group']]
    rows = [r for r in rows if r['group'] == T['group']]
    off = sorted({r['t_s'] for r in rows if min(abs(float(r['t_s']) - g) for g in grid) > 1e-9})
    if off:
        _fq3_refuse(f'{csv_path}: the times {off[:5]} s are not on the curve grid')
    for size in FQ3_SIZES:
        if size not in T['sizes']:
            _fq3_refuse(f'the timed CSV holds no rows of {size}')
        for col in ('policy_first_s', 'rule_first_s', 'policy_budget_s'):
            vals = sorted({r[col] for r in rows if r['size'] == size})
            ref = T['sizes'][size][col]
            got = _fq3_seconds(vals[0]) if len(vals) == 1 else None
            if len(vals) != 1 or (got is None) != (ref is None) or (
                    got is not None and abs(got - ref) > FQ3_TOL * max(1.0, abs(ref))):
                _fq3_refuse(f'{csv_path}, {size}: the column {col} holds {vals}, the timed CSV '
                            f'{ref} (group {T["group"]})')
    curves, checks, bad = {}, [], []
    for method in methods:
        arm = arm_of[method]
        for size in FQ3_SIZES:
            key = ('Sel', size, arm)
            rr = sorted((r for r in rows if r['size'] == size and r['method'] == method
                         and r['combination'] == pool), key=lambda r: float(r['t_s']))
            if not rr:
                _fq3_refuse(f'{csv_path} holds no pooled row ({pool}) of {method} at {size}; '
                            'the figure needs both methods at both sizes')
            if sorted({r['start'] for r in rr}) != [FQ3_ARMS[arm][1]]:
                _fq3_refuse(f'{csv_path}: {method} at {size} has the start '
                            f'{sorted({r["start"] for r in rr})}, not {FQ3_ARMS[arm][1]}')
            nc = sorted({int(r['n_combinations']) for r in rr})
            if nc != [len(members)]:
                _fq3_refuse(f'{csv_path}: {method} at {size} pools {nc} combinations, not '
                            f'{len(members)}')
            t = np.array([float(r['t_s']) for r in rr])
            if len(np.unique(t)) != len(t):
                _fq3_refuse(f'{csv_path}: {method} at {size}: a time appears twice')
            c = dict(t=t, held=np.zeros(len(t), bool),
                     **{k: np.array([float(r[f'{k}_pct']) for r in rr])
                        for k in ('mean', 'lo', 'hi')})
            rise = np.flatnonzero(np.diff(c['mean']) > FQ3_TOL)
            if rise.size:
                i = int(rise[0])
                _fq3_refuse(f'{csv_path}: {method} at {size}: the pooled mean rises from '
                            f'{c["mean"][i]:+.6f} at {t[i]:g} s to {c["mean"][i + 1]:+.6f} at '
                            f'{t[i + 1]:g} s, but an incumbent only improves')
            if ((c['lo'] > c['mean'] + FQ3_TOL) | (c['mean'] > c['hi'] + FQ3_TOL)).any():
                _fq3_refuse(f'{csv_path}: {method} at {size}: a mean lies outside its interval')
            start = ((C.get('start_s') or {}).get(size) or {}).get(method) if pool == 'Pool' else \
                (((C.get('group_start_s') or {}).get(size) or {}).get(method) or {}).get(pool)
            if start is None or abs(float(start) - t[0]) > 1e-9:
                _fq3_refuse(f'{csv_path}: {method} at {size} starts at {t[0]:g} s, the JSON at '
                            f'{start} s')
            rows_j = D['sizes'].get(size, {}).get('rows', {})
            for b in FQ3_MPAR_BUDGETS:
                where = f'decision_table.sizes.{size}.rows.{b:g}.{pool}.values.{arm}'
                text = ((rows_j.get(f'{b:g}') or {}).get(pool) or {}).get('values', {}).get(arm)
                got = _fq3_at(c, b)
                if got is None or text is None:
                    bad.append(f'{key} at {b:g} s: ' + ('no CSV row' if got is None else
                                                       f'{where} holds no number'))
                    continue
                ok = abs(got[0] - float(text)) <= 0.005 + FQ3_TOL
                if not ok:
                    bad.append(f'{key} at {b:g} s: CSV {got[0]:+.6f}, {where} {text}')
                checks.append((key, b, got[0], text, 'match' if ok else 'MISMATCH'))
            curves[key] = c
    if bad:
        _fq3_refuse('the curves file disagrees with the decision table on eight cores:\n  '
                    + '\n  '.join(bad))
    return dict(curves=curves, arms={arm_of[m] for m in methods}, preview=preview, dry=dry,
                csv=csv_path, checks=checks, json=J)


def load_memetic_minute(json_path, T, mpar):
    """The memetic algorithm with a one-minute budget, read one minute after the request, from
    the record of p137 (json_path): per size, the entries m60_vs_pol (started from the best rule)
    and mp60_vs_pol (started from the policy's first schedule) of results.<size>.comparisons,
    under the key of the curves' group (Pool, or the group's key), each the best of eight runs
    minus the policy's first schedule in percent of it, with its 95% interval. The record must
    name p137 as its generator and be a full run (a preview and a dry run are refused), on the
    request clock, read at 60 s with a budget of 60 s, and hold the selected training run, the
    number of instances and the number of bootstrap draws of the record of p124 (mpar, from
    load_memetic_parallel, which holds the training run of the timed record T). Every entry
    must pool the group's combinations over that number of instances, with finite values and
    its mean inside its interval. Returns {size: {start: (mean, lo, hi)}} and the keys read."""
    return _fq3_load_p137(json_path, T, mpar, FQ3_MMIN_AT, FQ3_MMIN_BUDGET, FQ3_MMIN_ENTRIES)


def load_memetic_five(json_path, T, mpar, preview=False):
    """The memetic algorithm with a five-minute budget, read five minutes after the request, from
    the record of p137 --budget 300 (json_path): per size, the entries m300_vs_pol (started from
    the best rule) and mp300_vs_pol (started from the policy's first schedule), with the gates of
    load_memetic_minute at 300 s in the place of 60 s; a record read at another time or with
    another budget (the one-minute record among them) is refused. preview: a preview of p137 is
    read as well (the test-only switch of fq3_time_budget, which the command line cannot reach); a
    dry run is refused in every case. Returns as load_memetic_minute, with the flag preview."""
    return _fq3_load_p137(json_path, T, mpar, FQ3_MFIV_AT, FQ3_MFIV_BUDGET, FQ3_MFIV_ENTRIES,
                          preview)


def _fq3_load_p137(json_path, T, mpar, at, budget, entries, preview=False):
    """The reader of load_memetic_minute and load_memetic_five: the record of p137 read `at` s
    after the request with a budget of `budget` s, the comparisons `entries` ((start, key) pairs)
    per size. preview: a preview of p137 is read as well."""
    word = FQ3_BUDGET_WORD[at]
    if not os.path.exists(json_path):
        _fq3_refuse(f'{json_path} not found (written by p137_memetic_one_minute.py'
                    + ('' if budget == FQ3_MMIN_BUDGET else f' --budget {budget:g}') + ')')
    with open(json_path) as f:
        J = json.load(f)
    if J.get('generator') != FQ3_MMIN_GENERATOR:
        _fq3_refuse(f'{json_path} names the generator {J.get("generator")!r}, not '
                    f'{FQ3_MMIN_GENERATOR}')
    run = J.get('run')
    if run == 'preview' and not preview:
        _fq3_refuse(f'{json_path} is a preview of p137 (not every seed of both sets); the '
                    f'{word} points are drawn from a full run only')
    if run != 'full' and not (preview and run == 'preview'):
        _fq3_refuse(f'{json_path} is not a full run of p137 (run {run!r})')
    if J.get('dry') is not False:
        _fq3_refuse(f'{json_path} is a dry run of p137 (synthetic records, field dry '
                    f'{J.get("dry")!r}); the {word} points are drawn from a full run only')
    clock = (J.get('origin'), J.get('read_at_s'), J.get('budget_s'))
    if clock[0] != 'request' or not all(isinstance(v, (int, float)) and
                                        abs(float(v) - w) < 1e-9 for v, w in
                                        zip(clock[1:], (at, budget))):
        _fq3_refuse(f'{json_path}: origin {clock[0]!r}, read_at_s {clock[1]!r} and budget_s '
                    f'{clock[2]!r}, not the request clock read at {at:g} s with a '
                    f'budget of {budget:g} s')
    P = mpar['json']
    n_inst = len(J.get('instances') or [])
    for what, mine, ref in (
            ('selected training run', J.get('selected_run'), P.get('selected_run')),
            ('number of instances', n_inst, len(P.get('instances') or [])),
            ('number of bootstrap draws', J.get('boot'), P.get('boot'))):
        if mine is None or mine != ref:
            _fq3_refuse(f'{json_path} has the {what} {mine!r}, the record of p124 that '
                        f'--memetic-parallel reads {ref!r}; the two records are not of one '
                        'setting')
    pool, members = FQ3_GROUPS[T['group']]
    R = J.get('results') if isinstance(J.get('results'), dict) else {}
    vals, keys = {}, []
    for size in FQ3_SIZES:
        for start, q in entries:
            where = f'results.{size}.comparisons.{q}.{pool}'
            e = _fq3_dig(R, (size, 'comparisons', q, pool))
            if not isinstance(e, dict) or any(k not in e for k in ('mean', 'lo', 'hi')):
                _fq3_refuse(f'{json_path}: {where} is not in the record or has no mean, lo and '
                            'hi')
            try:
                v = tuple(float(e[k]) for k in ('mean', 'lo', 'hi'))
            except (TypeError, ValueError):
                v = (float('nan'),) * 3
            if not np.isfinite(v).all() or not v[1] - FQ3_TOL <= v[0] <= v[2] + FQ3_TOL:
                _fq3_refuse(f'{json_path}: {where} holds mean {e.get("mean")!r} and interval '
                            f'[{e.get("lo")!r}, {e.get("hi")!r}], not a finite mean inside its '
                            'interval')
            if e.get('c') != len(members) or list(e.get('combinations') or []) != list(members) \
                    or e.get('n') != n_inst:
                _fq3_refuse(f'{json_path}: {where} pools {e.get("c")!r} combinations '
                            f'{e.get("combinations")!r} over {e.get("n")!r} instances, not the '
                            f'{len(members)} of group {T["group"]} over the record\'s {n_inst}')
            vals.setdefault(size, {})[start] = v
            keys.append(where)
    return dict(values=vals, keys=keys, path=json_path, pool=pool, preview=run == 'preview')


def load_policy_times(paths):
    """{size: (seconds, path)} from the valid entries of p75 records (sizes[<size>],
    worst-combination median) and of records that name one size themselves."""
    marks, notes = {}, []

    def put(size, seconds, p):
        if size in marks:
            _fq3_refuse(f'two records for {size}: {marks[size][1]} and {p}')
        marks[size] = (float(seconds), p)

    for p in paths:
        if not os.path.exists(p):
            notes.append(f'{p} not found')
            continue
        with open(p) as f:
            rec = json.load(f)
        if isinstance(rec.get('sizes'), dict):
            for size in FQ3_SIZES:
                shop = size[:-len('n30')] if size.endswith('n30') else size
                r = rec['sizes'].get(shop)
                if r is None:
                    notes.append(f'{p} holds no entry for {shop}, so {size} gets no mark')
                elif r.get('valid') is not True:
                    notes.append(f'{p} ({shop}): not a valid measurement (field valid = '
                                 f'{r.get("valid")!r}), no mark')
                else:
                    put(size, max(x['median_s'] for x in r['per_regime'].values()), p)
            continue
        size = rec.get('size')
        if size is None:
            notes.append(f'{p} names no size (fields sizes or size), so it gives no mark')
            continue
        if rec.get('valid') is not True:
            notes.append(f'{p} ({size}): not a valid measurement (field valid = '
                         f'{rec.get("valid")!r}), no mark')
            continue
        put(size, rec['worst_regime_median_s'], p)
    return marks, notes


def _fq3_xaxis(T, keys, marks):
    """From 1 s, or from 0.1 s when a curve moves or a mark lies before 1 s; the left
    end moves further left when a mark would sit within a sixth of a decade of the
    axis. Ticks at the reported budgets, and at 0.1 s when the axis starts there."""
    early = False
    for k in keys:
        c = T['curves'][k]
        s = c['t'] < 1.0
        for a in ('mean', 'lo', 'hi'):
            if s.any() and np.abs(c[a][s] - c[a][0]).max() > FQ3_TOL:
                early = True
    m_lo = min(marks.values(), default=np.inf)
    x0 = 0.1 if (early or m_lo < 1.0) else 1.0
    if m_lo < 1.5 * x0:
        x0 = float(m_lo / 1.5)
    ticks = sorted({b for b in T['budgets'] if b >= x0} | ({0.1} if x0 <= 0.1 else set()))
    return x0, max(T['budgets']) * 1.12, ticks


def _fq3_xaxis_request(T, keys, marks):
    """Request clock: from just below the earliest first schedule (the first time of a
    curve, divided by 1.25, or of a diamond, divided by 1.5 so that the diamond stays
    clear of the axis) to 1.12 times the largest budget. Ticks at 10 s and the reported
    budgets, and at 3 s (or else 2 s) when the earliest curve starts at most a factor
    FQ3_LEFT_SPAN above it; the axis then starts a little left of that tick."""
    s_curve = min(float(T['curves'][k]['t'][0]) for k in keys)
    x0 = min([s_curve / 1.25] + [m / 1.5 for m in marks.values()])
    left = set()
    for c in FQ3_LEFT_TICKS:
        if c <= s_curve + 1e-9 and s_curve / c <= FQ3_LEFT_SPAN:
            left, x0 = {c}, min(x0, c / 1.15)
            break
    top = max(T['budgets'])
    ticks = sorted({b for b in set(FQ3_REQUEST_TICKS) | set(T['budgets']) if x0 <= b <= top}
                   | left)
    return x0, top * 1.12, ticks


def _fq3_yaxis(vals):
    lo, hi = min(min(vals), 0.0), max(max(vals), 0.0)
    pad = max(0.06 * (hi - lo), 0.15)
    y0, y1 = lo - pad, hi + pad
    ticks = [t for t in MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]).tick_values(y0, y1)
             if y0 - 1e-9 <= t <= y1 + 1e-9]
    return (y0, y1), ticks


def _decimals(ticks):
    """The number of decimals that prints every tick exactly."""
    return max(len(f'{abs(round(t, 6)):g}'.partition('.')[2]) for t in ticks)


def _fq3_yaxis_request(vals):
    """As _fq3_yaxis; a data end more than 0.3 of a tick step beyond the outermost tick
    gets the next tick, so a curve that falls below zero meets a labelled tick. The lower
    limit then follows the data: the lowest band edge, rounded down to the step of the
    tick labels (0.1 when they print one decimal, 1 when they print none), and never
    below the limit the rule above gives; a tick below that limit is dropped, so no
    empty band remains under the curves. Returns the range, the ticks and their labels,
    all with one number of decimals."""
    (y0, y1), ticks = _fq3_yaxis(vals)
    lo, hi = min(min(vals), 0.0), max(max(vals), 0.0)
    if len(ticks) >= 2:
        step = ticks[1] - ticks[0]
        if lo < ticks[0] - 0.3 * step:
            ticks = [ticks[0] - step] + ticks
        if hi > ticks[-1] + 0.3 * step:
            ticks = ticks + [ticks[-1] + step]
        pad = 0.03 * (ticks[-1] - ticks[0])
        y0, y1 = min(y0, ticks[0] - pad), max(y1, ticks[-1] + pad)
    if lo < 0:
        q = 10.0 ** -_decimals(ticks)
        y0 = max(y0, float(np.floor(round(lo / q, 6))) * q)
        ticks = [t for t in ticks if t >= y0 - 1e-9]
    dec = _decimals(ticks)

    def text(v):
        v = round(v, 6)
        s = f'{abs(v):.{dec}f}'
        return f'{MINUS}{s}' if v < 0 else s
    return (y0, y1), ticks, text


def _fq3_panel(ax, T, arms, size, xlim, xticks, yticks, mark, points=None):
    """One panel. points: [(x, {start: (mean, lo, hi)})] of the fixed budgets at this size (the
    one-minute budget at x = FQ3_MMIN_AT, then the five-minute budget at x = FQ3_MFIV_AT), each
    drawn as circles in the memetic colour (filled: started from the best rule; open: from the
    policy's schedule), each over its 95% interval bar. Returns the circles and bars drawn,
    {(x, start): (circle, bar)}."""
    for b in xticks:
        ax.axvline(b, color=C_GRID, lw=0.4, zorder=0.5)
    for y in yticks:
        if abs(y) > 1e-9:
            ax.axhline(y, color=C_GRID, lw=0.4, zorder=0.5)
    ax.axhline(0, color=C_ZERO, lw=0.6, zorder=3)
    drawn = []
    for arm, row in arms:
        c = T['curves'][(row, size, arm)]
        k = c['t'] >= xlim[0] - 1e-12
        if arm == FQ3_SAMPLE:
            # the policy, black over a pale grey band, up to the budget of its run
            k = k & ~c['held']
            col, ls, z = 'black', '-', 5
        else:
            method, start = FQ3_ARMS[arm]
            col, ls, z = FQ3_METHOD[method][1], FQ3_START[start][1], 4
        ax.fill_between(c['t'][k], c['lo'][k], c['hi'][k], color=col,
                        alpha=FQ3_BAND_ALPHA, lw=0, zorder=1)
        drawn.append((c['t'][k], c['mean'][k], col, ls, z))
    for t, m, col, ls, z in drawn:
        ax.plot(t, m, color=col, lw=FQ3_LW, ls=ls, zorder=z, solid_capstyle='butt',
                dash_capstyle='butt')
    if mark is not None:
        ax.plot([mark], [0.0], ls='none', zorder=6, clip_on=False, **FQ3_MARK)
    drawn_minute = {}
    rose = FQ3_METHOD['memetic'][1]
    for at, values in points or []:
        for start, (m, lo, hi) in values.items():
            bar, = ax.plot([at, at], [lo, hi], color=rose, lw=FQ3_MMIN_BAR_LW,
                           zorder=FQ3_MMIN_Z, solid_capstyle='butt')
            dot, = ax.plot([at], [m], ls='none', color=rose, mec=rose,
                           mfc=FQ3_MMIN_FACE[start] or rose, zorder=FQ3_MMIN_Z + 0.1,
                           **FQ3_MMIN_MARK)
            drawn_minute[(at, start)] = (dot, bar)
    ax.set_xscale('log')
    ax.set_xlim(*xlim)
    ax.xaxis.set_major_locator(FixedLocator(xticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    ax.tick_params(axis='both', length=3, pad=2)
    return drawn_minute


def _fq3_spread(fig, axes, min_gap=6.0, most=2.5):
    """Two neighbouring time labels closer than min_gap points (300 and 600 on an axis
    from 0.1 s) each move apart by half the shortfall, at most `most` points."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    for ax in axes:
        labs = [t for t in ax.get_xticklabels() if t.get_visible() and t.get_text()]
        boxes = [t.get_window_extent(rend) for t in labs]
        for i in range(len(labs) - 1):
            short = min_gap - (boxes[i + 1].x0 - boxes[i].x1) * 72.0 / fig.dpi
            if short > 0:
                d = min(short / 2, most)
                for t, s in ((labs[i], -d), (labs[i + 1], d)):
                    t.set_transform(t.get_transform() + mtransforms.ScaledTranslation(
                        s / 72.0, 0, fig.dpi_scale_trans))


def _fq3_minute_check(fig, ax, drawn, values, mark, variant, size):
    """The circles of the fixed budgets in one panel (drawn and values keyed (x, start)): each
    drawn at its x (FQ3_MMIN_AT, FQ3_MFIV_AT) with the record's mean and its bar from the
    record's lo to hi; each inside the axes, clear of every other circle and of the policy's
    diamond (marks' extents in points)."""
    rend = fig.canvas.get_renderer()
    box = ax.get_window_extent(rend)
    px = fig.dpi / 72.0
    centres = {}
    for (at, start), (dot, bar) in drawn.items():
        m, lo, hi = values[(at, start)]
        name = f'the {FQ3_BUDGET_WORD[at]} circle started from the {start}'
        got = (float(dot.get_xdata()[0]), float(dot.get_ydata()[0]),
               *map(float, bar.get_xdata()), *map(float, bar.get_ydata()))
        want = (at, m, at, at, lo, hi)
        if max(abs(g - w) for g, w in zip(got, want)) > 1e-12:
            _fq3_refuse(f'variant {variant}, {size}: {name} is drawn at {got}, the record gives '
                        f'{want}')
        c = ax.transData.transform((at, m))
        r = (FQ3_MMIN_MARK['ms'] + FQ3_MMIN_MARK['mew']) / 2 * px
        if not (box.x0 + r <= c[0] <= box.x1 - r and box.y0 + r <= c[1] <= box.y1 - r):
            _fq3_refuse(f'variant {variant}, {size}: {name} leaves the axes or touches one')
        centres[name] = (c, r)
    others = list(centres.items())
    if mark is not None:
        others_d = (ax.transData.transform((mark, 0.0)),
                    (FQ3_MARK['ms'] * 0.71 + FQ3_MARK['mew'] / 2) * px)
    for i, (name, (c, r)) in enumerate(centres.items()):
        near = [(o, other) for other, o in others[i + 1:]]
        if mark is not None:
            near.append((others_d, "the policy's diamond"))
        for (o, ro), what in near:
            if np.hypot(*(c - o)) < r + ro:
                _fq3_refuse(f'variant {variant}, {size}: {name} touches {what}')


def fq3_time_budget(curves_csv, timed_json, policy_paths, out_dir, allow_partial=False,
                    group='all', memetic_parallel=None, memetic_minute=None, memetic_five=None,
                    *, test_only_five_preview=False):
    """Draw variants A and B and B's companion, then print the gated values. group: the
    rows of the CSV the curves come from (FQ3_GROUPS); the files of a group other than all
    carry its name (fq3_time_budget_A_setups.png), so the figure of all is never replaced.
    memetic_parallel: the record of p124 (load_memetic_parallel); its best-of-eight curves
    replace the memetic arms of row Sel after the timed record's own gate. A preview draws
    only the curves it holds, only the variants that show one of them, into files ending in
    _preview. memetic_minute: the record of p137 (load_memetic_minute), only with
    memetic_parallel; the variants of FQ3_MMIN_VARIANTS gain, at x = 60 s in each panel, the
    memetic algorithm with a one-minute budget from each start as a circle over its 95%
    interval bar, and a third legend row. memetic_five: the record of p137 --budget 300
    (load_memetic_five), only with memetic_minute; the same variants gain the same circles at
    x = 300 s with the five-minute budget, and the third legend row names both budgets.
    test_only_five_preview (keyword only, never set by the command line): the five-minute record
    may be a preview of p137, for a drawing outside the manuscript; the file names then end in
    _five_preview and out_dir must lie outside the manuscript's folder manuscript/."""
    if memetic_minute is not None and memetic_parallel is None:
        _fq3_refuse('--memetic-minute draws the one-minute budget beside the best of eight runs '
                    'of the record of p124; it needs --memetic-parallel')
    if memetic_five is not None and memetic_minute is None:
        _fq3_refuse('--memetic-five draws the five-minute budget beside the one-minute budget; '
                    'it needs --memetic-minute')
    if test_only_five_preview:
        paper = os.path.join(ROOT, 'manuscript')
        if os.path.commonpath([os.path.abspath(out_dir), paper]) == paper:
            _fq3_refuse(f'a drawing from a preview of p137 is written outside {paper}, '
                        f'not into {out_dir}')
    T = load_timed(curves_csv, timed_json, allow_partial, group)
    request = T['origin'] == 'request'
    print(f'  timed: {curves_csv} (run {T["run"]}, origin {T["origin"]}, budgets '
          f'{", ".join(f"{b:g}" for b in T["budgets"])} s)'
          + ('' if group == 'all' else f'; group {group}, pooled rows {T["pool"]} over '
             f'{", ".join(FQ3_GROUPS[group][1])}'))
    table = gate_timed(T)
    mpar = None
    if memetic_parallel is not None:
        mpar = load_memetic_parallel(memetic_parallel, T, curves_csv, allow_partial)
        for size in FQ3_SIZES:
            for arm in FQ3_MPAR_METHODS:
                T['curves'].pop(('Sel', size, arm), None)
        T['curves'].update(mpar['curves'])
        print(f'  memetic algorithm, row Sel: the best of {FQ3_MPAR_RUNS} runs from '
              f'{_shown(mpar["csv"])} ('
              + ', '.join(FQ3_MPAR_METHODS[a] for a in FQ3_MPAR_METHODS if a in mpar['arms'])
              + ('; PREVIEW, not for the paper' if mpar['preview'] else '')
              + ('; dry run of p124' if mpar['dry'] else '') + ')')
    mmin = None
    if memetic_minute is not None:
        mmin = load_memetic_minute(memetic_minute, T, mpar)
        print(f'  memetic algorithm, one-minute budget: the best of eight one-minute runs from '
              f'each start, read at {FQ3_MMIN_AT:g} s, from {_shown(memetic_minute)} (entries '
              + ' and '.join(q for _, q in FQ3_MMIN_ENTRIES) + f'.{mmin["pool"]}), variant '
              + ', '.join(FQ3_MMIN_VARIANTS))
    mfiv = None
    if memetic_five is not None:
        mfiv = load_memetic_five(memetic_five, T, mpar, preview=test_only_five_preview)
        print(f'  memetic algorithm, five-minute budget: the best of eight five-minute runs from '
              f'each start, read at {FQ3_MFIV_AT:g} s, from {_shown(memetic_five)} (entries '
              + ' and '.join(q for _, q in FQ3_MFIV_ENTRIES) + f'.{mfiv["pool"]}), variant '
              + ', '.join(FQ3_MMIN_VARIANTS)
              + ('; PREVIEW of p137, not for the paper' if mfiv['preview']
                 else ''))
    if request:
        # the diamond: the median time of the policy's first schedule, from the CSV
        secs = {s: T['sizes'][s]['policy_first_s'] for s in FQ3_SIZES
                if s in T['sizes'] and T['sizes'][s]['policy_first_s'] is not None}
        notes = []
        for size in FQ3_SIZES:
            z = {k: ('-' if v is None else f'{v:.3f}') for k, v in
                 T['sizes'].get(size, {}).items()}
            print(f'  policy mark, {size}: ' + (
                f'{secs[size]:.3f} s, the median time of its first schedule (policy_first_s; '
                f'rule schedules {z.get("rule_first_s")} s, sampled schedules up to '
                f'{z.get("policy_budget_s")} s)' if size in secs else 'none, left out'))
    else:
        marks, notes = load_policy_times(policy_paths)
        marks = {s: v for s, v in marks.items() if s in FQ3_SIZES}
        for size in FQ3_SIZES:
            print(f'  policy mark, {size}: ' + (f'{marks[size][0]:.3f} s from {marks[size][1]}'
                                               if size in marks else 'none, left out'))
        secs = {s: v for s, (v, _) in marks.items()}
    for n in notes + T['notes']:
        print(f'  note: {n}')
    suffix = '_dev' if T['partial'] or (mpar is not None and mpar['dry']) else ''
    if mpar is not None and mpar['preview']:
        suffix += '_preview'
    if mfiv is not None and mfiv['preview']:
        suffix += '_five_preview'

    os.makedirs(out_dir, exist_ok=True)
    written = []
    for variant, arms in FQ3_VARIANTS:
        if mpar is not None and mpar['preview']:
            # a preview: the memetic arms of row Sel it does not hold are left out, and a
            # variant without one of its curves is not drawn
            arms = [(a, r) for a, r in arms
                    if not (r == 'Sel' and a in FQ3_MPAR_METHODS) or a in mpar['arms']]
            if not any(r == 'Sel' and a in mpar['arms'] for a, r in arms):
                print(f'  variant {variant}: holds no curve of the preview, not drawn')
                continue
        sample = request and variant in FQ3_SAMPLE_VARIANTS
        if sample:
            arms = list(arms) + [(FQ3_SAMPLE, 'Sel')]
        keys = [(row, size, arm) for size in FQ3_SIZES for arm, row in arms]
        miss = [k for k in keys if k not in T['curves']]
        if miss:
            _fq3_refuse(f'variant {variant} needs {miss}, which the CSV does not hold')
        if request:
            x0, x1, xticks = _fq3_xaxis_request(T, keys, secs)
        else:
            x0, x1, xticks = _fq3_xaxis(T, keys, secs)
        vals = []
        for k in keys:
            c = T['curves'][k]
            s = c['t'] >= x0 - 1e-12
            if k[2] == FQ3_SAMPLE:
                s = s & ~c['held']
            vals += [*c['lo'][s], *c['hi'][s]]
        # the one-minute circles of this variant, {size: {start: (mean, lo, hi)}}; their
        # intervals join the values the value axis covers
        minute = mmin['values'] if mmin is not None and variant in FQ3_MMIN_VARIANTS else None
        if minute:
            vals += [v for s in FQ3_SIZES for _, lo, hi in minute[s].values() for v in (lo, hi)]
        # the five-minute circles, in the variants of the one-minute circles, the same way
        five = mfiv['values'] if mfiv is not None and minute else None
        if five:
            vals += [v for s in FQ3_SIZES for _, lo, hi in five[s].values() for v in (lo, hi)]
        if request:
            ylim, yticks, ytext = _fq3_yaxis_request(vals)
        else:
            ylim, yticks = _fq3_yaxis(vals)
            ytext = lambda v: _fmt(round(v, 6))  # noqa: E731

        search = [(a, r) for a, r in arms if a in FQ3_ARMS]
        methods = [m for m in FQ3_METHOD if any(FQ3_ARMS[a][0] == m for a, _ in search)]
        starts = [s for s in FQ3_START if any(FQ3_ARMS[a][1] == s for a, _ in search)]
        if request and starts == ['rule']:
            starts = []          # every curve is solid: the caption names the start schedule
        hs = [((Patch(facecolor=FQ3_METHOD[m][1], alpha=FQ3_BAND_ALPHA, edgecolor='none'),
                Line2D([0], [0], color=FQ3_METHOD[m][1], lw=FQ3_LW)), FQ3_METHOD[m][0])
              for m in methods]
        # line style keys: black, or grey on the request clock, where black is the policy;
        # with the one-minute circles each key carries the circle of its start (filled or open)
        key_col = FQ3_STYLE_KEY if request else 'black'
        hs += [(Line2D([0], [0], color=key_col, lw=FQ3_LW, ls=FQ3_START[s][1],
                       **(dict(FQ3_MMIN_MARK, mec=key_col, mfc=FQ3_MMIN_FACE[s] or key_col)
                          if minute else {})), FQ3_START[s][0]) for s in starts]
        # The policy: the zero line, with the mark of its time when one is drawn.
        mk = {k: v for k, v in FQ3_MARK.items() if k != 'color'}
        hs.append((Line2D([0], [0], color=C_ZERO, lw=0.6,
                          **(dict(mk, mfc='black') if secs else {})),
                   FQ3_FIRST_TEXT if request else 'policy'))
        if sample:
            hs.append(((Patch(facecolor='black', alpha=FQ3_BAND_ALPHA, edgecolor='none'),
                        Line2D([0], [0], color='black', lw=FQ3_LW)), FQ3_SAMPLE_TEXT))
        ncol = len(hs) if len(hs) <= 4 else 3
        leg_h = 0.25 if ncol == len(hs) else 0.41
        hmap = None
        if request:
            # columns of two rows, filled column by column: the methods, the start
            # schedules, the policy (the last two share a column when they hold two
            # entries together); a short column gets an empty entry. With the one-minute
            # circles the columns have three rows, the third of column FQ3_MMIN_LEG_COL the
            # one-minute budget (its two circles side by side), or both budgets when the
            # five-minute circles are drawn too
            blank = (Line2D([0], [0], ls='none'), '')
            st = hs[len(methods):len(methods) + len(starts)]
            po = hs[len(methods) + len(starts):]
            mm = []
            if minute:
                rose = FQ3_METHOD['memetic'][1]
                pair = tuple(Line2D([0], [0], ls='none', color=rose, mec=rose,
                                    mfc=FQ3_MMIN_FACE[s] or rose, **FQ3_MMIN_MARK)
                             for s, _ in FQ3_MMIN_ENTRIES)
                mm, hmap = ([(pair, FQ3_MFIV_TEXT if five else FQ3_MMIN_TEXT)],
                            {pair: HandlerTuple(ndivide=None)})
            rows = FQ3_MMIN_ROWS if minute else 2
            cols = [hs[:len(methods)]] + ([st + po] if len(st) + len(po) <= 2 else [st, po])
            if mm:
                cols[FQ3_MMIN_LEG_COL] = cols[FQ3_MMIN_LEG_COL] + mm
            if max(len(x) for x in cols) > rows:
                _fq3_refuse(f'variant {variant}: a legend column holds more than '
                            f'{("two", "three")[rows - 2]} entries')
            hs = [e for x in cols for e in x + [blank] * (rows - len(x))]
            ncol, leg_h = len(cols), 0.41 + FQ3_LEG_ROW * (rows - 2)

        # Layout in inches (printed size), bottom to top.
        fig_h = (2.70 if variant == 'B_supplement' else 3.05) + (leg_h - 0.25)
        left, right, gap = 0.66, 0.13, 0.26
        pw = (FIG_W - left - right - gap) / 2
        leg_y = 0.04
        ax_y0 = leg_y + leg_h + 0.46
        ax_h = fig_h - ax_y0 - 0.28
        fig = plt.figure(figsize=(FIG_W, fig_h), dpi=300)
        ax = fig.add_axes([left / FIG_W, ax_y0 / fig_h, pw / FIG_W, ax_h / fig_h])
        bx = fig.add_axes([(left + pw + gap) / FIG_W, ax_y0 / fig_h, pw / FIG_W,
                           ax_h / fig_h], sharey=ax)
        # the circles of the fixed budgets: (x, the comparisons of the record) per budget drawn
        fixed = [(FQ3_MMIN_AT, FQ3_MMIN_ENTRIES)] if minute else []
        if five:
            fixed.append((FQ3_MFIV_AT, FQ3_MFIV_ENTRIES))
        circles = {}
        for a, size, tag in ((ax, FQ3_SIZES[0], 'a'), (bx, FQ3_SIZES[1], 'b')):
            points = [(at, (minute if at == FQ3_MMIN_AT else five)[size])
                      for at, _ in fixed] or None
            circles[size] = (a, _fq3_panel(a, T, arms, size, (x0, x1), xticks, yticks,
                                           secs.get(size), points), points)
            a.set_title(f'({tag}) {FQ3_SIZE_TEXT[size]}', fontsize=FS_TITLE, pad=7)
        ax.set_ylim(*ylim)
        ax.yaxis.set_major_locator(FixedLocator(yticks))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: ytext(v)))
        ax.set_ylabel('makespan difference from the\npolicy’s first schedule (%)' if request
                      else 'makespan difference,\nsearch method minus policy (%)',
                      multialignment='center', labelpad=5)
        plt.setp(bx.get_yticklabels(), visible=False)

        xc = (left + pw + gap / 2) / FIG_W
        fig.text(xc, (ax_y0 - 0.33) / fig_h,
                 'time since the request for a schedule (s), log scale' if request
                 else 'time budget per instance (s), log scale',
                 ha='center', va='center', fontsize=FS_LABEL)
        leg = fig.legend([h for h, _ in hs], [t for _, t in hs], loc='lower center',
                         bbox_to_anchor=(xc, leg_y / fig_h), ncol=ncol,
                         fontsize=FS_TICK, frameon=True, fancybox=False,
                         edgecolor='0.6', framealpha=1.0, handlelength=2.2,
                         handletextpad=0.5, columnspacing=1.6, borderpad=0.4,
                         handler_map=hmap)
        leg.get_frame().set_linewidth(0.5)

        _fq3_spread(fig, (ax, bx))
        n_text = _check_layout(fig, (ax, bx), leg, tag='fq3')
        if minute:
            for size, (a, drawn, points) in circles.items():
                want = sorted((at, s) for at, entries in fixed for s, _ in entries)
                if sorted(drawn) != want:
                    _fq3_refuse(f'variant {variant}, {size}: circles {sorted(drawn)} drawn, '
                                f'{want} expected')
                values = {(at, s): v for at, vs in points for s, v in vs.items()}
                _fq3_minute_check(fig, a, drawn, values, secs.get(size), variant, size)
            # the legend against the span of the two panels and the canvas
            rend = fig.canvas.get_renderer()
            lb, pa, pb = (o.get_window_extent(rend) for o in (leg, ax, bx))
            if lb.x0 < pa.x0 - 0.5 or lb.x1 > pb.x1 + 0.5:
                print(f'  note: variant {variant}: the legend is {lb.width / fig.dpi:.2f} in '
                      f'wide, the two panels span {(pb.x1 - pa.x0) / fig.dpi:.2f} in'
                      + (' and the legend leaves the canvas' if lb.x0 < 0 or
                         lb.x1 > fig.bbox.width else ''))
        stem = os.path.join(out_dir, f'fq3_time_budget_{variant}'
                            + ('' if group == 'all' else f'_{group}') + suffix)
        fig.savefig(stem + '.pdf')
        fig.savefig(stem + '.png', dpi=300)
        plt.close(fig)
        written += [stem + '.pdf', stem + '.png']
        print(f'  variant {variant}: {FIG_W:.2f} x {fig_h:.2f} in, x {x0:g} to {x1:g} s, '
              f'y {ylim[0]:+.2f} to {ylim[1]:+.2f} %, {n_text} texts checked, no overlap')

    # ---- what is plotted at the reported budgets, and the JSON gate
    print('\n  size       arm             row   ' + '  '.join(f'{b:>18g} s' for b in
                                                           T['budgets']))
    for (row, size, arm), c in sorted(T['curves'].items(), key=lambda kv: kv[0][1:] + kv[0][:1]):
        if request and size not in FQ3_SIZES:
            continue
        cells = []
        for b in T['budgets']:
            v = _fq3_at(c, b)
            held = request and v is not None and bool(c['held'][np.flatnonzero(
                np.abs(c['t'] - b) < 1e-9)[0]])
            cells.append('-' if v is None else f'{v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}]'
                         + ('h' if held else ''))
        first = f'  from {c["t"][0]:g} s' if request else ''
        print(f'  {size:10s} {arm:15s} {row:4s}  ' + '  '.join(f'{x:>20s}' for x in cells)
              + first)
    if request:
        n = {s: sum(x == s for *_, x in table) for s in ('match', 'no schedule yet',
                                                          'absent, partial run')}
        print('  (h: a held value of the sampled schedules, not drawn)')
        print(f'\n  JSON gate: {n["match"]} of {len(table)} checks match (the times of the '
              'diamond, the rule schedules and the sampling budget; ii, vi and vii at every '
              f'budget of budgets_per_size); {n["no schedule yet"]} entries hold no schedule '
              f'yet and {n["absent, partial run"]} times are absent from a partial run')
        if mpar is not None:
            print(f'  (the memetic rows of row Sel are the best of {FQ3_MPAR_RUNS} runs, after '
                  'the JSON gate above held the single run against the timed record)')
            print(f'  decision-table gate: {sum(s == "match" for *_, s in mpar["checks"])} of '
                  f'{len(mpar["checks"])} values of the best of eight match the decision table on '
                  f'eight cores at its printed digits (at '
                  f'{", ".join(f"{b:g}" for b in FQ3_MPAR_BUDGETS)} s, entry {T["pool"]}, both '
                  'sizes, ' + ' and '.join(sorted(mpar['arms'])) + ')')
        if mmin is not None:
            def iv(v):
                return f'{v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}]'
            for size in FQ3_SIZES:
                v = mmin['values'][size]
                print(f'  one-minute budget, {size}, at {FQ3_MMIN_AT:g} s: started from the best '
                      f'rule {iv(v["rule"])} (filled circle), started from the policy’s schedule '
                      f'{iv(v["policy"])} (open circle); results.{size}.comparisons.'
                      + ' and '.join(f'{q}.{mmin["pool"]}' for _, q in FQ3_MMIN_ENTRIES)
                      + ', drawn as read in variant ' + ', '.join(FQ3_MMIN_VARIANTS))
        if mfiv is not None:
            # read back from the record: the values printed are the file's, and each must be
            # the value its circle was drawn with (checked against the artists above)
            with open(memetic_five) as f:
                R5 = json.load(f).get('results') or {}
            for size in FQ3_SIZES:
                v = {}
                for start, q in FQ3_MFIV_ENTRIES:
                    e = _fq3_dig(R5, (size, 'comparisons', q, mfiv['pool']))
                    v[start] = tuple(float(e[k]) for k in ('mean', 'lo', 'hi'))
                    if v[start] != mfiv['values'][size][start]:
                        _fq3_refuse(f'{memetic_five}: results.{size}.comparisons.{q}.'
                                    f'{mfiv["pool"]} reads {v[start]} now, the circle was drawn '
                                    f'with {mfiv["values"][size][start]}')
                print(f'  five-minute budget, {size}, at {FQ3_MFIV_AT:g} s: started from the best '
                      f'rule {iv(v["rule"])} (filled circle), started from the policy’s schedule '
                      f'{iv(v["policy"])} (open circle); results.{size}.comparisons.'
                      + ' and '.join(f'{q}.{mfiv["pool"]}' for _, q in FQ3_MFIV_ENTRIES)
                      + ', read back from the record, drawn in variant '
                      + ', '.join(FQ3_MMIN_VARIANTS)
                      + (' (PREVIEW of p137, not for the paper)' if mfiv['preview'] else ''))
    else:
        print(f'\n  JSON gate: {sum(s == "match" for *_, s in table)} of {len(table)} values '
              'match (policy-started arms at every budget, rule-started arms at time zero)')
    return written


# ------------------------------------------------------------------ fq4: boundary
# variant -> the form of p93 that panel (b) plots (FORMS of p93_rule_comparison.py)
FQ4_FORMS = {'A': 'sample4', 'B': 'greedy_best4'}
FQ4_FORM_WORD = {'sample4': 'SampleFour', 'greedy_best4': 'GreedyBestFour'}
FQ4_HEADLINE = 'pub_SD2_test_10x5+mix'           # rulecmp_plan.HEADLINE
FQ4_VERDICTS = ('ahead', 'behind', 'not distinguishable')
C_FACTORY = '#d07a87'                             # the medium rose of fq3
# Benchmark by marker shape and colour; the changed-setup levels are smaller open
# markers of the factory shape.
FQ4_STYLE = {
    'factory': dict(text='factory benchmark', marker='s', ms=4.2, color=C_FACTORY,
                    open=False, z=4),
    'public': dict(text='public instances', marker='o', ms=4.6, color=C_EQUIV,
                   open=False, z=5),
    'sens': dict(text='factory benchmark, changed setups', marker='s', ms=3.0,
                 color=C_FACTORY, open=True, z=3),
}
FQ4_CI_LW = 1.1
# Rows of the sensitivity CSV that the figure reads: the levels that change the
# setups (parameter column), the policy against the best dispatching rule, on the
# combinations the level reaches. Level words of p87 LEVEL_WORDS, for its macros.
FQ4_SENS_PARAMS = ('setup scale', 'stations charged with setups')
FQ4_SENS_WORDS = {'setup_x050': 'SetupHalf', 'setup_x075': 'SetupThreeQuarters',
                  'setup_x125': 'SetupFiveQuarters', 'setup_x150': 'SetupThreeHalves',
                  'setup_sub_strict': 'SetupShell', 'setup_sub_wide': 'SetupProduction'}
FQ4_SENS_COMBOS = ('S', 'LS', 'SW', 'LSW')       # the combinations a setup level reaches
FQ4_SENS_NEEDED = ('level', 'parameter', 'comparison', 'row', 'reached', 'mean', 'lo',
                   'hi', 'pol_one_pair', 'pol_setup_share')
FQ4_GAP, FQ4_PAD = 1.8, 0.6                       # pt: marker to label; clearance
# A spread label moves at most 0.6 label heights from the height of its point, and a
# label in a row at most 1.2 widths of the row's widest label from the x of its point.
FQ4_MOVE_V, FQ4_MOVE_H = 0.6, 1.2
FQ4_SIDE = {'public': 'right', 'factory': 'left'}  # the side each benchmark tries first


def _fq4_refuse(msg):
    raise SystemExit(f'fq4 REFUSING: {msg}')


def load_rulecmp(path, family, allow_partial):
    """Per combination, the policy's two measured variables (mean over the training
    runs) and, per form of FQ4_FORMS, the released policy against the best
    dispatching rule (mean, 95% interval, verdict), as p93 wrote them."""
    with open(path) as f:
        d = json.load(f)
    st = d.get('settings', {})
    if st.get('family') != family:
        _fq4_refuse(f'{path} holds the family {st.get("family")!r}, not {family!r}')
    if family == 'public' and st.get('test_set') != FQ4_HEADLINE:
        _fq4_refuse(f'{path} holds the test set {st.get("test_set")!r}, not {FQ4_HEADLINE}')
    partial = bool(st.get('partial'))
    if partial and not allow_partial:
        _fq4_refuse(f'{path} comes from a partial run of p93 (settings.partial); it is drawn '
                    'only with --allow-partial')
    pts = {}
    for r in COMBOS:
        c = d.get('combos', {}).get(r)
        if c is None:
            _fq4_refuse(f'{path} has no combos.{r}')
        tr = c.get('traces', {})
        m = {}
        for k in ('policy_one_pair', 'policy_setup_share'):
            if 'mean' not in tr.get(k, {}):
                _fq4_refuse(f'{path} has no combos.{r}.traces.{k}.mean')
            m[k] = float(tr[k]['mean'])
        forms = {}
        for form in FQ4_FORMS.values():
            it = c.get('arms', {}).get('released', {}).get(form)
            where = f'combos.{r}.arms.released.{form}'
            if it is None:
                _fq4_refuse(f'{path} has no {where}')
            for k in ('mean', 'lo', 'hi', 'verdict'):
                if k not in it:
                    _fq4_refuse(f'{path}: {where} has no field {k!r}')
            mean, lo, hi, v = float(it['mean']), float(it['lo']), float(it['hi']), it['verdict']
            if not lo <= mean <= hi:
                _fq4_refuse(f'{where}: the mean lies outside its interval')
            if v not in FQ4_VERDICTS or (v == 'ahead' and not hi < 0) or \
                    (v == 'behind' and not lo > 0):
                _fq4_refuse(f'{where}: verdict {v!r} disagrees with its interval '
                            f'[{lo:+.3f}, {hi:+.3f}]')
            forms[form] = dict(mean=mean, lo=lo, hi=hi, verdict=v)
        pts[r] = dict(one_pair=m['policy_one_pair'], setup_share=m['policy_setup_share'],
                      forms=forms)
    return dict(points=pts, partial=partial, path=path,
                setting=f'{st.get("test_set")}, {st.get("n_test")} instances, seeds '
                        f'{",".join(map(str, st.get("seeds", [])))}, K = {st.get("K")}')


def load_sensitivity(csv_path, allow_partial):
    """The levels that change the setups, from the CSV of p87 infer; its JSON, written
    with it, says whether the run was partial."""
    json_path = os.path.splitext(csv_path)[0] + '.json'
    if not os.path.exists(json_path):
        _fq4_refuse(f'{json_path} not found; p87 infer writes it with the CSV, and it says '
                    'whether the run was partial')
    with open(json_path) as f:
        J = json.load(f)
    partial = bool(J.get('partial'))
    if partial and not allow_partial:
        _fq4_refuse(f'{csv_path} comes from a partial run of p87 infer; it is drawn only '
                    'with --allow-partial')
    if not J.get('measured_variables', {}).get('computed', False):
        _fq4_refuse(f'{csv_path} holds no measured variables (p87 evaluate was run without '
                    '--trace_counts)')
    with open(csv_path, newline='') as f:
        rd = csv.DictReader(f)
        lack = [c for c in FQ4_SENS_NEEDED if c not in (rd.fieldnames or [])]
        if lack:
            _fq4_refuse(f'{csv_path} has no column {lack}')
        rows = [r for r in rd if r['parameter'] in FQ4_SENS_PARAMS
                and r['comparison'] == 'pol_rule' and r['row'] in COMBOS
                and r['reached'] == '1']
    pts, seen = [], {}
    for r in rows:
        if any(r[k] == '' for k in ('mean', 'lo', 'hi', 'pol_one_pair', 'pol_setup_share')):
            _fq4_refuse(f'{csv_path}, level {r["level"]}, {r["row"]}: an empty value')
        if r['row'] not in FQ4_SENS_COMBOS:
            _fq4_refuse(f'{csv_path}: level {r["level"]} reaches {r["row"]}, which has no '
                        'setups')
        key = (r['level'], r['row'])
        if key in seen:
            _fq4_refuse(f'{csv_path}: {key} appears twice')
        seen[key] = True
        mean, lo, hi = float(r['mean']), float(r['lo']), float(r['hi'])
        if not lo - 1e-6 <= mean <= hi + 1e-6:
            _fq4_refuse(f'{csv_path}, {key}: the mean lies outside its interval')
        pts.append(dict(level=r['level'], combo=r['row'], mean=mean, lo=lo, hi=hi,
                        one_pair=float(r['pol_one_pair']),
                        setup_share=float(r['pol_setup_share'])))
    want = {(lv, c) for lv in FQ4_SENS_WORDS for c in FQ4_SENS_COMBOS}
    if not partial and set(seen) != want:
        _fq4_refuse(f'{csv_path}: the setup levels hold {len(seen)} rows, want the '
                    f'{len(want)} of {", ".join(FQ4_SENS_WORDS)} on {", ".join(FQ4_SENS_COMBOS)}'
                    + (f' (missing {sorted(want - set(seen))[:4]})' if want - set(seen) else '')
                    + (f' (unexpected {sorted(set(seen) - want)[:4]})' if set(seen) - want
                       else ''))
    if not pts:
        _fq4_refuse(f'{csv_path} holds no row of a level that changes the setups')
    return dict(points=pts, partial=partial, path=csv_path,
                levels=sorted({p['level'] for p in pts}))


def gate_rulecmp(D, fam, prefix, path):
    """The plotted values against the macros of p93 (prefix Rcp or Rcf): mean, Lo, Hi
    and Verdict of both forms at two decimals, OnePair and SetupShare at one."""
    rows, bad = [], []
    have = bool(path) and os.path.exists(path)
    if not have:
        print(f'  {fam} macro check skipped: {path} not found')
    mac = _macros(path) if have else {}
    for r in COMBOS:
        p = D['points'][r]
        checks = [(f'{prefix}OnePair{r}', p['one_pair'], 0.05),
                  (f'{prefix}SetupShare{r}', p['setup_share'], 0.05)]
        for form, it in p['forms'].items():
            stem = f'{prefix}Pol{FQ4_FORM_WORD[form]}{r}'
            checks += [(stem, it['mean'], 0.005), (stem + 'Lo', it['lo'], 0.005),
                       (stem + 'Hi', it['hi'], 0.005), (stem + 'Verdict', it['verdict'], None)]
        for name, v, tol in checks:
            got = mac.get(name)
            if not have:
                status = 'no macro file'
            elif got is None:
                status = 'no macro'
                bad.append(f'\\{name} not in {path}')
            elif tol is None:
                status = 'match' if got == v else 'MISMATCH'
            else:
                status = 'match' if abs(v - float(got)) <= tol + 1e-9 else 'MISMATCH'
            if status == 'MISMATCH':
                bad.append(f'\\{name} = {got}, file {v}')
            rows.append((fam, name, v, got, status))
    if bad:
        _fq4_refuse(f'{path} disagrees with the plotted file:\n  ' + '\n  '.join(bad))
    return rows


def gate_sensitivity(S, path):
    """The measured variables of the plotted levels against the macros of p87 infer
    (Sens<level>PolOnePair<R>, Sens<level>PolSetupShare<R>, one decimal); the
    policy-against-rule means of these levels have no macro."""
    rows, bad = [], []
    have = bool(path) and os.path.exists(path)
    if not have:
        print(f'  sensitivity macro check skipped: {path} not found')
    mac = _macros(path) if have else {}
    for p in S['points']:
        w = FQ4_SENS_WORDS.get(p['level'])
        for word, v in (('PolOnePair', p['one_pair']), ('PolSetupShare', p['setup_share'])):
            name = f'Sens{w}{word}{p["combo"]}' if w else f'{p["level"]}:{word}:{p["combo"]}'
            got = mac.get(name) if w else None
            if not have:
                status = 'no macro file'
            elif got is None:
                status = 'no macro'
                bad.append(f'\\{name} not in {path}')
            else:
                status = 'match' if abs(v - float(got)) <= 0.05 + 1e-4 else 'MISMATCH'
                if status == 'MISMATCH':
                    bad.append(f'\\{name} = {got}, file {v:.4f}')
            rows.append(('sensitivity', name, v, got, status))
    if bad:
        _fq4_refuse(f'{path} disagrees with the plotted file:\n  ' + '\n  '.join(bad))
    return rows


# ---- direct labels: placed next to their points, proven clear of every mark
def _fq4_spread(c, size, gap, weight):
    """Positions closest to the wanted positions c (ascending) in weighted least
    squares, with neighbours at least (size_i + size_j) / 2 + gap apart: the pool
    adjacent violators algorithm on c minus the cumulative spacing. A heavy weight
    pins an entry (a mark the labels must pass around)."""
    s = [0.0]
    for i in range(1, len(c)):
        s.append(s[-1] + (size[i - 1] + size[i]) / 2 + gap)
    pools = []                                    # [sum of w * v, sum of w, count]
    for v, w in zip((ci - si for ci, si in zip(c, s)), weight):
        pools.append([w * v, w, 1])
        while len(pools) > 1 and pools[-2][0] / pools[-2][1] > pools[-1][0] / pools[-1][1]:
            b = pools.pop()
            a = pools.pop()
            pools.append([a[0] + b[0], a[1] + b[1], a[2] + b[2]])
    out = [p[0] / p[1] for p in pools for _ in range(p[2])]
    return [o + si for o, si in zip(out, s)]


def _fq4_hit(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _fq4_label_layouts(items, kind, frame, blocks=(), zero=None):
    """Candidate layouts of one group of labels (positions in points): each label
    beside its own point on one side, a column on one side with one left (or right)
    edge, or a row with one baseline above or below the group's intervals. Within a
    column or row the labels keep the order of their points and move apart as little
    as possible; the marks and labels in their way (blocks) and the frame stay fixed.
    A group that lies wholly above the zero line tries its row above first."""
    def put(it, ha, x, base):
        x0 = {'left': x, 'right': x - it['w'], 'center': x - it['w'] / 2}[ha]
        return dict(it=it, ha=ha, x=x, base=base,
                    box=(x0, base - it['desc'], x0 + it['w'], base + it['asc']))

    def fit(want, size, gap, fixed, lo_edge, hi_edge):
        """Spread the wanted centres around the fixed intervals (merged where no label
        fits between them) and inside the frame edges."""
        fixed = sorted(list(fixed) + [(lo_edge - 1e3, lo_edge), (hi_edge, hi_edge + 1e3)])
        merged = []
        for a, b in fixed:
            if merged and a - merged[-1][1] < min(size) + 2 * gap:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        chain = sorted([(c, h, 1.0, i) for i, (c, h) in enumerate(zip(want, size))] +
                       [((a + b) / 2, b - a, 1e6, None) for a, b in merged],
                       key=lambda e: e[0])
        got = _fq4_spread([e[0] for e in chain], [e[1] for e in chain], gap,
                          [e[2] for e in chain])
        return {e[3]: p for e, p in zip(chain, got) if e[3] is not None}

    def vspread(xs, ha):
        tmp = [put(it, ha, x, 0.0)['box'] for it, x in zip(items, xs)]
        x0, x1 = min(b[0] for b in tmp), max(b[2] for b in tmp)
        lift = [(it['asc'] - it['desc']) / 2 for it in items]
        want = [it['y'] - it['cap'] / 2 + d for it, d in zip(items, lift)]
        h = [it['asc'] + it['desc'] for it in items]
        got = fit(want, h, 0.3, [(b[1], b[3]) for b in blocks if b[0] < x1 and x0 < b[2]],
                  frame[1], frame[3])
        if max(abs(got[i] - want[i]) for i in range(len(items))) > FQ4_MOVE_V * max(h):
            return None                   # a label would leave the height of its point
        return [put(it, ha, x, got[i] - lift[i]) for i, (it, x) in enumerate(zip(items, xs))]

    def hspread(base):
        asc_, desc_ = max(it['asc'] for it in items), max(it['desc'] for it in items)
        got = fit([it['x'] for it in items], [it['w'] for it in items], 3.0,
                  [(b[0], b[2]) for b in blocks if b[1] < base + asc_ and base - desc_ < b[3]],
                  frame[0], frame[2])
        if max(abs(got[i] - it['x']) for i, it in enumerate(items)) > \
                FQ4_MOVE_H * max(it['w'] for it in items):
            return None                   # a label would leave the column of its point
        return [put(it, 'center', got[i], base) for i, it in enumerate(items)]

    right = [it['x'] + it['r'] + FQ4_GAP for it in items]
    left = [it['x'] - it['r'] - FQ4_GAP for it in items]
    asc = max(it['asc'] for it in items)
    desc = max(it['desc'] for it in items)
    below = min(min(it['y'] - it['r'], it['lo']) for it in items) - FQ4_GAP - asc
    above = max(max(it['y'] + it['r'], it['hi']) for it in items) + FQ4_GAP + desc
    col_r, col_l = [max(right)] * len(items), [min(left)] * len(items)
    side = [('beside, right', lambda: vspread(right, 'left')),
            ('beside, left', lambda: vspread(left, 'right'))]
    # a column only for points stacked at one x (beside a spread group a code would sit
    # level with a neighbour's point)
    stacked = max(it['x'] for it in items) - min(it['x'] for it in items) < 1.0
    column = [('column, right', lambda: vspread(col_r, 'left')),
              ('column, left', lambda: vspread(col_l, 'right'))] if stacked else []
    if FQ4_SIDE.get(items[0]['m']['style']) == 'left':
        side, column = side[::-1], column[::-1]
    # in panel (b) a row may step up to 12 pt further from its group, past the marks in
    # its way; it tries the side away from the zero line first, and it never lies
    # across the zero line from all of its points
    lo_y = min(it['y'] - it['r'] for it in items)
    hi_y = max(it['y'] + it['r'] for it in items)
    sides = ('above', 'below') if zero is not None and lo_y > zero else ('below', 'above')
    if zero is not None and hi_y < zero:
        sides = ('below',)
    elif zero is not None and lo_y > zero:
        sides = ('above',)
    row = [(f'row {w}' + (f', {d} pt further' if d else ''),
            lambda w=w, d=d: hspread(below - d if w == 'below' else above + d))
           for w in sides for d in ((0, 3, 6, 9, 12) if kind == 'ci' else (0,))]
    return side + row + column


def _fq4_place_labels(fig, ax, marks, kind):
    """Place the combination codes of one panel. marks: every drawn point, with its
    marker half-size and interval, and its label or None. Labels are grouped where
    labels beside their points would collide; each group takes the first layout
    whose labels clear every marker, interval, the zero line, the axes frame and the
    labels already placed. Returns the placed labels."""
    k = 72.0 / fig.dpi
    rend = fig.canvas.get_renderer()
    ax_b = ax.get_window_extent(rend)
    frame = (ax_b.x0 * k + 1, ax_b.y0 * k + 1, ax_b.x1 * k - 1, ax_b.y1 * k - 1)
    obst, items = [], []
    for m in marks:
        x, y = (v * k for v in ax.transData.transform((m['x'], m['y'])))
        r = m['ms'] / 2 + 0.5
        lo = hi = y
        if m.get('lo') is not None:
            lo = ax.transData.transform((m['x'], m['lo']))[1] * k
            hi = ax.transData.transform((m['x'], m['hi']))[1] * k
            obst.append((x - FQ4_CI_LW / 2, lo, x + FQ4_CI_LW / 2, hi))
        obst.append((x - r, y - r, x + r, y + r))
        if m.get('label'):
            t = fig.text(0.5, 0.5, comb_tex(m['label']), fontsize=FS_TICK, ha='left',
                         va='baseline')
            b = t.get_window_extent(rend)
            t.remove()
            base = 0.5 * fig.bbox.height
            items.append(dict(m=m, x=x, y=y, r=r, lo=min(lo, y - r), hi=max(hi, y + r),
                              w=b.width * k, asc=(b.y1 - base) * k, desc=(base - b.y0) * k,
                              cap=0.66 * FS_TICK))
    zero = None
    if kind == 'ci':
        zero = ax.transData.transform((0, 0))[1] * k
        obst.append((frame[0], zero - 0.4, frame[2], zero + 0.4))
    obst = [(a - FQ4_PAD, b - FQ4_PAD, c + FQ4_PAD, d + FQ4_PAD) for a, b, c, d in obst]

    def gap(box, c):
        return np.hypot((box[0] + box[2]) / 2 - c[0], (box[1] + box[3]) / 2 - c[1])

    def clear(lay, placed, group):
        for i, L in enumerate(lay):
            bx = L['box']
            if bx[0] < frame[0] or bx[1] < frame[1] or bx[2] > frame[2] or bx[3] > frame[3]:
                return False
            if any(_fq4_hit(bx, o) for o in obst):
                return False
            if any(_fq4_hit(bx, P['box']) for P in placed + lay[:i]):
                return False
            # the coded point nearest to a code (from the code's centre to the marker
            # centres) is one of its own group or carries the same code
            near = min(items, key=lambda it: gap(bx, (it['x'], it['y'])))
            if not any(near is g for g in group) and \
                    near['m']['label'] != L['it']['m']['label']:
                return False
        return True

    def blocks(placed):
        return obst + [(a - 0.3, b - 0.3, c + 0.3, d + 0.3)
                       for a, b, c, d in (P['box'] for P in placed)]

    # groups, within one benchmark: points at one x (the combinations without setups in
    # panel (a)) and labels that would collide when each sits beside its own point on
    # the benchmark's side. The public points are placed first, then the factory points.
    def beside_box(it):
        x0 = it['x'] + it['r'] + FQ4_GAP if FQ4_SIDE.get(it['m']['style']) != 'left' else \
            it['x'] - it['r'] - FQ4_GAP - it['w']
        base = it['y'] - it['cap'] / 2
        return (x0, base - it['desc'], x0 + it['w'], base + it['asc'])
    beside = [beside_box(it) for it in items]
    parent = list(range(len(items)))

    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i
    for i in range(len(items)):
        for j in range(len(items)):
            if i != j and items[i]['m']['style'] == items[j]['m']['style']:
                oj = (items[j]['x'] - items[j]['r'], items[j]['lo'],
                      items[j]['x'] + items[j]['r'], items[j]['hi'])
                if abs(items[i]['x'] - items[j]['x']) < 1.0 or _fq4_hit(beside[i], oj) \
                        or _fq4_hit(beside[i], beside[j]):
                    parent[root(i)] = root(j)
    groups = {}
    for i in range(len(items)):
        groups.setdefault(root(i), []).append(items[i])
    rank = {'public': 0, 'factory': 1}
    order = sorted(groups.values(), key=lambda g: (rank.get(g[0]['m']['style'], 2), -len(g),
                                                    min(it['x'] for it in g)))
    placed, notes, done = [], [], []
    for g in order:
        for name, make in _fq4_label_layouts(g, kind, frame, blocks(placed), zero):
            lay = make()
            if lay is not None and clear(lay, placed, g):
                placed += lay
                done.append(dict(g=g, lay=lay, name=name))
                break
        else:
            _fq4_refuse(f'panel ({"b" if kind == "ci" else "a"}): no tidy place next to their '
                        f'points for the {g[0]["m"]["style"]} labels '
                        + ', '.join(it['m']['label'] for it in g)
                        + ('; --fq4-codes b draws the codes in panel (b) only'
                           if kind != 'ci' else ''))
    # two rows on one side of their groups, side by side and a few points apart in
    # height, share one baseline: the nearer row moves out to the further one when its
    # labels stay clear
    for A in done:
        for B in done:
            wa, wb = A['name'][:9], B['name'][:9]
            if A is B or wa != wb or wa not in ('row below', 'row above'):
                continue
            ba, bb = A['lay'][0]['base'], B['lay'][0]['base']
            out = ba - bb if wa == 'row below' else bb - ba
            near = min(L['box'][0] for L in B['lay']) < max(L['box'][2] for L in A['lay']) + 30 \
                and min(L['box'][0] for L in A['lay']) - 30 < max(L['box'][2] for L in B['lay'])
            if not (0 < out <= 6 and near):
                continue
            new = [dict(L, base=bb, box=(L['box'][0], L['box'][1] + bb - ba, L['box'][2],
                                         L['box'][3] + bb - ba)) for L in A['lay']]
            rest = [P for P in placed if not any(P is L for L in A['lay'])]
            if clear(new, rest, A['g']):
                placed = rest + new
                A['lay'], A['name'] = new, A['name'] + ', on the baseline of the row beside it'
    for D in done:
        if len(D['g']) > 1:
            codes = '/'.join(it['m']['label'] for it in D['g'])
            notes.append(f'{D["g"][0]["m"]["style"]} {codes}: {D["name"]}')
    for L in placed:
        m, it = L['it']['m'], L['it']
        ax.annotate(comb_tex(m['label']), xy=(m['x'], m['y']),
                    xytext=(L['x'] - it['x'], L['base'] - it['y']), textcoords='offset points',
                    ha=L['ha'], va='baseline', fontsize=FS_TICK, annotation_clip=False)
    return placed, notes


def _fq4_check_marks(fig, axes_marks):
    """Independent of the placement: no text of a panel may touch a marker or an
    interval of that panel (the rendered extents)."""
    k = 72.0 / fig.dpi
    rend = fig.canvas.get_renderer()
    n = 0
    for ax, marks in axes_marks:
        boxes = []
        for m in marks:
            x, y = (v * k for v in ax.transData.transform((m['x'], m['y'])))
            r = m['ms'] / 2 + 0.5
            boxes.append((x - r, y - r, x + r, y + r))
            if m.get('lo') is not None:
                lo = ax.transData.transform((m['x'], m['lo']))[1] * k
                hi = ax.transData.transform((m['x'], m['hi']))[1] * k
                boxes.append((x - FQ4_CI_LW / 2, lo, x + FQ4_CI_LW / 2, hi))
        for t in ax.texts:
            b = t.get_window_extent(rend)
            tb = (b.x0 * k, b.y0 * k, b.x1 * k, b.y1 * k)
            for o in boxes:
                if _fq4_hit(tb, o):
                    raise SystemExit(f'fq4 REFUSING: the label {t.get_text()!r} touches a mark')
            n += 1
    return n


def _fq4_ticks(lo, hi, nbins=5):
    return [t for t in MaxNLocator(nbins=nbins, steps=[1, 2, 2.5, 5, 10]).tick_values(lo, hi)
            if lo - 1e-9 <= t <= hi + 1e-9]


def _fq4_point(ax, m):
    st = FQ4_STYLE[m['style']]
    if m.get('lo') is not None:
        ax.plot([m['x'], m['x']], [m['lo'], m['hi']], color=st['color'], lw=FQ4_CI_LW,
                solid_capstyle='butt', zorder=st['z'])
    face = dict(mfc='white', mec=st['color'], mew=0.8) if st['open'] else \
        dict(mfc=st['color'], mec='white', mew=0.5)
    ax.plot([m['x']], [m['y']], st['marker'], ms=st['ms'], zorder=st['z'] + 0.5, **face)


def fq4_boundary(public_json, factory_json, sens_csv, out_dir, public_macros=None,
                 factory_macros=None, sens_macros=None, allow_partial=False, codes='ab'):
    """Draw variants A and B, then print the plotted numbers against the macros.
    codes: the panels whose points carry their combination codes ('ab' or 'b')."""
    P = load_rulecmp(public_json, 'public', allow_partial)
    print(f'  public:  {public_json} ({P["setting"]})')
    F = S = None
    if factory_json and os.path.exists(factory_json):
        F = load_rulecmp(factory_json, 'factory', allow_partial)
        print(f'  factory: {factory_json} ({F["setting"]})')
    else:
        print(f'  factory points left out: {factory_json} not found')
    if sens_csv and os.path.exists(sens_csv):
        S = load_sensitivity(sens_csv, allow_partial)
        print(f'  changed setups: {sens_csv} (levels {", ".join(S["levels"])}, '
              f'{len(S["points"])} points)')
    else:
        print(f'  changed-setup points left out: {sens_csv} not found')
    table = gate_rulecmp(P, 'public', 'Rcp', public_macros)
    if F is not None:
        table += gate_rulecmp(F, 'factory', 'Rcf', factory_macros)
    if S is not None:
        table += gate_sensitivity(S, sens_macros)
    suffix = '_dev' if any(D is not None and D['partial'] for D in (P, F, S)) else ''
    benches = [(b, D) for b, D in (('factory', F), ('public', P)) if D is not None]

    os.makedirs(out_dir, exist_ok=True)
    written, notes = [], []
    for variant, form in FQ4_FORMS.items():
        # the points of each panel; the changed-setup levels unlabelled, below the rest
        ma, mb = [], []
        if S is not None:
            for p in S['points']:
                ma.append(dict(x=p['setup_share'], y=p['one_pair'], style='sens',
                               ms=FQ4_STYLE['sens']['ms']))
                if variant == 'B':
                    mb.append(dict(x=p['one_pair'], y=p['mean'], style='sens',
                                   ms=FQ4_STYLE['sens']['ms']))
        for b, D in benches:
            for r in COMBOS:
                p, it = D['points'][r], D['points'][r]['forms'][form]
                ms = FQ4_STYLE[b]['ms']
                ma.append(dict(x=p['setup_share'], y=p['one_pair'], style=b, ms=ms,
                               label=r if 'a' in codes else None))
                mb.append(dict(x=p['one_pair'], y=it['mean'], lo=it['lo'], hi=it['hi'],
                               style=b, ms=ms, label=r))

        # axes: the share of decisions with one pair is panel (a)'s y and (b)'s x
        top = 20.0 * np.ceil((max(m['y'] for m in ma) + 4.0) / 20.0)
        share_ticks = list(np.arange(0.0, top + 1e-9, 20.0))
        s_hi = 5.0 * np.ceil((max(m['x'] for m in ma) + 2.5) / 5.0)
        setup_ticks = _fq4_ticks(0.0, s_hi, nbins=4)

        # Layout in inches (printed size), bottom to top.
        fig_h = 3.10
        left, gap, right = 0.66, 0.80, 0.10
        wa = 2.05
        wb = FIG_W - left - gap - right - wa
        leg_y, leg_h = 0.04, 0.25
        ax_y0 = leg_y + leg_h + 0.50
        ax_h = fig_h - ax_y0 - 0.26

        # Panel (b): room for one row of codes (12 pt) beyond the outermost intervals.
        d_lo = min(m.get('lo', m['y']) for m in mb)
        d_hi = max(max(m.get('hi', m['y']) for m in mb), 0.0)
        span = d_hi - d_lo
        pad = max(0.07 * span, 12.0 * span / (ax_h * 72.0 - 24.0))
        dlim = (d_lo - pad, d_hi + pad)
        d_ticks = _fq4_ticks(*dlim, nbins=7)
        fig = plt.figure(figsize=(FIG_W, fig_h), dpi=300)
        # The combinations without setups sit at zero setup share, and the factory
        # points and the changed-setup levels lie close to their right: room left of
        # zero for labels.
        room = 0.05 * s_hi
        if (F is not None or S is not None) and 'a' in codes:
            rend = fig.canvas.get_renderer()
            w = 0.0
            for m in ma:
                if m.get('label') and m['x'] <= 1e-9:
                    t = fig.text(0, 0, comb_tex(m['label']), fontsize=FS_TICK)
                    w = max(w, t.get_window_extent(rend).width * 72.0 / fig.dpi)
                    t.remove()
            need = FQ4_STYLE['factory']['ms'] / 2 + 0.5 + FQ4_GAP + w + 2.0
            room = max(room, need * s_hi / (wa * 72.0 - need))
        setup_lim = (-room, s_hi)
        ax = fig.add_axes([left / FIG_W, ax_y0 / fig_h, wa / FIG_W, ax_h / fig_h])
        bx = fig.add_axes([(left + wa + gap) / FIG_W, ax_y0 / fig_h, wb / FIG_W,
                           ax_h / fig_h])
        for a, xt, yt in ((ax, setup_ticks, share_ticks), (bx, share_ticks, d_ticks)):
            for t in xt:
                a.axvline(t, color=C_GRID, lw=0.4, zorder=0.5)
            for t in yt:
                if abs(t) > 1e-9 or a is ax:
                    a.axhline(t, color=C_GRID, lw=0.4, zorder=0.5)
            a.xaxis.set_major_locator(FixedLocator(xt))
            a.yaxis.set_major_locator(FixedLocator(yt))
            a.xaxis.set_major_formatter(FuncFormatter(lambda v, _: _fmt(round(v, 6))))
            a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _fmt(round(v, 6))))
            a.tick_params(axis='both', length=3, pad=2)
        bx.axhline(0, color=C_ZERO, lw=0.6, zorder=2)
        ax.set_xlim(*setup_lim)
        ax.set_ylim(0.0, top)
        bx.set_xlim(0.0, top)
        bx.set_ylim(*dlim)
        for m in ma:
            _fq4_point(ax, m)
        for m in mb:
            _fq4_point(bx, m)
        ax.set_title('(a)', fontsize=FS_TITLE, pad=7)
        bx.set_title('(b)', fontsize=FS_TITLE, pad=7)
        ax.set_xlabel('setup share of machine capacity (%)', labelpad=4)
        ax.set_ylabel('decisions with one\nlegal pair (%)', multialignment='center',
                      labelpad=5)
        bx.set_xlabel('decisions with one legal pair (%)', labelpad=4)
        bx.set_ylabel('makespan difference,\npolicy minus best rule (%)',
                      multialignment='center', labelpad=5)

        fig.canvas.draw()
        _, na = _fq4_place_labels(fig, ax, ma, 'point')
        _, nb = _fq4_place_labels(fig, bx, mb, 'ci')
        notes.append(f'variant {variant}: panel (a) ' +
                     (('; '.join(na) or 'each beside its point') if 'a' in codes else
                      'no codes') + '; panel (b) ' + ('; '.join(nb) or 'each beside its point'))

        styles = [b for b, _ in benches] + (['sens'] if S is not None else [])
        hs = []
        for s in styles:
            st = FQ4_STYLE[s]
            face = dict(mfc='white', mec=st['color'], mew=0.8) if st['open'] else \
                dict(mfc=st['color'], mec='white', mew=0.5)
            hs.append(Line2D([0], [0], ls='none', marker=st['marker'], ms=st['ms'],
                             label=st['text'], **face))
        leg = fig.legend(handles=hs, loc='lower center',
                         bbox_to_anchor=(0.5, leg_y / fig_h), ncol=len(hs),
                         fontsize=FS_TICK, frameon=True, fancybox=False,
                         edgecolor='0.6', framealpha=1.0, handlelength=1.0,
                         handletextpad=0.5, columnspacing=1.6, borderpad=0.4)
        leg.get_frame().set_linewidth(0.5)

        n_text = _check_layout(fig, (ax, bx), leg, tag='fq4')
        n_lab = _fq4_check_marks(fig, ((ax, ma), (bx, mb)))
        stem = os.path.join(out_dir, f'fq4_boundary_{variant}{suffix}')
        fig.savefig(stem + '.pdf')
        fig.savefig(stem + '.png', dpi=300)
        plt.close(fig)
        written += [stem + '.pdf', stem + '.png']
        print(f'  variant {variant} ({form}): {FIG_W:.2f} x {fig_h:.2f} in, setup share 0 to '
              f'{s_hi:g} %, one pair 0 to {top:g} %, difference {dlim[0]:+.2f} to '
              f'{dlim[1]:+.2f} %, {n_text} texts checked, {n_lab} labels clear of every '
              'mark, no overlap')

    # ---- the plotted numbers, and the macro gate
    print('\n  benchmark  comb.  one pair  setup   form           mean     lo      hi     '
          'verdict')
    for b, D in benches:
        for r in COMBOS:
            p = D['points'][r]
            for form, it in p['forms'].items():
                print(f'  {b:9s}  {r:5s}  {p["one_pair"]:7.2f}  {p["setup_share"]:6.2f}  '
                      f'{form:13s}  {it["mean"]:+6.2f}  {it["lo"]:+6.2f}  {it["hi"]:+6.2f}  '
                      f'{it["verdict"]}')
    if S is not None:
        print('\n  changed setups (factory, one greedy schedule against the best rule)\n'
              '  level             comb.  one pair  setup    mean     lo      hi')
        for p in S['points']:
            print(f'  {p["level"]:17s} {p["combo"]:5s}  {p["one_pair"]:7.2f}  '
                  f'{p["setup_share"]:6.2f}  {p["mean"]:+6.2f}  {p["lo"]:+6.2f}  '
                  f'{p["hi"]:+6.2f}')
    print('\n  source       macro                                file         macro     check')
    for fam, name, v, got, status in table:
        fv = v if isinstance(v, str) else f'{v:+.4f}'
        print(f'  {fam:11s}  {name:36s} {fv:>12s}  {str(got):>9s}  {status}')
    nm = sum(s == 'match' for *_, s in table)
    nc = sum(s != 'no macro file' for *_, s in table)
    print(f'\n  macro gate: {nm} of {nc} checked values match'
          + (f', {len(table) - nc} values without a macro file' if len(table) > nc else ''))
    for n in notes:
        print(f'  labels, {n}')
    return written


# ------------------------------------------------------------------ fs_convergence
FS_CONV_JSON = 'results/convergence.json'
FS_CONV_MACROS = 'manuscript/macros_convergence.tex'
FS_CONV_GENERATOR = 'scripts/p116_convergence.py'
# Both panels: the validation score itself (rel of train_family.py, the greedy validation
# makespan divided by the MWKR rule's on the same instances) against the training updates, 0 to
# 2000, and a thin dashed vertical line at update 1000, where the specialists' training ends.
# Panel (a): the specialists trained for 2000 updates, each combination one thin solid line in a
# light slate (at each validation the median over its seeds), the mean of these eight lines as one
# bold line in the dark slate of the specialists, and the mean of the eight specialists trained
# for 1000 updates (the same per-combination medians) as one bold line in the medium green of
# fq4_limits, which ends at update 1000; a legend of three entries names the three kinds of line.
# Panel (b): the policy in the house blue and the single-rule control in the medium rose of fq3
# and fq4, each its median over the seeds over its range across the seeds as a pale band of its
# own colour.
FS_CONV_SPEC = (('N', 'L', 'S', 'W'), ('LS', 'LW', 'SW', 'LSW'))
FS_CONV_SPEC_COLORS = ('#3e4a56', '#8d99a6')
FS_CONV_ONE_COLOR, FS_CONV_ONE_LW = '#9ba6b1', 0.75     # one specialist: thin, light slate
FS_CONV_MED_COLOR, FS_CONV_MED_LW = FS_CONV_SPEC_COLORS[0], 1.7   # their mean: bold, dark
FS_CONV_STD_COLOR, FS_CONV_STD_LW = '#4a8f5c', 1.7      # mean of the 1000-update runs: green
FS_CONV_ONE_TEXT = 'one specialist, 2,000 updates'
FS_CONV_MED_TEXT = 'mean of the {}, 2,000 updates'
FS_CONV_STD_TEXT = 'mean of the {}, 1,000 updates'
FS_CONV_COUNT_WORDS = ('one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight')
FS_CONV_LW = 1.3
FS_CONV_B = (('released', 'policy', C_EQUIV), ('control', 'single-rule control', '#d07a87'))
FS_CONV_BAND_ALPHA = 0.18
FS_CONV_XMAX = 2000                              # updates of the policy and of arm long
FS_CONV_XTICKS = (0, 500, 1000, 1500, 2000)
FS_CONV_SPLIT = 1000                             # the specialists' updates: the dashed line
FS_CONV_SPLIT_STYLE = dict(color=C_ZERO, lw=0.7, ls=(0, (3, 2)))
# The y axis of a panel covers every line and band of the panel from this update on; a value
# before it that lies higher leaves the axis at the top, and the run prints it.
FS_CONV_FROM = 200
# The panels share the y axis when the span of each panel's values covers at least this share
# of the span of both panels' values together; otherwise each panel has its own tick labels.
FS_CONV_SHARE_MIN = 2.0 / 3.0
# Tick steps added at the top of a panel, at most, until its legend sits clear of every line
FS_CONV_HEADROOM_MAX = 4
FS_CONV_TOL = 1e-9


def _fs_conv_refuse(msg):
    raise SystemExit(f'fs_convergence REFUSING: {msg}')


def _fs_conv_group(J, key):
    """The stored curve group of key ('released', 'control' or a combination) and its runs."""
    if key in ('released', 'control'):
        return J['curves'].get(key), [r for r in J['runs'] if r['arm'] == key]
    return (J['curves'].get('specialist', {}).get(key),
            [r for r in J['runs'] if r['arm'] == 'specialist' and r['combo'] == key])


def _fs_conv_raw_check(g, rows, what):
    """A stored raw curve group against its runs: the validation updates, and the median, minimum
    and maximum over the seeds of the score at each of them."""
    grids = {tuple(int(u) for u, _ in r['curve']) for r in rows}
    if len(grids) != 1:
        _fs_conv_refuse(f'the runs of {what} do not validate at the same updates')
    if [int(u) for u in g['update']] != list(grids.pop()):
        _fs_conv_refuse(f'the stored updates of {what} differ from those of its runs')
    a = np.array([[s for _, s in r['curve']] for r in rows], dtype=float)
    for q, f in (('median', np.median), ('min', np.min), ('max', np.max)):
        if not np.allclose(f(a, axis=0), g[q], rtol=0, atol=FS_CONV_TOL):
            _fs_conv_refuse(f'the stored {q} of {what} is not the {q} over its runs')
    if int(g['n']) != len(rows):
        _fs_conv_refuse(f'{what} counts {g["n"]} runs, the record holds {len(rows)}')


def _fs_conv_change(curve, windows):
    """100 (mean score over windows['late'] / mean score over windows['ref'] - 1), each window
    (a, b] of updates."""
    u = np.array([x for x, _ in curve], dtype=float)
    s = np.array([y for _, y in curve], dtype=float)
    (ra, rb), (la, lb) = windows['ref'], windows['late']
    ref, late = s[(u > ra) & (u <= rb)], s[(u > la) & (u <= lb)]
    if not len(ref) or not len(late):
        _fs_conv_refuse(f'a curve has no validation in {windows["ref"]} or in {windows["late"]}')
    return float(100.0 * (late.mean() / ref.mean() - 1.0))


def _fs_conv_tex_list(combos):
    t = [f'${c}$' for c in combos]
    if not t:
        return 'none'
    return t[0] if len(t) == 1 else ', '.join(t[:-1]) + ' and ' + t[-1]


def load_convergence(path, allow_partial):
    """The record of p116, checked: its generator, partial only with allow_partial, every
    group of a full record present, the specialists trained for 2000 updates (arm long) present,
    every stored curve group equal to the median, minimum and maximum over its runs of
    100 (score / best - 1), every raw curve group equal to the median, minimum and maximum over
    its runs of the score, the means of the specialists' per-combination medians, and the
    change after update 1000 of arm long recomputed from the runs' curves."""
    if not os.path.exists(path):
        _fs_conv_refuse(f'{path} not found (written by {FS_CONV_GENERATOR})')
    J = json.load(open(path))
    if J.get('generator') != FS_CONV_GENERATOR:
        _fs_conv_refuse(f'{path} was written by {J.get("generator")!r}, not {FS_CONV_GENERATOR}')
    for k in ('seeds', 'partial', 'runs', 'arms', 'curves', 'raw_curves'):
        if k not in J:
            _fs_conv_refuse(f'{path} has no key {k!r}' + (
                ' (a record without the raw curves and the specialists trained for 2000 updates)'
                if k == 'raw_curves' else ''))
    if J['partial'] and not allow_partial:
        _fs_conv_refuse(f'{path} is the record of a partial run ({J.get("present")} of '
                        f'{J.get("expected")} runs); it is drawn for review only, into a folder '
                        'outside the repository')
    keys = ['released', 'control'] + [c for grp in FS_CONV_SPEC for c in grp]
    have = []
    for key in keys:
        g, rows = _fs_conv_group(J, key)
        if g is None or not rows:
            if not J['partial']:
                _fs_conv_refuse(f'{path} holds no curve group or no run of {key}')
            continue
        a = []
        for r in rows:
            u = np.array([x for x, _ in r['curve']], dtype=float)
            s = np.array([y for _, y in r['curve']], dtype=float)
            if not np.allclose(100.0 * u / r['updates'], g['share'], rtol=0, atol=FS_CONV_TOL):
                _fs_conv_refuse(f'{r["name"]}: shares of training differ from its group {key}')
            v = 100.0 * (s / s.min() - 1.0)
            if v.min() != 0.0:
                _fs_conv_refuse(f'{r["name"]}: the curve does not reach its own best')
            a.append(v)
        a = np.array(a)
        for q, f in (('median', np.median), ('min', np.min), ('max', np.max)):
            if not np.allclose(f(a, axis=0), g[q], rtol=0, atol=FS_CONV_TOL):
                _fs_conv_refuse(f'the stored {q} of group {key} is not the {q} over its runs')
        if int(g['n']) != len(rows):
            _fs_conv_refuse(f'group {key} counts {g["n"]} runs, the record holds {len(rows)}')
        have.append(key)
    # the raw curves: the policy and the control, and per combination the specialists of both
    # lengths with the mean of their per-combination medians
    R = J['raw_curves']
    long_rows = [r for r in J['runs'] if r['arm'] == 'long']
    if not long_rows or 'long' not in R or 'long' not in J['arms']:
        _fs_conv_refuse(f'{path} holds no run of the specialists trained for 2000 updates')
    for arm in ('released', 'control'):
        rows = [r for r in J['runs'] if r['arm'] == arm]
        if not rows:
            if not J['partial']:
                _fs_conv_refuse(f'{path} holds no run of {arm}')
            if arm in R:
                _fs_conv_refuse(f'{path} holds a raw curve of {arm} without its runs')
            continue
        if arm not in R:
            _fs_conv_refuse(f'{path} holds no raw curve of {arm}')
        _fs_conv_raw_check(R[arm], rows, f'the raw curve of {arm}')
    for arm in ('specialist', 'long'):
        if arm not in R:
            if not J['partial'] or arm == 'long':
                _fs_conv_refuse(f'{path} holds no raw curves of arm {arm}')
            continue
        G = R[arm]
        present = [c for c in COMBOS if any(r['arm'] == arm and r['combo'] == c
                                            for r in J['runs'])]
        if sorted(G['combos']) != sorted(present):
            _fs_conv_refuse(f'the raw curves of arm {arm} cover {sorted(G["combos"])}, its runs '
                            f'{present}')
        if not J['partial'] and present != COMBOS:
            _fs_conv_refuse(f'arm {arm} lacks the combinations '
                            f'{[c for c in COMBOS if c not in present]}')
        for c in present:
            _fs_conv_raw_check(G['combos'][c], [r for r in J['runs'] if r['arm'] == arm
                                                and r['combo'] == c], f'{arm} {c}')
        M = G['mean']
        if list(M['combos']) != present or int(M['n']) != len(present):
            _fs_conv_refuse(f'the mean of arm {arm} covers {M["combos"]}, its runs {present}')
        if [int(u) for u in M['update']] != [int(u) for u in G['combos'][present[0]]['update']]:
            _fs_conv_refuse(f'the mean of arm {arm} has other updates than its combinations')
        m = np.mean([G['combos'][c]['median'] for c in present], axis=0)
        if not np.allclose(m, M['mean'], rtol=0, atol=FS_CONV_TOL):
            _fs_conv_refuse(f'the stored mean of arm {arm} is not the mean of its '
                            'per-combination medians')
    # arm long: the change after update 1000, per run, per combination and over all runs
    L = J['arms']['long']
    win = L['windows']
    for r in long_rows:
        v = _fs_conv_change(r['curve'], win)
        if abs(v - r['change_after_1000']) > FS_CONV_TOL:
            _fs_conv_refuse(f'{r["name"]}: the stored change after update 1000 is not that of '
                            'its curve')
    v = np.array([r['change_after_1000'] for r in long_rows], dtype=float)
    want = dict(median=float(np.median(v)), min=float(v.min()), max=float(v.max()))
    for q, x in want.items():
        if abs(L['pooled'][q] - x) > FS_CONV_TOL:
            _fs_conv_refuse(f'the stored {q} of arm long is not the {q} over its runs')
    order = sorted(long_rows, key=lambda r: (COMBOS.index(r['combo']), r['seed']))
    for key, run in (('min_run', min(order, key=lambda r: r['change_after_1000'])),
                     ('max_run', max(order, key=lambda r: r['change_after_1000']))):
        if (L[key]['combo'], int(L[key]['seed'])) != (run['combo'], int(run['seed'])):
            _fs_conv_refuse(f'the stored {key} of arm long is not the run {run["name"]}')
    per = {c: float(np.median([r['change_after_1000'] for r in long_rows if r['combo'] == c]))
           for c in COMBOS if any(r['combo'] == c for r in long_rows)}
    if sorted(per) != sorted(L['combos']) or any(
            abs(L['combos'][c]['median'] - per[c]) > FS_CONV_TOL for c in per):
        _fs_conv_refuse('the stored medians per combination of arm long are not those of its runs')
    neg = [c for c in COMBOS if c in per and per[c] < 0]
    if list(L['negative']['combos']) != neg or int(L['negative']['n']) != len(neg):
        _fs_conv_refuse('the stored combinations of arm long with a negative median are not '
                        'those of its runs')
    if int(L['n']) != len(long_rows) or sorted(L['seeds']) != sorted({r['seed']
                                                                      for r in long_rows}):
        _fs_conv_refuse('the stored count of runs or seeds of arm long is not that of its runs')
    return J, have


def gate_convergence(J, path):
    """The record against the macro file of the same run (prefix Cvg): counts exactly, signed
    values at their two printed decimals, shares of training at the printed digits and the
    combinations as letters; the macros CvgLong of the specialists trained for 2000 updates
    likewise, their lists of combinations as names and as running text. Refuses a missing file
    or any disagreement."""
    if not path or not os.path.exists(path):
        _fs_conv_refuse(f'the macro file {path} not found (written by {FS_CONV_GENERATOR})')
    mac = _macros(path)
    A = J['arms']
    want = [('CvgSeeds', len(J['seeds']), 'count')]
    words = (('Spec', 'specialist'), ('Pol', 'released'), ('Ctl', 'control'))
    for w, arm in words:
        if arm not in A:
            continue
        S = A[arm]['pooled'] if arm == 'specialist' else A[arm]
        if arm == 'specialist':
            want.append(('CvgSpecRuns', S['n'], 'count'))
        for q, qw in (('median', 'Med'), ('min', 'Min'), ('max', 'Max')):
            want.append((f'Cvg{w}Late{qw}', S['late_change'][q], 'signed'))
        for q, qw in (('min', 'Min'), ('max', 'Max')):
            want.append((f'Cvg{w}Gain{qw}', S['gain_second_half'][q], 'signed'))
        want.append((f'Cvg{w}BestAtMed', 100.0 * S['best_at']['median'], 'share'))
    if 'specialist' in A:
        want += [('CvgSpecLateMinCombo', A['specialist']['largest_late_improvement']['combo'],
                  'text'),
                 ('CvgSpecLateMedLow', A['specialist']['lowest_combo_median']['late_change'],
                  'signed'),
                 ('CvgSpecLateMedLowCombo', A['specialist']['lowest_combo_median']['combo'],
                  'text')]
    if 'long' in A:
        L = A['long']
        neg = list(L['negative']['combos'])
        want += [('CvgLongRuns', L['n'], 'count'), ('CvgLongSeeds', len(L['seeds']), 'count'),
                 ('CvgLongMed', L['pooled']['median'], 'signed'),
                 ('CvgLongMin', L['pooled']['min'], 'signed'),
                 ('CvgLongMax', L['pooled']['max'], 'signed'),
                 ('CvgLongMinCombo', L['min_run']['combo'], 'text'),
                 ('CvgLongMaxCombo', L['max_run']['combo'], 'text')]
        want += [(f'CvgLongMed{c}', L['combos'][c]['median'], 'signed')
                 for c in COMBOS if c in L['combos']]
        want += [('CvgLongNegN', len(neg), 'count'),
                 ('CvgLongNegCombos', ', '.join(neg) if neg else 'none', 'text'),
                 ('CvgLongNegList', _fs_conv_tex_list(neg), 'text')]
    bad, rows = [], []
    for name, v, kind in want:
        got = mac.get(name)
        if got is None:
            bad.append(f'\\{name}: no macro')
            continue
        if kind == 'count':
            ok = got == str(int(v))
        elif kind == 'text':
            ok = got == v
        else:
            try:
                ok = abs(float(got) - float(v)) <= 0.005 + 1e-9
            except ValueError:
                ok = False
            if ok and kind == 'signed':
                ok = re.fullmatch(r'[+-]\d+\.\d\d', got) is not None
        rows.append((name, v, got, 'match' if ok else 'MISMATCH'))
        if not ok:
            bad.append(f'\\{name} = {got}, record {v}')
    if bad:
        _fs_conv_refuse(f'{path} disagrees with the record:\n  ' + '\n  '.join(bad))
    return rows


def _fs_conv_clear(ax, leg, series, rend):
    """True when no line or band of `series` [(x, lo, hi)] enters the legend's box (padded by
    2 pt) and the box lies inside the axes."""
    pad = 2.0 * ax.figure.dpi / 72.0
    bb = leg.get_window_extent(rend)
    ab = ax.get_window_extent(rend)
    if bb.x0 < ab.x0 or bb.y0 < ab.y0 or bb.x1 > ab.x1 or bb.y1 > ab.y1:
        return False
    inv = ax.transData.inverted()
    (x0, y0), (x1, y1) = inv.transform([(bb.x0 - pad, bb.y0 - pad), (bb.x1 + pad, bb.y1 + pad)])
    grid = np.linspace(x0, x1, 400)
    for x, lo, hi in series:
        g = grid[(grid >= x[0]) & (grid <= x[-1])]
        if not len(g):
            continue
        ylo, yhi = np.interp(g, x, lo), np.interp(g, x, hi)
        if np.any((yhi >= y0) & (ylo <= y1)):
            return False
    return True


def _fs_conv_try_legend(fig, ax, handles, labels, series, ncol):
    """A legend inside the axes in a thin box, in the first corner where no line or band enters
    it, as (legend, corner); None when no corner is clear."""
    rend = fig.canvas.get_renderer()
    for loc in ('upper right', 'center right', 'upper center'):
        leg = ax.legend(handles, labels, loc=loc, ncol=ncol, fontsize=FS_TICK, frameon=True,
                        fancybox=False, edgecolor='0.6', framealpha=1.0, handlelength=2.6,
                        handletextpad=0.5, columnspacing=1.2, borderpad=0.4, labelspacing=0.3,
                        borderaxespad=0.5)
        leg.get_frame().set_linewidth(0.5)
        leg.set_zorder(6)
        fig.canvas.draw()
        if _fs_conv_clear(ax, leg, series, rend):
            return leg, loc
        leg.remove()
    return None


def _fs_conv_ticks(lo, hi):
    """(ticks, step): ticks at a round step from the step below lo to the step above hi."""
    raw = MaxNLocator(nbins=5, steps=[1, 2, 5, 10]).tick_values(lo, hi)
    step = float(raw[1] - raw[0])
    k0, k1 = int(np.floor(lo / step + 1e-9)), int(np.ceil(hi / step - 1e-9))
    return [round(k * step, 6) for k in range(k0, k1 + 1)], step


def _fs_conv_xfmt(v, _):
    return f'{int(round(v)):,}'


def fs_convergence(conv_json, conv_macros, out_dir, *, test_only_partial_review=False):
    """Draw the figure from the record of p116 after its gates, then print the gate table.
    test_only_partial_review (keyword only, never set by the command line): the record may be
    that of p116 --partial, for a drawing outside the manuscript; out_dir must then lie outside
    the repository, and the file names of a partial record end in _review."""
    if test_only_partial_review:
        out_abs = os.path.abspath(out_dir)
        if os.path.commonpath([out_abs, ROOT]) == ROOT:
            _fs_conv_refuse(f'a review drawing is written outside the repository {ROOT}, not '
                            f'into {out_dir}')
    J, have = load_convergence(conv_json, test_only_partial_review)
    table = gate_convergence(J, conv_macros)
    suffix = '_review' if J['partial'] else ''
    print(f'  convergence: {_shown(conv_json)} ({J.get("setting", "?")}; {J.get("present")} of '
          f'{J.get("expected")} runs' + (', partial, for review only' if J['partial'] else '')
          + ')')
    R = J['raw_curves']

    def arr(g, k):
        return np.asarray(g[k], dtype=float)

    # (a) the specialists of 2000 updates per combination, their mean, and the mean of the
    # specialists of 1000 updates; (b) the policy and the control over their range
    long_c = [c for grp in FS_CONV_SPEC for c in grp if c in R['long']['combos']]
    lines_a = [(arr(R['long']['combos'][c], 'update'), arr(R['long']['combos'][c], 'median'))
               for c in long_c]
    mean_long = (arr(R['long']['mean'], 'update'), arr(R['long']['mean'], 'mean'))
    std = R.get('specialist')
    mean_std = None if std is None else (arr(std['mean'], 'update'), arr(std['mean'], 'mean'))
    pb = [(arm, text, col) for arm, text, col in FS_CONV_B if arm in R]
    bands_b = [(text, col, arr(R[arm], 'update'), arr(R[arm], 'median'), arr(R[arm], 'min'),
                arr(R[arm], 'max')) for arm, text, col in pb]

    # y range of each panel: every value from update FS_CONV_FROM on, and every lower value
    va = [(x, y) for x, y in lines_a + [mean_long] + ([mean_std] if mean_std else [])]
    vb = [(x, y) for _, _, x, m, lo, hi in bands_b for y in (m, lo, hi)]

    def span(vals):
        lo = min(float(y.min()) for _, y in vals)
        hi = max(float(y[x >= FS_CONV_FROM].max()) for x, y in vals)
        return lo, hi

    (lo_a, hi_a), (lo_b, hi_b) = span(va), span(vb)
    lo_c, hi_c = min(lo_a, lo_b), max(hi_a, hi_b)
    shared = min(hi_a - lo_a, hi_b - lo_b) >= FS_CONV_SHARE_MIN * (hi_c - lo_c)
    if shared:
        lo_a = lo_b = lo_c
        hi_a = hi_b = hi_c

    fig_h = 2.75
    left, right = 0.66, 0.13
    gap = 0.30 if shared else 0.52                  # room for the tick labels of panel (b)
    pw = (FIG_W - left - right - gap) / 2
    ax_y0 = 0.46
    ax_h = fig_h - ax_y0 - 0.28
    fig = plt.figure(figsize=(FIG_W, fig_h), dpi=300)
    ax = fig.add_axes([left / FIG_W, ax_y0 / fig_h, pw / FIG_W, ax_h / fig_h])
    bx = fig.add_axes([(left + pw + gap) / FIG_W, ax_y0 / fig_h, pw / FIG_W, ax_h / fig_h],
                      sharey=ax if shared else None)
    for a in (ax, bx):
        for t in FS_CONV_XTICKS[1:]:
            if t != FS_CONV_SPLIT:
                a.axvline(t, color=C_GRID, lw=0.4, zorder=0.5)
        a.axvline(FS_CONV_SPLIT, zorder=2, **FS_CONV_SPLIT_STYLE)
        a.set_xlim(0, 1.01 * FS_CONV_XMAX)
        a.xaxis.set_major_locator(FixedLocator(list(FS_CONV_XTICKS)))
        a.xaxis.set_major_formatter(FuncFormatter(_fs_conv_xfmt))
        a.tick_params(axis='both', length=3, pad=2)

    # (a): each line thin and light, all in one style; over them, in bold, the mean of these lines
    # at each validation, and the mean of the specialists of 1000 updates, which ends at 1000
    series_a = []
    for x, m in lines_a:
        ax.plot(x, m, color=FS_CONV_ONE_COLOR, lw=FS_CONV_ONE_LW, zorder=3,
                solid_capstyle='butt', solid_joinstyle='round')
        series_a.append((x, m, m))
    ax.plot(*mean_long, color=FS_CONV_MED_COLOR, lw=FS_CONV_MED_LW, zorder=4,
            solid_capstyle='butt', solid_joinstyle='round')
    series_a.append((mean_long[0], mean_long[1], mean_long[1]))
    ha = [Line2D([0], [0], color=FS_CONV_ONE_COLOR, lw=FS_CONV_ONE_LW),
          Line2D([0], [0], color=FS_CONV_MED_COLOR, lw=FS_CONV_MED_LW)]
    la = [FS_CONV_ONE_TEXT, FS_CONV_MED_TEXT.format(FS_CONV_COUNT_WORDS[len(long_c) - 1])]
    if mean_std is not None:
        ax.plot(*mean_std, color=FS_CONV_STD_COLOR, lw=FS_CONV_STD_LW, zorder=5,
                solid_capstyle='butt', solid_joinstyle='round')
        series_a.append((mean_std[0], mean_std[1], mean_std[1]))
        ha.append(Line2D([0], [0], color=FS_CONV_STD_COLOR, lw=FS_CONV_STD_LW))
        la.append(FS_CONV_STD_TEXT.format(FS_CONV_COUNT_WORDS[int(std['mean']['n']) - 1]))
    # (b): the policy and the single-rule control, median over the range across the seeds
    series_b, hb, lb_ = [], [], []
    for text, col, x, m, lo, hi in bands_b:
        bx.fill_between(x, lo, hi, color=col, alpha=FS_CONV_BAND_ALPHA, lw=0, zorder=1)
        bx.plot(x, m, color=col, lw=FS_CONV_LW, zorder=4, solid_capstyle='butt')
        series_b.append((x, lo, hi))
        hb.append((Patch(facecolor=col, alpha=FS_CONV_BAND_ALPHA, edgecolor='none'),
                   Line2D([0], [0], color=col, lw=FS_CONV_LW)))
        lb_.append(text)

    ax.set_ylabel('validation makespan relative to\nthe most-work-remaining rule',
                  multialignment='center',
                  labelpad=5)
    ax.set_title('(a) Specialists', fontsize=FS_TITLE, pad=7)
    bx.set_title('(b) Policy and single-rule control', fontsize=FS_TITLE, pad=7)
    xc = (left + pw + gap / 2) / FIG_W
    fig.text(xc, (ax_y0 - 0.32) / fig_h, 'training updates', ha='center', va='center',
             fontsize=FS_LABEL)

    # each panel's ticks, raised by whole steps at the top until its legend sits clear
    ticks_a, step_a = _fs_conv_ticks(lo_a, hi_a)
    ticks_b, step_b = _fs_conv_ticks(lo_b, hi_b)
    grid_lines = []

    def limits(a, ticks, step, extra):
        tk = ticks + [round(ticks[-1] + k * step, 6) for k in range(1, extra + 1)]
        pad = 0.04 * (tk[-1] - tk[0])
        a.set_ylim(tk[0] - pad, tk[-1] + pad)
        a.yaxis.set_major_locator(FixedLocator(tk))
        a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.2f}'))
        return tk

    def place(extra_a, extra_b):
        tk_a = limits(ax, ticks_a, step_a, extra_a)
        tk_b = tk_a if shared else limits(bx, ticks_b, step_b, extra_b)
        got_a = _fs_conv_try_legend(fig, ax, ha, la, series_a, ncol=1)
        got_b = _fs_conv_try_legend(fig, bx, hb, lb_, series_b, ncol=1)
        return tk_a, tk_b, got_a, got_b

    extra_a = extra_b = 0
    while True:
        tk_a, tk_b, got_a, got_b = place(extra_a, extra_b)
        if got_a is not None and got_b is not None:
            break
        for g in (got_a, got_b):
            if g is not None:
                g[0].remove()
        if ((got_a is None and extra_a >= FS_CONV_HEADROOM_MAX)
                or (got_b is None and extra_b >= FS_CONV_HEADROOM_MAX)):
            _fs_conv_refuse('no corner of a panel leaves the legend clear of the lines, with '
                            f'{FS_CONV_HEADROOM_MAX} tick steps added at the top')
        if shared:
            extra_a = extra_b = extra_a + 1
        else:
            extra_a += got_a is None
            extra_b += got_b is None
    (leg_a, loc_a), (leg_b, loc_b) = got_a, got_b
    for a, tk in ((ax, tk_a), (bx, tk_b)):
        for t in tk:
            grid_lines.append(a.axhline(t, color=C_GRID, lw=0.4, zorder=0.5))
    if shared:
        plt.setp(bx.get_yticklabels(), visible=False)
    clipped = []
    for name, a, vals in (('(a)', ax, va), ('(b)', bx, vb)):
        top = a.get_ylim()[1]
        for x, y in vals:
            for xi, yi in zip(x[y > top], y[y > top]):
                if xi >= FS_CONV_FROM:
                    _fs_conv_refuse(f'a value of panel {name} after update {FS_CONV_FROM} '
                                    'leaves the axis')
                clipped.append(f'{name} update {xi:.0f}: {yi:.4f}')
    n_text = _check_layout(fig, (), leg_a, tag='fs_convergence')

    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f'fs_convergence{suffix}')
    fig.savefig(stem + '.pdf')
    fig.savefig(stem + '.png', dpi=300)
    plt.close(fig)
    ya, yb = ax.get_ylim(), bx.get_ylim()
    print(f'  fs_convergence: {FIG_W:.2f} x {fig_h:.2f} in, y axis '
          + ('shared' if shared else 'per panel') + f', (a) {ya[0]:.3f} to {ya[1]:.3f}, '
          f'(b) {yb[0]:.3f} to {yb[1]:.3f} ({extra_a} and {extra_b} tick steps above the data '
          f'for the legends), legends {loc_a} and {loc_b} clear of the lines, {n_text} texts '
          'checked, no overlap')
    for c in clipped:
        print(f'  value above the axis (before update {FS_CONV_FROM}): {c}')
    print('\n  macro                      record     macro      check')
    for name, v, got, status in table:
        vs = f'{v:+.4f}' if isinstance(v, float) else str(v)
        print(f'  {name:26s} {vs:>10s} {got:>8s}   {status}')
    return [stem + '.pdf', stem + '.png']


# ------------------------------------------------------------------ fq2_reuse
FQ2_JSON = 'results/cross_eval.json'
FQ2_MACROS = 'manuscript/macros_crosseval.tex'
FQ2_GENERATOR = 'scripts/p100_cross_eval.py'
FQ2_SIGN = 'd = 100 (specialist - policy) / policy; positive = specialist longer'
FQ2_PREFIX = 'Xev'
FQ2_ALL = 'LSW'                                   # the specialist of all three rules
FQ2_CELL_KEYS = ('mean', 'lo', 'hi', 'v1.0', 'direction')
FQ2_COUNTS = (('XevSeedN', 'seeds'), ('XevInstN', 'n'), ('XevBoot', 'boot'))
# The three kinds of mark in a column, all specialists, so all in the slate of the
# specialists (fs_convergence): the two with intervals dark, the others light and small.
FQ2_STYLE = {
    'own': dict(text='specialist of the combination', marker='o', ms=MS_CIRCLE,
                color=FS_CONV_SPEC_COLORS[0], open=False),
    'all': dict(text='specialist of all three rules', marker='s', ms=MS_SQUARE,
                color=FS_CONV_SPEC_COLORS[0], open=True),
    'other': dict(text='other specialists', marker='o', ms=3.0,
                  color=FS_CONV_SPEC_COLORS[1], open=True),
}
FQ2_MEW_OPEN, FQ2_MEW_FILLED, FQ2_MEW_OTHER = 1.0, 0.5, 0.7
FQ2_CI_LW = 1.2
# Horizontal places in a column, in pt from the tick: the specialist of the combination and
# the specialist of all three rules side by side left of the tick, the other specialists
# right of it, each at the free place nearest to FQ2_SWARM_X0 among the places
# FQ2_SWARM_X0 + k * FQ2_SWARM_STEP (k = 0, 1, -1, 2, -2, ...) inside the column, so that
# no two of them overlap.
FQ2_DX = {'own': -16.0, 'all': -7.0}
FQ2_SWARM_X0, FQ2_SWARM_STEP = 9.0, 0.9
FQ2_SWARM_SLOTS = tuple(s * k for k in range(41) for s in ((1,) if k == 0 else (1, -1)))
FQ2_CLEAR = 0.8                                   # pt between two marks of a column


def _fq2_refuse(msg):
    raise SystemExit(f'fq2_reuse REFUSING: {msg}')


def load_crosseval(path):
    """The 8 x 8 record of p100: per cell (trained for, run under) the mean, the 95% interval,
    the verdict at the 1% margin and the direction, checked: its generator, environment and
    sign, all 64 cells present with finite values, each mean inside its interval, and each
    verdict and direction those that stats_family.verdict and stats_family.direction give
    for the interval (so a misread key cannot reach the plot). Refuses a missing or
    incomplete record and names what is missing."""
    import stats_family as SF
    if not path or not os.path.exists(path):
        _fq2_refuse(f'{path} not found (written by {FQ2_GENERATOR} under PACT_ENV=exact; the '
                    'figure reads it through arm_registry.out, so give the setting of that run, '
                    'or --crosseval)')
    with open(path) as f:
        J = json.load(f)
    lack = [k for k in ('generator', 'environment', 'sign', 'seeds', 'n', 'boot', 'cells')
            if k not in J]
    if lack:
        _fq2_refuse(f'{path} has no key {lack}')
    if J['generator'] != FQ2_GENERATOR:
        _fq2_refuse(f'{path} was written by {J["generator"]!r}, not {FQ2_GENERATOR}')
    if J['environment'] != 'exact':
        _fq2_refuse(f'{path} holds the environment {J["environment"]!r}, not exact')
    if J['sign'] != FQ2_SIGN:
        _fq2_refuse(f'{path} states the sign {J["sign"]!r}, not {FQ2_SIGN!r}')
    cells, missing, bad = {}, [], []
    for r in COMBOS:
        for c in COMBOS:
            key = f'{r}|{c}'
            x = J['cells'].get(key)
            if not isinstance(x, dict) or any(k not in x for k in FQ2_CELL_KEYS):
                missing.append(key)
                continue
            try:
                mean, lo, hi = (float(x[k]) for k in ('mean', 'lo', 'hi'))
            except (TypeError, ValueError):
                bad.append(f'{key}: a value that is not a number')
                continue
            if not all(np.isfinite(v) for v in (mean, lo, hi)):
                bad.append(f'{key}: a value that is not finite')
            elif not lo <= mean <= hi:
                bad.append(f'{key}: the mean {mean:+.4f} lies outside [{lo:+.4f}, {hi:+.4f}]')
            elif x['v1.0'] != SF.verdict(lo, hi, MARGIN):
                bad.append(f'{key}: verdict {x["v1.0"]!r} at the 1% margin, the interval '
                           f'[{lo:+.4f}, {hi:+.4f}] gives {SF.verdict(lo, hi, MARGIN)!r}')
            elif x['direction'] != SF.direction(lo, hi):
                bad.append(f'{key}: direction {x["direction"]!r}, the interval gives '
                           f'{SF.direction(lo, hi)!r}')
            cells[(r, c)] = dict(mean=mean, lo=lo, hi=hi, verdict=x['v1.0'],
                                 direction=x['direction'], key=key)
    if missing:
        _fq2_refuse(f'{path} is incomplete: {len(missing)} of {len(COMBOS) ** 2} cells missing '
                    f'or without {"/".join(FQ2_CELL_KEYS)} ({", ".join(missing[:8])}'
                    + (', ...' if len(missing) > 8 else '') + ')')
    if bad:
        _fq2_refuse(f'{path} holds cells that disagree with themselves:\n  ' + '\n  '.join(bad))
    return dict(cells=cells, seeds=list(J['seeds']), n=J['n'], boot=J['boot'], path=path)


def fq2_plan(X):
    """What each column draws, in the order of COMBOS: the specialist of the combination
    (own) with its interval; the specialist of all three rules (all) with its interval,
    except in its own column, where it is the own mark; the other specialists (other),
    means only. Returns {column: {'own': cell, 'all': cell or None, 'other': [(r, cell)]}}."""
    out = {}
    for c in COMBOS:
        rest = [r for r in COMBOS if r not in (c, FQ2_ALL)]
        out[c] = dict(own=X['cells'][(c, c)],
                      all=None if c == FQ2_ALL else X['cells'][(FQ2_ALL, c)],
                      other=[(r, X['cells'][(r, c)]) for r in rest])
    return out


def gate_crosseval(X, path):
    """Every plotted value against the macro of macros_crosseval.tex that prints it, at its two
    printed decimals and in the signed form of p100 (Xev<R>On<C> for a mean, with Lo and Hi for
    an interval); the counts XevSeedN, XevInstN and XevBoot exactly. A missing macro file, a
    missing count or a disagreement is refused; a plotted cell without a macro is compared
    with the record only and listed. Returns the table rows (kind, cell, field, value, macro,
    status)."""
    if not path or not os.path.exists(path):
        _fq2_refuse(f'the macro file {path} not found (written by {FQ2_GENERATOR})')
    mac = _macros(path)
    rows, bad = [], []
    for name, k in FQ2_COUNTS:
        want = len(X[k]) if k == 'seeds' else int(X[k])
        got = mac.get(name)
        if got is None:
            bad.append(f'\\{name}: no macro')
            continue
        ok = got == str(want)
        rows.append(('count', name, '', want, got, 'match' if ok else 'MISMATCH'))
        if not ok:
            bad.append(f'\\{name} = {got}, record {want}')
    for c, P in fq2_plan(X).items():
        marks = [('own', c, P['own'], ('mean', 'lo', 'hi'))]
        if P['all'] is not None:
            marks.append(('all', FQ2_ALL, P['all'], ('mean', 'lo', 'hi')))
        marks += [('other', r, it, ('mean',)) for r, it in P['other']]
        for kind, r, it, fields in marks:
            for fld in fields:
                name = f'{FQ2_PREFIX}{r}On{c}' + {'mean': '', 'lo': 'Lo', 'hi': 'Hi'}[fld]
                v, got = it[fld], mac.get(name)
                if got is None:
                    rows.append((kind, f'{r} under {c}', fld, v, None, 'no macro, record only'))
                    continue
                ok = re.fullmatch(r'[+-]\d+\.\d\d', got) is not None and \
                    abs(v - float(got)) <= 0.005 + 1e-9
                rows.append((kind, f'{r} under {c}', fld, v, got, 'match' if ok else 'MISMATCH'))
                if not ok:
                    bad.append(f'\\{name} = {got}, record {v:+.4f}')
    if bad:
        _fq2_refuse(f'{path} disagrees with the record:\n  ' + '\n  '.join(bad))
    return rows


def _fq2_swarm(ys, x_min, x_max, size):
    """x (pt) of points at heights ys (pt), each at the first place of FQ2_SWARM_SLOTS inside
    [x_min, x_max] that keeps it size + FQ2_CLEAR from every point placed before it, the
    points taken from the top down. Refuses when a point finds no place."""
    slots = [FQ2_SWARM_X0 + k * FQ2_SWARM_STEP for k in FQ2_SWARM_SLOTS
             if x_min <= FQ2_SWARM_X0 + k * FQ2_SWARM_STEP <= x_max]
    xs, placed = [None] * len(ys), []
    for i in sorted(range(len(ys)), key=lambda i: -ys[i]):
        for x in slots:
            if all(np.hypot(x - px, ys[i] - py) >= size + FQ2_CLEAR for px, py in placed):
                xs[i] = x
                placed.append((x, ys[i]))
                break
        else:
            _fq2_refuse('no free place beside the column for a mark of the other specialists')
    return xs


def _fq2_face(kind):
    """Face and edge of a marker: open markers white with a coloured edge, the filled marker
    coloured with a thin white edge."""
    st = FQ2_STYLE[kind]
    if st['open']:
        return dict(mfc='white', mec=st['color'],
                    mew=FQ2_MEW_OTHER if kind == 'other' else FQ2_MEW_OPEN)
    return dict(mfc=st['color'], mec='white', mew=FQ2_MEW_FILLED)


def _fq2_mark(ax, kind, x, it, with_ci):
    st = FQ2_STYLE[kind]
    if with_ci:
        ax.plot([x, x], [it['lo'], it['hi']], color=st['color'], lw=FQ2_CI_LW,
                solid_capstyle='butt', zorder=4)
    ax.plot([x], [it['mean']], st['marker'], ms=st['ms'], ls='none',
            zorder=3.5 if kind == 'other' else 5, **_fq2_face(kind))


def _fq2_radius(kind):
    """Radius (pt) of the circle that holds a marker with its edge: half the size for a
    circle, half the diagonal for a square, plus half the edge width."""
    st = FQ2_STYLE[kind]
    mew = FQ2_MEW_OTHER if kind == 'other' else (FQ2_MEW_OPEN if st['open'] else FQ2_MEW_FILLED)
    return st['ms'] / 2 * (np.sqrt(2.0) if st['marker'] == 's' else 1.0) + mew / 2


def _fq2_check_marks(fig, ax, drawn):
    """Independent of the placement, in display points: no two markers overlap (the circles
    that hold them), no marker touches an interval other than its own, no two intervals touch,
    and every mark lies inside the axes. drawn: (column, kind, x, mean, lo, hi). Returns the
    number of pairs checked."""
    k = 72.0 / fig.dpi
    y0, y1 = ax.get_ylim()
    circles, rects = [], []
    for col, kind, x, m, lo, hi in drawn:
        if not (y0 < min(m, lo) and max(m, hi) < y1):
            _fq2_refuse(f'the mark of {kind} in column {col} at {m:+.2f} leaves the y axis')
        px, py = (v * k for v in ax.transData.transform((x, m)))
        circles.append(((col, kind), px, py, _fq2_radius(kind)))
        if kind != 'other':
            plo = ax.transData.transform((x, lo))[1] * k
            phi = ax.transData.transform((x, hi))[1] * k
            rects.append(((col, kind), (px - FQ2_CI_LW / 2, plo, px + FQ2_CI_LW / 2, phi)))
    n = 0
    for i, (ida, xa, ya, ra) in enumerate(circles):
        for idb, xb, yb, rb in circles[i + 1:]:
            n += 1
            if np.hypot(xa - xb, ya - yb) < ra + rb:
                _fq2_refuse(f'the markers of {ida[1]} and {idb[1]} in column {ida[0]} overlap')
        for idb, (x0, b0, x1, b1) in rects:
            if idb == ida:
                continue                     # a marker on its own interval
            n += 1
            dx = max(x0 - xa, 0.0, xa - x1)
            dy = max(b0 - ya, 0.0, ya - b1)
            if np.hypot(dx, dy) < ra:
                _fq2_refuse(f'the marker of {ida[1]} touches the interval of {idb[1]} in column '
                            f'{idb[0]}')
    for i, (ida, a) in enumerate(rects):
        for idb, b in rects[i + 1:]:
            n += 1
            if _fq4_hit(a, b):
                _fq2_refuse(f'the intervals of {ida[1]} and {idb[1]} in column {ida[0]} touch')
    return n


def fq2_reuse(cross_json, cross_macros, out_dir):
    """Draw the figure from the record of p100 after its gates, then print the plotted values
    against the macros."""
    X = load_crosseval(cross_json)
    table = gate_crosseval(X, cross_macros)
    print(f'  cross evaluation: {_shown(cross_json)} (training seeds '
          f'{",".join(map(str, X["seeds"]))}, {X["n"]} test instances, {X["boot"]} bootstrap '
          'replicates)')
    plan = fq2_plan(X)

    # y range: the intervals drawn and every mean, never narrower than the 1% band
    vals = [MARGIN, -MARGIN]
    for P in plan.values():
        for it in [P['own']] + ([P['all']] if P['all'] else []):
            vals += [it['lo'], it['hi']]
        vals += [it['mean'] for _, it in P['other']]
    lo, hi = min(vals), max(vals)
    pad = 0.05 * (hi - lo)
    ylim = (lo - pad, hi + pad)
    yticks = _fq4_ticks(*ylim, nbins=6)

    # Layout in inches (printed size), bottom to top.
    fig_h = 2.70
    left, right, top = 0.66, 0.06, 0.08
    leg_y, leg_h = 0.04, 0.25
    ax_y0 = leg_y + leg_h + 0.50
    ax_h = fig_h - ax_y0 - top
    aw = FIG_W - left - right
    fig = plt.figure(figsize=(FIG_W, fig_h), dpi=300)
    ax = fig.add_axes([left / FIG_W, ax_y0 / fig_h, aw / FIG_W, ax_h / fig_h])
    ax.set_xlim(-0.5, len(COMBOS) - 0.5)
    ax.set_ylim(*ylim)
    ax.axhspan(-MARGIN, MARGIN, color=BAND_FILL, lw=0, zorder=0)
    for t in yticks:
        if abs(t) > 1e-9:
            ax.axhline(t, color=C_GRID, lw=0.4, zorder=0.5)
    ax.axhline(0, color=C_ZERO, lw=0.6, zorder=1)

    pitch = aw * 72.0 / len(COMBOS)               # pt per column
    to_x = 1.0 / pitch                            # data units per pt
    k = 72.0 / fig.dpi
    size = 2 * _fq2_radius('other')
    x_min = FQ2_DX['all'] + _fq2_radius('all') + FQ2_CLEAR + size / 2
    x_max = pitch + FQ2_DX['own'] - _fq2_radius('own') - FQ2_CLEAR - size / 2
    drawn = []
    for j, c in enumerate(COMBOS):
        P = plan[c]
        for kind in ('own', 'all'):
            it = P[kind]
            if it is not None:
                _fq2_mark(ax, kind, j + FQ2_DX[kind] * to_x, it, True)
                drawn.append((c, kind, j + FQ2_DX[kind] * to_x, it['mean'], it['lo'], it['hi']))
        ys = [ax.transData.transform((0, it['mean']))[1] * k for _, it in P['other']]
        xs = _fq2_swarm(ys, x_min, x_max, size)
        for x, (_, it) in zip(xs, P['other']):
            _fq2_mark(ax, 'other', j + x * to_x, it, False)
            drawn.append((c, 'other', j + x * to_x, it['mean'], it['mean'], it['mean']))

    ax.set_xticks(range(len(COMBOS)))
    ax.set_xticklabels([comb_tex(c) for c in COMBOS])
    ax.yaxis.set_major_locator(FixedLocator(yticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _fmt(round(v, 6))))
    ax.tick_params(axis='x', length=0, pad=4)
    ax.tick_params(axis='y', length=3, pad=2)
    ax.set_xlabel('rule combination of the test instances', labelpad=5)
    ax.set_ylabel('makespan difference,\nspecialist minus policy (%)', multialignment='center',
                  labelpad=5)

    # One legend below the axes in a thin box; the two kinds with an interval show a vertical
    # line through the marker, as the plot draws them.
    from matplotlib.collections import LineCollection
    from matplotlib.container import ErrorbarContainer
    from matplotlib.legend_handler import HandlerErrorbar
    hs, labels = [], []
    for kind in ('own', 'all', 'other'):
        st = FQ2_STYLE[kind]
        mk = Line2D([], [], ls='none', marker=st['marker'], ms=st['ms'], **_fq2_face(kind))
        if kind != 'other':
            bar = LineCollection([], colors=[st['color']], linewidths=[FQ2_CI_LW])
            mk = ErrorbarContainer((mk, (), (bar,)), has_xerr=False, has_yerr=True)
        hs.append(mk)
        labels.append(st['text'])
    hs.append(Patch(facecolor=BAND_FILL, edgecolor='none'))
    labels.append('±1% margin')
    leg = fig.legend(hs, labels, loc='lower center',
                     bbox_to_anchor=((left + aw / 2) / FIG_W, leg_y / fig_h), ncol=len(hs),
                     fontsize=FS_TICK, frameon=True, fancybox=False, edgecolor='0.6',
                     framealpha=1.0, handlelength=1.8, handletextpad=0.5, columnspacing=1.4,
                     borderpad=0.4,
                     handler_map={ErrorbarContainer: HandlerErrorbar(xerr_size=0.65)})
    leg.get_frame().set_linewidth(0.5)

    n_text = _check_layout(fig, (ax,), leg, tag='fq2_reuse')
    n_pairs = _fq2_check_marks(fig, ax, drawn)
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, 'fq2_reuse')
    fig.savefig(stem + '.pdf')
    fig.savefig(stem + '.png', dpi=300)
    plt.close(fig)
    print(f'  fq2_reuse: {FIG_W:.2f} x {fig_h:.2f} in, y {ylim[0]:+.2f} to {ylim[1]:+.2f} %, '
          f'{len(drawn)} marks, {n_pairs} pairs of marks clear of each other, {n_text} texts '
          'checked, no overlap')

    # ---- the plotted values against the macros
    print('\n  column  specialist of the combination    specialist of all three rules    '
          'other specialists (mean)')
    for c, P in plan.items():
        own, al = P['own'], P['all']
        a = '(the own mark)' if al is None else \
            f'{al["mean"]:+.2f} [{al["lo"]:+.2f}, {al["hi"]:+.2f}]'
        print(f'  {c:6s}  {own["mean"]:+.2f} [{own["lo"]:+.2f}, {own["hi"]:+.2f}]'
              f'{"":13s}{a:33s}' + ', '.join(f'{r} {it["mean"]:+.2f}' for r, it in P['other']))
    n = {s: sum(row[-1] == s for row in table) for s in ('match', 'no macro, record only')}
    print(f'\n  macro gate: {n["match"]} of {len(table)} values match '
          f'(\\{FQ2_PREFIX}<R>On<C> with Lo and Hi for the two intervals of a column, the mean '
          'for the other specialists, and the counts)'
          + (f'; {n["no macro, record only"]} values without a macro, compared with the record '
             'only: ' + ', '.join(f'{row[1]} {row[2]}' for row in table
                                  if row[-1] == 'no macro, record only')
             if n['no macro, record only'] else '; every plotted value has its macro'))
    return [stem + '.pdf', stem + '.png']


# ------------------------------------------------------------------ fq2_heldout
FQH_COLUMNS = ('LS', 'LW', 'SW', 'LSW')           # the combinations of two or three rules
FQH_SPLITS_JSON = 'results/heldout_splits.json'    # p122, read through arm_registry.out
FQH_SPLITS_MACROS = 'manuscript/macros_heldout.tex'
FQH_SPLITS_GENERATOR = 'scripts/p122_heldout_splits.py'
FQH_SPLITS_SIGN = '100 (policy - specialist) / specialist'
FQH_SPLITS_PREFIX = 'Hsp'
FQH_SPLIT_WORD = {'b': 'B', 'c': 'C'}              # macro words of policies b and c in p122
FQH_SEC_JSON = 'results/secondary_inference.json'  # p95 writes at the repository root
FQH_SEC_MACROS = 'manuscript/macros_secondary.tex'
FQH_SEC_GENERATOR = 'scripts/p95_secondary_inference.py'
FQH_SEC_SIGN = 'd = 100 (a - b) / b, a the first-named method'
FQH_ALL8 = ('all8', 'xmask-all8', 'AllEight')      # block, run stem and macro word in p95
FQH_SEC_FOLDER = '10x25'
FQH_CONTROL_COLOR = dict((a, c) for a, _, c in FS_CONV_B)['control']   # the rose of the control
# The models in the order of every column: key, legend text (policies b and c: built from the
# held-out pair of their record), marker and size in pt. Shape tells the models apart; the four
# policies share the house blue and the single-rule control keeps its rose.
FQH_MODELS = (('policy', 'policy', 'o', MS_CIRCLE),
              ('b', None, '^', 5.6),
              ('c', None, 'v', 5.6),
              ('all8', 'policy trained on all eight', 'D', 3.9),
              ('control', 'single-rule control', 's', MS_SQUARE))
FQH_FILL_TEXT = ('trained on the combination', 'held out of training')
FQH_FILL_COLOR = '0.25'                            # the two fill entries: neutral dark grey
FQH_MEW_OPEN, FQH_MEW_FILLED = 1.0, 0.5
FQH_CI_LW = FQ2_CI_LW
FQH_STEP = 9.5                                     # pt between neighbouring models in a column
FQH_MACRO_KEYS = ('', 'Lo', 'Hi', 'VOne')


def _fqh_refuse(msg):
    raise SystemExit(f'fq2_heldout REFUSING: {msg}')


def _fqh_where(path):
    """A path as the messages print it: relative to the repository when inside it."""
    return _shown(path) if path else '(no path given)'


def _fqh_cell(src, where):
    """mean, lo, hi, verdict at the 1% margin and direction of one comparison of p122 or p95,
    checked: numbers that are finite, the mean inside its interval, and the verdict and the
    direction those that stats_family.verdict and stats_family.direction give the interval."""
    import stats_family as SF
    if not isinstance(src, dict):
        _fqh_refuse(f'{where} is missing')
    lack = [k for k in ('mean', 'lo', 'hi', 'v1.0', 'direction') if k not in src]
    if lack:
        _fqh_refuse(f'{where} has no field {lack}')
    try:
        mean, lo, hi = (float(src[k]) for k in ('mean', 'lo', 'hi'))
    except (TypeError, ValueError):
        _fqh_refuse(f'{where}: a value that is not a number')
    if not all(np.isfinite(v) for v in (mean, lo, hi)):
        _fqh_refuse(f'{where}: a value that is not finite')
    if not lo <= mean <= hi:
        _fqh_refuse(f'{where}: the mean {mean:+.4f} lies outside [{lo:+.4f}, {hi:+.4f}]')
    if src['v1.0'] != SF.verdict(lo, hi, MARGIN):
        _fqh_refuse(f'{where}: verdict {src["v1.0"]!r} at the 1% margin, the interval '
                    f'[{lo:+.4f}, {hi:+.4f}] gives {SF.verdict(lo, hi, MARGIN)!r}')
    if src['direction'] != SF.direction(lo, hi):
        _fqh_refuse(f'{where}: direction {src["direction"]!r}, the interval gives '
                    f'{SF.direction(lo, hi)!r}')
    return dict(mean=mean, lo=lo, hi=hi, verdict=src['v1.0'], equiv=src['v1.0'] == 'equivalent',
                key=where)


def fqh_policy_control(factory_json, factory_macros, factory_seeds='auto', seedext_json=None,
                       seedext_macros=None, allow_partial=False):
    """The policy and the single-rule control, read and gated exactly as panel (a) of fq1 reads
    them (fq1_inputs): the choice of fq1_seed_mode, load_factory, and in mode ten load_seedext
    with gate_seedext for the policy rows and gate_factory(policy=False) for the control rows;
    in mode five gate_factory. Returns dict(models, problems, table, dev, mode, sources); a
    problem is (kind, text), kind absent (the model is left out)."""
    out = dict(models=[], problems=[], table=[], dev=False, mode=None, sources={})
    if not factory_json or not os.path.exists(factory_json):
        out['problems'] = [('absent', f'{t}: the five-seed record {_fqh_where(factory_json)} '
                                      'not found (written by p88_family_inference.py)')
                           for t in ('policy', 'single-rule control')]
        return out
    mode = fq1_seed_mode(factory_seeds, seedext_json)
    F = load_factory(factory_json)
    five_seeds = list(F['seeds'] or [])
    if mode == 'ten':
        T = load_seedext(seedext_json, F)
        if T['dry']:
            if not allow_partial:
                _fqh_refuse(f'{_fqh_where(seedext_json)} is the record of a dry run (synthetic '
                            'arrays); it is drawn only with --allow-partial')
            out['dev'] = True
        F = dict(F, rows=T['rows'], pool=T['pool'], seeds=T['seeds'], setting=T['setting'])
        table = gate_seedext(F, seedext_macros) + gate_factory(F, factory_macros, policy=False)
        out['sources']['policy'] = (f'{_shown(seedext_json)}, ten[R] ({_n_seeds(T["seeds"])} '
                                    f'training seeds' + (', dry run' if out['dev'] else '') + ')')
    else:
        table = gate_factory(F, factory_macros)
        out['sources']['policy'] = (f'{_shown(factory_json)}, parity.components[R] '
                                    f'({_n_seeds(F["seeds"])} training seeds)')
    out['mode'] = mode
    gated = {lab: (got, status) for fam, lab, _, got, status in table}
    out['models'].append(dict(key='policy', train=tuple(REG.TRAIN_COMBOS),
                              rows={c: F['rows'][c] for c in FQH_COLUMNS},
                              seeds=list(F['seeds'] or [])))
    for c in FQH_COLUMNS:
        got, status = gated.get(c, (None, 'no macro'))
        out['table'].append(('policy', c, F['rows'][c], got, status))
    lack = [c for c in FQH_COLUMNS if F['ctl'].get(c) is None]
    if lack:
        out['problems'].append(('absent', f'single-rule control: {_shown(factory_json)} has no '
                                f'control.components.Spec.{",".join(lack)}'))
    else:
        out['models'].append(dict(key='control', train=tuple(REG.CONTROL_COMBOS),
                                  rows={c: F['ctl'][c] for c in FQH_COLUMNS}, seeds=five_seeds))
        for c in FQH_COLUMNS:
            got, status = gated.get(f'{c} control', (None, 'no macro'))
            out['table'].append(('control', c, F['ctl'][c], got, status))
        out['sources']['control'] = (f'{_shown(factory_json)}, control.components.Spec[R] '
                                     f'({_n_seeds(five_seeds)} training seeds)')
    return out


def load_heldout_splits(path, allow_partial=False):
    """Policies b and c from the record of p122 (quantity (a): each against the specialist on
    every combination), checked: generator, sign, the training and held-out lists of
    xsplit_runs, every plotted cell (_fqh_cell). A record of a dry run is refused unless
    allow_partial. Returns (models, problems, dev); a missing record is a problem of kind absent,
    a record of other seeds than xsplit_runs.SEEDS one of kind incomplete (drawn in the partial
    mode only)."""
    import xsplit_runs as XS
    where = _fqh_where(path)
    if not path or not os.path.exists(path):
        return [], [('absent', f'policies b and c: {where} not found (written by '
                               f'{FQH_SPLITS_GENERATOR})')], False
    with open(path) as f:
        J = json.load(f)
    lack = [k for k in ('generator', 'seeds', 'dry', 'definitions', 'split') if k not in J]
    if lack:
        _fqh_refuse(f'{where} has no key {lack}')
    if J['generator'] != FQH_SPLITS_GENERATOR:
        _fqh_refuse(f'{where} was written by {J["generator"]!r}, not {FQH_SPLITS_GENERATOR}')
    sign = str((J['definitions'] or {}).get('against_specialist', ''))
    if not sign.startswith(FQH_SPLITS_SIGN):
        _fqh_refuse(f'{where} defines the difference as {sign!r}, not {FQH_SPLITS_SIGN} ...')
    dev = bool(J['dry'])
    if dev and not allow_partial:
        _fqh_refuse(f'{where} is the record of a dry run of p122 (synthetic arrays); it is drawn '
                    'only with --allow-partial')
    seeds = [int(s) for s in J['seeds']]
    problems = []
    if seeds != list(XS.SEEDS):
        problems.append(('incomplete', f'policies b and c: {_shown(path)} holds the training seeds '
                                       f'{",".join(map(str, seeds))}, the runs are trained for '
                                       f'{",".join(map(str, XS.SEEDS))}'))
    models = []
    for x in FQH_SPLIT_WORD:
        S = (J['split'] or {}).get(x)
        if not isinstance(S, dict) or any(k not in S for k in ('train', 'held',
                                                                'against_specialist')):
            _fqh_refuse(f'{where} has no split.{x} with train, held and against_specialist')
        if list(S['train']) != list(XS.SPLITS[x]) or list(S['held']) != list(XS.HELD[x]):
            _fqh_refuse(f'{where}: split.{x} trains on {S["train"]} and holds out {S["held"]}, '
                        f'xsplit_runs trains it on {list(XS.SPLITS[x])} and holds out '
                        f'{list(XS.HELD[x])}')
        rows = {c: _fqh_cell(S['against_specialist'].get(c),
                             f'{_shown(path)}: split.{x}.against_specialist.{c}')
                for c in FQH_COLUMNS}
        models.append(dict(key=x, text=_fqh_held_text(S['held']),
                           legend=_fqh_held_text(S['held'], comb_tex),
                           train=tuple(S['train']), held=tuple(S['held']), rows=rows,
                           seeds=seeds))
    return models, problems, dev


def _fqh_held_text(held, form=str):
    """The name of a policy trained without the combinations `held`: 'policy, LS and LSW held
    out'. form sets each combination name (comb_tex for the legend, the plain name for the
    printed lines)."""
    return f'policy, {" and ".join(form(c) for c in held)} held out'


def load_all_eight(path):
    """The policy trained on all eight combinations from the record of p95: block all8, its
    rows against the specialist of each combination (vs_specialist.components) and its training
    combinations (effort.combos, read from the configurations of its runs), checked: generator,
    environment, check mode, sign, the run stem and folder, every plotted cell (_fqh_cell).
    Returns (models, problems); a missing record or a block that p95 did not compute is a
    problem of kind absent, a block of other seeds than arm_registry.SECONDARY_SEEDS one of
    kind incomplete."""
    text = 'policy trained on all eight'
    where = _fqh_where(path)
    if not path or not os.path.exists(path):
        return [], [('absent', f'{text}: {where} not found (written by {FQH_SEC_GENERATOR})')]
    with open(path) as f:
        J = json.load(f)
    lack = [k for k in ('generator', 'environment', 'check', 'sign', 'blocks') if k not in J]
    if lack:
        _fqh_refuse(f'{where} has no key {lack}')
    if J['generator'] != FQH_SEC_GENERATOR:
        _fqh_refuse(f'{where} was written by {J["generator"]!r}, not {FQH_SEC_GENERATOR}')
    if J['check'] or J['environment'] != 'exact':
        _fqh_refuse(f'{where} is the check on the {J["environment"]} environment of p95 '
                    '(--earlier-check), not the exact-environment record')
    if not str(J['sign']).startswith(FQH_SEC_SIGN):
        _fqh_refuse(f'{where} states the sign {J["sign"]!r}, not {FQH_SEC_SIGN} ...')
    block, stem, _ = FQH_ALL8
    B = (J['blocks'] or {}).get(block)
    if not isinstance(B, dict) or 'status' not in B:
        _fqh_refuse(f'{where} has no block {block}')
    if B['status'] != 'computed':
        why = B.get('reason') or '; '.join((B.get('problems') or [])[:2]) or '-'
        return [], [('absent', f'{text}: block {block} of {_shown(path)} is {B["status"]} '
                               f'({why})')]
    R = B.get('result') or {}
    for k in ('stem', 'folder', 'seeds', 'effort', 'vs_specialist'):
        if k not in R:
            _fqh_refuse(f'{where}: blocks.{block}.result has no key {k!r}')
    if R['stem'] != stem or R['folder'] != FQH_SEC_FOLDER:
        _fqh_refuse(f'{where}: blocks.{block} reads {R["stem"]} at {R["folder"]}, not {stem} at '
                    f'{FQH_SEC_FOLDER}')
    train = [str(r) for r in (R['effort'] or {}).get('combos') or []]
    if not train or any(r not in COMBOS for r in train) or not R['effort'].get('agree', True):
        _fqh_refuse(f'{where}: blocks.{block}.result.effort.combos is {train}, not a list of '
                    'combinations that every run records')
    seeds = [int(s) for s in R['seeds']]
    problems = []
    if seeds != list(REG.SECONDARY_SEEDS):
        problems.append(('incomplete', f'{text}: {_shown(path)} holds the training seeds '
                                       f'{",".join(map(str, seeds))}, the runs are trained for '
                                       f'{",".join(map(str, REG.SECONDARY_SEEDS))}'))
    comps = (R['vs_specialist'] or {}).get('components') or {}
    rows = {c: _fqh_cell(comps.get(c), f'{_shown(path)}: blocks.{block}.result.vs_specialist'
                                       f'.components.{c}') for c in FQH_COLUMNS}
    return [dict(key='all8', train=tuple(train), rows=rows, seeds=seeds)], problems


def _fqh_macro_check(mac, stem, it):
    """(macros found, disagreements) of one plotted comparison against <stem>, <stem>Lo,
    <stem>Hi (signed two decimals, within half a unit of the last decimal) and <stem>VOne (the
    verdict word of stats_family.TEX)."""
    import stats_family as SF
    got = {k: mac.get(stem + k) for k in FQH_MACRO_KEYS}
    bad = [f'\\{stem}{k}: no macro' for k, v in got.items() if v is None]
    for k, v in (('', it['mean']), ('Lo', it['lo']), ('Hi', it['hi'])):
        if got[k] is not None and (re.fullmatch(r'[+-]\d+\.\d\d', got[k]) is None
                                   or abs(v - float(got[k])) > 0.005 + 1e-9):
            bad.append(f'\\{stem}{k} = {got[k]}, record {v:+.4f}')
    if got['VOne'] is not None and got['VOne'] != SF.TEX[it['verdict']]:
        bad.append(f'\\{stem}VOne = {got["VOne"]}, record {it["verdict"]}')
    return got, bad


def gate_heldout_splits(models, path):
    """Policies b and c against the macro file of p122: HspSeeds equal to the record's seeds,
    and every plotted value with a macro (Hsp<B|C><R>, Lo, Hi, VOne) equal to it; p122 prints
    every combination of a split, a held-out cell without its macros is refused, and a cell of a
    training combination without a macro is compared with the record only. Returns
    the table rows (model, column, item, macros, status)."""
    where = _fqh_where(path)
    if not path or not os.path.exists(path):
        _fqh_refuse(f'the macro file {where} not found (written by {FQH_SPLITS_GENERATOR})')
    mac = _macros(path)
    bad, rows = [], []
    k = str(len(models[0]['seeds']))
    if mac.get(f'{FQH_SPLITS_PREFIX}Seeds') != k:
        bad.append(f'\\{FQH_SPLITS_PREFIX}Seeds = {mac.get(FQH_SPLITS_PREFIX + "Seeds")}, '
                   f'record {k}')
    for M in models:
        w = FQH_SPLITS_PREFIX + FQH_SPLIT_WORD[M['key']]
        for c in FQH_COLUMNS:
            it, stem = M['rows'][c], f'{w}{c}'
            if c not in M['held'] and all(mac.get(stem + x) is None for x in FQH_MACRO_KEYS):
                rows.append((M['key'], c, it, None, 'no macro, record only'))
                continue
            got, err = _fqh_macro_check(mac, stem, it)
            bad += err
            rows.append((M['key'], c, it, got, 'match' if not err else 'MISMATCH'))
    if bad:
        _fqh_refuse(f'{where} disagrees with the record:\n  ' + '\n  '.join(bad))
    return rows


def gate_all_eight(model, path):
    """The policy trained on all eight against the macro file of p95: SecSeedN equal to the
    record's seeds and every plotted value equal to SecAllEightPar<R> with Lo, Hi and VOne (p95
    prints them for every combination it evaluates). Returns the table rows."""
    where = _fqh_where(path)
    if not path or not os.path.exists(path):
        _fqh_refuse(f'the macro file {where} not found (written by {FQH_SEC_GENERATOR})')
    mac = _macros(path)
    bad, rows = [], []
    if mac.get('SecSeedN') != str(len(model['seeds'])):
        bad.append(f'\\SecSeedN = {mac.get("SecSeedN")}, record {len(model["seeds"])}')
    for c in FQH_COLUMNS:
        it = model['rows'][c]
        got, err = _fqh_macro_check(mac, f'Sec{FQH_ALL8[2]}Par{c}', it)
        bad += err
        rows.append(('all8', c, it, got, 'match' if not err else 'MISMATCH'))
    if bad:
        _fqh_refuse(f'{where} disagrees with the record:\n  ' + '\n  '.join(bad))
    return rows


def fq2_heldout_inputs(factory_json, factory_macros, factory_seeds, seedext_json,
                       seedext_macros, splits_json, splits_macros, sec_json, sec_macros,
                       partial=False, allow_partial=False):
    """The models fq2_heldout draws, in the order of FQH_MODELS, after every gate. In the full
    mode (partial False) a model that is absent or incomplete refuses the figure with the list
    of every such problem; in the partial mode the models whose records exist are returned and
    the problems are listed. Returns dict(models, problems, table, dev, partial, lines)."""
    PC = fqh_policy_control(factory_json, factory_macros, factory_seeds, seedext_json,
                            seedext_macros, allow_partial)
    splits, prob_s, dev_s = load_heldout_splits(splits_json, allow_partial)
    all8, prob_a = load_all_eight(sec_json)
    problems = PC['problems'] + prob_s + prob_a
    if problems and not partial:
        _fqh_refuse('the full figure needs every model with every training seed; missing or '
                    'incomplete:\n  ' + '\n  '.join(t for _, t in problems)
                    + '\n(--heldout-partial draws the models whose records exist)')
    table = list(PC['table'])
    if splits:
        table += gate_heldout_splits(splits, splits_macros)
    if all8:
        table += gate_all_eight(all8[0], sec_macros)
    have = {M['key']: M for M in PC['models'] + splits + all8}
    if not have:
        _fqh_refuse('no model has a record:\n  ' + '\n  '.join(t for _, t in problems))
    models = []
    for key, text, marker, ms in FQH_MODELS:
        if key in have:
            M = dict(have[key])
            M.update(text=M.get('text') or text, marker=marker, ms=ms,
                     color=FQH_CONTROL_COLOR if key == 'control' else C_EQUIV)
            M.setdefault('legend', M['text'])      # policies b and c bring their own legend
            models.append(M)
    src = dict(PC['sources'])
    if splits:
        src['b'] = src['c'] = (f'{_shown(splits_json)}, split.<b|c>.against_specialist[R] '
                               f'({_n_seeds(splits[0]["seeds"])} training seeds'
                               + (', dry run' if dev_s else '') + ')')
    if all8:
        src['all8'] = (f'{_shown(sec_json)}, blocks.all8.result.vs_specialist.components[R] '
                       f'({_n_seeds(all8[0]["seeds"])} training seeds)')
    lines = [f'  fq2_heldout records (--factory-seeds {factory_seeds}'
             + (f': {PC["mode"]}' if PC['mode'] else '') + '):']
    lines += [f'    {M["text"]}: {src[M["key"]]}; trained on {",".join(M["train"])}'
              for M in models]
    for kind, t in problems:
        lines.append(f'    {"left out" if kind == "absent" else "drawn, but"}: {t}')
    return dict(models=models, problems=problems, table=table, dev=PC['dev'] or dev_s,
                partial=partial, lines=lines)


def _fqh_face(M, filled):
    """Filled: the model's colour with a thin white edge; open: white with a coloured edge."""
    if filled:
        return dict(mfc=M['color'], mec='white', mew=FQH_MEW_FILLED)
    return dict(mfc='white', mec=M['color'], mew=FQH_MEW_OPEN)


def _fqh_radius(marker, ms, mew):
    """Radius (pt) of the circle that holds a marker with its edge: half the size for a circle,
    half the diagonal of its box for the other shapes (square, triangle, diamond)."""
    return ms / 2 * (1.0 if marker == 'o' else np.sqrt(2.0)) + mew / 2


def _fqh_check_marks(fig, ax, drawn):
    """Independent of the placement, in display points: no two markers overlap (the circles
    that hold them), no marker touches an interval other than its own, no two intervals touch,
    and every mark lies inside the axes. drawn: (column, model, x, mean, lo, hi, radius).
    Returns the number of pairs checked."""
    k = 72.0 / fig.dpi
    y0, y1 = ax.get_ylim()
    x0, x1 = ax.get_xlim()
    circles, rects = [], []
    for col, key, x, m, lo, hi, r in drawn:
        if not (y0 < lo and hi < y1 and x0 < x < x1):
            _fqh_refuse(f'the mark of {key} in column {col} leaves the axes')
        px, py = (v * k for v in ax.transData.transform((x, m)))
        circles.append(((col, key), px, py, r))
        plo = ax.transData.transform((x, lo))[1] * k
        phi = ax.transData.transform((x, hi))[1] * k
        rects.append(((col, key), (px - FQH_CI_LW / 2, plo, px + FQH_CI_LW / 2, phi)))
    n = 0
    for i, (ida, xa, ya, ra) in enumerate(circles):
        for idb, xb, yb, rb in circles[i + 1:]:
            n += 1
            if np.hypot(xa - xb, ya - yb) < ra + rb:
                _fqh_refuse(f'the markers of {ida[1]} and {idb[1]} in column {ida[0]} overlap')
        for idb, (bx0, b0, bx1, b1) in rects:
            if idb == ida:
                continue
            n += 1
            dx = max(bx0 - xa, 0.0, xa - bx1)
            dy = max(b0 - ya, 0.0, ya - b1)
            if np.hypot(dx, dy) < ra:
                _fqh_refuse(f'the marker of {ida[1]} touches the interval of {idb[1]} in column '
                            f'{idb[0]}')
    for i, (ida, a) in enumerate(rects):
        for idb, b in rects[i + 1:]:
            n += 1
            if _fq4_hit(a, b):
                _fqh_refuse(f'the intervals of {ida[1]} and {idb[1]} in column {ida[0]} touch')
    return n


def fq2_heldout(factory_json, factory_macros, factory_seeds, seedext_json, seedext_macros,
                splits_json, splits_macros, sec_json, sec_macros, out_dir, partial=False,
                allow_partial=False):
    """Draw the figure after its gates (fq2_heldout_inputs), then print the plotted values
    against the macros. Full mode: fq2_heldout.pdf/.png; partial mode: fq2_heldout_partial.*;
    a dry-run record drawn with allow_partial adds _dev."""
    X = fq2_heldout_inputs(factory_json, factory_macros, factory_seeds, seedext_json,
                           seedext_macros, splits_json, splits_macros, sec_json, sec_macros,
                           partial, allow_partial)
    print('\n'.join(X['lines']))
    models = X['models']

    # y range: every interval drawn, never narrower than the 1% band plus half a percent (the
    # rule of fq1), so the edges of the band stay visible
    vals = [v for M in models for it in M['rows'].values() for v in (it['lo'], it['hi'])]
    lo, hi = min(vals + [-MARGIN - 0.5]), max(vals + [MARGIN + 0.5])
    pad = 0.05 * (hi - lo)
    ylim = (lo - pad, hi + pad)
    yticks = _fq4_ticks(*ylim, nbins=8)

    # Layout in inches (printed size): the legend right of the axes, centred on them; its
    # width, measured once drawn, fixes the width of the axes.
    fig_h = 2.80
    left, right, gap, top, ax_y0 = 0.66, 0.04, 0.20, 0.08, 0.50
    ax_h = fig_h - ax_y0 - top
    fig = plt.figure(figsize=(FIG_W, fig_h), dpi=300)
    # Every model has its interval, so its legend entry is the marker alone (filled); the fill
    # has its own two entries.
    hs, labels = [], []
    for M in models:
        hs.append(Line2D([], [], ls='none', marker=M['marker'], ms=M['ms'],
                         **_fqh_face(M, True)))
        labels.append(M['legend'])
    for text, filled in zip(FQH_FILL_TEXT, (True, False)):
        face = dict(mfc=FQH_FILL_COLOR, mec='white', mew=FQH_MEW_FILLED) if filled else \
            dict(mfc='white', mec=FQH_FILL_COLOR, mew=FQH_MEW_OPEN)
        hs.append(Line2D([], [], ls='none', marker='o', ms=MS_CIRCLE, **face))
        labels.append(text)
    hs.append(Patch(facecolor=BAND_FILL, edgecolor='none'))
    labels.append('±1% margin')
    leg = fig.legend(hs, labels, loc='center right',
                     bbox_to_anchor=((FIG_W - right) / FIG_W, (ax_y0 + ax_h / 2) / fig_h),
                     ncol=1, fontsize=FS_TICK, frameon=True, fancybox=False, edgecolor='0.6',
                     framealpha=1.0, handlelength=1.8, handletextpad=0.6, labelspacing=0.6,
                     borderpad=0.5, borderaxespad=0.0)
    leg.get_frame().set_linewidth(0.5)
    fig.canvas.draw()
    leg_w = leg.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
    aw = FIG_W - left - gap - leg_w - right
    ax = fig.add_axes([left / FIG_W, ax_y0 / fig_h, aw / FIG_W, ax_h / fig_h])
    ax.set_xlim(-0.5, len(FQH_COLUMNS) - 0.5)
    ax.set_ylim(*ylim)
    ax.axhspan(-MARGIN, MARGIN, color=BAND_FILL, lw=0, zorder=0)
    for t in yticks:
        if abs(t) > 1e-9:
            ax.axhline(t, color=C_GRID, lw=0.4, zorder=0.5)
    ax.axhline(0, color=C_ZERO, lw=0.6, zorder=1)

    pitch = aw * 72.0 / len(FQH_COLUMNS)          # pt per column
    span = FQH_STEP * (len(models) - 1)
    if span > 0.75 * pitch:
        _fqh_refuse(f'the {len(models)} models of a column span {span:.0f} pt of a column of '
                    f'{pitch:.0f} pt')
    drawn = []
    for j, c in enumerate(FQH_COLUMNS):
        for i, M in enumerate(models):
            it = M['rows'][c]
            filled = c in M['train']
            x = j + (i - (len(models) - 1) / 2) * FQH_STEP / pitch
            ax.plot([x, x], [it['lo'], it['hi']], color=M['color'], lw=FQH_CI_LW,
                    solid_capstyle='butt', zorder=4)
            ax.plot([x], [it['mean']], M['marker'], ms=M['ms'], ls='none', zorder=5,
                    **_fqh_face(M, filled))
            mew = FQH_MEW_FILLED if filled else FQH_MEW_OPEN
            drawn.append((c, M['key'], x, it['mean'], it['lo'], it['hi'],
                          _fqh_radius(M['marker'], M['ms'], mew)))

    ax.set_xticks(range(len(FQH_COLUMNS)))
    ax.set_xticklabels([comb_tex(c) for c in FQH_COLUMNS])
    ax.yaxis.set_major_locator(FixedLocator(yticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _fmt(round(v, 6))))
    ax.tick_params(axis='x', length=0, pad=4)
    ax.tick_params(axis='y', length=3, pad=2)
    ax.set_xlabel('rule combination of the test instances', labelpad=5)
    ax.set_ylabel('difference from the specialist (%)', labelpad=5)

    n_text = _check_layout(fig, (ax,), leg, tag='fq2_heldout')
    n_pairs = _fqh_check_marks(fig, ax, drawn)
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, 'fq2_heldout' + ('_partial' if partial else '')
                        + ('_dev' if X['dev'] else ''))
    fig.savefig(stem + '.pdf')
    fig.savefig(stem + '.png', dpi=300)
    plt.close(fig)
    print(f'  fq2_heldout{" (partial)" if partial else ""}: {FIG_W:.2f} x {fig_h:.2f} in, axes '
          f'{aw:.2f} in wide, y {ylim[0]:+.2f} to {ylim[1]:+.2f} %, {len(models)} models, '
          f'{len(drawn)} marks, {n_pairs} pairs of marks clear of each other, {n_text} texts '
          'checked, no overlap')

    # ---- the plotted values against the macros
    text = {M['key']: M['text'] for M in models}
    train = {M['key']: M['train'] for M in models}
    print('\n  column  model                           training   mean    95% interval      '
          'macro mean [Lo, Hi] VOne              check')
    order = [M['key'] for M in models]
    for c in FQH_COLUMNS:
        rows = sorted((r for r in X['table'] if r[1] == c and r[0] in text),
                      key=lambda r: order.index(r[0]))
        for key, _, it, got, status in rows:
            m = '-' if not got or got.get('') is None else \
                f'{got[""]} [{got["Lo"]}, {got["Hi"]}] {got["VOne"]}'
            print(f'  {c:6s}  {text[key]:30s}  {"trained" if c in train[key] else "held out":9s}  '
                  f'{it["mean"]:+.2f}  [{it["lo"]:+.2f}, {it["hi"]:+.2f}]  {m:34s} {status}')
    shown = [r for r in X['table'] if r[0] in text]
    n = sum(r[-1] == 'match' for r in shown)
    rest = [f'{text[r[0]]} on {r[1]} ({r[-1]})' for r in shown if r[-1] != 'match']
    print(f'\n  macro gate: {n} of {len(shown)} plotted comparisons match their macros'
          + ('; ' + '; '.join(rest) if rest else ''))
    return [stem + '.pdf', stem + '.png']


# ------------------------------------------------------------------ fq5_rule_change
FQ5_JSON = 'results/rule_change_cost.json'          # p99 infer, read through arm_registry.out
FQ5_MACROS = 'manuscript/macros_rulechange_cost.tex'
FQ5_GENERATOR = 'scripts/p99_rule_change_cost.py infer'
FQ5_DEPLOY_JSON = 'results/deploy_cost_cpu_1thread.json'   # p75, read through arm_registry.out
FQ5_DEPLOY_MACROS = 'paper/macros_deploy.tex'
FQ5_DEPLOY_GENERATOR = 'scripts/p75_deploy_cost.py'
FQ5_ENV = 'exact'                                   # the environment of the paper
FQ5_ROWS = ('descriptor checks', 're-planning', 'schedule validation')   # top to bottom
FQ5_LW = FS_CONV_LW
# Panel (a): the specialists in their slate (fq2_reuse, fs_convergence) with squares, their
# range as a pale band of the same slate; the policy in the house blue with larger circles,
# drawn under the specialists' squares, so that where the two lines cross both stay visible.
FQ5_SPEC = dict(color=FS_CONV_SPEC_COLORS[0], marker='s', ms=3.6)
FQ5_POL = dict(color=C_EQUIV, marker='o', ms=5.6)
FQ5_BAND_ALPHA = FS_CONV_BAND_ALPHA
FQ5_LEGEND_A = ('one specialist per combination', 'range over the choice\nof combinations',
                'one policy')
# Panel (b): one marker shape per batch size, in the neutral dark grey of fq2_heldout.
FQ5_SHAPE = {10: ('o', MS_CIRCLE), 50: ('^', 5.6), 80: ('D', 3.9)}
FQ5_COLOR_B = FQH_FILL_COLOR
FQ5_MEW = 0.5
FQ5_NUDGE = 8.0             # pt between the line of a row and a marker moved off it
FQ5_CLEAR = 0.6             # pt of white kept between two markers of a row
FQ5_VISIBLE = 0.3           # least share of a policy marker left visible under a square
FQ5_LEG_PAD = 2.0           # pt between a legend box and the nearest mark
FQ5_HEADROOM = 4            # y steps panel (a) may add above the data for its legend
FQ5_FIG_H = 2.50


def _fq5_refuse(msg):
    raise SystemExit(f'fq5_rule_change REFUSING: {msg}')


def _fq5_scripts():
    """p99_rule_change_cost and p75_deploy_cost, imported for their names and helpers. The import
    of p75 moves to the repository root and empties sys.argv; both are put back."""
    cwd, argv = os.getcwd(), sys.argv
    try:
        import p75_deploy_cost as P75
        import p99_rule_change_cost as P99
    finally:
        os.chdir(cwd)
        sys.argv = argv
    return P99, P75


def _fq5_shop(size):
    """The shop size p75 times for a size of the checks: the campaign's instance sets 50x25n30
    and 80x25n30 come from the shops 50x25 and 80x25 (the reading of fq3)."""
    return size[:-len('n30')] if size.endswith('n30') else size


def _fq5_decade(e):
    """Text of the tick at 10**e seconds, in a unit a reader knows: 1 ms, 10 ms, 0.1 s, 1 s."""
    if e >= 0:
        return f'{10 ** e:g} s'
    if e == -1:
        return '0.1 s'
    return f'{10.0 ** (e + 3):g} ms'


def _fq5_number(v, where):
    try:
        x = float(v)
    except (TypeError, ValueError):
        _fq5_refuse(f'{where} = {v!r} is not a number')
    if not np.isfinite(x) or x <= 0:
        _fq5_refuse(f'{where} = {v!r} is not a positive finite number')
    return x


def load_rule_change(path):
    """The training hours and the check times from the record of p99 infer: generator and
    environment checked; the training hours of the policy (records.released.hours) and of each
    of the eight specialists (records['specialist <R>'].hours), the time of the descriptor checks
    and of the schedule check at the three sizes of p99 (macros RccDescChecksMs<W> and
    RccScheduleCheckMs<W>, in ms, the median over the test instances), and the macros the gate
    holds against the macro file. Refuses a missing record, and a record that lacks any input of
    the figure, with the list of every missing input."""
    P99, _ = _fq5_scripts()
    where = _fqh_where(path)
    if not path or not os.path.exists(path):
        _fq5_refuse(f'{where} not found (written by {FQ5_GENERATOR} under PACT_ENV={FQ5_ENV}; '
                    'the figure reads it through arm_registry.out, so give the setting of that '
                    'run, or --rule-change)')
    with open(path) as f:
        J = json.load(f)
    lack = [k for k in ('generator', 'environment', 'macros', 'records', 'missing') if k not in J]
    if lack:
        _fq5_refuse(f'{where} has no key {lack}')
    if J['generator'] != FQ5_GENERATOR:
        _fq5_refuse(f'{where} was written by {J["generator"]!r}, not {FQ5_GENERATOR}')
    if J['environment'] != FQ5_ENV:
        _fq5_refuse(f'{where} holds the environment {J["environment"]!r}, not {FQ5_ENV}')
    mac, recs = dict(J['macros'] or {}), dict(J['records'] or {})
    missing = []

    def hours(key, what):
        r = recs.get(key)
        if not isinstance(r, dict) or 'hours' not in r:
            missing.append(f'the training record of {what} (records[{key!r}])')
            return None
        return _fq5_number(r['hours'], f'{where}: records[{key!r}].hours')

    pol = hours('released', 'the policy')
    spec = {R: hours(f'specialist {R}', f'the specialist of {R}') for R in COMBOS}
    sizes = P99.DEFAULT_SIZES.split(',')
    need = (['RccPolicyHours'] + [f'RccSpecHours{R}' for R in COMBOS]
            + ['RccSpecMeasuredN', 'RccCatalogueHours'])
    for s in sizes:
        w = P99.size_word(s)
        need += [f'RccDescChecksMs{w}', f'RccScheduleCheckMs{w}']
    gone = [n for n in need if n not in mac]
    if gone:
        missing.append('the macros ' + ', '.join(gone))
    if missing:
        own = '; '.join(J['missing']) if J['missing'] else 'nothing'
        _fq5_refuse(f'the figure needs the policy, all eight specialists and the checks at '
                    f'{", ".join(sizes)}; {where} lacks:\n  ' + '\n  '.join(missing)
                    + f'\n(the record lists as missing: {own})')
    checks = {}
    for s in sizes:
        w = P99.size_word(s)
        checks[s] = tuple(_fq5_number(mac[f'Rcc{stem}Ms{w}'], f'{where}: macros.Rcc{stem}Ms{w}')
                          for stem in ('DescChecks', 'ScheduleCheck'))
    return dict(policy=pol, spec=spec, total=float(sum(spec[R] for R in COMBOS)), checks=checks,
                sizes=sizes, macros=mac, need=need, schedules=J.get('checks_schedules'),
                instances={s: mac.get(f'RccChecksInstances{P99.size_word(s)}') for s in sizes},
                gpu=sorted({str(r.get('gpu')) for r in recs.values() if isinstance(r, dict)}),
                path=path)


def gate_rule_change(C, path):
    """The record of p99 infer against the macro file of the same run: every macro the figure
    reads is the one the file prints, the plotted training hours equal RccPolicyHours,
    RccSpecHours<R> and RccCatalogueHours (the sum, the mean line at eight) at their two printed
    decimals, RccSpecMeasuredN is 8, and the plotted check times are the printed values. A
    missing macro file or any disagreement is refused. Returns the table rows (quantity, value,
    macro, printed, status)."""
    where = _fqh_where(path)
    if not path or not os.path.exists(path):
        _fq5_refuse(f'the macro file {where} not found (written by {FQ5_GENERATOR})')
    mac = _macros(path)
    bad, rows = [], []
    for name in C['need']:
        if mac.get(name) != C['macros'][name]:
            bad.append(f'\\{name} = {mac.get(name)}, the record of the same run '
                       f'{C["macros"][name]}')
    if mac.get('RccSpecMeasuredN') != str(len(COMBOS)):
        bad.append(f'\\RccSpecMeasuredN = {mac.get("RccSpecMeasuredN")}, want {len(COMBOS)}')
    plan = ([('policy, training', 'RccPolicyHours', C['policy'])]
            + [(f'specialist {R}, training', f'RccSpecHours{R}', C['spec'][R]) for R in COMBOS]
            + [('eight specialists, training', 'RccCatalogueHours', C['total'])])
    for label, name, v in plan:
        got = mac.get(name)
        ok = (got is not None and re.fullmatch(r'\d+\.\d\d', got) is not None
              and abs(v - float(got)) <= 0.005 + 1e-9)
        rows.append((label, v, name, got, 'match' if ok else 'MISMATCH'))
        if not ok:
            bad.append(f'\\{name} = {got}, record {v:.4f} h')
    _, P75 = _fq5_scripts()
    for s in C['sizes']:
        w = P75.size_word(int(s.split('x')[0]))
        for k, (label, stem) in enumerate((('descriptor checks', 'DescChecks'),
                                           ('schedule validation', 'ScheduleCheck'))):
            name = f'Rcc{stem}Ms{w}'
            got = mac.get(name)
            ok = got is not None and abs(C['checks'][s][k] - float(got)) == 0.0
            rows.append((f'{label}, {s} (ms)', C['checks'][s][k], name, got,
                         'match' if ok else 'MISMATCH'))
            if not ok:
                bad.append(f'\\{name} = {got}, plotted {C["checks"][s][k]} ms')
    if bad:
        _fq5_refuse(f'{where} disagrees with the record {_fqh_where(C["path"])}:\n  '
                    + '\n  '.join(bad))
    return rows


def load_deploy(path):
    """The re-planning time of one batch from the record of p75_deploy_cost.py, by shop size
    (10x25, 50x25, 80x25): the median over the test instances of the combination with the
    largest median (p75_deploy_cost.worst_regime), the quantity of DeployLatency<Word>. The
    record must be one of the paper's environment, which p75 marks in the record itself: field
    environment exact at the top and in each size entry, the policy of that environment (model
    10x25+family+xmask-v1-s<seed>), one torch thread (torch_num_threads and
    torch_num_interop_threads 1, OMP_NUM_THREADS and MKL_NUM_THREADS 1), measured on idle cores
    (allow_busy false), and valid at the top and at each size, with cpu_share at least
    p75_deploy_cost.SHARE_MIN and all eight combinations timed. A record of an earlier
    environment (no field environment, four threads) is refused with every reason."""
    P99, P75 = _fq5_scripts()
    where = _fqh_where(path)
    if not path or not os.path.exists(path):
        _fq5_refuse(f'{where} not found (written by {FQ5_DEPLOY_GENERATOR} --seed select under '
                    f'PACT_ENV={FQ5_ENV} on one reserved core with one torch thread; the figure '
                    'reads it through arm_registry.out, or --deploy)')
    with open(path) as f:
        D = json.load(f)
    probs = []
    if D.get('environment') != FQ5_ENV:
        probs.append(f'environment {D.get("environment")!r}, not {FQ5_ENV!r} (a record without '
                     'the field comes from an earlier environment)')
    if D.get('device') != 'cpu':
        probs.append(f'device {D.get("device")!r}, not cpu')
    if D.get('valid') is not True:
        probs.append(f'field valid = {D.get("valid")!r}')
    if D.get('allow_busy') is not False:
        probs.append(f'allow_busy = {D.get("allow_busy")!r} (a measurement on busy cores, or a '
                     'record without the field)')
    for k in ('torch_num_threads', 'torch_num_interop_threads'):
        if D.get(k) != 1:
            probs.append(f'{k} = {D.get(k)!r}, not one thread')
    for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        if str(D.get(k)) != '1':
            probs.append(f'{k} = {D.get(k)!r}, not one thread')
    models = sorted(REG.PREFIX + REG.model('released', None, s, environment=FQ5_ENV)
                    for s in REG.PLANNED_SEEDS[FQ5_ENV])
    if D.get('model') not in models:
        probs.append(f'model {D.get("model")!r} is not the policy of the {FQ5_ENV} environment '
                     f'({models[0]} to {models[-1]})')
    times = {}
    for s in P99.DEFAULT_SIZES.split(','):
        shop, mods = _fq5_shop(s), P99.modules(s)
        r = (D.get('sizes') or {}).get(shop)
        if not isinstance(r, dict):
            probs.append(f'no entry sizes[{shop!r}] (re-planning at {mods} modules)')
            continue
        sub = []
        if r.get('valid') is not True:
            sub.append(f'valid = {r.get("valid")!r}')
        if r.get('environment') != FQ5_ENV:
            sub.append(f'environment {r.get("environment")!r}')
        if r.get('torch_num_threads') != 1:
            sub.append(f'torch_num_threads = {r.get("torch_num_threads")!r}')
        share = r.get('cpu_share')
        if not isinstance(share, (int, float)) or not share >= P75.SHARE_MIN:
            sub.append(f'cpu_share = {share!r}, below {P75.SHARE_MIN}')
        if r.get('model') != D.get('model'):
            sub.append(f'model {r.get("model")!r}')
        regs = r.get('per_regime') or {}
        if sorted(regs) != sorted(P75.REGIMES):
            sub.append(f'combinations timed {sorted(regs)}, want the eight')
        if sub:
            probs.append(f'sizes[{shop!r}]: ' + '; '.join(sub))
            continue
        reg, st = P75.worst_regime(r)
        times[mods] = dict(seconds=_fq5_number(st.get('median_s'),
                                               f'{where}: sizes[{shop!r}].per_regime[{reg!r}]'
                                               '.median_s'),
                           regime=reg, size=shop, n=st.get('n'))
    if probs:
        _fq5_refuse(f'{where} is not a record of the paper\'s environment (the {FQ5_ENV} '
                    'environment, one torch thread, valid at every size):\n  '
                    + '\n  '.join(probs))
    return dict(times=times, model=D['model'], path=path)


def gate_deploy(D, path):
    """The re-planning times against the macro file p75 wrote from the same record: its header
    names the record as its source (a macro file of another record, such as the copy of an
    earlier environment, is refused), the same model and one torch thread, and every plotted
    time equals DeployLatency<Word> at its two printed decimals. Returns the table rows."""
    _, P75 = _fq5_scripts()
    where = _fqh_where(path)
    if not path or not os.path.exists(path):
        _fq5_refuse(f'the macro file {where} not found (written by {FQ5_DEPLOY_GENERATOR} from '
                    'the record)')
    with open(path) as f:
        text = f.read()
    mac = _macros(path)
    bad, rows = [], []
    m = re.search(r'^% source: (.+)$', text, re.M)
    src = os.path.normpath(os.path.join(ROOT, m.group(1).strip())) if m else None
    if src != os.path.normpath(os.path.abspath(D['path'])):
        bad.append(f'the header names the source {m.group(1).strip() if m else None!r}, not '
                   f'{_shown(D["path"])} (a macro file of another record)')
    m = re.search(r'^% model (\S+), CPU, torch threads (\d+) \(OMP_NUM_THREADS=(\w+)\)', text,
                  re.M)
    if not m or (m.group(1), m.group(2), m.group(3)) != (D['model'], '1', '1'):
        bad.append(f'the header names {m.group(0)[2:] if m else "no model"!r}, not '
                   f'{D["model"]} with one torch thread')
    for mods, t in sorted(D['times'].items()):
        name = f'DeployLatency{P75.size_word(mods)}'
        got = mac.get(name)
        ok = (got is not None and re.fullmatch(r'\d+\.\d\d', got) is not None
              and abs(t['seconds'] - float(got)) <= 0.005 + 1e-9)
        rows.append((f're-planning, {mods} modules (s)', t['seconds'], name, got,
                     'match' if ok else 'MISMATCH'))
        if not ok:
            bad.append(f'\\{name} = {got}, record {t["seconds"]:.4f} s')
    if bad:
        _fq5_refuse(f'{where} disagrees with the record {_shown(D["path"])}:\n  '
                    + '\n  '.join(bad))
    return rows


def _fq5_marker(marker, ms, xy, dpi, grow=0.0):
    """The outline of a marker with its edge (FQ5_MEW), grown by `grow` pt on every side, in
    display pixels at xy."""
    from matplotlib.markers import MarkerStyle
    from matplotlib.transforms import Affine2D
    st = MarkerStyle(marker)
    size = (ms + FQ5_MEW + 2.0 * grow) * dpi / 72.0
    return st.get_path().transformed(st.get_transform() + Affine2D().scale(size).translate(*xy))


def _fq5_visible(lower, upper):
    """Share of the area of path `lower` that path `upper` does not cover."""
    e = lower.get_extents()
    gx, gy = np.meshgrid(np.linspace(e.x0, e.x1, 41), np.linspace(e.y0, e.y1, 41))
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    inside = lower.contains_points(pts)
    if not inside.any():
        return 1.0
    return float((inside & ~upper.contains_points(pts)).sum()) / float(inside.sum())


def _fq5_clear(leg, shapes, rend, dpi):
    """True when no mark enters the legend's box grown by FQ5_LEG_PAD pt. shapes: (kind, path in
    display pixels, half the line width in pixels); kind 'line' (an open path), 'area' or
    'marker' (filled paths)."""
    from matplotlib.transforms import Bbox
    bb = leg.get_window_extent(rend)
    for kind, p, half in shapes:
        g = FQ5_LEG_PAD * dpi / 72.0 + half
        box = Bbox([[bb.x0 - g, bb.y0 - g], [bb.x1 + g, bb.y1 + g]])
        if p.intersects_bbox(box, filled=kind != 'line'):
            return False
    return True


def _fq5_legend(fig, ax, handles, labels, shapes, locs, **kw):
    """A legend inside the axes in a thin box, at the first of `locs` where no mark enters it
    and the box lies inside the axes; None when none is clear."""
    rend = fig.canvas.get_renderer()
    for loc in locs:
        leg = ax.legend(handles, labels, loc=loc, fontsize=FS_TICK, frameon=True, fancybox=False,
                        edgecolor='0.6', framealpha=1.0, handletextpad=0.5, borderpad=0.4,
                        labelspacing=0.3, borderaxespad=0.5, **kw)
        leg.get_frame().set_linewidth(0.5)
        leg.set_zorder(7)
        fig.canvas.draw()
        lb, ab = leg.get_window_extent(rend), ax.get_window_extent(rend)
        inside = lb.x0 >= ab.x0 and lb.y0 >= ab.y0 and lb.x1 <= ab.x1 and lb.y1 <= ab.y1
        if inside and _fq5_clear(leg, shapes(), rend, fig.dpi):
            return leg, loc
        leg.remove()
    return None, None


def fq5_rule_change(cost_json, cost_macros, deploy_json, deploy_macros, out_dir, suffix=''):
    """Draw the figure of the cost of a rule change after its gates, then print the plotted
    values against the macros. Full mode only: every input must be present and valid, or the
    figure is refused with the list of what is missing. Writes fq5_rule_change<suffix>.pdf and
    .png (300 dpi); main passes no suffix, a development render of synthetic records does."""
    C = load_rule_change(cost_json)
    table = gate_rule_change(C, cost_macros)
    D = load_deploy(deploy_json)
    table += gate_deploy(D, deploy_macros)
    P99, P75 = _fq5_scripts()
    sizes = C['sizes']
    mods = [P99.modules(s) for s in sizes]
    if sorted(mods) != sorted(FQ5_SHAPE):
        _fq5_refuse(f'the sizes {sizes} of p99 are not the batches {sorted(FQ5_SHAPE)} of the '
                    'figure')
    print(f'  rule change: {_shown(cost_json)} (training records on {", ".join(C["gpu"])}; '
          f'check schedules {C["schedules"]}); re-planning: {_shown(deploy_json)} '
          f'({D["model"]}, one torch thread)')

    # ---- values
    n = len(COMBOS)
    k = np.arange(1, n + 1, dtype=float)
    h = np.sort(np.array([C['spec'][R] for R in COMBOS]))
    mean = C['total'] / n                       # the sum p99 prints, over the eight
    line_s = k * mean
    band_lo, band_hi = np.cumsum(h), np.cumsum(h[::-1])
    pol = C['policy']
    T = {}                                      # {modules: {row: seconds}}
    for s, m in zip(sizes, mods):
        T[m] = {FQ5_ROWS[0]: C['checks'][s][0] / 1000.0, FQ5_ROWS[1]: D['times'][m]['seconds'],
                FQ5_ROWS[2]: C['checks'][s][1] / 1000.0}

    # ---- layout in inches (printed size): the row labels of (b) measured first, so that each
    # panel with its labels takes the same width
    fig_h = FQ5_FIG_H
    left, right, gap, top, ax_y0, xlab = 0.62, 0.16, 0.30, 0.27, 0.42, 0.25
    ax_h = fig_h - ax_y0 - top
    fig = plt.figure(figsize=(FIG_W, fig_h), dpi=300)
    rend = fig.canvas.get_renderer()
    probe = [fig.text(0, 0, r, fontsize=FS_TICK) for r in FQ5_ROWS]
    lab_w = max(t.get_window_extent(rend).width for t in probe) / fig.dpi + 4.0 / 72
    for t in probe:
        t.remove()
    rest = FIG_W - left - right - gap - lab_w
    wa = (rest + lab_w - left) / 2
    wb = rest - wa
    ax = fig.add_axes([left / FIG_W, ax_y0 / fig_h, wa / FIG_W, ax_h / fig_h])
    bx = fig.add_axes([(left + wa + gap + lab_w) / FIG_W, ax_y0 / fig_h, wb / FIG_W,
                       ax_h / fig_h])
    px = fig.dpi / 72.0

    # ---- (a) training before the first use
    ln_s, = ax.plot(k, line_s, color=FQ5_SPEC['color'], lw=FQ5_LW, zorder=4,
                    solid_capstyle='butt')
    mk_s, = ax.plot(k, line_s, FQ5_SPEC['marker'], ms=FQ5_SPEC['ms'], ls='none', zorder=6,
                    mfc=FQ5_SPEC['color'], mec='white', mew=FQ5_MEW)
    ln_p, = ax.plot([k[0], k[-1]], [pol, pol], color=FQ5_POL['color'], lw=FQ5_LW, zorder=3,
                    solid_capstyle='butt')
    mk_p, = ax.plot(k, np.full(n, pol), FQ5_POL['marker'], ms=FQ5_POL['ms'], ls='none',
                    zorder=5, mfc=FQ5_POL['color'], mec='white', mew=FQ5_MEW)
    ax.set_xlim(0.5, n + 0.5)
    ax.set_xticks(k)
    ax.set_xticklabels([str(int(i)) for i in k])
    data_top = float(max(band_hi.max(), line_s.max(), pol))
    raw = MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]).tick_values(0, data_top)
    step = float(raw[1] - raw[0])
    n_steps = int(np.ceil(data_top / step - 1e-9))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _fmt(round(v, 6))))
    ax.tick_params(axis='both', length=3, pad=2)
    ax.set_ylabel('training (GPU hours)', labelpad=5)
    ax.set_xlabel('number of rule combinations served')
    ax.set_title('(a) Training before the first use', fontsize=FS_TITLE, pad=7)
    h_s = Line2D([], [], color=FQ5_SPEC['color'], lw=FQ5_LW, marker=FQ5_SPEC['marker'],
                 ms=FQ5_SPEC['ms'], mfc=FQ5_SPEC['color'], mec='white', mew=FQ5_MEW)
    h_b = Patch(facecolor=FQ5_SPEC['color'], alpha=FQ5_BAND_ALPHA, edgecolor='none')
    h_p = Line2D([], [], color=FQ5_POL['color'], lw=FQ5_LW, marker=FQ5_POL['marker'],
                 ms=FQ5_POL['ms'], mfc=FQ5_POL['color'], mec='white', mew=FQ5_MEW)

    def shapes_a(band):
        out = []
        for ln in (ln_s, ln_p):
            out.append(('line', ln.get_transform().transform_path(ln.get_path()),
                        FQ5_LW / 2 * px))
        for m_, st in ((mk_s, FQ5_SPEC), (mk_p, FQ5_POL)):
            for xy in ax.transData.transform(np.column_stack([m_.get_xdata(), m_.get_ydata()])):
                out.append(('marker', _fq5_marker(st['marker'], st['ms'], xy, fig.dpi), 0.0))
        if band is not None:
            out.append(('area', ax.transData.transform_path(band.get_paths()[0]), 0.0))
        return out

    band, grid, leg_a, loc_a, band_pt = None, [], None, None, 0.0
    for extra in range(FQ5_HEADROOM + 1):
        ticks = step * np.arange(n_steps + extra + 1)
        ax.set_ylim(0, ticks[-1] * 1.04)
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        for g in grid:
            g.remove()
        grid = [ax.axhline(t, color=C_GRID, lw=0.4, zorder=0.5) for t in ticks[1:]]
        fig.canvas.draw()
        # the band is drawn only where it is thicker than the line at print size
        band_pt = float((band_hi - band_lo).max()) * (ax_h * 72.0) / (ticks[-1] * 1.04)
        if band is not None:
            band.remove()
            band = None
        if band_pt > FQ5_LW:
            band = ax.fill_between(k, band_lo, band_hi, color=FQ5_SPEC['color'],
                                   alpha=FQ5_BAND_ALPHA, lw=0, zorder=1)
        keep = (True, band is not None, True)
        hs = [x for x, y in zip((h_s, h_b, h_p), keep) if y]
        labs = [x for x, y in zip(FQ5_LEGEND_A, keep) if y]
        leg_a, loc_a = _fq5_legend(fig, ax, hs, labs, lambda: shapes_a(band), ('upper left',),
                                   handlelength=1.8)
        if leg_a is not None:
            break
    if leg_a is None:
        _fq5_refuse('no place in panel (a) leaves its legend clear of the lines, the band and '
                    'the markers')

    # ---- (b) steps at a rule change: ticks at the decades that enclose the data, the axis
    # padded beyond them by the part of a decade that keeps every marker clear of its ends
    ys = {r: float(len(FQ5_ROWS) - 1 - i) for i, r in enumerate(FQ5_ROWS)}
    vals = [v for m in T for v in T[m].values()]
    v_lo, v_hi = np.log10(min(vals)), np.log10(max(vals))
    lo_e, hi_e = int(np.floor(v_lo)), int(np.ceil(v_hi))
    hi_e = max(hi_e, lo_e + 1)
    reach = max(_fq5_marker(mk, ms, (0, 0), 72.0).get_extents().x1 for mk, ms in
                FQ5_SHAPE.values()) + 1.0     # pt from a marker's centre to clear an end
    pad_lo = pad_hi = 0.0
    for _ in range(50):
        need = reach / (wb * 72.0 / (hi_e - lo_e + pad_lo + pad_hi))   # in decades
        new = (max(0.0, need - (v_lo - lo_e)), max(0.0, need - (hi_e - v_hi)))
        if np.allclose(new, (pad_lo, pad_hi), rtol=0, atol=1e-9):
            break
        pad_lo, pad_hi = new
    bx.set_xscale('log')
    bx.set_xlim(10.0 ** (lo_e - pad_lo), 10.0 ** (hi_e + pad_hi))
    decs = [10.0 ** e for e in range(lo_e, hi_e + 1)]
    bx.xaxis.set_major_locator(FixedLocator(decs))
    bx.xaxis.set_minor_locator(NullLocator())
    bx.xaxis.set_major_formatter(FuncFormatter(lambda v, _: _fq5_decade(int(round(np.log10(v))))))
    for e in range(lo_e, hi_e + 1):
        if lo_e - pad_lo + 1e-6 < e < hi_e + pad_hi - 1e-6:
            bx.axvline(10.0 ** e, color=C_GRID, lw=0.4, zorder=0.5)
    bx.set_ylim(-0.6, len(FQ5_ROWS) - 0.4)
    bx.set_yticks([ys[r] for r in FQ5_ROWS])
    bx.set_yticklabels(list(FQ5_ROWS))
    bx.spines['left'].set_visible(False)
    bx.tick_params(axis='y', length=0, pad=4)
    bx.tick_params(axis='x', length=3, pad=2)
    bx.set_xlabel('time per batch')
    bx.set_title('(b) Steps at a rule change', fontsize=FS_TITLE, pad=7)
    fig.canvas.draw()
    # each marker on the line of its row, or, where it would touch a marker placed before it in
    # the row, FQ5_NUDGE pt above or below
    inv = bx.transData.inverted()
    drawn_b, nudged = [], []
    for r in FQ5_ROWS:
        placed = []
        for m in sorted(T):
            mk, ms = FQ5_SHAPE[m]
            x0, y0 = bx.transData.transform((T[m][r], ys[r]))
            for j in (0, 1, -1, 2, -2):
                xy = (x0, y0 + j * FQ5_NUDGE * px)
                p = _fq5_marker(mk, ms, xy, fig.dpi, grow=FQ5_CLEAR / 2)
                if not any(p.intersects_path(q, filled=True) for q in placed):
                    break
            else:
                _fq5_refuse(f'no free place in the row {r!r} for the marker of {m} modules')
            placed.append(p)
            y = float(inv.transform(xy)[1])
            if j:
                nudged.append(f'{r}, {m} modules, {j * FQ5_NUDGE:+g} pt')
            bx.plot([T[m][r]], [y], mk, ms=ms, ls='none', zorder=5, mfc=FQ5_COLOR_B, mec='white',
                    mew=FQ5_MEW)
            drawn_b.append((r, m, T[m][r], y))

    def shapes_b():
        return [('marker', _fq5_marker(FQ5_SHAPE[m][0], FQ5_SHAPE[m][1],
                                       bx.transData.transform((x, y)), fig.dpi), 0.0)
                for _, m, x, y in drawn_b]

    hb = [Line2D([], [], ls='none', marker=FQ5_SHAPE[m][0], ms=FQ5_SHAPE[m][1],
                 mfc=FQ5_COLOR_B, mec='white', mew=FQ5_MEW) for m in sorted(FQ5_SHAPE)]
    leg_b, loc_b = _fq5_legend(fig, bx, hb, [f'{m} modules' for m in sorted(FQ5_SHAPE)],
                               shapes_b, ('upper right', 'lower right', 'upper left',
                                          'lower left', 'center right', 'center left'),
                               handlelength=1.2)
    if leg_b is None:
        _fq5_refuse('no corner of panel (b) leaves its legend clear of the markers')

    # ---- one baseline for the two axis titles under the panels
    for a in (ax, bx):
        a.xaxis.set_label_coords(0.5, -xlab / ax_h)
        a.xaxis.label.set_verticalalignment('top')
    fig.canvas.draw()

    # ---- checks independent of the placement
    ea, eb = ax.get_window_extent(rend), bx.get_window_extent(rend)
    if abs(ea.y0 - eb.y0) > 0.5 or abs(ea.height - eb.height) > 0.5:
        _fq5_refuse('the two panels do not share one height and one baseline')
    for what, pair in (('axis titles', (ax.xaxis.label, bx.xaxis.label)),
                       ('panel titles', (ax.title, bx.title))):
        y0s = [t.get_window_extent(rend).y0 for t in pair]
        if abs(y0s[0] - y0s[1]) > 0.5:
            _fq5_refuse(f'the {what} of the two panels are not on one baseline')
    for t in bx.get_yticklabels():
        if t.get_window_extent(rend).x0 < ax.get_tightbbox(rend).x1 + 0.1 * fig.dpi:
            _fq5_refuse(f'the row label {t.get_text()!r} comes within 0.1 in of panel (a)')
    sa = shapes_a(band)
    pol_marks = [p for kind, p, _ in sa if kind == 'marker'][n:]
    spec_marks = [p for kind, p, _ in sa if kind == 'marker'][:n]
    least = 1.0
    for i, p in enumerate(pol_marks):              # drawn under the specialists' squares
        for q in spec_marks:
            if p.intersects_path(q, filled=True):
                v = _fq5_visible(p, q)
                least = min(least, v)
                if v < FQ5_VISIBLE:
                    _fq5_refuse(f'a specialist marker hides {100 * (1 - v):.0f}% of the policy '
                                f'marker at {i + 1} combinations')
    for a, kind_shapes in ((ax, sa), (bx, shapes_b())):
        e = a.get_window_extent(rend)
        for kind, p, _ in kind_shapes:
            if kind == 'marker':
                b = p.get_extents()
                if b.x0 < e.x0 or b.x1 > e.x1 or b.y0 < e.y0 or b.y1 > e.y1:
                    _fq5_refuse('a marker leaves its panel')
    ms_b = [p for _, p, _ in shapes_b()]
    for i in range(len(ms_b)):
        for j in range(i + 1, len(ms_b)):
            if ms_b[i].intersects_path(ms_b[j], filled=True):
                _fq5_refuse(f'the markers of {drawn_b[i][:2]} and {drawn_b[j][:2]} overlap')
    n_text = _check_layout(fig, (), leg_a, tag='fq5_rule_change')

    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, 'fq5_rule_change' + suffix)
    fig.savefig(stem + '.pdf')
    fig.savefig(stem + '.png', dpi=300)
    plt.close(fig)
    yl = ax.get_ylim()
    print(f'  fq5_rule_change: {FIG_W:.2f} x {fig_h:.2f} in, panels {wa:.2f} and {wb:.2f} in wide, '
          f'{ax_h:.2f} in high; (a) y 0 to {yl[1]:g} h, legend {loc_a}; (b) '
          f'{_fq5_decade(lo_e)} to {_fq5_decade(hi_e)}, legend {loc_b}; {n_text} texts checked, '
          'no overlap')
    print('  band of the choice of combinations: '
          + (f'drawn, {band_pt:.1f} pt at its widest' if band is not None else
             f'left out, {band_pt:.2f} pt at its widest, not thicker than the line '
             f'({FQ5_LW} pt)'))
    if least < 1.0:
        print(f'  where the two lines cross, at least {100 * least:.0f}% of each policy marker '
              'stays visible under the specialist marker')
    print('  markers moved off the line of their row: ' + ('; '.join(nudged) or 'none'))

    # ---- the plotted values against the macros
    print(f'\n  {"quantity":38s} {"plotted":>10s}   {"macro":28s} {"printed":>8s}   check')
    for label, v, name, got, status in table:
        print(f'  {label:38s} {v:10.4f}   \\{name:27s} {str(got):>8s}   {status}')
    print(f'  mean line at k combinations: k x {mean:.4f} h (the sum \\RccCatalogueHours over '
          f'eight); band from the sum of the k smallest to the sum of the k largest hours')
    for m in sorted(D['times']):
        t = D['times'][m]
        print(f'  re-planning at {m} modules: {_shown(deploy_json)} sizes[{t["size"]}], '
              f'combination {t["regime"]} (largest median), {t["n"]} instances')
    print(f'\n  macro gate: {sum(r[-1] == "match" for r in table)} of {len(table)} plotted '
          'values match their macros')
    return [stem + '.pdf', stem + '.png']


# ------------------------------------------------------------------ fq4_limits
FQL_SEEDS = (301, 302, 303, 304, 305)          # the training seeds of every record it reads
FQL_ENV = FQ5_ENV                              # the environment of the paper
FQL_FAMILY_GENERATOR = 'scripts/p88_family_inference.py'
FQL_PERFLOOR_JSON = 'results/per_floor.json'    # p118, read through arm_registry.out
FQL_PERFLOOR_MACROS = 'manuscript/macros_perfloor.tex'
FQL_PERFLOOR_GENERATOR = 'scripts/p118_per_floor.py'
FQL_PROD_JSON = 'results/production_specialists.json'   # p128, read through arm_registry.out
FQL_PROD_MACROS = 'manuscript/macros_prodspec.tex'
FQL_PROD_GENERATOR = 'scripts/p128_production_specialists.py'
FQL_SENS_CSV = 'results/sensitivity_inference.csv'    # p87 infer writes its JSON beside it
FQL_SENS_MACROS = 'manuscript/macros_sensitivity.tex'
FQL_SENS_GENERATOR = 'scripts/p87_sensitivity.py infer'
FQL_SIZES = (10, 20, 27, 30, 50, 80)           # modules in the batch, panel (a)
FQL_POOLED = len(COMBOS)                       # combinations in a pooled row of panel (a)
FQL_KINDS = ('spec', 'rule')
FQL_VARIANTS = ('A', 'B')
_FQL_LAG, _FQL_SETUP, _FQL_WIN = ('L', 'LS', 'LW', 'LSW'), ('S', 'LS', 'SW', 'LSW'), \
    ('W', 'LW', 'SW', 'LSW')
# Panel (b): the 22 changed levels of p87 in the order and the nine groups of Table
# tab:sn-sensitivity. Per group: the label under the group in variant A, the label lines of
# variant B (line k beside row k of the group), the combinations its levels change; per level:
# the key of p87, its macro word (p87 LEVEL_WORDS), the value label, and the parameter and value
# the record states for it.
FQL_GROUPS = (
    dict(a='processing-\ntime noise', b=('processing-time noise',), reach=tuple(COMBOS),
         levels=(('dur_cv10', 'DurCvTen', '0.1', 'duration coefficient of variation', 0.1),
                 ('dur_cv20', 'DurCvTwenty', '0.2', 'duration coefficient of variation', 0.2),
                 ('dur_cv30', 'DurCvThirty', '0.3', 'duration coefficient of variation', 0.3))),
    dict(a='finishing,\nassembly\ntimes', b=('finishing and', 'assembly times'),
         reach=tuple(COMBOS),
         levels=(('dur_long_m25', 'DurLongDown', '0.75',
                  'duration scale of the finishing and assembly stations', 0.75),
                 ('dur_long_p25', 'DurLongUp', '1.25',
                  'duration scale of the finishing and assembly stations', 1.25))),
    dict(a='lag scale', b=('lag scale',), reach=_FQL_LAG,
         levels=(('lag_x050', 'LagHalf', '0.5', 'lag scale', 0.5),
                 ('lag_x075', 'LagThreeQuarters', '0.75', 'lag scale', 0.75),
                 ('lag_x125', 'LagFiveQuarters', '1.25', 'lag scale', 1.25),
                 ('lag_x150', 'LagThreeHalves', '1.5', 'lag scale', 1.5))),
    dict(a='setup scale', b=('setup scale',), reach=_FQL_SETUP,
         levels=(('setup_x050', 'SetupHalf', '0.5', 'setup scale', 0.5),
                 ('setup_x075', 'SetupThreeQuarters', '0.75', 'setup scale', 0.75),
                 ('setup_x125', 'SetupFiveQuarters', '1.25', 'setup scale', 1.25),
                 ('setup_x150', 'SetupThreeHalves', '1.5', 'setup scale', 1.5))),
    dict(a='same-\nclass\nsetups', b=('same-class setups',), reach=_FQL_SETUP,
         levels=(('setup_same0', 'SetupSameZero', 'zero', 'same-class setup scale', 0.0),)),
    dict(a='stations\nwith\nsetups', b=('stations with setups',), reach=_FQL_SETUP,
         levels=(('setup_sub_strict', 'SetupShell', '7 of 25', 'stations charged with setups', 7),
                 ('setup_sub_wide', 'SetupProduction', '21 of 25',
                  'stations charged with setups', 21))),
    dict(a='window\ndensity', b=('window density',), reach=_FQL_WIN,
         levels=(('win_x050', 'WinHalf', '0.5', 'outage density scale', 0.5),
                 ('win_x150', 'WinThreeHalves', '1.5', 'outage density scale', 1.5),
                 ('win_x200', 'WinDouble', '2', 'outage density scale', 2.0))),
    dict(a='stations\nin the\nfactory', b=('stations in the factory',), reach=tuple(COMBOS),
         levels=(('tight15', 'Tight', '15', 'stations in the factory', 15),)),
    dict(a='one\nshell\ntype', b=('shell type of', 'all modules'), reach=tuple(COMBOS),
         levels=(('mix_rc', 'MixConcrete', 'concrete', 'project mix', 'reinforced concrete'),
                 ('mix_steel', 'MixSteel', 'steel', 'project mix', 'steel'))),
)
FQL_N_LEVELS = sum(len(G['levels']) for G in FQL_GROUPS)       # 22
# The two series: the specialists' slate (fq2_reuse, fs_convergence, fq5_rule_change) for the
# policy against the specialists, filled circles; a medium green that no other figure uses for
# the policy against the best dispatching rule, open squares. Shape, fill and grey value tell
# them apart in a greyscale print.
FQL_RULE_COLOR = '#4a8f5c'
FQL_STYLE = {
    'spec': dict(text='against the specialists', marker='o', ms=4.2,
                 color=FS_CONV_SPEC_COLORS[0], open=False),
    'rule': dict(text='against the best dispatching rule', marker='s', ms=3.5,
                 color=FQL_RULE_COLOR, open=True),
}
FQL_MEW_OPEN, FQL_MEW_FILLED = 1.0, 0.5
FQL_CI_LW = FQ2_CI_LW
# pt from a position to each of its two marks: small against the pitch of the positions, so the
# two marks of a position read as a pair and stay apart from the next position's
FQL_DX = 2.1
# panel (a), whose sizes stand further apart: the two markers of a size never overlap, whatever
# their values, because twice this offset exceeds their two radii and the clearance (checked when
# the figure is drawn); at six sizes in variant A the next size's marks stay clear as well
FQL_DX_SIZES = 2.9
FQL_CLEAR = 0.4              # pt of white kept between two marks
FQL_TEXT_CLEAR = {'x': 2.0, 'y': 1.0}    # pt between neighbouring labels of a row, a column
FQL_VALUE_LABEL = 'makespan difference of the policy (%)'
FQL_SIZE_LABEL = 'modules in the batch'
FQL_TITLES = ('(a) Batch size', '(b) Benchmark parameters')
FQL_MARGIN_TEXT = '±1% margin'
# Variant A: one row, the y axis shared; variant B: panel (b) as rows (a forest plot). The
# height of variant A holds group labels of FQL_A_LINES lines; labels of more lines add their
# extra height below the axes (_fql_draw_a).
FQL_A_H, FQL_B_H = 2.95, 4.55
FQL_A_LINES = 2
FQL_PAPER_DIR = os.path.join(ROOT, 'manuscript')    # the macro files the manuscript inputs
FQL_A_GAP, FQL_B_GAP = 0.8, 0.45            # positions (rows) between two groups of panel (b)


def _fql_refuse(msg):
    raise SystemExit(f'fq4_limits REFUSING: {msg}')


def fql_sens_json(path):
    """The JSON of p87 infer: the path itself when it names a .json file, else the file beside the
    CSV with the extension .json."""
    return path if str(path).endswith('.json') else os.path.splitext(path)[0] + '.json'


def _fql_open(path, generator):
    """The record at path, or None when the file does not exist; refuses a file that is not a JSON
    object written by generator."""
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            J = json.load(f)
    except (OSError, ValueError) as e:
        _fql_refuse(f'{_shown(path)} is not readable JSON: {e}')
    if not isinstance(J, dict) or J.get('generator') != generator:
        _fql_refuse(f'{_shown(path)} was written by '
                    f'{J.get("generator") if isinstance(J, dict) else None!r}, not {generator}')
    return J


def _fql_setting(J, path, env_ok, env_text, dry, allow_partial):
    """The marks a record carries: the five training seeds of the paper and its environment, or a
    refusal; a record of a dry or partial run only with allow_partial. Returns True for such a
    record (the file names then end in _dev)."""
    try:
        seeds = [int(s) for s in J.get('seeds') or []]
    except (TypeError, ValueError):
        seeds = J.get('seeds')
    if seeds != list(FQL_SEEDS):
        _fql_refuse(f'{_shown(path)} holds the training seeds {seeds}, the figure needs '
                    f'{",".join(map(str, FQL_SEEDS))}')
    if not env_ok:
        _fql_refuse(f'{_shown(path)} is a record of {env_text}, not of the {FQL_ENV} environment')
    if dry and not allow_partial:
        _fql_refuse(f'{_shown(path)} is the record of a dry or partial run; it is drawn only with '
                    '--allow-partial')
    return bool(dry)


def _fql_row(src, where, c):
    """mean, lo and hi of one pooled row, checked: finite numbers, the mean inside its interval, and
    c combinations pooled."""
    if not isinstance(src, dict):
        _fql_refuse(f'{where} is missing')
    lack = [k for k in ('mean', 'lo', 'hi', 'c') if k not in src]
    if lack:
        _fql_refuse(f'{where} has no field {lack}')
    try:
        mean, lo, hi = (float(src[k]) for k in ('mean', 'lo', 'hi'))
    except (TypeError, ValueError):
        _fql_refuse(f'{where}: a value that is not a number')
    if not all(np.isfinite(v) for v in (mean, lo, hi)):
        _fql_refuse(f'{where}: a value that is not finite')
    if not lo <= mean <= hi:
        _fql_refuse(f'{where}: the mean {mean:+.4f} lies outside [{lo:+.4f}, {hi:+.4f}]')
    if int(src['c']) != c:
        _fql_refuse(f'{where} pools {src["c"]} combinations, not {c}')
    return dict(mean=mean, lo=lo, hi=hi, where=where)


def _fql_same(a, b, what, tol=1e-9):
    if any(abs(a[k] - b[k]) > tol for k in ('mean', 'lo', 'hi')):
        _fql_refuse(f'{what}: {a["where"]} is {a["mean"]:+.6f} [{a["lo"]:+.6f}, {a["hi"]:+.6f}] '
                    f'and {b["where"]} is {b["mean"]:+.6f} [{b["lo"]:+.6f}, {b["hi"]:+.6f}], so the '
                    'records are not of one run')


def fql_inputs(family_json, family_macros, perfloor_json, perfloor_macros, prod_json, prod_macros,
               sens_json, sens_macros, partial=False, allow_partial=False, paper_dir=None):
    """Every value the figure plots, after every check and gate. Panel (a): the pooled rows over the
    eight combinations at 10 modules (the policy against the specialists from the record of p88,
    parity.pooled.All; against the best dispatching rule from the default level of p87, whose
    pooled row against the specialists must have the same mean as that of p88), at 20 and 30
    modules (p88, larger_shops), 27 (p118) and 50 and 80 (p128). Panel (b): the pooled rows of the
    22 changed levels of p87 over the combinations each level changes. In the full mode every size
    must be present; in the partial mode a size whose record is absent is left out and listed.
    paper_dir: the folder of the macro files the manuscript inputs (see _fql_gate). Returns
    dict(sizes, groups, absent, table, copies, dev, partial, sources)."""
    absent, dev, sources, checks = [], False, {}, []
    exact = f'environment {FQL_ENV},'

    # ---- p87: panel (b) and the 10-module row against the best dispatching rule
    S = _fql_open(sens_json, FQL_SENS_GENERATOR)
    if S is None:
        _fql_refuse(f'the record of p87 infer {_fqh_where(sens_json)} not found; panel (b) and the '
                    '10-module point need it in every mode')
    dev |= _fql_setting(S, sens_json, S.get('mask') == FQL_ENV, f'the mask {S.get("mask")!r}',
                        S.get('partial'), allow_partial)
    levels = S.get('levels') or {}

    def pool_of(key, reach, kind):
        L = levels.get(key)
        where = f'{_shown(sens_json)}: levels.{key}'
        if not isinstance(L, dict):
            _fql_refuse(f'{where} is missing')
        if tuple(L.get('reached') or ()) != tuple(reach):
            _fql_refuse(f'{where} changes {L.get("reached")}, the figure expects {list(reach)}')
        name = 'pool' if len(reach) == len(COMBOS) else 'reached'
        P = (L.get('pools') or {}).get(name)
        if not isinstance(P, dict) or tuple(P.get('combos') or ()) != tuple(reach):
            _fql_refuse(f'{where}.pools.{name} is missing or pools other combinations than '
                        f'{list(reach)}')
        return _fql_row(P.get(kind), f'{where}.pools.{name}.{kind}', len(reach)), \
            'Pool' if name == 'pool' else 'Reached'

    groups = []
    for G in FQL_GROUPS:
        items = []
        for key, word, label, par, val in G['levels']:
            L = levels.get(key) or {}
            ok_val = (L.get('value') == val if isinstance(val, str) else
                      isinstance(L.get('value'), (int, float)) and abs(L['value'] - val) < 1e-9)
            if key in levels and (L.get('parameter') != par or not ok_val):
                _fql_refuse(f'{_shown(sens_json)}: levels.{key} sets {L.get("parameter")!r} to '
                            f'{L.get("value")!r}, the figure expects {par!r} {val!r}')
            spec, row = pool_of(key, G['reach'], 'pol_spec')
            rule, _ = pool_of(key, G['reach'], 'pol_rule')
            items.append(dict(key=key, word=word, label=label, spec=spec, rule=rule, row=row))
            for kind, it in (('spec', spec), ('rule', rule)):
                checks.append(('b', f'{" ".join(G["b"])} {label}', kind, it, sens_macros,
                               f'Sens{word}Pol{"Spec" if kind == "spec" else "Rule"}{row}'))
        groups.append(dict(G, items=items))
    rule10, _ = pool_of('default', tuple(COMBOS), 'pol_rule')
    spec10_p87, _ = pool_of('default', tuple(COMBOS), 'pol_spec')
    sources['sensitivity'] = f'{_shown(sens_json)} (p87 infer)'

    sizes = {}

    def add(m, spec, rule, gates, src):
        sizes[m] = dict(modules=m, spec=spec, rule=rule, src=src)
        for path, s_spec, s_rule in gates:
            if s_spec:
                checks.append(('a', f'{m} modules', 'spec', spec, path, s_spec))
            if s_rule:
                checks.append(('a', f'{m} modules', 'rule', rule, path, s_rule))

    # ---- p88: 10, 20 and 30 modules
    F = _fql_open(family_json, FQL_FAMILY_GENERATOR)
    if F is None:
        absent += [(m, f'the record of p88 {_fqh_where(family_json)} not found')
                   for m in (10, 20, 30)]
    else:
        _fql_setting(F, family_json, str(F.get('setting', '')).startswith(exact),
                     repr(F.get('setting')), False, allow_partial)
        where = _shown(family_json)
        spec10 = _fql_row(((F.get('parity') or {}).get('pooled') or {}).get('All'),
                          f'{where}: parity.pooled.All', FQL_POOLED)
        if abs(spec10['mean'] - spec10_p87['mean']) > 1e-6:
            _fql_refuse(f'the 10-module rows come from two records: the pooled difference from the '
                        f'specialists is {spec10["mean"]:+.6f} in {where} and '
                        f'{spec10_p87["mean"]:+.6f} at the default level of {_shown(sens_json)}, so '
                        'the two records do not hold one evaluation of the 10-module instances')
        add(10, spec10, rule10, [(family_macros, 'FamParPoolAll', None),
                                 (sens_macros, None, 'SensDefaultPolRulePool')],
            dict(spec=f'{where}: parity.pooled.All',
                 rule=f'{_shown(sens_json)}: levels.default.pools.pool.pol_rule'))
        H = ((F.get('sections') or {}).get('H') or {}).get('computed')
        for m, key, word in ((20, '20x25', 'Twenty'), (30, '30x25', 'Thirty')):
            x = (F.get('larger_shops') or {}).get(key)
            if not H or not isinstance(x, dict):
                absent.append((m, f'section H of {where} holds no {key}'))
                continue
            add(m, _fql_row((x.get('parity_pooled') or {}).get('All'),
                            f'{where}: larger_shops.{key}.parity_pooled.All', FQL_POOLED),
                _fql_row(x.get('rule_pool_pooled'), f'{where}: larger_shops.{key}.rule_pool_pooled',
                         FQL_POOLED),
                [(family_macros, f'FamShop{word}ParPoolAll', f'FamShop{word}RulePool')],
                dict(spec=f'{where}: larger_shops.{key}.parity_pooled.All',
                     rule=f'{where}: larger_shops.{key}.rule_pool_pooled'))
        sources['family'] = f'{where} (p88)'

    # ---- p118: 27 modules, and 20 and 30 side by side
    P1 = _fql_open(perfloor_json, FQL_PERFLOOR_GENERATOR)
    if P1 is None:
        absent.append((27, f'the record of p118 {_fqh_where(perfloor_json)} not found'))
    else:
        dev |= _fql_setting(P1, perfloor_json, str(P1.get('setting', '')).startswith(exact),
                            repr(P1.get('setting')), P1.get('dry'), allow_partial)
        where = _shown(perfloor_json)
        blocks = P1.get('sizes') or {}
        x = blocks.get('27x25')
        if not isinstance(x, dict):
            _fql_refuse(f'{where} holds no sizes.27x25')
        add(27, _fql_row((x.get('parity_pooled') or {}).get('All'),
                         f'{where}: sizes.27x25.parity_pooled.All', FQL_POOLED),
            _fql_row(x.get('rule_pool_pooled'), f'{where}: sizes.27x25.rule_pool_pooled',
                     FQL_POOLED),
            [(perfloor_macros, 'PflParPoolAll', 'PflRulePool')],
            dict(spec=f'{where}: sizes.27x25.parity_pooled.All',
                 rule=f'{where}: sizes.27x25.rule_pool_pooled'))
        for m, key, word in ((20, '20x25', 'Twenty'), (30, '30x25', 'Thirty')):
            if m not in sizes:
                continue
            y = blocks.get(key) or {}
            _fql_same(sizes[m]['spec'], _fql_row((y.get('parity_pooled') or {}).get('All'),
                                                 f'{where}: sizes.{key}.parity_pooled.All',
                                                 FQL_POOLED), f'{m} modules')
            _fql_same(sizes[m]['rule'], _fql_row(y.get('rule_pool_pooled'),
                                                 f'{where}: sizes.{key}.rule_pool_pooled',
                                                 FQL_POOLED), f'{m} modules')
            checks += [('a', f'{m} modules', 'spec', sizes[m]['spec'], perfloor_macros,
                        f'Pfl{word}ParPoolAll'),
                       ('a', f'{m} modules', 'rule', sizes[m]['rule'], perfloor_macros,
                        f'Pfl{word}RulePool')]
        sources['per_floor'] = f'{where} (p118)'

    # ---- p128: 50 and 80 modules
    P2 = _fql_open(prod_json, FQL_PROD_GENERATOR)
    if P2 is None:
        absent += [(m, f'the record of p128 {_fqh_where(prod_json)} not found')
                   for m in (50, 80)]
    else:
        dev |= _fql_setting(P2, prod_json, str(P2.get('setting', '')).startswith(exact),
                            repr(P2.get('setting')), P2.get('dry'), allow_partial)
        where = _shown(prod_json)
        for m, key, word in ((50, '50x25n30', 'Fifty'), (80, '80x25n30', 'Eighty')):
            x = (P2.get('sizes') or {}).get(key)
            if not isinstance(x, dict):
                _fql_refuse(f'{where} holds no sizes.{key}')
            add(m, _fql_row((x.get('parity_pooled') or {}).get('All'),
                            f'{where}: sizes.{key}.parity_pooled.All', FQL_POOLED),
                _fql_row(x.get('rule_pool_pooled'), f'{where}: sizes.{key}.rule_pool_pooled',
                         FQL_POOLED),
                [(prod_macros, f'Pbs{word}ParPoolAll', f'Pbs{word}RulePool')],
                dict(spec=f'{where}: sizes.{key}.parity_pooled.All',
                     rule=f'{where}: sizes.{key}.rule_pool_pooled'))
        if 10 in sizes:                    # p128 prints the 10-module row against the rule too
            y = ((P2.get('side_by_side') or {}).get('rows') or {}).get('10x25') or {}
            _fql_same(sizes[10]['rule'], _fql_row(y.get('rule'),
                                                  f'{where}: side_by_side.rows.10x25.rule',
                                                  FQL_POOLED),
                      '10 modules against the best dispatching rule')
            checks.append(('a', '10 modules', 'rule', sizes[10]['rule'], prod_macros,
                           'PbsTenRulePool'))
        sources['production'] = f'{where} (p128)'

    absent.sort()
    if absent and not partial:
        _fql_refuse('the full figure needs all six sizes; missing:\n  '
                    + '\n  '.join(f'{m} modules: {why}' for m, why in absent)
                    + '\n(--fq4-limits-partial draws the sizes whose records exist)')
    if not sizes:
        _fql_refuse('no size of panel (a) has a record:\n  '
                    + '\n  '.join(f'{m} modules: {why}' for m, why in absent))
    table, copies = _fql_gate(checks, paper_dir)
    return dict(sizes=[sizes[m] for m in FQL_SIZES if m in sizes], groups=groups, absent=absent,
                table=table, copies=copies, dev=dev, partial=partial, sources=sources)


def _fql_gate(checks, paper_dir=None):
    """Every plotted mean and interval end against the macro that prints it, at its two signed
    decimals (<stem>, <stem>Lo, <stem>Hi). A missing macro file, a missing macro or a disagreement
    is refused. With paper_dir (the folder of the macro files the manuscript inputs), the copy of
    each macro file there, when it exists at another path, must print the same text for every
    macro read; a copy that does not exist is listed. Returns the table rows (panel, label, kind,
    field, value, macro, printed) and the lines on the copies."""
    cache, rows, bad, used = {}, [], [], {}
    for panel, label, kind, it, path, stem in checks:
        if path not in cache:
            if not path or not os.path.exists(path):
                _fql_refuse(f'the macro file {_fqh_where(path)} not found (it prints \\{stem})')
            cache[path] = _macros(path)
        mac = cache[path]
        for fld, suffix in (('mean', ''), ('lo', 'Lo'), ('hi', 'Hi')):
            name, v = stem + suffix, it[fld]
            got = mac.get(name)
            if got is None:
                bad.append(f'\\{name}: no macro in {_shown(path)}')
                continue
            if re.fullmatch(r'[+-]\d+\.\d\d', got) is None or abs(v - float(got)) > 0.005 + 1e-9:
                bad.append(f'\\{name} = {got} in {_shown(path)}, record {v:+.4f}')
            rows.append((panel, label, kind, fld, v, name, got))
            used.setdefault(path, {})[name] = got
    copies = []
    for path, names in (used.items() if paper_dir else ()):
        twin = os.path.join(paper_dir, os.path.basename(path))
        if os.path.abspath(twin) == os.path.abspath(path):
            continue
        if not os.path.exists(twin):
            copies.append(f'{_shown(twin)} not found: {len(names)} macros held against '
                          f'{_shown(path)} only')
            continue
        tm = _macros(twin)
        diff = [f'\\{n} = {tm.get(n)} in {_shown(twin)}, {g} in {_shown(path)}'
                for n, g in sorted(names.items()) if tm.get(n) != g]
        bad += diff
        if not diff:
            copies.append(f'{_shown(twin)} prints the same {len(names)} macros as {_shown(path)}')
    if bad:
        _fql_refuse('the macros disagree with the records or with the copies the manuscript '
                    'inputs:\n  ' + '\n  '.join(bad))
    return rows, copies


def _fql_face(kind):
    st = FQL_STYLE[kind]
    if st['open']:
        return dict(mfc='white', mec=st['color'], mew=FQL_MEW_OPEN)
    return dict(mfc=st['color'], mec='white', mew=FQL_MEW_FILLED)


def _fql_radius(kind):
    """Radius (pt) of the circle that holds a marker with its edge."""
    st = FQL_STYLE[kind]
    mew = FQL_MEW_OPEN if st['open'] else FQL_MEW_FILLED
    return _fqh_radius(st['marker'], st['ms'], mew)


def _fql_mark(ax, kind, pos, it, vertical):
    """One interval and its marker at position pos (x when vertical, y otherwise)."""
    st = FQL_STYLE[kind]
    a, b = ([pos, pos], [it['lo'], it['hi']]) if vertical else ([it['lo'], it['hi']], [pos, pos])
    ax.plot(a, b, color=st['color'], lw=FQL_CI_LW, solid_capstyle='butt', zorder=4)
    xy = ([pos], [it['mean']]) if vertical else ([it['mean']], [pos])
    ax.plot(*xy, st['marker'], ms=st['ms'], ls='none', zorder=5, **_fql_face(kind))


def _fql_check_marks(fig, ax, drawn, vertical):
    """Independent of the placement, in display points: every mark inside the axes, no two
    markers overlapping (the circles that hold them, FQL_CLEAR apart), no marker touching an
    interval other than its own, no two intervals touching. drawn: (label, kind, pos, mean, lo,
    hi). Returns the number of pairs checked."""
    k = 72.0 / fig.dpi
    x0, x1 = sorted(ax.get_xlim())
    y0, y1 = sorted(ax.get_ylim())
    circles, rects = [], []
    for label, kind, pos, m, lo, hi in drawn:
        p_ok = (x0 < pos < x1) if vertical else (y0 < pos < y1)
        v0, v1 = (y0, y1) if vertical else (x0, x1)
        if not (p_ok and v0 < lo and hi < v1):
            _fql_refuse(f'the mark {kind} of {label} leaves the axes')

        def pt(v):
            xy = (pos, v) if vertical else (v, pos)
            return ax.transData.transform(xy) * k
        c = pt(m)
        circles.append(((label, kind), c[0], c[1], _fql_radius(kind)))
        a, b = pt(lo), pt(hi)
        h = FQL_CI_LW / 2
        if vertical:
            rects.append(((label, kind), (c[0] - h, min(a[1], b[1]), c[0] + h, max(a[1], b[1]))))
        else:
            rects.append(((label, kind), (min(a[0], b[0]), c[1] - h, max(a[0], b[0]), c[1] + h)))
    n = 0
    for i, (ida, xa, ya, ra) in enumerate(circles):
        for idb, xb, yb, rb in circles[i + 1:]:
            n += 1
            if np.hypot(xa - xb, ya - yb) < ra + rb + FQL_CLEAR:
                _fql_refuse(f'the markers {ida[1]} of {ida[0]} and {idb[1]} of {idb[0]} overlap')
        for idb, (bx0, by0, bx1, by1) in rects:
            if idb == ida:
                continue
            n += 1
            dx = max(bx0 - xa, 0.0, xa - bx1)
            dy = max(by0 - ya, 0.0, ya - by1)
            if np.hypot(dx, dy) < ra:
                _fql_refuse(f'the marker {ida[1]} of {ida[0]} touches the interval {idb[1]} of '
                            f'{idb[0]}')
    for i, (ida, a) in enumerate(rects):
        for idb, b in rects[i + 1:]:
            n += 1
            if _fq4_hit(a, b):
                _fql_refuse(f'the intervals {ida[1]} of {ida[0]} and {idb[1]} of {idb[0]} touch')
    return n


def _fql_check_row(fig, texts, what, axis='x'):
    """Neighbouring labels of one row (axis x) or one column (axis y) at least FQL_TEXT_CLEAR pt
    apart (in a column the boxes of the texts hold the space of the descenders). A label is a
    text or a list of texts, its lines, taken together. Returns the smallest gap in pt."""
    from matplotlib.transforms import Bbox
    rend = fig.canvas.get_renderer()
    boxes = []
    for t in texts:
        parts = [p for p in (t if isinstance(t, (list, tuple)) else [t])
                 if p.get_visible() and p.get_text().strip()]
        if parts:
            boxes.append((Bbox.union([p.get_window_extent(rend) for p in parts]),
                          '\n'.join(p.get_text() for p in parts)))
    boxes = sorted(boxes, key=lambda b: b[0].x0 if axis == 'x' else b[0].y0)
    k = 72.0 / fig.dpi
    least = np.inf
    for (a, ta), (b, tb) in zip(boxes, boxes[1:]):
        gap = ((b.x0 - a.x1) if axis == 'x' else (b.y0 - a.y1)) * k
        least = min(least, gap)
        if gap < FQL_TEXT_CLEAR[axis]:
            _fql_refuse(f'the {what} {ta!r} and {tb!r} lie {gap:.1f} pt apart, less than '
                        f'{FQL_TEXT_CLEAR[axis]} pt')
    return least


def _fql_value_axis(X):
    """Range and ticks of the difference: every interval drawn and the 1% band, with a margin of
    5% of the span on each side."""
    vals = [MARGIN, -MARGIN]
    for it in [S[k] for S in X['sizes'] for k in FQL_KINDS] + \
            [I[k] for G in X['groups'] for I in G['items'] for k in FQL_KINDS]:
        vals += [it['lo'], it['hi']]
    lo, hi = min(vals), max(vals)
    pad = 0.05 * (hi - lo)
    lim = (lo - pad, hi + pad)
    return lim, _fq4_ticks(*lim, nbins=7)


def _fql_frame(ax, lim, ticks, vertical):
    """Band, grid lines, zero line, range and ticks of the value axis of one panel."""
    span, line = (ax.axhspan, ax.axhline) if vertical else (ax.axvspan, ax.axvline)
    span(-MARGIN, MARGIN, color=BAND_FILL, lw=0, zorder=0)
    for t in ticks:
        if abs(t) > 1e-9:
            line(t, color=C_GRID, lw=0.4, zorder=0.5)
    line(0, color=C_ZERO, lw=0.6, zorder=1)
    axis = ax.yaxis if vertical else ax.xaxis
    (ax.set_ylim if vertical else ax.set_xlim)(*lim)
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(FuncFormatter(lambda v, _: _fmt(round(v, 6))))


def _fql_handles(variant):
    """Legend entries: the two series (in variant A the marker with a short vertical interval, as
    the panels draw it; in variant B, whose panels draw intervals in two orientations, the marker
    alone) and the band."""
    from matplotlib.collections import LineCollection
    from matplotlib.container import ErrorbarContainer
    hs, labels = [], []
    for kind in FQL_KINDS:
        st = FQL_STYLE[kind]
        mk = Line2D([], [], ls='none', marker=st['marker'], ms=st['ms'], **_fql_face(kind))
        if variant == 'A':
            bar = LineCollection([], colors=[st['color']], linewidths=[FQL_CI_LW])
            mk = ErrorbarContainer((mk, (), (bar,)), has_xerr=False, has_yerr=True)
        hs.append(mk)
        labels.append(st['text'])
    hs.append(Patch(facecolor=BAND_FILL, edgecolor='none'))
    labels.append(FQL_MARGIN_TEXT)
    return hs, labels


def _fql_legend(fig, x_center, y_bottom, variant):
    """One legend for the figure in a thin box, one row of three entries below the panels."""
    from matplotlib.container import ErrorbarContainer
    from matplotlib.legend_handler import HandlerErrorbar
    hs, labels = _fql_handles(variant)
    leg = fig.legend(hs, labels, loc='lower center', bbox_to_anchor=(x_center, y_bottom),
                     ncol=len(hs), fontsize=FS_TICK, frameon=True, fancybox=False,
                     edgecolor='0.6', framealpha=1.0, handlelength=1.8, handletextpad=0.5,
                     columnspacing=1.6, borderpad=0.4,
                     handler_map={ErrorbarContainer: HandlerErrorbar(xerr_size=0.65)})
    leg.get_frame().set_linewidth(0.5)
    return leg


def _fql_size(fig, rend, text, fontsize=FS_TICK, rotation=0):
    """Width and height (in) of a text as the figure draws it."""
    t = fig.text(0, 0, text, fontsize=fontsize, rotation=rotation, linespacing=1.15,
                 multialignment='center')
    b = t.get_window_extent(rend)
    t.remove()
    return b.width / fig.dpi, b.height / fig.dpi


def _fql_ascent(fig, rend, line, fontsize=FS_TICK, linespacing=1.15):
    """Distance (in) from the baseline of a one-line text to the top of its box, as the figure
    lays the line out at this line spacing (the same as the first line of a longer text)."""
    t = fig.text(0, 0, line, fontsize=fontsize, va='baseline', linespacing=linespacing)
    top = t.get_window_extent(rend).y1
    t.remove()
    return (top - fig.transFigure.transform((0, 0))[1]) / fig.dpi


def _fql_dx_sizes():
    """FQL_DX_SIZES, after the check that the two markers of a size cannot overlap."""
    if 2 * FQL_DX_SIZES < _fql_radius('spec') + _fql_radius('rule') + FQL_CLEAR:
        _fql_refuse(f'the offset {FQL_DX_SIZES} pt of panel (a) lets the two markers of a size '
                    'overlap')
    return FQL_DX_SIZES


def _fql_positions(groups, gap):
    """Positions of the levels of panel (b), group by group, with gap positions between groups."""
    out, p = [], 0.0
    for i, G in enumerate(groups):
        if i:
            p += gap
        out.append([p + j for j in range(len(G['items']))])
        p += len(G['items'])
    return out


def _fql_draw_a(X, lim, ticks):
    """Variant A: one row, the y axis shared; panel (b) positions in groups with rotated value
    labels and one label per group below them. Returns (fig, axes, legend, marks, labels)."""
    fig = plt.figure(figsize=(FIG_W, FQL_A_H), dpi=300)
    rend = fig.canvas.get_renderer()
    # the group labels are drawn line by line, one line height (pitch) apart; the row of group
    # labels is as high as its longest label. FQL_A_H holds labels of FQL_A_LINES lines, and
    # each further line adds its height to the figure below the axes, so the two panels keep
    # their size.
    pitch = _fql_size(fig, rend, 'lp')[1]
    n_lines = max(len(G['a'].split('\n')) for G in X['groups'])
    gh = n_lines * pitch
    fig_h = FQL_A_H + max(0, n_lines - FQL_A_LINES) * pitch
    fig.set_size_inches(FIG_W, fig_h)
    rend = fig.canvas.get_renderer()
    left, right, gap, top, leg_y = 0.60, 0.04, 0.12, 0.25, 0.04
    wa = 1.00
    wb = FIG_W - left - wa - gap - right
    leg = _fql_legend(fig, (left + (FIG_W - left - right) / 2) / FIG_W, leg_y / fig_h, 'A')
    fig.canvas.draw()
    lh = leg.get_window_extent(rend).height / fig.dpi
    th = max(_fql_size(fig, rend, I['label'], rotation=90)[1]
             for G in X['groups'] for I in G['items'])
    tick_pad = 3.0 / 72
    y_group = leg_y + lh + 0.10 + gh                 # top of the row of group labels
    ax_y0 = y_group + 0.06 + th + tick_pad
    ax_h = fig_h - ax_y0 - top
    ax = fig.add_axes([left / FIG_W, ax_y0 / fig_h, wa / FIG_W, ax_h / fig_h])
    bx = fig.add_axes([(left + wa + gap) / FIG_W, ax_y0 / fig_h, wb / FIG_W, ax_h / fig_h],
                      sharey=ax)
    for a in (ax, bx):
        _fql_frame(a, lim, ticks, True)
        a.tick_params(axis='x', length=0, pad=3)
        a.tick_params(axis='y', length=3, pad=2)
    plt.setp(bx.get_yticklabels(), visible=False)

    # panel (a)
    sizes = X['sizes']
    ax.set_xlim(-0.6, len(sizes) - 0.4)
    dxa = _fql_dx_sizes() / (wa * 72.0 / (len(sizes) + 0.2))
    marks_a = []
    for j, S in enumerate(sizes):
        for kind, s in (('spec', -1), ('rule', 1)):
            it = S[kind]
            _fql_mark(ax, kind, j + s * dxa, it, True)
            marks_a.append((f'{S["modules"]} modules', kind, j + s * dxa, it['mean'], it['lo'],
                            it['hi']))
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([str(S['modules']) for S in sizes])
    ax.set_ylabel('makespan difference\nof the policy (%)', multialignment='center', labelpad=5)
    ax.set_title(FQL_TITLES[0], fontsize=FS_TITLE, pad=7)

    # panel (b)
    pos = _fql_positions(X['groups'], FQL_A_GAP)
    last = pos[-1][-1]
    bx.set_xlim(-0.6, last + 0.6)
    dxb = FQL_DX / (wb * 72.0 / (last + 1.2))
    marks_b, xt, xl = [], [], []
    for G, ps in zip(X['groups'], pos):
        for I, p in zip(G['items'], ps):
            for kind, s in (('spec', -1), ('rule', 1)):
                it = I[kind]
                _fql_mark(bx, kind, p + s * dxb, it, True)
                marks_b.append((f'{" ".join(G["b"])} {I["label"]}', kind, p + s * dxb, it['mean'],
                                it['lo'], it['hi']))
            xt.append(p)
            xl.append(I['label'])
    bx.set_xticks(xt)
    bx.set_xticklabels(xl, rotation=90, ha='center', va='top')
    bx.set_title(FQL_TITLES[1], fontsize=FS_TITLE, pad=7)

    # the group labels, one row below the value labels, and the size label of (a) on the same
    # row: line k of every group label stands on the baseline of row k, the rows one pitch
    # apart, so the first lines of all labels and the size label share one baseline and every
    # further line shares the baseline of its row (one text per line, each centred under its
    # group: matplotlib spaces the lines of one text by their glyphs, which leaves the lines of
    # a three-line label unevenly spaced)
    asc = max(_fql_ascent(fig, rend, G['a'].split('\n')[0]) for G in X['groups'])
    y_base = y_group - asc
    inv = fig.transFigure.inverted()
    labels = []
    for G, ps in zip(X['groups'], pos):
        xc = inv.transform(bx.transData.transform(((ps[0] + ps[-1]) / 2, 0)))[0]
        labels.append([fig.text(xc, (y_base - k * pitch) / fig_h, line, ha='center',
                                va='baseline', fontsize=FS_TICK)
                       for k, line in enumerate(G['a'].split('\n'))])
    size_label = fig.text((left + wa / 2) / FIG_W, y_base / fig_h, FQL_SIZE_LABEL,
                          ha='center', va='baseline', fontsize=FS_LABEL)
    return fig, (ax, bx), leg, ((ax, marks_a, True), (bx, marks_b, True)), \
        dict(groups=labels, size=size_label, ticks=bx.get_xticklabels(), fig_h=fig_h,
             widths=(wa, wb), ax_h=ax_h)


def _fql_draw_b(X, lim, ticks):
    """Variant B: panel (a) as in variant A, panel (b) a forest plot with one row per level, the
    group label and the value as row labels on its left. Returns (fig, axes, legend, marks,
    labels)."""
    fig_h = FQL_B_H
    fig = plt.figure(figsize=(FIG_W, fig_h), dpi=300)
    rend = fig.canvas.get_renderer()
    left, right, gap, top, leg_y = 0.64, 0.06, 0.26, 0.25, 0.04
    wa = 1.18
    col_gap, val_pad = 7.0 / 72, 4.0 / 72
    gw = max(_fql_size(fig, rend, ln)[0] for G in X['groups'] for ln in G['b'])
    vw = max(_fql_size(fig, rend, I['label'])[0] for G in X['groups'] for I in G['items'])
    lab_w = gw + col_gap + vw + val_pad
    wb = FIG_W - left - wa - gap - lab_w - right
    xb0 = left + wa + gap + lab_w
    leg = _fql_legend(fig, (left + (FIG_W - left - right) / 2) / FIG_W, leg_y / fig_h, 'B')
    fig.canvas.draw()
    lh = leg.get_window_extent(rend).height / fig.dpi
    ax_y0 = leg_y + lh + 0.46
    ax_h = fig_h - ax_y0 - top
    ax = fig.add_axes([left / FIG_W, ax_y0 / fig_h, wa / FIG_W, ax_h / fig_h])
    bx = fig.add_axes([xb0 / FIG_W, ax_y0 / fig_h, wb / FIG_W, ax_h / fig_h])
    _fql_frame(ax, lim, ticks, True)
    _fql_frame(bx, lim, ticks, False)
    ax.tick_params(axis='x', length=0, pad=3)
    ax.tick_params(axis='y', length=3, pad=2)
    bx.tick_params(axis='x', length=3, pad=2)
    bx.tick_params(axis='y', length=0)
    bx.spines['left'].set_visible(False)

    # panel (a)
    sizes = X['sizes']
    ax.set_xlim(-0.6, len(sizes) - 0.4)
    dxa = _fql_dx_sizes() / (wa * 72.0 / (len(sizes) + 0.2))
    marks_a = []
    for j, S in enumerate(sizes):
        for kind, s in (('spec', -1), ('rule', 1)):
            it = S[kind]
            _fql_mark(ax, kind, j + s * dxa, it, True)
            marks_a.append((f'{S["modules"]} modules', kind, j + s * dxa, it['mean'], it['lo'],
                            it['hi']))
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([str(S['modules']) for S in sizes])
    ax.set_xlabel(FQL_SIZE_LABEL, labelpad=5)
    ax.set_ylabel('makespan difference\nof the policy (%)', multialignment='center', labelpad=5)
    ax.set_title(FQL_TITLES[0], fontsize=FS_TITLE, pad=7)

    # panel (b): rows top to bottom in the order of the table
    pos = _fql_positions(X['groups'], FQL_B_GAP)
    last = pos[-1][-1]
    bx.set_ylim(last + 0.7, -0.7)
    dyb = FQL_DX / (ax_h * 72.0 / (last + 1.4))
    marks_b = []
    # row labels stand on a baseline half a digit height below the row, so that every label is
    # centred on its row whatever its glyphs
    from matplotlib.font_manager import FontProperties
    h0, d0 = rend.get_text_width_height_descent('0', FontProperties(size=FS_TICK),
                                                ismath=False)[1:]
    tr = mtransforms.blended_transform_factory(fig.transFigure, bx.transData) + \
        mtransforms.ScaledTranslation(0, -(h0 - d0) / 2 / fig.dpi, fig.dpi_scale_trans)
    groups_t, values_t = [], []
    for G, ps in zip(X['groups'], pos):
        for k, (I, p) in enumerate(zip(G['items'], ps)):
            for kind, s in (('spec', -1), ('rule', 1)):
                it = I[kind]
                _fql_mark(bx, kind, p + s * dyb, it, False)
                marks_b.append((f'{" ".join(G["b"])} {I["label"]}', kind, p + s * dyb, it['mean'],
                                it['lo'], it['hi']))
            values_t.append(bx.text((xb0 - val_pad) / FIG_W, p, I['label'], transform=tr,
                                    ha='right', va='baseline', fontsize=FS_TICK))
            if k < len(G['b']):
                groups_t.append(bx.text((xb0 - lab_w) / FIG_W, p, G['b'][k], transform=tr,
                                        ha='left', va='baseline', fontsize=FS_TICK))
    bx.set_yticks([])
    bx.set_xlabel(FQL_VALUE_LABEL, labelpad=5)
    bx.set_title(FQL_TITLES[1], fontsize=FS_TITLE, pad=7)
    return fig, (ax, bx), leg, ((ax, marks_a, True), (bx, marks_b, False)), \
        dict(groups=groups_t, values=values_t, fig_h=fig_h, widths=(wa, wb), ax_h=ax_h,
             label_w=lab_w)


def fq4_limits(family_json, family_macros, perfloor_json, perfloor_macros, prod_json, prod_macros,
               sens_json, sens_macros, out_dir, variant='A', partial=False, allow_partial=False,
               paper_dir=FQL_PAPER_DIR):
    """Draw the figure of the policy across shop sizes and benchmark parameters after its checks
    and gates (fql_inputs), then print the plotted values against the macros. Full mode:
    fq4_limits_<variant>.pdf/.png; partial mode: fq4_limits_<variant>_partial.*; a record of a dry
    or partial run drawn with allow_partial adds _dev. paper_dir: the folder of the macro files
    the manuscript inputs, whose copies must print what the run's macro files print (None: not
    compared)."""
    if variant not in FQL_VARIANTS:
        _fql_refuse(f'variant {variant!r} is not one of {FQL_VARIANTS}')
    X = fql_inputs(family_json, family_macros, perfloor_json, perfloor_macros, prod_json,
                   prod_macros, sens_json, sens_macros, partial, allow_partial, paper_dir)
    print('  fq4_limits records: ' + '; '.join(X['sources'].values()))
    for m, why in X['absent']:
        print(f'    left out: {m} modules ({why})')
    lim, ticks = _fql_value_axis(X)
    draw = _fql_draw_a if variant == 'A' else _fql_draw_b
    fig, axes, leg, marks, L = draw(X, lim, ticks)

    n_text = _check_layout(fig, axes, leg, tag='fq4_limits')
    rend = fig.canvas.get_renderer()
    if variant == 'A':
        least_g = _fql_check_row(fig, L['groups'] + [L['size']], 'group labels')
        least_t = _fql_check_row(fig, L['ticks'], 'value labels')
        for lines in L['groups']:              # a group label stays below the value labels
            if any(t.get_window_extent(rend).y1 > v.get_window_extent(rend).y0 - 1.0
                   for t in lines for v in L['ticks'] if v.get_text()):
                _fql_refuse(f'the group label {" ".join(t.get_text() for t in lines)!r} '
                            'touches a value label')
        clear = f'labels at least {min(least_g, least_t):.1f} pt apart'
    else:
        least = min(_fql_check_row(fig, L['values'], 'value labels', axis='y'),
                    _fql_check_row(fig, L['groups'], 'group labels', axis='y'))
        right = max(t.get_window_extent(rend).x1 for t in L['groups'])
        if right > min(t.get_window_extent(rend).x0 for t in L['values']) - 1.0:
            _fql_refuse('a group label of panel (b) reaches the column of the values')
        clear = f'row labels at least {least:.1f} pt apart'
    n_pairs = sum(_fql_check_marks(fig, a, d, v) for a, d, v in marks)

    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f'fq4_limits_{variant}' + ('_partial' if partial else '')
                        + ('_dev' if X['dev'] else ''))
    fig.savefig(stem + '.pdf')
    fig.savefig(stem + '.png', dpi=300)
    plt.close(fig)
    n_marks = sum(len(d) for _, d, _ in marks)
    print(f'  fq4_limits {variant}{" (partial)" if partial else ""}: {FIG_W:.2f} x '
          f'{L["fig_h"]:.2f} in, panels {L["widths"][0]:.2f} and {L["widths"][1]:.2f} in wide, '
          f'axes {L["ax_h"]:.2f} in high, difference {lim[0]:+.2f} to {lim[1]:+.2f} %, '
          f'{len(X["sizes"])} sizes and {sum(len(G["items"]) for G in X["groups"])} levels, '
          f'{n_marks} marks, {n_pairs} pairs of marks clear of each other, {n_text} texts '
          f'checked, no overlap, {clear}')

    # ---- the plotted values against the macros
    print(f'\n  {"panel":5s} {"position":37s} {"series":33s} {"mean":>7s} {"95% interval":>17s}'
          '   macro (mean, Lo, Hi)')
    got = {}
    for panel, label, kind, fld, v, name, printed in X['table']:
        got.setdefault((panel, label, kind), []).append((fld, name, printed))
    rows = [('a', f'{S["modules"]} modules', S) for S in X['sizes']]
    rows += [('b', f'{" ".join(G["b"])} {I["label"]}', I) for G in X['groups'] for I in G['items']]
    for panel, label, D in rows:
        for kind in FQL_KINDS:
            it = D[kind]
            ms = got.get((panel, label, kind), [])
            stems = sorted({n for f, n, _ in ms if f == 'mean'})
            m = '; '.join(f'\\{s} ' + ', '.join(p for f, n, p in ms if n.startswith(s)
                                                 and n[len(s):] in ('', 'Lo', 'Hi'))
                          for s in stems)
            print(f'  ({panel})   {label:37s} {FQL_STYLE[kind]["text"]:33s} {it["mean"]:+7.2f} '
                  f'[{it["lo"]:+.2f}, {it["hi"]:+.2f}]   {m}')
    for S in X['sizes']:
        print(f'  sources at {S["modules"]} modules: {S["src"]["spec"]}; {S["src"]["rule"]}')
    print(f'\n  macro gate: {len(X["table"])} of {len(X["table"])} plotted means and interval '
          'ends match their macros')
    for line in X['copies']:
        print(f'  manuscript copy: {line}')
    if X['absent']:
        print('  sizes missing from panel (a): '
              + ', '.join(f'{m} modules' for m, _ in X['absent']))
    return [stem + '.pdf', stem + '.png']


def assert_times_only(paths):
    allowed = _SERIF.replace(' ', '')
    bad = []
    for pdf in paths:
        if not pdf.endswith('.pdf'):
            continue
        try:
            out = subprocess.run(['pdffonts', pdf], capture_output=True, text=True,
                                 check=True).stdout
        except (OSError, subprocess.CalledProcessError):
            print('  font check skipped (pdffonts unavailable)')
            return
        for line in out.splitlines()[2:]:
            name = line.split()[0].split('+')[-1] if line.split() else ''
            if name and not name.startswith(allowed):
                bad.append(f'{os.path.basename(pdf)}: {name}')
    if bad:
        raise SystemExit('NON-TIMES FONT IN A FIGURE:\n  ' + '\n  '.join(bad))
    print(f'  font check: every figure PDF embeds {_SERIF} only')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--only', choices=['fq1', 'fq3', 'fq4', 'fs_convergence', 'fq2_reuse',
                                       'fq2_heldout', 'fq5_rule_change', 'fq4_limits'],
                    default=None)
    ap.add_argument('--factory', default=None,
                    help='factory inference JSON (default: arm_registry.out of '
                         'results/family_inference.json for the active setting)')
    ap.add_argument('--factory-macros', default=None,
                    help='macros of the same run (default: arm_registry.out of '
                         'paper/macros_family.tex)')
    ap.add_argument('--factory-seeds', choices=list(FQ1_SEED_MODES), default='auto',
                    help='fq1: the training seeds behind the policy rows of panel (a): '
                         'five (the factory JSON), ten (the ten-seed record of '
                         'p110_seed_extension.py, refused when it or its macro file is '
                         'missing), auto (ten when the ten-seed record exists, five '
                         'otherwise); the control rows of variant B come from the factory '
                         'JSON in every mode')
    ap.add_argument('--seedext', default=None,
                    help='fq1: the ten-seed record (default: arm_registry.out of '
                         'results/seed_extension.json)')
    ap.add_argument('--seedext-macros', default=None,
                    help='fq1: the macro file of the ten-seed record (default: '
                         'arm_registry.out of manuscript/macros_seedext.tex)')
    ap.add_argument('--public-family', choices=list(FQ1_PUBLIC), default='active',
                    help='fq1: the public family of p89: active (default; '
                         'results/public_family_act.json, manuscript/macros_pubfam_act.tex) '
                         'or nondelay (results/public_family_inference.json, '
                         'manuscript/macros_public.tex)')
    ap.add_argument('--public', default=None,
                    help='public inference JSON (default: the record of --public-family)')
    ap.add_argument('--public-set', default='pub_SD2_test_10x5+mix')
    ap.add_argument('--public-macros', default=None,
                    help='default: the macro file of --public-family')
    ap.add_argument('--timed-curves', default=None,
                    help='fq3: the CSV of p94 (default: results/timed_curves.csv)')
    ap.add_argument('--timed-json', default=None,
                    help='fq3: the JSON of the same p94 run (default: '
                         'results/timed_inference.json)')
    ap.add_argument('--policy-time', nargs='+', default=None,
                    help="fq3, start clock only: records of the policy's time for one "
                         'schedule (the p75 record with its entries per size; default: '
                         'arm_registry.out of results/deploy_cost_cpu_1thread.json); the '
                         "request clock marks the policy's first schedule from the CSV")
    ap.add_argument('--rulecmp-public', default=None,
                    help='fq4: the public JSON of p93 (default: results/rulecmp_public.json)')
    ap.add_argument('--rulecmp-factory', default=None,
                    help='fq4: the factory JSON of p93 (default: '
                         'results/rulecmp_factory.json; left out when absent)')
    ap.add_argument('--rulecmp-public-macros', default=None,
                    help='fq4: default: manuscript/macros_rulecmp_public.tex')
    ap.add_argument('--rulecmp-factory-macros', default=None,
                    help='fq4: default: manuscript/macros_rulecmp_factory.tex')
    ap.add_argument('--sensitivity', default=None,
                    help='fq4: the CSV of p87 infer, with its JSON beside it (default: '
                         'results/sensitivity_inference.csv; left out when absent)')
    ap.add_argument('--sensitivity-macros', default=None,
                    help='fq4: default: manuscript/macros_sensitivity.tex')
    ap.add_argument('--fq3-group', choices=list(FQ3_GROUPS), default='all',
                    help='fq3: the pooled rows the curves come from: all (Pool, the eight '
                         'combinations), no_setups (N, L, W, LW) or setups (S, LS, SW, LSW); '
                         'the gate reads the JSON entries of the same group')
    ap.add_argument('--memetic-parallel', default=None,
                    help='fq3, request clock: the record of p124_memetic_parallel.py '
                         '(results/memetic_parallel.json under PACT_OUT); variant A and the '
                         'supplement companion draw the memetic algorithm of row Sel as the best '
                         'of eight runs from the curves file the record names beside it '
                         '(default: none, the single run of the timed CSV)')
    ap.add_argument('--memetic-minute', default=None,
                    help='fq3, with --memetic-parallel only: the record of '
                         'p137_memetic_one_minute.py (results/memetic_one_minute.json under '
                         'PACT_OUT); variant A gains, at 60 s in each panel, the memetic '
                         'algorithm with a one-minute budget from each start as a circle with '
                         'its 95%% interval (default: none)')
    ap.add_argument('--memetic-five', default=None,
                    help='fq3, with --memetic-minute only: the record of '
                         'p137_memetic_one_minute.py --budget 300 '
                         '(results/memetic_five_minute.json under PACT_OUT); variant A gains, at '
                         '300 s in each panel, the memetic algorithm with a five-minute budget '
                         'from each start as the same circles with their 95%% intervals, and the '
                         'legend row of the one-minute budget names both budgets; a preview or '
                         'a dry run of p137 is refused (default: none)')
    ap.add_argument('--fq4-codes', choices=['ab', 'b'], default='ab',
                    help='fq4: the panels whose points carry their combination codes')
    ap.add_argument('--convergence', default=None,
                    help='fs_convergence: the record of p116 (default: arm_registry.out of '
                         f'{FS_CONV_JSON})')
    ap.add_argument('--convergence-macros', default=None,
                    help='fs_convergence: the macro file of the same run (default: '
                         f'arm_registry.out of {FS_CONV_MACROS})')
    ap.add_argument('--crosseval', default=None,
                    help='fq2_reuse: the record of p100 (default: arm_registry.out of '
                         f'{FQ2_JSON})')
    ap.add_argument('--crosseval-macros', default=None,
                    help='fq2_reuse: the macro file of the same run (default: '
                         f'arm_registry.out of {FQ2_MACROS})')
    ap.add_argument('--heldout-splits', default=None,
                    help='fq2_heldout: the record of p122 (default: arm_registry.out of '
                         f'{FQH_SPLITS_JSON})')
    ap.add_argument('--heldout-splits-macros', default=None,
                    help='fq2_heldout: the macro file of the same run (default: '
                         f'arm_registry.out of {FQH_SPLITS_MACROS})')
    ap.add_argument('--secondary', default=None,
                    help=f'fq2_heldout: the record of p95 (default: {FQH_SEC_JSON}, the '
                         'repository path p95 writes)')
    ap.add_argument('--secondary-macros', default=None,
                    help=f'fq2_heldout: the macro file of the same run (default: {FQH_SEC_MACROS})')
    ap.add_argument('--rule-change', default=None,
                    help='fq5_rule_change: the record of p99_rule_change_cost.py infer (default: '
                         f'arm_registry.out of {FQ5_JSON})')
    ap.add_argument('--rule-change-macros', default=None,
                    help='fq5_rule_change: the macro file of the same run (default: '
                         f'arm_registry.out of {FQ5_MACROS})')
    ap.add_argument('--deploy', default=None,
                    help='fq5_rule_change: the record of p75_deploy_cost.py, re-planning time per '
                         f'shop size (default: arm_registry.out of {FQ5_DEPLOY_JSON})')
    ap.add_argument('--deploy-macros', default=None,
                    help='fq5_rule_change: the macro file p75 wrote from that record (default: '
                         f'arm_registry.out of {FQ5_DEPLOY_MACROS})')
    ap.add_argument('--heldout-partial', action='store_true',
                    help='fq2_heldout: draw the models whose records exist, list the absent '
                         'ones, and write fq2_heldout_partial.* instead of fq2_heldout.*')
    ap.add_argument('--per-floor', default=None,
                    help='fq4_limits: the record of p118_per_floor.py (default: arm_registry.out '
                         f'of {FQL_PERFLOOR_JSON})')
    ap.add_argument('--per-floor-macros', default=None,
                    help='fq4_limits: the macro file of the same run (default: arm_registry.out '
                         f'of {FQL_PERFLOOR_MACROS})')
    ap.add_argument('--prodspec', default=None,
                    help='fq4_limits: the record of p128_production_specialists.py (default: '
                         f'arm_registry.out of {FQL_PROD_JSON})')
    ap.add_argument('--prodspec-macros', default=None,
                    help='fq4_limits: the macro file of the same run (default: arm_registry.out '
                         f'of {FQL_PROD_MACROS})')
    ap.add_argument('--fq4-limits-variant', choices=list(FQL_VARIANTS), default='A',
                    help='fq4_limits: A, one row with the y axis shared; B, panel (b) as a forest '
                         'plot with one row per level')
    ap.add_argument('--fq4-limits-partial', action='store_true',
                    help='fq4_limits: draw panel (a) without the sizes whose record is absent, '
                         'list them, and write fq4_limits_<variant>_partial.*')
    ap.add_argument('--allow-partial', action='store_true',
                    help='fq1: draw a ten-seed record of a dry run of p110; fq3: draw a '
                         'CSV whose run is not full, or a record of p124 of a dry run or a '
                         'preview (file names end in _preview); fq4: draw files of a partial run of '
                         'p93 or p87; '
                         'fq2_heldout: draw a record of a dry run of p122 or p110; '
                         'file names end in _dev')
    ap.add_argument('--out', default=None, help=f'default: {OUT_DEFAULT}')
    a = ap.parse_args()

    # A path given on the command line is taken from the current folder; a
    # default is a repository path.
    def path(given, default):
        if given is not None:
            return os.path.abspath(given)
        return default if os.path.isabs(default) else os.path.join(ROOT, default)

    given = {k: getattr(a, k) for k in ('factory', 'factory_macros', 'seedext',
                                         'seedext_macros', 'public',
                                         'public_macros', 'out', 'timed_curves',
                                         'timed_json', 'rulecmp_public', 'rulecmp_factory',
                                         'rulecmp_public_macros', 'rulecmp_factory_macros',
                                         'sensitivity', 'sensitivity_macros', 'convergence',
                                         'convergence_macros', 'crosseval',
                                         'crosseval_macros', 'heldout_splits',
                                         'heldout_splits_macros', 'secondary',
                                         'secondary_macros')}
    given = {k: (None if v is None else os.path.abspath(v)) for k, v in given.items()}
    ptime = None if a.policy_time is None else [os.path.abspath(p) for p in a.policy_time]
    mpar = None if a.memetic_parallel is None else os.path.abspath(a.memetic_parallel)
    mmin = None if a.memetic_minute is None else os.path.abspath(a.memetic_minute)
    mfiv = None if a.memetic_five is None else os.path.abspath(a.memetic_five)
    fq5_given = {k: (None if getattr(a, k) is None else os.path.abspath(getattr(a, k)))
                 for k in ('rule_change', 'rule_change_macros', 'deploy', 'deploy_macros')}
    fql_given = {k: (None if getattr(a, k) is None else os.path.abspath(getattr(a, k)))
                 for k in ('per_floor', 'per_floor_macros', 'prodspec', 'prodspec_macros')}
    os.chdir(ROOT)
    factory = given['factory'] or path(None, REG.out('results/family_inference.json'))
    fmac = given['factory_macros'] or path(None, REG.out('paper/macros_family.tex'))
    public = path(given['public'], FQ1_PUBLIC[a.public_family][0])
    pmac = path(given['public_macros'], FQ1_PUBLIC[a.public_family][1])
    out = path(given['out'], OUT_DEFAULT)
    print(f'setting: {REG.describe()}')
    written = []
    if a.only in (None, 'fq1'):
        sx = given['seedext'] or path(None, REG.out('results/seed_extension.json'))
        sxm = given['seedext_macros']
        if sxm is None and fq1_seed_mode(a.factory_seeds, sx) == 'ten':
            sxm = path(None, REG.out('manuscript/macros_seedext.tex'))
        written += fq1_specialists(factory, public, a.public_set, out, fmac, pmac,
                                   a.factory_seeds, sx, sxm, a.public_family,
                                   a.allow_partial)
    if a.only in (None, 'fq3'):
        tc = path(given['timed_curves'], 'results/timed_curves.csv')
        tj = path(given['timed_json'], 'results/timed_inference.json')
        if a.only is None and not os.path.exists(tc):
            print(f'  fq3 skipped: {tc} not found')
        else:
            pt = ptime or [path(None, REG.out('results/deploy_cost_cpu_1thread.json'))]
            written += fq3_time_budget(tc, tj, pt, out, a.allow_partial, a.fq3_group, mpar,
                                       mmin, mfiv)
    if a.only in (None, 'fq4'):
        rp = path(given['rulecmp_public'], 'results/rulecmp_public.json')
        if a.only is None and not os.path.exists(rp):
            print(f'  fq4 skipped: {rp} not found')
        else:
            written += fq4_boundary(
                rp, path(given['rulecmp_factory'], 'results/rulecmp_factory.json'),
                path(given['sensitivity'], 'results/sensitivity_inference.csv'), out,
                path(given['rulecmp_public_macros'], 'manuscript/macros_rulecmp_public.tex'),
                path(given['rulecmp_factory_macros'], 'manuscript/macros_rulecmp_factory.tex'),
                path(given['sensitivity_macros'], 'manuscript/macros_sensitivity.tex'),
                a.allow_partial, a.fq4_codes)
    if a.only in (None, 'fs_convergence'):
        cj = given['convergence'] or path(None, REG.out(FS_CONV_JSON))
        cm = given['convergence_macros'] or path(None, REG.out(FS_CONV_MACROS))
        if a.only is None and not os.path.exists(cj):
            print(f'  fs_convergence skipped: {cj} not found')
        else:
            written += fs_convergence(cj, cm, out)
    if a.only == 'fq2_reuse':                     # drawn only when asked for
        xj = given['crosseval'] or path(None, REG.out(FQ2_JSON))
        xm = given['crosseval_macros'] or path(None, REG.out(FQ2_MACROS))
        written += fq2_reuse(xj, xm, out)
    if a.only == 'fq2_heldout':                   # drawn only when asked for
        sx = given['seedext'] or path(None, REG.out('results/seed_extension.json'))
        sxm = given['seedext_macros']
        if sxm is None and fq1_seed_mode(a.factory_seeds, sx) == 'ten':
            sxm = path(None, REG.out('manuscript/macros_seedext.tex'))
        hj = given['heldout_splits'] or path(None, REG.out(FQH_SPLITS_JSON))
        hm = given['heldout_splits_macros'] or path(None, REG.out(FQH_SPLITS_MACROS))
        written += fq2_heldout(factory, fmac, a.factory_seeds, sx, sxm, hj, hm,
                               path(given['secondary'], FQH_SEC_JSON),
                               path(given['secondary_macros'], FQH_SEC_MACROS), out,
                               a.heldout_partial, a.allow_partial)
    if a.only == 'fq5_rule_change':               # drawn only when asked for
        written += fq5_rule_change(
            fq5_given['rule_change'] or path(None, REG.out(FQ5_JSON)),
            fq5_given['rule_change_macros'] or path(None, REG.out(FQ5_MACROS)),
            fq5_given['deploy'] or path(None, REG.out(FQ5_DEPLOY_JSON)),
            fq5_given['deploy_macros'] or path(None, REG.out(FQ5_DEPLOY_MACROS)), out)
    if a.only == 'fq4_limits':                    # drawn only when asked for
        written += fq4_limits(
            factory, fmac,
            fql_given['per_floor'] or path(None, REG.out(FQL_PERFLOOR_JSON)),
            fql_given['per_floor_macros'] or path(None, REG.out(FQL_PERFLOOR_MACROS)),
            fql_given['prodspec'] or path(None, REG.out(FQL_PROD_JSON)),
            fql_given['prodspec_macros'] or path(None, REG.out(FQL_PROD_MACROS)),
            fql_sens_json(path(given['sensitivity'], FQL_SENS_CSV)),
            path(given['sensitivity_macros'], FQL_SENS_MACROS), out, a.fq4_limits_variant,
            a.fq4_limits_partial, a.allow_partial)
    assert_times_only(written)
    for w in written:
        print(f'  wrote {os.path.relpath(w, ROOT)}')


if __name__ == '__main__':
    main()
