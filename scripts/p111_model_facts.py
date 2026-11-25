r"""P111: counts and file facts that the text states, and a check of the design constants typed by
hand in the macro files (exact environment).

COUNTS AND FACTS (macro file manuscript/macros_modelfacts.tex)
  BlindParams         parameters of the policy network, read from the line
                      '[model] params total=' of the training log of every principal run (the
                      policy, the control and the specialists of the active seeds; the runs of
                      the second five seeds of scripts/seed_extension.py whose logs exist) and
                      required equal across them
  DescriptorParams    parameters of the variant with descriptor input (registry arm
                      xmask-film, seeds 301 to 303), from the training logs on disk
  HgnnNetworkParams   parameters of the published heterogeneous-graph scheduler (HGNN) trained
                      under the non-delay mask (runs hgnn-<reward>-<part>-s<seed>-xmask of
                      scripts/eval_newseeds_plan.py), from the training logs on disk
  PolicyCheckpointKB  size of the checkpoint file of the selected policy run
                      (arm_registry.select_seed), in KiB (bytes / 1024, no decimals)
  OpDurMedian         median processing time in hours over every compatible (operation,
                      machine) pair of the base instances of the factory test set at 10x25 (the
                      base instances the descriptors of data/FAMILY/10x25 name; the definition
                      of scripts/p28_exposure_and_certification.py), no decimals
  A count whose runs are not on disk is left out of the macro file and named in the output;
  a count that differs across the runs on disk stops the script (exit 1, nothing written).
  Parameter counts print as plain digits (the text groups them with \grp).

CONSTANTS CHECK
  Every macro of CONSTANTS is looked up in CONSTANT_FILES (manuscript/macros_design.tex, the
  constants typed by hand, one comment per macro naming its source); each definition found
  must equal the value in force, read from the stored configurations of the runs on disk
  (updates per run, parallel training episodes per update, training episodes per training
  combination as max_updates x num_envs over the combinations of family_regimes, rounded;
  the batch size and parameter count of the published HGNN scheduler, runs
  hgnn-<reward>-joint-s<seed>-xmask), from the base instances of the data folders (stations
  and modules) and from scripts/timing_validity.py (the minimum CPU share of a valid timed
  record). A difference, or a name defined in none of the files, stops the script (exit 1,
  nothing written). Values compare after removing thousands separators.

OUTPUTS (arm_registry.out under PACT_OUT)
  manuscript/macros_modelfacts.tex   macro_io.emit
  results/model_facts.json          every count with its runs, every constant with its
                                    definitions and sources

CHECK (--check, read only)
  exit 0 when the inputs exist: the training log of every principal run of the active seeds,
  their configurations, the configurations of the longer-trained policy and of the published
  HGNN scheduler, the checkpoint of the selected policy run, the test descriptors and the
  files of CONSTANT_FILES.

Usage: PACT_ENV=exact PACT_OUT=out/exact python scripts/p111_model_facts.py
       python scripts/p111_model_facts.py --check
"""
import argparse
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import numpy as np  # noqa: E402

import arm_registry as REG  # noqa: E402
from macro_io import emit  # noqa: E402

ENV = 'exact'
OUT_FILES = {'macros': 'manuscript/macros_modelfacts.tex', 'json': 'results/model_facts.json'}
LOG_PARAMS = re.compile(r'\[model\] params total=([0-9]+)')
FILM_STEM = 'xmask-film'
EQ_STEM = 'xmask-eq'
HGNN_GLOB = 'hgnn-*-xmask*'
HGNN_PUBLISHED_GLOB = 'hgnn-*-joint-s*-xmask'     # the published scheduler, one run per seed
TEST_FOLDER = 'data/FAMILY/10x25'
SIZES = {'10x25': 'data/FAMILY/10x25', '20x25': 'data/FAMILY/20x25',
         '30x25': 'data/FAMILY/30x25'}
# names that scripts/p28_exposure_and_certification.py also writes (manuscript/macros_exposure.tex,
# the only other file that defines them); every other name must be new under manuscript/
KEPT = {'BlindParams': 'manuscript/macros_exposure.tex',
        'OpDurMedian': 'manuscript/macros_exposure.tex'}
CONSTANT_FILES = ('manuscript/macros_design.tex',)
# (macro, meaning, key of the value in force); every macro that manuscript/macros_design.tex
# defines, in its order
CONSTANTS = (
    ('ExpEnvs', 'parallel training episodes per update', 'envs'),
    ('ExpSpecUpdates', 'updates of a specialist', 'specialist_updates'),
    ('ExpEqUpdates', 'updates of the longer-trained policy', 'eq_updates'),
    ('ExpRelUpdates', 'updates of the policy', 'policy_updates'),
    ('ExpSpecPerRegime', 'training episodes of a specialist on its combination',
     'specialist_episodes'),
    ('ExpEqPerRegime', 'training episodes of the longer-trained policy per combination',
     'eq_episodes'),
    ('ExpRelPerRegime', 'training episodes of the policy per training combination',
     'policy_episodes'),
    ('LoadTenJobs', 'modules at the training size', 'jobs_10x25'),
    ('LoadStations', 'stations per shop', 'stations'),
    ('HgnnBatch', 'instances per iteration of the published HGNN scheduler', 'hgnn_batch'),
    ('HgnnParams', 'parameters of the published HGNN scheduler', 'hgnn_params'),
    ('CpuShareMin', 'minimum CPU share of a valid timed record', 'share_min'),
)


class Refused(Exception):
    """An input or a value that this script does not accept."""


# ----------------------------------------------------------------- layout
def _layout(path_of):
    """Path template of the registry (with the run name as {m}) for runs of other arms."""
    m = REG.model('released', None, 301, ENV)
    p = path_of('released', None, 301, ENV)
    if p.count(m) != 1:
        raise Refused(f'cannot derive a path layout from {p!r}')
    return p.replace(m, '{m}')


def principal_runs():
    """(name, log, config) of every principal run of the active seeds, then of the second five
    seeds whose training logs exist."""
    out = []
    for s in REG.seeds():
        for arm in REG.ARMS:
            for c in (REG.COMBOS if arm == 'specialist' else [None]):
                out.append((REG.model(arm, c, s, ENV), REG.train_log(arm, c, s, ENV),
                            REG.config_path(arm, c, s, ENV), arm))
    import seed_extension as SE
    for a, c, s in SE.runs():
        if s in REG.seeds():
            continue
        lg = SE.train_log(a, c, s)
        if os.path.exists(lg):
            out.append((SE.name(a, c, s), lg, SE.config_path(a, c, s), a))
    return out


def secondary_runs(stem):
    cfg = _layout(REG.config_path)
    return [(REG.secondary_model(stem, s), REG.secondary_paths(stem, s)[1],
             cfg.format(m=REG.secondary_model(stem, s))) for s in REG.SECONDARY_SEEDS]


def hgnn_logs():
    tmpl = _layout(REG.train_log)
    return sorted(glob.glob(tmpl.format(m=HGNN_GLOB)))


def hgnn_configs():
    """Stored configurations of the published HGNN scheduler (one run per seed)."""
    return sorted(glob.glob(_layout(REG.config_path).format(m=HGNN_PUBLISHED_GLOB)))


# ----------------------------------------------------------------- counts
def log_params(path):
    """The parameter count of a training log; None without the line; refused when one log
    holds two different counts."""
    found = set(LOG_PARAMS.findall(open(path, errors='replace').read()))
    if len(found) > 1:
        raise Refused(f'{path} reports more than one parameter count: {sorted(found)}')
    return int(found.pop()) if found else None


def count(logs, what, required):
    """dict(value, runs, missing): one parameter count over the logs on disk. required: a
    missing log or line is refused; otherwise it is named and left out."""
    runs, missing = {}, []
    for p in logs:
        if not os.path.exists(p):
            missing.append(f'no training log {p}')
            continue
        v = log_params(p)
        if v is None:
            missing.append(f'{p} has no line [model] params total=')
            continue
        runs[p] = v
    if required and missing:
        raise Refused(f'{what}: {missing[0]} ({len(missing)} of {len(logs)} logs)')
    vals = sorted(set(runs.values()))
    if len(vals) > 1:
        by = {v: [p for p, x in runs.items() if x == v] for v in vals}
        raise Refused(f'{what}: the logs on disk disagree: '
                      + '; '.join(f'{v} in {len(ps)} (first {ps[0]})' for v, ps in by.items()))
    return dict(value=vals[0] if vals else None, runs=runs, missing=missing)


def checkpoint_kib():
    seed, scores = REG.select_seed('released')
    p = REG.checkpoint('released', None, seed, ENV)
    if not os.path.exists(p):
        raise Refused(f'the checkpoint of the selected policy run {p} does not exist')
    return dict(seed=seed, path=p, bytes=os.path.getsize(p),
                kib=os.path.getsize(p) / 1024.0)


# ----------------------------------------------------------------- data
def fjs_times(path):
    """Header (jobs, machines) and every compatible (operation, machine) processing time of
    one .fjs file (the parsing of scripts/p28_exposure_and_certification.py)."""
    t = open(path).read().split()
    nj, nm = int(t[0]), int(t[1])
    i, vals = 3, []
    for _ in range(nj):
        n_ops = int(t[i])
        i += 1
        for _ in range(n_ops):
            k = int(t[i])
            i += 1
            for _ in range(k):
                i += 1                          # machine index
                vals.append(int(t[i]))
                i += 1
    return nj, nm, vals


def base_instances(folder):
    """The base .fjs files the descriptors of a family folder name, over every combination."""
    meta = json.load(open(os.path.join(folder, 'dataset_meta.json')))
    names = set()
    for f in glob.glob(os.path.join(folder, 'desc', '*', '*.desc.json')):
        names.add(json.load(open(f))['base_instance'])
    if not names:
        raise Refused(f'no descriptors under {folder}/desc')
    files = sorted(n + '.fjs' for n in names)
    miss = [f for f in files if not os.path.exists(f)]
    if miss:
        raise Refused(f'{folder}: base instance {miss[0]} does not exist')
    return meta.get('base_dir'), files


def op_duration_median():
    base, files = base_instances(TEST_FOLDER)
    vals = [v for f in files for v in fjs_times(f)[2]]
    if not vals:
        raise Refused(f'no processing times in the base instances of {TEST_FOLDER}')
    return dict(value=float(np.median(vals)), pairs=len(vals), instances=len(files),
                base_dir=base)


def shop_size(folder):
    """(modules, stations) of every base instance of a family folder; refused when they
    differ across the instances."""
    _, files = base_instances(folder)
    sizes = sorted({fjs_times(f)[:2] for f in files})
    if len(sizes) != 1:
        raise Refused(f'{folder}: the base instances have different sizes {sizes[:3]}')
    return sizes[0], len(files)


# ----------------------------------------------------------------- values in force
def config_value(paths, key, what):
    """The value of key (dotted for a nested field) in the stored configurations on disk;
    refused when none is on disk or they disagree."""
    vals = {}
    for p in paths:
        if not os.path.exists(p):
            continue
        v = json.load(open(p))
        for k in key.split('.'):
            v = v.get(k) if isinstance(v, dict) else None
        vals[p] = v
    if not vals:
        raise Refused(f'{what}: no stored configuration on disk '
                      f'({paths[0] if paths else "no path"} and others)')
    got = sorted(set(map(str, vals.values())))
    if len(got) != 1:
        raise Refused(f'{what}: the stored configurations disagree on {key}: {got}')
    return next(iter(vals.values())), sorted(vals)


def episodes_per_combination(cfg):
    """Training episodes per training combination of one stored configuration: max_updates x
    num_envs over the combinations of family_regimes, rounded to the integer."""
    combos = [x for x in str(cfg.get('family_regimes') or '').split(',') if x.strip()]
    if not combos or cfg.get('max_updates') is None or cfg.get('num_envs') is None:
        raise Refused('a stored configuration lacks max_updates, num_envs or family_regimes')
    return round(cfg['max_updates'] * cfg['num_envs'] / len(combos))


def episodes_value(paths, what):
    vals = {p: episodes_per_combination(json.load(open(p))) for p in paths if os.path.exists(p)}
    if not vals:
        raise Refused(f'{what}: no stored configuration on disk '
                      f'({paths[0] if paths else "no path"} and others)')
    if len(set(vals.values())) != 1:
        raise Refused(f'{what}: the stored configurations disagree: {sorted(set(vals.values()))}')
    return next(iter(vals.values())), sorted(vals)


def values_in_force(runs):
    import timing_validity as TV
    cfg = {arm: [c for _, _, c, a in runs if a == arm] for arm in REG.ARMS}
    eq = [c for _, _, c in secondary_runs(EQ_STEM)]
    out = {}
    out['policy_updates'] = config_value(cfg['released'], 'max_updates', 'updates of the policy')
    out['specialist_updates'] = config_value(cfg['specialist'], 'max_updates',
                                             'updates of a specialist')
    out['eq_updates'] = config_value(eq, 'max_updates', 'updates of the longer-trained policy')
    out['envs'] = config_value([c for _, _, c, _ in runs], 'num_envs',
                               'parallel training episodes per update')
    out['policy_episodes'] = episodes_value(cfg['released'], 'training episodes of the policy')
    out['specialist_episodes'] = episodes_value(cfg['specialist'],
                                                'training episodes of a specialist')
    out['eq_episodes'] = episodes_value(eq, 'training episodes of the longer-trained policy')
    hg = hgnn_configs()
    out['hgnn_batch'] = config_value(hg, 'env_paras.batch_size',
                                     'instances per iteration of the published HGNN scheduler')
    out['hgnn_params'] = config_value(hg, 'n_params',
                                      'parameters of the published HGNN scheduler')
    stations = {}
    for size, folder in SIZES.items():
        (nj, nm), n = shop_size(folder)
        out[f'jobs_{size}'] = (nj, [f'{folder}: {n} base instances'])
        stations[folder] = nm
    if len(set(stations.values())) != 1:
        raise Refused(f'stations per shop differ across the data folders: {stations}')
    out['stations'] = (next(iter(stations.values())), sorted(stations))
    out['share_min'] = (TV.SHARE_MIN, ['scripts/timing_validity.py SHARE_MIN'])
    return out


def number(text):
    s = re.sub(r'\\,|\{,\}|,|\s|~', '', text)
    try:
        return float(s)
    except ValueError:
        return None


def definitions(name):
    """[(file, value text)] of every definition of the macro in CONSTANT_FILES."""
    rx = re.compile(r'\\(?:newcommand|renewcommand|providecommand)\*?\s*\{?\\' + name
                    + r'\}?\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}')
    out = []
    for f in CONSTANT_FILES:
        p = os.path.join(ROOT, f)
        if not os.path.exists(p):
            continue
        text = '\n'.join(re.sub(r'(?<!\\)%.*', '', x)
                         for x in open(p, errors='replace').read().splitlines())
        out += [(f, v) for v in rx.findall(text)]
    return out


def check_constants(force):
    """[(macro, meaning, value in force, [(file, text, equal)])] and the list of
    differences."""
    rows, bad = [], []
    for name, meaning, key in CONSTANTS:
        want = force[key][0]
        defs = [(f, v, number(v) is not None and abs(number(v) - float(want)) < 1e-12)
                for f, v in definitions(name)]
        rows.append(dict(macro=name, meaning=meaning, value=want, sources=force[key][1],
                         definitions=[dict(file=f, text=v, equal=ok) for f, v, ok in defs]))
        if not defs:
            bad.append(f'\\{name} ({meaning}) is defined in none of {", ".join(CONSTANT_FILES)}')
        for f, v, ok in defs:
            if not ok:
                bad.append(f'\\{name} ({meaning}) is {v!r} in {f}; the value in force is {want}')
    return rows, bad


# ----------------------------------------------------------------- names
def defined_macros():
    """{name: [files]} of every macro defined in a .tex file under manuscript/, the output
    file left out."""
    rx = re.compile(r'\\(?:newcommand|renewcommand|providecommand)\*?\s*\{?\s*\\([A-Za-z@]+)'
                    r'|\\def\s*\\([A-Za-z@]+)')
    own = os.path.basename(OUT_FILES['macros'])
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, 'manuscript', '**', '*.tex'), recursive=True)):
        if os.path.basename(f) == own:
            continue
        text = '\n'.join(re.sub(r'(?<!\\)%.*', '', x)
                         for x in open(f, errors='replace').read().splitlines())
        for a, b in rx.findall(text):
            out.setdefault(a or b, []).append(os.path.relpath(f, ROOT))
    return out


def check_names(names):
    have = defined_macros()
    bad = []
    for x in names:
        if not re.fullmatch(r'[A-Za-z]+', x):
            bad.append(f'\\{x} is not letters only')
            continue
        other = sorted(set(have.get(x, [])) - ({KEPT[x]} if x in KEPT else set()))
        if other:
            bad.append(f'\\{x} is also defined in {", ".join(other)}')
    if bad:
        raise Refused('; '.join(bad))


# ----------------------------------------------------------------- run
def require_exact():
    if REG.env() != ENV:
        raise Refused(f'the counts are those of the {ENV} environment; the active environment '
                      f'is {REG.env()} (set PACT_ENV={ENV})')


def analyse():
    require_exact()
    runs = principal_runs()
    active = set(REG.train_log(a, c, sd, ENV) for sd in REG.seeds() for a in REG.ARMS
                 for c in (REG.COMBOS if a == 'specialist' else [None]))
    policy = count([lg for _, lg, _, _ in runs if lg in active],
                   'parameters of the policy network', True)
    more = count([lg for _, lg, _, _ in runs if lg not in active],
                 'parameters of the policy network (second five seeds)', False)
    if more['value'] is not None and more['value'] != policy['value']:
        raise Refused(f'parameters of the policy network: {more["value"]} in the logs of the '
                      f'second five seeds, {policy["value"]} in those of the active seeds')
    policy['runs'].update(more['runs'])
    policy['missing'] = more['missing']
    film = count([lg for _, lg, _ in secondary_runs(FILM_STEM)],
                 'parameters of the variant with descriptor input', False)
    hg = hgnn_logs()
    hgnn = count(hg, 'parameters of the HGNN scheduler', False)
    if not hg:
        hgnn['missing'] = [f'no training log matching '
                           f'{_layout(REG.train_log).format(m=HGNN_GLOB)}']
    facts = dict(policy=policy, descriptor=film, hgnn=hgnn, checkpoint=checkpoint_kib(),
                 op_duration=op_duration_median())
    force = values_in_force(runs)
    rows, bad = check_constants(force)
    return dict(generator='scripts/p111_model_facts.py', setting=REG.describe(), facts=facts,
                constants=rows, differences=bad)


def macro_lines(res):
    f = res['facts']
    L = [f'% Generated by scripts/p111_model_facts.py ({res["setting"]}) -- do not hand-edit']
    left = []

    def put(name, value, comment):
        L.append(f'\\newcommand{{\\{name}}}{{{value}}}  % {comment}')

    put('BlindParams', f['policy']['value'],
        f'parameters of the policy network, equal in the training logs of '
        f'{len(f["policy"]["runs"])} principal runs')
    for name, key, what in (('DescriptorParams', 'descriptor',
                             'parameters of the variant with descriptor input'),
                            ('HgnnNetworkParams', 'hgnn',
                             'parameters of the HGNN scheduler trained under the non-delay mask')):
        if f[key]['value'] is None:
            left.append(f'\\{name} ({what}): {f[key]["missing"][0]}')
        else:
            put(name, f[key]['value'], f'{what}, equal in {len(f[key]["runs"])} training logs'
                + (f'; {len(f[key]["missing"])} runs not on disk' if f[key]['missing'] else ''))
    c = f['checkpoint']
    put('PolicyCheckpointKB', f'{c["kib"]:.0f}',
        f'KiB (bytes / 1024) of {c["path"]}, the selected policy run (seed {c["seed"]})')
    o = f['op_duration']
    put('OpDurMedian', f'{o["value"]:.0f}',
        f'h, median processing time over {o["pairs"]} compatible (operation, machine) pairs '
        f'of the {o["instances"]} base instances of the test set at 10x25')
    for x in left:
        L.append(f'% Left out, runs not on disk: {x}')
    return L, left


def outputs():
    return {k: REG.out(v) for k, v in OUT_FILES.items()}


def remove_outputs(outs):
    for p in outs.values():
        if os.path.exists(p):
            os.remove(p)


def run():
    require_exact()
    if os.environ.get('PACT_OUT') is None:
        raise Refused('set PACT_OUT (the output root of the chain, e.g. out/exact)')
    outs = outputs()
    remove_outputs(outs)
    res = analyse()
    if res['differences']:
        raise Refused('design constants that differ from the values in force:\n  '
                      + '\n  '.join(res['differences']))
    lines, left = macro_lines(res)
    names = re.findall(r'\\newcommand\{\\([^}]*)\}', '\n'.join(lines))
    check_names(names)
    res['left_out'] = left
    try:
        emit(outs['macros'], lines)
        with open(outs['json'], 'w') as fh:
            json.dump(res, fh, indent=1, default=str)
    except BaseException:
        remove_outputs(outs)
        raise
    return res, names


def check():
    """Read-only precondition of the chain step; returns the exit code."""
    require_exact()
    miss = []
    for _, lg, cfg, _ in principal_runs():
        miss += [p for p in (lg, cfg) if not os.path.exists(p)]
    miss += [c for _, _, c in secondary_runs(EQ_STEM) if not os.path.exists(c)]
    if not hgnn_configs():
        miss.append('no configuration of the published HGNN scheduler '
                    + _layout(REG.config_path).format(m=HGNN_PUBLISHED_GLOB))
    miss += [f for f in CONSTANT_FILES if not os.path.exists(os.path.join(ROOT, f))]
    try:
        c = checkpoint_kib()
        print(f'p111: checkpoint of the selected policy run {c["path"]}')
    except (Refused, SystemExit) as e:
        miss.append(str(e).splitlines()[0])
    if not glob.glob(os.path.join(TEST_FOLDER, 'desc', '*', '*.desc.json')):
        miss.append(f'no descriptors under {TEST_FOLDER}/desc')
    print('p111: inputs ' + ('present' if not miss else
                             f'missing: {len(miss)}, first {miss[0]}'))
    return 0 if not miss else 1


def report(res, names):
    f = res['facts']
    print(f'policy network: {f["policy"]["value"]} parameters in {len(f["policy"]["runs"])} logs')
    for key, what in (('descriptor', 'variant with descriptor input'), ('hgnn', 'HGNN scheduler')):
        x = f[key]
        print(f'{what}: ' + (f'{x["value"]} parameters in {len(x["runs"])} logs'
                             if x['value'] is not None else 'left out')
              + (f'; not on disk: {"; ".join(x["missing"])}' if x['missing'] else ''))
    c = f['checkpoint']
    print(f'checkpoint of the selected policy run (seed {c["seed"]}): {c["bytes"]} bytes, '
          f'{c["kib"]:.1f} KiB')
    o = f['op_duration']
    print(f'median processing time: {o["value"]:g} h over {o["pairs"]} pairs, '
          f'{o["instances"]} base instances')
    for r in res['constants']:
        print(f'  \\{r["macro"]:<15s} {r["meaning"]:<45s} in force {r["value"]}: '
              + ', '.join(f'{d["file"]} {d["text"]!r} {"equal" if d["equal"] else "DIFFERS"}'
                          for d in r['definitions']))
    print(f'{len(names)} macros; wrote ' + ', '.join(outputs().values()))


def main(argv=None):
    ap = argparse.ArgumentParser(description='counts, file facts and design constants')
    ap.add_argument('--check', action='store_true', help='read-only precondition of the chain')
    a = ap.parse_args(argv)
    os.chdir(ROOT)
    try:
        if a.check:
            sys.exit(check())
        REG.banner()
        res, names = run()
    except Refused as e:
        sys.exit(f'*** REFUSING TO CERTIFY ***\n{e}')
    report(res, names)


if __name__ == '__main__':
    main()
