"""Build the FAMILY benchmark descriptors on top of the published PPVC
instances.

For every base instance and every regime, a ConstraintDescriptor is generated
deterministically from DESC_SEED0 + instance index. Member parameters are
drawn from member-specific streams (descriptor.make_descriptor), so a given
(instance, member) has IDENTICAL parameters in every regime containing that
member — cross-regime comparisons are paired and family monotonicity holds
exactly.

An existing output folder (one that holds dataset_meta.json) is never rewritten:
the command line regenerates every descriptor and the meta file in memory and
compares them byte for byte with the files on disk. It reports "already built and
identical" when every file matches, and refuses, writing nothing, when a file
differs or is missing. build() itself, which scripts/p86_extend_scale_sets.py
calls on a fresh folder, is unchanged.

Usage:  python scripts/p1_build_family_data.py [--base data/PPVC/10x25+ppvc-mixed]
                                               [--out data/FAMILY/10x25]
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# params.py parses sys.argv at import time; strip our flags first
_ARGV = sys.argv[1:]
sys.argv = sys.argv[:1]

import numpy as np

from constraint_family.descriptor import make_descriptor, REGIMES_ALL
from ppvc_instance_generator import load_instance

DESC_SEED0 = 20260706


def build(base_dir, out_dir):
    stems = sorted(glob.glob(os.path.join(base_dir, 'instance_*.fjs')))
    stems = [s[:-len('.fjs')] for s in stems]
    assert stems, f'no instances under {base_dir}'
    os.makedirs(out_dir, exist_ok=True)

    for idx, stem in enumerate(stems):
        jl, pt, meta = load_instance(stem)
        M = pt.shape[1]
        name = os.path.basename(stem)
        for regime in REGIMES_ALL:
            desc = make_descriptor(regime, jl, pt, meta, M,
                                   seed=DESC_SEED0 + idx)
            d = desc.to_json()
            d['base_instance'] = os.path.relpath(stem)
            d['regime'] = regime
            d['desc_seed'] = DESC_SEED0 + idx
            rdir = os.path.join(out_dir, 'desc', regime)
            os.makedirs(rdir, exist_ok=True)
            with open(os.path.join(rdir, f'{name}.desc.json'), 'w') as f:
                json.dump(d, f)

    with open(os.path.join(out_dir, 'dataset_meta.json'), 'w') as f:
        json.dump({'base_dir': os.path.relpath(base_dir),
                   'n_instances': len(stems),
                   'regimes': list(REGIMES_ALL),
                   'desc_seed0': DESC_SEED0,
                   'convention': 'attached setups; contiguous setup+proc '
                                 'block avoids outages; idle-row cold-start '
                                 'setup; integer hours'}, f, indent=1)
    print(f'wrote descriptors for {len(stems)} instances x '
          f'{len(REGIMES_ALL)} regimes under {out_dir}')


def expected_files(base_dir):
    """{path relative to the output folder: text} of every file build() writes, computed
    in memory exactly as build() serializes it."""
    stems = sorted(glob.glob(os.path.join(base_dir, 'instance_*.fjs')))
    stems = [s[:-len('.fjs')] for s in stems]
    assert stems, f'no instances under {base_dir}'
    out = {}
    for idx, stem in enumerate(stems):
        jl, pt, meta = load_instance(stem)
        M = pt.shape[1]
        name = os.path.basename(stem)
        for regime in REGIMES_ALL:
            d = make_descriptor(regime, jl, pt, meta, M, seed=DESC_SEED0 + idx).to_json()
            d['base_instance'] = os.path.relpath(stem)
            d['regime'] = regime
            d['desc_seed'] = DESC_SEED0 + idx
            out[os.path.join('desc', regime, f'{name}.desc.json')] = json.dumps(d)
    out['dataset_meta.json'] = json.dumps(
        {'base_dir': os.path.relpath(base_dir), 'n_instances': len(stems),
         'regimes': list(REGIMES_ALL), 'desc_seed0': DESC_SEED0,
         'convention': 'attached setups; contiguous setup+proc block avoids outages; '
                       'idle-row cold-start setup; integer hours'}, indent=1)
    return out


def verify(base_dir, out_dir):
    """Compare an existing output folder with what build() would write; nothing is
    written. Returns (differences, files compared); no difference means identical."""
    want = expected_files(base_dir)
    bad = []
    for rel, text in want.items():
        p = os.path.join(out_dir, rel)
        if not os.path.exists(p):
            bad.append(f'missing {p}')
        elif open(p).read() != text:
            bad.append(f'differs {p}')
    have = {os.path.relpath(p, out_dir)
            for p in glob.glob(os.path.join(out_dir, 'desc', '*', '*.desc.json'))}
    bad += [f'not written by this build {os.path.join(out_dir, x)}'
            for x in sorted(have - set(want))]
    return bad, len(want)


if __name__ == '__main__':
    base = 'data/PPVC/10x25+ppvc-mixed'
    out = 'data/FAMILY/10x25'
    for i, a in enumerate(_ARGV):
        if a == '--base':
            base = _ARGV[i + 1]
        if a == '--out':
            out = _ARGV[i + 1]
    if os.path.exists(os.path.join(out, 'dataset_meta.json')):
        bad, n_files = verify(base, out)
        if bad:
            sys.exit(f'REFUSING: {out} exists and {len(bad)} of its files are not what '
                     f'this build writes (first: {bad[0]}). Nothing was written.')
        print(f'{out}: already built and identical ({n_files} files compared byte for '
              f'byte) -- nothing to do')
    elif os.path.isdir(out) and os.listdir(out):
        sys.exit(f'REFUSING: {out} holds files but no dataset_meta.json (an interrupted '
                 f'build?). Inspect and remove it by hand; nothing was written.')
    else:
        build(base, out)
