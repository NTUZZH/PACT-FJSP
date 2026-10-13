"""P65: counts behind the configuration-checks table (constraint_family/checks.py).

Runs the checks over three descriptor sets of the 10x25 test set:
  published       data/FAMILY/10x25          all 8 regimes          800 records
  shifted setup   data/FAMILY/10x25_oodsetup S, LS, SW, LSW         400 records
  shifted window  data/FAMILY/10x25_oodwindow W, LW, SW, LSW        400 records
(the shifted arms were written by scripts/p29_ood.py), and emits
paper/macros_misc_aei.tex:

  \\CheckFourShiftedFlagged / \\CheckFourShiftedOf   C4 = WARN on the shifted-setup arm
  \\CheckFourWindowFlagged  / \\CheckFourWindowOf    C4 = WARN on the shifted-window arm
  \\CheckFourPublishedFlagged / \\CheckFourPublishedOf  C4 = WARN on the published set
  \\CheckOneToFivePublishedPass   published records with C1, C2, C3 and C5 all PASS
  \\ChecksImplementedN            checks implemented in code (C1..C6; C7 is a procedure)
  \\OodSetupCrossX                mean cross-class setup entry, shifted over published
                                  (ratio of the pooled means over the 100 S descriptors)

Refuses to certify (removes the macro file) when a record count is not the
expected one. CPU only, a few seconds.

Usage (repository root):  python3 scripts/p65_check_counts.py
"""
import glob
import json
import os
import sys

sys.argv = sys.argv[:1]          # params.py (imported via data_utils) parses argv
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np  # noqa: E402

from constraint_family import checks  # noqa: E402
from constraint_family.descriptor import (N_CLASSES, REGIMES_ALL,  # noqa: E402
                                          ConstraintDescriptor)
from macro_io import emit, refuse  # noqa: E402
from ppvc_instance_generator import load_instance  # noqa: E402

OUT = 'paper/macros_misc_aei.tex'
PUBLISHED = 'data/FAMILY/10x25'
OOD_SETUP = 'data/FAMILY/10x25_oodsetup'
OOD_WINDOW = 'data/FAMILY/10x25_oodwindow'
SETUP_REGIMES = tuple(r for r in REGIMES_ALL if 'S' in r)
WINDOW_REGIMES = tuple(r for r in REGIMES_ALL if 'W' in r)

_BASE = {}


def records(root, regimes):
    for regime in regimes:
        for p in sorted(glob.glob(os.path.join(root, 'desc', regime, '*.desc.json'))):
            with open(p) as f:
                rec = json.load(f)
            stem = rec['base_instance']
            if stem not in _BASE:
                _BASE[stem] = load_instance(stem)
            yield regime, os.path.basename(p), rec, _BASE[stem][1]


def c4_counts(root, regimes):
    flagged = n = 0
    for _, _, rec, pt in records(root, regimes):
        desc = ConstraintDescriptor.from_json(rec)
        flagged += checks.check_magnitude_range(desc, pt).status == checks.WARN
        n += 1
    return flagged, n


def cross_mean(rec):
    S = np.asarray(rec['setup'], dtype=float)[:N_CLASSES]
    return S[~np.eye(N_CLASSES, dtype=bool)]


def main():
    # published: full run_all per record
    pub_flag = pub_n = pub_pass15 = 0
    for _, _, rec, pt in records(PUBLISHED, REGIMES_ALL):
        st = {r.id: r.status for r in checks.run_all(ConstraintDescriptor.from_json(rec), pt)}
        pub_n += 1
        pub_flag += st['C4'] == checks.WARN
        pub_pass15 += all(st[c] == checks.PASS for c in ('C1', 'C2', 'C3', 'C5'))
    sh_flag, sh_n = c4_counts(OOD_SETUP, SETUP_REGIMES)
    wi_flag, wi_n = c4_counts(OOD_WINDOW, WINDOW_REGIMES)

    for label, n, want in (('published', pub_n, 800), ('shifted setup', sh_n, 400),
                           ('shifted window', wi_n, 400)):
        if n != want:
            refuse(OUT, f'{label}: found {n} descriptor records, expected {want}. '
                        f'Check that the data directories are populated.')

    # cross-class setup ratio, paired on the 100 S descriptors
    shifted, published, per_inst = [], [], []
    for _, name, rec, _ in records(OOD_SETUP, ('S',)):
        with open(os.path.join(PUBLISHED, 'desc', 'S', name)) as f:
            ref = json.load(f)
        a, b = cross_mean(rec), cross_mean(ref)
        shifted.append(a)
        published.append(b)
        per_inst.append(a.mean() / b.mean())
    if len(shifted) != 100:
        refuse(OUT, f'found {len(shifted)} paired S descriptors, expected 100')
    ratio = float(np.concatenate(shifted).mean() / np.concatenate(published).mean())

    implemented = 6   # C1..C6 are functions in checks.py; C7 returns MANUAL by design
    assert all(hasattr(checks, f) for f in (
        'check_schema_and_values', 'check_setup_convention', 'check_family_membership',
        'check_magnitude_range', 'check_applicability', 'check_schedule_validation'))

    print(f'published: C4 WARN {pub_flag}/{pub_n}; C1,C2,C3,C5 all PASS {pub_pass15}/{pub_n}')
    print(f'shifted setup: C4 WARN {sh_flag}/{sh_n}')
    print(f'shifted window: C4 WARN {wi_flag}/{wi_n}')
    print(f'cross-class setup ratio: pooled {ratio:.4f}, '
          f'mean of per-instance ratios {np.mean(per_inst):.4f}')

    emit(OUT, [
        ('CheckFourShiftedFlagged', f'{sh_flag:d}'),
        ('CheckFourShiftedOf', f'{sh_n:d}'),
        ('CheckFourWindowFlagged', f'{wi_flag:d}'),
        ('CheckFourWindowOf', f'{wi_n:d}'),
        ('CheckFourPublishedFlagged', f'{pub_flag:d}'),
        ('CheckFourPublishedOf', f'{pub_n:d}'),
        ('CheckOneToFivePublishedPass', f'{pub_pass15:d}'),
        ('ChecksImplementedN', f'{implemented:d}'),
        ('OodSetupCrossX', f'{ratio:.2f}'),
    ], header='Generated by scripts/p65_check_counts.py from data/FAMILY/10x25, '
              '10x25_oodsetup and 10x25_oodwindow via constraint_family/checks.py')


if __name__ == '__main__':
    main()
