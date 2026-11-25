"""The one way a generator is allowed to put a number into the manuscript.

Two rules, both enforced here:

  emit()   never writes a non-finite value. 'nan' is number-shaped, so a nan
           that reaches LaTeX renders as a result; a generator that cannot
           compute its number must fail loudly, not launder the symbol.

  refuse() removes the target file entirely before exiting, so every macro in
           it falls through to the \\providecommand fallback and renders as a
           conspicuous '??'. A generator that declines to certify must leave
           NOTHING behind: a partial macro file is indistinguishable from a
           successful run.
"""
import math
import os
import re
import sys

_NUM = re.compile(r'-?\d+\.?\d*(?:[eE][-+]?\d+)?')


def _bad(v):
    if isinstance(v, (int, float)):
        return not math.isfinite(v)
    # Word-boundary match, not substring: 'inf' as a substring hits words like
    # "non-inferior" in trailing comments and rejects a perfectly good file.
    return bool(re.search(r'(?<![A-Za-z-])(nan|inf)(?![A-Za-z-])', str(v),
                          re.IGNORECASE))


def emit(path, lines, header=None):
    """Write a macro file, or die loudly rather than ship a nan.

    `lines` is a list of '\\newcommand{\\Foo}{...}' strings (or (name, value)
    pairs). Any non-finite value is a hard failure: a generator that cannot
    compute its number must say so, not launder it through LaTeX.
    """
    out = []
    for ln in lines:
        if isinstance(ln, (tuple, list)):
            name, val, *rest = ln
            if _bad(val):
                sys.exit(f'REFUSING to write {path}: \\{name} would be {val!r}. '
                         f'A non-finite macro means the data behind it is empty '
                         f'or broken. Fix the data, do not ship the symbol.')
            cmt = f'  % {rest[0]}' if rest else ''
            out.append(f'\\newcommand{{\\{name}}}{{{val}}}{cmt}')
        else:
            if _bad(ln):
                sys.exit(f'REFUSING to write {path}: a macro line is non-finite:\n'
                         f'  {ln}')
            out.append(str(ln))
    if not out:
        sys.exit(f'REFUSING to write {path}: no macros to emit. An empty macro '
                 f'file renders as ?? and looks like a build problem rather than '
                 f'the missing experiment it actually is.')
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        if header:
            f.write(f'% {header}\n')
        f.write('\n'.join(out) + '\n')
    print(f'wrote {path} ({len(out)} macros)')


def refuse(path, why):
    """Refuse to certify: remove the file so the manuscript shows ?? and stops.

    Removing the file guarantees no stale or partial macro survives to render
    a number this generator no longer certifies.
    """
    if os.path.exists(path):
        os.remove(path)
        print(f'removed {path} (refusing to certify)')
    sys.exit(f'*** REFUSING TO CERTIFY ***\n{why}')
