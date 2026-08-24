"""P35: give the main paper a stable, directed way to cite supplementary sections.

The main text defers a great deal to the supplement, and every one of those
pointers used to read "(supplementary material)" with no section behind it. A
reader who wanted the proof, the reward pseudocode, or the out-of-distribution
table had to hunt through nineteen-odd sections to find it. Directing each pointer
by hand is worse than not directing it: the supplement is renumbered whenever a
section is added, moved, or removed, and a literal "Sec. S-XI" silently rots
into a pointer to the wrong section the first time that happens.

So the section numbers are GENERATED, exactly like every rerun-movable number in
this paper. This script reads supplementary.tex top to bottom, inlines the
\\input'd section files in the position they are pulled in (so a \\section living
in sections/sm_baselines.tex is counted where it actually renders, not skipped),
assigns the S-Roman numerals IEEEtran will assign, and emits one structural macro
per section: \\SmMdp -> S-I, \\SmProofs -> S-IX, and so on. The main paper then
writes \\SmProofs and cannot point at the wrong section: if the supplement is
renumbered, this script is rerun and every pointer follows.

Two failure modes abort the run rather than emit a quietly wrong table:
  * a \\section with no sec:sm-* label (nothing to name a macro after, and a sign
    the section was added without wiring it in); and
  * two sections sharing a label (the second would overwrite the first's number).

Unlike the other p-scripts this reads no result files and computes no statistics;
the "numbers" it owns are section ordinals, which are structural, not empirical.
A missing macros_smrefs.tex therefore does not print "??": it leaves every \\Sm*
undefined and the build fails loudly, which is the right behavior for a
structural reference that must never be silently wrong.

The same rot runs in the other direction, and it bit once. The supplement cites
main-paper tables and figures, LaTeX cannot resolve a label across documents, so
those pointers were typed by hand as "Table~I of the main paper". Inserting the
E1 budget table ahead of the E5 table renumbered E5 to Table II and left nine
supplement sentences pointing at a table about training budgets. Nothing in the
build could see it: both documents compiled clean, and "Table~I" is valid prose.
So this script also reads main.aux, which carries the number LaTeX actually
assigned to every labelled float, and emits one macro per float: \\MainTabEfive
-> II, \\MainFigAnytime -> 4. The supplement writes those and follows the main
paper automatically. main.aux is written by the main build, which the build gate
runs first; if it is absent this script refuses rather than guess.

Emits paper/macros_smrefs.tex (supplementary section numerals, used by both
documents) and paper/macros_mainrefs.tex (main-paper float numerals, used by the
supplement only, so a clean tree can still build main.tex before main.aux exists).

Usage: python scripts/p35_sm_refs.py
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUPP = os.path.join(ROOT, 'paper', 'supplementary.tex')
OUT = os.path.join(ROOT, 'paper', 'macros_smrefs.tex')
MAIN_AUX = os.path.join(ROOT, 'paper', 'main.aux')
MAIN_OUT = os.path.join(ROOT, 'paper', 'macros_mainrefs.tex')

# \newlabel{tab:e5}{{II}{8}{...}} -- group 1 the label, group 2 the printed
# number. IEEEtran writes tables in Roman and figures in Arabic; both are taken
# verbatim, so this never has to know which convention a class uses.
NEWLABEL_RE = re.compile(r'\\newlabel\{((?:tab|fig):[^}]+)\}\{\{([^}]*)\}')

# \section{...}, allowing one level of nested braces in the title; the trailing
# \{ (after an optional *) is what keeps this from matching inside \subsection.
SECTION_RE = re.compile(r'\\section\*?\{((?:[^{}]|\{[^}]*\})*)\}')
LABEL_RE = re.compile(r'\\label\{(sec:sm-[^}]+)\}')
INPUT_RE = re.compile(r'^\s*\\input\{sections/([^}]+)\}')


def roman(n):
    table = [(1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'), (100, 'C'),
             (90, 'XC'), (50, 'L'), (40, 'XL'), (10, 'X'), (9, 'IX'),
             (5, 'V'), (4, 'IV'), (1, 'I')]
    out = []
    for value, sym in table:
        while n >= value:
            out.append(sym)
            n -= value
    return ''.join(out)


DIGIT_WORD = {'0': 'Zero', '1': 'One', '2': 'Two', '3': 'Three', '4': 'Four',
              '5': 'Five', '6': 'Six', '7': 'Seven', '8': 'Eight', '9': 'Nine'}


def camel(tail):
    """'wcause' -> Wcause; 'cpsat' -> Cpsat; 'e7' -> ESeven. Split on hyphens,
    then for each piece run over runs of letters (capitalize the first, lowercase
    the rest) and runs of digits (spell each digit, capitalized). A control
    sequence is letters only, so a digit left in the name would make \\SmE7 parse
    as \\SmE followed by the text '7' and take a hundred errors with it, exactly
    as p30 warns."""
    out = []
    for part in tail.split('-'):
        if not part:
            continue
        piece = ''
        for run in re.findall(r'[A-Za-z]+|[0-9]+', part):
            if run[0].isdigit():
                piece += ''.join(DIGIT_WORD[d] for d in run)
            else:
                piece += run[:1].upper() + run[1:].lower()
        out.append(piece)
    return ''.join(out)


def macro_name(label):
    """sec:sm-wcause -> SmWcause; sec:sm-e7 -> SmESeven."""
    return 'Sm' + camel(label[len('sec:sm-'):])


def main_float_macros():
    """One macro per labelled float of the main paper, read from main.aux, so a
    supplement sentence can cite "Table~\\MainTabEfive{}" and never name a
    numeral the main paper has since reassigned."""
    if not os.path.exists(MAIN_AUX):
        sys.exit('REFUSING: paper/main.aux does not exist, so the numbers the '
                 'main paper assigns to its tables and figures are unknown. '
                 'Build main.tex first (the build gate does, before the '
                 'supplement), then rerun this script.')
    lines, seen = [], {}
    with open(MAIN_AUX) as f:
        for label, num in NEWLABEL_RE.findall(f.read()):
            kind, tail = label.split(':', 1)
            name = ('MainTab' if kind == 'tab' else 'MainFig') + camel(tail)
            if name in seen and seen[name] != num:
                sys.exit(f'REFUSING: {label} resolves to both {seen[name]} and '
                         f'{num} in main.aux; the build did not converge.')
            if name in seen:
                continue
            seen[name] = num
            lines.append(f'\\newcommand{{\\{name}}}{{{num}}}  % {label}')
    if not lines:
        sys.exit('REFUSING: main.aux names no tab: or fig: label. Either the '
                 'main paper lost every float or the aux file is truncated; '
                 'neither should pass silently.')
    header = ('% auto-generated by scripts/p35_sm_refs.py -- do not hand-edit\n'
              '% Main-paper float numbers, read from main.aux. The supplement\n'
              '% cites these because LaTeX cannot resolve a label across two\n'
              '% documents, and a literal numeral rots when a float is added.\n')
    with open(MAIN_OUT, 'w') as f:
        f.write(header)
        f.write('\n'.join(sorted(lines)) + '\n')
    print(f'wrote {os.path.relpath(MAIN_OUT, ROOT)} ({len(lines)} main floats): '
          + ', '.join(f'\\{k}={v}' for k, v in sorted(seen.items())))


def inlined_lines(path):
    """(text, origin_file) for every line of supplementary.tex, with the bodies
    of \\input{sections/X} spliced in at the point of inclusion so a \\section in
    an input'd file is seen in document order. Only plain \\input is expanded;
    \\InputIfFileExists (the generated-macro guards) is left alone."""
    out = []
    base = os.path.basename(path)
    with open(path) as f:
        for raw in f:
            m = INPUT_RE.match(raw)
            if m:
                stem = m.group(1)
                sub = os.path.join(ROOT, 'paper', 'sections', stem + '.tex')
                with open(sub) as g:
                    for sraw in g:
                        out.append((sraw, stem + '.tex'))
            else:
                out.append((raw, base))
    return out


def main():
    sections = []          # {title, label, origin}
    pending = None
    for text, origin in inlined_lines(SUPP):
        for m in SECTION_RE.finditer(text):
            if pending is not None and pending['label'] is None:
                sys.exit(f"REFUSING: section {pending['title']!r} (from "
                         f"{pending['origin']}) has no sec:sm-* label. Every "
                         f"supplementary section must carry one so the main "
                         f"paper can point at it; add \\label{{sec:sm-...}} to it.")
            pending = {'title': m.group(1), 'label': None, 'origin': origin}
            sections.append(pending)
        for m in LABEL_RE.finditer(text):
            if pending is not None and pending['label'] is None:
                pending['label'] = m.group(1)
            # sec:sm-* labels on \subsections come after the section's own label
            # is already set, so this guard leaves them for referencing elsewhere.
    if pending is not None and pending['label'] is None:
        sys.exit(f"REFUSING: section {pending['title']!r} (from "
                 f"{pending['origin']}) has no sec:sm-* label.")

    seen = {}
    lines = []
    print(f'{"S-num":7} {"macro":14} {"origin":22} title')
    print('-' * 78)
    for i, sec in enumerate(sections, start=1):
        label, title, origin = sec['label'], sec['title'], sec['origin']
        if label in seen:
            sys.exit(f"REFUSING: label {label} is used twice, on section "
                     f"{seen[label]!r} and on {title!r}. The second definition "
                     f"would silently overwrite the first's number.")
        seen[label] = title
        snum = 'S-' + roman(i)
        name = macro_name(label)
        lines.append(f'\\newcommand{{\\{name}}}{{{snum}}}'
                     f'  % {label}: {title}')
        print(f'{snum:7} \\{name:13} {origin:22} {title}')

    hosts = sorted({s['origin'] for s in sections
                    if s['origin'] != 'supplementary.tex'})
    print()
    print('sections contributed by \\input files: '
          + (', '.join(hosts) if hosts else '(none)'))

    header = ('% auto-generated by scripts/p35_sm_refs.py -- do not hand-edit\n'
              '% Supplementary section cross-references. Each \\Sm* expands to the\n'
              '% S-Roman numeral IEEEtran assigns, in the order sections appear in\n'
              '% supplementary.tex with \\input\'d section files inlined in place.\n')
    with open(OUT, 'w') as f:
        f.write(header)
        f.write('\n'.join(lines) + '\n')
    print(f'\nwrote {os.path.relpath(OUT, ROOT)} ({len(lines)} sections)')
    main_float_macros()


if __name__ == '__main__':
    main()
