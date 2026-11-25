# Brandimarte (1993) instances Mk01 to Mk10

## Source

- File set `Monaldo/Fjsp/Job_Data/Brandimarte_Data/Text/Mk01.fjs` to `Mk10.fjs` of the
  archive `TextData.zip` (FJSPLIB), linked from https://people.idsia.ch/~monaldo/fjsp.html
  (instance library of M. Mastrolilli, IDSIA). Downloaded on 2026-11-11; archive SHA-256
  `e6637b4f84e2ab188a0c90ce27062e8ee50f11f34334d12c07931227e474f62f`. The files are those
  of the archive, byte for byte (CRLF line ends kept), and `SHA256SUMS` lists their
  checksums.
- The copies in the dual-attention network release (github.com/wrqccc/FJSP-DRL,
  `data/BenchData/Brandimarte`) contain the same numbers, except that the header of Mk06
  there declares 10 machines instead of 15.
- Instances Mk11 to Mk15 of the original article are not in this archive.

## Citation

P. Brandimarte. Routing and scheduling in a flexible job shop by tabu search. Annals of
Operations Research, 41(3):157-183, 1993. doi:10.1007/BF02023073.

## Licence

No licence is stated in the archive or on the web page. Cite the article above and the
FJSPLIB page when the files are used. The repository does not redistribute the files.
`scripts/fetch_public_instances.py` downloads
https://people.idsia.ch/~monaldo/fjspresults/TextData.zip, checks it against the SHA-256
above and extracts these ten files (see "Obtaining the instance files" in
`data/PUBLIC/README.md`).

## Format

Standard `.fjs` text: first line `J M f` (jobs, machines, average number of eligible
machines; the third number is rounded and is not used); then one line per job with the
number of operations followed, for each operation, by the number of eligible machines k
and k pairs (machine, processing time), machines numbered from 1. Read with
`constraint_family/public_adapter.py` (`read_fjs`), which returns the arrays the
environment uses (`job_length [J]`, `op_pt [N, M]`, 0 = machine not eligible).

## Instances

| instance | jobs | machines declared | machines used | operations | operations per job | mean eligible machines | max eligible | processing times |
|---|---|---|---|---|---|---|---|---|
| Mk01 | 10 | 6 | 6 | 55 | 5-6 | 2.09 | 3 | 1-6 |
| Mk02 | 10 | 6 | 6 | 58 | 5-6 | 4.10 | 6 | 1-6 |
| Mk03 | 15 | 8 | 8 | 150 | 10-10 | 3.01 | 5 | 1-19 |
| Mk04 | 15 | 8 | 8 | 90 | 3-9 | 1.91 | 3 | 1-9 |
| Mk05 | 15 | 4 | 4 | 106 | 5-9 | 1.71 | 2 | 5-9 |
| Mk06 | 10 | 15 | 10 | 150 | 15-15 | 3.27 | 5 | 1-9 |
| Mk07 | 20 | 5 | 5 | 100 | 5-5 | 2.83 | 5 | 1-19 |
| Mk08 | 20 | 10 | 9 | 225 | 10-14 | 1.43 | 2 | 5-19 |
| Mk09 | 20 | 10 | 10 | 240 | 10-14 | 2.52 | 5 | 5-19 |
| Mk10 | 20 | 15 | 11 | 240 | 10-14 | 2.98 | 5 | 5-19 |

Jobs and declared machines agree with the published sizes (Behnke and Geiger 2012,
Table 4). In Mk06, Mk08 and Mk10 some declared machines are eligible for no operation;
the adapter keeps them (they never receive work and do not change any makespan) and
`PublicInstance.without_unused_machines()` removes them when wanted.

## Best-known values: `best_known.csv`

Columns: `instance, best_known_makespan, lower_bound_if_known, source, optimal, verified,
note, fjsplib_lb, fjsplib_ub, fjsplib_credit`.

- `best_known_makespan`, `lower_bound_if_known` and `source` come from D. Behnke and
  M. J. Geiger, Test instances for the flexible job shop scheduling problem with work
  centers, Research Report RR-12-01-01, Helmut-Schmidt-University Hamburg, 2012,
  Table 4. The table credits each upper bound to its original publication (named in
  `source`) and adds a 10-minute IBM ILOG CP Optimizer solution; the smaller of the two is
  taken. `verified = yes` means the number was read from that table; the credited original
  papers were not checked one by one.
- `optimal = yes` when the table marks the literature value optimal or the lower bound
  equals it; `claimed by CP Optimizer in the table` when only the CP run claims it.
- Mk01: the table lists an upper bound of 39 (Xing et al. 2010) and flags it as
  inconsistent with the CP-proved optimum 40; 40 is used.
- `fjsplib_*` columns: bounds listed by FJSPLib (scheduleopt.github.io/benchmarks/fjsplib,
  accessed 2026-11-11). They are newer (for example Mk06 = 57, Mk10 in [189, 193]) but
  credited to solver runs or a company web page, not to a publication, so they are marked
  unverified and are not used as best-known values.

## Public family

Descriptors for the eight regimes are under `data/PUBLIC/family/pub_Brandimarte`
(see `data/PUBLIC/README.md`).
