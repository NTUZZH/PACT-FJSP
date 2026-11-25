# Hurink, Jurisch and Thole (1994) instances, vdata

## Source

- File set `Monaldo/Fjsp/Job_Data/Hurink_Data/Text/vdata/*.fjs` (66 files) of the archive
  `TextData.zip` (FJSPLIB), linked from https://people.idsia.ch/~monaldo/fjsp.html
  (instance library of M. Mastrolilli, IDSIA). Downloaded on 2026-11-11; archive SHA-256
  `e6637b4f84e2ab188a0c90ce27062e8ee50f11f34334d12c07931227e474f62f`. The files are those
  of the archive, byte for byte (CRLF line ends kept), and `SHA256SUMS` lists their
  checksums.
- The dual-attention network release (github.com/wrqccc/FJSP-DRL, `data/BenchData`)
  holds la01 to la40 of this set under the names `HurinkVdata4` to
  `HurinkVdata43`; their numbers are identical to these files.

## Citation

J. Hurink, B. Jurisch and M. Thole. Tabu search for the job-shop scheduling problem with
multi-purpose machines. OR Spektrum, 15(4):205-215, 1994. doi:10.1007/BF01719451.

The instances extend classical job-shop instances (the `sdata` set of the same archive)
by enlarging the set of eligible machines: in `vdata`, every operation can be assigned to several machines. An operation has the same
processing time on every eligible machine.

## Licence

No licence is stated in the archive or on the web page. The repository does not
redistribute the files. `scripts/fetch_public_instances.py` downloads
https://people.idsia.ch/~monaldo/fjspresults/TextData.zip, checks it against the SHA-256
above and extracts these 66 files (see "Obtaining the instance files" in
`data/PUBLIC/README.md`).

## Instances

Every job has M operations, so N = J x M.

| instances | jobs J | machines M | job-shop source |
|---|---|---|---|
| mt06 | 6 | 6 | Fisher and Thompson (1963) |
| mt10 | 10 | 10 | Fisher and Thompson (1963) |
| mt20 | 20 | 5 | Fisher and Thompson (1963) |
| la01-la05 | 10 | 5 | Lawrence (1984) |
| la06-la10 | 15 | 5 | Lawrence (1984) |
| la11-la15 | 20 | 5 | Lawrence (1984) |
| la16-la20 | 10 | 10 | Lawrence (1984) |
| la21-la25 | 15 | 10 | Lawrence (1984) |
| la26-la30 | 20 | 10 | Lawrence (1984) |
| la31-la35 | 30 | 10 | Lawrence (1984) |
| la36-la40 | 15 | 15 | Lawrence (1984) |
| abz5-abz6 | 10 | 10 | Adams, Balas and Zawack (1988) |
| abz7-abz9 | 20 | 15 | Adams, Balas and Zawack (1988) |
| car1, car2, car3, car4 | 11, 13, 12, 14 | 5, 4, 5, 4 | Carlier and Pinson (1989) |
| car5, car6, car7, car8 | 10, 8, 7, 8 | 6, 9, 7, 8 | Carlier and Pinson (1989) |
| orb1-orb10 | 10 | 10 | Applegate and Cook (1991) |

Mean number of eligible machines per operation: about M/2 (2.09 to 6.58 per group); maximum per operation: between 3 and 13.

`orb7.fjs` contains an operation with processing time 0 (job 10, operation 10). The
environment uses 0 as the "machine not eligible" code, so the adapter raises
`UnrepresentableInstance` for this file and the public family leaves it out (65 base
instances).

## Best-known values: `best_known.csv`

Columns: `instance, best_known_makespan, lower_bound_if_known, source, optimal, verified,
note, fjsplib_lb, fjsplib_ub, fjsplib_credit`.

- mt, la: D. Behnke and M. J. Geiger, Test instances for the flexible job shop scheduling
  problem with work centers, Research Report RR-12-01-01, Helmut-Schmidt-University
  Hamburg, 2012, Table 8. Upper bounds are credited there to Jurisch (1992),
  Dauzere-Peres and Paulli (1994, 1997) or Mastrolilli and Gambardella (2000); all lower
  bounds to B. Jurisch, Scheduling jobs in shops with multi-purpose machines, PhD thesis,
  University of Osnabrueck, 1992. The table also gives a 10-minute IBM ILOG CP Optimizer
  solution; the smaller of the two upper bounds is taken and `source` says which.
- abz, car, orb: the same report, Table 10 (CP Optimizer solution and a lower bound).
- `verified = yes` means the number was read from these tables; the credited original
  papers were not checked one by one. `optimal = yes` when the table marks the literature
  value optimal or the lower bound equals it; `claimed by CP Optimizer in the table` when
  only the CP run claims it.
- Conflicts are recorded in `note` (for example rdata la20, vdata abz5, edata orb7).
- `fjsplib_*` columns: bounds listed by FJSPLib (scheduleopt.github.io/benchmarks/fjsplib,
  accessed 2026-11-11). Many are tighter than the 2012 values, but most are credited to
  solver runs or web pages and not to a publication, so they are marked unverified and are
  not used as best-known values.

## Public family

Descriptors for the eight regimes are under `data/PUBLIC/family/pub_Hurink_vdata`
(see `data/PUBLIC/README.md`).
