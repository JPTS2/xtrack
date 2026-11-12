<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# Benchmarks

Measure execution cost, numerical convergence and statistical precision.
These studies run deliberately chosen workloads and are excluded from ordinary
example execution. Each guide links back to the corresponding subject lessons.

| Folder | Question |
| --- | --- |
| [performance/](performance/README.md) | How does cost change with workload, configuration and execution context? |
| [convergence/](convergence/README.md) | Which error decreases on refinement, and what limits further improvement? |
| [statistics/](statistics/README.md) | How precisely does a finite stochastic sample determine an observable? |

All studies use native Xtrack operations. TPSA uses `madng_tpsa`/GTPSA algebra but
does not run an external tracking code. Ring studies reuse the
[demonstration model](../../test_data/example_lattices/demo_ring/README.md);
single-element studies state their fields, coordinates and reference explicitly.

Parameters near the top select the workload and resolution. Timings are local
measurements: retain the configuration, package versions and hardware details
when comparing runs. Use an otherwise idle machine, repeat fresh-process runs,
and distinguish setup from warmed execution. Within-process timing spread does
not quantify all variation between machines or runs.

Convergence studies state their reference and refinement checks. Agreement with
a refined calculation is evidence within the stated model, rather than
independent validation of that model. Statistical studies show uncertainty and
distinguish pointwise from simultaneous confidence bands.
