# Prepared FCC runtime: measured costs and remaining failures

The [review](../../../designs/2026-11-20_prepared_runtime_review.md) explains the
model, freeze/clock contracts, numerical evidence and next architecture.
[Cost figure](prepared_costs.svg). All measurements below use the unmodified
parameterized thick FCC Z model pinned in the earlier
[input record](../fcc_scaling/README.md). Engineering girder assignments remain
provisional. This is serial CPU, not a measured GPU result.

Each JSON includes script/native Python/C-header/tracker hashes, compiler and
thread settings, sample timings, payload sizes, all-particle survival and the
explicit twofold target. Matching stdout/stderr files are retained beside it.
All five cases were run sequentially after correctness/example jobs completed.
Medians use three 128-turn calls. Dynamic call timing includes packed entry
validation, static-setting checks and final ownership bookkeeping; first kernel
compilation/preparation is separated below. Small ratios below 1 reflect timing
scatter, not a physical speedup claim.

| Workload | Bodies | Update only [ms] | 1-particle tracking ratio | 32-particle tracking ratio |
| --- | ---: | ---: | ---: | ---: |
| [Quad sites, six analytic coordinates](quads_cpu.json) | 2,663 | 0.218 | 1.165× | 0.996× |
| [All magnets, six analytic coordinates](all_magnets_cpu.json) | 12,041 | 0.864 | 1.806× | 1.085× |
| [Support/magnet hierarchy + residuals](all_magnets_cpu_nested.json) | 16,055 | 1.035 | 1.884× | 1.089× |
| [Joint ground + three response classes, sampled](quads_cpu_sampled_ground.json) | 2,663 | 0.181 | 1.189× | 1.020× |
| [Direct continuous field at anchors](quads_cpu_direct_ground.json) | 2,663 | 4.792 | 2.740× | not measured |

**The twofold gate is not universally met.** Direct continuous spatial evaluation
fails even with 128-turn calls. Warm one-turn calls fail because entry guards and
bookkeeping are not amortized; the all-magnet 16-turn case also fails.

| Workload | Turns/call, 1 particle | Ratio | Twofold ceiling |
| --- | ---: | ---: | --- |
| quads_cpu | 1 | 4.848× | **fail** |
| quads_cpu | 4 | 2.070× | **fail** |
| quads_cpu | 16 | 1.401× | pass |
| all_magnets_cpu | 1 | 9.203× | **fail** |
| all_magnets_cpu | 4 | 3.536× | **fail** |
| all_magnets_cpu | 16 | 2.219× | **fail** |

| Workload | Public body definition [s] | Source preparation [s] | Cold runtime preparation [s] |
| --- | ---: | ---: | ---: |
| quads_cpu | 3.716 | 0.002 | 1.853 |
| all_magnets_cpu | 4.298 | 0.011 | 2.659 |
| all_magnets_cpu_nested | 5.447 | 1.638 | 2.942 |
| quads_cpu_sampled_ground | 3.765 | 0.809 | 1.824 |
| quads_cpu_direct_ground | 3.698 | 0.000 | 1.801 |

The sampled-ground source is one joint field at 10,652 feet, 128 random modes,
128 samples at 2048 Hz, band 2–40 Hz, seed 956, correlation length 30 m, four-contact rigid
fits and three synthetic 8/12/18 Hz response classes. Filters start at zero. The
short record is explicit and resident on the host; only the current 6-coordinate
input packet is uploaded. This is a cost configuration, not a converged FCC
stationary noise model. Direct ground is a separate scalar-anchor shortcut with
no feet/class pipeline; its cost should not be presented as the same physics.

Source preparation matters: adding this short block's preparation to its
one-particle 128-turn tracking gives about **3.32×** the baseline, excluding
body definition and cold compilation. Fast consumption does not establish a
sub-twofold end-to-end streaming study. Fixed-foot projections, stationary
mode-domain responses or bounded streaming preparation remain necessary.

The per-update buffer-method counters are zero host scalar reads/writes, but
the sampled/custom path still has a real packed transfer. Array payload metadata
lists both host staging and context arrays by attribute; it is not a GPU-memory
measurement. No whole-run or bunch-by-whole-ring cache is allocated. Packed
bunch phases, arbitrary direct sources and actual GPU throughput are not covered
by a passing CPU row.

Reproduce from the Xtrack checkout, with source first on PYTHONPATH and one CPU
thread for these serial measurements:

```bash
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    python examples/mechanics/review/fcc_prepared_runtime.py --members quads --turns 128 --short-calls
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    python examples/mechanics/review/fcc_prepared_runtime.py --members all_magnets --turns 128 --short-calls
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    python examples/mechanics/review/fcc_prepared_runtime.py --members all_magnets --hierarchy nested --turns 128
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    python examples/mechanics/review/fcc_prepared_runtime.py --source sampled_ground --turns 128
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    python examples/mechanics/review/fcc_prepared_runtime.py --source direct_ground --turns 128 --particles 1
python examples/mechanics/review/fcc_prepared_plots.py
```

Replace CPU with an explicitly available `--context cuda` or `--context opencl`
to measure that hardware; no automatic CPU substitution is used. Local macOS
compiler overrides were CC=/opt/miniconda3/envs/xsuite/bin/clang and
LDSHARED='/opt/miniconda3/envs/xsuite/bin/clang -bundle -undefined dynamic_lookup'.
Those environment repairs are not portable model requirements. CUDA sampled
uploads explicitly finish consuming reusable host staging; a bounded pinned
staging/event optimization can be considered after real hardware measurements.
