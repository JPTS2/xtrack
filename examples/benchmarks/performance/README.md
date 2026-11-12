<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# Performance

Run these scripts separately on an otherwise idle machine. Each excludes an
initial warmup and reports repeated measurements. Tables or plots show timing
spread; no fixed speedup is an acceptance condition. Plotting uses Matplotlib.

| Example | Measured operation |
| --- | --- |
| [001_line_construction.py](001_line_construction.py) | Resolve anchored placements, then measure appends versus initial line size and in the stored HL-LHC model |
| [002_model_io.py](002_model_io.py) | Save/load one environment as JSON or gzip JSON, and rebuild it from its native Python input |
| [003_tracking_throughput.py](003_tracking_throughput.py) | Scan bounded particle counts at a fixed tracking horizon and execution context |
| [004_optics_runtime.py](004_optics_runtime.py) | Compare complete periodic 4D, chromatic 4D, periodic 6D and supplied-initial-condition open optics calculations |
| [005_tracking_optimization.py](005_tracking_optimization.py) | Compare an independent optimized snapshot with the source, including physical-coordinate agreement |
| [006_radiation_cost.py](006_radiation_cost.py) | Compare four radiation modes with the same RF, tapered fields and entrance particles |
| [007_tpsa_scaling.py](007_tpsa_scaling.py) | Separate input setup from scalar/TPSA tracking and scan map order on one nonlinear cell |
| [008_tpsa_workload_scaling.py](008_tpsa_workload_scaling.py) | Scan lattice length and independently active quadrupole parameters at fixed map order |

Construction excludes preparation of anchored components. Appends start from
fresh lines of 1, 1000 or 10000 drifts, and separately from the stored HL-LHC
model with its strength expressions. Construction or loading is outside the
append timer; marker creation and individual append calls are inside.
I/O includes serialization and buffered
file access with operating-system caches warm, without forcing writes to disk.
Native Python loading constructs the model, so its work differs from restoring
serialized objects.

Tracking uses the public synchronized `time_last_track` measurement. It includes
the tracking call's overhead and is not labelled pure device-kernel time.
Particle copies, host/device preparation and final inspection are outside those
timers. Default execution is serial CPU. Optional contexts require their backend
dependencies and hardware; an unavailable selection is not replaced with CPU.

The radiation-off comparison retains the compensated RF and tapered magnetic
fields. It is a fixed-input cost baseline rather than a periodic radiation-free
machine. Stochastic timing repeats copy initialized streams and repeat one
realization; they are not independent statistical samples.

TPSA tracking is experimental and CPU-only, using `madng_tpsa`/GTPSA algebra
without a MAD-NG process. Higher orders compute additional coefficients; scalar
transport and map transport are different workloads. Constant-part agreement
checks the common physical launch, not every map coefficient.

The workload scan varies length at fixed parameter count, then parameter count
at fixed length. Parameters drive individual quadrupole gradients; each response
is checked against a scalar finite difference. Descriptor creation, field
binding and fresh input setup are timed separately from transport. The unchanged
nominal strengths provide an independent scalar-orbit control for every case.

For the APIs, see [line construction](../../line_construction/README.md),
[tracking](../../tracking/README.md), [optics](../../optics/README.md),
[radiation](../../radiation/README.md) and
[contexts and memory](../../contexts_and_memory/README.md).
