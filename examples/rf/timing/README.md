<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# RF clocks and timing

Choose the RF clock, account for arrival delays, then change the frequency or
the tracking clock. Each script runs independently.

| Example | Lesson |
| --- | --- |
| [001_absolute_time_cavities.py](001_absolute_time_cavities.py) | Compare relative and laboratory-time RF phases at different cavity locations |
| [002_time_delays_off_momentum.py](002_time_delays_off_momentum.py) | Apply a native time delay and find the resulting off-momentum synchronous orbit |
| [003_rf_frequency_steering.py](003_rf_frequency_steering.py) | Find the synchronous momentum and orbit after a small RF frequency change |
| [004_frequency_ramps_and_phase.py](004_frequency_ramps_and_phase.py) | Integrate a frequency program into a continuous RF phase |
| [005_synchronized_time_frames.py](005_synchronized_time_frames.py) | Advance shared frames and measure differing revolution frequencies at a native pickup |

Positive `zeta` means earlier arrival at a fixed location. The laboratory
arrival time represented by a particle is
`time_s + (s - zeta)/(beta0*c)`. `time_s` is the common reference clock,
rather than an array of individual arrival times. RF frequencies are in hertz;
`Cavity.phase` is in radians.

`002` changes arrival coordinates with a fixed `TimeDelay` per passage; `003`
changes the frequency of a laboratory-time cavity. These are distinct timing
models. Both use small perturbations of the shared electron demonstration ring;
their synchronous solutions do not establish momentum acceptance.

`004` compares actual kicks with the integral of a linear frequency chirp.
Settings update once per reference turn: the programmed phase is exact at the
update epoch and uses a tangent approximation at other arrival times.

`005` uses a drift-only proton loop to isolate clock scheduling. It has no
transverse confinement. Install the native synchronization boundaries after
construction and slicing. With `SyncTime`, `num_turns` counts complete frames,
`reference_turns` follows the common clock, and each particle's `at_turn` counts
its completed revolutions at its stored event. Stored events span frame windows;
they do not form a simultaneous laboratory snapshot at `time_s`.
Negative synchronization states represent waiting,
rather than losses. The frame window must move faster than every particle;
this example uses the CPU controller.

The final acquisition continues the same probes across 128 additional frames.
Three native coasting `BeamStatsMonitor` pickups separate the momentum classes
by particle ID and bin their actual passage times. `time_centers` supplies the
local pickup clock; its constant flight-time offset cancels in intervals.
Frequencies come from more than 100 detected passages per probe, rather than
from the stored frame counters. Endpoint-bin widths give a deterministic
frequency bound, not a statistical error bar or an FFT resolution claim.
The speed comparison `beta*c/C` applies to this fixed-path model; a magnetic
ring also changes orbit length with momentum.

For fixed-frequency motion and slipping RF systems, see
[longitudinal dynamics](../dynamics/README.md). For the single-element kick
conventions, see [cavities](../../elements/cavities/README.md).
