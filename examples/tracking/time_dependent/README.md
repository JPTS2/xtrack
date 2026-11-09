<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# Time-dependent tracking

Change machine settings during tracking, then apply a sampled transverse drive.
Each script loads a fresh model and runs independently.

| Example | Lesson |
| --- | --- |
| [001_time_dependent_knobs.py](001_time_dependent_knobs.py) | Drive a live strength expression from the simulation clock and inspect the applied settings |
| [002_excitation.py](002_excitation.py) | Apply a sampled transverse excitation with a defined turn and arrival-time convention |

Both use the [shared demonstration ring](../../../test_data/example_lattices/demo_ring/README.md)
with its 1.5 km circumference and 1 GeV/c electron reference. Reference momentum
and geometry stay fixed. RF voltage and nonlinear strengths are zero; radiation
and collective forces are absent.

`001` enables machine-variable updates: the particle container's `time_s`
drives `t_turn_s` and expressions depending on it. These are simulation times
in seconds, not wall-clock runtimes. At the end of a complete turn, `time_s`
has advanced to the next entrance while `t_turn_s` retains the time used for
the last applied machine setting. Tracking continues the particle clock across
calls. Disabling machine-variable updates does not stop that clock.
The example requests updates at every turn entrance and holds each setting
through that turn; it does not vary the fields continuously inside a turn.
The 0.5% focusing-family ramp uses a small momentum offset `delta=1e-5`.
Its sampled optics remain stable; the bare ring's large chromaticity and nearby
vertical integer make larger offsets unsuitable for this ramp. This launch
does not measure momentum acceptance.

`002` uses a different mechanism. `Exciter` selects waveform samples from
`(at_turn - start_turn)/frev - zeta/(beta0*c)` at the element. It does not need
`enable_time_dependent_vars` and does not read `time_s`. Positive zeta means
earlier arrival. The waveform is referenced to the reference particle's arrival
at the exciter on `start_turn`, rather than a global laboratory RF phase.

Samples are stored as float32 and selected without interpolation; the sample
index is converted to an integer by truncation toward zero. The waveform can
repeat within its configured duration. This is prescribed transverse excitation,
rather than an RF acceleration model.

The finite cosine drive compares near-resonant and detuned responses. Its
coasting particle starts half a sample late to avoid sample-boundary ambiguity;
the addressed samples are reconstructed using the tracked zeta.
With eight samples per turn, that arrival offset is `zeta=-93.75 m` in this ring;
it specifies a coasting particle's sampling phase, not a bunch length.
Drive frequencies are stated in cycles per reference turn; observations alias
frequencies modulo the revolution frequency. The finite envelope does not
guarantee adiabatic switch-off or define a stability threshold.

For the tracking path and counters, see [core tracking](../README.md).
Laboratory RF clocks and acceleration have separate routes in the
[RF plan](../../README.md#rf).
