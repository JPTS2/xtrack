<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# Matching

Match periodic and transport optics, control the optimizer, then build shared
objectives and reusable knobs. Read in order; each script runs independently.
The [getting-started tour](../getting_started/05_match_tunes.py) offers a shorter
first pass.

| Example | Lesson |
| --- | --- |
| [001_match_tunes.py](001_match_tunes.py) | Match two full periodic tunes and independently recompute the result |
| [002_match_transport.py](002_match_transport.py) | Match exit beta functions with fixed entrance optics |
| [003_optimizer_control.py](003_optimizer_control.py) | Step and inspect the optimizer, retarget active controls and reload tagged strengths |
| [004_orbit_bump.py](004_orbit_bump.py) | Match a selected orbit displacement and close both position and momentum with three correctors |
| [005_targets_from_twiss.py](005_targets_from_twiss.py) | Restore selected local optics using target values from a reference Twiss table |
| [006_inequalities_and_callables.py](006_inequalities_and_callables.py) | Combine a tune interval with a callable beta-ratio target |
| [007_match_multiple_lines.py](007_match_multiple_lines.py) | Match two configurations jointly with shared and independent variables |
| [008_custom_actions.py](008_custom_actions.py) | Supply derived observables through a custom Action |
| [009_generate_knobs.py](009_generate_knobs.py) | Match a tune-knob endpoint and measure its response between settings |
| [010_match_chromaticity.py](010_match_chromaticity.py) | Match first-order tune derivatives with two sextupole families |
| [011_coupling_bumps.py](011_coupling_bumps.py) | Open local betatron coupling and close the transverse cross-plane map |
| [012_coupling_knobs.py](012_coupling_knobs.py) | Build two superposable knobs for a fixed-reference coupling phasor |
| [013_correct_optics_errors.py](013_correct_optics_errors.py) | Fit quadrupole trims to reference beta functions and tunes |
| [014_match_demo_ring.py](014_match_demo_ring.py) | Derive the demonstration ring's arc and straight-section strengths and save a setting |

The ring lessons load the same [demonstration geometry](../../test_data/example_lattices/demo_ring/)
with a 1 GeV/c electron reference, RF voltage zero, radiation off and 4D optics.
The nominal full tunes are approximately 8.7548 and 10.01745; the tune-matching
examples request nearby values 8.760 and 10.020. The arc family controls leave
final-focus and utility-straight settings fixed. Tune targets include the integer
part. `002` retains its distinct open transport, while `014` derives the demo
operating strengths from geometry alone.

`001` demonstrates the immediate solve performed by default by `line.match`.
`003` defers the solve to inspect the initial state, take a fixed number of
iterations, then continue to convergence. It temporarily disables one target
and one control, changes the remaining target, then restores the original
problem and a tagged solution. Reloading history restores variable values and
logged enabled states; changed target values must be restored explicitly.
Its numerical family-strength bounds are not hardware limits or a guarantee
that every intermediate lattice is stable.
The finite-difference strength step has units m^-2; optimizer steps are iterations.

`002` fixes entrance beta and alpha functions and varies two family variables,
which control the transport's three physical quadrupoles. `xt.END` is the actual
line exit at 7.1 m. Matching its two beta functions does not also constrain alpha
functions or dispersion.

`004` binds three independent knobs to the existing insertion correctors and
adds zero-length boundary markers at 160 and 250 m. It fixes the clean entrance
trajectory, requests a 0.1 mm horizontal displacement at the IP, and restores `x`
and normalized canonical `px` at the downstream boundary. The full periodic
orbit is recomputed to check the deviation outside the bump. The selected
displacement is not its peak height, and beta functions are not constrained.
The positive-kick knob uses `knl[0] = -kick` in the Multipole convention.

`005` first records a reference, gives two magnets individual strength knobs,
then perturbs those magnets. `Target(value=reference_table, at=...)` takes the
named scalar values from that table. The example restores local horizontal
beta and alpha; it does not request whole-table or full-lattice recovery.

`006` combines a beta-ratio equality at the passive `bpm.s1.f` plane with lower
and upper tune targets. The IP beta ratio is too insensitive to these arc-family
variations for this demonstration.
The inequalities have zero penalty inside the interval; they do not select a
unique tune or enforce hard feasibility during optimizer iterations.

`007` imports independent magnet copies into one environment. The focusing
family variable is shared; each configuration has its own defocusing variable.
Three independent controls satisfy three tune targets. The second configuration's
vertical tune is reported without being constrained.

`008` returns two length-weighted beta averages from an `xt.Action`. These use
entrance values in a left-endpoint quadrature, not exact longitudinal integrals.
Matching their reference values does not constrain peak beta or beta beating.

`009` separates endpoint matching from binding the corrections to the new knob.
Magnet strengths interpolate linearly; tunes need not. Recomputed intermediate
settings show the horizontal calibration error and vertical tune leakage.

`010` powers the existing dispersed arc sextupole circuits `k2lf` and `k2ld`,
then matches `dqx=1` and `dqy=2` with all quadrupoles held fixed. The bounded
solution has integrated strengths approximately +0.073 and -0.299 m^-2. Separate
off-momentum Twiss calculations check the derivatives at a different step.
Higher-order chromaticity and dynamic aperture are not targets; see
[nonlinear optics](../optics/nonlinear/) for momentum-dependent analysis.

`011` gives eight existing final-focus insertion quadrupoles independent skew
components between 160 and 250 m. It requests one
nonzero interior cross-plane map term and closes four independent cross-plane
terms at the section exit. Recomputed periodic mode projections demonstrate
decoupling outside. This is a nonunique betatron-coupling bump at `delta=0`;
diagonal optics and dispersion are not restored by those constraints.

`012` binds thin skew-quadrupole components on `cor.s1` and `cor.ip.2` and uses
a first-order coupling phasor with
weights from the uncoupled reference optics. Two generated knobs superpose in
that fixed reference convention. Actual coupled Twiss results are recomputed
to show the difference between the proxy and the physical optics.

`013` uses eight existing arc-QD error/trim circuit pairs. Fixed seeded errors
have k1 RMS scale 1e-5 m^-2, and the independent trims start at zero. It
fits the trims to reference beta functions and tunes without varying the error
variables. This is an ideal noiseless, fully actuated correction: sparse
actuators, measurement noise and error identifiability need additional analysis.

For initialization and transport conventions, see [optics](../optics/).

`014` separates geometry from the operating strengths. It starts from declared
seeds, matches the arc cell and open half-straights, and exports a strength
file. The underdetermined straight matches also depend on their seeds, bounds
and continuation path; endpoint targets alone do not select a unique setting.

For RF voltage, synchronous energy and phase targets, see
[longitudinal matching](longitudinal/README.md).

For correction using orbit-response matrices, see
the [errors and correction plan](../README.md#errors_and_correction).
