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

These top-level scripts use the
[native teaching models](../../test_data/example_lattices/README.md) with their
1 GeV/c electron references and 4D optics. There is no RF, radiation or collective
effect. Ring tune targets are full tunes, including the integer part.

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

`004` adds three independent thin dipole correctors and zero-length boundary
markers to a fresh ring load. It fixes the clean entrance trajectory, requests
a 1 mm horizontal displacement at one selected point, and restores both `x`
and normalized canonical `px` at the downstream boundary. The full periodic
orbit is recomputed to check the deviation outside the bump. The selected
displacement is not its peak height, and beta functions are not constrained.
The positive-kick knob uses `knl[0] = -kick` in the Multipole convention.

`005` first records a reference, gives two magnets individual strength knobs,
then perturbs those magnets. `Target(value=reference_table, at=...)` takes the
named scalar values from that table. The example restores local horizontal
beta and alpha; it does not request whole-table or full-lattice recovery.

`006` combines a local beta-ratio equality with lower and upper tune targets.
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

`010` adds zero-length sextupole families at dispersive quadrupole entrances,
then matches `dqx` and `dqy` with the quadrupoles held fixed. Separate
off-momentum Twiss calculations check the derivatives at a different step.
Higher-order chromaticity and dynamic aperture are not targets; see
[nonlinear optics](../optics/nonlinear/) for momentum-dependent analysis.

`011` gives eight quadrupoles independent skew components. It requests one
nonzero interior cross-plane map term and closes four independent cross-plane
terms at the section exit. Recomputed periodic mode projections demonstrate
decoupling outside. This is a nonunique betatron-coupling bump at `delta=0`;
diagonal optics and dispersion are not restored by those constraints.

`012` uses two thin skew quadrupoles and a first-order coupling phasor with
weights from the uncoupled reference optics. Two generated knobs superpose in
that fixed reference convention. Actual coupled Twiss results are recomputed
to show the difference between the proxy and the physical optics.

`013` adds fixed seeded quadrupole errors and separate correction trims. It
fits the trims to reference beta functions and tunes without varying the error
variables. This is an ideal noiseless, fully actuated correction: sparse
actuators, measurement noise and error identifiability need additional analysis.

For initialization and transport conventions, see [optics](../optics/).

For RF voltage, synchronous energy and phase targets, see
[longitudinal matching](longitudinal/README.md).

For correction using orbit-response matrices, see
the [errors and correction plan](../README.md#errors_and_correction).
