<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# Matching

Match periodic and transport optics, control the optimizer, then build shared
objectives and reusable knobs. Each script runs independently.
The [getting-started tour](../getting_started/05_match_tunes.py) offers a shorter
first pass.

| Example | Lesson |
| --- | --- |
| [001_match_tunes.py](001_match_tunes.py) | Match two full periodic tunes and independently recompute the result |
| [002_match_transport.py](002_match_transport.py) | Match exit beta functions with fixed entrance optics |
| [003_optimizer_control.py](003_optimizer_control.py) | Inspect history, clone constraints, match relative phase and restore tagged settings |
| [004_orbit_bump.py](004_orbit_bump.py) | Match IP height/slope with four correctors, boundary variants and a shared optics twin |
| [005_targets_from_twiss.py](005_targets_from_twiss.py) | Combine Twiss-table targets with variable and element-field constraints |
| [006_inequalities_and_callables.py](006_inequalities_and_callables.py) | Combine inequalities/callables, freeze targets and allocate weighted controls |
| [007_match_multiple_lines.py](007_match_multiple_lines.py) | Match configurations with common settings, then routes with a shared magnet |
| [008_custom_actions.py](008_custom_actions.py) | Match cheap optics averages and tracking-derived magnetic detuning |
| [009_generate_knobs.py](009_generate_knobs.py) | Compose physical tune knobs at nonzero origins and inspect calibration/reset |
| [010_match_chromaticity.py](010_match_chromaticity.py) | Match global chromaticities, then local IP responses with distributed sextupole trims |
| [011_coupling_bumps.py](011_coupling_bumps.py) | Match an exact-map bump, then correct a separate source using first-order integrals |
| [012_coupling_knobs.py](012_coupling_knobs.py) | Calibrate knobs against native coupling targets, inspect combinations and correct a skew error |
| [013_correct_optics_errors.py](013_correct_optics_errors.py) | Fit quadrupole trims to reference beta functions and tunes |
| [014_match_demo_ring.py](014_match_demo_ring.py) | Derive the demonstration ring's arc and straight-section strengths and save a setting |

The ring lessons load the same [demonstration geometry](../../test_data/example_lattices/demo_ring/)
with a 1 GeV/c electron reference, RF voltage zero, radiation off and 4D optics.
The nominal full tunes are approximately 8.7548 and 10.01745; the tune-matching
examples request nearby values 8.760 and 10.020. The arc family controls leave
final-focus and utility-straight settings fixed. Tune targets include the integer
part. `002` retains its distinct open transport, while `014` derives the demo
operating strengths from geometry alone.

Start with `001` for periodic tunes or `002` for transport optics. Then choose
the physical task: `004` for an orbit bump, `005` for local optics targets,
`010` for chromaticity, `011`–`012` for coupling, or `013` for optics correction.
The optimizer and reusable-control route is `003`, `006`–`009`, then `012`.
These capabilities are useful tools rather than prerequisites for every task.

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

`004` uses four independent horizontal correctors around the low-beta IP,
including one inserted at 230 m in this example. It requests 0.1 mm displacement
and canonical `px = 2e-6` at the IP, and closes both coordinates outside
160–250 m. Entrance, downstream, interior and table boundary formulations start
from fresh zero kicks and give the same solution. A shallow optics twin uses a
marker where the physical line retains a deliberately offset aperture; shared
corrector objects and variables transfer the match to tracking. The unsteered
particle is lost and the matched one survives. Other optics are not constrained.

`005` first records a reference, gives two magnets individual strength knobs,
then perturbs those magnets. `Target(value=reference_table, at=...)` takes the
named scalar values from that table. The example restores local horizontal
beta and alpha; it does not request whole-table or full-lattice recovery.
Additional bounds use `env.vars.target` and `line.target`, exposing variable,
element-field and Twiss actions without an optics calculation for a pure
variable or field constraint.

`006` combines a beta-ratio equality at the passive `bpm.s1.f` plane with lower
and upper tune targets. The IP beta ratio is too insensitive to these arc-family
variations for this demonstration.
The inequalities have zero penalty inside the interval; they do not select a
unique tune or enforce hard feasibility during optimizer iterations.
Freeze changes a target to its current observable, not a magnet setting. Later
sections use smooth inequalities and compare underdetermined allocations:
larger `Vary.weight` favors that control in the scaled minimum-norm step;
`Target.weight` instead scales the residual.

`007` imports independent magnet copies into one environment. The focusing
family variable is shared; each configuration has its own defocusing variable.
Three independent controls satisfy three tune targets. The second configuration's
vertical tune is reported without being constrained.
Object identity checks make the independent magnets explicit: sharing a scalar
setting does not make them one physical element. The second model has two forward-going electron transport routes containing
the same entrance quadrupole object, followed by independent downstream optics.
A joint callable target restores the sum of their exit horizontal betas, along
with both exit betas of route a. Both routes use the same rigidity and local
axes, so positive k1 focuses x in both. The routing switch and global branch
geometry are omitted; this is a transport-optics model. Counter-circulating
common-magnet signs and collider IP constraints belong to the separate HL-LHC
machine workflow.

`008` returns two length-weighted beta averages from an `xt.Action`. These use
entrance values in a left-endpoint quadrature, not exact longitudinal integrals.
Matching their reference values does not constrain peak beta or beta beating.
The second action measures actual octupole-induced detuning using native
tracking and optional `nafflib`. A fresh DEMO is matched to 8.28/10.19 so its
real-signal estimator does not fold fractional tunes above 0.5. Four probes and
512 turns supply each action evaluation; a 1024-turn, smaller-amplitude check
qualifies the finite-action target tolerance. Slopes use geometric actions in
metres and are not a zero-action normal-form calculation.

`009` separates endpoint matching from binding the corrections to the new knob.
Magnet strengths interpolate linearly; tunes need not. Recomputed intermediate
settings show calibration leakage. Both knobs use nonzero physical tune origins
and add corrections to the same quadrupole-family expressions. Endpoints and
mixed settings are recomputed; returning both origins restores the original
strengths. Mixed calibration need not be exact.

`010` powers the existing dispersed arc sextupole circuits `k2lf` and `k2ld`,
then matches `dqx=1` and `dqy=2` with all quadrupoles held fixed. The bounded
solution has integrated strengths approximately +0.073 and -0.299 m^-2. Separate
off-momentum Twiss calculations check the derivatives at a different step.
The second stage adds independent trims to six existing, distributed sextupoles.
It reduces both IP chromatic beta responses by 10%, holds chromatic alpha and
both global derivatives, and checks the changes at half the chromatic step.
Targets are local IP quantities; the correction is not spatially confined.
No new sextupoles or canonical model edits are needed. Higher-order chromaticity
and dynamic aperture are not targets; see
[nonlinear optics](../optics/nonlinear/) for momentum-dependent analysis.

`011` gives eight existing final-focus insertion quadrupoles independent skew
components between 160 and 250 m. It requests one
nonzero interior cross-plane map term and closes four independent cross-plane
terms at the section exit. Recomputed periodic mode projections demonstrate
decoupling outside. This is a nonunique betatron-coupling bump at `delta=0`;
diagonal optics and dispersion are not restored by those constraints.
A separate fresh-source correction then uses native `IntegralOptimization`.
Sixteen slices resolve each selected skew source; named coupling integrals and
real/imaginary callables use fixed reference optics with live strengths. These
are dimensionless integral sums, not periodic RDT amplitudes. Generated-knob
reset and actual transport/mode projections qualify the first-order correction;
it has a different objective from the preceding requested interior bump.

`012` uses the installed `ks_s2` and `ks_s5` skew-quadrupole controls and native
Twiss targets `c_minus_re_0` and `c_minus_im_0`. The coefficients follow Twiss's
closest-tune-approach and entrance-phase convention; they are not a thin-source
resonance-driving-term sum. Each generated knob is calibrated at its endpoint.
Recomputed mixed settings expose departures from independent linear control.
A subsequent nonlinear match uses the generated knobs to cancel a fixed skew
error without varying that error. This is noiseless, fully actuated correction,
not identification of an unknown distributed error pattern.

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

To carry a chromaticity-corrected setting through off-momentum optics and actual
tracking, continue with the [demo-ring workflow](../machine_studies/demo_ring_workflow.py).
