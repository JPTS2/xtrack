<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# Matching

Start with periodic tune or open-transport matching. Then choose a physical
application or continue through optimizer controls and custom observables.
Every script runs independently; no generated setting is required by another.

## Optics and orbit applications

| Example | Lesson |
| --- | --- |
| [001_match_tunes.py](001_match_tunes.py) | Match two full periodic tunes and independently recompute the result |
| [002_match_transport.py](002_match_transport.py) | Match exit beta functions with fixed entrance optics |
| [004_orbit_bump.py](004_orbit_bump.py) | Match a four-corrector bump with entrance, downstream, interior and table boundary conditions |
| [005_targets_from_twiss.py](005_targets_from_twiss.py) | Combine reference-Twiss targets with variable and element-field constraints |
| [010_match_chromaticity.py](010_match_chromaticity.py) | Match global chromaticities and local IP chromatic responses |
| [011_coupling_bumps.py](011_coupling_bumps.py) | Excite coupling inside an insertion and close both cross-plane transport blocks outside |
| [012_coupling_knobs.py](012_coupling_knobs.py) | Calibrate native coupling controls and inspect combinations and correction |
| [013_correct_optics_errors.py](013_correct_optics_errors.py) | Infer quadrupole trims with nonlinear beta/tune matching |
| [014_match_demo_ring.py](014_match_demo_ring.py) | Derive nominal arc/straight-section strengths from the demonstration geometry |
| [018_match_with_aperture_free_twin.py](018_match_with_aperture_free_twin.py) | Match on a shared optics twin while retaining the physical aperture for tracking |
| [019_match_coupled_orbit_bump.py](019_match_coupled_orbit_bump.py) | Use six correctors to close leakage into the other plane through a fixed skew source |

Most ring lessons use the [demonstration ring](../../test_data/example_lattices/demo_ring/README.md)
with 1 GeV/c electrons, zero RF voltage and radiation off. The arc-family
controls leave insertion settings fixed. Tune targets include integer parts.
`002` retains its distinct open transport. The twin lesson constructs a small
proton transport so that sharing and aperture transmission remain visible.

Orbit closure, optics restoration and nonlinear acceptance are different
objectives. Each example prints the quantities it targets and the residuals it
leaves unconstrained. Numerical variable bounds are not hardware limits.
The six-corrector example compares vertical closure alone with closure in both
planes; a vertical orbit in a skew field drives a horizontal response.

## Optimizer tools and derived observables

| Example | Lesson |
| --- | --- |
| [003_optimizer_control.py](003_optimizer_control.py) | Inspect history, clone constraints, match relative phase and restore settings |
| [006_inequalities_and_callables.py](006_inequalities_and_callables.py) | Use inequalities, freeze targets and compare weighted allocations |
| [008_custom_actions.py](008_custom_actions.py) | Match named outputs of a cheap custom optics Action |
| [009_generate_knobs.py](009_generate_knobs.py) | Compose tune knobs with nonzero origins and inspect calibration/reset |
| [016_match_tracked_detuning.py](016_match_tracked_detuning.py) | Match actual octupoles against tracking-derived detuning and remeasure independently |
| [017_correct_coupling_integrals.py](017_correct_coupling_integrals.py) | Cancel a fixed skew source using named/callable first-order integrals and a generated knob |

Custom Actions expose observables from a calculation; the optimizer reevaluates
them as controls change. `008` uses a stated longitudinal quadrature for average
beta, with identical sampling for reference and trial optics. `016` requires
optional **nafflib** and uses finite-amplitude tracked slopes. Its longer-record,
smaller-launch check addresses sampling bias rather than proving a zero-action
normal form. This cost/dependency does not gate the introductory Action example.

Integral cancellation in `017` holds its supplied optics fixed and resolves
thick sources by slicing. It is first order and differs from the direct map
constraints in `011`; neither implies restoration of every diagonal optical
function. [Optics response and SVD](../errors_and_correction/012_optics_response_and_svd.py)
shows an explicit measured linear response/rank/cutoff calculation, separate
from nonlinear Twiss matching in `013`.

## Shared constraints and longitudinal matching

| Example | Lesson |
| --- | --- |
| [007_match_multiple_lines.py](007_match_multiple_lines.py) | Match independent ring configurations whose strengths depend on common variables |
| [015_match_shared_transport.py](015_match_shared_transport.py) | Match two transport routes containing one genuinely shared quadrupole |
| [Longitudinal matching](longitudinal/README.md) | Match RF focusing, prescribed synchronous state or frequency from an orbit measurement |

Sharing a variable between independent magnet copies differs from sharing an
element object. `007` constrains three tunes with three variables; the fourth
tune is free. `015` instead has one physical common quadrupole and two
forward-going routes with the same rigidity and axes. Counter-rotating physical
field conventions belong with the [LHC workflow](../machine_studies/lhc/README.md).

For the shorter first pass, see [getting started](../getting_started/05_match_tunes.py).
