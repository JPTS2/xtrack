<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# Errors and correction

Introduce alignment and field errors, then steer injected trajectories and
closed orbits with the native response-based correction tools. The ring examples
use the shared [demonstration ring](../../test_data/example_lattices/demo_ring/README.md);
`011` isolates 3D alignment on one bend. Each script runs independently.

| Example | Lesson |
| --- | --- |
| [001_alignment_errors.py](001_alignment_errors.py) | Apply transverse magnet offsets and rolls; inspect orbit changes, coupled modes and surveyed element frames |
| [002_multipole_errors.py](002_multipole_errors.py) | Set random or shared expression-based relative multipoles; distinguish absolute errors, live coefficients and normal/skew main-strength references |
| [003_closed_orbit_correction.py](003_closed_orbit_correction.py) | Steer the periodic orbit at named BPMs and inspect the applied kicks |
| [004_trajectory_correction.py](004_trajectory_correction.py) | Steer a ring section as a transport line, keeping entrance coordinates fixed |
| [005_micado_and_selected_correctors.py](005_micado_and_selected_correctors.py) | Compare automatic sparse corrector selection with restricting the available hardware |
| [006_response_and_svd.py](006_response_and_svd.py) | Inspect the response spectrum and compare singular-value truncations |
| [007_threading.py](007_threading.py) | Extend a corrected injected pass through the ring, then refine the periodic orbit |
| [008_bpm_alignment.py](008_bpm_alignment.py) | Distinguish physical positions from shifted and rolled BPM readings, including steering reset and threading |
| [009_off_momentum_correction.py](009_off_momentum_correction.py) | Correct one fixed momentum offset and inspect the on-momentum tradeoff |
| [010_corrector_limits.py](010_corrector_limits.py) | Apply accumulated steering bounds and compare the residual with an unbounded solve |
| [011_three_dimensional_alignment.py](011_three_dimensional_alignment.py) | Apply shifts and rotations about an anchor on a bend; connect native survey frames with exit coordinates and arrival time |
| [012_optics_response_and_svd.py](012_optics_response_and_svd.py) | Build a scaled beta/tune response, inspect its singular values and iteratively correct fixed quadrupole errors |

The reference is a 1 GeV/c electron. RF voltage and radiation are zero; correction
uses 4D optics with fixed momentum. The orbit/trajectory lessons use a nominal response calculated before applying
errors; `012` remeasures its optics response after each correction. Magnet rolls in `001` demonstrate coupling; the correction
lessons use nominal uncoupled responses and small perturbations.

`001` also locates a misaligned magnet in the reference survey and shows its
actual entrance frame. The transverse-frame plot retains true offsets and
enlarges only the drawn roll, with the factor labelled. Terminal output reports
the actual surveyed offsets and angle. See
[element frames](../geometry_and_survey/004_element_frames.py) for more detail.

The ring has fourteen ideal BPMs and nine correctors per plane. Steering reduces
BPM residuals; it cannot generally centre the orbit everywhere. The response has
nearly null modes, so the full-device solves use an explicit singular-value
cutoff. `006` shows the spectrum and distinguishes truncating response modes from
selecting hardware.

Relative multipole coefficients multiply the signed main integrated strength.
They are not reference-radius harmonic coefficients; their units depend on
multipole order. `002` inspects integrated strengths and their scaling, rather
than fringe-map effects. Field integrals and fringe activation are separate
settings.

In `002`, relative coefficients can be driven by shared expressions without
building expressions for the main-field integrals. Selecting `main_is_skew`
changes the reference strength; it does not rotate the magnet. The comparison
uses an independent copy, leaving the ring's main-field convention unchanged.

`011` applies all six alignment coordinates at an anchor measured in metres
along the reference arc. Its pure alignment roll leaves the reference geometry
unchanged. Native survey frames locate the displaced element; tracking shows
the corresponding exit-coordinate and arrival-time changes. Accumulated `s`
remains the design coordinate, rather than the actual distance travelled.

Threading imposes an injected entrance condition, not periodic closure. BPM
alignment changes the coordinates being steered towards zero. Correcting one
momentum does not correct dispersion. Corrector bounds apply to accumulated
correction kicks; the model starts with zero independent steering settings.

For correction of beta beating and tunes with quadrupole trims, see
[optics-error matching](../matching/013_correct_optics_errors.py). For the effects
of offsets and rotations on element tracking, see
[element transformations](../elements/transformations/README.md).

`012` separates fixed quadrupole errors from independent trim controls. Its
measurement vector combines relative horizontal/vertical beta beating and tune
errors with explicit scales. Truncated SVD controls the retained response modes;
two model-based correction iterations do not establish robustness to measurement
noise, calibration errors or a real machine's available correction circuits.
