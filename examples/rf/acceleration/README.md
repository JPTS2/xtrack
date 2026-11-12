<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# Acceleration and reference momentum

Coordinate physical RF kicks with the reference used for tracking. Each script
runs independently.

| Example | Lesson |
| --- | --- |
| [001_ring_energy_ramp.py](001_ring_energy_ramp.py) | Coordinate a native energy programme with RF settings and follow the physical particle energy |
| [002_linac_energy_gain.py](002_linac_energy_gain.py) | Distinguish an accelerating cavity kick from a reference-momentum change |
| [003_linac_optics_and_tracking.py](003_linac_optics_and_tracking.py) | Propagate open 6D optics through acceleration and compare native beam sizes with tracking |
| [004_mixed_species_reference_clock.py](004_mixed_species_reference_clock.py) | Accelerate two species with one reference clock and two laboratory-time RF systems |

Changing reference momentum updates the coordinate description while preserving
physical particle energy. RF cavities provide physical energy gain. A reference
programme alone does not accelerate the beam.

`001` uses the shared 1 GeV/c electron
[demonstration ring](../../../test_data/example_lattices/demo_ring/README.md).
Native cavity harmonic mode follows the changing reference revolution frequency.
Fixed normalized magnet strengths represent magnetic fields following reference
rigidity. The RF-off control retains its physical energy despite the ramped
reference. Radiation is off; the small ramp is not an acceptance study.

The second part of `001` reuses that geometry for 64 protons accelerating from
0.5 towards 0.6 GeV/c. A separate tune match moves this illustrative setting
away from the bare near-integer working point. The 300 kV RF uses the focusing
phase below transition; the normalized magnet strengths remain fixed while
physical fields follow rigidity. A native matched Gaussian bunch is tracked
for 512 turns. Its final reference momentum is about 0.593 GeV/c: the programme
continues beyond the recorded interval.

Seventeen optics snapshots use the current applied reference to recover native
normalized coordinates. This diagnostic uses instantaneous 4D transverse
optics, suppressing RF energy kicks only in that optics calculation; the actual
tracking retains them. Slow reference changes and weak transverse-longitudinal
coupling permit an approximate adiabatic-action comparison. Centered action
moments remove coherent centroid motion. They remain approximately constant
while the phase-averaged betatron sizes shrink by about 8%, consistent with
inverse square-root beta-gamma scaling. Raw sample rms sizes also include
dispersion and finite-sample phase effects; they are plotted separately. This is a bounded linear-matching demonstration, not
an acceptance study or a precision emittance-conservation result.

The LINAC examples describe ideal cavity/transport maps. `003` retains the
entrance reference momentum: acceleration appears in the tracked and Twiss
`delta`, while canonical transverse momenta retain that common normalization.
It uses open **6D** Twiss; 4D Twiss freezes RF energy kicks.

`004` uses Fe-56 (25+) and Bi-209 (68+) probes in a constant-path drift loop,
without transverse confinement or collective forces. The native database
approximates the Bi-209 mass as 209 atomic mass units. Energies are per ion.
Equal rigidity requires `delta = chi - 1`, with
`chi = charge_ratio/mass_ratio`, rather than zero `delta` for both species.
Both probes see both RF systems; the equal-rigidity design curves are not
independent exact solutions of that combined field.

The mixed-species programmes update once per complete frame and use a tangent
RF phase within each frame. Particle event times can differ from the common
clock. For the clock conventions and frame-window requirements, see
[RF timing](../timing/README.md); for RF motion at fixed reference momentum, see
[longitudinal dynamics](../dynamics/README.md).
