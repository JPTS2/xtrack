<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# PSB injection chicane and collapse

Follow a proton injection model through the chicane field, its programmed
collapse, optics feed-forward and multiturn injection. The same live circuits
are retained from the thick optics model into thin tracking.

Run these scripts in order. Each generated model is saved in `outputs/` beside
the scripts; the next stage reads it. Xtrack, NumPy, SciPy and Matplotlib are
used. The sequence and strength files are stored legacy data, loaded with the
native parser; no MAD-X executable is used.

| Example | Lesson |
| --- | --- |
| [001_injection_chicane.py](001_injection_chicane.py) | Connect the four bumper fields and induced sextupoles, then compare orbit and beta beating |
| [002_correct_chicane_collapse.py](002_correct_chicane_collapse.py) | Match tune and local vertical-optics feed-forward through the tabulated collapse; check interpolation between matched knots |
| [003_injection_tracking.py](003_injection_tracking.py) | Inject a finite incoming sample, turn the source off and follow the stored beam through the collapse |

The reference momentum is 0.570830155 GeV/c, approximately 160 MeV proton kinetic
energy. All four bumpers have horizontal shifts of −44.2 mm.

The first script separates dipole-field/edge-focusing effects from sextupole
feed-down. The stored sextupole programme represents an induced field; the
examples do not calculate eddy currents electromagnetically. The second script
restores both tunes and vertical beta/alpha at one location. It prints residuals
at independent times because interpolating matched circuits does not make the
optics exact between knots or restore every beta function around the ring.

The third script compares thick and thin tunes at three programme times before
tracking with two bend slices and eight quadrupole slices. This is a finite
integration approximation; the printed differences should be considered when
changing the working point. The incoming file supplies positions, slopes and
kinetic energy. It is placed directly in the stored coordinates, without
rematching to the displaced closed orbit; the resulting coherent mismatch is
visible against the instantaneous closed-orbit samples. Slopes are converted
to native momenta in the paraxial approximation. The example interprets the post-stripping circulating
particles as protons; it does not model charge exchange, foil scattering or space charge.
The native injection sampler draws a fresh random subset from the input each
turn without depleting the source. The simulation covers the full collapse,
but does not establish injection efficiency or aperture acceptance.

This route develops the physical workflow also used in the
[Xsuite PSB chicane tutorial](https://github.com/xsuite/tutorial_psb_injection_chicane).
For the individual APIs, see [matching](../../matching/),
[time-dependent tracking](../../tracking/time_dependent/) and
[particle preparation](../../particles/).
