<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# PIMMS: mixed-species acceleration and resonant extraction

Accelerate a carbon bunch and an equal-rigidity proton bunch through the same
magnetic lattice and two RF systems. Follow their different passage frequencies
with species-selected pickups, then reconstruct the recorded energy evolution.
A separate proton configuration connects extraction optics, sextupole controls
and a programmed spill.

## Mixed-species acceleration

| Example | Lesson |
| --- | --- |
| [001_accelerate_and_record_pickups.py](001_accelerate_and_record_pickups.py) | Prepare the two species, programme the common rigidity and RF clocks, then save their pickup records |
| [002_plot_pickups_and_energy.py](002_plot_pickups_and_energy.py) | Compare initial/later passages, fold the pickup record and reconstruct complete-passage energy moments |

Run `001` before `002`. The producer writes
`outputs/004_pimm_carbon_protons.npz` relative to this folder; the plotter reads
that file without importing Xsuite or repeating the tracking. Its time, current,
longitudinal-moment and reference-programme fields supply the plotting inputs.

## Slow extraction

The [slow-extraction route](slow_extraction/README.md) uses three independently
runnable scripts: inspect the near-third-integer phase space; compare sextupole
controls of the bounded region and outgoing branch; then follow a programmed
spill at an ideal absorbing boundary. Its compact 200 MeV proton lattice has
distinct geometry details and strengths from the acceleration model below.
The native model is adapted from the Xsuite slow-extraction tutorial, with its
assumptions and source recorded beside the model. An aperture crossing is not a
prediction of extraction efficiency.

## Model and acceleration

The static lattice and strengths are in
[`test_data/pimms`](../../../test_data/pimms/). The native parser reads these
legacy-format data files without running MAD-X. This 75.24 m demonstration is
not an operational machine model.

The reference is fully stripped carbon-12 at 7 MeV/u. Equal magnetic rigidity
gives the protons about 27.5 MeV kinetic energy at injection. They share the
carbon reference definition through explicit mass and charge ratios; equal
rigidity corresponds to `delta = chi - 1`, with `chi` the charge ratio divided
by the mass ratio. Their speeds and revolution periods differ.

Carbon is programmed from 7 to 9 MeV/u over 256 **injection** carbon periods,
with a smooth initial energy-slope turn-on. Fixed normalized magnet strengths
represent physical fields rising with the common reference rigidity. Both beams
see both h=1 cavities, at 40 kV and 30 kV. Their chirped phases integrate the
frequency programmes; frequency multiplied by time would give a different phase.
The synchronized frame clock keeps particle arrival times distinct from the
reference clock. Inside each frame, the phase is continued at that frame's
frequency.

The initial Gaussian bunches use native transverse matching. Their longitudinal
preparation uses a small-amplitude match to each species' own RF system; it is
not a stationary solution of the combined two-RF potential. A horizontal hard
aperture at a high-dispersion station introduces geometric losses without
material interactions.

## Pickup interpretation

Here `coasting=True` selects full-period acquisition, although the beams are
bunched. Recorded rows index accumulated carbon **reference** turns, rather
than synchronization frames or each species' completed revolutions. Native
energy-programme time conversion supplies laboratory bin centres and widths.
The monitor's `num_particles` is weighted population; multiplying by species
charge and dividing by bin duration gives the ideal pickup current. Equal total
bunch charge makes the two traces comparable. Particle-ID filters separate the
species' contributions for analysis; a physical charge pickup would see their
sum.

The folded image shifts the continuous record, leaving acquisition ends blank.
Its arrival coordinate is positive for later arrival, opposite to `zeta`.
Energy moments combine complete passages using each species' own RF phase and
include both within-slice spread and variations of slice means. The conversion
uses the recorded mean reference momentum within each slice; a slice spanning a
reference update makes both the energy mean and RMS spread approximate. The
programmed energy curve is a reference comparison, not a constraint on the
tracked bunch mean.

The default producer uses 1,000 macroparticles per species and serial CPU
tracking. The plotter reads the saved acquisition without repeating tracking.
Sampling and the tracking margin affect the recorded details.
For the individual APIs, see [acceleration](../../rf/acceleration/README.md),
[RF timing](../../rf/timing/README.md) and
[beam statistics](../../monitors/beam_statistics/README.md).
