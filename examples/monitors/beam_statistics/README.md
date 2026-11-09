<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# Beam statistics

`BeamStatsMonitor` accumulates weighted population moments and coordinate
profiles. Read these standalone lessons in order; the diagnostic distributions
are small and explicitly prepared, without an RF-equilibrium assumption.

| Example | Lesson |
| --- | --- |
| [001_whole_beam_statistics.py](001_whole_beam_statistics.py) | Record whole-population moments and compare finite-sample covariance optics with their preparation model |
| [002_derived_statistics.py](002_derived_statistics.py) | Derive projected and normal-mode emittances from recorded covariance |
| [003_bunch_statistics.py](003_bunch_statistics.py) | Select physical slots, distinguish an empty slot and pool weighted moments |
| [004_slice_statistics.py](004_slice_statistics.py) | Record longitudinal slices and understand recorded-window reductions |
| [005_coasting_statistics.py](005_coasting_statistics.py) | Interpret periodic full-turn acquisition bins and the local arrival-time grid |
| [006_beam_profiles.py](006_beam_profiles.py) | Compare weighted histograms, population moments and conditional profile density |

All moments use `particles.weight`. `num_particles` is the sum of weights,
not a macroparticle count. RMS widths are population moments about the weighted
centroid, without a sample-estimator correction. Only accepted live particles
contribute. An empty bin returns zero for ordinary means and widths: mask it
using `num_particles > 0` before interpreting or plotting those quantities.

`001`, `002` and `005` use the shared 1 GeV/c electron
[demonstration ring](../../../test_data/example_lattices/demo_ring/), with RF
voltage zero and radiation off. `001` prepares independent longitudinal
coordinates and uncoupled transverse coordinates with first-order dispersion.
`002` instead prescribes its longitudinal spread independently, without the
dispersive preparation used in `001`. The fitted beta, alpha and dispersion
diagnostics in `001` describe the finite weighted
sample; they are not constrained to equal the lattice Twiss. Its momentum
spread is small and does not qualify the ring's momentum acceptance.

`002` measures one instantaneous population through the native monitor, then
rotates an independent copy and measures it again. Its covariance and projected
emittances are the lesson; no intervening ring transport is needed. The prepared
longitudinal spread gives full-rank covariance without an RF-equilibrium claim.

`003`, `004` and `006` acquire one explicitly prepared snapshot directly, without
lattice transport. Their slot, slice and profile semantics do not require a
matched bunch, a machine model or an RF system. Slot j is centered at
`zeta=-j*bunch_spacing_zeta`; this spacing is not automatically an RF bucket.
Only configured selected slots enter beam reductions. `003` includes a selected
but empty slot, as distinct from a slot excluded by the filling pattern.

In `004`, the local slice window is half-open. Bunch and beam reductions pool
only moments recorded inside the selected slots and slices; out-of-window
particles do not contribute. A pooled RMS includes centroid differences and
is not an average of the individual bunch or slice RMS widths.

In `005`, the arrival coordinate is `reference_turns-zeta/C`, referred to the
monitor. The nearest integer gives the record turn; the residual selects a
full-turn slice. Slice order follows increasing arrival phase and decreasing
zeta. Particles outside the central zeta interval can populate neighboring
records, giving partial edge frames. The unwrapped time helper uses the local
monitor clock, without adding flight time from the line entrance. The initial
coherent pattern is not asserted to be a collective eigenmode.

`006` records coordinate profiles and scalar moments simultaneously. Particles
outside a profile's coordinate range still contribute to accepted scalar
statistics and other profiles. Histogram counts sum particle weights; density
normalization is an explicit operation. The plotted density is conditional on
the recorded range and does not reconstruct its tails.

For persistence and chunked acquisition, see
[storage](../storage/README.md). For individual particle trajectories,
see [particle records](../particle_records/README.md).
