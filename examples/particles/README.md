<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# Particles

Start with reference quantities and coordinates (`001`–`003`), then particle
preparation (`004`–`006`). Container operations follow in `007`–`010`; `011`
extends the reference conventions to mixed species. Each script runs
independently in your Xsuite environment.
`012` returns to preparation, extending the single-bunch RF recipe to a train
with gaps and a selected subset of its bunches.

| Example | Subject |
| --- | --- |
| [001_reference_quantities.py](001_reference_quantities.py) | Species, reference momentum, total/kinetic energy and signed rigidity |
| [002_coordinates.py](002_coordinates.py) | Linked energy coordinates, automatic updates, slopes and arrival time |
| [003_normalized_coordinates.py](003_normalized_coordinates.py) | Courant-Snyder coordinates, emittance-scaled launches and round trips |
| [004_gaussian_transverse_bunch.py](004_gaussian_transverse_bunch.py) | Build a weighted Gaussian bunch and check its physical beam sizes |
| [005_distribution_generators.py](005_distribution_generators.py) | Generate Gaussian, uniform-area, grid and pencil coordinates with Xpart, then build a physical launch |
| [006_matched_rf_bunch.py](006_matched_rf_bunch.py) | Prepare a Gaussian bunch matched to a linear RF map and compare it after tracking |
| [007_copy_particles.py](007_copy_particles.py) | Keep an initial launch while tracking independent copies through two strengths |
| [008_save_and_load_particles.py](008_save_and_load_particles.py) | Save coordinates, reference quantities, IDs and weights as a dictionary or JSON |
| [009_select_and_merge.py](009_select_and_merge.py) | Follow particle IDs while sorting, selecting and combining subsets |
| [010_capacity_and_insertion.py](010_capacity_and_insertion.py) | Reserve particle storage, append a population and distinguish lost histories from unused slots |
| [011_multispecies_kicks.py](011_multispecies_kicks.py) | Compare proton, deuteron and helium deflections at equal velocity or momentum |
| [012_multibunch_train.py](012_multibunch_train.py) | Prepare a matched train with gaps; distinguish RF buckets, physical filling slots and compact bunch selection |

The reference and coordinate examples use a 1 GeV/c proton reference.
`003`, `004` and `005` use the
[shared demo ring](../../test_data/example_lattices/demo_ring/README.md), with its
1 GeV/c electron reference and 4D optics. RF voltage is zero, with no RF
confinement: the imposed longitudinal Gaussian in `004` is not a matched RF
distribution. A launch scan is not a statistical bunch, and a finite random
sample does not have exact target moments. `004` prepares the prescribed
distribution without qualifying its off-momentum stability or tracking it.

`003_normalized_coordinates.py` distinguishes unscaled Courant-Snyder
(Floquet) coordinates from emittance-scaled coordinates. Omitting emittances
means unit geometric scaling: `gemitt_x = 1` and `gemitt_y = 1` m. This
convention does not specify a physical beam emittance. In this uncoupled example,
the unscaled transverse coordinates have units sqrt(m), with radians treated as
dimensionless.

`011_multispecies_kicks.py` uses tabulated nuclear masses and charges for bare
protons, deuterons and helium-4 nuclei. A single thin dipole kick illustrates the
species normalization. No fragmentation model is involved.

`008` writes `outputs/particles.json` beside the script. The snapshot contains
particle data; save the associated line separately for a tracking restart.
Default particle serialization includes stored RNG state. `005` and `006` use
Xpart's distribution helpers, which consume NumPy's seeded legacy random stream.

`006` uses its own ideal one-turn map with a 3 GeV/c proton reference, assigned
transverse optics and linearized RF focusing. Its momentum compaction places it
above transition; this map mode assumes stable focusing, rather than choosing
stability from an RF phase. Xpart samples Gaussian `delta`; the map rotates
`pzeta`, which equals `delta` only to first order. Matching is therefore to
linear, small-spread accuracy, without a nonlinear bucket or separatrix.
No acceleration, radiation or collective force is included. Finite random samples
have fluctuating moments even when drawn from a matched distribution.

`012` uses the same ideal RF model. A filling slot spans two RF buckets; the
ten-slot occupancy array covers the ring. Positive slot numbers have negative
`zeta`. Xpart's `bunch_selection` indexes the ordered list of occupied slots,
whereas monitor `selected_slots` names physical slots. Selecting bunches creates
a fresh sample rather than filtering an existing particle container. See
[bunch statistics](../monitors/beam_statistics/003_bunch_statistics.py) for
recording the train, and
[bunch-dependent beam-beam optics](../optics/collective/002_bunch_dependent_beambeam_optics.py)
for a filling-dependent force model.

These recipes cover Xtrack particle containers and Xpart preparation for a line.
`010` concerns container storage and insertion, without an interaction model.
For creating particles from a Python tracking callback, see
[custom interaction processes](../elements/custom/004_python_interaction_process.py).
Continue with [tracking](../tracking/) and [particle records](../monitors/).
For a reference controlled by model expressions, see
[named reference particles](../line_construction/013_reference_particles.py).
