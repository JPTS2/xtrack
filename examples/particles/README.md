<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# Particles

A numbered introduction to reference quantities, coordinates and particle
preparation. Each script runs independently in your Xsuite environment.

| Example | Subject |
| --- | --- |
| [001_reference_quantities.py](001_reference_quantities.py) | Species, reference momentum, total/kinetic energy and signed rigidity |
| [002_coordinates.py](002_coordinates.py) | Linked energy coordinates, automatic updates, slopes and arrival time |
| [003_normalized_coordinates.py](003_normalized_coordinates.py) | Courant-Snyder coordinates, emittance-scaled launches and round trips |
| [004_gaussian_transverse_bunch.py](004_gaussian_transverse_bunch.py) | Build a weighted Gaussian bunch and check its physical beam sizes |
| [005_select_and_merge.py](005_select_and_merge.py) | Follow particle IDs while sorting, selecting and combining subsets |
| [006_multispecies_kicks.py](006_multispecies_kicks.py) | Compare proton, deuteron and helium deflections at equal velocity or momentum |
| [007_copy_particles.py](007_copy_particles.py) | Keep an initial launch while tracking independent copies through two strengths |
| [008_save_and_load_particles.py](008_save_and_load_particles.py) | Save coordinates, reference quantities, IDs and weights as a dictionary or JSON |
| [009_distribution_generators.py](009_distribution_generators.py) | Generate Gaussian, uniform-area, grid and pencil coordinates with Xpart, then build a physical launch |
| [010_matched_rf_bunch.py](010_matched_rf_bunch.py) | Prepare a Gaussian bunch matched to a linear RF map and compare it after tracking |

The reference and coordinate examples use a 1 GeV/c proton reference.
`003`, `004` and `009` use the
[native magnetic ring](../../test_data/example_lattices/README.md), with its
1 GeV/c electron reference and 4D optics. These native-ring examples have no RF
confinement: the imposed longitudinal Gaussian in `004` is not a matched RF
distribution. A launch scan is not a statistical bunch, and a finite random
sample does not have exact target moments.

`003_normalized_coordinates.py` distinguishes unscaled Courant-Snyder
(Floquet) coordinates from emittance-scaled coordinates. Omitting emittances
means unit geometric scaling: `gemitt_x = 1` and `gemitt_y = 1` m. This
convention does not specify a physical beam emittance. In this uncoupled example,
the unscaled transverse coordinates have units sqrt(m), with radians treated as
dimensionless.

`006_multispecies_kicks.py` uses tabulated nuclear masses and charges for bare
protons, deuterons and helium-4 nuclei. A single thin dipole kick illustrates the
species normalization. No fragmentation model is involved.

`008` writes `outputs/particles.json` beside the script. The snapshot contains
particle data; save the associated line separately for a tracking restart.
Default particle serialization includes stored RNG state. `009` and `010` use
Xpart's distribution helpers, which consume NumPy's seeded legacy random stream.

`010` uses its own ideal one-turn map with a 3 GeV/c proton reference, assigned
transverse optics and linearized RF focusing. Its momentum compaction places it
above transition; this map mode assumes stable focusing, rather than choosing
stability from an RF phase. Xpart samples Gaussian `delta`; the map rotates
`pzeta`, which equals `delta` only to first order. Matching is therefore to
linear, small-spread accuracy, without a nonlinear bucket or separatrix.
No acceleration, radiation or collective force is included. Finite random samples
have fluctuating moments even when drawn from a matched distribution.

These recipes cover Xtrack particle containers and Xpart preparation for a line.
Continue with [tracking](../tracking/) and [particle records](../monitors/).
For a reference controlled by model expressions, see
[named reference particles](../line_construction/013_reference_particles.py).
