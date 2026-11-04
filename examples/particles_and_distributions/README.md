<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# Particles and distributions

A numbered introduction to reference quantities, coordinates and particle
preparation. Each script runs independently in your Xsuite environment.

| Example | Subject |
| --- | --- |
| [001_reference_quantities.py](001_reference_quantities.py) | Species, reference momentum, total/kinetic energy and signed rigidity |
| [002_coordinates.py](002_coordinates.py) | Linked energy coordinates, automatic updates, slopes and arrival time |
| [003_normalized_coordinates.py](003_normalized_coordinates.py) | Courant-Snyder coordinates, emittance-scaled launches and round trips |
| [004_gaussian_bunch.py](004_gaussian_bunch.py) | Build a weighted Gaussian bunch and check its physical beam sizes |
| [005_select_and_merge_particles.py](005_select_and_merge_particles.py) | Follow particle IDs while sorting, selecting and combining subsets |
| [006_multispecies_kicks.py](006_multispecies_kicks.py) | Compare proton, deuteron and helium deflections at equal velocity or momentum |

The reference and coordinate examples use a 1 GeV/c proton reference. The two
optics-based preparations use the
[native magnetic ring](../../test_data/example_lattices/README.md), with its
1 GeV/c electron reference and 4D optics. There is no RF confinement: the imposed
longitudinal Gaussian is not a matched RF distribution. A launch scan is not a
statistical bunch, and a finite random sample does not have exact target moments.

`003_normalized_coordinates.py` distinguishes unscaled Courant-Snyder
(Floquet) coordinates from emittance-scaled coordinates. Omitting emittances
means unit geometric scaling: `gemitt_x = 1` and `gemitt_y = 1` m. This
convention does not specify a physical beam emittance. In this uncoupled example,
the unscaled transverse coordinates have units sqrt(m), with radians treated as
dimensionless.

`006_multispecies_kicks.py` uses tabulated nuclear masses and charges for bare
protons, deuterons and helium-4 nuclei. A single thin dipole kick illustrates the
species normalization. No fragmentation model is involved.

These recipes cover Xtrack particle containers and their preparation for a line.
For distribution generators and longitudinal matching, see Xpart. Continue with
[tracking](../tracking/) and
[particle records](../getting_started/06_track_particles.py).
For a reference controlled by model expressions, see
[named reference particles](../line_construction/010_reference_particles.py).
