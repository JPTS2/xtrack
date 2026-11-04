<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# Line construction

An introductory route through native line definitions, placement, expressions,
editing and shared environments. The numbering suggests a reading order; every
script runs independently. Begin with the
[ordered component list](../getting_started/01_build_line.py) and
[element-table inspection](../getting_started/02_load_and_inspect.py).

| Example | Subject |
| --- | --- |
| [001_place_elements.py](001_place_elements.py) | Element start/centre/end anchors and relative positions |
| [002_strength_expressions.py](002_strength_expressions.py) | Scalar/array strengths, values versus references and numeric overrides |
| [003_line_algebra.py](003_line_algebra.py) | Combine assembled lines by concatenation, repetition and reversal |
| [004_place_lines.py](004_place_lines.py) | Subline start/centre/end anchors and relative placement |
| [005_parametric_geometry.py](005_parametric_geometry.py) | Compose mode: retain and regenerate placement expressions |
| [006_clones_and_replicas.py](006_clones_and_replicas.py) | Repeated objects, named replicas and independent clones |
| [007_edit_line.py](007_edit_line.py) | Insert, replace, remove and append with explicit geometry |
| [008_shared_lines.py](008_shared_lines.py) | Shared selections versus independent environment copies |
| [009_import_line.py](009_import_line.py) | Import between environments: variable conflicts, independence and slice parents |
| [010_reference_particles.py](010_reference_particles.py) | Named references and a field-normalized spectrometer |

`003_line_algebra.py` combines already assembled element sequences.
`005_parametric_geometry.py` uses `compose = True` to retain placement expressions
and rebuild the geometry when a variable changes.

Run the scripts in your Xsuite environment.

`002_strength_expressions.py` introduces environment variables and expressions
for element strengths. Parametric placements are covered in
`005_parametric_geometry.py`. Expression-based matching and time-dependent
tracking are developed with those subjects.

`009_import_line.py` transfers a line's elements and variable dependencies into
a destination environment, including the parents of sliced elements.

The inline lattices teach geometry and object ownership; no periodic optics or
closed-ring geometry is implied.
Lengths and positions are in metres, quadrupole `k1` in m^-2, and rotations in
radians. These recipes do not track. `010_reference_particles.py` uses a proton
reference and relates a signed magnetic field to its rigidity.
The other recipes need no reference particle.

For reusable native Python definitions, see the
[teaching models](../../test_data/example_lattices/README.md). For persistence,
see [save and load](../getting_started/08_save_and_load.py).

For shared-line applications in optics matching, see the
[matching examples](../match/).

For slicing strategies and slice counts, see the
[slicing examples](../slicing/).
