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
| [001_environment_and_elements.py](001_environment_and_elements.py) | Element definitions, named access and the environment registry |
| [002_assemble_line.py](002_assemble_line.py) | Ordered components, named lines and entrance positions |
| [003_place_elements.py](003_place_elements.py) | Element start/centre/end anchors and relative positions |
| [004_strength_expressions.py](004_strength_expressions.py) | Scalar/array strengths, values versus references and numeric overrides |
| [005_line_algebra.py](005_line_algebra.py) | Combine assembled lines by concatenation, repetition and reversal |
| [006_place_lines.py](006_place_lines.py) | Subline start/centre/end anchors and relative placement |
| [007_parametric_geometry.py](007_parametric_geometry.py) | Compose mode: retain and regenerate placement expressions |
| [008_clones_and_replicas.py](008_clones_and_replicas.py) | Repeated objects, named replicas and independent clones |
| [009_edit_line.py](009_edit_line.py) | Insert, replace, remove and append with explicit geometry |
| [010_replace_elements.py](010_replace_elements.py) | Replace one magnet, every use of a name, or names matching a pattern |
| [011_shared_lines.py](011_shared_lines.py) | Shared selections versus independent environment copies |
| [012_import_line.py](012_import_line.py) | Import between environments: variable conflicts, independence and slice parents |
| [013_reference_particles.py](013_reference_particles.py) | Named references and a field-normalized spectrometer |

`005_line_algebra.py` combines already assembled element sequences.
`007_parametric_geometry.py` uses `compose = True` to retain placement expressions
and rebuild the geometry when a variable changes.

Run the scripts in your Xsuite environment.

`004_strength_expressions.py` introduces environment variables and expressions
for element strengths. Parametric placements are covered in
`007_parametric_geometry.py`. Expression-based matching and time-dependent
tracking are developed with those subjects.

`012_import_line.py` transfers a line's elements and variable dependencies into
a destination environment, including the parents of sliced elements. It then
imports two sliced lines with the same element names and shows how their
parents are renamed. Separate knob changes expose which source or imported
line responds.

`010_replace_elements.py` answers: how do I replace just the second magnet when
several positions share the same element? It then replaces every use of a name
and both screens selected by a pattern. All replacements preserve length.

The inline lattices teach geometry and object ownership; no periodic optics or
closed-ring geometry is implied.
Lengths and positions are in metres, quadrupole `k1` in m^-2, and rotations in
radians. These recipes do not track. `013_reference_particles.py` uses a proton
reference and relates a signed magnetic field to its rigidity.
The other recipes need no reference particle.

For reusable native Python definitions, see the
[teaching models](../../test_data/example_lattices/README.md). For persistence,
see [save and load](../getting_started/08_save_and_load.py).

For shared-line applications in optics matching, see the
[matching examples](../match/).

For slicing strategies and slice counts, see the
[slicing examples](../slicing/).
