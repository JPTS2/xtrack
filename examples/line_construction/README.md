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
| [004_strength_expressions.py](004_strength_expressions.py) | Scalar/array strengths, expression-preserving corrections and array extension |
| [005_line_algebra.py](005_line_algebra.py) | Combine assembled lines by concatenation, repetition and reversal |
| [006_place_lines.py](006_place_lines.py) | Subline anchors, relative placement and independent clones |
| [007_parametric_geometry.py](007_parametric_geometry.py) | Retain, combine, restore and regenerate placement expressions |
| [008_clones_and_replicas.py](008_clones_and_replicas.py) | Repeated objects, named replicas, clones and generated drifts |
| [009_edit_line.py](009_edit_line.py) | Single/batch/subline insertion, removal and appending with explicit geometry |
| [010_replace_elements.py](010_replace_elements.py) | Replace one magnet, every use of a name, or names matching a pattern |
| [011_shared_lines.py](011_shared_lines.py) | Shared selections versus independent environment copies |
| [012_import_line.py](012_import_line.py) | Import between environments: variable conflicts, independence and slice parents |
| [013_reference_particles.py](013_reference_particles.py) | Named versus independent references and a field-normalized spectrometer |
| [014_cut_at_positions.py](014_cut_at_positions.py) | Cut at specified positions and inspect parent-linked thick slices |
| [015_save_and_load_environment.py](015_save_and_load_environment.py) | Save shared lines, slice parents, named references and live expressions |
| [016_read_legacy_madx_data.py](016_read_legacy_madx_data.py) | Load separate legacy lattice/strength files and reverse a stored line |
| [017_build_demo_ring.py](017_build_demo_ring.py) | Build an instrumented ring with a low-beta IP and live machine controls |

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

The small inline lattices in `001`–`016` teach geometry and object ownership;
no periodic optics or closed-ring geometry is implied for those definitions.
Lengths and positions are in metres, quadrupole `k1` in m^-2, and rotations in
radians. These recipes do not track. `013_reference_particles.py` uses a proton
reference and relates a signed magnetic field to its rigidity.
`015` also preserves a named proton reference through copying and reload.
The other small recipes need no reference particle.

For reusable native Python definitions, see the
[teaching models](../../test_data/example_lattices/README.md). For persistence,
see [save and load an environment](015_save_and_load_environment.py).

For shared-line applications in optics matching, see the
[matching examples](../matching/).

`014_cut_at_positions.py` teaches geometric boundary placement and parent-linked
thick slices. For choosing slicing strategies and slice counts, see the
[magnet slicing example](../elements/magnets/003_thin_and_thick_slices.py).

`015_save_and_load_environment.py` saves three named lines in one environment,
including sliced and unsliced lines sharing parent magnets. Restored strength
expressions and the named reference remain live inside the loaded copy; the
original and an in-memory copy stay independent. It writes
`outputs/shared_environment.json` beside the script.

`016_read_legacy_madx_data.py` is a legacy data-input example, not a recommended
way to author new models. It uses Xtrack's native parser without `cpymad`, MAD-X
macros or a MAD-X process, and writes `outputs/converted_legacy_environment.json`.
It writes two visible legacy definitions to `outputs/`, then loads the lattice
and strength files together and reverses a second stored line. The definition
is magnetic geometry with strength expressions; no reference particle, optics
or tracking configuration is implied.

`017` inspects the shared demonstration-machine families and sections, assembles
the full ring, then compares its survey and optics. The hardware definitions
and operating strengths are separate native inputs under
[`test_data/example_lattices/demo_ring/`](../../test_data/example_lattices/demo_ring/README.md).
From the repository root, other lessons load the nominal machine with:

```python
env = xt.load('test_data/example_lattices/demo_ring/lattice.py')
line = env['demo']
```

Each load creates an independent environment without executing plots or a
matching solve. `017` itself runs its inspection and plotting at module level.
The [matching recipe](../matching/014_match_demo_ring.py) derives strengths and
writes a separate output setting; canonical inputs are not overwritten.
The model README states the qualified IP optics, device conventions and the
limits of the nominal nonlinear and RF configurations.
