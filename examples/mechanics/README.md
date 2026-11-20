<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# Experimental rigid-body mechanics

Attach existing thick elements to surveyed supports with `env.mechanics` and
`line.mechanics`. Deliberate element alignment, installation errors and dynamic
motion remain separate. These examples are the main interface for reviewing
and extending the experimental feature.

Start with **001** for placement and cleanup, **004** for a surveyed ring,
**015** for bunch phases and driven tracking, and **016** for the FCC input.
Each script runs independently and leaves ordinary Xsuite objects available
for inspection after `%run`. Edit its **User parameters** first. Main teaching
stages use 80-character headings; individual steps use 40-character headings.

Run from this checkout in an Xsuite environment:

```bash
PYTHONPATH=. XSUITE_ALLOW_KERNEL_COMPILATION=1 python examples/mechanics/001_rigid_girder.py
```

Or start IPython with this checkout on `PYTHONPATH`, then use `%run`.
Matplotlib provides figures; 014 also uses Pillow to save a GIF at the printed,
editable output path. The other scripts display figures without saving files by default.
Lessons001–015 use bundled inputs. Required setup, diagnostics, source preparation
and tracking are timed separately. First use can compile kernels.

| Example | Lesson |
| --- | --- |
| [001_rigid_girder.py](001_rigid_girder.py) | Compare design placement, static installation error and vibration; remove vibration, then clear mechanics |
| [002_sources_and_vibrations.py](002_sources_and_vibrations.py) | Harmonic, measured or PSD translation/rotation; slow drift and split tracking |
| [003_shared_girder.py](003_shared_girder.py) | One physical support, opposite beam axes and independent revolution clocks |
| [004_demo_ring.py](004_demo_ring.py) | Drive 168 surveyed quadrupole supports and inspect magnified motion |
| [005_ground_and_feet.py](005_ground_and_feet.py) | Sample the ground at four feet per support; fit all rigid coordinates and report contact mismatch |
| [006_transfer_chain.py](006_transfer_chain.py) | Loaded floor/girder/magnet/BPM responses, phase, spectra and causal startup |
| [007_local_sources.py](007_local_sources.py) | Mount a pump; supply direct and floor paths with gain and delay |
| [008_regional_ground.py](008_regional_ground.py) | Shared background plus local excess PSD; vector components and reproducible chunks |
| [009_moving_bpm.py](009_moving_bpm.py) | Compare nominal beam coordinates with a moving electrical pickup and frozen correction |
| [010_execution_costs.py](010_execution_costs.py) | Separate load/build/attachment/update/tracking costs and test passage-time sampling |
| [011_installation_and_correction.py](011_installation_and_correction.py) | Static installation seeds and orbit correction, preserving intended off-axis quadrupoles |
| [012_matrix_response.py](012_matrix_response.py) | Complex 6x6 response, mixed units, lever arms and all rigid coordinates |
| [013_combined_mechanics.py](013_combined_mechanics.py) | Whole-ring ground/region/pump, feet, conditional responses, installation, drift and tracking |
| [014_tables_and_quad_animation.py](014_tables_and_quad_animation.py) | Select a quad from a table, plot its composed six-coordinate history and save a cuboid GIF |
| [015_bunch_phases_and_twiss.py](015_bunch_phases_and_twiss.py) | Fill buckets, frozen per-bunch Twiss and driven orbit, with an inspectable bunch table |
| [016_fcc_ground_motion.py](016_fcc_ground_motion.py) | Actual FCC Z line, 2,663 supports, common ground/feet, conditional girder/magnet/pickup responses and full tracking costs |

## Model and inspection

The environment owns physical bodies and sources. Lines bind element occurrences
and explicit survey placements in that world; there is no environment turn
clock. Body coordinates use metres and survey radians, with rotation
`Ry(yaw) Rx(-pitch) Rz(roll)`. Sources can be body-local rigid coordinates or
world displacement fields. Multiple feet and response operators are explicit
model inputs; a floor is not one rigid parent.

```python
mech = env.mechanics
line.mechanics.adopt_reference()
girder = line.mechanics.new_girder('g1', elements=['q1', 'q2'])
girder.installation.shift_y = 20e-6
mech.add_source('floor_y', xt.mechanics.Harmonic(
    amplitude=50e-9, frequency_hz=5.), targets=['g1'], component='y')
mech.apply(time_s=.05)

mech.get_source_table().show()
girders = mech.get_table(time_s=.05).rows.match(kind='girder')
quad = line.mechanics.get_table(time_s=.05).rows['q1']
history = quad.get_time_series(time_s=times)
history.plot()
```

Tables/plots do not move the line. Reference adoption preserves existing numeric
inputs without retyping them. `enable_vibrations()` opts into automatic turn-held
updates using particle physical time; `disable_vibrations()` removes dynamics
while retaining installation. `clear()` restores adopted element inputs and
removes mechanics. `reference='absolute'` replaces inherited dynamic motion at a
child while retaining static placement; relative sources compose down the tree.

## FCC input

016 tracks the actual positron Z line from the
[FCC development model](https://gitlab.cern.ch/acc-models/fcc/acc-models-fcc-ee/-/tree/dev/108.1.0),
commit `8805a5bb8bd568f5de2e1558037371905ecfc985`. Select the external input:

```bash
XTRACK_MECHANICS_FCC_INPUT=/path/to/lattices/z/fccee_z_beam_1_positrons.json \
    PYTHONPATH=. XSUITE_ALLOW_KERNEL_COMPILATION=1 python examples/mechanics/016_fcc_ground_motion.py
```

The line is `fccee_p_ring`; input SHA-256 is
`dd7c7749d1bedfa5faf41bbfd885312bbc38bec8a817460bc0627f3c4b6094fe`.
The example has provisional supports for all 2,663 quads, a common ground field,
conditional girder/magnet responses and a selected virtual pickup. Other
magnets retain nominal placement. Assignment, spectra and responses are teaching
inputs; they do not establish engineering FCC performance or alignment tolerances.

## Current scope

The implementation supports prescribed rigid motion, bounded source records,
static installation cases, frozen optics and turn-held tracking. 015 compares
independent bunch runs at different physical phases; packed multibunch maps are
not implemented. Sampling/mode convergence must be checked for a study.

The numeric-field backend guards slices/replicas, shared attached element
objects, expression-driven alignments or xdeps consumers of their outputs, changed reference geometry and built
aperture application. Strength changes that leave reference geometry unchanged are supported. Whole-environment copy/JSON preserves mechanics for registered lines; attached line-only
copy/export is guarded. Disable dynamics and apply a fixed time before Twiss.

CPU/OpenMP execution is tested. Actual GPU execution and throughput remain
unvalidated. Long-run streaming, engineering support data and online moving-BPM /
feedback coupling are subsequent work. The bundled
[demo ring](../../test_data/example_lattices/demo_ring/README.md) is a teaching
lattice, with a sensitive near-integer vertical tune.
