<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# Experimental rigid-body mechanics

Attach existing thick elements to surveyed supports with `env.mechanics` and
`line.mechanics`. Deliberate element alignment, installation errors and dynamic
motion remain separate. These examples are the main interface for reviewing
and extending the experimental feature.

Start with **001** for shared mechanics, the three placement layers and table
inspection, then **002** for mounts and rigid sway. **003** adds reversed beam
coordinates. Follow the feature route below; the examples remain in one flat
folder. **014** is an optional animation rather than a prerequisite.
Each script runs independently and leaves ordinary Xsuite objects available
for inspection after `%run`. Edit its **User parameters** first. Main teaching
stages use 80-character headings; individual steps use 40-character headings.

For a short developer walkthrough, run **001 → 002 → 006 → 016**. This takes
the same UI from three placement layers, through finite mounted transforms and
relative response inputs, to tracking an actual FCC lattice. Use **003** for a
shared support on reversed lines and **015** for independent bunch phases.

Run from this checkout in an Xsuite environment:

```bash
PYTHONPATH=. XSUITE_ALLOW_KERNEL_COMPILATION=1 python examples/mechanics/001_rigid_girder.py
```

Or start IPython with this checkout on `PYTHONPATH`, then use `%run`.
Matplotlib provides figures; 014 also uses Pillow to save a GIF at the printed,
editable output path. The other scripts display figures without saving files by default.
Lessons other than 016 use bundled inputs. Most retain separate timing diagnostics
for setup, inspection, source preparation and tracking. Set `PRINT_TIMINGS = True`
where provided to see those details; 010 and 016 also report their main execution
comparisons. First use can compile kernels.

## Suggested learning route

| Stage | Examples | Physical question |
| --- | --- | --- |
| Shared mechanics and tables | **001** | Which motion is shared, and what is deliberate, installed or dynamic? |
| Mounts and rigid geometry | **002** | How do heights, reference points and rotations move the magnets? |
| Shared physical space | **003** | How does one support serve forward and reversed beam coordinates? |
| Supplying motion | **002** source alternatives | How do stated samples, harmonics and seeded PSDs reach the same mechanics interface? |
| Spatial sources | **004 → 008 → 007** | How do directional waves, correlated ground, noisy regions and local sources differ? |
| Mechanical responses | **006**, then optional **012** | How does floor motion become girder, magnet and pickup motion? |
| Pipe ownership | **018** | What changes when the chamber is fixed, girder-owned or magnet-owned? |
| Prescribed girder shapes | **019 → 020** | How do bending and torsion move rigid magnets, and how do a floor wave and pump drive the same support? |
| Overlapping solenoid and multipole fields | **021** | How can one parallel longitudinal-field axis move relative to a common rigid multipole chart? |
| Tracking and physical time | **015** | Which motion does each bunch encounter, and what does a frozen Twiss calculation mean? |

Further lessons cover assumed contact footprints (**005**), moving-pickup
readout and offline correction (**009**, **011**), sampling accuracy and costs
(**010**) and live inputs and applied histories (**017**).
**013** and **016** are complete ring/FCC studies.
**014** adds table-selected animation. The current filenames are retained while
the teaching sequence is reviewed; some scripts cover more than one step.

Each lesson starts from a physical setup and compares specified changes to that
setup. Diagrams identify the geometry and reference points; reported beam or
mechanical changes use physical units. Display gains affect presentation only.

| Example | Lesson |
| --- | --- |
| [001_rigid_girder.py](001_rigid_girder.py) | Compare design placement, static installation error and vibration; remove vibration, then clear mechanics |
| [002_sources_and_vibrations.py](002_sources_and_vibrations.py) | Declare girder/magnet heights and base mounts; compare rigid sway, beam-axis rotation and tracked beam responses |
| [003_shared_girder.py](003_shared_girder.py) | One physical support, opposite beam axes and independent revolution clocks |
| [004_demo_ring.py](004_demo_ring.py) | Locate 168 supports in the surveyed ring and compare installed versus displaced frozen closed orbits |
| [005_ground_and_feet.py](005_ground_and_feet.py) | Sample the ground at four feet per support; fit all rigid coordinates and report contact mismatch |
| [006_transfer_chain.py](006_transfer_chain.py) | Loaded floor/girder/magnet/BPM responses, phase, spectra and causal startup |
| [007_local_sources.py](007_local_sources.py) | Locate a pump; compare direct, propagated and combined paths in support motion and frozen beam response |
| [008_regional_ground.py](008_regional_ground.py) | Add a noisy region to the same background; compare near/far receivers, RMS, spectra and correlation |
| [009_moving_bpm.py](009_moving_bpm.py) | Compare correcting a moving electrical zero with restoring the beam to design coordinates |
| [010_execution_costs.py](010_execution_costs.py) | Compare actual beam transport with turn-held versus passage-time poses, then measure update/tracking costs |
| [011_installation_and_correction.py](011_installation_and_correction.py) | Correct installation seeds; distinguish electrical residual from true orbit error while preserving deliberate offsets |
| [012_matrix_response.py](012_matrix_response.py) | Separate mixed-coordinate mechanical response from rigid transport to another port; inspect the tracked beam effect |
| [013_combined_mechanics.py](013_combined_mechanics.py) | Combine ground, region, pump, contacts and responses; compare frozen-initial and moving-map beam tracking |
| [014_tables_and_quad_animation.py](014_tables_and_quad_animation.py) | Select a quad from a table, plot its composed six-coordinate history and save a cuboid GIF |
| [015_bunch_phases_and_twiss.py](015_bunch_phases_and_twiss.py) | Fill buckets, frozen per-bunch Twiss and driven orbit, with an inspectable bunch table |
| [016_fcc_ground_motion.py](016_fcc_ground_motion.py) | Track the actual FCC Z line with 2,663 provisional supports; compare design, installed and moving beams and measure costs |
| [017_live_inputs_and_history.py](017_live_inputs_and_history.py) | Supply applied carriage positions, remove vibration while controls continue, and replay captured history without reading the live provider |
| [018_pipe_ownership.py](018_pipe_ownership.py) | Compare transverse and 3D chamber windows for fixed, girder-owned and magnet-owned pipes, using the same tracked probe grid |
| [019_deformation_at_mounts.py](019_deformation_at_mounts.py) | Resolve coherent prescribed bending/torsion at several mounts while each carried magnet stays rigid |
| [020_driven_girder_response.py](020_driven_girder_response.py) | Feed a floor wave and pump force into one bending mode, then compare magnet motion and electrical BPM readout |
| [021_solenoid_axis.py](021_solenoid_axis.py) | Assign a common rigid VariableSolenoid chart and a separate transverse axis, preserving its existing x0/y0 and multipoles |

## Model and inspection

The environment owns physical bodies and sources. Lines bind element occurrences
and explicit survey placements in that world; there is no environment turn
clock. Body coordinates use metres and survey radians, with rotation
`Ry(yaw) Rx(-pitch) Rz(roll)`. Sources can be body-local rigid coordinates or
world displacement fields. Multiple feet and response operators are explicit
model inputs; a floor is not one rigid parent.

```python
import numpy as np
import xtrack as xt

env = xt.Environment()
env.new('q1', xt.Quadrupole, length=.4, k1=.1, shift_y=2e-6)
env.new('q2', xt.Quadrupole, length=.4, k1=-.1)
line = env.new_line(name='demo', components=['q1', 'q2'])
line.set_particle_ref('proton', p0c=7e12)

mech = env.mechanics
line.mechanics.adopt_reference()
girder = line.mechanics.new_girder('g1', elements=['q1', 'q2'])
girder.installation.shift_y = 20e-6
mech.add_source('floor_y', xt.mechanics.Harmonic(
    amplitude=50e-9, frequency_hz=5.), targets=['g1'], component='y')

line.mechanics.freeze_at(time_s=.05)
line.mechanics.get_applied_table().show(cols=['name', 'Y'])
particles = line.build_particles()
line.track(particles)

quad = line.mechanics.get_table().rows['q1']
quad.get_time_series(time_s=np.linspace(0., .2, 101)).plot()
```

Tables/plots do not move the line. Reference adoption preserves existing numeric
inputs without retyping them. `enable_vibrations()` opts into automatic turn-held
updates using particle physical time; `disable_vibrations()` removes disturbance
motion while retaining installation and applied controls. `enable_controls()`
enables controls without enabling disturbances; `freeze_updates()` holds the
current maps. `line.mechanics.clear()` restores that line's adopted inputs and
detaches it, retaining shared physical definitions. `env.mechanics.clear()`
restores all attached lines and removes the bodies and sources.
A mounted body always follows its parent. Bind extra motion at its mount. If a
transfer function gives total motion, first express the carried parent motion
at the same port, then bind `total.relative_to(carried)`, as in 006. If the
input is a history of world poses including installation, use
`mech.relative_motion_from_poses(time_s=..., targets=..., total_poses=...)`.
Define an independently moving body without a parent.

Use `disabled_sources='floor_y'` (or several names) with source preparation,
automatic tracking, explicit application, pose tables, BPMs and frozen cuts.
Pass the same selection when comparing results. A selected table row carries the
mask into its history. Read `line.mechanics.get_applied_table()` to inspect the
element frames currently installed on the line without sampling sources.
Its `pose_time_s` column includes each occurrence's reference arrival delay
when the last update used passage-time sampling.
An ordinary `get_table(time_s=...)` evaluates the requested model and time.

Pose rows retain their deliberate/installation case. After changing that case,
request a fresh table before querying a row's history or live element; old numeric
columns remain readable. Batch `new_bodies` and `new_girders` accept the same
parent-local `mount=xt.Frame(...)` convention as scalar construction.

Prescribed sources support offline `compute_time_series`. By default, queries
inside the computed window evaluate continuous sources at the requested time,
reusing the same Fourier realization for FFT PSD inputs. Measured inputs keep
their original time stamps and declared interpolation. Pass
`interpolation='linear'` or `'previous'` to replay samples on the supplied
grid. Custom prescribed sources use linear sample replay by default. Tracking
holds each evaluated pose until the next freeze.
Changing the duration or step of an FFT record changes its frequency bins.
Use `SpectralGroundMotion` when separately evaluated blocks must share one
continuous seeded field.
For a separately processed causal response, `get_state()` and `set_state()`
carry filter delays between consecutive input blocks. Keep the time grid,
ordered ports and any source state with that checkpoint.

Live sources supply
the current applied position; they do not implicitly model an actuator or a
controller. For historical inspection of live inputs, use a bounded
`record_applied_sources` recorder and pass its immutable snapshot to
`get_time_series(applied_history=...)`, as in 017. Start a new recording segment
when changing which sources are enabled. The result retains the requested
`time_s` and the recorded `pose_time_s` used for all inputs. Restoring or
clearing mechanics stops the active recording.

`mech.plot(...)` labels a point query involving live inputs as a current-held
pose. The multi-body `mech.animate(...)` is for prescribed motion; it rejects
live sources affecting the selection before sampling them. Use an applied
source history and a selected row's `animate(...)` for a recorded live pose,
or disable those sources explicitly for a prescribed-motion comparison.

For supplied contact responses, `TabulatedResponse` also accepts rectangular
matrices `(frequency, output_channels, input_channels)`. Apply them to explicitly
ordered numeric channels, then construct a six-coordinate `RigidMotion` for
body replay. State port frames and units; the matrix dimensions infer no support
mechanics. Filtering remains a host-side periodic preparation operation.

## Solenoid field axes and overlapping magnetic fields

For a group of independent `VariableSolenoid` tracking segments, first attach
the entire rigid chart (including all its multipole coefficients). Then declare
one additional longitudinal-field axis:

```python
chart = line.mechanics.new_girder('field_chart', elements=segment_names)
axis = line.mechanics.new_solenoid_axis('solenoid_axis', elements=segment_names)
axis.installation.shift_x = 10e-6
mech.add_source('axis_y', xt.mechanics.Harmonic(30e-9, 20.),
                targets=axis.name, component='y')
line.mechanics.get_solenoid_axis_table(time_s=.05).show()
```

The axis inherits the chart's full rigid motion and adds only local transverse
x/y translations to its segments' adopted `x0/y0`. Parallel or antiparallel
charts may have different rolls; the axis displacement is projected into each
segment's adopted chart. The multipoles and `ks_profile` remain unchanged.
Use `attach_solenoid_axis` for more segments or reversed lines sharing the same
physical chart and axis body. Disabling disturbances, restoring, clearing and
whole-environment serialization preserve the original adopted axis offsets.
The prepared tracker evaluates and writes those offsets on its context.

Independent tilt or longitudinal displacement of the solenoid field relative to
the chart requires a different field representation; it is rejected here.
All multipoles in one chart still move together. Independently supported
quadrupole, higher multipoles and solenoid fields must not be inferred from
already-summed `knl/ksl` or from integration-segment names. Example 021 uses only
the explicitly supported two-axis case.

## Pipes and levels of mechanical knowledge

Declare ownership of installed pipe placements independently of their shared
shape definitions and of intentional magnetic-axis offsets:

```python
line.mechanics.attach_pipe('installed_chamber', body=girder)
# body=None declares an adopted installed pipe stationary in the world.
# attach_pipes({'pipe1': girder, 'pipe2': None}) declares a batch.
line.mechanics.get_pipe_table().show()
frozen = line.mechanics.freeze_at(time_s=.05, aperture_at_s=[s_check])
limit = frozen.limits[s_check]
magnet_pose = frozen.element_poses['q1']
```

Declare every installed placement, including fixed ones. `line.aperture` stays
the adopted reference geometry. `freeze_at` captures placements, shapes, bounds and
fixed-survey-plane cuts from the same body state as the applied magnet fields.
For read-only cuts use `get_aperture(time_s=...)`. Copy the lowered elements
and generated limits into the tracking line for a frozen loss study, as in 018.
Fixed pipes and pipes on static bodies can
coexist with automatic motion on other bodies. Enabled sources on a pipe's
owner or its ancestors require a frozen-time workflow. Export ownership with
`env.to_dict(include_aperture=True)`.
Explicit-time queries report prescribed poses; pass `disabled_sources` for a
matched source-removed comparison. They do not infer a historical applied pose
from the current automatic enable flags.
For live inputs, pass `applied_history=recorder.snapshot()` to query the most
recent recorded freeze at or before the requested time. Read `pose_time_s`
in the table to see the selected freeze. For full-line replay with `freeze_at`,
record with `case_scope='line'` and include every enabled live source relevant
to that line. A new `freeze_at` without history samples current live inputs once.

Moving finite pipes need at least two profiles. Profile planes without a
bracketed survey intersection reject before geometry preparation; this is a
conservative coverage check, not support for extrapolation. Tracking-limit
extraction also rejects uncovered or nonfinite/degenerate cuts. Existing
cross-pipe gap interpolation and curved-pipe approximations still apply.
Longitudinal and polygon sampling need convergence checks, including when
comparing reversed beam views.

A girder does not require feet. Direct rigid motion, assumed support footprints
and supplied responses are useful at different levels of input knowledge.
Contact fitting is an explicit kinematic approximation, not a stiffness model.
In 019, `ModalMotion` takes six-coordinate shapes at named nominal mount
frames and a time series of amplitudes. Amplitudes can be shared or grouped by
support. Place a port on the neutral axis and put the rigid magnet at its actual
height to include the rotational lever arm. A direct `RigidMotion` record can
still supply measured or externally computed six-coordinate mount motion.
020 adds conditional transfer functions from floor displacement and local pump
force through the girder mode, magnet and pickup.

The model uses supplied linear modal kinematics and finite rigid poses at
the mounts. Elastic loads, contact stiffness, warping and field deformation
along a magnet require external models. A finite pipe can follow one rigid
pose. Continuous pipe deformation, bellows and automatic moving-pipe loss
checks are outside this workflow.

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
The default 128-mode ground field is a cost/UI demonstration with known spatial
covariance nonconvergence. Increase and validate the spatial sampling before
using it for FCC beam-quality predictions. This standalone mechanics lesson has
no feedback or commissioning dependency.

## Current scope

The implementation supports prescribed rigid motion, bounded source records,
static installation cases, frozen optics and turn-held tracking. Corrections in
009 and 011 are offline frozen-orbit calculations. 015 compares
independent bunch runs at different physical phases; packed multibunch maps are
not implemented. Sampling/mode convergence must be checked for a study.

The numeric-field backend rejects selected native slices and replicas, and
shared attached element objects (including their use as a hidden native-slice
parent or replica target on another physical line). Unrelated genuine native
slices may remain in the line; their parent geometry, weight and offsets are
guarded. Replicas on a bound line remain unsupported. Standalone
`VariableSolenoid` segments, rather than native parent-linked slices, may be
grouped as in 021. Expression-driven alignments or xdeps consumers of their
outputs, changed reference geometry and undeclared built-aperture application
are rejected before lowering.
Ownership checks cover physical lines in the same Environment. Explicitly
sharing writable Xobject storage between different environments is unsupported;
use `env.copy()` for independent mechanics scenarios. Independent elements can
share an allocation buffer without sharing writable element storage.

Strength changes that leave reference geometry unchanged
are supported. Whole-environment copy/JSON preserves mechanics definitions for
registered lines with serializable sources; custom live providers such as 017's
are rejected. Automatic enable flags, computed records and causal filter state
are not restored. Attached line-only copy/export is guarded. Freeze automatic
updates and apply a fixed time before Twiss.

CPU/OpenMP execution is tested. Actual GPU execution and throughput remain
unvalidated. Long-run streaming, engineering support data and online moving-BPM /
feedback coupling are subsequent work. The bundled
[demo ring](../../test_data/example_lattices/demo_ring/README.md) is a teaching
lattice, with a sensitive near-integer vertical tune.

On a known constant clock, a complete tracking request checks finite built-in
records before advancing particles, including reference passage or source
propagation delays. Live provider failures and changing clocks or callbacks can
still interrupt a call. Check `particles.at_turn`, `particles.time_s` and the
line's applied table before resuming from partial progress.
