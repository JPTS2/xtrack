<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# Mechanics

Start with the three classes of placement: deliberate reference, installation
errors and dynamic motion. Continue through the same closed demo ring used by
feedback, floor sampling at support feet, mechanical response paths, located
sources, regional spectra and moving pickups. Each script runs independently.
The numbered lessons use the feedback examples' section formatting and editable
**User parameters**, and leave ordinary Xsuite objects available after `%run`.

An 80-character `#` heading introduces a main teaching stage. Within each stage,
40-character headings separate the individual steps: timing setup, line
construction, reference adoption, body assignment, source definition, sampling,
application and inspection. The longer correction and timing loops use the same
subheadings inside the loop. Parameters are grouped by purpose so that a reader
can find the input to change before following its effect through the lesson.

| Example | Lesson | Approximate CPU time |
| --- | --- | --- |
| [001_rigid_girder.py](001_rigid_girder.py) | Adopt deliberate offsets; add installation and vibration; disable and clear | About 1 s |
| [002_sources_and_vibrations.py](002_sources_and_vibrations.py) | Harmonic, measured or PSD translation/rotation; slow drift and split tracking | About 1 s |
| [003_shared_girder.py](003_shared_girder.py) | One physical support, opposite beam axes and independent revolution clocks | About 1 s |
| [004_demo_ring.py](004_demo_ring.py) | Drive 168 surveyed quadrupole supports and inspect magnified motion | About 3 s |
| [005_ground_and_feet.py](005_ground_and_feet.py) | Sample the ground at four feet per support; fit all rigid coordinates and report contact mismatch | About 1 s |
| [006_transfer_chain.py](006_transfer_chain.py) | Loaded floor/girder/magnet/BPM responses, phase, spectra and causal startup | About 1 s |
| [007_local_sources.py](007_local_sources.py) | Mount a pump; supply direct and floor paths with gain and delay | About 1 s |
| [008_regional_ground.py](008_regional_ground.py) | Shared background plus local excess PSD; vector components and reproducible chunks | About 4 s |
| [009_moving_bpm.py](009_moving_bpm.py) | Compare nominal beam coordinates with a moving electrical pickup and frozen correction | About 3 s |
| [010_execution_costs.py](010_execution_costs.py) | Separate load/build/attachment/update/tracking costs and test passage-time sampling | About 16 s |
| [011_installation_and_correction.py](011_installation_and_correction.py) | Static installation seeds and orbit correction, preserving intended off-axis quadrupoles | About 6 s |
| [012_matrix_response.py](012_matrix_response.py) | Complex 6x6 response, mixed units, lever arms and all rigid coordinates | About 1 s |
| [013_combined_mechanics.py](013_combined_mechanics.py) | Whole-ring ground/region/pump, feet, conditional responses, installation, drift and tracking | About 6 s |
| [014_tables_and_quad_animation.py](014_tables_and_quad_animation.py) | Select a quad from a table, plot its composed six-coordinate history and save a cuboid GIF | About 4 s, including GIF |
| [015_bunch_phases_and_twiss.py](015_bunch_phases_and_twiss.py) | Fill buckets, frozen per-bunch Twiss and driven orbit, with an inspectable bunch table | About 15–25 s with cold Twiss kernels |

Times are local CPU examples, with available kernel caches, excluding time spent
interacting with a figure. See the [captured walkthrough](review/README.md) for
terminal output, figures, detailed timing definitions and measured limitations.
This is an exploration branch; these are teaching models, not an FCC-ee layout
or mechanical performance prediction.

The heavier [girder assembly review](review/demos/README.md) sits outside this
short lesson set. It animates a rigid girder, two 0.5 m quadrupoles and a 1.5 m
bend with independent magnet/girder motion, and saves a GIF.

The [FCC readiness review](designs/2026-11-20_fcc_scaling_review.md) benchmarks the
requested development lattice and identifies what must change before full FCC
studies. The small teaching examples do not establish full-machine readiness.

The subsequent [prepared-runtime and bunch review](designs/2026-11-20_prepared_runtime_review.md)
keeps these scripts and their UI unchanged while adding native bulk assignment,
support classes, precision-preserving local motion and context-native updates.
[Fresh captures](review/results/prepared_examples/manifest.json) include source,
kernel and tracker hashes. Cold preparation and steady tracking are measured
separately in the FCC review; a first example track may compile mechanics kernels.

The [first-demo readiness review](designs/2026-11-20_demo_pr_readiness.md)
distinguishes the successful prepared tracking path from the remaining GPU,
source-streaming and packed-bunch work. Its recommendation is to keep the UI and
prepare a focused experimental draft, with targeted backend comparisons.

Example 015 runs on the bundled ring. Its editable parameters also select the
FCC model used in the review. It leaves `bunch_table`, `high_frequency_bunches`,
the two scenario results and the last live mechanics source available after
`%run`; automatic vibrations are disabled and the line is frozen at one selected
bunch phase. The [captured lesson](review/results/prepared_examples/015_bunch_phases_and_twiss/README.md)
explains the figures and links the verified FCC variant.

Run from this checkout with its source first on `PYTHONPATH`:

```bash
PYTHONPATH=. python examples/mechanics/001_rigid_girder.py
PYTHONPATH=. ipython
# Then: %run examples/mechanics/006_transfer_chain.py
```

Matplotlib provides the figures. NumPy/SciPy and the normal Xsuite dependencies
are required; saving the GIF in 014 also uses Pillow. No code from the feedback checkout or the earlier scratch design
is imported. The shared [demo ring](../../test_data/example_lattices/demo_ring/README.md)
is included here. Its near-integer vertical tune and large insertion beta make
it particularly sensitive to installation errors; its correction numbers should
not be interpreted as FCC-ee predictions.

## Three placement classes

```python
# Existing intentional line offsets are authoritative reference inputs.
line['q1'].shift_x = 1e-3
line.mechanics.adopt_reference()
girder = line.mechanics.new_girder('g1', elements=['q1', 'q2'])

# Installation is static, independent of dynamic source seeds.
girder.installation.shift_y = 20e-6
errors = env.mechanics.generate_installation_errors(
    seed=956, bodies=['g1'], translation_sigma_m=(20e-6, 20e-6, 0.))
errors.apply(env.mechanics)

# Dynamic input includes vibrations and prescribed slow drift.
env.mechanics.add_source('floor', xt.mechanics.Harmonic(
    amplitude=50e-9, frequency_hz=5), targets=['g1'])
env.mechanics.apply(time_s=.05)
```

Reference adoption keeps the existing line values; it does not zero or require
retyping them. Attachments also capture current numeric inputs without an
explicit adoption call. Explicit adoption freezes/checks the classification
before binding. Design rolls remain distinct from mechanical output angles.

Installation generators support Gaussian/uniform draws, Gaussian truncation,
per-coordinate translation/angle scales and named common fixtures. By default
only root bodies are selected. Draws have stable body/group identities, so
receiver selection/order and dynamic seeds do not change them. The returned
installation case is immutable; applying a corrected case can retain the
original as-installed record for comparisons. Gaussian sigma describes the
parent distribution, before truncation.

Installation defines the static baseline. A stationary PSD process generally
has nonzero motion at time zero; it is not silently shifted to start at zero.
Measured traces, slow drifts or causal startup can supply explicit initial
conditions when those are needed.

`disable_vibrations()` removes dynamic input and retains installation placement.
`restore()` restores captured element inputs while retaining mechanics definitions.
`clear()` additionally removes bodies, sources, attachments and automatic updates.
The deliberate line offsets/rolls that were adopted before binding survive.
Body reference settings defined inside mechanics are removed with that model.
Beam-corrector settings are separate knobs and remain under user control.

## Geometry, signals and response paths

`env.mechanics` owns bodies/sources; `line.mechanics` binds line occurrences and
world placement. Use `new_girder` for supports and `new_body` for magnets,
pickups/equipment. A child may have a parent-local `mount=xt.Frame(...)` or an
explicit world anchor. Feet live on the support as `feet_local_m`; the same
geometry is sampled, fitted and drawn. Only explicitly attached lines join.

A floor is a spatial field, not one rigid parent. `sample_field` samples one
joint source at world receivers. `fit_rigid_bodies` returns rigid motion and
contact residuals; `to_rigid_motion` projects anchor translations without
inventing rotation. `RigidMotion` carries six local coordinates in metres and
survey radians. `through` applies a complex response; `transport` handles the
reference-point lever arm and axes separately. `reference='absolute'` replaces
inherited vibration while preserving static placement, avoiding double counting.
Named `select` and coframed `superpose` compose these records explicitly; example
013 connects the full pipeline on 168 supports, 168 magnet bodies and 14 pickups.

`DisplacementPSD` is a finite periodic Fourier reference with dense exponential
spatial correlation. `SpectralGroundMotion` is a continuous random-mode
approximation with bounded source state and stable time chunks/receiver subsets.
`RegionalGroundMotion` adds independent excess spectra with squared amplitude
weights; `VectorGroundMotion` supplies world components and explicit mixing.
The mode approximation needs convergence checks; it cannot represent every
measured covariance accurately with a small rank.

## Survey, figures and execution

The source inventory is a normal table and does not sample the sources:

```python
sources = mech.get_source_table()
sources.cols['name model frame component reference target_names'].show()
source_model = sources['source', 'floor']
```

It lists registered motion bindings. Intermediate floor fields and response
operators used to build an offline response record remain explicit objects in
the example; they are not automatically registered as additional drives.

At one physical time, select bodies and element occurrences independently:

```python
bodies = mech.get_table(time_s=.125, time_series=motion)
girders = bodies.rows.match(kind='girder')

elements = line.mechanics.get_table(time_s=.125, time_series=motion)
quads = elements.rows.match(element_type='Quadrupole')
selected = quads.rows['q1']
quad = selected.get_element()
history = selected.get_time_series()  # Uses the bound motion record's grid
history_fig, history_axes = history.plot()
```

Body kinds are explicit metadata: `new_girder` sets `kind='girder'`, while
`new_body(..., kind='magnet')` or `kind='pickup'` labels another rigid body.
The line table contains attached occurrences, with one row per element even
when several magnets share one support. Its `body_kind` identifies the support;
`element_type` identifies the Xtrack element. Unattached elements remain in
`line.get_table()`.

No-argument mechanics tables inspect configuration without evaluating sources;
supplying `time_s` requests composed world poses. `reference_*`, `static_*` and
`total_*` columns carry positions, survey angles and full rigid matrices.
`installation_X/Y/Z` and `dynamic_X/Y/Z` are world displacement contributions.
Relative angles are extracted from relative rotation matrices, not differences
of Euler angles. In a history plot, pitch/yaw/roll are relative to the installed
element axes, using the Xsuite survey convention.

`get_time_series` queries one selected entity. For analytic sources without a
bound record, supply `time_s=times`. The history includes static placement,
inherited support motion and the element's rotation lever arm; a source record's
`get_table(body)` instead shows direct input coordinates. Histories are explicit
bounded host allocations, with `max_memory_bytes`, and never create a tracking
or GPU history cache. They use the current static configuration; changes to
body/source topology require a fresh table. Ordinary numeric snapshot columns
remain available after such changes.

Example 014 also uses the selected row to animate an illustrative quadrupole:

```python
animation = selected.animate(
    time_s=frame_times, size_m=(.3, .3, quad.length), baseline='static',
    displacement_scale=2e5, rotation_scale=4e5)
animation.save('quad.gif', writer='pillow', fps=32)
```

The dimensions are local width, height and length in metres. The hollow cuboid
is display geometry; it does not define a field or aperture. Equal world axis
scales, a dashed baseline and a moving local triad expose rigid translations
and rotations. Neither table queries, history plots nor animations apply motion
to the line. `get_element` gives access to the live Xtrack element, so subsequent
edits still obey the normal mechanics alignment guards.

`mech.survey(time_s=...)` reads nominal/reference/static/total body frames without
writing elements. `mech.plot(...)` uses separate translation and rotation gains;
element spans and feet remain rigid. `mech.animate(...)` updates translations,
orientations and geometry, with bounded frame state. Example 012 exposes an
optional animation. Source records and installation cases have their own tables.
Use `view='bodies'` for a local view and `baseline='static'` to magnify dynamic
motion about the installed placement, without magnifying installation errors.

After `apply`, actual element poses are available through
`line.survey(include_element_frames=True)` and `get_frame(..., which='elem_start')`
/ `which='elem_end'`. The ordinary survey `X/Y/Z` path remains the nominal path.

Automatic turn-held tracking uses prepared context kernels for analytic sources,
rigid composition and alignment writes. Sampled/custom inputs supply a bounded
packed block; the reference backend remains available. Reference-speed passage
sampling is also demonstrated; shared
prescribed sources require an explicit common `time_origin_s` for each line.
No causal shared-response/controller event scheduler is provided. The separate
causal filter has bounded state and preserves it across consecutive blocks.

The reference lowering temporarily writes effective alignment fields and rejects
conflicting raw edits. Slices, shared element aliases, alignment expressions and
built aperture models remain guarded. An independent ring copy is presently
needed to avoid shared construction sublines. Actual GPU performance has not
been validated here. The [design notes](DESIGN.md) explain these boundaries and
the next backend experiments.

Operation timings separate line load/copy, explicit tracker preparation,
reference adoption, mechanics definition and optional geometry/table inspection.
Required surveys inside attachment remain part of mechanics definition. There
is no inclusive `setup_s` comparison across the teaching scripts. Example 010
isolates repeated update/tracking costs; its load and selection phases are also
separate. GIF rendering/encoding in 014 has its own timing. These are individual
operation measurements; the runner separately captures complete script elapsed
time and process memory.

All work stays on the local exploration branch. Publishing and clean commit
slicing remain for later review.
