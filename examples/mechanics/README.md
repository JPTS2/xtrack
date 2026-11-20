<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# Experimental rigid-body mechanics

Build prescribed motion around an existing Xtrack line with `env.mechanics`
and `line.mechanics`. Keep deliberate magnet offsets, installation errors and
dynamic motion separate. Inspect the result with normal tables, survey frames
and magnified plots. The examples are the main entry point for reviewing this
experimental UI.

Start with **001** for the three placement layers, **004** for an existing
surveyed ring, and **015** for bunch arrival phases, frozen Twiss and driven
tracking. **016** runs the requested thick FCC-ee input with a common floor,
girder/magnet responses and a selected virtual pickup. Each script leaves its
ordinary Xsuite objects available after `%run`. The portable lessons use the
same bundled teaching ring as feedback and require no external machine download
or feedback implementation. The FCC lesson uses the explicitly supplied model.

```bash
PYTHONPATH=. XSUITE_ALLOW_KERNEL_COMPILATION=1 python examples/mechanics/001_rigid_girder.py
PYTHONPATH=. XSUITE_ALLOW_KERNEL_COMPILATION=1 python examples/mechanics/004_demo_ring.py
PYTHONPATH=. XSUITE_ALLOW_KERNEL_COMPILATION=1 python examples/mechanics/015_bunch_phases_and_twiss.py
```

Or start IPython with this checkout first on `PYTHONPATH`, then use e.g.
`%run examples/mechanics/004_demo_ring.py`. Edit the **User parameters** at the
start of each lesson. An 80-character `#` heading introduces a teaching stage;
40-character headings separate setup, model construction, application and
inspection. Timings distinguish required setup from optional diagnostics.

NumPy/SciPy and the normal Xsuite dependencies are required. Matplotlib provides
figures; 014 also uses Pillow to save its GIF. GUI interaction is excluded from
the approximate CPU times below. Cold kernel compilation can dominate a first
run. These are teaching models, not FCC-ee engineering predictions.

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
| [016_fcc_ground_motion.py](016_fcc_ground_motion.py) | Actual FCC Z line, 2,663 supports, common ground/feet, conditional girder/magnet/pickup responses and full tracking costs | About 24 s, including setup and four figures |


Read the [API contract](API.md) for ownership, coordinate/sign conventions,
source references, clocks, lifecycle and current restrictions. The shared
[demo-ring description](../../test_data/example_lattices/demo_ring/README.md)
states its optics and limitations. Its near-integer vertical tune makes it
particularly sensitive to installation errors.

Lessons 001–015 use bundled inputs. For 016, the [FCC input instructions](review/FCC_INPUT.md)
pin the requested development model and give a one-command input override.
The default path already works in this workspace. Its support grouping and
response data are explicit demonstration assumptions, not engineering assignments.
The [FCC walkthrough](FCC_DEMO.md) explains its terminal costs and figures; the
default takes about 24 s locally, including source preparation and plotting.

A typical setup keeps the machine's existing offsets:

```python
mech = env.mechanics
line.mechanics.adopt_reference()
girder = line.mechanics.new_girder('g1', elements=['q1', 'q2'])
girder.installation.shift_y = 20e-6
mech.add_source('floor_y', xt.mechanics.Harmonic(
    amplitude=50e-9, frequency_hz=5.), targets=['g1'], component='y')
mech.apply(time_s=.05)
```

Tables are useful during model building and after application:

```python
mech.get_source_table().show()
girders = mech.get_table(time_s=.05).rows.match(kind='girder')
quad = line.mechanics.get_table(time_s=.05).rows['q1']
quad.get_element()
history = quad.get_time_series(time_s=times)
history.plot()
```

These queries do not move the line. To track turn-held motion, explicitly call
`line.mechanics.enable_vibrations(...)`, then ordinary `line.track(...)`.
`disable_vibrations()` removes dynamic motion while keeping installation;
`env.mechanics.clear()` removes mechanics and restores adopted element inputs.

The current backend supports prescribed rigid motion and bounded source records.
It rejects sliced/shared element objects, live alignment expressions, stale
reference geometry and built-aperture application. Passage-time sampling is an
explicit slower comparison. Lesson 015 uses independent bunch runs with a common
source and different time origins; packed multibunch maps are later work.

The [hardware acceptance instructions](review/HARDWARE_ACCEPTANCE.md) provide
explicit CPU/OpenMP/CUDA/OpenCL commands and a parameterized FCC timing probe.
CPU execution is validated locally. Actual GPU validation remains pending;
portable kernel source and CPU timings do not establish a GPU result.
The [delivery evidence](DELIVERY.md) records the current scope and reproducible
checks. A first working draft can expose this UI with those boundaries.

On the local exploration branch, `designs/` and `review/results/` retain design
comparisons, captures and benchmark history. They are separate from the focused
draft file selection. Publishing and clean commit slicing remain user-controlled.
