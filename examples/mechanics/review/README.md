<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# Captured mechanics walkthrough

CPU capture on 20 November 2026, native Xtrack 0.119.2, Python 3.14 and NumPy
2.4.1 on macOS/ARM. The fourteen lessons follow the feedback script formatting.
The native implementation is in `xtrack/mechanics`; no scratch/feedback module
is imported. These are demonstration inputs and explicit physical assumptions,
not an FCC-ee mechanical model or performance prediction.

Each result directory contains terminal output, SVG/PNG figures and result.json.
The [manifest](results/manifest.json) records the current script and native source
hashes, individual phase timings and generated artifacts. Worker elapsed time
excludes initial imports but includes figure capture and requested GIF rendering;
instrumented phases exclude interactive display. Process peak RSS includes the
scientific Python stack and rendering/encoding allocations.

The setup boundaries now separate line load/copy, explicit tracker preparation,
reference adoption, mechanics definition and optional geometry/table inspection.
Surveys required inside attachment remain part of its cost. Application and
subsequent pose inspection are separately measured. These are named operation
costs, rather than an inclusive `setup_s` comparison between teaching workflows.
Example 010 provides the repeated runtime comparison; 014 separately measures
selected-history queries, plotting and GIF encoding.

No CuPy or PyOpenCL runtime is installed locally, so no GPU result is claimed.

```bash
PYTHONPATH=. python examples/mechanics/review/run_examples.py
# Or capture one review slice, preserving the other manifest entries:
PYTHONPATH=. python examples/mechanics/review/run_examples.py --examples 014
```

This revision passed 53 focused mechanics checks, including table selection,
finite rigid composition, source inspection, memory limits, environment copy,
numeric exports and GIF rendering. All fourteen defaults passed and produced
21 figures plus the new 64-frame quadrupole GIF. The earlier branch review also
ran environment/API/aperture regression checks and the five alternative input
configurations linked below; those alternative captures retain their original
19 November source hashes.

| Slice | Captured script + figures [s] | Peak process RSS [MiB] |
| --- | ---: | ---: |
| [001](../001_rigid_girder.py) | 0.43 | 248 |
| [002](../002_sources_and_vibrations.py) | 0.48 | 251 |
| [003](../003_shared_girder.py) | 0.34 | 244 |
| [004](../004_demo_ring.py) | 3.47 | 324 |
| [005](../005_ground_and_feet.py) | 0.70 | 274 |
| [006](../006_transfer_chain.py) | 1.25 | 309 |
| [007](../007_local_sources.py) | 0.70 | 278 |
| [008](../008_regional_ground.py) | 4.12 | 449 |
| [009](../009_moving_bpm.py) | 3.76 | 267 |
| [010](../010_execution_costs.py) | 18.63 | 255 |
| [011](../011_installation_and_correction.py) | 6.51 | 271 |
| [012](../012_matrix_response.py) | 0.59 | 260 |
| [013](../013_combined_mechanics.py) | 7.36 | 827 |
| [014](../014_tables_and_quad_animation.py) | 4.43 | 519 |

## 001 — Reference, installation and dynamics

[source](../001_rigid_girder.py), [terminal](results/001_rigid_girder/terminal.txt), [figure 1](results/001_rigid_girder/figure_01.svg).

The terminal retains the existing intentional quadrupole x/y inputs of 1 mm / 3 µm, adds a 20 µm girder installation error and samples 50 nm dynamic motion. Disabling leaves q1 at 23 µm; clearing restores its original 3 µm and preserves q2's 0.1 rad design roll. The 1 mm intentional x offset is never retyped into a mechanics definition.

In the figure, grey is the reference and blue is the installed/moving support and element spans. The separate orientation panel uses a proper amplified rotation. Translation/rotation display gains never enter tracking. This lesson clears mechanics at the end deliberately; the saved survey still describes the applied state.

Measured operations: `line_construction_s` 17.119 ms; `tracker_build_s` 0.180 s; `reference_adoption_s` 0.047 ms; `reference_inspection_s` 0.023 ms; `mechanics_definition_s` 21.360 ms; `installation_definition_s` 0.000 ms; `source_definition_s` 0.024 ms; `apply_s` 0.275 ms; `pose_inspection_s` 2.605 ms; `one_pass_tracking_s` 0.158 ms.

## 002 — Interchangeable sources, six coordinates and slow drift

[source](../002_sources_and_vibrations.py), [terminal](results/002_sources_and_vibrations/terminal.txt), [figure 1](results/002_sources_and_vibrations/figure_01.svg).

The default is a 50 nm, 500 Hz vertical tone, with pitch/yaw/roll peaks of 0.2 / 0.3 / 0.4 µrad and an independent slow longitudinal drift. The 256 tracked turns span 0.000344985 s. Disabling vibration leaves the 20 µm installation placement. The reusable five-input record occupies 960 KiB.

The first two panels show the first 10 ms; the drift panel intentionally shows the full 0.4095 s record. Installation is excluded from these input plots. The PSD and measured translation/rotation alternatives were also executed, not just left as commented suggestions. Frozen Twiss is a separate explicitly applied state, with automatic vibration disabled.

Measured operations: `line_construction_s` 17.293 ms; `tracker_build_s` 0.131 s; `reference_adoption_s` 0.064 ms; `mechanics_definition_s` 21.606 ms; `installation_definition_s` 0.000 ms; `sample_motion_s` 8.211 ms; `dynamic_tracking_s` 95.019 ms.

## 003 — Shared physical body, different line axes and clocks

[source](../003_shared_girder.py), [terminal](results/003_shared_girder/terminal.txt), [figure 1](results/003_shared_girder/figure_01.svg).

The beams have 0.336900 and 0.603751 µs revolution intervals. At the same physical time, local x is -50 / +50 nm for the opposite traversal directions. Equal turn numbers therefore do not mean equal physical time. The quadrupoles are unpowered in this bookkeeping lesson.

The upper panel converts both samples into world X, where they lie on the same prescribed curve. The lower panel plots line-local values against each beam's turn number and exposes both the sign and clock differences. Each line updates its own fields; the immutable source is not advanced by the other beam. Stateful shared controllers/filters are outside this scheduling example.

Measured operations: `line_construction_s` 17.776 ms; `forward_tracker_build_s` 0.128 s; `backward_tracker_build_s` 1.254 ms; `mechanics_definition_s` 35.498 ms; `source_definition_s` 0.053 ms; `sample_motion_s` 0.337 ms; `forward_tracking_s` 15.138 ms; `backward_tracking_s` 14.630 ms.

## 004 — Whole surveyed demo ring

[source](../004_demo_ring.py), [terminal](results/004_demo_ring/terminal.txt), [figure 1](results/004_demo_ring/figure_01.svg), [figure 2](results/004_demo_ring/figure_02.svg).

This is the same 1.5 km demonstration lattice as feedback, with 168 illustrative independent quadrupole supports. The applied entrance frames agree with requested poses to 1.14e-13 m and 2.82e-16 rad. Those are numerical transform checks, not site-model accuracy. The short six-coordinate input payload is 15.75 MiB on the host.

Figure 1 shows spatial phase on the ring and separately magnifies the first support's installation orientation. Figure 2 shows real world displacement versus s and local input history. Wave direction, polarization and speed are independent parameters. The PSD/measured switches were executed too. Bends/sextupoles are intentionally not assigned in this teaching layout; actual FCC-ee girder assignments remain machine data.

Measured operations: `line_load_and_copy_s` 60.653 ms; `tracker_build_s` 0.140 s; `reference_adoption_s` 1.775 ms; `geometry_selection_s` 71.204 ms; `mechanics_definition_s` 2.597 s; `installation_definition_s` 0.001 ms; `sample_motion_s` 17.351 ms; `pose_prediction_s` 6.255 ms; `apply_s` 15.468 ms; `survey_verification_s` 31.266 ms.

## 005 — Floor contacts to rigid support motion

[source](../005_ground_and_feet.py), [terminal](results/005_ground_and_feet/terminal.txt), [figure 1](results/005_ground_and_feet/figure_01.svg), [figure 2](results/005_ground_and_feet/figure_02.svg).

The common floor is sampled at four stored feet per support. Weighted rigid fitting produces all six coordinates, including approximately 7.31 nrad pitch and 7.20 nrad roll RMS for the first support. The overall contact mismatch is 1.709 nm RMS: the rigid body cannot satisfy every imposed floor displacement exactly.

Figure 1 is a local view of one support, its feet and element span; the orientation gain is explicit. Figure 2 shows translations, survey angles and the residual. The residual is retained rather than converted into girder bending. This is a kinematic fit followed by a chosen response model, not a prediction of contact forces or support dynamics. The correlated-PSD floor option also ran.

Measured operations: `line_load_and_copy_s` 58.838 ms; `tracker_build_s` 0.147 s; `reference_adoption_s` 1.673 ms; `mechanics_definition_s` 0.104 s; `source_geometry_s` 0.021 ms; `sample_and_fit_s` 22.190 ms; `apply_s` 1.876 ms; `pose_inspection_s` 2.877 ms.

## 006 — Loaded floor/girder/magnet/BPM response

[source](../006_transfer_chain.py), [terminal](results/006_transfer_chain/terminal.txt), [figure 1](results/006_transfer_chain/figure_01.svg), [figure 2](results/006_transfer_chain/figure_02.svg).

One matched floor record gives 9.103 nm RMS at the floor, 7.847 at the loaded girder, 8.681 at the magnet and 8.902 at the pickup. The independently unloaded cascade gives 9.390 nm at the magnet: a different physical model. The consistent loaded cascade agrees with the direct two-mass solution within 1.99e-23 m.

Figure 1 separates gain, phase, PSD and the small BPM-minus-magnet motion. All outputs are bound as absolute responses, so parent motion is not added twice. Figure 2 is a distinct causal held-step experiment, starting at rest and settling to 50 nm. Whole-record and split-block filtering agree exactly. Offline FFT response is periodic; causal startup uses a stateful ZOH-discretized model. The pickup's reaction loading is neglected explicitly.

Measured operations: `line_load_and_copy_s` 58.924 ms; `tracker_build_s` 96.473 ms; `reference_adoption_s` 1.672 ms; `geometry_selection_s` 76.164 ms; `mechanics_definition_s` 27.177 ms; `response_definition_s` 0.028 ms; `sample_and_response_s` 32.364 ms; `apply_s` 2.050 ms; `pose_inspection_s` 3.146 ms; `causal_filter_s` 0.847 ms.

## 007 — Mounted pump and supplied propagation

[source](../007_local_sources.py), [terminal](results/007_local_sources/terminal.txt), [figure 1](results/007_local_sources/figure_01.svg).

The pump location is a child mount of the first girder. Receiver distances are 0.943, 19.790 and 39.463 m, with amplitude gains 0.954, 0.372 and 0.139 and delays 18.87, 395.81 and 789.26 ms. One latent waveform drives every receiver. An additional 25 nm direct support tone is bound separately.

The left panel is the stated exponential floor footprint; the right isolates floor-path traces and compares an alternative pump mount. The retained survey contains the direct-plus-floor result. Position alone does not supply elastic propagation: decay, polarization, speed and direct coupling are user-supplied assumptions. The source is prescribed displacement, not a force-source calculation.

Measured operations: `line_load_and_copy_s` 58.994 ms; `tracker_build_s` 0.144 s; `reference_adoption_s` 1.830 ms; `mechanics_definition_s` 0.105 s; `source_geometry_s` 0.014 ms; `sample_field_s` 0.583 ms; `apply_s` 2.103 ms; `pose_inspection_s` 3.004 ms.

## 008 — Regional PSD, correlation length and continuous random modes

[source](../008_regional_ground.py), [terminal](results/008_regional_ground/terminal.txt), [figure 1](results/008_regional_ground/figure_01.svg), [figure 2](results/008_regional_ground/figure_02.svg).

The observed vertical RMS is 19.066 nm near the excess region and 6.059 nm far away. Time chunks and reordered receiver subsets reproduce the same field (zero discrepancy in this CPU capture). The world-vector field payload is 7.894 MiB; an independent horizontal component is projected into each support frame too.

Figure 1 compares spatial RMS, local PSDs and band-integrated amplitude correlation with the ensemble target. The correlation RMS deviation is 0.0341 in this finite-mode, finite-time realization; it is not a machine-precision kernel claim. Increasing modes and record length is a model-convergence experiment. The regional PSD contribution is gain squared times excess PSD, retaining the shared background. Figure 2 shows the resulting ring motion. The region reappears near both ends of s because those locations are close in world space on the closed ring.

Measured operations: `line_load_and_copy_s` 58.751 ms; `tracker_build_s` 96.681 ms; `reference_adoption_s` 1.731 ms; `geometry_selection_s` 8.556 ms; `mechanics_definition_s` 2.710 s; `source_geometry_s` 0.017 ms; `source_definition_s` 0.489 ms; `sample_field_s` 0.121 s; `chunk_and_order_check_s` 17.692 ms.

## 009 — Moving electrical pickup changes the correction target

[source](../009_moving_bpm.py), [terminal](results/009_moving_bpm/terminal.txt), [figure 1](results/009_moving_bpm/figure_01.svg).

The nominal-frame beam RMS is 168.297 nm; the pickup itself moves by 67.835 nm RMS, giving an actual BPM signal of 235.523 nm RMS. Correction based on nominal beam y leaves 67.729 nm at the real pickup. Using the moved electrical plane removes that residual to the displayed precision in this ideal noiseless frozen model.

The upper panel separates beam, pickup and measured motion. The lower panel compares the two correction targets. The signed response is measured, approximately -115.818 m/rad, rather than assumed positive. Ray/plane intersection includes orientation and beam slope. This is a sequence of frozen Twiss states, not online feedback tracking; it does not import the feedback implementation.

Measured operations: `line_load_and_copy_s` 67.216 ms; `tracker_build_s` 0.135 s; `reference_adoption_s` 1.734 ms; `geometry_selection_s` 72.579 ms; `mechanics_definition_s` 27.172 ms; `readout_calibration_s` 0.124 s; `response_records_s` 0.951 ms; `frozen_states_and_correction_s` 3.173 s.

## 010 — What the reference backend costs

[source](../010_execution_costs.py), [terminal](results/010_execution_costs/terminal.txt), [figure 1](results/010_execution_costs/figure_01.svg).

The workload is 128 particles and 128 turns on the same ring. The table below separates attachment, pose and write costs from frozen/dynamic tracking. Kernel first use is timed separately. These are one local CPU run with available kernel caches, not hardware-independent guarantees or a comparison of identical motion semantics.

The left panel shows that source-independent line validation and Python writes already cost milliseconds, then grow with support count. The right shows the sampling assumption: at 20 kHz, 2π f T_rev = 0.629 rad and turn-held versus passage-prepared offsets differ by 30.715 nm. The 168-support passage preparation is about 0.52 s because the reference repeats hierarchy evaluation. A projected million-turn full 4x4 history would occupy 20.027 GiB; it is not allocated.

| Supports | Load/copy [s] | Select [ms] | Build [s] | Attach [s] | Pose [ms] | Apply [ms] | Frozen 128 turns [s] | Dynamic 128 turns [s] | Passage [ms] |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.064 | 2.344 | 0.163 | 0.078 | 0.059 | 1.581 | 1.911 | 2.154 | 2.549 |
| 24 | 0.049 | 2.121 | 0.018 | 0.387 | 0.428 | 3.112 | 2.031 | 2.365 | 13.642 |
| 168 | 0.049 | 2.042 | 0.019 | 2.737 | 3.145 | 13.367 | 2.023 | 3.613 | 519.672 |

## 011 — Installation/correction seed studies

[source](../011_installation_and_correction.py), [terminal](results/011_installation_and_correction/terminal.txt), [figure 1](results/011_installation_and_correction/figure_01.svg).

The intended 1 mm quad shift remains the adopted reference. Independent static draws give about 4.49–5.29 µm quad y RMS; changing dynamic seeds is not part of these draws. Four initial BPM RMS errors of approximately 4490, 9786, 131 and 861 µm reduce to 10.70, 13.12, 8.09 and 10.58 µm with the available correctors.

The figure compares seeds, a final seed's spatial residual and the SVD spectrum. Seven response modes are retained from 14 BPMs and 9 correctors. The bare teaching ring's vertical tune is 10.01745 with large insertion beta, so its large uncorrected sensitivity is not an FCC-ee prediction. Installation records are immutable; corrector knobs remain a separate control layer. Clearing mechanics leaves the deliberate quad offset unchanged.

Measured operations: `line_load_and_copy_s` 65.501 ms; `tracker_build_s` 0.133 s; `reference_adoption_s` 1.685 ms; `geometry_selection_s` 7.602 ms; `mechanics_definition_s` 3.034 s; `response_calibration_s` 1.842 s; `installation_cases_and_correction_s` 1.199 s.

## 012 — Matrix response versus geometric lever arm

[source](../012_matrix_response.py), [terminal](results/012_matrix_response/terminal.txt), [figure 1](results/012_matrix_response/figure_01.svg), [figure 2](results/012_matrix_response/figure_02.svg).

The supplied 6x6 complex matrix includes yaw-to-x coupling in m/rad and y-to-pitch coupling in rad/m. Support/magnet x RMS is 8.677 / 10.249 nm, and y RMS 46.742 / 56.272 nm. Their difference comes from the separate 1 m mount lever arm. Pitch is 9.533 nrad RMS at both ports in the common nominal axes.

Figure 1 separates input, responded support and transported magnet histories. Figure 2 is a local geometry/orientation view. Matrix coupling and coordinate transport are different operations: the response is a supplied linear model and transport is exact rigid geometry. The optional animation was rendered with 40 frames and updates geometry as well as orientation.

Measured operations: `line_load_and_copy_s` 58.341 ms; `tracker_build_s` 0.142 s; `reference_adoption_s` 1.740 ms; `geometry_selection_s` 71.802 ms; `mechanics_definition_s` 13.015 ms; `matrix_response_and_transport_s` 3.092 ms.

## 013 — Combined prescribed-mechanics ring

[source](../013_combined_mechanics.py), [terminal](results/013_combined_mechanics/terminal.txt), [figure 1](results/013_combined_mechanics/figure_01.svg), [figure 2](results/013_combined_mechanics/figure_02.svg).

The capstone connects 168 supports, 168 magnet bodies, 14 pickups and 672 feet. Background and regional PSD fields plus a mounted pump are sampled jointly; fitted motion passes through supplied conditional causal responses, with port transport, independent installation errors and slow drift. The 1.5 s prehistory is explicit. The supplied response laws are assumed to represent the loaded ports; they are not inferred from equipment coordinates.

The contact mismatch after warm-up is 0.392 nm RMS. First-channel support/magnet/pickup y RMS is 15.289 / 20.493 / 18.675 nm over this short realization. Figure 1 shows the combined response histories. Figure 2 uses the installed placement as its baseline, so installation errors do not mask magnified dynamics. The 32 tracked turns span only 0.000160111 s: this is a scheduling smoke check. The waveform study spans 0.5 s. At 1024 Hz replay, a 40 Hz sinusoid has 0.752% worst half-step amplitude attenuation; this sampling error is separate from transform accuracy.

The process peak is about 827 MiB, including Python, figures and temporary batched fitting/transport arrays. Explicit prepared support/magnet/pickup payloads total about 32.8 MiB on the host; there is no full GPU history. This demonstrates the UI composition and exposes the need for bounded streaming before long whole-machine studies.

Measured operations: `line_load_and_copy_s` 59.870 ms; `tracker_build_s` 0.148 s; `reference_adoption_s` 1.697 ms; `geometry_selection_s` 76.364 ms; `mechanics_definition_s` 2.917 s; `installation_definition_s` 2.090 ms; `floor_and_rigid_fit_s` 0.732 s; `response_paths_s` 1.233 s; `initial_frozen_twiss_s` 80.138 ms; `dynamic_tracking_s` 0.828 s.

## 014 — Select a quadrupole, plot its history and watch it move

[source](../014_tables_and_quad_animation.py), [terminal](results/014_tables_and_quad_animation/terminal.txt), [history](results/014_tables_and_quad_animation/figure_01.svg), [cuboid view](results/014_tables_and_quad_animation/figure_02.svg), [64-frame GIF](results/014_tables_and_quad_animation/motion.gif).

The source inventory shows six harmonic bindings. Selecting `kind='girder'`
gives one physical support; selecting `element_type='Quadrupole'` gives two
separate element rows on that support. The selected `q1` retains its intentional
1 mm horizontal offset and 0.15 rad design roll. The installed support has a
20 µm vertical placement error and 2 µrad yaw. Table queries never apply motion.

`selected.get_element()` accesses the live quadrupole.
`selected.get_time_series()` composes 257 samples over the bound two-second
record. The largest difference between magnet and support vertical inputs is
200.227 nm, due to the finite rotation lever arm. The history figure shows
world XYZ displacement in nm and pitch/yaw/roll in nrad relative to the installed
element axes; there is no display magnification in those curves.

The GIF uses a 0.3 × 0.3 × 0.5 m illustrative hollow cuboid, a moving local triad,
and equal world axis scales. Translations are magnified by 200,000 and rotations
by 400,000 about the installed baseline. The dashed box is that baseline. Every
surface follows the same proper rigid transform. The cuboid defines neither a
magnetic field nor an aperture, and rendering leaves all line inputs unchanged.

Selected-history preparation took 20.36 ms; animation
preparation took 13.96 ms. Rendering/encoding the
64-frame GIF took 3.99 s. Process peak RSS was
519 MiB, including scientific Python, plots and Pillow's
encoding frames; this is an explicit diagnostic artifact, not per-turn tracking
memory or a GPU pose cache.

Measured operations: `line_construction_s` 18.159 ms; `tracker_build_s` 0.132 s; `reference_adoption_s` 0.065 ms; `mechanics_definition_s` 20.561 ms; `source_definition_s` 0.063 ms; `sample_motion_s` 0.113 ms; `table_inspection_s` 3.401 ms; `selected_history_s` 20.364 ms; `history_plot_s` 35.249 ms; `animation_preparation_s` 13.958 ms; `gif_render_and_encode_s` 3.992 s.

## Alternative inputs and animation

The additional [local assembly review](demos/README.md) uses the native model to
animate a 3.5 m quad/drift/bend/drift/quad sequence on one rigid girder, with
separate relative magnet motion. Its [100-frame GIF](results/015_girder_assembly/girder_assembly.gif)
and [terminal output](results/015_girder_assembly/terminal.txt) are separate from
the fourteen default captures and their manifest. Two local presentation checks
bring the combined focused test run to 55 passing cases.

These earlier captures (19 November) changed only the stated user parameters in memory; their result
metadata records the overrides, original/executed code hashes and native source
hashes. The numbered files remain ordinary editable scripts.

- [002 displacement + angular PSD](results/variants/002_psd/terminal.txt).
- [002 measured displacement + angular traces](results/variants/002_measured/terminal.txt).
- [004 whole-ring PSD](results/variants/004_psd/terminal.txt).
- [004 stated whole-ring motion](results/variants/004_measured/terminal.txt).
- [005 correlated floor PSD and feet](results/variants/005_psd/terminal.txt).
- [012 animated six-coordinate response](results/variants/012_animation/motion.gif).

## What the costs point to next

The next runtime slice should compile reusable geometry/binding indices and
compare occurrence-indexed effective buffers with girder boundary transforms.
Keep deliberate/installation definitions authoritative, and make update cost
scale with the current state rather than Python setters or repeated full-line
surveys. Passage sampling particularly needs pointwise/batched evaluation.

Long records need chunked field generation and retained causal state, with
selected/decimated diagnostics rather than whole-machine histories. The random
mode approximation needs convergence against the spatial and spectral model;
an arbitrary measured field is not necessarily low rank. Actual GPU measurement
is required before selecting a fast path.

Geometry integration remains a separate slice: shared/sliced occurrences,
pipe/aperture ownership, tracking monitors reading their moved electrical frames,
and joint mechanics/feedback event scheduling. The observation geometry is native
now, but the frozen correction lessons are not a completed online feedback
integration. [Design decisions](../DESIGN.md) retain the conventions and limits.
