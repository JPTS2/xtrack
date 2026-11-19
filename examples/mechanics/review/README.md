<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# Captured mechanics walkthrough

CPU capture on 19 November 2026, native Xtrack 0.119.2, Python 3.14 and NumPy
2.4.1 on macOS/ARM. The thirteen lessons follow the feedback script formatting.
The native implementation is in `xtrack/mechanics`; no scratch/feedback module
is imported. These are demonstration inputs and explicit physical assumptions,
not an FCC-ee mechanical model or performance prediction.

Each result directory contains terminal output, SVG/PNG figures and result.json.
The [manifest](results/manifest.json) records source hashes and phase timings.
Worker elapsed time excludes initial imports but includes figure capture;
instrumented phases exclude interactive display. Setup may include implicit
tracker preparation through survey; 010 builds explicitly before attachment to
separate those costs. Process peak RSS includes the scientific Python stack.
No CuPy or PyOpenCL runtime is installed locally, so no GPU result is claimed.

```bash
PYTHONPATH=. python examples/mechanics/review/run_examples.py
# Or capture one review slice, preserving the other manifest entries:
PYTHONPATH=. python examples/mechanics/review/run_examples.py --examples 006
```

The physics/API checks passed 68 focused cases, including existing environment
and aperture regression checks. All thirteen defaults and five alternative input
configurations passed; the optional 40-frame animation rendered successfully.

| Slice | Captured script + figures [s] | Peak process RSS [MiB] |
| --- | ---: | ---: |
| [001](../001_rigid_girder.py) | 0.29 | 241 |
| [002](../002_sources_and_vibrations.py) | 0.39 | 246 |
| [003](../003_shared_girder.py) | 0.28 | 240 |
| [004](../004_demo_ring.py) | 2.96 | 316 |
| [005](../005_ground_and_feet.py) | 0.56 | 263 |
| [006](../006_transfer_chain.py) | 0.94 | 297 |
| [007](../007_local_sources.py) | 0.57 | 258 |
| [008](../008_regional_ground.py) | 3.42 | 426 |
| [009](../009_moving_bpm.py) | 2.81 | 262 |
| [010](../010_execution_costs.py) | 15.76 | 268 |
| [011](../011_installation_and_correction.py) | 5.47 | 266 |
| [012](../012_matrix_response.py) | 0.50 | 253 |
| [013](../013_combined_mechanics.py) | 6.19 | 808 |

## 001 — Reference, installation and dynamics

[source](../001_rigid_girder.py), [terminal](results/001_rigid_girder/terminal.txt), [figure 1](results/001_rigid_girder/figure_01.svg).

The terminal retains the existing intentional quadrupole x/y inputs of 1 mm / 3 µm, adds a 20 µm girder installation error and samples 50 nm dynamic motion. Disabling leaves q1 at 23 µm; clearing restores its original 3 µm and preserves q2's 0.1 rad design roll. The 1 mm intentional x offset is never retyped into a mechanics definition.

In the figure, grey is the reference and blue is the installed/moving support and element spans. The separate orientation panel uses a proper amplified rotation. Translation/rotation display gains never enter tracking. This lesson clears mechanics at the end deliberately; the saved survey still describes the applied state.

Instrumented costs: `setup_s` 0.1339 s; `apply_and_survey_s` 0.0018 s; `tracker_build_s` 0.0011 s; `one_pass_tracking_s` 0.0001 s.

## 002 — Interchangeable sources, six coordinates and slow drift

[source](../002_sources_and_vibrations.py), [terminal](results/002_sources_and_vibrations/terminal.txt), [figure 1](results/002_sources_and_vibrations/figure_01.svg).

The default is a 50 nm, 500 Hz vertical tone, with pitch/yaw/roll peaks of 0.2 / 0.3 / 0.4 µrad and an independent slow longitudinal drift. The 256 tracked turns span 0.000344985 s. Disabling vibration leaves the 20 µm installation placement. The reusable five-input record occupies 960 KiB.

The first two panels show the first 10 ms; the drift panel intentionally shows the full 0.4095 s record. Installation is excluded from these input plots. The PSD and measured translation/rotation alternatives were also executed, not just left as commented suggestions. Frozen Twiss is a separate explicitly applied state, with automatic vibration disabled.

Instrumented costs: `setup_s` 0.1400 s; `sample_motion_s` 0.0063 s; `tracker_build_s` 0.0000 s; `dynamic_tracking_s` 0.0835 s.

## 003 — Shared physical body, different line axes and clocks

[source](../003_shared_girder.py), [terminal](results/003_shared_girder/terminal.txt), [figure 1](results/003_shared_girder/figure_01.svg).

The beams have 0.336900 and 0.603751 µs revolution intervals. At the same physical time, local x is -50 / +50 nm for the opposite traversal directions. Equal turn numbers therefore do not mean equal physical time. The quadrupoles are unpowered in this bookkeeping lesson.

The upper panel converts both samples into world X, where they lie on the same prescribed curve. The lower panel plots line-local values against each beam's turn number and exposes both the sign and clock differences. Each line updates its own fields; the immutable source is not advanced by the other beam. Stateful shared controllers/filters are outside this scheduling example.

Instrumented costs: `setup_and_record_s` 0.1456 s; `forward_tracker_build_s` 0.0000 s; `forward_tracking_s` 0.0127 s; `backward_tracker_build_s` 0.0000 s; `backward_tracking_s` 0.0128 s.

## 004 — Whole surveyed demo ring

[source](../004_demo_ring.py), [terminal](results/004_demo_ring/terminal.txt), [figure 1](results/004_demo_ring/figure_01.svg), [figure 2](results/004_demo_ring/figure_02.svg).

This is the same 1.5 km demonstration lattice as feedback, with 168 illustrative independent quadrupole supports. The applied entrance frames agree with requested poses to 1.14e-13 m and 2.82e-16 rad. Those are numerical transform checks, not site-model accuracy. The short six-coordinate input payload is 15.75 MiB on the host.

Figure 1 shows spatial phase on the ring and separately magnifies the first support's installation orientation. Figure 2 shows real world displacement versus s and local input history. Wave direction, polarization and speed are independent parameters. The PSD/measured switches were executed too. Bends/sextupoles are intentionally not assigned in this teaching layout; actual FCC-ee girder assignments remain machine data.

Instrumented costs: `setup_s` 2.4530 s; `sample_motion_s` 0.0163 s; `apply_and_survey_s` 0.0507 s.

## 005 — Floor contacts to rigid support motion

[source](../005_ground_and_feet.py), [terminal](results/005_ground_and_feet/terminal.txt), [figure 1](results/005_ground_and_feet/figure_01.svg), [figure 2](results/005_ground_and_feet/figure_02.svg).

The common floor is sampled at four stored feet per support. Weighted rigid fitting produces all six coordinates, including approximately 7.31 nrad pitch and 7.20 nrad roll RMS for the first support. The overall contact mismatch is 1.709 nm RMS: the rigid body cannot satisfy every imposed floor displacement exactly.

Figure 1 is a local view of one support, its feet and element span; the orientation gain is explicit. Figure 2 shows translations, survey angles and the residual. The residual is retained rather than converted into girder bending. This is a kinematic fit followed by a chosen response model, not a prediction of contact forces or support dynamics. The correlated-PSD floor option also ran.

Instrumented costs: `setup_s` 0.2563 s; `sample_and_fit_s` 0.0189 s; `apply_and_survey_s` 0.0034 s.

## 006 — Loaded floor/girder/magnet/BPM response

[source](../006_transfer_chain.py), [terminal](results/006_transfer_chain/terminal.txt), [figure 1](results/006_transfer_chain/figure_01.svg), [figure 2](results/006_transfer_chain/figure_02.svg).

One matched floor record gives 9.103 nm RMS at the floor, 7.847 at the loaded girder, 8.681 at the magnet and 8.902 at the pickup. The independently unloaded cascade gives 9.390 nm at the magnet: a different physical model. The consistent loaded cascade agrees with the direct two-mass solution within 1.99e-23 m.

Figure 1 separates gain, phase, PSD and the small BPM-minus-magnet motion. All outputs are bound as absolute responses, so parent motion is not added twice. Figure 2 is a distinct causal held-step experiment, starting at rest and settling to 50 nm. Whole-record and split-block filtering agree exactly. Offline FFT response is periodic; causal startup uses a stateful ZOH-discretized model. The pickup's reaction loading is neglected explicitly.

Instrumented costs: `setup_s` 0.2180 s; `sample_and_response_s` 0.0293 s; `apply_and_survey_s` 0.0039 s; `causal_filter_s` 0.0004 s.

## 007 — Mounted pump and supplied propagation

[source](../007_local_sources.py), [terminal](results/007_local_sources/terminal.txt), [figure 1](results/007_local_sources/figure_01.svg).

The pump location is a child mount of the first girder. Receiver distances are 0.943, 19.790 and 39.463 m, with amplitude gains 0.954, 0.372 and 0.139 and delays 18.87, 395.81 and 789.26 ms. One latent waveform drives every receiver. An additional 25 nm direct support tone is bound separately.

The left panel is the stated exponential floor footprint; the right isolates floor-path traces and compares an alternative pump mount. The retained survey contains the direct-plus-floor result. Position alone does not supply elastic propagation: decay, polarization, speed and direct coupling are user-supplied assumptions. The source is prescribed displacement, not a force-source calculation.

Instrumented costs: `setup_s` 0.2584 s; `sample_field_s` 0.0005 s; `apply_and_survey_s` 0.0037 s.

## 008 — Regional PSD, correlation length and continuous random modes

[source](../008_regional_ground.py), [terminal](results/008_regional_ground/terminal.txt), [figure 1](results/008_regional_ground/figure_01.svg), [figure 2](results/008_regional_ground/figure_02.svg).

The observed vertical RMS is 19.066 nm near the excess region and 6.059 nm far away. Time chunks and reordered receiver subsets reproduce the same field (zero discrepancy in this CPU capture). The world-vector field payload is 7.894 MiB; an independent horizontal component is projected into each support frame too.

Figure 1 compares spatial RMS, local PSDs and band-integrated amplitude correlation with the ensemble target. The correlation RMS deviation is 0.0341 in this finite-mode, finite-time realization; it is not a machine-precision kernel claim. Increasing modes and record length is a model-convergence experiment. The regional PSD contribution is gain squared times excess PSD, retaining the shared background. Figure 2 shows the resulting ring motion. The region reappears near both ends of s because those locations are close in world space on the closed ring.

Instrumented costs: `setup_s` 2.4288 s; `sample_field_s` 0.0810 s; `chunk_and_order_check_s` 0.0089 s.

## 009 — Moving electrical pickup changes the correction target

[source](../009_moving_bpm.py), [terminal](results/009_moving_bpm/terminal.txt), [figure 1](results/009_moving_bpm/figure_01.svg).

The nominal-frame beam RMS is 168.297 nm; the pickup itself moves by 67.835 nm RMS, giving an actual BPM signal of 235.523 nm RMS. Correction based on nominal beam y leaves 67.729 nm at the real pickup. Using the moved electrical plane removes that residual to the displayed precision in this ideal noiseless frozen model.

The upper panel separates beam, pickup and measured motion. The lower panel compares the two correction targets. The signed response is measured, approximately -115.818 m/rad, rather than assumed positive. Ray/plane intersection includes orientation and beam slope. This is a sequence of frozen Twiss states, not online feedback tracking; it does not import the feedback implementation.

Instrumented costs: `setup_and_calibration_s` 0.3505 s; `response_records_s` 0.0008 s; `frozen_states_and_correction_s` 2.3407 s.

## 010 — What the reference backend costs

[source](../010_execution_costs.py), [terminal](results/010_execution_costs/terminal.txt), [figure 1](results/010_execution_costs/figure_01.svg).

The workload is 128 particles and 128 turns on the same ring. The table below separates attachment, pose and write costs from frozen/dynamic tracking. Kernel first use is timed separately. These are one local CPU run with available kernel caches, not hardware-independent guarantees or a comparison of identical motion semantics.

The left panel shows that source-independent line validation and Python writes already cost milliseconds, then grow with support count. The right shows the sampling assumption: at 20 kHz, 2π f T_rev = 0.629 rad and turn-held versus passage-prepared offsets differ by 30.715 nm. The 168-support passage preparation is about 0.41 s because the reference repeats hierarchy evaluation. A projected million-turn full 4x4 history would occupy 20.027 GiB; it is not allocated.

| Supports | Attach [s] | Pose [ms] | Apply [ms] | Frozen 128 turns [s] | Dynamic 128 turns [s] | Passage prepare [ms] |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.063 | 0.034 | 1.312 | 1.675 | 1.836 | 2.2 |
| 24 | 0.332 | 0.352 | 2.700 | 1.665 | 2.011 | 11.9 |
| 168 | 2.327 | 2.430 | 11.088 | 1.677 | 3.109 | 412.3 |

## 011 — Installation/correction seed studies

[source](../011_installation_and_correction.py), [terminal](results/011_installation_and_correction/terminal.txt), [figure 1](results/011_installation_and_correction/figure_01.svg).

The intended 1 mm quad shift remains the adopted reference. Independent static draws give about 4.49–5.29 µm quad y RMS; changing dynamic seeds is not part of these draws. Four initial BPM RMS errors of approximately 4490, 9786, 131 and 861 µm reduce to 10.70, 13.12, 8.09 and 10.58 µm with the available correctors.

The figure compares seeds, a final seed's spatial residual and the SVD spectrum. Seven response modes are retained from 14 BPMs and 9 correctors. The bare teaching ring's vertical tune is 10.01745 with large insertion beta, so its large uncorrected sensitivity is not an FCC-ee prediction. Installation records are immutable; corrector knobs remain a separate control layer. Clearing mechanics leaves the deliberate quad offset unchanged.

Instrumented costs: `setup_and_tracker_s` 2.7274 s; `response_calibration_s` 1.5209 s; `installation_cases_and_correction_s` 1.0289 s.

## 012 — Matrix response versus geometric lever arm

[source](../012_matrix_response.py), [terminal](results/012_matrix_response/terminal.txt), [figure 1](results/012_matrix_response/figure_01.svg), [figure 2](results/012_matrix_response/figure_02.svg).

The supplied 6x6 complex matrix includes yaw-to-x coupling in m/rad and y-to-pitch coupling in rad/m. Support/magnet x RMS is 8.677 / 10.249 nm, and y RMS 46.742 / 56.272 nm. Their difference comes from the separate 1 m mount lever arm. Pitch is 9.533 nrad RMS at both ports in the common nominal axes.

Figure 1 separates input, responded support and transported magnet histories. Figure 2 is a local geometry/orientation view. Matrix coupling and coordinate transport are different operations: the response is a supplied linear model and transport is exact rigid geometry. The optional animation was rendered with 40 frames and updates geometry as well as orientation.

Instrumented costs: `setup_s` 0.2450 s; `matrix_response_and_transport_s` 0.0025 s.

## 013 — Combined prescribed-mechanics ring

[source](../013_combined_mechanics.py), [terminal](results/013_combined_mechanics/terminal.txt), [figure 1](results/013_combined_mechanics/figure_01.svg), [figure 2](results/013_combined_mechanics/figure_02.svg).

The capstone connects 168 supports, 168 magnet bodies, 14 pickups and 672 feet. Background and regional PSD fields plus a mounted pump are sampled jointly; fitted motion passes through supplied conditional causal responses, with port transport, independent installation errors and slow drift. The 1.5 s prehistory is explicit. The supplied response laws are assumed to represent the loaded ports; they are not inferred from equipment coordinates.

The contact mismatch after warm-up is 0.392 nm RMS. First-channel support/magnet/pickup y RMS is 15.289 / 20.493 / 18.675 nm over this short realization. Figure 1 shows the combined response histories. Figure 2 uses the installed placement as its baseline, so installation errors do not mask magnified dynamics. The 32 tracked turns span only 0.000160111 s: this is a scheduling smoke check. The waveform study spans 0.5 s. At 1024 Hz replay, a 40 Hz sinusoid has 0.752% worst half-step amplitude attenuation; this sampling error is separate from transform accuracy.

The process peak is about 808 MiB, including Python, figures and temporary batched fitting/transport arrays. Explicit prepared support/magnet/pickup payloads total about 32.8 MiB on the host; there is no full GPU history. This demonstrates the UI composition and exposes the need for bounded streaming before long whole-machine studies.

Instrumented costs: `layout_and_installation_s` 2.7533 s; `floor_and_rigid_fit_s` 0.6499 s; `response_paths_s` 0.8870 s; `initial_frozen_twiss_s` 0.0742 s; `dynamic_tracking_s` 0.7170 s.

## Alternative inputs and animation

These runs changed only the stated user parameters in memory; their result
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
