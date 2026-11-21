<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# Finding examples after the reorganization

Start with the [examples index](README.md) for the current reading routes.
The tables below connect the earlier subject folders to those routes. A lesson
may have been combined with another example, split into smaller lessons, or
rewritten on a simpler native model. Most original scripts and inputs remain
during the transition while their tests, documentation and data consumers are
reviewed. Fifteen original files are already absent from this branch; they are
identified below and remain part of the file-by-file review.

## Construction, particles and individual elements

| Earlier location | Current teaching route |
| --- | --- |
| `toy_ring/` | [Getting started](getting_started/README.md), then [building the demonstration ring](line_construction/017_build_demo_ring.py) |
| `lattice_design/`, `sequence/`, `insert_elements/`, `knobs/` | [Line construction](line_construction/README.md): environments, expressions, placement, editing, sharing and imports |
| `native_madloader/`, `mad_loader/`, `lhc_native_loader_and_py_lattice/`, `lhc_with_env/`, `lattice_and_optics_files/`, `to_json/` | [Line construction](line_construction/README.md): native model files, stored legacy input and persistence |
| `table_custom/`, `twiss_table_file/` | [Tables](tables/README.md); [Twiss result inspection and persistence](optics/transverse/README.md) |
| `transition_crossing/001_check_energy_variables.py` | [Particle coordinates](particles/README.md) |
| `ion_fragments/` | The unfinished ion-fragment setup has no executed fragmentation lesson; [particles](particles/README.md) covers species normalization without a fragmentation model |
| `magnet_models_and_integrators/`, `bend_api/`, `slicing/`, `magnet/`, `true_rbend/` | [Magnets](elements/magnets/README.md); [bend frames](geometry_and_survey/006_rectangular_bend_frames.py) |
| `solenoid/`, `splineboris/`, `bfieldexpansion/`, `boris_spatial/`, `dipole_fringe/`, `nonlinear_lens/` | [Magnetic elements and field profiles](elements/magnets/README.md); specialized field fits and validation remain separate |
| `transfer_matrix/`, `taylor_map/` | [Maps](elements/maps/README.md): assigned optics, Taylor maps and RF-map arrays |
| `element_transformations/` | [Spatial transformations](elements/transformations/README.md), plus [magnet axes and slicing](elements/magnets/README.md) |
| `element_internal_record/`, custom-element lessons in `lattice_design/` and `collimation/` | [Custom elements](elements/custom/README.md) |
| `electron_cooler/` | [Single-element cooling kick](elements/electron_cooler.py) |

## Optics, matching and tracking

| Earlier location | Current teaching route |
| --- | --- |
| `twiss/`, `coupling_edwards_teng/` | [Transverse optics](optics/transverse/README.md): periodic/open optics, closed orbit, dispersion, coupled modes and matrices |
| 6D Twiss in `twiss/`, `crab_dispersion/` | [Longitudinal optics](optics/longitudinal/README.md) |
| `amplitude_detuning/`, `nonlinear_tunes/`, `rdt/`, chromatic studies in `twiss/` | [Nonlinear optics](optics/nonlinear/README.md); [matching](matching/README.md) for controls and targets |
| `footprint/` | [Tracking footprints](tracking/nonlinear/README.md) and [nonlinear optics](optics/nonlinear/README.md) for stability diagrams |
| `match/`, `coupling_knobs/`, `optics_correction/`, `match_octupole_phase_knob/`, `optimize_function/` | [Matching](matching/README.md) |
| `tracker/`, `full_ring/`, `track_flags/`, `freeze_longitudinal/` | [Tracking](tracking/README.md) |
| Clock-driven lessons in `toy_ring/`, `multisetter/` | [Time-dependent tracking](tracking/time_dependent/README.md) |
| `dynamic_aperture/` | [Nonlinear tracking](tracking/nonlinear/README.md) |
| `pipeline/`, `spacecharge/` | [Collective tracking boundaries](tracking/collective/README.md); full source-distribution models belong with Xfields |
| `monitor/`, `monitor_multi_element/` | [Particle records](monitors/particle_records/README.md) |
| `beam_stats_monitor/`, `coasting/` | [Beam statistics](monitors/beam_statistics/README.md) and [record storage](monitors/storage/README.md) |
| `misalignments/`, `relative_errors/`, `multipole_errors_relative/`, `orbit_and_tracjectory_correction/` | [Errors and correction](errors_and_correction/README.md) |
| `survey/`, `survey_ccs/` | [Geometry and survey](geometry_and_survey/README.md); machine survey conversion and external comparisons retain their own inputs |
| `apertures/`, `aperture_measurement/`, generic lessons in `aperture_model/` and `collimation/` | [Apertures](apertures/README.md): profiles, SVG input, tracking limits, losses and refinement |

## RF, radiation, spin and external codes

| Earlier location | Current teaching route |
| --- | --- |
| `rfbucket/`, `transition_crossing/`, `slipstacking/` | [RF dynamics](rf/dynamics/README.md) |
| `cavity_absolute_time/`, `radial_steering/` | [RF timing](rf/timing/README.md); [longitudinal matching](matching/longitudinal/README.md) |
| `acceleration/` | [Acceleration](rf/acceleration/README.md); the paired PIMMS producer/plotter is under [PIMMS](machine_studies/pimms/README.md) |
| `radiation/`, `tapering/`, `radiation_integrals/`, radiation lessons in `fcc_ee/` | [Radiation](radiation/README.md); numerical/statistical studies have [benchmark routes](benchmarks/README.md) |
| `spin/`, `spin_lep/` | [Spin](spin/README.md) for individual methods and [LEP](machine_studies/lep/README.md) for a connected machine workflow |
| `madng_interface/`, supported export lessons in `mad_writer/` | [MAD-NG interface](interfaces/madng/README.md) |
| `rf_track/` | [RF-Track interface](interfaces/rf_track/README.md) |
| `pyheadtail_interface/`, `lhc_against_sixtrack/` | Retained compatibility or cross-code validation; these do not form a new recommended interface route |

## Connected machine studies and benchmarks

The [machine-studies index](machine_studies/README.md) separates connected
FCC-ee, LHC, ELENA, PIMMS, PSB and LEP routes. A new machine lesson does not
replace every earlier input model, measurement, field fit or validation study.
In particular, the FCC detector insertion, historical FCC-Z solenoid scans,
and FCC-ee error/correction workflow use different models and operating points.

| Earlier location | Current teaching route |
| --- | --- |
| `match_optics/` | [LHC two-beam workflows](machine_studies/lhc/README.md); exact older correction studies remain separately reviewed |
| `fcc_ee_solenoid/` | [FCC-ee detector insertion](machine_studies/fcc_ee/README.md); historical analytic scans and their result consumers remain distinct |
| ELENA field studies in `bfieldexpansion/` | [ELENA](machine_studies/elena/README.md) |
| `acceleration/004a_pimm_carbon_protons.py` and `004b_plot_pimm_carbon_protons.py` | [PIMMS acceleration and pickups](machine_studies/pimms/README.md) |
| `psb/`, machine-import pipelines in `aperture_model/` | [Machine studies](machine_studies/README.md) for curated workflows; retained model builders and external references need their own disposition |
| `tracking_time/`, `time_tracking/`, `optimized_tracker/`, `tpsa/` | [Performance](benchmarks/performance/README.md) |
| `bend_convergence/`, `magnet_vs_numerical_integrator/`, numerical field/map studies | [Convergence](benchmarks/convergence/README.md) |
| Statistical radiation/RNG studies | [Statistics](benchmarks/statistics/README.md) |

Benchmarks are separate from ordinary example execution. Their workload,
reference calculation and timing boundaries matter as much as their filenames.

## Selected file-level mappings

These entries identify the lesson's new home; the accompanying guides state
where its model or scope differs from the original.

| Earlier file | Current example(s) |
| --- | --- |
| `twiss/000_twiss.py` | [Periodic transverse optics](optics/transverse/001_periodic_twiss.py), [6D optics](optics/longitudinal/001_six_dimensional_twiss.py) |
| `twiss/000a_twiss_range.py` | [Open Twiss](optics/transverse/002_open_twiss.py), [initial conditions](optics/transverse/003_twiss_initial_conditions.py) |
| `match/000_match_basic.py` | [Tune matching](matching/001_match_tunes.py), [chromaticity matching](matching/010_match_chromaticity.py) |
| `monitor/000_example_quick_monitor.py` | [Turn-by-turn records](monitors/particle_records/001_turn_by_turn.py), [recording windows](monitors/particle_records/005_recording_windows.py) |
| `apertures/001_rect_aperture_withinline.py` | [Install tracking limits](apertures/005_install_tracking_limits.py) |
| `radiation/007_record_photons.py` | [Photon records and spectrum](radiation/007_record_photons_and_spectrum.py), with an explicit thin-kick experiment |
| `spin_lep/004_spin_tune_vs_energy.py` | [Spin tune versus energy](spin/006_spin_tune_vs_energy.py) |

## Before retiring an original file

The current branch already removes these source groups:

| Earlier sources already absent | Current ownership to review |
| --- | --- |
| `insert_elements/000_multiple_thick_insertions.py`; `lattice_design/{011_remove_element_env.py,012_replace_element.py,017_import_env_with_slices.py,018d_set_particle_ref.py}` | Line editing, imported sliced models and named reference particles in [line construction](line_construction/README.md), plus the corresponding source/test checks |
| `monitor/{000_example_quick_monitor.py,001_example_custom_monitor.py,002_example_custom_monitor_multiframe.py,003_monitors_as_beam_elements.py,004_monitor_standalone.py}`; `monitor_multi_element/000_multi_element_monitor.py` | [Particle records](monitors/particle_records/README.md), with documentation-extract migration still requiring review |
| `tracker/{000_track.py,001_tracker_start_stop.py,002_backtrack.py,003_preserve_context.py}` | [Tracking](tracking/README.md) and [contexts and memory](contexts_and_memory/README.md) |

The file-by-file disposition review is still in progress. Its categories are:
**moved** for substantially the same lesson, **merged** for lessons combined or
reworked, **removed** for a proposed omission with a reason, and **unresolved**
when the teaching, validation or consumer decision remains open.

For each retirement, retain or migrate its unique regression cases and its
inputs, helpers and generated-data consumers together. Documentation extracts
also need their new source paths and surrounding explanations checked. Passing
a generic replacement is insufficient evidence for an exact machine or
cross-code configuration. Private APIs, deprecated compatibility layers and
unsupported formats need an explicit decision rather than an implied new home.

The detailed ledger covers 725 original paths, including one input subsequently
removed upstream. It must be reconciled with the final diff and supplied with
the PR as small subject-by-subject file mappings before further retirements.
The tables here provide the navigation overview while that review is completed.
