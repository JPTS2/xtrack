<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# Xtrack examples

Start with [getting started](getting_started/README.md) for a runnable tour through
construction, inspection, optics, matching and tracking. Each step points to
more detailed examples. Subject folders also provide independent starting points.

For installation, reference APIs and the physics guide, see the
[Xsuite documentation](https://xsuite.readthedocs.io/en/latest/index.html).

For detailed companion-package physics examples, see:

- [Xfields](https://github.com/xsuite/xfields/tree/main/examples): space charge
  and beam–beam interactions.
- [Xcoll](https://github.com/xsuite/xcoll/tree/main/examples): collimators,
  material interactions and collimation studies.
- [Xwakes](https://github.com/xsuite/xwakes/tree/main/examples): wakefields and
  impedance-driven beam dynamics.

## Folder overview

This is the working structure for the collection and the reference for its
implementation. It includes planned folders and filenames, rather than claiming
a completed migration. Existing examples may still have different names or
locations. File boundaries and physical configurations are reviewed as each
subject is implemented.

The organisation follows operations, with integrated RF, radiation and spin
workflows kept together. The overview shows the folder hierarchy; the sections
below give the proposed files. Each folder and subfolder has its own `README.md`.

| Folder | Scope or subfolders |
| --- | --- |
| [getting_started/](#getting_started) | Guided introductory tour |
| [tables/](#tables) | Access, selections, calculations and storage |
| [line_construction/](#line_construction) | Environments, assembly, expressions and editing |
| [elements/](#elements) | magnets/, cavities/, maps/, transformations/, custom/ |
| [geometry_and_survey/](#geometry_and_survey) | Reference geometry and element frames |
| [particles/](#particles) | Coordinates, distributions, identity and storage |
| [optics/](#optics) | transverse/, longitudinal/, nonlinear/, collective/ |
| [matching/](#matching) | Transverse matching and knobs; longitudinal/ |
| [tracking/](#tracking) | Propagation; time_dependent/, nonlinear/, collective/ |
| [monitors/](#monitors) | particle_records/, beam_statistics/, storage/ |
| [rf/](#rf) | dynamics/, timing/, acceleration/ |
| [radiation/](#radiation) | Losses, compensation, equilibrium and photons |
| [spin/](#spin) | Transport, tune, polarization and depolarization |
| [errors_and_correction/](#errors_and_correction) | Errors, responses and orbit/trajectory correction |
| [apertures/](#apertures) | Profiles, pipes, clearance, limits and losses |
| [contexts_and_memory/](#contexts_and_memory) | CPU/GPU contexts, arrays and buffers |
| [interfaces/](#interfaces) | madng/, rf_track/ |
| [machine_studies/](#machine_studies) | Integrated machine applications |
| [benchmarks/](#benchmarks) | Runtime, convergence and statistical studies |

## getting_started

A runnable tour through construction, inspection, open and periodic optics,
matching, tracking, measurements and model persistence. Each step links to more
detailed lessons; the subject folders remain independently usable.

- `01_build_line.py`
- `02_load_and_inspect.py`
- `03_open_optics.py`
- `04_periodic_optics.py`
- `05_match_tunes.py`
- `06_track_particles.py`
- `07_monitors_at_elements.py`
- `08_save_and_load.py`

## tables

See the [table examples](tables/README.md) for the runnable learning route.

Table access, selections, calculations and table formats. Saving and loading
models, particles, Twiss results, aperture models and measurements stays with
the corresponding subject.

- `001_create_and_access_tables.py`
- `002_select_and_calculate.py`
- `003_save_and_load_tables.py`
- `004_hdf5_and_tfs_tables.py`

## line_construction

Create environments and lines, place and share elements, manage expressions,
edit models and save native definitions. MAD-X reading is explicitly legacy data
input, without live MAD-X execution or teaching `cpymad` imports.

- `001_environment_and_elements.py`
- `002_assemble_line.py`
- `003_place_elements.py`
- `004_strength_expressions.py`
- `005_line_algebra.py`
- `006_place_lines.py`
- `007_parametric_geometry.py`
- `008_clones_and_replicas.py`
- `009_edit_line.py`
- `010_replace_elements.py`
- `011_shared_lines.py`
- `012_import_line.py`
- `013_reference_particles.py`
- `014_cut_at_positions.py`
- `015_save_and_load_environment.py`
- `016_read_legacy_madx_data.py`
- `017_build_demo_ring.py`

## elements

See the [element examples](elements/README.md) for the available learning routes.

Single-element behaviour, grouped by physical role. Apertures belong with
apertures; machine-level RF behaviour belongs with RF. Custom elements form an
advanced branch rather than an introduction to element usage.

- [electron_cooler.py](elements/electron_cooler.py): one-pass friction kicks,
  compared with zero current.

### magnets

- `001_fields_and_strengths.py`
- `002_models_and_integrators.py`
- `003_thin_and_thick_slices.py`
- `004_bend_geometry_and_edges.py`
- `005_fringe_fields_and_drifts.py`
- `006_solenoids.py`
- `007_spline_fields.py`
- `008_field_expansion.py`

### cavities

- `001_cavity_kick.py`
- `002_finite_length_cavity_transport.py`
- `003_cavity_models.py`
- `004_transverse_rf_kicks.py`

### maps

- `001_taylor_maps.py`
- `002_line_segment_map.py`

### transformations

- `001_translations_and_rotations.py`

### custom

- `001_custom_element.py`
- `002_random_kicks.py`
- `003_internal_records.py`
- `004_python_interaction_process.py`

## geometry_and_survey

Reference geometry, ring closure, selected survey origins and actual element
frames.

Follow the [geometry and survey lessons](geometry_and_survey/README.md).

- `001_survey_a_line.py`
- `002_ring_closure.py`
- `003_survey_from_a_selected_point.py`
- `004_element_frames.py`

## particles

Reference quantities, coordinate conventions, normalized launches, distributions,
particle identity and persistence. Matched RF preparation requires a qualified
longitudinal configuration.

Start with the [particle lessons](particles/README.md).

- `001_reference_quantities.py`
- `002_coordinates.py`
- `003_normalized_coordinates.py`
- `004_gaussian_transverse_bunch.py`
- `005_select_and_merge.py`
- `006_multispecies_kicks.py`
- `007_copy_particles.py`
- `008_save_and_load_particles.py`
- `009_distribution_generators.py`
- `010_matched_rf_bunch.py`
- `011_capacity_and_insertion.py`

## optics

Calculate and interpret optics. Transverse, longitudinal and nonlinear lessons
have separate routes. Coupled-mode and Edwards–Teng examples require actual
coupled optics; constructing coupling bumps and knobs belongs with matching.

### transverse

Begin with [transverse optics](optics/transverse/README.md).

- `001_periodic_twiss.py`
- `002_open_twiss.py`
- `003_inspect_twiss.py`
- `004_beam_covariance.py`
- `005_coupled_mode_optics.py`
- `006_edwards_teng.py`
- `007_twiss_initial_conditions.py`
- `008_periodic_cells.py`
- `009_change_starting_point.py`
- `010_reverse_reference_frame.py`
- `011_closed_orbit_search.py`
- `012_closed_orbit_dispersion.py`
- `013_transfer_matrices.py`
- `014_twiss_settings.py`
- `015_save_and_load_twiss.py`

### longitudinal

Follow the [longitudinal optics lessons](optics/longitudinal/README.md).

- `001_six_dimensional_twiss.py`
- `002_longitudinal_covariance.py`
- `003_revolution_period.py`

### nonlinear

Chromatic optics, from first-order derivatives to higher-order momentum
dependence, followed by amplitude-dependent and other nonlinear studies.

Begin with the [chromatic optics lessons](optics/nonlinear/README.md).

- `001_chromatic_optics.py`
- `002_higher_order_chromaticity.py`
- `003_amplitude_detuning.py`
- `004_multiturn_optics.py`
- `005_stability_diagram.py`

### collective

See [optics with collective elements](optics/collective/README.md).

- `001_optics_with_collective_elements.py`

## matching

Choose targets and controls, operate the optimizer, match bumps and multiple
lines, and generate knobs. Longitudinal matching has its own subfolder.

Start with the [matching lessons](matching/README.md).

- `001_match_tunes.py`
- `002_match_transport.py`
- `003_optimizer_control.py`
- `004_orbit_bump.py`
- `005_targets_from_twiss.py`
- `006_inequalities_and_callables.py`
- `007_match_multiple_lines.py`
- `008_custom_actions.py`
- `009_generate_knobs.py`
- `010_match_chromaticity.py`
- `011_coupling_bumps.py`
- `012_coupling_knobs.py`
- `013_correct_optics_errors.py`
- `014_match_demo_ring.py`

### longitudinal

See [longitudinal matching](matching/longitudinal/README.md).

- `001_match_synchrotron_tune.py`
- `002_match_phase_and_energy.py`

## tracking

Propagation controls, time-dependent and nonlinear studies, and Xtrack collective
tracking contracts. Nonlinear tracking and nonlinear optics link to
one another rather than duplicate the same recipe.

Start with the [tracking lessons](tracking/README.md).

- `001_track_line.py`
- `002_partial_tracking.py`
- `003_backtrack.py`
- `004_freeze_longitudinal.py`
- `005_optimize_for_tracking.py`

### time_dependent

See [time-dependent tracking](tracking/time_dependent/README.md).

- `001_time_dependent_knobs.py`
- `002_excitation.py`

### nonlinear

See [nonlinear tracking](tracking/nonlinear/README.md).

- `001_tune_footprint.py`
- `002_dynamic_aperture.py`

### collective

See [collective tracking](tracking/collective/README.md).

- `001_track_collective_elements.py`
- `002_pipeline_scheduling.py`

## monitors

See the [monitor examples](monitors/README.md) for the learning route.

Individual particle records, derived beam statistics and measurement storage.
The element-by-element lesson includes `ONE_TURN_EBE`.

### particle_records

See [particle records](monitors/particle_records/README.md).

- `001_turn_by_turn.py`
- `002_element_by_element.py`
- `003_observation_points.py`
- `004_valid_samples.py`
- `005_recording_windows.py`

### beam_statistics

See [beam statistics](monitors/beam_statistics/README.md).

- `001_whole_beam_statistics.py`
- `002_derived_statistics.py`
- `003_bunch_statistics.py`
- `004_slice_statistics.py`
- `005_coasting_statistics.py`
- `006_beam_profiles.py`

### storage

See [measurement storage](monitors/storage/README.md).

- `001_save_statistics_during_tracking.py`
- `002_reuse_recording_frames.py`

## rf

Start with [RF and longitudinal motion](rf/README.md).

Longitudinal motion, laboratory-time RF, off-momentum delays and acceleration.
Generic longitudinal Twiss and matching remain under optics and matching, with
cross-links. LINAC optics and tracking stay with their complete acceleration
workflow. The folder name `rf/` remains provisional.

### dynamics

See [longitudinal dynamics](rf/dynamics/README.md).

- `001_synchrotron_motion.py`
- `002_rf_bucket_and_separatrix.py`
- `003_coasting_arrival_times.py`
- `004_transition_crossing.py`
- `005_slip_stacking.py`

### timing

See [RF clocks and timing](rf/timing/README.md).

- `001_absolute_time_cavities.py`
- `002_time_delays_off_momentum.py`
- `003_rf_frequency_steering.py`
- `004_frequency_ramps_and_phase.py`
- `005_synchronized_time_frames.py`

### acceleration

See [acceleration and reference momentum](rf/acceleration/README.md).

- `001_ring_energy_ramp.py`
- `002_linac_energy_gain.py`
- `003_linac_optics_and_tracking.py`
- `004_mixed_species_reference_clock.py`

## radiation

See [synchrotron radiation](radiation/README.md).

Energy loss, compensation, tapering, damping, equilibrium and photon emission.
Keep model configuration, specialised optics and tracking interpretation together;
the operation indexes link here.

- `001_mean_energy_loss.py`
- `002_compensate_radiation_and_taper.py`
- `003_radiation_integrals.py`
- `004_mean_radiation_damping.py`
- `005_equilibrium_emittances.py`
- `006_quantum_radiation.py`
- `007_record_photons_and_spectrum.py`

## spin

Spin transport, closed solutions, tune, polarization and depolarization.
Keep the physical configuration and related optics/tracking calculations together.

- `001_spin_precession.py`
- `002_closed_spin_solution.py`
- `003_track_spin_in_a_ring.py`
- `004_spin_tune_vs_energy.py`
- `005_resonant_depolarization.py`
- `006_equilibrium_polarization.py`
- `007_radiative_depolarization.py`
- `008_invariant_spin_field.py`

## errors_and_correction

Alignment and multipole errors, closed-orbit and trajectory correction, response
selection and threading. Optics-matching procedures link to matching.

- `001_alignment_errors.py`
- `002_multipole_errors.py`
- `003_closed_orbit_correction.py`
- `004_trajectory_correction.py`
- `005_micado_and_selected_correctors.py`
- `006_custom_response.py`
- `007_threading.py`

## apertures

Chamber shapes, profiles and pipes, geometry, clearance, tracking limits and
loss-location refinement.

- `001_aperture_shapes.py`
- `002_profiles_and_pipes.py`
- `003_place_and_transform_pipes.py`
- `004_interpolate_profiles_on_curves.py`
- `005_install_tracking_limits.py`
- `006_beam_clearance.py`
- `007_save_and_reload_apertures.py`
- `008_refine_loss_locations.py`

## contexts_and_memory

Execution contexts, arrays, buffers, shared storage and data movement.

For an existing CPU-context and tracker-rebuild example, see
[execution context](tracking/004_execution_context.py).

- `001_cpu_context_and_rebuilds.py`
- `002_move_particles_between_contexts.py`
- `003_buffers_and_shared_storage.py`

## interfaces

Supported external-code connections. Xsuite companion-package integrations stay
with their subject; full solver applications are linked to their implementing
package. Optional external engines require their own API and version checks.

### madng

- `001_madng_optics.py`
- `002_resonance_driving_terms.py`

### rf_track

- `001_track_an_external_element.py`
- `002_rf_fieldmap_transport.py`

## machine_studies

Integrated applications whose machine geometry or operating conditions are
essential. Start with the
[demo-ring workflow](machine_studies/demo_ring_workflow.py), which carries a
chromaticity-corrected setting into off-momentum optics and finite tracking.
The three machine studies below are candidates; they do not account for all
existing machine-specific material.

- `elena_field_transport.py`
- `fcc_ee_solenoid_compensation.py`
- `lep_polarization.py`

## benchmarks

Runtime, convergence and statistical studies, excluded from ordinary example
execution.

- `field_tracking_convergence.py`
- `integrator_convergence.py`
- `optics_runtime.py`
- `radiation_statistics.py`
- `tracking_backends.py`
