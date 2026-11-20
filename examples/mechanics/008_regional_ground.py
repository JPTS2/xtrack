# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

# Add a local excess ground spectrum to one shared background over the demo ring.
# Use continuous random modes so receiver order and time chunks keep one process.

################################################################################
# Required packages
################################################################################

from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch
from scipy.integrate import trapezoid
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Demonstration lattice
########################################
LATTICE_FILE = 'test_data/example_lattices/demo_ring/lattice.py'

########################################
# Background and regional excess spectra
########################################
BACKGROUND_DENSITY = 1e-18            # [m²/Hz], one-sided
EXCESS_DENSITY     = 1e-17            # Additional PSD at the region center

########################################
# Spatial correlation lengths and region envelope
########################################
BACKGROUND_LENGTH_M = 100.
EXCESS_LENGTH_M     = 30.
REGION_WIDTH_M      = 80.             # Gaussian amplitude width, not a hard boundary

########################################
# Independent latent-source seeds
########################################
BACKGROUND_SEED = 956
REGION_SEED     = 957

########################################
# Random-mode approximation
########################################
N_MODES = 512                         # Approximation rank; increase and compare

########################################
# Record sampling
########################################
N_SAMPLES     = 2048
SAMPLE_STEP_S = 1 / 128

########################################
# Inspection and display
########################################
INSPECT_TIME_S     = .5
DISPLACEMENT_SCALE = 1e8

################################################################################
# Set up the ordinary Xtrack line
################################################################################

########################################
# Start performance monitoring
########################################
timings = {}

########################################
# Time line load and copy
########################################
started = perf_counter()

########################################
# Locate and load the demonstration ring
########################################
repo_root = Path(__file__).resolve().parents[2]
nominal_env = xt.load(repo_root / LATTICE_FILE)
line = nominal_env['demo'].copy()
env = line.env
env['demo'] = line

timings['line_load_and_copy_s'] = perf_counter() - started

########################################
# Prepare and time the tracker before mechanics queries
########################################
started = perf_counter()
line.build_tracker()
timings['tracker_build_s'] = perf_counter() - started

################################################################################
# Assign supports and choose the regional center
################################################################################

########################################
# Access the environment mechanics model
########################################
mech = env.mechanics

########################################
# Time reference adoption
########################################
started = perf_counter()

########################################
# Adopt existing offsets as the deliberate reference
########################################
line.mechanics.adopt_reference()

timings['reference_adoption_s'] = perf_counter() - started

########################################
# Time geometry selection
########################################
started = perf_counter()

########################################
# Assign a support to each surveyed quadrupole
########################################
quads = list(line.get_table().rows.match(element_type = 'Quadrupole').name)

timings['geometry_selection_s'] = perf_counter() - started
########################################
# Time mechanics definition
########################################
started = perf_counter()

girders = [line.mechanics.new_girder(f'g.{name}', elements = [name]) for name in quads]

########################################
# Collect world receivers and define the region envelope
########################################
positions = np.array([g.nominal_frame.XYZ for g in girders])
center = positions[0]

timings['mechanics_definition_s'] = perf_counter() - started
########################################
# Time source geometry preparation
########################################
started = perf_counter()

region_gain = np.exp(-np.sum((positions - center)**2, axis = 1) / (2 * REGION_WIDTH_M**2))

timings['source_geometry_s'] = perf_counter() - started

################################################################################
# Define the joint regional ground field
################################################################################

########################################
# Time source definition
########################################
started = perf_counter()

########################################
# Construct the shared background process
########################################
background = xt.mechanics.SpectralGroundMotion(
    frequency_hz = [0., 1., 2., 30., 40.],
    density = [0., 0., BACKGROUND_DENSITY, BACKGROUND_DENSITY, 0.],
    seed = BACKGROUND_SEED, correlation_length_m = BACKGROUND_LENGTH_M, n_modes = N_MODES,
)

########################################
# Construct an independent excess process
########################################
excess = xt.mechanics.SpectralGroundMotion(
    frequency_hz = [0., 1., 2., 30., 40.],
    density = [0., 0., EXCESS_DENSITY, EXCESS_DENSITY, 0.],
    seed = REGION_SEED, correlation_length_m = EXCESS_LENGTH_M, n_modes = N_MODES,
)

########################################
# Combine the background and localized vertical excess
########################################
vertical = xt.mechanics.RegionalGroundMotion(
    background = background,
    regions = (xt.mechanics.GaussianRegion(excess, tuple(center), REGION_WIDTH_M),),
)

########################################
# Specify the independent horizontal component
########################################
horizontal = xt.mechanics.SpectralGroundMotion(
    frequency_hz = [0., 1., 2., 30., 40.],
    density = [0., 0., .1 * BACKGROUND_DENSITY, .1 * BACKGROUND_DENSITY, 0.],
    seed = REGION_SEED + 1, correlation_length_m = BACKGROUND_LENGTH_M, n_modes = N_MODES,
)

########################################
# Assemble the world-vector ground field
########################################
source = xt.mechanics.VectorGroundMotion(components = (horizontal, vertical, None))

timings['source_definition_s'] = perf_counter() - started

################################################################################
# Sample the same field over the ring
################################################################################

########################################
# Choose the physical sample grid
########################################
times = np.arange(N_SAMPLES) * SAMPLE_STEP_S

########################################
# Start timing field sampling
########################################
started = perf_counter()

########################################
# Sample every receiver jointly
########################################
floor = xt.mechanics.sample_field(source, time_s = times, positions_m = positions)

########################################
# Record field sampling cost
########################################
timings['sample_field_s'] = perf_counter() - started

################################################################################
# Check reproducibility under chunking and selection
################################################################################

########################################
# Start timing the reproducibility checks
########################################
started = perf_counter()

########################################
# Choose a short time window and a receiver subset
########################################
probe_times = times[:128]
probe_positions = positions[[0, 30, -1]]

########################################
# Evaluate whole, split and reordered queries
########################################
whole = source.evaluate(probe_times, probe_positions)
chunks = np.concatenate((source.evaluate(probe_times[:51], probe_positions),
                         source.evaluate(probe_times[51:], probe_positions)))
reordered = source.evaluate(probe_times, probe_positions[[2, 0]])

########################################
# Check that query shape does not change the process
########################################
chunk_error = np.max(np.abs(whole - chunks))
order_error = np.max(np.abs(whole[:, [2, 0]] - reordered))
assert max(chunk_error, order_error) < 1e-20

########################################
# Record the check cost
########################################
timings['chunk_and_order_check_s'] = perf_counter() - started

################################################################################
# Bind and assess the sampled motion
################################################################################

########################################
# Project world motion into each support's nominal axes
########################################
motion = floor.to_rigid_motion(
    frames = [g.nominal_frame for g in girders], target_names = tuple(g.name for g in girders),
)

########################################
# Bind the sampled site motion to the supports
########################################
mech.add_source('regional_site', motion, targets = motion.target_names)

########################################
# Select representative receivers and compare observed RMS
########################################
near, far = 0, int(np.argmin(region_gain))
print('Background and regional excess PSD\n-----------------------------------')
print(f'Supports: {len(girders)}; modes per independent field: {N_MODES}')
print('Receiver       Region amplitude gain    Observed vertical RMS [nm]')

for label, index in [('Near region', near), ('Far from region', far)]:
    print(f'{label:<18} {region_gain[index]:16.6f} {np.std(floor.values[:, index, 1]) * 1e9:29.6f}')
print(f'Chunk / receiver-order discrepancies [m]: {chunk_error:.3g} / {order_error:.3g}')
print(f'World field payload [MiB]: {floor.nbytes / 1024**2:.3f}')
print('The regional contribution is gain² × excess PSD, plus the shared background.')
print('An independent world-X field is also sampled and projected into each support frame.')
print('Finite random modes approximate spectra/coherence; this is not an exact dense covariance.')

########################################
# Compute the ensemble target for amplitude correlation
########################################
distance = np.linalg.norm(positions - positions[near], axis = 1)
variance_background = trapezoid(background.density, background.frequency_hz)
variance_excess = trapezoid(excess.density, excess.frequency_hz)
variance = variance_background + region_gain**2 * variance_excess
expected_correlation = (
    variance_background * np.exp(-distance / BACKGROUND_LENGTH_M)
    + region_gain[near] * region_gain * variance_excess * np.exp(-distance / EXCESS_LENGTH_M)
) / np.sqrt(variance[near] * variance)

########################################
# Compare the finite realization with that target
########################################
observed_correlation = np.corrcoef(floor.values[..., 1].T)[near]
correlation_error = np.sqrt(np.mean((observed_correlation - expected_correlation)**2))
print(f'Finite-record amplitude-correlation deviation, RMS: {correlation_error:.6f}')

################################################################################
# Inspect spatial RMS, spectra and correlation
################################################################################

########################################
# Prepare the three diagnostic panels
########################################
fig, axes = plt.subplots(1, 3, figsize = (13, 4))

########################################
# Plot motion strength along the ring
########################################
path_s = line.get_table()['s', quads]
axes[0].plot(path_s, np.std(floor.values[..., 1], axis = 0) * 1e9, '.-', label = 'Observed RMS')
axes[0].set(xlabel = 'Nominal s [m]', ylabel = 'Ground y RMS [nm]', title = 'One continuous site field')

########################################
# Compare measured spectra with the supplied PSDs
########################################
for label, index in [('Near', near), ('Far', far)]:
    frequency, density = welch(floor.values[:, index, 1], fs = 1 / SAMPLE_STEP_S, nperseg = 512)
    line_plot, = axes[1].loglog(frequency[1:], density[1:], label = f'{label}: finite record')
    expected = np.interp(frequency, background.frequency_hz, background.density)
    expected += region_gain[index]**2 * np.interp(frequency, excess.frequency_hz, excess.density)
    axes[1].loglog(frequency[1:], expected[1:], '--', color = line_plot.get_color(), label = f'{label}: target PSD')
axes[1].set(xlabel = 'Frequency [Hz]', ylabel = 'PSD [m²/Hz]', xlim = (.2, 50), ylim = (1e-22, 1e-16))

########################################
# Compare spatial correlation against Euclidean distance
########################################
order = np.argsort(distance)
axes[2].plot(distance[order], expected_correlation[order], color = '0.4', label = 'Ensemble target')
axes[2].plot(distance, observed_correlation, '.', label = 'Finite modes + record')
axes[2].set(xlabel = 'Euclidean distance from first support [m]',
            ylabel = 'Band-integrated amplitude correlation', ylim = (-.3, 1.05))

########################################
# Finish the diagnostic figure
########################################
axes[0].legend()
axes[1].legend(fontsize = 8)
axes[2].legend(fontsize = 8)
fig.tight_layout()

########################################
# Draw the magnified whole-ring motion
########################################
layout_fig, layout_axes = mech.plot(
    time_s = INSPECT_TIME_S, line = line, displacement_scale = DISPLACEMENT_SCALE,
)

################################################################################
# Review execution costs and figures
################################################################################

########################################
# Print the measured costs
########################################
print('\nReference CPU costs\n-------------------')
for name, seconds in timings.items():
    print(f'{name:<28} {seconds:9.4f} s')

########################################
# Display the figures
########################################
plt.show()
