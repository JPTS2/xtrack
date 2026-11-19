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
# Site model and independent excess region
########################################
LATTICE_FILE = 'test_data/example_lattices/demo_ring/lattice.py'
BACKGROUND_DENSITY = 1e-18              # [m²/Hz], one-sided
EXCESS_DENSITY     = 1e-17              # Additional PSD at the region center
BACKGROUND_LENGTH_M = 100.
EXCESS_LENGTH_M     = 30.
REGION_WIDTH_M      = 80.               # Gaussian amplitude width, not a hard boundary
BACKGROUND_SEED     = 956
REGION_SEED         = 957
N_MODES             = 512              # Approximation rank; increase and compare

########################################
# Sampling and inspection
########################################
N_SAMPLES     = 2048
SAMPLE_STEP_S = 1 / 128
INSPECT_TIME_S = .5
DISPLACEMENT_SCALE = 1e8

################################################################################
# Sample a shared site field at surveyed ring supports
################################################################################
started = perf_counter()
repo_root = Path(__file__).resolve().parents[2]
nominal_env = xt.load(repo_root / LATTICE_FILE)
line = nominal_env['demo'].copy()
env = line.env
env['demo'] = line
mech = env.mechanics
line.mechanics.adopt_reference()
quads = list(line.get_table().rows.match(element_type = 'Quadrupole').name)
girders = [line.mechanics.new_girder(f'g.{name}', elements = [name]) for name in quads]
positions = np.array([g.nominal_frame.XYZ for g in girders])
center = positions[0]
region_gain = np.exp(-np.sum((positions - center)**2, axis = 1) / (2 * REGION_WIDTH_M**2))
background = xt.mechanics.SpectralGroundMotion(
    frequency_hz = [0., 1., 2., 30., 40.],
    density = [0., 0., BACKGROUND_DENSITY, BACKGROUND_DENSITY, 0.],
    seed = BACKGROUND_SEED, correlation_length_m = BACKGROUND_LENGTH_M, n_modes = N_MODES,
)
excess = xt.mechanics.SpectralGroundMotion(
    frequency_hz = [0., 1., 2., 30., 40.],
    density = [0., 0., EXCESS_DENSITY, EXCESS_DENSITY, 0.],
    seed = REGION_SEED, correlation_length_m = EXCESS_LENGTH_M, n_modes = N_MODES,
)
vertical = xt.mechanics.RegionalGroundMotion(
    background = background,
    regions = (xt.mechanics.GaussianRegion(excess, tuple(center), REGION_WIDTH_M),),
)
horizontal = xt.mechanics.SpectralGroundMotion(
    frequency_hz = [0., 1., 2., 30., 40.],
    density = [0., 0., .1 * BACKGROUND_DENSITY, .1 * BACKGROUND_DENSITY, 0.],
    seed = REGION_SEED + 1, correlation_length_m = BACKGROUND_LENGTH_M, n_modes = N_MODES,
)
source = xt.mechanics.VectorGroundMotion(components = (horizontal, vertical, None))
timings = {'setup_s': perf_counter() - started}
times = np.arange(N_SAMPLES) * SAMPLE_STEP_S
started = perf_counter()
floor = xt.mechanics.sample_field(source, time_s = times, positions_m = positions)
timings['sample_field_s'] = perf_counter() - started

################################################################################
# Check that chunking and receiver selection do not change the field
################################################################################
started = perf_counter()
probe_times = times[:128]
probe_positions = positions[[0, 30, -1]]
whole = source.evaluate(probe_times, probe_positions)
chunks = np.concatenate((source.evaluate(probe_times[:51], probe_positions),
                         source.evaluate(probe_times[51:], probe_positions)))
reordered = source.evaluate(probe_times, probe_positions[[2, 0]])
chunk_error = np.max(np.abs(whole - chunks))
order_error = np.max(np.abs(whole[:, [2, 0]] - reordered))
assert max(chunk_error, order_error) < 1e-20
timings['chunk_and_order_check_s'] = perf_counter() - started

########################################
# Bind the sampled input to rigid supports
########################################
motion = floor.to_rigid_motion(
    frames = [g.nominal_frame for g in girders], target_names = tuple(g.name for g in girders),
)
mech.add_source('regional_site', motion, targets = motion.target_names)
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
distance = np.linalg.norm(positions - positions[near], axis = 1)
variance_background = trapezoid(background.density, background.frequency_hz)
variance_excess = trapezoid(excess.density, excess.frequency_hz)
variance = variance_background + region_gain**2 * variance_excess
expected_correlation = (
    variance_background * np.exp(-distance / BACKGROUND_LENGTH_M)
    + region_gain[near] * region_gain * variance_excess * np.exp(-distance / EXCESS_LENGTH_M)
) / np.sqrt(variance[near] * variance)
observed_correlation = np.corrcoef(floor.values[..., 1].T)[near]
correlation_error = np.sqrt(np.mean((observed_correlation - expected_correlation)**2))
print(f'Finite-record amplitude-correlation deviation, RMS: {correlation_error:.6f}')

################################################################################
# Compare local spectra and inspect the spatial envelope
################################################################################
fig, axes = plt.subplots(1, 3, figsize = (13, 4))
path_s = line.get_table()['s', quads]
axes[0].plot(path_s, np.std(floor.values[..., 1], axis = 0) * 1e9, '.-', label = 'Observed RMS')
axes[0].set(xlabel = 'Nominal s [m]', ylabel = 'Ground y RMS [nm]', title = 'One continuous site field')
for label, index in [('Near', near), ('Far', far)]:
    frequency, density = welch(floor.values[:, index, 1], fs = 1 / SAMPLE_STEP_S, nperseg = 512)
    line_plot, = axes[1].loglog(frequency[1:], density[1:], label = f'{label}: finite record')
    expected = np.interp(frequency, background.frequency_hz, background.density)
    expected += region_gain[index]**2 * np.interp(frequency, excess.frequency_hz, excess.density)
    axes[1].loglog(frequency[1:], expected[1:], '--', color = line_plot.get_color(), label = f'{label}: target PSD')
axes[1].set(xlabel = 'Frequency [Hz]', ylabel = 'PSD [m²/Hz]', xlim = (.2, 50), ylim = (1e-22, 1e-16))
order = np.argsort(distance)
axes[2].plot(distance[order], expected_correlation[order], color = '0.4', label = 'Ensemble target')
axes[2].plot(distance, observed_correlation, '.', label = 'Finite modes + record')
axes[2].set(xlabel = 'Euclidean distance from first support [m]',
            ylabel = 'Band-integrated amplitude correlation', ylim = (-.3, 1.05))
axes[0].legend()
axes[1].legend(fontsize = 8)
axes[2].legend(fontsize = 8)
fig.tight_layout()
layout_fig, layout_axes = mech.plot(
    time_s = INSPECT_TIME_S, line = line, displacement_scale = DISPLACEMENT_SCALE,
)
print('\nReference CPU costs\n-------------------')
for name, seconds in timings.items():
    print(f'{name:<28} {seconds:9.4f} s')
plt.show()
