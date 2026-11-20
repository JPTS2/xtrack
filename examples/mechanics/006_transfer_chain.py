# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

# Follow one floor record through a loaded girder, magnet and mounted BPM.
# Distinguish absolute response from inherited motion and causal startup.

################################################################################
# Required packages
################################################################################

from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Line, quadrupole and pickup selection
########################################
LATTICE_FILE = 'test_data/example_lattices/demo_ring/lattice.py'
QUADRUPOLE   = 'arc.s1.c1.qf.in'
BPM          = 'bpm.s1.f'

########################################
# Loaded girder and magnet assembly
########################################
GIRDER_MASS_KG = 500.
MAGNET_MASS_KG = 150.
GIRDER_HZ      = 12.                  # Unloaded stage parameter, not coupled mode
MAGNET_HZ      = 35.                  # Fixed-base stage parameter
GIRDER_DAMPING = .15
MAGNET_DAMPING = .15

########################################
# Pickup response with negligible reaction loading
########################################
BPM_HZ      = 80.                     # Negligible pickup loading in this example
BPM_DAMPING = .2

########################################
# Stationary floor record
########################################
N_SAMPLES     = 16384
SAMPLE_STEP_S = 1 / 256
PSD_SEED      = 956
FLOOR_DENSITY = 1e-18                 # One-sided displacement PSD [m²/Hz]

########################################
# Frozen inspection
########################################
INSPECT_TIME_S = .125

########################################
# Causal held-step sampling
########################################
CAUSAL_SAMPLE_STEP_S = 1 / 4096       # ZOH response to a held step, initially at rest

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
# Build the mechanical mounting hierarchy
################################################################################

########################################
# Access the shared mechanics model
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
# Survey the nominal attachment frames
########################################
nominal = line.survey(include_element_frames = True)

timings['geometry_selection_s'] = perf_counter() - started

########################################
# Time mechanics definition
########################################
started = perf_counter()

########################################
# Place the girder at the selected quadrupole
########################################
girder = mech.new_girder('girder', anchor = nominal.get_frame(QUADRUPOLE, which = 'elem_start'))

########################################
# Mount the quadrupole on the girder
########################################
magnet = mech.new_body('magnet', kind = 'magnet', parent = girder, mount = xt.Frame(),
                       elements = [QUADRUPOLE], line = line)

########################################
# Mount the pickup on the quadrupole body
########################################
bpm = mech.new_body('pickup', kind = 'pickup', parent = magnet,
                    anchor = nominal.get_frame(BPM, which = 'elem_start'), elements = [BPM], line = line)

timings['mechanics_definition_s'] = perf_counter() - started

################################################################################
# Define consistent loaded response paths
################################################################################

########################################
# Time the response-model definitions
########################################
started = perf_counter()

########################################
# Convert stage frequencies into assembly parameters
########################################
omega_g = 2 * np.pi * GIRDER_HZ
omega_m = 2 * np.pi * MAGNET_HZ
parameters = dict(
    girder_mass_kg = GIRDER_MASS_KG, magnet_mass_kg = MAGNET_MASS_KG,
    floor_stiffness_n_m = GIRDER_MASS_KG * omega_g**2,
    magnet_stiffness_n_m = MAGNET_MASS_KG * omega_m**2,
    floor_damping_ns_m = 2 * GIRDER_DAMPING * GIRDER_MASS_KG * omega_g,
    magnet_damping_ns_m = 2 * MAGNET_DAMPING * MAGNET_MASS_KG * omega_m,
)

########################################
# Construct the loaded and conditional two-mass responses
########################################
floor_to_girder = xt.mechanics.LoadedTwoMass(**parameters, stage = 'floor_to_girder')
girder_to_magnet = xt.mechanics.LoadedTwoMass(**parameters, stage = 'girder_to_magnet')
floor_to_magnet = xt.mechanics.LoadedTwoMass(**parameters, stage = 'floor_to_magnet')

########################################
# Specify the pickup response and unloaded comparison
########################################
magnet_to_bpm = xt.mechanics.BaseExcitedOscillator(BPM_HZ, BPM_DAMPING)
unloaded_girder = xt.mechanics.BaseExcitedOscillator(GIRDER_HZ, GIRDER_DAMPING)
timings['response_definition_s'] = perf_counter() - started

################################################################################
# Propagate one common floor realization
################################################################################

########################################
# Start timing source and response preparation
########################################
started = perf_counter()

########################################
# Define the seeded floor displacement source
########################################
times = np.arange(N_SAMPLES) * SAMPLE_STEP_S
source = xt.mechanics.DisplacementPSD(
    frequency_hz = [0., 1., 2., 80., 90.],
    density = [0., 0., FLOOR_DENSITY, FLOOR_DENSITY, 0.], seed = PSD_SEED,
)

########################################
# Express the floor drive at a six-coordinate input port
########################################
floor_values = np.zeros((N_SAMPLES, 1, 6))
floor_values[:, 0, 1] = source.evaluate(times, np.zeros((1, 3)))[:, 0]
floor = xt.mechanics.RigidMotion(times, floor_values, ('girder',))

########################################
# Apply the loaded floor-to-girder response
########################################
girder_motion = floor.through(floor_to_girder)

########################################
# Transport to the magnet port and apply its conditional response
########################################
magnet_motion = girder_motion.transport(
    from_frames = [girder.nominal_frame], to_frames = [magnet.nominal_frame], target_names = ('magnet',),
).through(girder_to_magnet)

########################################
# Transport to the pickup port and apply its response
########################################
bpm_motion = magnet_motion.transport(
    from_frames = [magnet.nominal_frame], to_frames = [bpm.nominal_frame], target_names = ('pickup',),
).through(magnet_to_bpm)

########################################
# Check the loaded cascade against the direct assembly solution
########################################
direct = floor.through(floor_to_magnet)
naive = floor.through(unloaded_girder).through(girder_to_magnet)
np.testing.assert_allclose(magnet_motion.values, direct.values, rtol = 2e-12, atol = 1e-21)

########################################
# Bind each output as absolute motion
########################################
for name, motion in [('girder_response', girder_motion), ('magnet_response', magnet_motion), ('bpm_response', bpm_motion)]:
    mech.add_source(name, motion, targets = motion.target_names, reference = 'absolute')

########################################
# Record the input and response preparation cost
########################################
timings['sample_and_response_s'] = perf_counter() - started

################################################################################
# Verify absolute response without double counting
################################################################################

########################################
# Start timing pose application
########################################
started = perf_counter()

########################################
# Apply the resolved response state
########################################
mech.apply(time_s = INSPECT_TIME_S)
timings['apply_s'] = perf_counter() - started

########################################
# Time the read-only pose inspection separately
########################################
started = perf_counter()
survey = mech.survey(time_s = INSPECT_TIME_S)

########################################
# Compare each body with its prescribed absolute output
########################################
expected = [np.interp(INSPECT_TIME_S, times, m.values[:, 0, 1])
            for m in (girder_motion, magnet_motion, bpm_motion)]
np.testing.assert_allclose(survey.dynamic_Y, expected, rtol = 0, atol = 1e-12)

########################################
# Record the pose-inspection cost
########################################
timings['pose_inspection_s'] = perf_counter() - started

########################################
# Compare the floor, loaded assembly and unloaded approximation
########################################
print('Loaded support response\n-----------------------')
print('Stage             Vertical RMS [nm]')
records = {'Floor': floor, 'Girder': girder_motion, 'Magnet': magnet_motion,
           'BPM': bpm_motion, 'Unloaded cascade': naive}

for label, record in records.items():
    print(f'{label:<18} {np.std(record.values[:, 0, 1]) * 1e9:12.6f}')
print(f'Loaded cascade/direct mismatch [m]: {np.max(np.abs(magnet_motion.values-direct.values)):.3g}')
print('The pickup response is absolute; rigid inheritance is not added again.')
print('FFT replay is periodic offline response, not a causal startup simulation.')

################################################################################
# Inspect gain, phase, spectra and relative pickup motion
################################################################################

########################################
# Prepare diagnostic frequencies and panels
########################################
frequency = np.geomspace(.5, 120., 600)
fig, axes = plt.subplots(2, 2, figsize = (10, 7))

########################################
# Plot complex response gain and phase
########################################
for transfer, label, style in [(floor_to_girder, 'Floor → loaded girder', '-'),
                                (floor_to_magnet, 'Floor → magnet', '-'),
                                (unloaded_girder, 'Floor → unloaded girder', '--')]:
    response = transfer.evaluate(frequency)
    axes[0, 0].loglog(frequency, np.abs(response), style, label = label)
    axes[1, 0].semilogx(frequency, np.unwrap(np.angle(response)) * 180 / np.pi, style)

########################################
# Compare the output spectra
########################################
for label, record in records.items():
    f, density = welch(record.values[:, 0, 1], fs = 1 / SAMPLE_STEP_S, nperseg = 2048)
    axes[0, 1].loglog(f[1:], density[1:], label = label)

########################################
# Inspect magnet motion and pickup motion relative to the magnet
########################################
axes[1, 1].plot(times, magnet_motion.values[:, 0, 1] * 1e9, label = 'Magnet')
axes[1, 1].plot(times, (bpm_motion.values[:, 0, 1]-magnet_motion.values[:, 0, 1]) * 1e9,
                 label = 'BPM minus magnet')

########################################
# Label the spectral and time-domain comparisons
########################################
axes[0, 0].set(ylabel = 'Absolute displacement gain [m/m]')
axes[1, 0].set(xlabel = 'Frequency [Hz]', ylabel = 'Phase [deg]')
axes[0, 1].set(xlabel = 'Frequency [Hz]', ylabel = 'PSD [m²/Hz]', xlim = (.5, 120), ylim = (1e-24, 1e-16))
axes[1, 1].set(xlabel = 'Physical time [s]', ylabel = 'Displacement [nm]', xlim = (0, 1))
axes[0, 0].legend(fontsize = 8)
axes[0, 1].legend(fontsize = 8)
axes[1, 1].legend(fontsize = 8)
fig.tight_layout()

################################################################################
# Demonstrate causal startup and persistent state
################################################################################

########################################
# Start timing causal replay
########################################
started = perf_counter()

########################################
# Define a held base step and a causal pickup model
########################################
step = np.full((2048, 1), 50e-9)
causal = xt.mechanics.CausalResponse(oscillator = magnet_to_bpm, sample_step_s = CAUSAL_SAMPLE_STEP_S)

########################################
# Process the step as one record
########################################
whole = causal.process(step)

########################################
# Repeat from the same initial state using two consecutive blocks
########################################
causal.reset()
split = np.concatenate((causal.process(step[:700]), causal.process(step[700:])))

########################################
# Verify that splitting the record preserves filter state
########################################
np.testing.assert_array_equal(whole, split)
timings['causal_filter_s'] = perf_counter() - started

########################################
# Report replay agreement and the final displacement
########################################
print(f'Causal whole/split mismatch [m]: {np.max(np.abs(whole-split)):.3g}')
print(f'Causal final pickup after 50 nm step [nm]: {whole[-1, 0] * 1e9:.6f}')

########################################
# Plot the causal transient and its static limit
########################################
startup_fig, startup_ax = plt.subplots(figsize = (8, 3.8))
startup_ax.plot(np.arange(len(step)) * CAUSAL_SAMPLE_STEP_S, whole[:, 0] * 1e9, label = 'Causal pickup, initially at rest')
startup_ax.axhline(50., color = '0.5', ls = '--', label = 'Static base displacement')
startup_ax.set(xlabel = 'Time after step [s]', ylabel = 'Pickup [nm]', xlim = (0, .15))
startup_ax.legend()
startup_fig.tight_layout()

################################################################################
# Review execution costs and figures
################################################################################

########################################
# Print the measured costs
########################################
print('\nReference CPU costs\n-------------------')
for name, seconds in timings.items():
    print(f'{name:<24} {seconds:9.4f} s')

########################################
# Display the figures
########################################
plt.show()
