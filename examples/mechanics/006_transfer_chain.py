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
# Loaded two-body assembly
########################################
LATTICE_FILE   = 'test_data/example_lattices/demo_ring/lattice.py'
QUADRUPOLE    = 'arc.s1.c1.qf.in'
BPM           = 'bpm.s1.f'
GIRDER_MASS_KG = 500.
MAGNET_MASS_KG = 150.
GIRDER_HZ      = 12.                    # Unloaded stage parameter, not coupled mode
MAGNET_HZ      = 35.                    # Fixed-base stage parameter
GIRDER_DAMPING = .15
MAGNET_DAMPING = .15
BPM_HZ         = 80.                    # Negligible pickup loading in this example
BPM_DAMPING    = .2

########################################
# Floor spectrum and offline record
########################################
N_SAMPLES     = 16384
SAMPLE_STEP_S = 1 / 256
PSD_SEED      = 956
FLOOR_DENSITY = 1e-18                   # One-sided displacement PSD [m²/Hz]
INSPECT_TIME_S = .125
CAUSAL_SAMPLE_STEP_S = 1 / 4096         # ZOH response to a held step, initially at rest

################################################################################
# Mount the magnet and pickup on surveyed rigid bodies
################################################################################
started = perf_counter()
repo_root = Path(__file__).resolve().parents[2]
nominal_env = xt.load(repo_root / LATTICE_FILE)
line = nominal_env['demo'].copy()
env = line.env
env['demo'] = line
mech = env.mechanics
line.mechanics.adopt_reference()
nominal = line.survey(include_element_frames = True)
girder = mech.new_girder('girder', anchor = nominal.get_frame(QUADRUPOLE, which = 'elem_start'))
magnet = mech.new_body('magnet', parent = girder, mount = xt.Frame(),
                       elements = [QUADRUPOLE], line = line)
bpm = mech.new_body('pickup', parent = magnet,
                    anchor = nominal.get_frame(BPM, which = 'elem_start'), elements = [BPM], line = line)
timings = {'setup_s': perf_counter() - started}

################################################################################
# Build consistent loaded response paths
################################################################################
omega_g = 2 * np.pi * GIRDER_HZ
omega_m = 2 * np.pi * MAGNET_HZ
parameters = dict(
    girder_mass_kg = GIRDER_MASS_KG, magnet_mass_kg = MAGNET_MASS_KG,
    floor_stiffness_n_m = GIRDER_MASS_KG * omega_g**2,
    magnet_stiffness_n_m = MAGNET_MASS_KG * omega_m**2,
    floor_damping_ns_m = 2 * GIRDER_DAMPING * GIRDER_MASS_KG * omega_g,
    magnet_damping_ns_m = 2 * MAGNET_DAMPING * MAGNET_MASS_KG * omega_m,
)
floor_to_girder = xt.mechanics.LoadedTwoMass(**parameters, stage = 'floor_to_girder')
girder_to_magnet = xt.mechanics.LoadedTwoMass(**parameters, stage = 'girder_to_magnet')
floor_to_magnet = xt.mechanics.LoadedTwoMass(**parameters, stage = 'floor_to_magnet')
magnet_to_bpm = xt.mechanics.BaseExcitedOscillator(BPM_HZ, BPM_DAMPING)
unloaded_girder = xt.mechanics.BaseExcitedOscillator(GIRDER_HZ, GIRDER_DAMPING)

########################################
# Reuse one floor realization at every stage
########################################
started = perf_counter()
times = np.arange(N_SAMPLES) * SAMPLE_STEP_S
source = xt.mechanics.DisplacementPSD(
    frequency_hz = [0., 1., 2., 80., 90.],
    density = [0., 0., FLOOR_DENSITY, FLOOR_DENSITY, 0.], seed = PSD_SEED,
)
floor_values = np.zeros((N_SAMPLES, 1, 6))
floor_values[:, 0, 1] = source.evaluate(times, np.zeros((1, 3)))[:, 0]
floor = xt.mechanics.RigidMotion(times, floor_values, ('girder',))
girder_motion = floor.through(floor_to_girder)
magnet_motion = girder_motion.transport(
    from_frames = [girder.nominal_frame], to_frames = [magnet.nominal_frame], target_names = ('magnet',),
).through(girder_to_magnet)
bpm_motion = magnet_motion.transport(
    from_frames = [magnet.nominal_frame], to_frames = [bpm.nominal_frame], target_names = ('pickup',),
).through(magnet_to_bpm)
direct = floor.through(floor_to_magnet)
naive = floor.through(unloaded_girder).through(girder_to_magnet)
np.testing.assert_allclose(magnet_motion.values, direct.values, rtol = 2e-12, atol = 1e-21)
for name, motion in [('girder_response', girder_motion), ('magnet_response', magnet_motion), ('bpm_response', bpm_motion)]:
    mech.add_source(name, motion, targets = motion.target_names, reference = 'absolute')
timings['sample_and_response_s'] = perf_counter() - started

########################################
# Apply absolute outputs without adding the parent a second time
########################################
started = perf_counter()
mech.apply(time_s = INSPECT_TIME_S)
survey = mech.survey(time_s = INSPECT_TIME_S)
expected = [np.interp(INSPECT_TIME_S, times, m.values[:, 0, 1])
            for m in (girder_motion, magnet_motion, bpm_motion)]
np.testing.assert_allclose(survey.dynamic_Y, expected, rtol = 0, atol = 1e-12)
timings['apply_and_survey_s'] = perf_counter() - started
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
# Compare transfer gain, phase and output spectra
################################################################################
frequency = np.geomspace(.5, 120., 600)
fig, axes = plt.subplots(2, 2, figsize = (10, 7))
for transfer, label, style in [(floor_to_girder, 'Floor → loaded girder', '-'),
                                (floor_to_magnet, 'Floor → magnet', '-'),
                                (unloaded_girder, 'Floor → unloaded girder', '--')]:
    response = transfer.evaluate(frequency)
    axes[0, 0].loglog(frequency, np.abs(response), style, label = label)
    axes[1, 0].semilogx(frequency, np.unwrap(np.angle(response)) * 180 / np.pi, style)
for label, record in records.items():
    f, density = welch(record.values[:, 0, 1], fs = 1 / SAMPLE_STEP_S, nperseg = 2048)
    axes[0, 1].loglog(f[1:], density[1:], label = label)
axes[1, 1].plot(times, magnet_motion.values[:, 0, 1] * 1e9, label = 'Magnet')
axes[1, 1].plot(times, (bpm_motion.values[:, 0, 1]-magnet_motion.values[:, 0, 1]) * 1e9,
                 label = 'BPM minus magnet')
axes[0, 0].set(ylabel = 'Absolute displacement gain [m/m]')
axes[1, 0].set(xlabel = 'Frequency [Hz]', ylabel = 'Phase [deg]')
axes[0, 1].set(xlabel = 'Frequency [Hz]', ylabel = 'PSD [m²/Hz]', xlim = (.5, 120), ylim = (1e-24, 1e-16))
axes[1, 1].set(xlabel = 'Physical time [s]', ylabel = 'Displacement [nm]', xlim = (0, 1))
axes[0, 0].legend(fontsize = 8)
axes[0, 1].legend(fontsize = 8)
axes[1, 1].legend(fontsize = 8)
fig.tight_layout()

################################################################################
# Keep causal response state across consecutive blocks
################################################################################
started = perf_counter()
step = np.full((2048, 1), 50e-9)
causal = xt.mechanics.CausalResponse(oscillator = magnet_to_bpm, sample_step_s = CAUSAL_SAMPLE_STEP_S)
whole = causal.process(step)
causal.reset()
split = np.concatenate((causal.process(step[:700]), causal.process(step[700:])))
np.testing.assert_array_equal(whole, split)
timings['causal_filter_s'] = perf_counter() - started
print(f'Causal whole/split mismatch [m]: {np.max(np.abs(whole-split)):.3g}')
print(f'Causal final pickup after 50 nm step [nm]: {whole[-1, 0] * 1e9:.6f}')
startup_fig, startup_ax = plt.subplots(figsize = (8, 3.8))
startup_ax.plot(np.arange(len(step)) * CAUSAL_SAMPLE_STEP_S, whole[:, 0] * 1e9, label = 'Causal pickup, initially at rest')
startup_ax.axhline(50., color = '0.5', ls = '--', label = 'Static base displacement')
startup_ax.set(xlabel = 'Time after step [s]', ylabel = 'Pickup [nm]', xlim = (0, .15))
startup_ax.legend()
startup_fig.tight_layout()
print('\nReference CPU costs\n-------------------')
for name, seconds in timings.items():
    print(f'{name:<24} {seconds:9.4f} s')
plt.show()
