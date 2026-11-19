# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

# Mount a pump in girder-local coordinates and prescribe its two coupling paths.
# Changing the mount changes the floor footprint, gain and travel delays.

################################################################################
# Required packages
################################################################################
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Mounting and prescribed coupling
########################################
LATTICE_FILE = 'test_data/example_lattices/demo_ring/lattice.py'
QUADRUPOLES  = ['arc.s1.c1.qf.in', 'arc.s1.c1.qd', 'arc.s1.c1.qf.out']
PUMP_LOCAL_M = (.8, -.5, 0.)            # Local x,y,s on the first girder
ALTERNATIVE_LOCAL_M = (-.8, -.5, 2.)
FREQUENCY_HZ = 25.
EMITTED_PEAK_M = 50e-9                 # Prescribed displacement, not force
DIRECT_MOUNT_PEAK_M = 25e-9            # Same tone, direct support path [m]
DECAY_LENGTH_M = 20.                   # Amplitude gain exp(-distance/length)
PROPAGATION_SPEED_M_S = 50.
POLARIZATION = (0., 1., 0.)            # World XYZ

########################################
# Record and display
########################################
N_SAMPLES     = 4096
SAMPLE_STEP_S = 1 / 512
INSPECT_TIME_S = .01

################################################################################
# Put the source on a surveyed support
################################################################################
started = perf_counter()
repo_root = Path(__file__).resolve().parents[2]
nominal_env = xt.load(repo_root / LATTICE_FILE)
line = nominal_env['demo'].copy()
env = line.env
env['demo'] = line
mech = env.mechanics
line.mechanics.adopt_reference()
girders = [line.mechanics.new_girder(f'g.{name}', elements = [name]) for name in QUADRUPOLES]
pump = mech.new_body('pump', parent = girders[0], mount = xt.Frame.from_survey_angles(
    X = PUMP_LOCAL_M[0], Y = PUMP_LOCAL_M[1], Z = PUMP_LOCAL_M[2],
))
alternative = girders[0].nominal_frame @ xt.Frame.from_survey_angles(
    X = ALTERNATIVE_LOCAL_M[0], Y = ALTERNATIVE_LOCAL_M[1], Z = ALTERNATIVE_LOCAL_M[2],
)
positions = np.array([g.nominal_frame.XYZ for g in girders])
timings = {'setup_s': perf_counter() - started}

################################################################################
# Keep one latent pump tone at every receiver
################################################################################
started = perf_counter()
waveform = xt.mechanics.Harmonic(amplitude = EMITTED_PEAK_M, frequency_hz = FREQUENCY_HZ)
source = xt.mechanics.LocatedSource(
    waveform = waveform, position_m = tuple(pump.nominal_frame.XYZ),
    decay_length_m = DECAY_LENGTH_M, speed_m_s = PROPAGATION_SPEED_M_S,
    polarization = POLARIZATION,
)
other_source = xt.mechanics.LocatedSource(
    waveform = waveform, position_m = tuple(alternative.XYZ),
    decay_length_m = DECAY_LENGTH_M, speed_m_s = PROPAGATION_SPEED_M_S,
    polarization = POLARIZATION,
)
times = np.arange(N_SAMPLES) * SAMPLE_STEP_S
field = xt.mechanics.sample_field(source, time_s = times, positions_m = positions)
other_field = xt.mechanics.sample_field(other_source, time_s = times, positions_m = positions)
mech.add_source('pump_floor_path', source, targets = [g.name for g in girders])
mech.add_source('pump_mount_path', xt.mechanics.Harmonic(
    amplitude = DIRECT_MOUNT_PEAK_M, frequency_hz = FREQUENCY_HZ,
), targets = [girders[0].name], component = 'y')
timings['sample_field_s'] = perf_counter() - started

########################################
# Inspect the prescribed coupling assumptions
########################################
distance = np.linalg.norm(positions - pump.nominal_frame.XYZ, axis = 1)
print('Mounted pump and supplied propagation\n--------------------------------------')
print('Support                 Distance [m]    Amplitude gain    Delay [ms]')
for girder, radius in zip(girders, distance):
    print(f'{girder.name:<24} {radius:10.4f} {np.exp(-radius / DECAY_LENGTH_M):17.6f}'
          f' {radius / PROPAGATION_SPEED_M_S * 1e3:13.5f}')
print(f'Pump mount local x/y/s [m]: {PUMP_LOCAL_M}')
print(f'Direct mount tone [nm peak]: {DIRECT_MOUNT_PEAK_M * 1e9:.3f}')
print('The floor law and direct coupling are supplied models, not inferred elasticity.')
print('The pump body fixes nominal mounting; propagation uses that reference geometry.')

################################################################################
# Plot the footprint and the effect of changing the mount
################################################################################
offset = np.linspace(-50, 50, 101)
xx, zz = np.meshgrid(offset, offset)
grid = pump.nominal_frame.XYZ + np.column_stack((xx.ravel(), np.zeros(xx.size), zz.ravel()))
gain = np.exp(-np.linalg.norm(grid - pump.nominal_frame.XYZ, axis = 1) / DECAY_LENGTH_M)
fig, axes = plt.subplots(1, 2, figsize = (11, 4.2))
image = axes[0].pcolormesh(xx, zz, gain.reshape(xx.shape), shading = 'auto', vmin = 0, vmax = 1)
axes[0].plot(0, 0, 'r*', ms = 12, label = 'Pump')
relative = positions - pump.nominal_frame.XYZ
axes[0].plot(relative[:, 0], relative[:, 2], 'wo', mec = 'k', label = 'Support anchors')
axes[0].set(xlabel = 'World X relative to pump [m]', ylabel = 'World Z relative to pump [m]',
            aspect = 'equal', title = 'Supplied floor amplitude footprint')
axes[0].legend(fontsize = 8)
fig.colorbar(image, ax = axes[0], label = 'Displacement gain [m/m]')
for index, girder in enumerate(girders):
    axes[1].plot(times, field.values[:, index, 1] * 1e9, label = girder.name)
axes[1].plot(times, other_field.values[:, 0, 1] * 1e9, '--', color = '0.4', label = 'First support, alternative mount')
axes[1].set(xlabel = 'Physical time [s]', ylabel = 'Floor-path y [nm]', xlim = (0, .12))
axes[1].legend(fontsize = 8)
fig.tight_layout()
started = perf_counter()
mech.apply(time_s = INSPECT_TIME_S)
survey = mech.survey(time_s = INSPECT_TIME_S)
timings['apply_and_survey_s'] = perf_counter() - started
print('\nReference CPU costs\n-------------------')
for name, seconds in timings.items():
    print(f'{name:<24} {seconds:9.4f} s')
plt.show()
