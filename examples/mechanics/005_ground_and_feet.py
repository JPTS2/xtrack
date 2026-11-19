# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

# Sample one floor field at four feet per girder and reconstruct rigid motion.
# Show the part a rigid body can follow and the remaining contact mismatch.

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
# Ring, supports and floor field
########################################
LATTICE_FILE = 'test_data/example_lattices/demo_ring/lattice.py'
QUADRUPOLES  = ['arc.s1.c1.qf.in', 'arc.s1.c1.qd', 'arc.s1.c1.qf.out']
FEET_LOCAL_M = [(-.3, -.5, -1.), (.3, -.5, -1.),
                (-.3, -.5, 1.), (.3, -.5, 1.)]
SOURCE       = 'wave'                   # 'wave' or 'psd'
AMPLITUDE_M  = 50e-9                    # Peak vector displacement [m]
FREQUENCY_HZ = 5.
SPEED_M_S    = 100.
DIRECTION    = (1., 0., 1.)             # World XYZ
POLARIZATION = (.3, 1., .2)
PSD_SEED     = 956
CORRELATION_LENGTH_M = 30.

########################################
# Sampling and inspection
########################################
N_SAMPLES          = 4096
SAMPLE_STEP_S      = 1 / 256
INSPECT_TIME_S     = .125
DISPLACEMENT_SCALE = 1e8
ROTATION_SCALE     = 1e7

################################################################################
# Place the feet in the surveyed world
################################################################################
started = perf_counter()
repo_root = Path(__file__).resolve().parents[2]
nominal_env = xt.load(repo_root / LATTICE_FILE)
line = nominal_env['demo'].copy()        # Independent elements for reference lowering
env = line.env
env['demo'] = line
mech = env.mechanics
line.mechanics.adopt_reference()
girders = [line.mechanics.new_girder(f'g.{name}', elements = [name], feet_local_m = FEET_LOCAL_M)
           for name in QUADRUPOLES]
frames = [g.nominal_frame for g in girders]
feet = np.array([g.feet_local_m for g in girders])
positions = np.concatenate([g.nominal_feet_m for g in girders])
timings = {'setup_s': perf_counter() - started}

################################################################################
# Sample the common floor and fit rigid support poses
################################################################################
started = perf_counter()
if SOURCE == 'wave':
    source = xt.mechanics.PlaneWave(
        amplitude = AMPLITUDE_M, frequency_hz = FREQUENCY_HZ,
        direction = DIRECTION, polarization = POLARIZATION, speed_m_s = SPEED_M_S,
    )
elif SOURCE == 'psd':
    source = xt.mechanics.DisplacementPSD(
        frequency_hz = [0., 1., 2., 30., 40.],
        density = [0., 0., 1e-17, 1e-17, 0.], seed = PSD_SEED,
        correlation_length_m = CORRELATION_LENGTH_M,
    )
else:
    raise ValueError("SOURCE must be 'wave' or 'psd'")
times = np.arange(N_SAMPLES) * SAMPLE_STEP_S
floor = xt.mechanics.sample_field(source, time_s = times, positions_m = positions)
motion, residual = floor.fit_rigid_bodies(
    frames = frames, feet_local_m = feet, target_names = [g.name for g in girders],
)
mech.add_source('floor_following', motion, targets = motion.target_names, reference = 'absolute')
timings['sample_and_fit_s'] = perf_counter() - started

########################################
# Compare input feet and fitted motion
########################################
table = motion.get_table(girders[0].name)
print('Floor motion and rigid supports\n-------------------------------')
print(f'Field: {SOURCE}; supports: {len(girders)}; feet per support: {feet.shape[1]}')
print(f'Record length / step [s]: {N_SAMPLES * SAMPLE_STEP_S:g} / {SAMPLE_STEP_S:g}')
print('Coordinate      First-girder RMS')
for name in ('shift_x', 'shift_y', 'shift_s', 'rot_x_rad', 'rot_y_rad', 'rot_s_rad'):
    scale, unit = (1e9, 'nm') if name.startswith('shift') else (1e6, 'µrad')
    print(f'{name:<15} {np.std(table[name]) * scale:12.6f} {unit}')
print(f'Contact mismatch RMS [nm]: {np.sqrt(np.mean(residual**2)) * 1e9:.6f}')
print('The contact mismatch is reported; the girder does not deform to remove it.')

################################################################################
# Inspect magnified geometry and actual six-coordinate histories
################################################################################
started = perf_counter()
mech.apply(time_s = INSPECT_TIME_S)
survey = mech.survey(time_s = INSPECT_TIME_S)
timings['apply_and_survey_s'] = perf_counter() - started
layout_fig, layout_axes = mech.plot(
    time_s = INSPECT_TIME_S, line = line, bodies = [girders[0].name], view = 'bodies',
    displacement_scale = DISPLACEMENT_SCALE, rotation_scale = ROTATION_SCALE,
)
fig, axes = plt.subplots(3, 1, sharex = True, figsize = (8, 7))
for name, label in [('shift_x', 'x'), ('shift_y', 'y'), ('shift_s', 's')]:
    axes[0].plot(times, table[name] * 1e9, label = label)
for name, label in [('rot_x_rad', 'Pitch'), ('rot_y_rad', 'Yaw'), ('rot_s_rad', 'Roll')]:
    axes[1].plot(times, table[name] * 1e6, label = label)
axes[2].plot(times, residual[:, 0] * 1e9)
axes[0].set(ylabel = 'Translation [nm]', title = 'First support: finite rigid fit to moving feet')
axes[1].set(ylabel = 'Survey angle [µrad]')
axes[2].set(xlabel = 'Physical time [s]', ylabel = 'Contact mismatch [nm]', xlim = (0, 1))
axes[0].legend()
axes[1].legend()
fig.tight_layout()
print('\nReference CPU costs\n-------------------')
for name, seconds in timings.items():
    print(f'{name:<24} {seconds:9.4f} s')
plt.show()
