# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

# Adopt deliberate magnet offsets, then add installation and dynamic motion.
# Remove mechanics and verify that the intended magnetic geometry survives.

################################################################################
# Required packages
################################################################################
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import xobjects as xo
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Deliberate reference and rigid support
########################################
CONTEXT           = xo.ContextCpu()
DELIBERATE_X_M    = 1e-3                 # Intentional off-axis quadrupole [m]
DELIBERATE_Y_M    = 3e-6                 # Existing input field; never copied by hand
DELIBERATE_ROLL   = .1                  # Design roll [rad]
INSTALLATION_Y_M  = 20e-6                # Static support placement error [m]
INSTALLATION_YAW  = 2e-6                 # Static support placement error [rad]
DYNAMIC_Y_M       = 50e-9                # Peak vibration [m]
DYNAMIC_HZ        = 5.
INSPECT_TIME_S    = .05
DISPLACEMENT_SCALE = 1e4
ROTATION_SCALE     = 1e5

################################################################################
# Build the intended magnetic geometry and adopt its existing offsets
################################################################################
started = perf_counter()
env = xt.Environment()
env.new('q1', xt.Quadrupole, length = .2, k1 = .2,
        shift_x = DELIBERATE_X_M, shift_y = DELIBERATE_Y_M)
env.new('gap', xt.Drift, length = 2.)
env.new('q2', xt.Quadrupole, length = .2, k1 = -.2, rot_s_rad = DELIBERATE_ROLL)
line = env.new_line(name = 'line', components = ['q1', 'gap', 'q2'])
line.set_particle_ref('proton', p0c = 7e12)
line.mechanics.adopt_reference()
reference = line.mechanics.get_reference_table()
mech = env.mechanics
girder = line.mechanics.new_girder('girder', elements = ['q1', 'q2'])
timings = {'setup_s': perf_counter() - started}

################################################################################
# Keep installation and dynamic inputs separate
################################################################################
girder.installation.shift_y = INSTALLATION_Y_M
girder.installation.rot_y_rad = INSTALLATION_YAW
mech.add_source('vibration', xt.mechanics.Harmonic(
    amplitude = DYNAMIC_Y_M, frequency_hz = DYNAMIC_HZ,
), targets = ['girder'], component = 'y')
started = perf_counter()
mech.apply(time_s = INSPECT_TIME_S)
survey = mech.survey(time_s = INSPECT_TIME_S)
timings['apply_and_survey_s'] = perf_counter() - started
print('Reference, installation and dynamic placement\n---------------------------------------------')
print(f'Adopted q1 deliberate x / y [µm]: {reference["shift_x", "q1"] * 1e6:.3f} / '
      f'{reference["shift_y", "q1"] * 1e6:.3f}')
print(f'Girder installation y [µm]: {girder.installation.shift_y * 1e6:.3f}')
print(f'Girder dynamic world Y [nm]: {survey.dynamic_Y[0] * 1e9:.3f}')
print(f'q2 design roll [rad]: {env["q2"].rot_s_rad:.6f}')
layout_fig, layout_axes = mech.plot(
    time_s = INSPECT_TIME_S, line = line,
    displacement_scale = DISPLACEMENT_SCALE, rotation_scale = ROTATION_SCALE,
)

################################################################################
# Track the effective geometry and remove only the mechanical layers
################################################################################
started = perf_counter()
line.build_tracker(_context = CONTEXT)
timings['tracker_build_s'] = perf_counter() - started
particles = line.build_particles()
started = perf_counter()
line.track(particles)
timings['one_pass_tracking_s'] = perf_counter() - started
print(f'Tracked exit x / y [µm]: {float(particles.x[0]) * 1e6:.6f} / '
      f'{float(particles.y[0]) * 1e6:.6f}')
line.mechanics.disable_vibrations()
static_y = env['q1'].shift_y
mech.clear()
np.testing.assert_equal(env['q1'].shift_x, DELIBERATE_X_M)
np.testing.assert_equal(env['q1'].shift_y, DELIBERATE_Y_M)
np.testing.assert_equal(env['q2'].rot_s_rad, DELIBERATE_ROLL)
print(f'q1 y after disabling / clearing [µm]: {static_y * 1e6:.6f} / {env["q1"].shift_y * 1e6:.6f}')
print('Disable keeps installation. Clear restores the adopted deliberate inputs.')
print('\nReference CPU costs\n-------------------')
for name, seconds in timings.items():
    print(f'{name:<24} {seconds:9.4f} s')
plt.show()
