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
# Execution context
########################################
CONTEXT = xo.ContextCpu()

########################################
# Deliberate magnet offsets and design roll
########################################
DELIBERATE_X_M  = 1e-3                # Intentional off-axis quadrupole [m]
DELIBERATE_Y_M  = 3e-6                # Existing input field; never copied by hand
DELIBERATE_ROLL = .1                  # Design roll [rad]

########################################
# Static installation errors
########################################
INSTALLATION_Y_M = 20e-6              # Static support placement error [m]
INSTALLATION_YAW = 2e-6               # Static support placement error [rad]

########################################
# Dynamic vibration
########################################
DYNAMIC_Y_M = 50e-9                   # Peak vibration [m]
DYNAMIC_HZ  = 5.

########################################
# Inspection time and display magnification
########################################
INSPECT_TIME_S     = .05
DISPLACEMENT_SCALE = 1e4
ROTATION_SCALE     = 1e5

################################################################################
# Set up the ordinary Xtrack line
################################################################################

########################################
# Start performance monitoring
########################################
timings = {}

########################################
# Time line construction
########################################
started = perf_counter()

########################################
# Construct the line and its reference particle
########################################
env = xt.Environment()
env.new('q1', xt.Quadrupole, length = .2, k1 = .2,
        shift_x = DELIBERATE_X_M, shift_y = DELIBERATE_Y_M)
env.new('gap', xt.Drift, length = 2.)
env.new('q2', xt.Quadrupole, length = .2, k1 = -.2, rot_s_rad = DELIBERATE_ROLL)
line = env.new_line(name = 'line', components = ['q1', 'gap', 'q2'])
line.set_particle_ref('proton', p0c = 7e12)

timings['line_construction_s'] = perf_counter() - started

########################################
# Prepare and time the tracker before mechanics queries
########################################
started = perf_counter()
line.build_tracker(_context = CONTEXT)
timings['tracker_build_s'] = perf_counter() - started

################################################################################
# Define the mechanics reference and girder
################################################################################

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
# Time reference inspection
########################################
started = perf_counter()

########################################
# Inspect the adopted reference inputs
########################################
reference = line.mechanics.get_reference_table()

timings['reference_inspection_s'] = perf_counter() - started

########################################
# Time mechanics definition
########################################
started = perf_counter()

########################################
# Access the shared mechanics model
########################################
mech = env.mechanics

########################################
# Attach the two existing quadrupoles to one girder
########################################
girder = line.mechanics.new_girder('girder', elements = ['q1', 'q2'])

timings['mechanics_definition_s'] = perf_counter() - started

################################################################################
# Define installation and vibration inputs
################################################################################

########################################
# Time the static installation definition
########################################
started = perf_counter()

########################################
# Set the static installation placement
########################################
girder.installation.shift_y = INSTALLATION_Y_M
girder.installation.rot_y_rad = INSTALLATION_YAW
timings['installation_definition_s'] = perf_counter() - started

########################################
# Time the dynamic source definition
########################################
started = perf_counter()

########################################
# Bind a vertical harmonic vibration to the girder
########################################
mech.add_source('vibration', xt.mechanics.Harmonic(
    amplitude = DYNAMIC_Y_M, frequency_hz = DYNAMIC_HZ,
), targets = ['girder'], component = 'y')
timings['source_definition_s'] = perf_counter() - started

################################################################################
# Apply and inspect one mechanical state
################################################################################

########################################
# Start timing the pose update
########################################
started = perf_counter()

########################################
# Apply motion at the selected physical time
########################################
mech.apply(time_s = INSPECT_TIME_S)
timings['apply_s'] = perf_counter() - started

########################################
# Time the read-only pose inspection separately
########################################
started = perf_counter()
survey = mech.survey(time_s = INSPECT_TIME_S)

########################################
# Record the pose-inspection cost
########################################
timings['pose_inspection_s'] = perf_counter() - started

########################################
# Compare deliberate, installed and dynamic positions
########################################
print('Reference, installation and dynamic placement\n---------------------------------------------')
print(f'Adopted q1 deliberate x / y [µm]: {reference["shift_x", "q1"] * 1e6:.3f} / '
      f'{reference["shift_y", "q1"] * 1e6:.3f}')
print(f'Girder installation y [µm]: {girder.installation.shift_y * 1e6:.3f}')
print(f'Girder dynamic world Y [nm]: {survey.dynamic_Y[0] * 1e9:.3f}')
print(f'q2 design roll [rad]: {env["q2"].rot_s_rad:.6f}')

########################################
# Draw the magnified support geometry
########################################
layout_fig, layout_axes = mech.plot(
    time_s = INSPECT_TIME_S, line = line,
    displacement_scale = DISPLACEMENT_SCALE, rotation_scale = ROTATION_SCALE,
)

################################################################################
# Track through the displaced magnets
################################################################################

########################################
# Create a particle at the nominal line entrance
########################################
particles = line.build_particles()

########################################
# Start timing one traversal
########################################
started = perf_counter()

########################################
# Track and inspect the exit coordinates
########################################
line.track(particles)
timings['one_pass_tracking_s'] = perf_counter() - started
print(f'Tracked exit x / y [µm]: {float(particles.x[0]) * 1e6:.6f} / '
      f'{float(particles.y[0]) * 1e6:.6f}')

################################################################################
# Remove dynamic and installation effects in turn
################################################################################

########################################
# Disable vibration while retaining installation errors
########################################
line.mechanics.disable_vibrations()
static_y = env['q1'].shift_y

########################################
# Clear mechanics and restore the adopted magnet inputs
########################################
mech.clear()

########################################
# Verify that deliberate offsets and roll survive
########################################
np.testing.assert_equal(env['q1'].shift_x, DELIBERATE_X_M)
np.testing.assert_equal(env['q1'].shift_y, DELIBERATE_Y_M)
np.testing.assert_equal(env['q2'].rot_s_rad, DELIBERATE_ROLL)

########################################
# Compare the two cleanup operations
########################################
print(f'q1 y after disabling / clearing [µm]: {static_y * 1e6:.6f} / {env["q1"].shift_y * 1e6:.6f}')
print('Disable keeps installation. Clear restores the adopted deliberate inputs.')

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
