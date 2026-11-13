# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt

################################################################################
# User parameters
################################################################################
P0C             = 1e9                   # Proton reference p0*c [eV]
LENGTH          = 10.0                  # Bookkeeping path length [m]
YAW             = 0.05                  # [rad]
PITCH           = 0.04                  # [rad]
ROLL            = 0.1                   # [rad]
X_START         = 1e-3                  # Field-free entrance x [m]
Y_START         = 2e-3                  # Field-free entrance y [m]

################################################################################
# Build a closed sequence of nonplanar coordinate changes
################################################################################

env = xt.Environment()
components = []
for index, (position, axis, angle) in enumerate([
    (1.0, 'y', YAW), (2.0, 'y', -YAW),
    (3.0, 'x', PITCH), (4.0, 'x', -PITCH),
    (4.5, 's', ROLL), (5.5, 's', -ROLL),
    (6.0, 'x', -PITCH), (7.0, 'x', PITCH),
    (8.0, 'y', -YAW), (9.0, 'y', YAW),
]):
    components.append(env.new(
        f'rotation.{index}', xt.Rotation,
        at = position, **{f'rot_{axis}_rad': angle}))
line = env.new_line(length = LENGTH, components = components)
line.set_particle_ref('proton', p0c = P0C)
line.configure_drift_model(model = 'exact')
# These passive coordinate changes do not apply magnetic forces. The reference
# polyline bends out of the horizontal plane, while the physical ray is free.

################################################################################
# Reconstruct the physical ray in global coordinates
################################################################################

########################################
# Combine the local orbit with the surveyed reference basis
########################################
survey = line.survey()
tw = line.twiss4d(betx = 1.0, bety = 1.0, x = X_START, y = Y_START)
trajectory = (survey.XYZ + tw.x[:, None] * survey.ex
               + tw.y[:, None] * survey.ey)
print('Maximum physical X/Y change [m]:',
      np.ptp(trajectory[:, 0]), np.ptp(trajectory[:, 1]))
print('Reference exit XYZ [m]:', survey.XYZ[-1])
print('Ray exit XYZ [m]:', trajectory[-1])
# Samples lie on differently oriented observation planes. Their global Z
# coordinates need not equal the reference bookkeeping s.

########################################
# Separate coordinate excursions from a physical orbit bump
########################################
fig, axes = plt.subplots(3, 1, figsize = (9, 8), layout = 'constrained')
axes[0].plot(tw.s, tw.x * 1e3, label = 'Local x')
axes[0].plot(tw.s, tw.y * 1e3, label = 'Local y')
axes[0].set(xlabel = 'Reference s [m]', ylabel = 'Local position [mm]')
for axis, coordinate, index in [(axes[1], 'X', 0), (axes[2], 'Y', 1)]:
    axis.plot(survey.Z, survey.XYZ[:, index], label = 'Reference path')
    axis.plot(trajectory[:, 2], trajectory[:, index], 'o-',
              label = 'Physical field-free ray')
    axis.set(xlabel = 'Global Z [m]', ylabel = f'Global {coordinate} [m]')
for axis in axes:
    axis.legend()
    axis.grid()
plt.show()
