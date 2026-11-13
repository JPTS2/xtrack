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

########################################
# Common straight reference and exterior planes
########################################
P0C             = 1e9                   # Proton reference p0*c [eV]
K0              = 0.15                  # Vertical field / rigidity [m^-1]
MAGNET_LENGTH   = 1.0                   # Hard-edge magnetic length [m]
DRIFT_LENGTH    = 0.5                   # Each surrounding drift [m]
FRINGE_LENGTH   = 0.05                  # Each quintic ramp [m]
Y_START         = np.array([0, 0.001, 0.002, 0.003])  # [m]

########################################
# Integration controls
########################################
RAMP_STEPS      = 64
BODY_STEPS      = 128

################################################################################
# Keep the field integral and exterior planes fixed
################################################################################

########################################
# Hard-edge field with full dipole edges
########################################
hard_edge = xt.Line(elements = [
    xt.Drift(length = DRIFT_LENGTH),
    xt.Bend(
        length = MAGNET_LENGTH, angle = 0.0, k0 = K0,
        edge_entry_model = 'full', edge_exit_model = 'full'),
    xt.Drift(length = DRIFT_LENGTH),
])
# Reference curvature is zero in both models; the powered field bends the ray.
# The comparison uses common field-free entrance/exit planes, not a curved
# reference line in one model and a straight one in the other.

########################################
# Smooth C2 field ramps and a shortened plateau
########################################
rising = K0 * np.array([
    0.0, 0.0, 0.0, 10 / FRINGE_LENGTH**3,
    -15 / FRINGE_LENGTH**4, 6 / FRINGE_LENGTH**5])
falling = -rising.copy()
falling[0] = K0
outputs = {}
for steps in [RAMP_STEPS, 2 * RAMP_STEPS]:
    smooth = xt.Line(elements = [
        xt.Drift(length = DRIFT_LENGTH - FRINGE_LENGTH / 2),
        xt.BFieldExpansion(
            length = FRINGE_LENGTH, knc = [rising],
            num_integration_steps = steps, pkin_const = True),
        xt.BFieldExpansion(
            length = MAGNET_LENGTH - FRINGE_LENGTH, knc = [[K0]],
            num_integration_steps = BODY_STEPS, pkin_const = True),
        xt.BFieldExpansion(
            length = FRINGE_LENGTH, knc = [falling],
            num_integration_steps = steps, pkin_const = True),
        xt.Drift(length = DRIFT_LENGTH - FRINGE_LENGTH / 2),
    ])
    particles = xt.Particles('proton', p0c = P0C, y = Y_START)
    smooth.track(particles)
    outputs[steps] = particles
initial = xt.Particles('proton', p0c = P0C, y = Y_START)
hard_edge.track(initial)
print('Common length [m]:', hard_edge.get_length(), smooth.get_length())
print('Common integral k0 ds:', K0 * MAGNET_LENGTH)

################################################################################
# Separate finite-profile changes from ramp integration error
################################################################################

########################################
# Compare the common field-free outputs
########################################
resolved = outputs[2 * RAMP_STEPS]
print('Ramp-step refinement: largest x/py difference:',
      np.max(np.abs(outputs[RAMP_STEPS].x - resolved.x)),
      np.max(np.abs(outputs[RAMP_STEPS].py - resolved.py)))
print('Smooth minus hard-edge exit py:', resolved.py - initial.py)
# Equal magnetic integrals do not imply equal fringe focusing. The body RK4
# setting is held fixed; the refinement above checks the ramp step only.

########################################
# Inspect the profile and its vertical kick
########################################
fig, axes = plt.subplots(1, 2, figsize = (10, 4), layout = 'constrained')
z = np.linspace(0, FRINGE_LENGTH, 100)
for polynomial, start in [(rising, DRIFT_LENGTH - FRINGE_LENGTH / 2),
                           (falling, DRIFT_LENGTH + MAGNET_LENGTH
                            - FRINGE_LENGTH / 2)]:
    axes[0].plot(start + z,
                 np.polynomial.polynomial.polyval(z, polynomial), 'C0')
axes[0].plot([DRIFT_LENGTH + FRINGE_LENGTH / 2,
              DRIFT_LENGTH + MAGNET_LENGTH - FRINGE_LENGTH / 2],
             [K0, K0], 'C0', label = 'Smooth on-axis field')
axes[0].plot([DRIFT_LENGTH, DRIFT_LENGTH,
              DRIFT_LENGTH + MAGNET_LENGTH,
              DRIFT_LENGTH + MAGNET_LENGTH], [0, K0, K0, 0],
             'k--', label = 'Hard edge')
axes[0].set(xlabel = 's [m]', ylabel = r'$B_y/(B\rho)$ [m$^{-1}$]')
axes[1].plot(Y_START * 1e3, (resolved.py - initial.py) * 1e6, 'o-')
axes[1].set(xlabel = 'Entrance y [mm]',
             ylabel = r'Smooth minus hard-edge $p_y$ [$10^{-6}$]')
axes[0].legend()
for axis in axes:
    axis.grid()
plt.show()
