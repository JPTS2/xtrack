# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
import numpy as np
import xtrack as xt

################################################################################
# User parameters
################################################################################
P0C             = 1e9                  # Proton reference p0*c [eV]
LENS_STRENGTH   = 1e-6                 # Integrated knll [m]
LENS_SCALE      = 1e-2                 # Singularities at x = +/-cnll [m]
LENS_AMPLITUDES = np.array([0.1, 0.3, 0.6])  # x/cnll; y = 0

################################################################################
# Track an elliptic nonlinear lens
################################################################################

########################################
# Launch inside the two singular points
########################################
# This potential is non-polynomial. Keep |x/cnll| < 1 at y = 0; its small-
# amplitude quadrupole term has integrated strength 2*knll/cnll**2 [m^-1].
lens = xt.NonLinearLens(knll = LENS_STRENGTH, cnll = LENS_SCALE)
lens_line = xt.Line(elements = [lens])
lens_line.set_particle_ref('proton', p0c = P0C)
lens_particles = lens_line.build_particles(x = LENS_SCALE * LENS_AMPLITUDES)
lens_line.track(lens_particles)

########################################
# Compare with the exact horizontal-axis derivative
########################################
# For u = x/cnll, the potential on this interval is
# U(u) = -u*asin(u)/sqrt(1-u**2); the kick is knll/cnll * dU/du.
u = LENS_AMPLITUDES
lens_expected_px = -LENS_STRENGTH / LENS_SCALE * (
    np.arcsin(u) / (1 - u**2)**1.5 + u / (1 - u**2))
linear_px = -2 * LENS_STRENGTH / LENS_SCALE * u
print('Elliptic lens x/cnll:', u)
print('Exact elliptic kick P_x/p0:', lens_expected_px)
print('Tracked elliptic kick P_x/p0:', lens_particles.px)
print('Small-amplitude quadrupole approximation P_x/p0:', linear_px)
