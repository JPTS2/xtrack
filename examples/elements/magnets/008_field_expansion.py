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
# Reference and physical field
########################################
P0C             = 1.0e9                 # Proton reference p0*c [eV]
LENGTH          = 0.4                   # Local longitudinal interval [m]
MEAN_BY         = 0.02                  # Mean vertical field [T]

########################################
# Numerical settings
########################################
N_STEPS         = 64                    # RK4 steps across the element
N_FIELD_POINTS  = 101

################################################################################
# Express the field as normalized polynomial coefficients
################################################################################

########################################
# Normalize to the reference rigidity
########################################
reference   = xt.Particles(mass0 = xt.PROTON_MASS_EV, p0c = P0C)
rigidity    = reference.rigidity0[0]
k_mean      = MEAN_BY / rigidity

########################################
# Set a smooth normal-dipole profile
########################################
# For t = s_local/LENGTH, B_y/B_rho = 30*k_mean*t^2*(1-t)^2 on axis.
# knc[i, j] multiplies s^j*x^i/i!: row 0 is the normal dipole.
# This is the same physical profile used in the spline-field lesson.
coefficients = [
    0.0, 0.0, 30 * k_mean / LENGTH**2,
    -60 * k_mean / LENGTH**3, 30 * k_mean / LENGTH**4]
element = xt.BFieldExpansion(
    length                  = LENGTH,
    knc                     = [coefficients],
    num_integration_steps   = N_STEPS,
    pkin_const              = True,
)

################################################################################
# Inspect field reconstruction and integrated strength
################################################################################

########################################
# Evaluate the physical field on axis
########################################
s_local = np.linspace(0, LENGTH, N_FIELD_POINTS)
field   = element.get_field(x = 0.0, y = 0.0, s_local = s_local)
# get_field returns normalized fields; multiply by B_rho to obtain tesla.
by      = rigidity * field['By']
print('Resolved vertical expansion order:', element.num_phi)
print('RK4 integration steps:', element.num_integration_steps)

########################################
# Inspect the integrated normal coefficients
########################################
knl, ksl = element.get_total_knl_ksl()
print('Integrated dipole coefficient [1]:', knl[0])
print('Expected integral(B_y ds)/B_rho [1]:', k_mean * LENGTH)
fig, ax = plt.subplots(figsize = (8, 4))
ax.plot(s_local, by, label = 'BFieldExpansion B_y at x = y = 0')
ax.set_xlabel('s_local [m]')
ax.set_ylabel('B_y [T]')
ax.legend()
ax.grid()
fig.tight_layout()

################################################################################
# Track with an explicit boundary-momentum convention
################################################################################

########################################
# Set the field-free entrance state
########################################
# pkin_const=True preserves kinetic momentum at boundary conversions and
# returns zero transverse vector potential at exit. RK4 is not symplectic.
particles   = reference.copy()
expected_px = -MEAN_BY * LENGTH / rigidity

########################################
# Inspect exit position and kinetic momentum
########################################
line = xt.Line(elements = [element])
line.track(particles)
print('Expected P_x/p0:', expected_px)
print('Tracked kinetic P_x/p0:', particles.kin_px[0])
print('Momentum difference:', particles.kin_px[0] - expected_px)
print('Exit x [m]:', particles.x[0])
print('Exit normalized transverse vector potential:',
      particles.ax[0], particles.ay[0])
plt.show()
