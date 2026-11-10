# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Shared planar geometry at a local 10 GeV/c electron setting
########################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
P0C             = 10e9                  # Reference p0*c [eV]

########################################
# Number of representative bending-element rows to display
########################################
N_ROWS          = 6

################################################################################
# Request native radiation-integral predictions from radiation-free optics
################################################################################

########################################
# Compute periodic 4D optics, including native integral and integrand fields
########################################
repo_root   = Path(__file__).resolve().parents[2]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env['demo']
env['p0c']  = P0C
line.set_particle_ref('electron', p0c = env['p0c'])
twiss       = line.twiss4d(radiation_integrals = True)
print('I2 [m^-1] / I3 [m^-2] / I4x [m^-1] / I5x [m^-1]:',
      twiss.rad_int_i2, twiss.rad_int_i3,
      twiss.rad_int_i4x, twiss.rad_int_i5x)
print('Native energy loss per turn [eV]:', twiss.rad_int_energy_loss)
print('Native equilibrium geometric emittance x / y [m]:',
      twiss.rad_int_eq_gemitt_x, twiss.rad_int_eq_gemitt_y)
print('Native equilibrium sigma_delta:', twiss.rad_int_sigma_delta)
print('Native damping rates x / y / zeta [s^-1]:',
      twiss.rad_int_damping_constant_x_s,
      twiss.rad_int_damping_constant_y_s,
      twiss.rad_int_damping_constant_zeta_s)

# Geometry and normalized strengths are unchanged: physical magnet fields
# scale tenfold with reference rigidity. This is not a hardware qualification.

# No radiation tracking or RF compensation is required for these predictions.
# They describe linear radiation-integral equilibrium, not a tracked bunch.
# The ideal planar uncoupled model has no vertical quantum-excitation floor.
# Equilibrium predictions do not establish momentum acceptance or survival.

########################################
# Inspect where the energy-loss and horizontal-excitation integrands are large
########################################
bending = twiss.rows[twiss.rad_int_i2_integrand
                     > 0.01 * np.max(twiss.rad_int_i2_integrand)]
bending.rows[:N_ROWS].cols[
    's length rad_int_curly_hx rad_int_i2_integrand '
    'rad_int_i5x_integrand'].show(maxwidth = 'full')

################################################################################
# Locate the sources of I2 and I5x around the ring
################################################################################

########################################
# Accumulate native element integrands with their native length weights
########################################
s_end           = twiss.s + twiss.length
cumulative_i2   = np.cumsum(twiss.rad_int_i2_integrand * twiss.length)
cumulative_i5x  = np.cumsum(twiss.rad_int_i5x_integrand * twiss.length)
print('Cumulative endpoints I2 / I5x [m^-1]:',
      cumulative_i2[-1], cumulative_i5x[-1])

# Integrands are per unit length, not already-integrated element contributions.
# These sums use the same element weights as the native scalar integrals.

########################################
# Compare cumulative energy-loss and horizontal-excitation contributions
########################################
fig, axes = plt.subplots(2, 1, figsize = (8, 6), sharex = True)
axes[0].plot(s_end, cumulative_i2)
axes[1].plot(s_end, cumulative_i5x)
axes[0].set_ylabel('Cumulative I2 [m$^{-1}$]')
axes[1].set_ylabel('Cumulative I5x [m$^{-1}$]')
axes[1].set_xlabel('s [m]')
for axis in axes:
    axis.grid()
fig.tight_layout()
plt.show()

# RF and magnet tapering are treated separately in 002. Integral predictions
# here are based on the nominal radiation-free lattice and its thick-element
# representation; this is not an integration-step convergence study.
