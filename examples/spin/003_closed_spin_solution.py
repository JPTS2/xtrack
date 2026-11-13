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
# Shared electron geometry with radiation and RF off
########################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
P0C             = 10e9                  # Reference p0*c [eV]

########################################
# Controlled longitudinal field in an existing drift
########################################
SOLENOID_S      = 2.0                   # Entrance s [m]
SOLENOID_LENGTH = 0.2                   # [m]
SOLENOID_KS     = 0.05                  # B_s/(B*rho) [m^-1]

################################################################################
# Obtain the native closed spin direction before and after adding a solenoid
################################################################################

########################################
# Establish the planar reference solution
########################################
repo_root   = Path(__file__).resolve().parents[2]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env['demo']
line.set_particle_ref('electron', p0c = P0C)
line.configure_spin('auto')
planar      = line.twiss4d(spin = True)
# Local spin axes are horizontal x, vertical y and longitudinal z.
print('Planar entrance n0 [x,y,z]:',
      planar.spin_x[0], planar.spin_y[0], planar.spin_z[0])

# Fixed normalized strengths imply physical fields scaled with rigidity.
# Spin is a unit vector in the local reference frame; n0 and -n0 are equivalent
# invariant axes. The native solution chooses a positive vertical orientation.

########################################
# Add a real solenoid and solve the new coupled orbit and closed spin
########################################
line.insert(
    'spin.solenoid',
    xt.UniformSolenoid(
        length  = SOLENOID_LENGTH,
        ks      = SOLENOID_KS,
    ),
    at      = SOLENOID_S,
    anchor  = 'start',
)
twiss       = line.twiss4d(spin = True)

########################################
# Track the closed-orbit spin through one full turn
########################################
initial     = twiss.particle_on_co.copy()
returned    = initial.copy()
line.track(returned)
print('Ring entrance n0 with solenoid [x,y,z]:',
      twiss.spin_x[0], twiss.spin_y[0], twiss.spin_z[0])
print('One-turn tracked spin return residual [x,y,z]:',
      returned.spin_x[0] - initial.spin_x[0],
      returned.spin_y[0] - initial.spin_y[0],
      returned.spin_z[0] - initial.spin_z[0])
print('Orbital tunes before / after:',
      (planar.qx, planar.qy), (twiss.qx, twiss.qy))

# The solenoid changes orbital coupling as well as spin transport.
# This axis is invariant on the closed orbit; off-orbit spins can differ.

################################################################################
# Inspect periodicity and the nonvertical spin components around the ring
################################################################################

########################################
# Show native spin columns and the returned particle state
########################################
twiss.rows['ip.1|spin.solenoid|_end_point'].cols[
    's x y spin_x spin_y spin_z'].show(maxwidth = 'full')
print('Returned particle state:', returned.state[0])
print('Maximum unit-spin error:', np.max(np.abs(
    twiss.spin_x**2 + twiss.spin_y**2 + twiss.spin_z**2 - 1)))

########################################
# Resolve the small horizontal and longitudinal components
########################################
# These are element-entrance samples, not resolved spin motion inside bends.
fig, axis   = plt.subplots(figsize = (8, 4))
axis.plot(twiss.s, twiss.spin_x, '.', label = 'n0 x')
axis.plot(twiss.s, twiss.spin_z, '.', label = 'n0 z')
axis.set_xlabel('s [m]')
axis.set_ylabel('Element-entrance closed spin components')
axis.grid()
axis.legend()
fig.tight_layout()
plt.show()
