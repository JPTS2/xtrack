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
# Electron reference
########################################
P0C                 = 10e9              # Reference momentum [eV/c]

########################################
# Sector-bend scan
########################################
BEND_LENGTH         = 1.0               # [m]
BEND_ANGLES         = np.linspace(0, 0.08, 9)  # [rad]

################################################################################
# Enable native spin transport through a sector bend
################################################################################

########################################
# Prepare the named electron reference and an initially radial spin
########################################
line    = xt.Line(elements = {
    'bend': xt.Bend(
        length      = BEND_LENGTH,
        k0          = 'from_h',
        model       = 'bend-kick-bend',
    ),
})
line.set_particle_ref('electron', p0c = P0C)
line.configure_spin('auto')
initial = line.build_particles(spin_x = 1.0)
# Spin components use local horizontal x, vertical y and longitudinal z axes.
# Named species supply their tabulated magnetic-moment anomaly.
# k0 follows angle/length, keeping the reference particle on the design orbit.

########################################
# Change bend angle and track the same on-axis reference particle each time
########################################
spins   = []
for angle in BEND_ANGLES:
    line['bend'].angle   = float(angle)
    particles           = initial.copy()
    line.track(particles, num_turns = 1)
    spins.append([
        particles.spin_x[0], particles.spin_y[0], particles.spin_z[0]])
spins   = np.asarray(spins)

################################################################################
# Compare native precession with the planar Thomas–BMT prediction
################################################################################

########################################
# Measure rotation relative to the moving reference frame
########################################
anomaly         = initial.anomalous_magnetic_moment[0]
angle_tracked   = np.arctan2(spins[:, 2], spins[:, 0])
angle_expected  = anomaly * initial.gamma0[0] * BEND_ANGLES
spin_norm       = np.linalg.norm(spins, axis = 1)
xt.Table(
    data    = {
        'bend_angle':       BEND_ANGLES,
        'spin_x':           spins[:, 0],
        'spin_y':           spins[:, 1],
        'spin_z':           spins[:, 2],
        'spin_norm':        spin_norm,
        'spin_angle':       angle_tracked,
        'bmt_angle':        angle_expected},
    index   = 'bend_angle').show(maxwidth = 'full')
# For this sign convention, atan2(Sz,Sx)=a*gamma*theta. The laboratory-frame
# angle also includes the reference-frame rotation. Radiation is off.
# The chosen scan stays below pi, so atan2 gives an unambiguous angle.

########################################
# Resolve the precession angle over the selected bend-angle range
########################################
fig, ax     = plt.subplots(figsize = (8, 4))
ax.plot(BEND_ANGLES, angle_tracked, 'o', label = 'Native spin tracking')
ax.plot(BEND_ANGLES, angle_expected, '--', label = 'a gamma theta')
ax.set_xlabel('Bend angle [rad]')
ax.set_ylabel('Spin angle atan2(Sz, Sx) [rad]')
ax.grid()
ax.legend()
fig.tight_layout()
plt.show()
