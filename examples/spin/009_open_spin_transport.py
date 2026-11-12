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
# Electron reference and one rectangular bend
########################################
ENERGY          = 5e9                  # Total reference energy [eV]
CHORD_LENGTH    = 2.0                  # Rectangular bend straight length [m]
BEND_ANGLE      = 0.1                  # Reference deflection [rad]

########################################
# Incoming orbital optics and a radial unit spin
########################################
BETX_ENTRANCE   = 1.0                  # [m]; alpha_x = 0
BETY_ENTRANCE   = 1.0                  # [m]; alpha_y = 0
SPIN_ENTRANCE   = [1.0, 0.0, 0.0]     # Local x, y, longitudinal z components

################################################################################
# Propagate specified entrance conditions through an open line
################################################################################

########################################
# Use native rectangular-bend body and edge models
########################################
line = xt.Line(elements = {
    'bend': xt.RBend(
        length_straight     = CHORD_LENGTH,
        angle               = BEND_ANGLE,
        rbend_model         = 'curved-body',
        edge_entry_model    = 'full',
        edge_exit_model     = 'full',
    ),
})
line.set_particle_ref('electron', energy0 = ENERGY)
line.configure_spin('auto')
# The named electron supplies its magnetic-moment anomaly. Radiation is off.
# The native RBend model handles the rectangular faces and curved body.

########################################
# Supply incoming orbital optics and spin instead of a periodic spin axis
########################################
entrance = xt.TwissInit(
    betx    = BETX_ENTRANCE,
    bety    = BETY_ENTRANCE,
    spin_x  = SPIN_ENTRANCE[0],
    spin_y  = SPIN_ENTRANCE[1],
    spin_z  = SPIN_ENTRANCE[2],
)
transport = line.twiss4d(init = entrance, spin = True)
transport.cols['name s x px spin_x spin_y spin_z'].show(maxwidth = 'full')
# This is initial-value spin transport. It does not search for a periodic
# invariant spin direction or assign a closed-ring spin tune.

################################################################################
# Compare Twiss propagation, direct tracking and the planar BMT angle
################################################################################

########################################
# Track the same entrance orbit and spin
########################################
tracked = transport.get_twiss_init(at_element = 'bend').particle_on_co.copy()
line.track(tracked, num_turns = 1)
spin_twiss      = np.array([
    transport.spin_x[-1], transport.spin_y[-1], transport.spin_z[-1]])
spin_tracked    = np.array([
    tracked.spin_x[0], tracked.spin_y[0], tracked.spin_z[0]])
print('Exit spin: open Twiss / direct tracking:', spin_twiss, spin_tracked)
print('Maximum spin-component difference:',
      np.max(np.abs(spin_twiss - spin_tracked)))
print('Exit spin norm:', np.linalg.norm(spin_twiss))

########################################
# Measure precession relative to the moving reference frame
########################################
anomaly         = line.particle_ref.anomalous_magnetic_moment[0]
angle_expected  = anomaly * line.particle_ref.gamma0[0] * BEND_ANGLE
angle_twiss     = np.arctan2(spin_twiss[2], spin_twiss[0])
angle_tracked   = np.arctan2(spin_tracked[2], spin_tracked[0])
print('Local exit angle: Twiss / tracking / a*gamma*theta [rad]:',
      angle_twiss, angle_tracked, angle_expected)
print('Twiss minus analytic local angle [rad]:', angle_twiss - angle_expected)
# For the radial input and this sign convention, atan2(Sz,Sx)=a*gamma*theta.
# The selected deflection keeps that angle below pi, avoiding atan2 wrapping.

################################################################################
# Distinguish the entrance and exit reference frames
################################################################################

########################################
# Transform both local spins to the common survey frame
########################################
frames          = line.survey(
    include_element_frames  = True,
).get_all_frames('bend')
spin_global_in  = (SPIN_ENTRANCE[0] * frames['ref_start'].ex
                  + SPIN_ENTRANCE[1] * frames['ref_start'].ey
                  + SPIN_ENTRANCE[2] * frames['ref_start'].es)
spin_global_out = (spin_twiss[0] * frames['ref_end'].ex
                   + spin_twiss[1] * frames['ref_end'].ey
                   + spin_twiss[2] * frames['ref_end'].es)
print('Entrance / exit longitudinal reference axes in global XYZ:',
      frames['ref_start'].es, frames['ref_end'].es)
print('Entrance / exit spins in global XYZ:', spin_global_in, spin_global_out)
print('Global exit angle atan2(SZ,SX) [rad]:',
      np.arctan2(spin_global_out[2], spin_global_out[0]))
# The spin components above use each endpoint's reference axes, not the
# rectangular pole-face axes. The global angle also includes the reference
# frame rotation; it is (1+a*gamma)*theta for this planar, on-axis example.

########################################
# Display the same outgoing spin in the two coordinate descriptions
########################################
fig, axes = plt.subplots(1, 2, figsize = (9, 4))
for axis, vector, title, labels in [
    (axes[0], spin_twiss, 'Exit local reference axes', ('Sx', 'Sz')),
    (axes[1], spin_global_out, 'Common survey axes', ('SX', 'SZ')),
]:
    axis.quiver(0.0, 0.0, vector[0], vector[2],
                 angles = 'xy', scale_units = 'xy', scale = 1.0)
    axis.set_xlim(-1.1, 1.1)
    axis.set_ylim(-1.1, 1.1)
    axis.set_aspect('equal')
    axis.set_xlabel(labels[0])
    axis.set_ylabel(labels[1])
    axis.set_title(title)
    axis.text(0.04, 0.04,
              f'Angle = {np.arctan2(vector[2], vector[0]):.6f} rad',
              transform = axis.transAxes)
    axis.grid()
fig.tight_layout()
plt.show()
