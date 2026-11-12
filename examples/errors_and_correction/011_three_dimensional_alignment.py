# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# One curved reference element and three entrance probes
########################################
P0C             = 1e9                  # Electron reference p0*c [eV]
BEND_LENGTH     = 2.0                  # Reference arc length [m]
BEND_ANGLE      = 0.05                 # Horizontal reference deflection [rad]
DELTA_VALUES    = np.array([-1e-4, 0.0, 1e-4])  # Include the delta=0 probe

########################################
# Translations and rotations at an anchor on the reference arc
########################################
ANCHOR_S        = 0.75                 # Arc distance from reference entry [m]
SHIFT_X         = 50e-6                # Anchor-local horizontal shift [m]
SHIFT_Y         = -30e-6               # Anchor-local vertical shift [m]
SHIFT_S         = 100e-6               # Anchor-local longitudinal shift [m]
PITCH           = 20e-6                # rot_x_rad [rad]
YAW             = -40e-6               # rot_y_rad [rad]
ROLL            = 60e-6                # Pure alignment roll [rad]

################################################################################
# Compare the same particle launches before and after 3D alignment errors
################################################################################

########################################
# Isolate the body map without dipole-edge focusing
########################################
line = xt.Line(elements = {
    'bend': xt.Bend(
        length              = BEND_LENGTH,
        angle               = BEND_ANGLE,
        k0                  = 'from_h',
        model               = 'bend-kick-bend',
        edge_entry_model    = 'suppressed',
        edge_exit_model     = 'suppressed',
    ),
})
line.set_particle_ref('electron', p0c = P0C)
line.skip_end_turn_actions = True
# This is one open passage, so skip ring turn-boundary resets and clock shifts.
# Retained s is the design coordinate, not the actual travelled distance.
initial         = line.build_particles(delta = DELTA_VALUES)
nominal         = initial.copy()
line.track(nominal, num_turns = 1)
nominal_survey  = line.survey(include_element_frames = True)

########################################
# Apply all six alignment coordinates through public element attributes
########################################
line['bend'].rot_shift_anchor       = ANCHOR_S
line['bend'].shift_x                = SHIFT_X
line['bend'].shift_y                = SHIFT_Y
line['bend'].shift_s                = SHIFT_S
line['bend'].rot_x_rad              = PITCH
line['bend'].rot_y_rad              = YAW
line['bend'].rot_s_rad_no_frame     = ROLL
# rot_s_rad_no_frame is a pure alignment roll. The geometric tilt rot_s_rad
# remains zero, keeping the horizontal reference arc and its end frames fixed.
# At the reference anchor, positive pitch turns the longitudinal axis toward
# +y, positive yaw toward +x, and positive roll turns the x axis toward +y.
# The anchor is a distance in metres along that arc, not a length fraction.
misaligned = initial.copy()
line.track(misaligned, num_turns = 1)
print('Nominal exit coordinates:')
nominal.get_table().cols['particle_id x px y py zeta delta s state'].show(
    maxwidth = 'full')
print('Misaligned exit coordinates in the same reference frame:')
misaligned.get_table().cols['particle_id x px y py zeta delta s state'].show(
    maxwidth = 'full')
print('Exit s: nominal / misaligned [m]:', nominal.s, misaligned.s)
print('Misaligned minus nominal zeta [m]:', misaligned.zeta - nominal.zeta)
print('Corresponding arrival-time difference at the same s [s]:',
      -(misaligned.zeta - nominal.zeta) / (initial.beta0[0] * c))
# The entrance and exit alignment maps include changes in longitudinal
# coordinates. shift_s is not an extra straight drift appended to the line.

################################################################################
# Locate the translated anchor using native reference and element frames
################################################################################
survey              = line.survey(include_element_frames = True)
frames              = survey.get_all_frames('bend')
reference_anchor    = frames['ref_start'].copy().arc(
    length  = ANCHOR_S,
    angle   = BEND_ANGLE * ANCHOR_S / BEND_LENGTH,
)
element_anchor      = frames['elem_start'].copy().arc(
    length  = ANCHOR_S,
    angle   = BEND_ANGLE * ANCHOR_S / BEND_LENGTH,
)
print('Reference / actual element frame origins in global XYZ [m]:')
for name, frame in frames.items():
    print(name, frame.XYZ)
anchor_displacement = element_anchor.XYZ - reference_anchor.XYZ
print('Anchor displacement in its reference-local x/y/s axes [m]:',
      np.dot(anchor_displacement, reference_anchor.ex),
      np.dot(anchor_displacement, reference_anchor.ey),
      np.dot(anchor_displacement, reference_anchor.es))
print('Reference survey endpoint change XYZ [m]:',
      survey.XYZ[-1] - nominal_survey.XYZ[-1])
# Native Frame.arc follows the curvature to the anchor. A straight-line
# extrapolation of the entrance tangent would locate a different point.

################################################################################
# Backtrack the same finite 6D map
################################################################################
returned    = misaligned.copy()
line.track(returned, num_turns = 1, backtrack = True)
print('Maximum forward/backward residual for each coordinate:')
for coordinate in ['x', 'px', 'y', 'py', 'zeta', 'delta', 's']:
    print(coordinate, np.max(np.abs(
        getattr(returned, coordinate) - getattr(initial, coordinate))))
# This checks the configured native map's reversibility. It is not an
# independent survey or cross-code validation of arbitrary large rotations.

################################################################################
# Relate the element displacement to its tracked effect
################################################################################

########################################
# Resolve the actual centreline offset along the reference arc
########################################
arc_samples         = np.linspace(0.0, BEND_LENGTH, 41)
centreline_offsets  = []
for position in arc_samples:
    reference_frame = frames['ref_start'].copy().arc(
        length  = position,
        angle   = BEND_ANGLE * position / BEND_LENGTH,
    )
    element_frame   = frames['elem_start'].copy().arc(
        length  = position,
        angle   = BEND_ANGLE * position / BEND_LENGTH,
    )
    displacement    = element_frame.XYZ - reference_frame.XYZ
    centreline_offsets.append([
        np.dot(displacement, reference_frame.ex),
        np.dot(displacement, reference_frame.ey),
        np.dot(displacement, reference_frame.es),
    ])
centreline_offsets  = np.array(centreline_offsets)
fig, axes           = plt.subplots(1, 2, figsize = (11, 4))
for index, label in enumerate(['Local x', 'Local y', 'Local s']):
    axes[0].plot(arc_samples, centreline_offsets[:, index] * 1e6, label = label)
axes[0].axvline(ANCHOR_S, color = '0.5', linestyle = '--', label = 'Anchor')
axes[0].set_xlabel('Reference arc coordinate [m]')
axes[0].set_ylabel('Actual centreline displacement [um]')

########################################
# Compare output changes at the unchanged reference exit plane
########################################
zero_probe  = np.flatnonzero(DELTA_VALUES == 0.0)[0]
coordinates = ['x', 'y', 'zeta']
changes     = [getattr(misaligned, name)[zero_probe]
               - getattr(nominal, name)[zero_probe] for name in coordinates]
axes[1].bar(coordinates, np.array(changes) * 1e6)
axes[1].axhline(0.0, color = '0.5', linewidth = 0.8)
axes[1].set_title('On-reference-momentum probe')
axes[1].set_ylabel('Misaligned minus nominal exit coordinate [um]')
axes[0].legend()
for axis in axes:
    axis.grid(axis = 'y')
fig.tight_layout()
plt.show()
