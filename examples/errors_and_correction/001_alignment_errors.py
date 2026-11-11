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
# Ring and alignment errors
########################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
SEED            = 31
SHIFT_RMS       = 5e-6                  # Independent x/y offsets [m]
ROLL_RMS        = 5e-4                  # Roll about the reference s axis [rad]

########################################
# Survey-frame illustration
########################################
FRAME_ELEMENT       = 'arc.s1.c1.qd'
FRAME_ARROW_LENGTH  = 10.0              # Drawing length [µm]
DISPLAY_ROLL_FACTOR = 1000.0            # Drawing only; offsets stay unscaled

################################################################################
# Load the aligned ring and select arc quadrupoles
################################################################################
repo_root   = Path(__file__).resolve().parents[2]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env['demo']
nominal     = line.twiss4d()
table       = line.get_table()
quad_names  = table.rows[r'arc\.s[1-6]\.c[1-4]\.qd'].name
rng         = np.random.default_rng(SEED)

# The reference is a 1 GeV/c electron; RF voltage and radiation are zero.
# Only the arc defocusing quadrupoles receive errors in this independent load.

################################################################################
# Displace the magnetic centres, then roll the magnets
################################################################################

########################################
# Apply fixed, reproducible transverse offsets
########################################
offsets = rng.normal(scale = SHIFT_RMS, size = (len(quad_names), 2))
for name, (dx, dy) in zip(quad_names, offsets):
    line[name].shift_x  = dx
    line[name].shift_y  = dy
shifted = line.twiss4d()

# Positive shift_x moves the magnet centre towards positive line x. The local
# magnetic coordinate is x - shift_x. An offset quadrupole therefore steers.

########################################
# Add independent rolls without changing the offsets or body strengths
########################################
rolls       = rng.normal(scale = ROLL_RMS, size = len(quad_names))
for name, roll in zip(quad_names, rolls):
    line[name].rot_s_rad    = roll
misaligned  = line.twiss4d()

# rot_s_rad rotates the magnet's transverse frame. A rolled normal quadrupole
# couples the two transverse planes; it does not change the reference geometry.

################################################################################
# Inspect the installed errors and their optical consequences
################################################################################

########################################
# Inspect attributes in the line frame and the coupled mode projections
########################################
line.get_table(attr = True).rows[quad_names].cols[
    'name s shift_x shift_y rot_s_rad'].show(maxwidth = 'full')
for label, tw in [('Nominal', nominal), ('Shifted', shifted),
                  ('Shifted and rolled', misaligned)]:
    print(label, 'sampled orbit RMS x/y [m]:',
          np.sqrt(np.mean(tw.x**2)), np.sqrt(np.mean(tw.y**2)))
    tw.rows[['bpm.s1.f', 'bpm.s4.d']].cols[
        'name betx1 betx2 bety1 bety2'].show(maxwidth = 'full')

# betx2 and bety1 show the cross-plane projections of the two normal modes.
# Mode optics and coupling interpretation are developed in optics/transverse.

########################################
# Plot the resulting closed orbit relative to the aligned reference
########################################
fig, axes   = plt.subplots(2, 1, sharex = True, figsize = (9, 6))
for axis, coordinate in zip(axes, ['x', 'y']):
    axis.plot(
        misaligned.s,
        1e3 * (misaligned[coordinate] - nominal[coordinate]),
    )
    axis.set_ylabel(coordinate + ' orbit change [mm]')
    axis.grid()
axes[-1].set_xlabel('s [m]')
fig.tight_layout()

################################################################################
# Locate a displaced magnet and inspect its surveyed frames
################################################################################

########################################
# Express the actual entrance frame in the reference entrance frame
########################################
survey      = line.survey(include_element_frames = True)
frames      = survey.get_all_frames(FRAME_ELEMENT)
reference   = frames['ref_start']
actual      = frames['elem_start']
relative    = reference.inverse() @ actual
offset_um   = 1e6 * relative.XYZ
frame_roll  = np.arctan2(relative.ex[1], relative.ex[0])
print('Surveyed element:', FRAME_ELEMENT)
print('Entrance offset in reference (x, y, s) [µm]:', offset_um)
print('Entrance-frame roll [mrad]:', 1e3 * frame_roll)
# The reference survey stays fixed. Element frames include the actual offsets
# and rotations; these micrometre shifts are invisible on the full-ring plot.

########################################
# Show the ring location and a magnified view of the transverse axes
########################################
frame_fig, frame_axes       = plt.subplots(1, 2, figsize = (11, 5))
frame_axes[0].plot(survey.Z, survey.X, color = 'grey')
frame_axes[0].plot(reference.Z, reference.X, 'o', label = FRAME_ELEMENT)
frame_axes[0].set_xlabel('Global Z [m]')
frame_axes[0].set_ylabel('Global X [m]')
frame_axes[0].set_title('Reference ring: element-boundary polyline')

# Only the drawn roll is enlarged. Frame origins use the true surveyed shifts,
# and the terminal output above reports the true angle.
display_frame   = xt.Frame().rotate_s(DISPLAY_ROLL_FACTOR * frame_roll)
for origin, axes_frame, label, color in [
    (np.zeros(2), xt.Frame(), 'Reference entrance', 'C0'),
    (offset_um[:2], display_frame, 'Actual entrance', 'C1'),
]:
    frame_axes[1].plot(*origin, 'o', color = color, label = label)
    for axis_name, direction in [('x', axes_frame.ex), ('y', axes_frame.ey)]:
        arrow   = FRAME_ARROW_LENGTH * direction[:2]
        frame_axes[1].annotate(
            '',
            xy          = origin + arrow,
            xytext      = origin,
            arrowprops  = {'arrowstyle': '->', 'color': color},
        )
        frame_axes[1].text(*(origin + arrow), axis_name, color = color)
# Set limits from the origins and arrow length, including annotation margins.
margin      = 1.4 * FRAME_ARROW_LENGTH
frame_axes[1].set_xlim(
    min(0, offset_um[0]) - margin, max(0, offset_um[0]) + margin)
frame_axes[1].set_ylim(
    min(0, offset_um[1]) - margin, max(0, offset_um[1]) + margin)
frame_axes[1].set_xlabel('Reference entrance x [µm]')
frame_axes[1].set_ylabel('Reference entrance y [µm]')
frame_axes[1].set_title(f'Entrance frames: roll ×{DISPLAY_ROLL_FACTOR:g}')
for axis in frame_axes:
    axis.set_aspect('equal', adjustable = 'box')
    axis.grid()
    axis.legend()
frame_fig.tight_layout()
plt.show()
