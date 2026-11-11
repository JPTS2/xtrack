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
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
ELEMENT         = 'arc.s1.c1.qd'
HALF_X          = 0.02                  # Elliptical chamber semiaxis [m]
HALF_Y          = 0.01                  # Elliptical chamber semiaxis [m]
PIPE_SHIFT_X    = 0.002                 # Chamber translation [m]
OUTPUT_DIR      = Path(__file__).resolve().parent / 'outputs'

################################################################################
# Add a named chamber to the shared ring
################################################################################
repo_root   = Path(__file__).resolve().parents[2]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env['demo']
profile     = env.aperture.new_profile(
    'ellipse', 'Ellipse',
    a   = HALF_X,
    b   = HALF_Y,
)
pipe        = env.aperture.new_pipe('quad.chamber', positions = [
    profile.place(),
    profile.place(at = line[ELEMENT].length),
])
line.aperture.place_pipe(
    'installed.chamber', pipe,
    at      = ELEMENT,
    shift_x = PIPE_SHIFT_X,
)
line.aperture.build_model()
centre_s    = line.get_table()['s', ELEMENT] + line[ELEMENT].length / 2
original    = line.aperture.get_cross_sections_at_s([centre_s])

################################################################################
# Save either the complete model or the aperture data separately
################################################################################

########################################
# Explicitly include aperture definitions and placements with the line
########################################
OUTPUT_DIR.mkdir(exist_ok = True)
line_file   = OUTPUT_DIR / 'demo_with_apertures.json'
line.to_json(line_file, include_aperture = True)
restored    = xt.load(line_file)
restored.aperture.build_model()
# Ordinary line/environment JSON excludes the aperture model unless requested.
# Reloading restores definitions and placements; build_model prepares geometry.

########################################
# Carry the aperture model onto another copy of the same named lattice
########################################
aperture_file   = OUTPUT_DIR / 'demo_apertures.json'
env.aperture.to_json(aperture_file)
target_env      = xt.load(repo_root / LATTICE_FILE)
target_env.aperture.from_json(aperture_file)
target          = target_env['demo']
target.aperture.build_model()
# An aperture-only export includes shared profiles, pipes and named lines'
# placements. Corresponding line names and placement references must exist.
# The target aperture container starts empty; existing geometry is not erased.

################################################################################
# Inspect the restored definitions and physical cross-sections
################################################################################

########################################
# Show names and installed geometry after both reload routes
########################################
print('Complete-line file:', line_file)
print('Aperture-only file:', aperture_file)
print('Restored profiles:', restored.env.aperture.profiles.keys())
print('Restored pipes:', restored.env.aperture.pipes.keys())
print('Restored placements:', restored.aperture.pipe_positions.keys())
restored.aperture.get_extents_at_s([centre_s]).show(maxwidth = 'full')

########################################
# Compare the same section from both native persistence paths
########################################
fig, ax     = plt.subplots(figsize = (7, 5))
for table, label, style in [
    (original, 'Before saving', '-'),
    (restored.aperture.get_cross_sections_at_s([centre_s]),
     'Complete line', '--'),
    (target.aperture.get_cross_sections_at_s([centre_s]), 'Aperture-only', ':'),
]:
    points  = table.cross_section[0]
    closed  = np.vstack([points, points[0]])
    ax.plot(1e3 * closed[:, 0], 1e3 * closed[:, 1], style, label = label)
ax.set_xlabel('Local reference x [mm]')
ax.set_ylabel('Local reference y [mm]')
ax.set_title('Identical chamber geometry through both reload routes')
ax.set_aspect('equal')
ax.grid()
ax.legend()
fig.tight_layout()
plt.show()
