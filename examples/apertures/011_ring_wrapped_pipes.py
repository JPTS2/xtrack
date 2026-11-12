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
# A straight taper spanning the shared ring's s=0 boundary
########################################
LATTICE_FILE        = 'test_data/example_lattices/demo_ring/lattice.py'
PIPE_LENGTH         = 0.4                   # Total chamber length [m]
RADIUS_BEFORE       = 0.020                 # Upstream profile radius [m]
RADIUS_AFTER        = 0.012                 # Downstream profile radius [m]
QUERY_OFFSET        = 0.1                   # Distance before/after the seam [m]
PROFILE_POINTS      = 128

################################################################################
# Place one chamber across the boundary of a closed survey
################################################################################

########################################
# Use the straight reference on both sides of the ring seam
########################################
repo_root       = Path(__file__).resolve().parents[2]
env             = xt.load(repo_root / LATTICE_FILE)
line            = env['demo']
circumference   = line.get_length()
upstream        = env.aperture.new_profile(
    'upstream', 'Circle', radius = RADIUS_BEFORE)
downstream      = env.aperture.new_profile(
    'downstream', 'Circle', radius = RADIUS_AFTER)
pipe            = env.aperture.new_pipe(
    'seam_taper',
    positions = [upstream.place(at = 0), downstream.place(at = PIPE_LENGTH)],
)
line.aperture.place_pipe(
    'seam.chamber', pipe,
    at      = 'straight.s6.end',
    shift_z = -PIPE_LENGTH / 2,
)
line.aperture.num_profile_points = PROFILE_POINTS
line.aperture.build_model()
print('Closed survey recognized as a ring:', line.aperture.is_ring)
line.aperture.get_bounds_table().cols[
    'name s s_start s_end'].show(maxwidth = 'full')
# Pipe coordinates increase continuously across the placement. Installed
# survey coordinates wrap to [0,C). The chamber remains one straight taper;
# there is no bend or physical end cap at the numerical seam.

########################################
# Split a longitudinal inspection window at the native ring boundary
########################################
print('Native wrapped interval [m]:', line.aperture.get_wrapped_s_interval(
    circumference - PIPE_LENGTH / 2, PIPE_LENGTH / 2))

################################################################################
# Query equivalent coordinates on either side of s=0
################################################################################
requested_s = np.array([-QUERY_OFFSET, 0.0, circumference + QUERY_OFFSET])
labels      = np.array(['Before seam', 'At seam', 'After seam'])
query_s     = np.mod(requested_s, circumference)
order       = np.argsort(query_s)
# Query along increasing canonical survey s; retain the requested positions
# and their labels separately, including the negative and next-turn requests.
sections    = line.aperture.get_cross_sections_at_s(
    s_positions = query_s[order],
    extents     = True,
)
xt.Table(
    data    = {
        'station':      labels[order],
        'requested_s':  requested_s[order],
        'survey_s':     sections.s,
        'min_x':        sections.min_x,
        'max_x':        sections.max_x,
    },
    index   = 'station',
).show(maxwidth = 'full')
# Normalization is explicit: these queries all lie inside the chamber. The
# aperture model supplies the neighbouring stations across the ring boundary.
# This does not define chamber geometry around the rest of the ring.

################################################################################
# Inspect the continuous taper and its three local circular sections
################################################################################

########################################
# Draw the native placement in a frame centred before the seam
########################################
fig, axes   = plt.subplots(1, 2, figsize = (11, 4))
line.aperture.plot_floor_projection(
    ax      = axes[0],
    origin  = 'seam.chamber',
    s_range = (0, PIPE_LENGTH),
    aspect  = 'auto',
)
axes[0].set_title('One chamber across the numerical seam')

########################################
# Compare before-seam, seam and after-seam cuts
########################################
for label, polygon in zip(labels[order], sections.cross_section):
    axes[1].plot(polygon[:, 0] * 1e3, polygon[:, 1] * 1e3, label = label)
axes[1].set_xlabel('Local survey x [mm]')
axes[1].set_ylabel('Local survey y [mm]')
axes[1].set_aspect('equal')
axes[1].legend()
axes[1].grid()
fig.tight_layout()
plt.show()
