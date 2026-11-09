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
from scipy.constants import c
import xtrack as xt

################################################################################
# User parameters
################################################################################
MODEL_DIR   = 'test_data/example_lattices/demo_ring'  # Relative to repo root

################################################################################
# Load geometry and an independently maintained operating point
################################################################################

########################################
# Define hardware, then apply nominal quadrupole strengths
########################################
repo_root   = Path(__file__).resolve().parents[2]
model_dir   = repo_root / MODEL_DIR
env = xt.Environment()
env.call(model_dir / 'geometry.py')
env.call(model_dir / 'strengths.py')

########################################
# Inspect reusable cells, half-straights and named full sections
########################################
env.lines['arc.cell'].get_table().cols['name s element_type'].show(
    maxwidth = 'full')
for name in ['arc.cell', 'arc.s1', 'ff.half', 'utility.half', 'insertion']:
    print(name, 'length [m]:', env.lines[name].get_length())

################################################################################
# Assemble the ring and set its reference clock
################################################################################

########################################
# Order the six arcs and their intervening straight sections
########################################
demo    = env.new_line(name = 'demo', components = [
    'arc.s1', 'insertion',
    'arc.s2', 'utility.s2',
    'arc.s3', 'utility.s3',
    'arc.s4', 'utility.s4',
    'arc.s5', 'utility.s5',
    'arc.s6', 'utility.s6',
])

########################################
# Set the reference species, RF clock and ideal steering selections
########################################
demo.set_particle_ref('electron', p0c = env['p0c'])
env['frequency_rf'] = (
    env['harmonic_rf'] * demo.particle_ref.beta0[0] * c / demo.get_length())
table       = demo.get_table()
bpms        = list(table.rows['bpm[.].*'].name)
correctors  = list(table.rows['cor[.].*'].name)
demo.steering_monitors_x = bpms
demo.steering_monitors_y = bpms
demo.steering_correctors_x = correctors
demo.steering_correctors_y = correctors
# RF voltage remains zero; radiation is off. No 6D RF confinement is assumed.

################################################################################
# Inspect the reference geometry and the selected optics
################################################################################

########################################
# Check reference-path closure independently of particle tracking
########################################
survey          = demo.survey()
position_error  = np.linalg.norm(survey.XYZ[-1] - survey.XYZ[0])
frame_error     = np.max(np.abs(survey.E_matrix[-1] - survey.E_matrix[0]))
print('Circumference [m]:', demo.get_length())
print('Survey closure [m], frame residual:', position_error, frame_error)

########################################
# Inspect integrated strengths and the low-beta interaction point
########################################
print('Integrated quadrupole strengths k1l [m^-1]:')
demo.get_table(attr = True).rows[
    ['arc.s1.c1.qf.in', 'arc.s1.c1.qd', 'straight.s1.r.q0', 'rf.1']
].cols['name s element_type k1l'].show(maxwidth = 'full')
tw  = demo.twiss4d()
print('Full tunes:', tw.qx, tw.qy)
tw.rows[['ip.1']].cols['name s betx bety alfx alfy dx dpx'].show(
    maxwidth    = 'full')

# Arcs span beta roughly 12--65 m; the low-beta insertion peaks near 535 m.
# Two-cell sextupole pairs have on-momentum geometric -I maps, but unmatched
# dispersion. Equal-D chromatic cancellation and crab-waist optics are not set.

################################################################################
# Plot the geometry and magnetic optics
################################################################################

########################################
# Connect surveyed element boundaries, drawing bends as chords
########################################
fig, axes = plt.subplots(1, 2, figsize = (11, 5))
axes[0].plot(survey.Z, survey.X, '.-', markersize = 2)
axes[0].set_xlabel('Z [m]')
axes[0].set_ylabel('X [m]')
axes[0].set_aspect('equal', adjustable = 'datalim')
axes[0].set_title('Element-boundary polyline')

########################################
# Show the insertion beta peak together with arc optics
########################################
axes[1].plot(tw.s, tw.betx, label = 'betx')
axes[1].plot(tw.s, tw.bety, label = 'bety')
axes[1].set_xlabel('s [m]')
axes[1].set_ylabel('Beta [m]')
axes[1].legend()
fig.tight_layout()
plt.show()
