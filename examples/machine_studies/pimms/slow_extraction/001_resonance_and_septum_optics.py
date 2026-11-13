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
# Model and resonance controls
########################################
LATTICE_FILE    = 'test_data/example_lattices/pimms_extraction/lattice.py'
KSE1            = 1.0                # Extraction sextupole k2 [m^-3]
KSE2            = -6.5               # Extraction sextupole k2 [m^-3]
SEPTUM_X        = 0.035              # Observation plane threshold [m]

########################################
# On-momentum horizontal probes
########################################
N_PARTICLES = 20
MAX_X       = 0.025              # Launch displacement [m]
N_TURNS     = 768

################################################################################
# Inspect the extraction operating point
################################################################################

########################################
# Load the native 200 MeV proton model
########################################
repo_root   = Path(__file__).resolve().parents[4]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env['pimms']
tw          = line.twiss4d()
print('Tunes Qx/Qy:', tw.qx, tw.qy)
print('Horizontal distance below the third integer:', 5 / 3 - tw.qx)
print('Chromaticities:', tw.dqx, tw.dqy)
tw.rows['extr_septum'].cols['name betx alfx dx dpx'].show()

########################################
# Plot the sextupoles and the extraction-plane optics
########################################
fig, axes   = plt.subplots(2, 1, sharex = True, figsize = (9, 6))
axes[0].plot(tw.s, tw.betx, label = 'Horizontal')
axes[0].plot(tw.s, tw.bety, label = 'Vertical')
axes[0].set_ylabel(r'$\beta$ [m]')
axes[1].plot(tw.s, tw.dx)
axes[1].set(xlabel = 's [m]', ylabel = 'Horizontal dispersion [m]')
for name in ['se1', 'se2']:
    axes[0].axvline(
        tw['s', name], color = 'grey', linestyle = ':', label = name)
axes[0].legend()
for axis in axes:
    axis.grid(alpha = 0.3)
fig.tight_layout()

################################################################################
# Excite the third-integer resonance and follow its three branches
################################################################################

########################################
# Activate both extraction sextupoles
########################################
env['kse1'] = KSE1
env['kse2'] = KSE2
# The extraction sextupoles have no linear kick on the centred reference orbit.
particles = line.build_particles(x = np.linspace(0, MAX_X, N_PARTICLES))
line.track(particles, num_turns = N_TURNS, turn_by_turn_monitor = True)
record      = line.record_last_track
normalized  = tw.get_normalized_coordinates(record)

########################################
# Compare physical and Courant-Snyder phase space at the septum
########################################
# Without nemitt arguments, normalized coordinates use geometric emittance 1 m.
# No intercepting septum is installed here: the line marks an observation plane.
fig, axes = plt.subplots(1, 2, figsize = (10, 4))
for index in range(N_PARTICLES):
    visible = (record.state[index] > 0) & (np.abs(record.x[index]) < 0.06)
    axes[0].plot(record.x[index, visible] * 1e3,
                 record.px[index, visible] * 1e3, '.', markersize = 1)
    axes[1].plot(normalized.x_norm[index, visible],
                 normalized.px_norm[index, visible], '.', markersize = 1)
axes[0].axvline(SEPTUM_X * 1e3, color = 'k', linestyle = '--',
                label = 'Proposed septum boundary')
axes[0].set(xlabel = 'x [mm]', ylabel = r'$p_x$ [$10^{-3}$]',
            xlim = (-40, 45), ylim = (-4, 4))
axes[1].set(xlabel = r'$x/\sqrt{\beta_x}$ [$\sqrt{m}$]',
            ylabel = r'$(\alpha_x x+\beta_x p_x)/\sqrt{\beta_x}$ [$\sqrt{m}$]',
            xlim = (-0.015, 0.015), ylim = (-0.015, 0.015))
axes[1].set_aspect('equal')
axes[0].legend(fontsize = 'small')
for axis in axes:
    axis.grid(alpha = 0.3)
fig.tight_layout()
plt.show()
