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
LINE_NAME       = 'demo'
K1S_S2          = 1.0e-3                 # Sector-2 cell-2 skew k1 [m^-2]
K1S_S5          = -0.8e-3                # Sector-5 cell-2 skew k1 [m^-2]

################################################################################
# Compute Edwards-Teng optics for the coupled ring
################################################################################

########################################
# Load and excite the same two skew components
########################################
repo_root       = Path(__file__).resolve().parents[3]
env             = xt.load(repo_root / LATTICE_FILE)
line            = env[LINE_NAME]
env['ks_s2']    = K1S_S2
env['ks_s5']    = K1S_S5

########################################
# Request the Edwards-Teng coupling matrix
########################################
# This option supplies r11/r12/r21/r22_edw_teng for a forward periodic solution.
tw = line.twiss4d(coupling_edw_teng = True)

################################################################################
# Distinguish mode projections from decoupled Twiss parameters
################################################################################

########################################
# Inspect the two beta conventions
########################################
# betx1/bety2 are principal Mais-Ripken projections into physical x/y.
# beta_edw_teng describes the decoupled coordinate pair for each mode.
tw.rows[['ip.1', 'bpm.s4.d']].cols[
    'name betx1 betx_edw_teng bety2 bety_edw_teng g_edw_teng'].show(
        maxwidth = 'full')

########################################
# Relate them through the Edwards-Teng normalization
########################################
# In this branch, betx1 = g^2*betx_edw_teng, and likewise for bety2.
print('At the first entrance, beta_x1 [m]:', tw.betx1[0])
print('g^2 times beta_x_ET [m]:',
      tw.g_edw_teng[0]**2 * tw.betx_edw_teng[0])

################################################################################
# Build the physical-to-decoupled coordinate transformation
################################################################################

########################################
# Read the 2 by 2 coupling matrix at the first entrance
########################################
# R_ET is a coupling parameter, not the one-turn transport matrix.
# Its diagonal entries are dimensionless, R12 is in m and R21 in m^-1.
r_et = np.array([
    [tw.r11_edw_teng[0], tw.r12_edw_teng[0]],
    [tw.r21_edw_teng[0], tw.r22_edw_teng[0]]])
print('Edwards-Teng coupling matrix at', tw.name[0], ':')
print(r_et)

########################################
# Assemble the decoupling basis
########################################
# For u = (x, px, y, py), define u = T*v, with v the decoupled coordinates.
# For a 2 by 2 matrix, the symplectic conjugate is [[d, -b], [-c, a]].
r_bar       = np.array([
    [r_et[1, 1], -r_et[0, 1]], [-r_et[1, 0], r_et[0, 0]]])
g           = 1 / np.sqrt(1 + np.linalg.det(r_et))
transform   = g * np.block([
    [np.eye(2), r_bar], [-r_et, np.eye(2)]])

################################################################################
# Express one turn in the decoupled coordinates
################################################################################

########################################
# Transform the transverse one-turn matrix
########################################
# R_matrix is linearized about the closed orbit, in canonical coordinates.
# At fixed delta, its leading transverse block is a closed 4D map.
one_turn    = tw.R_matrix[:4, :4]
modal_turn  = np.linalg.solve(transform, one_turn @ transform)
print('One-turn map in (x, px, y, py):')
print(one_turn)

########################################
# Inspect the two blocks and the remaining cross terms
########################################
print('Mode 1 one-turn block:')
print(modal_turn[:2, :2])
print('Mode 2 one-turn block:')
print(modal_turn[2:, 2:])
print('Largest off-block entries after decoupling:',
      np.max(np.abs(modal_turn[:2, 2:])),
      np.max(np.abs(modal_turn[2:, :2])))
# Entries carry the coordinate-dependent units of a transfer matrix. Residual
# cross terms reflect numerical precision, not an additional physical coupling.

################################################################################
# Follow the changing Edwards-Teng basis along the ring
################################################################################

########################################
# Use the native propagated parameters at a second observation plane
########################################
end             = 'bpm.s4.d'
r_end           = np.array([
    [tw['r11_edw_teng', end], tw['r12_edw_teng', end]],
    [tw['r21_edw_teng', end], tw['r22_edw_teng', end]],
])
r_end_bar       = np.array([
    [r_end[1, 1], -r_end[0, 1]], [-r_end[1, 0], r_end[0, 0]],
])
g_end           = tw['g_edw_teng', end]
transform_end   = g_end * np.block([
    [np.eye(2), r_end_bar], [-r_end, np.eye(2)],
])
physical_map    = tw.get_R_matrix(tw.name[0], end)[:4, :4]
modal_map       = np.linalg.solve(transform_end, physical_map @ transform)
print('Between the two local decoupling bases, maximum cross blocks:',
      np.max(np.abs(modal_map[:2, 2:])),
      np.max(np.abs(modal_map[2:, :2])))
# A position-dependent basis is essential: one entrance transformation does
# not decouple the physical coordinates at every downstream location.

########################################
# Propagate the decoupled mode-1 covariance shape
########################################
beta        = tw.betx_edw_teng[0]
alpha       = tw.alfx_edw_teng[0]
sigma_shape = np.array([[beta, -alpha], [-alpha, (1 + alpha**2) / beta]])
transported = modal_map[:2, :2] @ sigma_shape @ modal_map[:2, :2].T
print('Transported/native downstream beta_ET [m]:',
      transported[0, 0], tw['betx_edw_teng', end])
print('Transported/native downstream alpha_ET:',
      -transported[0, 1], tw['alfx_edw_teng', end])
# This is a consistency relation among native Twiss/maps, not an independent
# field-integration or external-code validation.

################################################################################
# Inspect the native element-by-element Edwards-Teng functions
################################################################################
fig, axes = plt.subplots(2, 1, sharex = True, figsize = (9, 6),
                         layout = 'constrained')
axes[0].plot(tw.s, tw.betx_edw_teng, label = 'Mode 1')
axes[0].plot(tw.s, tw.bety_edw_teng, label = 'Mode 2')
axes[0].set_ylabel('Decoupled beta [m]')
axes[1].plot(tw.s, tw.r11_edw_teng, label = 'R_ET,11')
axes[1].plot(tw.s, tw.r22_edw_teng, label = 'R_ET,22')
axes[1].set(xlabel = 's [m]', ylabel = 'Dimensionless coupling entries')
for axis in axes:
    axis.grid()
    axis.legend()
plt.show()
