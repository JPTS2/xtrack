# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
from pathlib import Path

import numpy as np
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Model and boundaries
########################################
LATTICE_FILE    = 'test_data/example_lattices/transport/lattice.py'
LINE_NAME       = 'transport'
# This example starts at the line entrance. Choose later boundaries in order.
MIDDLE  = 'qd.1'
STOP    = 'qf.2'

########################################
# Initial optics and small transverse displacement
########################################
BETX            = 1.0                   # [m]
BETY            = 1.0                   # [m]
DISPLACEMENT    = np.array([1e-6, 2e-7, -2e-6, 1e-7])
# Coordinates are x [m], px [1], y [m], py [1], at delta = 0.

########################################
# Periodic 6D comparison
########################################
RING_FILE   = 'test_data/example_lattices/demo_ring/lattice.py'
RF_VOLTAGE  = 1e5                    # [V]
RING_MIDDLE = 'bpm.s4.f'

################################################################################
# Load the line and calculate open 4D optics
################################################################################
repo_root   = Path(__file__).resolve().parents[3]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env[LINE_NAME]
start       = line.element_names[0]
tw          = line.twiss(method = '4d', betx = BETX, bety = BETY)

################################################################################
# Compose the transverse maps between element entrances
################################################################################

########################################
# Obtain the three maps
########################################
# The full canonical basis is (x, px, y, py, zeta, pzeta). Here use its
# transverse block at fixed momentum, with column vectors (x, px, y, py).
r_start_stop    = tw.get_R_matrix(start = start, end = STOP)[:4, :4]
r_start_middle  = tw.get_R_matrix(start = start, end = MIDDLE)[:4, :4]
r_middle_stop   = tw.get_R_matrix(start = MIDDLE, end = STOP)[:4, :4]

########################################
# Check the order of multiplication
########################################
# Column vectors propagate as u_stop = R_middle_stop @ R_start_middle @ u_start.
r_composed  = r_middle_stop @ r_start_middle
print('Largest matrix-composition residual:',
      np.max(np.abs(r_composed - r_start_stop)))
print('Direct transverse matrix:')
with np.printoptions(precision = 6, suppress = True):
    print(r_start_stop)

########################################
# Inspect maps from the common entrance to several locations
########################################
maps = tw.get_R_matrix_table()
# Every row maps from the first Twiss row to its sampled plane: an element
# entrance, or the final exit for _end_point.
# r12 is in metres and r21 in m^-1; r11 and r22 are dimensionless.
maps.rows[[start, MIDDLE, STOP]].cols[
    'name s r11 r12 r21 r22'].show(maxwidth = 'full')

################################################################################
# Compute the same map directly by finite-difference tracking
################################################################################
# line.get_R_matrix returns a dictionary and linearizes about the supplied
# trajectory. Its stop boundary, like the Twiss-table map, is an entrance.
finite_difference   = line.get_R_matrix(
    particle_on_co  = tw.particle_on_co,
    start           = start,
    end             = STOP)
r_from_tracking     = finite_difference['R_matrix'][:4, :4]
print('Largest Twiss/tracking matrix difference:',
      np.max(np.abs(r_start_stop - r_from_tracking)))

# Matrix entries have coordinate-dependent units; this is not a physical norm.

################################################################################
# Compare the linear prediction with particle tracking
################################################################################

########################################
# Launch a reference trajectory and one displaced particle
########################################
# This transport's reference trajectory starts at zero transverse coordinates.
# Track both to subtract the reference and compare displacements, not orbits.
particles = line.build_particles(
    x           = [0.0, DISPLACEMENT[0]],
    px          = [0.0, DISPLACEMENT[1]],
    y           = [0.0, DISPLACEMENT[2]],
    py          = [0.0, DISPLACEMENT[3]],
    delta       = 0.0,
)
line.track(particles, ele_stop = STOP)

########################################
# Inspect the residual of the linear approximation
########################################
coordinates = ['x', 'px', 'y', 'py']
tracked     = np.array([getattr(particles, nn)[1] - getattr(particles, nn)[0]
                        for nn in coordinates])
predicted   = r_start_stop @ DISPLACEMENT
result      = xt.Table(
    data    = {
        'coordinate': np.array(coordinates),
        'unit':       np.array(['m', '1', 'm', '1']),
        'tracked':    tracked,
        'linear':     predicted,
        'difference': tracked - predicted,
    },
    index   = 'coordinate',
)

# Differences include nonlinear and numerical transport terms. Reduce the
# displacement to examine the linear regime. Both particles stop before STOP,
# at its entrance.
result.show(maxwidth = 'full')
print('Particle stopping indices:', particles.at_element)

################################################################################
# Compose full 6D maps with RF focusing
################################################################################

########################################
# Solve three periodic modes in an independent ring
########################################
# The shared ring has a 1 GeV/c electron reference and no radiation.
ring_env = xt.load(repo_root / RING_FILE)
ring = ring_env['demo']
ring_env['voltage_rf'] = RF_VOLTAGE
tw6 = ring.twiss6d()
ring_start = ring.element_names[0]
r6_direct = tw6.get_R_matrix(start = ring_start, end = '_end_point')
r6_first = tw6.get_R_matrix(start = ring_start, end = RING_MIDDLE)
r6_second = tw6.get_R_matrix(start = RING_MIDDLE, end = '_end_point')
print('6D matrix composition residual:',
      np.max(np.abs(r6_second @ r6_first - r6_direct)))
print('One-turn reconstruction versus tracked linearization:',
      np.max(np.abs(r6_direct - tw6.R_matrix)))

########################################
# Transport mode shapes without comparing arbitrary phase origins
########################################
# W uses (x, px, y, py, zeta, pzeta), with three pairs of mode columns.
# Propagated columns acquire phase rotations. Their pairwise outer products
# remove that phase, so compare mode shapes instead of raw W eigenvectors.
# These are unit-action shapes, not an emittance-weighted beam covariance.
w_start     = tw6.W_matrix[0]
w_end       = tw6.W_matrix[-1]
mode_shapes = []
for mode in range(3):
    propagated          = r6_direct @ w_start[:, 2 * mode:2 * mode + 2]
    target              = w_end[:, 2 * mode:2 * mode + 2]
    shape_propagated    = propagated @ propagated.T
    shape_target        = target @ target.T
    mode_shapes.append((shape_propagated, shape_target))
    print('Mode', mode + 1, 'maximum shape-entry residual:',
          np.max(np.abs(shape_propagated - shape_target)))
# Shape entries have coordinate-product units; this maximum is a numerical
# comparison in the stated coordinate convention, not a physical norm.
