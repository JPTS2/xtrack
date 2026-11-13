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
STRENGTH_SCALES = np.linspace(0.05, 2.0, 80)

################################################################################
# Use the reusable arc cell as an optical period
################################################################################
repo_root           = Path(__file__).resolve().parents[3]
env                 = xt.load(repo_root / LATTICE_FILE)
cell                = env['arc.cell']
cell.particle_ref   = env['demo'].particle_ref.copy()
kqf_nominal         = env['kqf']
kqd_nominal         = env['kqd']
# The cell has no skew field, RF or radiation. It bends through 15 degrees,
# but periodic transverse optics does not require floor-plan closure.

################################################################################
# Track linear maps even where periodic Twiss is unstable
################################################################################

########################################
# Read the two transverse traces
########################################
half_traces = []
for scale in STRENGTH_SCALES:
    env['kqf']  = scale * kqf_nominal
    env['kqd']  = scale * kqd_nominal
    matrix      = cell.get_R_matrix(
        particle_on_co = cell.particle_ref.copy())['R_matrix']
    half_traces.append([np.trace(matrix[:2, :2]) / 2,
                        np.trace(matrix[2:4, 2:4]) / 2])
half_traces = np.array(half_traces)
stable      = np.all(np.abs(half_traces) < 1, axis = 1)
# Each uncoupled, symplectic 2 by 2 block is elliptic when |Tr(M)/2|<1.
# At +/-1 the limit is marginal; this is not nonlinear dynamic aperture.
print('Scanned scale range with both planes linearly stable:',
      STRENGTH_SCALES[stable].min(), STRENGTH_SCALES[stable].max())

########################################
# Compare with native Twiss at the nominal setting
########################################
env['kqf']  = kqf_nominal
env['kqd']  = kqd_nominal
tw          = cell.twiss4d()
nominal_map = cell.get_R_matrix(
    particle_on_co = tw.particle_on_co)['R_matrix']
print('Nominal Twiss phase advances [turns]:', tw.qx, tw.qy)
print('Nominal half-traces / cos(2*pi*Q):',
      np.trace(nominal_map[:2, :2]) / 2, np.cos(2 * np.pi * tw.qx),
      np.trace(nominal_map[2:4, 2:4]) / 2, np.cos(2 * np.pi * tw.qy))

################################################################################
# Display the linear stability boundary
################################################################################
fig, axis = plt.subplots(figsize = (9, 4))
axis.plot(STRENGTH_SCALES, half_traces[:, 0], label = 'Horizontal')
axis.plot(STRENGTH_SCALES, half_traces[:, 1], label = 'Vertical')
axis.axhspan(-1, 1, color = 'grey', alpha = 0.15,
             label = 'Elliptic interval for each block')
axis.axhline(-1, color = 'grey', linestyle = ':')
axis.axhline(1, color = 'grey', linestyle = ':')
axis.set(xlabel = 'Common quadrupole strength scale',
         ylabel = 'Half trace of the transverse cell map')
axis.grid()
axis.legend()
fig.tight_layout()
plt.show()
