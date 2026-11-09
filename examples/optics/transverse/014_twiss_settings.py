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
# Native ring
########################################
LATTICE_FILE    = 'test_data/example_lattices/ring/lattice.py'
LINE_NAME       = 'ring'

########################################
# Alternative finite-difference steps
########################################
STEPS_R         = {
    'dx':     2e-6,                      # [m]
    'dpx':    2e-7,                      # Normalized canonical momentum
    'dy':     2e-6,                      # [m]
    'dpy':    2e-7,
    'dzeta':  2e-5,                      # [m]
    'ddelta': 2e-6}                      # Relative momentum

################################################################################
# Set defaults on this loaded line
################################################################################
repo_root   = Path(__file__).resolve().parents[3]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env[LINE_NAME]
# These defaults apply to this line object, not to other loads or installations.
line.twiss_default['method'] = '4d'
line.twiss_default['chrom']  = False

################################################################################
# Request the results needed for the calculation
################################################################################

########################################
# Compute optics using the line defaults
########################################
baseline = line.twiss()
print('Method:', baseline.method)
print('Chromatic derivatives included:', 'dqx' in baseline.keys())
print('Tunes:', baseline.qx, baseline.qy)

########################################
# Request only the orbit
########################################
orbit = line.twiss(only_orbit = True)
orbit.rows[r'obs\..*'].cols['name s x px y py'].show(maxwidth = 'full')
print('Beta functions included in orbit-only result:', 'betx' in orbit.keys())

########################################
# Attach magnetic strengths to the optics table
########################################
with_strengths = line.twiss(strengths = True)
with_strengths.rows[['qf.1', 'qd.1', 'bf.1']].cols[
    'name s betx bety k0l k1l angle'].show(maxwidth = 'full')

# k0l and bend angle are dimensionless; k1l is in m^-1.

################################################################################
# Compare numerical step choices
################################################################################

########################################
# Supply finite-difference steps for the linearized map
########################################
# Steps control the probes used to construct R; they are not beam emittances.
alternative = line.twiss(steps_R_matrix = STEPS_R)
print('Baseline steps:', baseline.steps_R_matrix)
print('Steps used after override:', alternative.steps_R_matrix)

########################################
# Check the sensitivity of physical outputs
########################################
print('Tune changes after changing the steps:',
      alternative.qx - baseline.qx, alternative.qy - baseline.qy)
print('Largest relative beta changes (x, y):',
      np.max(np.abs(alternative.betx / baseline.betx - 1)),
      np.max(np.abs(alternative.bety / baseline.bety - 1)))

# This stable, smooth model should give consistent optics over these small
# step changes. Numerical convergence needs its own check for other models.
