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
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
LINE_NAME       = 'demo'
OBSERVATION     = 'bpm.s4.d'
RANGE_START     = 'arc.s1.c1.qf.in'
RANGE_END       = 'arc.s1.c1.qd'

################################################################################
# Calculate the ring optics
################################################################################
repo_root   = Path(__file__).resolve().parents[3]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env[LINE_NAME]
tw          = line.twiss4d()

################################################################################
# Distinguish global results from row quantities
################################################################################

########################################
# Global optics and chromaticities
########################################
print('Twiss method:', tw.method)
print(f'Tunes: Qx = {tw.qx:.6f}, Qy = {tw.qy:.6f}')
print(f'Chromaticities: dQx/d(delta) = {tw.dqx:.6f}, '
      f'dQy/d(delta) = {tw.dqy:.6f}')

########################################
# A scalar at a named entrance and full column arrays
########################################
print(f'beta_x at {OBSERVATION}: {tw["betx", OBSERVATION]:.6f} m')
print(f'Maximum sampled beta_x: {np.max(tw.betx):.6f} m')
print('Available quantities:', list(tw.keys()))

################################################################################
# Select rows and columns for the task
################################################################################

########################################
# Select ideal BPM read planes by name pattern
########################################
# Regular expressions select table rows, not a new optics calculation.
tw.rows[r'bpm\..*'].cols[
    'name s x px y py betx bety dx dy'].show(maxwidth = 'full')

########################################
# Inspect a contiguous range of element entrances
########################################
# Named table slices include both endpoints. They select an existing result;
# a ranged line.twiss call instead calculates optics through selected elements.
tw.rows[RANGE_START:RANGE_END].cols[
    'name s betx alfx bety alfy dx dpx mux muy'].show(maxwidth = 'full')
