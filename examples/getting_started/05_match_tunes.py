# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

# Further examples: examples/matching/

################################################################################
# Required packages
################################################################################
from pathlib import Path

import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Model
########################################
# Lattice file relative to the repository root.
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
LINE_NAME       = 'demo'

########################################
# Target tunes
########################################
QX_TARGET       = 8.75683249
QY_TARGET       = 10.01945242

########################################
# Matching settings
########################################
TUNE_TOL        = 1.0e-8
K1_STEP         = 1.0e-5                # Finite-difference step [m^-2]

################################################################################
# Load the model
################################################################################
repo_root   = Path(__file__).resolve().parents[2]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env[LINE_NAME]

################################################################################
# Calculate the starting optics
################################################################################
tw0 = line.twiss(method = '4d')
print('Starting tunes\n--------------')
print(f'Before: Qx = {tw0.qx:.6f}, Qy = {tw0.qy:.6f}')

################################################################################
# Match the tunes
################################################################################
# Both additive trims control arc families through the lattice's expressions.
# solve = False separates the problem definition from its execution.
print('\nMatch the tunes\n---------------')
opt = line.match(
    method  = '4d',
    solve   = False,
    vary    = xt.VaryList(['dkqf', 'dkqd'], step = K1_STEP),
    targets = xt.TargetSet(qx = QX_TARGET, qy = QY_TARGET, tol = TUNE_TOL),
)
opt.solve()

print('\nTarget tolerances\n-----------------')
opt.target_status()

################################################################################
# Inspect the matched tunes and arc-family trims
################################################################################
tw  = line.twiss(method = '4d')
print('\nMatched tunes and trims\n-----------------------')
print(f'After:  Qx = {tw.qx:.6f}, Qy = {tw.qy:.6f}')
print(f'Matched dkqf [m^-2]: {env["dkqf"]:.9f}')
print(f'Matched dkqd [m^-2]: {env["dkqd"]:.9f}')
