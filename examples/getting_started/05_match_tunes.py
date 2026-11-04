# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

# Further examples: examples/match/

################################################################################
# Required packages
################################################################################
from pathlib import Path

import matplotlib.pyplot as plt
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Model
########################################
# Lattice file relative to the repository root.
LATTICE_FILE    = 'test_data/example_lattices/ring/lattice.py'
LINE_NAME       = 'ring'

########################################
# Target tunes
########################################
QX_TARGET       = 1.70
QY_TARGET       = 1.65

########################################
# Matching settings
########################################
TUNE_TOL        = 1.0e-8
K1_STEP         = 1.0e-5                # Finite-difference step [m^-2]

################################################################################
# Load the model
################################################################################
repo_root = Path(__file__).resolve().parents[2]
env       = xt.load(repo_root / LATTICE_FILE)
line      = env[LINE_NAME]

################################################################################
# Calculate the starting optics
################################################################################
tw0 = line.twiss(method = '4d')
print(f'Before: Qx = {tw0.qx:.6f}, Qy = {tw0.qy:.6f}')

################################################################################
# Match the tunes
################################################################################
# Both variables control whole families through the lattice's expressions.
# solve = False separates the problem definition from its execution.
opt = line.match(
    method  = '4d',
    solve   = False,
    vary    = xt.VaryList(['kqf', 'kqd'], step = K1_STEP),
    targets = xt.TargetSet(qx = QX_TARGET, qy = QY_TARGET, tol = TUNE_TOL),
)
opt.solve()
opt.target_status()

################################################################################
# Inspect and plot the matched optics
################################################################################
tw = line.twiss(method = '4d')
print(f'After:  Qx = {tw.qx:.6f}, Qy = {tw.qy:.6f}')
print(f'Matched kqf = {env["kqf"]:.9f} m^-2')
print(f'Matched kqd = {env["kqd"]:.9f} m^-2')

fig, axes = plt.subplots(2, 1, sharex = True, figsize = (8, 6))
axes[0].plot(tw0.s, tw0.betx, label = 'Before')
axes[0].plot(tw.s, tw.betx, label = 'After')
axes[0].set_ylabel(r'$\beta_x$ [m]')
axes[0].legend()

axes[1].plot(tw0.s, tw0.bety, label = 'Before')
axes[1].plot(tw.s, tw.bety, label = 'After')
axes[1].set_ylabel(r'$\beta_y$ [m]')
axes[1].set_xlabel('s [m]')
fig.suptitle('Tune matching with two quadrupole families')
fig.tight_layout()
plt.show()
