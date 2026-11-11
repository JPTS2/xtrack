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
# Shared magnetic ring: RF and radiation off
########################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
LINE_NAME       = 'demo'

########################################
# Weak integrated sextupole strengths and requested terms
########################################
K2L_F           = 0.002                  # [m^-2]
K2L_D           = -0.002                 # [m^-2]
RDTS            = ['f3000', 'f1020']
# f_jklm has the resonance harmonic (j-k)*Qx + (l-m)*Qy:
# these terms describe 3*Qx and Qx + 2*Qy, respectively.

################################################################################
# Evaluate the bare ring with MAD-NG
################################################################################
repo_root   = Path(__file__).resolve().parents[3]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env[LINE_NAME]

# This public helper exports the line and returns MAD-NG columns alongside
# Xtrack optics. RDTs use MAD-NG's complex normal-form convention.
model   = line.build_madng_model()
bare    = line.madng_twiss(rdts = RDTS, normal_form = False)

################################################################################
# Power the sextupoles and recompute the resonance driving terms
################################################################################

########################################
# Change the existing symbolic circuits
########################################
line['k2lf']    = K2L_F
line['k2ld']    = K2L_D
# Changes to these Xtrack variables also update the exported MAD-NG model.
powered = line.madng_twiss(rdts = RDTS, normal_form = False)
model.close()

########################################
# Inspect the largest magnitudes around the ring
########################################
# The unpowered ring can already have geometric nonlinearities in bends.
# Keep its RDTs as a baseline rather than assuming they vanish.
bare_max    = np.array([np.max(np.abs(bare[rdt])) for rdt in RDTS])
powered_max = np.array([np.max(np.abs(powered[rdt])) for rdt in RDTS])
change_max  = np.array([
    np.max(np.abs(powered[rdt] - bare[rdt])) for rdt in RDTS])
print('RDT magnitudes and magnitudes of their differences are in m^-1/2.')
xt.Table(
    data    = {
        'rdt':                 np.array(RDTS),
        'max_abs_bare':         bare_max,
        'max_abs_powered':      powered_max,
        'max_abs_difference':   change_max},
    index   = 'rdt').show(maxwidth = 'full')

################################################################################
# Plot the resonance driving terms around the ring
################################################################################
# The ring pairs sextupoles across -I: leading geometric terms cancel outside
# each pair, so small values in the straight sections are expected.
fig, axes   = plt.subplots(
    len(RDTS), 1,
    sharex  = True,
    figsize = (9, 6),
    squeeze = False,
)
for ax, rdt in zip(axes[:, 0], RDTS):
    ax.plot(bare.s, np.abs(bare[rdt]), label = 'Sextupoles off')
    ax.plot(powered.s, np.abs(powered[rdt]), label = 'Sextupoles powered')
    ax.set_ylabel(rf'$|{rdt}|$ [m$^{{-1/2}}$]')
    ax.legend()
axes[-1, 0].set_xlabel('s [m]')
fig.tight_layout()
plt.show()
