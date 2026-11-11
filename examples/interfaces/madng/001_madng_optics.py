# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

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
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
LINE_NAME       = 'demo'

########################################
# Quadrupole family trim
########################################
DKQF            = 1e-5                  # Additional arc focusing k1 [m^-2]

################################################################################
# Load the model
################################################################################
repo_root   = Path(__file__).resolve().parents[3]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env[LINE_NAME]
line.twiss_default['method']    = '4d'

################################################################################
# Compare periodic optics
################################################################################

########################################
# Build the MAD-NG model
########################################
# Native export preserves the line's expressions and reference particle.
# No MAD-X executable or cpymad session is involved.
model   = line.build_madng_model()
tw      = line.madng_twiss()

########################################
# Inspect the BPM optics
########################################
# The returned table contains Xtrack columns and MAD-NG columns ending in _ng.
# This uncoupled ring has beta11 = betx and beta22 = bety.
bpms    = tw.rows[line.steering_monitors_x]
bpms.cols['name s betx beta11_ng bety beta22_ng'].show(maxwidth = 'full')
print(f'Xtrack tunes: Qx = {tw.qx:.8f}, Qy = {tw.qy:.8f}')
qx_ng   = tw.mu1_ng[-1] - tw.mu1_ng[0]
qy_ng   = tw.mu2_ng[-1] - tw.mu2_ng[0]
print(f'MAD-NG tunes: Qx = {qx_ng:.8f}, Qy = {qy_ng:.8f}')

################################################################################
# Change a shared family knob
################################################################################
# Updating an existing variable also updates the attached MAD-NG model.
# Changing the lattice structure would instead require rebuilding that model.
line['dkqf']    = DKQF
tw_trimmed      = line.madng_twiss()
print(f'Focusing family trim: dkqf = {DKQF:.3g} m^-2')
print(f'Xtrack tune shifts: dQx = {tw_trimmed.qx - tw.qx:.8f}, '
      f'dQy = {tw_trimmed.qy - tw.qy:.8f}')
qx_trimmed_ng   = tw_trimmed.mu1_ng[-1] - tw_trimmed.mu1_ng[0]
qy_trimmed_ng   = tw_trimmed.mu2_ng[-1] - tw_trimmed.mu2_ng[0]
print('MAD-NG tune shifts: '
      f'dQx = {qx_trimmed_ng - qx_ng:.8f}, '
      f'dQy = {qy_trimmed_ng - qy_ng:.8f}')
model.close()

################################################################################
# Plot the nominal optics comparison
################################################################################
fig, axes   = plt.subplots(2, 1, sharex = True, figsize = (8, 6))

########################################
# Beta functions at BPMs
########################################
axes[0].plot(bpms.s, bpms.betx, 'C0-', label = r'Xtrack $\beta_x$')
axes[0].plot(bpms.s, bpms.bety, 'C1-', label = r'Xtrack $\beta_y$')
axes[0].plot(bpms.s, bpms.beta11_ng, 'C0x', label = r'MAD-NG $\beta_{11}$')
axes[0].plot(bpms.s, bpms.beta22_ng, 'C1x', label = r'MAD-NG $\beta_{22}$')
axes[0].set_ylabel(r'$\beta$ [m]')
axes[0].legend(ncol = 2)

########################################
# Relative differences
########################################
axes[1].plot(
    bpms.s, (bpms.beta11_ng / bpms.betx - 1) * 100,
    'C0o-',
    label   = r'$x$ plane',
)
axes[1].plot(
    bpms.s, (bpms.beta22_ng / bpms.bety - 1) * 100,
    'C1o-',
    label   = r'$y$ plane',
)
axes[1].axhline(0, color = 'black', linewidth = 0.5)
axes[1].set_ylabel(r'$(\beta_{\mathrm{NG}}/\beta_{\mathrm{XT}} - 1)$ [%]')
axes[1].set_xlabel('s [m]')
axes[1].legend()
fig.suptitle('Xtrack and MAD-NG: nominal uncoupled 4D optics')
fig.tight_layout()
plt.show()
