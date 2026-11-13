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
# Native RF ring
########################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
LINE_NAME       = 'demo'
RF_VOLTAGE      = 1e5                    # Thin-cavity voltage [V]

########################################
# Mode populations
########################################
NEMITT_X        = 1e-6                    # Normalized mode 1 emittance [m rad]
NEMITT_Y        = 2e-6                    # Normalized mode 2 emittance [m rad]
SIGMA_ZETA_3    = 0.02                    # Mode 3 rms zeta at the entrance [m]

################################################################################
# Calculate the 6D optical modes
################################################################################
repo_root           = Path(__file__).resolve().parents[3]
env                 = xt.load(repo_root / LATTICE_FILE)
line                = env[LINE_NAME]
env['voltage_rf']   = RF_VOLTAGE
tw                  = line.twiss6d()

################################################################################
# Specify longitudinal emittance in the canonical coordinate convention
################################################################################

########################################
# Set the mode 3 contribution to the entrance bunch length
########################################
# bets0 is the signed mode-3 zeta beta at the entrance. Its magnitude gives
# sigma_zeta,3^2 = abs(bets0)*gemitt_zeta. The emittance unit is m in
# (zeta,pzeta), rather than eV*s. This specifies linear covariance.
gemitt_zeta = SIGMA_ZETA_3**2 / abs(tw.bets0)
mode3       = tw.get_beam_covariance(gemitt_zeta = gemitt_zeta)

########################################
# Include all three independent mode populations
########################################
covariance = tw.get_beam_covariance(
    nemitt_x        = NEMITT_X,
    nemitt_y        = NEMITT_Y,
    gemitt_zeta     = gemitt_zeta)
print('Geometric longitudinal emittance [m]:', gemitt_zeta)
print('Mode-3 entrance sigma_zeta [m]:', mode3.sigma_zeta[0])
print('Full entrance sigma_zeta [m]:', covariance.sigma_zeta[0])

# With 6D coupling, transverse modes can also contribute to zeta variance.

################################################################################
# Inspect the longitudinal covariance and its contour
################################################################################

########################################
# Read rms sizes and correlation
########################################
# The longitudinal block has diagonal units m^2 and 1, and covariance in m.
# pzeta agrees with delta only to first order near the reference particle.
covariance.rows[['arc.s1.c1.qf.in', 'ip.1', 'rf.1']].cols[
    'name sigma_zeta sigma_pzeta'].show(maxwidth = 'full')
longitudinal = covariance.Sigma[0, 4:6, 4:6]
print('Entrance covariance block (zeta, pzeta):')
print(longitudinal)

########################################
# Draw the entrance contour u^T Sigma^-1 u = 1
########################################
# u contains offsets (zeta-zeta_co, pzeta-pzeta_co), centered on the orbit.
theta   = np.linspace(0, 2 * np.pi, 200)
circle  = np.array([np.cos(theta), np.sin(theta)])
contour = np.linalg.cholesky(longitudinal) @ circle
fig, ax = plt.subplots(figsize = (7, 5))
ax.plot(contour[0] * 1e3, contour[1] * 1e3)
ax.set_xlabel('zeta - zeta_co [mm]')
ax.set_ylabel('pzeta - pzeta_co [1e-3]')
ax.set_title('Entrance covariance contour: u^T Sigma^-1 u = 1')
ax.grid()
fig.tight_layout()
plt.show()
