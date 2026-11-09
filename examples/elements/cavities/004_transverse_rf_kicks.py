# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c as clight
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Reference and transverse RF
########################################
P0C             = 1.0e9                 # Proton reference p0*c [eV]
CRAB_VOLTAGE    = 1.0e6                 # [V]
FREQUENCY       = 400.0e6               # [Hz]
PHASE           = 0.0                   # [rad]

########################################
# Arrival scan and horizontal offset
########################################
ZETA_SCAN       = np.linspace(-0.02, 0.02, 11)  # [m]; positive arrives earlier
X_START         = 2.0e-3                # [m]; exposes the companion energy kick

################################################################################
# Define equivalent thin transverse RF kicks
################################################################################

########################################
# Use voltage and sine phase for CrabCavity
########################################
crab = xt.CrabCavity(
    crab_voltage    = CRAB_VOLTAGE,
    frequency       = FREQUENCY,
    phase           = PHASE,
)

########################################
# Use integrated strength and cosine phase for RFMultipole
########################################
# Unit-charge protons: CrabCavity gives +V/p0c*sin(phase_seen) in P_x/p0.
# RFMultipole gives -knl[0]*cos(phase_n_seen), so add pi/2 to match it.
k0l = CRAB_VOLTAGE / P0C
rf_multipole = xt.RFMultipole(
    knl         = [k0l],
    frequency   = FREQUENCY,
    phase_n     = [PHASE + np.pi / 2],
)

################################################################################
# Track identical off-axis entrance particles
################################################################################
initial = xt.Particles(
    mass0   = xt.PROTON_MASS_EV,
    p0c     = P0C,
    x       = X_START,
    zeta    = ZETA_SCAN,
)
crab_particles  = initial.copy()
rf_particles    = initial.copy()
crab.track(crab_particles)
rf_multipole.track(rf_particles)

################################################################################
# Inspect transverse and companion longitudinal kicks
################################################################################

########################################
# Calculate the thin-map expectations
########################################
wave_number = 2 * np.pi * FREQUENCY / clight
phase_seen  = PHASE - wave_number * ZETA_SCAN / initial.beta0[0]
expected_px = k0l * np.sin(phase_seen)
# The off-axis companion energy kick depends on the phase derivative.
expected_energy = -CRAB_VOLTAGE * wave_number * X_START * np.cos(phase_seen)
crab_energy     = (crab_particles.ptau - initial.ptau) * initial.p0c
rf_energy       = (rf_particles.ptau - initial.ptau) * initial.p0c

########################################
# Plot both descriptions and the analytic kicks
########################################
fig, axes = plt.subplots(1, 2, figsize = (11, 4))
axes[0].plot(ZETA_SCAN * 1e3, expected_px * 1e6, label = 'Analytic')
axes[0].plot(
    ZETA_SCAN * 1e3, crab_particles.px * 1e6, 'o', label = 'CrabCavity',
)
axes[0].plot(ZETA_SCAN * 1e3, rf_particles.px * 1e6, 'x', label = 'RFMultipole')
axes[0].set_ylabel('Horizontal kick P_x/p0 [1e-6]')
axes[1].plot(ZETA_SCAN * 1e3, expected_energy * 1e-3, label = 'Analytic')
axes[1].plot(ZETA_SCAN * 1e3, crab_energy * 1e-3, 'o', label = 'CrabCavity')
axes[1].plot(ZETA_SCAN * 1e3, rf_energy * 1e-3, 'x', label = 'RFMultipole')
axes[1].set_ylabel('Companion energy kick [keV]')
for ax in axes:
    ax.set_xlabel('Entrance zeta [mm]')
    ax.legend()
    ax.grid()
fig.tight_layout()
plt.show()
