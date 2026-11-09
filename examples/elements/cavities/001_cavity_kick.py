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
# Reference particle and RF
########################################
P0C             = 1.0e9                 # Proton reference p0*c [eV]
VOLTAGE         = 1.0e6                 # [V]
FREQUENCY       = 400.0e6               # [Hz]
PHASE           = np.pi / 4             # [rad], at zeta = 0
VOLTAGE_SCALE   = 1.5

########################################
# Phase and arrival scans
########################################
PHASE_SCAN      = np.linspace(-np.pi, np.pi, 17)  # [rad], at zeta = 0
ZETA_SCAN       = np.linspace(-0.05, 0.05, 11)    # [m]; positive is earlier

################################################################################
# Inspect the zero-length energy kick
################################################################################

########################################
# Define the RF element and reference
########################################
reference   = xt.Particles(mass0 = xt.PROTON_MASS_EV, p0c = P0C)
cavity      = xt.Cavity(voltage = VOLTAGE, frequency = FREQUENCY, phase = PHASE)

########################################
# Scan the RF phase at zeta = 0
########################################
# For this unit-charge proton, Delta E [eV] = voltage [V] * sin(phase).
phase_kicks = []
for phase in PHASE_SCAN:
    cavity.phase = phase
    particles = reference.copy()
    cavity.track(particles)
    phase_kicks.append((particles.ptau[0] - reference.ptau[0]) * P0C)
phase_kicks             = np.array(phase_kicks)
expected_phase_kicks    = VOLTAGE * np.sin(PHASE_SCAN)

################################################################################
# Inspect arrival-coordinate and voltage dependence
################################################################################

########################################
# Track particles at the chosen phase
########################################
cavity.phase = PHASE
initial = xt.Particles(mass0 = xt.PROTON_MASS_EV, p0c = P0C, zeta = ZETA_SCAN)
particles = initial.copy()
cavity.track(particles)

# With the relative RF clock, phase_seen = phase - omega*zeta/(beta0*c).
phase_seen = PHASE - 2 * np.pi * FREQUENCY * ZETA_SCAN / (
    reference.beta0[0] * clight)
expected_kicks  = VOLTAGE * np.sin(phase_seen)
energy_kicks    = (particles.ptau - initial.ptau) * initial.p0c

########################################
# Scale the voltage with the same entrance particles
########################################
scaled_cavity = cavity.copy()
scaled_cavity.voltage = VOLTAGE_SCALE * VOLTAGE
scaled_particles = initial.copy()
scaled_cavity.track(scaled_particles)
scaled_kicks = (scaled_particles.ptau - initial.ptau) * initial.p0c

################################################################################
# Inspect the reference and particle quantities
################################################################################
# The RF kick changes particle energy, delta and ptau; p0c stays fixed.
i = len(ZETA_SCAN) // 2
print('At zeta = 0, energy kick [MeV]:', energy_kicks[i] * 1e-6)
print('Reference p0*c before and after [eV]:',
      initial.p0c[i], particles.p0c[i])
print('Particle delta before and after:', initial.delta[i], particles.delta[i])
print('Particle ptau before and after:', initial.ptau[i], particles.ptau[i])

################################################################################
# Plot the phase, arrival and voltage dependence
################################################################################
fig, axes = plt.subplots(1, 2, figsize = (11, 4))
axes[0].plot(PHASE_SCAN, expected_phase_kicks * 1e-6, label = 'V sin(phase)')
axes[0].plot(PHASE_SCAN, phase_kicks * 1e-6, 'o', label = 'Tracked')
axes[0].set_xlabel('RF phase at zeta = 0 [rad]')
axes[1].plot(
    ZETA_SCAN * 1e3, expected_kicks * 1e-6, label = 'V sin(phase_seen)',
)
axes[1].plot(ZETA_SCAN * 1e3, energy_kicks * 1e-6, 'o', label = 'Tracked')
axes[1].plot(ZETA_SCAN * 1e3, scaled_kicks * 1e-6, 'x',
             label = f'Tracked, voltage scaled by {VOLTAGE_SCALE:g}')
axes[1].set_xlabel('Entrance zeta [mm]')
for ax in axes:
    ax.set_ylabel('Energy kick [MeV]')
    ax.legend()
    ax.grid()
fig.tight_layout()
plt.show()
