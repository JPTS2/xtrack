# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
import numpy as np
from scipy.constants import c as clight
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Reference particle and finite-length RF
########################################
P0C             = 200.0e6               # Proton reference p0*c [eV]
LENGTH          = 0.5                   # [m]
VOLTAGE         = 1.0e6                 # Total voltage [V]
FREQUENCY       = 20.0e6                # [Hz]
PHASE           = np.pi / 4             # [rad], relative RF clock

########################################
# Entrance state
########################################
X_START         = 1.0e-3                # [m]
PX_START        = 0.01                  # P_x / p0
ZETA_START      = 0.02                  # [m]

########################################
# Numerical settings
########################################
N_KICKS         = 8                     # Uniform kick/drift steps

################################################################################
# Track through the same length with voltage off and on
################################################################################

########################################
# Define the distributed kick/drift map
########################################
cavity = xt.Cavity(
    length      = LENGTH,
    voltage     = VOLTAGE,
    frequency   = FREQUENCY,
    phase       = PHASE,
    model       = 'drift-kick-drift-exact',
    integrator  = 'uniform',
    num_kicks   = N_KICKS,
)
voltage_off = cavity.copy()
voltage_off.voltage = 0.0

########################################
# Use identical entrance particles
########################################
initial = xt.Particles(
    mass0   = xt.PROTON_MASS_EV,
    p0c     = P0C,
    x       = X_START,
    px      = PX_START,
    zeta    = ZETA_START,
)
# Standalone element tracking retains the exit s = LENGTH for this open pass.
off = initial.copy()
on  = initial.copy()
voltage_off.track(off)
cavity.track(on)

################################################################################
# Inspect longitudinal transport and time of flight
################################################################################

########################################
# Calculate elapsed time and relative RF phase
########################################
states  = {'entrance': initial, 'voltage off': off, 'powered': on}
beta0   = initial.beta0[0]

# t = (s-zeta)/(beta0*c) here; time_s is zero and p0c is unchanged.
times   = np.array([
    ((p.s[0] - initial.s[0]) - (p.zeta[0] - initial.zeta[0])) / (beta0 * clight)
    for p in states.values()])
phases  = np.array([
    PHASE - 2 * np.pi * FREQUENCY * p.zeta[0] / (beta0 * clight)
    for p in states.values()])

########################################
# Display entrance and exit quantities
########################################
table = xt.Table(
    data    = {
        'name':         np.array(list(states)),
        's_m':          np.array([p.s[0] for p in states.values()]),
        'kinetic_MeV':  np.array([
            (p.energy[0] - p.mass0) * 1e-6 for p in states.values()]),
        'beta':         np.array([
            p.beta0[0] * p.rvv[0] for p in states.values()]),
        'zeta_mm':      np.array([
            p.zeta[0] * 1e3 for p in states.values()]),
        'time_ns':      times * 1e9,
        'phase_rad':    phases},
    index   = 'name')
table.show()
print('Reference flight time over LENGTH [ns]:',
      LENGTH / (beta0 * clight) * 1e9)
print('Reference p0*c before and after [eV]:', initial.p0c[0], on.p0c[0])

################################################################################
# Inspect transverse momentum and slope
################################################################################
# The longitudinal kick preserves canonical P_x/p0. Acceleration changes
# P_s/p0, so the geometrical slope dx/ds changes between transport steps.
transverse = xt.Table(
    data    = {
        'name':         np.array(list(states)),
        'x_mm':         np.array([p.x[0] * 1e3 for p in states.values()]),
        'px':           np.array([p.px[0] for p in states.values()]),
        'dx_ds':        np.array([p.kin_xp[0] for p in states.values()])},
    index   = 'name')
transverse.show()
