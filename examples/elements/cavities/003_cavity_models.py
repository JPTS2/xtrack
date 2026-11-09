# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
import numpy as np
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Reference and RF
########################################
P0C             = 1.0e9                 # Proton reference p0*c [eV]
LENGTH          = 1.0                   # [m]
VOLTAGE         = 5.0e6                 # [V]
FREQUENCY       = 200.0e6               # [Hz]
PHASE           = 0.7                   # [rad]

########################################
# Entrance state
########################################
PX_START        = 0.05                  # P_x / p0
PY_START        = 0.03                  # P_y / p0
ZETA_START      = 0.01                  # [m]
DELTA_START     = 0.02                  # (p - p0) / p0

########################################
# Common numerical settings
########################################
INTEGRATOR      = 'uniform'
N_KICKS         = 4
BASELINE_MODEL  = 'drift-kick-drift-exact'

################################################################################
# Inspect supported map choices
################################################################################
models = xt.Cavity.get_available_models()
print('Available cavity maps:', models)
print('Available integrators:', xt.Cavity.get_available_integrators())
# Here model selects the transport-map approximation, not an RF field profile.

################################################################################
# Compare maps with fixed voltage, geometry and kick settings
################################################################################

########################################
# Prepare the common entrance state
########################################
initial = xt.Particles(
    mass0   = xt.PROTON_MASS_EV,
    p0c     = P0C,
    px      = PX_START,
    py      = PY_START,
    zeta    = ZETA_START,
    delta   = DELTA_START,
)

########################################
# Track each public model
########################################
outputs = {}
for model in models:
    cavity = xt.Cavity(
        length      = LENGTH,
        voltage     = VOLTAGE,
        frequency   = FREQUENCY,
        phase       = PHASE,
        model       = model,
        integrator  = INTEGRATOR,
        num_kicks   = N_KICKS,
    )
    particles = initial.copy()
    cavity.track(particles)
    outputs[model] = particles

################################################################################
# Inspect differences from the chosen baseline
################################################################################
# Exact refers to the drift treatment. This selected map and kick count
# provide a comparison baseline, rather than an RF-field accuracy standard.
baseline    = outputs[BASELINE_MODEL]
dx          = np.array([p.x[0] - baseline.x[0] for p in outputs.values()])
dzeta       = np.array([
    p.zeta[0] - baseline.zeta[0] for p in outputs.values()])
denergy     = np.array([
    (p.ptau[0] - baseline.ptau[0]) * P0C for p in outputs.values()])
table       = xt.Table(
    data    = {
        'name':     np.array(list(outputs)),
        'dx_um':    dx * 1e6,
        'dzeta_um': dzeta * 1e6,
        'dE_eV':    denergy},
    index   = 'name')
table.show()
print('Baseline model, integrator and kick count:',
      BASELINE_MODEL, INTEGRATOR, N_KICKS)
