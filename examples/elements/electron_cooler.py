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
# Homogeneous positive-ion reference and signed input momenta
########################################
MASS0           = 193729.0248722061e6   # Illustrative ion rest energy [eV]
Q0              = 54                   # Reference charge in units of e
GAMMA0          = 1.004469679
X               = np.array([-3, -2, -1, 1, 2, 3]) * 1e-3  # [m]
PX              = np.array([-4, -2, -1, 1, 2, 4]) * 1e-4  # P_x/p0
PY              = np.array([1, -2, 3, -3, 2, -1]) * 1e-4
DELTA           = np.array([3, -2, 1, -1, 2, -3]) * 1e-4  # (p-p0)/p0

########################################
# Native Parkhomchuk friction configuration
########################################
CURRENT         = 0.6                  # Electron current magnitude [A]
COOLER_LENGTH   = 2.5                  # Effective interaction length [m]
RADIUS          = 25e-3                # Electron-beam radius [m]
TEMP_PERP       = 0.1                  # Electron k_B*T_perp [eV]
TEMP_LONG       = 1e-3                 # Electron k_B*T_long [eV]
MAGNETIC_FIELD  = 0.075                # Longitudinal guiding field [T]

################################################################################
# Apply one native cooler passage to independent copies
################################################################################

########################################
# Prepare ions inside the electron beam and define the interaction
########################################
initial = xt.Particles(
    mass0   = MASS0,
    q0      = Q0,
    gamma0  = GAMMA0,
    x       = X,
    y       = 0.0,
    px      = PX,
    py      = PY,
    delta   = DELTA,
)
cooler  = xt.ElectronCooler(
    current                 = CURRENT,
    length                  = COOLER_LENGTH,
    radius_e_beam           = RADIUS,
    temp_perp               = TEMP_PERP,
    temp_long               = TEMP_LONG,
    magnetic_field          = MAGNETIC_FIELD,
    magnetic_field_ratio    = 0.0,
    space_charge_factor     = 0.0,
    offset_energy           = 0.0,
)

# The electrons' mean speed matches the reference ion at zero energy offset.
# Electron space charge and magnetic-field imperfections are disabled.

########################################
# Compare zero current with the configured electron current
########################################
cooler_off          = cooler.copy()
cooler_off.current  = 0.0
unpowered           = initial.copy()
powered             = initial.copy()
cooler_off.track(unpowered)
cooler.track(powered)

################################################################################
# Inspect friction kicks, without a ring-optics or equilibrium interpretation
################################################################################
# px/py are canonical momenta normalized to p0, not geometric slopes.
# ElectronCooler uses length to integrate its force; it does not drift x/y or
# advance s. The guiding field enters the friction model, not a solenoid map.
for coordinate in ['px', 'py', 'delta']:
    before      = np.asarray(getattr(initial, coordinate))
    after       = np.asarray(getattr(powered, coordinate))
    off         = np.asarray(getattr(unpowered, coordinate))
    print(coordinate, '(dimensionless):')
    xt.Table(
        data    = {
            'particle_id':       initial.particle_id,
            'before':            before,
            'powered_after':     after,
            'powered_kick':      after - before,
            'zero_current_kick': off - before,
        },
        index   = 'particle_id',
    ).show(maxwidth = 'full')
print('Final states:', powered.state)

# This homogeneous positive-ion example demonstrates one deterministic friction
# passage. It does not determine a cooling time or a heating-balanced state.
