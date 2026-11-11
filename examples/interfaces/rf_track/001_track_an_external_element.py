# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
import numpy as np
import RF_Track as rft
import xobjects as xo
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Element and reference
########################################
DRIFT_LENGTH    = 2.0                   # [m]
P0C             = 1.0e9                 # Electron reference p0*c [eV]

########################################
# Five entrance probes
########################################
X               = [0, 1e-3, -1e-3, 0, 0]           # [m]
PX              = [0, 2e-3, -2e-3, 1e-3, -1e-3]    # P_x / p0
Y               = [0, 0, 0, 2e-3, -2e-3]           # [m]
PY              = [0, -1e-3, 1e-3, 3e-3, -3e-3]    # P_y / p0
ZETA            = [0, 1e-3, -1e-3, 2e-3, -2e-3]    # [m]
DELTA           = [0, 0.01, -0.01, 0.02, -0.02]     # (p - p0) / p0

################################################################################
# Wrap an RF-Track element
################################################################################
# RFT_Element calls RF-Track on CPU; keep its particles on a CPU context.
context     = xo.ContextCpu()
external    = xt.RFT_Element(
    element     = rft.Drift(DRIFT_LENGTH),
    update_ref  = False,
)
# Keep the entrance reference momentum fixed in this comparison.
native      = xt.Drift(length = DRIFT_LENGTH, model = 'exact')

################################################################################
# Track the same incoming particles
################################################################################

########################################
# Supply Xtrack coordinates
########################################
# RF-Track uses mm, mrad slopes, time in mm/c and momentum in MeV/c.
# The bridge converts x, px=P_x/p0, y, py=P_y/p0, zeta and delta for us.
# In particular, slopes are P_x/P_s and time is -zeta/(beta0*c).
particles   = xt.Particles(
    'electron',
    _context    = context,
    p0c         = P0C,
    x           = X,
    px          = PX,
    y           = Y,
    py          = PY,
    zeta        = ZETA,
    delta       = DELTA,
)
particles_rft   = particles.copy()
particles_xt    = particles.copy()

########################################
# Compare exact drift transport
########################################
# An exact native drift includes the path-length effect of finite slopes.
external.track(particles_rft)
native.track(particles_xt)

################################################################################
# Inspect the transported coordinates and agreement
################################################################################

########################################
# External-element output
########################################
output  = xt.Table(
    data    = {
        'particle_id':  particles_rft.particle_id,
        'x_mm':         particles_rft.x * 1e3,
        'px':           particles_rft.px,
        'y_mm':         particles_rft.y * 1e3,
        'py':           particles_rft.py,
        'zeta_mm':      particles_rft.zeta * 1e3,
        'delta':        particles_rft.delta,
    },
    index   = 'particle_id',
)
output.show(maxwidth = 'full')

########################################
# Coordinate differences from native Xtrack
########################################
for coordinate, unit in [
    ('x', 'm'), ('px', '1'), ('y', 'm'), ('py', '1'),
    ('zeta', 'm'), ('delta', '1'), ('s', 'm'),
]:
    difference  = (getattr(particles_rft, coordinate)
                   - getattr(particles_xt, coordinate))
    print(f'Max |RF-Track - Xtrack| in {coordinate} [{unit}]: '
          f'{np.max(np.abs(difference)):.3e}')
print(f'Initial p0*c: {P0C * 1e-9:.9f} GeV')
print(f'Final p0*c:   {particles_rft.p0c[0] * 1e-9:.9f} GeV')
