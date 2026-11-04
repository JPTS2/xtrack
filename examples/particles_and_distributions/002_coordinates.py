# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
import numpy as np
from scipy.constants import c

import xtrack as xt

################################################################################
# User parameters
################################################################################
P0C         = 1.0e9                     # Proton reference p0*c [eV]
X           = np.array([-1e-3, 0.0, 1e-3])  # [m]
PX          = np.array([0.05, 0.10, 0.15])  # Normalized canonical px
ZETA        = np.array([-0.01, 0.0, 0.01])  # [m]
DELTA       = np.array([-0.1, 0.0, 0.1])  # (p - p0) / p0
DELTA_AFTER = np.array([-0.05, 0.05, 0.15])

################################################################################
# Supply momentum deviation and inspect linked quantities
################################################################################

########################################
# Specify delta, leaving the energy coordinates to Xtrack
########################################
# All particles have the reference proton mass and charge. Only one of
# delta, ptau or pzeta is needed; Xtrack supplies the linked quantities.
particles = xt.Particles(
    pdg_id_0    = 'proton',
    p0c         = P0C,
    x           = X,
    px          = PX,
    zeta        = ZETA,
    delta       = DELTA,
)

########################################
# Inspect the derived properties
########################################
# For this species: ptau = (E - E0)/(p0*c), pzeta = ptau/beta0,
# rpp = 1/(1 + delta), and rvv = beta/beta0.
energy_table = xt.Table(
    data    = {
        'particle_id': particles.particle_id,
        'delta_given': DELTA,
        'ptau':        particles.ptau,
        'pzeta':       particles.pzeta,
        'rpp':         particles.rpp,
        'rvv':         particles.rvv,
        'energy_GeV':  particles.energy * 1e-9,
    },
    index   = 'particle_id',
)
energy_table.show()

################################################################################
# Use another energy coordinate or update delta
################################################################################

########################################
# Reconstruct from ptau alone
########################################
particles_from_ptau = xt.Particles(
    pdg_id_0    = 'proton',
    p0c         = P0C,
    ptau        = particles.ptau,
)
comparison = xt.Table(
    data    = {
        'particle_id':     particles.particle_id,
        'delta_given':     DELTA,
        'delta_from_ptau': particles_from_ptau.delta,
        'pzeta_original':  particles.pzeta,
        'pzeta_from_ptau': particles_from_ptau.pzeta,
    },
    index   = 'particle_id',
)
comparison.show()

########################################
# Update a coordinate through its public setter
########################################
particles_changed = particles.copy()
particles_changed.delta = DELTA_AFTER
print('After assigning delta, Xtrack updates the linked energy coordinates:')
updated = xt.Table(
    data    = {
        'particle_id': particles_changed.particle_id,
        'delta':       particles_changed.delta,
        'ptau':        particles_changed.ptau,
        'pzeta':       particles_changed.pzeta,
        'rpp':         particles_changed.rpp,
        'rvv':         particles_changed.rvv,
        'energy_GeV':  particles_changed.energy * 1e-9,
    },
    index   = 'particle_id',
)
updated.show()

################################################################################
# Interpret transverse momentum and longitudinal position
################################################################################

########################################
# Compare canonical momentum and the derived slope
########################################
# Unspecified py, ax and ay are zero. In this straight field-free frame,
# kin_ps = sqrt((1 + delta)^2 - px^2 - py^2), and dx/ds = kin_px/kin_ps.
# The deliberately large px values make the difference visible.
transverse = xt.Table(
    data    = {
        'particle_id': particles.particle_id,
        'delta':       particles.delta,
        'px':          particles.px,
        'kin_ps':      particles.kin_ps,
        'dx_ds':       particles.kin_xp,
    },
    index   = 'particle_id',
)
transverse.show()

########################################
# Convert zeta to an arrival-time offset
########################################
# t = time_s + (s - zeta)/(beta0*c). At fixed s and time_s, positive
# zeta means earlier arrival relative to the reference particle.
arrival_offset_ns   = -particles.zeta / (particles.beta0 * c) * 1e9
arrival             = xt.Table(
    data    = {
        'particle_id':       particles.particle_id,
        'zeta_m':            particles.zeta,
        'arrival_offset_ns': arrival_offset_ns,
    },
    index   = 'particle_id',
)
arrival.show()
