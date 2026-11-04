# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
from pathlib import Path

import numpy as np
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Model
########################################
# Lattice file relative to the repository root.
LATTICE_FILE    = 'test_data/example_lattices/ring/lattice.py'
LINE_NAME       = 'ring'

########################################
# Launch coordinates
########################################
X_NORM          = np.array([-2.0, 0.0, 2.0])
PX_NORM         = np.array([0.5, 0.0, -0.5])
Y_NORM          = 0.5

########################################
# Normalized rms emittances
########################################
NEMITT_X        = 1.0e-6                # [m rad]
NEMITT_Y        = 2.0e-6                # [m rad]

################################################################################
# Load the model
################################################################################
repo_root   = Path(__file__).resolve().parents[2]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env[LINE_NAME]

################################################################################
# Build a normalized launch
################################################################################

########################################
# Transform normalized to physical coordinates
########################################
# This is a launch scan, not a statistical bunch. With no RF, use 4D
# periodic optics. Unspecified longitudinal coordinates are zero.
particles = line.build_particles(
    method      = '4d',
    x_norm      = X_NORM,
    px_norm     = PX_NORM,
    y_norm      = Y_NORM,
    nemitt_x    = NEMITT_X,
    nemitt_y    = NEMITT_Y,
)
particles.get_table().cols['particle_id x px y py delta'].show()

########################################
# Recover the normalized coordinates
########################################
tw          = line.twiss(method = '4d')
normalized  = tw.get_normalized_coordinates(
    particles,
    nemitt_x    = NEMITT_X,
    nemitt_y    = NEMITT_Y,
)
print('Requested x_norm:', X_NORM)
print('Requested px_norm:', PX_NORM)
print('Requested y_norm:', Y_NORM)
normalized.cols['particle_id x_norm px_norm y_norm py_norm'].show()

################################################################################
# Use equivalent geometric emittances
################################################################################

########################################
# Convert the emittance convention
########################################
beta_gamma  = line.particle_ref.beta0[0] * line.particle_ref.gamma0[0]
gemitt_x    = NEMITT_X / beta_gamma
gemitt_y    = NEMITT_Y / beta_gamma
print(f'Geometric emittances: {gemitt_x:.9e}, {gemitt_y:.9e} m rad')

########################################
# Build the same launch with geometric emittances
########################################
particles_geometric = line.build_particles(
    method      = '4d',
    x_norm      = X_NORM,
    px_norm     = PX_NORM,
    y_norm      = Y_NORM,
    gemitt_x    = gemitt_x,
    gemitt_y    = gemitt_y,
)

########################################
# Compare the physical coordinates
########################################
particles_geometric.get_table().cols['particle_id x px y py delta'].show()
print('Largest x difference [m]:',
      np.max(np.abs(particles_geometric.x - particles.x)))
print('Largest px difference:',
      np.max(np.abs(particles_geometric.px - particles.px)))

################################################################################
# Use Courant-Snyder coordinates without emittance scaling
################################################################################

########################################
# Recover the unscaled Floquet coordinates
########################################
# Omitting emittances gives unit geometric scaling: numerically gemitt = 1 m
# in each transverse plane. It does not assign a physical beam emittance.
# For this uncoupled, on-momentum lattice these are Courant-Snyder (Floquet)
# coordinates, with units sqrt(m), treating radians as dimensionless.
cs              = tw.get_normalized_coordinates(particles)
x_beta          = particles.x - tw.x[0]
px_beta         = particles.px - tw.px[0]
x_cs_formula    = x_beta / np.sqrt(tw.betx[0])
px_cs_formula   = (
    tw.alfx[0] * x_beta + tw.betx[0] * px_beta
) / np.sqrt(tw.betx[0])
cs_table = xt.Table(
    data    = {
        'particle_id': particles.particle_id,
        'X_CS':        cs.x_norm,
        'X_formula':   x_cs_formula,
        'P_CS':        cs.px_norm,
        'P_formula':   px_cs_formula,
    },
    index   = 'particle_id',
)
print('Unscaled Courant-Snyder coordinates [sqrt(m)]:')
cs_table.show()

########################################
# Build particles without nemitt or gemitt
########################################
# Reuse the small recovered CS coordinates, rather than interpreting the
# earlier sigma-scale numbers as sqrt(m) launch amplitudes.
particles_cs = line.build_particles(
    method  = '4d',
    x_norm  = cs.x_norm,
    px_norm = cs.px_norm,
    y_norm  = cs.y_norm,
    py_norm = cs.py_norm,
)

########################################
# Compare with explicit unit geometric scaling
########################################
particles_unit = line.build_particles(
    method      = '4d',
    x_norm      = cs.x_norm,
    px_norm     = cs.px_norm,
    y_norm      = cs.y_norm,
    py_norm     = cs.py_norm,
    gemitt_x    = 1.0,
    gemitt_y    = 1.0,
)
unit_comparison = xt.Table(
    data    = {
        'particle_id':  particles.particle_id,
        'x_original_m': particles.x,
        'x_no_emitt_m': particles_cs.x,
        'x_gemitt_1_m': particles_unit.x,
        'px_original':  particles.px,
        'px_no_emitt':  particles_cs.px,
        'px_gemitt_1':  particles_unit.px,
    },
    index   = 'particle_id',
)
unit_comparison.show()
