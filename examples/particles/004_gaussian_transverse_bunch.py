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
# Model
########################################
# Lattice file relative to the repository root.
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
LINE_NAME       = 'demo'

########################################
# Bunch
########################################
SEED            = 12345
N_PARTICLES     = 4096
TOTAL_INTENSITY = 2.0e10                # Real particles
NEMITT_X        = 1.0e-6                # Normalized rms emittance [m rad]
NEMITT_Y        = 2.0e-6                # Normalized rms emittance [m rad]
SIGMA_ZETA      = 1.0e-2                # [m]
SIGMA_DELTA     = 1.0e-4                # Relative momentum rms

################################################################################
# Load the model and optics
################################################################################
repo_root   = Path(__file__).resolve().parents[2]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env[LINE_NAME]
tw          = line.twiss(method = '4d')

################################################################################
# Build a weighted Gaussian bunch
################################################################################

########################################
# Draw the launch coordinates
########################################
# The seed fixes this input sample. Draw the transverse coordinates from
# unit-rms Gaussians; zeta and delta are imposed longitudinal inputs.
rng     = np.random.default_rng(SEED)
x_norm  = rng.standard_normal(N_PARTICLES)
px_norm = rng.standard_normal(N_PARTICLES)
y_norm  = rng.standard_normal(N_PARTICLES)
py_norm = rng.standard_normal(N_PARTICLES)
zeta    = SIGMA_ZETA * rng.standard_normal(N_PARTICLES)
delta   = SIGMA_DELTA * rng.standard_normal(N_PARTICLES)

########################################
# Transform the coordinates and assign macroparticle weights
########################################
particles = line.build_particles(
    method      = '4d',
    x_norm      = x_norm,
    px_norm     = px_norm,
    y_norm      = y_norm,
    py_norm     = py_norm,
    nemitt_x    = NEMITT_X,
    nemitt_y    = NEMITT_Y,
    zeta        = zeta,
    delta       = delta,
    weight      = TOTAL_INTENSITY / N_PARTICLES,
)
print(f'{len(particles.x)} macroparticles represent '
      f'{np.sum(particles.weight):.6e} real particles')

################################################################################
# Check the prepared beam sizes
################################################################################

########################################
# Calculate the input rms sizes
########################################
# In linear optics for this uncoupled lattice, independent betatron and
# dispersion contributions add in quadrature. Finite-sample moments fluctuate.
beta_gamma  = line.particle_ref.beta0[0] * line.particle_ref.gamma0[0]
gemitt_x    = NEMITT_X / beta_gamma
gemitt_y    = NEMITT_Y / beta_gamma
sigma_x     = np.sqrt(
    tw.betx[0] * gemitt_x + tw.dx[0]**2 * SIGMA_DELTA**2
)
sigma_y     = np.sqrt(
    tw.bety[0] * gemitt_y + tw.dy[0]**2 * SIGMA_DELTA**2
)

########################################
# Compare inputs with the particle sample
########################################
quantities  = np.array([
    'sigma_x [m]', 'sigma_y [m]', 'sigma_zeta [m]', 'sigma_delta',
])
input_rms   = np.array([sigma_x, sigma_y, SIGMA_ZETA, SIGMA_DELTA])
sample_rms  = np.array([
    np.std(particles.x), np.std(particles.y),
    np.std(particles.zeta), np.std(particles.delta),
])
moments = xt.Table(
    data    = {
        'quantity':   quantities,
        'input_rms':  input_rms,
        'sample_rms': sample_rms,
    },
    index   = 'quantity',
)
moments.show(maxwidth = 'full')

################################################################################
# Inspect the horizontal distribution
################################################################################
# Center the input Gaussian on the entrance closed orbit.
mean_x_mm   = tw.x[0] * 1e3
sigma_x_mm  = sigma_x * 1e3
x_grid      = np.linspace(
    mean_x_mm - 4 * sigma_x_mm, mean_x_mm + 4 * sigma_x_mm, 200,
)
density     = np.exp(-0.5 * ((x_grid - mean_x_mm) / sigma_x_mm)**2)
density /= np.sqrt(2 * np.pi) * sigma_x_mm

fig, ax = plt.subplots(figsize = (7, 4))
ax.hist(
    particles.x * 1e3,
    bins    = 50,
    density = True,
    alpha   = 0.5,
    label   = 'Particle sample',
)
ax.plot(x_grid, density, label = 'Input Gaussian')
ax.set_xlabel('x [mm]')
ax.set_ylabel('Probability density [1/mm]')
ax.set_title('Prepared bunch at the ring entrance')
ax.legend()
fig.tight_layout()
plt.show()
