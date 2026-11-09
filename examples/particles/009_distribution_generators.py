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
import xpart as xp
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Distributions in normalized phase space
########################################
SEED            = 12345
NUM_PARTICLES   = 1024
RADII           = np.array([1.0, 2.0, 3.0])
NUM_PHASES      = 16
SECTOR_RADII    = (1.0, 3.0)
PENCIL_CUT      = 4.0                     # Positive position cut [sigma]
PENCIL_WIDTH    = 0.2                     # Radial thickness [sigma]

########################################
# Transform the grid to a physical launch
########################################
LATTICE_FILE    = 'test_data/example_lattices/ring/lattice.py'
NEMITT_X        = 1e-6                    # Normalized rms emittance [m rad]
NEMITT_Y        = 2e-6                    # [m rad]

################################################################################
# Generate sample clouds and an amplitude scan
################################################################################
# These Xpart helpers use NumPy's legacy random stream, seeded here explicitly.
np.random.seed(SEED)

########################################
# Independent unit Gaussian coordinates
########################################
gaussian_x, gaussian_px = xp.generate_2D_gaussian(NUM_PARTICLES)

########################################
# A uniform annulus in phase-space area
########################################
# Uniform area means uniform r**2, rather than uniform radius.
sector_x, sector_px, sector_r, sector_phase = (
    xp.generate_2D_uniform_circular_sector(
        num_particles   = NUM_PARTICLES,
        r_range         = SECTOR_RADII))

########################################
# A regular grid in amplitude and phase
########################################
# Avoid duplicating phase zero at 2*pi. This grid is a deterministic scan,
# rather than a random bunch or an assignment of macroparticle weights.
phases = np.linspace(0, 2 * np.pi, NUM_PHASES, endpoint = False)
grid_x, grid_px, grid_r, grid_phase = xp.generate_2D_polar_grid(
    r_grid      = RADII,
    theta_grid  = phases)

########################################
# A pencil beyond a positive position cut
########################################
pencil_x, pencil_px, pencil_r, pencil_phase = xp.generate_2D_pencil(
    num_particles   = NUM_PARTICLES,
    pos_cut_sigmas  = PENCIL_CUT,
    dr_sigmas       = PENCIL_WIDTH,
    side            = '+')

################################################################################
# Inspect the normalized distributions
################################################################################
fig, axes = plt.subplots(2, 2, figsize = (9, 8))
for ax, x, px, title in [
    (axes[0, 0], gaussian_x, gaussian_px, 'Unit Gaussian'),
    (axes[0, 1], sector_x, sector_px, 'Uniform phase-space area'),
    (axes[1, 0], grid_x, grid_px, 'Amplitude and phase grid'),
    (axes[1, 1], pencil_x, pencil_px, 'Positive-side pencil'),
]:
    ax.scatter(x, px, s = 5)
    ax.set_xlabel('x_norm')
    ax.set_ylabel('px_norm')
    ax.set_title(title)
    ax.set_aspect('equal', adjustable = 'datalim')
    ax.grid()
axes[1, 1].axvline(PENCIL_CUT, color = 'C1', label = 'Position cut')
axes[1, 1].legend()
fig.tight_layout()
plt.show()

################################################################################
# Build physical particles from the normalized scan
################################################################################
repo_root   = Path(__file__).resolve().parents[2]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env['ring']
# With these emittances, x_norm and px_norm are dimensionless sigma-scaled
# Floquet coordinates. The vertical launch and longitudinal offsets are zero.
particles = line.build_particles(
    method      = '4d',
    x_norm      = grid_x,
    px_norm     = grid_px,
    nemitt_x    = NEMITT_X,
    nemitt_y    = NEMITT_Y)
print('First eight grid particles in physical coordinates:')
particles.get_table().rows[:8].cols[
    'particle_id x px y py'].show(maxwidth = 'full')
