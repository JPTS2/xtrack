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
LATTICE_FILE    = 'test_data/example_lattices/ring/lattice.py'
LINE_NAME       = 'ring'

########################################
# Beam
########################################
SEED            = 12345
N_PARTICLES     = 4096
TOTAL_INTENSITY = 2.0e10                # Real particles
NEMITT_X        = 1.0e-6                # Normalized rms emittance [m rad]
NEMITT_Y        = 2.0e-6                # Normalized rms emittance [m rad]
X_CENTROID      = 2.0e-4                # Coherent horizontal offset [m]
SIGMA_ZETA      = 1.0e-2                # Imposed longitudinal rms [m]
SIGMA_DELTA     = 1.0e-5                # Imposed relative momentum rms

########################################
# Observation and covariance comparison
########################################
N_TURNS         = 32
ROTATION        = np.deg2rad(20)        # Transverse canonical rotation [rad]

################################################################################
# Load the model
################################################################################
repo_root = Path(__file__).resolve().parents[2]
env       = xt.load(repo_root / LATTICE_FILE)
line      = env[LINE_NAME]

################################################################################
# Prepare one weighted distribution
################################################################################

########################################
# Build the transverse shape and offset the centroid
########################################
# Use on-momentum optics for the transverse shape, without a dispersion term.
rng = np.random.default_rng(SEED)
particles = line.build_particles(
    method      = '4d',
    x_norm      = rng.standard_normal(N_PARTICLES),
    px_norm     = rng.standard_normal(N_PARTICLES),
    y_norm      = rng.standard_normal(N_PARTICLES),
    py_norm     = rng.standard_normal(N_PARTICLES),
    nemitt_x    = NEMITT_X,
    nemitt_y    = NEMITT_Y,
    zeta        = 0.0,
    delta       = 0.0,
)
particles.x += X_CENTROID

########################################
# Give the instantaneous distribution full longitudinal rank
########################################
# Full 6D covariance modes require longitudinal spread. Add it independently:
# this synthetic distribution is neither RF nor dispersively matched.
particles.zeta  = SIGMA_ZETA * rng.standard_normal(N_PARTICLES)
particles.delta = SIGMA_DELTA * rng.standard_normal(N_PARTICLES)

########################################
# Assign nonuniform macroparticle weights
########################################
# Independent weights distinguish intensity weighting from simply counting
# macroparticles. Every statistic below uses this same weighted population.
weights             = rng.uniform(0.5, 1.5, N_PARTICLES)
weights            *= TOTAL_INTENSITY / np.sum(weights)
particles.weight    = weights
initial             = particles.copy()

################################################################################
# Record moments and quantities derived from covariance
################################################################################

########################################
# Choose the statistics
########################################
emittance_stats = [
    'nemitt_x_projected', 'nemitt_y_projected',
    'nemitt_x', 'nemitt_y', 'nemitt_zeta',
]
monitor = xt.BeamStatsMonitor(
    start_at_turn   = 0,
    stop_at_turn    = N_TURNS,
    stats           = [
        'num_particles', 'mean_x', 'sigma_x',
        *emittance_stats,
    ],
)

########################################
# Sample the beam at the ring entrance
########################################
line.insert('beam_stats', monitor, at = 'obs.0@start')
line.track(particles, num_turns = N_TURNS)

################################################################################
# Derive the first sample directly from the input particles
################################################################################

########################################
# Calculate weighted central moments
########################################
# num_particles is sum(weight), not the macroparticle count. Central moments
# subtract the centroid, so a coherent offset does not inflate beam size.
mean_x      = np.average(initial.x, weights = initial.weight)
mean_px     = np.average(initial.px, weights = initial.weight)
var_x       = np.average((initial.x - mean_x)**2, weights = initial.weight)
var_px      = np.average((initial.px - mean_px)**2, weights = initial.weight)
cov_x_px    = np.average(
    (initial.x - mean_x) * (initial.px - mean_px),
    weights = initial.weight,
)
sigma_x = np.sqrt(var_x)

########################################
# Form a projected emittance from those moments
########################################
# sqrt(det of the plane's 2x2 covariance) is geometric projected emittance.
# Multiplication by reference beta0*gamma0 gives normalized emittance.
beta_gamma  = line.particle_ref.beta0[0] * line.particle_ref.gamma0[0]
nemitt_x    = beta_gamma * np.sqrt(var_x * var_px - cov_x_px**2)
print('Macroparticles and represented intensity:',
      N_PARTICLES, monitor.num_particles[0])
print('Direct and monitored centroid [m]:', mean_x, monitor.mean_x[0])
print('Direct and monitored rms size [m]:', sigma_x, monitor.sigma_x[0])
print('Direct and monitored projected emittance [m rad]:',
      nemitt_x, monitor.nemitt_x_projected[0])

################################################################################
# Distinguish centroid motion from rms beam size
################################################################################
# The initial beam is not dispersively matched, so its size can evolve too.
# Centroid motion and size evolution remain distinct observables.
fig, axes = plt.subplots(2, 1, sharex = True, figsize = (8, 6))
axes[0].plot(monitor.turns, monitor.mean_x * 1e3, '.-')
axes[0].set_ylabel('Centroid x [mm]')
axes[1].plot(monitor.turns, monitor.sigma_x * 1e3, '.-')
axes[1].set_ylabel('Rms size x [mm]')
axes[1].set_ylim(0, 1.15 * np.max(monitor.sigma_x) * 1e3)
axes[1].set_xlabel('Turn at ring entrance')
fig.tight_layout()

################################################################################
# Distinguish plane projections from covariance normal modes
################################################################################

########################################
# Rotate the same initial distribution canonically
########################################
# This is a separate instantaneous comparison, not a change to the tracked line.
# Rotating both position and canonical momentum pairs mixes the transverse
# planes while preserving the eigenemittances of the full covariance matrix.
rotated = initial.copy()
c       = np.cos(ROTATION)
s       = np.sin(ROTATION)
rotated.x   = c * initial.x + s * initial.y
rotated.y   = -s * initial.x + c * initial.y
rotated.px  = c * initial.px + s * initial.py
rotated.py  = -s * initial.px + c * initial.py

########################################
# Measure the rotated covariance
########################################
after = xt.BeamStatsMonitor(
    start_at_turn   = 0,
    stop_at_turn    = 1,
    stats           = emittance_stats,
)
after.track(rotated)

########################################
# Compare projections and mode invariants
########################################
# _projected names describe plane projections. Unqualified nemitt_x/y/zeta
# name covariance modes, labelled using their eigenvectors. In strongly mixed
# cases compare the set of eigenemittances as well as their individual labels.
result = xt.Table(
    data    = {
        'statistic': np.array(emittance_stats),
        'before':    np.array([getattr(monitor, nn)[0]
                               for nn in emittance_stats]),
        'after':     np.array([getattr(after, nn)[0]
                               for nn in emittance_stats]),
    },
    index   = 'statistic',
)
print('Normalized emittances before and after rotation [m rad]:')
result.show()
print('Covariance calculation status:',
      monitor.optics_from_covariance(turn = 0)['status'],
      after.optics_from_covariance(turn = 0)['status'])
plt.show()
