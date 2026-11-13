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
from scipy.constants import c
import xpart as xp
import xtrack as xt

################################################################################
# User parameters
################################################################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
PROTON_P0C      = 0.5e9                 # Initial p0*c [eV]
PROTON_P0C_END  = 0.6e9                 # Program endpoint beyond recorded turns
PROTON_VOLTAGE  = 3e5                   # [V], below transition
PROTON_TURNS    = 512
SAMPLE_EVERY    = 32                    # Track between optics snapshots
NUM_PARTICLES   = 64
SEED            = 287
NEMITT_X        = 1e-9                  # Normalized rms emittance [m]
NEMITT_Y        = 1e-9                  # [m]
SIGMA_ZETA      = 1e-3                  # Initial rms bunch length [m]
PROTON_QX       = 8.28                  # Retune away from the bare integer
PROTON_QY       = 10.19

################################################################################
# Locate the native ring definition
################################################################################
repo_root   = Path(__file__).resolve().parents[3]
################################################################################
# Follow transverse action and beam size during proton acceleration
################################################################################

########################################
# Reuse the geometry at a bounded below-transition operating point
########################################
proton_env  = xt.load(repo_root / LATTICE_FILE)
proton_line = proton_env['demo']
proton_line.set_particle_ref('proton', p0c = PROTON_P0C)
# Move away from the nominal near-integer vertical tune before acceleration.
# The normalized magnet strengths then stay fixed: physical fields follow p0.
proton_line.match(
    method  = '4d',
    vary    = xt.VaryList(['kqf', 'kqd'], step = 1e-5),
    targets = xt.TargetSet(qx = PROTON_QX, qy = PROTON_QY, tol = 1e-8),
)
proton_duration                 = ((PROTON_TURNS + 2) * proton_line.get_length()
                   / (proton_line.particle_ref.beta0[0] * c))
proton_line.energy_program      = xt.EnergyProgram(
    t_s     = [0.0, proton_duration],
    p0c     = [PROTON_P0C, PROTON_P0C_END],
)
proton_env['frequency_rf']      = 0.0
proton_line['rf.1'].harmonic    = proton_env['harmonic_rf']
proton_env['voltage_rf']        = PROTON_VOLTAGE
proton_gain                     = (
    proton_line.energy_program.get_kinetic_energy0_at_t_s(
        proton_line.energy_program.get_t_s_at_turn(1))
    - proton_line.energy_program.get_kinetic_energy0_at_t_s(0.0))
proton_env['phase_rf']          = np.arcsin(proton_gain / PROTON_VOLTAGE)
proton_env['t_turn_s']          = 0.0
proton_initial_optics           = proton_line.twiss6d(chrom = False)
print('Proton reference gain per turn [eV], RF phase [rad], slip factor:',
      proton_gain, proton_env['phase_rf'], proton_initial_optics.slip_factor)
# For positive protons below transition, asin(DeltaE/(q0*V)) is focusing.
# This is a second illustrative setting, not a proton machine design.

########################################
# Generate a small matched Gaussian bunch with native Xpart
########################################
np.random.seed(SEED)
bunch = xp.generate_matched_gaussian_bunch(
    line            = proton_line,
    engine          = 'linear',
    num_particles   = NUM_PARTICLES,
    nemitt_x        = NEMITT_X,
    nemitt_y        = NEMITT_Y,
    sigma_z         = SIGMA_ZETA,
)
# Linear matching and a small longitudinal spread isolate transverse damping;
# this sample does not qualify bucket acceptance or nonlinear matching.

########################################
# Recompute normalized coordinates at the current applied reference
########################################
transverse_samples = []
for turn in range(0, PROTON_TURNS + 1, SAMPLE_EVERY):
    if turn:
        proton_line.enable_time_dependent_vars = True
        proton_line.track(bunch, num_turns = SAMPLE_EVERY)
    proton_line.enable_time_dependent_vars = False
    # Use instantaneous 4D transverse optics only for this diagnostic. RF
    # kicks remain active in tracking; slow acceleration and weak coupling
    # make the transverse action an approximate adiabatic invariant.
    optics      = proton_line.twiss4d(chrom = False)
    normalized  = optics.get_normalized_coordinates(
        bunch, nemitt_x = NEMITT_X, nemitt_y = NEMITT_Y)
    beta_gamma  = (optics.particle_on_co.beta0[0]
                  * optics.particle_on_co.gamma0[0])
    # Centered action moments exclude coherent centroid motion. Equal particle
    # weights make ordinary population variances appropriate here.
    action_x    = (np.var(normalized.x_norm) + np.var(normalized.px_norm)) / 2
    action_y    = (np.var(normalized.y_norm) + np.var(normalized.py_norm)) / 2
    transverse_samples.append((turn, beta_gamma, action_x, action_y,
                               np.std(bunch.x), np.std(bunch.y),
                               optics.betx[0], optics.bety[0]))
# Twiss uses the last applied reference, as do the tracked particle momenta.
# Normalizing with a single initial table would miss the beta*gamma change.
transverse_samples = np.array(transverse_samples)

########################################
# Compare phase-averaged betatron sizes with beta-gamma scaling
########################################
turns       = transverse_samples[:, 0]
beta_gamma  = transverse_samples[:, 1]
size_ratio  = np.sqrt(beta_gamma[0] / beta_gamma)
fig, axes   = plt.subplots(2, 1, sharex = True, figsize = (8, 6))
for label, action_column, size_column, beta_column, emittance in [
    ('x', 2, 4, 6, NEMITT_X),
    ('y', 3, 5, 7, NEMITT_Y),
]:
    actions         = transverse_samples[:, action_column]
    betatron_size   = np.sqrt(
        transverse_samples[:, beta_column] * emittance * actions / beta_gamma)
    axes[0].plot(turns, actions / actions[0], label = label)
    axes[1].plot(turns, betatron_size / betatron_size[0],
                 label = label + ': betatron (phase average)')
    axes[1].plot(turns,
                 transverse_samples[:, size_column]
                 / transverse_samples[0, size_column],
                 ':', alpha = 0.5, label = label + ': sample rms size')
    print(label, 'normalized centered-action min/max relative to initial:',
          np.min(actions / actions[0]), np.max(actions / actions[0]))
axes[1].plot(turns, size_ratio, 'k--',
             label = 'sqrt(initial / current beta*gamma)')
axes[0].set_ylabel('Centered normalized action / initial')
axes[1].set_ylabel('Beam size / initial')
axes[1].set_xlabel('Tracked turns')
for axis in axes:
    axis.grid()
    axis.legend()
fig.tight_layout()
print('Proton final state set / reference p0*c [GeV]:',
      np.unique(bunch.state), bunch.p0c[0] * 1e-9)
print('Expected phase-averaged betatron size ratio:', size_ratio[-1])
# Phase-averaged size uses the measured centered action and local beta.
# A finite Gaussian sample has phase-dependent rms-size oscillations. The
# approximate action comparison does not remove chromatic or RF coupling.
plt.show()
