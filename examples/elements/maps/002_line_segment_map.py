# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt
from scipy.constants import c

################################################################################
# User parameters
################################################################################

########################################
# Reference and assigned transverse optics
########################################
P0C             = 1.0e9                 # Proton reference p0*c [eV]
LENGTH          = 100.0                 # Surrogate segment length [m]
QX              = 0.31                  # Horizontal phase advance [turns/pass]
QY              = 0.32                  # Vertical phase advance [turns/pass]
BETX            = 12.0                  # Entrance = exit beta_x [m]
BETY            = 8.0                   # Entrance = exit beta_y [m]
ALFX            = 0.2
ALFY            = -0.3

########################################
# Assigned longitudinal rotation
########################################
QS              = 0.02                  # Longitudinal advance [turns/pass]
BETS            = 100.0                 # Longitudinal beta parameter [m]

########################################
# Physical entrance coordinates and recording
########################################
X_START         = 1.0e-3                # [m]
PX_START        = 1.0e-4                # P_x / p0
Y_START         = -0.7e-3               # [m]
PY_START        = 2.0e-5                # P_y / p0
ZETA_START      = 0.03                  # [m]
DELTA_START     = 2.0e-4                # (p - p0) / p0
N_TURNS         = 64

########################################
# Different optics and orbit at the two ends
########################################
BETX_EXIT       = 24.0                  # [m]
X_REF           = [1e-3, -2e-3]          # Entrance/exit reference x [m]
PX_REF          = [2e-5, -3e-5]          # Entrance/exit reference px

########################################
# Repeated linear RF buckets
########################################
RF_FREQUENCY    = 100e6                 # [Hz]
RF_VOLTAGE      = 1e6                   # [V]
ALPHA_C         = 0.01                  # Momentum compaction

########################################
# Prescribed damping and excitation
########################################
DAMPING_RATE    = 0.1                   # Fractional x reduction per pass
NOISE_SIGMA     = 1e-4                  # Added Gaussian x kick RMS [m]
N_PARTICLES     = 2048

################################################################################
# Define the idealized segment
################################################################################
# Scalar beta/alpha values apply at both ends. The three phase advances are
# assigned inputs; this map does not define a magnetic lattice or RF bucket.
segment = xt.LineSegmentMap(
    length              = LENGTH,
    qx                  = QX,
    qy                  = QY,
    betx                = BETX,
    bety                = BETY,
    alfx                = ALFX,
    alfy                = ALFY,
    longitudinal_mode   = 'linear_fixed_qs',
    qs                  = QS,
    bets                = BETS,
)
line = xt.Line(elements = [segment], element_names = ['segment'])
line.set_particle_ref('proton', p0c = P0C)

################################################################################
# Inspect periodic optics of the assigned map
################################################################################
tw = line.twiss()
print('Assigned qx, qy, qs [turns/pass]:', QX, QY, QS)
print('Twiss qx, qy, qs [turns/pass]:', tw.qx, tw.qy, tw.qs)
tw.cols['s betx alfx bety alfy'].show()

################################################################################
# Track repeated passes
################################################################################
initial = line.build_particles(
    x       = X_START,
    px      = PX_START,
    y       = Y_START,
    py      = PY_START,
    zeta    = ZETA_START,
    delta   = DELTA_START,
)
particles   = initial.copy()
line.track(particles, num_turns = N_TURNS, turn_by_turn_monitor = True)
record      = line.record_last_track
turns       = record.at_turn[0]

################################################################################
# Compare records with the assigned rotations
################################################################################

########################################
# Transverse rotations with the chosen beta and alpha
########################################
mux         = 2 * np.pi * QX * turns
muy         = 2 * np.pi * QY * turns
expected_x  = (np.cos(mux) + ALFX * np.sin(mux)) * X_START + (
    BETX * np.sin(mux) * PX_START)
expected_y  = (np.cos(muy) + ALFY * np.sin(muy)) * Y_START + (
    BETY * np.sin(muy) * PY_START)

########################################
# Longitudinal rotation in zeta and pzeta
########################################
# This convention gives zeta_out = cos(mu_s)*zeta - BETS*sin(mu_s)*pzeta.
mus = 2 * np.pi * QS * turns
expected_zeta = ZETA_START * np.cos(mus) - (
    BETS * np.sin(mus) * initial.pzeta[0])

################################################################################
# Plot turn-entrance coordinates
################################################################################
fig, axes = plt.subplots(3, 1, sharex = True, figsize = (9, 8))
for ax, coordinate, expected in zip(
    axes, ['x', 'y', 'zeta'], [expected_x, expected_y, expected_zeta]
):
    ax.plot(turns, expected * 1e3, label = 'Assigned rotation')
    ax.plot(turns, getattr(record, coordinate)[0] * 1e3, '.', label = 'Tracked')
    ax.set_ylabel(coordinate + ' [mm]')
    ax.legend()
    ax.grid()
axes[-1].set_xlabel('Pass number (recorded at entrance)')
fig.tight_layout()

################################################################################
# Assign different entrance and exit optics
################################################################################

########################################
# Separate the reference orbit from motion about it
########################################
transport = xt.LineSegmentMap(
    length              = LENGTH,
    qx                  = QX,
    qy                  = QY,
    betx                = [BETX, BETX_EXIT],
    bety                = BETY,
    alfx                = [ALFX, -ALFX],
    alfy                = ALFY,
    x_ref               = X_REF,
    px_ref              = PX_REF,
    longitudinal_mode   = 'frozen',
)
transport_line = xt.Line(elements = [transport])
transport_line.particle_ref = line.particle_ref.copy()
open_twiss = transport_line.twiss(
    method  = '4d',
    betx    = BETX,
    alfx    = ALFX,
    bety    = BETY,
    alfy    = ALFY,
    x       = X_REF[0],
    px      = PX_REF[0],
)
open_twiss.cols['s x px betx alfx'].show()
reference_probe = transport_line.build_particles(x = X_REF[0], px = PX_REF[0])
transport.track(reference_probe)
print('Assigned and tracked reference exit x [m]:',
      X_REF[1], reference_probe.x[0])
print('Assigned and tracked reference exit px:',
      PX_REF[1], reference_probe.px[0])

################################################################################
# Repeat a linear RF rotation in neighboring buckets
################################################################################

########################################
# Derive the rotation from fixed RF parameters
########################################
# Here eta < 0, so the derived longitudinal beta is negative.
# The model repeats a linear rotation; it does not supply a separatrix.
rf_segment = xt.LineSegmentMap(
    length                      = LENGTH,
    qx                          = QX,
    qy                          = QY,
    longitudinal_mode           = 'linear_fixed_rf',
    voltage_rf                  = RF_VOLTAGE,
    frequency_rf                = RF_FREQUENCY,
    momentum_compaction_factor  = ALPHA_C,
)
rf_line = xt.Line(elements = [rf_segment])
rf_line.particle_ref = line.particle_ref.copy()
bucket_spacing = line.particle_ref.beta0[0] * c / RF_FREQUENCY
eta = ALPHA_C - 1 / line.particle_ref.gamma0[0]**2
print('Slip factor and bucket spacing [m]:', eta, bucket_spacing)

########################################
# Compare the same launch relative to two bucket centres
########################################
rf_particles = rf_line.build_particles(
    zeta    = [ZETA_START, ZETA_START + bucket_spacing],
    delta   = DELTA_START,
)
rf_line.track(rf_particles, num_turns = N_TURNS, turn_by_turn_monitor = True)
rf_record = rf_line.record_last_track
print('Largest bucket-centred zeta difference [m]:',
      np.max(np.abs(rf_record.zeta[1] - bucket_spacing - rf_record.zeta[0])))
print('Largest inter-bucket momentum difference:',
      np.max(np.abs(rf_record.pzeta[1] - rf_record.pzeta[0])))

################################################################################
# Prescribe damping and excitation independently of a physical lattice
################################################################################

########################################
# Prepare native random streams and track the prescribed process
########################################
# This map applies x <- (1-DAMPING_RATE)*x + Gaussian(NOISE_SIGMA).
# These are assigned coefficients, not damping/excitation computed from magnets.
noisy_segment = xt.LineSegmentMap(
    longitudinal_mode   = 'frozen',
    damping_rate_x      = DAMPING_RATE,
    gauss_noise_ampl_x  = NOISE_SIGMA,
)
noisy_line = xt.Line(elements = [noisy_segment])
noisy_line.particle_ref = line.particle_ref.copy()
noisy_particles = noisy_line.build_particles(x = np.zeros(N_PARTICLES))
# Sampling initializes each particle's stream before the stochastic map uses it.
normal_sampler = xt.RandomNormal()
first_draw = normal_sampler.generate(
    n_samples   = N_PARTICLES,
    particles   = noisy_particles,
)
print('Native standard-normal sample mean and RMS:',
      np.mean(first_draw), np.std(first_draw))
noisy_line.track(
    noisy_particles,
    num_turns               = N_TURNS,
    turn_by_turn_monitor    = True,
)
noisy_record = noisy_line.record_last_track
noise_turns = noisy_record.at_turn[0]
measured_sigma = np.std(noisy_record.x, axis = 0)
expected_sigma = NOISE_SIGMA * np.sqrt(
    (1 - (1 - DAMPING_RATE)**(2 * noise_turns))
    / (1 - (1 - DAMPING_RATE)**2))
print('Assigned stationary x RMS [m]:',
      NOISE_SIGMA / np.sqrt(1 - (1 - DAMPING_RATE)**2))
print('Final recorded x RMS [m]:', measured_sigma[-1])

########################################
# Compare the ensemble RMS with the discrete process
########################################
fig_noise, ax_noise = plt.subplots(figsize = (8, 4))
ax_noise.plot(noise_turns, expected_sigma * 1e3, label = 'Assigned process')
ax_noise.plot(noise_turns, measured_sigma * 1e3, '.', label = 'Native tracking')
ax_noise.set_xlabel('Pass number (recorded at entrance)')
ax_noise.set_ylabel('Horizontal RMS [mm]')
ax_noise.set_title('Prescribed damping and Gaussian excitation')
ax_noise.legend()
ax_noise.grid()
fig_noise.tight_layout()
plt.show()
