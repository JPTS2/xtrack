# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
# get_amplitude_detuning_coefficients additionally requires nafflib.
import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# One-turn surrogate and its geometric-action detuning
########################################
P0C             = 7.0e9                 # Proton reference p0*c [eV]
QX              = 0.28
QY              = 0.31
BETX            = 2.0                   # [m], alpha_x = 0
BETY            = 3.0                   # [m], alpha_y = 0
DET_XX          = 1000.0                # dQx/dJx [m^-1]
DET_XY          = 200.0                 # dQx/dJy = dQy/dJx [m^-1]
DET_YY          = 1500.0                # dQy/dJy [m^-1]

########################################
# Tracking amplitudes and emittance scale
########################################
N_TURNS         = 256
NEMITT          = 1.0e-6                # Normalized emittance scale [m]
A_SMALL         = 0.05                  # Nonzero amplitude in the other plane
A1              = 0.5                   # First finite-difference amplitude
A2              = 1.0                   # Second finite-difference amplitude
A_SCAN          = np.array([0.5, 1.0, 1.5, 2.0])  # Normalized x amplitudes

################################################################################
# Construct an explicitly amplitude-dependent map
################################################################################
# This is an analytic one-turn surrogate, not an octupole lattice. It isolates
# Qx = QX + DET_XX*Jx + DET_XY*Jy and the corresponding y-plane relation.
# Equal cross coefficients give a common action-dependent Hamiltonian.
line = xt.Line(elements = [xt.LineSegmentMap(
    qx      = QX,
    qy      = QY,
    betx    = BETX,
    bety    = BETY,
    det_xx  = DET_XX,
    det_xy  = DET_XY,
    det_yx  = DET_XY,
    det_yy  = DET_YY,
)])
line.set_particle_ref('proton', p0c = P0C)
# No RF, radiation, damping or excitation is included; delta = 0 throughout.

################################################################################
# Measure the four amplitude-detuning coefficients
################################################################################

########################################
# Define the action convention
########################################
# J = (gamma_Twiss*x^2 + 2*alpha*x*px + beta*px^2)/2 is geometric action [m],
# with gamma_Twiss = (1 + alpha^2)/beta, and no extra factor of 2*pi.
# For a normalized launch (x_norm, px_norm) = (a, 0), J = a^2*gemitt/2.
# The emittance sets the launch scale, rather than a statistical beam sample.
beta_gamma  = float(line.particle_ref.beta0[0] * line.particle_ref.gamma0[0])
gemitt      = NEMITT / beta_gamma
print(f'Geometric emittance scale: {gemitt:.6e} m')
print('Finite-difference actions [m]:', gemitt * np.array([A1, A2])**2 / 2)

########################################
# Extract finite differences of tracked tunes
########################################
detuning = line.get_amplitude_detuning_coefficients(
    nemitt_x    = NEMITT,
    nemitt_y    = NEMITT,
    num_turns   = N_TURNS,
    a0_sigmas   = A_SMALL,
    a1_sigmas   = A1,
    a2_sigmas   = A2,
)
print('Coefficient    Measured [m^-1]    Map input [m^-1]')
for name, expected in [
    ('det_xx', DET_XX), ('det_xy', DET_XY),
    ('det_yx', DET_XY), ('det_yy', DET_YY),
]:
    print(f'{name:10s} {detuning[name]:18.6f} {expected:19.6f}')

################################################################################
# Compare the action-frequency relation with an independent phase measurement
################################################################################

########################################
# Track a horizontal action scan at fixed vertical action
########################################
particles = line.build_particles(
    method      = '4d',
    x_norm      = A_SCAN,
    y_norm      = A_SMALL,
    nemitt_x    = NEMITT,
    nemitt_y    = NEMITT,
    delta       = 0.0,
)
line.track(particles, num_turns = N_TURNS, turn_by_turn_monitor = True)
record  = line.record_last_track
jx      = gemitt * A_SCAN**2 / 2
jy      = gemitt * A_SMALL**2 / 2

########################################
# Unwrap the phase and compare with the measured coefficients
########################################
# alpha = 0 here, so atan2(-beta*px, x) advances by 2*pi*Q per turn.
# These tunes stay between 0 and 0.5, avoiding phase and real-signal aliases.
# A general lattice requires a suitable tune estimator and resonance checks.
phase_x         = np.unwrap(np.arctan2(-BETX * record.px, record.x), axis = 1)
phase_y         = np.unwrap(np.arctan2(-BETY * record.py, record.y), axis = 1)
qx_scan         = np.mean(np.diff(phase_x, axis = 1), axis = 1) / (2 * np.pi)
qy_scan         = np.mean(np.diff(phase_y, axis = 1), axis = 1) / (2 * np.pi)
qx_prediction   = QX + detuning['det_xx'] * jx + detuning['det_xy'] * jy
qy_prediction   = QY + detuning['det_yx'] * jx + detuning['det_yy'] * jy
print('Jx [m]          Measured Qx      Measured Qy')
for action, qx, qy in zip(jx, qx_scan, qy_scan):
    print(f'{action:.6e}    {qx:.10f}     {qy:.10f}')
print('Maximum tune prediction errors:',
      np.max(np.abs(qx_scan - qx_prediction)),
      np.max(np.abs(qy_scan - qy_prediction)))

# The API uses finite amplitudes and a finite record. In a magnet lattice,
# higher action terms and tune-estimation precision can affect the result;
# coefficients should be checked against amplitude and tracking-length changes.

################################################################################
# Plot tune shifts versus geometric action
################################################################################
fig, axes = plt.subplots(2, 1, sharex = True, figsize = (7, 6))
for ax, measured, predicted, tune0, plane in [
    (axes[0], qx_scan, qx_prediction, QX, 'x'),
    (axes[1], qy_scan, qy_prediction, QY, 'y'),
]:
    ax.plot(jx * 1e9, measured - tune0, 'o', label = 'Tracked phase advance')
    ax.plot(jx * 1e9, predicted - tune0, '-', label = 'Measured detuning slope')
    ax.set_ylabel(f'Q_{plane} - Q_{plane}(0, 0)')
    ax.legend()
axes[-1].set_xlabel('Geometric Jx [nm], Jy fixed')
fig.tight_layout()
plt.show()
