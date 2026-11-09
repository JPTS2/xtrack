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
plt.show()
