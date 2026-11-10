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
# Same ideal proton gap, followed by explicit magnetic transport
########################################
KINETIC_START   = 100e6                 # Entrance reference kinetic energy [eV]
VOLTAGE         = 5e6                   # [V]
FREQUENCY       = 100e6                 # [Hz]
PHASE           = np.pi / 2             # Crest at zeta=0 [rad]
DRIFT_LENGTHS   = [0.5, 1.0, 0.5]       # [m]
QUAD_LENGTH     = 0.2                   # [m]
K1F             = 0.8                   # [m^-2], normalized to entrance p0
K1D             = -0.6                  # [m^-2]

########################################
# Open entrance optics and deterministic transverse population
########################################
BETX_START      = 2.0                   # Canonical beta_x [m]
BETY_START      = 3.0                   # [m]
ALFX_START      = 0.2
ALFY_START      = -0.1
NEMITT_X        = 1e-8                  # Entrance-normalized emittance [m]
NEMITT_Y        = 2e-8                  # [m]
N_PARTICLES     = 64

################################################################################
# Calculate open optics with the RF energy kick active
################################################################################

########################################
# Keep the entrance p0 fixed throughout this accelerating line
########################################
line    = xt.Line(elements = {
    'entrance': xt.Marker(),
    'gap': xt.Cavity(voltage = VOLTAGE, frequency = FREQUENCY, phase = PHASE),
    'd0': xt.Drift(length = DRIFT_LENGTHS[0]),
    'qf': xt.Quadrupole(length = QUAD_LENGTH, k1 = K1F),
    'd1': xt.Drift(length = DRIFT_LENGTHS[1]),
    'qd': xt.Quadrupole(length = QUAD_LENGTH, k1 = K1D),
    'd2': xt.Drift(length = DRIFT_LENGTHS[2]),
    'exit': xt.Marker(),
})
line.set_particle_ref('proton', kinetic_energy0 = KINETIC_START)
# Physical acceleration is represented by increasing delta at this FIXED p0.
# No reference rebasing is performed; all px/py retain the entrance reference.
# The thin gap omits transverse RF focusing and any realistic field map.

########################################
# Use 6D open Twiss rather than suppressing the RF energy kick
########################################
tw          = line.twiss(
    method  = '6d',
    chrom   = False,
    betx    = BETX_START,
    bety    = BETY_START,
    alfx    = ALFX_START,
    alfy    = ALFY_START,
)

# method='4d' suppresses cavity energy kicks and would describe another map.
# These are propagated entrance conditions, not a periodic RF equilibrium.
tw.cols['name s delta betx bety alfx alfy'].show(maxwidth = 'full')
covariance  = tw.get_beam_covariance(nemitt_x = NEMITT_X, nemitt_y = NEMITT_Y)

################################################################################
# Prepare particles using the native entrance optical matrix
################################################################################
phase       = 2 * np.pi * np.arange(N_PARTICLES) / N_PARTICLES
particles   = line.build_particles(
    particle_on_co  = tw.particle_on_co,
    W_matrix        = tw.W_matrix[0],
    x_norm          = np.sqrt(2) * np.cos(phase),
    px_norm         = np.sqrt(2) * np.sin(phase),
    y_norm          = np.sqrt(2) * np.cos(3 * phase),
    py_norm         = np.sqrt(2) * np.sin(3 * phase),
    nemitt_x        = NEMITT_X,
    nemitt_y        = NEMITT_Y,
    zeta            = 0.0,
    delta           = 0.0,
)
initial     = particles.copy()

# Orthogonal phase harmonics give the intended transverse second moments.
# This is a deterministic transverse sample, not a Gaussian or RF-matched bunch.

################################################################################
# Track one physical passage and compare population moments with optics
################################################################################

########################################
# Record element entrances and the final exit using the native monitor
########################################
line.track(particles, num_turns = 1, turn_by_turn_monitor = 'ONE_TURN_EBE')
record      = line.record_last_track
sigma_x     = np.std(record.x, axis = 0)
sigma_y     = np.std(record.y, axis = 0)
print('Physical RF energy gain, min/max [eV]:',
      np.min(particles.energy - initial.energy),
      np.max(particles.energy - initial.energy))
print('Fixed reference p0*c [eV]:', particles.p0c[0])

########################################
# Distinguish normalized momentum from geometric slope at the gap
########################################
print('Rms px before/after gap:',
      np.std(record.px[:, 1]), np.std(record.px[:, 2]))
print('Rms geometric x slope before/after gap:',
      np.std(record.kin_xp[:, 1]), np.std(record.kin_xp[:, 2]))
print('Maximum relative covariance/tracking width differences:',
      np.max(np.abs(sigma_x / covariance.sigma_x - 1)),
      np.max(np.abs(sigma_y / covariance.sigma_y - 1)))

# Longitudinal acceleration leaves transverse physical momentum unchanged in
# this ideal gap, while the slope decreases. The covariance uses canonical
# momenta normalized to the SAME entrance p0 as the tracking records.

################################################################################
# Plot predicted and measured transverse rms sizes along the accelerating line
################################################################################
fig, axes = plt.subplots(2, 1, sharex = True, figsize = (8, 6))
for axis, predicted, measured, plane in [
    (axes[0], covariance.sigma_x, sigma_x, 'x'),
    (axes[1], covariance.sigma_y, sigma_y, 'y'),
]:
    axis.plot(tw.s, 1e3 * predicted, label = 'Native covariance')
    axis.plot(record.s[0], 1e3 * measured, 'o', label = 'Tracked population')
    axis.set_ylabel(f'rms {plane} [mm]')
    axis.grid()
    axis.legend()
axes[1].set_xlabel('s [m]')
fig.tight_layout()
plt.show()
