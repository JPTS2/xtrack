# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c

import xpart as xp
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Ideal ring with linearized RF focusing
########################################
P0C             = 3e9                     # Proton reference p0*c [eV]
CIRCUMFERENCE   = 100.0                   # [m]
ALPHA_C         = 0.2                     # Momentum compaction
HARMONIC        = 20
VOLTAGE         = 1e5                     # RF voltage [V]
QX              = 0.31
QY              = 0.32
BETX            = 10.0                    # Assigned transverse beta [m]
BETY            = 12.0                    # [m]

########################################
# Matched Gaussian bunch
########################################
SEED            = 12345
NUM_PARTICLES   = 4096
INTENSITY       = 1e10                    # Real particles
NEMITT_X        = 1e-6                    # Normalized rms emittance [m rad]
NEMITT_Y        = 2e-6                    # [m rad]
SIGMA_ZETA      = 0.02                    # Target rms bunch length [m]
NUM_TURNS       = 256

################################################################################
# Define a stable linear longitudinal model
################################################################################

########################################
# Set reference and RF frequency
########################################
reference   = xt.Particles(pdg_id_0 = 'proton', p0c = P0C)
frequency   = HARMONIC * reference.beta0[0] * c / CIRCUMFERENCE

########################################
# Assign transverse optics and linear RF transport
########################################
# This one-turn surrogate has no individual magnets or cavity locations.
# RF is linearized about its stable fixed point: there is no finite bucket
# boundary, acceleration, radiation, collective force or aperture model.
# This mode assumes stable focusing; RF phase is not a stability control here.
line = xt.Line(elements = [xt.LineSegmentMap(
    length                      = CIRCUMFERENCE,
    qx                          = QX,
    qy                          = QY,
    betx                        = BETX,
    bety                        = BETY,
    longitudinal_mode           = 'linear_fixed_rf',
    voltage_rf                  = VOLTAGE,
    frequency_rf                = frequency,
    slippage_length             = CIRCUMFERENCE,
    momentum_compaction_factor  = ALPHA_C)])
line.particle_ref = reference
tw = line.twiss()
print('Slip factor:', tw.slip_factor)
print('Synchrotron tune:', tw.qs)
print('Longitudinal beta bets0 [m]:', tw.bets0)

################################################################################
# Generate a bunch matched to the linear map
################################################################################

########################################
# Generate transverse and longitudinal coordinates together
########################################
# The linear engine uses sigma_delta = sigma_zeta / abs(bets0); the two
# longitudinal coordinates are independent Gaussians at this map's entrance.
np.random.seed(SEED)
particles = xp.generate_matched_gaussian_bunch(
    line                        = line,
    engine                      = 'linear',
    num_particles               = NUM_PARTICLES,
    total_intensity_particles   = INTENSITY,
    nemitt_x                    = NEMITT_X,
    nemitt_y                    = NEMITT_Y,
    sigma_z                     = SIGMA_ZETA)
initial = particles.copy()

########################################
# Track the same ensemble
########################################
line.track(particles, num_turns = NUM_TURNS)

################################################################################
# Compare input sizes with the prepared and tracked bunch
################################################################################
# Matching here is to linear, small-momentum-spread accuracy: Xpart samples
# delta, while this map rotates pzeta, equal to delta only to first order.
# Finite-sample moments also fluctuate as individual particles oscillate.
sigma_delta = SIGMA_ZETA / abs(tw.bets0)
xt.Table(
    data    = {
        'quantity': np.array(['sigma_zeta [m]', 'sigma_delta']),
        'target':   np.array([SIGMA_ZETA, sigma_delta]),
        'prepared': np.array([np.std(initial.zeta), np.std(initial.delta)]),
        'tracked':  np.array([np.std(particles.zeta),
                              np.std(particles.delta)])},
    index   = 'quantity').show(maxwidth = 'full')
print('Total represented intensity:', np.sum(particles.weight))

################################################################################
# Plot longitudinal phase space before and after tracking
################################################################################
fig, axes = plt.subplots(1, 2, figsize = (10, 4), sharex = True, sharey = True)
for ax, batch, title in [
    (axes[0], initial, 'Prepared bunch'),
    (axes[1], particles, f'After {NUM_TURNS} turns'),
]:
    ax.scatter(batch.zeta * 1e3, batch.delta * 1e3, s = 2, alpha = 0.4)
    ax.set_xlabel('zeta [mm]')
    ax.set_title(title)
    ax.grid()
axes[0].set_ylabel('delta [1e-3]')
fig.tight_layout()
plt.show()
