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
# Ideal ring and single-harmonic RF
########################################
P0C             = 1e9               # Electron reference p0*c [eV]
CIRCUMFERENCE   = 100.0             # [m]
ALPHA_C         = 0.02              # Constant momentum compaction
HARMONIC        = 20
RF_VOLTAGE      = 1e5               # [V]
RF_PHASE        = np.pi            # Above-transition stable phase [rad]

########################################
# Reference gain and native matched sample
########################################
ENERGY_GAIN         = 50e3              # Reference energy increment [eV/turn]
SIGMA_Z_ACCEL       = 0.2               # Nonlinear matched RMS length [m]
N_ACCEL_PARTICLES   = 256
ACCEL_TURNS         = 64
SEED                = 753
NEMITT              = 1e-6              # Transverse launch scale [m]
################################################################################
# Define a native nonlinear RF map with independent transverse transport
################################################################################

########################################
# Specify the reference and its RF frequency
########################################
reference   = xt.Particles(pdg_id_0 = 'electron', p0c = P0C)
frequency   = HARMONIC * reference.beta0[0] * c / CIRCUMFERENCE

########################################
# Use sinusoidal longitudinal motion without transverse chromatic limitations
########################################
line    = xt.Line(
    elements        = [
        xt.LineSegmentMap(
            length                      = CIRCUMFERENCE,
            qx                          = 0.31,
            qy                          = 0.32,
            betx                        = 10.0,
            bety                        = 12.0,
            longitudinal_mode           = 'nonlinear',
            momentum_compaction_factor  = ALPHA_C,
            voltage_rf                  = RF_VOLTAGE,
            frequency_rf                = frequency,
            phase_rf                    = RF_PHASE,
        ),
    ],
    element_names   = ['turn_map'],
    particle_ref    = reference,
)

# The ideal map isolates longitudinal dynamics from transverse acceptance.

################################################################################
# Tilt the ideal bucket for reference acceleration
################################################################################

########################################
# Supply the reference gain and choose its focusing RF phase
########################################
accelerating_line = line.copy()
accelerating_line['turn_map'].energy_ref_increment = ENERGY_GAIN
accelerating_line['turn_map'].phase_rf[0] = (
    np.pi - np.arcsin(ENERGY_GAIN / RF_VOLTAGE))
accelerating_bucket = xp.longitudinal.get_bucket(accelerating_line)
print('Accelerating bucket interval [m]:',
      accelerating_bucket.z_left, accelerating_bucket.z_right)
print('Effective force at its stable point [N]:',
      accelerating_bucket.total_force(accelerating_bucket.z_sfp))
print('Stable-point Hamiltonian / direct evaluation [m/s]:',
      accelerating_bucket.h_sfp(), accelerating_bucket.hamiltonian(
          accelerating_bucket.z_sfp, accelerating_bucket.dp0))
# The native effective force subtracts the reference momentum gain. The
# resulting bucket is tilted, even with unchanged RF voltage and frequency.

########################################
# Match a finite-length bunch to the accelerating bucket
########################################
np.random.seed(SEED)
accelerating_bunch = xp.generate_matched_gaussian_bunch(
    line            = accelerating_line,
    num_particles   = N_ACCEL_PARTICLES,
    nemitt_x        = NEMITT,
    nemitt_y        = NEMITT,
    sigma_z         = SIGMA_Z_ACCEL,
)
print('Accelerating sample RMS length [m], fraction inside entrance bucket:',
      np.std(accelerating_bunch.zeta), np.mean(
          accelerating_bucket.is_in_separatrix(
              accelerating_bunch.zeta, accelerating_bunch.delta)))
accelerating_zeta   = np.linspace(
    accelerating_bucket.z_left, accelerating_bucket.z_right, 401)
fig, axes           = plt.subplots(1, 2, figsize = (11, 4.5))
axes[0].scatter(accelerating_bunch.zeta, accelerating_bunch.delta,
                 s = 4, alpha = 0.4, label = 'Nonlinearly matched sample')
for sign in [-1, 1]:
    axes[0].plot(accelerating_zeta,
                 accelerating_bucket.separatrix(accelerating_zeta, sgn = sign),
                 'k--')

########################################
# Track a bounded gain without assuming exact Hamiltonian conservation
########################################
accelerating_line.track(accelerating_bunch, num_turns = ACCEL_TURNS,
                        turn_by_turn_monitor = True)
accelerating_record = accelerating_line.record_last_track
print('After acceleration: states / reference p0*c [GeV]:',
      np.unique(accelerating_bunch.state), accelerating_bunch.p0c[0] * 1e-9)
print('Fraction still inside the initial bucket:', np.mean(
    accelerating_bucket.is_in_separatrix(
        accelerating_bunch.zeta, accelerating_bunch.delta)))
# This bucket is evaluated at the entrance reference, not updated every turn.
# Its effective fixed point need not equal the discrete map's exact orbit.
# Reference acceleration also prevents an exact conserved-Hamiltonian claim.
axes[0].plot(accelerating_record.zeta[0], accelerating_record.delta[0],
             linewidth = 0.8, label = 'One tracked particle')
axes[0].set_xlabel('zeta [m]')
axes[0].set_ylabel('delta')
axes[0].legend(fontsize = 8)
axes[1].plot(accelerating_zeta, accelerating_bucket.hamiltonian(
    accelerating_zeta, accelerating_bucket.dp0) - accelerating_bucket.h_sfp())
axes[1].set_xlabel('zeta [m]')
axes[1].set_ylabel('Entrance H(zeta, dp0) - H(stable point) [m/s]')
for axis in axes:
    axis.grid()
fig.tight_layout()

plt.show()
