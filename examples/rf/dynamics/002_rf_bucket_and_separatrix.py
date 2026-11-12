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

########################################
# Ideal ring with a stationary single-harmonic RF system
########################################
P0C                 = 1e9               # Electron reference p0*c [eV]
CIRCUMFERENCE       = 100.0             # [m]
ALPHA_C             = 0.02              # Constant momentum compaction
HARMONIC            = 20
RF_VOLTAGE          = 1e5               # [V]
RF_PHASE            = np.pi            # Above-transition stable phase [rad]

########################################
# Launches relative to the stationary bucket's momentum height
########################################
HEIGHT_FRACTIONS    = np.array([-0.8, 0.8, -1.2, 1.2])
N_TURNS             = 1024

########################################
# Accelerating bucket on the same ideal map
########################################
ENERGY_GAIN         = 50e3              # Reference energy increment [eV/turn]
SIGMA_Z_ACCEL       = 0.2               # Nonlinear matched RMS length [m]
N_ACCEL_PARTICLES   = 256
ACCEL_TURNS         = 64

########################################
# Compensated mean-radiation operating point on the shared ring
########################################
LATTICE_FILE            = 'test_data/example_lattices/demo_ring/lattice.py'
P0C_RADIATION           = 10e9          # Electron reference p0*c [eV]
RF_VOLTAGE_RADIATION    = 50e6          # Above the mean loss per turn [V]
HARMONIC_RADIATION      = 200
N_PARTICLES             = 512
NEMITT                  = 1e-6          # Transverse launch scale [m]
SIGMA_Z                 = 0.02          # Small-amplitude matched RMS [m]
SEED                    = 753

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

# This explicit ideal map isolates longitudinal dynamics. It does not inherit
# the shared demonstration ring's chromaticity or transverse acceptance.
# Reference energy is constant; acceleration, radiation and collective forces
# are absent. The single RF harmonic gives a stationary symmetric bucket.

################################################################################
# Obtain the native RF-bucket object and its separatrix
################################################################################

########################################
# Use Xpart's public bucket entry point
########################################
tw      = line.twiss6d()
bucket  = xp.longitudinal.get_bucket(line)
zeta    = np.linspace(bucket.z_left, bucket.z_right, 401)
upper   = bucket.separatrix(zeta)
lower   = bucket.separatrix(zeta, sgn = -1)
print('Selected bucket zeta interval [m]:', bucket.z_left, bucket.z_right)
print('Stable point zeta [m], delta:', bucket.z_sfp, bucket.dp0)

########################################
# Launch inside and outside the separatrix, avoiding the boundary itself
########################################
height          = bucket.separatrix(tw.zeta[0]) - bucket.dp0
zeta_start      = np.full(len(HEIGHT_FRACTIONS), tw.zeta[0])
delta_start     = bucket.dp0 + HEIGHT_FRACTIONS * height
particles       = line.build_particles(zeta = zeta_start, delta = delta_start)
print('Initially inside the selected bucket:',
      bucket.is_in_separatrix(particles.zeta, particles.delta))

# delta=(p-p0)/p0; positive zeta means earlier arrival at a fixed position.
# The separatrix is centered on dp0. The bucket calculation retains leading-
# order momentum compaction; it is not a 6D acceptance model.

################################################################################
# Compare librating trajectories with circulating longitudinal motion
################################################################################

########################################
# Record the native map and retain physical particle states
########################################
line.track(particles, num_turns = N_TURNS, turn_by_turn_monitor = True)
record  = line.record_last_track
print('Final states:', particles.state)
print('Sampled zeta ranges [m]:', np.ptp(record.zeta, axis = 1))

# Outside-bucket particles slip in RF phase but remain live: this boundary is
# not a material aperture or a loss condition. The tracked map has discrete
# kicks; its trajectories need not exactly follow a continuous separatrix.

########################################
# Draw the separatrix and show unwrapped longitudinal slippage
########################################
fig, axes = plt.subplots(1, 2, figsize = (11, 4.5))
axes[0].plot(zeta, upper, 'k--', label = 'Native bucket separatrix')
axes[0].plot(zeta, lower, 'k--')
for slot, fraction in enumerate(HEIGHT_FRACTIONS):
    valid   = record.state[slot] > 0
    axes[0].plot(record.zeta[slot, valid], record.delta[slot, valid])
    axes[1].plot(record.at_turn[slot, valid], record.zeta[slot, valid],
                 label  = f'Initial height fraction {fraction:g}')
axes[0].set_xlim(bucket.z_left, bucket.z_right)
axes[0].set_xlabel('zeta [m]')
axes[0].set_ylabel('delta')
axes[0].legend()
axes[1].set_xlabel('Turn')
axes[1].set_ylabel('Unwrapped zeta [m]')
axes[1].legend()
for axis in axes:
    axis.grid()
fig.tight_layout()

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
accelerating_zeta = np.linspace(
    accelerating_bucket.z_left, accelerating_bucket.z_right, 401)
fig, axes = plt.subplots(1, 2, figsize = (11, 4.5))
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

################################################################################
# Inspect a tilted bucket with a radiation-shifted momentum centre
################################################################################

########################################
# Compensate actual mean radiation and taper the shared geometry
########################################
repo_root           = Path(__file__).resolve().parents[3]
radiation_env       = xt.load(repo_root / LATTICE_FILE)
radiation_line      = radiation_env['demo']
radiation_env['p0c'] = P0C_RADIATION
radiation_line.set_particle_ref('electron', p0c = P0C_RADIATION)
radiation_env['frequency_rf'] = 0.0
radiation_env.set('rf.1', harmonic = HARMONIC_RADIATION)
radiation_env['voltage_rf'] = RF_VOLTAGE_RADIATION
radiation_line.configure_radiation(model = 'mean')
radiation_line.compensate_radiation_energy_loss()
radiation_twiss = radiation_line.twiss6d(
    chrom               = False,
    radiation_analysis  = True,
)
radiation_bucket = xp.longitudinal.get_bucket(radiation_line)

# The RF restores the actual mean energy loss; native tapering follows the
# local momentum sawtooth. This illustrative 10 GeV/c setting uses the same
# geometry and normalized strengths, not hardware-qualified magnet fields.
print('Compensated mean energy loss per turn [eV]:',
      radiation_twiss.energy_loss)
print('Compensated Twiss zeta [m] / delta:',
      radiation_twiss.zeta[0], radiation_twiss.delta[0])
print('Native bucket stable zeta [m] / dp0:',
      radiation_bucket.z_sfp, radiation_bucket.dp0)
print('Stable-point Hamiltonian, evaluated at delta=dp0 [m/s]:',
      radiation_bucket.h_sfp(),
      radiation_bucket.hamiltonian(
          radiation_bucket.z_sfp, radiation_bucket.dp0))

########################################
# Generate a native small-amplitude RF-bucket-matched distribution
########################################
np.random.seed(SEED)
bunch = xp.generate_matched_gaussian_bunch(
    line            = radiation_line,
    num_particles   = N_PARTICLES,
    nemitt_x        = NEMITT,
    nemitt_y        = NEMITT,
    sigma_z         = SIGMA_Z,
)
print('Bunch mean zeta [m] / delta and RMS zeta [m]:',
      np.mean(bunch.zeta), np.mean(bunch.delta), np.std(bunch.zeta))
print('Fraction initially inside the effective bucket:',
      np.mean(radiation_bucket.is_in_separatrix(bunch.zeta, bunch.delta)))

# The short-bunch branch uses linear optics around the effective bucket's
# fixed point. It prepares a distribution, not a quantum-radiation equilibrium.
# Radiation damping makes that Hamiltonian non-invariant along real tracks.
# The asymmetric bucket need not support a symmetric long distribution.
# This snapshot does not qualify transverse acceptance or stored-beam lifetime.

########################################
# Plot the asymmetric separatrix around its shifted momentum centre
########################################
radiation_zeta = np.linspace(
    radiation_bucket.z_left, radiation_bucket.z_right, 401)
fig, axes = plt.subplots(1, 2, figsize = (11, 4.5))
axes[0].scatter(bunch.zeta, bunch.delta, s = 4, alpha = 0.4,
                 label = 'Linearly matched sample')
for sign in [-1, 1]:
    axes[0].plot(radiation_zeta,
                 radiation_bucket.separatrix(radiation_zeta, sgn = sign),
                 'k--')
axes[0].plot(radiation_twiss.zeta[0], radiation_twiss.delta[0],
             'rx', label = 'Radiating entrance closed orbit')
axes[0].axhline(radiation_bucket.dp0, color = '0.5', linewidth = 0.7)
axes[0].set_xlabel('zeta [m]')
axes[0].set_ylabel('delta')
axes[0].legend(fontsize = 8)
# Resolve the short sample that is barely visible on the full bucket scale.
axes[1].scatter((bunch.zeta - radiation_bucket.z_sfp) * 1e3,
                 bunch.delta - radiation_bucket.dp0, s = 4, alpha = 0.4,
                 label = 'Linearly matched sample')
axes[1].plot(0.0, 0.0, 'rx', label = 'Effective stable point')
axes[1].set_xlabel('zeta - stable zeta [mm]')
axes[1].set_ylabel('delta - dp0')
axes[1].legend(fontsize = 8)
for axis in axes:
    axis.grid()
fig.tight_layout()
plt.show()
