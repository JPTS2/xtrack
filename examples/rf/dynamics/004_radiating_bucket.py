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
# Shared ring and compensated RF
########################################
LATTICE_FILE            = 'test_data/example_lattices/demo_ring/lattice.py'
P0C_RADIATION           = 10e9          # Electron reference p0*c [eV]
RF_VOLTAGE_RADIATION    = 50e6          # Above the mean loss per turn [V]
HARMONIC_RADIATION      = 200

########################################
# Small-amplitude matched sample
########################################
N_PARTICLES = 512
NEMITT      = 1e-6          # Transverse launch scale [m]
SIGMA_Z     = 0.02          # Small-amplitude matched RMS [m]
SEED        = 753

################################################################################
# Inspect a tilted bucket with a radiation-shifted momentum centre
################################################################################

########################################
# Compensate actual mean radiation and taper the shared geometry
########################################
repo_root               = Path(__file__).resolve().parents[3]
radiation_env           = xt.load(repo_root / LATTICE_FILE)
radiation_line          = radiation_env['demo']
radiation_env['p0c']    = P0C_RADIATION
radiation_line.set_particle_ref('electron', p0c = P0C_RADIATION)
radiation_env['frequency_rf'] = 0.0
radiation_env.set('rf.1', harmonic = HARMONIC_RADIATION)
radiation_env['voltage_rf'] = RF_VOLTAGE_RADIATION
radiation_line.configure_radiation(model = 'mean')
radiation_line.compensate_radiation_energy_loss()
radiation_twiss     = radiation_line.twiss6d(
    chrom               = False,
    radiation_analysis  = True,
)
radiation_bucket    = xp.longitudinal.get_bucket(radiation_line)

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
radiation_zeta  = np.linspace(
    radiation_bucket.z_left, radiation_bucket.z_right, 401)
fig, axes       = plt.subplots(1, 2, figsize = (11, 4.5))
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
