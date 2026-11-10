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
# Shared planar electron ring
########################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
P0C             = 10e9                  # Electron reference p0*c [eV]
RF_VOLTAGE      = 50e6                  # [V]
HARMONIC        = 200

########################################
# Observation planes
########################################
OBSERVATIONS    = ['ip.1', 'bpm.s1.d']

################################################################################
# Analyse mean damping together with the model's quantum excitation
################################################################################

########################################
# Compensate mean radiation without adding coupling or vertical dispersion
########################################
repo_root           = Path(__file__).resolve().parents[2]
env                 = xt.load(repo_root / LATTICE_FILE)
line                = env['demo']
env['p0c']          = P0C
line.set_particle_ref('electron', p0c = env['p0c'])
env['frequency_rf'] = 0.0
env.set('rf.1', harmonic = HARMONIC)
env['voltage_rf']   = RF_VOLTAGE
line.configure_radiation(model = 'mean')
line.compensate_radiation_energy_loss()
# The same geometry is used at 10 GeV/c, with fixed normalized strengths
# and therefore scaled physical fields. This is not a tracked equilibrium bunch.

########################################
# Request full native radiation analysis of the periodic 6D mean map
########################################
tw      = line.twiss(
    method              = '6d',
    chrom               = False,
    radiation_analysis  = True,
    radiation_method    = 'full',
)
emittances  = np.array([tw.eq_gemitt_x, tw.eq_gemitt_y, tw.eq_gemitt_zeta])
print('Mode; equilibrium geometric emittance [m]; amplitude decrement [1/turn]')
for mode, emittance, rate in zip(
    ['horizontal', 'vertical', 'longitudinal'],
    emittances, tw.damping_constants_turns,
):
    print(mode, emittance, rate)
# With collinear photon recoil and no vertical dispersion/coupling, this model
# has no vertical excitation; photon-opening-angle excitation is omitted.
# Its tiny numerical remainder is not a physical prediction of beam height.
# Mean tracking supplies damping; radiation_analysis also evaluates quantum
# excitation and balances its second moments against the linear damping map.

################################################################################
# Inspect the predicted equilibrium covariance through native APIs
################################################################################

########################################
# Use the full-analysis covariance, including its inter-mode correlations
########################################
covariance  = tw.eq_beam_covariance_matrix
print('Sizes [m] and normalized canonical momentum spread [dimensionless]:')
covariance.rows[OBSERVATIONS].cols[
    'name sigma_x sigma_y sigma_zeta sigma_pzeta'].show(maxwidth = 'full')

########################################
# Reconstruct the diagonal-mode approximation from its emittances
########################################
modal       = tw.get_beam_covariance(
    gemitt_x    = tw.eq_gemitt_x,
    gemitt_y    = tw.eq_gemitt_y,
    gemitt_zeta = tw.eq_gemitt_zeta,
)
print('Largest relative full/modal horizontal size difference:',
      np.max(np.abs(modal.sigma_x / covariance.sigma_x - 1)))
# This reconstruction populates the three modes independently. The full native
# covariance can retain small cross-mode correlations; equality is not imposed.

################################################################################
# Plot predicted equilibrium sizes, without claiming stochastic equilibration
################################################################################
fig, axes = plt.subplots(2, 1, sharex = True, figsize = (9, 6))
for axis, coordinate, unit in [
    (axes[0], 'sigma_x', 'mm'),
    (axes[1], 'sigma_zeta', 'mm'),
]:
    axis.plot(
        tw.s, 1e3 * covariance[coordinate],
        label   = 'Full radiation analysis',
    )
    axis.plot(tw.s, 1e3 * modal[coordinate], '--', label = 'Independent modes')
    axis.set_ylabel(f'{coordinate} [{unit}]')
    axis.grid()
    axis.legend()
axes[1].set_xlabel('s [m]')
fig.tight_layout()
plt.show()

# This is a linearized damping/excitation prediction. It does not qualify the
# nonlinear momentum acceptance, dynamic aperture or tails of an actual bunch.
# The bare ring has large chromaticity: its predicted momentum spread is not
# evidence that such a population can be stored in the nonlinear lattice.
# Deterministic mean tracking alone cannot produce this stochastic equilibrium.
