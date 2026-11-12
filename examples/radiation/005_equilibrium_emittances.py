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
import xpart as xp
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

########################################
# Bounded stochastic comparison, not an equilibration run
########################################
N_PARTICLES     = 256
N_TURNS         = 64
SEEDS           = (753, 754)            # Independent prepared populations

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
# and therefore scaled physical fields. The analysis predicts a linear
# equilibrium; the short tracking comparison below does not establish it.

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

################################################################################
# Prepare native Gaussian bunches from the equilibrium prediction
################################################################################

########################################
# Record population moments at the line entrance
########################################
quantities = ['sigma_x', 'sigma_zeta', 'sigma_pzeta']
monitor = xt.BeamStatsMonitor(
    start_at_turn   = 0,
    stop_at_turn    = N_TURNS,
    stats           = quantities,
)
line.insert('equilibrium.monitor', monitor, at = 0)
predicted           = {name: covariance[name][0] for name in quantities}
moment_histories    = {}
tracked_populations = {}
print('Tracked amplitude damping times x/y/z:',
      N_TURNS * tw.damping_constants_turns)
print('Approximate single-snapshot Gaussian RMS sampling scale:',
      1 / np.sqrt(2 * (N_PARTICLES - 1)))

########################################
# Compare mean and quantum tracking from the same first sample
########################################
for sample, seed in enumerate(SEEDS):
    line.configure_radiation(model = 'mean')
    np.random.seed(seed)
    initial = xp.generate_matched_gaussian_bunch(
        line            = line,
        num_particles   = N_PARTICLES,
        nemitt_x        = tw.eq_nemitt_x,
        nemitt_y        = 0.0,
        sigma_z         = covariance.sigma_zeta[0],
    )
    # The planar collinear-recoil model has zero physical vertical excitation.
    # Use zero, rather than adding a floor to its numerical emittance remainder.
    # Native generation uses independent modes and RF matching; it does not
    # impose the full analysis's small inter-mode covariance correlations.
    models = ['mean', 'quantum'] if sample == 0 else ['quantum']
    for model in models:
        particles = initial.copy()
        monitor.clear()
        line.configure_radiation(model = model)
        line.track(particles, num_turns = N_TURNS)
        label = f'{model}, sample {sample + 1}'
        moment_histories[label] = {
            name: getattr(monitor, name).copy() for name in quantities
        }
        tracked_populations[label] = particles
        print(label, 'final live particles / initial:',
              np.count_nonzero(particles.state > 0), N_PARTICLES)
        print('Final maximum |y| [m]:', np.max(np.abs(particles.y)))
        print('Recorded RMS / linear-equilibrium prediction at final entrance:',
              {name: moment_histories[label][name][-1] / predicted[name]
               for name in quantities})

# Both quantum runs start near the predicted covariance; they do not start
# cold and demonstrate convergence. Mean tracking damps without excitation.
# The short interval is a fraction of an amplitude damping time. Snapshot
# sampling fluctuations are correlated between turns, so the number of turns
# must not be used as an independent-sample count.

################################################################################
# Compare measured moments with the native linear predictions
################################################################################

########################################
# Inspect the bounded quantum histories and their mean-radiation control
########################################
fig, axes = plt.subplots(3, 1, sharex = True, figsize = (9, 8))
for axis, name in zip(axes, quantities):
    for label, history in moment_histories.items():
        axis.plot(monitor.turns, history[name] / predicted[name], label = label)
    axis.axhline(1.0, color = 'k', linestyle = '--',
                  label = 'Linear equilibrium prediction')
    axis.set_ylabel(name + ' / predicted')
    axis.grid()
axes[0].legend(fontsize = 8)
axes[-1].set_xlabel('Turn at the entrance monitor')
fig.tight_layout()
plt.show()

# Surviving these samples does not qualify long-term storage or momentum tails.
# In particular, a zero-vertical-emittance launch does not test acceptance
# against the bare ring's large chromaticity. Photon-opening-angle excitation
# and a physical finite vertical equilibrium are absent from this model.
