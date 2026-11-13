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
STATE_DIR   = 'outputs/correction'   # Produced by 003, relative here
STATES      = ['aligned', 'errors', 'orbit']
IP          = 'ipd'

################################################################################
# Calculate radiating equilibrium for each saved correction state
################################################################################

########################################
# Compensate energy loss and taper each state independently
########################################
state_dir   = Path(__file__).resolve().parent / STATE_DIR
results     = []
for state in STATES:
    line = xt.load(state_dir / f'{state}.json')
    line.configure_radiation(model = 'mean')
    line.compensate_radiation_energy_loss()
    tw = line.twiss(
        method = '6d', radiation_method = 'full', radiation_analysis = True)
    results.append(tw)
    print('\n' + state)
    print('Energy loss [MeV/turn]:', tw.energy_loss * 1e-6)
    print('Equilibrium mode emittances x/y [m]:',
          tw.eq_gemitt_x, tw.eq_gemitt_y)
    print('Damping constants [s^-1]:', tw.damping_constants_s)
# The full linear radiation model includes coupling and vertical dispersion.
# These are equilibrium predictions, not a quantum-tracking measurement or
# an operational FCC emittance budget. No beam-beam, collective or IBS effects.

########################################
# Use the full equilibrium covariance for projected IP beam sizes
########################################
sigma_y = []
for state, tw in zip(STATES, results):
    covariance  = tw.eq_beam_covariance_matrix
    sigma       = covariance['sigma_y', IP]
    sigma_y.append(sigma)
    print(state, f'projected sigma_y at {IP.upper()} [nm]:', sigma * 1e9)
# The full covariance includes cross-mode correlations. Reconstructing it
# from W and diagonal mode emittances would be an approximation.
# The ideal planar result is a model/numerical floor, not a credible budget.

################################################################################
# Compare emittance recovery and the actual projected size
################################################################################
fig, axes   = plt.subplots(1, 2, figsize = (10, 4), layout = 'constrained')
values      = [[tw.eq_gemitt_y * 1e12 for tw in results],
          np.array(sigma_y) * 1e9]
labels      = ['Vertical-like equilibrium mode emittance [pm]',
          f'Projected sigma_y at {IP.upper()} [nm]']
for axis, numbers, label in zip(axes, values, labels):
    bars = axis.bar(STATES, numbers)
    axis.bar_label(bars, fmt = '%.3g', padding = 3)
    axis.set_yscale('log')
    axis.set(ylabel = label, ylim = (min(numbers) / 4, max(numbers) * 6))
    axis.tick_params(axis = 'x', rotation = 25)
    axis.grid(axis = 'y')
plt.show()
