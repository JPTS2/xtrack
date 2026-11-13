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
# One installed spin-bump control in the LEP model
########################################
LATTICE_FILE    = 'test_data/spin_refs_bmad/lep_lattice_to_bmad.json'
IPS             = [2, 4, 6, 8]
BUMP_IP         = 2
BUMP_SCALES     = np.array([0.0, 0.5, 1.0])
CORRECTIONS     = [('Bump only', False), ('Paired skew correction', True)]

################################################################################
# Start from compensated solenoids without spin bumps
################################################################################
repo_root   = Path(__file__).resolve().parents[3]
line        = xt.load(repo_root / LATTICE_FILE)
line.configure_spin('auto')
for ip in IPS:
    line[f'on_sol.{ip}']            = 1.0
    line[f'on_spin_bump.{ip}']      = 0.0
    line[f'on_coupl_sol.{ip}']      = 1.0
    line[f'on_coupl_sol_bump.{ip}'] = 0.0
tw0 = line.twiss4d(polarization_analysis = True)
# The stored vertical-orbit bump and its skew-quadrupole correction belong to
# this specific model. Scaling a correction is not rematching at each setting.
# These fixed-energy orbital/spin responses do not predict polarization alone.

################################################################################
# Vary one spin bump with and without its paired coupling correction
################################################################################

########################################
# Preserve all other insertion settings during the comparison
########################################
results = {}
for label, paired in CORRECTIONS:
    samples = []
    for scale in BUMP_SCALES:
        line[f'on_spin_bump.{BUMP_IP}']      = scale
        line[f'on_coupl_sol_bump.{BUMP_IP}'] = scale if paired else 0.0
        tw  = line.twiss4d(polarization_analysis = True)
        samples.append(tw)
        print(f'{label}, bump scale {scale:.1f}:')
        print('  Maximum vertical-orbit change [mm]:',
              1e3 * np.max(np.abs(tw.y - tw0.y)))
        print('  Coupling c_minus:', tw.c_minus)
        print('  Fractional spin tune:', tw.spin_tune_fractional)
    results[label] = samples

########################################
# Compare the local orbit response with the global coupling response
########################################
# Keeping the vertical orbit bump does not require keeping its added coupling.
# Conversely, reducing c_minus does not establish a polarization optimum; the
# following lesson uses ring integrals to predict equilibrium polarization.
fig, axes = plt.subplots(1, 3, figsize = (13, 4), layout = 'constrained')
for label, samples in results.items():
    axes[0].plot(BUMP_SCALES,
                 [1e3 * np.max(np.abs(tw.y - tw0.y)) for tw in samples],
                 'o-', label = label)
    axes[1].plot(BUMP_SCALES, [tw.c_minus for tw in samples], 'o-',
                 label = label)
axes[0].set(
    xlabel  = 'Installed spin-bump scale',
    ylabel  = 'Maximum vertical-orbit change [mm]',
)
axes[1].set(xlabel = 'Installed spin-bump scale', ylabel = 'Coupling c_minus')

################################################################################
# Inspect the spin-field energy sensitivity at full bump strength
################################################################################
sensitivity0    = np.sqrt(tw0.spin_dn_ddelta_x**2 + tw0.spin_dn_ddelta_y**2
                       + tw0.spin_dn_ddelta_z**2)
axes[2].plot(tw0.s * 1e-3, sensitivity0, ':', label = 'No spin bump')
for label, samples in results.items():
    tw          = samples[-1]
    sensitivity = np.sqrt(tw.spin_dn_ddelta_x**2 + tw.spin_dn_ddelta_y**2
                           + tw.spin_dn_ddelta_z**2)
    axes[2].plot(tw.s * 1e-3, sensitivity, label = label)
axes[2].set(xlabel = 's [km]', ylabel = '|partial n / partial delta|')
for ax in axes:
    ax.legend()
    ax.grid(alpha = 0.25)
plt.show()
