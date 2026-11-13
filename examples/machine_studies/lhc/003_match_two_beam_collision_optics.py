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
# Native two-beam model and collision targets
########################################
MODEL_FILE      = 'test_data/hllhc15_thick/hllhc15_collider_thick.json'
BETA_IP8        = 1.65                   # Both planes and beams [m]
GRADIENT_STEP   = 1e-6                   # Matching step [m^-2]

################################################################################
# Retain the nominal two-beam collision optics
################################################################################
repo_root   = Path(__file__).resolve().parents[3]
env         = xt.load(repo_root / MODEL_FILE)
tw0         = env.twiss(method = '4d', chrom = False)
# This is a modest beta increase from the stored 1.5 m IP8 optics, at 7 TeV/c.
# The two beams must see the same common-triplet circuit changes.

################################################################################
# Assemble one optimizer for both beams
################################################################################

########################################
# Distinguish common triplets from separate matching quadrupoles
########################################
shared_circuits = ['kqx.l8', 'kqx.r8']
beam_circuits   = [
    f'kq{index}.{side}8b{beam}'
    for beam in [1, 2] for side in ['l', 'r'] for index in [4, 5, 6, 7]
]
circuits        = shared_circuits + beam_circuits
initial_values  = np.array([env[name] for name in circuits])

########################################
# Impose collision waists and preserve both full-ring tunes
########################################
targets = []
for name in ['lhcb1', 'lhcb2']:
    targets.extend([
        xt.TargetSet(
            betx = BETA_IP8, bety = BETA_IP8, alfx = 0, alfy = 0,
            at = 'ip8', line = name, tol = 1e-5),
        xt.TargetSet(
            qx = tw0[name].qx, qy = tw0[name].qy,
            line = name, tol = 1e-6),
    ])
opt = env.match(
    solve       = False,
    lines       = ['lhcb1', 'lhcb2'],
    method      = '4d',
    chrom       = False,
    vary        = xt.VaryList(circuits, step = GRADIENT_STEP),
    targets     = targets,
    n_steps_max = 30,
)
opt.solve()
opt.target_status()

################################################################################
# Check the periodic optics and the unconstrained consequences
################################################################################

########################################
# Recalculate independently from the optimizer
########################################
tw = env.twiss(method = '4d')
for name in ['lhcb1', 'lhcb2']:
    print('\n' + name)
    tw[name].rows[['ip1', 'ip5', 'ip8']].cols[
        'name betx bety alfx alfy dx dpx'].show()
    print('Tune residuals:', tw[name].qx - tw0[name].qx,
          tw[name].qy - tw0[name].qy)
    print('Resulting chromaticities:', tw[name].dqx, tw[name].dqy)

########################################
# Inspect the shared and independent circuit changes
########################################
final_values = np.array([env[name] for name in circuits])
xt.Table(
    data    = {
        'circuit':         np.array(circuits),
        'before_k1':       initial_values,
        'after_k1':        final_values,
        'relative_change': final_values / initial_values - 1,
    },
    index   = 'circuit',
).show()
# This teaches the joint constraints, not an operational squeeze: dispersion,
# chromaticities, other IPs, aperture and hardware limits are not targets.
# Fixed tunes do not establish matching to every external insertion boundary.

################################################################################
# Compare the two beam optics around the matched interaction point
################################################################################
fig, axes = plt.subplots(2, 2, figsize = (12, 8), sharex = True)
for column, name in enumerate(['lhcb1', 'lhcb2']):
    for label, optics, style in [('Nominal', tw0[name], '--'),
                                  ('Matched', tw[name], '-')]:
        region      = np.abs(optics.s - optics['s', 'ip8']) < 300
        distance    = optics.s[region] - optics['s', 'ip8']
        axes[0, column].plot(distance, optics.betx[region], style,
                             label = label)
        axes[1, column].plot(distance, optics.bety[region], style,
                             label = label)
    axes[0, column].set_title(name)
    axes[0, column].set_ylabel('Horizontal beta [m]')
    axes[1, column].set_ylabel('Vertical beta [m]')
    axes[1, column].set_xlabel('s - s(IP8) [m]')
    axes[0, column].legend()
for axis in axes.flat:
    axis.grid(alpha = 0.3)
fig.tight_layout()
plt.show()
