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
# Common error realization
########################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
SHIFT_RMS       = 20e-6                 # Arc quadrupole alignment rms [m]
SEED            = 12345

########################################
# Sparse automatic choice or explicit available hardware
########################################
N_MICADO        = 3                     # Correctors chosen per plane
SELECTED        = ['cor.s1', 'cor.s3', 'cor.s5']

################################################################################
# Prepare one error realization for both correction strategies
################################################################################
repo_root   = Path(__file__).resolve().parents[2]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env['demo']
line.twiss_default['method']    = '4d'
tw_ref      = line.twiss4d()
bpms        = line.steering_monitors_x
quads       = line.get_table().rows['arc[.]s.*[.]q[fd].*'].name
rng         = np.random.default_rng(SEED)
for name in quads:
    line[name].shift_x  = rng.normal(scale = SHIFT_RMS)
    line[name].shift_y  = rng.normal(scale = SHIFT_RMS)
tw_before   = line.twiss4d()

################################################################################
# Let MICADO choose a small corrector subset
################################################################################

########################################
# One pass keeps the requested sparsity explicit
########################################
# Repeated MICADO passes can select additional correctors. Here both planes
# choose at most N_MICADO from the full ring's available hardware in one pass.
micado      = line.correct_trajectory(
    twiss_table = tw_ref,
    n_micado    = N_MICADO,
    n_iter      = 1,
)
tw_micado   = line.twiss4d()
kicks_x     = micado.x_correction.get_kick_values()
kicks_y     = micado.y_correction.get_kick_values()
for plane, plane_correction, kicks in zip(
        ['x', 'y'], [micado.x_correction, micado.y_correction],
        [kicks_x, kicks_y]):
    names   = np.array(plane_correction.corrector_names)
    print(f'MICADO {plane} selected:', names[np.abs(kicks) > 0])

########################################
# Restore the same uncorrected machine for the comparison
########################################
micado.clear_correction_knobs()

################################################################################
# Restrict the available hardware explicitly
################################################################################
# All three named correctors are available to this solve; MICADO is not used.
selected    = line.correct_trajectory(
    twiss_table         = tw_ref,
    corrector_names_x   = SELECTED,
    corrector_names_y   = SELECTED,
    n_iter              = 1,
)
tw_selected = line.twiss4d()
for label, tw in [('Before', tw_before), ('MICADO', tw_micado),
                  ('Selected hardware', tw_selected)]:
    print(label, 'BPM rms x / y [m]:',
          np.sqrt(np.mean(tw.rows[bpms].x**2)),
          np.sqrt(np.mean(tw.rows[bpms].y**2)))

################################################################################
# Compare residuals and the chosen horizontal hardware
################################################################################

########################################
# Evaluate both strategies at the same BPMs
########################################
fig, axes   = plt.subplots(3, 1, figsize = (9, 8))
for axis, plane in zip(axes[:2], ['x', 'y']):
    for label, tw in [('Before', tw_before), ('MICADO', tw_micado),
                      ('Selected hardware', tw_selected)]:
        axis.plot(tw.rows[bpms].s, tw.rows[bpms][plane] * 1e3,
                  'o-',
                  label = label)
    axis.set_ylabel(f'BPM {plane} [mm]')
    axis.set_xlabel('s [m]')
    axis.legend()

########################################
# The same corrector count does not imply the same subset
########################################
axes[2].stem(
    micado.x_correction.s_correctors, kicks_x * 1e6,
    linefmt     = 'C0-',
    markerfmt   = 'C0o',
    label       = 'MICADO',
)
axes[2].stem(
    selected.x_correction.s_correctors,
    selected.x_correction.get_kick_values() * 1e6,
    linefmt     = 'C1--',
    markerfmt   = 'C1s',
    label       = 'Selected hardware',
)
axes[2].set_ylabel('Horizontal kick [µrad]')
axes[2].set_xlabel('s [m]')
axes[2].legend()
for axis in axes:
    axis.grid()
fig.tight_layout()
plt.show()
