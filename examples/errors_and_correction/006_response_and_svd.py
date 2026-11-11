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
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
SEED            = 42
SHIFT_RMS       = 2e-6                  # Arc-quadrupole offset RMS [m]
RETAINED_MODES  = [3, 6, 9]             # Per transverse plane
RESPONSE_CUTOFF = 1e-3                  # Reject weak modes in every solve

################################################################################
# Build a response from the aligned ring, then introduce errors
################################################################################

########################################
# Keep the ideal optics for the linear correction model
########################################
repo_root   = Path(__file__).resolve().parents[2]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env['demo']
line.twiss_default['method']    = '4d'
reference   = line.twiss4d()
correction  = line.correct_trajectory(twiss_table = reference, run = False)

########################################
# Apply independent horizontal and vertical alignment errors
########################################
rng         = np.random.default_rng(SEED)
quad_names  = line.get_table().rows['arc[.].*[.]qd'].name
for name in quad_names:
    line[name].shift_x  = rng.normal(scale = SHIFT_RMS)
    line[name].shift_y  = rng.normal(scale = SHIFT_RMS)

################################################################################
# Inspect the response and compare singular-value truncations
################################################################################

########################################
# Each matrix maps corrector momentum kicks to BPM positions
########################################
fig, axes   = plt.subplots(1, 2, figsize = (10, 4))
for plane in ['x', 'y']:
    setup   = getattr(correction, plane + '_correction')
    print(plane, 'response shape (BPMs, correctors):',
          setup.response_matrix.shape)
    singular_values = setup.singular_values
    print(plane, 'relative singular values:',
          singular_values / singular_values[0])
    axes[0].semilogy(
        np.arange(1, len(singular_values) + 1), singular_values, 'o-',
        label   = plane,
    )
# Singular values have units m per unit normalized transverse momentum.
# A small singular value needs a large kick to change its BPM pattern.

########################################
# Start every comparison with the same errors and zero correction
########################################
residuals   = {'x': [], 'y': []}
for n_modes in RETAINED_MODES:
    correction.clear_correction_knobs()
    correction.correct(
        n_singular_values   = n_modes,
        rcond               = RESPONSE_CUTOFF,
        n_iter              = 1,
    )
    corrected   = line.twiss4d()
    for plane in ['x', 'y']:
        setup       = getattr(correction, plane + '_correction')
        positions   = corrected[plane, setup.monitor_names]
        rms         = np.sqrt(np.mean(positions**2))
        kick_rms    = np.sqrt(np.mean(setup.get_kick_values()**2))
        residuals[plane].append(rms)
        print(f'{n_modes} modes, {plane}: BPM RMS = {rms:.3e} m; '
              f'kick RMS = {kick_rms:.3e}')
# This noiseless example compares residuals, not an optimal noise cutoff.
# The relative cutoff also rejects near-null modes, even in the nine-mode
# case. Truncation discards BPM patterns; it does not select devices.

################################################################################
# Compare response conditioning and residual orbit
################################################################################
for plane in ['x', 'y']:
    axes[1].semilogy(
        RETAINED_MODES, 1e6 * np.asarray(residuals[plane]), 'o-',
        label   = plane,
    )
axes[0].set_xlabel('Singular-value index')
axes[0].set_ylabel('Singular value [m]')
axes[1].set_xlabel('Requested maximum modes per plane')
axes[1].set_ylabel('BPM position RMS [µm]')
axes[1].set_xticks(RETAINED_MODES)
for axis in axes:
    axis.grid()
    axis.legend()
fig.tight_layout()
plt.show()
