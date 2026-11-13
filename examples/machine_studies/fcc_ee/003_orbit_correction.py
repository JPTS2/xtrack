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
# Nominal optics and reproducible section offsets
########################################
MODEL_DIR   = 'test_data/example_lattices/fcc_ee_z'
OUTPUT_DIR  = 'outputs/correction'    # Relative to this script
SIGMA_SHIFT = 0.2e-6                 # Independent x/y offsets [m]
SEED        = 2026

########################################
# Closed-orbit correction
########################################
N_ARC_BPMS  = 64
RCOND       = 1e-5
ORBIT_ITER  = 3

################################################################################
# Load the model and resolve small-amplitude optics
################################################################################

########################################
# Use a stable error scale for full optics, not just closed orbit
########################################
repo_root   = Path(__file__).resolve().parents[3]
model_dir   = repo_root / MODEL_DIR
output_dir  = Path(__file__).resolve().parent / OUTPUT_DIR
output_dir.mkdir(parents = True, exist_ok = True)
env     = xt.load(model_dir / 'lattice.json')
line    = env.fccee_p_ring
line.configure_radiation(model = None)
env.call(model_dir / 'strengths.py')
line.twiss_default.update(
    method          = '4d',
    nemitt_y        = 1e-4,
    step_W_sigma    = 0.1,
    steps_R_matrix  = dict(
        dx      = 1e-7,
        dpx     = 1e-8,
        dy      = 1e-8,
        dpy     = 1e-9,
        dzeta   = 1e-4,
        ddelta  = 1e-5,
    ),
)
# The emittance here sets numerical optics-probe amplitudes, not the beam
# emittance. It allows the requested map steps near the narrow vertical waist.
# Radiation is off throughout correction; RF is ignored.
tw0         = line.twiss4d(chrom = False)
line.to_json(output_dir / 'aligned.json')

########################################
# Apply smaller versions of the same seeded final-doublet errors
########################################
table   = line.get_table()
quads   = table.rows[r'(qd0[abc]|qf1[abcd])[lr]\.[0-3]'].name
rng     = np.random.default_rng(SEED)
shift_x = rng.normal(scale = SIGMA_SHIFT, size = len(quads))
shift_y = rng.normal(scale = SIGMA_SHIFT, size = len(quads))
for name, dx, dy in zip(quads, shift_x, shift_y):
    line[name].shift_x  = dx
    line[name].shift_y  = dy
tw_errors   = line.twiss4d(chrom = False)
line.to_json(output_dir / 'errors.json')

################################################################################
# Correct the closed orbit using installed steering elements
################################################################################

########################################
# Select final-doublet stations and a bounded sample around the ring
########################################
all_bpms                    = list(table.rows[r'bpm_.*'].name)
available                   = [
    name for name in all_bpms
    if 'hcor_' + name[4:] in env.elements
    and 'vcor_' + name[4:] in env.elements
]
arc_stride                  = max(1, len(available) // N_ARC_BPMS)
bpms                        = list(dict.fromkeys(
    [name for name in available if name[4:] in quads]
    + available[::arc_stride]))
line.steering_monitors_x    = bpms
line.steering_monitors_y    = bpms
line.steering_correctors_x  = ['hcor_' + name[4:] for name in bpms]
line.steering_correctors_y  = ['vcor_' + name[4:] for name in bpms]
# Ideal BPMs, no calibration errors or kick limits. This selection illustrates
# correction; it is not a proposed FCC commissioning configuration.

########################################
# Reuse the aligned response and iterate the measured orbit
########################################
line.correct_trajectory(
    twiss_table = tw0,
    rcond       = RCOND,
    n_iter      = ORBIT_ITER,
)
tw_orbit    = line.twiss4d(chrom = False)
line.to_json(output_dir / 'orbit.json')

################################################################################
# Inspect orbit correction and the remaining dispersion
################################################################################

########################################
# Print physical residuals and saved states for the radiation study
########################################
results = [('Aligned', tw0), ('Errors', tw_errors), ('Orbit', tw_orbit)]
for label, tw in results:
    print('\n' + label)
    for coordinate in ['x', 'y', 'dy']:
        rms = np.sqrt(np.trapezoid(tw[coordinate]**2, tw.s) / line.get_length())
        print(coordinate, 'distance-weighted RMS [m]:', rms)
    tw.rows[['ipa', 'ipd', 'ipg', 'ipj']].cols['name y dy'].show()
# A small orbit residual does not guarantee zero dispersion or a minimum
# equilibrium emittance. The next example evaluates the radiating covariance.
print('Saved correction states:', output_dir)

########################################
# Separate orbit improvement from remaining vertical dispersion
########################################
fig, axes = plt.subplots(2, 2, sharex = True, figsize = (12, 7),
                         layout = 'constrained')
for column, selected in enumerate([results[1:], results[2:]]):
    for label, tw in selected:
        axes[0, column].plot(tw.s * 1e-3, tw.y * 1e6, label = label)
        axes[1, column].plot(tw.s * 1e-3, tw.dy * 1e3, label = label)
    axes[0, column].set_ylabel('Closed orbit y [µm]')
    axes[1, column].set_ylabel('Vertical dispersion [mm]')
    axes[1, column].set_xlabel('s [km]')
axes[0, 0].set_title('Uncorrected and corrected states')
axes[0, 1].set_title('Corrected-state zoom')
for axis in axes.ravel():
    axis.grid()
    axis.legend()
plt.show()
