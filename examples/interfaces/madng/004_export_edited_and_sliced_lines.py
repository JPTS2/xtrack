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
# Shared ring and one directly aligned quadrupole
########################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
QUADRUPOLE      = 'arc.s2.c1.qd'
SHIFT_X         = 5e-6                  # [m]
SHIFT_Y         = -3e-6                 # [m]
ROLL            = 1e-3                  # About the longitudinal axis [rad]

########################################
# Independent structural edit inside a clean drift
########################################
DRIFT_NAME      = 'arc.s1.c1.d0'
SPLIT_FRACTION  = 0.5                   # Marker position / drift length

################################################################################
# Preserve the baseline and prepare two independent edited versions
################################################################################

########################################
# Apply direct alignment attributes on a separate copy
########################################
repo_root       = Path(__file__).resolve().parents[3]
env             = xt.load(repo_root / LATTICE_FILE)
line            = env['demo']
line.twiss_default['method']    = '4d'
aligned_line    = line.copy()
aligned_line[QUADRUPOLE].shift_x    = SHIFT_X
aligned_line[QUADRUPOLE].shift_y    = SHIFT_Y
aligned_line[QUADRUPOLE].rot_s_rad  = ROLL
# The 1 GeV/c electron reference, RF-off magnets and family expressions are
# preserved. The direct attributes belong to this edited version only.

########################################
# Split a clean drift by inserting a marker on another baseline copy
########################################
sliced_line = line.copy()
split_s     = (
    sliced_line.get_table()['s', DRIFT_NAME]
    + SPLIT_FRACTION * sliced_line[DRIFT_NAME].length
)
sliced_line.insert('export.marker', xt.Marker(), at = split_s)
sliced_line.get_table().rows[DRIFT_NAME + '.*|export[.]marker'].cols[
    'name s element_type parent_name'].show(maxwidth = 'full')
# Insertion creates native DriftSlice elements with the original drift as
# their parent. The structural edit uses an independent clean baseline;
# the original BPMs remain observation sites.

################################################################################
# Build a fresh MAD-NG model for each configured line
################################################################################

########################################
# Compare orbit and mode projections at common BPM names
########################################
# Existing live variables can update an attached model, as shown in001.
# Direct element attributes and line structure are captured when exporting:
# each version here is fully configured before its own model is built.
cases       = {
    'Baseline':     line,
    'Aligned':      aligned_line,
    'Drift split':  sliced_line,
}
models      = []
results     = {}
bpms        = line.steering_monitors_x
for label, case in cases.items():
    models.append(case.build_madng_model())
    optics          = case.madng_twiss(
        normal_form         = False,
        coupling_edw_teng   = True,
    )
    results[label]  = optics.rows[bpms]
    print(label, 'native / MAD-NG coordinates and projected mode betas [m]:')
    results[label].cols[
        'name s x x_ng y y_ng '
        'betx2 beta12_ng bety1 beta21_ng'].show(maxwidth = 'full')
    print('Maximum native/MAD-NG x and y differences [m]:',
          np.max(np.abs(results[label].x - results[label].x_ng)),
          np.max(np.abs(results[label].y - results[label].y_ng)))
    print('Maximum native/MAD-NG mode-beta differences [m]:',
          np.max(np.abs(results[label].betx2 - results[label].beta12_ng)),
          np.max(np.abs(results[label].bety1 - results[label].beta21_ng)))

########################################
# Close the external models after all calculations on their lines
########################################
for model in models:
    model.close()
# No attached line is edited or queried after closing its model. The plots
# below use the already returned numerical tables, without a live process.

################################################################################
# Inspect the alignment response and clean drift-slice agreement
################################################################################
fig, axes   = plt.subplots(2, 1, sharex = True, figsize = (9, 6))
for color, (label, table) in enumerate(results.items()):
    for axis, coordinate in zip(axes, ['x', 'y']):
        axis.plot(
            table.s, table[coordinate] * 1e6,
            color   = f'C{color}',
            label   = label + ': Xtrack',
        )
        axis.plot(
            table.s, table[coordinate + '_ng'] * 1e6, 'x',
            color   = f'C{color}',
            label   = label + ': MAD-NG',
        )
axes[0].set_ylabel('BPM x [um]')
axes[1].set_ylabel('BPM y [um]')
axes[1].set_xlabel('s [m]')
axes[0].legend(ncol = 2, fontsize = 8)
for axis in axes:
    axis.grid()
fig.tight_layout()
plt.show()
