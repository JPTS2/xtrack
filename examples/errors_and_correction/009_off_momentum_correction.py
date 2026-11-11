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
DELTA0          = 1e-5                  # (p - p0) / p0, held fixed
RCOND           = 1e-3                  # Relative response cutoff

################################################################################
# Compare the on-momentum and off-momentum closed orbits
################################################################################

########################################
# Calculate the nominal response at the reference momentum
########################################
repo_root   = Path(__file__).resolve().parents[2]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env['demo']
line.twiss_default['method']    = '4d'      # No RF or synchrotron motion here.
tw_ref      = line.twiss4d(delta0 = 0)
bpms        = line.steering_monitors_x

########################################
# Dispersion displaces a beam with fixed momentum error
########################################
tw_before   = line.twiss4d(delta0 = DELTA0)
# No alignment errors are added. The uncorrected orbit is approximately
# x = Dx * DELTA0; DELTA0 is a momentum error, not a relative energy error.

################################################################################
# Correct the horizontal orbit of the off-momentum beam
################################################################################

########################################
# Reuse the nominal response but measure at the beam's momentum
########################################
correction  = line.correct_trajectory(
    twiss_table = tw_ref,
    rcond       = RCOND,
    run         = False,
)
correction.correct(planes = 'x', delta0 = DELTA0)
tw_after    = line.twiss4d(delta0 = DELTA0)

########################################
# Keep the same kicks and inspect their on-momentum effect
########################################
tw_on_after = line.twiss4d(delta0 = 0)
for label, tw in [('Off-momentum before', tw_before),
                  ('Off-momentum after', tw_after),
                  ('On-momentum before', tw_ref),
                  ('On-momentum after', tw_on_after)]:
    positions   = tw.rows[bpms].x
    print(label, 'horizontal BPM RMS [m]:',
          np.sqrt(np.mean(positions**2)))
# Steering one momentum towards zero does not correct dispersion or centre all
# momenta simultaneously. The on-momentum beam now sees the same added kicks.

################################################################################
# Compare the two momenta under the same steering settings
################################################################################

########################################
# Orbit improvement at one momentum can cost orbit quality at another
########################################
fig, axes   = plt.subplots(3, 1, sharex = True, figsize = (9, 8))
axes[0].plot(tw_before.s, tw_before.x * 1e6, label = 'Before')
axes[0].plot(tw_after.s, tw_after.x * 1e6, label = 'After')
axes[0].plot(tw_after.rows[bpms].s, tw_after.rows[bpms].x * 1e6,
             'o',
        label   = 'Corrected BPM readings',
    )
axes[0].set_title(f'Off-momentum closed orbit: delta = {DELTA0:g}')
axes[0].set_ylabel('x [µm]')
axes[0].legend()
axes[1].plot(tw_ref.s, tw_ref.x * 1e6, label = 'Before')
axes[1].plot(tw_on_after.s, tw_on_after.x * 1e6, label = 'After')
axes[1].set_title('On-momentum closed orbit: delta = 0')
axes[1].set_ylabel('x [µm]')
axes[1].legend()

########################################
# The correctors retain the same settings for both beams
########################################
axes[2].stem(
    correction.x_correction.s_correctors,
    correction.x_correction.get_kick_values() * 1e6,
)
axes[2].set_ylabel('Horizontal kick [µrad]')
axes[2].set_xlabel('s [m]')
for axis in axes:
    axis.grid()
fig.tight_layout()
plt.show()
