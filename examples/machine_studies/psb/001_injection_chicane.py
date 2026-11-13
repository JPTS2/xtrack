# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
from pathlib import Path

import matplotlib.pyplot as plt
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Stored injection model
########################################
SEQUENCE_FILE   = 'test_data/psb_chicane/psb.seq'
STRENGTH_FILE   = 'test_data/psb_chicane/psb_fb_lhc.str'
P0C             = 0.5708301551893517e9    # Proton reference p0*c [eV]
BUMPER_SHIFT    = -0.0442                # All four bumper offsets [m]
OUTPUT_FILE     = 'outputs/psb_chicane.json'

########################################
# Chicane and induced sextupole field
########################################
BUMP_ANGLE      = 6.6e-2                 # Signed integrated k0 of bumper1 [rad]
EDDY_K2L        = -9.7429e-2             # Integrated k2 of bumper1 [m^-2]
BUMPER_NAMES    = [f'bi1.bsw1l1.{index}' for index in range(1, 5)]
SIGNS           = [1, -1, -1, 1]

################################################################################
# Load the stored sequence with the native parser
################################################################################
repo_root                       = Path(__file__).resolve().parents[3]
env                             = xt.load([
    repo_root / SEQUENCE_FILE,
    repo_root / STRENGTH_FILE,
])
line                            = env.psb1
line.particle_ref               = xt.Particles('proton', p0c = P0C)
line.twiss_default['method']    = '4d'
line.configure_bend_model(core = 'adaptive')
# These legacy files store a machine definition; MAD-X is not executed.
# The same four -44.2 mm shifts define the native regression configuration.

################################################################################
# Connect the four bumpers to the two physical circuits
################################################################################

########################################
# Dipole field and eddy-current sextupole component
########################################
env['bsw_k0l']  = 0.0
env['bsw_k2l']  = 0.0
for index, (name, sign) in enumerate(zip(BUMPER_NAMES, SIGNS), start = 1):
    line[name].shift_x          = BUMPER_SHIFT
    env[f'k0bi1bsw1l1{index}']  = (
        sign * env.ref['bsw_k0l'] / line[name].length)
    env.ref[name].knl[2]        = sign * env.ref['bsw_k2l']
# k0 changes the field, not the reference curvature h. The stored edge model
# therefore contributes focusing as the chicane is powered.

########################################
# Keep the injection tunes as the chicane-off reference
########################################
line.match(
    vary    = xt.VaryList(['kbrqf', 'kbrqd'], step = 1e-5),
    targets = xt.TargetSet(qx = 4.4, qy = 4.45, tol = 1e-7),
)
tw0 = line.twiss()

################################################################################
# Compare the chicane orbit, edge focusing and eddy-current feed-down
################################################################################
settings    = [
    ('Chicane off', 0.0, 0.0),
    ('Dipole field', BUMP_ANGLE, 0.0),
    ('With eddy sextupole', BUMP_ANGLE, EDDY_K2L),
]
results     = []
for label, angle, k2l in settings:
    env['bsw_k0l']  = angle
    env['bsw_k2l']  = k2l
    tw              = line.twiss()
    results.append((label, tw))
    print('\n' + label, 'tunes:', tw.qx, tw.qy)
    tw.rows[['bi1.tstr1l1', 'br.stscrap22']].cols[
        'name x betx bety'].show()

fig, axes = plt.subplots(3, 1, figsize = (10, 9), sharex = True)
for label, tw in results:
    axes[0].plot(tw.s, tw.x * 1e3, label = label)
    axes[1].plot(tw.s, 100 * (tw.betx / tw0.betx - 1), label = label)
    axes[2].plot(tw.s, 100 * (tw.bety / tw0.bety - 1), label = label)
axes[0].set_ylabel('Closed orbit x [mm]')
axes[1].set_ylabel('Horizontal beta beating [%]')
axes[2].set_ylabel('Vertical beta beating [%]')
axes[2].set_xlabel('s [m]')
for axis in axes:
    axis.grid(alpha = 0.3)
axes[0].legend()
fig.tight_layout()

################################################################################
# Save the configured model for the collapse programme
################################################################################
env['bsw_k0l']  = 0.0
env['bsw_k2l']  = 0.0
save_path       = Path(__file__).resolve().parent / OUTPUT_FILE
save_path.parent.mkdir(parents = True, exist_ok = True)
line.to_json(save_path)
print('Saved chicane-off model:', save_path)
plt.show()
