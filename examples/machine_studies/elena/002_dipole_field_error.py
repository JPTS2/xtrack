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
# Fitted ELENA dipole and current error
########################################
LATTICE_FILE    = 'test_data/example_lattices/elena_field/lattice.py'
BEND_NAME       = 'lnr.mbhek.0135'
FIELD_ERRORS    = np.array([-1e-4, 0.0, 1e-4])  # Relative field scale

################################################################################
# Establish the fitted-field operating point
################################################################################
repo_root   = Path(__file__).resolve().parents[3]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env['elena']
tw0         = line.twiss4d()
survey0     = line.survey()

# Scale the complete fitted normal field in one dipole, including its fringes.
# This is an illustrative current error at fixed 100 MeV/c antiproton rigidity.
# Native kscale changes the complete field; h and all reference positions
# stay unchanged. The fitted input coefficients themselves are not rewritten.
names                       = [name for name in line.element_names
               if name.startswith(BEND_NAME + '.')]
env['elena_dipole_scale']   = 1.0
env.set(names, kscale = 'elena_dipole_scale')

################################################################################
# Compare periodic orbit and optics at the same reference geometry
################################################################################

########################################
# Recompute the orbit and linear optics for each field scale
########################################
results = []
for error in FIELD_ERRORS:
    env['elena_dipole_scale']   = 1 + error
    tw                          = line.twiss4d()
    results.append((error, tw))
    print(f'Field error {error:+.1e}: Qx={tw.qx:.8f}, Qy={tw.qy:.8f}')
    print('  Maximum orbit change [mm]:',
          1e3 * np.max(np.abs(tw.x - tw0.x)))
    print('  Maximum horizontal beta beating [%]:',
          100 * np.max(np.abs(tw.betx / tw0.betx - 1)))

########################################
# Check the reference path while the last field error is still applied
########################################
survey  = line.survey()
print('Reference-path change [m]:',
      np.max(np.linalg.norm(survey.XYZ - survey0.XYZ, axis = 1)))
# The closed orbit responds although the survey is unchanged. This is not a
# magnet displacement or a recalculation of the reference curvature.
env['elena_dipole_scale'] = 1.0

################################################################################
# Inspect the nonlocal response to one dipole error
################################################################################

########################################
# Plot orbit displacement and beta beating around the full ring
########################################
fig, axes = plt.subplots(2, 1, figsize = (9, 7), sharex = True,
                         layout = 'constrained')
for error, tw in results:
    label   = f'Field scale error {error:+.0e}'
    axes[0].plot(tw.s, 1e3 * (tw.x - tw0.x), label = label)
    axes[1].plot(tw.s, 100 * (tw.betx / tw0.betx - 1), label = label)
axes[0].set_ylabel('Orbit change [mm]')
axes[1].set(xlabel = 's [m]', ylabel = 'Horizontal beta beating [%]')
for ax in axes:
    ax.legend()
    ax.grid(alpha = 0.25)
plt.show()
