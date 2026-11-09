# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Magnetic transport
########################################
P0C             = 1.0e9                 # Proton reference p0*c [eV]
QUAD_LENGTH     = 0.5                   # [m]
DRIFT_LENGTH    = 0.5                   # Length on each side [m]
K1              = 0.8                   # [m^-2]

########################################
# Common entrance scan
########################################
X_START         = np.linspace(-0.02, 0.02, 5)  # [m]
Y_START         = 0.01                  # [m]
PX_START        = 0.03                  # P_x / p0
PY_START        = 0.02                  # P_y / p0
DELTA           = 0.02                  # (p - p0) / p0

################################################################################
# Define three controlled comparisons
################################################################################

########################################
# Choose fringe and drift settings
########################################
# Keep the quadrupole body model and strength fixed in all three cases.
# Compare one setting at a time with the fringe-on/exact-drift baseline.
settings = {
    'fringes on, exact drifts':     (True, 'exact'),
    'fringes off, exact drifts':    (False, 'exact'),
    'fringes on, expanded drifts':  (True, 'expanded')}

########################################
# Build the transport lines
########################################
lines = {}
for label, (fringes, drift_model) in settings.items():
    lines[label] = xt.Line(
        elements = [
            xt.Drift(length = DRIFT_LENGTH, model = drift_model),
            xt.Quadrupole(
                length              = QUAD_LENGTH,
                k1                  = K1,
                model               = 'mat-kick-mat',
                edge_entry_active   = fringes,
                edge_exit_active    = fringes),
            xt.Drift(length = DRIFT_LENGTH, model = drift_model)],
    )

################################################################################
# Track the common entrance scan
################################################################################
initial = xt.Particles(
    mass0   = xt.PROTON_MASS_EV,
    p0c     = P0C,
    x       = X_START,
    y       = Y_START,
    px      = PX_START,
    py      = PY_START,
    delta   = DELTA,
)
outputs = {}
for label, line in lines.items():
    particles = initial.copy()
    line.track(particles)
    outputs[label] = particles
    print(label, 'exit x [m]:', particles.x)
    print(label, 'exit P_x/p0:', particles.px)

################################################################################
# Inspect differences from the selected baseline
################################################################################
# Quadrupole fringes use the hard-edge model. Changing the drift model changes
# the surrounding drift maps only; the selected quadrupole body stays fixed.
baseline = outputs['fringes on, exact drifts']
fig, axes = plt.subplots(1, 2, figsize = (11, 4))
for label in ['fringes off, exact drifts', 'fringes on, expanded drifts']:
    particles = outputs[label]
    axes[0].plot(1e3 * X_START, 1e6 * (particles.x - baseline.x),
                 '.-', label = label)
    axes[1].plot(1e3 * X_START, 1e6 * (particles.px - baseline.px),
                 '.-', label = label)
axes[0].set_ylabel('Exit x difference [um]')
axes[1].set_ylabel('Exit P_x/p0 difference [1e-6]')
for ax in axes:
    ax.set_xlabel('Entrance x [mm]')
    ax.grid()
    ax.legend()
fig.suptitle('Differences from fringes on, exact surrounding drifts')
fig.tight_layout()
plt.show()
