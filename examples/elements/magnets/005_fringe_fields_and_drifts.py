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
FRINGE_LENGTH   = 0.05                  # Length of each smooth ramp [m]
N_FRINGE_STEPS  = 32                    # RK4 steps per ramp
FIELD_PROBE_X   = 1.0e-3                # Field-sampling offset [m]

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

################################################################################
# Resolve a finite longitudinal quadrupole profile
################################################################################

########################################
# Keep magnetic length and exterior planes fixed
########################################
# Each quintic ramp has half its peak-field integral and zero first/second
# derivatives at its ends. Shorten the plateau and surrounding drifts so the
# smooth and hard-edge lines have the same integral and entrance/exit planes.
ramp = K1 * np.array([
    0.0, 0.0, 0.0,
    10 / FRINGE_LENGTH**3,
    -15 / FRINGE_LENGTH**4,
    6 / FRINGE_LENGTH**5,
])
falling = -ramp.copy()
falling[0] = K1
smooth_outputs = {}
for steps in [N_FRINGE_STEPS, 2 * N_FRINGE_STEPS]:
    smooth = xt.Line(
        elements = [
            xt.Drift(length = DRIFT_LENGTH - FRINGE_LENGTH / 2,
                     model = 'exact'),
            xt.BFieldExpansion(
                length                  = FRINGE_LENGTH,
                knc                     = [np.zeros_like(ramp), ramp],
                num_integration_steps   = steps,
                pkin_const              = True),
            xt.Quadrupole(
                length              = QUAD_LENGTH - FRINGE_LENGTH,
                k1                  = K1,
                model               = 'mat-kick-mat',
                edge_entry_active   = False,
                edge_exit_active    = False),
            xt.BFieldExpansion(
                length                  = FRINGE_LENGTH,
                knc                     = [np.zeros_like(ramp), falling],
                num_integration_steps   = steps,
                pkin_const              = True),
            xt.Drift(length = DRIFT_LENGTH - FRINGE_LENGTH / 2,
                     model = 'exact')],
    )
    particles = initial.copy()
    smooth.track(particles)
    smooth_outputs[steps] = particles
print('Hard-edge and smooth lengths [m]:',
      lines['fringes off, exact drifts'].get_length(), smooth.get_length())
print('Hard-edge and smooth integrated k1 [m^-1]:',
      K1 * QUAD_LENGTH, np.sum(smooth.get_table(attr = True).k1l))

########################################
# Distinguish profile effects from integration error
########################################
# Equal field integrals do not imply identical finite-amplitude transport.
# This is not a validation of a thin Bend edge map or a fringe integral.
resolved = smooth_outputs[N_FRINGE_STEPS]
refined = smooth_outputs[2 * N_FRINGE_STEPS]
hard_edge = outputs['fringes off, exact drifts']
print('Largest exit x change on doubling fringe steps [m]:',
      np.max(np.abs(resolved.x - refined.x)))
print('Largest exit P_x/p0 change on doubling fringe steps:',
      np.max(np.abs(resolved.px - refined.px)))

fig, axes = plt.subplots(1, 2, figsize = (11, 4))
s_ramp = np.linspace(0, FRINGE_LENGTH, 101)
s_entrance = DRIFT_LENGTH - FRINGE_LENGTH / 2
s_exit = s_entrance + QUAD_LENGTH
for element_index, offset in [(1, s_entrance), (3, s_exit)]:
    field = smooth[element_index].get_field(
        x = FIELD_PROBE_X, y = 0.0, s_local = s_ramp)
    axes[0].plot(offset + s_ramp, field['By'] / FIELD_PROBE_X, 'C0')
axes[0].plot([s_entrance + FRINGE_LENGTH, s_exit], [K1, K1],
             'C0', label = f'Smooth field at x = {FIELD_PROBE_X * 1e3:g} mm')
axes[0].plot([DRIFT_LENGTH, DRIFT_LENGTH, DRIFT_LENGTH + QUAD_LENGTH,
              DRIFT_LENGTH + QUAD_LENGTH], [0, K1, K1, 0],
             'k--', label = 'Hard-edge k1')
axes[0].set_xlabel('s from common entrance [m]')
axes[0].set_ylabel('B_y / (B_rho x) [m^-2]')
axes[1].plot(1e3 * X_START, 1e6 * (refined.px - hard_edge.px), '.-')
axes[1].set_xlabel('Entrance x [mm]')
axes[1].set_ylabel('Smooth minus hard-edge P_x/p0 [1e-6]')
for ax in axes:
    ax.grid()
axes[0].legend()
fig.suptitle('Equal integrated gradient and common exterior planes')
fig.tight_layout()
plt.show()
