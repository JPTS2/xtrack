# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
import matplotlib.pyplot as plt
import numpy as np
import xfields as xf
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Proton reference and prescribed source distribution
########################################
P0C             = 1.0e9                 # Reference p0*c [eV]
BUNCH_INTENSITY = 1.0e11                # Number of source protons
SIGMA_X         = 2.0e-3                # Prescribed transverse rms size [m]
SIGMA_Y         = 2.0e-3                # [m]
SIGMA_Z         = 0.2                   # Gaussian longitudinal rms size [m]
SC_LENGTH       = 1.0                   # Integrated interaction length [m]

########################################
# Open-line geometry and entrance optics
########################################
DRIFT_LENGTH    = 1.0                   # Drift on each side of the kick [m]
BETX_START      = 2.0                   # [m]
BETY_START      = 3.0                   # [m]

################################################################################
# Build frozen-field and genuinely updating configurations
################################################################################
# Each configuration owns its source profile and field. The source is centered
# and round; no longitudinal space-charge kick, RF or radiation is included.
lines   = {}
fields  = {}
for label, intensity, updating in [
    ('Zero charge', 0.0, False),
    ('Frozen field', BUNCH_INTENSITY, False),
    ('Updating field', BUNCH_INTENSITY, True),
]:
    ########################################
    # Prescribe an independent source profile and field
    ########################################
    profile = xf.LongitudinalProfileQGaussian(
        number_of_particles = intensity,
        sigma_z             = SIGMA_Z,
        z0                  = 0.0,
        q_parameter         = 1.0,
    )
    field = xf.SpaceChargeBiGaussian(
        longitudinal_profile        = profile,
        update_on_track             = updating,
        length                      = SC_LENGTH,
        sigma_x                     = SIGMA_X,
        sigma_y                     = SIGMA_Y,
        z_kick_num_integ_per_sigma   = 0,
    )

    ########################################
    # Compose the open line around the thin interaction
    ########################################
    line = xt.Line(
        elements        = [
            xt.Drift(length = DRIFT_LENGTH), field,
            xt.Drift(length = DRIFT_LENGTH), xt.Marker(),
        ],
        element_names   = ['upstream', 'spacecharge', 'downstream', 'end'],
    )
    line.set_particle_ref('proton', p0c = P0C)
    lines[label] = line
    fields[label] = field

# SpaceChargeBiGaussian is a thin kick: SC_LENGTH weights the force, while
# the surveyed line length is supplied only by the two drifts.

################################################################################
# Calculate open optics with the default collective handling
################################################################################

########################################
# Compare the three default calculations
########################################
# Frozen fields do not update from tracked particles and are noncollective in
# Xfields. Their prescribed force therefore remains active in default Twiss.
# The genuinely updating field is collective and is replaced by a drift here.
optics = {}
for label, line in lines.items():
    optics[label] = line.twiss4d(
        betx    = BETX_START,
        bety    = BETY_START,
        alfx    = 0.0,
        alfy    = 0.0,
        zeta    = 0.0,
        delta   = 0.0,
    )
    print(f'{label}: iscollective = {line.iscollective}, '
          f'exit betx = {optics[label].betx[-1]:.9f} m, '
          f'exit bety = {optics[label].bety[-1]:.9f} m')

########################################
# Check the skipped force against the zero-charge control
########################################
zero    = optics['Zero charge']
frozen  = optics['Frozen field']
skipped = optics['Updating field']
print('Maximum skipped/zero-charge beta differences [m]:',
      np.max(np.abs(skipped.betx - zero.betx)),
      np.max(np.abs(skipped.bety - zero.bety)))
print('Frozen-field exit beta changes [m]:',
      frozen.betx[-1] - zero.betx[-1], frozen.bety[-1] - zero.bety[-1])

# The updating source is unchanged because its element was skipped. Including
# it would infer a source bunch from numerical Twiss probes; that configuration
# needs a separately qualified physical model.
print('Updating-field source intensity after skipped Twiss:',
      fields['Updating field'].longitudinal_profile.number_of_particles)

################################################################################
# Plot the prescribed-field response and the skipped collective configuration
################################################################################
fig, axes = plt.subplots(2, 1, sharex = True, figsize = (8, 6))
for ax, quantity, plane in [(axes[0], 'betx', 'x'), (axes[1], 'bety', 'y')]:
    for label, tw in optics.items():
        style = 'o--' if label == 'Updating field' else '-'
        ax.plot(tw.s, tw[quantity], style, label = label)
    ax.set_ylabel(f'beta_{plane} [m]')
    ax.legend()
axes[-1].set_xlabel('s [m]')
fig.suptitle('Twiss: prescribed frozen force versus skipped updating force')
fig.tight_layout()
plt.show()
