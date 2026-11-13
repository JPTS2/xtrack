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
# Combined solenoid and quadrupole
########################################
P0C             = 20e9                  # Electron reference p0*c [eV]
BS              = 2.0                   # Longitudinal field [T]
LENGTH          = 1.0                   # [m]
K1L             = 0.15                  # Embedded integrated quadrupole [m^-1]
AXIS_X          = 5e-3                  # Transverse field displacement [m]
AXIS_Y          = -4e-3                 # [m]

########################################
# Common exterior launches and variable-field ramps
########################################
X_START         = np.linspace(-0.02, 0.03, 5)  # [m]
Y_START         = 0.02                  # [m]
RAMP_LENGTH     = 0.1                   # [m]
N_KICKS         = 32

################################################################################
# Distinguish solenoid, multipole and whole-element axes
################################################################################

########################################
# Move one field component or both together
########################################
initial = xt.Particles('electron', p0c = P0C, x = X_START, y = Y_START)
ks      = BS / initial.rigidity0[0]
cases   = {
    'solenoid axis only': dict(x0 = AXIS_X, y0 = AXIS_Y),
    'whole element': dict(shift_x = AXIS_X, shift_y = AXIS_Y),
}
outputs = {}
for label, offsets in cases.items():
    magnet = xt.UniformSolenoid(
        length = LENGTH, ks = ks, knl = [0.0, K1L],
        num_multipole_kicks = N_KICKS, **offsets)
    particles = initial.copy()
    magnet.track(particles)
    outputs[label] = particles
    print(label, 'exit px:', particles.px)
# x0/y0 do not move the embedded quadrupole. Translating the whole element
# moves both fields and their edge maps; field-axis shifts alone do not.

########################################
# Plot the different feed-down effects
########################################
fig, ax = plt.subplots(figsize = (8, 4), layout = 'constrained')
for label, particles in outputs.items():
    ax.plot(X_START * 1e3, particles.px * 1e3, '.-', label = label)
ax.set(xlabel = 'Entrance x [mm]', ylabel = r'Exit $p_x$ [$10^{-3}$]')
ax.legend()
ax.grid()

################################################################################
# Translate a variable field and follow mean-radiation loss
################################################################################

lines = {}
for label, dx, dy in [('Centred', 0.0, 0.0),
                       ('Translated', AXIS_X, AXIS_Y)]:
    line = xt.Line(elements = [
        xt.VariableSolenoid(
            length = RAMP_LENGTH, ks_profile = [0.0, ks],
            x0 = dx, y0 = dy, num_multipole_kicks = N_KICKS),
        xt.UniformSolenoid(
            length = LENGTH, ks = ks, x0 = dx, y0 = dy,
            edge_entry_active = False, edge_exit_active = False),
        xt.VariableSolenoid(
            length = RAMP_LENGTH, ks_profile = [ks, 0.0],
            x0 = dx, y0 = dy, num_multipole_kicks = N_KICKS),
    ])
    line.configure_radiation(model = 'mean')
    particles = xt.Particles(
        'electron', p0c = P0C, x = X_START + dx, y = Y_START + dy)
    line.track(particles)
    lines[label] = particles
    print(label, 'mean energy lost [eV]:',
          initial.energy - particles.energy)
print('Translation-corrected exit x/y differences [m]:',
      lines['Translated'].x - AXIS_X - lines['Centred'].x,
      lines['Translated'].y - AXIS_Y - lines['Centred'].y)
print('Difference of radiating exit delta:',
      lines['Translated'].delta - lines['Centred'].delta)
# Translation covariance is checked for equal relative entrance coordinates.
# This is open magnetic transport, without RF compensation or equilibrium.
plt.show()
