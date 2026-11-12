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
# Reference and field profile
########################################
P0C             = 1.0e9                 # Proton reference p0*c [eV]
BS              = 0.5                   # Peak longitudinal field [T]
BODY_LENGTH     = 1.0                   # Uniform body length [m]
FRINGE_LENGTH   = 0.1                   # Each linear field ramp [m]
AXIS_X          = 0.3e-3                # Solenoid field-axis offset [m]
AXIS_Y          = -0.2e-3               # [m]

########################################
# Numerical settings
########################################
FRINGE_KICKS    = 16                    # Integration setting for each ramp

########################################
# Horizontal entrance scan
########################################
# Exterior field-free launches with y = px = py = delta = 0.
X_START         = np.linspace(-1e-3, 1e-3, 5)  # [m]

################################################################################
# Build uniform-field cases
################################################################################

########################################
# Include the hard-edge entrance and exit
########################################
reference   = xt.Particles(mass0 = xt.PROTON_MASS_EV, p0c = P0C)
ks          = BS / reference.rigidity0[0]
print('Normalized solenoid strength ks [m^-1]:', ks)
uniform = xt.Line(
    elements        = [xt.UniformSolenoid(length = BODY_LENGTH, ks = ks)],
    element_names   = ['solenoid'],
)

########################################
# Switch off or reverse the field
########################################
field_off = uniform.copy()
field_off['solenoid'].ks = 0.0
reversed_field = uniform.copy()
reversed_field['solenoid'].ks = -ks

################################################################################
# Join longitudinal-field ramps to a uniform body
################################################################################

########################################
# Use VariableSolenoid for the linear ramps
########################################
# VariableSolenoid includes transverse fields from dB_s/ds. B_s is continuous
# at joins, but its derivative and transverse fields jump there. The body
# adds no hard-edge maps; the adjacent longitudinal boundary fields match.
tapered = xt.Line(
    elements = [
        xt.VariableSolenoid(
            length              = FRINGE_LENGTH,
            ks_profile          = [0.0, ks],
            num_multipole_kicks = FRINGE_KICKS),
        xt.UniformSolenoid(
            length              = BODY_LENGTH,
            ks                  = ks,
            edge_entry_active   = False,
            edge_exit_active    = False),
        xt.VariableSolenoid(
            length              = FRINGE_LENGTH,
            ks_profile          = [ks, 0.0],
            num_multipole_kicks = FRINGE_KICKS)],
)

########################################
# Inspect length and integrated field
########################################
# The ramped profile has a longer magnetic integral and geometric length.
# Its exit is a different plane from the uniform case's exit.
print('Uniform length [m] and integral(ks ds):',
      uniform.get_length(), ks * BODY_LENGTH)
print('Ramped length [m] and integral(ks ds):',
      tapered.get_length(), ks * (BODY_LENGTH + FRINGE_LENGTH))

########################################
# Match the magnetic integral and exterior planes
########################################
# Replace each linear ramp by equal lengths of drift and uniform field.
# Equal integral and exterior planes do not imply identical fringe focusing.
sharp_equivalent = xt.Line(elements = [
    xt.Drift(length = FRINGE_LENGTH / 2),
    xt.UniformSolenoid(length = BODY_LENGTH + FRINGE_LENGTH, ks = ks),
    xt.Drift(length = FRINGE_LENGTH / 2),
])

################################################################################
# Inspect transverse coupling
################################################################################

########################################
# Track the same field-free launches
########################################
initial = xt.Particles(mass0 = xt.PROTON_MASS_EV, p0c = P0C, x = X_START)
lines = {
    'field off':        field_off,
    'uniform':          uniform,
    'reversed uniform': reversed_field,
    'ramped':           tapered,
    'matched sharp':    sharp_equivalent}
outputs = {}
for label, line in lines.items():
    particles = initial.copy()
    line.track(particles)
    outputs[label] = particles
    print(label, 'exit y [mm]:', particles.y * 1e3)
    print(label, 'exit P_y/p0:', particles.py)
print('Ramped minus matched sharp exit y [nm]:',
      (outputs['ramped'].y - outputs['matched sharp'].y) * 1e9)

########################################
# Plot the cross-plane response
########################################
fig, ax = plt.subplots(figsize = (8, 4))
for label, particles in outputs.items():
    plot_label = f'{label}, L = {lines[label].get_length():.1f} m'
    ax.plot(1e3 * X_START, 1e3 * particles.y, '.-', label = plot_label)
ax.set_xlabel('Entrance x [mm]')
ax.set_ylabel('Exit y [mm]')
ax.set_title('Solenoid coupling from a horizontal launch')
ax.legend()
ax.grid()
fig.tight_layout()

################################################################################
# Move the solenoid field axis
################################################################################

########################################
# Compare field offsets with a whole-element shift
########################################
# For a pure uniform solenoid these describe the same translated field.
# x0/y0 move its field axis; shift_x/shift_y move the whole element, including
# any embedded multipoles. Do not equate them for combined elements.
axis_offset = xt.Line(elements = [xt.UniformSolenoid(
    length  = BODY_LENGTH,
    ks      = ks,
    x0      = AXIS_X,
    y0      = AXIS_Y,
)])
element_shift = xt.Line(elements = [xt.UniformSolenoid(
    length  = BODY_LENGTH,
    ks      = ks,
    shift_x = AXIS_X,
    shift_y = AXIS_Y,
)])
axis_particles = initial.copy()
shift_particles = initial.copy()
axis_offset.track(axis_particles)
element_shift.track(shift_particles)
print('Field-axis and whole-element shifts: exit x differences [m]:',
      axis_particles.x - shift_particles.x)
print('Field-axis and whole-element shifts: exit y differences [m]:',
      axis_particles.y - shift_particles.y)

plt.show()
