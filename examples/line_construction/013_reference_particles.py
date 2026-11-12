# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
import xtrack as xt

################################################################################
# User parameters
################################################################################
P0C_START   = 1.0e9                     # Proton reference p0*c [eV]
P0C_AFTER   = 2.0e9                     # [eV]
P0C_OBJECT  = 0.5e9                     # Independent reference p0*c [eV]
B_FIELD     = 0.2                       # Signed field parameter B_y [T]
LENGTH      = 1.0                       # [m]

################################################################################
# Define a named reference particle
################################################################################

########################################
# Define an expression-controlled reference
########################################
env = xt.Environment()
env['p0c'] = P0C_START

env.new_particle(
    'proton_reference',
    mass0   = xt.PROTON_MASS_EV,
    q0      = 1,
    p0c     = ['p0c'],
)
env.particle_ref = 'proton_reference'

########################################
# Normalize a fixed field to the reference rigidity
########################################
# For this q0 = +1 reference, signed B_y / rigidity0 sets k0 [m^-1].
# Keep angle = 0: the design trajectory is straight even with a nonzero field.
# This is a fixed-field spectrometer definition, not an accelerating lattice.
env['field_tesla'] = B_FIELD
env.new(
    'spectrometer', xt.Bend,
    length  = LENGTH,
    angle   = 0.0,
    k0      = env.ref['field_tesla'] / env.ref['proton_reference'].rigidity0[0],
)
line = env.new_line(
    name        = 'spectrometer_line',
    components  = ['spectrometer'],
)

########################################
# Inspect the initial momentum and strength
########################################
print('Starting reference p0*c [GeV]:', line.particle_ref.p0c[0] * 1e-9)
print('Starting k0 [m^-1]:', line['spectrometer'].k0)

################################################################################
# Change the shared reference
################################################################################

########################################
# Change the momentum variable
########################################
# Both the line reference and the field-normalized strength depend on the
# same named particle. Changing p0c therefore preserves the stated field.
env['p0c'] = P0C_AFTER

########################################
# Recover the field from the updated strength
########################################
print('Changed reference p0*c [GeV]:', line.particle_ref.p0c[0] * 1e-9)
print('Changed k0 [m^-1]:', line['spectrometer'].k0)
print('B_y recovered from k0 * rigidity0 [T]:',
      line['spectrometer'].k0 * line.particle_ref.rigidity0[0])

################################################################################
# Use an independent reference object
################################################################################

########################################
# Copy the particle values into a separate line
########################################
# A copied particle object carries values, not the named particle's expression.
independent = xt.Line(
    elements        = [xt.Drift(length = LENGTH)],
    particle_ref    = line.particle_ref.copy(),
)
env['p0c'] = P0C_START
print('Named and copied reference p0*c [GeV]:',
      line.particle_ref.p0c[0] * 1e-9,
      independent.particle_ref.p0c[0] * 1e-9)

########################################
# Change only the independent reference
########################################
independent.particle_ref.p0c = P0C_OBJECT
print('After changing only the independent reference [GeV]:',
      line.particle_ref.p0c[0] * 1e-9,
      independent.particle_ref.p0c[0] * 1e-9)

########################################
# Restore the independent reference values
########################################
restored = xt.Line.from_dict(independent.to_dict())
print('Independent and restored reference p0*c [GeV]:',
      independent.particle_ref.p0c[0] * 1e-9,
      restored.particle_ref.p0c[0] * 1e-9)
restored.particle_ref.p0c = P0C_AFTER
print('After changing only the restored reference [GeV]:',
      independent.particle_ref.p0c[0] * 1e-9,
      restored.particle_ref.p0c[0] * 1e-9)
