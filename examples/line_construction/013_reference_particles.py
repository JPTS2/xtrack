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
P0C_START   = 1.0e9                     # Proton reference momentum times c [eV]
P0C_AFTER   = 2.0e9                     # [eV]
P0C_OBJECT  = 0.5e9                     # Independent momentum times c [eV]
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

env.new_particle('proton_reference', 'proton', p0c = 'p0c')

env.particle_ref = 'proton_reference'

########################################
# Normalize a fixed field to the reference rigidity
########################################
# For this q0 = +1 reference, signed B_y / rigidity0 sets k0 [m^-1].
# Keep angle = 0: the reference geometry is straight even with a nonzero field.
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
print('Start with a fixed-field spectrometer'
      '\n-------------------------------------')
print('Proton reference; straight reference geometry (angle = 0).')
print('k0 = B_y / rigidity0 links the strength to the named reference.')
print(f'Reference charge [e]: {line.particle_ref.q0:g}')
print(f'Reference momentum [GeV/c]: {line.particle_ref.p0c[0] * 1e-9:.3f}')
print(f'Reference rigidity [T m]: {line.particle_ref.rigidity0[0]:.6f}')
print(f'B_y [T]: {B_FIELD:.3f}')
print(f'k0 [m^-1]: {line["spectrometer"].k0:.6f}')

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
print('\nChange reference momentum at fixed field'
      '\n----------------------------------------')
print('k0 changes with rigidity; the recovered physical field stays fixed.')
print(f'Reference momentum [GeV/c]: {line.particle_ref.p0c[0] * 1e-9:.3f}')
print(f'Reference rigidity [T m]: {line.particle_ref.rigidity0[0]:.6f}')
print(f'k0 [m^-1]: {line["spectrometer"].k0:.6f}')
print('Recovered B_y = k0 * rigidity0 [T]: '
      f'{line["spectrometer"].k0 * line.particle_ref.rigidity0[0]:.3f}')

################################################################################
# Use an independent reference object
################################################################################

########################################
# Copy the particle values into a separate line
########################################
# A copied particle object carries values, not the named particle's expression.
print('\nCopy reference values into a separate line'
      '\n------------------------------------------')
independent = xt.Line(
    elements        = [xt.Drift(length = LENGTH)],
    particle_ref    = line.particle_ref.copy(),
)
print(f'Copied momentum [GeV/c]: '
      f'{independent.particle_ref.p0c[0] * 1e-9:.3f}')
env['p0c'] = P0C_START
print('After changing the original momentum variable:')
print(f'Named reference momentum [GeV/c]: '
      f'{line.particle_ref.p0c[0] * 1e-9:.3f}')
print(f'Unchanged copied momentum [GeV/c]: '
      f'{independent.particle_ref.p0c[0] * 1e-9:.3f}')

########################################
# Change only the independent reference
########################################
independent.particle_ref.p0c = P0C_OBJECT
print('\nChange only the copied reference\n--------------------------------')
print(f'Named reference momentum [GeV/c]: '
      f'{line.particle_ref.p0c[0] * 1e-9:.3f}')
print(f'Copied reference momentum [GeV/c]: '
      f'{independent.particle_ref.p0c[0] * 1e-9:.3f}')

########################################
# Restore the independent reference values
########################################
print('\nReconstruct from a dictionary\n-----------------------------')
restored = xt.Line.from_dict(independent.to_dict())
print(f'Independent reference momentum [GeV/c]: '
      f'{independent.particle_ref.p0c[0] * 1e-9:.3f}')
print(f'Restored reference momentum [GeV/c]: '
      f'{restored.particle_ref.p0c[0] * 1e-9:.3f}')
restored.particle_ref.p0c = P0C_AFTER
print('\nChange only the restored reference'
      '\n----------------------------------')
print(f'Unchanged independent momentum [GeV/c]: '
      f'{independent.particle_ref.p0c[0] * 1e-9:.3f}')
print(f'Modified restored momentum [GeV/c]: '
      f'{restored.particle_ref.p0c[0] * 1e-9:.3f}')
