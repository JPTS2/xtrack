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

########################################
# Scalar strengths
########################################
QUAD_LENGTH     = 0.4                   # [m]
KQ_START        = 0.2                   # [m^-2]
KQ_AFTER        = 0.3                   # [m^-2]
KQ_PROBE        = 0.4                   # [m^-2]
KQ_CORRECTION   = 0.01                  # [m^-2]
QD_OVERRIDE     = -0.1                  # [m^-2]

########################################
# Integrated multipole strengths
########################################
K1L_START       = 0.03                  # Normal quadrupole [m^-1]
K1L_AFTER       = 0.05                  # [m^-1]
K2L_START       = 0.01                  # Normal sextupole [m^-2]
K2L_AFTER       = 0.015                 # [m^-2]
K1L_OVERRIDE    = 0.02                  # Independent quadrupole [m^-1]

################################################################################
# Control two magnets with one variable
################################################################################

########################################
# Define expressions instead of numerical strengths
########################################
env = xt.Environment()
env['kq'] = KQ_START
env.new('qf', xt.Quadrupole, length = QUAD_LENGTH, k1 = 'kq')
env.new('qd', xt.Quadrupole, length = QUAD_LENGTH, k1 = '-kq')

print('Two magnets controlled by kq\n----------------------------')
print(f'Initial kq [m^-2]: {env["kq"]:.3f}')
print(f'qf.k1 [m^-2]: {env["qf"].k1:.3f}')
print(f'qd.k1 [m^-2]: {env["qd"].k1:.3f}')

########################################
# Change the source and observe both magnets
########################################
env['kq'] = KQ_AFTER
print(f'\nAfter setting kq [m^-2]: {env["kq"]:.3f}')
print(f'qf.k1 [m^-2]: {env["qf"].k1:.3f}')
print(f'qd.k1 [m^-2]: {env["qd"].k1:.3f}')
print('qf.k1 expression:', env.ref['qf'].k1.xdeps.expr)
print('qd.k1 expression:', env.ref['qd'].k1.xdeps.expr)

########################################
# Inspect which attributes the variable controls
########################################
print('\nSource variable and dependents\n------------------------------')
env.info('kq')

################################################################################
# Copy a value or retain a dependency
################################################################################

########################################
# Compare ordinary access with reference access
########################################
# Ordinary access reads the current number. Reference access keeps the link.
env['kq_snapshot'] = env['kq']
env['kq_linked']   = env.ref['kq']
print('\nCopy a value or retain a dependency'
      '\n-----------------------------------')
print('Before changing kq; values [m^-2]:')
env.vars.get_table().rows[['kq', 'kq_snapshot', 'kq_linked']].show()

########################################
# Change the source after making the copies
########################################
env['kq'] = KQ_PROBE
print('\nAfter changing kq; values [m^-2]:')
env.vars.get_table().rows[['kq', 'kq_snapshot', 'kq_linked']].show()
print(f'Evaluate 2 * kq [m^-2]: {env.eval("2 * kq"):.3f}')

################################################################################
# Detach one magnet from the source
################################################################################

########################################
# Replace its expression with a number
########################################
env['qd'].k1 = QD_OVERRIDE
print('\nReplace qd.k1 with a number\n---------------------------')
print(f'Before changing kq again [m^-2]: {env["kq"]:.3f}')
print(f'qf.k1 [m^-2]: {env["qf"].k1:.3f}')
print(f'qd.k1 [m^-2]: {env["qd"].k1:.3f}')

########################################
# Change kq and check which magnet still follows it
########################################
env['kq'] = KQ_START
print(f'\nAfter changing kq [m^-2]: {env["kq"]:.3f}')
print(f'qf.k1 [m^-2]: {env["qf"].k1:.3f}')
print(f'qd.k1 [m^-2]: {env["qd"].k1:.3f}')
print('qf.k1 expression:', env.ref['qf'].k1.xdeps.expr)
print('qd.k1 expression:', env.ref['qd'].k1.xdeps.expr)

################################################################################
# Add a correction while preserving the expression
################################################################################

########################################
# Apply the same correction through values and references
########################################
env.new('q_numeric', xt.Quadrupole, length = QUAD_LENGTH, k1 = 'kq')
env.new('q_linked', xt.Quadrupole, length = QUAD_LENGTH, k1 = 'kq')
# Numerical += replaces the expression with the corrected current value.
# Reference += adds the correction to the expression itself.
env['q_numeric'].k1 += KQ_CORRECTION
env.ref['q_linked'].k1 += KQ_CORRECTION

print('\nCorrect a value or an expression\n--------------------------------')
print(f'Initial kq [m^-2]: {env["kq"]:.3f}')
print(f'Correction [m^-2]: {KQ_CORRECTION:.3f}')
print('After the corrections; k1 [m^-2]:')
print(f'q_numeric.k1 [m^-2]: {env["q_numeric"].k1:.3f}')
print(f'q_linked.k1 [m^-2]: {env["q_linked"].k1:.3f}')
print('q_numeric.k1 expression:', env.ref['q_numeric'].k1.xdeps.expr)
print('q_linked.k1 expression:', env.ref['q_linked'].k1.xdeps.expr)

########################################
# Change the source to distinguish the two assignments
########################################
env['kq'] = KQ_AFTER
print(f'\nAfter changing kq [m^-2]: {env["kq"]:.3f}')
print(f'q_numeric.k1 [m^-2]: {env["q_numeric"].k1:.3f}')
print(f'q_linked.k1 [m^-2]: {env["q_linked"].k1:.3f}')

################################################################################
# Use expressions in multipole arrays
################################################################################

########################################
# Define two elements with linked array entries
########################################
env['k1l'] = K1L_START
env['k2l'] = K2L_START
# knl[n] and ksl[n] are integrated normal and skew strengths of order n.
# The skew quadrupole shares k1l only to demonstrate an array dependency.
env.new('multipole', xt.Multipole,
        knl = [0.0, 'k1l', 'k2l'], ksl = [0.0, 'k1l'])
env.new('second_multipole', xt.Multipole, knl = [0.0, 'k1l', 'k2l'])

print('\nLinked multipole arrays\n-----------------------')
print('Index 0: dipole [1]; index 1: quadrupole [m^-1];')
print('index 2: sextupole [m^-2].')
print('Initial multipole.knl:', env['multipole'].knl.tolist())
print('Initial second_multipole.knl:', env['second_multipole'].knl.tolist())

########################################
# Change a shared strength
########################################
env['k1l'] = K1L_AFTER
print(f'\nAfter setting k1l [m^-1]: {env["k1l"]:.3f}')
print('multipole.knl:', env['multipole'].knl.tolist())
print('second_multipole.knl:', env['second_multipole'].knl.tolist())

########################################
# Detach one entry, then change both source variables
########################################
# Numerical assignment clears only the selected entry's expression.
env['second_multipole'].knl[1] = K1L_OVERRIDE
env['k1l'] = K1L_START
env['k2l'] = K2L_AFTER
print('\nOverride second_multipole.knl[1], then change k1l and k2l')
print(f'k1l [m^-1]: {env["k1l"]:.3f}; k2l [m^-2]: {env["k2l"]:.3f}')
print('multipole.knl:', env['multipole'].knl.tolist())
print('second_multipole.knl:', env['second_multipole'].knl.tolist())
print('second_multipole.knl[1] expression:',
      env.ref['second_multipole'].knl[1].xdeps.expr)
print('second_multipole.knl[2] expression:',
      env.ref['second_multipole'].knl[2].xdeps.expr)

################################################################################
# Extend arrays without losing their dependencies
################################################################################

########################################
# Add zero entries through order four
########################################
multipole_line = env.new_line(components = ['multipole', 'second_multipole'])
print('\nExtend multipole arrays\n-----------------------')
print('Before extension:')
print('multipole.knl:', env['multipole'].knl.tolist())
print('multipole.ksl:', env['multipole'].ksl.tolist())
multipole_line.extend_knl_ksl(order = 4)
print('\nAfter extension through index 4:')
print('multipole.knl:', env['multipole'].knl.tolist())
print('multipole.ksl:', env['multipole'].ksl.tolist())

########################################
# Change the source again after extending the arrays
########################################
env['k1l'] = K1L_AFTER
print(f'\nAfter setting k1l [m^-1]: {env["k1l"]:.3f}')
print('multipole.knl:', env['multipole'].knl.tolist())
print('multipole.ksl:', env['multipole'].ksl.tolist())
print('second_multipole.knl:', env['second_multipole'].knl.tolist())
