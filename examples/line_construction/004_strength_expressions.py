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
KQ_START        = 0.2                   # [m^-2]
KQ_AFTER        = 0.3                   # [m^-2]
KQ_CORRECTION   = 0.01                  # [m^-2]
QD_OVERRIDE     = -0.1                  # [m^-2]

########################################
# Integrated multipole strengths
########################################
K1L_START       = 0.03                  # Normal quadrupole [m^-1]
K1L_AFTER       = 0.05                  # [m^-1]
K2L             = 0.01                  # Normal sextupole [m^-2]
K1L_OVERRIDE    = 0.02                  # Independent quadrupole [m^-1]

################################################################################
# Connect strengths to a variable
################################################################################

########################################
# Define the strength expressions
########################################
env = xt.Environment()
env['kq'] = KQ_START
line = env.new_line(
    components  = [
        env.new('qf', xt.Quadrupole, length = 0.4, k1 = 'kq'),
        env.new('drift', xt.Drift, length = 1.0),
        env.new('qd', xt.Quadrupole, length = 0.4, k1 = '-kq'),
    ],
)

########################################
# Numeric values and live references
########################################
# Ordinary access reads a number. A reference retains the dependency.
env['kq_snapshot']  = env['kq']
env['kq_linked']    = env.ref['kq']
print('Starting snapshot and linked value [m^-2]:',
      env['kq_snapshot'], env['kq_linked'])

########################################
# Change the controlling variable
########################################
print('Starting strengths [m^-2]:', line['qf'].k1, line['qd'].k1)
env['kq'] = KQ_AFTER
print('After changing kq [m^-2]:', line['qf'].k1, line['qd'].k1)
print('Snapshot and linked value after the change [m^-2]:',
      env['kq_snapshot'], env['kq_linked'])

########################################
# Inspect the dependencies
########################################
env.info('kq')
env.info('qd')

########################################
# Inspect variables and evaluate a formula
########################################
env.vars.get_table().show()
print('Twice the current kq [m^-2]:', env.eval('2 * kq'))

################################################################################
# Add a correction to a live expression
################################################################################

########################################
# Compare numerical and reference assignment
########################################
# Numerical += reads the current value and replaces the expression.
# Reference += adds to the expression itself.
env.new('q_numeric', xt.Quadrupole, length = 0.4, k1 = 'kq')
env.new('q_linked', xt.Quadrupole, length = 0.4, k1 = 'kq')
env['q_numeric'].k1 += KQ_CORRECTION
env.ref['q_linked'].k1 += KQ_CORRECTION
env.info('q_numeric')
env.info('q_linked')

########################################
# Change the source variable again
########################################
env['kq'] = KQ_START
print('Numerical and linked corrected k1 [m^-2]:',
      env['q_numeric'].k1, env['q_linked'].k1)

################################################################################
# Replace an expression with a number
################################################################################
# A numerical assignment detaches this attribute from its former expression.
line['qd'].k1 = QD_OVERRIDE
env['kq'] = KQ_START
print('After detaching qd and restoring kq [m^-2]:',
      line['qf'].k1, line['qd'].k1)
env.info('qd')

################################################################################
# Expressions in array fields
################################################################################

########################################
# Define integrated strengths
########################################
env['k1l'] = K1L_START
env['k2l'] = K2L

# Multipole knl[n] is the integrated normal strength of order n. This is a
# thin element: knl[1] has units m^-1 and knl[2] has units m^-2.
env.new(
    'multipole', xt.Multipole,
    knl = [0.0, 'k1l', 'k2l'],
    ksl = [0.0, 'k1l'],
)
env.new('second_multipole', xt.Multipole, knl = [0.0, 'k1l', 'k2l'])
print('Starting arrays:',
      list(env['multipole'].knl), list(env['second_multipole'].knl))

########################################
# Change the controlling variable
########################################
env['k1l'] = K1L_AFTER
print('Both follow k1l:',
      list(env['multipole'].knl), list(env['second_multipole'].knl))

########################################
# Override one array entry
########################################
# Assigning a number to one entry clears only that entry's expression.
env['second_multipole'].knl[1] = K1L_OVERRIDE
env['k1l'] = K1L_START
print('Original and detached second multipole:',
      list(env['multipole'].knl), list(env['second_multipole'].knl))
env.info('second_multipole')

################################################################################
# Extend the multipole order while retaining expressions
################################################################################

########################################
# Grow both normal and skew arrays
########################################
multipole_line = env.new_line(components = ['multipole', 'second_multipole'])
multipole_line.extend_knl_ksl(order = 4)
print('Normal and skew arrays after extension:')
print(list(env['multipole'].knl), list(env['multipole'].ksl))

########################################
# Check that the original strengths still respond
########################################
env['k1l'] = K1L_AFTER
print('Original and detached arrays after changing k1l:')
print(list(env['multipole'].knl), list(env['second_multipole'].knl))
print('Linked skew array:', list(env['multipole'].ksl))
env.info('multipole')
