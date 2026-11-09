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
env.new('multipole', xt.Multipole, knl = [0.0, 'k1l', 'k2l'])
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
