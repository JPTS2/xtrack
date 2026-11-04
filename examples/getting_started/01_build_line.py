# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

# Further examples: examples/line_construction/

################################################################################
# Required packages
################################################################################
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Reference particle
########################################
P0C             = 1.0e9                 # Electron reference p0*c [eV]

########################################
# Elements
########################################
QUAD_LENGTH     = 0.25                  # [m]
DRIFT_LENGTH    = 1.0                   # [m]
KQ              = 0.8                   # Quadrupole k1 magnitude [m^-2]

################################################################################
# Build the line
################################################################################
env = xt.Environment()
env['kq'] = KQ

env.new('drift', xt.Drift, length = DRIFT_LENGTH)
env.new('qf', xt.Quadrupole, length = QUAD_LENGTH, k1 = 'kq')
env.new('qd', xt.Quadrupole, length = QUAD_LENGTH, k1 = '-kq')

line = env.new_line(
    name       = 'short_line',
    components = ['drift', 'qf', 'drift', 'qd', 'drift'],
)
line.set_particle_ref('electron', p0c = P0C)

################################################################################
# Inspect the result
################################################################################
# s is measured at element entrances, in metres.
line.get_table().cols['name s_start s_end element_type'].show(maxrows = 12)
print(f'Line length: {line.get_length():.3f} m')
print(f'Reference p0*c: {line.particle_ref.p0c[0] * 1e-9:.3f} GeV')

################################################################################
# Change the quadrupole strengths
################################################################################
k1f_before = line['qf'].k1
k1d_before = line['qd'].k1

# Both strengths are expressions of kq, so changing it updates both magnets.
env['kq'] = 1.1 * KQ

print('\nQuadrupole k1 [m^-2]       Before       After')
print(f'qf                  {k1f_before:12.3f} {line["qf"].k1:11.3f}')
print(f'qd                  {k1d_before:12.3f} {line["qd"].k1:11.3f}')
