# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
import xtrack as xt

################################################################################
# Model parameters
################################################################################
P0C = 1.0e9                              # Reference p0*c [eV]
KQF = 0.35                               # Focusing quadrupole k1 [m^-2]
KQD = -0.30                              # Defocusing quadrupole k1 [m^-2]

################################################################################
# Elements and strength variables
################################################################################
env        = xt.get_environment()
env['kqf'] = KQF
env['kqd'] = KQD

env.new('qf', xt.Quadrupole, length = 0.25, k1 = 'kqf')
env.new('qd', xt.Quadrupole, length = 0.25, k1 = 'kqd')
env.new(
    'b1', xt.Bend, length = 0.9, angle = 0.045,
    model = 'adaptive', edge_entry_model = 'linear', edge_exit_model = 'linear',
)
env.new(
    'b2', xt.Bend, length = 0.9, angle = -0.030,
    model = 'adaptive', edge_entry_model = 'linear', edge_exit_model = 'linear',
)

################################################################################
# Transport-line layout
################################################################################
transport = env.new_line(
    name       = 'transport',
    length     = 7.1,
    refer      = 'start',
    components = [
        env.new('qf.1', 'qf', at = 0.0),
        env.new('b1.1', 'b1', at = 1.0),
        env.new('qd.1', 'qd', at = 3.05),
        env.new('b2.1', 'b2', at = 4.05),
        env.new('qf.2', 'qf', at = 6.1),
    ],
)
transport.set_particle_ref('electron', p0c = P0C)
