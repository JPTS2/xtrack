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
P0C             = 1.0e9                  # Reference momentum times c [eV]
KQF             = 1.8                    # Focusing quadrupole k1 [m^-2]
KQD             = -1.8                   # Defocusing quadrupole k1 [m^-2]
CELL_LENGTH     = 6.1                    # [m]
N_CELLS         = 3

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
components = []
for cell in range(N_CELLS):
    offset = cell * CELL_LENGTH
    components.extend([
        env.new(f'qf.{cell + 1}', 'qf', at = offset),
        env.new(f'b1.{cell + 1}', 'b1', at = offset + 1.0),
        env.new(f'qd.{cell + 1}', 'qd', at = offset + 3.05),
        env.new(f'b2.{cell + 1}', 'b2', at = offset + 4.05),
    ])

transport = env.new_line(
    name       = 'transport',
    length     = N_CELLS * CELL_LENGTH,
    refer      = 'start',
    components = components,
)
transport.set_particle_ref('electron', p0c = P0C)
