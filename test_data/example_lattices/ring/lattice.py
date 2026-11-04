# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
import numpy as np
import xtrack as xt

################################################################################
# Model parameters
################################################################################
P0C         = 1.0e9                      # Reference p0*c [eV]
N_CELLS     = 8
CELL_LENGTH = 5.0                        # [m]
KQF         = 1.194864989644536          # Focusing quadrupole k1 [m^-2]
KQD         = -1.4872100226207832        # Defocusing quadrupole k1 [m^-2]

################################################################################
# Elements and strength variables
################################################################################
env        = xt.get_environment()
env['kqf'] = KQF
env['kqd'] = KQD

env.new('qf', xt.Quadrupole, length = 0.3, k1 = 'kqf')
env.new('qd', xt.Quadrupole, length = 0.3, k1 = 'kqd')
env.new(
    'bend', xt.Bend,
    length           = 1.5,
    angle            = 2 * np.pi / (2 * N_CELLS),
    model            = 'adaptive',
    edge_entry_model = 'linear',
    edge_exit_model  = 'linear',
)
env.new('observation_point', xt.Marker)

################################################################################
# Ring layout
################################################################################
# Positions refer to element entrances. Each 5 m cell contains two bends.
components = []
for cell in range(N_CELLS):
    s_start = cell * CELL_LENGTH
    if cell in (0, 4):
        components.append(
            env.new(f'obs.{cell}', 'observation_point', at = s_start)
        )
    components.extend([
        env.new(f'qf.{cell + 1}', 'qf', at = s_start),
        env.new(f'bf.{cell + 1}', 'bend', at = s_start + 0.8),
        env.new(f'qd.{cell + 1}', 'qd', at = s_start + 2.5),
        env.new(f'bd.{cell + 1}', 'bend', at = s_start + 3.3),
    ])

ring = env.new_line(
    name       = 'ring',
    components = components,
    length     = N_CELLS * CELL_LENGTH,
    refer      = 'start',
)
ring.set_particle_ref('electron', p0c = P0C)
