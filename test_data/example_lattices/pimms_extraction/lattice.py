# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

from pathlib import Path

import xtrack as xt

################################################################################
# Load the extraction operating point
################################################################################
env         = xt.get_environment()
model_path  = Path(__file__).resolve().parent
env.call(model_path / 'strengths.py')

################################################################################
# Define the compact tutorial geometry
################################################################################

########################################
# Three quadrupole families and sixteen sector bends
########################################
env.new('qfa', xt.Quadrupole, length = 0.35, k1 = 'kqfa')
env.new('qfb', xt.Quadrupole, length = 0.35, k1 = 'kqfb')
env.new('qd', xt.Quadrupole, length = 0.35, k1 = 'kqd')
env.new(
    'mb', xt.Bend,
    length              = 1.661,
    angle               = 0.3926990817,
    edge_entry_angle    = 0.19634954085,
    edge_exit_angle     = 0.19634954085,
)

########################################
# Chromatic and extraction sextupoles
########################################
for name, knob in [('sd1', 'ksd'), ('sd2', 'ksd'),
                   ('sf1', 'ksf'), ('sf2', 'ksf'),
                   ('se1', 'kse1'), ('se2', 'kse2')]:
    env.new(name, xt.Sextupole, length = 0.2, k2 = knob)
env.new('extr_septum', xt.Marker)

################################################################################
# Place the original tutorial magnets at their stated centres
################################################################################
# Positions [m] are from the tutorial's PIMM.seq, based on CERN-PS-99-010-DI.
placements  = [
    ('extr_septum', 0.0),
    ('qfa.1', 'qfa', 2.3875), ('mb', 3.8125), ('qd.1', 'qd', 5.2925),
    ('mb', 7.0475), ('qfa.2', 'qfa', 8.3275), ('sd1', 10.29),
    ('qfb.1', 'qfb', 10.6775), ('mb', 12.1325), ('qd.2', 'qd', 14.2625),
    ('mb', 15.9175), ('qfb.2', 'qfb', 17.1975), ('qfb.3', 'qfb', 20.2475),
    ('mb', 21.7025), ('qd.3', 'qd', 23.1825), ('sf1', 24.045),
    ('mb', 25.4875), ('qfb.4', 'qfb', 26.7675), ('qfa.3', 'qfa', 29.1175),
    ('mb', 30.5725), ('qd.4', 'qd', 32.1525), ('mb', 33.8075),
    ('qfa.4', 'qfa', 35.0575), ('se1', 35.6575), ('se2', 39.4075),
    ('qfa.5', 'qfa', 40.0075), ('mb', 41.4325), ('qd.5', 'qd', 42.9125),
    ('mb', 44.6675), ('qfa.6', 'qfa', 45.9475), ('sd2', 47.91),
    ('qfb.5', 'qfb', 48.2975), ('mb', 49.7525), ('qd.6', 'qd', 51.8825),
    ('mb', 53.5375), ('qfb.6', 'qfb', 54.8175), ('qfb.7', 'qfb', 57.8675),
    ('mb', 59.3225), ('qd.7', 'qd', 60.8025), ('sf2', 61.665),
    ('mb', 63.1075), ('qfb.8', 'qfb', 64.3875), ('qfa.7', 'qfa', 66.7375),
    ('mb', 68.1925), ('qd.8', 'qd', 69.7725), ('mb', 71.4275),
    ('qfa.8', 'qfa', 72.6775),
]
components  = []
for item in placements:
    if len(item) == 3:
        name, prototype, position = item
        env.new(name, prototype)
    else:
        name, position = item
    components.append(env.place(name, at = position))

line = env.new_line(
    name        = 'pimms',
    length      = 75.24,
    refer       = 'centre',
    components  = components,
)
line.set_particle_ref('proton', kinetic_energy0 = 200e6)
line.configure_bend_model(
    edge = 'full', core = 'adaptive', num_multipole_kicks = 5)
