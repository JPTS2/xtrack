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
# User parameters
################################################################################

########################################
# Geometry
########################################
CHORD_LENGTH    = 1.0                   # Rectangular magnet chord [m]
BEND_ANGLE      = 0.2                   # Reference bending angle [rad]
CHANGED_ANGLE   = 0.3                   # Fixed-chord geometry change [rad]

########################################
# Reference particle and vertical probe
########################################
P0C             = 1.0e9                 # Proton reference p0*c [eV]
Y_START         = 1.0e-3                # [m]; x = px = py = delta = 0

################################################################################
# Compare sector and rectangular bend geometry
################################################################################

########################################
# Use the same reference arc and angle
########################################
rectangular = xt.RBend(
    length_straight = CHORD_LENGTH,
    angle           = BEND_ANGLE,
    rbend_model     = 'curved-body',
)
sector = xt.Bend(length = rectangular.length, angle = BEND_ANGLE)
# RBend is specified by its chord. Bend is specified by its reference arc.
# Both use k0 = from_h here: the normalized field follows angle / arc length.
expected_arc = CHORD_LENGTH / np.sinc(BEND_ANGLE / (2 * np.pi))
print('Rectangular chord and arc lengths [m]:',
      rectangular.length_straight, rectangular.length)
print('Analytic arc length [m]:', expected_arc)
print('Sector arc length [m]:', sector.length)
print('Common reference curvature [m^-1]:', BEND_ANGLE / sector.length)

########################################
# Change angle at fixed chord
########################################
changed = rectangular.copy()
changed.angle = CHANGED_ANGLE
print('Changed chord and arc lengths [m]:',
      changed.length_straight, changed.length)
print('Changed analytic arc length [m]:',
      CHORD_LENGTH / np.sinc(CHANGED_ANGLE / (2 * np.pi)))
print('Changed reference curvature and field k0 [m^-1]:',
      CHANGED_ANGLE / changed.length)

################################################################################
# Inspect edge focusing at the original geometry
################################################################################

########################################
# Interpret zero user edge angles
########################################
# For this curved-body RBend, effective face angles include angle/2.
# A sector bend with zero user edge angles has zero face incidence instead.
# This comparison uses linear edges with zero fringe integrals.
print('Sector effective entry angle [rad]:', sector.edge_entry_angle)
print('RBend effective entry angle [rad]:',
      rectangular.edge_entry_angle + rectangular.angle / 2)

########################################
# Compare enabled and disabled edge maps
########################################
outputs = {}
for label, magnet in [('sector', sector), ('rectangular', rectangular)]:
    for edges_active in [False, True]:
        element = magnet.copy()
        element.edge_entry_active   = edges_active
        element.edge_exit_active    = edges_active
        element.edge_entry_model    = 'linear'
        element.edge_exit_model     = 'linear'
        line = xt.Line(elements = [element])
        line.set_particle_ref('proton', p0c = P0C)
        particles = line.build_particles(y = Y_START)
        line.track(particles)
        outputs[label, edges_active] = particles
        print(label, 'edges active:', edges_active,
              'exit y [mm]:', particles.y[0] * 1e3,
              'exit P_y/p0 [1e-6]:', particles.py[0] * 1e6)
