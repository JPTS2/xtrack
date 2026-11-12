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
      changed.h, changed.h)

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

################################################################################
# Distinguish geometric curvature from the powered field
################################################################################

########################################
# Keep the field fixed while changing reference geometry
########################################
powered = rectangular.copy()
powered.k0 = rectangular.h
powered.angle = CHANGED_ANGLE
print('Geometry-following flag, h and effective k0 [m^-1]:',
      changed.k0_from_h, changed.h, changed.h)
print('Independently powered k0_from_h, h and k0 [m^-1]:',
      powered.k0_from_h, powered.h, powered.k0)

########################################
# Set geometry in either order and restore both bend types
########################################
angle_first = xt.RBend(angle = BEND_ANGLE)
angle_first.length_straight = CHORD_LENGTH
length_first = xt.RBend(length_straight = CHORD_LENGTH)
length_first.angle = BEND_ANGLE
print('Angle-first and length-first reference arcs [m]:',
      angle_first.length, length_first.length)
for magnet in [sector, rectangular, powered]:
    restored = type(magnet).from_dict(magnet.to_dict())
    print(type(magnet).__name__, 'restored arc, h, k0 setting, k0_from_h:',
          restored.length, restored.h, restored.k0, restored.k0_from_h)

################################################################################
# Compare rectangular body models at common reference frames
################################################################################

########################################
# Track the same off-axis launch through each model
########################################
# Full edge maps are used here, unlike the linear-edge illustration above.
# The geometry and field are identical; rbend_model selects the body map.
body_outputs = {}
for body in ['curved-body', 'straight-body']:
    magnet = rectangular.copy()
    magnet.rbend_model = body
    magnet.edge_entry_model = 'full'
    magnet.edge_exit_model = 'full'
    model_line = xt.Line(elements = {'bend': magnet})
    model_line.set_particle_ref('proton', p0c = P0C)
    probe = model_line.build_particles(x = 1e-3, y = Y_START, delta = 1e-3)
    model_line.track(probe)
    body_outputs[body] = probe
    survey = model_line.survey(include_element_frames = True)
    exit_frame = survey.get_all_frames('bend')['ref_end']
    global_exit = (exit_frame.XYZ + probe.x[0] * exit_frame.ex
                   + probe.y[0] * exit_frame.ey)
    print(body, 'exit (x, px, y, py):',
          probe.x[0], probe.px[0], probe.y[0], probe.py[0])
    print(body, 'global exit (X, Y, Z) [m]:', global_exit)
    print(body, 'reference exit tangent:', exit_frame.es)

########################################
# Inspect the body-model difference
########################################
for coordinate in ['x', 'px', 'y', 'py']:
    difference = (getattr(body_outputs['straight-body'], coordinate)
                  - getattr(body_outputs['curved-body'], coordinate))
    print('Straight minus curved body', coordinate, ':', difference[0])
