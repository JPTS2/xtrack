# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
import numpy as np
import xobjects as xo
import xtrack as xt

################################################################################
# Choose the integrated quadrupole strength
################################################################################
P0C     = 1e9                    # Reference momentum [eV/c], proton
K1L     = 0.2                    # Normal quadrupole integrated strength [m^-1]
X       = np.array([-2e-3, 0, 1e-3])  # Horizontal positions [m]
Y       = np.array([1e-3, 0, -2e-3])  # Vertical positions [m]

################################################################################
# Define the element and its tracking kernel
################################################################################

########################################
# Declare stored parameters
########################################
# These underscored class attributes are the BeamElement authoring hooks.
class ThinQuadrupole(xt.BeamElement):
    _xofields = {'k1l': xo.Float64}

    ########################################
    # Apply a normal thin quadrupole kick
    ########################################
    # px and py are canonical momenta divided by the reference momentum p0.
    # This reference-species map assumes mass_ratio = charge_ratio = 1.
    # It leaves positions, delta and the reference momentum unchanged.
    _extra_c_sources = [r'''
        #include "xtrack/headers/track.h"

        GPUFUN
        void ThinQuadrupole_track_local_particle(
                ThinQuadrupoleData el, LocalParticle* part0) {
            double k1l = ThinQuadrupoleData_get_k1l(el);
            START_PER_PARTICLE_BLOCK(part0, part);
                LocalParticle_add_to_px(part,
                    -k1l * LocalParticle_get_x(part));
                LocalParticle_add_to_py(part,
                     k1l * LocalParticle_get_y(part));
            END_PER_PARTICLE_BLOCK;
        }
    ''']

################################################################################
# Compare with the built-in magnetic kick
################################################################################

########################################
# Compile and track the custom element
########################################
context   = xo.ContextCpu()
context.allow_kernel_compilation = True
element   = ThinQuadrupole(_context = context, k1l = K1L)
particles = xt.Particles(
    _context    = context,
    p0c         = P0C,
    x           = X,
    y           = Y)
built_in = particles.copy()
# The first call compiles the C kernel; subsequent calls reuse it.
element.track(particles)

########################################
# Inspect both maps at the same coordinates
########################################
xt.Multipole(_context = context, knl = [0, K1L]).track(built_in)
xt.Table(
    data    = {
        'particle_id':   particles.particle_id,
        'x':             particles.x,
        'y':             particles.y,
        'px_custom':     particles.px,
        'px_builtin':    built_in.px,
        'py_custom':     particles.py,
        'py_builtin':    built_in.py},
    index   = 'particle_id').show(maxwidth = 'full')
