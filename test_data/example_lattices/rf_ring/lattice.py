# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
from pathlib import Path

import numpy as np
from scipy.constants import c
import xtrack as xt

################################################################################
# RF parameters
################################################################################
HARMONIC    = 20
VOLTAGE     = 1e5                        # [V]
PHASE       = np.pi                      # Stable above transition [rad]

################################################################################
# Load the native magnetic ring into this environment
################################################################################
env = xt.get_environment()
env.call(Path(__file__).resolve().parents[1] / 'ring/lattice.py')
magnetic = env['ring']

################################################################################
# Add one zero-length RF cavity at the end of the magnetic sequence
################################################################################
# The cavity is at s = 40 m, just before the ring entrance on the next turn.
# The public Cavity convention is abs(q0)*V*sin(phase - omega*zeta/(beta0*c))
# for the reference species. With this positive slip factor, phase pi focuses.
frequency   = (
    HARMONIC * magnetic.particle_ref.beta0[0] * c / magnetic.get_length())
rf_ring     = env.new_line(
    name        = 'rf_ring',
    components  = [
        magnetic,
        env.new('rf', xt.Cavity,
                voltage     = VOLTAGE,
                frequency   = frequency,
                phase       = PHASE)])
rf_ring.particle_ref = magnetic.particle_ref.copy()
