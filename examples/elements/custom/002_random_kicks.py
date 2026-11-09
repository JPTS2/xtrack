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
# Choose the random kick ensemble
################################################################################
P0C             = 1e9             # Reference momentum [eV/c], proton
SIGMA_PX        = 1e-6            # RMS normalized horizontal kick per passage
NUM_PARTICLES   = 4096
NUM_PASSAGES    = 16
SEED_OFFSET     = 1234            # Each particle receives its own nonzero seed

################################################################################
# Define independent Gaussian kicks
################################################################################

########################################
# Include the Xtrack random generator
########################################
class GaussianKick(xt.BeamElement):
    _xofields   = {'sigma_px': xo.Float64}
    _depends_on = [xt.RandomNormal]

    ########################################
    # Draw from the current particle's stream
    ########################################
    _extra_c_sources = [r'''
        #include "xtrack/headers/track.h"

        GPUFUN
        void GaussianKick_track_local_particle(
                GaussianKickData el, LocalParticle* part0) {
            double sigma = GaussianKickData_get_sigma_px(el);
            START_PER_PARTICLE_BLOCK(part0, part);
                double kick = sigma * RandomNormal_generate(part);
                LocalParticle_add_to_px(part, kick);
            END_PER_PARTICLE_BLOCK;
        }
    ''']

################################################################################
# Seed the streams and track the ensemble
################################################################################

########################################
# Initialize and copy the random state
########################################
context = xo.ContextCpu()
context.allow_kernel_compilation = True
element = GaussianKick(_context = context, sigma_px = SIGMA_PX)
initial = xt.Particles(
    _context    = context,
    p0c         = P0C,
    x           = np.zeros(NUM_PARTICLES))
seeds = SEED_OFFSET + np.arange(NUM_PARTICLES, dtype = np.uint32)
# Xtrack currently exposes explicit per-particle seeding through this
# underscored initializer. Copying Particles also copies their RNG states.
initial._init_random_number_generator(seeds = seeds)
particles = initial.copy()
replayed  = initial.copy()

########################################
# Repeat the same sequence on the same CPU
########################################
for _ in range(NUM_PASSAGES):
    element.track(particles)
    element.track(replayed)

################################################################################
# Inspect diffusion and replay
################################################################################
# Independent zero-mean kicks add variances. This is a stochastic kick model,
# without damping or an associated longitudinal energy exchange.
expected_rms = SIGMA_PX * np.sqrt(NUM_PASSAGES)
print('Expected RMS px:', expected_rms)
print('Measured RMS px:', np.std(particles.px))
print('Mean px:', np.mean(particles.px))
print('Maximum replay difference:',
      np.max(np.abs(particles.px - replayed.px)))
