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

################################################################################
# Define Gaussian and bounded-uniform kicks
################################################################################

########################################
# Include the Xtrack random generator
########################################
class RandomKick(xt.BeamElement):
    _xofields   = {'sigma_px': xo.Float64, 'use_uniform': xo.Int64}
    _depends_on = [xt.RandomNormal, xt.RandomUniform]
    needs_rng   = True

    ########################################
    # Draw from the current particle's stream
    ########################################
    _extra_c_sources = [r'''
        #include "xtrack/headers/track.h"

        GPUFUN
        void RandomKick_track_local_particle(
                RandomKickData el, LocalParticle* part0) {
            double sigma = RandomKickData_get_sigma_px(el);
            int use_uniform = RandomKickData_get_use_uniform(el);
            START_PER_PARTICLE_BLOCK(part0, part);
                double kick;
                if (use_uniform) {
                    // Uniform[-sqrt(3)*sigma, sqrt(3)*sigma] has RMS sigma.
                    kick = sigma * sqrt(12.)
                        * (RandomUniform_generate(part) - 0.5);
                } else {
                    kick = sigma * RandomNormal_generate(part);
                }
                LocalParticle_add_to_px(part, kick);
            END_PER_PARTICLE_BLOCK;
        }
    ''']

################################################################################
# Initialize the streams and track the ensemble
################################################################################

########################################
# Initialize and copy the random state
########################################
context = xo.ContextCpu()
context.allow_kernel_compilation = True
element = RandomKick(
    _context    = context,
    sigma_px    = SIGMA_PX,
    use_uniform = False)
initial = xt.Particles(
    _context    = context,
    p0c         = P0C,
    x           = np.zeros(NUM_PARTICLES))
# Public sampling initializes the particle-local streams without changing px.
# Copying the particles then preserves the RNG state for exact replay.
samples = xt.RandomNormal(_context = context).generate(
    n_samples   = NUM_PARTICLES,
    particles   = initial)
print('Initial standard-normal sample RMS:', np.std(samples))
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

################################################################################
# Change the single-passage distribution
################################################################################

########################################
# Keep the same RMS with bounded-uniform kicks
########################################
uniform_element = RandomKick(
    _context    = context,
    sigma_px    = SIGMA_PX,
    use_uniform = True)
uniform_particles = initial.copy()
uniform_element.track(uniform_particles)
print('Uniform single-passage expected RMS:', SIGMA_PX)
print('Uniform single-passage measured RMS:', np.std(uniform_particles.px))
print('Uniform kick bounds:', -np.sqrt(3) * SIGMA_PX,
      np.sqrt(3) * SIGMA_PX)
print('Observed minimum and maximum:',
      np.min(uniform_particles.px), np.max(uniform_particles.px))

########################################
# Distinguish the generator contracts
########################################
# RandomUniformAccurate gives finer unit-interval resolution. The exponential
# distribution is positive with unit mean, rather than zero mean and unit RMS.
for generator in [xt.RandomUniformAccurate, xt.RandomExponential]:
    samples = generator(_context = context).generate(
        n_samples   = NUM_PARTICLES,
        particles   = initial.copy())
    print(generator.__name__, 'mean / std:', np.mean(samples), np.std(samples))
# Rutherford scattering needs its own physical parameters; see its studies in
# examples/random_number_generator/ rather than treating it as generic noise.
