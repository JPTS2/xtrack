# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt

################################################################################
# User parameters
################################################################################
P0C             = 1e9                   # Proton reference p0*c [eV]
DAMPING_RATE    = 0.1                   # Fractional x reduction per pass
NOISE_SIGMA     = 1e-4                  # Added Gaussian x kick RMS [m]
N_PARTICLES     = 2048
N_TURNS         = 64

################################################################################
# Prescribe damping and excitation independently of a physical lattice
################################################################################

########################################
# Prepare native random streams and track the prescribed process
########################################
# This map applies x <- (1-DAMPING_RATE)*x + Gaussian(NOISE_SIGMA).
# These are assigned coefficients, not damping/excitation computed from magnets.
noisy_segment = xt.LineSegmentMap(
    longitudinal_mode   = 'frozen',
    damping_rate_x      = DAMPING_RATE,
    gauss_noise_ampl_x  = NOISE_SIGMA,
)
noisy_line = xt.Line(elements = [noisy_segment])
noisy_line.set_particle_ref('proton', p0c = P0C)
noisy_particles = noisy_line.build_particles(x = np.zeros(N_PARTICLES))
# Sampling initializes each particle's stream before the stochastic map uses it.
normal_sampler = xt.RandomNormal()
first_draw = normal_sampler.generate(
    n_samples   = N_PARTICLES,
    particles   = noisy_particles,
)
print('Native standard-normal sample mean and RMS:',
      np.mean(first_draw), np.std(first_draw))
noisy_line.track(
    noisy_particles,
    num_turns               = N_TURNS,
    turn_by_turn_monitor    = True,
)
noisy_record = noisy_line.record_last_track
noise_turns = noisy_record.at_turn[0]
measured_sigma = np.std(noisy_record.x, axis = 0)
expected_sigma = NOISE_SIGMA * np.sqrt(
    (1 - (1 - DAMPING_RATE)**(2 * noise_turns))
    / (1 - (1 - DAMPING_RATE)**2))
print('Assigned stationary x RMS [m]:',
      NOISE_SIGMA / np.sqrt(1 - (1 - DAMPING_RATE)**2))
print('Final recorded x RMS [m]:', measured_sigma[-1])

########################################
# Compare the ensemble RMS with the discrete process
########################################
fig_noise, ax_noise = plt.subplots(figsize = (8, 4))
ax_noise.plot(noise_turns, expected_sigma * 1e3, label = 'Assigned process')
ax_noise.plot(noise_turns, measured_sigma * 1e3, '.', label = 'Native tracking')
ax_noise.set_xlabel('Pass number (recorded at entrance)')
ax_noise.set_ylabel('Horizontal RMS [mm]')
ax_noise.set_title('Prescribed damping and Gaussian excitation')
ax_noise.legend()
ax_noise.grid()
fig_noise.tight_layout()
plt.show()
