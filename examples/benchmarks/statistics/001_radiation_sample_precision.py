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

########################################
# Fixed physical and numerical single-bend settings
########################################
P0C             = 10e9                  # Reference p0*c [eV]
BEND_LENGTH     = 1.0                   # [m]
BEND_ANGLE      = 0.005                 # [rad]
N_KICKS         = 8

########################################
# Independent samples and bootstrap precision estimates
########################################
SAMPLE_SIZES    = np.array([256, 1024, 4096, 16384])
N_REPEATS       = 24
N_REFERENCE     = 65536
N_BOOTSTRAPS    = 400
BOOTSTRAP_SEED  = 18273                 # Resampling only, not native radiation

################################################################################
# Construct one native stochastic-radiation experiment
################################################################################
line    = xt.Line(elements = {
    'bend': xt.Bend(
        length                  = BEND_LENGTH,
        angle                   = BEND_ANGLE,
        model                   = 'bend-kick-bend',
        num_multipole_kicks     = N_KICKS)})
line.set_particle_ref('electron', p0c = P0C)
line.configure_radiation(model = 'quantum')

################################################################################
# Estimate the one-passage loss distribution with an independent large sample
################################################################################

########################################
# Track fresh particles with native automatically initialized RNG streams
########################################
reference   = line.build_particles(x = np.zeros(N_REFERENCE))
energy_in   = reference.energy.copy()
line.track(reference)
loss        = np.asarray(energy_in - reference.energy) / 1e3
# Loss is in keV. Fresh batches get fresh native streams; realized values vary.
# The NumPy generator below controls bootstrap resampling only.

########################################
# Predict sampling errors without assuming Gaussian photon-energy losses
########################################
mean_ref        = np.mean(loss)
variance        = np.var(loss, ddof = 1)
rms_ref         = np.sqrt(variance)
moment4         = np.mean((loss - mean_ref)**4)
mean_error      = rms_ref / np.sqrt(SAMPLE_SIZES)
variance_error  = np.sqrt(np.maximum(0,
    (moment4 - (SAMPLE_SIZES - 3) / (SAMPLE_SIZES - 1) * variance**2)
    / SAMPLE_SIZES))
rms_error       = variance_error / (2 * rms_ref)
# The mean is unbiased; ddof=1 makes the variance unbiased. Its square root
# (the rms spread) retains finite-sample bias. rms_error uses a large-N
# delta-method approximation and the independent reference fourth moment.
print('Independent reference mean / rms spread [keV]:', mean_ref, rms_ref)
print('Reference mean standard error [keV]:', rms_ref / np.sqrt(N_REFERENCE))

################################################################################
# Repeat each sample size rather than subdividing a single realization
################################################################################

########################################
# Collect independent estimates of the mean and rms spread
########################################
means       = np.empty((len(SAMPLE_SIZES), N_REPEATS))
spreads     = np.empty_like(means)
for index, size in enumerate(SAMPLE_SIZES):
    for repeat in range(N_REPEATS):
        particles   = line.build_particles(x = np.zeros(size))
        energy_in   = particles.energy.copy()
        line.track(particles)
        loss        = np.asarray(energy_in - particles.energy) / 1e3
        means[index, repeat]    = np.mean(loss)
        spreads[index, repeat]  = np.std(loss, ddof = 1)

########################################
# Quantify uncertainty in the observed repeat-to-repeat scatter itself
########################################
rng         = np.random.default_rng(BOOTSTRAP_SEED)
indices     = rng.integers(0, N_REPEATS, (N_BOOTSTRAPS, N_REPEATS))
scatter     = []
intervals   = []
for estimates in [means, spreads]:
    scatter.append(np.std(estimates, axis = 1, ddof = 1))
    bootstraps  = np.std(estimates[:, indices], axis = 2, ddof = 1)
    intervals.append(np.quantile(bootstraps, [0.025, 0.975], axis = 1))
# These percentile intervals describe uncertainty from only N_REPEATS batches.
# They are approximate, and do not certify native-model accuracy or rare tails.
xt.Table(
    data    = {
        'N':                    SAMPLE_SIZES,
        'mean_of_means_keV':    np.mean(means, axis = 1),
        'mean_scatter_keV':     scatter[0],
        'mean_SE_prediction':   mean_error,
        'mean_rms_keV':         np.mean(spreads, axis = 1),
        'rms_scatter_keV':      scatter[1],
        'rms_SE_approx':        rms_error},
    index   = 'N').show(maxwidth = 'full')

################################################################################
# Compare measured precision with the expected sample-size scaling
################################################################################
fig, axes   = plt.subplots(1, 2, figsize = (11, 4))
predictions = [mean_error, rms_error]
names       = ['Sample mean', 'Sample rms spread']
for axis, observed, bounds, prediction, name in zip(
        axes, scatter, intervals, predictions, names):
    axis.plot(SAMPLE_SIZES, observed, 'o',
        label = 'Repeat scatter; bootstrap 95% interval')
    axis.vlines(SAMPLE_SIZES, bounds[0], bounds[1])
    axis.hlines(bounds[0], SAMPLE_SIZES * 0.95, SAMPLE_SIZES * 1.05)
    axis.hlines(bounds[1], SAMPLE_SIZES * 0.95, SAMPLE_SIZES * 1.05)
    axis.plot(SAMPLE_SIZES, prediction, '--', label = 'Moment prediction')
    axis.set_xscale('log')
    axis.set_yscale('log')
    axis.set_xlabel('Particles per independent batch')
    axis.set_ylabel(name + ' precision [keV]')
    axis.grid()
    axis.legend(fontsize = 'small')

fig.suptitle(f'{N_REPEATS} independent batches at each sample size')
fig.tight_layout()
plt.show()
