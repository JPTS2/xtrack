# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c
import xpart as xp
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Ideal proton ring and two RF systems
########################################
P0C                 = 26e9       # Proton reference p0*c [eV]
CIRCUMFERENCE       = 1000.0     # [m]
MOMENTUM_COMPACTION = 0.02
HARMONIC            = 100
VOLTAGE             = 1e4        # [V] per RF system
FREQUENCY_OFFSET    = 500.0     # [Hz] on either side of h*f_rev0

########################################
# Two isolated-system matched bunch trains
########################################
N_PER_BUNCH = 128
N_BUNCHES   = 3
BUCKET_GAP  = 2                     # Bunch spacing in RF buckets
SIGMA_Z     = 0.15                  # RMS bunch length [m]
SEED        = 857
N_TURNS     = 512

################################################################################
# Construct a ring that isolates longitudinal RF slipping
################################################################################

########################################
# Derive the reference clock and RF frequencies
########################################
reference   = xt.Particles(pdg_id_0 = 'proton', p0c = P0C)
beta0       = float(reference.beta0[0])
t_rev0      = CIRCUMFERENCE / (beta0 * c)
f_rf0       = HARMONIC / t_rev0

# This assigned map has first-order momentum slip and linear transverse motion.
# It is not an SPS lattice or a transverse momentum-acceptance model.

########################################
# Put both absolute-time cavities at the common entrance
########################################
line                = xt.Line(
    elements        = [
        xt.Cavity(
            voltage         = VOLTAGE,
            frequency       = f_rf0 + FREQUENCY_OFFSET,
            phase           = np.pi,
            absolute_time   = True,
        ),
        xt.Cavity(
            voltage         = 0,
            frequency       = f_rf0 - FREQUENCY_OFFSET,
            phase           = np.pi,
            absolute_time   = True,
        ),
        xt.LineSegmentMap(
            length                      = CIRCUMFERENCE,
            qx                          = 0.31,
            qy                          = 0.32,
            betx                        = 10,
            bety                        = 12,
            longitudinal_mode           = 'nonlinear',
            momentum_compaction_factor  = MOMENTUM_COMPACTION,
            voltage_rf                  = 0,
            frequency_rf                = 0,
        ),
    ],
    element_names   = ['rf_high', 'rf_low', 'arc'],
)
line.particle_ref   = reference

# Both phases are pi at laboratory t=0. At s=0, a particle samples the field at
# time_s-zeta/(beta0*c). The RF-off map supplies slip; the cavities supply RF.

################################################################################
# Find the synchronous center of each RF system separately
################################################################################

########################################
# Search the native revolution period with the other RF switched off
########################################
synchronous = {}
for name in ['rf_high', 'rf_low']:
    line['rf_high'].voltage = VOLTAGE if name == 'rf_high' else 0
    line['rf_low'].voltage  = VOLTAGE if name == 'rf_low' else 0
    synchronous[name]       = line.twiss(
        method                  = '6d',
        search_for_t_rev        = True,
        num_turns_search_t_rev  = 4,
        chrom                   = False,
    )

########################################
# Display the two energy centers and their synchronism
########################################
names   = list(synchronous)
data    = {
    'rf_system':            np.array(names),
    'frequency_Hz':         np.array([line[name].frequency for name in names]),
    'delta':                np.array([synchronous[name].delta[0]
                                      for name in names]),
    'energy_offset_MeV':    np.array([
        synchronous[name].particle_on_co.ptau[0] * P0C / 1e6
        for name in names]),
    'period_us':            np.array([1e6 * synchronous[name].t_rev
                                      for name in names]),
    'rf_cycles_per_turn':   np.array([
        line[name].frequency * synchronous[name].t_rev for name in names]),
    'qs':                   np.array([synchronous[name].qs for name in names]),
}
centers = xt.Table(data = data, index = 'rf_system')
centers.show(maxwidth = 'full')
print('Each isolated RF makes', HARMONIC, 'cycles per synchronous turn.')

################################################################################
# Follow both populations with both RF systems active
################################################################################

########################################
# Match each train with only its own RF system powered
########################################
np.random.seed(SEED)
line.twiss_default['search_for_t_rev'] = True
line.twiss_default['num_turns_search_t_rev'] = 4
filling = np.zeros((N_BUNCHES - 1) * BUCKET_GAP + 1, dtype = int)
filling[::BUCKET_GAP] = 1
populations = []
for name in names:
    line['rf_high'].voltage = VOLTAGE if name == 'rf_high' else 0
    line['rf_low'].voltage  = VOLTAGE if name == 'rf_low' else 0
    bucket_length           = beta0 * c / line[name].frequency
    train                   = xp.generate_matched_gaussian_multibunch_beam(
        line                        = line,
        filling_scheme              = filling,
        bunch_num_particles         = N_PER_BUNCH,
        bunch_intensity_particles   = 1e10,
        bucket_length               = bucket_length,
        engine                      = 'linear',
        nemitt_x                    = 0,
        nemitt_y                    = 0,
        sigma_z                     = SIGMA_Z,
    )
    populations.append(train)
particles       = xt.Particles.merge(populations)
population_ids  = particles.particle_id.copy().reshape(
    2, N_BUNCHES * N_PER_BUNCH)

########################################
# Record the entrance coordinates once per reference turn
########################################
line['rf_high'].voltage = VOLTAGE
line['rf_low'].voltage  = VOLTAGE
line.track(particles, num_turns = N_TURNS, turn_by_turn_monitor = True)
record  = line.record_last_track
print('Final states / counts:',
      np.unique(particles.state, return_counts = True))

# Single-system centers are references, not joint fixed points of the two RFs.
# Each train is matched only to its isolated RF; both active systems create
# a time-dependent potential. No final recapture or stacking yield is modelled.

################################################################################
# Compare population motion with the two RF clocks
################################################################################

########################################
# Extract centroids by particle ID and the single-system slip references
########################################
turns           = record.at_turn[0]
clock_s         = turns * t_rev0
mean_zeta       = []
mean_delta      = []
slip_reference  = []
for ids, tw in zip(population_ids, synchronous.values()):
    rows    = np.isin(record.particle_id[:, 0], ids)
    mean_zeta.append(np.mean(record.zeta[rows], axis = 0))
    mean_delta.append(np.mean(record.delta[rows], axis = 0))
    slip_reference.append(
        mean_zeta[-1][0] + beta0 * c * (t_rev0 - tw.t_rev) * turns)

frequency_difference    = line['rf_high'].frequency - line['rf_low'].frequency
rf_beat_cycles          = frequency_difference * clock_s
rf_spacing              = CIRCUMFERENCE / HARMONIC
relative_slip           = ((mean_zeta[0] - mean_zeta[1])
                   - (mean_zeta[0][0] - mean_zeta[1][0])) / rf_spacing
print('Final common-clock RF phase difference [cycles]:', rf_beat_cycles[-1])
print('Final centroid separation change [nominal RF periods]:',
      relative_slip[-1])

########################################
# Show energy separation, longitudinal slip and the relative RF phase
########################################
fig, axes = plt.subplots(3, 1, figsize = (9, 9), sharex = True)
for index, name in enumerate(names):
    axes[0].plot(1e3 * clock_s, mean_delta[index], label = name)
    axes[0].axhline(synchronous[name].delta[0], color = f'C{index}', ls = ':')
    axes[1].plot(1e3 * clock_s, mean_zeta[index], label = name)
    axes[1].plot(1e3 * clock_s, slip_reference[index],
                 color  = f'C{index}',
                 ls     = '--',
                 label  = f'{name}: isolated-RF period reference')
axes[0].set_ylabel('Mean delta')
axes[0].legend()
axes[1].set_ylabel('Mean zeta [m]')
axes[1].legend(fontsize = 'small')
axes[2].plot(1e3 * clock_s, relative_slip, label = 'Centroid separation change')
axes[2].plot(1e3 * clock_s, rf_beat_cycles, ls = '--',
             label  = 'RF phase difference at reference clock')
axes[2].set_ylabel('Nominal RF periods / cycles')
axes[2].set_xlabel('Reference clock time [ms]')
axes[2].legend()
fig.suptitle('Two RF systems: slipping energy-separated proton trains')
fig.tight_layout()
fig, axes = plt.subplots(1, 2, figsize = (11, 4), layout = 'constrained')
for ids, name in zip(population_ids, names):
    rows = np.isin(record.particle_id[:, 0], ids)
    for axis, turn in zip(axes, [0, N_TURNS - 1]):
        axis.scatter(record.zeta[rows, turn], record.delta[rows, turn],
                     s = 3, alpha = 0.4, label = name)
        axis.set(title = f'Entrance record, turn {turn}',
                 xlabel = 'zeta [m]', ylabel = 'delta')
for axis in axes:
    axis.grid()
    axis.legend()
plt.show()
