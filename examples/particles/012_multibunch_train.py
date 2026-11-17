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
# Ideal ring and RF bucket grid
########################################
P0C             = 3e9                    # Proton reference momentum [eV/c]
CIRCUMFERENCE   = 100.0                  # [m]
HARMONIC        = 20
VOLTAGE         = 1e5                    # [V]
ALPHA_C         = 0.2                    # Momentum compaction
BUCKET_GAP      = 2                      # RF buckets per filling slot
FILLING         = np.array([1, 1, 0, 1, 0, 0, 1, 0, 0, 0])

########################################
# Population and compact bunch selection
########################################
SEED            = 12345
N_PER_BUNCH     = 256
INTENSITY       = 1e10                   # Real protons per filled slot
NEMITT_X        = 1e-6                   # Normalized rms emittance [m]
NEMITT_Y        = 2e-6                   # [m]
SIGMA_ZETA      = 0.03                   # Within-bunch rms length [m]
BUNCH_SELECTION = [2, 0]                 # Indices in the occupied-slot list

################################################################################
# Define a periodic slot grid and a linear RF model
################################################################################

########################################
# Assign stable transverse and longitudinal optics
########################################
# As in006, this is an ideal optical map, with RF linearized about a stable
# fixed point. Generation is matched to linear, small-spread accuracy.
reference   = xt.Particles('proton', p0c = P0C)
frequency   = HARMONIC * reference.beta0[0] * c / CIRCUMFERENCE
line = xt.Line(elements = [xt.LineSegmentMap(
    length                      = CIRCUMFERENCE,
    qx                          = 0.31,
    qy                          = 0.32,
    betx                        = 10.0,
    bety                        = 12.0,
    longitudinal_mode           = 'linear_fixed_rf',
    voltage_rf                  = VOLTAGE,
    frequency_rf                = frequency,
    slippage_length             = CIRCUMFERENCE,
    momentum_compaction_factor  = ALPHA_C)])
line.particle_ref = reference
line.build_tracker()

########################################
# Relate filling slots to RF buckets and zeta
########################################
bucket_length   = CIRCUMFERENCE / HARMONIC
slot_spacing    = BUCKET_GAP * bucket_length
filled_slots    = np.flatnonzero(FILLING)
print('Relate filling slots to RF buckets and zeta\n'
      '------------------------------------------')
print('RF bucket length [m]:', bucket_length)
print('Filling-slot spacing [m]:', slot_spacing)
print('Occupied physical slots:', filled_slots)
print('A positive slot number has center zeta = -slot * slot_spacing.')

################################################################################
# Prepare the complete train with one native generation call
################################################################################
# Each occupied slot receives the same population and matched distribution.
# Gaps are empty slots, not zero-weight macroparticles. There are no collective
# fields, acceleration or radiation; no multi-turn confinement is tested here.
np.random.seed(SEED)
particles = xp.generate_matched_gaussian_multibunch_beam(
    line                        = line,
    filling_pattern             = FILLING,
    bunch_spacing_buckets       = BUCKET_GAP,
    bunch_num_particles         = N_PER_BUNCH,
    bunch_intensity_particles   = INTENSITY,
    nemitt_x                    = NEMITT_X,
    nemitt_y                    = NEMITT_Y,
    sigma_z                     = SIGMA_ZETA,
    engine                      = 'linear')

################################################################################
# Inspect each occupied slot rather than the train's overall rms length
################################################################################
zeta        = particles.zeta.reshape(len(filled_slots), N_PER_BUNCH)
weights     = particles.weight.reshape(len(filled_slots), N_PER_BUNCH)
centers     = -filled_slots * slot_spacing
print('\nInspect the generated train\n'
      '---------------------------')
xt.Table(
    data    = {
        'slot':             filled_slots,
        'rf_bucket':        filled_slots * BUCKET_GAP,
        'center_m':         centers,
        'mean_zeta_m':      np.mean(zeta, axis = 1),
        'rms_length_m':     np.std(zeta, axis = 1),
        'intensity':        np.sum(weights, axis = 1)},
    index   = 'slot').show(maxwidth = 'full')
print('Total represented intensity:', np.sum(particles.weight))
print('Within-bunch rms lengths fluctuate about', SIGMA_ZETA, 'm.')

################################################################################
# Generate only selected bunches using their compact indices
################################################################################
# This creates a fresh sample, not a filtered copy of particles. Compact index 2
# means the third occupied slot (physical slot 3), not physical slot 2.
selected = xp.generate_matched_gaussian_multibunch_beam(
    line                        = line,
    filling_pattern             = FILLING,
    bunch_selection             = BUNCH_SELECTION,
    bunch_spacing_buckets       = BUCKET_GAP,
    bunch_num_particles         = N_PER_BUNCH,
    bunch_intensity_particles   = INTENSITY,
    nemitt_x                    = NEMITT_X,
    nemitt_y                    = NEMITT_Y,
    sigma_z                     = SIGMA_ZETA,
    engine                      = 'linear')
selected_slots = filled_slots[BUNCH_SELECTION]
print('\nDistinguish compact bunch indices from physical slots\n'
      '-----------------------------------------------------')
print('Requested compact indices:', BUNCH_SELECTION)
print('Corresponding physical slots, in generated order:', selected_slots)
print('Generated mean zeta [m]:',
      selected.zeta.reshape(len(selected_slots), N_PER_BUNCH).mean(axis = 1))

################################################################################
# Show the train gaps and the selected physical slots
################################################################################
fig, axes = plt.subplots(2, 1, sharex = True, figsize = (8, 6))
axes[0].scatter(-particles.zeta / slot_spacing, particles.x * 1e3,
                s = 5, alpha = 0.4)
axes[0].set_ylabel('x [mm]')
axes[0].set_title('Four bunches on a ten-slot grid')
axes[1].bar(np.arange(len(FILLING)), FILLING, label = 'Complete train')
axes[1].bar(selected_slots, np.ones(len(selected_slots)),
            width = 0.4, label = 'Selected slots')
axes[1].set_ylabel('Occupancy')
axes[1].set_xlabel('Physical slot (-zeta / slot spacing)')
axes[1].set_xticks(np.arange(len(FILLING)))
axes[1].legend()
for axis in axes:
    axis.grid()
fig.tight_layout()
plt.show()
