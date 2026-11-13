# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Shared geometry and reference
########################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
P0C             = 10e9                  # Reference p0*c [eV]

########################################
# Available RF voltage, above the energy loss per turn
########################################
RF_VOLTAGE      = 50e6                  # [V]
HARMONIC        = 200
DECEL_VOLTAGE   = 2e6                   # Separate weak RF station [V]
DECEL_AT        = 'bpm.s4.f'

################################################################################
# Configure the original compensated operating point
################################################################################
repo_root   = Path(__file__).resolve().parents[2]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env['demo']
env['p0c']  = P0C
line.set_particle_ref('electron', p0c = P0C)
env['frequency_rf'] = 0.0
env.set('rf.1', harmonic = HARMONIC)
env['voltage_rf']   = RF_VOLTAGE
line.configure_radiation(model = 'mean')
line.compensate_radiation_energy_loss()
twiss = line.twiss6d(
    chrom = False, radiation_analysis = True, radiation_method = 'full')
################################################################################
# Add a weak decelerating RF station with the tapered magnets held fixed
################################################################################

########################################
# Find the new mean-radiation closed orbit without retapering the fields
########################################
decelerating_line = line.copy()
decelerating_line.insert(
    'rf.decelerating',
    xt.Cavity(
        voltage     = DECEL_VOLTAGE,
        harmonic    = HARMONIC,
        phase       = line['rf.1'].phase + line['rf.1'].phase_taper + np.pi,
    ),
    at = DECEL_AT,
)
decelerating_optics = decelerating_line.twiss6d(
    chrom               = False,
    radiation_analysis  = True,
    radiation_method    = 'full',
)
# The powered original station still supplies the total radiation loss plus
# the energy extracted here. The new orbit balances the modified RF system;
# no claim is made that the old taper remains optimal.

########################################
# Measure the negative kick and inspect the damping partitions
########################################
station_kicks = {}
for station in ['rf.1', 'rf.decelerating']:
    probe           = decelerating_optics.get_twiss_init(
        at_element = station).particle_on_co.copy()
    energy_before   = probe.energy[0]
    decelerating_line.track(probe, ele_start = station, num_elements = 1)
    station_kicks[station] = probe.energy[0] - energy_before
    print(station, 'actual energy change [eV]:', station_kicks[station])
print('Sum of RF gains minus mean radiation loss [eV]:',
      sum(station_kicks.values()) - decelerating_optics.energy_loss)
for label, optics in [('Original RF', twiss),
                       ('With decelerating station', decelerating_optics)]:
    print(label, 'mean energy loss [eV] / damping partitions / sum:',
          optics.energy_loss, optics.partition_numbers,
          np.sum(optics.partition_numbers))

################################################################################
# Compare the two RF gains and damping partitions
################################################################################
fig, axes = plt.subplots(1, 2, figsize = (10, 4), layout = 'constrained')
axes[0].bar(list(station_kicks), np.array(list(station_kicks.values())) * 1e-6)
axes[0].axhline(0, color = 'grey', linewidth = 0.8)
axes[0].set_ylabel('Actual RF energy change [MeV]')
for label, optics in [('Original RF', twiss),
                       ('With extraction', decelerating_optics)]:
    axes[1].plot(['x', 'y', 'longitudinal'], optics.partition_numbers,
                 'o-', label = label)
axes[1].set_ylabel('Damping partition')
axes[1].legend()
for axis in axes:
    axis.grid(axis = 'y')
plt.show()
