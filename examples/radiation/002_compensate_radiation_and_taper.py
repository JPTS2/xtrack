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
# Shared geometry at a local 10 GeV/c electron operating point
########################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
P0C             = 10e9                  # Reference p0*c [eV]

########################################
# Available RF voltage, above the energy loss per turn
########################################
RF_VOLTAGE  = 50e6                  # [V]
HARMONIC    = 200

################################################################################
# Compare uncompensated energy loss with native compensation and tapering
################################################################################

########################################
# Follow the radiation-free closed orbit for one radiating turn
########################################
repo_root   = Path(__file__).resolve().parents[2]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env['demo']
env['p0c']  = P0C
line.set_particle_ref('electron', p0c = env['p0c'])
env['frequency_rf'] = 0.0
env.set('rf.1', harmonic = HARMONIC)
env['voltage_rf']   = RF_VOLTAGE
reference           = line.twiss6d(chrom = False)
before              = reference.particle_on_co.copy()
line.configure_radiation(model = 'mean')
line.track(before, turn_by_turn_monitor = 'ONE_TURN_EBE')
uncompensated       = line.record_last_track
print('Uncompensated one-turn energy change [eV]:',
      before.energy[0] - reference.particle_on_co.energy[0])

# Fixed normalized strengths imply tenfold physical fields at tenfold p0c.
# This is an illustrative setting, not a hardware-qualified ring. Native
# harmonic RF follows the reference beta; shared model files remain unchanged.

# The initial RF phase pi supplies no synchronous energy gain. Mean radiation
# is deterministic; this is not photon sampling or a quantum equilibrium bunch.

########################################
# Let Xsuite set RF phase offsets and local magnet taper offsets
########################################
entrance_line = line.copy()
line.compensate_radiation_energy_loss()
twiss       = line.twiss6d(
    chrom               = False,
    radiation_analysis  = True,
    radiation_method    = 'full',
)
initial     = twiss.particle_on_co.copy()
returned    = initial.copy()
line.track(returned, turn_by_turn_monitor = 'ONE_TURN_EBE')
compensated     = line.record_last_track
print('Compensated energy loss per turn [eV]:', twiss.energy_loss)
print('RF nominal phase / phase_taper [rad]:',
      line['rf.1'].phase, line['rf.1'].phase_taper)
fields  = line.get_table(attr = True)
fields.rows['arc.s1.c1.*'].cols[
    's element_type delta_taper'].show(maxwidth = 'full')

# The default requests zero length-averaged delta during the taper solve.
# Local delta still varies; restored RF need not give an exactly zero mean.
# Native delta_taper follows the local energy sawtooth; phase_taper supplies
# the RF energy gain. Nominal circuit strengths and reference p0c stay fixed.

################################################################################
# Inspect periodic closure and the energy sawtooth around the ring
################################################################################

########################################
# Compare orbit displacement and all six returned coordinates
########################################
print('Maximum closed-orbit change x / y [m]:',
      np.max(np.abs(twiss.x - reference.x)),
      np.max(np.abs(twiss.y - reference.y)))
print('Closed-orbit delta range:', np.min(twiss.delta), np.max(twiss.delta))
print('Returned state:', returned.state[0])
for coordinate in ['x', 'px', 'y', 'py', 'zeta', 'delta']:
    print(coordinate, 'one-turn residual:',
          getattr(returned, coordinate)[0] - getattr(initial, coordinate)[0])

########################################
# Plot actual one-turn energy changes before and after compensation
########################################
fig, axis = plt.subplots(figsize = (8, 4))
for label, record in [('Uncompensated', uncompensated),
                      ('Compensated', compensated)]:
    axis.plot(
        record.s[0],
        1e-6 * line.particle_ref.p0c[0] * (record.ptau[0] - record.ptau[0, 0]),
        label   = label,
    )
axis.set_xlabel('s [m]')
axis.set_ylabel('Energy change from entrance [MeV]')
axis.grid()
axis.legend()
fig.tight_layout()

# Compensation closes this mean-radiation orbit; it does not remove the local
# energy sawtooth or establish a sampled equilibrium distribution.

################################################################################
# Choose an entrance or a mean momentum convention
################################################################################
########################################
# Contrast a fixed entrance momentum with the default mean convention
########################################
entrance_line.build_tracker()
entrance_line.compensate_radiation_energy_loss(delta0 = 0.0)
entrance_optics = entrance_line.twiss6d(
    chrom               = False,
    radiation_method    = 'full',
)
print('Taper convention / entrance delta / length-weighted delta estimate:')
for label, optics in [('Default zero mean', twiss),
                       ('Requested entrance delta=0', entrance_optics)]:
    mean_delta = np.sum(
        0.5 * (optics.delta[:-1] + optics.delta[1:]) * np.diff(optics.s)
    ) / optics.s[-1]
    print(label, optics.delta[0], mean_delta)
# The mean estimate uses endpoint trapezoids; thick-element evolution is not
# resolved continuously. The recomputed orbit can differ slightly from the
# requested convention; changing it changes the operating point.

plt.show()
