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
# Two independent rings with the demonstration optics
########################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
P0C             = 7e12                   # Proton reference momentum [eV/c]
IP              = 'ip.1'
N_SLOTS         = 8
FILLED_CW       = [0, 1, 2]
FILLED_ACW      = [0, 2]                 # Remove the partner of CW slot 1

########################################
# Prescribed Gaussian bunch sizes and populations
########################################
INTENSITY       = 1e10                   # Protons per occupied slot
NEMITT_X        = 2e-6                   # Normalized rms emittance [m]
NEMITT_Y        = 2e-6                   # [m]
MAX_ITERATIONS  = 5
COLLIDING_SLOT  = 0
UNPAIRED_SLOT   = 1

################################################################################
# Make clockwise and anticlockwise optical copies
################################################################################

########################################
# Reuse the magnetic lattice with a proton reference
########################################
repo_root   = Path(__file__).resolve().parents[3]
clockwise   = xt.load(repo_root / LATTICE_FILE)['demo']
clockwise.set_particle_ref('proton', p0c = P0C)
# RF stays off. The stored normalized magnet strengths retain the transverse
# optics at this new rigidity; physical fields are not held at electron values.

########################################
# Mirror an independent ring and keep the IP inside both lines
########################################
anticlockwise = clockwise.copy()
anticlockwise.mirror()
# There are no skew fields or powered correctors. These are independent optical
# rings, not two proton beams sharing the same physical magnet fields.
env = xt.Environment(lines = {'cw': clockwise, 'acw': anticlockwise})
for name in ['cw', 'acw']:
    env[name].twiss_default['method'] = '4d'

################################################################################
# Install one head-on encounter with slot-by-slot bunch pairing
################################################################################

########################################
# Let Xfields configure the collision geometry and bunch representatives
########################################
# There are eight equally spaced timing slots and one IP with zero pairing
# offset: CW slot j meets ACW slot j. No RF confinement is modelled. There are
# no long-range encounters, crossing angle, radiation or coherent-mode analysis.
env.xfields.install_beambeam_interactions(
    clockwise_line                      = 'cw',
    anticlockwise_line                   = 'acw',
    ip_names                            = [IP],
    num_long_range_encounters_per_side   = 0,
    harmonic_number                     = N_SLOTS,
    bunch_spacing_buckets               = 1,
    mode                                = 'rigid_bunch')
study = env.xfields.configure_beambeam_interactions(
    num_particles   = INTENSITY,
    nemitt_x        = NEMITT_X,
    nemitt_y        = NEMITT_Y,
    filled_slots_cw = FILLED_CW,
    filled_slots_acw = FILLED_CW)

########################################
# Calculate a no-beam-beam reference on the same installed lattice
########################################
env['beambeam_scale'] = 0.0
baseline = env['cw'].twiss4d()
env['beambeam_scale'] = 1.0
print('Prepare the collision schedule\n'
      '------------------------------')
print('Slot spacing [m]:', env['cw'].get_length() / N_SLOTS)
print('Clockwise filled slots:', study.filled_slots_cw)
print('Initially every clockwise bunch has a head-on partner.')
print('No-BB tunes:', baseline.qx, baseline.qy)

################################################################################
# Compare complete pairing with one missing opposing bunch
################################################################################

########################################
# Solve the two-beam closed orbit with every partner present
########################################
# solve iterates opposing-beam centroids to self-consistency. Here symmetry
# makes the orbit unchanged. dynamic_beta=False keeps source sizes fixed at
# their no-BB values; the test-particle optics still respond to the BB lens.
paired = study.solve(
    max_iterations  = MAX_ITERATIONS,
    dynamic_beta    = False,
    twiss_mode      = 'fast',
    show_progress   = False)

########################################
# Change the filling without reinstalling elements and solve again
########################################
study.apply_filling_pattern(
    filled_slots_cw = FILLED_CW,
    filled_slots_acw = FILLED_ACW)
gapped = study.solve(
    max_iterations  = MAX_ITERATIONS,
    dynamic_beta    = False,
    twiss_mode      = 'fast',
    show_progress   = False)
print('\nRemove one opposing bunch\n'
      '-------------------------')
print('Anticlockwise filled slots:', study.filled_slots_acw)
print('Paired / gapped solutions converged:',
      paired.converged, gapped.converged)
gapped.cw.cols['slot zeta qx qy'].show(digits = 10, maxwidth = 'full')
print('Slot 1 has no head-on partner; slots 0 and 2 retain the BB lens.')

################################################################################
# Select one bunch and compare all bunches at the IP
################################################################################

########################################
# Access ordinary lattice-indexed Twiss tables by physical slot
########################################
colliding   = gapped.cw.bunch(slot = COLLIDING_SLOT)
unpaired    = gapped.cw.bunch(slot = UNPAIRED_SLOT)
at_ip       = gapped.cw.at_element(IP)
print('\nCompare bunch optics at the IP\n'
      '------------------------------')
at_ip.cols['slot x y betx bety'].show(maxwidth = 'full')
print('No-BB IP beta functions [m]:', baseline['betx', IP],
      baseline['bety', IP])
print('Unpaired-minus-no-BB tunes:', unpaired.qx - baseline.qx,
      unpaired.qy - baseline.qy)
print('Symmetric head-on beams have zero orbit shift here.')

########################################
# Recompute optics with the already loaded opposing-beam state
########################################
# twiss does not iterate the two beams. After solve it reproduces the optics
# of that solution; it is not a request to solve a new filling pattern.
loaded = study.twiss(mode = 'fast', show_progress = False)
print('Maximum loaded-state/solution tune difference:',
      np.max(np.abs(loaded.cw.qx - gapped.cw.qx)),
      np.max(np.abs(loaded.cw.qy - gapped.cw.qy)))

################################################################################
# Plot tune shifts by slot and the resulting vertical beta beating
################################################################################

########################################
# Show which tune shifts disappear when the opposing slot is empty
########################################
fig, axes = plt.subplots(2, 1, sharex = True, figsize = (8, 6))
for axis, quantity, q0 in zip(axes, ['qx', 'qy'],
                             [baseline.qx, baseline.qy]):
    axis.plot(paired.cw.slot, 1e3 * (paired.cw[quantity] - q0), 's--',
              label = 'All partners present')
    axis.plot(gapped.cw.slot, 1e3 * (gapped.cw[quantity] - q0), 'o-',
              label = 'Opposing slot 1 empty')
    axis.axhline(0.0, color = 'gray', linewidth = 0.8)
    axis.set_ylabel(f'Delta {quantity} [1e-3]')
    axis.grid()
axes[0].legend()
axes[1].set_xticks(FILLED_CW)
axes[1].set_xlabel('Clockwise physical slot')
fig.tight_layout()

########################################
# Compare the full lattice functions rather than only the tunes
########################################
fig, axis = plt.subplots(figsize = (8, 4))
for tw, slot in [(colliding, COLLIDING_SLOT), (unpaired, UNPAIRED_SLOT)]:
    axis.plot(tw.s, 100 * (tw.bety / baseline.bety - 1),
              label = f'CW slot {slot}')
axis.set_xlabel('s [m]')
axis.set_ylabel('Vertical beta beating [%]')
axis.legend()
axis.grid()
fig.tight_layout()
plt.show()
