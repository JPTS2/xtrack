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
from scipy.constants import c
import xtrack as xt

################################################################################
# User parameters
################################################################################
MODEL_DIR   = 'test_data/example_lattices/demo_ring'  # Relative to repo root

################################################################################
# Load geometry and an independently maintained operating point
################################################################################

########################################
# Define hardware, then apply nominal quadrupole strengths
########################################
repo_root   = Path(__file__).resolve().parents[2]
model_dir   = repo_root / MODEL_DIR
print('Load hardware definitions and an operating point'
      '\n------------------------------------------------')
print('geometry.py defines elements and reusable sections.')
print('strengths.py supplies the nominal strengths, reference momentum and RF.')
env = xt.Environment()
env.call(model_dir / 'geometry.py')
env.call(model_dir / 'strengths.py')

########################################
# Inspect reusable cells, half-straights and named full sections
########################################
print('\nReusable arc cell; positions [m]:')
env.lines['arc.cell'].get_table().cols['name s element_type'].show(
    maxwidth = 'full')
print('\nReusable section       Length [m]')
for name in ['arc.cell', 'arc.s1', 'ff.half', 'utility.half', 'insertion']:
    print(f'{name:<20} {env.lines[name].get_length():12.3f}')

################################################################################
# Assemble the ring and set its reference clock
################################################################################

########################################
# Order the six arcs and their intervening straight sections
########################################
print('\nAssemble six arcs with six intervening straights'
      '\n------------------------------------------------')
print('The insertion is the low-beta straight; '
      'the other five are utility sections.')
demo    = env.new_line(name = 'demo', components = [
    'arc.s1', 'insertion',
    'arc.s2', 'utility.s2',
    'arc.s3', 'utility.s3',
    'arc.s4', 'utility.s4',
    'arc.s5', 'utility.s5',
    'arc.s6', 'utility.s6',
])

########################################
# Set the reference species, RF clock and ideal steering selections
########################################
demo.set_particle_ref('electron', p0c = env['p0c'])
env['frequency_rf'] = (
    env['harmonic_rf'] * demo.particle_ref.beta0[0] * c / demo.get_length())
table       = demo.get_table()
bpms        = list(table.rows['bpm[.].*'].name)
correctors  = list(table.rows['cor[.].*'].name)
demo.steering_monitors_x = bpms
demo.steering_monitors_y = bpms
demo.steering_correctors_x = correctors
demo.steering_correctors_y = correctors
# The nominal setting has zero RF voltage and no radiation.
print(f'Reference momentum [GeV/c]: {demo.particle_ref.p0c[0] * 1e-9:.3f}')
print(f'RF clock [MHz]: {env["frequency_rf"] * 1e-6:.6f}')
print(f'RF voltage [V]: {env["voltage_rf"]:.1f}')
print(f'Ideal BPM locations: {len(bpms)}; '
      f'corrector locations: {len(correctors)}')

################################################################################
# Inspect the reference geometry and the selected optics
################################################################################

########################################
# Check reference-path closure independently of particle tracking
########################################
survey          = demo.survey()
position_error  = np.linalg.norm(survey.XYZ[-1] - survey.XYZ[0])
frame_error     = np.max(np.abs(survey.E_matrix[-1] - survey.E_matrix[0]))
print('\nCheck reference-geometry closure\n--------------------------------')
print('Survey follows the design geometry, '
      'independently of an orbit calculation.')
print(f'Circumference [m]: {demo.get_length():.3f}')
print(f'Position closure residual [m]: {position_error:.3e}')
print(f'Frame closure residual [1]: {frame_error:.3e}')

########################################
# Inspect integrated strengths and the low-beta interaction point
########################################
print('\nInspect selected magnets and periodic optics'
      '\n--------------------------------------------')
print('Units: s [m]; k1l [m^-1].')
demo.get_table(attr = True).rows[
    ['arc.s1.c1.qf.in', 'arc.s1.c1.qd', 'straight.s1.r.q0']
].cols['name s element_type k1l'].show(maxwidth = 'full')
tw  = demo.twiss4d()
print(f'Horizontal tune: {tw.qx:.6f}')
print(f'Vertical tune: {tw.qy:.6f}')
print('\nIP optics: s, betx, bety, dx [m]; alfx, alfy, dpx dimensionless.')
tw.rows[['ip.1']].cols['name s betx bety alfx alfy dx dpx'].show(
    maxwidth    = 'full')

# The ring-wide plot resolves the insertion peaks, not the centimetre IP waist.

################################################################################
# Plot the geometry and magnetic optics
################################################################################

########################################
# Connect surveyed element boundaries, drawing bends as chords
########################################
fig, axes = plt.subplots(1, 2, figsize = (11, 5))
axes[0].plot(survey.Z, survey.X, '.-', markersize = 2)
axes[0].set_xlabel('Z [m]')
axes[0].set_ylabel('X [m]')
axes[0].set_aspect('equal', adjustable = 'datalim')
axes[0].set_title('Reference geometry')
axes[0].plot(survey['Z', 'ip.1'], survey['X', 'ip.1'], 'o', color = 'tab:red')
axes[0].annotate(
    'IP1', xy = (survey['Z', 'ip.1'], survey['X', 'ip.1']),
    xytext = (8, 8), textcoords = 'offset points', color = 'tab:red',
)

########################################
# Show the insertion beta peak together with arc optics
########################################
tw.plot(
    yl      = 'betx bety',
    yr      = '',
    lattice = False,
    ax      = axes[1],
)
axes[1].set_title('Periodic 4D optics')
axes[1].legend(loc = 'upper right')
print('\nThe floor plot joins element boundaries, drawing bends as chords.')
print('The full-ring optics plot shows insertion peaks; the IP table resolves')
print('the small beta functions at the waist.')
fig.tight_layout()
plt.show()
