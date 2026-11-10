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

########################################
# Fixed-reference ring and static RF settings
########################################
# Lattice path is relative to the repository root.
LATTICE_FILE        = 'test_data/example_lattices/demo_ring/lattice.py'
VOLTAGE             = 1e5                   # [V]; stable reference phase pi
FREQUENCY_SHIFTS    = np.array([-40.0, 0.0, 40.0])  # [Hz]

########################################
# Native synchronous-period search
########################################
SEARCH_TURNS        = 4

################################################################################
# Anchor the RF phase to the cavity's actual laboratory position
################################################################################

########################################
# Load a fresh fixed 1 GeV/c electron reference
########################################
repo_root   = Path(__file__).resolve().parents[3]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env['demo']
beta0       = float(line.particle_ref.beta0[0])
period0     = line.get_length() / (beta0 * c)
frequency0  = float(line['rf.1'].frequency)
harmonic    = round(frequency0 * period0)
cavity_s    = float(line.get_table()['s', 'rf.1'])
arrival0    = cavity_s / (beta0 * c)

env['voltage_rf'] = VOLTAGE
# The cavity is at s=250 m, not at the 1500 m line exit. Radiation is off.

########################################
# Preserve phase pi at reference arrival for every static frequency setting
########################################
env.set(
    'rf.1',
    absolute_time   = True,
    phase   = f'{np.pi}-2*{np.pi}*frequency_rf*{arrival0}',
)
nominal = line.twiss(
    method  = '6d',
    chrom   = False,
    search_for_t_rev    = True,
    num_turns_search_t_rev  = SEARCH_TURNS,
)

################################################################################
# Solve the actual synchronous period for each static frequency offset
################################################################################
rows    = []
for shift in FREQUENCY_SHIFTS:
    env['frequency_rf'] = frequency0 + shift
    tw  = line.twiss(
        method  = '6d',
        chrom   = False,
        search_for_t_rev    = True,
        num_turns_search_t_rev  = SEARCH_TURNS,
    )
    linear_delta    = -shift / (frequency0 * nominal.slip_factor)
    # Positive eta gives lower synchronous momentum when RF frequency rises.
    particle    = tw.particle_on_co.copy()
    initial = particle.copy()
    line.track(particle, num_turns = 1)
    slip    = float(particle.zeta[0] - initial.zeta[0])
    rows.append((shift, tw.delta[0], linear_delta, tw.t_rev,
                 line['rf.1'].frequency * tw.t_rev, slip))
    print('df [Hz], transverse/delta return residuals:', shift,
          [float(getattr(particle, name)[0] - getattr(initial, name)[0])
           for name in ['x', 'px', 'y', 'py', 'delta']])

################################################################################
# Compare the native solution with the local slip-factor prediction
################################################################################
columns = ['df_hz', 'delta', 'linear_delta', 'period_s', 'rf_cycles', 'dzeta_m']
xt.Table(
    data    = {name: np.array(values)
               for name, values in zip(columns, zip(*rows))},
    index   = 'df_hz',
).show(maxwidth = 'full')
print('Harmonic / fixed reference p0*c [eV]:',
      harmonic, line.particle_ref.p0c[0])
# f_rf*T_synchronous=h. The transverse orbit and momentum return, while zeta
# slips against the fixed reference clock by beta0*c*(T0-T_synchronous).
# These offsets are a small tested sample, not a momentum-acceptance scan.

################################################################################
# Plot momentum steering without changing the reference momentum
################################################################################
values  = np.asarray(rows)
fig, ax = plt.subplots(figsize = (8, 4))
ax.plot(
    values[:, 0], 1e4 * values[:, 1], 'o-',
    label   = 'Native synchronous orbit',
)
ax.plot(
    values[:, 0], 1e4 * values[:, 2], '--',
    label   = 'Linear slip prediction',
)
ax.set_xlabel('RF frequency offset [Hz]')
ax.set_ylabel('Synchronous delta [1e-4]')
ax.grid()
ax.legend()
fig.tight_layout()
plt.show()
