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
# Compensated electron ring
########################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
P0C             = 10e9                  # Reference p0*c [eV]
RF_VOLTAGE      = 50e6                  # [V]
HARMONIC        = 200

########################################
# Controlled solenoid inside an existing drift
########################################
SOLENOID_S      = 2.0                   # [m]
SOLENOID_LENGTH = 0.2                   # [m], replaces this much drift
SOLENOID_KS     = 0.05                  # Bz/(B*rho) [m^-1]

################################################################################
# Prepare the native mean-radiation and spin operating point
################################################################################

########################################
# Load the shared ring and add a solenoid in a drift
########################################
repo_root           = Path(__file__).resolve().parents[2]
env                 = xt.load(repo_root / LATTICE_FILE)
line                = env['demo']
line.set_particle_ref('electron', p0c = P0C)
env['frequency_rf']  = 0.0
env.set('rf.1', harmonic = HARMONIC)
env['voltage_rf']    = RF_VOLTAGE
line.insert(
    'spin.solenoid',
    xt.UniformSolenoid(
        length  = SOLENOID_LENGTH,
        ks      = 0.0,
    ),
    at      = SOLENOID_S,
    anchor  = 'start',
)

########################################
# Enable native spin transport and compensate mean radiation
########################################
line.configure_spin('auto')
line.configure_radiation(model = 'mean')
line.compensate_radiation_energy_loss()
table       = line.get_table()
bend_name   = table.rows[table.element_type == 'Bend'].name[0]

# As in the radiation examples, compensation sets RF phase and magnet taper.
# Fixed normalized strengths scale the physical fields with reference rigidity.

################################################################################
# Compare the planar limit with a spin-orbit-sensitive solenoid setting
################################################################################
# The invariant axis n0 has an arbitrary sign. The physical polarization is
# P = spin_polarization_eq * n0: reversing n0 reverses the scalar coefficient,
# leaving P unchanged. spin_n0_ib projects n0 on the physical magnetic field.
results = []
for strength in [0.0, SOLENOID_KS]:
    ########################################
    # Find the invariant spin field and its orbital dependence
    ########################################
    line['spin.solenoid'].ks = strength
    tw          = line.twiss(
        method                  = '6d',
        chrom                   = False,
        polarization_analysis   = True,
    )
    derivative  = np.sqrt(
        tw.spin_dn_ddelta_x**2 + tw.spin_dn_ddelta_y**2
        + tw.spin_dn_ddelta_z**2)

    ########################################
    # Resolve the signed physical polarization at the first radiating bend
    ########################################
    bend        = tw.rows[bend_name]
    projection  = tw.spin_polarization_eq * bend.spin_n0_ib[0]
    no_depol    = tw.spin_polarization_inf_no_depol * bend.spin_n0_ib[0]
    n0          = np.array([tw.spin_x[0], tw.spin_y[0], tw.spin_z[0]])
    print('Solenoid ks [m^-1]:', strength)
    print('Entrance n0 / physical equilibrium P:',
          n0, tw.spin_polarization_eq * n0)
    print('P dot B_hat at', bend_name, ':', projection)
    print('Polarization / depolarization component times [s]:',
          tw.spin_t_pol_component_s, tw.spin_t_depol_component_s)
    results.append((strength, tw, derivative, projection, no_depol))

################################################################################
# Compare signed equilibrium polarization and spin-orbit sensitivity
################################################################################

########################################
# Recover electron polarization antiparallel to B in the planar limit
########################################
# The planar Sokolov-Ternov limit is negative along physical B for electrons.
# The solenoid tilts n0 and makes it energy-dependent; the native linearized
# calculation includes the resulting radiative depolarization contribution.
limit       = -8 / (5 * np.sqrt(3))
strengths   = np.array([row[0] for row in results])
fig, axes   = plt.subplots(2, 1, figsize = (9, 7))
axes[0].plot(strengths, [row[3] for row in results], 'o',
             label = 'Equilibrium')
axes[0].plot(strengths, [row[4] for row in results], 's',
             label = 'Closed-orbit polarization only')
axes[0].axhline(limit, color = 'k', linestyle = ':',
                label = 'Planar electron Sokolov-Ternov limit')
axes[0].set_xlabel('Solenoid ks [m^-1]')
axes[0].set_xticks(strengths)
axes[0].set_ylabel('P dot B_hat at first radiating bend')

########################################
# Locate energy sensitivity of the invariant spin field around the ring
########################################
for strength, tw, derivative, projection, no_depol in results:
    axes[1].plot(tw.s, derivative, label = f'ks={strength:g} m^-1')
axes[1].set_xlabel('s [m]')
axes[1].set_ylabel('|partial n / partial delta|')
for axis in axes:
    axis.grid()
    axis.legend()
fig.tight_layout()
plt.show()
