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
LATTICE_FILE    = 'test_data/spin_refs_bmad/lep_lattice_to_bmad.json'
BEND_NAME       = 'b2l.ql12.r1'         # Powered arc bend, local B along +y
IPS             = [2, 4, 6, 8]
SETTINGS        = [
    ('Solenoids',                0, 0),
    ('Add spin bumps',           1, 0),
    ('Add coupling correction',  1, 1),
]
# Each setting specifies (label, spin-bump scale, coupling-correction scale).

################################################################################
# Load the LEP comparison model and inspect its physical reference
################################################################################
repo_root   = Path(__file__).resolve().parents[3]
line        = xt.load(repo_root / LATTICE_FILE)
reference   = line.particle_ref
print('Stored reference q0 [e], mass0 [eV/c^2], p0c [GeV]:',
      reference.q0, reference.mass0, reference.p0c[0] * 1e-9)
print('Stored reference gamma0 / magnetic-moment anomaly:',
      reference.gamma0[0], reference.anomalous_magnetic_moment[0])
# The stored positive-charge electron-mass reference represents positrons.
# This regression model has simplified bend fields and edges for comparison
# with Bmad. It is an illustrative LEP model, not an operational machine setup.

################################################################################
# Apply the installed spin bumps and coupling-correction knobs in sequence
################################################################################
# Fixed-energy 4D optics leaves RF and mean radiation transport out of the
# orbital calculation. Native radiation integrals predict polarization and
# time constants; no polarization build-up is tracked in this example.
results = []
for label, spin_bump, coupling_correction in SETTINGS:
    ########################################
    # Set all four experimental insertions explicitly
    ########################################
    for ip in IPS:
        line[f'on_sol.{ip}']            = 1
        line[f'on_spin_bump.{ip}']      = spin_bump
        line[f'on_coupl_sol.{ip}']      = coupling_correction
        line[f'on_coupl_sol_bump.{ip}'] = coupling_correction
    # The bumps use vertical orbit correctors. The correction knobs use the
    # stored skew-quadrupole settings for solenoids, then for the added bumps.

    ########################################
    # Calculate the invariant spin field and native polarization predictions
    ########################################
    tw              = line.twiss4d(polarization_analysis = True)
    sensitivity     = np.sqrt(
        tw.spin_dn_ddelta_x**2 + tw.spin_dn_ddelta_y**2
        + tw.spin_dn_ddelta_z**2)
    n0              = np.array([tw.spin_x[0], tw.spin_y[0], tw.spin_z[0]])
    # n0 has an arbitrary sign; the physical vector P_eq*n0 does not.
    polarization    = tw.spin_polarization_eq * n0
    bend            = tw.rows[BEND_NAME]
    projection      = tw.spin_polarization_eq * bend.spin_n0_ib[0]
    results.append((label, tw, sensitivity, projection))

    ########################################
    # Inspect recovery of polarization and its rate components
    ########################################
    print('\n' + label)
    print('Entrance equilibrium polarization vector [x, y, z]:', polarization)
    print('P dot B_hat at', BEND_NAME, ':', projection)
    print('Polarization / depolarization component times [s]:',
          tw.spin_t_pol_component_s, tw.spin_t_depol_component_s)
    print('Predicted combined build-up time [s]:', tw.spin_t_pol_buildup_s)
    print('Spin-orbit depolarization integral [m^-2]:',
          tw.spin_int_kappa3_11_18_dn_ddelta_sq)

################################################################################
# Relate the orbit bumps and spin-field sensitivity to recovered polarization
################################################################################

########################################
# Compare the orbit and energy sensitivity around the ring
########################################
fig, axes = plt.subplots(3, 1, figsize = (10, 10))
for label, tw, sensitivity, projection in results:
    axes[0].plot(tw.s * 1e-3, tw.y * 1e3, label = label)
    axes[1].plot(tw.s * 1e-3, sensitivity, label = label)
axes[0].set_ylabel('Closed orbit y [mm]')
axes[1].set_ylabel('|partial n / partial delta|')
for axis in axes[:2]:
    axis.set_xlabel('s [km]')
    axis.grid()
    axis.legend()

########################################
# Compare gauge-independent polarization along the local bending field
########################################
# Positrons polarize parallel to B in the planar Sokolov-Ternov limit.
# The actual native prediction also includes spin-orbit depolarization.
labels      = [row[0] for row in results]
projections = [row[3] for row in results]
axes[2].bar(labels, projections)
axes[2].axhline(8 / (5 * np.sqrt(3)), color = 'k', linestyle = ':',
                label = 'Planar positron limit')
axes[2].set_ylabel('P dot B_hat at ' + BEND_NAME)
axes[2].set_ylim(-1, 1)
axes[2].grid(axis = 'y')
axes[2].legend()
fig.tight_layout()
plt.show()
