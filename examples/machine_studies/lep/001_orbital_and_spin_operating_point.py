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
# Stored LEP operating point and detector insertions
########################################
LATTICE_FILE    = 'test_data/spin_refs_bmad/lep_lattice_to_bmad.json'
IPS             = [2, 4, 6, 8]
SETTINGS        = [('Planar ring', 0.0), ('Detector solenoids', 1.0)]

################################################################################
# Establish the positron operating point
################################################################################

########################################
# Retain the model's reference and fixed-energy comparison conventions
########################################
repo_root   = Path(__file__).resolve().parents[3]
line        = xt.load(repo_root / LATTICE_FILE)
reference   = line.particle_ref
line.configure_spin('auto')
# This 45.585 GeV/c positron model has simplified bend fields/edges for Bmad
# comparisons. It is an illustrative LEP setting, not operational optics.
# 4D optics excludes synchrotron motion and radiation energy-loss transport.
print('Reference p0c [GeV/c]:', reference.p0c[0] * 1e-9)
print('Charge [e], mass [eV/c^2]:', reference.q0, reference.mass0)
print('Anomalous magnetic moment:', reference.anomalous_magnetic_moment[0])

########################################
# Disable spin bumps and their coupling corrections in all insertions
########################################
for ip in IPS:
    line[f'on_spin_bump.{ip}']      = 0.0
    line[f'on_coupl_sol.{ip}']      = 0.0
    line[f'on_coupl_sol_bump.{ip}'] = 0.0

################################################################################
# Compare the planar and detector-solenoid invariant spin fields
################################################################################

########################################
# Solve orbital optics and the closed-orbit spin at each setting
########################################
results = []
for label, solenoid_scale in SETTINGS:
    for ip in IPS:
        line[f'on_sol.{ip}'] = solenoid_scale
    tw      = line.twiss4d(polarization_analysis = True)
    tilt    = np.arctan2(np.hypot(tw.spin_x, tw.spin_z), np.abs(tw.spin_y))
    results.append((label, tw, tilt))
    print('\n' + label)
    print('Orbital tunes Qx,Qy:', tw.qx, tw.qy)
    print('Coupling c_minus:', tw.c_minus)
    print('Fractional spin tune:', tw.spin_tune_fractional)
    print('Maximum invariant-axis tilt from vertical [mrad]:',
          1e3 * np.max(tilt))
    print('Maximum vertical orbit [mm]:', 1e3 * np.max(np.abs(tw.y)))

########################################
# Inspect orbital and spin quantities at the experimental points
########################################
for label, tw, tilt in results:
    print('\n' + label + ': experimental points')
    tw.rows[['ip2', 'ip4', 'ip6', 'ip8']].cols[
        'name s betx bety bety1 spin_x spin_y spin_z'].show(maxwidth = 'full')
# A solenoid couples the orbital modes and rotates spin even when the on-axis
# vertical closed orbit stays zero. n0 and -n0 describe the same invariant axis;
# the plotted acute tilt is independent of that sign choice.

################################################################################
# Relate local detector fields to ring-wide spin and orbital coupling
################################################################################

########################################
# Plot the invariant-axis tilt and horizontal-mode leakage into y
########################################
fig, axes = plt.subplots(2, 1, figsize = (10, 7), sharex = True,
                         layout = 'constrained')
for label, tw, tilt in results:
    axes[0].plot(tw.s * 1e-3, tilt * 1e3, label = label)
    axes[1].plot(tw.s * 1e-3, tw.bety1, label = label)
axes[0].set_ylabel('Invariant-axis tilt from vertical [mrad]')
axes[1].set(xlabel = 's [km]', ylabel = r'$\beta_{y,1}$ [m]')
for ax in axes:
    ax.legend()
    ax.grid(alpha = 0.25)
plt.show()
