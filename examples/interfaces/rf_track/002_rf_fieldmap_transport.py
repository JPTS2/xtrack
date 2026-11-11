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
from scipy.constants import c as clight
import RF_Track as rft
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Field map
########################################
MAP_FILE        = ('examples/rf_track/'
                   '400_Ez_on-axis_TD26__vg1p8_r05_CC_EDMS.txt')
FREQUENCY       = 11.994e9              # [Hz]
PEAK_EZ         = 10e6                  # Chosen peak |complex Ez| [V/m]

########################################
# Electron probes
########################################
P0C             = 100e6                 # Reference p0*c [eV]
N_PROBES        = 13
PHASE_RANGE     = (-180, 180)           # Entrance arrival-time phases [deg]

################################################################################
# Load and scale the complex field map
################################################################################
# Keep the CLIC map's shape, not its original power calibration. The file does
# not specify that calibration: PEAK_EZ defines this illustrative field instead.
repo_root   = Path(__file__).resolve().parents[3]
data        = np.loadtxt(repo_root / MAP_FILE)
s_map       = data[:, 0] * 1e-3          # File longitudinal positions: mm -> m
ez_shape    = data[:, 1] + 1j * data[:, 2]
ez          = ez_shape * PEAK_EZ / np.max(np.abs(ez_shape))
mesh_step   = s_map[1] - s_map[0]

################################################################################
# Wrap the RF-Track field map
################################################################################
# Arguments: complex Ez [V/m], mesh step [m], length [m], frequency [Hz],
# propagation direction (+1). Length -1 uses the field-map length.
rf_map      = rft.RF_FieldMap_1d(ez, mesh_step, -1, FREQUENCY, +1)
# Keep power rescaling unity; PEAK_EZ sets the field amplitude.
rf_map.set_P_map(1.0)
rf_map.set_P_actual(1.0)
rf_map.set_t0(0)                         # Fixed clock origin [mm/c]
rf_map.set_phid(0)                       # No additional RF phase [deg]

# Keep p0*c fixed: acceleration appears in delta and ptau. Updating p0*c would
# change the momentum used to normalize the particle coordinates.
line        = xt.Line(elements = [xt.RFT_Element(
    element     = rf_map,
    update_ref  = False,
)])
line.set_particle_ref('electron', p0c = P0C)

################################################################################
# Scan the entrance arrival time
################################################################################
# phase = 360*f*t and zeta = -beta0*c*t: positive zeta arrives earlier.
# Zero phase follows the map and clock convention; it need not be RF crest.
phase_deg   = np.linspace(*PHASE_RANGE, N_PROBES)
arrival_t   = phase_deg / (360 * FREQUENCY)
beta0       = line.particle_ref.beta0[0]
particles   = line.build_particles(zeta = -beta0 * clight * arrival_t)
energy_in   = particles.energy.copy()
line.track(particles)
energy_out  = particles.energy
gain_mev    = (energy_out - energy_in) * 1e-6

print(f'Field-map length: {line.get_length():.6f} m')
print(f'Chosen peak |complex Ez|: {PEAK_EZ * 1e-6:.3f} MV/m')
print(f'Fixed reference p0*c: {particles.p0c[0] * 1e-6:.3f} MeV')
xt.Table(
    data    = {
        'phase_deg': phase_deg,
        'gain_MeV':  gain_mev,
        'delta':     particles.delta,
        'state':     particles.state,
    },
    index   = 'phase_deg',
).show()

################################################################################
# Plot the imported field shape and energy gain
################################################################################
fig, axes   = plt.subplots(2, 1, figsize = (8, 7))

########################################
# Complex field amplitudes
########################################
axes[0].plot(s_map, ez.real * 1e-6, label = 'Re Ez')
axes[0].plot(s_map, ez.imag * 1e-6, label = 'Im Ez')
axes[0].set_xlabel('Position in map [m]')
axes[0].set_ylabel('Complex field amplitude [MV/m]')
axes[0].legend()

########################################
# Entrance arrival-time scan
########################################
axes[1].plot(phase_deg, gain_mev, 'o-')
axes[1].axhline(0, color = 'black', linewidth = 0.5)
axes[1].set_xlabel('Entrance arrival-time phase [deg]')
axes[1].set_ylabel('Electron energy gain [MeV]')
fig.tight_layout()
plt.show()
