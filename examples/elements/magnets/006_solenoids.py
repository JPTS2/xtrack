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
# Uniform solenoid pair
########################################
P0C             = 200e6                 # Proton reference momentum [eV/c]
BS              = 2.0                   # Axial magnetic field [T]
LENGTH          = 3.0                   # Length of each solenoid [m]
PX_START        = 0.035                 # p_x / p0 for the orbit probe
N_SLICES        = 96                    # Sample the orbit and spin along s
PROTON_ANOMALY  = 1.79284734463          # (g - 2) / 2

########################################
# Field-axis offset with an embedded quadrupole
########################################
AXIS_LENGTH     = 0.3                   # Length of the combined element [m]
AXIS_X          = 5e-3                  # Horizontal displacement [m]
QUAD_K1L        = 0.15                  # Integrated gradient [m^-1]

########################################
# Tabulated detector field
########################################
FIELD_FILE      = 'test_data/fcc_ee/Bz_closed_before_quads.dat'
DETECTOR_P0C    = 182.5e9               # Positron reference momentum [eV/c]
HALF_SPAN       = 2.2                   # Half-span of the field model [m]
DETECTOR_PX     = 1e-3                  # Reference-normalized entrance momentum

################################################################################
# Add an opposite-polarity compensating solenoid
################################################################################
reference = xt.Particles('proton', p0c = P0C)
ks        = BS / reference.rigidity0[0]
line      = xt.Line(elements = {
    'main':     xt.UniformSolenoid(length = LENGTH, ks = ks),
    'opposite': xt.UniformSolenoid(length = LENGTH, ks = -ks),
})
# Each magnet keeps its default hard edges. Thick slices sample the motion
# inside each magnet without adding a fringe at every slice boundary.
line.slice_thick_elements(
    [xt.Strategy(slicing = xt.Uniform(N_SLICES, mode = 'thick'))],
    with_progress = False)
line.configure_spin('auto')

################################################################################
# Follow a tilted orbit and a transverse spin on the axis
################################################################################
# Both protons enter on axis, where the transverse vector potential vanishes.
# The tilted launch makes a helix. The second proton stays on axis, so its
# spin precesses about the longitudinal field with a simple analytic phase.
particles = xt.Particles(
    'proton',
    p0c                         = P0C,
    px                          = [PX_START, 0.0],
    spin_x                      = [1.0, 1.0],
    anomalous_magnetic_moment   = PROTON_ANOMALY)
samples = [[0.0, 0.0, 1.0, 0.0]]
for index in range(len(line.element_names)):
    line.track(particles, ele_start = index, num_elements = 1)
    samples.append([
        particles.x[0], particles.y[0],
        particles.spin_x[1], particles.spin_y[1]])
s            = np.r_[0.0, line.get_table().s_end[:-1]]
x, y, sx, sy = np.array(samples).T

################################################################################
# Compare the orbit radius and spin phase with the uniform-field predictions
################################################################################
radius          = PX_START / abs(ks)
main            = s <= LENGTH
tracked_radius  = np.hypot(x[main], y[main] + PX_START / ks)
spin_phase      = np.unwrap(np.arctan2(sy, sx))
expected_phase  = -(1 + PROTON_ANOMALY) * ks * np.minimum(s, 2 * LENGTH - s)
print('Helical orbit in the solenoid pair')
print('----------------------------------')
print(f'Axial field [T]: {BS:g}')
print(f'Helix radius [mm]: {1e3 * radius:.3f}')
print(f'Maximum radius error [m]: {np.max(abs(tracked_radius - radius)):.3e}')
print(f'Tilted probe exit x, y [mm]: {1e3 * x[-1]:.3f}, {1e3 * y[-1]:.3f}')
print('\nCompensate the on-axis spin rotation')
print('-------------------------------------')
print(f'After the main solenoid [rad]: {spin_phase[main][-1]:.6f}')
print(f'After the compensator [rad]: {spin_phase[-1]:.3e}')
phase_error = np.max(abs(spin_phase - expected_phase))
print(f'Maximum phase error [rad]: {phase_error:.3e}')

################################################################################
# Show the helix and the on-axis spin rotation
################################################################################
fig = plt.figure(figsize = (11, 4), layout = 'constrained')
orbit_ax = fig.add_subplot(121, projection = '3d')
orbit_ax.plot(s[main], 1e3 * x[main], 1e3 * y[main], label = 'Main')
compensator = s >= LENGTH
orbit_ax.plot(s[compensator], 1e3 * x[compensator], 1e3 * y[compensator],
              label = 'Compensator')
orbit_ax.set(xlabel = 's [m]', ylabel = 'x [mm]', zlabel = 'y [mm]',
             title = 'Tilted launch: helical orbit')
orbit_ax.legend()

spin_ax = fig.add_subplot(122)
spin_ax.plot(s, spin_phase, label = 'Tracked')
spin_ax.plot(s, expected_phase, '--', label = 'Analytic')
spin_ax.axvline(LENGTH, color = '0.5', linestyle = ':')
spin_ax.set(xlabel = 's [m]', ylabel = 'Unwrapped spin phase [rad]',
            title = 'On-axis spin rotation cancels')
spin_ax.grid()
spin_ax.legend()

################################################################################
# Move the solenoid axis alone or move the whole element
################################################################################
# Launch on the displaced solenoid axis in both cases. With x0 only, the
# embedded quadrupole remains centred at x = 0 and kicks the offset ray.
print('\nMove the solenoid axis or both fields')
print('-------------------------------------')
print(f'Horizontal shift and launch x [mm]: {1e3 * AXIS_X:g}')
for label, offsets in [
    ('Solenoid axis only (x0)', dict(x0 = AXIS_X)),
    ('Whole element (shift_x)', dict(shift_x = AXIS_X)),
]:
    magnet = xt.UniformSolenoid(
        length  = AXIS_LENGTH,
        ks      = ks,
        knl     = [0.0, QUAD_K1L],
        **offsets)
    probe = xt.Particles('proton', p0c = P0C, x = AXIS_X)
    magnet.track(probe)
    print(f'{label}, exit px [1]: {probe.px[0]:.3e}')

################################################################################
# Load the detector and screening field along their common axis
################################################################################
# This FCC-ee detector study input is in metres and tesla. Add a short
# linear closure to zero at each end. The beam frame is aligned with the
# field axis here, so the example needs only the local insertion.
repo_root = Path(__file__).resolve().parents[3]
z_tab, bz_tab = np.loadtxt(repo_root / FIELD_FILE, skiprows = 1, unpack = True)
z       = np.r_[-HALF_SPAN, z_tab, HALF_SPAN]
bz      = np.r_[0.0, bz_tab, 0.0]
initial = xt.Particles('positron', p0c = DETECTOR_P0C, px = DETECTOR_PX)
ks      = bz / initial.rigidity0[0]

################################################################################
# Make a VariableSolenoid segment between every pair of field samples
################################################################################
# Neighbours share an endpoint strength. The orbital map uses the segment
# mean, so the longitudinal field sampling sets the orbital resolution.
detector = xt.Line(elements = [
    xt.VariableSolenoid(length = length, ks_profile = [left, right])
    for length, left, right in zip(np.diff(z), ks[:-1], ks[1:])])
ray = initial.copy()
detector.track(ray, turn_by_turn_monitor = 'ONE_TURN_EBE')
record = detector.record_last_track

################################################################################
# Compare the transported ray with a field-free drift
################################################################################
drift_ray = initial.copy()
xt.Drift(length = detector.get_length()).track(drift_ray)
integral = np.trapezoid(bz, z)
print('\nAssemble the tabulated detector field')
print('-------------------------------------')
print(f'VariableSolenoid segments: {len(detector.element_names)}')
print(f'Length [m]: {detector.get_length():.3f}')
print(f'Signed field integral [T m]: {integral:.6g}')
print('\nCompare the field-free exits')
print('----------------------------')
print(f'Detector exit y [µm]: {1e6 * ray.y[0]:.3f}')
print(f'Drift exit y [µm]: {1e6 * drift_ray.y[0]:.3f}')
print(f'Exit vector potential ax, ay: {ray.ax[0]}, {ray.ay[0]}')

fig, axes = plt.subplots(2, 1, sharex = True, layout = 'constrained')
axes[0].plot(z, bz, '.-', markersize = 2)
axes[0].axhline(0, color = '0.5', linewidth = 0.8)
axes[0].set_ylabel(r'On-axis $B_z$ [T]')
axes[0].set_title('FCC-ee detector and screening field')
axes[1].plot(z, 1e6 * record.y[0], label = 'Detector field')
axes[1].axhline(0, color = '0.5', linestyle = '--', label = 'Drift')
axes[1].set(xlabel = 'z along the detector axis [m]', ylabel = 'Ray y [µm]')
axes[1].legend()
for ax in axes:
    ax.grid()
plt.show()
