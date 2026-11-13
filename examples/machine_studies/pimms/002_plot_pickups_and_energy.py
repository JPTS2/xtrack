# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgba
import numpy as np

################################################################################
# User parameters
################################################################################
DATA_FILE                   = 'outputs/004_pimm_carbon_protons.npz'
COLORS                      = ('#007f86', '#d65b32')
LABELS                      = (r'$^{12}$C$^{6+}$', 'protons')
WINDOW_INJECTION_PERIODS    = 8       # Window in carbon injection periods

################################################################################
# Load the recorded pickup data
################################################################################
# Run 001_accelerate_and_record_pickups.py first. Plotting needs only NumPy
# and Matplotlib; it neither imports Xsuite nor repeats the tracking.
data_path   = Path(__file__).resolve().parent / DATA_FILE
with np.load(data_path, allow_pickle = False) as data:
    result  = dict(data)

time_us     = result['time_s'].ravel() * 1e6
currents    = result['current_A'] * 1e3
period_us   = result['periods_s'][0] * 1e6
frequency   = result['frequencies_Hz'] / 1e3
window_us   = WINDOW_INJECTION_PERIODS * period_us
end_us      = float(result['acquisition_duration_s']) * 1e6

################################################################################
# Compare initial and later pickup passages
################################################################################
fig     = plt.figure(figsize = (10, 8), layout = 'constrained')
grid    = fig.add_gridspec(3, 2, height_ratios = (1, 1, 1.5))
first   = fig.add_subplot(grid[0, :])
last    = fig.add_subplot(grid[1, :], sharey = first)
fig.suptitle('Two species at the same magnetic rigidity')

for ax, start, title in (
        (first, 0., 'Initial passages'),
        (last, end_us - window_us, 'Later passages')):
    selected    = (time_us >= start) & (time_us <= start + window_us)
    for ii in range(2):
        ax.plot(
            time_us[selected], currents[ii].ravel()[selected],
            color   = COLORS[ii],
            label   = f'{LABELS[ii]}: {frequency[ii]:.1f} kHz at injection')
    ax.set(
        xlim    = (start, start + window_us),
        xlabel  = 'Laboratory time [µs]',
        ylabel  = 'Pickup current [mA]',
        title   = title)
    ax.grid(alpha = .2)

carbon_mev  = float(result['carbon_kinetic_energy_per_nucleon_eV']) / 1e6
proton_mev  = float(result['proton_kinetic_energy_eV']) / 1e6
length      = float(result['circumference_m'])
first.text(
    .01, .95,
    f'Injection: carbon {carbon_mev:g} MeV/u; protons {proton_mev:.2f} MeV\n'
    f'Circumference {length:.2f} m; '
    f'injection frequency ratio {frequency[1] / frequency[0]:.3f}',
    transform   = first.transAxes,
    va          = 'top')
first.legend(loc = 'upper right', frameon = False)
first.set_ylim(0, 1.55 * currents.max())

################################################################################
# Fold the continuous record around the mean carbon arrival
################################################################################

########################################
# Choose one common circular origin
########################################
# Use the circular mean because a bunch can straddle a row boundary. One
# fixed origin retains the carbon-centroid motion and relative proton timing.
# Increasing arrival coordinate means later arrival, opposite to physical zeta.
num_slices      = currents.shape[-1]
slice_phase     = 2 * np.pi * (
    (np.arange(num_slices) + .5) / num_slices - .5)
carbon_profile  = result['counts'][0].sum(axis = 0)
carbon_phase    = np.angle(np.sum(carbon_profile * np.exp(1j * slice_phase)))
shift           = -int(np.rint(carbon_phase * num_slices / (2 * np.pi)))
flat_currents   = currents.reshape(2, -1)
centered_flat   = np.full_like(flat_currents, np.nan)

# Shift the continuous acquisition, not each row independently: crossings
# belong in the adjacent row. Leave the acquisition ends blank, without wrap.
if shift > 0:
    centered_flat[:, shift:]    = flat_currents[:, :-shift]
elif shift < 0:
    centered_flat[:, :shift]    = flat_currents[:, -shift:]
else:
    centered_flat[:]    = flat_currents
centered_currents   = centered_flat.reshape(currents.shape)

########################################
# Overlay both species without hiding empty bins
########################################
pickup_grid = grid[2, :].subgridspec(1, 3, width_ratios = (1, .025, .025))
folded      = fig.add_subplot(pickup_grid[0, 0])
folded.set(
    xlabel  = 'Reference arrival coordinate [m]; positive later',
    ylabel  = 'Carbon reference-turn index',
    title   = 'All recorded passages in the carbon reference frame')

for ii in range(2):
    cmap            = LinearSegmentedColormap.from_list(
        f'species_{ii}', [to_rgba(COLORS[ii], 0), to_rgba(COLORS[ii], 1)])
    picture         = folded.imshow(
        centered_currents[ii],
        origin          = 'lower',
        aspect          = 'auto',
        extent          = (-.5 * length, .5 * length,
                           -.5, currents.shape[1] - .5),
        cmap            = cmap,
        vmin            = 0,
        vmax            = currents.max(),
        interpolation   = 'nearest')
    colorbar_axis   = fig.add_subplot(pickup_grid[0, ii + 1])
    fig.colorbar(
        picture, cax = colorbar_axis, label = f'{LABELS[ii]} current [mA]')

################################################################################
# Compare the programmed energy with complete bunch passages
################################################################################
energy_fig, axes    = plt.subplots(
    2, 1, figsize = (10, 6), sharex = True, layout = 'constrained')
energy_fig.suptitle('Energy measured at the pickup')

for ii in range(2):
    ########################################
    # Convert slice pzeta moments to kinetic energy
    ########################################
    scale       = (12., 1.)[ii] * 1e6
    mass        = result['species_masses_eV'][ii]
    mass0       = result['species_masses_eV'][0]
    pc          = result['ramp_p0c'] * result['species_charge_ratios'][ii]
    target      = (np.hypot(pc, mass) - mass) / scale
    occupied    = ((result['counts'][ii] > 0)
                   & (result['time_s'] >= 0)
                   & (result['time_s'] <= result['acquisition_duration_s']))
    weights     = result['counts'][ii][occupied]
    sample_time = result['time_s'][occupied]
    p0c         = result['reference_p0c_eV'][ii][occupied]
    energy0     = np.hypot(p0c, mass0)
    conversion  = mass / mass0 * p0c**2 / energy0 / scale
    # E = mass_ratio * (E0 + beta0*p0c*pzeta). The recorded mean p0c
    # approximates the reference within slices straddling a SyncTime boundary.
    slice_mean  = (mass / mass0 * (energy0 - mass0) / scale
                   + conversion * result['mean_pzeta'][ii][occupied])
    slice_sigma = conversion * result['sigma_pzeta'][ii][occupied]

    ########################################
    # Group by each species' own RF phase
    ########################################
    # Use successive h=1 RF cycles to group each bunch passage. These differ
    # from carbon reference rows and from particles' completed-turn counters.
    bucket          = np.floor(
        result['pickup_rf_phase_rad'][ii][occupied] / (2 * np.pi) + .5
    ).astype(int)
    phase_limits    = np.interp(
        [0, result['acquisition_duration_s']], result['time_s'].ravel(),
        result['pickup_rf_phase_rad'][ii].ravel()) / (2 * np.pi)
    passage         = []
    for number in np.unique(bucket):
        if (number - .5 < phase_limits[0]
                or number + .5 > phase_limits[1]):
            continue
        selected    = bucket == number
        w           = weights[selected]
        mean        = np.average(slice_mean[selected], weights = w)
        # Total variance includes both within-slice spread and slice means.
        variance    = np.average(
            slice_sigma[selected]**2 + (slice_mean[selected] - mean)**2,
            weights = w)
        arrival     = np.average(sample_time[selected], weights = w)
        passage.append((arrival, mean, np.sqrt(variance)))
    energy_time, mean, sigma    = np.asarray(passage).reshape(-1, 3).T

    ########################################
    # Plot the passage mean and rms spread
    ########################################
    ax  = axes[ii]
    ax.plot(
        result['ramp_times_s'] * 1e6, target, 'k--',
        label   = 'Programmed energy at equal rigidity')
    ax.plot(
        energy_time * 1e6, mean,
        color   = COLORS[ii],
        label   = 'Passage mean')
    ax.fill_between(
        energy_time * 1e6, mean - sigma, mean + sigma,
        color   = COLORS[ii],
        alpha   = .25,
        label   = 'Passage rms spread')
    ax.set(
        ylabel  = ('Carbon [MeV/u]', 'Protons [MeV]')[ii],
        title   = f'{LABELS[ii]}: {result["survivors"][ii]:.1%} surviving')
    ax.grid(alpha = .2)

axes[0].legend(frameon = False)
axes[1].set(xlabel = 'Laboratory time [µs]', xlim = (0, end_us))
plt.show()
