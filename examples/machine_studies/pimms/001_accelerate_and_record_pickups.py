# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
from pathlib import Path

import numpy as np
from scipy.constants import c as clight, elementary_charge
from scipy.integrate import cumulative_trapezoid

import xobjects as xo
import xtrack as xt
import xtrack.synctime as st

################################################################################
# User parameters
################################################################################

########################################
# Common-rigidity carbon acceleration programme
########################################
CARBON_EKIN_PER_NUCLEON     = 7e6      # Injection kinetic energy [eV/u]
FINAL_ENERGY_PER_NUCLEON    = 9e6      # End of the acquisition [eV/u]
N_CARBON_PERIODS            = 256      # Duration in INJECTION carbon periods
TURN_ON_FRACTION            = 0.2     # Smooth energy-slope turn-on
FRAME_FRACTION              = 0.45    # Reference revolutions per frame
RF_VOLTAGES                 = (40e3, 30e3)  # Carbon / proton cavities [V]
RF_POSITIONS                = (0.001, 0.01)  # Cavity locations [m]

########################################
# Two reproducible bunches of equal total charge
########################################
N_PARTICLES                 = 1000    # Per species
SEED                        = 20261009
NEMITT                      = 1e-7    # Normalized transverse emittance [m]
SIGMA_TIME_FRACTION         = 0.02    # RMS duration / carbon injection period
ARRIVAL_FRACTIONS           = (0.58, 0.78)  # Injection carbon periods
PROTON_ID_START             = 1_000_000

########################################
# Pickup acquisition and dispersive physical aperture
########################################
N_SLICES                    = 512     # Fixed-zeta bins per reference period
APERTURE_ELEMENT            = 'qfb.3'
APERTURE_HALF_WIDTH         = 0.10    # Horizontal half-width [m]
OUTPUT_FILE                 = 'outputs/004_pimm_carbon_protons.npz'

################################################################################
# Load the native PIMMS demonstration and define equal-rigidity species
################################################################################

########################################
# Read static lattice and strength data without an external MAD-X process
########################################
repo_root   = Path(__file__).resolve().parents[3]
data        = repo_root / 'test_data' / 'pimms'
env         = xt.load(
    [data / 'PIMMS.seq', data / 'pimms_optics.str'])
line        = env['pimms']
line.set_particle_ref('Carbon-12', kinetic_energy0 = 12*CARBON_EKIN_PER_NUCLEON)
line.configure_bend_model(core = 'bend-kick-bend', edge = 'full')
# This repository PIMMS demonstration is not an operational machine model.

########################################
# Compare the injection speeds on the common magnetic-rigidity orbit
########################################
carbon          = line.particle_ref
proton          = xt.Particles('proton', p0c = carbon.p0c[0] / carbon.q0)
circumference   = line.get_length()
beta_c          = carbon.beta0[0]
beta_p          = proton.beta0[0]
periods         = circumference / (clight * np.array([beta_c, beta_p]))
frequencies     = 1 / periods
mass_ratio      = proton.mass0 / carbon.mass0
charge_ratio    = proton.q0 / carbon.q0
chi_p           = charge_ratio / mass_ratio

################################################################################
# Define the smooth common-rigidity acceleration programme
################################################################################

########################################
# Integrate a raised-cosine turn-on of the carbon energy slope
########################################
# Keep the normalized magnetic strengths fixed: the physical dipole and
# quadrupole fields then rise with the common reference rigidity.
# Duration is measured in INJECTION carbon periods, not actual turns.
duration            = N_CARBON_PERIODS * periods[0]
turn_on_duration    = TURN_ON_FRACTION * duration
times               = np.linspace(0, duration + 8*periods[0], 20001)
# A raised-cosine energy slope joins zero smoothly to a constant slope.
# Its integral is t/2 - tau*sin(pi*t/tau)/(2*pi) during turn-on, then
# t - tau/2. Normalize to reach the requested energy at t=duration.
# Continue the linear ramp through the short tracking/acquisition margin.
turn_on             = np.minimum(times/turn_on_duration, 1.)
ramp_time           = times - .5*turn_on_duration*(
    turn_on + np.sin(np.pi*turn_on)/np.pi)
energy_slope        = (
    12*(FINAL_ENERGY_PER_NUCLEON - CARBON_EKIN_PER_NUCLEON)
    / (duration - .5*turn_on_duration))
kinetic_energy      = 12*CARBON_EKIN_PER_NUCLEON + energy_slope*ramp_time
kinetic_energy_rate = energy_slope * .5*(1 - np.cos(np.pi*turn_on))
momentum            = np.sqrt(kinetic_energy*(kinetic_energy + 2*carbon.mass0))
momentum_rate       = ((kinetic_energy + carbon.mass0)/momentum
                       * kinetic_energy_rate)

########################################
# Compare the required energy gain per charge with the available RF
########################################
voltage_acc         = circumference/clight * momentum_rate/carbon.q0
# Both beams require the same energy gain per charge per revolution.
if np.max(np.abs(voltage_acc)) >= min(RF_VOLTAGES):
    raise ValueError(
        'Ramp exceeds the RF voltage; increase N_CARBON_PERIODS.')

################################################################################
# Prepare both bunches before enabling the RF and reference ramp
################################################################################

########################################
# Find the closed orbit for each species at equal rigidity
########################################
# Find the two transverse closed orbits with RF off. Equal rigidity means
# delta=chi-1 in Xsuite's shared reference, including the proton near +1.
tw_c    = line.twiss4d()
tw_p    = line.twiss4d(
    delta0          = chi_p - 1,
    mass_ratio      = mass_ratio,
    charge_ratio    = charge_ratio)
rng     = np.random.default_rng(SEED)
sigma_t = SIGMA_TIME_FRACTION * periods[0]
# Delay both bunches so the longer Gaussian tails fit after the leading
# edge of the first SyncTime window. Keep their relative timing unchanged.
arrival_offsets = np.array(ARRIVAL_FRACTIONS) * periods[0]

########################################
# Use native transverse matching and small-amplitude own-RF preparation
########################################
beams           = []
for ii, (tw, species, chi, beta, voltage) in enumerate(zip(
        (tw_c, tw_p), (carbon, proton), (1., chi_p),
        (beta_c, beta_p), RF_VOLTAGES)):
    # Approximate small-amplitude longitudinal matching to each own RF.
    # alpha refers to the common equal-rigidity orbit. With both RF systems
    # on this is only an initial match, not a stationary two-RF solution.
    eta             = tw_c.momentum_compaction_factor - 1/species.gamma0[0]**2
    sigma_phase     = 2*np.pi*frequencies[ii]*sigma_t
    sigma_dp        = sigma_phase * np.sqrt(
        species.q0 * voltage
        / (2*np.pi*abs(eta)*beta**2*species.energy0[0]))
    dp              = sigma_dp * rng.normal(size = N_PARTICLES)
    bunch           = line.build_particles(
        particle_on_co  = tw.particle_on_co,
        W_matrix        = tw.W_matrix[0],
        x_norm          = rng.normal(size = N_PARTICLES),
        px_norm         = rng.normal(size = N_PARTICLES),
        y_norm          = rng.normal(size = N_PARTICLES),
        py_norm         = rng.normal(size = N_PARTICLES),
        nemitt_x        = NEMITT,
        nemitt_y        = NEMITT,
        delta           = chi*(1 + dp) - 1,
        zeta            = -beta_c*clight*(
            arrival_offsets[ii] + sigma_t*rng.normal(size = N_PARTICLES)),
        # Equal total bunch charge makes the two current traces comparable.
        weight          = 1e8 / species.q0 / N_PARTICLES)
    bunch.pdg_id    = species.pdg_id[0]
    if ii == 1:
        bunch.particle_id += PROTON_ID_START
        bunch.parent_particle_id += PROTON_ID_START
    beams.append(bunch)
particles   = xt.Particles.merge(beams)

################################################################################
# Install the dispersive aperture and two absolute-time RF systems
################################################################################

########################################
# Place the horizontal limit at a large-dispersion station
########################################
line.discard_tracker()
aperture_name   = 'dispersive_aperture'
aperture_s      = float(tw_c['s', APERTURE_ELEMENT])
aperture_dx     = float(tw_c['dx', APERTURE_ELEMENT])
line.insert(
    aperture_name,
    xt.LimitRect(
        min_x   = -APERTURE_HALF_WIDTH,
        max_x   = APERTURE_HALF_WIDTH),
    at = aperture_s)

########################################
# Integrate each frequency programme and bind it to the native clock
########################################
rf_phases   = []
for ii, (name, position, beta, voltage) in enumerate(zip(
        ('rf_carbon', 'rf_proton'), RF_POSITIONS,
        (beta_c, beta_p), RF_VOLTAGES)):
    species         = (carbon, proton)[ii]
    pc              = momentum * species.q0/carbon.q0
    rf_frequency    = (clight/circumference
                       * pc/np.sqrt(pc**2 + species.mass0**2))
    # Integrate frequency; frequency(t)*t is not the chirped RF phase.
    rf_phase        = (
        2*np.pi*cumulative_trapezoid(rf_frequency, times, initial = 0)
        + np.arcsin(voltage_acc/voltage)
        - 2*np.pi*frequencies[ii]*arrival_offsets[ii]
        - 2*np.pi*position/circumference)
    rf_phases.append(rf_phase)
    line.insert(
        name, xt.Cavity(voltage = voltage, absolute_time = True),
        at = position)
    line.functions[f'frequency_{name}'] = xt.FunctionPieceWiseLinear(
        x = times, y = rf_frequency)
    line.functions[f'phase_{name}']     = xt.FunctionPieceWiseLinear(
        x = times, y = rf_phase)

    clock                   = line.ref['t_turn_s']
    line[name].frequency    = line.functions[f'frequency_{name}'](clock)
    line[name].phase        = (line.functions[f'phase_{name}'](clock)
                              - 2*np.pi*line.ref[name].frequency*clock)

################################################################################
# Record species-selected pickups on a common synchronized frame clock
################################################################################

########################################
# Record population-weighted longitudinal slices and their moments
########################################
monitor_names       = ('pickup_carbon', 'pickup_proton')
# Fixed zeta slices, indexed by accumulated carbon reference turns.
reference_turn_grid = cumulative_trapezoid(
    clight/circumference * momentum/np.sqrt(momentum**2 + carbon.mass0**2),
    times, initial = 0)
num_reference_turns = int(np.ceil(
    np.interp(duration, times, reference_turn_grid)))
monitors            = []
for name, first_id in zip(monitor_names, (0, PROTON_ID_START)):
    monitor                 = xt.BeamStatsMonitor(
        start_at_turn       = 0,
        stop_at_turn        = num_reference_turns + 1,
        coasting            = True,
        num_slices          = N_SLICES,
        particle_id_range   = (first_id, first_id + N_PARTICLES),
        stats               = ['num_particles', 'mean_pzeta', 'sigma_pzeta'])
    line.env.elements[name] = monitor
    monitors.append(monitor)
line.insert(list(monitor_names), at = 0)

########################################
# Attach the rigidity programme and install the synchronized boundaries
########################################
# Install the EnergyProgram after lattice insertions, which create
# intermediate lines sharing the environment's element dictionary.
line.energy_program = xt.EnergyProgram(t_s = times, p0c = momentum)
st.install_sync_time_at_collective_elements(
    line, frame_relative_length = FRAME_FRACTION,
    at_element_names = monitor_names)
line.enable_time_dependent_vars = True
# Use one serial CPU context for the bounded two-species tracking run.
context                         = xo.ContextCpu()
particles.move(_context = context)
line.build_tracker(_context = context)

reference_turns = line.energy_program.get_turn_at_t_s(duration + periods[0])
num_frames      = int(np.ceil(reference_turns / FRAME_FRACTION)) + 1
print('Tracking on one serial CPU thread')
print(f'Ring length: {circumference:.2f} m')
print(f'Horizontal aperture: +/-{APERTURE_HALF_WIDTH*100:g} cm at '
      f's={aperture_s:.4f} m, Dx={aperture_dx:.3f} m '
      f'(about +/-{APERTURE_HALF_WIDTH/abs(aperture_dx):.2%} '
      f'rigidity acceptance)')
print(f'Carbon: {CARBON_EKIN_PER_NUCLEON/1e6:g} MeV/u, '
      f'f_rev = {frequencies[0]/1e3:.3f} kHz, '
      f'T_rev = {periods[0]*1e6:.4f} us')
print(f'Proton: {proton.kinetic_energy0[0]/1e6:.4f} MeV, '
      f'f_rev = {frequencies[1]/1e3:.3f} kHz, '
      f'T_rev = {periods[1]*1e6:.4f} us')
print(f'f_proton / f_carbon = {frequencies[1]/frequencies[0]:.6f}')
print(f'{N_SLICES} samples per carbon period '
      f'({periods[0]/N_SLICES*1e9:.2f} ns bins)')
print(f'Carbon ramp: {CARBON_EKIN_PER_NUCLEON/1e6:g} -> '
      f'{FINAL_ENERGY_PER_NUCLEON/1e6:g} MeV/u '
      f'in {duration*1e6:.1f} us; '
      f'smooth turn-on over {turn_on_duration*1e6:.1f} us')
print(f'Peak accelerating voltage per charge: {voltage_acc.max():.1f} V')

################################################################################
# Track complete frames through the finite acquisition
################################################################################
line.track(particles, num_turns = num_frames, with_progress = 20)

################################################################################
# Convert pickup records to laboratory time and electrical current
################################################################################

########################################
# Recover the recorded moments and the reference used at each acquisition
########################################
# Convert the fixed zeta grid to lab time with the reference-turn/time map.
# Rows are NOT SyncTime frames or individual particles' turn counters.
time        = monitors[0].time_centers(
    line_length     = circumference,
    energy_program  = line.energy_program)
counts      = np.stack([np.asarray(mon.num_particles) for mon in monitors])
mean_pzeta  = np.stack([np.asarray(mon.mean_pzeta) for mon in monitors])
sigma_pzeta = np.stack([np.asarray(mon.sigma_pzeta) for mon in monitors])
# This moment is always recorded by BeamStatsMonitor. Use the reference
# momentum seen by the particles, rather than evaluating the ramp at bin
# centers.
sum_p0c         = np.stack([
    np.asarray(mon.data.sum_beta0_gamma0).reshape(time.shape)
    for mon in monitors]) * carbon.mass0
reference_p0c   = np.full_like(counts, np.nan)
np.divide(sum_p0c, counts, out = reference_p0c, where = counts > 0)

########################################
# Convert reference-turn bins to their changing laboratory-time widths
########################################
# Own-RF phase identifies successive bunch passages in the plotting script.
pickup_rf_phase             = np.stack([
    np.interp(time, times, phase) + 2*np.pi*position/circumference
    for phase, position in zip(rf_phases, RF_POSITIONS)])
turn_edges                  = (np.arange(counts[0].size + 1)/N_SLICES - .5)
time_edges                  = line.energy_program.get_t_s_at_turn(turn_edges)
time_edges[turn_edges < 0]  = turn_edges[turn_edges < 0]*periods[0]
dt                          = np.diff(time_edges).reshape(counts[0].shape)
# Counts represent ions; electrical current additionally uses species charge.
currents                    = (
    counts * np.array([carbon.q0, proton.q0])[:, None, None])
currents *= elementary_charge / dt

################################################################################
# Inspect survival and the physical energy of each species
################################################################################
survivors       = []
aperture_losses = []
aperture_index  = line.element_names.index(aperture_name)
for ii, first_id in enumerate((0, PROTON_ID_START)):
    mask        = ((particles.particle_id >= first_id)
                   & (particles.particle_id < first_id + N_PARTICLES))
    # Waiting SyncTime particles have reserved negative states; they live.
    alive       = ((particles.state > 0)
                   | (particles.state < -st.COAST_STATE_RANGE_START))
    survivors.append(np.count_nonzero(mask & alive) / N_PARTICLES)
    lost_here   = ((particles.state == 0)
                   & (particles.at_element == aperture_index))
    aperture_losses.append(np.count_nonzero(mask & lost_here)/N_PARTICLES)
    turns   = particles.at_turn[mask & alive]
    kinetic = (particles.energy
                   - particles.mass0*particles.mass_ratio)[mask & alive]
    print(f'{("Carbon", "Proton")[ii]}: {survivors[-1]:.1%} surviving, '
          f'{turns.min() if len(turns) else 0}--'
          f'{turns.max() if len(turns) else 0} '
          f'completed revolutions; final mean kinetic energy '
          f'{np.mean(kinetic)/1e6:.6f} MeV, '
          f'rms {np.std(kinetic)/1e3:.3f} keV; '
          f'{aperture_losses[-1]:.1%} lost at the dispersive aperture')

################################################################################
# Save the numerical record for the paired plotting example
################################################################################
# Keep the established NPZ keys, units and species order: carbon, then proton.
output_file = Path(__file__).resolve().parent / OUTPUT_FILE
output_file.parent.mkdir(parents = True, exist_ok = True)
np.savez_compressed(
    output_file,
    time_s                                  = time,
    current_A                               = currents,
    counts                                  = counts,
    periods_s                               = periods,
    frequencies_Hz                          = frequencies,
    carbon_kinetic_energy_per_nucleon_eV    = CARBON_EKIN_PER_NUCLEON,
    proton_kinetic_energy_eV                = proton.kinetic_energy0[0],
    num_carbon_turns                        = num_reference_turns,
    mean_pzeta                              = mean_pzeta,
    sigma_pzeta                             = sigma_pzeta,
    reference_p0c_eV                        = reference_p0c,
    pickup_rf_phase_rad                     = pickup_rf_phase,
    acquisition_duration_s                  = duration,
    circumference_m                         = circumference,
    survivors                               = np.array(survivors),
    ramp_times_s                            = times,
    ramp_p0c                                = momentum,
    species_masses_eV                       = np.array(
        [carbon.mass0, proton.mass0]),
    species_charge_ratios                   = np.array([1.0, charge_ratio]),
    final_energy_per_nucleon_eV             = FINAL_ENERGY_PER_NUCLEON,
    aperture_s_m                            = aperture_s,
    aperture_dx_m                           = aperture_dx,
    aperture_half_width_m                   = APERTURE_HALF_WIDTH,
    aperture_losses                         = np.array(aperture_losses),
    num_particles_per_species               = N_PARTICLES,
    num_threads                             = 1,
)
print('Data saved in', output_file)
