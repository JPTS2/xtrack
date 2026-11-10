# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c
import xtrack as xt
import xtrack.synctime as st

################################################################################
# User parameters
################################################################################

########################################
# Two species at equal initial magnetic rigidity
########################################
P0C                 = 20e9              # Fe-56 reference momentum [eV/c]
CIRCUMFERENCE       = 216.0             # Constant flight path [m]
VOLTAGES            = [15000.0, 18000.0] # Fe and Bi RF systems [V]
HARMONIC            = 2

########################################
# Common rigidity ramp and frame clock
########################################
RAMP_DURATION       = 2e-3              # [s]
MOMENTUM_INCREASE   = 0.02              # Relative change over full programme
N_RF_SAMPLES        = 513
FRAME_FRACTION      = 0.9
N_FRAMES            = 256

################################################################################
# Describe both species using one particle reference
################################################################################

########################################
# Keep the native species masses and charges explicit
########################################
iron            = xt.Particles('Iron-56', q0 = 25, p0c = P0C)
bismuth         = xt.Particles('Bismuth-209', q0 = 68)
mass_ratio      = bismuth.mass0 / iron.mass0
charge_ratio    = bismuth.q0 / iron.q0
chi             = charge_ratio / mass_ratio

# In the shared reference, equal rigidity means (1+delta)/chi = 1.
# It does not mean delta=0 for both species. The native database approximates
# the Bi-209 mass as 209 atomic mass units.
particles   = xt.Particles(
    mass0           = iron.mass0,
    q0              = iron.q0,
    p0c             = P0C,
    mass_ratio      = [1.0, mass_ratio],
    charge_ratio    = [1.0, charge_ratio],
    delta           = [0.0, chi - 1.0])

########################################
# Construct a constant-path timing model with two RF locations
########################################
# This drift-only loop isolates longitudinal timing, not a machine's optics.
# Both probes traverse both cavities; no collective or inter-species force acts.
line    = xt.Line(elements = {
    'first_flight':  xt.Drift(length = 0.25 * CIRCUMFERENCE),
    'rf_fe':         xt.Cavity(absolute_time = True),
    'middle_flight': xt.Drift(length = 0.40 * CIRCUMFERENCE),
    'rf_bi':         xt.Cavity(absolute_time = True),
    'last_flight':   xt.Drift(length = 0.35 * CIRCUMFERENCE)})
line.particle_ref   = iron.copy()

################################################################################
# Coordinate the reference ramp and each species' RF clock
################################################################################

########################################
# Let the native energy programme update the common reference and frame times
########################################
line.energy_program = xt.EnergyProgram(
    t_s     = [0.0, RAMP_DURATION],
    p0c     = [P0C, P0C * (1 + MOMENTUM_INCREASE)])
times           = np.linspace(0, RAMP_DURATION, N_RF_SAMPLES)
reference_pc    = line.energy_program.get_p0c_at_t_s(times)
momentum_rate   = P0C * MOMENTUM_INCREASE / RAMP_DURATION
voltage_acc     = CIRCUMFERENCE / c * momentum_rate / iron.q0
if abs(voltage_acc) >= min(VOLTAGES):
    raise ValueError('Increase RF voltages or slow the rigidity ramp.')
print('Required accelerating voltage per unit charge [V]:', voltage_acc)

# Changing the reference preserves physical particle energies. The cavities
# must supply the energy gains required by the common rigidity ramp.

########################################
# Integrate the phase for each RF system, with one laboratory origin
########################################
for name, species, ratio, location, voltage in [
        ('rf_fe', iron, 1.0, 0.25, VOLTAGES[0]),
        ('rf_bi', bismuth, charge_ratio, 0.65, VOLTAGES[1])]:
    pc          = ratio * reference_pc
    energy      = np.hypot(pc, species.mass0)
    frequency   = HARMONIC * pc / energy * c / CIRCUMFERENCE
    sync_phase  = np.arcsin(voltage_acc / voltage)

    # For a linear pc ramp, integral(beta*dt) = (E(t)-E(0))/d(pc)/dt.
    # Subtract the flight phase to this cavity's actual location.
    phase   = (2 * np.pi * HARMONIC * c / CIRCUMFERENCE
             * (energy - energy[0]) / (ratio * momentum_rate)
             + sync_phase - 2 * np.pi * HARMONIC * location)
    line.functions[f'frequency_{name}'] = xt.FunctionPieceWiseLinear(
        x   = times,
        y   = frequency)
    line.functions[f'phase_{name}'] = xt.FunctionPieceWiseLinear(
        x   = times,
        y   = phase)
    clock                   = line.ref['t_turn_s']
    line[name].frequency    = line.functions[f'frequency_{name}'](clock)
    line[name].phase        = (line.functions[f'phase_{name}'](clock)
                              - 2 * np.pi * line.ref[name].frequency * clock)
    line[name].voltage      = voltage

# These are sampled phase/frequency programmes: within a frame the RF phase
# uses the tangent at the frame epoch; see the frequency-ramp timing example.
# Each system is designed for its own species; both fields act on both probes.

################################################################################
# Track the two probes on a common native frame clock
################################################################################

########################################
# Install synchronization after constructing the complete flight path
########################################
st.install_sync_time_at_collective_elements(
    line,
    frame_relative_length   = FRAME_FRACTION,
    with_progress           = False)
line.enable_time_dependent_vars         = True
line.dt_update_time_dependent_vars      = 0.0
line.build_tracker()

########################################
# Record individual event arrival times and physical energies
########################################
line.track(particles, num_turns = N_FRAMES, log = xt.Log(
    arrival     = lambda line, p: (
        p.time_s + (p.s - p.zeta) / (p.beta0 * c)),
    energy      = lambda line, p: p.energy.copy(),
    particle_id = lambda line, p: p.particle_id.copy()))
logged      = line.log_last_track
order       = np.argsort(logged['particle_id'], axis = 1)
arrival     = np.take_along_axis(np.array(logged['arrival']), order, axis = 1)
energy      = np.take_along_axis(np.array(logged['energy']), order, axis = 1)

# SyncTime can store waiting probes with negative states. Their events span
# frame windows, rather than forming a simultaneous snapshot at time_s.
order   = np.argsort(particles.particle_id)
xt.Table(
    data    = {
        'species':          np.array(['Fe-56(25+)', 'Bi-209(68+)']),
        'mass_ratio':       particles.mass_ratio[order],
        'charge_ratio':     particles.charge_ratio[order],
        'delta':            particles.delta[order],
        'stored_turn':      particles.at_turn[order],
        'kinetic_MeV':      (particles.energy[order]
                             - iron.mass0 * particles.mass_ratio[order]) / 1e6,
        'state':            particles.state[order]},
    index   = 'species').show(maxwidth = 'full')
print('Shared reference p0*c [GeV]:', particles.p0c / 1e9)
print('Last RF/reference update epoch [us]:', line['t_turn_s'] * 1e6)
print('Common clock [us] / reference turns:',
      particles.time_s * 1e6, particles.reference_turns)

################################################################################
# Compare energy gains with each species' equal-rigidity design trajectory
################################################################################
fig, axes   = plt.subplots(2, 1, sharex = True, figsize = (8, 6))
for index, (species, ratio, label) in enumerate([
        (iron, 1.0, 'Fe-56 (25+)'), (bismuth, charge_ratio, 'Bi-209 (68+)')]):
    reference_pc    = line.energy_program.get_p0c_at_t_s(arrival[:, index])
    pc_design       = ratio * reference_pc
    energy_design   = np.hypot(pc_design, species.mass0)
    energy_initial  = np.hypot(ratio * P0C, species.mass0)
    axes[index].plot(arrival[:, index] * 1e6,
                     (energy[:, index] - energy_initial) / 1e6,
                     label = 'Tracked, both RF systems')
    axes[index].plot(arrival[:, index] * 1e6,
                     (energy_design - energy_initial) / 1e6, '--',
                     label = 'Equal-rigidity design')
    axes[index].set_title(label)
    axes[index].set_ylabel('Energy gain [MeV/ion]')
    axes[index].grid()
    axes[index].legend()
axes[-1].set_xlabel('Arrival time at stored event [us]')
fig.tight_layout()
plt.show()
