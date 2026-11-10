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
# Shared electron ring and a gentle reference-momentum ramp
########################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
P0C_END         = 1.00004e9             # Final program p0*c [eV]
N_TURNS         = 256                   # Two extra turns leave a program margin

########################################
# RF energy supply
########################################
RF_VOLTAGE      = 1e5                   # [V], installed harmonic is 200

################################################################################
# Configure native reference evolution and harmonic RF synchronization
################################################################################

########################################
# Attach a linear-in-momentum EnergyProgram
########################################
repo_root       = Path(__file__).resolve().parents[3]
env             = xt.load(repo_root / LATTICE_FILE)
line            = env['demo']
p0c_start       = float(line.particle_ref.p0c[0])
beta0_start     = float(line.particle_ref.beta0[0])
ramp_duration   = (N_TURNS + 2) * line.get_length() / (beta0_start * c)
line.energy_program = xt.EnergyProgram(
    t_s     = [0.0, ramp_duration],
    p0c     = [p0c_start, P0C_END],
)
program         = line.energy_program

# Constant normalized magnet strengths represent fields following reference
# rigidity. Radiation and collective forces are off; no hardware model is used.

########################################
# Use the native harmonic mode and select the accelerating phase
########################################
env['frequency_rf']     = 0.0
line['rf.1'].harmonic   = env['harmonic_rf']
env['voltage_rf']       = RF_VOLTAGE

first_turn_time = program.get_t_s_at_turn(1)
energy_per_turn = (program.get_kinetic_energy0_at_t_s(first_turn_time)
                    - program.get_kinetic_energy0_at_t_s(0.0))
env['phase_rf'] = np.pi - np.arcsin(
    energy_per_turn / (abs(line.particle_ref.q0) * RF_VOLTAGE))
env['t_turn_s'] = 0.0
print('Required reference energy gain per turn [eV]:', energy_per_turn)
print('RF phase [rad], stored frequency [Hz], harmonic:',
      env['phase_rf'], line['rf.1'].frequency, line['rf.1'].harmonic)

# Nonzero harmonic with stored frequency zero computes f=h*beta0*c/C natively.
# The phase is relative to each local reference arrival, not an absolute-time
# frequency integral. This linear p0c(t) program has constant energy gain/turn.
# Above transition, pi-asin(DeltaE/(abs(q0)*V)) selects the focusing branch.

################################################################################
# Compare identical initial particles with and without physical RF kicks
################################################################################

########################################
# Use the native accelerating fixed point as a near-synchronous launch
########################################
initial = line.twiss6d(chrom = False).particle_on_co.copy()
control = line.copy()
control.vars['voltage_rf']      = 0.0

########################################
# Record physical energy, reference energy and the last applied RF setting
########################################
histories   = {}
for label, case in [('RF on', line), ('RF off', control)]:
    particles   = initial.copy()
    samples     = []
    case.enable_time_dependent_vars         = True
    case.dt_update_time_dependent_vars      = None
    for turn in range(N_TURNS + 1):
        if turn:
            case.track(particles, num_turns = 1)
        applied_time    = float(case['t_turn_s'])
        effective_rf    = (case['rf.1'].harmonic
                        * case.energy_program.get_frev_at_t_s(applied_time))
        samples.append((particles.time_s, applied_time, particles.energy[0],
                        particles.energy0[0], particles.delta[0], effective_rf,
                        particles.state[0]))
    samples     = np.array(samples)
    histories[label]    = xt.Table(
        data        = {
            'turn':         np.arange(N_TURNS + 1),
            'time_s':       samples[:, 0],
            'applied_s':    samples[:, 1],
            'energy_eV':    samples[:, 2],
            'reference_eV': samples[:, 3],
            'delta':        samples[:, 4],
            'rf_Hz':        samples[:, 5],
            'state':        samples[:, 6],
        },
        index       = 'turn',
    )
    print(label, 'final state / physical gain [keV] / delta:',
          particles.state[0], 1e-3 * (particles.energy[0] - initial.energy[0]),
          particles.delta[0])

# EnergyProgram changes the reference and renormalizes coordinates while
# preserving physical energy. Only the powered cavity supplies acceleration.
# After a turn, time_s is the next entrance; applied_s is the last update.

################################################################################
# Inspect energy following and the effective RF frequency
################################################################################

########################################
# Compare physical energy gains with the requested reference program
########################################
powered         = histories['RF on']
program_gain    = (program.get_kinetic_energy0_at_t_s(powered.time_s)
                   - program.get_kinetic_energy0_at_t_s(0.0))
following_error = powered.energy_eV - powered.energy_eV[0] - program_gain
print('Final program gain at the particle clock [keV]:',
      1e-3 * program_gain[-1])
print('Maximum / final energy-following error [eV]:',
      np.max(np.abs(following_error)), following_error[-1])
print('Next entrance / last applied clock [s]:',
      powered.time_s[-1], powered.applied_s[-1])
fig, axes = plt.subplots(2, 1, figsize = (8, 6))
for label, history in histories.items():
    axes[0].plot(
        1e3 * history.time_s,
        1e-3 * (history.energy_eV - history.energy_eV[0]),
        label   = label,
    )
axes[0].plot(
    1e3 * powered.time_s,
    1e-3 * program_gain,
    'k--',
    label   = 'Program reference',
)
axes[0].set_ylabel('Physical energy gain [keV]')
axes[0].set_xlabel('Particle reference clock [ms]')
axes[0].legend()

########################################
# Resolve the small frequency change of this ultrarelativistic electron ramp
########################################
axes[1].plot(1e3 * powered.applied_s, 1e3 * (powered.rf_Hz - powered.rf_Hz[0]))
axes[1].set_ylabel('Effective RF frequency change [mHz]')
axes[1].set_xlabel('Last applied entrance clock [ms]')
for axis in axes:
    axis.grid()
fig.tight_layout()
plt.show()

# These small launches and reference changes are qualified, not a momentum-
# acceptance or long-term acceleration study. No radiation loss is compensated.
