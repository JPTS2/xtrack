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

################################################################################
# User parameters
################################################################################

########################################
# Weak RF timing probe on a repeated straight flight path
########################################
P0C                 = 1e9               # Electron reference p0*c [eV]
PATH_LENGTH         = 30.0              # [m]; not a closed ring geometry
VOLTAGE             = 100.0             # [V]; small kicks, fixed reference p0
N_TURNS             = 64

########################################
# Linear laboratory frequency chirp
########################################
FREQUENCY_START     = 1e6               # [Hz]
CHIRP_RATE          = 2e10              # df/dt [Hz/s]
PHASE_START         = 0.3               # Integrated phase at t=0 [rad]

################################################################################
# Configure the native clock-driven RF frequency
################################################################################

########################################
# Put the absolute-time cavity at the timing plane s=0
########################################
base    = xt.Line(elements = {
    'rf': xt.Cavity(voltage = VOLTAGE, absolute_time = True),
    'flight': xt.Drift(length = PATH_LENGTH),
})
base.set_particle_ref('electron', p0c = P0C)
base['f0']                          = FREQUENCY_START
base['chirp']                       = CHIRP_RATE
clock                               = base.ref['t_turn_s']
base['rf'].frequency                = (
    base.ref['f0'] + base.ref['chirp'] * clock)
base.enable_time_dependent_vars     = True
base.dt_update_time_dependent_vars  = 0.0

########################################
# Express the accumulated phase, not the instantaneous frequency times time
########################################
# For f(t)=f0+a*t, Theta(t)=phi0+2*pi*(f0*t+a*t^2/2).
# A native absolute-time cavity evaluates 2*pi*f_frame*t_arrival+phase_frame.
# Its phase offset must therefore subtract 2*pi*f_frame*t_frame.
base['phase_integral'] = 2 * np.pi * (
    base.ref['f0'] * clock + 0.5 * base.ref['chirp'] * clock**2)

################################################################################
# Compare the integrated-phase programme with frequency changes alone
################################################################################
results = []
for label, integrate_phase in [('Integrated phase', True),
                                ('Frequency only', False)]:
    ########################################
    # Bind the phase correction in this independent line's environment
    ########################################
    line    = base.copy()
    clock   = line.ref['t_turn_s']
    if integrate_phase:
        line['rf'].phase = (
            PHASE_START + line.ref['phase_integral']
            - 2 * np.pi * line.ref['rf'].frequency * clock)
    else:
        line['rf'].phase = PHASE_START
    particle    = line.build_particles(zeta = 0.0, delta = 0.0)

    ########################################
    # Record native time, settings and energy before each cavity passage
    ########################################
    line.track(particle, num_turns = N_TURNS, log = xt.Log(
        time        = lambda line, p: p.time_s,
        frequency   = lambda line, p: float(line['rf'].frequency),
        phase       = lambda line, p: float(line['rf'].phase),
        energy      = lambda line, p: float(p.energy[0]),
        zeta        = lambda line, p: float(p.zeta[0]),
    ))
    logged      = line.log_last_track
    time        = np.asarray(logged['time'])
    frequency   = np.asarray(logged['frequency'])
    offset      = np.asarray(logged['phase'])
    energy      = np.asarray(logged['energy'])
    kicks       = np.diff(np.r_[energy, particle.energy[0]])
    beta0       = float(line.particle_ref.beta0[0])
    arrival     = time - np.asarray(logged['zeta']) / (beta0 * c)
    ideal_phase = PHASE_START + 2 * np.pi * (
        FREQUENCY_START * arrival + 0.5 * CHIRP_RATE * arrival**2)
    # Native Cavity uses abs(q0)*charge_ratio; here that factor is +1.
    ideal_kicks = VOLTAGE * np.sin(ideal_phase)
    print('Maximum arrival/update time offset [s]:',
          np.max(np.abs(arrival - time)))
    print(label, 'maximum kick error vs continuous chirp [eV]:',
          np.max(np.abs(kicks - ideal_kicks)))
    print('Final state / fixed reference p0*c [eV]:',
          particle.state, particle.p0c)
    results.append((label, time, frequency, offset, kicks, ideal_kicks,
                    arrival, particle))

# Settings update once per reference turn. Within a turn the cavity uses a
# tangent approximation to the chirp phase, exact at its update time. The phase
# error away from that time is pi*CHIRP_RATE*(t_arrival-t_frame)^2 in magnitude.
# Here a near-reference particle samples the cavity at s=0, making it tiny.
# This timing probe has no RF equilibrium or prescribed acceleration ramp.

################################################################################
# Plot the frequency programme and the actual energy kicks
################################################################################
fig, axes = plt.subplots(3, 1, sharex = True, figsize = (9, 8))
axes[0].plot(1e6 * results[0][1], results[0][2] / 1e6)
for (label, time, frequency, offset, kicks, ideal_kicks,
     arrival, particle) in results:
    axes[1].plot(1e6 * time, kicks, label = label)
    axes[2].plot(1e6 * time, kicks - ideal_kicks, label = label)
axes[0].set_ylabel('Programmed frequency [MHz]')
axes[1].set_ylabel('Measured energy kick [eV]')
axes[2].set_ylabel('Kick error vs chirp integral [eV]')
axes[2].set_xlabel('Native reference time [us]')
for axis in axes:
    axis.grid()
for axis in axes[1:]:
    axis.legend()
fig.tight_layout()
plt.show()
