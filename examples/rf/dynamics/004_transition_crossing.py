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
# Ideal proton turn map and bounded reference-energy ramp
########################################
CIRCUMFERENCE       = 1000.0               # [m]
GAMMA_START         = 3.9
GAMMA_END           = 4.1
GAMMA_TRANSITION    = 4.0
N_TURNS             = 1024
HARMONIC            = 40
VOLTAGE             = 300e3                # RF voltage [V], proton charge +e

########################################
# Small independent longitudinal probes around an initially central particle
########################################
ZETA_START          = [0.0, 0.05, -0.05, 0.0, 0.0]  # [m]
DELTA_START         = [0.0, 0.0, 0.0, 1e-5, -1e-5]  # (p-p0)/p0

################################################################################
# Configure native acceleration and the two phase branches
################################################################################

########################################
# Balance the nominal RF energy gain with the native reference increment
########################################
reference   = xt.Particles('proton', gamma0 = GAMMA_START)
alpha_p     = 1 / GAMMA_TRANSITION**2
energy_step = xt.PROTON_MASS_EV * (GAMMA_END - GAMMA_START) / N_TURNS
phi_below   = np.arcsin(energy_step / VOLTAGE)
phi_above   = np.pi - phi_below

# Both phases give the same nominal +e*V*sin(phi) energy gain [eV].
# Local linear phase focusing requires eta*cos(phi)<0 in this map convention.

########################################
# Use a nonlinear RF map with native reference-energy updates
########################################
base    = xt.Line(
    elements        = {
        'turn.map': xt.LineSegmentMap(
            length                      = CIRCUMFERENCE,
            qx                          = 0.31,
            qy                          = 0.32,
            betx                        = 1.0,
            bety                        = 1.0,
            longitudinal_mode           = 'nonlinear',
            momentum_compaction_factor  = alpha_p,
            voltage_rf                  = VOLTAGE,
            frequency_rf                = HARMONIC * reference.beta0[0]
                                          * c / CIRCUMFERENCE,
            phase_rf                    = phi_below,
            energy_ref_increment        = energy_step,
        ),
    },
    particle_ref    = reference,
)
initial = base.build_particles(zeta = ZETA_START, delta = DELTA_START)

# This ideal turn map supplies RF kicks and momentum-compaction slip.
# It excludes transition optics, radiation, collective forces and lab timing.

################################################################################
# Compare the phase switch with retaining the below-transition phase
################################################################################
results = []
for label, switch_phase in [('Phase switch', True), ('No switch', False)]:
    ########################################
    # Prepare an independent map and native turn-by-turn recording
    ########################################
    line        = base.copy()
    particles   = initial.copy()
    monitor     = xt.ParticlesMonitor(
        start_at_turn   = 0,
        stop_at_turn    = N_TURNS,
        num_particles   = len(ZETA_START),
    )
    eta_history     = np.zeros(N_TURNS)
    phase_history   = np.zeros(N_TURNS)

    ########################################
    # Set RF controls from the actual native reference before each passage
    ########################################
    for turn in range(N_TURNS):
        eta     = alpha_p - 1 / particles.gamma0[0]**2
        phase   = phi_above if switch_phase and eta > 0 else phi_below
        line['turn.map'].phase_rf[0] = phase
        line['turn.map'].frequency_rf[0] = (
            HARMONIC * particles.beta0[0] * c / CIRCUMFERENCE)
        eta_history[turn] = eta
        phase_history[turn] = phase
        line.track(particles, num_turns = 1, turn_by_turn_monitor = monitor)

    ########################################
    # Measure probe separation from the tracked central particle
    ########################################
    # The finite-step accelerating map does not keep the initially central
    # particle exactly at zeta=delta=0. Subtract its tracked trajectory here.
    ddelta      = monitor.delta[1:] - monitor.delta[0]
    spread      = np.sqrt(np.mean(ddelta**2, axis = 0))
    crossing    = np.flatnonzero(eta_history > 0)[0]
    print(label, 'first above-transition entrance turn:', crossing)
    print('Final native reference gamma:', particles.gamma0[0])
    print('Final probe RMS zeta/delta separation [m,1]:',
          np.sqrt(np.mean((particles.zeta[1:] - particles.zeta[0])**2)),
          np.sqrt(np.mean((particles.delta[1:] - particles.delta[0])**2)))
    print('Final states:', particles.state)
    results.append((label, eta_history, phase_history, spread,
                    monitor, particles))

# The switch restores the local phase-focusing sign after eta changes sign.
# At eta=0 the linear synchrotron tune vanishes. No adiabatic capture,
# separatrix validity or machine transition-loss prediction is implied.
# energy_ref_increment supplies an element-driven ramp, not an EnergyProgram
# with a coordinated laboratory clock; results are compared by turn number.

################################################################################
# Plot the crossing, RF phase choice and finite probe response
################################################################################
turns   = np.arange(N_TURNS)
fig, axes = plt.subplots(3, 1, sharex = True, figsize = (9, 8))
axes[0].plot(turns, results[0][1])
axes[0].axhline(0, color = 'k', linestyle = ':')
for label, eta_history, phase_history, spread, monitor, particles in results:
    axes[1].plot(turns, phase_history / np.pi, label = label)
    axes[2].plot(turns, spread, label = label)
axes[0].set_ylabel('Slip factor eta')
axes[1].set_ylabel('RF phase / pi')
axes[2].set_ylabel('Probe RMS delta separation')
axes[2].set_xlabel('Turn entrance')
for axis in axes:
    axis.axvline(crossing, color = 'k', linestyle = ':')
    axis.grid()
for axis in axes[1:]:
    axis.legend()
fig.tight_layout()
plt.show()
