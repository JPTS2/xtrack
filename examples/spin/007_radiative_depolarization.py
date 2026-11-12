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
# Near-integer spin-energy choice on the shared ring
########################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
A_GAMMA         = 23.01                 # Near-integer spin sensitivity
RF_VOLTAGE      = 50e6                  # [V]
HARMONIC        = 200
SOLENOID_S      = 2.0                   # [m], existing drift
SOLENOID_LENGTH = 0.2                   # [m]
SOLENOID_KS     = 0.05                  # Bz/(B*rho) [m^-1]

########################################
# Identical closed-orbit probes and one stochastic realization
########################################
N_PARTICLES     = 32
N_TURNS         = 128
SEED            = 761

################################################################################
# Prepare a sensitive mean-orbit spin reference through native analysis
################################################################################

########################################
# Choose the reference energy from the planar spin tune
########################################
repo_root   = Path(__file__).resolve().parents[2]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env['demo']
electron    = xt.Particles('electron')
anomaly     = electron.anomalous_magnetic_moment[0]
line.set_particle_ref('electron', gamma0 = A_GAMMA / anomaly)
env['frequency_rf']  = 0.0
env.set('rf.1', harmonic = HARMONIC)
env['voltage_rf']    = RF_VOLTAGE
line.insert(
    'spin.solenoid',
    xt.UniformSolenoid(
        length  = SOLENOID_LENGTH,
        ks      = SOLENOID_KS,
    ),
    at      = SOLENOID_S,
    anchor  = 'start',
)
line.configure_spin('auto')
line.configure_radiation(model = 'mean')
line.compensate_radiation_energy_loss()

# The planar spin tune a*gamma is near an integer at about 10.139 GeV/c.
# The solenoid makes the spin sensitive to photon-driven orbital excursions.
# Compensation supplies the mean energy loss and local magnet taper.

########################################
# Initialise identical orbits with spins along the native invariant axis
########################################
tw      = line.twiss(
    method                  = '6d',
    chrom                   = False,
    polarization_analysis   = True,
)
n0      = np.array([tw.spin_x[0], tw.spin_y[0], tw.spin_z[0]])
initial = line.build_particles(
    particle_on_co  = tw.particle_on_co,
    mode            = 'shift',
    num_particles   = N_PARTICLES,
)
initial.spin_x  = n0[0]
initial.spin_y  = n0[1]
initial.spin_z  = n0[2]
print('Reference p0*c [eV], a*gamma, native fractional spin tune:',
      line.particle_ref.p0c[0], A_GAMMA, tw.spin_tune_fractional)
print('Native max |partial n / partial delta|:', np.max(np.sqrt(
    tw.spin_dn_ddelta_x**2 + tw.spin_dn_ddelta_y**2 + tw.spin_dn_ddelta_z**2)))

########################################
# Compare the analytical component time with the observation window
########################################
window_s    = N_TURNS * tw.t_rev0
depol_time  = tw.spin_t_depol_component_s
print('Native analytical depolarization-component time [s]:', depol_time)
print('Tracking window [s] / fraction of that component time:',
      window_s, window_s / depol_time)
# The analytical rate assumes a stationary orbital distribution. These cold
# closed-orbit probes instead acquire their momentum spread during tracking.
# Their transient coherence is not an estimator of that analytical time.

################################################################################
# Compare deterministic mean transport with quantum-orbit-driven spin dispersion
################################################################################
results = []
for model in ['mean', 'quantum']:
    ########################################
    # Use independent configurations and identical initial spins and orbits
    ########################################
    tracked     = line.copy()
    tracked.configure_radiation(model = model)
    particles   = initial.copy()
    # Native per-particle photon RNG startup draws its seeds from NumPy.
    np.random.seed(SEED)
    tracked.track(particles, num_turns = N_TURNS, turn_by_turn_monitor = True)

    ########################################
    # Measure ensemble coherence separately from individual spin length
    ########################################
    record          = tracked.record_last_track
    spins           = np.stack(
        [record.spin_x, record.spin_y, record.spin_z], axis = -1)
    final_spins     = np.column_stack(
        [particles.spin_x, particles.spin_y, particles.spin_z])
    spins           = np.concatenate([spins, final_spins[:, None, :]], axis = 1)
    live            = np.column_stack([record.state > 0, particles.state > 0])
    live_count      = np.sum(live, axis = 0)
    if np.any(live_count == 0):
        raise RuntimeError('No live spin probes at one or more observations.')
    # All probes have equal weight. Exclude lost particles at each observation.
    mean_spin       = np.sum(np.where(live[:, :, None], spins, 0.0), axis = 0)
    mean_spin      /= live_count[:, None]
    coherence       = np.linalg.norm(mean_spin, axis = 1)
    norm_error      = np.max(np.abs(np.linalg.norm(spins, axis = 2)[live] - 1))
    delta_history   = np.column_stack([record.delta, particles.delta])
    sigma_delta     = np.nanstd(np.where(live, delta_history, np.nan), axis = 0)
    survivors       = np.count_nonzero(particles.state > 0)
    print(model, 'surviving particles:', survivors, '/', N_PARTICLES)
    print('Final |mean spin| / maximum individual spin-length error:',
          coherence[-1], norm_error)
    print('Final rms delta:', sigma_delta[-1])
    results.append((model, coherence, sigma_delta))

# Photon recoil changes the orbits, so BMT rotations differ among particles.
# Individual spins retain unit length. This is ensemble dephasing, not spin-flip
# tracking or Sokolov-Ternov build-up. Survivor selection can also change the
# ensemble mean; inspect the reported population. No lifetime is fitted.

################################################################################
# Display the finite-window loss of polarization coherence
################################################################################
turns       = np.arange(N_TURNS + 1)
fig, axes   = plt.subplots(2, 1, figsize = (8, 7), sharex = True)
for model, coherence, sigma_delta in results:
    axes[0].plot(turns, coherence, label = model)
    axes[1].plot(turns, sigma_delta, label = model)
axes[0].set_ylabel('|mean spin vector|')
axes[1].set_ylabel('rms delta')
axes[1].set_xlabel('Turns')
for axis in axes:
    axis.grid()
    axis.legend()
fig.tight_layout()
plt.show()
