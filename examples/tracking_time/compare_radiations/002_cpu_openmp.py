# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import xtrack as xt
import xobjects as xo

################################################################################
# User parameters
################################################################################

########################################
# Number of Threads
########################################
OPEN_MP_THREADS     = 8

########################################
# Tracking Types
########################################
TRACK_NONE          = True
TRACK_MEAN          = True
TRACK_QUANTUM       = True
TRACK_EFFICIENT     = True
TRACK_TABLE32       = True
TRACK_TABLE32DIRECT = True

########################################
# Time Limit
########################################
TIME_LIMIT          = 20
N_TURNS             = int(5E1)
N_PARTICLES_INIT    = int(1)

########################################
# Line
########################################
REPO_ROOT           = Path(__file__).resolve().parents[3]
ENV_PATH            = REPO_ROOT / "examples" / "fcc_ee_solenoid" / "fccee_z_lcc.json"

################################################################################
# Contexts
################################################################################
CONTEXT_CPU_SINGLE  = xo.context_default
CONTEXT_CPU_OPENMP  = xo.ContextCpu(omp_num_threads = OPEN_MP_THREADS)

################################################################################
# Lattice setup
################################################################################
print("\n" + "#"*80 + "\n" + "Loading Line" + "\n" + "#"*80 + "\n")

########################################
# Load line from JSON
########################################
env     = xt.load(ENV_PATH)
line    = env.lines["fccee_p_ring"]
line.build_tracker(_context = CONTEXT_CPU_SINGLE)

########################################
# Taper Line
########################################
line_taper  = line.copy()
line_taper.configure_radiation(model = "mean")
line_taper.compensate_radiation_energy_loss()
line_taper.discard_tracker()

################################################################################
# Per radiation line setup
################################################################################
print("\n" + "#"*80 + "\n" + "Building Lines" + "\n" + "#"*80 + "\n")

########################################
# No radiation
########################################
if TRACK_NONE:
    print("Creating line for radiation mode: None")

    line_none = line.copy()
    line_none.configure_radiation(model = None)
    line_none.build_tracker(_context = CONTEXT_CPU_OPENMP)

########################################
# Mean radiation
########################################
if TRACK_MEAN:
    print("Creating line for radiation mode: Mean")

    line_mean = line_taper.copy()
    line_mean.configure_radiation(model = "mean")
    line_mean.build_tracker(_context = CONTEXT_CPU_OPENMP)

########################################
# Quantum radiation
########################################
if TRACK_QUANTUM:
    print("Creating line for radiation mode: Quantum")

    line_quantum = line_taper.copy()
    line_quantum.configure_radiation(model = "quantum")
    line_quantum.build_tracker(_context = CONTEXT_CPU_OPENMP)

########################################
# Quantum Efficient radiation
########################################
if TRACK_EFFICIENT:
    print("Creating line for radiation mode: Quantum Efficient")

    line_efficient = line_taper.copy()
    line_efficient.configure_radiation(model = "quantum-efficient")
    line_efficient.build_tracker(_context = CONTEXT_CPU_OPENMP)

########################################
# Quantum Efficient Table32 radiation
########################################
if TRACK_TABLE32:
    print("Creating line for radiation mode: Quantum Efficient Table32")

    line_table32 = line_taper.copy()
    line_table32.configure_radiation(model = "quantum-efficient-table32")
    line_table32.build_tracker(_context = CONTEXT_CPU_OPENMP)

########################################
# Quantum Efficient Table32 Direct radiation
########################################
if TRACK_TABLE32DIRECT:
    print("Creating line for radiation mode: Quantum Efficient Table32 Direct")

    line_direct = line_taper.copy()
    line_direct.configure_radiation(model = "quantum-efficient-table32-directsearch")
    line_direct.build_tracker(_context = CONTEXT_CPU_OPENMP)

################################################################################
# Track
################################################################################
print("\n" + "#"*80 + "\n" + "Tracking" + "\n" + "#"*80 + "\n")

########################################
# No Radiation
########################################
if TRACK_NONE:
    print("#"*40 + "\n" + "Tracking with radiation mode: None" + "\n" + "#"*40)

    tracking_times_none   = []
    n_particles_none      = []
    time_last_track_none  = 0
    iteration_none        = 0

    while time_last_track_none < TIME_LIMIT:

        n_particles_track   = int(N_PARTICLES_INIT * 2**iteration_none)
        print(f"Tracking with {n_particles_track} particles...")

        particles_none    = line_none.build_particles(
            _context    = CONTEXT_CPU_OPENMP,
            x           = np.zeros(n_particles_track),
            px          = np.zeros(n_particles_track),
            y           = np.zeros(n_particles_track),
            py          = np.zeros(n_particles_track),
            zeta        = np.zeros(n_particles_track),
            delta       = np.zeros(n_particles_track))

        line_none.track(
            particles       = particles_none,
            num_turns       = N_TURNS,
            time            = True)

        assert np.all(particles_none.state == 1)

        time_last_track_none  = line_none.time_last_track
        n_particles_none.append(n_particles_track)
        tracking_times_none.append(line_none.time_last_track)
        iteration_none        += 1

    tracking_times_none     = np.array(tracking_times_none)
    n_particles_none        = np.array(n_particles_none)
    particle_turn_time_none = tracking_times_none \
        / N_TURNS / n_particles_none

########################################
# Mean Radiation
########################################
if TRACK_MEAN:
    print("#"*40 + "\n" + "Tracking with radiation mode: Mean" + "\n" + "#"*40)

    tracking_times_mean   = []
    n_particles_mean      = []
    time_last_track_mean  = 0
    iteration_mean        = 0

    while time_last_track_mean < TIME_LIMIT:

        n_particles_track   = int(N_PARTICLES_INIT * 2**iteration_mean)
        print(f"Tracking with {n_particles_track} particles...")

        particles_mean    = line_mean.build_particles(
            _context    = CONTEXT_CPU_OPENMP,
            x           = np.zeros(n_particles_track),
            px          = np.zeros(n_particles_track),
            y           = np.zeros(n_particles_track),
            py          = np.zeros(n_particles_track),
            zeta        = np.zeros(n_particles_track),
            delta       = np.zeros(n_particles_track))

        line_mean.track(
            particles       = particles_mean,
            num_turns       = N_TURNS,
            time            = True)

        assert np.all(particles_mean.state == 1)

        time_last_track_mean  = line_mean.time_last_track
        n_particles_mean.append(n_particles_track)
        tracking_times_mean.append(line_mean.time_last_track)
        iteration_mean        += 1

    tracking_times_mean     = np.array(tracking_times_mean)
    n_particles_mean        = np.array(n_particles_mean)
    particle_turn_time_mean = tracking_times_mean \
        / N_TURNS / n_particles_mean
    
########################################
# Quantum Radiation
########################################
if TRACK_QUANTUM:
    print("#"*40 + "\n" + "Tracking with radiation mode: Quantum" + "\n" + "#"*40)

    tracking_times_quantum   = []
    n_particles_quantum      = []
    time_last_track_quantum  = 0
    iteration_quantum        = 0

    while time_last_track_quantum < TIME_LIMIT:

        n_particles_track   = int(N_PARTICLES_INIT * 2**iteration_quantum)
        print(f"Tracking with {n_particles_track} particles...")

        particles_quantum    = line_quantum.build_particles(
            _context    = CONTEXT_CPU_OPENMP,
            x           = np.zeros(n_particles_track),
            px          = np.zeros(n_particles_track),
            y           = np.zeros(n_particles_track),
            py          = np.zeros(n_particles_track),
            zeta        = np.zeros(n_particles_track),
            delta       = np.zeros(n_particles_track))
        particles_quantum._init_random_number_generator()

        line_quantum.track(
            particles       = particles_quantum,
            num_turns       = N_TURNS,
            time            = True)

        assert np.all(particles_quantum.state == 1)

        time_last_track_quantum  = line_quantum.time_last_track
        n_particles_quantum.append(n_particles_track)
        tracking_times_quantum.append(line_quantum.time_last_track)
        iteration_quantum        += 1

    tracking_times_quantum     = np.array(tracking_times_quantum)
    n_particles_quantum        = np.array(n_particles_quantum)
    particle_turn_time_quantum = tracking_times_quantum \
        / N_TURNS / n_particles_quantum
    
########################################
# Quantum Efficient Radiation
########################################
if TRACK_EFFICIENT:
    print("#"*40 + "\n" + "Tracking with radiation mode: Quantum Efficient" + "\n" + "#"*40)

    tracking_times_efficient   = []
    n_particles_efficient      = []
    time_last_track_efficient  = 0
    iteration_efficient        = 0

    while time_last_track_efficient < TIME_LIMIT:

        n_particles_track   = int(N_PARTICLES_INIT * 2**iteration_efficient)
        print(f"Tracking with {n_particles_track} particles...")

        particles_efficient    = line_efficient.build_particles(
            _context    = CONTEXT_CPU_OPENMP,
            x           = np.zeros(n_particles_track),
            px          = np.zeros(n_particles_track),
            y           = np.zeros(n_particles_track),
            py          = np.zeros(n_particles_track),
            zeta        = np.zeros(n_particles_track),
            delta       = np.zeros(n_particles_track))
        particles_efficient._init_random_number_generator()

        line_efficient.track(
            particles       = particles_efficient,
            num_turns       = N_TURNS,
            time            = True)

        assert np.all(particles_efficient.state == 1)

        time_last_track_efficient  = line_efficient.time_last_track
        n_particles_efficient.append(n_particles_track)
        tracking_times_efficient.append(line_efficient.time_last_track)
        iteration_efficient        += 1

    tracking_times_efficient     = np.array(tracking_times_efficient)
    n_particles_efficient        = np.array(n_particles_efficient)
    particle_turn_time_efficient = tracking_times_efficient \
        / N_TURNS / n_particles_efficient
    
########################################
# Quantum Efficient Table 32 Radiation
########################################
if TRACK_TABLE32:
    print("#"*40 + "\n" + "Tracking with radiation mode: Quantum Efficient Table 32" + "\n" + "#"*40)

    tracking_times_table32   = []
    n_particles_table32      = []
    time_last_track_table32  = 0
    iteration_table32        = 0

    while time_last_track_table32 < TIME_LIMIT:

        n_particles_track   = int(N_PARTICLES_INIT * 2**iteration_table32)
        print(f"Tracking with {n_particles_track} particles...")

        particles_table32    = line_table32.build_particles(
            _context    = CONTEXT_CPU_OPENMP,
            x           = np.zeros(n_particles_track),
            px          = np.zeros(n_particles_track),
            y           = np.zeros(n_particles_track),
            py          = np.zeros(n_particles_track),
            zeta        = np.zeros(n_particles_track),
            delta       = np.zeros(n_particles_track))
        particles_table32._init_random_number_generator()

        line_table32.track(
            particles       = particles_table32,
            num_turns       = N_TURNS,
            time            = True)

        assert np.all(particles_table32.state == 1)

        time_last_track_table32  = line_table32.time_last_track
        n_particles_table32.append(n_particles_track)
        tracking_times_table32.append(line_table32.time_last_track)
        iteration_table32        += 1

    tracking_times_table32     = np.array(tracking_times_table32)
    n_particles_table32        = np.array(n_particles_table32)
    particle_turn_time_table32 = tracking_times_table32 \
        / N_TURNS / n_particles_table32
    
########################################
# Quantum Efficient Table 32 Direct Radiation
########################################
if TRACK_TABLE32DIRECT:
    print("#"*40 + "\n" + "Tracking with radiation mode: Quantum Efficient Table 32 Direct Search" + "\n" + "#"*40)

    tracking_times_direct   = []
    n_particles_direct      = []
    time_last_track_direct  = 0
    iteration_direct        = 0

    while time_last_track_direct < TIME_LIMIT:

        n_particles_track   = int(N_PARTICLES_INIT * 2**iteration_direct)
        print(f"Tracking with {n_particles_track} particles...")

        particles_direct    = line_direct.build_particles(
            _context    = CONTEXT_CPU_OPENMP,
            x           = np.zeros(n_particles_track),
            px          = np.zeros(n_particles_track),
            y           = np.zeros(n_particles_track),
            py          = np.zeros(n_particles_track),
            zeta        = np.zeros(n_particles_track),
            delta       = np.zeros(n_particles_track))
        particles_direct._init_random_number_generator()

        line_direct.track(
            particles       = particles_direct,
            num_turns       = N_TURNS,
            time            = True)

        assert np.all(particles_direct.state == 1)

        time_last_track_direct  = line_direct.time_last_track
        n_particles_direct.append(n_particles_track)
        tracking_times_direct.append(line_direct.time_last_track)
        iteration_direct        += 1

    tracking_times_direct     = np.array(tracking_times_direct)
    n_particles_direct        = np.array(n_particles_direct)
    particle_turn_time_direct = tracking_times_direct \
        / N_TURNS / n_particles_direct

################################################################################
# Plot overlayed
################################################################################
fig, ax = plt.subplots(figsize = (10, 6))

if TRACK_NONE:
    ax.plot(
        n_particles_none,
        particle_turn_time_none * 1E6,
        label   = "None",
        marker  = "o")

if TRACK_MEAN:
    ax.plot(
        n_particles_mean,
        particle_turn_time_mean * 1E6,
        label   = "Mean",
        marker  = "o")

if TRACK_QUANTUM:
    ax.plot(
        n_particles_quantum,
        particle_turn_time_quantum * 1E6,
        label   = "Quantum",
        marker  = "o")
    
if TRACK_EFFICIENT:
    ax.plot(
        n_particles_efficient,
        particle_turn_time_efficient * 1E6,
        label   = "Quantum Efficient",
        marker  = "o")

if TRACK_TABLE32:
    ax.plot(
        n_particles_table32,
        particle_turn_time_table32 * 1E6,
        label   = "Quantum Efficient Table 32",
        marker  = "o")

if TRACK_TABLE32DIRECT:
    ax.plot(
        n_particles_direct,
        particle_turn_time_direct * 1E6,
        label   = "Quantum Efficient Table 32 Direct",
        marker  = "o")

ax.set_xlabel("Number of Particles")
ax.set_ylabel("Mean Tracking Time per Particle per Turn [us]")

ax.set_xscale("log")
ax.set_yscale("log")

ax.legend()

fig.suptitle(f"Tracking Time Comparison (CPU OpenMP) - {N_TURNS} Turns")

plt.show()
