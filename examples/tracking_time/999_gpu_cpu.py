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
# Time Limit
########################################
TIME_LIMIT          = 120
N_TURNS             = int(2E1)
N_PARTICLES_INIT    = int(1)

########################################
# Line
########################################
REPO_ROOT           = Path(__file__).resolve().parents[2]
LINE_PATH           = REPO_ROOT / "test_data" / "fcc_ee" / "fccee_h_thick.json"

################################################################################
# Contexts
################################################################################
CONTEXT_CPU_SINGLE  = xo.context_default
CONTEXT_GPU_CUPY    = xo.ContextCupy()

################################################################################
# Lattice setup
################################################################################

########################################
# Load line from JSON
########################################
line = xt.load(LINE_PATH)
line.build_tracker(_context = CONTEXT_CPU_SINGLE)

########################################
# Taper Line
########################################
line.configure_radiation(model = "mean")
line.compensate_radiation_energy_loss()
line.discard_tracker()

################################################################################
# Per context line setup
################################################################################

########################################
# CPU Single Thread
########################################
line_cpu_single = line.copy()
line_cpu_single.build_tracker(_context = CONTEXT_CPU_SINGLE)

########################################
# GPU CuPy
########################################
line_gpu_cupy   = line.copy()
line_gpu_cupy.build_tracker(_context = CONTEXT_GPU_CUPY)

################################################################################
# Deal with the single thread compilation here
################################################################################
test_part = line_cpu_single.build_particles(_context    = CONTEXT_CPU_SINGLE)
line_cpu_single.track(test_part, num_turns = 1)

################################################################################
# Track quantum
################################################################################
line_cpu_single.configure_radiation(model = "quantum")
line_gpu_cupy.configure_radiation(model = "quantum")

########################################
# CPU Single Thread
########################################
print("Tracking with CPU single thread")

tracking_times_cpu_single_quantum   = []
n_particles_cpu_single_quantum      = []
time_last_track_cpu_single          = 0
iteration_cpu_single                = 0

while time_last_track_cpu_single < TIME_LIMIT:

    n_particles_track   = int(N_PARTICLES_INIT * 2**iteration_cpu_single)
    n_particles_cpu_single_quantum.append(n_particles_track)

    particles_cpu_single    = line_cpu_single.build_particles(
        _context    = CONTEXT_CPU_SINGLE,
        x           = np.zeros(n_particles_track),
        px          = np.zeros(n_particles_track),
        y           = np.zeros(n_particles_track),
        py          = np.zeros(n_particles_track),
        zeta        = np.zeros(n_particles_track),
        delta       = np.zeros(n_particles_track))
    particles_cpu_single._init_random_number_generator()

    line_cpu_single.track(
        particles       = particles_cpu_single,
        num_turns       = N_TURNS,
        time            = True)
    
    tracking_times_cpu_single_quantum.append(line_cpu_single.time_last_track)
    time_last_track_cpu_single  = line_cpu_single.time_last_track
    iteration_cpu_single        += 1

tracking_times_cpu_single_quantum       = np.array(tracking_times_cpu_single_quantum)
n_particles_cpu_single_quantum          = np.array(n_particles_cpu_single_quantum)
particle_turn_time_cpu_single_quantum   = tracking_times_cpu_single_quantum \
    / N_TURNS / n_particles_cpu_single_quantum

########################################
# GPU CuPy
########################################
print("Tracking with GPU CuPy")

tracking_times_gpu_cupy_quantum = []
n_particles_gpu_cupy_quantum    = []
time_last_track_gpu_cupy        = 0
iteration_gpu_cupy              = 0

while time_last_track_gpu_cupy < TIME_LIMIT:

    n_particles_track   = int(N_PARTICLES_INIT * 2**iteration_gpu_cupy)
    n_particles_gpu_cupy_quantum.append(n_particles_track)

    particles_gpu_cupy    = line_gpu_cupy.build_particles(
        _context    = CONTEXT_GPU_CUPY,
        x           = np.zeros(n_particles_track),
        px          = np.zeros(n_particles_track),
        y           = np.zeros(n_particles_track),
        py          = np.zeros(n_particles_track),
        zeta        = np.zeros(n_particles_track),
        delta       = np.zeros(n_particles_track))
    particles_gpu_cupy._init_random_number_generator()

    line_gpu_cupy.track(
        particles       = particles_gpu_cupy,
        num_turns       = N_TURNS,
        time            = True)
    
    tracking_times_gpu_cupy_quantum.append(line_gpu_cupy.time_last_track)
    time_last_track_gpu_cupy  = line_gpu_cupy.time_last_track
    iteration_gpu_cupy        += 1

tracking_times_gpu_cupy_quantum     = np.array(tracking_times_gpu_cupy_quantum)
n_particles_gpu_cupy_quantum        = np.array(n_particles_gpu_cupy_quantum)
particle_turn_time_gpu_cupy_quantum = tracking_times_gpu_cupy_quantum \
    / N_TURNS / n_particles_gpu_cupy_quantum

################################################################################
# Track quantum Efficient
################################################################################
line_cpu_single.configure_radiation(model = "quantum-efficient")
line_gpu_cupy.configure_radiation(model = "quantum-efficient")

########################################
# CPU Single Thread
########################################
print("Tracking with CPU single thread")

tracking_times_cpu_single_efficient = []
n_particles_cpu_single_efficient    = []
time_last_track_cpu_single          = 0
iteration_cpu_single                = 0

while time_last_track_cpu_single < TIME_LIMIT:

    n_particles_track   = int(N_PARTICLES_INIT * 2**iteration_cpu_single)
    n_particles_cpu_single_efficient.append(n_particles_track)

    particles_cpu_single    = line_cpu_single.build_particles(
        _context    = CONTEXT_CPU_SINGLE,
        x           = np.zeros(n_particles_track),
        px          = np.zeros(n_particles_track),
        y           = np.zeros(n_particles_track),
        py          = np.zeros(n_particles_track),
        zeta        = np.zeros(n_particles_track),
        delta       = np.zeros(n_particles_track))
    particles_cpu_single._init_random_number_generator()

    line_cpu_single.track(
        particles       = particles_cpu_single,
        num_turns       = N_TURNS,
        time            = True)
    
    tracking_times_cpu_single_efficient.append(line_cpu_single.time_last_track)
    time_last_track_cpu_single  = line_cpu_single.time_last_track
    iteration_cpu_single        += 1

tracking_times_cpu_single_efficient     = np.array(tracking_times_cpu_single_efficient)
n_particles_cpu_single_efficient        = np.array(n_particles_cpu_single_efficient)
particle_turn_time_cpu_single_efficient = tracking_times_cpu_single_efficient \
    / N_TURNS / n_particles_cpu_single_efficient

########################################
# GPU CuPy
########################################
print("Tracking with GPU CuPy")

tracking_times_gpu_cupy_efficient   = []
n_particles_gpu_cupy_efficient      = []
time_last_track_gpu_cupy            = 0
iteration_gpu_cupy                  = 0

while time_last_track_gpu_cupy < TIME_LIMIT:

    n_particles_track   = int(N_PARTICLES_INIT * 2**iteration_gpu_cupy)
    n_particles_gpu_cupy_efficient.append(n_particles_track)

    particles_gpu_cupy    = line_gpu_cupy.build_particles(
        _context    = CONTEXT_GPU_CUPY,
        x           = np.zeros(n_particles_track),
        px          = np.zeros(n_particles_track),
        y           = np.zeros(n_particles_track),
        py          = np.zeros(n_particles_track),
        zeta        = np.zeros(n_particles_track),
        delta       = np.zeros(n_particles_track))
    particles_gpu_cupy._init_random_number_generator()

    line_gpu_cupy.track(
        particles       = particles_gpu_cupy,
        num_turns       = N_TURNS,
        time            = True)
    
    tracking_times_gpu_cupy_efficient.append(line_gpu_cupy.time_last_track)
    time_last_track_gpu_cupy  = line_gpu_cupy.time_last_track
    iteration_gpu_cupy        += 1

tracking_times_gpu_cupy_efficient       = np.array(tracking_times_gpu_cupy_efficient)
n_particles_gpu_cupy_efficient          = np.array(n_particles_gpu_cupy_efficient)
particle_turn_time_gpu_cupy_efficient   = tracking_times_gpu_cupy_efficient \
    / N_TURNS / n_particles_gpu_cupy_efficient

################################################################################
# Plot overlayed
################################################################################
fig, ax = plt.subplots(figsize = (10, 6))

ax.plot(
    n_particles_cpu_single_quantum,
    particle_turn_time_cpu_single_quantum * 1E6,
    label   = "CPU Single Thread: Quantum",
    marker  = "o")
ax.plot(
    n_particles_gpu_cupy_quantum,
    particle_turn_time_gpu_cupy_quantum * 1E6,
    label   = "GPU CuPy: Quantum",
    marker  = "o")

ax.plot(
    n_particles_cpu_single_efficient,
    particle_turn_time_cpu_single_efficient * 1E6,
    label   = "CPU Single Thread: Quantum Efficient",
    marker  = "o")
ax.plot(
    n_particles_gpu_cupy_efficient,
    particle_turn_time_gpu_cupy_efficient * 1E6,
    label   = "GPU CuPy: Quantum Efficient",
    marker  = "o")

ax.set_xlabel("Number of Particles")
ax.set_ylabel("Mean Tracking Time per Particle per Turn [us]")

ax.set_xscale("log")
ax.set_yscale("log")

ax.legend()

fig.suptitle(f"Tracking Time Comparison (Quantum Radiation) - {N_TURNS} Turns")

fig.savefig("temp.png")

plt.show()
