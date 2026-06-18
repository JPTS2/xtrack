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
# Tracking Types
########################################
TRACK_CPU_SINGLE    = True
TRACK_CPU_OPENMP    = False
TRACK_GPU_CUPY      = False

OPEN_MP_THREADS     = 4

########################################
# Time Limit
########################################
TIME_LIMIT          = 20
N_TURNS             = int(1E2)
N_PARTICLES_INIT    = int(1E1)

########################################
# Line
########################################
REPO_ROOT           = Path(__file__).resolve().parents[2]
LINE_PATH           = REPO_ROOT / "test_data" / "fcc_ee" / "fccee_h_thick.json"

################################################################################
# Contexts
################################################################################
CONTEXT_CPU_SINGLE  = xo.context_default
if TRACK_CPU_OPENMP:
    CONTEXT_CPU_OPENMP  = xo.ContextCpu(omp_num_threads = OPEN_MP_THREADS)
if TRACK_GPU_CUPY:
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
# Per context/radiation line setup
################################################################################

########################################
# CPU Single Thread
########################################
if TRACK_CPU_SINGLE:
    print("Creating lines for CPU Single Thread")

    line_cpu_single = line.copy()
    line_cpu_single.configure_radiation(model = "quantum-efficient")
    line_cpu_single.build_tracker(_context = CONTEXT_CPU_SINGLE)

########################################
# CPU OpenMP
########################################
if TRACK_CPU_OPENMP:
    print("Creating lines for CPU OpenMP")

    line_cpu_openmp = line.copy()
    line_cpu_openmp.configure_radiation(model = "quantum-efficient")
    line_cpu_openmp.build_tracker(_context = CONTEXT_CPU_OPENMP)

########################################
# GPU CuPy
########################################
if TRACK_GPU_CUPY:
    print("Creating lines for GPU CuPy")

    line_gpu_cupy   = line.copy()
    line_gpu_cupy.configure_radiation(model = "quantum-efficient")
    line_gpu_cupy.build_tracker(_context = CONTEXT_GPU_CUPY)

################################################################################
# Track
################################################################################

########################################
# CPU Single Thread
########################################
if TRACK_CPU_SINGLE:
    print("Tracking with CPU single thread")

    tracking_times_cpu_single   = []
    n_particles_cpu_single      = []
    time_last_track_cpu_single  = 0
    iteration_cpu_single        = 0

    while time_last_track_cpu_single < TIME_LIMIT:

        n_particles_track   = int(N_PARTICLES_INIT * 2**iteration_cpu_single)
        n_particles_cpu_single.append(n_particles_track)

        particles_cpu_single    = line_cpu_single.build_particles(
            _context    = CONTEXT_CPU_SINGLE,
            x           = np.zeros(n_particles_track),
            px          = np.zeros(n_particles_track),
            y           = np.zeros(n_particles_track),
            py          = np.zeros(n_particles_track),
            zeta        = np.zeros(n_particles_track),
            delta       = np.zeros(n_particles_track))

        line_cpu_single.track(
            particles       = particles_cpu_single,
            num_turns       = N_TURNS,
            time            = True)
        
        tracking_times_cpu_single.append(line_cpu_single.time_last_track)
        time_last_track_cpu_single  = line_cpu_single.time_last_track
        iteration_cpu_single        += 1

    tracking_times_cpu_single     = np.array(tracking_times_cpu_single)
    n_particles_cpu_single        = np.array(n_particles_cpu_single)
    particle_turn_time_cpu_single = tracking_times_cpu_single \
        / N_TURNS / n_particles_cpu_single

########################################
# CPU OpenMP
########################################
if TRACK_CPU_OPENMP:
    print("Tracking with CPU OpenMP")

    tracking_times_cpu_openmp   = []
    n_particles_cpu_openmp      = []
    time_last_track_cpu_openmp  = 0
    iteration_cpu_openmp        = 0

    while time_last_track_cpu_openmp < TIME_LIMIT:

        n_particles_track   = int(N_PARTICLES_INIT * 2**iteration_cpu_openmp)
        n_particles_cpu_openmp.append(n_particles_track)

        particles_cpu_openmp    = line_cpu_openmp.build_particles(
            _context    = CONTEXT_CPU_OPENMP,
            x           = np.zeros(n_particles_track),
            px          = np.zeros(n_particles_track),
            y           = np.zeros(n_particles_track),
            py          = np.zeros(n_particles_track),
            zeta        = np.zeros(n_particles_track),
            delta       = np.zeros(n_particles_track))

        line_cpu_openmp.track(
            particles       = particles_cpu_openmp,
            num_turns       = N_TURNS,
            time            = True)
        
        tracking_times_cpu_openmp.append(line_cpu_openmp.time_last_track)
        time_last_track_cpu_openmp  = line_cpu_openmp.time_last_track
        iteration_cpu_openmp        += 1
    
    tracking_times_cpu_openmp     = np.array(tracking_times_cpu_openmp)
    n_particles_cpu_openmp        = np.array(n_particles_cpu_openmp)
    particle_turn_time_cpu_openmp = tracking_times_cpu_openmp \
        / N_TURNS / n_particles_cpu_openmp

########################################
# GPU CuPy
########################################
if TRACK_GPU_CUPY:
    print("Tracking with GPU CuPy")

    tracking_times_gpu_cupy     = []
    n_particles_gpu_cupy        = []
    time_last_track_gpu_cupy    = 0
    iteration_gpu_cupy          = 0

    while time_last_track_gpu_cupy < TIME_LIMIT:

        n_particles_track   = int(N_PARTICLES_INIT * 2**iteration_gpu_cupy)
        n_particles_gpu_cupy.append(n_particles_track)

        particles_gpu_cupy    = line_gpu_cupy.build_particles(
            _context    = CONTEXT_GPU_CUPY,
            x           = np.zeros(n_particles_track),
            px          = np.zeros(n_particles_track),
            y           = np.zeros(n_particles_track),
            py          = np.zeros(n_particles_track),
            zeta        = np.zeros(n_particles_track),
            delta       = np.zeros(n_particles_track))

        line_gpu_cupy.track(
            particles       = particles_gpu_cupy,
            num_turns       = N_TURNS,
            time            = True)
        
        tracking_times_gpu_cupy.append(line_gpu_cupy.time_last_track)
        time_last_track_gpu_cupy  = line_gpu_cupy.time_last_track
        iteration_gpu_cupy        += 1
    
    tracking_times_gpu_cupy     = np.array(tracking_times_gpu_cupy)
    n_particles_gpu_cupy        = np.array(n_particles_gpu_cupy)
    particle_turn_time_gpu_cupy = tracking_times_gpu_cupy \
        / N_TURNS / n_particles_gpu_cupy

################################################################################
# Plot overlayed
################################################################################
fig, ax = plt.subplots(figsize = (10, 6))

if TRACK_CPU_SINGLE:
    ax.plot(
        n_particles_cpu_single,
        particle_turn_time_cpu_single * 1E6,
        label   = "CPU Single Thread",
        marker  = "o")
    
if TRACK_CPU_OPENMP:
    ax.plot(
        n_particles_cpu_openmp,
        particle_turn_time_cpu_openmp * 1E6,
        label   = f"CPU OpenMP ({OPEN_MP_THREADS} Threads)",
        marker  = "o")

if TRACK_GPU_CUPY:
    ax.plot(
        n_particles_gpu_cupy,
        particle_turn_time_gpu_cupy * 1E6,
        label   = "GPU CuPy",
        marker  = "o")

ax.set_xlabel("Number of Particles")
ax.set_ylabel("Mean Tracking Time per Particle per Turn [us]")

ax.set_xscale("log")
ax.set_yscale("log")

ax.legend()

fig.suptitle(f"Tracking Time Comparison (Quantum Radiation) - {N_TURNS} Turns")

plt.show()
