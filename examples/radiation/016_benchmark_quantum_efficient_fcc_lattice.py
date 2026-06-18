# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
from pathlib import Path

import numpy as np

import xtrack as xt
import xcoll as xc
import xobjects as xo

################################################################################
# User parameters
################################################################################
OPEN_MP_THREADS     = 8

N_PARTICLES         = int(1E3)
N_TURNS             = int(5E2)

# The FCC-ee H lattice is the same local test lattice used by the radiation
# equilibrium tests. It is loaded fresh for each radiation mode so that the only
# intentional difference between runs is the selected radiation model.
REPO_ROOT           = Path(__file__).resolve().parents[2]
LINE_PATH           = REPO_ROOT / "test_data" / "fcc_ee" / "fccee_h_thick.json"

################################################################################
# Contexts
################################################################################
CONTEXT_CPU_SINGLE  = xo.ContextDefault()
CONTEXT_CPU_OPENMP  = xo.ContextCpu(omp_num_threads = OPEN_MP_THREADS)
CONTEXT_GPU_CUPY    = xo.ContextCupy()

################################################################################
# Helpers
################################################################################
def to_cpu(run_context, array):
    """Copy a context array to a NumPy array on the host."""
    return run_context.nparray_from_context_array(array)

################################################################################
# Lattice setup
################################################################################

########################################
# Load line from JSON
########################################
line = xt.load(LINE_PATH)
line.build_tracker(context = CONTEXT_CPU_SINGLE)

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
# Mean radiation
########################################
line_cpu_single_mean        = line.copy()
line_cpu_openmp_mean        = line.copy()
line_gpu_cupy_mean          = line.copy()

xc.EmittanceMonitor.install(
    line_cpu_single_mean,
    name            = "emittance_monitor",
    start_at_turn   = 0,
    stop_at_turn    = N_TURNS)
xc.EmittanceMonitor.install(
    line_cpu_openmp_mean,
    name            = "emittance_monitor",
    start_at_turn   = 0,
    stop_at_turn    = N_TURNS)
xc.EmittanceMonitor.install(
    line_gpu_cupy_mean, 
    name            = "emittance_monitor",
    start_at_turn   = 0,
    stop_at_turn    = N_TURNS)

line_cpu_single_mean.configure_radiation(model = "mean")
line_cpu_openmp_mean.configure_radiation(model = "mean")
line_gpu_cupy_mean.configure_radiation(model = "mean")

line_cpu_single_mean.build_tracker(_context = CONTEXT_CPU_SINGLE)
line_cpu_openmp_mean.build_tracker(_context = CONTEXT_CPU_OPENMP)
line_gpu_cupy_mean.build_tracker(_context = CONTEXT_GPU_CUPY)

########################################
# Quantum radiation
########################################
line_cpu_single_quantum     = line.copy()
line_cpu_openmp_quantum     = line.copy()
line_gpu_cupy_quantum       = line.copy()

xc.EmittanceMonitor.install(
    line_cpu_single_quantum,
    name            = "emittance_monitor",
    start_at_turn   = 0,
    stop_at_turn    = N_TURNS)
xc.EmittanceMonitor.install(
    line_cpu_openmp_quantum,
    name            = "emittance_monitor",
    start_at_turn   = 0,
    stop_at_turn    = N_TURNS)
xc.EmittanceMonitor.install(
    line_gpu_cupy_quantum,
    name            = "emittance_monitor",
    start_at_turn   = 0,
    stop_at_turn    = N_TURNS)

line_cpu_single_quantum.configure_radiation(model = "quantum")
line_cpu_openmp_quantum.configure_radiation(model = "quantum")
line_gpu_cupy_quantum.configure_radiation(model = "quantum")

line_cpu_single_quantum.build_tracker(_context = CONTEXT_CPU_SINGLE)
line_cpu_openmp_quantum.build_tracker(_context = CONTEXT_CPU_OPENMP)
line_gpu_cupy_quantum.build_tracker(_context = CONTEXT_GPU_CUPY)

########################################
# Quantum Efficient radiation
########################################
line_cpu_single_efficient   = line.copy()
line_cpu_openmp_efficient   = line.copy()
line_gpu_cupy_efficient     = line.copy()

xc.EmittanceMonitor.install(
    line_cpu_single_efficient,
    name            = "emittance_monitor",
    start_at_turn   = 0,
    stop_at_turn    = N_TURNS)
xc.EmittanceMonitor.install(
    line_cpu_openmp_efficient,
    name            = "emittance_monitor",
    start_at_turn   = 0,
    stop_at_turn    = N_TURNS)
xc.EmittanceMonitor.install(
    line_gpu_cupy_efficient,
    name            = "emittance_monitor",
    start_at_turn   = 0,
    stop_at_turn    = N_TURNS)

line_cpu_single_efficient.configure_radiation(model = "quantum-efficient")
line_cpu_openmp_efficient.configure_radiation(model = "quantum-efficient")
line_gpu_cupy_efficient.configure_radiation(model = "quantum-efficient")

line_cpu_single_efficient.build_tracker(_context = CONTEXT_CPU_SINGLE)
line_cpu_openmp_efficient.build_tracker(_context = CONTEXT_CPU_OPENMP)
line_gpu_cupy_efficient.build_tracker(_context = CONTEXT_GPU_CUPY)

################################################################################
# Per context/radiation particles setup
################################################################################

########################################
# Mean radiation
########################################
particles_cpu_single_mean   = line_cpu_single_mean.build_particles(
        _context    = CONTEXT_CPU_SINGLE,
        x           = np.zeros(N_PARTICLES),
        px          = np.zeros(N_PARTICLES),
        y           = np.zeros(N_PARTICLES),
        py          = np.zeros(N_PARTICLES),
        zeta        = np.zeros(N_PARTICLES),
        delta       = np.zeros(N_PARTICLES))
particles_cpu_openmp_mean   = line_cpu_openmp_mean.build_particles(
        _context    = CONTEXT_CPU_OPENMP,
        x           = np.zeros(N_PARTICLES),
        px          = np.zeros(N_PARTICLES),
        y           = np.zeros(N_PARTICLES),
        py          = np.zeros(N_PARTICLES),
        zeta        = np.zeros(N_PARTICLES),
        delta       = np.zeros(N_PARTICLES))
particles_gpu_cupy_mean     = line_gpu_cupy_mean.build_particles(
        _context    = CONTEXT_GPU_CUPY,
        x           = np.zeros(N_PARTICLES),
        px          = np.zeros(N_PARTICLES),
        y           = np.zeros(N_PARTICLES),
        py          = np.zeros(N_PARTICLES),
        zeta        = np.zeros(N_PARTICLES),
        delta       = np.zeros(N_PARTICLES))

particles_cpu_single_mean._init_random_number_generator()
particles_cpu_openmp_mean._init_random_number_generator()
particles_gpu_cupy_mean._init_random_number_generator()

########################################
# Quantum radiation
########################################
particles_cpu_single_quantum    = line_cpu_single_quantum.build_particles(
        _context    = CONTEXT_CPU_SINGLE,
        x           = np.zeros(N_PARTICLES),
        px          = np.zeros(N_PARTICLES),
        y           = np.zeros(N_PARTICLES),
        py          = np.zeros(N_PARTICLES),
        zeta        = np.zeros(N_PARTICLES),
        delta       = np.zeros(N_PARTICLES))
particles_cpu_openmp_quantum    = line_cpu_openmp_quantum.build_particles(
        _context    = CONTEXT_CPU_OPENMP,
        x           = np.zeros(N_PARTICLES),
        px          = np.zeros(N_PARTICLES),
        y           = np.zeros(N_PARTICLES),
        py          = np.zeros(N_PARTICLES),
        zeta        = np.zeros(N_PARTICLES),
        delta       = np.zeros(N_PARTICLES))
particles_gpu_cupy_quantum  = line_gpu_cupy_quantum.build_particles(
        _context    = CONTEXT_GPU_CUPY,
        x           = np.zeros(N_PARTICLES),
        px          = np.zeros(N_PARTICLES),
        y           = np.zeros(N_PARTICLES),
        py          = np.zeros(N_PARTICLES),
        zeta        = np.zeros(N_PARTICLES),
        delta       = np.zeros(N_PARTICLES))

particles_cpu_single_quantum._init_random_number_generator()
particles_cpu_openmp_quantum._init_random_number_generator()
particles_gpu_cupy_quantum._init_random_number_generator()

########################################
# Efficient radiation
########################################
particles_cpu_single_efficient  = line_cpu_single_efficient.build_particles(
        _context    = CONTEXT_CPU_SINGLE,
        x           = np.zeros(N_PARTICLES),
        px          = np.zeros(N_PARTICLES),
        y           = np.zeros(N_PARTICLES),
        py          = np.zeros(N_PARTICLES),
        zeta        = np.zeros(N_PARTICLES),
        delta       = np.zeros(N_PARTICLES))
particles_cpu_openmp_efficient  = line_cpu_openmp_efficient.build_particles(
        _context    = CONTEXT_CPU_OPENMP,
        x           = np.zeros(N_PARTICLES),
        px          = np.zeros(N_PARTICLES),
        y           = np.zeros(N_PARTICLES),
        py          = np.zeros(N_PARTICLES),
        zeta        = np.zeros(N_PARTICLES),
        delta       = np.zeros(N_PARTICLES))
particles_gpu_cupy_efficient    = line_gpu_cupy_efficient.build_particles(
        _context    = CONTEXT_GPU_CUPY,
        x           = np.zeros(N_PARTICLES),
        px          = np.zeros(N_PARTICLES),
        y           = np.zeros(N_PARTICLES),
        py          = np.zeros(N_PARTICLES),
        zeta        = np.zeros(N_PARTICLES),
        delta       = np.zeros(N_PARTICLES))

particles_cpu_single_efficient._init_random_number_generator()
particles_cpu_openmp_efficient._init_random_number_generator()
particles_gpu_cupy_efficient._init_random_number_generator()

################################################################################
# Track
################################################################################

########################################
# Mean radiation
########################################
print("Tracking with mean radiation model")

line_cpu_single_mean.track(
    particles       = particles_cpu_single_mean,
    num_turns       = N_TURNS,
    time            = True,
    with_progress   = True)
line_cpu_openmp_mean.track(
    particles       = particles_cpu_openmp_mean,
    num_turns       = N_TURNS,
    time            = True,
    with_progress   = True)
line_gpu_cupy_mean.track(
    particles       = particles_gpu_cupy_mean,
    num_turns       = N_TURNS,
    time            = True,
    with_progress   = True)

print("Tracking times for mean radiation model:")
print(f"CPU single: {line_cpu_single_mean.time_last_track:.2f} s")
print(f"CPU openmp: {line_cpu_openmp_mean.time_last_track:.2f} s")
print(f"GPU cupy: {line_gpu_cupy_mean.time_last_track:.2f} s")

########################################
# Quantum radiation
########################################
print("Tracking with quantum radiation model")

line_cpu_single_quantum.track(
    particles       = particles_cpu_single_quantum,
    num_turns       = N_TURNS,
    time            = True,
    with_progress   = True)
line_cpu_openmp_quantum.track(
    particles       = particles_cpu_openmp_quantum,
    num_turns       = N_TURNS,
    time            = True,
    with_progress   = True)
line_gpu_cupy_quantum.track(
    particles       = particles_gpu_cupy_quantum,
    num_turns       = N_TURNS,
    time            = True,
    with_progress   = True)

print("Tracking times for quantum radiation model:")
print(f"CPU single: {line_cpu_single_quantum.time_last_track:.2f} s")
print(f"CPU openmp: {line_cpu_openmp_quantum.time_last_track:.2f} s")
print(f"GPU cupy: {line_gpu_cupy_quantum.time_last_track:.2f} s")

########################################
# Quantum-efficient radiation
########################################
print("Tracking with quantum-efficient radiation model")

line_cpu_single_efficient.track(
    particles       = particles_cpu_single_efficient,
    num_turns       = N_TURNS,
    time            = True,
    with_progress   = True)
line_cpu_openmp_efficient.track(
    particles       = particles_cpu_openmp_efficient,
    num_turns       = N_TURNS,
    time            = True,
    with_progress   = True)
line_gpu_cupy_efficient.track(
    particles       = particles_gpu_cupy_efficient,
    num_turns       = N_TURNS,
    time            = True,
    with_progress   = True)

print("Tracking times for quantum efficient radiation model:")
print(f"CPU single: {line_cpu_single_efficient.time_last_track:.2f} s")
print(f"CPU openmp: {line_cpu_openmp_efficient.time_last_track:.2f} s")
print(f"GPU cupy: {line_gpu_cupy_efficient.time_last_track:.2f} s")
