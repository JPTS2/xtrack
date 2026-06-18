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
OPEN_MP_THREADS     = 2

N_PARTICLES_CPU     = int(5E1)
N_PARTICLES_GPU     = int(5E4)
N_TURNS             = int(1E2)

MONITOR_POINT       = "ca1.1"

# The FCC-ee H lattice is the same local test lattice used by the radiation
# equilibrium tests. It is loaded fresh for each radiation mode so that the only
# intentional difference between runs is the selected radiation model.
REPO_ROOT           = Path(__file__).resolve().parents[2]
LINE_PATH           = REPO_ROOT / "test_data" / "fcc_ee" / "fccee_h_thick.json"

TRACK_CPU_SINGLE    = False
TRACK_CPU_OPENMP    = False
TRACK_GPU_CUPY      = True

MONITOR_EMITTANCE   = False

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
line_taper  = line.copy()
line_taper.configure_radiation(model = "mean")
line_taper.compensate_radiation_energy_loss()
line_taper.discard_tracker()

################################################################################
# Per context/radiation line setup
################################################################################

########################################
# CPU Single Thread
########################################
if TRACK_CPU_SINGLE:
    print("Creating lines for CPU Single Thread")

    line_cpu_single_none        = line.copy()
    line_cpu_single_mean        = line_taper.copy()
    line_cpu_single_quantum     = line_taper.copy()
    line_cpu_single_efficient   = line_taper.copy()
    line_cpu_single_table32     = line_taper.copy()
    line_cpu_single_direct      = line_taper.copy()

    if MONITOR_EMITTANCE:
        moni_cpu_single_none        = xc.EmittanceMonitor.install(
            line                = line_cpu_single_none,
            name                = "emit_moni",
            at                  = MONITOR_POINT,
            stop_at_turn        = N_TURNS,
            suppress_warnings   = True)
        moni_cpu_single_mean        = xc.EmittanceMonitor.install(
            line                = line_cpu_single_mean,
            name                = "emit_moni",
            at                  = MONITOR_POINT,
            stop_at_turn        = N_TURNS,
            suppress_warnings   = True)
        moni_cpu_single_quantum     = xc.EmittanceMonitor.install(
            line                = line_cpu_single_quantum,
            name                = "emit_moni",
            at                  = MONITOR_POINT,
            stop_at_turn        = N_TURNS,
            suppress_warnings   = True)
        moni_cpu_single_efficient   = xc.EmittanceMonitor.install(
            line                = line_cpu_single_efficient,
            name                = "emit_moni",
            at                  = MONITOR_POINT,
            stop_at_turn        = N_TURNS,
            suppress_warnings   = True)
        moni_cpu_single_table32     = xc.EmittanceMonitor.install(
            line                = line_cpu_single_table32,
            name                = "emit_moni",
            at                  = MONITOR_POINT,
            stop_at_turn        = N_TURNS,
            suppress_warnings   = True)
        moni_cpu_single_direct      = xc.EmittanceMonitor.install(
            line                = line_cpu_single_direct,
            name                = "emit_moni",
            at                  = MONITOR_POINT,
            stop_at_turn        = N_TURNS,
            suppress_warnings   = True)

    line_cpu_single_none.configure_radiation(model = None)
    line_cpu_single_mean.configure_radiation(model = "mean")
    line_cpu_single_quantum.configure_radiation(model = "quantum")
    line_cpu_single_efficient.configure_radiation(model = "quantum-efficient")
    line_cpu_single_table32.configure_radiation(model = "quantum-efficient-table32")
    line_cpu_single_direct.configure_radiation(model = "quantum-efficient-table32-directsearch")

    line_cpu_single_none.build_tracker(_context = CONTEXT_CPU_SINGLE)
    line_cpu_single_mean.build_tracker(_context = CONTEXT_CPU_SINGLE)
    line_cpu_single_quantum.build_tracker(_context = CONTEXT_CPU_SINGLE)
    line_cpu_single_efficient.build_tracker(_context = CONTEXT_CPU_SINGLE)
    line_cpu_single_table32.build_tracker(_context = CONTEXT_CPU_SINGLE)
    line_cpu_single_direct.build_tracker(_context = CONTEXT_CPU_SINGLE)

########################################
# CPU OpenMP
########################################
if TRACK_CPU_OPENMP:
    print("Creating lines for CPU OpenMP")

    line_cpu_openmp_none        = line.copy()
    line_cpu_openmp_mean        = line_taper.copy()
    line_cpu_openmp_quantum     = line_taper.copy()
    line_cpu_openmp_efficient   = line_taper.copy()
    line_cpu_openmp_table32     = line_taper.copy()
    line_cpu_openmp_direct      = line_taper.copy()

    if MONITOR_EMITTANCE:
        moni_cpu_openmp_none        = xc.EmittanceMonitor.install(
            line                = line_cpu_openmp_none,
            name                = "emit_moni",
            at                  = MONITOR_POINT,
            stop_at_turn        = N_TURNS,
            suppress_warnings   = True)
        moni_cpu_openmp_mean        = xc.EmittanceMonitor.install(
            line                = line_cpu_openmp_mean,
            name                = "emit_moni",
            at                  = MONITOR_POINT,
            stop_at_turn        = N_TURNS,
            suppress_warnings   = True)
        moni_cpu_openmp_quantum     = xc.EmittanceMonitor.install(
            line                = line_cpu_openmp_quantum,
            name                = "emit_moni",
            at                  = MONITOR_POINT,
            stop_at_turn        = N_TURNS,
            suppress_warnings   = True)
        moni_cpu_openmp_efficient   = xc.EmittanceMonitor.install(
            line                = line_cpu_openmp_efficient,
            name                = "emit_moni",
            at                  = MONITOR_POINT,
            stop_at_turn        = N_TURNS,
            suppress_warnings   = True)
        moni_cpu_openmp_table32     = xc.EmittanceMonitor.install(
            line                = line_cpu_openmp_table32,
            name                = "emit_moni",
            at                  = MONITOR_POINT,
            stop_at_turn        = N_TURNS,
            suppress_warnings   = True)
        moni_cpu_openmp_direct      = xc.EmittanceMonitor.install(
            line                = line_cpu_openmp_direct,
            name                = "emit_moni",
            at                  = MONITOR_POINT,
            stop_at_turn        = N_TURNS,
            suppress_warnings   = True)

    line_cpu_openmp_none.configure_radiation(model = None)
    line_cpu_openmp_mean.configure_radiation(model = "mean")
    line_cpu_openmp_quantum.configure_radiation(model = "quantum")
    line_cpu_openmp_efficient.configure_radiation(model = "quantum-efficient")
    line_cpu_openmp_table32.configure_radiation(model = "quantum-efficient-table32")
    line_cpu_openmp_direct.configure_radiation(model = "quantum-efficient-table32-directsearch")

    line_cpu_openmp_none.build_tracker(_context = CONTEXT_CPU_OPENMP)
    line_cpu_openmp_mean.build_tracker(_context = CONTEXT_CPU_OPENMP)
    line_cpu_openmp_quantum.build_tracker(_context = CONTEXT_CPU_OPENMP)
    line_cpu_openmp_efficient.build_tracker(_context = CONTEXT_CPU_OPENMP)
    line_cpu_openmp_table32.build_tracker(_context = CONTEXT_CPU_OPENMP)
    line_cpu_openmp_direct.build_tracker(_context = CONTEXT_CPU_OPENMP)

########################################
# GPU CuPy
########################################
if TRACK_GPU_CUPY:
    print("Creating lines for GPU CuPy")

    line_gpu_cupy_none        = line.copy()
    line_gpu_cupy_mean        = line_taper.copy()
    line_gpu_cupy_quantum     = line_taper.copy()
    line_gpu_cupy_efficient   = line_taper.copy()
    line_gpu_cupy_table32     = line_taper.copy()
    line_gpu_cupy_direct      = line_taper.copy()

    if MONITOR_EMITTANCE:
        moni_gpu_cupy_none        = xc.EmittanceMonitor.install(
            line                = line_gpu_cupy_none,
            name                = "emit_moni",
            at                  = MONITOR_POINT,
            stop_at_turn        = N_TURNS,
            suppress_warnings   = True)
        moni_gpu_cupy_mean        = xc.EmittanceMonitor.install(
            line                = line_gpu_cupy_mean,
            name                = "emit_moni",
            at                  = MONITOR_POINT,
            stop_at_turn        = N_TURNS,
            suppress_warnings   = True)
        moni_gpu_cupy_quantum     = xc.EmittanceMonitor.install(
            line                = line_gpu_cupy_quantum,
            name                = "emit_moni",
            at                  = MONITOR_POINT,
            stop_at_turn        = N_TURNS,
            suppress_warnings   = True)
        moni_gpu_cupy_efficient   = xc.EmittanceMonitor.install(
            line                = line_gpu_cupy_efficient,
            name                = "emit_moni",
            at                  = MONITOR_POINT,
            stop_at_turn        = N_TURNS,
            suppress_warnings   = True)
        moni_gpu_cupy_table32     = xc.EmittanceMonitor.install(
            line                = line_gpu_cupy_table32,
            name                = "emit_moni",
            at                  = MONITOR_POINT,
            stop_at_turn        = N_TURNS,
            suppress_warnings   = True)
        moni_gpu_cupy_direct      = xc.EmittanceMonitor.install(
            line                = line_gpu_cupy_direct,
            name                = "emit_moni",
            at                  = MONITOR_POINT,
            stop_at_turn        = N_TURNS,
            suppress_warnings   = True)

    line_gpu_cupy_none.configure_radiation(model = None)
    line_gpu_cupy_mean.configure_radiation(model = "mean")
    line_gpu_cupy_quantum.configure_radiation(model = "quantum")
    line_gpu_cupy_efficient.configure_radiation(model = "quantum-efficient")
    line_gpu_cupy_table32.configure_radiation(model = "quantum-efficient-table32")
    line_gpu_cupy_direct.configure_radiation(model = "quantum-efficient-table32-directsearch")

    line_gpu_cupy_none.build_tracker(_context = CONTEXT_GPU_CUPY)
    line_gpu_cupy_mean.build_tracker(_context = CONTEXT_GPU_CUPY)
    line_gpu_cupy_quantum.build_tracker(_context = CONTEXT_GPU_CUPY)
    line_gpu_cupy_efficient.build_tracker(_context = CONTEXT_GPU_CUPY)
    line_gpu_cupy_table32.build_tracker(_context = CONTEXT_GPU_CUPY)
    line_gpu_cupy_direct.build_tracker(_context = CONTEXT_GPU_CUPY)

################################################################################
# Per context/radiation particles setup
################################################################################

########################################
# CPU Single Thread
########################################
if TRACK_CPU_SINGLE:
    print("Building particles for CPU Single Thread")

    particles_cpu_single_none   = line_cpu_single_none.build_particles(
        _context    = CONTEXT_CPU_SINGLE,
        x           = np.zeros(N_PARTICLES_CPU),
        px          = np.zeros(N_PARTICLES_CPU),
        y           = np.zeros(N_PARTICLES_CPU),
        py          = np.zeros(N_PARTICLES_CPU),
        zeta        = np.zeros(N_PARTICLES_CPU),
        delta       = np.zeros(N_PARTICLES_CPU))
    particles_cpu_single_mean   = line_cpu_single_mean.build_particles(
        _context    = CONTEXT_CPU_SINGLE,
        x           = np.zeros(N_PARTICLES_CPU),
        px          = np.zeros(N_PARTICLES_CPU),
        y           = np.zeros(N_PARTICLES_CPU),
        py          = np.zeros(N_PARTICLES_CPU),
        zeta        = np.zeros(N_PARTICLES_CPU),
        delta       = np.zeros(N_PARTICLES_CPU))
    particles_cpu_single_quantum    = line_cpu_single_quantum.build_particles(
        _context    = CONTEXT_CPU_SINGLE,
        x           = np.zeros(N_PARTICLES_CPU),
        px          = np.zeros(N_PARTICLES_CPU),
        y           = np.zeros(N_PARTICLES_CPU),
        py          = np.zeros(N_PARTICLES_CPU),
        zeta        = np.zeros(N_PARTICLES_CPU),
        delta       = np.zeros(N_PARTICLES_CPU))
    particles_cpu_single_efficient  = line_cpu_single_efficient.build_particles(
        _context    = CONTEXT_CPU_SINGLE,
        x           = np.zeros(N_PARTICLES_CPU),
        px          = np.zeros(N_PARTICLES_CPU),
        y           = np.zeros(N_PARTICLES_CPU),
        py          = np.zeros(N_PARTICLES_CPU),
        zeta        = np.zeros(N_PARTICLES_CPU),
        delta       = np.zeros(N_PARTICLES_CPU))
    particles_cpu_single_table32  = line_cpu_single_table32.build_particles(
        _context    = CONTEXT_CPU_SINGLE,
        x           = np.zeros(N_PARTICLES_CPU),
        px          = np.zeros(N_PARTICLES_CPU),
        y           = np.zeros(N_PARTICLES_CPU),
        py          = np.zeros(N_PARTICLES_CPU),
        zeta        = np.zeros(N_PARTICLES_CPU),
        delta       = np.zeros(N_PARTICLES_CPU))
    particles_cpu_single_direct   = line_cpu_single_direct.build_particles(
        _context    = CONTEXT_CPU_SINGLE,
        x           = np.zeros(N_PARTICLES_CPU),
        px          = np.zeros(N_PARTICLES_CPU),
        y           = np.zeros(N_PARTICLES_CPU),
        py          = np.zeros(N_PARTICLES_CPU),
        zeta        = np.zeros(N_PARTICLES_CPU),
        delta       = np.zeros(N_PARTICLES_CPU))

    particles_cpu_single_none._init_random_number_generator()
    particles_cpu_single_mean._init_random_number_generator()
    particles_cpu_single_quantum._init_random_number_generator()
    particles_cpu_single_efficient._init_random_number_generator()
    particles_cpu_single_table32._init_random_number_generator()
    particles_cpu_single_direct._init_random_number_generator()

########################################
# CPU OpenMP
########################################
if TRACK_CPU_OPENMP:
    print("Building particles for CPU OpenMP")

    particles_cpu_openmp_none   = line_cpu_openmp_none.build_particles(
        _context    = CONTEXT_CPU_OPENMP,
        x           = np.zeros(N_PARTICLES_CPU),
        px          = np.zeros(N_PARTICLES_CPU),
        y           = np.zeros(N_PARTICLES_CPU),
        py          = np.zeros(N_PARTICLES_CPU),
        zeta        = np.zeros(N_PARTICLES_CPU),
        delta       = np.zeros(N_PARTICLES_CPU))
    particles_cpu_openmp_mean   = line_cpu_openmp_mean.build_particles(
        _context    = CONTEXT_CPU_OPENMP,
        x           = np.zeros(N_PARTICLES_CPU),
        px          = np.zeros(N_PARTICLES_CPU),
        y           = np.zeros(N_PARTICLES_CPU),
        py          = np.zeros(N_PARTICLES_CPU),
        zeta        = np.zeros(N_PARTICLES_CPU),
        delta       = np.zeros(N_PARTICLES_CPU))
    particles_cpu_openmp_quantum    = line_cpu_openmp_quantum.build_particles(
        _context    = CONTEXT_CPU_OPENMP,
        x           = np.zeros(N_PARTICLES_CPU),
        px          = np.zeros(N_PARTICLES_CPU),
        y           = np.zeros(N_PARTICLES_CPU),
        py          = np.zeros(N_PARTICLES_CPU),
        zeta        = np.zeros(N_PARTICLES_CPU),
        delta       = np.zeros(N_PARTICLES_CPU))
    particles_cpu_openmp_efficient    = line_cpu_openmp_efficient.build_particles(
        _context    = CONTEXT_CPU_OPENMP,
        x           = np.zeros(N_PARTICLES_CPU),
        px          = np.zeros(N_PARTICLES_CPU),
        y           = np.zeros(N_PARTICLES_CPU),
        py          = np.zeros(N_PARTICLES_CPU),
        zeta        = np.zeros(N_PARTICLES_CPU),
        delta       = np.zeros(N_PARTICLES_CPU))
    particles_cpu_openmp_table32    = line_cpu_openmp_table32.build_particles(
        _context    = CONTEXT_CPU_OPENMP,
        x           = np.zeros(N_PARTICLES_CPU),
        px          = np.zeros(N_PARTICLES_CPU),
        y           = np.zeros(N_PARTICLES_CPU),
        py          = np.zeros(N_PARTICLES_CPU),
        zeta        = np.zeros(N_PARTICLES_CPU),
        delta       = np.zeros(N_PARTICLES_CPU))
    particles_cpu_openmp_direct     = line_cpu_openmp_direct.build_particles(
        _context    = CONTEXT_CPU_OPENMP,
        x           = np.zeros(N_PARTICLES_CPU),
        px          = np.zeros(N_PARTICLES_CPU),
        y           = np.zeros(N_PARTICLES_CPU),
        py          = np.zeros(N_PARTICLES_CPU),
        zeta        = np.zeros(N_PARTICLES_CPU),
        delta       = np.zeros(N_PARTICLES_CPU))

    particles_cpu_openmp_none._init_random_number_generator()
    particles_cpu_openmp_mean._init_random_number_generator()
    particles_cpu_openmp_quantum._init_random_number_generator()
    particles_cpu_openmp_efficient._init_random_number_generator()
    particles_cpu_openmp_table32._init_random_number_generator()
    particles_cpu_openmp_direct._init_random_number_generator()

########################################
# GPU CuPy
########################################
if TRACK_GPU_CUPY:
    print("Building particles for GPU CuPy")

    particles_gpu_cupy_none     = line_gpu_cupy_none.build_particles(
        _context    = CONTEXT_GPU_CUPY,
        x           = np.zeros(N_PARTICLES_GPU),
        px          = np.zeros(N_PARTICLES_GPU),
        y           = np.zeros(N_PARTICLES_GPU),
        py          = np.zeros(N_PARTICLES_GPU),
        zeta        = np.zeros(N_PARTICLES_GPU),
        delta       = np.zeros(N_PARTICLES_GPU))
    particles_gpu_cupy_mean     = line_gpu_cupy_mean.build_particles(
        _context    = CONTEXT_GPU_CUPY,
        x           = np.zeros(N_PARTICLES_GPU),
        px          = np.zeros(N_PARTICLES_GPU),
        y           = np.zeros(N_PARTICLES_GPU),
        py          = np.zeros(N_PARTICLES_GPU),
        zeta        = np.zeros(N_PARTICLES_GPU),
        delta       = np.zeros(N_PARTICLES_GPU))
    particles_gpu_cupy_quantum    = line_gpu_cupy_quantum.build_particles(
        _context    = CONTEXT_GPU_CUPY,
        x           = np.zeros(N_PARTICLES_GPU),
        px          = np.zeros(N_PARTICLES_GPU),
        y           = np.zeros(N_PARTICLES_GPU),
        py          = np.zeros(N_PARTICLES_GPU),
        zeta        = np.zeros(N_PARTICLES_GPU),
        delta       = np.zeros(N_PARTICLES_GPU))
    particles_gpu_cupy_efficient    = line_gpu_cupy_efficient.build_particles(
        _context    = CONTEXT_GPU_CUPY,
        x           = np.zeros(N_PARTICLES_GPU),
        px          = np.zeros(N_PARTICLES_GPU),
        y           = np.zeros(N_PARTICLES_GPU),
        py          = np.zeros(N_PARTICLES_GPU),
        zeta        = np.zeros(N_PARTICLES_GPU),
        delta       = np.zeros(N_PARTICLES_GPU))
    particles_gpu_cupy_table32    = line_gpu_cupy_table32.build_particles(
        _context    = CONTEXT_GPU_CUPY,
        x           = np.zeros(N_PARTICLES_GPU),
        px          = np.zeros(N_PARTICLES_GPU),
        y           = np.zeros(N_PARTICLES_GPU),
        py          = np.zeros(N_PARTICLES_GPU),
        zeta        = np.zeros(N_PARTICLES_GPU),
        delta       = np.zeros(N_PARTICLES_GPU))
    particles_gpu_cupy_direct     = line_gpu_cupy_direct.build_particles(
        _context    = CONTEXT_GPU_CUPY,
        x           = np.zeros(N_PARTICLES_GPU),
        px          = np.zeros(N_PARTICLES_GPU),
        y           = np.zeros(N_PARTICLES_GPU),
        py          = np.zeros(N_PARTICLES_GPU),
        zeta        = np.zeros(N_PARTICLES_GPU),
        delta       = np.zeros(N_PARTICLES_GPU))

    particles_gpu_cupy_none._init_random_number_generator()
    particles_gpu_cupy_mean._init_random_number_generator()
    particles_gpu_cupy_quantum._init_random_number_generator()
    particles_gpu_cupy_efficient._init_random_number_generator()
    particles_gpu_cupy_table32._init_random_number_generator()
    particles_gpu_cupy_direct._init_random_number_generator()

################################################################################
# Track
################################################################################

########################################
# CPU Single Thread
########################################
if TRACK_CPU_SINGLE:
    print("Tracking with CPU single thread")

    line_cpu_single_none.track(
        particles       = particles_cpu_single_none,
        num_turns       = N_TURNS,
        time            = True,
        with_progress   = 10)
    line_cpu_single_mean.track(
        particles       = particles_cpu_single_mean,
        num_turns       = N_TURNS,
        time            = True,
        with_progress   = 10)
    line_cpu_single_quantum.track(
        particles       = particles_cpu_single_quantum,
        num_turns       = N_TURNS,
        time            = True,
        with_progress   = 10)
    line_cpu_single_efficient.track(
        particles       = particles_cpu_single_efficient,
        num_turns       = N_TURNS,
        time            = True,
        with_progress   = 10)
    line_cpu_single_table32.track(
        particles       = particles_cpu_single_table32,
        num_turns       = N_TURNS,
        time            = True,
        with_progress   = 10)
    line_cpu_single_direct.track(
        particles       = particles_cpu_single_direct,
        num_turns       = N_TURNS,
        time            = True,
        with_progress   = 10)

    particle_turn_cpu_single_none       = line_cpu_single_none.time_last_track \
        / (N_PARTICLES_CPU * N_TURNS)
    particle_turn_cpu_single_mean       = line_cpu_single_mean.time_last_track \
        / (N_PARTICLES_CPU * N_TURNS)
    particle_turn_cpu_single_quantum    = line_cpu_single_quantum.time_last_track \
        / (N_PARTICLES_CPU * N_TURNS)
    particle_turn_cpu_single_efficient  = line_cpu_single_efficient.time_last_track \
        / (N_PARTICLES_CPU * N_TURNS)
    particle_turn_cpu_single_table32    = line_cpu_single_table32.time_last_track \
        / (N_PARTICLES_CPU * N_TURNS)
    particle_turn_cpu_single_direct     = line_cpu_single_direct.time_last_track \
        / (N_PARTICLES_CPU * N_TURNS)

    print("Tracking times for CPU single thread:")
    print(f"None: {particle_turn_cpu_single_none * 1E6:.2f} us per particle-turn")
    print(f"Mean: {particle_turn_cpu_single_mean * 1E6:.2f} us per particle-turn")
    print(f"Quantum: {particle_turn_cpu_single_quantum * 1E6:.2f} us per particle-turn")
    print(f"Quantum-efficient: {particle_turn_cpu_single_efficient * 1E6:.2f} us per particle-turn")
    print(f"Quantum-efficient-table32: {particle_turn_cpu_single_table32 * 1E6:.2f} us per particle-turn")
    print(f"Quantum-efficient-table32-directsearch: {particle_turn_cpu_single_direct * 1E6:.2f} us per particle-turn")

########################################
# CPU OpenMP
########################################
if TRACK_CPU_OPENMP:
    print("Tracking with CPU OpenMP")

    line_cpu_openmp_none.track(
        particles       = particles_cpu_openmp_none,
        num_turns       = N_TURNS,
        time            = True,
        with_progress   = 10)
    line_cpu_openmp_mean.track(
        particles       = particles_cpu_openmp_mean,
        num_turns       = N_TURNS,
        time            = True,
        with_progress   = 10)
    line_cpu_openmp_quantum.track(
        particles       = particles_cpu_openmp_quantum,
        num_turns       = N_TURNS,
        time            = True,
        with_progress   = 10)
    line_cpu_openmp_efficient.track(
        particles       = particles_cpu_openmp_efficient,
        num_turns       = N_TURNS,
        time            = True,
        with_progress   = 10)
    line_cpu_openmp_table32.track(
        particles       = particles_cpu_openmp_table32,
        num_turns       = N_TURNS,
        time            = True,
        with_progress   = 10)
    line_cpu_openmp_direct.track(
        particles       = particles_cpu_openmp_direct,
        num_turns       = N_TURNS,
        time            = True,
        with_progress   = 10)

    particle_turn_cpu_openmp_none       = line_cpu_openmp_none.time_last_track \
        / (N_PARTICLES_CPU * N_TURNS)
    particle_turn_cpu_openmp_mean       = line_cpu_openmp_mean.time_last_track \
        / (N_PARTICLES_CPU * N_TURNS)
    particle_turn_cpu_openmp_quantum    = line_cpu_openmp_quantum.time_last_track \
        / (N_PARTICLES_CPU * N_TURNS)
    particle_turn_cpu_openmp_efficient  = line_cpu_openmp_efficient.time_last_track \
        / (N_PARTICLES_CPU * N_TURNS)
    particle_turn_cpu_openmp_table32    = line_cpu_openmp_table32.time_last_track \
        / (N_PARTICLES_CPU * N_TURNS)
    particle_turn_cpu_openmp_direct     = line_cpu_openmp_direct.time_last_track \
        / (N_PARTICLES_CPU * N_TURNS)

    print("Tracking times for CPU OpenMP:")
    print(f"None: {particle_turn_cpu_openmp_none * 1E6:.2f} us per particle-turn")
    print(f"Mean: {particle_turn_cpu_openmp_mean * 1E6:.2f} us per particle-turn")
    print(f"Quantum: {particle_turn_cpu_openmp_quantum * 1E6:.2f} us per particle-turn")
    print(f"Quantum-efficient: {particle_turn_cpu_openmp_efficient * 1E6:.2f} us per particle-turn")
    print(f"Quantum-efficient-table32: {particle_turn_cpu_openmp_table32 * 1E6:.2f} us per particle-turn")
    print(f"Quantum-efficient-table32-directsearch: {particle_turn_cpu_openmp_direct * 1E6:.2f} us per particle-turn")

########################################
# GPU CuPy
########################################
if TRACK_GPU_CUPY:
    print("Tracking with GPU CuPy")

    line_gpu_cupy_none.track(
        particles       = particles_gpu_cupy_none,
        num_turns       = N_TURNS,
        time            = True,
        with_progress   = 10)
    line_gpu_cupy_mean.track(
        particles       = particles_gpu_cupy_mean,
        num_turns       = N_TURNS,
        time            = True,
        with_progress   = 10)
    line_gpu_cupy_quantum.track(
        particles       = particles_gpu_cupy_quantum,
        num_turns       = N_TURNS,
        time            = True,
        with_progress   = 10)
    line_gpu_cupy_efficient.track(
        particles       = particles_gpu_cupy_efficient,
        num_turns       = N_TURNS,
        time            = True,
        with_progress   = 10)
    line_gpu_cupy_table32.track(
        particles       = particles_gpu_cupy_table32,
        num_turns       = N_TURNS,
        time            = True,
        with_progress   = 10)
    line_gpu_cupy_direct.track(
        particles       = particles_gpu_cupy_direct,
        num_turns       = N_TURNS,
        time            = True,
        with_progress   = 10)

    particle_turn_gpu_cupy_none       = line_gpu_cupy_none.time_last_track \
        / (N_PARTICLES_GPU * N_TURNS)
    particle_turn_gpu_cupy_mean       = line_gpu_cupy_mean.time_last_track \
        / (N_PARTICLES_GPU * N_TURNS)
    particle_turn_gpu_cupy_quantum    = line_gpu_cupy_quantum.time_last_track \
        / (N_PARTICLES_GPU * N_TURNS)
    particle_turn_gpu_cupy_efficient  = line_gpu_cupy_efficient.time_last_track \
        / (N_PARTICLES_GPU * N_TURNS)
    particle_turn_gpu_cupy_table32    = line_gpu_cupy_table32.time_last_track \
        / (N_PARTICLES_GPU * N_TURNS)
    particle_turn_gpu_cupy_direct     = line_gpu_cupy_direct.time_last_track \
        / (N_PARTICLES_GPU * N_TURNS)

    print("Tracking times for GPU CuPy:")
    print(f"None: {particle_turn_gpu_cupy_none * 1E6:.2f} us per particle-turn")
    print(f"Mean: {particle_turn_gpu_cupy_mean * 1E6:.2f} us per particle-turn")
    print(f"Quantum: {particle_turn_gpu_cupy_quantum * 1E6:.2f} us per particle-turn")
    print(f"Quantum-efficient: {particle_turn_gpu_cupy_efficient * 1E6:.2f} us per particle-turn")
    print(f"Quantum-efficient-table32: {particle_turn_gpu_cupy_table32 * 1E6:.2f} us per particle-turn")
    print(f"Quantum-efficient-table32-directsearch: {particle_turn_gpu_cupy_direct * 1E6:.2f} us per particle-turn")

################################################################################
# Compare Times
################################################################################
