# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import xobjects as xo
from xobjects.context import get_context_from_string, get_user_context
import xtrack as xt


################################################################################
# User parameters
################################################################################

# ``None`` asks Xobjects to use the default user context for the GPU tracker.
# For this example that context must resolve to ``ContextCupy``. The intended
# command-line use is
#
#     XOBJECTS_USER_CONTEXT=ContextCupy:0 python 016_benchmark_quantum_efficient_fcc_lattice.py
#
# A fully explicit run can also set this variable to ``"ContextCupy:0"``.
GPU_CONTEXT = None
CPU_OMP_NUM_THREADS = "auto"

CONTEXTS_TO_RUN = [
    "cpu-openmp",
    "cupy"]

MODES_TO_RUN = [
    "quantum",
    "quantum-efficient"]

N_PARTICLES = int(1E4)
N_TURNS     = int(5E3)

# Beam sizes are sampled only at these turn intervals. This avoids allocating a
# full turn-by-turn monitor with N_PARTICLES * N_TURNS entries per coordinate.
SAMPLE_EVERY = 100

# The FCC-ee H lattice is the same local test lattice used by the radiation
# equilibrium tests. It is loaded fresh for each radiation mode so that the only
# intentional difference between runs is the selected radiation model.
REPO_ROOT = Path(__file__).resolve().parents[2]
LINE_PATH = REPO_ROOT / "test_data" / "fcc_ee" / "fccee_h_thick.json"

# The thick FCC-ee H lattice contains strong wiggler elements. The radiation
# equilibrium tests slice these for accurate equilibrium-emittance calculations.
# Here the objective is a direct tracking benchmark, so this is kept configurable
# and disabled by default to keep the benchmark close to the stored lattice.
SLICE_WIGGLERS = False
WIGGLER_SLICES = 20

# CFFI/CuPy kernel compilation can accidentally parse the repository
# ``pyproject.toml`` if the process runs from the repository root. The temporary
# directory is only a build location for generated kernels; benchmark results are
# independent of where this directory is placed.
COMPILE_WORKDIR = "auto"


################################################################################
# Purpose
################################################################################
# This example is a direct benchmark of the existing photon-by-photon
# synchrotron-radiation model against the compound-table quantum-efficient model
# in a realistic FCC-ee lattice, comparing both OpenMP CPU and CuPy GPU tracking.
#
# All particles are launched at zero coordinates. The beam-size evolution is
# therefore produced by radiation excitation and lattice dynamics rather than by
# an imposed initial distribution. The two modes are run independently on fresh
# copies of the same CPU-compensated FCC-ee H lattice:
#
#     quantum           model='quantum'
#     quantum-efficient model='quantum-efficient'
#
# The benchmark tracks 1e4 particles for 5e3 turns by default on both contexts.
# It samples beam sizes every ``SAMPLE_EVERY`` turns instead of using a full
# turn-by-turn monitor, because storing every particle at every turn would
# dominate memory use and can mask the tracking-kernel cost that this example is
# meant to measure.


################################################################################
# Mode and context setup
################################################################################

MODE_DESCRIPTIONS = {
    "quantum": (
        "existing photon-by-photon quantum synchrotron radiation"),
    "quantum-efficient": (
        "compound-table quantum radiation using total energy loss tables")}

UNKNOWN_MODES = sorted(set(MODES_TO_RUN) - set(MODE_DESCRIPTIONS))
if UNKNOWN_MODES:
    raise ValueError(f"Unknown modes: {UNKNOWN_MODES}")

if COMPILE_WORKDIR == "auto":
    COMPILE_WORKDIR = tempfile.mkdtemp(prefix="xtrack-fcc-synrad-kernels-")

if COMPILE_WORKDIR:
    os.makedirs(COMPILE_WORKDIR, exist_ok=True)
    os.chdir(COMPILE_WORKDIR)

def is_cupy_context(context):
    """Return True when the resolved Xobjects context is a CuPy GPU context."""
    return context.__class__.__name__ == "ContextCupy"


def resolve_gpu_context():
    """Resolve the configured GPU context and require it to be a CuPy context."""
    if GPU_CONTEXT is None:
        context = get_user_context()
    elif isinstance(GPU_CONTEXT, str):
        context = get_context_from_string(GPU_CONTEXT)
    else:
        context = GPU_CONTEXT

    if not is_cupy_context(context):
        raise RuntimeError(
            "The GPU context must resolve to ContextCupy. Run with, for "
            "example, XOBJECTS_USER_CONTEXT=ContextCupy:0.")
    return context


def describe_cupy_device():
    """Return the active CuPy device id and name for terminal logging."""
    import cupy as cp

    device_id = cp.cuda.Device().id
    properties = cp.cuda.runtime.getDeviceProperties(device_id)
    device_name = properties["name"]
    if isinstance(device_name, bytes):
        device_name = device_name.decode()
    return device_id, device_name


cpu_context = xo.ContextCpu(omp_num_threads=CPU_OMP_NUM_THREADS)
gpu_context = resolve_gpu_context()

CONTEXTS = {
    "cpu-openmp": cpu_context,
    "cupy":       gpu_context,
}

UNKNOWN_CONTEXTS = sorted(set(CONTEXTS_TO_RUN) - set(CONTEXTS))
if UNKNOWN_CONTEXTS:
    raise ValueError(f"Unknown contexts: {UNKNOWN_CONTEXTS}")


################################################################################
# Lattice and particle helpers
################################################################################

def prepare_compensated_line_on_cpu(model):
    """
    Prepare the FCC-ee H lattice on CPU and select a stochastic mode.

    The compensation is done with ``model='mean'`` before switching to the
    stochastic radiation model. This follows the pattern used by the FCC-ee
    radiation equilibrium tests: the ring RF system is adjusted for the average
    synchrotron-radiation loss, then quantum fluctuations are enabled for
    tracking.

    The preparation deliberately stays on CPU because Xtrack tapering currently
    asserts that ``compensate_radiation_energy_loss`` is CPU-only. The prepared
    line is returned without a tracker so that equivalent OpenMP and CuPy
    trackers can be built from the same compensated lattice state.

    Parameters
    ----------
    model : str
        Stochastic radiation model to use after compensation. The benchmark
        compares ``"quantum"`` and ``"quantum-efficient"``.
    """
    line = xt.load(LINE_PATH)
    line.build_tracker()

    if SLICE_WIGGLERS:
        line.slice_thick_elements(slicing_strategies=[
            xt.Strategy(slicing=None),
            xt.Strategy(
                slicing=xt.Teapot(WIGGLER_SLICES, mode="thick"),
                name=r"^mwi.*"),
        ])
        line.build_tracker()

    line.configure_radiation(model="mean")
    line.compensate_radiation_energy_loss()
    line.configure_radiation(model=model)
    line.discard_tracker()
    return line


def build_line_for_context(base_line, run_context):
    """
    Build one tracker from the CPU-prepared line on the selected context.

    A fresh line copy is used for every mode/context pair. This avoids sharing
    tracker state between OpenMP and CuPy runs while preserving the same lattice,
    tapering, RF compensation, and radiation-model configuration.
    """
    line = base_line.copy()
    line.build_tracker(_context=run_context)
    return line


def build_zero_particles(line, run_context):
    """
    Build the mono-energetic zero-amplitude particle sample on one context.

    Explicit zero arrays are used rather than relying on scalar defaults so that
    the benchmark is unambiguous: every particle starts at the same phase-space
    point, and any observed spread is generated during tracking.
    """
    particles = line.build_particles(
        _context=run_context,
        x=np.zeros(N_PARTICLES),
        px=np.zeros(N_PARTICLES),
        y=np.zeros(N_PARTICLES),
        py=np.zeros(N_PARTICLES),
        zeta=np.zeros(N_PARTICLES),
        delta=np.zeros(N_PARTICLES),
    )
    particles._init_random_number_generator()
    return particles


def to_cpu(run_context, array):
    """Copy a context array to a NumPy array on the host."""
    return run_context.nparray_from_context_array(array)


def particle_standard_deviation(particles, field, run_context, live=None):
    """
    Return the standard deviation of one live-particle coordinate.

    Lost particles are excluded from the beam-size estimate. The benchmark also
    records the live-particle count so that unexpected losses are visible in the
    terminal summary.
    """
    values = to_cpu(run_context, getattr(particles, field))
    if live is None:
        state = to_cpu(run_context, particles.state)
        live = state > 0
    if not np.any(live):
        return np.nan
    return float(np.std(values[live]))


def record_beam_sizes(turn, particles, history, run_context):
    """
    Append one beam-size sample to the history dictionary.

    The selected coordinates are the practical diagnostics for this comparison:
    horizontal and vertical beam size show transverse excitation and damping,
    while ``sigma_delta`` and ``sigma_zeta`` show the longitudinal response to
    stochastic energy loss.
    """
    state = to_cpu(run_context, particles.state)
    live = state > 0
    n_alive = int(np.sum(live))

    history["turn"].append(turn)
    history["n_alive"].append(n_alive)
    history["sigma_x"].append(
        particle_standard_deviation(particles, "x", run_context, live=live))
    history["sigma_y"].append(
        particle_standard_deviation(particles, "y", run_context, live=live))
    history["sigma_zeta"].append(
        particle_standard_deviation(
            particles, "zeta", run_context, live=live))
    history["sigma_delta"].append(
        particle_standard_deviation(
            particles, "delta", run_context, live=live))


################################################################################
# Benchmark helpers
################################################################################

def run_mode_on_context(model, context_name, run_context, base_line):
    """
    Track one stochastic radiation mode on one context.

    The timing window covers ``line.track`` plus ``context.synchronize`` only.
    Beam-size copies and statistics are outside the timing window. This keeps the
    reported number focused on tracking work, while still preserving enough
    host-side diagnostics to see whether the two stochastic modes evolve
    similarly on OpenMP and CuPy.
    """
    print()
    print("-" * 80)
    print(f"Preparing mode: {model} on context: {context_name}")
    print(f"description: {MODE_DESCRIPTIONS[model]}")

    t_setup_start = time.perf_counter()
    line = build_line_for_context(base_line, run_context)
    particles = build_zero_particles(line, run_context)
    run_context.synchronize()
    t_setup = time.perf_counter() - t_setup_start

    history = {
        "turn": [],
        "n_alive": [],
        "sigma_x": [],
        "sigma_y": [],
        "sigma_zeta": [],
        "sigma_delta": [],
    }
    record_beam_sizes(0, particles, history, run_context)

    turns_done = 0
    t_track_total = 0.0
    chunk_times = []
    chunk_turns = []

    progress = tqdm(
        total=N_TURNS,
        desc=f"tracking {model} / {context_name}",
        unit="turn",
        dynamic_ncols=True,
    )

    while turns_done < N_TURNS:
        turns_this = min(SAMPLE_EVERY, N_TURNS - turns_done)

        t_start = time.perf_counter()
        line.track(
            particles,
            num_turns=turns_this,
            turn_by_turn_monitor=False,
        )
        run_context.synchronize()
        t_chunk = time.perf_counter() - t_start

        t_track_total += t_chunk
        chunk_times.append(t_chunk)
        chunk_turns.append(turns_this)
        turns_done += turns_this
        record_beam_sizes(turns_done, particles, history, run_context)

        progress.update(turns_this)
        progress.set_postfix({
            "chunk_s": f"{t_chunk:.3f}",
            "us/particle/turn": (
                f"{1e6 * t_chunk / (N_PARTICLES * turns_this):.3g}"),
        })

    progress.close()

    chunk_times = np.array(chunk_times)
    chunk_turns = np.array(chunk_turns)
    particle_turns = N_PARTICLES * N_TURNS

    result = {
        "model": model,
        "context_name": context_name,
        "context": run_context,
        "setup_time_s": t_setup,
        "track_time_s": t_track_total,
        "chunk_times_s": chunk_times,
        "particle_turns": particle_turns,
        "particles_turns_per_s": particle_turns / t_track_total,
        "us_per_particle_turn": 1e6 * t_track_total / particle_turns,
        "chunk_us_per_particle_turn": (
            1e6 * chunk_times / (N_PARTICLES * chunk_turns)),
        "history": history,
    }

    print_mode_summary(result)
    return result


def sem(values):
    """Return the standard error of the mean for a one-dimensional sample."""
    values = np.asarray(values)
    if len(values) < 2:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def print_mode_summary(result):
    """Print one mode summary immediately after it finishes tracking."""
    history = result["history"]
    print(f"context           = {result['context_name']}")
    print(f"setup time        = {result['setup_time_s']:.6e} s")
    print(f"tracking time     = {result['track_time_s']:.6e} s")
    print(
        "tracking rate     = "
        f"{result['particles_turns_per_s']:.6e} particle-turns/s")
    print(
        "tracking cost     = "
        f"{result['us_per_particle_turn']:.6e} us/particle/turn")
    print(f"final live        = {history['n_alive'][-1]} / {N_PARTICLES}")
    print(f"final sigma_x     = {history['sigma_x'][-1]:.6e} m")
    print(f"final sigma_y     = {history['sigma_y'][-1]:.6e} m")
    print(f"final sigma_zeta  = {history['sigma_zeta'][-1]:.6e} m")
    print(f"final sigma_delta = {history['sigma_delta'][-1]:.6e}")


def print_configuration():
    """Print the resolved run configuration before any heavy work starts."""
    print(f"cpu_context = {cpu_context}")
    print(f"gpu_context = {gpu_context}")
    print(f"gpu_context_is_cupy = {is_cupy_context(gpu_context)}")
    device_id, device_name = describe_cupy_device()
    print(f"cupy_device_id = {device_id}")
    print(f"cupy_device_name = {device_name}")
    print(f"compile_workdir = {COMPILE_WORKDIR}")
    print(f"line_path = {LINE_PATH}")
    print(f"n_particles = {N_PARTICLES}")
    print(f"n_turns = {N_TURNS}")
    print(f"sample_every = {SAMPLE_EVERY}")
    print(f"slice_wigglers = {SLICE_WIGGLERS}")
    if SLICE_WIGGLERS:
        print(f"wiggler_slices = {WIGGLER_SLICES}")
    print("contexts:")
    for context_name in CONTEXTS_TO_RUN:
        print(f"  {context_name:10s} {CONTEXTS[context_name]}")
    print("modes:")
    for mode in MODES_TO_RUN:
        print(f"  {mode:17s} {MODE_DESCRIPTIONS[mode]}")


def print_comparison_summary(results):
    """
    Print the side-by-side timing and final beam-size comparison.

    This is the primary non-graphical output for SSH runs. The speedup is quoted
    as quantum tracking time divided by quantum-efficient tracking time, so a
    value larger than one means the compound-table mode is faster.
    """
    print()
    print("=" * 80)
    print("Timing comparison")
    print("=" * 80)
    print(
        f"{'mode':>17s} {'context':>12s} {'track [s]':>14s} {'us/p/t':>14s} "
        f"{'M particle-turn/s':>20s} {'SEM us/p/t':>14s}")
    for mode in MODES_TO_RUN:
        for context_name in CONTEXTS_TO_RUN:
            result = results[mode][context_name]
            print(
                f"{mode:>17s} "
                f"{context_name:>12s} "
                f"{result['track_time_s']:14.6e} "
                f"{result['us_per_particle_turn']:14.6e} "
                f"{result['particles_turns_per_s'] / 1e6:20.6e} "
                f"{sem(result['chunk_us_per_particle_turn']):14.6e}")

    print()
    print("Speedups")
    print("-" * 80)
    for context_name in CONTEXTS_TO_RUN:
        if set(("quantum", "quantum-efficient")).issubset(results):
            speedup = (
                results["quantum"][context_name]["track_time_s"]
                / results["quantum-efficient"][context_name]["track_time_s"]
            )
            print(
                f"quantum-efficient over quantum on {context_name}: "
                f"{speedup:.6e}x")

    for mode in MODES_TO_RUN:
        if set(("cpu-openmp", "cupy")).issubset(results[mode]):
            gpu_speedup = (
                results[mode]["cpu-openmp"]["track_time_s"]
                / results[mode]["cupy"]["track_time_s"]
            )
            print(
                f"CuPy over OpenMP for {mode}: "
                f"{gpu_speedup:.6e}x")

    print()
    print("Final beam-size comparison")
    print("-" * 80)
    print(
        f"{'mode':>17s} {'context':>12s} {'alive':>10s} {'sigma_x [m]':>14s} "
        f"{'sigma_y [m]':>14s} {'sigma_zeta [m]':>16s} "
        f"{'sigma_delta':>14s}")
    for mode in MODES_TO_RUN:
        for context_name in CONTEXTS_TO_RUN:
            history = results[mode][context_name]["history"]
            print(
                f"{mode:>17s} "
                f"{context_name:>12s} "
                f"{history['n_alive'][-1]:10d} "
                f"{history['sigma_x'][-1]:14.6e} "
                f"{history['sigma_y'][-1]:14.6e} "
                f"{history['sigma_zeta'][-1]:16.6e} "
                f"{history['sigma_delta'][-1]:14.6e}")


################################################################################
# Plotting
################################################################################

def plot_beam_size_evolution(results):
    """
    Plot beam-size evolution for both stochastic modes and both contexts.

    The same sampled turns are used for all curves. Agreement is not a formal
    proof of physics equivalence, but it is the most direct end-to-end check
    that the benchmark did not gain speed by changing the observable radiation
    excitation in the lattice.
    """
    plt.close("fcc beam size evolution")
    fig, axes = plt.subplots(
        2, 2, figsize=(12.5, 8.5),
        num="fcc beam size evolution",
        sharex=True,
    )

    specs = [
        ("sigma_x", r"$\sigma_x$ [m]"),
        ("sigma_y", r"$\sigma_y$ [m]"),
        ("sigma_zeta", r"$\sigma_\zeta$ [m]"),
        ("sigma_delta", r"$\sigma_\delta$ [1]"),
    ]

    for ax, (field, ylabel) in zip(axes.ravel(), specs):
        for mode in MODES_TO_RUN:
            for context_name in CONTEXTS_TO_RUN:
                history = results[mode][context_name]["history"]
                ax.plot(
                    history["turn"],
                    history[field],
                    label=f"{mode} / {context_name}")
        ax.set_ylabel(ylabel)
        ax.grid(True)

    for ax in axes[-1, :]:
        ax.set_xlabel("Turn [1]")

    axes[0, 0].legend(loc="best")
    fig.suptitle(
        "FCC-ee H lattice beam-size evolution from zero initial coordinates")
    fig.tight_layout()


def plot_timing_comparison(results):
    """
    Plot total tracking time and per-particle-turn cost for all comparisons.

    Error bars on the per-particle-turn plot are the standard errors of the
    chunk-level estimates. They indicate timing repeatability during this run,
    not statistical uncertainty in the radiation physics.
    """
    x_pos = np.arange(len(MODES_TO_RUN))
    width = 0.8 / len(CONTEXTS_TO_RUN)

    plt.close("fcc tracking timing")
    fig, axes = plt.subplots(
        1, 2, figsize=(12.0, 4.8),
        num="fcc tracking timing",
    )

    for i_context, context_name in enumerate(CONTEXTS_TO_RUN):
        offset = (i_context - 0.5 * (len(CONTEXTS_TO_RUN) - 1)) * width
        track_times = np.array([
            results[mode][context_name]["track_time_s"]
            for mode in MODES_TO_RUN
        ])
        us_per_particle_turn = np.array([
            results[mode][context_name]["us_per_particle_turn"]
            for mode in MODES_TO_RUN
        ])
        us_per_particle_turn_sem = np.array([
            sem(results[mode][context_name]["chunk_us_per_particle_turn"])
            for mode in MODES_TO_RUN
        ])

        axes[0].bar(
            x_pos + offset,
            track_times,
            width=width,
            label=context_name,
        )
        axes[1].bar(
            x_pos + offset,
            us_per_particle_turn,
            width=width,
            yerr=us_per_particle_turn_sem,
            capsize=4,
            label=context_name,
        )

    axes[0].set_ylabel("Tracking time [s]")
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(MODES_TO_RUN, rotation=20, ha="right")
    axes[0].grid(True, axis="y")
    axes[0].legend(loc="best")

    axes[1].set_ylabel(r"Tracking cost [$\mu$s/particle/turn]")
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(MODES_TO_RUN, rotation=20, ha="right")
    axes[1].grid(True, axis="y")
    axes[1].legend(loc="best")

    fig.suptitle(
        f"FCC-ee H lattice tracking benchmark: {N_PARTICLES} particles, "
        f"{N_TURNS} turns")
    fig.tight_layout()


################################################################################
# Run
################################################################################

print_configuration()

results = {}
for mode in MODES_TO_RUN:
    print()
    print("=" * 80)
    print(f"CPU lattice preparation for mode: {mode}")
    print("=" * 80)
    t_prepare_start = time.perf_counter()
    base_line = prepare_compensated_line_on_cpu(mode)
    t_prepare = time.perf_counter() - t_prepare_start
    print(f"CPU lattice preparation time = {t_prepare:.6e} s")

    results[mode] = {}
    for context_name in CONTEXTS_TO_RUN:
        results[mode][context_name] = run_mode_on_context(
            mode,
            context_name,
            CONTEXTS[context_name],
            base_line,
        )

print_comparison_summary(results)
plot_beam_size_evolution(results)
plot_timing_comparison(results)
plt.show()
