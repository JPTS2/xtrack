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

from xobjects.context import get_context_from_string, get_user_context
import xtrack as xt


################################################################################
# User parameters
################################################################################

# ``None`` asks Xobjects to use the default user context. For this example that
# context must resolve to ``ContextCupy``. The intended command-line use is
#
#     XOBJECTS_USER_CONTEXT=ContextCupy:0 python 016_benchmark_quantum_efficient_fcc_lattice.py
#
# A fully explicit run can also set this variable to ``"ContextCupy:0"``.
CONTEXT = None

MODES_TO_RUN = [
    "quantum",
    "quantum-efficient",
]

N_PARTICLES = int(1E4)
N_TURNS = int(5E3)

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
# This example is a direct GPU benchmark of the existing photon-by-photon
# synchrotron-radiation model against the compound-table quantum-efficient model
# in a realistic FCC-ee lattice.
#
# All particles are launched at zero coordinates. The beam-size evolution is
# therefore produced by radiation excitation and lattice dynamics rather than by
# an imposed initial distribution. The two modes are run independently on fresh
# copies of the same compensated FCC-ee H lattice:
#
#     quantum           model='quantum'
#     quantum-efficient model='quantum-efficient'
#
# The benchmark tracks 1e4 particles for 5e3 turns by default. It samples beam
# sizes every ``SAMPLE_EVERY`` turns instead of using a full turn-by-turn monitor,
# because storing every particle at every turn would dominate memory use and can
# mask the tracking-kernel cost that this example is meant to measure.


################################################################################
# Mode and context setup
################################################################################

MODE_DESCRIPTIONS = {
    "quantum": (
        "existing photon-by-photon quantum synchrotron radiation"),
    "quantum-efficient": (
        "compound-table quantum radiation using total energy loss tables"),
}

UNKNOWN_MODES = sorted(set(MODES_TO_RUN) - set(MODE_DESCRIPTIONS))
if UNKNOWN_MODES:
    raise ValueError(f"Unknown modes: {UNKNOWN_MODES}")

if COMPILE_WORKDIR == "auto":
    COMPILE_WORKDIR = tempfile.mkdtemp(prefix="xtrack-fcc-synrad-kernels-")

if COMPILE_WORKDIR:
    os.makedirs(COMPILE_WORKDIR, exist_ok=True)
    os.chdir(COMPILE_WORKDIR)

if CONTEXT is None:
    context = get_user_context()
elif isinstance(CONTEXT, str):
    context = get_context_from_string(CONTEXT)
else:
    context = CONTEXT


def is_cupy_context(context):
    """Return True when the resolved Xobjects context is a CuPy GPU context."""
    return context.__class__.__name__ == "ContextCupy"


def describe_cupy_device():
    """Return the active CuPy device id and name for terminal logging."""
    import cupy as cp

    device_id = cp.cuda.Device().id
    properties = cp.cuda.runtime.getDeviceProperties(device_id)
    device_name = properties["name"]
    if isinstance(device_name, bytes):
        device_name = device_name.decode()
    return device_id, device_name


if not is_cupy_context(context):
    raise RuntimeError(
        "This example is intentionally GPU-only. Run it with, for example, "
        "XOBJECTS_USER_CONTEXT=ContextCupy:0.")


################################################################################
# Lattice and particle helpers
################################################################################

def load_compensated_line(model):
    """
    Load the FCC-ee H lattice, compensate mean energy loss, and select a mode.

    The compensation is done with ``model='mean'`` before switching to the
    stochastic radiation model. This follows the pattern used by the FCC-ee
    radiation equilibrium tests: the ring RF system is adjusted for the average
    synchrotron-radiation loss, then quantum fluctuations are enabled for
    tracking.

    Parameters
    ----------
    model : str
        Stochastic radiation model to use after compensation. The benchmark
        compares ``"quantum"`` and ``"quantum-efficient"``.
    """
    line = xt.load(LINE_PATH)
    line.build_tracker(_context=context)

    if SLICE_WIGGLERS:
        line.slice_thick_elements(slicing_strategies=[
            xt.Strategy(slicing=None),
            xt.Strategy(
                slicing=xt.Teapot(WIGGLER_SLICES, mode="thick"),
                name=r"^mwi.*"),
        ])
        line.build_tracker(_context=context)

    line.configure_radiation(model="mean")
    line.compensate_radiation_energy_loss()
    line.configure_radiation(model=model)
    return line


def build_zero_particles(line):
    """
    Build the mono-energetic zero-amplitude particle sample on the GPU.

    Explicit zero arrays are used rather than relying on scalar defaults so that
    the benchmark is unambiguous: every particle starts at the same phase-space
    point, and any observed spread is generated during tracking.
    """
    particles = line.build_particles(
        _context=context,
        x=np.zeros(N_PARTICLES),
        px=np.zeros(N_PARTICLES),
        y=np.zeros(N_PARTICLES),
        py=np.zeros(N_PARTICLES),
        zeta=np.zeros(N_PARTICLES),
        delta=np.zeros(N_PARTICLES),
    )
    particles._init_random_number_generator()
    return particles


def to_cpu(array):
    """Copy a context array to a NumPy array on the host."""
    return context.nparray_from_context_array(array)


def particle_standard_deviation(particles, field):
    """
    Return the standard deviation of one live-particle coordinate.

    Lost particles are excluded from the beam-size estimate. The benchmark also
    records the live-particle count so that unexpected losses are visible in the
    terminal summary.
    """
    values = to_cpu(getattr(particles, field))
    state = to_cpu(particles.state)
    live = state > 0
    if not np.any(live):
        return np.nan
    return float(np.std(values[live]))


def record_beam_sizes(turn, particles, history):
    """
    Append one beam-size sample to the history dictionary.

    The selected coordinates are the practical diagnostics for this comparison:
    horizontal and vertical beam size show transverse excitation and damping,
    while ``sigma_delta`` and ``sigma_zeta`` show the longitudinal response to
    stochastic energy loss.
    """
    state = to_cpu(particles.state)
    n_alive = int(np.sum(state > 0))

    history["turn"].append(turn)
    history["n_alive"].append(n_alive)
    history["sigma_x"].append(particle_standard_deviation(particles, "x"))
    history["sigma_y"].append(particle_standard_deviation(particles, "y"))
    history["sigma_zeta"].append(
        particle_standard_deviation(particles, "zeta"))
    history["sigma_delta"].append(
        particle_standard_deviation(particles, "delta"))


################################################################################
# Benchmark helpers
################################################################################

def run_mode(model):
    """
    Track one stochastic radiation mode and return timing and beam-size history.

    The timing window covers ``line.track`` plus ``context.synchronize`` only.
    Beam-size copies and statistics are outside the timing window. This keeps the
    reported number focused on GPU tracking work, while still preserving enough
    host-side diagnostics to see whether the two stochastic modes evolve
    similarly.
    """
    print()
    print("-" * 80)
    print(f"Preparing mode: {model}")
    print(f"description: {MODE_DESCRIPTIONS[model]}")

    t_setup_start = time.perf_counter()
    line = load_compensated_line(model)
    particles = build_zero_particles(line)
    context.synchronize()
    t_setup = time.perf_counter() - t_setup_start

    history = {
        "turn": [],
        "n_alive": [],
        "sigma_x": [],
        "sigma_y": [],
        "sigma_zeta": [],
        "sigma_delta": [],
    }
    record_beam_sizes(0, particles, history)

    turns_done = 0
    t_track_total = 0.0
    chunk_times = []
    chunk_turns = []

    progress = tqdm(
        total=N_TURNS,
        desc=f"tracking {model}",
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
        context.synchronize()
        t_chunk = time.perf_counter() - t_start

        t_track_total += t_chunk
        chunk_times.append(t_chunk)
        chunk_turns.append(turns_this)
        turns_done += turns_this
        record_beam_sizes(turns_done, particles, history)

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
    print(f"context = {context}")
    print(f"context_is_cupy = {is_cupy_context(context)}")
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
        f"{'mode':>17s} {'track [s]':>14s} {'us/p/t':>14s} "
        f"{'M particle-turn/s':>20s} {'SEM us/p/t':>14s}")
    for mode in MODES_TO_RUN:
        result = results[mode]
        print(
            f"{mode:>17s} "
            f"{result['track_time_s']:14.6e} "
            f"{result['us_per_particle_turn']:14.6e} "
            f"{result['particles_turns_per_s'] / 1e6:20.6e} "
            f"{sem(result['chunk_us_per_particle_turn']):14.6e}")

    if set(("quantum", "quantum-efficient")).issubset(results):
        speedup = (
            results["quantum"]["track_time_s"]
            / results["quantum-efficient"]["track_time_s"]
        )
        print()
        print(
            "quantum-efficient speedup over quantum "
            f"= {speedup:.6e}x")

    print()
    print("Final beam-size comparison")
    print("-" * 80)
    print(
        f"{'mode':>17s} {'alive':>10s} {'sigma_x [m]':>14s} "
        f"{'sigma_y [m]':>14s} {'sigma_zeta [m]':>16s} "
        f"{'sigma_delta':>14s}")
    for mode in MODES_TO_RUN:
        history = results[mode]["history"]
        print(
            f"{mode:>17s} "
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
    Plot beam-size evolution for both stochastic radiation modes.

    The same sampled turns are used for both modes. Agreement in these curves is
    not a formal proof of physics equivalence, but it is the most direct
    end-to-end check that the benchmark did not gain speed by changing the
    observable radiation excitation in the lattice.
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
            history = results[mode]["history"]
            ax.plot(history["turn"], history[field], label=mode)
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
    Plot total tracking time and per-particle-turn cost for the two modes.

    Error bars on the per-particle-turn plot are the standard errors of the
    chunk-level estimates. They indicate timing repeatability during this run,
    not statistical uncertainty in the radiation physics.
    """
    labels = MODES_TO_RUN
    x_pos = np.arange(len(labels))

    track_times = np.array([results[mode]["track_time_s"] for mode in labels])
    us_per_particle_turn = np.array([
        results[mode]["us_per_particle_turn"] for mode in labels
    ])
    us_per_particle_turn_sem = np.array([
        sem(results[mode]["chunk_us_per_particle_turn"]) for mode in labels
    ])

    plt.close("fcc tracking timing")
    fig, axes = plt.subplots(
        1, 2, figsize=(12.0, 4.8),
        num="fcc tracking timing",
    )

    axes[0].bar(x_pos, track_times)
    axes[0].set_ylabel("Tracking time [s]")
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(labels, rotation=20, ha="right")
    axes[0].grid(True, axis="y")

    axes[1].bar(
        x_pos,
        us_per_particle_turn,
        yerr=us_per_particle_turn_sem,
        capsize=4,
    )
    axes[1].set_ylabel(r"Tracking cost [$\mu$s/particle/turn]")
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1].grid(True, axis="y")

    fig.suptitle(
        f"FCC-ee H lattice GPU benchmark: {N_PARTICLES} particles, "
        f"{N_TURNS} turns")
    fig.tight_layout()


################################################################################
# Run
################################################################################

print_configuration()

results = {}
for mode in MODES_TO_RUN:
    results[mode] = run_mode(mode)

print_comparison_summary(results)
plot_beam_size_evolution(results)
plot_timing_comparison(results)
plt.show()
