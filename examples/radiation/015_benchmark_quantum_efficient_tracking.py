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

import xobjects as xo
from xobjects.context import get_context_from_string, get_user_context
import xpart as xp
import xtrack as xt

################################################################################
# User parameters
################################################################################

# ``None`` asks Xobjects to use the default user context. This still allows the
# context to be selected from the shell through ``XOBJECTS_USER_CONTEXT``. For a
# fully explicit run, set this to a string such as ``"ContextCpu:auto"`` or
# ``"ContextCupy:0"``, or set it directly to an instantiated context.
CONTEXT         = None

MODES_TO_RUN    = [
    "no-radiation",
    "mean",
    "quantum",
    "quantum-efficient",
    "quantum-efficient-table32",
    "quantum-efficient-table32-directsearch"]

N_PARTICLES     = int(1E7)
BATCH_SIZE      = int(1E6)
PILOT_PARTICLES = int(1E6)

# The same physical bend can be split into slices. This is useful when checking
# performance against the per-slice photon multiplicity expected in a real line.
N_SLICES        = 1

CASES           = [
    {
        "name":   "SuperKEKB-like low lambda",
        "label":  "low lambda",
        "p0c":    4.0e9,
        "length": 22.653765579198428,
        "angle":  0.0012128729366015125},
    {
        "name":   "FCC-ee tt-like high lambda",
        "label":  "high lambda",
        "p0c":    182.5e9,
        "length": 22.653765579198428,
        "angle":  0.0022799344662676477}]

INITIAL_PX      = 1e-4
INITIAL_PY      = -1e-4
INITIAL_DELTA   = 0.0

N_BINS              = 400
TAIL_PROBABILITY    = 1E-6
DDELTA_LOW_SIGMAS   = 8.0

# CFFI CPU kernel compilation can accidentally parse the repository
# ``pyproject.toml`` if the process runs from the repository root. The temporary
# directory is only a build location for generated kernels; benchmark results are
# independent of where this directory is placed.
COMPILE_WORKDIR = "auto"

ALPHA_EM        = 7.2973525693e-3


################################################################################
# Purpose
################################################################################
# This example measures the tracking cost of the available synchrotron-radiation
# modes in two representative bends:
#
#     no-radiation      radiation_flag = 0
#     mean              radiation_flag = 1
#     quantum           radiation_flag = 2
#     quantum-efficient radiation_flag = 3
#
# The low-lambda case is dominated by zero- and one-photon events. The high-
# lambda case is the regime where replacing many photon-by-photon samples with
# compound total-energy table samples should be most beneficial.
#
# The physics-distribution comparison is handled in more detail by
# ``013_compare_quantum_efficient.py`` and the table-only distribution check is
# handled by ``014_validate_quantum_efficient_tables.py``. This file is instead
# focused on runtime: for each selected radiation mode it streams particle
# batches through the same element, times only the ``track`` call, and reports
# the resulting throughput.
#
# Histograms and moments are still accumulated because timing-only tests can hide
# obvious failures. They are deliberately lightweight diagnostics, not a full
# physics validation suite. Particle construction and host-side analysis are
# excluded from the timing so that CPU and GPU runs can be compared on the
# tracking kernel itself.


################################################################################
# Mode and context setup
################################################################################

RADIATION_FLAGS = {
    "no-radiation":         0,
    "mean":                 1,
    "quantum":              2,
    "quantum-efficient":    3,
    "quantum-efficient-table32": 4,
    "quantum-efficient-table32-directsearch": 5}

MODE_DESCRIPTIONS = {
    "no-radiation": (
        "pure tracking baseline with the radiation code path disabled"),
    "mean": (
        "deterministic average radiation kick, no quantum fluctuations"),
    "quantum": (
        "existing photon-by-photon quantum synchrotron radiation"),
    "quantum-efficient": (
        "compound-table quantum radiation using power-of-two total energy "
        "loss tables"),
    "quantum-efficient-table32": (
        "compound-table quantum radiation using direct fixed-count tables "
        "through N=32 and binary grid search"),
    "quantum-efficient-table32-directsearch": (
        "compound-table quantum radiation using direct fixed-count tables "
        "through N=32 and direct grid indexing")}

PLOT_SPECS = [
    ("dpx", r"$\Delta p_x$"),
    ("dpy", r"$\Delta p_y$"),
    ("ddelta", r"$\Delta\delta$")]

UNKNOWN_MODES = sorted(set(MODES_TO_RUN) - set(RADIATION_FLAGS))
if UNKNOWN_MODES:
    raise ValueError(f"Unknown modes: {UNKNOWN_MODES}")

REQUESTS_TABLE32 = any("table32" in mode for mode in MODES_TO_RUN)
HEADER_PATH = (
    Path(__file__).resolve().parents[2]
    / "xtrack" / "headers" / "synrad_total_energy_tables.h")
HEADER_TEXT = HEADER_PATH.read_text() if HEADER_PATH.exists() else ""
HAS_TABLE32_TABLES = (
    "XTRACK_SYNRAD_TOTAL_ENERGY_DIRECT_TABLE_MAX 32" in HEADER_TEXT
    and "XTRACK_SYNRAD_TOTAL_ENERGY_TAIL_PROBABILITY_MAX 9.80000000000000038e-02"
        in HEADER_TEXT)
if REQUESTS_TABLE32 and not HAS_TABLE32_TABLES:
    raise RuntimeError(
        "MODES_TO_RUN requests table32 radiation modes, but "
        "synrad_total_energy_tables.h has not been regenerated with direct "
        "N=1..32 tables and the current runtime probability grid.")

if COMPILE_WORKDIR == "auto":
    COMPILE_WORKDIR = tempfile.mkdtemp(prefix="xtrack-synrad-kernels-")

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
    """Return a human-readable description of the active CuPy device."""
    import cupy as cp

    device_id = cp.cuda.Device().id
    properties = cp.cuda.runtime.getDeviceProperties(device_id)
    device_name = properties["name"]
    if isinstance(device_name, bytes):
        device_name = device_name.decode()
    return device_id, device_name



################################################################################
# Beam and line helpers
################################################################################

def expected_average_photon_count(case):
    """
    Return the approximate mean emitted photon count for one configured bend.

    Xsuite's synchrotron-radiation model treats photon emission in a bend as a
    Poisson process. For a sector bend the expected number of photons is

        lambda = (5 / (2 sqrt(3))) alpha (E / m) |theta|

    where ``alpha`` is the fine-structure constant, ``E`` is the reference
    momentum in eV for ultra-relativistic electrons, ``m`` is the electron rest
    mass in eV, and ``theta`` is the bend angle in radians.

    This number is printed because it strongly affects the benchmark
    interpretation. A high-lambda bend favours the compound-table algorithm,
    while low-lambda slices can be dominated by zero- and one-photon events.
    """
    return (
        2.5 / np.sqrt(3)
        * ALPHA_EM
        * (case["p0c"] / xp.ELECTRON_MASS_EV)
        * abs(case["angle"] / N_SLICES))


def make_particles(case, n_batch, i_start):
    """
    Build one particle batch on the selected context.

    All particles start from the same transverse coordinates and momentum
    deviation. The only per-particle label is ``particle_id``. Keeping the beam
    otherwise mono-energetic makes the measured spread after tracking come from
    synchrotron-radiation sampling rather than from an imposed initial beam
    distribution.

    Parameters
    ----------
    case : dict
        Benchmark case providing the reference momentum.
    n_batch : int
        Number of particles to create for this streaming batch.
    i_start : int
        First particle id in the batch. Distinct ids avoid reusing the same
        random-number stream across batches in stochastic radiation modes.
    """
    return xp.Particles(
        _context=context,
        p0c=case["p0c"],
        mass0=xp.ELECTRON_MASS_EV,
        q0=-1,
        particle_id=np.arange(i_start, i_start + n_batch),
        x=np.zeros(n_batch),
        px=INITIAL_PX * np.ones(n_batch),
        y=np.zeros(n_batch),
        py=INITIAL_PY * np.ones(n_batch),
        delta=INITIAL_DELTA * np.ones(n_batch),
    )


def make_tracker(case, radiation_flag):
    """
    Build the element or line used for one radiation mode.

    With ``N_SLICES = 1`` the benchmark tracks through a single ``xt.Bend``.
    For larger ``N_SLICES`` the same total length and angle are split into a
    line of identical bend slices. This keeps the integrated bend unchanged
    while testing the per-slice multiplicity and repeated-element overhead that
    a real lattice would see.
    """
    if N_SLICES == 1:
        return xt.Bend(
            _context=context,
            length=case["length"],
            angle=case["angle"],
            k0_from_h=True,
            radiation_flag=radiation_flag,
        )

    elements = [
        xt.Bend(
            length=case["length"] / N_SLICES,
            angle=case["angle"] / N_SLICES,
            k0_from_h=True,
            radiation_flag=radiation_flag,
        )
        for _ in range(N_SLICES)
    ]
    line = xt.Line(elements=elements)
    line.build_tracker(_context=context)
    return line


def track_batch(case, tracker, radiation_flag, n_batch, i_start):
    """
    Track one batch and return host-side coordinate changes.

    The timer starts immediately before ``tracker.track`` and stops after
    ``context.synchronize``. Particle construction, random-number-generator
    initialisation, and host transfers for the returned arrays are outside the
    timing window. This is the quantity most relevant for comparing tracking
    kernels between radiation modes and between CPU/GPU contexts.
    """
    particles = make_particles(case, n_batch, i_start)

    t_start = time.perf_counter()
    tracker.track(particles)
    context.synchronize()
    t_track = time.perf_counter() - t_start

    return {
        "dpx": context.nparray_from_context_array(particles.px) - INITIAL_PX,
        "dpy": context.nparray_from_context_array(particles.py) - INITIAL_PY,
        "ddelta": (
            context.nparray_from_context_array(particles.delta)
            - INITIAL_DELTA
        ),
        "t_track": t_track,
    }


################################################################################
# Histogram and statistics helpers
################################################################################

def collect_pilot(case, trackers):
    """
    Run a small pilot sample used only to choose common histogram bins.

    The pilot avoids storing the full benchmark sample in memory. Bins are based
    on stochastic modes where possible, because the no-radiation and
    mean-radiation modes do not provide useful distribution widths for
    comparison plots.
    """
    pilot = {mode: {key: [] for key, _ in PLOT_SPECS} for mode in MODES_TO_RUN}
    n_pilot = min(PILOT_PARTICLES, N_PARTICLES)

    for mode in MODES_TO_RUN:
        radiation_flag = RADIATION_FLAGS[mode]
        n_done = 0
        while n_done < n_pilot:
            n_batch = min(BATCH_SIZE, n_pilot - n_done)
            values = track_batch(
                case, trackers[mode], radiation_flag, n_batch, n_done)
            for key in pilot[mode]:
                pilot[mode][key].append(values[key])
            n_done += n_batch

        for key in pilot[mode]:
            pilot[mode][key] = np.concatenate(pilot[mode][key])

    return pilot


def make_bins(pilot):
    """
    Choose common histogram edges for all selected modes.

    Common bins make CDF residuals meaningful because every mode is evaluated on
    the same grid. The ``ddelta`` lower edge is extended beyond a simple
    quantile estimate to keep rare large energy-loss events visible when plots
    are enabled.
    """
    bins = {}
    stochastic_modes = [
        mode for mode in MODES_TO_RUN
        if mode not in ("no-radiation", "mean")
    ]
    if not stochastic_modes:
        stochastic_modes = MODES_TO_RUN

    for key, _label in PLOT_SPECS:
        all_values = np.concatenate([
            pilot[mode][key] for mode in stochastic_modes
        ])
        if key == "ddelta":
            mean = np.mean(all_values)
            sigma = np.std(all_values)
            lo = min(
                np.quantile(all_values, TAIL_PROBABILITY),
                mean - DDELTA_LOW_SIGMAS * sigma,
            )
            hi = np.quantile(all_values, 1 - 1e-5)
        else:
            lo, hi = np.quantile(all_values, [1e-5, 1 - 1e-5])

        if hi <= lo:
            hi = lo + max(abs(lo), 1.0) * 1e-12
        bins[key] = np.linspace(lo, hi, N_BINS)

    return bins


def make_accumulators(bins):
    """
    Allocate streaming accumulators for timing, moments, and histograms.

    The benchmark can be run with large particle counts, especially on GPU. This
    structure stores only aggregate statistics, so memory use is controlled by
    ``N_BINS`` and the number of selected modes rather than by ``N_PARTICLES``.
    """
    acc = {}
    for mode in MODES_TO_RUN:
        acc[mode] = {
            "timing": {
                "n_batches": 0,
                "n_particles": 0,
                "total": 0.0,
                "min": np.inf,
                "max": 0.0,
                "batch_times": [],
                "batch_particles": [],
            }
        }
        for key, _label in PLOT_SPECS:
            acc[mode][key] = {
                "hist": np.zeros(bins[key].size - 1, dtype=np.int64),
                "below": 0,
                "above": 0,
                "n": 0,
                "sum": 0.0,
                "sum2": 0.0,
                "min": np.inf,
                "max": -np.inf,
            }
    return acc


def update_histogram(acc_item, values, bin_edges):
    """
    Add one batch to a histogram accumulator.

    Values outside the bin range are counted explicitly. This makes it clear
    when a pilot-derived plotting range has clipped a distribution tail.
    """
    hist, _ = np.histogram(values, bins=bin_edges)
    acc_item["hist"] += hist
    acc_item["below"] += np.count_nonzero(values < bin_edges[0])
    acc_item["above"] += np.count_nonzero(values >= bin_edges[-1])
    acc_item["n"] += values.size


def update_moments(acc_item, values):
    """
    Add one batch to the first- and second-moment accumulators.

    The moments are used for a compact numerical sanity check alongside the
    timing result. They are not intended to replace the more detailed
    distribution tests in examples 013 and 014.
    """
    acc_item["sum"] += np.sum(values)
    acc_item["sum2"] += np.sum(values * values)
    acc_item["min"] = min(acc_item["min"], np.min(values))
    acc_item["max"] = max(acc_item["max"], np.max(values))


def mean_and_rms(acc_item):
    """
    Return the streamed mean and RMS for one observable accumulator.

    The variance expression can become very slightly negative through floating
    point cancellation when a distribution is narrow, so it is clipped at zero
    before taking the square root.
    """
    mean = acc_item["sum"] / acc_item["n"]
    variance = acc_item["sum2"] / acc_item["n"] - mean * mean
    return mean, np.sqrt(max(variance, 0.0))


def timing_metrics(timing):
    """
    Convert one timing accumulator into absolute and batch-level statistics.

    The total throughput is computed from the summed tracking time because this
    is the best estimate of sustained benchmark performance. Batch-to-batch
    scatter is also reported because it shows warm-up effects, scheduling
    jitter, and GPU launch variability.
    """
    batch_times = np.array(timing["batch_times"], dtype=float)
    batch_particles = np.array(timing["batch_particles"], dtype=float)
    time_per_particle = batch_times / batch_particles
    batch_rates = batch_particles / batch_times

    n_batches = timing["n_batches"]
    if n_batches > 1:
        time_per_particle_sem = (
            np.std(time_per_particle, ddof=1) / np.sqrt(n_batches)
        )
        batch_rate_sem = np.std(batch_rates, ddof=1) / np.sqrt(n_batches)
    else:
        time_per_particle_sem = 0.0
        batch_rate_sem = 0.0

    return {
        "throughput": timing["n_particles"] / timing["total"],
        "batch_rate_mean": np.mean(batch_rates),
        "batch_rate_sem": batch_rate_sem,
        "time_per_particle": timing["total"] / timing["n_particles"],
        "time_per_particle_mean": np.mean(time_per_particle),
        "time_per_particle_sem": time_per_particle_sem,
        "mean_batch_time": timing["total"] / timing["n_batches"],
    }


def stream_statistics(case, trackers, bins):
    """
    Run the full benchmark and fill the streaming accumulators.

    Each mode is run in batches. The line printed for each batch reports the
    cumulative particle count, the tracked batch time, and the corresponding
    rate in particles per second. The rate is per tracked batch, while the final
    summary reports the rate over all batches in that mode.
    """
    acc = make_accumulators(bins)

    for mode in MODES_TO_RUN:
        radiation_flag = RADIATION_FLAGS[mode]
        n_done = 0

        while n_done < N_PARTICLES:
            n_batch = min(BATCH_SIZE, N_PARTICLES - n_done)
            values = track_batch(
                case, trackers[mode], radiation_flag, n_batch, n_done)

            timing = acc[mode]["timing"]
            timing["n_batches"] += 1
            timing["n_particles"] += n_batch
            timing["total"] += values["t_track"]
            timing["min"] = min(timing["min"], values["t_track"])
            timing["max"] = max(timing["max"], values["t_track"])
            timing["batch_times"].append(values["t_track"])
            timing["batch_particles"].append(n_batch)

            for key, _label in PLOT_SPECS:
                update_histogram(acc[mode][key], values[key], bins[key])
                update_moments(acc[mode][key], values[key])

            n_done += n_batch
            print(
                f"{case['label']:11s} {mode:17s}"
                f" {n_done:10d}/{N_PARTICLES}"
                f" batch={values['t_track']:.6f} s"
                f" rate={n_batch / values['t_track']:.6e} particles/s"
            )

    return acc


################################################################################
# Reporting helpers
################################################################################

def print_configuration():
    """
    Print the benchmark setup before any heavy tracking starts.

    This makes saved terminal output self-contained enough to interpret later,
    especially when comparing CPU and GPU runs or changing the number of bend
    slices.
    """
    print(f"context = {context}")
    print(f"context_is_cupy = {is_cupy_context(context)}")
    if is_cupy_context(context):
        device_id, device_name = describe_cupy_device()
        print(f"cupy_device_id = {device_id}")
        print(f"cupy_device_name = {device_name}")
    print(f"compile_workdir = {COMPILE_WORKDIR}")
    print("modes:")
    for mode in MODES_TO_RUN:
        print(
            f"  {mode:17s}"
            f" flag={RADIATION_FLAGS[mode]}"
            f"  {MODE_DESCRIPTIONS[mode]}"
        )
    print(f"n_particles = {N_PARTICLES}")
    print(f"batch_size = {BATCH_SIZE}")
    print(f"pilot_particles = {PILOT_PARTICLES}")
    print(f"n_slices = {N_SLICES}")
    print("cases:")
    for case in CASES:
        print(
            f"  {case['label']:11s}"
            f" p0c={case['p0c']:.6e} eV"
            f" length={case['length']:.6e} m"
            f" angle={case['angle']:.6e} rad"
            f" <N_gamma>/slice={expected_average_photon_count(case):.6e}"
        )


def print_summary(case, acc):
    """
    Print throughput and compact radiation-kick diagnostics for each mode.

    The timing columns are the benchmark result. The radiation-kick columns are
    quick checks that the selected modes have the expected qualitative
    behaviour: no radiation should stay near zero, mean radiation should have
    negligible RMS, and the two quantum modes should have comparable stochastic
    spreads.
    """
    print()
    print(
        "Absolute timing and radiation-kick summary: "
        f"{case['name']} "
        rf"(<N_gamma>/slice = {expected_average_photon_count(case):.6e})"
    )
    print(
        "mode                  flag     total[s]      us/particle"
        "       sem[us]      particles/s    batch sem"
        "       <ddelta>    rms(ddelta)     clipped"
    )

    for mode in MODES_TO_RUN:
        timing = acc[mode]["timing"]
        metrics = timing_metrics(timing)
        mean_ddelta, rms_ddelta = mean_and_rms(acc[mode]["ddelta"])
        clipped = sum(
            acc[mode][key]["below"] + acc[mode][key]["above"]
            for key, _label in PLOT_SPECS
        )

        print(
            f"{mode:20s}"
            f" {RADIATION_FLAGS[mode]:4d}"
            f" {timing['total']:12.6f}"
            f" {1e6 * metrics['time_per_particle']:16.6f}"
            f" {1e6 * metrics['time_per_particle_sem']:13.6f}"
            f" {metrics['throughput']:16.6e}"
            f" {metrics['batch_rate_sem']:12.5e}"
            f" {mean_ddelta:13.6e}"
            f" {rms_ddelta:13.6e}"
            f" {clipped:11d}"
        )


def print_relative_timing(case, acc):
    """
    Print speed ratios relative to the no-radiation and quantum baselines.

    The no-radiation ratio shows how much overhead each radiation model adds to
    plain tracking. The quantum ratio is the direct algorithmic comparison for
    the proposed efficient mode.
    """
    print()
    print(f"Relative timing: {case['name']}")
    print(
        "mode                  speed/no-rad     time/no-rad"
        "   speed/quantum   time/quantum"
    )

    no_rad = acc.get("no-radiation")
    quantum = acc.get("quantum")
    no_rad_time = None
    quantum_time = None
    if no_rad is not None:
        no_rad_time = timing_metrics(no_rad["timing"])["time_per_particle"]
    if quantum is not None:
        quantum_time = timing_metrics(quantum["timing"])["time_per_particle"]

    for mode in MODES_TO_RUN:
        mode_time = timing_metrics(acc[mode]["timing"])["time_per_particle"]

        if no_rad_time is None:
            speed_no_rad = np.nan
            time_no_rad = np.nan
        else:
            speed_no_rad = no_rad_time / mode_time
            time_no_rad = mode_time / no_rad_time

        if quantum_time is None:
            speed_quantum = np.nan
            time_quantum = np.nan
        else:
            speed_quantum = quantum_time / mode_time
            time_quantum = mode_time / quantum_time

        print(
            f"{mode:20s}"
            f" {speed_no_rad:14.6f}"
            f" {time_no_rad:15.6f}"
            f" {speed_quantum:15.6f}"
            f" {time_quantum:14.6f}"
        )


def print_quantum_efficient_physics_comparison(case, acc):
    """
    Print a compact comparison of efficient mode against photon-by-photon mode.

    This is only a benchmark-level sanity check. A discrepancy here should be
    investigated with examples 013 and 014, which use more targeted plots and
    validation statistics.
    """
    if "quantum" not in acc or "quantum-efficient" not in acc:
        return

    print()
    print(
        "quantum-efficient radiation kicks relative to quantum: "
        f"{case['name']}"
    )
    print("observable        mean rel.diff    rms rel.diff    max CDF abs.diff")

    for key, _label in PLOT_SPECS:
        q_mean, q_rms = mean_and_rms(acc["quantum"][key])
        e_mean, e_rms = mean_and_rms(acc["quantum-efficient"][key])
        q_cdf = (
            acc["quantum"][key]["below"]
            + np.cumsum(acc["quantum"][key]["hist"])
        ) / acc["quantum"][key]["n"]
        e_cdf = (
            acc["quantum-efficient"][key]["below"]
            + np.cumsum(acc["quantum-efficient"][key]["hist"])
        ) / acc["quantum-efficient"][key]["n"]

        print(
            f"{key:12s}"
            f" {(e_mean / q_mean - 1):+15.6e}"
            f" {(e_rms / q_rms - 1):+15.6e}"
            f" {np.max(np.abs(e_cdf - q_cdf)):18.6e}"
        )


################################################################################
# Plotting
################################################################################

def plot_timing_bars(results):
    """
    Plot timing comparisons with one column per benchmark case.

    Two figures are produced. The first shows sustained throughput, where higher
    is faster. The second shows time per particle, where lower is faster. Each
    figure has one column for the low-lambda case and one column for the
    high-lambda case. Error bars are standard errors of the batch-level
    estimates, so they reflect repeatability during this run rather than physics
    uncertainty.
    """
    import matplotlib.pyplot as plt

    labels = MODES_TO_RUN
    x_pos = np.arange(len(labels))

    plt.close("benchmark throughput")
    plt.close("benchmark time per particle")
    fig_throughput, axes_throughput = plt.subplots(
        1, len(CASES), figsize=(6.5 * len(CASES), 4.8),
        num="benchmark throughput", squeeze=False)
    fig_time, axes_time = plt.subplots(
        1, len(CASES), figsize=(6.5 * len(CASES), 4.8),
        num="benchmark time per particle", squeeze=False)

    for i_case, case in enumerate(CASES):
        acc = results[case["label"]]["acc"]
        metrics = [timing_metrics(acc[mode]["timing"]) for mode in labels]

        throughputs = np.array([item["throughput"] for item in metrics])
        throughput_errors = np.array([
            item["batch_rate_sem"] for item in metrics
        ])
        us_per_particle = np.array([
            1e6 * item["time_per_particle"] for item in metrics
        ])
        us_per_particle_errors = np.array([
            1e6 * item["time_per_particle_sem"] for item in metrics
        ])

        lambda_synrad = expected_average_photon_count(case)
        ax = axes_throughput[0, i_case]
        ax.bar(x_pos, throughputs, yerr=throughput_errors, capsize=4)
        ax.set_title(
            f"{case['label']}\n"
            rf"$\langle N_\gamma\rangle = {lambda_synrad:.3g}$ per slice")
        ax.set_ylabel("Throughput [particles/s]")
        ax.grid(True, axis="y")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=25, ha="right")

        ax = axes_time[0, i_case]
        ax.bar(
            x_pos, us_per_particle, yerr=us_per_particle_errors, capsize=4)
        ax.set_title(
            f"{case['label']}\n"
            rf"$\langle N_\gamma\rangle = {lambda_synrad:.3g}$ per slice")
        ax.set_ylabel(r"Tracking time [$\mu$s/particle]")
        ax.grid(True, axis="y")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=25, ha="right")

    fig_throughput.suptitle(
        "Synchrotron-radiation tracking throughput; higher is faster")
    fig_time.suptitle(
        "Synchrotron-radiation tracking time per particle; lower is faster")
    fig_throughput.tight_layout()
    fig_time.tight_layout()

################################################################################
# Run
################################################################################

print_configuration()

results = {}

for case in CASES:
    print()
    print("=" * 80)
    print(case["name"])
    print("=" * 80)
    print(f"p0c = {case['p0c']:.6e} eV")
    print(f"length = {case['length']:.6e} m")
    print(f"angle = {case['angle']:.6e} rad")
    print(
        "expected photons per slice = "
        f"{expected_average_photon_count(case):.6e}")

    trackers = {
        mode: make_tracker(case, RADIATION_FLAGS[mode])
        for mode in MODES_TO_RUN}

    pilot   = collect_pilot(case, trackers)
    bins    = make_bins(pilot)
    del pilot

    acc = stream_statistics(case, trackers, bins)
    results[case["label"]] = {
        "case": case,
        "bins": bins,
        "acc": acc}

    print_summary(case, acc)
    print_relative_timing(case, acc)
    print_quantum_efficient_physics_comparison(case, acc)

plot_timing_bars(results)
plt.show()
