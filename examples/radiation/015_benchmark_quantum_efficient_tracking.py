# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import os
import tempfile
import time

import numpy as np

import xobjects as xo
from xobjects.context import get_context_from_string, get_user_context
import xpart as xp
import xtrack as xt


# ********************************************************************************
# Purpose
# ********************************************************************************

# This example compares the tracking-level behavior of the three synchrotron
# radiation modes:
#
#   mean              radiation_flag = 1
#   quantum           radiation_flag = 2
#   quantum-efficient radiation_flag = 3
#
# It is intended to be run interactively on CPU and GPU machines. The same file
# can be used with:
#
#   XOBJECTS_USER_CONTEXT=ContextCpu:auto python 015_benchmark_...
#   XOBJECTS_USER_CONTEXT=ContextCupy:0  python 015_benchmark_...
#
# The printed timing measures only the tracking call, not particle construction
# or host/device transfer for analysis.


# ********************************************************************************
# Configuration
# ********************************************************************************

context_name = os.environ.get("XOBJECTS_USER_CONTEXT", None)
modes_to_run = os.environ.get(
    "MODES", "mean,quantum,quantum-efficient").split(",")

n_particles = int(os.environ.get("N_PARTICLES", "1_000_000"))
batch_size = int(os.environ.get("BATCH_SIZE", "200_000"))
pilot_particles = int(os.environ.get("PILOT_PARTICLES", "200_000"))

n_slices = int(os.environ.get("N_SLICES", "1"))
p0c = float(os.environ.get("P0C", "182.5e9"))
length = float(os.environ.get("BEND_LENGTH", "22.653765579198428"))
angle = float(os.environ.get("BEND_ANGLE", "0.0022799344662676477"))

initial_px = float(os.environ.get("INITIAL_PX", "1e-4"))
initial_py = float(os.environ.get("INITIAL_PY", "-1e-4"))
initial_delta = float(os.environ.get("INITIAL_DELTA", "0.0"))

n_bins = int(os.environ.get("N_BINS", "400"))
tail_probability = float(os.environ.get("TAIL_PROBABILITY", "1e-6"))
ddelta_low_sigmas = float(os.environ.get("DDELTA_LOW_SIGMAS", "8.0"))

make_plots = int(os.environ.get("MAKE_PLOTS", "0"))

# CFFI CPU kernel compilation can accidentally parse the repository
# pyproject.toml if the process runs from the repository root. Use a temporary
# compile directory by default. Set COMPILE_WORKDIR=. to disable this.
compile_workdir = os.environ.get("COMPILE_WORKDIR", "auto")


# ********************************************************************************
# Mode and Context Setup
# ********************************************************************************

radiation_flags = {
    "mean": 1,
    "quantum": 2,
    "quantum-efficient": 3,
}

plot_specs = [
    ("dpx", r"$\Delta p_x$"),
    ("dpy", r"$\Delta p_y$"),
    ("ddelta", r"$\Delta\delta$"),
]

modes_to_run = [mode.strip() for mode in modes_to_run if mode.strip()]
unknown_modes = sorted(set(modes_to_run) - set(radiation_flags))
if unknown_modes:
    raise ValueError(f"Unknown modes: {unknown_modes}")

if compile_workdir == "auto":
    compile_workdir = tempfile.mkdtemp(prefix="xtrack-synrad-kernels-")

if compile_workdir:
    os.makedirs(compile_workdir, exist_ok=True)
    os.chdir(compile_workdir)

if context_name:
    context = get_context_from_string(context_name)
else:
    context = get_user_context()


# ********************************************************************************
# Tracking Helpers
# ********************************************************************************

def make_particles(n_batch, i_start):
    """Build a particle batch with identical initial conditions."""
    return xp.Particles(
        _context=context,
        p0c=p0c,
        mass0=xp.ELECTRON_MASS_EV,
        q0=-1,
        particle_id=np.arange(i_start, i_start + n_batch),
        x=np.zeros(n_batch),
        px=initial_px * np.ones(n_batch),
        y=np.zeros(n_batch),
        py=initial_py * np.ones(n_batch),
        delta=initial_delta * np.ones(n_batch),
    )


def make_tracker(radiation_flag):
    """Build either one bend or a sliced line of identical bend pieces."""
    if n_slices == 1:
        return xt.Bend(
            _context=context,
            length=length,
            angle=angle,
            k0_from_h=True,
            radiation_flag=radiation_flag,
        )

    elements = [
        xt.Bend(
            length=length / n_slices,
            angle=angle / n_slices,
            k0_from_h=True,
            radiation_flag=radiation_flag,
        )
        for _ in range(n_slices)
    ]
    line = xt.Line(elements=elements)
    line.build_tracker(_context=context)
    return line


def track_batch(tracker, radiation_flag, n_batch, i_start):
    """Track one batch and return host-side coordinate changes."""
    particles = make_particles(n_batch, i_start)
    if radiation_flag in (2, 3):
        particles._init_random_number_generator()

    t_start = time.perf_counter()
    tracker.track(particles)
    context.synchronize()
    t_track = time.perf_counter() - t_start

    return {
        "dpx": context.nparray_from_context_array(particles.px) - initial_px,
        "dpy": context.nparray_from_context_array(particles.py) - initial_py,
        "ddelta": (
            context.nparray_from_context_array(particles.delta)
            - initial_delta
        ),
        "t_track": t_track,
    }


# ********************************************************************************
# Histogram and Statistics Helpers
# ********************************************************************************

def collect_pilot(trackers):
    """Collect a smaller sample used only to choose common histogram bins."""
    pilot = {mode: {key: [] for key, _ in plot_specs} for mode in modes_to_run}
    n_pilot = min(pilot_particles, n_particles)

    for mode in modes_to_run:
        radiation_flag = radiation_flags[mode]
        n_done = 0
        while n_done < n_pilot:
            n_batch = min(batch_size, n_pilot - n_done)
            values = track_batch(trackers[mode], radiation_flag, n_batch, n_done)
            for key in pilot[mode]:
                pilot[mode][key].append(values[key])
            n_done += n_batch

        for key in pilot[mode]:
            pilot[mode][key] = np.concatenate(pilot[mode][key])

    return pilot


def make_bins(pilot):
    """Choose common bins from stochastic modes for fair CDF comparison."""
    bins = {}
    stochastic_modes = [mm for mm in modes_to_run if mm != "mean"]
    if not stochastic_modes:
        stochastic_modes = modes_to_run

    for key, _label in plot_specs:
        all_values = np.concatenate([pilot[mode][key]
                                     for mode in stochastic_modes])
        if key == "ddelta":
            mean = np.mean(all_values)
            sigma = np.std(all_values)
            lo = min(
                np.quantile(all_values, tail_probability),
                mean - ddelta_low_sigmas * sigma,
            )
            hi = np.quantile(all_values, 1 - 1e-5)
        else:
            lo, hi = np.quantile(all_values, [1e-5, 1 - 1e-5])

        if hi <= lo:
            hi = lo + max(abs(lo), 1.0) * 1e-12
        bins[key] = np.linspace(lo, hi, n_bins)

    return bins


def make_accumulators(bins):
    acc = {}
    for mode in modes_to_run:
        acc[mode] = {
            "timing": {
                "n_batches": 0,
                "n_particles": 0,
                "total": 0.0,
                "min": np.inf,
                "max": 0.0,
            }
        }
        for key, _label in plot_specs:
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
    hist, _ = np.histogram(values, bins=bin_edges)
    acc_item["hist"] += hist
    acc_item["below"] += np.count_nonzero(values < bin_edges[0])
    acc_item["above"] += np.count_nonzero(values >= bin_edges[-1])
    acc_item["n"] += values.size


def update_moments(acc_item, values):
    acc_item["sum"] += np.sum(values)
    acc_item["sum2"] += np.sum(values * values)
    acc_item["min"] = min(acc_item["min"], np.min(values))
    acc_item["max"] = max(acc_item["max"], np.max(values))


def mean_and_rms(acc_item):
    mean = acc_item["sum"] / acc_item["n"]
    variance = acc_item["sum2"] / acc_item["n"] - mean * mean
    return mean, np.sqrt(max(variance, 0.0))


def stream_statistics(trackers, bins):
    """Run the full benchmark and accumulate histograms without storing all data."""
    acc = make_accumulators(bins)

    for mode in modes_to_run:
        radiation_flag = radiation_flags[mode]
        n_done = 0

        while n_done < n_particles:
            n_batch = min(batch_size, n_particles - n_done)
            values = track_batch(trackers[mode], radiation_flag, n_batch, n_done)

            timing = acc[mode]["timing"]
            timing["n_batches"] += 1
            timing["n_particles"] += n_batch
            timing["total"] += values["t_track"]
            timing["min"] = min(timing["min"], values["t_track"])
            timing["max"] = max(timing["max"], values["t_track"])

            for key, _label in plot_specs:
                update_histogram(acc[mode][key], values[key], bins[key])
                update_moments(acc[mode][key], values[key])

            n_done += n_batch
            print(
                f"{mode:17s} {n_done:10d}/{n_particles}"
                f" batch={values['t_track']:.6f} s"
                f" rate={n_batch / values['t_track']:.6e} particles/s"
            )

    return acc


def print_summary(acc):
    print()
    print("Tracking summary")
    print(
        "mode                 particles/s     total[s]"
        "       <ddelta>    rms(ddelta)"
        "       <dpx>        <dpy>"
    )

    for mode in modes_to_run:
        timing = acc[mode]["timing"]
        throughput = timing["n_particles"] / timing["total"]
        mean_ddelta, rms_ddelta = mean_and_rms(acc[mode]["ddelta"])
        mean_dpx, _ = mean_and_rms(acc[mode]["dpx"])
        mean_dpy, _ = mean_and_rms(acc[mode]["dpy"])

        print(
            f"{mode:20s}"
            f" {throughput:12.6e}"
            f" {timing['total']:12.6f}"
            f" {mean_ddelta:13.6e}"
            f" {rms_ddelta:13.6e}"
            f" {mean_dpx:12.5e}"
            f" {mean_dpy:12.5e}"
        )


def print_quantum_efficient_comparison(acc):
    if "quantum" not in acc or "quantum-efficient" not in acc:
        return

    print()
    print("quantum-efficient relative to quantum")
    print("observable        mean rel.diff    rms rel.diff    max CDF abs.diff")

    for key, _label in plot_specs:
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


# ********************************************************************************
# Plotting
# ********************************************************************************

def plot_results(acc, bins):
    import matplotlib.pyplot as plt

    plot_modes = [mode for mode in modes_to_run if mode != "mean"]
    if not plot_modes:
        plot_modes = modes_to_run

    plt.close("all")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig_cdf, axes_cdf = plt.subplots(1, 3, figsize=(14, 4.5))

    for ax, ax_cdf, (key, label) in zip(axes, axes_cdf, plot_specs):
        bin_edges = bins[key]
        widths = np.diff(bin_edges)

        for mode in plot_modes:
            item = acc[mode][key]
            density = item["hist"] / item["n"] / widths
            cdf = (item["below"] + np.cumsum(item["hist"])) / item["n"]
            ax.step(bin_edges[1:], density, where="post", label=mode)
            ax_cdf.step(bin_edges[1:], cdf, where="post", label=mode)

        ax.set_xlabel(label)
        ax.set_ylabel("density")
        ax.set_yscale("log")
        ax.grid(True)
        ax.legend()

        ax_cdf.set_xlabel(label)
        ax_cdf.set_ylabel("CDF")
        ax_cdf.grid(True)
        ax_cdf.legend()

    fig.suptitle("Synchrotron radiation kick distributions")
    fig_cdf.suptitle("Synchrotron radiation kick CDFs")
    fig.tight_layout()
    fig_cdf.tight_layout()
    plt.show()


# ********************************************************************************
# Run
# ********************************************************************************

print(f"context = {context}")
print(f"compile_workdir = {compile_workdir}")
print(f"modes = {modes_to_run}")
print(f"n_particles = {n_particles}")
print(f"batch_size = {batch_size}")
print(f"n_slices = {n_slices}")

trackers = {
    mode: make_tracker(radiation_flags[mode])
    for mode in modes_to_run
}

pilot = collect_pilot(trackers)
bins = make_bins(pilot)
del pilot

acc = stream_statistics(trackers, bins)
print_summary(acc)
print_quantum_efficient_comparison(acc)

if make_plots:
    plot_results(acc, bins)

