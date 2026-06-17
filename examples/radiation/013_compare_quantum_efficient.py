# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
import time

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import xobjects as xo
import xpart as xp
import xtrack as xt


################################################################################
# User parameters
################################################################################
CONTEXT = xo.ContextCpu()

N_PARTICLES         = int(1E8)
BATCH_SIZE          = int(1E6)
PILOT_PARTICLES     = int(1E6)

# The pilot sample is used only to choose plotting ranges. The full streamed
# sample is then accumulated without storing every particle in memory.
TAIL_PROBABILITY    = 1E-10
DDELTA_LOW_SIGMAS   = 8.0

P0C                 = 182.5e9
LENGTH              = 22.653765579198428
ANGLE               = 0.0022799344662676477
INITIAL_PX          = 1e-4
INITIAL_PY          = -1e-4
INITIAL_DELTA       = 0.0

MODES = {
    "quantum": 2,
    "quantum-efficient": 3}

PLOT_SPECS = [
    ("dpx", r"$\Delta p_x$"),
    ("dpy", r"$\Delta p_y$"),
    ("ddelta", r"$\Delta\delta$")]

################################################################################
# Purpose
################################################################################
# This example compares the existing photon-by-photon synchrotron-radiation
# mode, ``quantum``, against the new compound-table mode,
# ``quantum-efficient``.
#
# The physics question is whether the algorithmic change preserves the particle
# kicks produced by one representative FCC-ee tt-like bend. The efficient mode
# no longer samples every emitted photon, so the retained observables are the
# total radiation kick:
#
#     Delta p_x, Delta p_y, Delta delta
#
# The script streams many particles through the same bend, accumulates moments
# and histograms, and gives special attention to the lower tail of
# ``Delta delta`` because rare large energy-loss events are important for
# lifetime studies.

################################################################################
# Lattice and particle construction
################################################################################
def make_particles(n_batch, i_start):
    """Build a reproducible batch of identical test particles.

    Parameters
    ----------
    n_batch : int
        Number of particles in this batch.
    i_start : int
        First particle identifier. Consecutive identifiers make the batches
        easier to inspect and avoid reusing the same particle IDs.

    Returns
    -------
    particles : xpart.Particles
        Electron particles on the configured context. All particles start from
        the same coordinates so that the distributions after tracking are due
        to the stochastic radiation model rather than the initial beam
        distribution.
    """
    return xp.Particles(
        _context    = CONTEXT,
        p0c         = P0C,
        mass0       = xp.ELECTRON_MASS_EV,
        q0          = -1,
        particle_id = np.arange(i_start, i_start + n_batch),
        x           = np.zeros(n_batch),
        px          = INITIAL_PX * np.ones(n_batch),
        y           = np.zeros(n_batch),
        py          = INITIAL_PY * np.ones(n_batch),
        delta       = INITIAL_DELTA * np.ones(n_batch))


def make_bend(radiation_flag):
    """Build the single bend used for the comparison.

    Parameters
    ----------
    radiation_flag : int
        Xtrack radiation flag. In this branch, ``2`` is the existing
        photon-by-photon quantum mode and ``3`` is the proposed
        ``quantum-efficient`` mode.

    Returns
    -------
    bend : xtrack.Bend
        A standalone FCC-ee tt-like bend with the requested radiation mode.

    Notes
    -----
    This is deliberately a single-element test. It removes lattice-level
    complications and focuses on whether the two radiation algorithms produce
    statistically equivalent one-kick distributions.
    """
    return xt.Bend(
        _context        = CONTEXT,
        length          = LENGTH,
        angle           = ANGLE,
        k0_from_h       = True,
        radiation_flag  = radiation_flag)


def track_batch(bend, radiation_flag, n_batch, i_start):
    """Track one batch and return the radiation-induced coordinate changes.

    Parameters
    ----------
    bend : xtrack.Bend
        Element through which the particles are tracked.
    radiation_flag : int
        Radiation mode used by ``bend``. Random-number generation is
        initialized for stochastic modes.
    n_batch : int
        Number of particles to track in this call.
    i_start : int
        First particle identifier for this batch.

    Returns
    -------
    values : dict
        Arrays for ``dpx``, ``dpy``, and ``ddelta`` together with the elapsed
        tracking time. The coordinate differences are measured relative to the
        configured initial coordinates.
    """
    particles = make_particles(n_batch, i_start)
    particles._init_random_number_generator()

    t_start = time.perf_counter()
    bend.track(particles)
    t_track = time.perf_counter() - t_start

    return {
        "dpx":      CONTEXT.nparray_from_context_array(particles.px) - INITIAL_PX,
        "dpy":      CONTEXT.nparray_from_context_array(particles.py) - INITIAL_PY,
        "ddelta":   CONTEXT.nparray_from_context_array(particles.delta) - INITIAL_DELTA,
        "t_track":  t_track}

################################################################################
# Pilot pass and histogram ranges
################################################################################

########################################
# Collect pilot distributions
########################################
def collect_pilot():
    """Track a smaller pilot sample to choose stable histogram ranges.

    Returns
    -------
    pilot : dict
        Nested dictionary containing arrays of ``dpx``, ``dpy``, and
        ``ddelta`` for each radiation mode.

    Notes
    -----
    The full comparison can use millions of particles. Storing every value
    from the full run is unnecessary and memory expensive. Instead, this pilot
    pass estimates useful plotting ranges, after which the full run streams
    through fixed histogram bins.
    """
    pilot   = {mode: {key: [] for key, _ in PLOT_SPECS} for mode in MODES}
    n_pilot = min(PILOT_PARTICLES, N_PARTICLES)

    for mode, radiation_flag in MODES.items():
        bend    = make_bend(radiation_flag)
        n_done  = 0

        with tqdm(
                total       = n_pilot,
                desc        = f"{mode:17s} pilot",
                unit        = "particle",
                unit_scale  = True) as progress:
            while n_done < n_pilot:
                n_batch = min(BATCH_SIZE, n_pilot - n_done)
                values  = track_batch(bend, radiation_flag, n_batch, n_done)
                for key in pilot[mode]:
                    pilot[mode][key].append(values[key])

                n_done += n_batch
                progress.update(n_batch)
                progress.set_postfix(
                    batch_s=f"{values['t_track']:.3f}",
                    batch_rate=f"{n_batch / values['t_track']:.3e} p/s")

        for key in pilot[mode]:
            pilot[mode][key] = np.concatenate(pilot[mode][key])

    return pilot


########################################
# Choose plotting bins
########################################
def make_bins(pilot):
    """Define common histogram bins for both radiation modes.

    Parameters
    ----------
    pilot : dict
        Pilot distributions returned by :func:`collect_pilot`.

    Returns
    -------
    bins : dict
        Histogram bin edges for ``dpx``, ``dpy``, ``ddelta``, and the
        dedicated lower-tail ``ddelta_tail`` plot.

    Notes
    -----
    Common bins are essential for a meaningful visual comparison. The
    ``ddelta`` range is extended into the lower tail because large negative
    energy kicks correspond to rare high-radiation events and are one of the
    main physics risks for a table-based sampler.
    """
    bins = {}

    for key, _ in PLOT_SPECS:
        all_values = np.concatenate([pilot[mode][key] for mode in MODES])
        if key == "ddelta":
            mean = np.mean(all_values)
            sigma = np.std(all_values)
            lo_quantile = np.quantile(all_values, TAIL_PROBABILITY)
            lo_sigma = mean - DDELTA_LOW_SIGMAS * sigma
            lo = min(lo_quantile, lo_sigma)
            hi = np.quantile(all_values, 1 - 1e-5)
            bins[key] = np.linspace(lo, hi, 500)
        else:
            lo, hi = np.quantile(all_values, [1e-5, 1 - 1e-5])
            bins[key] = np.linspace(lo, hi, 300)

    all_ddelta  = np.concatenate([pilot[mode]["ddelta"] for mode in MODES])
    
    mean    = np.mean(all_ddelta)
    sigma   = np.std(all_ddelta)
    
    tail_lo_quantile    = np.quantile(all_ddelta, TAIL_PROBABILITY)
    tail_lo_sigma       = mean - DDELTA_LOW_SIGMAS * sigma
    
    tail_lo = min(tail_lo_quantile, tail_lo_sigma)
    tail_hi = np.quantile(all_ddelta, 5e-3)
    
    bins["ddelta_tail"] = np.linspace(tail_lo, tail_hi, 700)

    return bins


################################################################################
# Streaming statistics
################################################################################

########################################
# Allocate accumulators
########################################
def make_accumulators(bins):
    """Allocate histograms, moments, and timing accumulators.

    Parameters
    ----------
    bins : dict
        Histogram bin edges returned by :func:`make_bins`.

    Returns
    -------
    acc : dict
        Nested accumulator dictionary indexed first by radiation mode and then
        by observable.

    Notes
    -----
    The accumulator stores only histogram counts and running sums. This keeps
    the memory footprint small while still allowing millions of particles to
    contribute to smooth distributions and stable moments.
    """
    acc = {}
    for mode in MODES:
        acc[mode] = {}
        for key, _ in PLOT_SPECS:
            acc[mode][key] = {
                "hist":     np.zeros(bins[key].size - 1, dtype=np.int64),
                "below":    0,
                "above":    0,
                "n":        0,
                "sum":      0.0,
                "sum2":     0.0,
                "min":      np.inf,
                "max":      -np.inf}
        acc[mode]["ddelta_tail"] = {
            "hist":         np.zeros(bins["ddelta_tail"].size - 1, dtype=np.int64),
            "below":        0,
            "above":        0,
            "n":            0}
        acc[mode]["timing"] = {
            "n_batches":    0,
            "n_particles":  0,
            "total":        0.0,
            "min":          np.inf,
            "max":          0.0}
    return acc


########################################
# Update one histogram
########################################
def update_histogram(acc_item, values, bin_edges):
    """Accumulate a batch into a fixed histogram.

    Parameters
    ----------
    acc_item : dict
        Histogram accumulator containing counts inside, below, and above the
        chosen bin range.
    values : ndarray
        Batch values to histogram.
    bin_edges : ndarray
        Histogram bin edges.

    Notes
    -----
    The underflow and overflow counters are kept because the bins are chosen
    from a finite pilot sample. They make it explicit if later batches produce
    values outside the displayed range.
    """
    hist, _ = np.histogram(values, bins=bin_edges)
    
    acc_item["hist"]    += hist
    acc_item["below"]   += np.count_nonzero(values < bin_edges[0])
    acc_item["above"]   += np.count_nonzero(values >= bin_edges[-1])
    acc_item["n"]       += values.size


########################################
# Update running moments
########################################
def update_moments(acc_item, values):
    """Accumulate first and second moments for one observable.

    Parameters
    ----------
    acc_item : dict
        Moment accumulator for a single observable.
    values : ndarray
        Batch values.

    Notes
    -----
    The mean checks the average radiation kick. The RMS checks the retained
    quantum excitation. Both are necessary but not sufficient, which is why
    the script also compares the full histograms and CDFs.
    """
    acc_item["sum"]     += np.sum(values)
    acc_item["sum2"]    += np.sum(values * values)
    acc_item["min"]     = min(acc_item["min"], np.min(values))
    acc_item["max"]     = max(acc_item["max"], np.max(values))


########################################
# Convert moments
########################################
def mean_and_rms(acc_item):
    """Return the mean and RMS from a running-moment accumulator.

    Parameters
    ----------
    acc_item : dict
        Accumulator containing ``n``, ``sum``, and ``sum2``.

    Returns
    -------
    mean : float
        Mean value.
    rms : float
        Standard deviation computed from the accumulated first and second
        moments.
    """
    mean        = acc_item["sum"] / acc_item["n"]
    variance    = acc_item["sum2"] / acc_item["n"] - mean * mean
    return mean, np.sqrt(max(variance, 0.0))

########################################
# Update timing statistics
########################################
def update_timing(timing, n_batch, t_track):
    """Accumulate tracking-time diagnostics.

    Parameters
    ----------
    timing : dict
        Timing accumulator for one radiation mode.
    n_batch : int
        Number of particles in the tracked batch.
    t_track : float
        Elapsed tracking time in seconds.
    """
    timing["n_batches"]     += 1
    timing["n_particles"]   += n_batch
    timing["total"]         += t_track
    timing["min"]           = min(timing["min"], t_track)
    timing["max"]           = max(timing["max"], t_track)

########################################
# Print one-mode summary
########################################
def print_summary(mode, mode_acc):
    """Print moment and timing diagnostics for one radiation mode.

    Parameters
    ----------
    mode : str
        Human-readable radiation-mode name.
    mode_acc : dict
        Accumulators for this mode.

    Notes
    -----
    The printed values are intentionally redundant with the plots. During
    interactive work they provide a fast check that the two modes have similar
    mean kicks, similar RMS kicks, and comparable or different timing before
    inspecting the figures in detail.
    """
    mean_dpx, rms_dpx       = mean_and_rms(mode_acc["dpx"])
    mean_dpy, rms_dpy       = mean_and_rms(mode_acc["dpy"])
    mean_ddelta, rms_ddelta = mean_and_rms(mode_acc["ddelta"])

    timing          = mode_acc["timing"]
    mean_batch_time = timing["total"] / timing["n_batches"]
    throughput      = timing["n_particles"] / timing["total"]

    print(
        f"{mode:17s}"
        f" <dpx>={mean_dpx: .6e}"
        f" rms(dpx)={rms_dpx: .6e}"
        f" <dpy>={mean_dpy: .6e}"
        f" rms(dpy)={rms_dpy: .6e}"
        f" <ddelta>={mean_ddelta: .6e}"
        f" rms(ddelta)={rms_ddelta: .6e}"
        f" min(ddelta)={mode_acc['ddelta']['min']: .6e}")
    print(
        f"{mode:17s}"
        f" track_time_total={timing['total']:.6f} s"
        f" mean_batch={mean_batch_time:.6f} s"
        f" min_batch={timing['min']:.6f} s"
        f" max_batch={timing['max']:.6f} s"
        f" throughput={throughput:.6e} particles/s")


########################################
# Track the full sample
########################################
def stream_statistics(bins):
    """Track all requested particles and accumulate comparison statistics.

    Parameters
    ----------
    bins : dict
        Histogram bin edges returned by :func:`make_bins`.

    Returns
    -------
    acc : dict
        Accumulated histograms, moments, tail histograms, and timing
        information for each radiation mode.

    Notes
    -----
    The function streams the particle ensemble in batches. This is important
    for large validation runs because the script can generate smooth
    distributions without keeping every tracked particle in memory.
    """
    acc = make_accumulators(bins)

    for mode, radiation_flag in MODES.items():
        bend = make_bend(radiation_flag)
        n_done = 0

        with tqdm(
                total=N_PARTICLES,
                desc=f"{mode:17s} track",
                unit="particle",
                unit_scale=True) as progress:
            while n_done < N_PARTICLES:
                n_batch = min(BATCH_SIZE, N_PARTICLES - n_done)
                values = track_batch(bend, radiation_flag, n_batch, n_done)
                update_timing(acc[mode]["timing"], n_batch, values["t_track"])

                for key, _ in PLOT_SPECS:
                    update_histogram(acc[mode][key], values[key], bins[key])
                    update_moments(acc[mode][key], values[key])

                update_histogram(
                    acc[mode]["ddelta_tail"],
                    values["ddelta"],
                    bins["ddelta_tail"])

                n_done += n_batch
                progress.update(n_batch)
                progress.set_postfix(
                    batch_s=f"{values['t_track']:.3f}",
                    batch_rate=f"{n_batch / values['t_track']:.3e} p/s")

        print_summary(mode, acc[mode])

    return acc


################################################################################
# Plotting
################################################################################

########################################
# Plot accumulated results
########################################
def plot_results(acc, bins):
    """Plot distributions, CDFs, and the lower energy-loss tail.

    Parameters
    ----------
    acc : dict
        Accumulators returned by :func:`stream_statistics`.
    bins : dict
        Histogram bin edges returned by :func:`make_bins`.

    Notes
    -----
    The density plots show the overall kick distributions. The CDF plots are
    often easier to compare in the tails because they suppress some histogram
    noise. Residual panels make small systematic differences visible without
    adding extra lines to the main comparison axes. The dedicated lower-tail
    CDF for ``Delta delta`` focuses on rare high-energy radiation events,
    which are the most important events for lifetime-sensitive validation.
    """
    plt.close("all")
    reference_mode = "quantum"
    efficient_mode = "quantum-efficient"

    fig, axes = plt.subplots(
        2, 3, figsize=(14, 6.5), sharex="col",
        gridspec_kw={"height_ratios": [3, 1]})
    fig_cdf, axes_cdf = plt.subplots(
        2, 3, figsize=(14, 6.5), sharex="col",
        gridspec_kw={"height_ratios": [3, 1]})

    for ii, (key, label) in enumerate(PLOT_SPECS):
        ax = axes[0, ii]
        ax_res = axes[1, ii]
        ax_cdf = axes_cdf[0, ii]
        ax_cdf_res = axes_cdf[1, ii]
        bin_edges = bins[key]
        widths = np.diff(bin_edges)
        densities = {}
        cdfs = {}

        for mode in MODES:
            item = acc[mode][key]
            density = item["hist"] / item["n"] / widths
            cdf = (item["below"] + np.cumsum(item["hist"])) / item["n"]
            densities[mode] = density
            cdfs[mode] = cdf

            ax.step(
                bin_edges[1:],
                density,
                where="post",
                linewidth=1.5,
                label=mode,
            )
            ax_cdf.step(
                bin_edges[1:],
                cdf,
                where="post",
                linewidth=1.5,
                label=mode,
            )

        density_ref = densities[reference_mode]
        density_eff = densities[efficient_mode]
        density_residual = np.full_like(density_ref, np.nan, dtype=float)
        populated = density_ref > 0.0
        density_residual[populated] = (
            (density_eff[populated] - density_ref[populated])
            / density_ref[populated])
        cdf_residual = cdfs[efficient_mode] - cdfs[reference_mode]

        ax_res.step(
            bin_edges[1:],
            density_residual,
            where="post",
            linewidth=1.2,
            color="black")
        ax_res.axhline(0.0, color="0.5", linewidth=0.8)

        ax_cdf_res.step(
            bin_edges[1:],
            cdf_residual,
            where="post",
            linewidth=1.2,
            color="black")
        ax_cdf_res.axhline(0.0, color="0.5", linewidth=0.8)

        ax.set_ylabel("density")
        ax.set_yscale("log")
        ax.grid(True)
        ax.legend()

        ax_res.set_xlabel(label)
        ax_res.set_ylabel(r"$\Delta\rho/\rho_q$")
        ax_res.grid(True)

        ax_cdf.set_ylabel("CDF")
        ax_cdf.grid(True)
        ax_cdf.legend()

        ax_cdf_res.set_xlabel(label)
        ax_cdf_res.set_ylabel(r"$\Delta$CDF")
        ax_cdf_res.grid(True)

    fig.suptitle(
        "Synchrotron radiation kick distributions in one FCC-ee tt-like bend")
    fig_cdf.suptitle(
        "Synchrotron radiation kick CDFs in one FCC-ee tt-like bend")
    fig.tight_layout()
    fig_cdf.tight_layout()

    fig_tail, (ax_tail, ax_tail_res) = plt.subplots(
        2, 1, figsize=(7, 6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]})
    tail_bins = bins["ddelta_tail"]
    tail_cdfs = {}

    for mode in MODES:
        item = acc[mode]["ddelta_tail"]
        cdf = (item["below"] + np.cumsum(item["hist"])) / item["n"]
        tail_cdfs[mode] = cdf
        ax_tail.step(
            tail_bins[1:],
            cdf,
            where       = "post",
            linewidth   = 1.5,
            label       = mode)

    tail_residual = tail_cdfs[efficient_mode] - tail_cdfs[reference_mode]
    ax_tail_res.step(
        tail_bins[1:],
        tail_residual,
        where       = "post",
        linewidth   = 1.2,
        color       = "black")
    ax_tail_res.axhline(0.0, color="0.5", linewidth=0.8)

    ax_tail.set_ylabel(r"$P(\Delta\delta \leq x)$")
    ax_tail.set_yscale("log")
    ax_tail.grid(True)
    ax_tail.legend()

    ax_tail_res.set_xlabel(r"$\Delta\delta$")
    ax_tail_res.set_ylabel(r"$\Delta$CDF")
    ax_tail_res.grid(True)

    fig_tail.suptitle("Lower-tail CDF of the energy kick")
    fig_tail.tight_layout()


################################################################################
# Run
################################################################################
pilot   = collect_pilot()
bins    = make_bins(pilot)
del pilot
acc     = stream_statistics(bins)
plot_results(acc, bins)

plt.show()
