# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
import importlib.util
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


################################################################################
# User parameters
################################################################################
FIXED_COUNTS        = [1, 2, 3, 7, 9, 16, 32]
LAMBDAS             = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 9.0, 30.0]
N_SAMPLES           = int(1E6)
MAX_POWER           = 256
PLOT_COUNTS         = [1, 2, 4, 8, 32]
PLOT_LAMBDAS_LOW    = [0.01, 0.03, 0.1, 0.3]
PLOT_LAMBDAS_HIGH   = [1.0, 3.0, 9.0, 30.0]

SEED                = 12345

# The mean check is compared against the expected Monte Carlo uncertainty from
# the two independent samples. The KS check compares full empirical CDFs.
MEAN_Z_LIMIT        = 5.0
KS_LIMIT_FACTOR     = 1.95

################################################################################
# Purpose
################################################################################

# This example validates the total-energy table sampler used by the
# ``quantum-efficient`` synchrotron-radiation mode before any tracking element
# is involved.
#
# The central physics question is whether replacing individual photon-energy
# sampling with precomputed compound tables changes the distribution of the
# total emitted energy. This file isolates that question from bends, particle
# coordinates, recoil application, and kernel timing.
#
# It compares two ways of sampling the total normalized energy emitted by a
# fixed number of photons:
#
#     1. brute force: sample every photon energy and sum them;
#     2. table mode: sample the precomputed inverse CDF for the total energy.
#
# It also compares the compound-Poisson process:
#
#     N_gamma ~ Poisson(lambda)
#     S = x_1 + ... + x_N_gamma
#
# The lambda scan deliberately includes values well below one. This is
# important for low-radiation regimes, where most integration steps emit no
# photon and the few nonzero events are dominated by ``N_gamma = 1``. A table
# error in the one-photon distribution can therefore matter even if the same
# table looks acceptable for high-photon-count FCC-ee tt-like cases.
#
# This file intentionally reuses the production table generator in
# ``xtrack/headers/_generate_synrad_total_energy_tables.py``. The point is to
# make this example a validation of the generated-table machinery itself,
# rather than an independent reimplementation of the same physics.
#
# The adopted generator builds all power-of-two runtime tables through the same
# high-accuracy offline route: a carefully integrated one-photon CDF followed
# by repeated deterministic self-convolution in quantile/CDF space. This
# example therefore validates the table-only runtime sampler against an
# independent brute-force photon-by-photon reference.
#
# This is complementary to ``013_compare_quantum_efficient.py``. Example 013
# asks whether the full Xtrack radiation kick agrees after tracking through a
# bend. This example asks the more basic question: before tracking, do the
# tables reproduce the total-energy random variables they are meant to sample?


################################################################################
# Helper Loading
################################################################################

########################################
# Load a Python module from a file
########################################
def load_module(path, module_name):
    """Load a Python source file as a module.

    Parameters
    ----------
    path : pathlib.Path
        Source file to load.
    module_name : str
        Temporary module name used for importlib.

    Returns
    -------
    module
        Imported module object.

    Notes
    -----
    The table generator is a repository script rather than a normal public API.
    Loading it directly keeps this example tied to the exact generator used to
    build the production C tables.
    """
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


here        = Path(__file__).parent
repo_root   = here.parents[1]

table_generator = load_module(
    repo_root / "xtrack" / "headers" / "_generate_synrad_total_energy_tables.py",
    "synrad_table_generator")


################################################################################
# Table Construction
################################################################################

########################################
# Build validation tables
########################################
def build_log_inverse_cdf_tables(max_power = 256):
    """Build the log inverse-CDF tables used by the efficient sampler.

    Parameters
    ----------
    max_power : int
        Largest power-of-two photon count to include.

    Returns
    -------
    u_grid : ndarray
        Probability grid used by the inverse-CDF tables.
    tables : dict[int, ndarray]
        Mapping from photon count to ``log(X_N)`` inverse-CDF table, where
        ``X_N`` is the total normalized energy emitted by ``N`` photons.

    Notes
    -----
    This function calls the production table generator. The validation
    therefore checks the same deterministic table-building path used by the
    branch. The returned values are converted to logarithms because the runtime
    C header stores ``log(X_N)`` and interpolates in log-space.
    """
    u_grid, quantile_tables = table_generator.build_total_energy_quantile_tables(
        u_grid=table_generator.make_probability_grid())

    tables = {
        nn: np.log(values)
        for nn, values in sorted(quantile_tables.items())
        if nn <= max_power
    }

    return u_grid, tables


################################################################################
# Sampling
################################################################################

########################################
# Sample one-photon spectrum
########################################
def sample_photon_energy_normalized(rng, n_samples):
    """Sample photon energies using the same rejection strategy as Xtrack.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random number generator.
    n_samples : int
        Number of one-photon samples to draw.

    Returns
    -------
    samples : ndarray
        Samples of ``x = E_gamma / E_c``.

    Notes
    -----
    This function deliberately calls ``table_generator.synrad`` rather than
    copying the SynRad approximation into this example. The sampling envelope
    follows the same structure as the preceding examples: a low-energy branch
    handles the singular behavior near zero and a high-energy exponential
    branch handles the tail.
    """
    xlow    = 1.0
    a1      = 2.149528241534391
    a2      = 1.770750801624037
    ratio   = 0.908250405131381

    samples = np.empty(n_samples)
    n_done = 0
    while n_done < n_samples:
        n_try = max(1024, int(1.35 * (n_samples - n_done)))
        use_low = rng.random(n_try) < ratio
        candidate = np.empty(n_try)
        approx = np.empty(n_try)

        u_low = rng.random(np.count_nonzero(use_low))
        candidate[use_low] = u_low**3
        approx[use_low] = a1 / np.maximum(
            u_low * u_low, np.finfo(float).tiny)

        u_high = rng.random(np.count_nonzero(~use_low))
        candidate[~use_low] = (
            xlow - np.log(np.maximum(u_high, np.finfo(float).tiny)))
        approx[~use_low] = a2 * np.exp(-candidate[~use_low])

        accepted = table_generator.synrad(candidate) >= approx * rng.random(n_try)
        accepted_values = candidate[accepted]
        n_take = min(accepted_values.size, n_samples - n_done)
        samples[n_done:n_done + n_take] = accepted_values[:n_take]
        n_done += n_take

    return samples


########################################
# Sample compound-Poisson reference
########################################
def sample_total_loss_normalized(rng, lam, n_samples):
    """Sample the brute-force compound-Poisson total energy loss.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random number generator.
    lam : float
        Mean photon count, ``lambda = <N_gamma>``.
    n_samples : int
        Number of independent total-loss samples.

    Returns
    -------
    values : ndarray
        Total normalized energy loss samples.

    Notes
    -----
    This is the low-level physics reference for the table sampler. It draws
    the photon count from a Poisson distribution and then samples every photon
    energy explicitly before summing the photons belonging to each event.
    """
    counts = rng.poisson(lam, size=n_samples)
    total_photons = int(np.sum(counts))
    values = np.zeros(n_samples)
    if total_photons == 0:
        return values

    photon_energy = sample_photon_energy_normalized(rng, total_photons)
    particle_index = np.repeat(np.arange(n_samples), counts)
    np.add.at(values, particle_index, photon_energy)
    return values


########################################
# Decompose photon count into tables
########################################
def decompose_photon_count(n_photons, tables):
    """Return the table chunks used to sample a photon count.

    Parameters
    ----------
    n_photons : int
        Photon count to represent.
    tables : dict[int, ndarray]
        Available inverse-CDF tables.

    Returns
    -------
    chunks : list[int]
        Photon-count chunks whose independent table samples are summed.

    Notes
    -----
    The adopted runtime algorithm represents arbitrary photon counts as sums
    of independent power-of-two table samples. The tables themselves are all
    generated by the same high-accuracy offline method.
    """
    if n_photons == 0:
        return []

    chunks = []
    n_left = int(n_photons)
    max_power_available = max(nn for nn in tables if nn & (nn - 1) == 0)

    while n_left > 0:
        chunk = 1 << (n_left.bit_length() - 1)
        chunk = min(chunk, max_power_available)
        chunks.append(chunk)
        n_left -= chunk

    return chunks


########################################
# Sample one inverse CDF
########################################
def sample_one_table_chunk(rng, u_grid, log_table):
    """Sample one total-energy table by inverse-CDF interpolation.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random number generator.
    u_grid : ndarray
        Probability grid for the inverse-CDF table.
    log_table : ndarray
        Stored ``log(X_N)`` inverse-CDF values.

    Returns
    -------
    value : float
        One sample of ``X_N``.

    Notes
    -----
    The generated tables store logarithms to preserve tail accuracy. Sampling
    therefore interpolates in log-space and exponentiates at the end, matching
    the intended runtime behavior.
    """
    u = rng.random()
    return np.exp(np.interp(u, u_grid, log_table))


########################################
# Sample fixed photon count from tables
########################################
def sample_fixed_count_from_tables(rng, n_photons, u_grid, tables):
    """Sample total normalized energy for a fixed photon count.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random number generator.
    n_photons : int
        Fixed photon count.
    u_grid : ndarray
        Probability grid for all inverse-CDF tables.
    tables : dict[int, ndarray]
        Available log inverse-CDF tables.
    Returns
    -------
    total : float
        Total normalized emitted energy.
    """
    total = 0.0
    for chunk in decompose_photon_count(n_photons, tables):
        total += sample_one_table_chunk(rng, u_grid, tables[chunk])
    return total


########################################
# Sample many fixed-count events from tables
########################################
def sample_many_fixed_count_from_tables(rng, n_photons, n_samples, u_grid,
                                        tables):
    """Sample many fixed-count total losses from the table method.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random number generator.
    n_photons : int
        Fixed photon count per event.
    n_samples : int
        Number of independent total-loss samples.
    u_grid : ndarray
        Probability grid for all inverse-CDF tables.
    tables : dict[int, ndarray]
        Available log inverse-CDF tables.
    Returns
    -------
    values : ndarray
        Table-sampled total normalized energy values.

    Notes
    -----
    This deliberately loops over events because it mirrors the runtime
    algorithmic structure: each event has a photon count, that count is
    decomposed into available tables, and the corresponding independent table
    samples are summed.
    """
    values = np.empty(n_samples)
    for ii in range(n_samples):
        values[ii] = sample_fixed_count_from_tables(
            rng, n_photons, u_grid, tables)
    return values


########################################
# Sample many fixed-count brute-force events
########################################
def sample_many_fixed_count_bruteforce(rng, n_photons, n_samples):
    """Sample fixed-count total losses by summing individual photons.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random number generator.
    n_photons : int
        Fixed photon count per event.
    n_samples : int
        Number of independent total-loss samples.

    Returns
    -------
    values : ndarray
        Brute-force total normalized energy values.
    """
    if n_photons == 0:
        return np.zeros(n_samples)

    photon_energy = sample_photon_energy_normalized(
        rng, n_photons * n_samples)
    return photon_energy.reshape(n_samples, n_photons).sum(axis=1)


########################################
# Sample compound-Poisson events from tables
########################################
def sample_compound_poisson_from_tables(rng, lam, n_samples, u_grid, tables):
    """Sample the compound-Poisson process using total-energy tables.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random number generator.
    lam : float
        Mean photon count.
    n_samples : int
        Number of independent total-loss samples.
    u_grid : ndarray
        Probability grid for all inverse-CDF tables.
    tables : dict[int, ndarray]
        Available log inverse-CDF tables.
    Returns
    -------
    values : ndarray
        Table-sampled total normalized energy values.
    counts : ndarray
        Poisson photon counts used for each event.
    n_chunks : ndarray
        Number of table chunks used for each event.

    Notes
    -----
    ``n_chunks`` is an algorithm diagnostic: it shows how much table work is
    needed after photon-count decomposition.
    """
    counts = rng.poisson(lam, size=n_samples)
    values = np.empty(n_samples)
    n_chunks = np.empty(n_samples, dtype=np.int64)

    for ii, count in enumerate(counts):
        chunks = decompose_photon_count(count, tables)
        n_chunks[ii] = len(chunks)
        values[ii] = sum(
            sample_one_table_chunk(rng, u_grid, tables[chunk])
            for chunk in chunks
        )

    return values, counts, n_chunks


################################################################################
# Reporting
################################################################################

########################################
# Print validation context
########################################
def print_validation_context():
    """Print the interpretation of the validation output.

    Notes
    -----
    The numerical comparisons are stochastic because both the brute-force
    reference and the table sampler are Monte Carlo samples. The printed
    checks therefore use tolerances tied to sample size rather than expecting
    exact agreement row by row.
    """
    ks_limit = KS_LIMIT_FACTOR * np.sqrt(2.0 / N_SAMPLES)

    print()
    print("What this example is checking")
    print("-----------------------------")
    print(
        "For each case, the brute-force reference samples every photon energy "
        "and sums the photons. The table sampler samples the same total "
        "energy directly from inverse-CDF tables.")
    print(
        "The lambda scan includes sub-unity values because low-radiation "
        "machines are dominated by zero- and one-photon events. In those "
        "cases the CDF has a real jump at S = 0 from P(N_gamma = 0).")
    print()
    print("A row is marked PASS when:")
    print(
        f"  mean_z < {MEAN_Z_LIMIT:g}, where mean_z is the mean difference "
        "divided by the expected Monte Carlo uncertainty;")
    print(
        f"  KS < {ks_limit:.4e}, the approximate two-sample CDF tolerance "
        f"for N_SAMPLES = {N_SAMPLES:g}.")
    print(
        "A row marked CHECK is not hidden or ignored: it means the table and "
        "reference distributions differ enough that the CDF residual plot "
        "should be inspected.")
    print()
    print(
        "The q99.9 column is a tail diagnostic, not a hard assert. With "
        f"{N_SAMPLES:g} samples, only about {0.001 * N_SAMPLES:g} samples "
        "sit above the 99.9% quantile, so it is visibly noisier than the "
        "mean, RMS, and KS checks.")
    print()


########################################
# Empirical CDF distance
########################################
def empirical_ks_distance(reference, candidate):
    """Compute the two-sample Kolmogorov-Smirnov distance.

    Parameters
    ----------
    reference : ndarray
        Brute-force photon-by-photon samples.
    candidate : ndarray
        Table-sampled values.

    Returns
    -------
    ks_distance : float
        Maximum absolute difference between the two empirical CDFs.
    """
    reference_sorted = np.sort(reference)
    candidate_sorted = np.sort(candidate)
    support = np.sort(np.concatenate([reference_sorted, candidate_sorted]))

    cdf_reference = (
        np.searchsorted(reference_sorted, support, side="right")
        / reference_sorted.size)
    cdf_candidate = (
        np.searchsorted(candidate_sorted, support, side="right")
        / candidate_sorted.size)

    return np.max(np.abs(cdf_candidate - cdf_reference))


########################################
# Compare two distributions
########################################
def compare_distributions(label, reference, candidate, mean_chunks=None):
    """Compute validation diagnostics for one reference/table comparison.

    Parameters
    ----------
    label : str
        Row label, usually a fixed photon count or a lambda value.
    reference : ndarray
        Brute-force photon-by-photon samples.
    candidate : ndarray
        Table-sampled values to compare against the reference.
    mean_chunks : float, optional
        Average number of table chunks used per event. This is relevant for
        compound-Poisson checks and for algorithm studies.

    Returns
    -------
    result : dict
        Samples and scalar diagnostics used for printing and plotting.

    Notes
    -----
    The mean checks total radiated power, the RMS checks excitation strength,
    the KS statistic checks the full CDF, and the 99.9% quantile gives a
    compact high-energy-tail diagnostic.
    """
    mean_ref = np.mean(reference)
    mean_cand = np.mean(candidate)
    rms_ref = np.std(reference)
    rms_cand = np.std(candidate)
    q999_ref = np.quantile(reference, 0.999)
    q999_cand = np.quantile(candidate, 0.999)
    ks_distance = empirical_ks_distance(reference, candidate)

    mean_uncertainty = np.sqrt(
        np.var(reference) / reference.size
        + np.var(candidate) / candidate.size)
    mean_z = abs(mean_cand - mean_ref) / mean_uncertainty
    ks_limit = KS_LIMIT_FACTOR * np.sqrt(
        (reference.size + candidate.size)
        / (reference.size * candidate.size))

    mean_pass = mean_z < MEAN_Z_LIMIT
    ks_pass = ks_distance < ks_limit
    passed = mean_pass and ks_pass

    return {
        "label": label,
        "reference": reference,
        "candidate": candidate,
        "mean_ref": mean_ref,
        "mean_cand": mean_cand,
        "mean_rel": mean_cand / mean_ref - 1.0,
        "mean_z": mean_z,
        "rms_ref": rms_ref,
        "rms_cand": rms_cand,
        "rms_rel": rms_cand / rms_ref - 1.0,
        "q999_ref": q999_ref,
        "q999_cand": q999_cand,
        "q999_rel": q999_cand / q999_ref - 1.0,
        "ks_distance": ks_distance,
        "ks_limit": ks_limit,
        "mean_chunks": mean_chunks,
        "passed": passed,
    }


########################################
# Print comparison table
########################################
def print_comparison_table(title, results, include_chunks=False):
    """Print a readable validation table.

    Parameters
    ----------
    title : str
        Section title.
    results : list[dict]
        Distribution-comparison dictionaries from
        :func:`compare_distributions`.
    include_chunks : bool
        Whether to include average table chunks per event.
    """
    print()
    print(title)
    print("-" * len(title))

    header = (
        f"{'case':>10s}"
        f" {'mean rel':>11s}"
        f" {'mean z':>8s}"
        f" {'rms rel':>11s}"
        f" {'q99.9 rel':>11s}"
        f" {'KS':>10s}"
        f" {'KS limit':>10s}")
    if include_chunks:
        header += f" {'<chunks>':>10s}"
    header += f" {'status':>8s}"
    print(header)

    for result in results:
        status = "PASS" if result["passed"] else "CHECK"
        row = (
            f"{result['label']:>10s}"
            f" {result['mean_rel']:+11.4e}"
            f" {result['mean_z']:8.3f}"
            f" {result['rms_rel']:+11.4e}"
            f" {result['q999_rel']:+11.4e}"
            f" {result['ks_distance']:10.4e}"
            f" {result['ks_limit']:10.4e}")
        if include_chunks:
            row += f" {result['mean_chunks']:10.4f}"
        row += f" {status:>8s}"
        print(row)


########################################
# Run fixed-count checks
########################################
def run_fixed_count_checks(u_grid, tables):
    """Compare fixed-photon-count total-energy distributions.

    Parameters
    ----------
    u_grid : ndarray
        Probability grid for all inverse-CDF tables.
    tables : dict[int, ndarray]
        Available log inverse-CDF tables.

    Notes
    -----
    This is the most direct test of the table construction. For each requested
    fixed count ``N``, the table sampler should reproduce the distribution of
    a brute-force sum of ``N`` independent synchrotron-radiation photons.
    """
    rng_ref = np.random.default_rng(SEED)
    rng_table = np.random.default_rng(SEED + 1)
    results = []

    for count in FIXED_COUNTS:
        reference = sample_many_fixed_count_bruteforce(
            rng_ref, count, N_SAMPLES)
        candidate = sample_many_fixed_count_from_tables(
            rng_table, count, N_SAMPLES, u_grid, tables)
        results.append(compare_distributions(str(count), reference, candidate))

    print_comparison_table("Fixed photon count checks", results)

    return results


########################################
# Run compound-Poisson checks
########################################
def run_compound_poisson_checks(u_grid, tables):
    """Compare compound-Poisson total-energy distributions.

    Parameters
    ----------
    u_grid : ndarray
        Probability grid for all inverse-CDF tables.
    tables : dict[int, ndarray]
        Available log inverse-CDF tables.

    Notes
    -----
    This test adds the Poisson photon-count process back on top of the
    fixed-count table sampler. It is the same low-level random variable that
    the tracking kernel should use before converting total energy loss into a
    particle kick.
    """
    rng_ref = np.random.default_rng(SEED + 2)
    rng_table = np.random.default_rng(SEED + 3)
    results = []

    for lam in LAMBDAS:
        reference = sample_total_loss_normalized(rng_ref, lam, N_SAMPLES)
        candidate, _counts, n_chunks = sample_compound_poisson_from_tables(
            rng_table, lam, N_SAMPLES, u_grid, tables)
        results.append(compare_distributions(
            f"{lam:g}",
            reference,
            candidate,
            mean_chunks=np.mean(n_chunks)))

    print_comparison_table(
        "Compound-Poisson checks", results, include_chunks=True)

    return results


################################################################################
# Plotting
################################################################################

########################################
# Evaluate empirical CDFs on common grid
########################################
def empirical_cdfs_on_common_grid(reference, candidate, n_grid=700):
    """Evaluate both empirical CDFs on a shared support.

    Parameters
    ----------
    reference : ndarray
        Brute-force photon-by-photon samples.
    candidate : ndarray
        Table-sampled values.
    n_grid : int
        Number of support points used for the residual curve.

    Returns
    -------
    x_grid : ndarray
        Total normalized energy values.
    cdf_reference : ndarray
        Brute-force empirical CDF evaluated on ``x_grid``.
    cdf_candidate : ndarray
        Table-sampled empirical CDF evaluated on ``x_grid``.
    residual : ndarray
        ``CDF(table) - CDF(reference)`` at each support point.

    Notes
    -----
    The support is taken from central quantiles of the combined sample. This
    avoids using plot resolution on extreme single-sample outliers while still
    showing the distribution region constrained by the Monte Carlo statistics.
    Evaluating both CDFs on the same grid also avoids misleading plots where
    two independent empirical CDF lines appear to stop at different positions
    simply because their finite-sample maxima differ.
    """
    combined = np.concatenate([reference, candidate])
    probabilities = np.linspace(1e-3, 1.0 - 1e-3, n_grid)
    x_grid = np.quantile(combined, probabilities)

    reference_sorted = np.sort(reference)
    candidate_sorted = np.sort(candidate)
    cdf_reference = (
        np.searchsorted(reference_sorted, x_grid, side="right")
        / reference_sorted.size)
    cdf_candidate = (
        np.searchsorted(candidate_sorted, x_grid, side="right")
        / candidate_sorted.size)

    return x_grid, cdf_reference, cdf_candidate, cdf_candidate - cdf_reference


########################################
# Select plotted cases
########################################
def select_results(results, selected_labels):
    """Select validation results to show in the plots.

    Parameters
    ----------
    results : list[dict]
        Result dictionaries from the validation checks.
    selected_labels : list
        Labels requested by the corresponding user parameter.

    Returns
    -------
    selected : list[dict]
        Results whose labels match the requested values.
    """
    label_set = {f"{label:g}" if isinstance(label, float) else str(label)
                 for label in selected_labels}
    return [result for result in results if result["label"] in label_set]


########################################
# Plot CDF comparisons
########################################
def plot_cdf_comparisons(results, selected_labels, title, label_prefix):
    """Plot CDF overlays and CDF residuals for selected cases.

    Parameters
    ----------
    results : list[dict]
        Result dictionaries from the validation checks.
    selected_labels : list
        Cases to include in the figure.
    title : str
        Figure title.
    label_prefix : str
        Prefix used in subplot titles, for example ``N`` or ``lambda``.

    Notes
    -----
    The top row overlays the brute-force and table empirical CDFs. Agreement
    there should be nearly indistinguishable by eye. The bottom row shows
    ``CDF(table) - CDF(reference)``, which makes small systematic differences
    visible without obscuring the main CDF comparison. Each residual panel has
    its own y-scale because the low-count cases can have much larger residuals
    than the high-count cases.
    """
    selected = select_results(results, selected_labels)
    if len(selected) == 0:
        return

    fig, axes = plt.subplots(
        2, len(selected), figsize=(4.0 * len(selected), 6.0),
        gridspec_kw={"height_ratios": [3, 1]})

    if len(selected) == 1:
        axes = axes.reshape(2, 1)

    for ii, result in enumerate(selected):
        ax_cdf = axes[0, ii]
        ax_res = axes[1, ii]

        x_grid, cdf_reference, cdf_candidate, residual = (
            empirical_cdfs_on_common_grid(
                result["reference"], result["candidate"]))

        ax_cdf.plot(
            x_grid, cdf_reference, linewidth=1.4, label="brute force")
        ax_cdf.plot(
            x_grid, cdf_candidate, linewidth=1.2, label="table")

        ax_res.plot(x_grid, residual, color="black", linewidth=1.2)
        ax_res.axhline(0.0, color="0.5", linewidth=0.8)
        ax_res.axhline(
            result["ks_limit"], color="0.6", linestyle="--", linewidth=0.8)
        ax_res.axhline(
            -result["ks_limit"], color="0.6", linestyle="--", linewidth=0.8)

        residual_limit = max(
            np.nanmax(np.abs(residual)),
            result["ks_limit"])
        ax_res.set_ylim(-1.15 * residual_limit, 1.15 * residual_limit)

        ax_cdf.set_title(f"{label_prefix} = {result['label']}")
        ax_cdf.set_ylim(-0.02, 1.02)
        ax_cdf.set_ylabel("CDF")
        ax_cdf.grid(True)
        ax_cdf.legend()

        ax_res.set_xlabel("Total normalized energy loss, S [1]")
        ax_res.set_ylabel(r"$\Delta$CDF")
        ax_res.grid(True)

    fig.suptitle(title)
    fig.tight_layout()


########################################
# Plot validation results
########################################
def plot_validation_results(fixed_results, compound_results):
    """Plot the selected low-level validation checks.

    Parameters
    ----------
    fixed_results : list[dict]
        Fixed-photon-count validation results.
    compound_results : list[dict]
        Compound-Poisson validation results.
    """
    plt.close("all")
    plot_cdf_comparisons(
        fixed_results,
        PLOT_COUNTS,
        "Fixed photon count total-energy CDFs",
        r"$N_\gamma$")
    plot_cdf_comparisons(
        compound_results,
        PLOT_LAMBDAS_LOW,
        "Compound-Poisson total-energy CDFs, low lambda",
        r"$\lambda$")
    plot_cdf_comparisons(
        compound_results,
        PLOT_LAMBDAS_HIGH,
        "Compound-Poisson total-energy CDFs, high lambda",
        r"$\lambda$")


################################################################################
# Run
################################################################################
print("Building deterministic inverse-CDF tables")
print(f"MAX_POWER = {MAX_POWER}")
print(f"N_SAMPLES = {N_SAMPLES}")

u_grid, tables = build_log_inverse_cdf_tables(max_power=MAX_POWER)

print(f"available table counts = {sorted(tables)}")
print()

print_validation_context()
fixed_results       = run_fixed_count_checks(u_grid, tables)
compound_results    = run_compound_poisson_checks(u_grid, tables)
plot_validation_results(fixed_results, compound_results)

plt.show()
