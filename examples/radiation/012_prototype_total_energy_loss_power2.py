# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
import importlib.util
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

################################################################################
# User parameters
################################################################################
RNG_TABLES      = np.random.default_rng(12345)
RNG_REFERENCE   = np.random.default_rng(67890)
RNG_POWER2      = np.random.default_rng(24680)

# ``TEST_LAMBDAS`` drives the physics validation plots. These are mean photon
# counts in the compound-Poisson process:
#
#     N_gamma ~ Poisson(lambda)
#     S = x_1 + ... + x_N_gamma
#
# so this checks whether the power-of-two decomposition reproduces the total
# energy-loss distribution when the number of photons is itself random.
TEST_LAMBDAS    = np.array([2.86, 8.58, 20.0])
N_PARTICLES     = int(1E6)

# ``TIMING_COUNTS`` drives the algorithmic scaling plot. These are fixed photon
# counts:
#
#     S_N = x_1 + ... + x_N
#
# so this isolates the cost of replacing ``N`` individual photon samples with
# the greedy power-of-two table decomposition for the same fixed ``N``.
TIMING_COUNTS   = np.array([1, 2, 3, 5, 8, 13, 21, 32, 64, 128, 256])
TIMING_SAMPLES  = int(1E5)

################################################################################
# Purpose
################################################################################
# This example prototypes the idea behind the ``quantum-efficient``
# synchrotron-radiation mode.
#
# The photon-by-photon model samples:
#
#     N_gamma ~ Poisson(lambda)
#     S = x_1 + ... + x_N_gamma
#
# where each ``x_i = E_gamma / E_c`` is sampled from the synchrotron-radiation
# photon spectrum. This file tests a faster route: precompute inverse-CDF
# tables for fixed photon counts that are powers of two, then represent an
# arbitrary photon count as a sum of those powers of two.
#
# The tables in this prototype are empirical Monte Carlo tables. The production
# branch replaces them with deterministic convolution-generated tables in
# ``xtrack/headers/_generate_synrad_total_energy_tables.py``.

################################################################################
# Helper loading
################################################################################

########################################
# Load baseline SynRad sampler
########################################
def _load_synrad_helpers():
    """Load the baseline sampler from ``011_plot_total_energy_loss.py``.

    The helper module provides:

    - ``sample_photon_energy_normalized``: photon-by-photon samples of
      ``x = E_gamma / E_c``;
    - ``sample_total_loss_normalized``: the brute-force compound-Poisson
      reference process.

    Loading the file this way keeps this prototype close to the preceding
    example while avoiding package/import assumptions about the examples
    directory.
    """
    here = Path(__file__).parent
    helper_path = here / "011_plot_total_energy_loss.py"
    spec = importlib.util.spec_from_file_location("synrad_loss_helpers", helper_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


synrad_helpers = _load_synrad_helpers()


################################################################################
# Table construction
################################################################################

########################################
# Build power-of-two tables
########################################
def make_power2_tables(rng, max_power=128, n_table_samples=100_000,
                       batch_photons=1_000_000):
    """Build empirical inverse-CDF tables for powers of two photon counts.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random number generator used to build the Monte Carlo tables.
    max_power : int
        Largest photon count table to build. The generated counts are
        ``1, 2, 4, ..., max_power``.
    n_table_samples : int
        Number of total-energy samples stored in each empirical table.
    batch_photons : int
        Approximate cap on the number of individual photons sampled in one
        batch while building a table. This keeps temporary arrays bounded when
        building large ``N`` tables.

    Returns
    -------
    tables : dict[int, ndarray]
        Mapping from photon count ``N`` to a sorted array of samples of
        ``x_1 + ... + x_N``. The sorted arrays are used as empirical inverse
        CDFs by ``sample_from_inverse_cdf``.

    Notes
    -----
    This is deliberately a prototype. It uses Monte Carlo tables, so the tables
    themselves contain sampling noise. The production implementation avoids
    this by constructing deterministic distributions through convolution.
    """
    powers = []
    power = 1
    while power <= max_power:
        powers.append(power)
        power *= 2

    tables = {}
    for power in powers:
        values = np.empty(n_table_samples)
        n_done = 0

        while n_done < n_table_samples:
            n_batch = min(n_table_samples - n_done,
                          max(1, batch_photons // power))
            photon_energy = synrad_helpers.sample_photon_energy_normalized(
                rng, n_batch * power)
            values[n_done:n_done + n_batch] = photon_energy.reshape(
                n_batch, power).sum(axis=1)
            n_done += n_batch

        values.sort()
        tables[power] = values
        print(f"built table for {power:4d} photons")

    return tables


################################################################################
# Table sampling
################################################################################

########################################
# Sample one empirical inverse CDF
########################################
def sample_from_inverse_cdf(rng, sorted_values):
    """Sample from one empirical inverse-CDF table.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random number generator.
    sorted_values : ndarray
        Sorted samples representing an empirical inverse CDF.

    Returns
    -------
    value : float
        One sample from the distribution represented by ``sorted_values``.

    Notes
    -----
    A uniform random number selects a fractional table index. Linear
    interpolation between neighboring stored samples reduces quantization from
    the finite table size.
    """
    u = rng.random()
    q = u * (sorted_values.size - 1)
    i0 = int(q)
    i1 = min(i0 + 1, sorted_values.size - 1)
    w = q - i0
    return (1.0 - w) * sorted_values[i0] + w * sorted_values[i1]


########################################
# Decompose photon count
########################################
def decompose_power2(n_photons, max_power):
    """Decompose a photon count into greedy powers of two.

    Parameters
    ----------
    n_photons : int
        Photon count to represent.
    max_power : int
        Largest available power-of-two table.

    Returns
    -------
    chunks : list[int]
        Greedy decomposition of ``n_photons``. For example, with sufficient
        tables, ``19`` becomes ``[16, 2, 1]``.

    Notes
    -----
    The decomposition is exact as a distributional construction: the sum of an
    independent sample from the ``a``-photon table and an independent sample
    from the ``b``-photon table has the same distribution as the sum of
    ``a + b`` independent photons, assuming the tables are exact.
    """
    chunks = []
    n_left = int(n_photons)
    while n_left > 0:
        chunk = 1 << (n_left.bit_length() - 1)
        chunk = min(chunk, max_power)
        chunks.append(chunk)
        n_left -= chunk
    return chunks


########################################
# Sample fixed photon count
########################################
def sample_fixed_n_power2(rng, n_photons, tables):
    """Sample total normalized energy for a fixed photon count.

    The photon count is decomposed into powers of two, then one inverse-CDF
    sample is taken from each corresponding table. The returned value is the
    sum of those independent table samples.
    """
    if n_photons == 0:
        return 0.0

    max_power = max(tables)
    total = 0.0
    for chunk in decompose_power2(n_photons, max_power):
        total += sample_from_inverse_cdf(rng, tables[chunk])
    return total


########################################
# Sample compound-Poisson process
########################################
def sample_total_loss_power2(rng, lambdas, n_particles, tables):
    """Sample the table-based compound-Poisson total-energy process.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random number generator.
    lambdas : array-like
        Mean photon counts, ``lambda = <N_gamma>``.
    n_particles : int
        Number of independent events per ``lambda``.
    tables : dict[int, ndarray]
        Power-of-two inverse-CDF tables from ``make_power2_tables``.

    Returns
    -------
    out : ndarray
        Table-sampled total normalized losses with shape
        ``(len(lambdas), n_particles)``.
    counts_out : ndarray
        Sampled Poisson photon counts.
    chunks_out : ndarray
        Number of table lookups used for each particle/event.

    Notes
    -----
    ``chunks_out`` is important for algorithm studies: it estimates how many
    sampler calls the table method uses relative to the original
    photon-by-photon loop.
    """
    out = np.empty((len(lambdas), n_particles))
    counts_out = np.empty((len(lambdas), n_particles), dtype=np.int64)
    chunks_out = np.empty((len(lambdas), n_particles), dtype=np.int64)

    for ii, lam in enumerate(lambdas):
        n_photons = rng.poisson(lam, size=n_particles)
        counts_out[ii, :] = n_photons
        values = np.empty(n_particles)
        n_chunks = np.empty(n_particles, dtype=np.int64)

        for ipart, count in enumerate(n_photons):
            chunks = decompose_power2(count, max(tables))
            n_chunks[ipart] = len(chunks)
            values[ipart] = sum(
                sample_from_inverse_cdf(rng, tables[chunk])
                for chunk in chunks)

        out[ii, :] = values
        chunks_out[ii, :] = n_chunks

    return out, counts_out, chunks_out


################################################################################
# Timing helpers
################################################################################

########################################
# Time fixed photon count samplers
########################################
def time_fixed_n_samplers(rng_reference, rng_power2, photon_counts,
                          n_samples, tables):
    """Measure prototype sampling time for fixed photon counts.

    Parameters
    ----------
    rng_reference : numpy.random.Generator
        Random generator for the brute-force photon-by-photon sampler.
    rng_power2 : numpy.random.Generator
        Random generator for the power-of-two table sampler.
    photon_counts : array-like
        Fixed photon counts to benchmark.
    n_samples : int
        Number of total-loss samples generated for each photon count.
    tables : dict[int, ndarray]
        Power-of-two empirical inverse-CDF tables.

    Returns
    -------
    timing : dict[str, ndarray]
        Arrays with brute-force time, power-of-two time, and expected number
        of power-of-two table chunks for each photon count.

    Notes
    -----
    This is a Python prototype timing, not a prediction of GPU timing. It is
    still useful because it shows the algorithmic scaling: brute force grows
    roughly with the number of photons, while the table method grows with the
    number of power-of-two chunks.
    """
    photon_counts = np.asarray(photon_counts, dtype=np.int64)
    brute_force_time = np.empty(photon_counts.size)
    power2_time = np.empty(photon_counts.size)
    n_chunks = np.empty(photon_counts.size)

    for ii, count in enumerate(photon_counts):
        t_start = time.perf_counter()
        if count == 0:
            reference_values = np.zeros(n_samples)
        else:
            photon_energy = synrad_helpers.sample_photon_energy_normalized(
                rng_reference, n_samples * count)
            reference_values = photon_energy.reshape(n_samples, count).sum(axis=1)
        brute_force_time[ii] = time.perf_counter() - t_start

        t_start = time.perf_counter()
        power2_values = np.array([
            sample_fixed_n_power2(rng_power2, count, tables)
            for _ in range(n_samples)
        ])
        power2_time[ii] = time.perf_counter() - t_start

        n_chunks[ii] = len(decompose_power2(count, max(tables)))

        # Keep the arrays live until both samplers have run. This avoids making
        # the timing block look unused and keeps the prototype easy to inspect.
        assert reference_values.shape == power2_values.shape

    return {
        "photon_counts": photon_counts,
        "brute_force_time": brute_force_time,
        "power2_time": power2_time,
        "n_chunks": n_chunks,
    }


################################################################################
# Run
################################################################################

if __name__ == "__main__":
    ########################################
    # Build prototype tables
    ########################################
    tables = make_power2_tables(
        rng             = RNG_TABLES,
        max_power       = 256,
        n_table_samples = 40_000)

    ########################################
    # Show a few decompositions
    ########################################
    for count in [8, 19, 87, 312]:
        print(f"{count:3d} -> {decompose_power2(count, max(tables))}")

    ########################################
    # Compare brute force with table sampling
    ########################################
    reference = synrad_helpers.sample_total_loss_normalized(
        RNG_REFERENCE, TEST_LAMBDAS, N_PARTICLES)
    power2, counts, chunks = sample_total_loss_power2(
        RNG_POWER2, TEST_LAMBDAS, N_PARTICLES, tables)

    print()
    print("lambda      <N>   <chunks>   mean ref  mean p2    rms ref   rms p2")
    for ii, lam in enumerate(TEST_LAMBDAS):
        print(
            f"{lam:6.2f}"
            f"  {np.mean(counts[ii]):7.3f}"
            f"  {np.mean(chunks[ii]):8.3f}"
            f"  {np.mean(reference[ii]):8.4f}"
            f"  {np.mean(power2[ii]):8.4f}"
            f"  {np.std(reference[ii]):8.4f}"
            f"  {np.std(power2[ii]):8.4f}")

    ########################################
    # Measure fixed-count prototype timings
    ########################################
    timing = time_fixed_n_samplers(
        rng_reference  = np.random.default_rng(112233),
        rng_power2     = np.random.default_rng(445566),
        photon_counts  = TIMING_COUNTS,
        n_samples      = TIMING_SAMPLES,
        tables         = tables)

    print()
    print("fixed N   brute force [us/sample]   power2 [us/sample]   speedup   chunks")
    for ii, count in enumerate(timing["photon_counts"]):
        brute_us = timing["brute_force_time"][ii] / TIMING_SAMPLES * 1e6
        power2_us = timing["power2_time"][ii] / TIMING_SAMPLES * 1e6
        print(
            f"{count:7d}"
            f" {brute_us:24.4f}"
            f" {power2_us:20.4f}"
            f" {brute_us / power2_us:9.3f}"
            f" {timing['n_chunks'][ii]:8.0f}")

    ########################################
    # Plot distribution comparison
    ########################################
    plt.close("all")
    fig_dist, ax_dist = plt.subplots(1, 1, figsize=(8, 5))

    for lam, ref, p2 in zip(TEST_LAMBDAS, reference, power2):
        bins = np.linspace(
            0.0,
            np.quantile(np.concatenate([ref, p2]), 0.995),
            120,
        )
        ax_dist.hist(ref, bins=bins, density=True, histtype="step",
                     label=rf"brute force, $\lambda={lam:g}$")
        ax_dist.hist(p2, bins=bins, density=True, histtype="step",
                     linestyle="--",
                     label=rf"power-2 tables, $\lambda={lam:g}$")

    ax_dist.set_xlabel(
        r"Total normalized emitted energy, "
        r"$S=\sum_i E_{\gamma,i}/E_c$ [1]")
    ax_dist.set_ylabel("Probability density [1]")
    ax_dist.set_title("Brute-force photon loop vs power-of-two tables")
    ax_dist.set_yscale("log")
    ax_dist.grid(True)
    ax_dist.legend(fontsize=8)
    fig_dist.tight_layout()

    ########################################
    # Plot sampler-call counts
    ########################################
    max_count = int(np.quantile(counts, 0.999))
    count_values = np.arange(max_count + 1)
    chunk_values = [
        len(decompose_power2(count, max(tables)))
        for count in count_values
    ]

    fig_calls, ax_calls = plt.subplots(1, 1, figsize=(8, 5))
    ax_calls.plot(count_values, count_values, label="photon-by-photon loop")
    ax_calls.step(count_values, chunk_values, where="post",
                  label="power-of-two table loop")
    ax_calls.set_xlabel(r"Photon count, $N_\gamma$ [1]")
    ax_calls.set_ylabel("Number of sampler calls [1]")
    ax_calls.set_title("Algorithmic sampler-call count")
    ax_calls.grid(True)
    ax_calls.legend()
    fig_calls.tight_layout()

    ########################################
    # Plot prototype timings
    ########################################
    fig_time, ax_time = plt.subplots(1, 1, figsize=(8, 5))
    ax_time.loglog(
        timing["photon_counts"],
        timing["brute_force_time"] / TIMING_SAMPLES * 1e6,
        "o-",
        label="brute-force photon loop")
    ax_time.loglog(
        timing["photon_counts"],
        timing["power2_time"] / TIMING_SAMPLES * 1e6,
        "s-",
        label="power-of-two table sampler")
    ax_time.set_xlabel(r"Fixed photon count, $N_\gamma$ [1]")
    ax_time.set_ylabel("Python prototype time [us/sample]")
    ax_time.set_title("Prototype timing for fixed photon counts")
    ax_time.grid(True, which="both")
    ax_time.legend()
    fig_time.tight_layout()

    plt.show()
