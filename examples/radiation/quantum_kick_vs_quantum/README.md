# Quantum-kick versus quantum

This folder is for physics validation of `quantum-kick` against the
photon-by-photon `quantum` radiation model.

The scope is deliberately narrow:

- compare only the public models `quantum` and `quantum-kick`;
- validate total emitted-energy and kick distributions, including tails;
- check stored total-energy-loss tables without regenerating the full
  production table set;
- keep timing benchmarks out of this folder, except for algorithmic call-count
  diagnostics.

`quantum` generates individual photons. `quantum-kick` does not. It samples an
equivalent stochastic total-energy kick, so equivalence is expected only for
total radiation kicks and downstream tracking observables, not for photon
records.

## Validation sequence

The implemented active sequence currently contains six examples:

```text
001_review_total_energy_tables.py
002_validate_table_sampling.py
003_single_bend_kick_distribution.py
004_regime_scan_kick_distribution.py
005_sampler_call_counts.py
006_fcc_emittance_cpu.py
```

`007_lifetime_tail_proxy.py` is retained as a documented, non-executable
placeholder. It should become active only when a machine acceptance and loss
model are available for a defensible lattice-level lifetime comparison.

Together the active examples replace the useful `quantum-kick` validation
coverage from the historical `011` to `016` development files. If a very large
statistics certification run is needed later, it should either be exposed
through explicit user parameters in these examples or added as a deliberately
heavy, non-default certification script.

### 001 Review total-energy tables

Inspect the stored total-energy-loss tables before sampling or tracking:

- table metadata, grid size, direct-table range, and tail coverage;
- monotonicity and finite values of the stored inverse CDFs;
- selected CDF/inverse-CDF plots;
- regenerate a small direct subset, expected to be around `N=1..4`, and compare
  it against the stored defaults.

This gives a quick check that the table data and generator have not drifted in
the low-count region where mistakes would be most visible.

### 002 Validate table sampling

Compare table-sampled fixed-photon-count losses against brute-force
photon-by-photon Monte Carlo:

- fixed counts such as `1, 2, 3, 4, 8, 16, 32`;
- mean, rms, quantiles, and tail probabilities;
- explicit high-quantile checks such as `q99`, `q999`, and `q9999`.

This validates the random variables sampled by the `quantum-kick` machinery
before any tracking element is involved.

### 003 Single-bend kick distribution

Track many particles through one bend with `quantum` and `quantum-kick`:

- compare `dpx`, `dpy`, `ddelta`, and emitted-energy proxies;
- include both bulk and tail metrics;
- make the single default case easy to inspect with plots.

This checks the full element kick path in a controlled setting.

### 004 Regime scan of kick distributions

Repeat the single-bend kick comparison across emitted-photon regimes:

- very low lambda, mostly zero photons;
- one-photon dominated;
- medium lambda;
- high lambda;
- stress/fallback coverage.

Each regime should report zero-photon fraction, one-photon fraction,
direct-table usage, fallback usage, and tail metrics.

This checks that the physics agreement holds where the algorithm changes
behavior.

### 005 Sampler call counts

Compare algorithmic work rather than wall-clock timing:

- expected photon samples for `quantum`;
- expected table lookups for `quantum-kick`;
- direct-table versus fallback/decomposition use.

This explains why `quantum-kick` is faster while leaving real timing
comparisons to `examples/tracking_time`.

### 006 FCC emittance CPU check

Run a CPU-only FCC-ee tt tracking-observable check:

- `quantum` versus `quantum-kick`;
- around `1E2` particles and `3E2` turns by default;
- all particles initially at zero;
- compare normal modes I and III within explicit tolerances;
- use an `xcoll.EmittanceMonitor` and configurable turn-window smoothing.

This is a first lattice-level physics sanity check, not a high-statistics
equilibrium-emittance certification. The current native tt sequence is an
ideal planar lattice, so its equilibrium mode-II emittance is zero and only
modes I and III are meaningful here. This should be upgraded to a committed,
native-Xsuite, coupled FCC-ee tt LCC lattice when one becomes available, at
which point the full I/II/III comparison should be restored.

### 007 Lifetime tail proxy

This file is intentionally a future-work specification. Examples 002-004
already cover distribution and kick tails, so another standalone tail sampler
would be redundant. A useful 007 must instead provide a lattice-level loss
observable with:

- a documented RF bucket, momentum acceptance and physical aperture;
- an explicit loss definition;
- survival and loss-location comparisons for `quantum` and `quantum-kick`;
- confidence intervals and PASS, FAIL or LOWSTAT reporting.

It must not present an arbitrary kick threshold as a machine lifetime.

## Historical development files

The following files were moved here from the historical flat radiation folder
and remain temporarily for retirement review:

- `012_prototype_total_energy_loss_power2.py`
- `013_compare_quantum_efficient.py`
- `014_validate_quantum_efficient_tables.py`
- `015_benchmark_quantum_efficient_tracking.py`
- `016_benchmark_quantum_efficient_fcc_lattice.py`

These names still reflect the development branch and are not part of the active
example sequence.

### Retirement status

The retirement audit found that `012` to `016` can be removed:

- `012` prototypes the superseded empirical power-of-two approach;
- `013` and `014` are covered by examples 001-004;
- `015` timing is covered by `examples/tracking_time`, while its kick checks are
  covered by examples 003 and 004;
- `016` timing is covered by `examples/tracking_time`, while its lattice-level
  physics role is covered by example 006;
- `013`, `015` and `016` also refer to radiation modes that no longer exist.

The general educational content from `011` has been promoted to
`examples/radiation/001_synchrotron_radiation_spectrum.py`. Its
`quantum-kick`-specific sampling, compound-Poisson and tail checks are covered
by examples 001 and 002 in this folder. No active example imports any of the
remaining historical files.

## Completed migration

The useful `quantum-kick` validation content was migrated as follows:

- `011_plot_total_energy_loss.py`
  - its general single-photon spectrum and analytic theory plots were promoted
    to `examples/radiation/001_synchrotron_radiation_spectrum.py`;
  - its brute-force photon sampling and compound-Poisson validation are covered
    by `002_validate_table_sampling.py`.

- `012_prototype_total_energy_loss_power2.py`
  - its sampler-call scaling and photon-count decomposition are covered by
    `005_sampler_call_counts.py`;
  - its empirical power-of-two table construction is obsolete.

- `013_compare_quantum_efficient.py`
  - its single-bend Monte Carlo comparison is covered by
    `003_single_bend_kick_distribution.py`;
  - its broader case handling is covered by
    `004_regime_scan_kick_distribution.py`.

- `014_validate_quantum_efficient_tables.py`
  - its stored-table checks and table-sampling validation are covered by
    `001_review_total_energy_tables.py` and
    `002_validate_table_sampling.py`;
  - full table regeneration should not be part of the default example path.

- `015_benchmark_quantum_efficient_tracking.py`
  - its single-bend histogram and diagnostic accumulation are covered by
    `003_single_bend_kick_distribution.py` and
    `004_regime_scan_kick_distribution.py`;
  - real wall-clock timing belongs in `examples/tracking_time`, while only
    algorithmic call-count diagnostics belong here.

- `016_benchmark_quantum_efficient_fcc_lattice.py`
  - its lattice-level role is covered by `006_fcc_emittance_cpu.py`;
  - a genuine survival or loss observable remains explicitly deferred in
    `007_lifetime_tail_proxy.py`;
  - CPU/GPU timing comparisons belong in `examples/tracking_time`.
