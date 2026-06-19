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

## Planned validation sequence

The intended end state is seven active examples:

```text
001_review_total_energy_tables.py
002_validate_table_sampling.py
003_single_bend_kick_distribution.py
004_regime_scan_kick_distribution.py
005_sampler_call_counts.py
006_fcc_emittance_cpu.py
007_lifetime_tail_proxy.py
```

Together these should replace the useful validation coverage from the
historical `011` to `016` development files. An eighth file is not planned at
this stage. If a very large statistics certification run is needed later, it
should either be exposed through explicit user parameters in these examples or
added as a deliberately heavy, non-default certification script.

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

Run a CPU-only FCC tracking-observable check:

- `quantum` versus `quantum-kick`;
- around `1E2` particles and `1E4` turns by default;
- all particles initially at zero;
- compare emittance or beam-size observables within an explicit tolerance.

This is a first lattice-level physics sanity check, not a high-statistics
equilibrium-emittance certification.

### 007 Lifetime tail proxy

Add a tail-sensitive observable aimed at lifetime risk:

- compare the probability of exceeding selected energy/kick thresholds;
- use thresholds derived from the `quantum` reference distribution where useful;
- keep the default runtime reasonable, with clear knobs for certification-scale
  Monte Carlo.

This starts addressing the main remaining physics risk: rare events that can
drive losses and lifetime.

## Historical development files

The following files were moved here from the historical flat radiation folder
and are kept for reference while the new sequence is implemented:

- `011_plot_total_energy_loss.py`
- `012_prototype_total_energy_loss_power2.py`
- `013_compare_quantum_efficient.py`
- `014_validate_quantum_efficient_tables.py`
- `015_benchmark_quantum_efficient_tracking.py`
- `016_benchmark_quantum_efficient_fcc_lattice.py`

These names still reflect the development branch. They should be mined for
useful code and then either renamed, simplified, or removed from the active
example sequence.

## Migration from historical files

The useful content from the historical files should be folded into the seven
active examples as follows:

- `011_plot_total_energy_loss.py`
  - contributes the single-photon spectrum, brute-force photon sampling, and
    compound-Poisson total-loss Monte Carlo used by
    `002_validate_table_sampling.py`;
  - selected CDF or spectrum inspection may also feed
    `001_review_total_energy_tables.py`.

- `012_prototype_total_energy_loss_power2.py`
  - contributes the explanation of sampler-call scaling and photon-count
    decomposition to `005_sampler_call_counts.py`;
  - the old power-of-two prototype should not remain an active public example
    once the direct-table `quantum-kick` sequence is complete.

- `013_compare_quantum_efficient.py`
  - contributes the single-bend Monte Carlo comparison to
    `003_single_bend_kick_distribution.py`;
  - its broader case handling should feed
    `004_regime_scan_kick_distribution.py`.

- `014_validate_quantum_efficient_tables.py`
  - contributes stored-table checks and table-sampling validation to
    `001_review_total_energy_tables.py` and
    `002_validate_table_sampling.py`;
  - full table regeneration should not be part of the default example path.

- `015_benchmark_quantum_efficient_tracking.py`
  - contributes single-bend histogram and diagnostic accumulation to
    `003_single_bend_kick_distribution.py` and
    `004_regime_scan_kick_distribution.py`;
  - real wall-clock timing belongs in `examples/tracking_time`, while only
    algorithmic call-count diagnostics belong here.

- `016_benchmark_quantum_efficient_fcc_lattice.py`
  - contributes the lattice-level setup to `006_fcc_emittance_cpu.py`;
  - any tail-sensitive survival or loss observable should feed
    `007_lifetime_tail_proxy.py`;
  - CPU/GPU timing comparisons belong in `examples/tracking_time`.
