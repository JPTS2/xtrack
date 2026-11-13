<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# Statistical precision

Use native stochastic radiation to distinguish a physical distribution from
the uncertainty of a finite sample. These studies use a stated single-bend
experiment, without a machine acceptance or lifetime interpretation.
Plotting uses Matplotlib.

| Example | Lesson |
| --- | --- |
| [001_radiation_sample_precision.py](001_radiation_sample_precision.py) | Measure independent repeat-to-repeat scatter versus particle count, and compare mean/rms precision predictions |
| [002_radiation_model_distributions.py](002_radiation_model_distributions.py) | Compare native quantum and quantum-kick CDFs, zero-loss probabilities and preselected tails with confidence intervals |
| [003_radiation_regimes_and_vectors.py](003_radiation_regimes_and_vectors.py) | Compare two emission regimes and transverse exit correlations, beyond scalar energy-loss means |

The sample-precision study distinguishes an unbiased sample variance from its
finite-sample-biased square root. Its fourth-moment rms-precision estimate is a
large-sample approximation; bootstrap intervals describe uncertainty in the
measured repeat scatter. The independent reference sample has finite precision
as well.

Distribution comparisons show simultaneous distribution-free CDF bands and
pointwise Wilson intervals for selected event probabilities. Zero observed tail
events still give an upper confidence limit. Overlapping intervals do not
establish agreement to arbitrary accuracy, particularly in poorly sampled tails.

The regime study keeps both models identical within each bend setting, then
changes the bend angle. A stronger bend changes emission occurrence and photon
energy scales; the finite samples are not a rare-tail or ring-equilibrium test.

Fresh particle batches receive native RNG streams. NumPy seeds control only
explicit resampling in these scripts, not the native radiation realization.
For the underlying models and an introductory histogram, see
[quantum radiation](../../radiation/006_quantum_radiation.py).
