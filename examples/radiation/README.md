<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# Synchrotron radiation

Start with deterministic energy loss, compensate it, then study damping,
linear equilibrium predictions and stochastic photon emission. Each script
runs independently.

| Example | Lesson |
| --- | --- |
| [001_mean_energy_loss.py](001_mean_energy_loss.py) | Compare radiation off and mean energy loss during one uncompensated passage |
| [002_compensate_radiation_and_taper.py](002_compensate_radiation_and_taper.py) | Compensate and taper; compare entrance and mean momentum conventions |
| [003_radiation_integrals.py](003_radiation_integrals.py) | Inspect native integral predictions and locate the energy-loss/excitation sources |
| [004_mean_radiation_damping.py](004_mean_radiation_damping.py) | Follow two amplitude damping times and compare action decay with the native rate |
| [005_equilibrium_emittances.py](005_equilibrium_emittances.py) | Inspect equilibrium covariance, prepare native bunches and compare short mean/quantum histories |
| [006_quantum_radiation.py](006_quantum_radiation.py) | Contrast deterministic loss with photon-resolved and total-kick fluctuations |
| [007_record_photons_and_spectrum.py](007_record_photons_and_spectrum.py) | Record photons, check energy balance and compare analytical spectra and photon moments |
| [008_radiation_linearizations.py](008_radiation_linearizations.py) | Compare three native optics linearizations on one compensated setting |
| [009_decelerating_rf_station.py](009_decelerating_rf_station.py) | Extract energy at a second RF station and inspect the new balance and partitions |
| [010_wigglers_and_combined_fields.py](010_wigglers_and_combined_fields.py) | Change radiation sources with a closed vertical wiggler and an independent bend gradient |

All examples use a 10 GeV/c electron reference. `001`–`005` and `008`–`010` load the shared
[demonstration ring](../../test_data/example_lattices/demo_ring/README.md).
Its input reference momentum is 1 GeV/c; selecting 10 GeV/c with fixed normalized
magnet strengths corresponds to ten times the nominal physical fields.
The powered ring examples use 50 MV at harmonic 200, with about 23.17 MeV lost
per turn. This is an illustrative operating point, without a hardware claim.

`001` leaves RF off. `002` introduces native RF compensation and magnet tapering;
`004` and `005` reuse that operation on fresh copies of the same ring.
Compensation restores the energy each turn, while tapering keeps the operating
optics close to the intended setting along the energy sawtooth.

`002` compares requested mean and entrance-momentum conventions. Local momentum
still follows the energy sawtooth; either choice describes a compensated
operating point rather than a sampled equilibrium bunch.

`003` is the short radiation-integral introduction. `010` compares integral and
full mean-map predictions on the same compensated operating points. Its closed
vertical wiggler adds vertical dispersion/excitation while preserving the
surveyed reference path; its actual orbit and path length change. An independent
bend gradient isolates a signed I4 contribution. The ideal planar model has
no resolved physical vertical excitation; numerical remnants are not a beam-
height floor.

`008` compares `full`, `kick_as_co` and `scale_as_co` on identical compensated
fields. The comparison uses serial CPU; `scale_as_co` may compile a local kernel.
`009` instead changes the physical RF system, adding a weak decelerating station
while holding the old tapered fields fixed. It measures both station kicks and
the new damping partitions; a phase setting alone is not proof of energy loss.

`005` uses native Xpart generation to prepare samples from the predicted
horizontal emittance and longitudinal width, with zero vertical emittance.
This mode/RF matching does not impose every correlation in the full covariance.
Two independent samples and a same-launch mean-radiation control are tracked
for 64 turns and measured by `BeamStatsMonitor`. This is only about 0.065
horizontal and 0.157 longitudinal amplitude damping times. The quantum samples
start near the prediction; they do not demonstrate convergence from a cold
beam. Correlated turn samples are not independent statistical repeats.

Equilibrium calculations are linear predictions. They do not establish
nonlinear acceptance, survival or tails of a stored bunch.

`006` and `007` use explicit dipole probes at the same 10 GeV/c. `006` tracks a
finite-length `Bend`; `007` records a thin radiating `Multipole` kick with an
assigned bending length. The latter does not describe finite-length transport.
`quantum` generates individual photons, whereas `quantum-kick` samples a total
stochastic kick without photon records. Native tracking initializes the random
streams; stochastic realizations vary between runs.

`007` compares photon counts with the angle-integrated photon-number spectrum,
using actual logarithmic-bin probabilities normalized over all energies. It
also reports the first two energy moments and their conditional finite-count
uncertainty. The entrance critical energy is an approximation as electrons
recoil. The analytical convention follows the
[Geant4 synchrotron-radiation reference](https://geant4.web.cern.ch/documentation/dev/prm_html/PhysicsReferenceManual/electromagnetic/xray_production/InvSynFracInt.html).

For high-statistics distribution and tail comparisons, see
[quantum-kick versus quantum](quantum_kick_vs_quantum/README.md). Those studies
serve a different purpose from the short introductory tracking examples.

For the API and general workflow, see the
[Xsuite radiation guide](https://xsuite.readthedocs.io/en/latest/synchrotron_radiation.html).
