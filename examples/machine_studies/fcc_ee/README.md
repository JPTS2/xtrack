<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# FCC-ee: IP optics, alignment and equilibrium emittance

Follow a common FCC-ee Z model from IP beta control through alignment errors,
correction and a radiating equilibrium calculation. The model and strength files
come from the [Xsuite CAS November 2026 tutorial](https://github.com/xsuite/tutorial_cas_nov_2026/tree/main/02_fccee_optics_and_bb),
which also provides optics-comparison and beam-beam exercises.

| Example | Lesson |
| --- | --- |
| [001_ip_beta_control.py](001_ip_beta_control.py) | Generate a periodic beta-control knob, compensate endpoint chromaticities and inspect interpolation errors |
| [002_alignment_sensitivity.py](002_alignment_sensitivity.py) | Compare the same final-doublet offsets under nominal and supplied larger-beta optics |
| [003_orbit_correction.py](003_orbit_correction.py) | Correct the orbit with installed steering elements and inspect the residual orbit and dispersion |
| [004_equilibrium_after_correction.py](004_equilibrium_after_correction.py) | Compensate and taper each saved correction state; compare equilibrium emittances and projected IP beam sizes |

The [stored model](../../../test_data/example_lattices/fcc_ee_z/README.md) has a
45.6 GeV/c positron reference, four IPs, simplified magnet maps and no beam-beam
interaction. These are illustrative studies, not current FCC design settings
or a commissioning prescription. The generic [matching](../../matching/README.md),
[correction](../../errors_and_correction/README.md) and
[radiation](../../radiation/README.md) examples introduce the individual APIs.

## IP beta and sensitivity

`001` matches a modest 10% beta increase using periodic optics and shared
circuits. It preserves the endpoint tunes and linear chromaticities; it does
not hold insertion boundaries fixed. The native knob linearly interpolates
circuit changes, so intermediate beta functions and tunes need checking.
It does not independently control the four IPs.

`002` uses the CAS supplied larger-beta setting, not the small knob endpoint.
It applies identical independent 5 µm RMS section offsets to both optics.
The calculation finds the closed orbit only: convergence does not establish
stable optics around that orbit. It does not model correlated girder motion.

## Correction and radiation

Run `003` before `004`. The correction script writes three native
states under `outputs/correction/`; the radiation script loads those states
without repeating the correction.

Full periodic optics needs smaller errors than the orbit-only comparison.
`003` uses the same seed at 0.2 µm RMS, ideal BPMs, a bounded steering selection
and no hardware kick limits. The three saved states are aligned, misaligned
and orbit-corrected. Numerical probe settings resolve the linear optics near
the narrow vertical waist; their emittance parameter is not a physical bunch
emittance. Small orbit RMS alone does not establish zero vertical dispersion
or minimum equilibrium emittance.

`004` independently enables mean radiation, compensates RF energy loss and
tapers each state. It uses the full radiation linearization and obtains
equilibrium mode emittances and native projected
covariances from 6D optics. It does not perform a quantum-tracking measurement
or include collective effects, IBS or beam-beam forces. The ideal planar
vertical result is a model/numerical floor, not an achievable emittance budget.

## Detector solenoid: a separate insertion study

[detector_solenoid/001_compensate_insertion.py](detector_solenoid/001_compensate_insertion.py)
uses the retained FCC-t sequence at 182.5 GeV/c, not the CAS Z model. It installs
a tabulated tilted detector field and corrects orbit, coupling and optics
through the two half-insertions. It compares field-off, uncorrected and corrected
physical slopes, cross-plane transfer blocks and coupled-mode beta functions.
It is a fixed-energy 4D calculation with radiation off; it does not establish
the detector's radiation-induced emittance contribution. The different energy,
field profile and insertion geometry prevent silently combining its result
with the Z-ring correction sequence above.
