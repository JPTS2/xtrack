<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# Optics with collective elements

This lesson needs **Xfields**, alongside Xtrack, NumPy and Matplotlib.

| Example | Lesson |
| --- | --- |
| [001_optics_with_collective_elements.py](001_optics_with_collective_elements.py) | Compare prescribed frozen space-charge optics with default skipping of an updating collective field |

The inline model is an open proton line at 1 GeV/c, with two 1 m drifts around
a thin bi-Gaussian space-charge interaction. Its prescribed source has
`1e11` protons, transverse rms sizes of 2 mm and Gaussian longitudinal rms size
of 0.2 m. Optics are evaluated at `zeta = delta = 0`, with supplied entrance
beta and alpha. The interaction length weights the force; it adds no surveyed
length. There is no RF, radiation or longitudinal space-charge kick.

The three configurations own independent fields and source profiles:

- Zero charge provides a drift control.
- A frozen source is noncollective in the tracking classification. Its
  prescribed force remains active in default Twiss, giving visible defocusing.
- An updating source is collective. Default Twiss skips that force, reproducing
  the zero-charge result while leaving the prescribed source intensity intact.

The frozen result is single-particle optics linearized in a prescribed field.
Its nonlinear force and finite numerical probes affect the derivative precision.
It is not a self-consistent beam equilibrium calculation.

`include_collective=True` keeps collective elements active during Twiss tracking.
For this updating space-charge element, population, centroid and widths would
then be inferred from the numerical Twiss probes. Those probes do not define
the physical source bunch. That configuration is left unqualified here; using
the flag alone does not establish self-consistent collective optics.

Full source-distribution, quasi-frozen and PIC tracking workflows belong with
Xfields; see the [Xsuite space-charge guide](https://xsuite.readthedocs.io/en/latest/spacechargeauto.html).
