<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# Optics with collective elements

These lessons need **Xfields**, alongside Xtrack, NumPy and Matplotlib.

| Example | Lesson |
| --- | --- |
| [001_optics_with_collective_elements.py](001_optics_with_collective_elements.py) | Compare prescribed frozen space-charge optics with default skipping of an updating collective field |
| [002_bunch_dependent_beambeam_optics.py](002_bunch_dependent_beambeam_optics.py) | Remove one opposing bunch and compare per-slot tunes and beta functions in a rigid-bunch beam-beam study |

## Prescribed space-charge optics

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

## Bunch-dependent beam-beam optics

`002` reuses the demonstration ring's normalized magnetic strengths for two
independent 7 TeV/c proton rings, with RF off. The second ring mirrors the
element order; its normal magnetic optics has no skew fields or powered
correctors. This is an illustrative two-ring optical model, not a shared-magnet
collider design or a proton operating point qualified for acceleration.

There is one head-on IP and no long-range encounters or crossing angle. Three
occupied slots initially have partners. Removing the opposing slot 1 leaves
the corresponding clockwise bunch at the no-beam-beam optics, while slots 0
and 2 retain tune shifts and beta beating. The symmetric centroids stay on
axis: this lesson does not demonstrate bunch-dependent orbit offsets or
long-range PACMAN effects.

`study.solve()` iterates the opposing centroids; `study.twiss()` computes optics
in their currently loaded state. Source sizes stay at their no-beam-beam values
(`dynamic_beta=False`), although the resulting test-particle beta functions
change. The batched `fast` calculation is 4D. These per-bunch optics are not a
coupled coherent-mode spectrum, a particle simulation or an emittance-growth
prediction.

Start with [train preparation](../../particles/012_multibunch_train.py) for
filling slots and populations. Complete collider configuration, dynamic-beta
iteration and reduced-lattice workflows belong with Xfields; see the
[Xsuite rigid-bunch guide](https://xsuite.readthedocs.io/en/latest/collective.html#rigid-bunch).
