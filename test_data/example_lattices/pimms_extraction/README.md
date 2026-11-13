<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# PIMMS extraction model

A native version of the compact 75.24 m, 200 MeV proton ring used in the
[Xsuite slow-extraction tutorial](https://github.com/xsuite/slow_extraction_tutorial).
The tutorial's `PIMM.seq` attributes its geometry to CERN-PS-99-010-DI and the
TERA Foundation. Magnet centres, lengths, bend angles and pole-face angles are
retained here; no MAD-X executable or tutorial installation is required.

Load `lattice.py` and select `env['pimms']`. The three quadrupole families and
two chromatic sextupole families are controlled by the adjacent `strengths.py`.
They are rematched for the current adaptive bend maps to:

- tunes Qx=1.663 and Qy=1.72;
- horizontal dispersion zero at `extr_septum`;
- chromaticities dQx/d(delta)=dQy/d(delta)=-0.1.

The two extraction sextupoles, `se1` and `se2`, start off and have independent
`kse1`/`kse2` controls in m^-3. Full pole-face maps and five bend multipole kicks
are used. There is no RF, radiation, space charge, septum material or extracted
beamline. Examples install their own observation or absorbing boundary.

## Relation to the acceleration model

This is a different PIMMS teaching configuration from
[`test_data/pimms`](../../pimms/), used by the carbon/proton acceleration study.
The bend and main quadrupole stations share the same layout. The compact
extraction tutorial omits the extra `qa` quadrupole, RF and electrostatic-kicker
assembly and observes extraction at s=0. Its family names, strengths and proton
operating point are distinct. Loading extraction strengths into the acceleration
model would therefore not reproduce these studies.

## Interpretation

The [extraction lessons](../../../examples/machine_studies/pimms/slow_extraction/README.md)
show the third-integer phase-space topology, changes of its bounded region and
outgoing branch, and a programmed spill. Finite-turn boundary estimates are not
proofs of infinite-time stability. Counts at an ideal absorbing boundary are
not extraction efficiencies or septum-loss predictions.
