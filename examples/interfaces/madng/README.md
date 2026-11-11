<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# MAD-NG

Export a native Xtrack line to MAD-NG, compare periodic optics, then calculate
resonance driving terms with the public interface. Each script runs independently.

| Example | Lesson |
| --- | --- |
| [001_madng_optics.py](001_madng_optics.py) | Compare BPM optics and tunes, then propagate a live quadrupole-family change to MAD-NG |
| [002_resonance_driving_terms.py](002_resonance_driving_terms.py) | Compare MAD-NG geometric RDTs before and after powering sextupoles |

Both use the shared [demonstration ring](../../../test_data/example_lattices/demo_ring/README.md)
with a 1 GeV/c electron reference. RF voltage and radiation are zero. The
external backend requires PyMADNG and its MAD-NG executable. The examples were
verified with Xtrack 0.118.7 and PyMADNG 0.9.6. No MAD-X executable or `cpymad`
session is needed.

`line.madng_twiss()` returns Xtrack 4D optics alongside MAD-NG columns such as
`beta11_ng`, `beta22_ng`, `mu1_ng` and `mu2_ng`. In this uncoupled example, compare
those mode beta functions with `betx` and `bety`. Phase advances are in turns;
the full-ring end-minus-start advance gives the tune. Use the returned table's
row alignment and named BPMs.

Changing existing symbolic variables updates the attached MAD-NG model. Direct
changes to element attributes or lattice structure are not the same operation:
export a fresh model for those changes. Each example explicitly closes its
backend after the last calculation.

Export carries supported geometry, strengths, expressions and reference-particle
properties. It does not preserve every Xtrack integration/model setting; the
optics comparison measures agreement for this configuration. Exported aperture
limits become markers, so MAD-NG export does not transfer loss tracking.

The RDT columns retain names such as `f3000` and contain complex values in
MAD-NG's convention. These third-order geometric terms have units `m^-1/2` in
transverse Courant–Snyder coordinates. `f3000` describes the `3 Qx` harmonic;
`f1020` describes `Qx + 2 Qy`. Their magnitudes along the ring are
normal-form quantities, rather than a map of local sextupole strength. The bare
ring has geometric bend nonlinearities, so its RDTs are retained as a baseline.
The shared ring has sextupole pairs separated by a transverse `-I` transform;
their leading geometric contributions cancel outside each pair. Near-zero
values in the straight sections are therefore expected. The comparison does
not infer particle loss or dynamic aperture.

Xtrack also provides `xt.rdt_first_order_perturbation` for first-order estimates
from Twiss and multipole-strength tables, without an external engine.

For native nonlinear optics and finite-amplitude tracking, continue with
[nonlinear optics](../../optics/nonlinear/README.md) and
[nonlinear tracking](../../tracking/nonlinear/README.md).
