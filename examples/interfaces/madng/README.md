<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# MAD-NG

Export a native Xtrack line to MAD-NG, compare optics and nonlinear quantities,
then use external optics targets in the native optimizer. Each script runs
independently.

| Example | Lesson |
| --- | --- |
| [001_madng_optics.py](001_madng_optics.py) | Compare BPM optics and tunes, then propagate a live quadrupole-family change to MAD-NG |
| [002_resonance_driving_terms.py](002_resonance_driving_terms.py) | Compare MAD-NG geometric RDTs before and after powering sextupoles |
| [003_match_madng_optics.py](003_match_madng_optics.py) | Restore named BPM beta functions with MAD-NG targets in Xtrack's optimizer; verify the result independently |
| [004_export_edited_and_sliced_lines.py](004_export_edited_and_sliced_lines.py) | Export separately configured lines after alignment and structural edits; compare common observation points |
| [005_normal_form_derivatives.py](005_normal_form_derivatives.py) | Compare normal-form chromaticity and amplitude detuning with native numerical estimates |

All use the shared [demonstration ring](../../../test_data/example_lattices/demo_ring/README.md)
with a 1 GeV/c electron reference. RF voltage and radiation are zero. The
external backend requires PyMADNG and its MAD-NG executable. The examples were
verified with Xtrack 0.118.7 and PyMADNG 0.9.6. No MAD-X executable or `cpymad`
session is needed.

`005` also uses `nafflib` through Xtrack's native tracked-tune helper. It moves
the shared ring to an explicit working point before enabling its octupoles.

`line.madng_twiss()` returns Xtrack 4D optics alongside MAD-NG columns such as
`beta11_ng`, `beta22_ng`, `mu1_ng` and `mu2_ng`. In the uncoupled configurations,
compare those mode beta functions with `betx` and `bety`. `004` also compares
cross-plane mode projections after a small quadrupole roll. Phase advances are
in turns; the full-ring end-minus-start advance gives the tune. Use the returned table's
row alignment and named BPMs.

`003` records nominal MAD-NG beta functions, changes the arc-family trims,
then restores those two beta functions at one BPM. It does not constrain the
full-ring tunes or require the original strengths to be recovered.

Changing existing symbolic variables updates the attached MAD-NG model. Direct
changes to element attributes or lattice structure are not the same operation.
`004` preserves independently configured line copies and exports each after its
edits; agreement is inspected at common BPMs. Each example explicitly closes its
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

`005` compares normal-form derivatives with finite-action tracking and small
fixed-field momentum scans. The action is geometric `J` in metres, without a
`2*pi` factor. The interface's `dqxdjx_nf_ng` and related fields already convert
MAD-NG's internal `2J` convention to derivatives with respect to `J`.
The second chromatic derivatives are derivatives, not quadratic Taylor
coefficients. The example prints discrepancies and changes the probe scales;
its vertical self-detuning differs by about 3% for this operating point. Smaller
launches and a longer record do not remove that offset. This comparison does
not establish identical nonlinear maps in the two engines.

For native nonlinear optics and finite-amplitude tracking, continue with
[nonlinear optics](../../optics/nonlinear/README.md) and
[nonlinear tracking](../../tracking/nonlinear/README.md).
