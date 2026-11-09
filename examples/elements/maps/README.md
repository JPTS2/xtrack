<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# Maps

Represent transport directly, either by a local Taylor approximation to a known
line or by assigning an idealized segment's optics and phase advances. Both
examples run independently with a 1 GeV/c proton reference.

| Example | Lesson |
| --- | --- |
| [001_taylor_maps.py](001_taylor_maps.py) | Extract a quadratic map, construct the affine approximation in its coordinate basis, and inspect errors versus amplitude |
| [002_line_segment_map.py](002_line_segment_map.py) | Assign periodic optics and three phase advances, then compare tracked coordinates with the prescribed rotations |

`001` uses an exact drift followed by a thin quadrupole/sextupole/octupole kick
as the reference. Open-line Twiss conditions identify the on-axis expansion
trajectory. `SecondOrderTaylorMap.from_line` extracts coefficients numerically;
there is no periodic-solution assumption for this reference line.

The two Taylor elements use different longitudinal coordinate pairs:

| Element | Coordinates | Map |
| --- | --- | --- |
| `FirstOrderTaylorMap` | `(x, px, y, py, zeta/beta0, ptau)` | `m0 + m1 @ u` |
| `SecondOrderTaylorMap` | `(x, px, y, py, zeta, pzeta)` | `k + R @ z + sum_jk T[:,j,k]*z[j]*z[k]` |

Here `pzeta = ptau/beta0`. The script explicitly changes basis when constructing
the first-order coefficients, with the same fixed reference momentum at both
ends. A small momentum-offset probe demonstrates the converted longitudinal shear.
The tensor expression has no extra factor of `1/2`; mixed entries both
contribute. It prints extracted sextupole coefficients to show that convention.
The plotted residual norm combines only the two dimensionless transverse
canonical momenta. All transverse launch coordinates shrink towards the
expansion point together. Truncation and finite-difference errors limit the
amplitude range over which a Taylor approximation is useful.

`002` treats one segment as a repeatable one-turn surrogate. `qx`, `qy` and
`qs` are phase advances in cycles per pass, with angles `2*pi*q`. Scalar beta
and alpha parameters give the same optics at both ends. The selected
`linear_fixed_qs` mode is an assigned rotation in `(zeta, pzeta)`, with
`zeta_out = cos(mu_s)*zeta - bets*sin(mu_s)*pzeta` and `bets` in metres.
A nonzero initial momentum offset makes the sign and `bets` dependence visible.
The example compares periodic Twiss results and turn-entrance coordinates with
those assignments. It does not specify a magnetic layout or physical RF bucket.

For transfer-matrix calculations from physical lattices, see
[optics](../../twiss/). RF buckets, nonlinear longitudinal modes and acceleration
belong with the [RF learning routes](../../README.md#rf).
Retained larger studies are under [Taylor maps](../../taylor_map/) and
[segment maps](../../transfer_matrix/), with their own inputs and requirements.
