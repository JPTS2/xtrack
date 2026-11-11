<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# Simplified ELENA field-expansion model

`lattice.py` loads natively with `xt.load` and supplies `env['elena']`.
It retains the geometry, strengths and fitted dipole coefficients from
`examples/bfieldexpansion/elena.py`, without refitting or regenerating numbers.

The model describes 100 MeV/c antiprotons in the high-energy ELENA configuration:
six 60-degree bends of nominal radius 0.927 m and twelve powered quadrupoles.
Each dipole uses 33 polynomial field segments, including 0.285 m straight
extensions at both ends. RF, apertures, cooler fields, correctors, skew
quadrupoles and separate sextupoles are omitted. This is a 4D teaching model,
not a complete operational machine model.

The original adaptation used the `bpmeth` ELENA example and the
`acc-models-elena` sequence and high-energy strengths. The model header retains
the field-map hash. The original fit used `MagnetFromFieldmap(order=3,
hgap=0.038, symmetric=True, step=50, radius=0.0025,
design_field=1.000447/0.927)` on columns `[0, 1, 2, 7, 8, 9]` of
`dipole/ELENA_fieldmap.csv`. Knots crossing a nominal bend edge were split to
change reference curvature; coefficients were translated to each segment's
longitudinal origin. The original field map is not included,
so the fit cannot be regenerated from this repository alone. Construction and
execution need neither `bpmeth` nor MAD-X.

Normal coefficients multiply `s**j * x**i / i!` in `By / (B rho)`;
`knc[i, j]` therefore has units `m**(-(i+j+1))`. Curvature specifies the reference
geometry independently of field strength. Canonical momenta are carried between
segments (`pkin_const=False`), with exact drifts and ten RK4 steps per segment.
RK4 is not exactly symplectic; long-term stability is not established by the
one-turn machine-study example.
