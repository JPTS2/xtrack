<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# Native example lattices

| Model | Line name | Geometry |
| --- | --- | --- |
| [ring/lattice.py](ring/lattice.py) | `ring` | 40 m ring with eight FODO cells and observation markers at 0 and 20 m |
| [transport/lattice.py](transport/lattice.py) | `transport` | 7.1 m line with three quadrupoles and two bends |

Load a definition with `env = xt.load(path)` and select `env['ring']` or
`env['transport']`. Each model supplies its reference particle and strengths.

## Reference and modelling

Both models use a 1 GeV/c electron reference (`p0c = 1e9` eV). Lengths are in
metres, bend angles in radians and quadrupole `k1` in m^-2. The variables `kqf`
and `kqd` drive the corresponding quadrupole families through expressions.

Bodies use the adaptive model with linear bend edges, zero edge angles and
zero fringe integrals. There is no RF, radiation, spin tracking, collective
force, error or aperture model. Use 4D optics; the transport line needs entrance
Twiss conditions.

The ring baseline has `kqf = 1.194864989644536` and
`kqd = -1.4872100226207832` m^-2, giving `Qx = 1.62` and `Qy = 1.58`.
The transport baseline has `kqf = 0.35` and `kqd = -0.30` m^-2.

## Coordinates

`x`, `y`, `s` and `zeta` are in metres. `px` and `py` are canonical momenta
normalized to the reference momentum, not slopes. For the reference species,
`delta = (p - p0) / p0`. The time convention is
`t = time_s + (s - zeta)/(beta0*c)`; positive `zeta` means earlier arrival
at fixed `s`. Normalized launch coordinates use the stated normalized
rms emittances; a launch scan is not a statistical bunch distribution.

The strengths describe the model at its stated reference momentum. Changing
that reference alone does not represent acceleration with fixed magnet fields.
