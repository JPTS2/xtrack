<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# Native example lattices

| Model | Line name | Geometry |
| --- | --- | --- |
| [demo_ring/lattice.py](demo_ring/README.md) | `demo` | Instrumented 1.5 km ring with a low-beta IP and separate geometry/strength inputs |
| [ring/lattice.py](ring/lattice.py) | `ring` | 40 m ring with eight FODO cells and observation markers at 0 and 20 m |
| [transport/lattice.py](transport/lattice.py) | `transport` | 7.1 m line with three quadrupoles and two bends |
| [rf_ring/lattice.py](rf_ring/lattice.py) | `rf_ring` | Native 40 m magnetic ring followed by one zero-length sinusoidal RF cavity |

Load a definition with `env = xt.load(path)` and select `env['ring']` or
`env['transport']`. Each model supplies its reference particle and strengths.
The RF definition supplies both `env['ring']` (RF-free) and `env['rf_ring']`.

## Reference and modelling

All models use a 1 GeV/c electron reference (`p0c = 1e9` eV). Lengths are in
metres, bend angles in radians and quadrupole `k1` in m^-2. The variables `kqf`
and `kqd` drive the corresponding quadrupole families through expressions.

The compact ring and transport magnetic bodies use the adaptive model with
linear bend edges, zero edge angles and zero fringe integrals. The demonstration
ring uses `bend-kick-bend` bends; its configuration is documented in
[demo_ring/README.md](demo_ring/README.md).
The compact magnetic ring and transport line have no RF;
use 4D optics, supplying entrance conditions for the transport line. There is no
radiation, spin tracking, collective force, error or aperture model.

The ring baseline has `kqf = 1.194864989644536` and
`kqd = -1.4872100226207832` m^-2, giving `Qx = 1.62` and `Qy = 1.58`.
The transport baseline has `kqf = 0.35` and `kqd = -0.30` m^-2.

## RF-ring qualification

The RF ring retains those magnetic strengths and adds a 100 kV cavity at
`s = 40` m, just before the next turn's entrance. Its frequency is harmonic 20
of `beta0*c/40`, approximately 149.896 MHz. Phase is pi radians: the current
public `Cavity` reference-species kick uses `abs(q0)*V*sin(phase - omega*zeta /
(beta0*c))`. Together with the positive slip factor, this gives stable focusing.

The synchronous solution has synchrotron tune approximately 0.01139322. Use 6D
Twiss to solve all three modes. Tracking uses sinusoidal RF, while Twiss and
covariance describe the local linearization around the synchronous orbit.
There is no acceleration or radiation-loss compensation. The RF-free companion
remains available for fixed-momentum orbit and revolution-period studies.

## Coordinates

`x`, `y`, `s` and `zeta` are in metres. `px` and `py` are canonical momenta
normalized to the reference momentum, not slopes. For the reference species,
`delta = (p - p0) / p0`. The time convention is
`t = time_s + (s - zeta)/(beta0*c)`; positive `zeta` means earlier arrival
at fixed `s`. Normalized launch coordinates use the stated normalized
rms emittances; a launch scan is not a statistical bunch distribution.

The strengths describe the model at its stated reference momentum. Changing
that reference alone does not represent acceleration with fixed magnet fields.
