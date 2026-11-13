<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# Native example lattices

| Model | Line name | Geometry |
| --- | --- | --- |
| [demo_ring/lattice.py](demo_ring/README.md) | `demo` | Instrumented 1.5 km ring with a low-beta IP and separate geometry/strength inputs |
| [fcc_ee_z/lattice.json](fcc_ee_z/README.md) | `fccee_p_ring` | CAS four-IP FCC-ee Z model, with separate nominal/larger-beta strengths |
| [elena_field/lattice.py](elena_field/README.md) | `elena` | Compact antiproton ring with fitted curved dipole fields |
| [pimms_extraction/lattice.py](pimms_extraction/README.md) | `pimms` | Compact proton ring for resonant extraction |
| [transport/lattice.py](transport/lattice.py) | `transport` | 18.3 m, three-cell alternating-gradient transport |

Load a definition with `env = xt.load(path)` and select `env['demo']` or
`env['transport']`. Each model supplies its reference particle and strengths.
The demonstration ring uses one geometry for magnetic, powered-RF, skew and
nonlinear studies. Each example explicitly selects the setting it needs.

## Reference and modelling

The demonstration ring and transport use a 1 GeV/c electron reference (`p0c = 1e9` eV). Lengths are in
metres, bend angles in radians and quadrupole `k1` in m^-2. The variables `kqf`
and `kqd` drive the corresponding quadrupole families through expressions.

The demonstration ring uses `bend-kick-bend` bends and defaults to zero RF
voltage, skew fields, nonlinear strengths, errors and operating trims. Its
[model README](demo_ring/README.md) specifies the optics, device names,
strengths and qualified RF configuration. Radiation and collective fields are
not enabled by the model loader.

The transport uses adaptive magnetic bodies with linear bend edges, zero edge
angles and zero fringe integrals. Its family settings are `kqf=1.8` and
`kqd=-1.8` m^-2. Each 6.1 m cell has one focusing and one defocusing
quadrupole and two oppositely signed bends (+0.045 and -0.030 rad). It has no RF and needs entrance conditions for 4D Twiss.

## Coordinates

`x`, `y`, `s` and `zeta` are in metres. `px` and `py` are canonical momenta
normalized to the reference momentum, not slopes. For the reference species,
`delta = (p - p0) / p0`. The time convention is
`t = time_s + (s - zeta)/(beta0*c)`; positive `zeta` means earlier arrival
at fixed `s`. Normalized launch coordinates use the stated normalized
rms emittances; a launch scan is not a statistical bunch distribution.

The strengths describe the model at its stated reference momentum. Changing
that reference alone does not represent acceleration with fixed magnet fields.
