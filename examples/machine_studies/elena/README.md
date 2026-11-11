<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# ELENA field transport

Run [001_field_transport.py](001_field_transport.py) to inspect curved dipoles
with fitted fringe fields in a simplified ELENA extraction setting. It contrasts
magnetic field extent with reference curvature, computes periodic 4D optics and
survey geometry, then follows three small-amplitude probes for one turn.

The reference is a 100 MeV/c antiproton. Fields are plotted normalized to the
reference magnetic rigidity, not in tesla. The three trajectories are probes,
not a beam distribution or an acceptance calculation.

The [native model](../../../test_data/example_lattices/elena_field/README.md)
documents the retained field fit and omitted machine systems. RF, apertures and
cooling are absent. The example uses RK4 integration and does not establish
long-term stability.
