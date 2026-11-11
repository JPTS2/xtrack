<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# LEP: polarization recovery

[001_polarization_recovery.py](001_polarization_recovery.py) compares detector
solenoids alone, the addition of vertical spin bumps, and the addition of stored
skew-quadrupole coupling corrections. Follow the orbit and energy sensitivity of
the invariant spin field as the predicted equilibrium polarization recovers.

The native model in
[`test_data/spin_refs_bmad`](../../../test_data/spin_refs_bmad/) represents a
45.585 GeV/c positron ring. Its bend fields and edges were simplified for Bmad
comparisons; it is an illustrative LEP model, not an operational machine setup.
The stored reference includes its mass, positive charge and magnetic-moment
anomaly. The example preserves these inputs and uses no external tracking code.

The calculation uses fixed-energy 4D optics. Radiation integrals predict
equilibrium polarization and polarization/depolarization time constants;
polarization build-up is not tracked. The plotted projection along the local
bending field is independent of the arbitrary sign chosen for the invariant
spin axis. The reference Sokolov–Ternov limit is parallel to the magnetic field
for positrons.

The correction settings are already installed in the model; this example does
not rematch them. The corresponding base, spin-bump and corrected comparisons
are covered by [`tests/test_spin.py`](../../../tests/test_spin.py). The source
model is `test_data/lep/lep_sol.json`; the `004a`–`004c` generators in
`test_data/spin_refs_bmad` apply the Bmad comparison simplifications. The
underlying native spin-field and polarization operations are introduced in
[spin](../../spin/README.md).
