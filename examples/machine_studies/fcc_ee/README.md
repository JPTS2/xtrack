<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# FCC-ee detector-solenoid correction

[001_detector_solenoid_correction.py](001_detector_solenoid_correction.py)
follows one insertion through field installation, local correction and a new
periodic-optics calculation. It uses the stored FCC-t sequence at
182.5 GeV/c with positrons and corrects the region around `ip.1`.
The machine geometry matters: the final-doublet locations provide the skew,
normal-gradient and orbit-corrector controls.

The native Xtrack parser reads the legacy sequence in
[`test_data/fcc_ee`](../../../test_data/fcc_ee).
This is a stored machine snapshot, without MAD-X macros or a MAD-X runtime.
It is not a claim about the current FCC-ee design.

The supplied `Bz_closed_before_quads.dat` profile contains the detector field
and opposite-polarity screening regions. It is represented by piecewise-linear
`VariableSolenoid` slices in the detector frame. `ks = Bz / rigidity0` uses the
signed reference rigidity; here the positron charge is positive. The ±15 mrad
entry/exit rotations and transverse translations give the field-free beam
`kin_xp = tan(15 mrad)` at the IP in detector coordinates. The example reports
physical slopes as well as canonical-coordinate matching residuals.

The matching first separates orbit, coupling and normal optics, then solves
them together. Each half-insertion is initialized at its unperturbed external
boundary. The corrected periodic solution independently checks the resulting
orbit, projected normal-mode beta functions and cross-plane transfer blocks.
The plot shows the field profile, local vertical orbit and `bety1` leakage.
The corrected orbit closes at the boundaries and IP; its excursion inside the
detector remains.

RF voltages are zero and radiation is off. This lesson does not include
radiation-induced vertical emittance, field errors, higher transverse field
derivatives, full detector-field-map transport or correction at all four IPs.
In particular, cancelling a longitudinal field integral alone would not restore
the orbit and coupled optics demonstrated here.

The configuration and correction stages follow the existing
[`test_fcc_ee_solenoid_correction_new_optimize_api.py`](../../../tests/test_fcc_ee_solenoid_correction_new_optimize_api.py)
regression. The separate FCC-Z local/nonlocal compensation studies in the legacy
`examples/fcc_ee_solenoid` folder use another lattice and remain distinct.
For the generic map constraints, see
[coupling bumps](../../matching/011_coupling_bumps.py).
