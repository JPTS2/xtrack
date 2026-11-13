<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# FCC-ee Z teaching model

`lattice.json` is the unchanged native environment supplied with the
[Xsuite CAS November 2026 FCC-ee exercises](https://github.com/xsuite/tutorial_cas_nov_2026/tree/main/02_fccee_optics_and_bb).
Credit belongs to the tutorial contributors. This is a stored machine input,
not an authored Python geometry definition. The tutorial itself starts from a
JSON model; its preparation script does not construct the geometry from scratch.

Load the environment with `xt.load('lattice.json')`, select
`env.fccee_p_ring`, then apply `env.call('strengths.py')` or
`env.call('strengths_larger_ip_beta.py')` using the paths to this folder.
The two readable Python settings expose 19 quadrupole/sextupole controls.
The snapshot already contains nominal values and the remaining machine
parameters/expressions. The Python files reapply only these 19 controls, so
switching settings retains element errors and does not recreate the model.

The stored reference is a 45.6 GeV/c positron. There are four interaction points
(`ipa`, `ipd`, `ipg`, `ipj`); nominal beta functions are 0.09 m horizontally and
0.0008 m vertically. The supplied larger-beta setting raises both by about a
factor eight. Its circuit families repeat around the ring. The independent
right-side matching circuits are already present in this environment.

The CAS preparation uses one-kick `mat-kick-mat` bend/quadrupole models and
one-kick expanded drift-kick-drift sextupoles. These choices support a compact
teaching workload; they do not establish convergence of nonlinear acceptance,
error tolerances or a machine emittance budget. The stored RF cavity supports
6D radiation calculations after explicit energy-loss compensation/tapering.
Radiation and beam-beam interactions are off in the supplied setting.

The stored BPM markers and horizontal/vertical correctors provide observation
and steering sites. See the
[machine-study guide](../../../examples/machine_studies/fcc_ee/README.md)
for the assumptions and relationship between the two error scales.

The lattice SHA-256 is
`2e0c365c2bbbb93aad2def7d054819adc474a772e878c89e549eb2b5dff8786d`.
It identifies the exact tutorial input. The separate
FCC-t detector-solenoid study uses its own retained sequence and field table.
