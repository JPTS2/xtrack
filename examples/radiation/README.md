# Radiation examples

This folder contains examples related to synchrotron radiation in Xtrack.

The sequence starts with the underlying photon spectrum and then moves to
tracking examples:

- `001_synchrotron_radiation_spectrum.py` demonstrates the normalized
  single-photon SynRad spectrum, photon sampling, compound-Poisson total-energy
  loss, analytic moments and high-loss tails;
- `002_single_dipole.py` compares radiation in a tracked dipole against the
  analytical spectrum.

The current layout is historical and needs a wider cleanup. In particular,
the tapering examples and radiation-integral examples should be reviewed
together with this folder, because they are part of the same user workflow:

- enable mean radiation for optics and radiation analysis;
- compensate radiation energy loss when needed;
- study stochastic photon emission with `quantum`;
- validate and benchmark `quantum-kick` against `quantum`.

For now, the existing general radiation examples are kept in place. The
development examples specific to validating `quantum-kick` against `quantum`
have been collected in:

```text
quantum_kick_vs_quantum/
```

That subfolder is intentionally narrow. It should contain examples and checks
that compare only the public radiation models `quantum` and `quantum-kick`,
without reintroducing intermediate development modes. Some files currently in
that folder still have historical names from the development branch; these
should be renamed and simplified in the next cleanup pass.
