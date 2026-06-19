# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

"""CPU-only FCC emittance check for ``quantum`` versus ``quantum-kick``.

Goal
----
Run a small lattice-level tracking-observable comparison on CPU.

Default target for the implemented example:

- FCC lattice;
- CPU context only;
- ``1E2`` particles;
- ``1E4`` turns;
- all particles initially at zero;
- compare emittance or beam-size observables from ``quantum`` and
  ``quantum-kick`` within an explicit tolerance.

What this gives us
------------------
A real tracking-observable sanity check. This is not intended to be a
high-statistics equilibrium-emittance certification.
"""


if __name__ == "__main__":
    print(__doc__)
