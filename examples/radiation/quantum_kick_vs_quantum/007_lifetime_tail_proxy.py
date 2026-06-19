# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

"""Tail-sensitive lifetime proxy for ``quantum-kick`` validation.

Goal
----
Add a physics check focused on rare radiation events that could affect lifetime
or losses.

This example should eventually:

- define energy or kick thresholds relevant to loss/lifetime sensitivity;
- compare probabilities above those thresholds for ``quantum`` and
  ``quantum-kick``;
- include thresholds derived from high quantiles of the ``quantum`` reference;
- keep the default runtime reasonable while exposing knobs for larger
  certification-scale Monte Carlo.

What this gives us
------------------
A first observable-level tail check, separate from bulk agreement and separate
from full machine-specific lifetime studies.
"""


if __name__ == "__main__":
    print(__doc__)
