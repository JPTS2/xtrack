# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

"""Compare algorithmic sampler-call counts for the two models.

Goal
----
Explain the expected speedup mechanism without making this a wall-clock timing
benchmark.

This example should eventually:

- compute expected photon-by-photon samples for ``quantum``;
- compute expected table lookups for ``quantum-kick``;
- separate direct-table use from fallback/decomposition use;
- plot call counts as a function of mean emitted photon count.

What this gives us
------------------
A compact explanation of why ``quantum-kick`` can be faster, while actual CPU
and GPU timing comparisons remain in ``examples/tracking_time``.
"""


if __name__ == "__main__":
    print(__doc__)
