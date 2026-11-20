# Public bulk FCC setup

See [the design note](../../../designs/2026-11-20_bulk_geometry.md).
[`result.json`](result.json) measures the original selected development lattice,
with live length and strength expressions preserved. One support per quad is a
scaling proxy, not an engineering girder assignment.

Run from the Xtrack checkout:

```bash
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python examples/mechanics/review/fcc_bulk_setup.py
```

The isolated model checkout is the same pinned input documented in
[the earlier scaling probe](../fcc_scaling/README.md). Script/native source hashes
record the implementation present during this measurement. This file does not
claim timing or validation of later source revisions.
