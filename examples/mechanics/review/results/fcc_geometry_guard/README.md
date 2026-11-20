# Packed FCC tracking-entry validation

See [the design note](../../../designs/2026-11-20_bulk_geometry.md).
[`result.json`](result.json) separates full reference validation, initial guard
preparation and repeated packed tracking-entry checks. It does not time a GPU
or perform tracking. One provisional support per quadrupole is a setup proxy.

Run from the Xtrack checkout with the same pinned external input as
[the earlier scaling probe](../fcc_scaling/README.md):

```bash
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python examples/mechanics/review/fcc_geometry_guard.py
```

The local macOS process used the same compiler/linker overrides as the other
native-kernel probes. Hashes record the implementation at measurement time;
results are not silently relabelled as validation of later source revisions.
The guard is checked at tracking entry, not each turn.
