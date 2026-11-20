# FCC scaling evidence

See the [adversarial review](../../../designs/2026-11-20_fcc_scaling_review.md)
for interpretation and the distinction between supported operations and private
experiments. No native mechanics implementation was changed for this review.

Primary input: `fccee_z_beam_1_positrons.json`, from the user-selected CERN branch
`dev/108.1.0`, commit `8805a5bb8bd568f5de2e1558037371905ecfc985`. Its `VERSION`
reports `dev/LCC_108.3.0`. The JSON has SHA-256
`dd7c7749d1bedfa5faf41bbfd885312bbc38bec8a817460bc0627f3c4b6094fe`.

From the Xtrack checkout, place an isolated input checkout at
`../inputs/acc-models-fcc-ee-108.1.0`, then select that exact commit. These input
files are not copied into the Xtrack source tree. The model and catalogue
repositories are not modified by the probes.

```bash
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python examples/mechanics/review/fcc_scaling.py --phase native
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python examples/mechanics/review/fcc_scaling.py --phase bulk
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python examples/mechanics/review/fcc_scaling.py --phase field
PYTHONPATH=. python examples/mechanics/review/fcc_inventory.py
python examples/mechanics/review/fcc_scaling_plots.py
```

The local macOS environment also used its process-local compiler/linker overrides
and `XSUITE_ALLOW_KERNEL_COMPILATION=1`; these are not portable requirements of
the model. CPU/BLAS thread settings and package versions are recorded.

- [Native JSON](native.json), [stdout](native.txt), [stderr](native.stderr.txt):
  original parameterized input; attachment stops at the seventh quadrupole,
  while timings continue on the six supported roots. `status='passed'` means
  the probe completed, not that the full public FCC workflow is supported.
- [Bulk JSON](bulk.json), [stdout](bulk.txt), [stderr](bulk.stderr.txt): explicitly
  detached numeric snapshot and private bulk bookkeeping, then update, tracking,
  precision, buffer-I/O and selected-history measurements.
- [Field JSON](field.json), [stdout](field.txt), [stderr](field.stderr.txt): one
  finite-mode ground field, four feet per provisional support, rigid fitting,
  synthetic response classes and exact whole/split causal-filter agreement.
  The direct source call deliberately bypasses an overestimated public guard;
  it does not bypass a dense covariance allocation.
- [Inventory](inventory.json), [stdout](inventory.txt): expression coverage and
  Xobjects buffer capacities.
- [Candidate associations](candidate_associations.csv): catalogue-style name
  links and nearest-quad attribution only. Blank engineering-support/class
  columns mean actual girder rules are still required.
- [Measured costs](costs.svg): serial CPU measurements, not GPU predictions.

The imported optics, all quadrupole sites and arbitrary response classes are not
claimed as an engineering support layout. In particular the ground-noise cost
configuration (128 Hz, 256 modes) is not a converged FCC noise study. Process RSS
includes Python, lattice objects, transient arrays and experiments; it is not
device-memory usage.
