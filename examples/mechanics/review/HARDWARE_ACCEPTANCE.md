# Mechanics hardware acceptance

Run from the Xtrack checkout with its source on `PYTHONPATH`. The command uses
the native regression tests on exactly one requested context. It exercises
analytic and sampled motion, all six rigid coordinates, finite hierarchy and
absolute-child composition, very small offsets at large survey coordinates,
reference parity, monitors, split-call clocks and geometry/alignment guards.
The shared-support case checks opposite traversal axes, independent prepared
arrays and revolution clocks, and disabling one beam while the other keeps
running. Distinct element offsets can share the Environment's Xobjects storage.

```bash
PYTHONPATH=. XSUITE_ALLOW_KERNEL_COMPILATION=1 python \
    examples/mechanics/review/hardware_acceptance.py \
    --context cpu --output /tmp/mechanics-cpu

PYTHONPATH=. XSUITE_ALLOW_KERNEL_COMPILATION=1 python \
    examples/mechanics/review/hardware_acceptance.py \
    --context openmp --threads 2 --output /tmp/mechanics-openmp

PYTHONPATH=. XSUITE_ALLOW_KERNEL_COMPILATION=1 python \
    examples/mechanics/review/hardware_acceptance.py \
    --context cuda --device 0 --output /tmp/mechanics-cuda

PYTHONPATH=. XSUITE_ALLOW_KERNEL_COMPILATION=1 python \
    examples/mechanics/review/hardware_acceptance.py \
    --context opencl --device 0.0 --output /tmp/mechanics-opencl
```

Use the compiler settings required by your Xsuite installation. In this local
macOS environment, source compilation additionally uses the conda Clang:

```bash
export CC=/opt/miniconda3/envs/xsuite/bin/clang
export LDSHARED='/opt/miniconda3/envs/xsuite/bin/clang -bundle -undefined dynamic_lookup'
```

The output directory contains `acceptance.json`, `tests.txt` and `pytest.xml`.
Context-construction or allocation failure instead produces `preflight.txt`.
Missing dependencies/devices and construction failure are unavailable; a context
that constructs after device discovery but fails the allocation/transfer check
is a failure. CuPy availability is checked explicitly because its Xobjects
context can construct lazily without CuPy installed. Importing another Xtrack
installation is incomplete; local hashes must describe the code being exercised.
The JSON records the requested/actual context, device name when available,
source hashes, exact test command and test counts. Exit codes are:

| Code | Meaning |
| --- | --- |
| 0 | All selected tests passed on the requested context |
| 1 | Conformance failed |
| 2 | Requested context unavailable; no CPU fallback |
| 3 | Incomplete, including skips or sources changing during the run |

The selection currently contains 19 cases per context. `PYTEST_ADDOPTS` and
configured pytest `addopts` are neutralized, and an unexpected count is
incomplete. Inherited `PYTEST_ADDOPTS` is recorded for reproducibility.

An unavailable CUDA/OpenCL result establishes no GPU correctness or throughput
claim. OpenCL device selection can name a CPU implementation; inspect the
recorded device and platform before describing it as a GPU result.

## FCC throughput

Conformance is separate from the requested twofold tracking-cost ceiling. Use
the requested FCC development input on the actual hardware, keeping the input,
particle count, seeds, motion accuracy and comparison basis identical:

```bash
PYTHONPATH=. XSUITE_ALLOW_KERNEL_COMPILATION=1 python \
    examples/mechanics/review/fcc_prepared_runtime.py \
    --context cuda --device 0 \
    --input /path/to/lattices/z/fccee_z_beam_1_positrons.json \
    --line fccee_p_ring --members all_magnets --hierarchy nested \
    --turns 128 --particles 1 32 --repeats 3 --short-calls \
    --output /tmp/mechanics-fcc-cuda
```

Repeat with `--members quads --hierarchy flat --source sampled_ground` to
exercise packed host-record upload and context composition. The source is a
short illustrative ground/response record, not an accuracy-converged FCC ground
model. Direct-field evaluation is a distinct provider-cost probe. There is no
automatic fallback if the selected device cannot execute the model.

The benchmark reports complete baseline and dynamic tracking calls, cold
preparation, source preparation, update-only cost and prepared array bytes.
Short calls include repeated entry validation. Compilation is outside warm
measurements. `update_host_buffer_io` counts Python Xobjects buffer hooks; it is
not a device profiler or a measure of all host/device transfers. Prepared array
bytes are not peak process/device memory. The reported twofold criterion applies
to each measured warm tracking case, and does not imply that source preparation
or every short-call pattern meets it.

These are explicitly turn-held, single-packet measurements. Packed bunch phases,
passage-time accuracy, aperture coupling and a long-run streaming provider need
their own acceptance cases. An experimental UI draft can be reviewed while real
GPU measurements are pending, provided its status is stated accurately.
