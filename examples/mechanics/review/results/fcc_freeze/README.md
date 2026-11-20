# Freeze-scheduler and particle-map evidence

Interpretation is in the [GPU/freeze review](../../../designs/2026-11-20_gpu_freeze_runtime.md).
These measurements isolate costs; they are not physical girder assignments or
GPU performance predictions. The native implementation lives in Xtrack.

## Fixed-field tracking floor

[fcc_tracking_floor.py](../../fcc_tracking_floor.py) loads the user-selected
parameterized FCC Z positron input, changes small alignment scalars in memory,
and restores them on completion. It does not install mechanics or change the
input file. One-particle, 32-particle and 128-particle results are in
[tracking_floor_cpu.json](tracking_floor_cpu.json):

```bash
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python examples/mechanics/review/fcc_tracking_floor.py \
    --turns 8 --particles 1 32 128 --repeats 3
```

That initial run contains large host outliers in one single-particle case. The
separate [64-turn, seven-repeat single-particle capture](repeat_1p/tracking_floor_cpu.json)
is retained without replacing the original evidence. It used the same script,
with its output-directory constant redirected to `repeat_1p`:

```python
import importlib.util
from pathlib import Path
import sys

path = Path('examples/mechanics/review/fcc_tracking_floor.py')
spec = importlib.util.spec_from_file_location('fcc_floor', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.OUTPUT = module.OUTPUT / 'repeat_1p'
sys.argv = [str(path), '--turns', '64', '--particles', '1', '--repeats', '7']
module.main()
```

Both captures record script and lattice hashes, context, turn count, repeats,
individual durations and particle survival. They do not capture a physical
support map or uncertainty/confidence interval. CPU geometry sharing differs
from CUDA's per-particle thread execution.

## Particle-clock scalar access

[mechanics_clock_io.py](../../mechanics_clock_io.py) uses a small native line
after prepared mechanics initialization and counts accesses to its particle
buffer. No FCC load or geometry preparation occurs in the instrumented region.

```bash
PYTHONPATH=. python examples/mechanics/review/mechanics_clock_io.py
```

- [clock_io_collective_reference.json](clock_io_collective_reference.json)
  is the earlier capture, before the direct fixed-reference scheduler was added.
  It shows ten clock-field reads and five writes per turn, plus call overhead.
- [clock_io.json](clock_io.json) is the direct prepared-scheduler capture. Its
  counts are identical for one, four and sixteen turns: the host scalar-I/O
  slope is zero. Entry validation and bounded final bookkeeping remain.

These are CPU method-call counts used to audit the transfer pattern. They are
not CUDA latency measurements. The GPU eligibility reduction is performed once
per call, not per turn. Actual GPU execution and performance remain unverified
on this host.
