<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# First working draft: scope and validation

The deliverable is a native experimental rigid-mechanics UI with runnable
teaching examples. It is suitable for early developer review without finishing
packed multibunch tracking, long-run source streaming or the engineering FCC
model. The [API contract](API.md) states the supported behavior and guards.

Start with [001](001_rigid_girder.py) and the full
[FCC case 016](016_fcc_ground_motion.py); [004](004_demo_ring.py) supplies a
portable surveyed ring and [015](015_bunch_phases_and_twiss.py) adds bunch phases.
The other twelve lessons remain independent
and available. The implementation lives in `xtrack/mechanics`; examples do not
import a scratch implementation or feedback code.

## Integration delivered in this slice

- Standard Tracker and prepared mechanics now share kernel, TrackerData,
  monitor, I/O and flag binding. Range/scheduling choices remain with Tracker.
- Tracker configuration is restored even if mechanics cleanup raises.
- Copy/export validates geometry and alignment ownership before saving. A raw
  conflicting edit cannot be silently adopted by copying the environment.
- A cleared manager can reload definitions despite cached line accessors.
  Real reference configuration still prevents accidental overwrite.
- Public `new_bodies` batches magnet/pickup attachments with one survey, so
  the full FCC girder → magnet hierarchy does not repeat setup per element.
- Regression cases cover frozen copies, three placement layers, removal
  failures, source interruption/resume, both monitor types, EBE fallback and
  independent clocks on a shared support with opposite traversal.
- The explicit hardware command records actual context, source hashes and test
  counts. An unavailable device never becomes a CPU fallback success.

The numeric-field runtime remains private. Occurrence/phase buffers can replace
it behind the same model UI. No new public scheduling or source-replacement
framework is introduced just to open this first draft.

## Reproduction

Use a source checkout with ordinary Xsuite dependencies and pytest. These
commands do not require the external FCC input:

```bash
PYTHONPATH=. XSUITE_ALLOW_KERNEL_COMPILATION=1 python -m pytest -q \
    tests/test_mechanics*.py

PYTHONPATH=. XSUITE_ALLOW_KERNEL_COMPILATION=1 python \
    examples/mechanics/review/run_examples.py \
    --examples 001 004 015 --output /tmp/mechanics-examples

PYTHONPATH=. XSUITE_ALLOW_KERNEL_COMPILATION=1 python \
    examples/mechanics/review/hardware_acceptance.py \
    --context cpu --output /tmp/mechanics-cpu
```

Compiler selection is installation-specific. The
[hardware instructions](review/HARDWARE_ACCEPTANCE.md) include this local macOS
configuration, explicit OpenMP/CUDA/OpenCL choices and the FCC timing command.
Do not compare a cold example's total time with warm tracking throughput.

The selected draft patch was replayed on an archived clean baseline. The full
native suite ran there, importing that checkout rather than another installed
Xtrack. Validation on 20 November 2026:

| Check | Outcome |
| --- | --- |
| Complete mechanics regressions on the clean replay | **172 passed**, 260 s |
| Targeted ordinary Tracker regressions after shared launch binding | **60 passed**; partial turns, collective ranges, backtracking, monitors and I/O |
| Explicit hardware acceptance, serial CPU | **19 passed**, zero skips |
| Explicit hardware acceptance, two-thread OpenMP | **19 passed**, zero skips |
| CUDA / OpenCL preflight | **Unavailable**, exit 2; no CPU fallback |
| Wheel and source-distribution build | Passed; mechanics Python/header files included |
| Isolated wheel install and compiled tracking | Passed; installation/clear restoration checked |
| Portable entry examples 001 / 004 / 015 | Passed on the current native sources |
| Requested FCC lesson 016 | Passed; all default particles survived; four figures |
| FCC plane-wave branch | Passed with explicit wave-only parameter overrides |

These counts overlap and must not be summed as independent tests. The hardware
runner rejects skipped/selected-away cases and source changes during a run.
The extraction manifest separately pins every selected file and baseline/source
commits. Historical captures retain their original hashes.

## Performance interpretation

The requested FCC input is the positron Z line from `dev/108.1.0` in
`acc-models-fcc-ee`, commit
`8805a5bb8bd568f5de2e1558037371905ecfc985`. Its JSON SHA-256 is
`dd7c7749d1bedfa5faf41bbfd885312bbc38bec8a817460bc0627f3c4b6094fe`.
It has 32,204 occurrences and 2,663 quadrupoles. Cost-probe supports are
provisional groupings, not engineering girder assignments. Radiation is off;
the benchmark measures turn-held prescribed motion.

Warm tracking comparisons include entry guards and final ownership bookkeeping.
They exclude cold compilation and source preparation, which are reported
separately. The earlier 128-turn CPU cases passed the requested twofold ceiling;
this slice repeats representative cases after the integration change.
Direct spatial-provider evaluation and repeated short calls remain more
expensive. A pass on the measured warm path does not establish a universal
twofold guarantee.

The new [FCC lesson walkthrough](FCC_DEMO.md) reports the actual complete
floor/girder/magnet/virtual-pickup case: about **24 s** including preparation,
repeated measurements and plotting; default moving/nominal **1.415×**, moving /
frozen-installed **1.251×**. Its separate longer 32-particle comparison is
**1.049× nominal**. The whole lattice is tracked, with mechanics assigned to all
2,663 quads. It does not claim that every dipole/sextupole has an engineering
support assignment. Source preparation, peak host memory and finite startup are
explicit in the walkthrough.

Actual CUDA/OpenCL execution is unavailable on this workstation. CPU/OpenMP
tests, source inspection and portable syntax do not constitute GPU validation.
The first draft can be reviewed with that status explicit; performance claims
must name the measured hardware, workload and source model.

## Focused draft selection

The reviewable patch comprises the native implementation and four integration
files, mechanics regressions, all sixteen teaching lessons, the bundled demo
ring, these contract/evidence documents and three reproduction commands.
Historical captures, long design records, old experiments and provisional
catalogue exports stay on the local exploration branch. They are not required
dependencies of the implementation.

The local projection tool under `review/draft/` produces a verbatim selected
patch and per-file manifest, and can replay it on an archived baseline. It
creates no branch and changes no index, history or remote. Clean commit slicing
and publication remain the user's later review step.
