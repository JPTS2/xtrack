# Mechanics: approach decision and first-demo PR readiness

## Decision

Keep the physical model and UI. The exploration has established useful contracts
and demonstrated a fast implementation of them. It has also exposed where the
current execution backend stops being adequate. That calls for focused backend
work, not restarting the mechanical model or another broad survey of codes.

Prepared many-turn CPU tracking is already below the requested twofold ceiling
in the measured cases. That is a positive result and sufficient performance
evidence for an explicitly experimental first demo. Earlier wording about an
unmet gate combined this success with unoptimized providers, repeated short
calls and unmeasured GPU/packed-bunch operation. Those are different workloads.

The [recorded costs](../review/results/fcc_freeze_runtime/README.md) remain the
evidence: 128-turn calls are about 1.17/1.00 times baseline for quad supports,
1.81/1.09 for all magnets and 1.88/1.09 for a two-level hierarchy (one/32
particles). Consuming prepared ground/response records is about 1.19/1.02.
Direct continuous field evaluation is 2.74 times baseline for one particle;
short-record preparation plus tracking is 3.32; warm one-turn calls are
4.85–9.20. These last paths remain optimization targets. None of these CPU
measurements establishes a CUDA/OpenCL performance result.

## What is settled, and what remains provisional

| Layer | Current judgment | Consequence |
| --- | --- | --- |
| Environment-owned physical bodies, line-owned occurrence views | Keep | A shared support can have different traversal frames and explicit clocks. No implicit environment turn number is introduced. |
| Deliberate/reference, installation and dynamic layers | Keep | Preserve existing offsets and rolls, independent installation seeds, explicit removal and correction cases. |
| Rigid hierarchy and nominal geometry plus local `(d,R-I)` increments | Keep | Finite composition and tiny dynamic offsets coexist without moved-world subtraction. |
| Prescribed fields, measured records, port transport and conditional responses | Keep the concepts | Preserve units, frames, common realizations and relative/absolute semantics; separate provider definitions from run state. |
| Turn-held preparation at an explicit physical time | Keep as a selectable default | Passage/finer grids need an accuracy comparison. One turn number does not encode different bunch arrival phases. |
| Writing effective numeric element fields | Keep as an experimental/reference backend | It is fast for the demonstrated single-packet case. It cannot supply different simultaneous maps to packed bunches or resolve arbitrary shared/sliced occurrences. |
| Prepared occurrence/phase buffers | Preferred production direction | They address an identified representation requirement, rather than a hypothetical speed problem. Keep them behind the existing UI. |
| Girder-boundary transforms | Targeted optimization alternative | Compare on contiguous groups only if GPU map cost warrants it. Curvature, nested residuals, apertures and observers still need their frames. |
| Long-run source execution | Not settled | Direct mode evaluation is too expensive in the tested path; immutable full records are not a streaming design. |

The new [bunch example](../015_bunch_phases_and_twiss.py) uses the bundled ring
and normal `xt.Table` selection. It retains a last live mechanics scenario for
interactive inspection. The FCC configuration is exposed in its user parameters.
It computes actual frozen per-bunch Twiss and separately tracks the driven beam;
it does not label a sequence of frozen optics as dynamic equilibrium Twiss.

The example's bunches are independent sequential runs with different physical
time origins and a common prescribed source. This is already useful for API and
physics discussion. Packed multibunch tracking is a separate capability: one
shared element-field array cannot distinguish two bunch phases at once.

## What the additional code review found

The physical model is in fifteen native Python/header modules, approximately
3,800 lines at the preceding checkpoint. The integration outside that directory
is concentrated in the Environment/Line accessors, serialization and Tracker.
The branch's much larger diff also contains fifteen examples, review notes,
captured figures, logs and numerical evidence. Its raw line count is not the
size of the runtime to review; it is also not a suitable first PR diff as-is.

Two concrete collective integration issues were found and fixed in this review:

1. Temporary noncollective survey/Twiss views use distinct shallow environments
   but share the original environment's weak line set. Counting them as physical
   lines produced false shared-element errors. Ownership checks now exclude
   those foreign-environment views while retaining genuine same-world alias
   rejection. The regression test deliberately retains temporary views.
2. Automatic compiled preparation assumes geometry/static placement is frozen
   for a call. Arbitrary collective callbacks do not provide that guarantee.
   Auto mode now retains per-turn reference validation for collective lines;
   explicit compiled mode keeps the documented fixed-geometry contract. This
   distinction is tested with a geometry-changing callback and a fixed callback.

These are integration/lifecycle hardening findings. They do not require changing
the user-facing hierarchy, source vocabulary or three placement classes.
The [68-check targeted regression run](../review/results/demo_readiness/validation.json)
passed, including the new collective cases and the existing ownership, bulk,
runtime and geometry-guard cases. The new lesson was captured on both the bundled
ring and the requested FCC input. Actual GPU execution remains unavailable here.
The previous performance captures pin the earlier implementation; the two fixes
do not change the C kernels or the noncollective workloads used there. The raw
capture hashes are retained, not relabelled as newer measurements.

Other review findings remain relevant:

- The mechanics runtime has its own full-turn launch routine duplicating part
  of Tracker's setup. It is a useful measured prototype, but should be folded
  into a shared tracking hook/helper before a stable release. It must not grow
  into a second independent tracker implementation.
- Geometry dependency discovery currently traces Xobjects scalar reads during
  preparation. It fails closed for array-based geometry, and explicitly checks
  custom Python geometry. This is adequate for the demonstrated builtin scope;
  a reviewed geometry schema or explicit invalidation/session mechanism would
  be easier to maintain as a general feature.
- Advancing a source record/filter window is currently not a public run-state
  operation. Replacing a definition invalidates preparation. Long simulations
  need bounded mutable windows and filter state without rebuilding geometry.
- A resolved `RigidMotion` does not retain the entire floor → support → magnet
  → pickup response graph. Scripts remain inspectable, but a native response
  registry/provenance graph is still needed for a convenient full-machine UI.
- Slice/alias restrictions, attached standalone line-copy restrictions and
  aperture coupling are explicit. They should remain clear unsupported cases
  until their occurrence/frame handling exists.
- Deliberate numeric offsets are adopted without retyping. Expression-driven
  alignment fields remain rejected because the current backend writes those
  same fields. Preserving arbitrary live alignment expressions is another
  concrete reason for a non-overwriting occurrence backend.
- Sources are immutable definitions and there is no simple public replacement
  operation yet. The new comparison lesson rebuilds scenarios between source
  frequencies and leaves the last one live. A small replacement/removal API is
  an ergonomic decision for the draft; it need not become a large scenario engine.
- Moving electrical pickup helpers are not yet an online moving-BPM monitor
  and feedback integration. Those must be tested jointly before FCC correction
  studies claim that capability.
- The portable kernels have CPU/OpenMP tests and offline GPU syntax checks.
  Real GPU execution, memory ownership and throughput still need hardware
  validation; the fake upload-policy test is not a CUDA execution test.
- Actual FCC support assignments, measured ground cross spectra and mechanical
  response data remain physical inputs to obtain and validate. Catalogue naming
  associations and synthetic benchmark response classes do not replace them.

## Further investigation should be targeted

Another open-ended survey of accelerator/general vibration codes is unlikely to
change the ownership or rigid-motion decisions. Three bounded checks can settle
the remaining execution choices:

1. **Actual GPU acceptance:** compare nominal, frozen-error and moving cases
   with identical particles, seeds, source accuracy and observables. Include
   cold preparation, streaming preparation and peak device memory separately.
   Exercise all six coordinates, both frame directions and sampled inputs.
2. **Occurrence/phase consumption:** demonstrate a small phase-aware buffer
   behind the current UI; compare independent bunch runs against packed bunches.
   Check exact zero, small increments, slice/alias ownership and pickup frames.
   Only compare a boundary variant on the same fixture if map overhead remains
   significant. Prescribed occurrence-time preparation may avoid extra launches.
3. **Bounded source execution:** compare fixed-contact projection and stationary
   mode-domain response with causal block streaming. Require class coherence,
   source-selection invariance and chunk continuity at a stated accuracy.

These tests have acceptance criteria. They are not reasons to postpone all
developer feedback until a complete FCC tool exists.

## Proposed first draft PR

Purpose: obtain feedback on a native experimental rigid-mechanics UI, placement
ownership, inspection and prescribed motion, with a measured fast backend.
Do not title or describe it as a complete FCC ground-motion/feedback solution.

Suggested title: **Experimental rigid-body mechanics and prescribed motion**.

The reviewer entry points should be short and visible:

- 001: deliberate/reference, installation and vibration; disable and clear.
- 004: apply mechanics to an existing surveyed ring.
- 015: bunch clocks, frozen Twiss and driven tracking, with ordinary tables.
- 005/006/013 provide deeper feet/response/pipeline examples when wanted.

The local exploration retains all examples and evidence. A fresh review branch
should separate the native change and selected examples/tests from historical
captures, repeated benchmark logs, the provisional 20k-row catalogue export and
long design records. Retain reproducible benchmark commands and a small evidence
summary; keep the full exploratory material here. Do not rewrite this branch to
manufacture a clean history.

Remaining work for that draft is a small number of delivery slices:

1. **Extraction and API pass:** choose supported experimental scope; document
   clocks, source references, lifecycle and unsupported cases; review naming and
   remove stale host-only wording. Arrange imports/tests/docs for the clean PR.
2. **Integration pass:** review the Tracker hook and prepared lifecycle as one
   change; check failure/cleanup, ordinary tracking when mechanics is absent,
   environment copy/serialization and collective fallback. Keep runtime details
   private so production buffers can replace the current writer.
3. **Reproduction/GPU pass:** run the selected portable examples and tests from
   a clean environment. Obtain an actual GPU smoke/performance record before
   claiming GPU support. The UI can be shown in a draft with that validation
   explicitly pending; it must not be presented as completed.

Engineering estimate, not a measured completion time: **one to two focused days**
to extract a coherent experimental UI/demo draft; **roughly two to four focused
days** including integration cleanup and an initial GPU pass if hardware is
available. GPU findings can extend that estimate. Full FCC streaming, packed
bunches, engineering assignments and online correction are larger subsequent
work, not prerequisites for opening the first clearly scoped draft.

Recommendation: stop broad exploration, show the current UI through a focused
draft soon, and keep developing the occurrence/phase and source-run-state layers
on this exploratory branch. No PR is published by this review.
