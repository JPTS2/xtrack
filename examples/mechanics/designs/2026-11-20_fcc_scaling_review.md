# FCC mechanics: adversarial readiness and scaling review

The current public workflow is **not ready for a full FCC run with live lattice
knobs, global ground motion and class-dependent support responses**. The rigid
hierarchy and inspection UI are useful foundations. The execution and preparation
paths need changes now, before adding more demonstrations or relying on long
tracking jobs. This review changes no native mechanics implementation.

## Input and scope

The primary input is the user-selected
[FCC development branch](https://gitlab.cern.ch/acc-models/fcc/acc-models-fcc-ee/-/tree/dev/108.1.0),
pinned to `8805a5bb8bd568f5de2e1558037371905ecfc985`. The selected file is
[`lattices/z/fccee_z_beam_1_positrons.json`](https://gitlab.cern.ch/acc-models/fcc/acc-models-fcc-ee/-/blob/8805a5bb8bd568f5de2e1558037371905ecfc985/lattices/z/fccee_z_beam_1_positrons.json).
The branch name is `dev/108.1.0`; its own `VERSION` reports `dev/LCC_108.3.0`.
Both identifiers are retained rather than assigning a release label by inference.

This is a thick 45.6 GeV positron model with circumference 90,640.199866 m and
reference revolution time 302.343163 µs. It contains 32,204 occurrences:
2,663 quadrupoles, 7,218 rectangular bends, 2,112 sextupoles, 48 octupoles,
5,414 multipoles/correctors, 2,676 markers, 12,072 drifts and one cavity.
The optics already contains subdivided dipoles, mostly about 9.375 m, with the
longest about 13.303 m. Conclusions from the old bundled FCC lattice's different
magnet representation must not be substituted for this model.

All measurements used serial `ContextCpu`, Xtrack 0.119.2, Xobjects 0.7.3,
Python 3.14 and float64 on macOS/ARM. BLAS/OpenMP thread settings were fixed to
one. Neither CuPy nor PyOpenCL is installed; **there is no measured GPU result**.
The input has radiation disabled. These short timing workloads establish neither
emittance/luminosity performance nor a validated noise model.

Reproducible material:

- [Scaling probe](../review/fcc_scaling.py): fresh-process `native`, `bulk`, `field` phases.
- [Inventory and candidate-association generator](../review/fcc_inventory.py).
- [Measured costs figure](../review/results/fcc_scaling/costs.svg).
- [Native public path](../review/results/fcc_scaling/native.json),
  [numeric/bulk experiment](../review/results/fcc_scaling/bulk.json),
  [field/response probe](../review/results/fcc_scaling/field.json),
  [inventory](../review/results/fcc_scaling/inventory.json).

The probes and JSON record the source hashes, lattice hash, model commit,
individual timings and peak process RSS. A phase marked `passed` means its
measurement procedure completed; the public capability failures remain explicit
fields in its result.

## What happens with the unmodified model

Loading took 2.20 s, explicit tracker preparation 0.64 s. A girder can be attached
to the first quadrupole. The next five also attach. The seventh attempted
quadrupole stops with:

```text
qf2r.0.length: expression-driven geometry/alignment is not supported by numeric lowering
```

This is systemic, not an isolated awkward magnet:

| Element class | Occurrences | Protected expression-driven geometry |
| --- | ---: | --- |
| Quadrupole | 2,663 | 2,544 lengths |
| RBend | 7,218 | All 7,218 angles |
| Sextupole | 2,112 | All 2,112 lengths |

Those are legitimate lattice definitions. Removing their expressions is not an
acceptable production workaround. Mechanics should read their evaluated geometry,
compile its reference transforms and invalidate/rebuild that compilation when
geometry changes. Strength knobs must remain available.

The current `_candidates`, `_attach` and `_check` paths repeatedly survey or
inspect the entire line. The first support cost 3.73 s; subsequent supports
about 1.2 s each; six took 9.60 s. Continuing at that rate would take tens of
minutes for thousands of supports, even if the expression guard were removed.
This is an extrapolation, not a completed full-machine public setup.

## Girder assignments and response classes

The supplied [magnet catalogue](https://github.com/skostogl/fcc-ee-magnet-catalogue)
is useful for names, device associations and family/circuit information. Its
README identifies LCC_107.0.0, so its spreadsheets are not automatically an
engineering map for the selected development input.

The [generator](https://github.com/skostogl/fcc-ee-magnet-catalogue/blob/main/000_create_catalogues.py#L128)
uses a 3 m nearest-quadrupole rule for nearby sextupole cell attribution. BPMs
and correctors use the quadrupole encoded in their names, including explicitly
name-linked long-girder cases. A catalogue cell number is not by itself evidence
that every element in that cell belongs to one rigid support.

Applying those attribution rules to the selected model produced the explicitly
provisional [candidate associations](../review/results/fcc_scaling/candidate_associations.csv):
2,663 quadrupole anchors, 2,112 nearby sextupoles, 7,957 name-associated devices,
and 7,266 unassigned bend/octupole rows. Engineering-girder and support-class
columns are deliberately blank. The distance is periodic longitudinal distance
between element entrances. These rows help review assignments; they are not
accepted support definitions. The actual girder rules remain pending.

The desired UI should add one inspectable bulk assignment table containing
physical support identity, member occurrences, support class, anchor and feet.
`kind='girder'` should remain distinct from a class such as `arc_quad_support`.
Class selection should work through normal table operations. Keep
`new_girder` for small explicit examples and add a bulk preparation operation.

There is currently no native response-class registry or compiled floor-to-body
pipeline. The pieces exist: sample one joint ground field, fit support contacts,
select class channels, apply their responses and bind the resolved motion.
This is more manual than the FCC workflow should be. Do not draw an independent
ground realization for each class: that destroys cross-class coherence.

A full correction/feedback study also needs moving electrical-frame readouts in
the tracking-monitor path. The existing explicit `read_bpm` geometry helper and
frozen-Twiss examples are useful references, but the timing probes neither move
all physical FCC pickups nor implement an online feedback controller.

## Measured costs beyond the public guard

To separate import restrictions from runtime scaling, the `bulk` phase makes a
**detached numeric snapshot** of current element values, with no live knob links.
It then captures native candidates once and uses private bookkeeping to install
one provisional support per quadrupole. This is an isolated experiment, not a
supported API or an engineering grouping.

Capturing candidates once and installing all 2,663 supports took **4.61 s**.
The circumference is unchanged; quadrupole lengths/strengths are checked against
the original values. Short frozen/dynamic tracking tests retain active particles.
The maximum entrance-position discrepancy against native applied survey is
9.1 pm. That agreement is not an independent small-displacement precision test.

| Work, with 2,663 provisional supports | Measured CPU cost |
| --- | ---: |
| Compose all body poses | 38.57 ms/update |
| Validate geometry/attachments | 159.78 ms/update |
| Lower transforms, excluding validation | 44.06 ms/update |
| Write alignment scalars | 14.59 ms/update |
| Complete mechanics application | 268.76 ms/update |
| Frozen tracking, one particle | 2.87 ms/turn |
| Dynamic tracking, one particle | 277.33 ms/turn |
| Frozen tracking, 32 particles | 62.03 ms/turn |
| Dynamic tracking, 32 particles | 342.55 ms/turn |
| One selected element, 33 history samples | 1.57 s |

Update-component medians are independently measured; they should not be summed
as an exact decomposition. Tracking medians use two four-turn repetitions,
excluding launch allocation and kernel warmup. Frozen and moving fields have
different physical semantics, but the particle workload is the same.

Sustaining the measured one-particle dynamic rate for a million turns would
take about 3.2 CPU days to represent about 302 s of physical time. That is an
extrapolation from the short probe, not a measured long tracking job.

Even with only six supported quadrupoles, a complete update takes about 102 ms:
the current full-line validation dominates. Removing that scan is necessary but
insufficient: pose composition, lowering and scalar writes still cost about
100 ms combined at 2,663 supports.

Pointwise passage preparation repeats the entire body calculation for each
element. Eight pointwise queries cost 0.332 s; scaling that loop to 2,663 gives
about **110 s per prepared turn**. This is explicitly an extrapolation; a full
passage call was not run. It needs batched pointwise evaluation, not this loop.

The selected-history result is also a UI problem. Selection currently occurs
after all body poses have been computed at every sample. One magnet's history
must evaluate its ancestor closure and required source channels, rather than
the entire machine. A normal table interface alone does not guarantee an
interactive query.

## Global ground motion and class transfer functions

The field probe uses 2,663 provisional quad-site supports, four feet per support,
128 samples at 128 Hz and 256 seeded random modes. It samples one joint field,
fits all six rigid coordinates, then filters three synthetic response classes
(888/888/887 supports; 8/12/18 Hz oscillators). These classes are deliberately
not claimed as real FCC support categories.

| Preparation operation | Cost |
| --- | ---: |
| Direct random-mode field evaluation at 10,652 feet | 63.03 ms |
| Rigid contact reconstruction | 507.52 ms |
| All three class responses | 11.23 ms |

Class-filter whole-record and consecutive-block outputs agree exactly. The
response law is causal ZOH with zero initial filter state. This demonstrates
bounded filter-state continuation, not stationary filter initialization.

The public sampler rejects that same finite-mode field with an estimated
workspace of 9,340,040,448 bytes. It unconditionally charges a dense `80*N²`
term, although this source's own workspace estimate is 110,125,056 bytes.
Removing the false quadratic term would still leave a sampling estimate of
about 251 MiB, so this workload legitimately needs a larger explicit budget
than the default 128 MiB. Contact fitting has separate transient allocations.
`compute_time_series` similarly rejects a two-sample harmonic record on 2,663
bodies as 568 MB, although its six-coordinate sample payload is only 0.244 MiB.
The generic allocation guards must become source-specific. Dense PSD sampling
really can need quadratic storage and cubic factorization; the fix must retain
honest limits for that path.

The field, fitted motion and class outputs retain about 62.7 MiB of payload.
The field process peaked near 1,007 MiB including the Python/Xsuite model,
copies and fitting workspace. The native and bulk probe peaks were about 882
and 727 MiB; the bulk process also holds the original and numeric models.
These process peaks are not device-memory measurements. The active lattice
Xobjects buffer itself has capacity about 17.25 MiB.

This is a cost configuration, not a converged FCC ground model. At 128 Hz,
linear replay of a 40 Hz sinusoid can lose about 44% amplitude at a half-step;
Nyquist compliance alone is inadequate. The 256-mode spatial approximation also
requires convergence in relative-motion spectra and beam observables. The
favourable source/filter costs must not be presented as validated accuracy.

## GPU and numerical readiness

One complete 2,663-support application made **131,867 scalar buffer reads**
(1,054,936 bytes) and **18,641 scalar writes** (149,128 bytes), measured by
temporarily instrumenting the CPU buffer methods. With six supports, there were
still 81,384 scalar reads because the lattice was scanned.

Xobjects scalar access calls `to_bytearray` and `update_from_buffer`. In the
installed CuPy buffer these perform device `.get()` and host-to-device array
updates. Thus the current host path has a structurally hostile transfer pattern;
these CPU operation counts are not claimed as measured CUDA latency. CUDA's
[data-transfer guidance](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#data-transfer-between-host-and-device)
specifically supports retaining intermediate data on device and batching small
transfers. A GPU context switch will not fix this implementation.

The current motion hook also puts tracking through a Python turn loop. An
ordinary GPU particle thread cannot safely update shared mechanics state once
per turn and act as a grid-wide barrier. Scheduling must use supported kernel
boundaries, immutable time-addressable data, or bounded prepared chunks. This
must fit Xsuite's [collective/non-collective scheduling model](https://xsuite.readthedocs.io/en/latest/collective.html#how-does-it-work).

There is a second reason to change the representation: full-world matrix
round trips lose small local displacement precision on the FCC survey:

| Commanded local horizontal shift | Maximum error across 2,663 quads |
| --- | ---: |
| 100 nm | 15.31 pm |
| 1 nm | 15.00 pm |
| 100 pm | 16.42 pm |
| 10 pm | 15.47 pm; some realised values have the wrong sign |

These compare actual effective `shift_x` increments against the known command,
not one world-pose calculation against another. Vertical behaviour may differ
because global coordinate magnitudes differ. Keep dynamic increments and local
mount transforms near their reference frames, then compose for display/export.
Do not attempt to fix this by subtracting an empirical Euler-angle correction.

At this circumference, `2*pi*f*T_rev` is 0.019 rad at 10 Hz, 0.076 at 40 Hz and
0.190 at 100 Hz. Turn-held mechanics and finite interpolation need an explicit
accuracy budget. Installation, drift, vibration and feedback sample clocks need
not all have the same cadence.

## What to retain and what to change

Retain environment-owned physical bodies, line-owned occurrence views, explicit
reference adoption, the three placement layers, source definitions, response
ports and normal table selection. The prototype has earned those choices. Do
not make its Python scalar-writing implementation the production contract.

| Candidate | Assessment after implementation |
| --- | --- |
| Existing per-element numeric writes | Keep as a small-system correctness reference; unsuitable as the FCC GPU runtime. |
| Compiled per-occurrence pose buffers | Recommended initial fast backend. Preserve original fields/knobs, compose once per body/sample, publish bounded arrays and support fragmented groups/slices/aliases through occurrence indices. Requires native tracking integration. |
| Girder/subline boundary transforms | Serious A/B alternative for contiguous spans. Can reduce per-particle transforms, but internal BPMs, apertures, RF, nested magnet motion and fragmented membership must retain their frames. Whole-ring nominal curvature and all relevant particle coordinates need independent checks. |
| Entire-run pose cache | Reject as the default. One million samples of six doubles on 2,663 supports is 119 GiB; 4x4 matrices are 317 GiB, before magnets/BPMs. |
| Bounded pose chunks | Viable scheduling option, not an unlimited cache. A 64-sample block is 7.8 MiB for six coordinates or 20.8 MiB for matrices on this support set. Needs explicit budgets and chunk/restart invariance. |
| Stationary mode-domain response propagation | Particularly attractive for the first FCC use case: apply each class's complex `H(f_mode)` to common latent ground modes, then query continuously. Avoids full time histories and replay interpolation for stationary prescribed LTI motion. Spatial approximation and memory still need budgets/convergence; it does not supply causal startup or feedback by itself. |
| General causal streaming | Needed for measured traces, local source transients and evolving runs. Retain filter state and bounded time blocks rather than full-ring histories. |

Precomputing the small-signal rigid-contact projection for fixed feet is also
worth testing: nanometre dynamic displacements do not generally require a fresh
3x3 SVD per body per sample. Keep finite Kabsch reconstruction as the reference
and retain exact finite static/pose composition. Any linearized dynamic fit needs
an independently checked regime and contact residuals.

The source backend must remain selectable by physical assumptions. A separable
exponential correlation model cannot describe arbitrary frequency-dependent
complex cross spectra. An open-line longitudinal recursion is not automatically
a closed-ring Euclidean field. No universal small-rank fast representation is
justified for arbitrary measured site motion.

Static caching also needs a budget. Precomputing two float64 spatial quadratures
at 10,652 feet and 2,048 modes already costs about 333 MiB for one scalar field,
before vector components, regions or local sources. Do not turn the correct
rejection of an entire-run cache into an unbounded spatial-mode cache. Tile or
prepare only the required receiver projections where needed.

Keep source/response definitions and provenance separate from their sampled
records and mutable filter/run state. Today an offline chain is eventually
registered as a resolved `RigidMotion`, so the source inventory alone cannot
recover the floor-to-girder-to-magnet response graph. A class-aware runtime needs
that graph explicitly, without creating per-scalar xdeps expressions. Separate
topology, reference geometry, installation and source-schema invalidation so
ordinary strength/corrector changes do not rebuild the world geometry.

## Recommended next slices

1. **Real FCC setup and assignment UI.** Bulk-resolve occurrences and survey
   once, preserve live lattice expressions, add explicit support-class metadata,
   and import/review actual engineering assignments. Fix source-specific budget
   accounting. A class-aware FCC example should share one ground realization
   and expose its resolved membership/response tables.
2. **Compiled local runtime and selective diagnostics.** Cache only static
   geometry and binding indices with explicit invalidation. Preserve deliberate
   element fields; store small local dynamic increments. Remove per-turn
   full-line scans and scalar device access. Query only selected ancestors and
   channels for a single-element history. Compare a per-occurrence buffer and
   a boundary backend on the same inputs.
3. **Bounded sources, responses and scheduling.** Separate immutable definitions,
   prepared geometry and mutable run/filter state. Use mode-domain steady-state
   propagation where applicable and causal streaming elsewhere. Benchmark actual
   CUDA hardware with the same recorded inputs before selecting a default.

Acceptance should include the unmodified parameterized FCC input, exact
zero-motion preservation of deliberate fields, analytic pm/nrad increment tests,
invariance under a large change of world-coordinate origin,
class gain/phase and cross-class coherence, chunk/state continuity, moved BPM
and aperture frames, and CPU/GPU coordinate comparisons. Both-beam common
mechanics and online feedback need consistent event ordering; the present
one-beam timing tests do not establish those capabilities.

The next work should be these scaling/correctness slices, not another visual
demo. The measured source and class-response costs are encouraging; the current
validation, world-coordinate lowering and host update architecture are the
parts that must change early.
