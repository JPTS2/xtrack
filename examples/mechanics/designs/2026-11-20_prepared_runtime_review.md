# Prepared mechanics, freeze clocks and FCC bunch sampling

This slice retains the existing interface and fourteen teaching scripts. It
implements the first fast native backend, rather than treating the earlier
Python scalar-writing reference as the production architecture. The twofold
slowdown requirement is an acceptance ceiling, not a claim established for
every context, source, bunch representation or tracking-call length.

## What changed

`env.mechanics` still owns the physical hierarchy; `line.mechanics` owns its
survey placement and attachments. Deliberate element offsets, installation and
dynamic motion remain distinct. Existing `new_girder`, `add_source`,
`compute_time_series`, tables, plots, animation, enable/disable and clear remain.

The additions are:

- `new_girders(mapping)` resolves and surveys a batch once. It accepts live
  reference-geometry expressions without detaching knobs; numeric alignment
  expressions and ambiguous shared occurrences remain guarded.
- `support_class` is explicit, inspectable and serialized. The batch mapping
  can supply different classes, parents and anchors per group. This is metadata,
  not yet a native response-class registry or an inferred engineering layout.
- The prepared runtime freezes topology/reference geometry/static placement for
  one tracking call. A packed geometry/alignment guard runs at call entry;
  ordinary strength changes remain live. Explicit inspection synchronizes
  ownership bookkeeping on demand. Clearing releases the prepared buffers.
- Harmonic and plane-wave sources execute on the tracking context. Sampled and
  custom sources supply one packed six-coordinate block; source rotation
  composition, body hierarchy and lowering execute in portable C kernels.
  There is no per-particle source evaluation or implicit GPU run-history cache.
- A conservative fixed-reference, noncollective scheduler publishes the shared
  particle clock at kernel boundaries and tracks complete turns. It removes
  per-turn host clock reads/writes and the separate end-turn tracking launch.
  Ramping, collective/custom elements, logs and unsupported range/monitor modes
  retain the established scheduler. The reference mechanics backend remains
  selectable with `backend='reference'`.
- Source-specific allocation budgets replace the unconditional quadratic
  charge. Correlated dense PSDs retain their quadratic guard. Uncorrelated PSDs
  no longer allocate/multiply an identity covariance.

The [bulk note](2026-11-20_bulk_geometry.md),
[precision note](2026-11-20_mechanics_precision.md) and
[GPU/freeze investigation](2026-11-20_gpu_freeze_runtime.md) give the individual
contracts and adversarial checks. Native code is in `xtrack/mechanics`; the
tracking integration is in `xtrack/tracker.py`.

## Freeze semantics and kernel ordering

The accelerated default is turn-held motion. One physical freeze time prepares
the body/element maps, and all particles in that tracking packet use them for
the turn. Extra updates require actual tracking boundaries; an ordinary marker
cannot safely make a shared update inside the asynchronous GPU particle loop.
This slice does not introduce a new user-inserted freeze-element API. The older
explicit `sampling='passage'` reference calculation remains available, with its
previous cost limitations.

For prescribed fields, once-per-turn *preparation* need not imply one common
physical time for every occurrence. A batched kernel could prepare occurrence
maps at `t_b+s_e/(beta0*c)` before tracking starts. Each ancestor must then be
evaluated at that occurrence's encounter time; reusing one parent pose at a
different time would be wrong. This is a useful A/B alternative to extra
freeze-boundary launches: replace the reference passage loop's repeated whole
hierarchy work with independent occurrence/ancestor traversal. It still treats
each element traversal as frozen and needs accuracy checks. Stateful feedback
cannot generally prepare unknown future actuator values this way.

Each body kernel thread independently traverses its immutable ancestor path.
No thread consumes a parent output that another thread might not have written.
Subsequent lowering and tracking launches provide the required global ordering.
The fixed-reference scheduler writes shared clocks before tracking, never from
one particle thread while other blocks may still be reading the old clock.

The analytic route currently launches three mechanics kernels, one clock
publication kernel and one full-turn tracking kernel per turn. There is a final
clock publication at call end. This removes scalar synchronization from the
loop but does not eliminate launch overhead. Actual CUDA/OpenCL execution is
still required to establish the GPU performance ceiling.

Geometry-changing callbacks within a prepared tracking call violate the freeze
contract. Automatic mode conservatively retains reference scheduling for logs
and time-dependent variables. Explicit compiled mode permits observations with
fixed geometry; it does not make arbitrary mutation safe. Static body settings
are refreshed between calls. Geometry changes require rebuilding the surveyed
bindings; they are not silently absorbed into an old reference.

## Precision and the composer question

The problematic representation added nanometres to world positions of tens of
kilometres and subsequently subtracted those positions. Mechanics now retains
nominal geometry plus local translation `d` and rotation increment `E=R-I`.
It computes `E` directly with stable trigonometry and composes finite transforms
through `E1+E2+E1*E2`. This avoids losing motion before lowering. It also improves
relative table columns, port transport, pickup projection and amplified plots.

The composer's `CompensatedFloat` supports scalar addition/subtraction; it is
not compensated matrix arithmetic. It is useful for the reference clock:
the prepared scheduler retains its high/low elapsed-time sum across calls in
weak per-particle run state, with no change to the public particle layout.
External clock/reference/period edits start a new epoch at the published time.

For the FCC revolution period, ordinary repeated binary64 addition accumulates
about -4.15 ns after a million steps. At 1 kHz that is a -2.61e-5 rad phase
error. This is a scalar arithmetic experiment, not a million-turn tracking run.
The compensated result matches the rounded integer-period product; chunking,
manual time reset and period changes are tested separately.

The former approximately 15 pm horizontal lowering floor is removed. Independent
tests include tiny translations with deliberate millimetre offsets, nanoradian
lever terms, shifted world origins, finite hierarchy rotations and exact/near
Euler singularities. Exact zero motion restores the adopted fields verbatim.
World display matrices and final native scalar fields still have ordinary
binary64 resolution; compensation is not a promise of arbitrary absolute
survey accuracy or infinite precision in a large deliberate offset.

## Measured FCC costs and the twofold ceiling

The [prepared runtime records](../review/results/fcc_freeze_runtime/README.md)
use the pinned development Z input from the earlier
[FCC review](2026-11-20_fcc_scaling_review.md). All live geometry expressions
remain attached. They include original-lattice tracking, complete prepared
tracking calls, cold preparation, isolated updates, memory payloads, source and
kernel hashes, compiler/thread settings and particle survival checks.

The final tracking ratios for 128-turn calls are about 1.17/1.00 for quad sites,
1.81/1.09 for all magnets, and 1.88/1.09 for a two-level hierarchy (one/32
particles). The sampled ground/class record consumes at 1.19/1.02. However,
direct continuous field evaluation is 2.74 times baseline for one particle;
including the short ground record's preparation gives 3.32 times baseline.
Warm short calls also exceed the ceiling. The full twofold acceptance gate has
therefore **not** been met.

The probes distinguish flat provisional supports, a two-level support/magnet
hierarchy and a joint ground-field/response record. These are cost proxies,
not engineering girder assignments. Final measurements are tabulated in the
results README rather than copied into several competing timing tables.

The twofold ceiling must include the chosen call size and workload. Cold
preparation takes seconds and cannot be hidden in a one-turn timing comparison.
Entry validation and ownership checks also make frequent one-turn calls more
expensive than long calls. The short-call measurements explicitly expose that
limit. A passing 128-turn CPU measurement is not a universal performance claim.

The sampled-ground case uses one field at four feet per quad site, 128 random
modes, 128 samples at 2048 Hz, a 2–40 Hz spectrum and three synthetic response
classes. Causal filters start at zero. This is a short cost/configuration probe,
not a converged stationary FCC noise model. Arbitrary direct spatial providers
can still be expensive when called at every freeze; no twofold result is claimed
for those unmeasured paths.

There is also a critical accounting distinction: source/response preparation is
reported separately from tracking. Repeatedly rebuilding short contact-fit
records can exceed the total simulation budget even when record consumption is
fast. Long studies need prepared contact projections, mode-domain stationary
responses or bounded streaming with persistent filter state; storing a whole
run is not an acceptable workaround. End-to-end source preparation plus tracking
must be benchmarked at the intended horizon before calling the FCC study ready.

## Filling patterns and what per-bunch Twiss means

For a bunch delayed by `tau_b` and an element at reference distance `s_e`, the
physical encounter time is approximately

```text
t = t_origin + n*T_rev + tau_b + s_e/(beta0*c).
```

An additional within-bunch `-delta_zeta/(beta0*c)` can be retained when relevant.
The ground realization, phases and seed are common to every bunch. Empty
buckets remove observations; they do not reset motion or filter state.

The current shared scalar alignment backend cannot assign distinct mechanical
maps simultaneously to bunches packed into one particle array. Its semantics
remain one common frozen map per packet. The
[FCC bunch experiment](../review/results/fcc_bunch_sampling/README.md) instead
tracks independent reference bunches sequentially with explicit physical time
origins, preserving the same prescribed source. Its whole-line freeze omits the
additional passage delay; this is stated in the figures and records.

Actual frozen 4D Twiss is recomputed at six bunch phases. At 1 kHz those phases
span about 1.44 rad for the illustrative fill; at 10 Hz about 0.0144 rad.
These snapshots show different closed orbits/optics, but are not the response
of a beam evolving through the time-dependent lattice.
For one selected phase, zero-motion Twiss repeats exactly; halving/doubling
`steps_R_matrix` changes the roughly 1,045 ppm beta-beating signal by less than
0.000581 ppm. W-propagation steps were held fixed. This is a focused numerical
check, not a full optics/ground-model convergence study.

For the linear map `u[n+1]=M*u[n]+b*A*sin(omega*n*T+phase_b)`, the particular
response uses `(exp(i*omega*T)*I-M)^-1`; a static orbit uses `(I-M)^-1`.
The experiment retains the free transient from its common zero-error closed
orbit initial condition and checks tracked histories against that independent
prediction. Its particular response/static-orbit amplitude ratio is about
1.00076 at 10 Hz and 1.59684 at 1 kHz for the selected drive/observation.
Agreement with the tracked driven-plus-free solution is better than 0.03% RMS.

The baseline tunes give first betatron aliases around 552 Hz and 662 Hz. They
lie within the user's sub-kilohertz band: being below the revolution frequency
does not justify a quasistatic beam response. At 1 kHz, the source phase advances
about 109 degrees per FCC revolution. Turn-held, passage-resolved and finer
freeze grids require a convergence comparison in the relevant beam observable.

## Architecture to carry forward

Keep the current model/UI and fast prepared backend as the native reference for
the next stage. The fill requirement strengthens the case for occurrence-aware
tracking buffers rather than making shared element scalar mutation permanent.

1. Compile entry/exit maps per occurrence, preserving deliberate geometry and
   avoiding repeated curved-frame conjugation in each GPU particle thread.
   Compare against girder-boundary maps for contiguous spans, including bend
   curvature, fragmented members, nested residuals, apertures and pickup frames.
2. Add an explicit freeze-grid/packet clock contract. Use bounded exact bunch
   tiles or a converged phase grid; every particle selects the appropriate
   prepared occurrence map. Do not allocate bunch count times the whole ring
   without an explicit memory budget. Shared bodies keep one physical source.
3. Prepare source/response graphs independently from run state. Precompute
   fixed-foot small-signal projections where valid, retain finite reconstruction
   as a reference, and apply class `H(f)` to common latent modes for stationary
   prescribed LTI motion. Use bounded causal streaming for measured/transient
   sources. Preserve correlation across classes and source-selection invariance.
4. Gate on actual GPU measurements: identical lattice, bunch/freeze choices,
   particle counts, source accuracy and observables, including source preparation
   and memory. Offline GPU compilation and CPU timings do not establish that
   gate. Moving BPM readouts/feedback and both-beam event ordering remain
   separate acceptance cases.

The unresolved engineering girder rules are still needed. The catalogue's
provisional associations remain useful review data, not physical support truth.
