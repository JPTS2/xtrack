# Bulk geometry and protected placement during prepared tracking

The FCC use case now has a public bulk setup path on the original parameterized
lattice. `line.mechanics.new_girders(assignments)` surveys once, resolves aliases
and occurrence order once, validates all definitions before installation, and
captures the full-line reference signature once. `env.mechanics.new_girders(...,
line=line)` provides the same environment-owned definitions with explicit line
binding. Existing `new_girder` calls retain their behavior.

```python
supports = line.mechanics.new_girders({
    'g.arc.001': {'elements': ['q1', 's1'], 'support_class': 'arc_quad'},
    'g.arc.002': {'elements': ['q2'], 'support_class': 'arc_quad'},
})
arc_supports = env.mechanics.get_table().rows.match(support_class='arc_quad')
arc_elements = line.mechanics.get_table().rows.match(support_class='arc_quad')
```

`support_class` is independent of `kind='girder'`, survives environment copying
and serialization, and appears in both configuration and posed tables. It is
assignment metadata, not an implicit transfer function. The input mapping may
also specify anchor, an existing parent and support contacts. Defining parent
levels before child levels retains an explicit mechanical hierarchy. Group
members need not be contiguous; this is still rigid placement, not girder
bending. Group centres use the first entrance and last exit in line order.

Reference geometry expressions are accepted because mechanics only reads their
evaluated values; it never writes length, angle, curvature or design roll.
Deliberate alignment expressions remain rejected by this numeric alignment
backend because writing effective alignment would otherwise take ownership of
those expressions. Strength expressions remain live. Changing evaluated
reference geometry invalidates captured survey bindings, including changes to
unmounted drifts upstream of a mounted magnet. `line.mechanics.validate()` makes
this explicit. It does not silently reinterpret old support anchors.

The environment loader also restores all saved attachments with one candidate
capture and one bookkeeping pass, avoiding a repeated survey per saved member.

## Measured public FCC setup

[`fcc_bulk_setup.py`](../review/fcc_bulk_setup.py) loads the selected development
Z positron JSON unchanged and creates one provisional support at each of 2,663
quadrupoles. This is a scaling proxy, not the missing engineering girder map.
The recorded result is in
[`fcc_bulk_setup/result.json`](../review/results/fcc_bulk_setup/result.json).

On serial macOS/ARM CPU, public bulk definition took **4.72 s**; explicit full
validation took **0.201 s**, separately measured. Input loading was 2.31 s,
table inspection 0.120 s and deliberate-reference adoption 0.113 s. All 2,544
expression-backed quadrupole lengths retained their original expression object;
length and strength values were unchanged. No detached numeric snapshot or
private attachment bookkeeping was used. This resolves the previous seventh-
quadrupole expression guard and replaces the repeated per-support survey path
with a bounded full-line preparation operation. It makes no tracking-speed or
GPU-latency claim.

## Prepared geometry guard

Per-turn full-line validation is incompatible with the performance target. The
private `GeometryGuard` validates prepared tracking entry using one context
kernel and one contiguous context-to-host transfer:

1. At preparation, trace the scalar storage read by the public reference
   geometry getters. This records property dependencies such as private Bend
   length/angle storage and conditional selector fields, rather than guessing
   raw names from a public property name.
2. Cache those scalar byte offsets, their expected bit patterns and the seven
   effective-alignment offsets per attachment. No whole-run history is stored.
3. At tracking entry, check line sequence, element identity, alias ownership,
   aperture coupling and alignment-expression ownership using Python metadata.
   Gather protected scalar storage on the active CPU/CUDA/OpenCL context and
   compare its one packed output against the captured geometry and the last
   mechanics-written alignments.

Geometry dependencies are compared conservatively at storage level. A changed
raw dependency invalidates the prepared state even when derived values happen
to cancel. Ordinary strengths do not invalidate it unless they also affect
reference geometry through lattice expressions. A newly added alignment
expression is rejected even if its present numeric value equals the old field.

The guard does not inspect geometry at every turn or claim that arbitrary
concurrent mutation is safe. Geometry remains fixed within one tracking call;
geometry-changing callbacks require a different scheduler/explicit rebuild.
A strength-only callback need not rebuild geometry. Entry checks have a real
cost and must be reported separately from repeated turn costs, especially for
one-turn-at-a-time Python workloads. Initial preparation still performs full
validation and scalar tracing once; this is not hidden per-turn GPU work.

The unit tests parameterize all installed Xobjects test contexts and explicitly
forbid any scalar `buffer.to_bytearray` read during a prepared guard check.
Only CPU serial/OpenMP contexts are available locally; the existence of common
context kernels is not a measured GPU validation result.

The separate
[`fcc_geometry_guard.py`](../review/fcc_geometry_guard.py) measurement on the same
unmodified FCC model found **172 ms** median full validation versus **11.29 ms**
median prepared packed entry validation (seven repetitions). It gathered 55,094
geometry words plus 18,641 alignment words: **589,880 bytes per entry**, in one
context-to-host array, with zero scalar buffer reads. Initial guard preparation
was 1.45 s including compilation and scalar-storage tracing. These are serial
CPU results. The packed check still costs several one-particle FCC turns, so
short-call overhead must remain visible; long tracking calls amortize it.

The prepared guard now fails closed for an array-based geometry getter instead
of silently missing its dependencies; `backend='auto'` can retain the reference
implementation for such geometry. Custom Python and collective devices receive
an explicit entry check of their geometry, including plain mutable Python
lengths. This small custom subset can incur additional scalar reads; the
zero-scalar-read claim applies to the built-in non-collective FCC path.
Geometry tracing patches only the participating buffer instances and restores
their methods even when preparation rejects an unsupported getter.

Explicit table inspection or application during a compiled tracking callback
first synchronizes the packed last-written output on demand. That makes the
normal alignment conflict checks distinguish kernel writes from external
edits without introducing a synchronization every turn. Clearing bindings
releases the prepared runtime and its context arrays.
