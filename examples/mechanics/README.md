# Mechanics exploration

Active work lives directly in Xtrack on local branch `feature/mechanics-manager`.
This branch is for exploration and is expected to evolve. There is no PR or
publish step now. Once the design is useful, changes can be reviewed in slices
onto a fresh branch before the user opens a draft PR.

Only three examples are the main review surface:

1. [Rigid girder](001_rigid_girder.py): `env.mechanics`, `line.mechanics`,
   `new_girder`, a surveyed anchor, existing local errors and actual tracking.
2. [Sources and vibration lifecycle](002_sources_and_vibrations.py):
   `add_source`, interchangeable harmonic/measured/PSD models,
   `compute_time_series`, `enable_vibrations()` and `disable_vibrations()`.
3. [Shared girder](003_shared_girder.py): one physical body attached to two
   lines with different survey placements and opposite local directions.

Run with the checkout first on `PYTHONPATH`; `%run` leaves the objects available:

```bash
PYTHONPATH=. python examples/mechanics/001_rigid_girder.py
PYTHONPATH=. python examples/mechanics/002_sources_and_vibrations.py
PYTHONPATH=. python examples/mechanics/003_shared_girder.py
```

The former `install(env, line)` only attached the scratch prototype to existing
objects. It is gone from the native interface. The 19 scratch examples remain
reference material outside this checkout, rather than a review assignment.

## What lives where

```python
mech = env.mechanics
girder = line.mechanics.new_girder('g1', elements=['q1', 'q2'])
assert line.mechanics.manager is mech
```

The environment owns physical bodies and named sources. A line owns attachments
and its placement in that environment's world. Its accessor is a bound view,
not a second manager. `mech.new_girder(..., line=line)` is the equivalent form.
With several lines, the environment form requires `line=`. There is no need for
`env.mechanics.MechanicsManager(line=...)`; the instance already exists.

For a shared body, create it once and call `other_line.mechanics.attach(girder,
elements=[...])`. Place the second survey explicitly using
`set_reference_frame(xt.Frame(...))` before attachment. This avoids assuming
independent line origins coincide. The frames are ordinary Xtrack survey frames.
The reference-frame convention is local-to-world; translations are metres and
angles radians. Default centre is the chord midpoint of the selected entrance/
exit span, with axes from its first nominal entrance.

Opposite traversal changes the line-local representation of the same physical
motion. It does not create another physical girder. A source is evaluated at
nominal world anchors; sensor/foot sampling and body transfer paths can extend
that binding without changing ownership.

## Source models and verbs

```python
mech.add_source('floor', xt.mechanics.Harmonic(
    amplitude=50e-9, frequency_hz=5), targets=['g1'], component='y')
motion = mech.compute_time_series(time_s=sample_times_s)
line.mechanics.enable_vibrations(time_series=motion)
line.track(particles, num_turns=100)
line.mechanics.disable_vibrations()
```

`add_source` is the shared verb. Explicit source objects retain B's useful part
without replacing A's named factories. Harmonic amplitude is peak displacement;
PSD is one-sided displacement density in m²/Hz; measured inputs are metres.
`PlaneWave` has independent world direction, polarization and propagation speed.

`compute_time_series` creates reusable **input motion** at explicit seconds,
with an allocation budget. `motion.get_table(body)` reads direct local source
translation; it excludes static alignment and inherited parent motion.
`get_body_poses(time_s=..., time_series=...)` composes world poses without writes.
`apply(time_s=..., time_series=...)` applies one physical time to all bound lines.
These distinct operations replace the scratch `realize`/`snapshot` vocabulary.

The lifecycle follows Xcoll's explicit enable/disable style. Disable removes the
vibration contribution and preserves static girder alignment and local magnet
errors. Re-enable uses the particles' current physical time; it does not reset
source phases or regenerate a PSD record. For a frozen Twiss calculation,
disable automatic vibrations, explicitly apply the requested time, then Twiss.

## Deliberate alignment, vibration and removal

There are three distinct inputs:

- Original magnet-local alignment and deliberate `rot_s_rad` design rolls.
- Deliberate girder settings such as `girder.shift_y` and `girder.rot_s_rad`.
- Source-induced motion. No source writes back into those girder settings.

`line.mechanics.get_alignment_table()` shows captured `local_*` fields beside
current `effective_*` fields, with `design_rot_s_rad` separate. The reference
lowering temporarily writes effective fields into the elements for tracking;
the intended fast backend should keep those transforms separate so user element
definitions remain authoritative. Finite rotations are composed as matrices;
subtracting Euler angles is not a valid general contribution decomposition.

`disable_vibrations()` removes source motion but retains static girder alignment.
`mech.restore()` removes applied girder/motion effects while retaining the model.
`mech.clear()` restores captured original local alignment and removes all sources,
bodies, bindings and automatic updates. It never overwrites deliberate design
rolls. `line.mechanics.clear()` restores/detaches only that line, retaining shared
definitions for other lines. An applied source record is invalid after clearing.
Conflicting raw edits to effective alignment fields are rejected rather than
erased; clear mechanics before redefining those inputs in this reference backend.

Several named scenarios remain desirable. Initially one active configuration is
sufficient: store its parameters in the environment and use `env.call(...)` to
run a machine-specific builder that first clears and then rebuilds mechanics.
That does not require a new Environment or another manager for every line.

## Current execution and boundaries

The current tracking backend is a **host reference update once per turn** using
`particles.time_s`, before tracking that turn. It writes numeric effective
element alignments. It does not build xdeps chains or evaluate sources per
particle. Ordinary static tracking can use CPU/CUDA contexts; automatic updates
on GPU would still incur host scheduling and field-write costs. This is not the
final fast GPU implementation.

Built-in sources can be replaced by a user model with
pure `evaluate(time_s=..., positions_m=...)`. Scalar local sources return
`(times, targets)` displacements; a `frame='world'` model returns
`(times, targets, 3)` XYZ displacements. This leaves room for local pump fields
and regional models without another top-level verb. Custom source serialization
is not yet defined. Different source names do not decorrelate equal seeds;
use distinct seeds for independent PSD inputs or one jointly sampled source
for an explicitly correlated field.
With fixed record length/step/receivers, a PSD's Fourier phases are anchored at
physical time zero; changing the requested start time does not restart the
waveform. Changing record length or step changes that finite spectral model.

Explicit multi-line `apply(time_s=...)` works. Automatic multi-line advancement
is guarded until a common event/clock policy is chosen; an environment namespace
does not make different beam turn counters interchangeable. Automatic SyncTime,
partial turns, pipeline holds and vibration-enabled Twiss are also guarded.

Numeric lowering currently accepts unsliced alignment-capable elements and
preserves their captured local numeric errors. It handles a bent nominal path
using surveyed entry/exit frames and keeps `rot_s_rad` design geometry separate.
Shared/repeated underlying element objects, slices, expression-driven alignment
and post-binding geometry edits are rejected before writes. Boundary transforms
and occurrence-indexed effective buffers remain backend alternatives.

Aperture geometry uses the same survey machinery, but pipe-to-body coupling is
not implemented. Applying mechanics with a built aperture model raises rather
than silently using its stale survey. Full environment dict/JSON roundtrips and
`env.copy()` preserve the configuration and numeric baselines. Standalone line
copy/export and pickle are guarded while their ownership semantics are explored.
Copied/loaded environments have automatic vibrations disabled; bind a motion
record explicitly before enabling a new tracking run.

[DESIGN.md](DESIGN.md) records Xsuite precedents, current decisions and the next
experiments. These are exploration boundaries, not permanent API decisions.
