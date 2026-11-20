<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# Experimental mechanics contract

This draft implements prescribed rigid motion of existing thick elements. The
first review is about the UI, placement ownership and inspectable examples.
The runtime is private and can change without changing the model-building UI.

## Ownership and placement

`env.mechanics` owns physical bodies and named source bindings.
`line.mechanics` binds element occurrences to those bodies and places that line
in the world. Creating or querying another line does not attach it implicitly.
A shared girder is one body; different beams need distinct element objects and
explicit survey placements. The environment supplies no implicit common clock.

```python
mech = env.mechanics
line.mechanics.adopt_reference()
girder = line.mechanics.new_girder('g1', elements=['q1', 'q2'])
```

Reference adoption captures the existing numeric alignment inputs without
rewriting them. Binding also adopts current inputs if explicit adoption is
omitted. An intentional `shift_x` or design roll remains an authoritative input;
the user does not retype it into mechanics.

There are three placement layers:

| Layer | Input | Lifetime |
| --- | --- | --- |
| Deliberate reference | Existing element offsets/rolls and body reference settings | Independent of installation and dynamic seeds |
| Installation | `body.installation` or immutable generated installation case | Static across dynamic times and seeds |
| Dynamic | Named analytic, measured or sampled motion | Vibration and prescribed slow drift at physical time |

`generate_installation_errors` supports stable body/group identities, Gaussian
or uniform draws and optional Gaussian truncation. Selection/reordering does not
redraw another body's error. Root bodies are selected by default so a hierarchy
does not silently receive an error at every level. A stationary PSD generally
has a nonzero initial displacement; installation is not defined by subtracting
the first dynamic sample.

Use `new_body` for magnets, pickups and equipment. A child has a parent-local
`mount=xt.Frame(...)` or a nominal world `anchor`, with the usual rigid lever
arm. `new_girders(assignments)` resolves a whole batch with one survey;
`new_bodies(assignments, kind='magnet')` attaches a magnet/pickup batch without
repeating that full-line setup for each body. Parents must already exist. A
`support_class` is selection metadata, not a response inferred from its name.
Topology and nominal anchors are fixed after creation in this draft.

## Coordinates and composition

Frames map local `(x,y,s)` into world `(X,Y,Z)`. Translations are metres; angles
are radians. Survey pitch/yaw/roll use `Ry(yaw) Rx(-pitch) Rz(roll)`.
In particular, positive survey pitch is a negative right-handed x rotation.
`RigidMotion` columns are `[x,y,s,pitch,yaw,roll]` in nominal body axes, not
rotation-vector components.

The static body map is `T_static = T_deliberate @ T_installation`. Its dynamic
input acts about that statically placed anchor: `d_local = d_static + u_dynamic`
and `R_local = R_dynamic @ R_static`. The dynamic rotation does not rotate the
static translation. Parent/child maps then compose as rigid matrices. Relative
source translations add in their stated nominal axes; their rotations
left-compose in binding insertion order. A jointly specified orientation is
best supplied as one six-coordinate record. Angular records interpolate their
specified survey coordinates, so they require adequate temporal resolution.

`reference='absolute'` replaces inherited dynamic motion about the body's
static placement. Static installation remains. An absolute-driven body accepts
one explicitly combined input; this avoids counting a parent response twice.
Relative bindings cannot accidentally target both an ancestor and its child.
World-vector shortcuts drive roots; child response paths use explicit port
transport or absolute records.

## Sources, floor fields and responses

```python
mech.add_source('floor_y', xt.mechanics.Harmonic(
    amplitude=50e-9, frequency_hz=5.), targets=['g1'], component='y')
motion = mech.compute_time_series(time_s=times)
sources = mech.get_source_table()
```

`add_source` registers a motion binding independently of body construction.
Scalar sources specify `x/y/s` in metres or `pitch/yaw/roll` in radians.
`RigidMotion` supplies all six coordinates. Source tables inspect bindings
without evaluating a source. Intermediate floor/response objects used to build a
resolved record remain explicit Python objects rather than additional drives.

`DisplacementPSD` is a finite periodic Fourier reference with one-sided density
in m²/Hz, or rad²/Hz for `AngularPSD`. Dense spatial correlation is a small-model
reference. `SpectralGroundMotion` supplies continuous random modes with a stable
seed across receiver subsets and time chunks. Mode-count convergence remains a
physical accuracy requirement. Regional/vector fields and located sources
describe supplied amplitude, correlation and propagation models; a source
position alone does not define a mechanical force law.

Sample one common floor field at support feet before applying conditional
girder/magnet/pickup responses. `fit_rigid_bodies` returns a rigid fit and its
contact mismatch; it does not bend the support. `RigidMotion.transport` changes
ports and axes, including the lever arm. A complex response uses `exp(+iωt)`;
6×6 cross terms can have mixed units, e.g. m/rad. `through` is a periodic offline
frequency-domain operation. `CausalResponse` has explicit at-rest initialization
and bounded filter state carried across consecutive blocks. Neither implies an
online force/beam/controller coupling.

Records are immutable host arrays with explicit time bounds. Manager-computed
records and history queries expose allocation budgets; caller-supplied
`RigidMotion` arrays remain the caller's allocation responsibility.
Queries outside their range fail; there is no implicit looping/extrapolation.
Manager-computed records are tied to the source/topology revision. They remain
readable after cleanup, but cannot silently drive a different model. Static
placement changes can reuse a record. Replacing a source currently means
rebuilding the scenario; a named scenario registry is outside this draft.

## Inspection and application

Tables and plots do not modify elements:

```python
bodies = mech.get_table(time_s=.1, time_series=motion)
girders = bodies.rows.match(kind='girder')
elements = line.mechanics.get_table(time_s=.1, time_series=motion)
quad_row = elements.rows.match(element_type='Quadrupole').rows['q1']
quad = quad_row.get_element()
history = quad_row.get_time_series()
history.plot()
```

No-argument tables inspect configuration without sampling sources. Supplying a
physical time requests composed reference/static/total poses. Attached element
rows remain separate when they share a girder. Histories include inheritance
and lever arms, use current static settings and have explicit memory bounds.
For analytic sources without a bound record, pass `time_s=times` to the history
query. Numeric exports omit live element/query context. Stale live table
contexts reject source/topology changes.

`mech.plot` and `animate` magnify translations and rotations separately. These
gains affect presentation only. Use `baseline='static'` to magnify vibration
about the installed pose. After explicit `apply`,
`line.survey(include_element_frames=True)` exposes the applied element frames;
the ordinary survey trajectory remains the nominal path.

| Operation | Effect on the line | Definitions retained? |
| --- | --- | --- |
| `mech.apply(time_s=t)` | Apply one prescribed state to attached lines | Yes |
| `line.mechanics.apply(time_s=t)` | Apply one line only | Yes |
| `line.mechanics.disable_vibrations()` | Remove dynamic motion, retain deliberate + installation placement | Yes |
| `mech.restore()` | Restore captured element input fields | Yes; automatic-enable choice unchanged |
| `line.mechanics.clear()` | Restore and detach this line | Environment bodies/sources remain |
| `mech.clear()` | Restore and remove all mechanics effects/bindings | No |

Adopted deliberate element inputs survive removal. Body reference settings
belong to the mechanics model and are removed with it. Raw edits of effective
alignment fields are rejected instead of silently adopted or erased.
If automatic updates remain enabled after `restore`, the next tracking call
applies motion again. Use `disable_vibrations` or `clear` to stop those updates.

## Tracking and clocks

```python
line.mechanics.enable_vibrations(time_series=motion, time_origin_s=0.)
line.track(particles, num_turns=128)
line.mechanics.disable_vibrations()
```

The default freezes the entire line once per turn at
`particles.time_s + time_origin_s`. A later tracking call continues at particle
physical time. Different prescribed beam runs can use the same source and
explicit origins without advancing one another. On the supported fixed-period
path, compensated clock accumulation keeps splitting a run into calls from
changing its motion phase. Energy-changing cases use the normal tracker clock.

`backend='auto'` selects prepared context-native source/composition/lowering
where supported. Sampled/custom sources use a bounded host batch. Logs,
time-dependent variables and collective callbacks retain validation/fallback
semantics. Explicit `backend='compiled'` requires geometry and static placement
to remain fixed for the call; strength changes are independent of that contract.
The reference backend is available explicitly. End-of-call fields retain the
last tracked freeze, rather than applying an untracked future state.
Preparation currently includes all bodies in the mechanics environment, even
when some belong only to another attached line. The recorded single-line FCC
costs therefore do not establish scaling for several disjoint body trees.

`sampling='passage'` is a slower reference-speed comparison using
`t + s/(beta0*c)` at each attached occurrence. It excludes particle-specific
`zeta/delta` delays. Turn-held sampling is an approximation governed by
`2*pi*f*T_rev`; being below Nyquist alone does not make it quasi-static.
Lesson 015 compares independent bunch phases, actual frozen Twiss and driven
tracking. It does not implement packed multibunch maps or dynamic-equilibrium
Twiss. Disable automatic motion and apply a fixed time before ordinary Twiss.

Whole-environment JSON/copy preserves builtin definitions and the currently
applied state. Runtime buffers, computed manager records, enable flags and
filter state are not serialized. Standalone attached line copy/export and
environment pickle are guarded; use environment JSON/copy for this draft.
Export checks ownership and captured geometry before saving, so copying cannot
turn a conflicting raw alignment edit into an accepted mechanics output. A
frozen built-aperture snapshot can be saved; changing pose still requires its
normal discard/rebuild. An empty or explicitly cleared manager can load saved
definitions; merely accessing `line.mechanics` does not configure a scenario.

## Explicit draft boundaries

The current numeric-field backend rejects slices, shared/repeated element
objects, live alignment expressions, geometry edits after binding and built
aperture models during application. Live reference-geometry expressions are
retained, but their values must remain consistent with the adopted survey.
SyncTime, partial turns, backtracking, `line.track(..., freeze_longitudinal=True)`
and pipeline hold are guarded while vibrations are enabled. Copy construction rings
before attaching mechanics when their sections share element objects.

CPU and OpenMP kernels have execution tests. CUDA/OpenCL kernels use the same
portable source, but actual GPU correctness, memory ownership and performance
are pending hardware validation. Use the explicit acceptance harness before
claiming a device result. No whole-run device pose history is allocated.

Occurrence/phase buffers, long-run source streaming, engineering FCC girder
assignments, aperture coupling and moving-BPM/feedback scheduling remain later
work. These boundaries permit a useful working draft now; they are not promises
that the complete FCC ground-motion/correction workflow already exists.
