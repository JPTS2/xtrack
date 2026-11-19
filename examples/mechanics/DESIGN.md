# Ownership and lifecycle review

The current user preference is A's named manager/factory style, with explicit
source objects where their physical model has parameters. The main questions
are ownership, survey geometry and time, rather than which spelling requires
fewer lines. Only three examples need to lead this review.

## Verified Xsuite precedents

| Existing code | Observed pattern | Mechanics consequence |
| --- | --- | --- |
| `Environment.new`, `EnvAperture.new_profile/new_pipe` | Named factory methods on an owner | `mech.new_girder` fits existing style |
| `env.aperture` and `line.aperture` | Shared definitions versus line placements; survey-based models | Physical world on the environment; attachment/reference-frame view per line |
| `env.xfields` and `line.xfields` | Owner-bound facades, instantiated by accessors | No public installation shim or manager constructor inside an accessor |
| PIC `spacecharge_install_frozen/spacecharge_replace_with_pic` | Flat line operations with explicit profile/solver objects | Factory-style API and typed physics models coexist |
| `PICCollection` | Reuses compatible meshes/FFT plans; tracks context/buffer | Cache prepared geometry/work only when useful; not a full time × body history |
| `line.xcoll.scattering.enable()/disable()` | Explicit lifecycle, separate from construction | `enable_vibrations`/`disable_vibrations`, preserving static alignment |
| `Particles.time_s`, SyncTime | Physical clock distinct from per-particle revolutions | Do not infer a common multi-line clock from an Environment |

Local source inspection: Xtrack `environment.py`, `line.py`, `aperture/`,
`tracker.py`, `synctime.py`; Xfields `line_tools.py`, `environment_tools.py`,
`config_tools/spacecharge_config_tools.py`; Xcoll `line_tools.py`. The
[upstream Xfields line facade](https://github.com/xsuite/xfields/blob/main/xfields/line_tools.py)
also documents the flat-prefix choice explicitly. These precedents guide the
interface; their distinct physical ownership is not copied indiscriminately.

## Present model

One Environment owns one mechanics manager. Girders/bodies and source IDs are
physical definitions. Each Line has a bound view with selected element
attachments and a nominal survey placement in the shared world. A girder can
carry elements on several lines with different local axes. A line's optics
reporting convention must not silently redefine that placement.

The body tree handles exact rigid geometry. Source/response relationships are a
separate concern. This first exploration binds harmonic, measured, PSD and wave
inputs; local pumps, regions and transfer paths can reuse the source interface.
An absolute child response must not be added again to inherited parent motion.
The earlier numerical reference contains tests and derivations for this.

An Environment can also hold optics alternatives or disconnected scratch lines;
it is not inherently one physical machine. Only explicitly attached lines join
the mechanics model. The aperture precedent is about ownership/accessors, not
proof of a global survey frame. This exploration currently has one physical
world per manager. The user likes named alternative scenarios; one active setup
built by a parameterised machine script is not a blocker initially. Separate
run/input records should not require duplicating immutable geometry merely to
try another seed or controller gain.

The native lowering currently composes surveyed SE(3) frames and writes numeric
effective element alignment once per requested time. It is a correctness path.
The same model can later lower to cached boundary maps or occurrence buffers.
Neither choice belongs in an ordinary `new_girder` call.

Original element-local errors, deliberate girder alignment and source-induced
vibration must remain separate. Design `rot_s_rad` is not a mechanics output;
captured local errors are restored exactly by removal. The compatibility path
does temporarily write effective fields, with baseline/effective audit tables
and conflict checks. This must not become an excuse to overwrite input
definitions in the eventual fast backend. `clear()` is a first-class lifecycle
operation: remove all owned effects and bindings, preserve deliberate inputs,
invalidate sampled records and discard stale derived aperture models.

## Clock decision still open

Static shared geometry and explicit physical-time application are available.
For one line, the exploratory automatic lifecycle uses the existing collective
Python turn loop and `particles.time_s`, without toggling time-dependent xdeps.
This is CPU/GPU-compatible reference scheduling, not a GPU throughput claim.

For several beams, choose a common physical event clock and determine when each
passage samples each body. Sequentially running both lines for the same number
of turns is insufficient when revolution times differ. SyncTime and the feedback
work are relevant shared infrastructure; they need an explicit joint contract.
Until that exists, automatic multi-line enable fails clearly. No fake global
clock is attached to the environment.

That guard is not a claim that every prescribed multi-line study needs a joint
mutable clock. Pure prescribed sources can be queried at each beam's passage
times with an explicitly shared time origin, writing only that line's effective
buffers. Coupled mechanical/filter/controller state or collective encounters
need additional event ordering. This is a concrete next clock experiment; the
current single-line reference path should not dictate the final scheduler.

## Exploration sequence

1. Keep the native three-example surface small and edit it directly with user
   feedback. No further requirement to review all 19 reference scripts.
2. Establish static/shared survey geometry, copy/serialization and physical
   observation frames. Compare aperture coupling and mounted BPM semantics.
3. Exercise source records, enable/disable, split runs and explicit clock cases.
   Extend sources with pumps, regional PSDs and loaded transfer paths.
4. Compare cached girder boundaries and occurrence buffers under these same
   semantics. Existing numerical probes support both. Measure update cost,
   launch cost and memory on actual GPU hardware before selecting a fast path.
5. Let this local exploration branch become messy as needed. Much later, review
   slices onto a fresh branch; the user will decide when to open a draft PR.

No PR is being prepared or published now, and exploratory implementation chunks
are not being presented as a clean final commit series.
