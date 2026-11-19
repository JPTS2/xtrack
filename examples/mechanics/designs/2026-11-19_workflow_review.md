# Ownership and lifecycle review

The current user preference is A's named manager/factory style, with explicit
source objects where their physical model has parameters. The main questions
are ownership, survey geometry and time, rather than which spelling requires
fewer lines. The demo ring now leads the FCC-ee workflow; the three small
examples isolate ownership and lifecycle behavior.

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
invalidate replay bindings and discard stale derived aperture models. Stored
input motion remains readable after cleanup so different scenarios can be
compared without retaining the old active model.

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

1. Keep the native example surface small and edit it directly with user
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

## FCC-ee workflow review — 19 November 2026

Recommendation: develop the native interface through independently runnable
lessons on the same demo ring as feedback. Each lesson should expose user
parameters, ordinary named Xsuite objects, a physical comparison, a small table
and a figure. Keep scenario construction readable in the script; move only
reusable machine layout into an `env.call(...)` builder. Do not make the user
read test machinery or all the earlier research examples to judge the UI.

### Verified current capabilities

| Requirement | Native implementation today | What remains |
| --- | --- | --- |
| Assign elements to rigid girders after building a lattice | Named bodies, surveyed anchors, exact parent/child transforms and fixed mount poses inferred from survey | Explicit editable local mount frames, feet, coverage/overlap inspection and machine layout builder |
| Translations, pitch, yaw and roll | All six **static** body settings; finite SE(3) composition | Sampled angular motion, rotational PSDs, joint translational/angular channels |
| Ground displacement PSD and seed | Short host record at selected anchors; one scalar local channel, real exponential Euclidean correlation | A field sampled at arbitrary receivers/feet, frequency-dependent complex cross spectra, multiple components and scalable sampling |
| Plane waves | Direction, polarization, phase, speed and nominal world position | Native foot/body response and wave-driven rotations; physical-medium models supplied separately |
| Stated/measured vibration | Scalar trace, per-target channels, explicit interpolation and bounded replay; custom pure local/world source protocol | Six-coordinate traces, persistent channel identity and reusable records independent of target binding order |
| Floor → girder → magnet → BPM response | Rigid inheritance and static mount geometry only | Transfer paths, input/output frame and reference-point contracts, causal filter state and independent internal excitation |
| Located pumps and equipment | User can implement a prescribed vector displacement field in `evaluate` | Named equipment pose/mount, shared latent source channels and explicit propagation/coupling models |
| Survey and plots | Non-mutating body/element pose dictionaries; actual element-frame survey after `apply`; demo plots | A convenient time-selectable mechanics survey/table and shared plotting/animation interface |
| Beam study | Static frozen Twiss and single-line turn-boundary host updates | Within-turn passage sampling, mounted BPM readouts, apertures, slices/occurrences and actual GPU performance evidence |
| Disable/remove/rebuild | Source disable, input restore, line detach and complete clear; deliberate rolls preserved | Effective output buffers that leave element input definitions untouched; named scenario/run comparison |

The demo ring exposes a concrete integration restriction: its 16 named lines
include construction sections sharing elements with the assembled ring. The
current alias guard rejects attachment even though only the final ring is being
studied. `line.copy()` before adding mechanics produces an independent study and
is used explicitly in example 004. Removing that requirement is an acceptance
criterion for the occurrence/boundary backend; silently ignoring aliases while
writing their shared element fields would not resolve the ownership problem.

Example 004 assigns 168 independent supports to quadrupoles and switches among
a vector wave, a spatially correlated vertical PSD and stated samples. The
nominal 1.5 km ring closes to approximately 4.5e-13 m in the local survey probe.
The example verifies requested versus applied element entrance positions, plots
actual world displacement versus s and source history, and displays magnified
anchor translations and a separately magnified proper rotation. Magnification
is presentation-only. It does not supply an FCC-ee support layout: bends,
sextupoles and BPMs are not moving in this first lesson. No transfer function is
being hidden behind the word `floor_input`.

### The physical model we need

Keep four concerns explicit:

1. **Geometry and mounting.** Bodies, nominal world poses, local mount frames,
   support contacts and line occurrences. The mounting tree is rigid. A pump,
   girder, quadrupole and BPM may have different reference points and axes.
   Keep design roll/local errors, deliberate body alignment, and vibration as
   separate inputs. Survey provides poses; it does not define a response law.
2. **Excitation fields.** Site ground motion, waves, local equipment sources,
   stated/measured motion, and independent internal excitation. A spatially
   varying floor is a field, **not one rigid parent body**. Define the PSD,
   spatial cross spectra/coherency, frames, units, seed and phase/time origin.
   Sample it at the support locations, not just at magnet centers.
   A large-machine site model also needs an explicit local vertical convention;
   an arbitrary line-local y axis must not silently define the ground direction.
3. **Mechanical responses.** Named paths or multi-input response blocks turn
   input channels into body motion. A girder response can map 3N contact inputs
   to six rigid outputs; a scalar vertical transmissibility is the first useful
   special case. Response connections need not follow the mounting tree.
   User-supplied loaded measurements, analytic models and coupled mechanical
   solutions should all be usable without a structural solver inside Xtrack.
4. **Observation and tracking.** Lower the resolved motion to occurrence/boundary
   transforms; sample at physical passage times; project the beam into the BPM
   electrical reference frame. Electronics noise belongs to the measurement
   model, separately from movement of the pickup. Feedbacks consume that readout.

An initial spectral response can use `q_out(f) = H(f) q_in(f)`, including complex
phase. For multiple inputs, the spectral matrix transforms as
`S_out = H S_in H†`, with `S_ij = E[Q_i Q_j*]` under the chosen normalization.
Frames, reference points and units belong to the channels: m for translations,
rad for rotations, m/rad or rad/m for cross couplings. Validate against
independent cross-spectrum estimates; take care that
[SciPy csd](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.csd.html)
uses `conj(X) * Y`, the opposite ordered pair to this matrix convention.

For multi-foot input, rigid fitting is a specified kinematic approximation or
one part of a response model. It is not a prediction of forces, loading or
contact dynamics. A rough displacement field does not automatically have a
finite point slope; feet baselines/spatial regularization must be explicit
before claiming pitch/roll PSDs. Likewise, a scalar site PSD does not determine
directional coherence, polarization or propagation speed.

An absolute magnet/girder transmissibility cannot be added directly to inherited
girder motion. In a linearized channel representation, subtract the inherited
rigid transport `B`: the relative correction is `(H - B) q_parent`. `B` includes
the lever arm and axis changes, so it is not always the identity. For finite
poses, derive the exact relative transform from the desired absolute pose.
Chained conditional responses must refer to the same loaded system; a product
of independently unloaded oscillator models is a different physical model.
The earlier loaded two-body study is research evidence outside this Git
repository, not a native feature.

Support both offline prescribed records and eventually causal response filters.
An FFT response of a finite periodic record must advertise its boundary/startup
assumptions. Zero-phase forward/backward filtering is not a substitute for a
causal response. For time-domain filters, keep a bounded state per response and
an explicit initialization/warm-up contract; state can pass between consecutive
blocks, as in [sosfilt](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfilt.html).
Beam/feedback execution must not silently restart the filter at each track call.

A local pump needs an equipment pose, mounting point, excitation quantity and a
coupling path. A prescribed displacement source and an applied force source are
different inputs with different transfer units. A pump rigidly mounted on a
girder can excite that girder directly and the floor through separate paths;
declaring its coordinates does not compute either path. Multiple receivers
must see the same latent pump motion, with the specified phase/delay/gain.
Equipment coordinates should live in the mount frame, so changing a support
layout moves its attached sources consistently. State which reference geometry
the propagation is linearized about; nominal-receiver sampling is a small-motion
approximation, not a fully moving-medium model.
Regional PSDs must join through a valid joint spectral model, rather than
independently resetting channels at arbitrary region boundaries.

### Example sequence and acceptance criteria

The next three lessons should be implemented first, evolving the native API
only as each lesson needs it. These are planned capabilities, not runnable API
claims. Keep the existing small examples as focused reference scripts.

| Next lesson | What the reader edits | Evidence it must expose |
| --- | --- | --- |
| Ground motion and six rigid coordinates | Wave direction/polarization, scalar/vector/angle trace or PSD, support feet | Native survey at time, real-unit histories, separately scaled translation/rotation plots, rigid-distance check and reproducible common input |
| Floor/girder/magnet/BPM response chain | Loaded scalar complex TFs first, then matrices; mounts/reference points | Input/output PSD and cross phase; unity limit gives rigid common motion, zero absolute response stays fixed, no inherited-motion double count |
| Located pump and alternate support layouts | Equipment local mount, source type, supplied footprint/propagation, girder assignments | Same source channel at all receivers; position/delay and support changes alter the expected response, while geometry stays rigid |

Then add regional/frequency-dependent cross spectra and a two-beam shared-body
lesson; finally run a joint mechanics/feedback study with moving BPM frames.
Keep the current standalone mechanics checkout independent of the feedback
implementation until there is an explicit integration branch. The same lattice
does not imply copying the other branch's tracker/controller changes.

The survey/plot interface is part of these lessons, not a final presentation
task. A reader should be able to select physical time, bodies and a world/local
frame; inspect nominal/static/total poses and separate input records; plot
layout/axes/footprints plus displacement/rotation/PSD/coherence; and animate
a short decimated record. Use explicit display gains and a proper scaled relative
rotation, not scaled Euler-angle differences or distorted body geometry. Accept
matplotlib plots initially; decide the native convenience methods by using them
in the lessons. No need to commit to a separate graphical application.

### Performance gates alongside the lessons

Do not wait for a large FCC-ee model to discover the execution costs:

- Retain the host implementation as a numerical reference. Prototype a separate
  occurrence-buffer lowering and compare cached girder boundaries on the same
  pose/mount model. Preserve element definition fields and support real cloned,
  sliced and reused lattice occurrences. Check observer/monitor frames and
  starts inside a group explicitly for a boundary scheme.
- Compile geometry and binding indices only after layout edits. Dense spatial
  eigenfactorization is a small reference method, not the whole-machine default.
  Select structured, modal, tiled or matrix-free spatial models with a stated
  accuracy cost. No claim that every measured covariance is low rank.
- Keep current poses, bounded source/filter state and small transfer blocks on
  device. Generate/replay in chunks; request histories only for selected bodies
  or decimated diagnostics. Stable source/channel identities and sample indices
  must make chunking, traversal order and diagnostic selection leave the process
  unchanged. The current whole-record seeded FFT does not yet provide that.
- Measure setup time, update/launch cost, tracking time and host/device memory
  separately at fixed physics accuracy. Run actual CPU and GPU cases before
  calling a backend ready; the local GPU runtime is not available here.
- One sample for an entire turn is a low-frequency approximation. Quantify the
  phase error using `2*pi*f*T_rev`; use passage times `t0 + s/(beta0*c)` (or the
  chosen event clock) when that error is unacceptable. Shared prescribed fields
  need a common time origin, not identical turn numbers. Stateful coupled
  responses/feedbacks additionally require consistent event ordering.

Suggested next implementation slice: six-coordinate prescribed motion plus a
native read-only motion survey/plot, then one loaded scalar response-chain
lesson with an explicitly mounted BPM geometry. Establish its moving readout
before claiming a beam-feedback correction result. Evolve plotting and the
effective-transform backend alongside these lessons rather than hardening the
current element-field-writing path into the final architecture.

Review validation: example 004 ran on CPU with all three source selections;
maximum requested/applied entrance-position discrepancy was 1.14e-13 m and
orientation discrepancy 2.83e-16 rad. The first quadrupole's entrance/exit
distance is preserved within the 1e-12 m check. Frozen moved-ring `twiss4d()`
returned tunes 8.754832487 / 10.017452422 in the stated-motion case, and complete
cleanup removed all bodies. The 22 native mechanics tests passed. Wave and PSD
figures were rendered and visually inspected. This validates the geometry/input
lesson, not a ground-to-magnet dynamic response or GPU throughput.
