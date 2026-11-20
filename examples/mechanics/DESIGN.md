# Mechanics implementation and open decisions

This document describes the native exploration after the fourteen-example set.
The earlier [workflow review](designs/2026-11-19_workflow_review.md) is retained
as a dated design record; its capability table predates this implementation.

The earlier [FCC readiness review](designs/2026-11-20_fcc_scaling_review.md)
records the original reference backend's limitations on the requested lattice.
The [prepared-runtime review](designs/2026-11-20_prepared_runtime_review.md)
describes the subsequent native bulk setup, local precision and context-kernel
implementation, with a twofold tracking-slowdown ceiling. Its measurements and
fill-pattern investigation distinguish current capabilities from the remaining
GPU, long-run streaming and packed-bunch work. Support cost proxies are not
accepted engineering girder assignments.

The [first-demo readiness review](designs/2026-11-20_demo_pr_readiness.md) is the
current delivery assessment. The UI/physical model is retained; further studies
are restricted to GPU acceptance, occurrence/phase consumption and bounded source
execution. The new 015 lesson promotes the bunch/Twiss study to a portable example.

## Ownership and the example contract

The interface follows the verified Environment/Line facade pattern in Xtrack
and Xfields, named factories in Environment/aperture, and the explicit
Xcoll scattering lifecycle. `env.mechanics` owns one physical model, while
`line.mechanics` owns attachments and its explicit survey placement in that
world. An Environment is also a namespace for construction sections and optics
alternatives; it does not automatically assert common geometry or a common beam
clock. Several named scenarios remain useful; machine-specific parameterized
builders and immutable installation/motion records support comparisons now.

The feedback lessons are the formatting and interaction precedent: required
packages, editable user parameters, readable setup/physics sections, terminal
comparisons and figures. Model-specific scripts expose their assumptions.
Reusable physics lives in `xtrack/mechanics`, not in hidden example helpers.
The shared demo-ring Python files are included without importing feedback code.
The review runner only captures outputs and provenance; it does not implement
motion or tracking physics.
Example 013 assembles the whole prescribed-motion pipeline: a shared site field,
local excess spectrum and mounted pump at 672 feet, fitted support poses,
supplied conditional causal responses, magnet/pickup port transport, independent
installation errors and slow drift. Its 1.5 s prehistory is explicit. The short
source histories are on the host; no GPU pose history is allocated. It reports
the interpolation approximation separately from numerical map correctness.

## Three classes of placement

1. **Deliberate reference.** Existing element-local offsets and design rolls are
   captured from the line, including an intentionally off-axis quadrupole.
   `adopt_reference()` makes this classification explicit before attachment;
   it does not rewrite the line. Without it, binding still adopts current inputs.
   Mechanics body reference settings extend the intentional support geometry.
2. **Installation.** `body.installation` is a separate static rigid transform.
   Seeded installation records hold the realized errors independently of dynamic
   seeds, time sampling and diagnostics. Applying a corrected installation case
   can preserve the immutable as-installed case. Body-local Gaussian/uniform
   draws have explicit translation/angle scales, optional Gaussian rejection
   truncation and named common fixtures. Default root selection avoids silently
   adding errors at every hierarchy level. Stable hash-based identities make
   subset/reordering invariant. A shared local draw across differently oriented
   bodies is not a shared world displacement; grouping states that local rule.
3. **Dynamic.** Sources describe vibration, measured motion and slow drift.
   They never mutate deliberate or installation settings. A stationary random
   process need not vanish at time zero. Zero-start/ramp/transient behavior must
   be an explicit source/filter initial-condition model, not an unannounced
   seed-dependent DC subtraction.

For one body, the static local transform is the deliberate transform followed
by the installation transform. Parent/child static transforms compose exactly.
Reference tables retain original element inputs and effective-field audit
columns remain available. `disable_vibrations` preserves installation; `restore`
restores captured element inputs; `clear` removes owned mechanics definitions
and effects. Adopted intentional line offsets/rolls survive. Body reference
settings that belong to a removed mechanics model do not remain implicitly
baked into the line. Raw edits to effective fields are rejected rather than
silently adopted or erased. A general rebase/bake interface is not implemented.

## Coordinates and motion records

Frames map local `(x,y,s,1)` into world `(X,Y,Z,1)`. Positions are metres,
angles radians. Survey pitch/yaw/roll use
`Ry(yaw) Rx(-pitch) Rz(roll)`; pitch is not a right-handed positive x rotation.
RigidMotion stores `[x,y,s,pitch,yaw,roll]` in nominal body axes. These are not
rotation-vector coordinates. Angular traces interpolate those specified survey
coordinates; sufficiently resolved sampling remains the user's responsibility.

Geometry composes SE(3) matrices, including deliberate rolls and lever arms.
Several relative input translations add in the stated local axes; their finite
rotations left-compose in source insertion order, before the static orientation.
A joint six-coordinate record is preferable when a model defines one combined
orientation. No subtraction of Euler angles is used to remove parent motion.

An absolute input defines dynamic displacement/orientation about the body's
static placement, in nominal body axes. Its prescribed output replaces inherited
dynamic motion, retaining parent/child static placement. An absolute-driven
body accepts one explicitly combined input. This makes zero and unity absolute
response limits well-defined. A body/ancestor can share an absolute multi-body
record; relative sources cannot accidentally drive both in one binding.

`RigidMotion.transport` computes exact prescribed rigid transport from one
nominal port to another, including changed axes and reference-point lever arms.
Response matrices must be supplied at their specified input/output ports.
Dynamic multiplication by a linear frequency response is still a small-signal
model even though the resulting geometry is composed exactly.

Records are immutable bounded host arrays with explicit time limits. No implicit
periodic replay or extrapolation occurs. Manager-computed records validate the
binding revision and remain readable after cleanup. Explicit supplied RigidMotion
sources serialize their data; computed manager records and automatic enable
flags are not serialized. Causal filter state is separate run state.

## Ground fields and contact reconstruction

A spatially varying floor is a field. Receivers have nominal world positions;
contact locations are defined once in each support's local geometry. A source
is sampled jointly before processing support response. A large-machine site
model must define its vertical directions and world placement; arbitrary local
beam y axes do not establish site vertical.

The finite FFT PSD reference uses a one-sided density in m²/Hz (or rad²/Hz for
AngularPSD), zero DC and zero unpaired Nyquist power. Gaussian quadratures have
expected discrete mean square `df * sum(bin_density)`. A scalar exponential
Euclidean spatial kernel describes real amplitude coherency, not squared
coherence or traveling-wave phase. Dense spatial eigendecomposition is a small
reference method, including singular coincident receivers. Sampling checks the
whole interpolated PSD support against Nyquist, not only nonzero tabulation
points. For fixed count/step/receivers/seed, phases have physical origin zero;
changing record count/step or receivers changes that finite spectral model.

SpectralGroundMotion uses independent random temporal frequencies sampled from
the normalized PSD, isotropic multivariate-Cauchy wavevectors, and shared random
phases. The ensemble space-time covariance factors into the target temporal
covariance and exponential Euclidean spatial coherency. A finite number of
modes approximates this model: spectral lines, finite rank and spatial covariance
error remain visible. This model is continuous in space/time and does not change
with receiver subsets/order or time chunks. Its mode parameters are bounded;
NumPy evaluation uses temporary time-by-mode arrays, with explicit workspace
checks in field sampling. It is not an elastic-wave or measured site model.

RegionalGroundMotion adds independent excess fields with Gaussian amplitude
weights: regional PSD contribution is weight squared times its PSD. Background
cross spectra remain shared, including across a region boundary. Independent
seeded components must have distinct seeds, also through nested vector/region
models. VectorGroundMotion provides independent scalar latent components and a
specified real mixing matrix for world XYZ; mixing creates cross spectra.

A weighted proper Kabsch fit reconstructs one rigid pose from at least three
noncollinear contacts. It reports residual contact mismatch, never deforms a
body to match incompatible floor displacement. This is kinematics, not a force,
contact or loaded-support solver. Exponential ground displacement does not have
a finite point-slope variance; finite feet baselines or other regularization
are essential before claiming angular ground spectra.

LocatedSource has one scalar latent displacement waveform, an equipment position,
world polarization, a supplied exponential amplitude footprint and travel delay.
Mounting an equipment body positions the source consistently in the chosen
nominal layout. Neither force coupling nor material propagation follows from
coordinates alone. Direct support and emitted floor paths are explicitly supplied.
A measured waveform must include all delayed prehistory. Propagation here uses
nominal positions: it is a small-motion reference-geometry model, not a moving
medium or a reacting mounted machine.

## Response and observation

Complex frequency responses use `exp(+i omega t)`. The loaded two-mass example
solves one consistent girder/magnet assembly; its conditional magnet/girder
response cascades with the loaded floor/girder response. Replacing the first
stage with an independently unloaded oscillator changes the model. The BPM stage
in that lesson assumes negligible pickup loading. Supplied measured/analytic
conditional paths are also supported, with their physical loading assumptions
remaining explicit.

A scalar response is dimensionless displacement/displacement or angle/angle.
A 6x6 matrix has mixed SI units: off-diagonal terms can be m/rad or rad/m.
Tabulated responses interpolate complex real/imaginary parts within their supplied
frequency bounds. For multiple inputs, `S_out = H S_in H†`, using
`S_ij = E[Q_i Q_j*]`; SciPy csd uses the opposite ordered conjugation convention.
Do not infer independent inputs or their coherence from marginal PSDs alone.

Offline `through` filtering is circular frequency-domain response on the finite
record. It is not causal startup or implicit zero padding. A nonzero endpoint
whose requested response is complex is rejected rather than losing that phase.
CausalResponse instead retains bounded second-order filter state across blocks;
default zero-order-hold discretization matches the specified continuous LTI
response at sample times for held input. Optional bilinear discretization has
its own frequency mapping. Initialization is at rest; an arbitrary stationary
filter state is not silently inferred. Live coupled filter/controller scheduling
is not yet part of the tracking hook.

A BPM is geometry plus an electrical reference plane, not simply a nominal y
sample. `read_bpm` intersects the reference-frame beam ray with the actual plane,
including transverse/longitudinal offsets and rotations. px/py are normalized
to reference p0, with forward pz from delta. Electronics/noise are separate.
The moving-BPM correction lesson uses frozen Twiss states and explicit reads;
it is not an online feedback integration. Full common-rigid-transform covariance
of the ray/plane observation is tested.

## Execution, timing and remaining scope

Body and attached-element pose queries use normal Xsuite table selection.
`mech.get_table()` reads configured body inputs; `time_s=...` requests composed
world poses. Body `kind` labels are explicit, immutable topology metadata and
survive environment copy/JSON. Element rows retain `element_type` and `body_kind`:
several magnets on one girder remain separately selectable. Unattached line
elements are outside the mechanics table.

The source table inventories registered bindings without evaluating models or
copying waveform arrays. A selected pose-table row can access its live element
or body and compute a bounded host history, including inheritance and lever
arms. This is an explicit diagnostic, not an all-body tracking cache. Histories
use current static settings and an explicitly supplied or table-bound source
record. Numeric table exports omit live query context and source records;
history exports preserve their time index. Topology/source edits invalidate
live query context. Relative angles come from relative matrices in the preceding
layer's axes, never Euler-angle subtraction.

Example 014 plots this history and animates a single hollow cuboid using the
same element frames. The cuboid dimensions are illustrative, not aperture or
field data. Equal world scales and a local triad expose rigid motion; separate
translation/rotation magnifications affect presentation only. Preparing a small
selected history and rendering/encoding its GIF have distinct measured costs.

Explicit application retains a host correctness reference. Automatic turn-held
tracking now prepares context-native source/hierarchy/lowering kernels and writes
effective numeric fields on the tracking context. Both preserve captured input
data; occurrence-aware non-overwriting buffers remain the next backend. Slices, shared/repeated element objects,
alignment expressions and post-binding geometry edits remain guarded. Built
aperture models are guarded against stale geometry; pipe/body ownership and
mounted tracking monitors still need joint implementation. Whole-environment
JSON/copy works for builtin sources; standalone line copy/export remains guarded
after binding. Copy the demo ring before binding to avoid its shared construction
sublines. Individual factories include their required survey in setup cost;
`new_girders(mapping)` resolves a whole batch with one survey and retains live
reference-geometry expressions.

Pure prescribed sources can be sampled from several beams with explicit common
physical time origins and distinct line-local buffers. Running one line does
not advance another's state. Stateful responses/controllers and encounters need
consistent event ordering; there is no common turn counter inferred from env.
SyncTime, partial turns and pipeline-hold integration remain guarded.

Turn-held sampling prepares one time per line turn. Passage sampling instead
prepares each occurrence at `t0 + s/(beta0*c)`; it does not include particle-specific
zeta/delta arrival. Its current repeated hierarchy evaluation has poor scaling.
The dimensionless error scale is `2*pi*f*T_rev`, not source amplitude alone.

The lessons separate line load/copy, explicit tracker build before survey,
reference adoption, attachment, and optional geometry/table inspection. Surveys
required internally for attachment remain included in the attachment cost.
Application and subsequent diagnostic surveys are separately measured. The
timing lesson additionally isolates first kernel use, pure pose calculation,
numeric writes, frozen tracking and dynamic tracking. GPU timings require actual GPU hardware;
none are claimed here. Field generation and response preparation currently use
NumPy/SciPy on the host, even when static tracking uses a GPU context.

The next implementation stage should follow the captured costs: extend prepared
geometry/binding indices to occurrence-indexed effective buffers, compare them
against cached girder boundaries, and keep only current poses plus
bounded source/filter state on device. Do not cache a whole time-by-body matrix
history by default. A boundary scheme must handle starts inside a group and
observer frames. A buffer scheme must keep definition fields authoritative and
support ordinary cloned/sliced lattice occurrences. Spatial models require their
own stated approximation/structure; an arbitrary measured covariance need not
be low rank. Long-machine streaming, actual CPU/GPU throughput, aperture coupling,
physical BPM tracking monitors and joint mechanics/feedback scheduling remain
separate reviewable implementation slices. The prepared-runtime review also
records short-call failures against the twofold ceiling and the need for
phase-aware maps when multiple bunches are packed together.
