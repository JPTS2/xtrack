# Local mechanics increments and compensated arithmetic

The previous FCC horizontal precision floor was caused by a representation
choice: the code added small motion to kilometre-scale world coordinates, then
subtracted those coordinates to reconstruct the effective element alignment.
The correction is to carry the rigid deformation separately from the nominal
world placement throughout the mechanical hierarchy and tracking lowering.

The native reference implementation now does this without changing the public
UI or the three placement layers. World `xt.Frame` objects remain ordinary
float64 display/export values. Their rounded positions are not the tracking
input and are not subtracted to obtain the table's dynamic displacement.

## What the composer compensation does and does not provide

`xtrack.composer.positions.CompensatedFloat` retains a high/low pair for
longitudinal additions/subtractions, with division by two for centres. Its
`difference_from` cancels common large terms before rounding the remaining
scalar difference. That is useful for line composition.

It is not a compensated matrix algebra: it does not provide general products,
dot products, rotations or trigonometry. Wrapping input coordinates in this
class and subsequently converting them to an ordinary `Frame.matrix` would
discard the low part before the problematic subtraction. A complete high/low
world transform implementation would need compensated products and reductions
as well as additions, on CPU and GPU.

The principle of retaining separate parts is the right one. For mechanics,
nominal world geometry plus local affine increments is a smaller and more
direct representation. The compiled runtime can consume ordinary float64
arrays, without carrying two doubles for every world matrix entry.

## Finite local representation

For each body, retain displacement `d` [m] and `E = R - I`, about that body's
nominal local frame. `E` is computed directly, not by subtracting an identity
matrix from a rounded rotation. Each elementary rotation uses
`cos(a)-1 = -2*sin(a/2)**2`, retaining second-order terms for small angles.
The survey convention remains `Ry(yaw) Rx(-pitch) Rz(roll)`.

Composition uses

```text
d = d_left + d_right + E_left d_right
E = E_left + E_right + E_left E_right
```

These are finite rigid-transform identities, not a first-order approximation.
They allow finite deliberate or installation rotations together with tiny
dynamic increments. Relative/absolute source semantics and insertion order of
relative rotations are retained.

The final Euler extraction also needs care. At pitch close to ±pi/2,
independently extracting yaw and roll from vanishing matrix entries can corrupt
their well-determined combination. The near-lock branch recovers yaw from the
dominant first column after choosing roll, preserving the rigid matrix even at
exact lock. Individual yaw/roll labels remain nonunique there. Tests cover exact
lock, 1e-12 rad proximity on both sides, and random finite rotations.

For parent and child nominal axes `Rp`, `Rc`, define `C = Rp.T Rc` and the
fixed parent-local lever `r = Rp.T (pc-pp)`. The inherited child deformation is

```text
d_child = C.T (d_parent + E_parent r)
E_child = C.T E_parent C
```

Only fixed nominal coordinates are differenced. Small moved coordinates are
never added to and then subtracted from world positions. Child-local placement
and source increments are subsequently composed. Absolute-driven children use
the inherited static placement, replacing inherited vibration as before.

The implementation is in
[`local_deformations.py`](../../../xtrack/mechanics/local_deformations.py).
The public pose dictionary retains this local data for native lowering;
constructing the world matrices is a separate display operation.

## Element entry lowering and observation

The adopted baseline element fields, including design roll and rotation
anchor, remain the deliberate reference. The baseline entry transform `(t0,R0)`
comes directly from those input fields and native straight/curved entry-transform
rules, rather than from subtracting two surveyed world positions.

Let `A = R_ref.T R_body`, and let `lever` be the baseline element entrance in
nominal body coordinates. The moved entry transform is

```text
t = t0 + A (d + E lever)
R = R0 + (A E A.T) R0
```

Effective alignment fields are derived from this local result. At exact zero
deformation, the original seven fields are returned verbatim, including the
original rotation anchor. Applying zero mechanics therefore cannot silently
reclassify an intentional placement into another numeric parameterization.

Table installation/dynamic columns use differences of these local deformation
parts and lever arms. Absolute world matrix columns still have the resolution
of their world origin; the relative columns need not share that limitation.
`line.mechanics.read_bpm` similarly constructs the pickup frame relative to its
nominal beam entrance before intersecting the beam ray. The generic geometric
intersection now works in pickup-local coordinates, avoiding world-coordinate
addition of the beam's small transverse coordinates.

Rigid-motion port transport uses the same incremental rotation conjugation.
It no longer constructs a full world rotation and subtracts identity; zero
input therefore remains exactly zero even with finite port orientations.

Selected body/history queries compose only the requested ancestor closure.
Sampled motion records interpolate only the required target channels. Arbitrary
joint stochastic sources retain full receiver-set evaluation before selection,
so selection does not silently change their realization or correlation law.
The 33-sample single-element FCC history probe took about 0.17 s, with one
full-line geometry validation still included. This is an indicative UI cost;
the source type differs from the previous approximately 1.57 s harmonic probe.

## Numerical evidence

The targeted precision tests independently check:

- 1 nm, 10 pm and 100 fm translations, including a 3 mm deliberate element
  offset, with origins at zero, 20–50 km and 1e9 m;
- a 1 nrad yaw of a 0.5 m quadrupole: entrance shift
  `x=-0.25*sin(a)` and `s=0.5*sin(a/2)**2`, retaining the 1.25e-19 m
  second-order longitudinal term;
- exact zero-motion preservation of a deliberately shifted, rolled and
  anchored curved element;
- finite rotations, including configurations close to Euler singularity,
  against independent finite-frame conjugation;
- nested installation and relative vibration invariant under translation of
  the whole world origin, plus absolute-child semantics;
- 100 fm table-history and pickup signals at a 1e9 m world origin.
- exact zero and the nanoradian second-order lever term in rigid-motion port
  transport, plus selected-ancestor queries preserving joint source semantics.

The public bulk setup was also exercised on all 2,663 quad sites of the pinned
FCC Z model, retaining all 2,544 expression-driven quadrupole lengths. These
sites are provisional supports, not engineering girder assignments. The
[precision probe](../review/fcc_precision.py) and
[recorded results](../review/results/fcc_freeze_runtime/precision.json) show:

| Local horizontal command | Maximum effective-field error |
| --- | ---: |
| zero | exactly zero |
| 1 nm | 3.56e-22 m |
| 10 pm | 3.56e-24 m |
| 100 fm | 3.56e-26 m |
| 1 fm | 3.56e-28 m |

The residual scales with the command; the previous approximately 15 pm
additive floor is absent. The tiny proportional residual reflects finite
precision of the accumulated survey axes and their transpose products. This
probe validates prescribed translations, not the full physical ground-motion
model, nor GPU parity. Its concurrent setup timings are not used as performance
claims.

## Remaining limits

The effective scalar alignment fields still have ordinary float64 precision.
Adding a tiny increment to a very large deliberate effective shift can lose
that increment at the final field assignment. A 3 mm baseline has roughly
4e-19 m spacing, well below pm-scale requirements; an unphysical kilometre-scale
element-local offset would not. If such inputs become a requirement, the
particle map must consume baseline and incremental transforms separately.

The clock provides a separate scalar compensation use case. Repeated binary64
addition of the FCC reference period for one million turns differs from
`N*T_rev` by about -4.15 ns in a direct arithmetic probe. At 1 kHz this is
2.61e-5 rad, an upper 2.61 pm waveform error for a 100 nm sinusoid. This is an
arithmetic experiment, not a measured million-turn tracking job. The prepared
fixed-reference mechanics scheduler now retains a `CompensatedFloat` high/low
clock in weak-key per-particle-container state. It publishes the rounded
physical time while retaining the low part across split calls. External edits
of time, reference turns or revolution period start a new epoch at the current
published time. No particle/device layout is enlarged, and discarded particle
containers do not retain clock cache entries. General variable-period and
collective tracking paths still require their own clock-convergence assessment;
this does not replace the clock of every tracking mode.

Likewise, multiplying a small rotation by a genuinely long physical lever arm
can amplify uncertainty in the nominal geometry. Local decomposition removes
artificial dependence on the world origin; it does not improve the accuracy of
the supplied geometry. Finite Euler fields retain their usual principal-angle
nonuniqueness at singularity, while the reconstructed rigid map is retained.

Field-to-rigid contact fitting has its own SVD and local-coordinate numerical
path. It needs separate small-signal/contact validation; fixing the hierarchy
lowering and port transport does not certify every source preparation algorithm.
