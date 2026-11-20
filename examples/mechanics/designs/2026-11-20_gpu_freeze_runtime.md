# Mechanics freeze points and a GPU-first execution path

This note separates verified code behaviour, measured CPU costs and GPU design
inferences. No CUDA or OpenCL runtime/device is available on this machine.
The new kernels have serial/OpenMP conformance tests and offline GPU syntax
checks; those are not GPU performance measurements.

## Recommendation

Keep the existing environment/body/source/table UI and rigid hierarchy. Define
the default tracking contract as one explicit physical-time freeze at the
beginning of each reference turn. Compute the shared body state once, then
consume immutable effective element data until the next freeze. Additional
within-turn updates should be explicit scheduler points, rather than source
evaluation inside every magnet's particle map.

The immediate implementation uses context-native source, body and lowering
kernels followed by the existing particle tracking kernel for one turn. This
changes execution rather than the physical model. The existing host lowering
remains a correctness/reference path. A factor-of-two maximum is an acceptance
criterion for each measured workload, not a demonstrated universal bound.

Do not replace the default tracker with a GPU kernel in which particle thread
zero mutates common alignments at each turn. Particle threads can be on different
turns and CUDA blocks have no ordinary grid-wide barrier in that kernel. Do not
have every particle evaluate the same floor spectrum or girder hierarchy.

## What the tracker actually does

The ordinary non-collective `track_line` kernel in `xtrack/tracker.py` contains
the turn loop inside the particle thread. On CUDA, a thread may complete many
turns independently of another thread. On serial CPU the local element function
often loops over all particles, so some geometric calculations are shared there.
This difference matters when extrapolating CPU map costs to a GPU.

The previous mechanics implementation already updated once per turn. Its
277 ms/turn FCC result was not caused by evaluating a source in every magnet
for every particle. The main costs were full-line validation, Python world-pose
composition/lowering and scalar field access. Changing only the time-freeze
assumption therefore would not have removed those costs.

The valid portable sequence is:

1. Prepare topology, relative reference geometry, field storage offsets and
   source descriptors once. Validate the frozen geometry at tracking-call entry.
2. At a freeze time, evaluate sources into one body-input array. For supported
   analytic sources this is a context kernel; general sources retain an explicit
   bounded host fallback.
3. Compose all required body deformations and lower them into alignment storage
   on the same context. The particle kernel reads the completed results.
4. Track one turn, or one segment ending at an explicit extra freeze point.

Kernels in a single CUDA stream execute in enqueue order. This provides the
dependency between preparation, lowering and tracking without a host wait after
each kernel. Different streams need an explicit dependency; sharing an
environment is not a synchronization mechanism. [CUDA stream ordering](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#cuda-stream-ordering)

The body kernel traverses each body's immutable root-to-body path independently.
It does not read a parent output written concurrently by another thread. This
costs approximately O(BD) for B bodies of typical depth D, rather than requiring
one launch for every hierarchy level. A pathological chain of B nested bodies
would make that O(B²); that is not a suitable mechanical layout. A level-sorted
backend can remain available if deeply nested inputs become realistic.

### Removing the remaining particle-clock synchronization

Zero host scalar accesses inside the mechanics update does **not** mean zero
host scalar accesses in a tracking turn. The reference collective scheduler
still calls `_prepare_particle_time_s_and_reference_turns` and
`_advance_particle_time_s_and_reference_turns`. These read/write the scalar
particle clock and read reference beta0. In CuPy, scalar buffer access includes
device reads or host-to-device writes.

The independent [clock-I/O probe](../review/mechanics_clock_io.py) starts after
mechanics preparation and instruments only the particle buffer. Its
[recorded reference counts](../review/results/fcc_freeze/clock_io_collective_reference.json)
show this slope:

| Clock field | Host reads per turn | Host writes per turn |
| --- | ---: | ---: |
| reference_turns | 4 | 1 |
| beta0[0] | 3 | 0 |
| time_s | 2 | 1 |
| t_sim | 1 | 3 |
| Total clock accesses | 10 | 5 |

There is additional constant call overhead. The serial CPU probe counts
29 total scalar buffer accesses per turn, including metadata/counters; CPU-only
active/lost bookkeeping means that total is not an exact CUDA count. The clock
access pattern follows common scheduler code, so it remains a GPU concern even
after the earlier per-element transfers have been removed.

The reference collective path also launches a zero-element tracking kernel for
end-turn actions after the full non-collective part. Thus the analytic case
has three mechanics kernels plus two tracking kernels per turn,
before monitors or other collective work. This is a reference prepared backend,
not the final lowest-overhead GPU scheduler.

A dedicated fixed-energy, non-collective mechanics scheduler now reads the
initial clock, reference period and capacity once per call. It enqueues the
prepared update followed by a full-turn particle kernel with end-turn actions
enabled, using the known freeze time without device-to-host scalar reads each
turn. A prepared publication kernel writes the shared scalar clock before the
next particle kernel, and publishes the call's final clock after the last turn.
Other collective/ramping/SyncTime cases retain the reference contract.

Repeating the [same clock probe](../review/results/fcc_freeze/clock_io.json) on
this path gives identical host particle-buffer counts for one, four and sixteen
turns: 19 reads / 152 bytes and two writes / 16 bytes per call. The measured
per-turn scalar-I/O slope is zero. Context reductions used for eligibility
validation also occur at call entry; this is not a claim of zero setup traffic
or measured CUDA latency. The analytic path still launches three mechanics
kernels, one clock-publication kernel and one full-turn particle kernel per
turn. It removes synchronization without pretending all launch costs vanished.

Do not instead let particle thread zero publish the shared clock at the end of
the particle kernel: a later-scheduled block may not yet have loaded its input
clock. `LocalParticle_to_Particles` deliberately treats shared clock fields as
read-only during the kernel call. Kernel boundaries, a separate publication
stage or immutable time-addressable phase data retain the necessary ordering.
This scheduling improvement addresses scalar clock traffic, while the
occurrence/phase consumption path below addresses bunch resolution.

## Once per turn is an explicit physical approximation

For this FCC Z model, T_rev = 302.343163 microseconds and f_rev is about 3.31 kHz.
Sampling below the Nyquist frequency does not establish that a common
turn-frozen field reproduces each magnet's physical encounter time.

| Sinusoid frequency | Phase advance over one FCC turn | Box-hold fundamental gain |
| --- | ---: | ---: |
| 10 Hz | 0.018997 rad / 1.09 degrees | 0.999985 |
| 100 Hz | 0.189968 rad / 10.88 degrees | 0.998497 |
| 1 kHz | 1.899678 rad / 108.84 degrees | 0.856273 |

The gain column is `sin(pi*f*T_rev)/(pi*f*T_rev)` for a continuous left-sampled
zero-order hold, with a half-step phase lag. It is an illustration of the hold
operator, not a prediction of the beam's response to a distributed ring. For a
magnet at nominal arrival delay tau, the common-time encounter approximation
has phase error `2*pi*f*tau`. Ground power concentrated at a few Hz makes the
default attractive. Saying only "below 1 kHz" is not sufficient to establish
accuracy at FCC.

Extra freeze points divide the maximum interval. At 1 kHz, 19 equally spaced
points per turn would bound the interval's phase advance by 0.1 rad; the same
illustrative budget needs two points at 100 Hz. This is not a universal required
number: convergence must use relative motion, beam observables and the actual
spectra. Extra points should have clear physical times and deterministic order.

The alternative `sampling='passage'` reference path prepares each occurrence at
`time_s+s/(beta0*c)`. This is prescribed reference-speed encounter sampling,
not particle-specific arrival-time motion and not a synchronized continuously
moving world. It remains useful as an accuracy comparison. Its current whole-
hierarchy-per-occurrence loop is deliberately not the fast backend.

Additional freeze markers are not implemented by the new kernels alone. They
need scheduler integration, including partial segments, monitors and clock
advancement. Xsuite's collective parts already define synchronization boundaries
between asynchronous non-collective sections. [Xsuite collective scheduling](https://xsuite.readthedocs.io/en/latest/collective.html#how-does-it-work)

## Alternatives and their limits

| Execution choice | Main benefit | Constraint |
| --- | --- | --- |
| Separate update kernels once per turn | Simple portable dependencies; O(B+A) current state; existing magnet maps | Several launches per turn, Python scheduling, existing per-particle misalignment work |
| Bounded immutable pose chunks | Existing particle kernel can keep several turns inside one launch | Needs tracking-side row selection and bounded memory; causal feedback limits preparation horizon |
| CUDA Graph replay of update/track sequences | Reduces repeated launch setup while retaining dependencies | CUDA-specific optimization, does not reduce arithmetic or supply OpenCL/CPU support |
| Independent analytic source calculation in every magnet/particle | Avoids shared mutable update state | Duplicates spectra and hierarchy work, expensive nonlinear source models, unacceptable default |
| Girder/subline boundary transforms | Can reduce per-particle frame changes for contiguous supports | Nested magnet residuals, apertures, BPM/RF frames, bend curvature and fragmented supports need exact treatment |
| A persistent cooperative CUDA kernel | Could synchronize stages on device | Requires cooperative launch/residency restrictions, major tracker rewrite and nonportable scheduling |

CUDA Graphs should be considered only after an actual GPU benchmark identifies
launch setup as a limiting cost. They preserve explicitly specified dependency
graphs and are not a substitute for correct state ownership. [CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)

Do not allocate a complete run history. The current-body buffers are bounded by
the topology. A chunk backend would add an explicit configured horizon, rather
than multiplying storage by the requested tracking length. Stationary
mode-domain transfer functions and streaming causal filters are compatible
source providers for either scheduling choice.

## Independent particle-map and scheduling timing floor

The probe [fcc_tracking_floor.py](../review/fcc_tracking_floor.py) loads the same
unmodified parameterized FCC input as the scaling review. It does not install
mechanics. It assigns small, fixed numerical shifts/rotations to selected magnet
alignment fields in memory, then times either one multi-turn call or one launch
per turn. Original fields are restored; all tested particles remain active.

This deliberately separates particle-map and scheduler costs from the mechanics
update. It does not simulate motion, assign physical supports or measure CUDA.
The single-particle repeat uses 64 turns and seven repetitions:

| Fixed-field workload | ms/turn | Ratio to unmodified multi-turn baseline |
| --- | ---: | ---: |
| Unmodified, multi-turn | 2.966 | 1.000 |
| Unmodified, launch per turn | 3.041 | 1.025 |
| Quadrupoles, six coordinates, launch per turn | 3.249 | 1.095 |
| All Q/B/S/O magnets, vertical shift, launch per turn | 3.618 | 1.220 |
| All Q/B/S/O magnets, six coordinates, launch per turn | 4.422 | 1.491 |

The [initial probe](../review/results/fcc_freeze/tracking_floor_cpu.json) also
contains 32- and 128-particle workloads. At 32 particles the fixed six-coordinate
all-magnet case was about 1.075 times the unmodified baseline. The initial
single-particle all-magnet launch-per-turn measurement suffered large host
outliers; the longer [repeat](../review/results/fcc_freeze/repeat_1p/tracking_floor_cpu.json)
is retained separately rather than replacing the original evidence. These are
short serial CPU timings on a shared development host, not confidence limits.

The floor is encouraging for a twofold CPU limit. The all-magnet one-particle
case leaves considerably less headroom than quadrupole-only motion. End-to-end
measurements must include the update and distinguish first preparation from
steady tracking. A warmed internal kernel number is not the end-to-end ratio.

The current curved misalignment wrapper performs reference-frame conjugation
and Euler extraction when a bend's alignment is nonzero. On CPU some of this
work is outside the per-particle loop; on GPU it occurs in each particle thread.
GPU map overhead must therefore be measured directly. Compiling the body update
does not by itself prove a sub-twofold GPU result.

The subsequent [end-to-end prepared-runtime results](../review/results/fcc_freeze_runtime/README.md)
use the public API with live lattice expressions retained. That central table
records the final fixed-reference scheduler, quadrupole-only/all-magnet cases,
hierarchy and source choices, cold preparation, update-only and complete tracking
costs. It supersedes intermediate runtime captures made before the clock and
sampled-input changes. The support definitions remain cost probes, not the
engineering girder map.

The limit is not universal across arbitrary call sizes: **measured short-call
cases exceed the twofold target**, even after cold preparation. A cold one-turn
call adds preparation as well. Frequent one-turn calls amortize entry validation
and final ownership bookkeeping differently from a 128-turn call. Results must
state both preparation cost and turns per call. The present timings also do not
establish the twofold limit for GPU, packed bunches, arbitrary source complexity
or a chosen physics-accuracy convergence criterion.

## Kernel implementation and precision

The portable [update header](../../../xtrack/mechanics/runtime_src/update.h)
stores local affine deformations as translation d and rotation increment E=R-I.
It composes finite transforms through `E_left+E_right+E_left*E_right`. Small
angles use `cos(angle)-1=-2*sin(angle/2)^2`; there is no subtraction of large
world origins to recover a dynamic shift. This retains finite rigid composition;
it is not a first-order rotation-vector model.

Absolute child motion retains inherited static placement but replaces inherited
dynamic motion. Relative child motion inherits its parent's full deformation.
Analytic source rows preserve insertion order for noncommuting rotations. A
zero body deformation writes the exact adopted element alignment values,
including its original rotation anchor.

The header implements:

- `mechanics_sources`: deterministic harmonic/plane-wave descriptors,
  evaluated on the tracking context at the freeze time. The phase and projected
  polarization are prepared at nominal body positions.
- `mechanics_body_update`: independent root-to-body composition with no shared
  parent-output race.
- `mechanics_lower`: local lowering and numeric alignment writes on the same
  tracking buffer, preserving source/installation/reference ownership.
- `mechanics_source_values`: one bounded packed set of six-coordinate source
  values, with context-native ordered rotation composition. This supports
  sampled PSD/response records and custom-source compatibility without building
  per-body rotation matrices on the host. Replay interpolation/source sampling
  remains explicit; it is not an implicit complete-run GPU cache.
- `mechanics_publish_clock`: publication of the common reference clock at a
  kernel boundary, with no shared clock write in the particle kernel.

Packed source inputs also need host-buffer ownership across uploads. Xobjects'
CuPy conversion calls `cupy.array`, whose default H2D copy is asynchronous;
mutating reusable host staging before that copy finishes can race. The native
`context_arrays.to_context` helper therefore requests `blocking=True` for CuPy
bulk uploads. Sampled records retain one explicit bounded transfer/synchronization
per freeze; analytic inputs do not have a host payload. A later pinned staging
ring with completion events could overlap copies safely without an unbounded
queue of host arrays. This is a correctness requirement, not a reason to assume
particular pageable-memory copy behaviour. [CuPy array copy ordering](https://docs.cupy.dev/en/stable/reference/generated/cupy.array.html#cupy.array)

The current numeric field-writing backend still rejects shared/repeated element
objects whose occurrences need distinct mechanics. It does not solve this by
silently copying knobs or overwriting deliberate definitions. A future occurrence
buffer backend is the clean solution for such aliases.

## Filling patterns change the required runtime contract

The native particle clock is a shared scalar `Particles.time_s`. The documented
arrival-time convention is

```text
t_arrival = time_s + (s - zeta)/(beta0*c).
```

The installed Xpart multibunch generator constructs bunch slot k by subtracting
`k*bunch_spacing` from its particles' zeta. Thus, for a nominal bunch centre,

```text
tau_b = slot_b * bunch_spacing_zeta/(beta0*c)
t_b(n, s) = t_origin + n*T_rev + tau_b + s/(beta0*c).
```

The positive slot delay corresponds to a negative bunch-centre zeta. A residual
within-bunch longitudinal coordinate contributes `-delta_zeta/(beta0*c)`.
At kHz mechanical frequencies and ordinary mm-scale bunch lengths that final
term is usually negligible, but the bunch-to-bunch delay over a whole fill need
not be. This is a declared approximation, not an additional independent source.

Verified implementation pointers are `xtrack/particles/particles.py`,
`xtrack/_filling_pattern.py`, installed `xpart/matched_gaussian.py` and Xfields'
`slicers/slicers_src/uniform_bin_slicer.h`. Native Particles has no persistent
`bunch_id` field. Xfields derives slots from zeta and the slicer's configured
edges/spacing. Mechanics should reuse the dense `filling_pattern` and sparse
`filled_slots` conventions, rather than invent a conflicting bucket numbering.
The distinction between an RF bucket and a coarser filling slot must remain
explicit through `bunch_spacing_zeta` or a physical `bunch_spacing_s`.

There are three distinct studies:

1. **Common turn-index freeze.** All particles use the field sampled at one
   reference time n*T_rev throughout their indexed turn. The current fast
   numeric-alignment backend implements this. Packed bunches see identical
   alignments, regardless of their zeta or filling slot. This is a useful
   approximation for sufficiently slow motion.
2. **Bunch-reference freeze.** Each bunch uses motion at n*T_rev+tau_b for its
   entire turn. This retains the user's once-per-turn-per-bunch approximation
   while resolving fill phase. Independent sequential bunch runs with an
   explicit common time origin can provide a reference now. One shared mutable
   element-alignment array cannot represent these distinct maps for a packed
   multi-bunch Particles object.
3. **Reference-encounter sampling.** Each bunch/occurrence uses
   n*T_rev+tau_b+s_e/(beta0*c), or an explicit piecewise approximation using extra
   freeze points. This addresses within-turn phase advance. It is not a
   simultaneous mechanical survey, and still does not model individual
   particles' arrival-time deviations unless explicitly requested.

An actual global wall-clock zero-order hold is also not automatically the same
as a common turn-index freeze: late bunch encounters can cross the next global
time interval while their particle still has the previous indexed turn.
The implementation and examples must identify their clock convention.

All bunches must query the same physical realization, floor and response graph.
They must not get independently seeded ground fields. A filling gap removes
encounters and changes controller/collective timing; it does not restart the
mechanical process or reset its random phase. Oppositely directed beams share
world time and physical bodies, with their own surveyed occurrence frames and
arrival-time maps. A separate turn counter for each beam is not a common clock.

The gap is architectural and should be acknowledged now: a bunch-resolved
backend needs the tracking map to read immutable mechanics data indexed by
occurrence and clock/phase, rather than mutate one global set of element fields.
The existing numeric backend remains useful as a single-phase reference and
should not be advertised as supporting phase-resolved packed bunches.

### GPU storage and work for many bunches

Do not allocate a complete pose array for every bunch of an arbitrary fill.
For illustration, **10,000** bunches times 2,663 supports times twelve doubles
is **2.38 GiB for a single current turn**, before magnet fields and reference
geometry. This is a storage example, not an asserted FCC filling specification.
At all 12,041 Q/B/S/O occurrences, seven effective alignment values for 10,000
bunches would occupy about 6.28 GiB. Multiplying that by tracking history is
unacceptable.

A bounded prepared backend has two useful choices:

- **Exact bunch tiles.** Prepare only a configured K bunch phases, then track
  particles belonging to that tile. Geometry remains shared. With K=32, seven
  fields at 12,041 magnet occurrences cost about 20.6 MiB. This is exact for the
  selected bunch-reference/encounter convention, but generating the field
  independently for every bunch remains O(N_bunch*source_work), and many small
  tracking launches can be expensive. Collective feedback and beam-beam/wake
  events constrain which tiles may advance independently.
- **A bounded time/phase grid.** Prepare maps on a fixed physical-time grid
  covering the tile/turn, and let each bunch reference read/interpolate the
  appropriate rows. The number of rows is controlled by bandwidth and accuracy,
  rather than the number of occupied buckets. For a 1 kHz sinusoid over this
  FCC turn, 32 intervals limit the worst half-step scalar linear-interpolation
  amplitude loss to about 0.044%. This is only an interpolation illustration;
  spectra, relative motions, nonlinear rotations and beam observables require
  convergence. Linear interpolation of arbitrary full rotation matrices does
  not preserve orthogonality. Use the already specified rigid coordinates or
  a proper rotation interpolant, with a small-motion regime checked explicitly.

Both choices require a new native tracking consumption path. They cannot be
implemented correctly by overwriting the same element fields before launching
all bunches asynchronously. Candidate inputs are an immutable source provider,
occurrence binding indices, a bounded phase grid and a bunch/clock selection.
The body hierarchy, support definitions, class transfer functions and table UI
do not need to change.

For a stationary spectral provider, temporal phase factors can be shared across
receiver/class projections. Class responses apply to the same latent ground
modes, retaining cross-bunch and cross-class coherence. Repeating a full
floor-mode sum independently in every macroparticle defeats that structure.
No universal low-rank approximation should be assumed for arbitrary measured
motion; provider and interpolation convergence remain explicit.

The fast sampled-record probe is not yet a native long-run streaming provider.
`RigidMotion` is immutable and replacing a source definition invalidates the
prepared model. An advancing record window/filter state must therefore live in
separate run state, so a new chunk does not require recompiling geometry. Avoiding
an implicit GPU history is necessary but does not make a complete high-rate CPU
record affordable. Direct `SpectralGroundMotion` point queries avoid that record,
but currently recompute receiver/mode spatial phases on the host. Neither path
yet establishes a converged million-turn FCC ground-plus-response simulation.

Memory remains a limit even without any run history: two float64 spatial
quadratures for 10,652 receivers and 2,048 modes occupy about **333 MiB for one
scalar field**. Vector components, regional fields and local sources multiply
this. Prepared class/contact projections can reduce the receiver set, but a
dense body-by-mode-by-coordinate cache is not free either. Budget both retained
weights and transient work; tile spatial projections and mode sums where needed.
Compute the shared mode time factors once per freeze, and propagate the same
latent modes through class-dependent `H(f)` before receiver/body projection.
This is a candidate stationary prescribed-LTI provider, not an implemented
extension of the current harmonic/plane-wave source kernel. Measured traces,
transients and live feedback require the separate bounded causal path.

### What per-bunch Twiss can and cannot establish

A Twiss calculation on one simultaneously frozen machine is identical for all
bunches whose optics model is otherwise identical. To expose the fill phase,
one can compute a separate frozen optics/orbit at each bunch's reference time,
or build the reference-encounter map. The latter is a spatially sampled
surrogate for one passage, not one common static floor geometry.

Such Twiss results are useful diagnostics of instantaneous orbit, beta beating,
coupling, feed-down and error sensitivity. They are not a dynamical prediction
of the beam's response to kHz motion. In a linear driven model,

```text
u_(n+1) = M*u_n + b_hat_b(omega)*exp(i*omega*n*T_rev)
u_hat_b = (exp(i*omega*T_rev)*I - M)^(-1)*b_hat_b(omega)
```

For additive kicks at occurrences j,
`b_hat_b = exp(i*omega*tau_b) * sum_j M_(j->end)*k_hat_j*exp(i*omega*tau_j)`.
A static closed-orbit calculation instead uses `(I-M)^(-1)` and need not give
the correct driven amplitude or phase. Without damping, the free `M^n` transient
persists; it must be included when comparing against tracking launched from the
nominal orbit. Motion that changes the linear map also introduces parametric
effects; a periodic Floquet description needs a specified periodic drive.
General PSD motion requires tracking, stochastic response analysis or a
validated equivalent frequency-domain treatment.

For noninteracting identical bunches driven by a single stationary harmonic,
different bunch-reference phases give the same steady-state amplitude with a
phase factor `exp(i*omega*tau_b)`. The pattern selects which phases are occupied.
Different observed orbit values at one reference turn are not, by themselves,
evidence of bunch-dependent optics or different excitation power. Controllers,
collective coupling, transients and multiple source frequencies can make the
complete situation richer.

## GPU acceptance that remains necessary

The source and geometry kernels use portable Xobjects context macros. Context-
parametrized tests compare randomized nested transforms, absolute resets,
ordered finite rotations, picoradian increments and exact zero restoration.
Serial CPU and OpenMP tests passed. Offline Clang checks accepted specialized
OpenCL 1.2/SPIR64 and CUDA device source; the CUDA syntax check supplied the
expected built-in declarations without linking CUDA. Neither check executes a
device or measures NVRTC/register pressure.

A first GPU run must include:

1. Identical native FCC input, radiation configuration and surviving particle
   sets for frozen and moving cases; timing after warmup and final stream
   completion, plus the separate first-call preparation cost.
2. Small and large particle counts, quadrupole-only and whole-magnet motion,
   nonzero pitch/yaw/roll, and the actual PSD/response provider rather than only
   one harmonic. The twofold maximum applies to the stated workload.
3. No per-element scalar host/device transfers in the turn loop. General-source
   fallback transfers, scalar clock accesses, buffer allocations and final
   bookkeeping synchronization must be visible and counted.
4. CPU/GPU particle-coordinate comparisons, deterministic split-call clocks,
   exact deliberate-field preservation and analytic pm/nrad geometry tests.
5. Actual kernel profiling: launch overhead, register spills/local memory,
   random alignment writes and the existing misalignment maps. A kernel with
   several small matrix temporaries can be correct but still have poor occupancy.

The implementation removes the earlier O(N_line) validation and Python matrix
lowering from each turn. This is the correct direction for orders-of-magnitude
update speedup. It is not yet evidence that every GPU workload meets the limit.
