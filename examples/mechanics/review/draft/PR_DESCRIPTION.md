## Experimental rigid-body mechanics and prescribed motion

Existing thick elements can be attached to surveyed rigid supports through
`env.mechanics` and `line.mechanics`. Deliberate element offsets, static
installation errors and dynamic motion remain separate. Named sources supply
harmonics, plane waves, measured records or seeded ground-motion models;
explicit feet/response paths compose support, magnet and pickup motion.

The examples are the main review interface. Start with
`001_rigid_girder.py` for reference/installation/dynamic placement and cleanup,
`004_demo_ring.py` for a surveyed ring, and `015_bunch_phases_and_twiss.py` for
fill buckets, frozen per-bunch Twiss and driven tracking. Other short lessons
cover floor contacts, transfer responses, local sources, correction seeds,
ordinary table selection and magnified rigid animation.
`016_fcc_ground_motion.py` runs the actual requested FCC development lattice:
2,663 provisional quad supports, common floor/feet, three girder response
classes, magnet transfer and a selected virtual pickup, with complete tracking
costs against nominal and frozen installed inputs.

Tracking optionally freezes prescribed motion once per turn. Prepared kernels
compose rigid transforms and update effective numeric alignment on the tracking
context; sampled/custom inputs use a bounded host batch. Ordinary tracking and
mechanics share Tracker kernel/buffer binding. A Python reference path remains
available. No whole-run device pose history or scalar xdeps mechanics hierarchy
is installed.

This is an experimental UI draft. The current backend guards slices/shared
element objects, live alignment expressions, stale survey geometry, built
aperture application and unsupported tracking schedules. Bunch comparisons use
independent runs with explicit time origins. Packed bunch maps, bounded long-run
source streaming, engineering FCC support assignments and online moving-BPM /
feedback coupling are subsequent work. Coordinate, clock and lifecycle
contracts are in `examples/mechanics/API.md`.

Validation and its comparison basis are recorded in
`examples/mechanics/DELIVERY.md`. CPU/OpenMP execution is tested; actual GPU
validation remains pending hardware access. The explicit hardware acceptance
command records unavailable devices without falling back to CPU. FCC timings
separate warm tracking from source preparation, cold setup and short-call cost.

Related: xsuite/xsuite#956. This is the first demonstrator slice rather than
completion of every machine-mechanics and feedback requirement in that issue.
