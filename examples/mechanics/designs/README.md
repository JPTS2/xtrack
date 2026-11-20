# Design records

[Current implementation decisions](../DESIGN.md) describe the native example set.
[The workflow review](2026-11-19_workflow_review.md) preserves the preceding
FCC-ee requirements and proposed development sequence; it predates implementation
of the thirteen lessons and its capability table is historical.

[The FCC scaling review](2026-11-20_fcc_scaling_review.md) tests the user-selected
development Z lattice and catalogue associations. It records public-workflow
failures, isolated scaling experiments, GPU implications and the changes needed
before full FCC production studies.

[The prepared-runtime review](2026-11-20_prepared_runtime_review.md) records the
next implementation and the twofold performance ceiling. Its supporting notes
cover [bulk geometry](2026-11-20_bulk_geometry.md),
[precision and clock compensation](2026-11-20_mechanics_precision.md), and
[GPU freeze scheduling and bunch phases](2026-11-20_gpu_freeze_runtime.md).

[First-demo PR readiness](2026-11-20_demo_pr_readiness.md) records the subsequent
approach decision, focused integration hardening, the new portable bunch lesson
and a concrete scope/remaining-work estimate for an experimental draft.

Keep alternatives, physical contracts and decisions here as the local branch
evolves. The executable review surface is the numbered examples, with captured
terminal output and figures in [review](../review/README.md). Clean commit slicing
and publication remain later work.
