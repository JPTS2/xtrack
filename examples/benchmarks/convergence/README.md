<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# Numerical convergence

Refine one numerical choice while keeping the physical problem fixed. Each
study states its reference, tests its refinement and separates integration error
from model approximations or measurement resolution. Plotting uses Matplotlib.

| Example | Lesson |
| --- | --- |
| [001_magnet_integrators.py](001_magnet_integrators.py) | Compare integration orders with a pure-bend control and a nonlinear combined-field magnet |
| [002_accuracy_vs_runtime.py](002_accuracy_vs_runtime.py) | Compare warmed tracking costs at stated scaled errors, including an approximation floor |
| [003_field_integration.py](003_field_integration.py) | Separate RK4 step refinement from transverse field-expansion order |
| [004_field_segmentation.py](004_field_segmentation.py) | Separate longitudinal cubic fitting, off-axis field reconstruction and canonical/kinetic interfaces |
| [005_radiation_slicing.py](005_radiation_slicing.py) | Refine native mean-radiation kicks and inspect both energy loss and trajectory convergence |
| [006_ring_optics.py](006_ring_optics.py) | Propagate integration differences into tunes, beta functions and finite-difference chromaticities |
| [007_field_accuracy_vs_runtime.py](007_field_accuracy_vs_runtime.py) | Compare native RK4 and spline-Boris transport at matched exit accuracy in the same field |

Errors involving positions and momenta use explicit coordinate scales. Relative
error is not formed by dividing by a coordinate that may vanish. More slices can
eventually expose roundoff, and different body-map approximations need not
converge to the same model. Order estimates apply to the resolved range shown.

Field examples distinguish kinetic and canonical momenta at interfaces.
`BFieldExpansion` uses classical RK4: preserving canonical momentum at interfaces
does not make its finite-step map exactly symplectic. Convergence of an on-axis
fit also does not guarantee off-axis field convergence. These studies do not
establish long-term stability or a general symplecticity audit.

The matched-field study uses an independent SciPy DOP853 integration of the
stated analytical field. Its fixed-energy proton probes, boundary-momentum
convention and coordinate scales define the accuracy comparison. It measures
this single-pass problem, not a universal ranking of the integrators.

For element usage, see [magnets](../../elements/magnets/README.md); for the
machine application, see [ELENA](../../machine_studies/elena/README.md).
Mean-radiation assumptions are introduced in [radiation](../../radiation/README.md).
