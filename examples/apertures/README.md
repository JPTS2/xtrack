<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# Apertures

Start with local particle cuts, then build chamber geometry, place it in the
survey, install tracking checks and evaluate beam clearance. Each script runs
independently. Chamber geometry and particle tracking limits are distinct objects.

| Example | Lesson |
| --- | --- |
| [001_tracking_apertures.py](001_tracking_apertures.py) | Apply native tracking limits and compare which particles survive |
| [002_profiles_and_pipes.py](002_profiles_and_pipes.py) | Define analytic or SVG profiles, then assemble pipes from their longitudinal positions |
| [003_place_and_transform_pipes.py](003_place_and_transform_pipes.py) | Anchor pipes to a line and inspect offsets and rolls in survey frames |
| [004_interpolate_profiles_on_curves.py](004_interpolate_profiles_on_curves.py) | Follow a bend with a curved pipe and interpolate its changing cross-sections |
| [005_install_tracking_limits.py](005_install_tracking_limits.py) | Sample chamber sections into native tracking elements and inspect discrete loss locations |
| [006_beam_clearance.py](006_beam_clearance.py) | Calculate native envelope clearance and directional margins; display the beam and chamber together |
| [007_save_and_reload_apertures.py](007_save_and_reload_apertures.py) | Save a complete line with its chambers or transfer aperture data separately |
| [008_refine_loss_locations.py](008_refine_loss_locations.py) | Refine coarse losses under an explicit interpolated-wall assumption |

This route draws on [Szymon Łopaciuk's aperture-model tutorial](https://github.com/xsuite/aperture-model-tutorial),
particularly its reusable pipe definitions, per-line placements, native geometry
inspection and clearance plots. The notebooks provide a fuller machine workflow,
including transitions and loss maps.

Profiles and pipes belong to the environment; their placements belong to each
line. A transverse cut is expressed in its local survey plane. Its pose maps
that plane into global coordinates. Pipe placement transforms the chamber,
without moving magnets or changing the reference trajectory. For frame
conventions, see [geometry and survey](../geometry_and_survey/README.md).

`001` uses `LimitRect`, `LimitEllipse` and `LimitPolygon`: these are tracking
beam elements and can immediately mark particles as lost. `002` instead defines
reusable chamber profiles. Circle, Ellipse and SVGShape are alternative ways to
describe a transverse boundary. A pipe combines profile positions along its
length. `005` connects the two representations by turning chamber cross-sections
into tracking elements.

The SVG profile in `002` has a circular body and unequal horizontal winglets:
the larger is outward (+x for the shared ring's positive bends). SVG coordinates
are in millimetres, converted by `scale`; `resolution` is an intermediate
sampling distance in metres. The final point count controls polygon resampling.
The SVG y-axis is inverted into physical y-up coordinates. These are
illustrative chamber dimensions, without photon transport or absorber modelling.

`003` uses the maintained short transport line. `004`, `006` and `007` use the
shared [demonstration ring](../../test_data/example_lattices/demo_ring/README.md).
The small straight lines in `005` and `008` isolate sampling and refinement.
RF, radiation and material interactions are absent.

An aperture model alone does not remove particles. `005` explicitly installs
`LimitPolygon` snapshots at selected stations. Losses are recorded at those
checks, so a particle may have crossed the assumed wall earlier. `008` uses
native loss refinement to estimate that intersection; it does not reconstruct
an unknown chamber. Its CPU calculation has finite longitudinal, radial and
polygon resolutions.

`006` evaluates a sampled geometric envelope scale, `n1`. Its `delta_rms`
parameter contributes a bounded `±D*delta_rms` momentum excursion to the envelope;
it is not added in quadrature as a Gaussian RMS size. The result is neither
nonlinear acceptance nor beam transmission. For nonlinear tracking acceptance,
see [dynamic aperture](../tracking/nonlinear/002_dynamic_aperture.py). For material
interactions and collimation studies, see [Xcoll examples](https://github.com/xsuite/xcoll/tree/main/examples).

Advanced legacy geometry and machine applications remain available in
[aperture_model](../aperture_model/) and
[loss-refinement applications](../collimation/). These include overlapping and
ring-wrapped geometry, transformed refinement and machine-specific data models;
the introductory sequence above does not replace all of those lessons.
