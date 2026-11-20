# Prepared-backend teaching captures

The fifteen teaching scripts have headless captures on the native prepared
backend. Lessons 001–014 retain the preceding checkpoint captures; 015 is the
new portable bunch/Twiss lesson. Per-run source hashes identify each revision.
These headless captures include terminal output, figures and the
quadrupole GIF. The [manifest](manifest.json) records script, native Python,
C-header and tracker hashes.

Worker times include cold mechanics kernel compilation when first needed,
plots and GIF encoding, on a shared development host. They are compatibility
and teaching records, not the clean FCC performance measurements. Read the
[prepared-runtime review](../../../designs/2026-11-20_prepared_runtime_review.md)
for setup/steady timing definitions, physics and remaining limitations.

| Lesson source | Output | Figures | Worker seconds |
| --- | --- | --- | ---: |
| [001](../../../001_rigid_girder.py) | [terminal](001_rigid_girder/terminal.txt) | [figure 1](001_rigid_girder/figure_01.svg) | 0.35 |
| [002](../../../002_sources_and_vibrations.py) | [terminal](002_sources_and_vibrations/terminal.txt) | [figure 1](002_sources_and_vibrations/figure_01.svg) | 1.70 |
| [003](../../../003_shared_girder.py) | [terminal](003_shared_girder/terminal.txt) | [figure 1](003_shared_girder/figure_01.svg) | 0.33 |
| [004](../../../004_demo_ring.py) | [terminal](004_demo_ring/terminal.txt) | [figure 1](004_demo_ring/figure_01.svg), [figure 2](004_demo_ring/figure_02.svg) | 3.24 |
| [005](../../../005_ground_and_feet.py) | [terminal](005_ground_and_feet/terminal.txt) | [figure 1](005_ground_and_feet/figure_01.svg), [figure 2](005_ground_and_feet/figure_02.svg) | 0.66 |
| [006](../../../006_transfer_chain.py) | [terminal](006_transfer_chain/terminal.txt) | [figure 1](006_transfer_chain/figure_01.svg), [figure 2](006_transfer_chain/figure_02.svg) | 1.07 |
| [007](../../../007_local_sources.py) | [terminal](007_local_sources/terminal.txt) | [figure 1](007_local_sources/figure_01.svg) | 0.66 |
| [008](../../../008_regional_ground.py) | [terminal](008_regional_ground/terminal.txt) | [figure 1](008_regional_ground/figure_01.svg), [figure 2](008_regional_ground/figure_02.svg) | 3.81 |
| [009](../../../009_moving_bpm.py) | [terminal](009_moving_bpm/terminal.txt) | [figure 1](009_moving_bpm/figure_01.svg) | 3.32 |
| [010](../../../010_execution_costs.py) | [terminal](010_execution_costs/terminal.txt) | [figure 1](010_execution_costs/figure_01.svg) | 17.86 |
| [011](../../../011_installation_and_correction.py) | [terminal](011_installation_and_correction/terminal.txt) | [figure 1](011_installation_and_correction/figure_01.svg) | 4.63 |
| [012](../../../012_matrix_response.py) | [terminal](012_matrix_response/terminal.txt) | [figure 1](012_matrix_response/figure_01.svg), [figure 2](012_matrix_response/figure_02.svg) | 0.58 |
| [013](../../../013_combined_mechanics.py) | [terminal](013_combined_mechanics/terminal.txt) | [figure 1](013_combined_mechanics/figure_01.svg), [figure 2](013_combined_mechanics/figure_02.svg) | 7.87 |
| [014](../../../014_tables_and_quad_animation.py) | [terminal](014_tables_and_quad_animation/terminal.txt) | [figure 1](014_tables_and_quad_animation/figure_01.svg), [figure 2](014_tables_and_quad_animation/figure_02.svg) | 4.16 |
| [015](../../../015_bunch_phases_and_twiss.py) | [terminal](015_bunch_phases_and_twiss/terminal.txt) | [lesson and figures](015_bunch_phases_and_twiss/README.md) | 14.21 |

[Quadrupole animation](014_tables_and_quad_animation/motion.gif).
