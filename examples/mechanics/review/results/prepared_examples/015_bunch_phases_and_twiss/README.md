# 015: bunch phases, actual frozen Twiss and driven response

[Teaching source](../../../../015_bunch_phases_and_twiss.py),
[terminal output](terminal.txt), [phase/orbit figure](figure_01.svg),
[frozen beta figure](figure_02.svg).

The default is runnable from this checkout: the shared 1.5 km demonstration ring,
not an external FCC download. Editable user parameters specify the lattice,
quad, source frequencies, filled RF buckets and initial beam state. The script
uses the established 80-character main headings and 40-character teaching steps.

The frequencies default to 0.003 and 0.30 times the revolution frequency: about
599.585 Hz and 59,958.5 Hz in this compact ring. The high case is a clock/response
diagnostic, not a proposed ground-noise spectrum. Set the supplied FCC lattice
parameters and `FREQUENCIES_HZ=(10.,1000.)` for the corresponding FCC study.

Six bucket centroids form two illustrative trains with empty gaps. Each bunch
is an independent sequential run with an explicit physical arrival offset;
all sample the same prescribed harmonic. Gaps remove observations without
resetting the source. This is not packed multibunch tracking. The whole-line
turn-held approximation omits element passage and within-bunch time differences.

The first figure separates source phase, actual frozen 4D Twiss closed orbit,
and tracked beam response. The low-frequency driven/static amplitude ratio is
about 1.00018; the faster case is about 2.87167 for this lattice/observation.
The driven-map prediction and full tracking overlap closely. An instantaneous
closed-orbit sequence does not reproduce the high-frequency response.

The default initial beam state is the linear driven particular solution. This
makes the slow-drive agreement and faster-drive difference easy to see. Select
`INITIAL_CONDITION='zero_motion_closed_orbit'` to retain the undamped free
betatron transient from an abrupt start; the prediction accounts for it. These
models have longitudinal motion frozen, radiation off and no collective effects
or feedback. No damping/adiabatic startup is silently assumed.

The second figure recomputes actual beta functions for every selected bunch
phase. The demo explicitly enables weak integrated sextupoles of +0.1/−0.1 m^-2;
changing orbit then changes frozen optics through feed-down. A selected phase's
maximum beta beating is about 0.8864 ppm.
The [numerical step check](twiss_step_check.json) repeats the zero result exactly
and changes `steps_R_matrix` by 0.5/1/2. The maximum pointwise change is
0.000851 ppm.
W-propagation steps were held fixed; this is a focused numerical check.

After `%run`, the lesson leaves `bunch_table`, `high_frequency_bunches`,
`results`, `mech` and `body` available. `bunch_table` is an ordinary Xsuite table
containing bucket, frequency, arrival time, phase, orbit, beta functions and tunes.
Normal column/row selection works. Automatic vibrations are disabled; the last
source remains defined and the actual line is applied at the representative
bunch phase. For another frozen phase, apply its arrival time and call ordinary
`line.twiss(method='4d')`.

The capture took 14.21 s on the local CPU, including cold Twiss/
mechanics kernels and figure capture. Individual timings are in the terminal
and [metadata](result.json); they are not a throughput benchmark.

## Verified FCC parameter variant

The same teaching source was also run with only these user parameters changed:

```python
LATTICE_FILE = '../inputs/acc-models-fcc-ee-108.1.0/lattices/z/fccee_z_beam_1_positrons.json'
LINE_NAME = 'fccee_p_ring'
QUAD_NAME = 'qf2r.0'
FREQUENCIES_HZ = (10., 1000.)
```

[FCC terminal](fcc_variant/terminal.txt),
[phase/orbit figure](fcc_variant/figure_01.svg),
[beta figure](fcc_variant/figure_02.svg), [metadata](fcc_variant/result.json).
The external input is the pinned development model in the
[original input record](../../fcc_scaling/README.md), not copied into the example.

At 10 Hz and 1 kHz the particular/static amplitude ratios are 1.000757 and
1.596837, respectively. Maximum relative tracking/prediction RMS errors are
2.016e-06 and 3.204e-06. The initial-state choice differs
from the earlier abrupt-start review capture, whose free transient remains
useful reference evidence.
