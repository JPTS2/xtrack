"""FCC fill-pattern phases: frozen per-bunch optics versus driven response.

This bounded review experiment tracks bunch centroids independently, sequentially.
The mechanics field is frozen once per turn at each bunch's line-origin arrival.
It does not claim packed multibunch tracking or a dynamic Twiss calculation.
"""
from pathlib import Path
from time import perf_counter
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt


################################################################################
# Prepare the requested FCC line and a small illustrative fill
################################################################################

########################################
# Load the parameterized thick Z model
########################################

ROOT = Path(__file__).resolve().parents[3]
INPUT = ROOT.parent/'inputs/acc-models-fcc-ee-108.1.0/lattices/z/fccee_z_beam_1_positrons.json'
OUTPUT = Path(__file__).resolve().parent/'results/fcc_bunch_sampling'
OUTPUT.mkdir(parents=True,exist_ok=True)
env = xt.load(INPUT)
line = env.fccee_p_ring
line.build_tracker()
line.freeze_longitudinal()
baseline = line.twiss(method='4d')
closed_orbit = baseline.particle_on_co.copy()
matrix = np.asarray(baseline.R_matrix[:4,:4])
period = line.get_length()/(float(line.particle_ref.beta0[0])*299792458.)
table = line.get_table()
cavity = table.rows.match(element_type='Cavity').name[0]
rf_frequency = float(line[cavity].frequency)
harmonic = round(rf_frequency*period)
buckets = np.rint(harmonic*np.array([0.,.08,.16,.60,.68,.76])).astype(int)
arrivals = buckets/rf_frequency
amplitude = 100e-9
phase = np.pi/4
turns = 256
quad = 'qf2r.0'
coordinates = ('x','px','y','py')

########################################
# Define the reference clock and fill
########################################

# Positive arrival delay corresponds to a negative bunch-centre zeta in a
# packed bunch representation. These six reference centroids are separate runs.
# Empty RF buckets supply no observations; they do not alter the common motion.
# Freezing at line-origin arrival intentionally omits subturn passage phase.
metadata = dict(lattice_sha256=hashlib.sha256(INPUT.read_bytes()).hexdigest(),
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    tracker_sha256=hashlib.sha256((ROOT/'xtrack/tracker.py').read_bytes()).hexdigest(),
    mechanics_source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((ROOT/'xtrack/mechanics').rglob('*')) if p.suffix in ('.py','.h')},
    circumference_m=line.get_length(),revolution_period_s=period,rf_frequency_hz=rf_frequency,
    inferred_harmonic=harmonic,rf_turn_closure_error_s=harmonic/rf_frequency-period,
    qx=float(baseline.qx),qy=float(baseline.qy),quad=quad,
    quad_s_m=float(table['s',quad]),amplitude_m=amplitude,phase_rad=phase,
    bucket_ids=buckets.tolist(),arrival_offsets_s=arrivals.tolist(),turns=turns,
    observation=str(table.name[0]),longitudinal_frozen=True,radiation_enabled=False,
    sampling='whole line frozen at n*T_rev + bunch arrival offset',
    initial_condition='the same zero-vibration closed orbit for every bunch',
    engineering_fill=False,packed_multibunch_supported=False)


################################################################################
# Measure the linear drive map and inspect actual frozen per-bunch Twiss
################################################################################

########################################
# Differentiate the one-turn drive map
########################################

body = line.mechanics.new_girder('quad_support',elements=[quad])
probe = 1e-7
outputs = []
for offset in (-probe,probe):
    body.shift_x = offset
    line.mechanics.apply()
    particle = closed_orbit.copy()
    line.track(particle,num_turns=1)
    outputs.append(np.array([float(getattr(particle,name)[0]) for name in coordinates]))
drive = (outputs[1]-outputs[0])/(2*probe)
body.shift_x = 0.
env.mechanics.clear()

########################################
# Keep the finite-frequency response
########################################

# For u[n+1] = M u[n] + b*A*sin(mu*n+phase_b), the complex particular
# solution is (exp(i*mu)I-M)^-1*b*A. Starting at zero orbit perturbation adds
# the undamped free term -M^n Im(particular*exp(i*phase_b)).
results = []
for frequency in (10.,1000.):
    body = line.mechanics.new_girder('quad_support',elements=[quad])
    env.mechanics.add_source('quad_motion',xt.mechanics.Harmonic(amplitude,frequency,phase),
                            targets='quad_support',component='x')
    mu = 2*np.pi*frequency*period
    particular = np.linalg.solve(np.exp(1j*mu)*np.eye(4)-matrix,drive*amplitude)
    frozen_gain = np.linalg.solve(np.eye(4)-matrix,drive)
    frozen = []
    beta_curves = []
    sampled = np.arange(0,len(baseline.s),32)
    for arrival in arrivals:
        line.mechanics.apply(time_s=float(arrival))
        optics = line.twiss(method='4d')
        frozen.append([float(optics[name][0]-baseline[name][0]) for name in coordinates])
        beta_curves.append((optics.betx[sampled]/baseline.betx[sampled]-1).tolist())


    ############################################################################
    # Track the same prescribed motion at the bunch-dependent freeze times
    ############################################################################

    ########################################
    # Retain the free betatron transient
    ########################################

    bunches = []
    for bucket,arrival,frozen_state in zip(buckets,arrivals,frozen):
        particle = closed_orbit.copy()
        line.mechanics.enable_vibrations(time_origin_s=float(arrival))
        started = perf_counter()
        line.track(particle,num_turns=turns,turn_by_turn_monitor=True)
        elapsed = perf_counter()-started
        record = line.record_last_track
        actual = np.column_stack([np.asarray(getattr(record,name))[0]-float(getattr(closed_orbit,name)[0])
                                  for name in coordinates])
        line.mechanics.disable_vibrations()
        phase_b = phase+2*np.pi*frequency*arrival
        n = np.arange(turns)
        steady = np.imag(np.exp(1j*(mu*n+phase_b))[:,None]*particular[None,:])
        homogeneous = np.empty_like(steady)
        homogeneous[0] = -steady[0]
        for index in range(1,turns):
            homogeneous[index] = matrix @ homogeneous[index-1]
        prediction = steady+homogeneous
        snapshots = np.sin(mu*n+phase_b)[:,None]*amplitude*frozen_gain[None,:]
        rms_actual = float(np.sqrt(np.mean(actual[:,0]**2)))
        relative_error = float(np.sqrt(np.mean((actual[:,0]-prediction[:,0])**2))/max(rms_actual,1e-30))
        assert np.all(particle.state>0)
        assert relative_error < .03, (frequency,int(bucket),relative_error)
        bunches.append(dict(bucket=int(bucket),arrival_s=float(arrival),phase_rad=float(phase_b),
            frozen_twiss_state=frozen_state,tracking_s=elapsed,
            actual_x_m=actual[:,0].tolist(),linear_with_transient_x_m=prediction[:,0].tolist(),
            steady_particular_x_m=steady[:,0].tolist(),frozen_snapshot_x_m=snapshots[:,0].tolist(),
            rms_tracking_x_m=rms_actual,relative_linear_prediction_rms_error=relative_error))
    results.append(dict(frequency_hz=frequency,phase_advance_per_turn_rad=float(mu),
        tune_fraction=frequency*period,bunches=bunches,
        beta_s_m=baseline.s[sampled].tolist(),frozen_betx_beating=beta_curves,
        steady_to_static_amplitude_ratio=float(abs(particular[0])/(abs(amplitude*frozen_gain[0])))))
    env.mechanics.clear()


################################################################################
# Plot the fill phases, frozen optics and finite-frequency tracking response
################################################################################

########################################
# Separate snapshots from driven response
########################################

figure,axes = plt.subplots(2,2,figsize=(12,7.5),layout='constrained')
colors = ('tab:blue','tab:orange')
for result,color in zip(results,colors):
    frequency = result['frequency_hz']
    axes[0,0].plot(arrivals*1e6,2*np.pi*frequency*arrivals,'o-',color=color,label=f'{frequency:g} Hz')
    axes[0,1].plot(arrivals*1e6,np.array([b['frozen_twiss_state'][0] for b in result['bunches']])*1e9,
                   'o-',color=color,label=f'{frequency:g} Hz')
axes[0,0].set(xlabel='Bunch arrival offset [µs]',ylabel='Phase relative to first bunch [rad]',
              title='Two illustrative trains; empty buckets retain the common field')
axes[0,1].set(xlabel='Bunch arrival offset [µs]',ylabel='Frozen closed-orbit x [nm]',
              title=f'Actual per-bunch 4D Twiss at {metadata["observation"]}, turn 0')
for axis,result in zip(axes[1],results):
    bunch = result['bunches'][3]
    n = np.arange(turns)
    axis.plot(n,np.asarray(bunch['actual_x_m'])*1e9,lw=1.2,label='Tracked, zero-error CO initial')
    axis.plot(n,np.asarray(bunch['linear_with_transient_x_m'])*1e9,'--',lw=.9,label='Driven map + free transient')
    axis.plot(n,np.asarray(bunch['frozen_snapshot_x_m'])*1e9,color='0.4',alpha=.8,label='Frozen-orbit sequence')
    axis.set(xlabel='Turn',ylabel='Observation x [nm]',
        title=f'{result["frequency_hz"]:g} Hz; bucket {bunch["bucket"]}; amplitude ratio {result["steady_to_static_amplitude_ratio"]:.3f}')
    if result['frequency_hz'] >= 1000.:
        axis.set(xlim=(0,40),xlabel='Turn (first 40 shown)')
for axis in axes.flat:
    axis.legend(fontsize=7)
    axis.grid(alpha=.2)
figure.suptitle('FCC fill phases and driven orbit: per-bunch turn-held approximation')
figure.savefig(OUTPUT/'fill_and_response.png',dpi=150)
figure.savefig(OUTPUT/'fill_and_response.svg')

beta_figure,beta_axis = plt.subplots(figsize=(10,4),layout='constrained')
for bunch,beating in zip(results[1]['bunches'],results[1]['frozen_betx_beating']):
    beta_axis.plot(np.asarray(results[1]['beta_s_m'])/1000,np.asarray(beating)*1e6,
                   label=f'Bucket {bunch["bucket"]}')
beta_axis.set(xlabel='s [km]',ylabel='Frozen Δβx / βx [ppm]',
              title='1 kHz: actual frozen per-bunch Twiss; this is not a dynamic response function')
beta_axis.legend(ncol=3,fontsize=8)
beta_axis.grid(alpha=.2)
beta_figure.savefig(OUTPUT/'frozen_beta_beating.png',dpi=150)
beta_figure.savefig(OUTPUT/'frozen_beta_beating.svg')

for path in OUTPUT.glob('*.svg'):
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
payload = dict(metadata=metadata,baseline_one_turn_matrix=matrix.tolist(),drive_jacobian=drive.tolist(),results=results)
(OUTPUT/'bunch_sampling.json').write_text(json.dumps(payload,indent=2)+'\n')
print(json.dumps(dict(metadata=metadata,summary=[dict(frequency_hz=r['frequency_hz'],
    steady_to_static_amplitude_ratio=r['steady_to_static_amplitude_ratio'],
    max_relative_linear_prediction_rms_error=max(b['relative_linear_prediction_rms_error'] for b in r['bunches']))
    for r in results]),indent=2))
