"""Small numerical-step check of the FCC per-bunch frozen-beta illustration."""
from pathlib import Path
import hashlib
import json

import numpy as np
import xtrack as xt
from xtrack.twiss import DEFAULT_STEPS_R_MATRIX


ROOT = Path(__file__).resolve().parents[3]
INPUT = ROOT.parent/'inputs/acc-models-fcc-ee-108.1.0/lattices/z/fccee_z_beam_1_positrons.json'
OUTPUT = Path(__file__).resolve().parent/'results/fcc_bunch_sampling'
env = xt.load(INPUT)
line = env.fccee_p_ring
line.build_tracker()
line.freeze_longitudinal()
body = line.mechanics.new_girder('g',elements=['qf2r.0'])
reference = line.twiss(method='4d')
repeat = line.twiss(method='4d')
period = line.get_length()/(float(line.particle_ref.beta0[0])*299792458.)
rf_frequency = float(line['rf600'].frequency)
bucket = round(round(rf_frequency*period)*.6)
arrival = bucket/rf_frequency
offset = 100e-9*np.sin(np.pi/4+2*np.pi*1000.*arrival)
records = []
curves = []
sampled = np.arange(0,len(reference.s),32)
for factor in (.5,1.,2.):
    steps = {name:value*factor for name,value in DEFAULT_STEPS_R_MATRIX.items()}
    body.shift_x = 0.
    line.mechanics.apply()
    zero = line.twiss(method='4d',steps_R_matrix=steps)
    body.shift_x = offset
    line.mechanics.apply()
    displaced = line.twiss(method='4d',steps_R_matrix=steps)
    beating = displaced.betx/zero.betx-1.
    curves.append(beating)
    records.append(dict(step_factor=factor,
        max_zero_beta_change_from_default=float(np.max(np.abs(zero.betx/reference.betx-1.))),
        max_beta_beating=float(np.max(np.abs(beating))),
        observation_co_x_m=float(displaced.x[0]-zero.x[0]),
        tune_shift_x=float(displaced.qx-zero.qx),
        sampled_beta_beating=beating[sampled].tolist()))
for record,curve in zip(records,curves):
    record['max_beating_difference_from_default'] = float(np.max(np.abs(curve-curves[1])))
result = dict(lattice_sha256=hashlib.sha256(INPUT.read_bytes()).hexdigest(),
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    tracker_sha256=hashlib.sha256((ROOT/'xtrack/tracker.py').read_bytes()).hexdigest(),
    default_steps_R_matrix=DEFAULT_STEPS_R_MATRIX,
    default_W_propagation_step_unchanged=True,
    quad='qf2r.0',horizontal_offset_m=float(offset),phase_bunch_bucket=bucket,
    zero_repeat_max_beta_change=float(np.max(np.abs(repeat.betx/reference.betx-1.))),
    records=records,sampled_s_m=reference.s[sampled].tolist())
OUTPUT.mkdir(parents=True,exist_ok=True)
(OUTPUT/'twiss_convergence.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({key:value for key,value in result.items() if key!='sampled_s_m'
    and key!='records'},indent=2))
for record in records:
    print({key:value for key,value in record.items() if key!='sampled_beta_beating'})
