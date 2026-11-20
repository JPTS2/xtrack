"""Public parameterized FCC attachment and independent local precision probe."""
from pathlib import Path
from time import perf_counter
import hashlib
import json

import numpy as np
import xtrack as xt


ROOT = Path(__file__).resolve().parents[3]
INPUT = ROOT.parent/'inputs/acc-models-fcc-ee-108.1.0/lattices/z/fccee_z_beam_1_positrons.json'
OUTPUT = Path(__file__).resolve().parent/'results/fcc_freeze_runtime'

started = perf_counter()
env = xt.load(INPUT)
line = env.fccee_p_ring
quads = tuple(line.get_table().rows.match(element_type='Quadrupole').name)
loaded_s = perf_counter()-started
started = perf_counter()
line.mechanics.new_girders({f'g.{name}':[name] for name in quads},
                          support_class='provisional_quad_site')
attached_s = perf_counter()-started
targets = tuple(f'g.{name}' for name in quads)
amplitudes = (0., 1e-9, 1e-11, 1e-13, 1e-15)
coordinates = np.zeros((len(amplitudes), len(quads), 6))
coordinates[..., 0] = np.asarray(amplitudes)[:, None]
env.mechanics.add_source('precision', xt.mechanics.RigidMotion(
    np.arange(len(amplitudes), dtype=float), coordinates, target_names=targets), targets=targets)
records = []
for index, amplitude in enumerate(amplitudes):
    env.mechanics.apply(time_s=float(index))
    actual = np.asarray([line[name].shift_x-line.mechanics._attachments[name]['baseline'][0]
                         for name in quads])
    difference = actual-amplitude
    records.append(dict(command_m=amplitude,max_error_m=float(np.max(np.abs(difference))),
                        rms_error_m=float(np.sqrt(np.mean(difference**2))),
                        min_realised_m=float(actual.min()),max_realised_m=float(actual.max())))
expressions_preserved = sum(env.ref[name].length._expr is not None for name in quads)
selected = line.mechanics.get_table().rows[quads[len(quads)//2]]
started = perf_counter()
history = selected.get_time_series(time_s=np.linspace(0.,4.,33))
selected_history_s = perf_counter()-started
assert expressions_preserved == 2544
assert records[0]['max_error_m'] == 0.
assert records[2]['max_error_m'] < 1e-22
result = dict(lattice_sha256=hashlib.sha256(INPUT.read_bytes()).hexdigest(),
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    mechanics_source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted((ROOT/'xtrack/mechanics').glob('*.py'))},
    n_quad_sites=len(quads),engineering_assignment=False,
    load_s=loaded_s,attachment_s=attached_s,
    quadrupole_length_expressions_preserved=expressions_preserved,precision=records,
    selected_element_history_samples=len(history),selected_element_history_s=selected_history_s)
OUTPUT.mkdir(parents=True,exist_ok=True)
(OUTPUT/'precision.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
