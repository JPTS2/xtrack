"""Inspect the requested FCC input and catalogue-style candidate associations.

The output is NOT an engineering girder map. Device-name links and a 3 m
nearest-quadrupole rule reproduce the catalogue's attribution convention only.
No original model or catalogue file is modified.
"""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import xobjects as xo
import xtrack as xt


ROOT = Path(__file__).resolve().parents[3]
INPUT = ROOT.parent/'inputs/acc-models-fcc-ee-108.1.0/lattices/z/fccee_z_beam_1_positrons.json'
OUTPUT = Path(__file__).resolve().parent/'results/fcc_scaling'

env = xt.load(INPUT)
line = env.fccee_p_ring
line.build_tracker()
table = line.get_table()
quads = table.rows.match(element_type='Quadrupole')
quad_names = set(quads.name)
quad_s = np.asarray(quads.s)
circumference = line.get_length()
indices = {name:i for i,name in enumerate(quads.name)}

buffers = {id(element._buffer):element._buffer for element in env.element_dict.values()
           if hasattr(element,'_buffer')}
expression_counts = {}
for element_type in ('Quadrupole','RBend','Sextupole'):
    names = table.rows.match(element_type=element_type).name
    expression_counts[element_type] = {
        field:sum(getattr(getattr(env.ref[name],field),'_expr',None) is not None
                  for name in names if hasattr(env[name],field))
        for field in ('length','angle','rot_s_rad','shift_x','shift_y')}


def distance(a,b):
    difference = abs(float(a)-float(b)) % circumference
    return min(difference,circumference-difference)


rows = []
for name,element_type,s in zip(table.name[:-1],table.element_type[:-1],table.s[:-1]):
    quad = ''
    rule = 'unassigned'
    if element_type == 'Quadrupole':
        quad,rule = name,'quadrupole anchor candidate'
    elif name.startswith(('bpm_','hcor_','vcor_')):
        candidate = name.split('_',1)[1]
        if candidate in quad_names:
            quad,rule = candidate,'catalogue name association'
    elif element_type == 'Sextupole':
        index = np.searchsorted(quad_s,s)
        candidates = [int(index%len(quad_s)),int((index-1)%len(quad_s))]
        nearest = min(candidates,key=lambda i:distance(s,quad_s[i]))
        if distance(s,quad_s[nearest]) <= 3.+1e-6:
            quad,rule = quads.name[nearest],'catalogue nearest quad within 3 m'
    if element_type in ('Quadrupole','RBend','Sextupole','Octupole') or name.startswith(('bpm_','hcor_','vcor_')):
        rows.append(dict(element=name,element_type=element_type,s_m=float(s),
            associated_quad=quad,rule=rule,
            distance_s_m='' if not quad else distance(s,quad_s[indices[quad]]),
            engineering_girder='',support_class=''))

OUTPUT.mkdir(parents=True,exist_ok=True)
with (OUTPUT/'candidate_associations.csv').open('w') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
metadata=dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    lattice_sha256=hashlib.sha256(INPUT.read_bytes()).hexdigest(),
    xobjects_version=xo.__version__,xtrack_version=xt.__version__,
    n_buffers=len(buffers),buffer_capacity_bytes=sum(int(b.capacity) for b in buffers.values()),
    active_tracker_buffer_capacity_bytes=int(line._buffer.capacity),
    expression_counts=expression_counts,association_counts=dict(Counter(r['rule'] for r in rows)),
    n_association_rows=len(rows),distance_metric='periodic longitudinal s at element entrances',
    same_girder_max_distance_m=3.,engineering_assignment=False,
    catalogue_generator_url='https://github.com/skostogl/fcc-ee-magnet-catalogue/blob/main/000_create_catalogues.py',
    catalogue_generator_blob_sha='d1fd495e8b55145f8930e815a9aeb3f5d33069bd',
    catalogue_documented_model='LCC_107.0.0')
(OUTPUT/'inventory.json').write_text(json.dumps(metadata,indent=2)+'\n')
print(json.dumps(metadata,indent=2))
