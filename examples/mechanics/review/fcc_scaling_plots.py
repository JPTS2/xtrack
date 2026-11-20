"""Plot measured FCC reference costs; no tracking or model loading."""
import json
from pathlib import Path

import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
RESULTS = HERE/'results/fcc_scaling'
native = json.loads((RESULTS/'native.json').read_text())
bulk = json.loads((RESULTS/'bulk.json').read_text())
field = json.loads((RESULTS/'field.json').read_text())
n, b, f = (record['results'] for record in (native,bulk,field))

fig, axes = plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
fig.suptitle('FCC Z thick lattice — measured serial CPU reference costs',fontsize=13,y=.99)
names = ['Frozen tracking: 1 particle', 'Update: 6 supported quads',
         'Pose composition: 2,663 supports', 'Validation: 2,663 supports',
         'Transform lowering: 2,663 supports', 'Scalar writes: 2,663 supports',
         'Complete update: 2,663 supports']
values = [b['frozen_tracking'][0]['median_s']/4,n['apply']['median_s'],b['pose']['median_s'],
          b['line_validation']['median_s'],b['transform_lowering_without_validation']['median_s'],
          b['scalar_field_writes']['median_s'],b['apply']['median_s']]
colors = ['#536b83','#d29043','#648f83','#c26852','#648f83','#648f83','#b65441']
axes[0].barh(names,[1000*v for v in values],color=colors,height=.65)
axes[0].invert_yaxis()
axes[0].set_xscale('log')
axes[0].set(xlabel='Per turn / update [ms, logarithmic]',xlim=(1,700),
            title='Full-lattice checks dominate the current update')
for i,value in enumerate(values):
    axes[0].text(value*1000*1.07,i,f'{value*1000:.2f}',va='center',fontsize=8)
axes[0].grid(axis='x',alpha=.2)
axes[0].set_axisbelow(True)
axes[0].title.set_size(10)
axes[0].tick_params(axis='y',labelsize=8)

names = ['Ground random-mode evaluation', 'Rigid fit at four feet / support', 'Three conditional response classes']
values = [f['direct_source_evaluation']['median_s'],f['rigid_contact_fit']['median_s'],
          sum(v['median_s'] for v in f['class_responses'].values())]
axes[1].barh(names,[v*1000 for v in values],color=['#648f83','#c26852','#536b83'],height=.55)
axes[1].invert_yaxis()
axes[1].set_xscale('log')
axes[1].set(xlabel='128-sample preparation [ms, logarithmic]',xlim=(5,1200),
            title='10,652 feet; 256 random modes; 128 Hz sampling')
for i,value in enumerate(values):
    axes[1].text(value*1000*1.07,i,f'{value*1000:.2f}',va='center',fontsize=8)
axes[1].grid(axis='x',alpha=.2)
axes[1].set_axisbelow(True)
axes[1].title.set_size(10)
axes[1].tick_params(axis='y',labelsize=8)
fig.get_layout_engine().set(rect=(0,.13,1,.82))
fig.text(.5,.015,'2,663-support results use an experimental bulk binding on a detached numeric snapshot. '
         'Field evaluation bypasses a public allocation-guard defect.\n'
         'These are scaling probes, not validated engineering assignments, GPU measurements or a converged FCC noise model.',
         ha='center',fontsize=8)
fig.savefig(RESULTS/'costs.png',dpi=140)
p=RESULTS/'costs.svg'
fig.savefig(p)
p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
plt.close(fig)
