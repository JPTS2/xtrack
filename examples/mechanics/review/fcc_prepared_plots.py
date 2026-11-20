"""Plot the recorded prepared-runtime limits without running any tracking."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


base = Path(__file__).resolve().parent/'results/fcc_freeze_runtime'
keys = ['quads_cpu', 'all_magnets_cpu', 'all_magnets_cpu_nested',
        'quads_cpu_sampled_ground', 'quads_cpu_direct_ground']
records = [json.loads((base/(key+'.json')).read_text()) for key in keys]
labels = ['2,663 quads\nanalytic', '12,041 magnets\nanalytic', '16,055 bodies\ntwo levels',
          'Ground + classes\nsampled record', 'Direct ground\nanchor shortcut']
fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout='constrained')
fig.suptitle('FCC Z thick lattice: measured serial CPU tracking slowdown', fontsize=14)
x = np.arange(len(records))
for count, offset, color in [('1', -.18, '#46708d'), ('32', .18, '#719b8c')]:
    values = [record['slowdown'].get(count, np.nan) for record in records]
    bars = axes[0].bar(x+offset, values, width=.34, color=color, label=f'{count} particle'+('s' if count != '1' else ''))
    for bar, value in zip(bars, values):
        if np.isfinite(value):
            axes[0].text(bar.get_x()+bar.get_width()/2, value+.035, f'{value:.2f}×', ha='center', fontsize=9)
axes[0].set_xticks(x, labels, fontsize=9)
axes[0].set_title('128-turn calls; preparation reported separately')
axes[0].legend(loc='upper left')
for index, color in [(0, '#46708d'), (1, '#b86d56')]:
    record = records[index]
    values = [record['short_calls'][str(turns)]['slowdown'] for turns in (1, 4, 16)]
    values.append(record['slowdown']['1'])
    axes[1].plot([1, 4, 16, 128], values, 'o-', color=color,
                 label='Quad sites' if index == 0 else 'All magnets')
axes[1].set_xscale('log', base=2)
axes[1].set_xticks([1, 4, 16, 128], ['1', '4', '16', '128'])
axes[1].set_xlabel('Turns per call, one particle')
axes[1].set_title('Warm short calls still exceed the ceiling')
axes[1].legend(loc='upper right')
for axis in axes:
    axis.axhline(2., color='#b54436', linestyle='--', label='2× ceiling')
    axis.set_ylabel('Moving / unmodified tracking time')
    axis.grid(axis='y', alpha=.2)
    axis.set_axisbelow(True)
    axis.set_ylim(bottom=0)
fig.supxlabel('Provisional support assignments; no GPU timing or packed-bunch claim. Source accuracy is not converged.', fontsize=10)
fig.savefig(base/'prepared_costs.png', dpi=150)
path = base/'prepared_costs.svg'
fig.savefig(path)
path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
