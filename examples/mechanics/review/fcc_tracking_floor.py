"""Separate scheduler and particle-map costs from mechanics preparation.

This is a timing probe, not a mechanics implementation. It assigns fixed small
numeric alignment fields on the unmodified FCC input in memory. No source input
file is changed, and no mechanics model is installed. All tested particles must
survive. Use serial CPU or a real CUDA/OpenCL context selected explicitly.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
import xobjects as xo
import xtrack as xt


ROOT = Path(__file__).resolve().parents[3]
INPUT = ROOT.parent/'inputs/acc-models-fcc-ee-108.1.0/lattices/z/fccee_z_beam_1_positrons.json'
OUTPUT = Path(__file__).resolve().parent/'results/fcc_freeze'
FIELDS = ('shift_x', 'shift_y', 'shift_s', 'rot_x_rad', 'rot_y_rad',
          'rot_s_rad_no_frame', 'rot_shift_anchor')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context', choices=('cpu', 'cuda', 'opencl'), default='cpu')
    parser.add_argument('--turns', type=int, default=12)
    parser.add_argument('--particles', type=int, nargs='+', default=[1, 32, 256])
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    context = {'cpu': xo.ContextCpu, 'cuda': xo.ContextCupy,
               'opencl': xo.ContextPyopencl}[args.context]()
    env = xt.load(INPUT)
    line = env.fccee_p_ring
    line.build_tracker(_context=context)
    table = line.get_table()
    quads = list(table.rows.match(element_type='Quadrupole').name)
    magnet_names = list(table.rows.match(
        element_type='Quadrupole|RBend|Sextupole|Octupole').name)
    names = list(dict.fromkeys(magnet_names))
    baseline = {name: np.array([float(getattr(line[name], field)) for field in FIELDS])
                for name in names}
    line.track(line.build_particles(), num_turns=1)
    results = dict(metadata=dict(
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        lattice_sha256=hashlib.sha256(INPUT.read_bytes()).hexdigest(),
        xtrack_version=xt.__version__, xobjects_version=xo.__version__,
        platform=platform.platform(), python=sys.version,
        context=args.context, turns=args.turns, repeats=args.repeats,
        n_quadrupole_occurrences=len(quads),
        n_magnet_occurrences=len(magnet_names),
        meaning='Fixed-field particle-map and launch-per-turn timing floor; no mechanics update'),
        measurements=[])
    def restore():
        for name, values in baseline.items():
            for field, value in zip(FIELDS, values):
                setattr(line[name], field, value)
    cases = ('unmodified', 'quad_vertical', 'quad_six_coordinates',
             'all_magnets_vertical', 'all_magnets_six_coordinates')
    try:
        for case in cases:
            restore()
            selected = quads if case.startswith('quad_') else names
            if case != 'unmodified':
                for name in selected:
                    line[name].shift_y = baseline[name][1]+1e-9
                    if case.endswith('six_coordinates'):
                        line[name].shift_x = baseline[name][0]+2e-10
                        line[name].shift_s = baseline[name][2]+1e-10
                        line[name].rot_x_rad = baseline[name][3]+1e-10
                        line[name].rot_y_rad = baseline[name][4]+2e-10
                        line[name].rot_s_rad_no_frame = baseline[name][5]+3e-10
            for count in args.particles:
                for scheduler in ('multi_turn', 'launch_per_turn'):
                    durations = []
                    active = []
                    for _ in range(args.repeats):
                        particles = line.build_particles(
                            x=np.full(count, 1e-7), y=np.full(count, 1e-8))
                        context.synchronize()
                        started = perf_counter()
                        if scheduler == 'multi_turn':
                            line.track(particles, num_turns=args.turns)
                        else:
                            for _ in range(args.turns):
                                line.track(particles, num_turns=1)
                        context.synchronize()
                        durations.append(perf_counter()-started)
                        active.append(bool(np.all(context.nparray_from_context_array(
                            particles.state) > 0)))
                    assert all(active), (case, count, scheduler)
                    record = dict(case=case, particles=count, scheduler=scheduler,
                        median_s=float(np.median(durations)), samples_s=durations,
                        ms_per_turn=1000*float(np.median(durations))/args.turns,
                        all_active=active)
                    results['measurements'].append(record)
                    print(json.dumps(record), flush=True)
    finally:
        restore()
        OUTPUT.mkdir(exist_ok=True, parents=True)
        (OUTPUT/f'tracking_floor_{args.context}.json').write_text(
            json.dumps(results, indent=2)+'\n')


if __name__ == '__main__':
    main()
