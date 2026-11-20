"""Measure the public prepared runtime on the parameterized FCC Z input.

Supports here are provisional cost probes, not engineering assignments.
The baseline includes ordinary tracking on the same unmodified input. Dynamic
timings include complete track calls, guards and final ownership bookkeeping;
cold preparation and steady update-only costs are reported separately.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
import xobjects as xo
import xtrack as xt


ROOT = Path(__file__).resolve().parents[3]
INPUT = ROOT.parent/'inputs/acc-models-fcc-ee-108.1.0/lattices/z/fccee_z_beam_1_positrons.json'
OUTPUT = Path(__file__).resolve().parent/'results/fcc_freeze_runtime'


def count_buffer_io(line, call):
    """Count Python Xobjects buffer access hooks, not every device transfer."""
    buffer_type = type(line.element_dict[line.element_names[0]]._buffer)
    read, write = buffer_type.to_bytearray, buffer_type.update_from_buffer
    counts = dict(buffer_type=buffer_type.__name__, read_calls=0, read_bytes=0,
                  write_calls=0, write_bytes=0)
    def counted_read(self, offset, nbytes):
        counts['read_calls'] += 1
        counts['read_bytes'] += nbytes
        return read(self, offset, nbytes)
    def counted_write(self, offset, source):
        counts['write_calls'] += 1
        counts['write_bytes'] += len(source)
        return write(self, offset, source)
    try:
        buffer_type.to_bytearray = counted_read
        buffer_type.update_from_buffer = counted_write
        call()
    finally:
        buffer_type.to_bytearray = read
        buffer_type.update_from_buffer = write
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context', choices=('cpu', 'openmp', 'cuda', 'opencl'), default='cpu')
    parser.add_argument('--device', help='CUDA index or OpenCL platform.device')
    parser.add_argument('--threads', type=int, default=2, help='OpenMP threads')
    parser.add_argument('--input', type=Path, default=INPUT)
    parser.add_argument('--line', default='fccee_p_ring')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--members', choices=('quads', 'all_magnets'), default='quads')
    parser.add_argument('--turns', type=int, default=64)
    parser.add_argument('--particles', type=int, nargs='+', default=[1, 32])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--hierarchy', choices=('flat', 'nested'), default='flat')
    parser.add_argument('--short-calls', action='store_true')
    parser.add_argument('--source', choices=('analytic', 'sampled_ground', 'direct_ground'), default='analytic')
    args = parser.parse_args()
    if Path(xt.__file__).resolve() != (ROOT/'xtrack/__init__.py').resolve():
        parser.error('Imported another Xtrack installation; run from this checkout with PYTHONPATH=.')
    if args.turns <= 0 or args.repeats <= 0 or args.threads <= 0 or any(n <= 0 for n in args.particles):
        parser.error('Turns, repeats, threads and particle counts must be positive')
    if args.device is not None and args.context in ('cpu', 'openmp'):
        parser.error('--device applies only to CUDA/OpenCL')
    if args.source == 'sampled_ground' and (args.members != 'quads' or args.hierarchy != 'flat'):
        parser.error('The sampled ground/response cost probe uses flat provisional quad sites')
    context = {'cpu': lambda: xo.ContextCpu(),
               'openmp': lambda: xo.ContextCpu(omp_num_threads=args.threads),
               'cuda': lambda: xo.ContextCupy(device=int(args.device) if args.device is not None else None),
               'opencl': lambda: xo.ContextPyopencl(device=args.device)}[args.context]()
    env = xt.load(args.input)
    line = env[args.line]
    line.build_tracker(_context=context)
    line.track(line.build_particles(), num_turns=1)
    def measure(count, turns=None):
        turns = args.turns if turns is None else turns
        samples, active = [], []
        for _ in range(args.repeats):
            particles = line.build_particles(x=np.full(count, 1e-7), y=np.full(count, 1e-8))
            context.synchronize()
            start = perf_counter()
            line.track(particles, num_turns=turns)
            context.synchronize()
            samples.append(perf_counter()-start)
            active.append(bool(np.all(context.nparray_from_context_array(particles.state) > 0)))
        assert all(active)
        return dict(median_s=float(np.median(samples)), samples_s=samples, all_active=active,
                    ms_per_turn=1000*float(np.median(samples))/turns)
    baseline = {str(count): measure(count) for count in args.particles}
    short_baseline = {str(turns):measure(1, turns) for turns in (1, 4, 16)} if args.short_calls else {}
    table = line.get_table()
    pattern = 'Quadrupole' if args.members == 'quads' else 'Quadrupole|RBend|Sextupole|Octupole'
    names = list(table.rows.match(element_type=pattern).name)
    start = perf_counter()
    if args.hierarchy == 'flat':
        assignments = {f'g.{name}': [name] for name in names}
        parents = ()
    else:
        survey = line.survey(include_element_frames=True)
        assignments, parents = {}, []
        for i in range(0, len(names), 3):
            group = names[i:i+3]
            frame = survey.get_frame(group[0], which='ref_start')
            last = survey.get_frame(group[-1], which='ref_end')
            frame.matrix[:3, 3] = (frame.XYZ+last.XYZ)/2
            parent = env.mechanics.new_girder(f'support.{i//3}', anchor=frame,
                support_class='provisional_triplet_support')
            parents.append(parent.name)
            for name in group:
                assignments[f'g.{name}'] = dict(elements=[name], parent=parent.name)
    line.mechanics.new_girders(assignments, support_class='provisional_cost_probe')
    definition_s = perf_counter()-start
    bodies = tuple(assignments)
    start = perf_counter()
    if args.source == 'analytic':
        for index, component in enumerate(('x', 'y', 's', 'pitch', 'yaw', 'roll')):
            env.mechanics.add_source(component, xt.mechanics.Harmonic(
                amplitude=1e-9 if index < 3 else 1e-10,
                frequency_hz=10.+index, phase_rad=.2,
                unit='m' if index < 3 else 'rad'), targets=tuple(parents) if parents else bodies,
                component=component)
        if parents:
            for index, component in enumerate(('x', 'y', 'roll')):
                env.mechanics.add_source(f'local_{component}', xt.mechanics.Harmonic(
                    amplitude=1e-10 if index < 2 else 1e-11,
                    frequency_hz=100.+index, unit='m' if index < 2 else 'rad'),
                    targets=bodies, component=component)
    elif args.source == 'direct_ground':
        # A distinct provider-cost probe: field at anchors, without the feet/
        # response pipeline used by sampled_ground.
        env.mechanics.add_source('direct_ground', xt.mechanics.SpectralGroundMotion(
            [0., 2., 3., 30., 40.], [0., 0., 1e-18, 1e-18, 0.],
            seed=956, correlation_length_m=30., n_modes=128), targets=bodies)
    else:
        # One COMMON field at all feet, then three illustrative girder response
        # classes. Explicit short host record; no GPU time-history cache.
        times = np.arange(128)/2048
        frames = [env.mechanics[name].nominal_frame for name in bodies]
        feet = np.array([[-.3, -.5, -.5], [.3, -.5, -.5], [-.3, -.5, .5], [.3, -.5, .5]])
        positions = np.concatenate([feet @ frame.E_matrix.T + frame.XYZ for frame in frames])
        ground = xt.mechanics.SpectralGroundMotion([0., 2., 3., 30., 40.],
            [0., 0., 1e-18, 1e-18, 0.], seed=956, correlation_length_m=30., n_modes=128)
        field = xt.mechanics.sample_field(ground, time_s=times, positions_m=positions,
                                         max_memory_bytes=256*1024**2)
        rigid, residual = field.fit_rigid_bodies(frames=frames, feet_local_m=feet,
                                               target_names=bodies)
        values = np.empty_like(rigid.values)
        for index, resonance in enumerate((8., 12., 18.)):
            indices = np.arange(index, len(bodies), 3)
            response = xt.mechanics.CausalResponse(oscillator=xt.mechanics.BaseExcitedOscillator(
                resonance, .15), sample_step_s=1/2048)
            values[:, indices] = response.process(rigid.values[:, indices])
        env.mechanics.add_source('resolved_ground_response',
            xt.mechanics.RigidMotion(times, values, target_names=bodies), targets=bodies)
    source_prepare_s = perf_counter()-start
    line.mechanics.enable_vibrations(backend='compiled')
    start = perf_counter()
    line.mechanics._prepare_for_tracking()
    cold_prepare_s = perf_counter()-start
    runtime = line.mechanics._runtime
    runtime.apply(0.)
    runtime.finish()
    samples = []
    for index in range(30):
        context.synchronize()
        start = perf_counter()
        runtime.apply(index*1e-5)
        context.synchronize()
        samples.append(perf_counter()-start)
    runtime.finish()
    dynamic = {str(count): measure(count) for count in args.particles}
    short_dynamic = {str(turns):measure(1, turns) for turns in (1, 4, 16)} if args.short_calls else {}
    ratios = {key: dynamic[key]['median_s']/baseline[key]['median_s'] for key in baseline}
    # An update must never perform per-scalar buffer reads/writes on the host.
    io = count_buffer_io(line, lambda: runtime.apply(.01))
    runtime.finish()
    line.mechanics.disable_vibrations()
    line.mechanics.validate()
    metadata = dict(lattice=str(args.input.resolve()), line=args.line,
        lattice_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        tracker_sha256=hashlib.sha256((ROOT/'xtrack/tracker.py').read_bytes()).hexdigest(),
        mechanics_source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted((ROOT/'xtrack/mechanics').rglob('*'))
                                 if p.suffix in ('.py', '.h')},
        platform=platform.platform(), python=sys.version,
        imported_xtrack=str(Path(xt.__file__).resolve()),
        xtrack=xt.__version__, xobjects=xo.__version__, numpy=np.__version__,
        context=args.context, actual_context=str(context), device=args.device,
        members=args.members, n_attachments=len(names),
        n_supports=len(env.mechanics.bodies), hierarchy=args.hierarchy, turns=args.turns,
        thread_settings={key:os.environ.get(key) for key in
            ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')},
        compiler_settings={key:os.environ.get(key) for key in ('CC', 'LDSHARED')},
        public_api=True, live_expressions_preserved=True, engineering_assignment=False,
        update_sampling='turn-held', source=args.source,
        source_execution=({'analytic':'context-native analytic',
            'sampled_ground':'host short-record interpolation, one packed transfer, context-native composition',
            'direct_ground':'host direct spatial-field evaluation, one packed transfer, context-native composition'}[args.source]),
        source_prepare_s=source_prepare_s,
        definition_s=definition_s, cold_prepare_s=cold_prepare_s,
        update_median_s=float(np.median(samples)), update_samples_s=samples,
        prepared_array_bytes={name:int(value.nbytes) for name,value in vars(runtime).items()
                              if hasattr(value, 'nbytes')},
        guard_array_bytes={name:int(value.nbytes) for name,value in vars(runtime.guard).items()
                           if hasattr(value, 'nbytes')},
        update_host_buffer_io=io,
        baseline=baseline, dynamic=dynamic, slowdown=ratios,
        short_calls={key:dict(baseline=short_baseline[key], dynamic=short_dynamic[key],
            slowdown=short_dynamic[key]['median_s']/short_baseline[key]['median_s'])
            for key in short_baseline},
        target_max_slowdown=2., target_met={key:value <= 2. for key,value in ratios.items()})
    args.output.mkdir(parents=True, exist_ok=True)
    suffix = '' if args.hierarchy == 'flat' else '_nested'
    if args.source != 'analytic':
        suffix += '_'+args.source
    (args.output/f'{args.members}_{args.context}{suffix}.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
