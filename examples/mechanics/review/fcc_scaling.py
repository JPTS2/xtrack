"""Adversarial scaling probe on the user-selected FCC development lattice.

Run each phase in a fresh process, with serial BLAS/thread settings:
    PYTHONPATH=. python examples/mechanics/review/fcc_scaling.py --phase native
    PYTHONPATH=. python examples/mechanics/review/fcc_scaling.py --phase bulk
    PYTHONPATH=. python examples/mechanics/review/fcc_scaling.py --phase field

Bulk preparation is an isolated PRIVATE-API experiment, not a supported fast
path. The field phase explicitly distinguishes the public allocation guard
from bounded direct source evaluation. No input lattice is rewritten.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
from time import perf_counter

import numpy as np
import xobjects as xo
import xtrack as xt


ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT.parent / 'inputs/acc-models-fcc-ee-108.1.0'
LATTICE = MODEL / 'lattices/z/fccee_z_beam_1_positrons.json'
OUTPUT = Path(__file__).resolve().parent / 'results/fcc_scaling'


def timed(call, repeats=1):
    samples = []
    result = None
    for _ in range(repeats):
        started = perf_counter()
        result = call()
        samples.append(perf_counter()-started)
    return result, dict(median_s=float(np.median(samples)), samples_s=samples)


def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2 if sys.platform == 'darwin' else 1024)


def count_buffer_io(line, call):
    """Count CPU buffer accesses; GPU transfer consequences are source analysis."""
    buffer_type = type(line.element_dict[line.element_names[0]]._buffer)
    read, write = buffer_type.to_bytearray, buffer_type.update_from_buffer
    counts = dict(buffer_type=buffer_type.__name__,read_calls=0,read_bytes=0,write_calls=0,write_bytes=0)
    def counted_read(self,offset,nbytes):
        counts['read_calls'] += 1
        counts['read_bytes'] += nbytes
        return read(self,offset,nbytes)
    def counted_write(self,offset,source):
        counts['write_calls'] += 1
        counts['write_bytes'] += len(source)
        return write(self,offset,source)
    try:
        buffer_type.to_bytearray = counted_read
        buffer_type.update_from_buffer = counted_write
        call()
    finally:
        buffer_type.to_bytearray = read
        buffer_type.update_from_buffer = write
    return counts


def prepare():
    env, load = timed(lambda: xt.load(LATTICE))
    memory_after_load = rss()
    line = env.fccee_p_ring
    table, inspect = timed(line.get_table)
    quads = list(table.rows.match(element_type='Quadrupole').name)
    _, build = timed(lambda: line.build_tracker(_context=xo.ContextCpu()))
    metadata = dict(lattice=str(LATTICE), lattice_sha256=hashlib.sha256(LATTICE.read_bytes()).hexdigest(),
        model_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=MODEL, text=True).strip(),
        model_branch='dev/108.1.0', model_version=(MODEL/'VERSION').read_text().strip(),
        xtrack=xt.__version__, python=sys.version, platform=platform.platform(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        mechanics_source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted((ROOT/'xtrack/mechanics').glob('*.py'))},
        cuda_module=importlib.util.find_spec('cupy') is not None,
        opencl_module=importlib.util.find_spec('pyopencl') is not None,
        thread_settings={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS')},
        circumference_m=line.get_length(), p0c_ev=float(line.particle_ref.p0c[0]),
        revolution_time_s=line.get_length()/(float(line.particle_ref.beta0[0])*299792458),
        n_occurrences=len(line.element_names), n_unique_names=len(set(line.element_names)),
        n_quadrupoles=len(quads), types=dict(Counter(table.element_type[:-1])),
        peak_rss_after_load_mib=memory_after_load, peak_rss_after_tracker_mib=rss(),
        load=load, table_inspection=inspect, tracker_build=build)
    print(f'Loaded {len(line.element_names)} occurrences; {len(quads)} quads; C={line.get_length():.3f} m', flush=True)
    return env, line, table, quads, metadata


def tracking(line, turns=4):
    results = []
    for count in (1, 32):
        durations = []
        states = []
        for _ in range(2):
            particles = line.build_particles(x=np.full(count, 1e-7), y=np.full(count, 1e-8))
            started = perf_counter()
            line.track(particles, num_turns=turns)
            line._context.synchronize()
            durations.append(perf_counter()-started)
            states.append(bool(np.all(particles.state > 0)))
        results.append(dict(particles=count, turns=turns, median_s=float(np.median(durations)),
                            samples_s=durations, all_active=states))
    return results


def native(env, line, table, quads):
    result = {}
    _, result['kernel_warmup'] = timed(lambda: line.track(line.build_particles(), num_turns=1))
    result['frozen_tracking'] = tracking(line)
    _, result['reference_adoption'] = timed(line.mechanics.adopt_reference)
    records = []
    started_all = perf_counter()
    for index, name in enumerate(quads[:8]):
        started = perf_counter()
        try:
            _, duration = timed(lambda: line.mechanics.new_girder(f'g.{name}', elements=[name]))
        except ValueError as error:
            result['unsupported_attachment'] = dict(element=name, error=str(error),
                elapsed_s=perf_counter()-started, successful_groups=len(records))
            print('Native attachment blocked:',str(error),flush=True)
            break
        records.append(dict(supports=index+1, element=name, last_group_s=duration['median_s'],
                            cumulative_s=perf_counter()-started_all))
        print(f'Native {index+1} supports: last {duration["median_s"]:.3f} s', flush=True)
    result['attachment'] = records
    env.mechanics.add_source('tone', xt.mechanics.Harmonic(50e-9, 5.), targets=list(env.mechanics.bodies))
    _, result['pose'] = timed(lambda: env.mechanics.get_body_poses(time_s=.0123), repeats=3)
    _, result['line_validation'] = timed(line.mechanics._check, repeats=3)
    _, result['apply'] = timed(lambda: env.mechanics.apply(time_s=.0123), repeats=3)
    result['apply_cpu_buffer_io'] = count_buffer_io(line,lambda: env.mechanics.apply(time_s=.0123))
    line.mechanics.enable_vibrations()
    result['dynamic_tracking'] = tracking(line)
    line.mechanics.disable_vibrations()
    return result


def experimental_bulk(line, quads):
    """One native candidate capture; one final topology/geometry bookkeeping.

    This deliberately bypasses repeated _attach bookkeeping to isolate the
    architectural issue. It is NOT a public implementation or accepted fix.
    Native _candidates still checks occurrence ownership, expressions and poses.
    """
    view, mech = line.mechanics, line.env.mechanics
    candidates, candidate_cost = timed(lambda: view._candidates(quads))
    started = perf_counter()
    for candidate in candidates:
        name = candidate['element']
        anchor = xt.Frame(candidate['ref_start'])
        anchor.matrix[:3, 3] = (candidate['ref_start'][:3, 3]+candidate['ref_end'][:3, 3])/2
        body = mech.new_girder(f'g.{name}', anchor=anchor)
        candidate['body'] = body.name
        view._attachments[name] = candidate
    view._element_names = tuple(line.element_names)
    view._line_geometry = view._geometry_signature()
    mech._revision += 1
    install_s = perf_counter()-started
    _, validation = timed(view._check)
    return dict(native_candidate_capture=candidate_cost, install_and_bookkeeping_s=install_s,
                final_validation=validation, total_s=candidate_cost['median_s']+install_s+validation['median_s'])


def bulk(env, line, table, quads):
    result = {}
    # First exercise the unmodified imported model, preserving the rejection.
    try:
        line.mechanics._candidates(quads)
        result['unmodified_all_quad_candidates'] = 'accepted'
    except ValueError as error:
        result['unmodified_all_quad_candidates'] = str(error)
    # A detached NUMERIC snapshot is used only for the scaling experiment.
    # All current element values/maps are copied; live knob/expression links
    # are deliberately absent. This is not a solution for the user's model.
    started = perf_counter()
    unique = dict.fromkeys(line.element_names)
    elements = {name:line.element_dict[name].copy() for name in unique}
    numeric_line = xt.Line(elements=elements, element_names=list(line.element_names),
                            particle_ref=line.particle_ref.copy())
    numeric_line.config.data.update(line.config.data)
    result['numeric_snapshot_s'] = perf_counter()-started
    result['numeric_snapshot_has_live_knobs'] = False
    result['numeric_snapshot_circumference_difference_m'] = numeric_line.get_length()-line.get_length()
    # Boundaries/strengths are identical at the supplied input values.
    for name in quads:
        assert numeric_line[name].length == line[name].length
        assert numeric_line[name].k1 == line[name].k1
    line = numeric_line
    env = line.env
    _, result['numeric_tracker_build'] = timed(line.build_tracker)
    _, result['kernel_warmup'] = timed(lambda: line.track(line.build_particles(), num_turns=1))
    result['frozen_tracking'] = tracking(line)
    _, result['reference_adoption'] = timed(line.mechanics.adopt_reference)
    result['experimental_bulk_attachment'] = experimental_bulk(line, quads)
    print('Experimental bulk preparation:',result['experimental_bulk_attachment']['total_s'],flush=True)
    env.mechanics.add_source('tone', xt.mechanics.Harmonic(50e-9, 5.), targets=list(env.mechanics.bodies))
    _, result['pose'] = timed(lambda: env.mechanics.get_body_poses(time_s=.0123), repeats=3)
    _, result['line_validation'] = timed(line.mechanics._check, repeats=3)
    poses = env.mechanics.get_body_poses(time_s=.0123)
    updates, result['transform_lowering_without_validation'] = timed(
        lambda: line.mechanics._updates(poses, check=False), repeats=3)
    def write_fields():
        fields = ('shift_x','shift_y','shift_s','rot_x_rad','rot_y_rad','rot_s_rad_no_frame','rot_shift_anchor')
        for attachment,element,values in updates:
            for key,value in zip(fields,values):
                setattr(element,key,value)
            attachment['last_written'] = values.copy()
    _, result['scalar_field_writes'] = timed(write_fields,repeats=3)
    _, result['apply'] = timed(lambda: env.mechanics.apply(time_s=.0123), repeats=3)
    result['apply_cpu_buffer_io'] = count_buffer_io(line,lambda: env.mechanics.apply(time_s=.0123))
    nominal_table = line.get_table()
    speed = float(line.particle_ref.beta0[0])*299792458
    def pointwise_passage_probe():
        for name in quads[:8]:
            poses = env.mechanics.get_body_poses(time_s=.0123+float(nominal_table['s',name])/speed)
            line.mechanics._updates(poses,names=[name],check=False)
    _, result['eight_pointwise_passage_queries'] = timed(pointwise_passage_probe)
    result['full_passage_extrapolation_s'] = result['eight_pointwise_passage_queries']['median_s']*len(quads)/8
    try:
        env.mechanics.compute_time_series(time_s=[0., .01])
        result['two_sample_harmonic_record'] = 'accepted'
    except ValueError as error:
        result['two_sample_harmonic_record'] = str(error)
    line.mechanics.enable_vibrations()
    result['dynamic_tracking'] = tracking(line)
    line.mechanics.disable_vibrations()
    # Verify the actual lowering on a representative attached element.
    env.mechanics.apply(time_s=.0123)
    requested = line.mechanics.get_element_poses(time_s=.0123)
    actual, result['verification_survey'] = timed(lambda: line.survey(include_element_frames=True))
    result['entrance_position_error_m'] = float(max(np.max(np.abs(
        actual.get_frame(n, which='elem_start').XYZ-requested[n].XYZ)) for n in quads))
    # Small local translations should not lose precision by travelling through
    # world coordinates of tens of kilometres and back to the element frame.
    from dataclasses import replace
    original = env.mechanics._sources['tone']
    precision = []
    for amplitude in (1e-7,1e-9,1e-10,1e-11):
        env.mechanics._sources['tone'] = replace(original,
            source=xt.mechanics.Harmonic(amplitude,0.,np.pi/2), component='x')
        env.mechanics.apply(time_s=0.)
        realised = np.array([line[n].shift_x-line.mechanics._attachments[n]['baseline'][0] for n in quads])
        precision.append(dict(command_m=amplitude,max_error_m=float(np.max(np.abs(realised-amplitude))),
            rms_error_m=float(np.sqrt(np.mean((realised-amplitude)**2))),
            min_realised_m=float(realised.min()),max_realised_m=float(realised.max())))
    env.mechanics._sources['tone'] = original
    result['small_translation_precision'] = precision
    selected = line.mechanics.get_table(time_s=0.).rows[quads[len(quads)//2]]
    _, result['one_selected_element_33_sample_history'] = timed(
        lambda: selected.get_time_series(time_s=np.linspace(0.,.1,33)))
    env.mechanics.restore()
    return result


def field(env, line, table, quads):
    """Bounded field/canonical-response cost; NOT a validated FCC noise model."""
    result = {}
    survey, result['survey'] = timed(lambda: line.survey(include_element_frames=True))
    frames = []
    for name in quads:
        first, last = (survey.get_frame(name, which=k) for k in ('ref_start', 'ref_end'))
        frame = xt.Frame(first.matrix)
        frame.matrix[:3, 3] = (first.XYZ+last.XYZ)/2
        frames.append(frame)
    feet = np.array([[-.3,-.5,-.5],[.3,-.5,-.5],[-.3,-.5,.5],[.3,-.5,.5]])
    positions = np.concatenate([feet @ f.matrix[:3,:3].T+f.XYZ for f in frames])
    times = np.arange(128)/128
    source = xt.mechanics.SpectralGroundMotion([0.,1.,2.,30.,40.], [0.,0.,1e-18,1e-18,0.],
        seed=956, correlation_length_m=30., n_modes=256)
    result.update(n_receivers=len(positions), n_times=len(times), n_modes=source.n_modes,
        direct_source_workspace_estimate_bytes=source.workspace_bytes(len(times),len(positions)))
    try:
        xt.mechanics.sample_field(source, time_s=times, positions_m=positions)
        result['public_field_guard'] = 'accepted'
    except ValueError as error:
        result['public_field_guard'] = str(error)
    # Explicit bounded diagnostic bypass; do not bypass a possible large dense
    # covariance. This source uses finite random modes, whose workspace is known.
    if result['direct_source_workspace_estimate_bytes'] > 256*1024**2:
        raise MemoryError('Direct diagnostic source workspace exceeds 256 MiB')
    values, result['direct_source_evaluation'] = timed(lambda: source.evaluate(times, positions))
    world_values = np.zeros((*values.shape, 3))
    world_values[...,1] = values
    world = xt.mechanics.FieldMotion(times, positions, world_values)
    (rigid,residual), result['rigid_contact_fit'] = timed(lambda: world.fit_rigid_bodies(
        frames=frames, feet_local_m=feet, target_names=tuple(quads)))
    response_outputs = []
    response_costs = {}
    # Synthetic classes isolate grouping/response cost. They do not encode
    # engineering girder assignments, which are not supplied by this probe.
    for index, resonance in enumerate((8.,12.,18.)):
        indices = np.arange(index,len(quads),3)
        response = xt.mechanics.CausalResponse(oscillator=xt.mechanics.BaseExcitedOscillator(resonance,.15),
                                              sample_step_s=1/128)
        output, cost = timed(lambda: response.process(rigid.values[:,indices]))
        response_outputs.append(output)
        response_costs[f'benchmark_class_{index}'] = dict(count=len(indices),frequency_hz=resonance,**cost)
        split_response = xt.mechanics.CausalResponse(
            oscillator=xt.mechanics.BaseExcitedOscillator(resonance,.15), sample_step_s=1/128)
        split = np.concatenate((split_response.process(rigid.values[:64,indices]),
                                split_response.process(rigid.values[64:,indices])))
        mismatch = float(np.max(np.abs(output-split)))
        assert mismatch == 0.
        response_costs[f'benchmark_class_{index}']['whole_split_max_error'] = mismatch
    result['class_responses'] = response_costs
    result['field_payload_bytes'] = world.nbytes
    result['rigid_payload_bytes'] = rigid.nbytes
    result['response_payload_bytes'] = sum(v.nbytes for v in response_outputs)
    result['contact_residual_rms_m'] = float(np.sqrt(np.mean(residual**2)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('native','bulk','field'), required=True)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    env,line,table,quads,metadata = prepare()
    result = dict(phase=args.phase, metadata=metadata)
    try:
        result['results'] = globals()[args.phase](env,line,table,quads)
        result['status'] = 'passed'
    except Exception as error:
        result['status'] = 'failed'
        result['error'] = dict(type=type(error).__name__,message=str(error))
        raise
    finally:
        result['peak_rss_mib'] = rss()
        (OUTPUT/(args.phase+'.json')).write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    main()
