# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #
"""Prepared, bounded mechanics updates on the tracking context.

Topology and geometry are frozen for one tracking call. Analytic sources execute
on context; sampled/custom sources supply one packed six-coordinate input block.
Rotation composition, hierarchy and alignment writes execute in context kernels.
No particle evaluates a source, and no full-run pose history is allocated.
"""
from pathlib import Path
from weakref import WeakKeyDictionary

import numpy as np
import xobjects as xo
from ..composer.positions import CompensatedFloat

from .local_deformations import attachment_reference, static_local
from .motion import RigidMotion
from .sources import Harmonic, PlaneWave
from .context_arrays import to_context


FIELDS = ('shift_x', 'shift_y', 'shift_s', 'rot_x_rad', 'rot_y_rad',
          'rot_s_rad_no_frame', 'rot_shift_anchor')


def _rotation_increment(angles):
    """Vectorized, cancellation-free survey rotation minus identity."""
    phi, theta, psi = np.asarray(angles).T
    sx, sy, sz = np.sin(phi), np.sin(theta), np.sin(psi)
    cx, cy, cz = (-2*np.sin(a/2)**2 for a in (phi, theta, psi))
    shape = (len(phi), 3, 3)
    pitch, yaw, roll = (np.zeros(shape) for _ in range(3))
    pitch[:, 1, 1], pitch[:, 1, 2] = cx, sx
    pitch[:, 2, 1], pitch[:, 2, 2] = -sx, cx
    yaw[:, 0, 0], yaw[:, 0, 2] = cy, sy
    yaw[:, 2, 0], yaw[:, 2, 2] = -sy, cy
    roll[:, 0, 0], roll[:, 0, 1] = cz, -sz
    roll[:, 1, 0], roll[:, 1, 1] = sz, cz
    result = yaw + pitch + yaw @ pitch
    return result + roll + result @ roll


def _interpolate(times, values, time_s):
    if time_s < times[0] or time_s > times[-1]:
        raise ValueError('Requested time lies outside the computed motion record')
    index = min(int(np.searchsorted(times, time_s, side='right'))-1, len(times)-2)
    weight = (time_s-times[index])/(times[index+1]-times[index])
    return values[index] + weight*(values[index+1]-values[index])


def _compile(context):
    if all(name in context.kernels for name in ('mechanics_body_update', 'mechanics_lower',
            'mechanics_sources', 'mechanics_source_values', 'mechanics_publish_clock')):
        return
    def arg(dtype, name):
        return xo.Arg(dtype, name=name, pointer=True)
    body = [xo.Arg(xo.Int64, name='n')]
    body += [arg(xo.Int64, n) for n in ('paths_offsets', 'paths')]
    body += [arg(xo.Float64, n) for n in ('relations', 'static_local', 'dynamic_local')]
    body += [arg(xo.Int64, 'absolute'), arg(xo.Float64, 'out')]
    lower = [xo.Arg(xo.Int64, name='n'), arg(xo.Int64, 'body_indices')]
    lower += [arg(xo.Float64, n) for n in ('geometry', 'baseline')]
    lower += [arg(xo.Int64, 'offsets'), arg(xo.Float64, 'bodyout')]
    lower += [arg(xo.Int8, 'buffer'), arg(xo.Float64, 'values')]
    context.add_kernels(sources=[Path(__file__).parent/'runtime_src/update.h'], kernels={
        'mechanics_body_update': xo.Kernel(args=body, n_threads='n'),
        'mechanics_lower': xo.Kernel(args=lower, n_threads='n'),
        'mechanics_sources': xo.Kernel(args=[xo.Arg(xo.Int64, name='n'),
            arg(xo.Int64, 'source_starts'), arg(xo.Float64, 'source_data'),
            xo.Arg(xo.Float64, name='time_s'), arg(xo.Float64, 'dynamic_local')], n_threads='n'),
        'mechanics_publish_clock': xo.Kernel(args=[arg(xo.Int8, 'buffer'), arg(xo.Int64, 'offsets'),
            xo.Arg(xo.Float64, name='time_s'), xo.Arg(xo.Float64, name='reference_turns'),
            xo.Arg(xo.Float64, name='t_sim')], n_threads=1),
        'mechanics_source_values': xo.Kernel(args=[xo.Arg(xo.Int64, name='n'),
            arg(xo.Int64, 'source_starts'), arg(xo.Int64, 'source_rows'),
            arg(xo.Float64, 'values'), arg(xo.Float64, 'dynamic_local')], n_threads='n')},
        extra_include_dirs=[str(Path(xo.__file__).resolve().parents[1])])


class PreparedMechanics:
    """One current-pose buffer, never an implicit entire-run cache."""

    def __init__(self, view):
        self.view = view
        self.manager = view.manager
        self.revision = self.manager._revision
        self.buffer = view.line.tracker._buffer
        self.context = self.buffer.context
        self._clocks = WeakKeyDictionary()
        classes = {type(element) for element in view.line.tracker._tracker_data_base._elements}
        self.fixed_reference = all(cls.__module__.startswith(('xtrack.beam_elements', 'xtrack.monitors'))
            and cls.__name__ not in ('ReferenceEnergyIncrease', 'ReferenceEnergyChange', 'LineSegmentMap')
            for cls in classes)
        _compile(self.context)
        self.names = tuple(self.manager.bodies)
        self.indices = {name: i for i, name in enumerate(self.names)}
        n = len(self.names)
        relations, static = np.zeros((n, 12)), np.zeros((n, 12))
        self._static_settings = np.zeros((n, 12))
        paths, starts = [], [0]
        for name, body in self.manager.bodies.items():
            i = self.indices[name]
            chain, current = [], body
            while current is not None:
                chain.append(self.indices[current.name])
                current = None if current.parent is None else self.manager[current.parent]
            paths.extend(reversed(chain))
            starts.append(len(paths))
            relations[i, :9] = np.eye(3).ravel()
            if body.parent is not None:
                parent = self.manager[body.parent]._nominal
                relations[i, :9] = (parent[:3, :3].T @ body._nominal[:3, :3]).ravel()
                relations[i, 9:] = parent[:3, :3].T @ (body._nominal[:3, 3]-parent[:3, 3])
            values = np.array([body.shift_x, body.shift_y, body.shift_s,
                               body.rot_x_rad, body.rot_y_rad, body.rot_s_rad])
            if not np.all(np.isfinite(values)):
                raise ValueError(f'{name}: local alignment must be finite')
            installation = body.installation.values()
            self._static_settings[i] = np.concatenate((values, installation))
            d, e = static_local(values, installation)
            static[i, :3], static[i, 3:] = d, e.ravel()
        self.attachments = list(view._attachments.values())
        a = len(self.attachments)
        geometry, baseline, offsets = np.zeros((a, 24)), np.zeros((a, 7)), np.zeros((a, 7), dtype=np.int64)
        body_indices = []
        for i, attachment in enumerate(self.attachments):
            body = self.manager[attachment['body']]
            body_indices.append(self.indices[body.name])
            t0, r0 = attachment_reference(attachment)
            reference = attachment['ref_start']
            axes = reference[:3, :3].T @ body._nominal[:3, :3]
            lever = (body._nominal[:3, :3].T @ (reference[:3, 3]-body._nominal[:3, 3])
                     + axes.T @ t0)
            geometry[i] = np.concatenate((axes.ravel(), lever, t0, r0.ravel()))
            baseline[i] = attachment['baseline']
            element = view.line.element_dict[attachment['element']]
            if element._buffer is not self.buffer:
                raise ValueError('Prepared mechanics requires elements in the tracker buffer')
            offsets[i] = [element._xobject._get_offset(field) for field in FIELDS]
        array = lambda value: to_context(self.context, value)
        self.paths_offsets, self.paths = array(np.array(starts, dtype=np.int64)), array(np.array(paths, dtype=np.int64))
        self.relations, self.static_local = array(relations.ravel()), array(static.ravel())
        self.geometry, self.baseline = array(geometry.ravel()), array(baseline.ravel())
        self.offsets, self.body_indices = array(offsets.ravel()), array(np.array(body_indices, dtype=np.int64))
        self.input = np.zeros((n, 12))
        self.dynamic_local = array(self.input.ravel())
        self.absolute_host = np.zeros(n, dtype=np.int64)
        self.sources = []
        for name, binding in self.manager.sources.items():
            indices = np.array([self.indices[target] for target in binding.targets])
            frames = np.array([self.manager[target]._nominal for target in binding.targets])
            self.sources.append((name, binding, indices, frames[:, :3, 3], frames[:, :3, :3]))
            if binding.reference == 'absolute':
                self.absolute_host[indices] = 1
        self.absolute = array(self.absolute_host)
        self.analytic = all(type(binding.source) in (Harmonic, PlaneWave)
                            for binding in self.manager.sources.values())
        if self.analytic:
            rows = [[] for _ in self.names]
            for _, binding, indices, positions, axes in self.sources:
                source = binding.source
                omega = 2*np.pi*source.frequency_hz
                if isinstance(source, Harmonic):
                    component = ('x', 'y', 's', 'pitch', 'yaw', 'roll').index(binding.component)
                    for index in indices:
                        rows[index].append([component, source.amplitude, omega, source.phase_rad, 0., 0., 0.])
                else:
                    phase = source.phase_rad - omega*((positions-source.origin_m) @ np.asarray(source.direction))/source.speed_m_s
                    polarization = np.einsum('bji,j->bi', axes, source.polarization)
                    for index, angle, direction in zip(indices, phase, polarization):
                        rows[index].append([-1, source.amplitude, omega, angle, *direction])
            starts = np.concatenate(([0], np.cumsum([len(row) for row in rows])))
            data = np.array([item for row in rows for item in row], dtype=float).reshape(-1)
            self.source_starts = array(starts.astype(np.int64))
            self.source_data = array(data if len(data) else np.zeros(1))
        self.bodyout = self.context.zeros(n*12, dtype=np.float64)
        self.values = array(np.array([item['last_written'] for item in self.attachments]).ravel())
        self.applied = False
        self.raw_values = None
        from .geometry_guard import GeometryGuard
        self.guard = GeometryGuard(view)

    def _ensure_sampled_input(self):
        if self.raw_values is not None:
            return
        rows = [[] for _ in self.names]
        self.row_slices = {}
        offset = 0
        for name, _, indices, _, _ in self.sources:
            self.row_slices[name] = slice(offset, offset+len(indices))
            for index, body in enumerate(indices):
                rows[body].append(offset+index)
            offset += len(indices)
        self.raw_values = np.zeros((offset, 6))
        array = lambda value: to_context(self.context, value)
        self.sample_starts = array(np.concatenate(([0], np.cumsum([len(row) for row in rows]))).astype(np.int64))
        self.sample_rows = array(np.array([item for row in rows for item in row], dtype=np.int64))
        self.sample_values = array(self.raw_values.ravel())

    def refresh_static(self):
        """Read mutable body settings once per call, without world poses."""
        values = np.array([[body.shift_x, body.shift_y, body.shift_s,
                            body.rot_x_rad, body.rot_y_rad, body.rot_s_rad]
                           for body in self.manager.bodies.values()])
        installation = np.array([[body.installation.shift_x, body.installation.shift_y,
                                  body.installation.shift_s, body.installation.rot_x_rad,
                                  body.installation.rot_y_rad, body.installation.rot_s_rad]
                                 for body in self.manager.bodies.values()])
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(installation)):
            raise ValueError('Body local alignments must be finite')
        settings = np.column_stack((values, installation))
        if np.array_equal(settings, self._static_settings):
            return
        left, right = _rotation_increment(values[:, 3:]), _rotation_increment(installation[:, 3:])
        data = np.empty_like(self.input)
        data[:, :3] = values[:, :3] + installation[:, :3] + np.einsum('bij,bj->bi', left, installation[:, :3])
        data[:, 3:] = (left + right + left @ right).reshape(-1, 9)
        self.static_local[:] = to_context(self.context, data.ravel())
        self._static_settings = settings

    def apply(self, time_s, time_series=None):
        if self.manager._revision != self.revision or self.view.line.tracker._buffer is not self.buffer:
            raise ValueError('Mechanics definition changed inside a tracking call; prepare a new call')
        if not np.isscalar(time_s) or not np.isfinite(time_s):
            raise ValueError('time_s must be a finite scalar in seconds')
        if self.analytic and time_series is None:
            self.context.kernels.mechanics_sources(n=len(self.names), source_starts=self.source_starts,
                source_data=self.source_data, time_s=time_s, dynamic_local=self.dynamic_local)
            self._apply_kernels()
            return
        self._ensure_sampled_input()
        for name, binding, indices, positions, axes in self.sources:
            source = binding.source
            if time_series is not None:
                values = _interpolate(time_series.time_s, time_series.values[name], time_s)
            elif type(source) is RigidMotion:
                values = _interpolate(source.time_s, source.values, time_s)
            else:
                values = np.asarray(source.evaluate(time_s=np.array([time_s]), positions_m=positions))[0]
                if np.iscomplexobj(values) or not np.all(np.isfinite(values)):
                    raise ValueError(f'{name}: source must return finite real motion')
                frame = getattr(source, 'frame', 'body_local')
                if frame == 'world':
                    values = np.column_stack((np.einsum('bji,bj->bi', axes, values), np.zeros((len(indices), 3))))
                elif frame != 'body_local_rigid':
                    output = np.zeros((len(indices), 6))
                    output[:, ('x', 'y', 's', 'pitch', 'yaw', 'roll').index(binding.component)] = values
                    values = output
            if values.shape != (len(indices), 6) or np.iscomplexobj(values) or not np.all(np.isfinite(values)):
                raise ValueError(f'{name}: source must return finite real motion with matching targets')
            self.raw_values[self.row_slices[name]] = values
        self.sample_values[:] = to_context(self.context, self.raw_values.ravel())
        self.context.kernels.mechanics_source_values(n=len(self.names), source_starts=self.sample_starts,
            source_rows=self.sample_rows, values=self.sample_values, dynamic_local=self.dynamic_local)
        self._apply_kernels()

    def _apply_kernels(self):
        self.context.kernels.mechanics_body_update(n=len(self.names), paths_offsets=self.paths_offsets,
            paths=self.paths, relations=self.relations, static_local=self.static_local,
            dynamic_local=self.dynamic_local, absolute=self.absolute, out=self.bodyout)
        self.context.kernels.mechanics_lower(n=len(self.attachments), body_indices=self.body_indices,
            geometry=self.geometry, baseline=self.baseline, offsets=self.offsets,
            bodyout=self.bodyout, buffer=self.buffer.buffer, values=self.values)
        self.applied = True

    def finish(self):
        """Synchronize conflict-detection values once per call, in one read."""
        if self.applied:
            values = self.context.nparray_from_context_array(self.values).reshape(-1, 7)
            for attachment, value in zip(self.attachments, values):
                attachment['last_written'] = value.copy()
            self.applied = False


def can_track_prepared(tracker, particles, kwargs):
    """Conservative fixed-reference scheduler eligibility; others retain fallback."""
    allowed = {'ele_start', 'ele_stop', 'num_elements', 'num_turns', 'turn_by_turn_monitor',
               'freeze_longitudinal', 'backtrack', 'log', '_reset_log', '_force_no_end_turn_actions'}
    if set(kwargs)-allowed:
        return False
    monitor = kwargs.get('turn_by_turn_monitor')
    if (tracker.iscollective or tracker.line.energy_program is not None
            or tracker.line.enable_time_dependent_vars or tracker._sync_time_controller is not None
            or tracker.skip_end_turn_actions or not tracker.reset_s_at_end_turn
            or tracker.enable_pipeline_hold or particles.start_tracking_at_element >= 0
            or kwargs.get('ele_start') not in (None, 0) or kwargs.get('ele_stop') is not None
            or kwargs.get('num_elements') is not None or kwargs.get('backtrack', False)
            or kwargs.get('freeze_longitudinal', False) or kwargs.get('log') is not None
            or kwargs.get('_force_no_end_turn_actions', False)
            or monitor == 'ONE_TURN_EBE' or getattr(monitor, 'ebe_mode', 0) == 1
            or tracker.track_flags.XS_FLAG_BACKTRACK
            or not tracker.line.mechanics._runtime.fixed_reference):
        return False
    # A shared reference clock requires a common reference speed. Reduce on
    # context and transfer one result, rather than copying the particle array.
    beta = float(particles.beta0[0])
    if not np.isfinite(beta) or beta <= 0:
        return False
    uniform = (particles.beta0 == beta).all()
    if hasattr(uniform, 'get'):
        uniform = uniform.get()
    return bool(uniform)


def track_prepared(tracker, particles, *, num_turns=1, turn_by_turn_monitor=None,
                   multi_element_monitor=None, **kwargs):
    """Fixed-reference complete turns with no host clock I/O inside the loop."""
    if num_turns is None:
        num_turns = 1
    if not isinstance(num_turns, (int, np.integer)) or num_turns <= 0:
        raise ValueError('num_turns must be a positive integer')
    tracker._check_invalidated()
    if isinstance(tracker._context, xo.ContextCpu):
        if particles._num_active_particles < 0 or particles._num_lost_particles < 0:
            raise ValueError('Reorganize particles before CPU tracking')
    if tracker.line._needs_rng and not particles._has_valid_rng_state():
        particles._init_random_number_generator()
    tracker._prepare_particle_time_s_and_reference_turns(particles)
    time_s, reference_turns, step = (float(getattr(particles, name))
                                   for name in ('time_s', 'reference_turns', 't_sim'))
    runtime = tracker.line.mechanics._runtime
    previous = runtime._clocks.get(particles)
    if (previous is not None and float(previous[0]) == time_s
            and previous[1] == reference_turns and previous[2] == step):
        clock = previous[0]
    else:
        clock = CompensatedFloat(time_s)
    context = tracker._context
    offsets = to_context(context, np.array([
        particles._xobject._get_offset(name) for name in ('time_s', 'reference_turns', 't_sim')], dtype=np.int64))
    def publish():
        context.kernels.mechanics_publish_clock(buffer=particles._buffer.buffer, offsets=offsets,
            time_s=time_s, reference_turns=reference_turns, t_sim=step)
    flag_monitor, monitor, buffer_monitor, offset_monitor = tracker._get_monitor(
        particles, turn_by_turn_monitor, num_turns)
    kernel, args = tracker._prepare_track_kernel_arguments(
        particles, flag_monitor=flag_monitor, buffer_monitor=buffer_monitor,
        offset_monitor=offset_monitor, multi_element_monitor=multi_element_monitor)
    args.update(num_turns=1, ele_start=0, num_ele_track=tracker.num_elements,
                flag_end_turn_actions=True)
    try:
        for _ in range(num_turns):
            runtime.apply(time_s+tracker.line.mechanics._time_origin_s, tracker.line.mechanics._time_series)
            publish()
            kernel(**args)
            # Keep the low part across calls without changing the Particles
            # layout. An externally edited time/reference/step starts a new
            # clock epoch at the published physical time.
            clock = clock + step
            time_s = float(clock)
            reference_turns += 1.
    finally:
        publish()
        runtime._clocks[particles] = (clock, reference_turns, step)
    tracker.record_last_track = monitor
    tracker.record_multi_element_last_track = multi_element_monitor
