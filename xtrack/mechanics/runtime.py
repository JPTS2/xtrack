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
from dataclasses import dataclass
from weakref import WeakKeyDictionary

import numpy as np
import xobjects as xo
from ..composer.positions import CompensatedFloat

from .local_deformations import attachment_reference, rotation_increment, static_local
from .motion import RigidMotion, sample_at
from .sources import (
    Harmonic, PlaneWave, prepare_spectral_receivers, requires_time_series,
)
from .context_arrays import to_context, upload_into


FIELDS = (
    'shift_x',
    'shift_y',
    'shift_s',
    'rot_x_rad',
    'rot_y_rad',
    'rot_s_rad_no_frame',
    'rot_shift_anchor',
)

# Bound receiver coefficients on the host. This storage is independent of
# track duration and is never copied to the tracking context.
_SPECTRAL_RECEIVER_BUDGET = 32 * 1024**2


def _compile(context):
    if all(
        name in context.kernels
        for name in (
            'mechanics_body_update',
            'mechanics_lower',
            'mechanics_lower_solenoid_axis',
            'mechanics_sources',
            'mechanics_sources_mixed',
            'mechanics_source_values',
            'mechanics_publish_clock',
        )
    ):
        return

    def arg(dtype, name):
        return xo.Arg(dtype, name=name, pointer=True)

    body = [xo.Arg(xo.Int64, name='n')]
    body += [arg(xo.Int64, n) for n in ('paths_offsets', 'paths')]
    body += [arg(xo.Float64, n) for n in ('relations', 'static_local', 'dynamic_local')]
    body += [arg(xo.Float64, 'out')]
    lower = [xo.Arg(xo.Int64, name='n'), arg(xo.Int64, 'body_indices')]
    lower += [arg(xo.Float64, n) for n in ('geometry', 'baseline')]
    lower += [arg(xo.Int64, 'offsets'), arg(xo.Float64, 'bodyout')]
    lower += [arg(xo.Int8, 'buffer'), arg(xo.Float64, 'values')]
    axis_lower = [xo.Arg(xo.Int64, name='n'), arg(xo.Int64, 'body_indices')]
    axis_lower += [arg(xo.Float64, n) for n in ('projection', 'baseline')]
    axis_lower += [arg(xo.Int64, 'offsets')]
    axis_lower += [arg(xo.Float64, n) for n in ('static_local', 'dynamic_local')]
    axis_lower += [arg(xo.Int8, 'buffer'), arg(xo.Float64, 'values')]
    context.add_kernels(
        sources=[Path(__file__).parent / 'runtime_src/update.h'],
        kernels={
            'mechanics_body_update': xo.Kernel(args=body, n_threads='n'),
            'mechanics_lower': xo.Kernel(args=lower, n_threads='n'),
            'mechanics_lower_solenoid_axis': xo.Kernel(args=axis_lower, n_threads='n'),
            'mechanics_sources': xo.Kernel(
                args=[
                    xo.Arg(xo.Int64, name='n'),
                    arg(xo.Int64, 'source_starts'),
                    arg(xo.Float64, 'source_data'),
                    xo.Arg(xo.Float64, name='time_s'),
                    arg(xo.Float64, 'dynamic_local'),
                ],
                n_threads='n',
            ),
            'mechanics_publish_clock': xo.Kernel(
                args=[
                    arg(xo.Int8, 'buffer'),
                    arg(xo.Int64, 'offsets'),
                    xo.Arg(xo.Float64, name='time_s'),
                    xo.Arg(xo.Float64, name='reference_turns'),
                    xo.Arg(xo.Float64, name='t_sim'),
                ],
                n_threads=1,
            ),
            'mechanics_source_values': xo.Kernel(
                args=[
                    xo.Arg(xo.Int64, name='n'),
                    arg(xo.Int64, 'source_starts'),
                    arg(xo.Int64, 'source_rows'),
                    arg(xo.Float64, 'values'),
                    arg(xo.Float64, 'dynamic_local'),
                ],
                n_threads='n',
            ),
            'mechanics_sources_mixed': xo.Kernel(
                args=[
                    xo.Arg(xo.Int64, name='n'),
                    arg(xo.Int64, 'source_starts'),
                    arg(xo.Int64, 'source_rows'),
                    arg(xo.Float64, 'source_data'),
                    arg(xo.Float64, 'sample_values'),
                    xo.Arg(xo.Float64, name='time_s'),
                    arg(xo.Float64, 'dynamic_local'),
                ],
                n_threads='n',
            ),
        },
    )


@dataclass(frozen=True)
class _ConsumedSource:
    name: str
    binding: object
    indices: np.ndarray
    selected: np.ndarray
    positions: np.ndarray
    axes: np.ndarray


def has_fixed_reference(tracker):
    classes = {type(element) for element in tracker._tracker_data_base._elements}
    return all(
        cls.__module__.startswith(('xtrack.beam_elements', 'xtrack.monitors'))
        and cls.__name__ not in (
            'ReferenceEnergyIncrease', 'ReferenceEnergyChange', 'LineSegmentMap',
        )
        for cls in classes
    )


class PreparedMechanics:
    """One current-pose buffer, never an implicit entire-run cache."""

    def __init__(self, view):
        from .model import _check_transverse_values

        self.view = view
        self.manager = view.manager
        self.revision = self.manager._revision
        self.buffer = view.line.tracker._buffer
        self.context = self.buffer.context
        self._clocks = WeakKeyDictionary()
        self._source_modes = self._current_source_modes()
        self.fixed_reference = has_fixed_reference(view.line.tracker)
        _compile(self.context)
        needed = self.manager._body_closure(view._consumed_bodies())
        self.names = tuple(name for name in self.manager.bodies if name in needed)
        self.indices = {name: i for i, name in enumerate(self.names)}
        self.transverse_indices = [
            i for i, name in enumerate(self.names) if self.manager[name]._transverse_only
        ]
        n = len(self.names)
        relations, static = np.zeros((n, 12)), np.zeros((n, 12))
        self._static_settings = np.zeros((n, 12))
        paths, starts = [], [0]
        for name in self.names:
            body = self.manager[name]
            i = self.indices[name]
            chain, current = [], body
            while current is not None:
                chain.append(self.indices[current.name])
                current = (
                    None if current.parent is None else self.manager[current.parent]
                )
            paths.extend(reversed(chain))
            starts.append(len(paths))
            relations[i, :9] = np.eye(3).ravel()
            if body.parent is not None:
                parent = self.manager[body.parent]._nominal
                relations[i, :9] = (parent[:3, :3].T @ body._nominal[:3, :3]).ravel()
                relations[i, 9:] = parent[:3, :3].T @ (
                    body._nominal[:3, 3] - parent[:3, 3]
                )
            values = np.array(
                [
                    body.shift_x,
                    body.shift_y,
                    body.shift_s,
                    body.rot_x_rad,
                    body.rot_y_rad,
                    body.rot_s_rad,
                ]
            )
            if np.iscomplexobj(values):
                raise ValueError(f'{name}: local alignment must be real')
            if not np.all(np.isfinite(values)):
                raise ValueError(f'{name}: local alignment must be finite')
            installation = body.installation.values()
            _check_transverse_values(body, values)
            _check_transverse_values(body, installation)
            self._static_settings[i] = np.concatenate((values, installation))
            d, e = static_local(values, installation)
            static[i, :3], static[i, 3:] = d, e.ravel()
        self.attachments = list(view._attachments.values())
        a = len(self.attachments)
        geometry, baseline, offsets = (
            np.zeros((a, 24)),
            np.zeros((a, 7)),
            np.zeros((a, 7), dtype=np.int64),
        )
        body_indices = []
        for i, attachment in enumerate(self.attachments):
            body = self.manager[attachment['body']]
            body_indices.append(self.indices[body.name])
            t0, r0 = attachment_reference(attachment)
            reference = attachment['ref_start']
            axes = reference[:3, :3].T @ body._nominal[:3, :3]
            lever = (
                body._nominal[:3, :3].T @ (reference[:3, 3] - body._nominal[:3, 3])
                + axes.T @ t0
            )
            geometry[i] = np.concatenate((axes.ravel(), lever, t0, r0.ravel()))
            baseline[i] = attachment['baseline']
            element = view.line.element_dict[attachment['element']]
            if element._buffer is not self.buffer:
                raise ValueError(
                    'Prepared mechanics requires elements in the tracker buffer'
                )
            offsets[i] = [element._xobject._get_offset(field) for field in FIELDS]
        def array(value):
            return to_context(self.context, value)

        self.paths_offsets, self.paths = (
            array(np.array(starts, dtype=np.int64)),
            array(np.array(paths, dtype=np.int64)),
        )
        self.relations, self.static_local = (
            array(relations.ravel()),
            array(static.ravel()),
        )
        self.geometry, self.baseline = array(geometry.ravel()), array(baseline.ravel())
        self.offsets, self.body_indices = (
            array(offsets.ravel()),
            array(np.array(body_indices, dtype=np.int64)),
        )
        self.input = np.zeros((n, 12))
        self.dynamic_local = array(self.input.ravel())
        self.disabled_sources = view._disabled_tracking_sources()
        self.sources = []
        for name, binding in self.manager.sources.items():
            if not view._source_enabled(binding):
                continue
            selected = np.array(
                [
                    index
                    for index, target in enumerate(binding.targets)
                    if target in self.indices
                ],
                dtype=np.int64,
            )
            if not len(selected):
                continue
            indices = np.array(
                [self.indices[binding.targets[index]] for index in selected],
                dtype=np.int64,
            )
            # Keep the original receiver set for every relevant binding.
            # Custom stochastic realizations may depend on its length/order.
            frames = np.array(
                [self.manager[target]._nominal for target in binding.targets]
            )
            self.sources.append(
                _ConsumedSource(
                    name,
                    binding,
                    indices,
                    selected,
                    frames[:, :3, 3],
                    frames[:, :3, :3],
                )
            )
        self._dispatch_key = None
        self._dispatch_record = object()
        self._dispatch_values = None
        self._spectral_receivers = {}
        self._spectral_receiver_bytes = 0
        self._prepare_dispatch(None)
        self.bodyout = self.context.zeros(n * 12, dtype=np.float64)
        self.values = array(
            np.array([item['last_written'] for item in self.attachments]).ravel()
        )
        self.solenoid_axes = list(view._solenoid_axes.values())
        if self.solenoid_axes:
            self.axis_indices = array(np.array(
                [self.indices[a['body']] for a in self.solenoid_axes], dtype=np.int64,
            ))
            self.axis_projection = array(np.array(
                [a['projection'] for a in self.solenoid_axes]
            ).ravel())
            self.axis_baseline = array(np.array(
                [a['baseline'] for a in self.solenoid_axes]
            ).ravel())
            self.axis_offsets = array(np.array([
                [view.line.element_dict[a['element']]._xobject._get_offset(field)
                 for field in ('x0', 'y0')]
                for a in self.solenoid_axes
            ], dtype=np.int64).ravel())
            self.axis_values = array(np.array(
                [a['last_written'] for a in self.solenoid_axes]
            ).ravel())
        self.applied = False
        from .geometry_guard import GeometryGuard

        self.guard = GeometryGuard(view)

    def _prepare_dispatch(self, time_series):
        """Bind native and uploaded rows once for a particular record layout."""
        values = None if time_series is None else time_series.values
        if time_series is self._dispatch_record and values is self._dispatch_values:
            return
        # A composed exact provider may also contain finite Fourier replay
        # belonging to this particular record, not just continuous modes.
        self._spectral_receivers.clear()
        self._spectral_receiver_bytes = 0
        recorded = frozenset() if time_series is None else frozenset(time_series.values)
        exact = frozenset() if time_series is None else frozenset(
            name for name in recorded if time_series.replay_modes[name] == 'source'
        )
        layout = (recorded, exact)
        self._dispatch_record, self._dispatch_values = time_series, values
        if layout == self._dispatch_key:
            return
        rows = [[] for _ in self.names]
        self.row_slices = {}
        self.host_sources = []
        self.native_sources = []
        self._bounded_native = False
        self.live_values = {}
        offset = 0
        empty_analytic = (0.0,) * 8
        for item in self.sources:
            binding, source = item.binding, item.binding.source
            native = (
                not binding.live
                and type(source) in (Harmonic, PlaneWave)
                and (item.name not in recorded or item.name in exact)
            )
            if not native:
                self.host_sources.append(item)
                if binding.live:
                    self.live_values[item.name] = np.empty((len(item.positions), 6))
                self.row_slices[item.name] = slice(offset, offset + len(item.indices))
                for index, body in enumerate(item.indices):
                    rows[body].append((offset + index, empty_analytic))
                offset += len(item.indices)
                continue
            self.native_sources.append(item)
            self._bounded_native |= item.name in recorded
            omega = 2 * np.pi * source.frequency_hz
            if type(source) is Harmonic:
                component = ('x', 'y', 's', 'pitch', 'yaw', 'roll').index(
                    binding.component
                )
                for body in item.indices:
                    rows[body].append(
                        (
                            -1,
                            [
                                component,
                                source.amplitude,
                                omega,
                                source.phase_rad,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                            ],
                        )
                    )
            else:
                positions, axes = (
                    item.positions[item.selected],
                    item.axes[item.selected],
                )
                delay = (
                    (positions - source.origin_m) @ np.asarray(source.direction)
                    / source.speed_m_s
                )
                polarization = np.einsum('bji,j->bi', axes, source.polarization)
                for body, delay_s, direction in zip(item.indices, delay, polarization):
                    rows[body].append(
                        (-1, [-1, source.amplitude, omega, source.phase_rad,
                              delay_s, *direction])
                    )
        def array(value):
            return to_context(self.context, value)

        self.source_starts = array(
            np.concatenate(([0], np.cumsum([len(row) for row in rows]))).astype(
                np.int64
            )
        )
        entries = [entry for row in rows for entry in row]
        data = (
            np.array([entry[1] for entry in entries], dtype=float).ravel()
            if self.native_sources and entries
            else np.zeros(1)
        )
        self.source_data = array(data)
        self.analytic = not self.host_sources
        self.raw_values = None
        self.sample_values = None
        self.source_rows = None
        if not self.analytic:
            self.source_rows = array(
                np.array([entry[0] for entry in entries], dtype=np.int64)
            )
            self.raw_values = np.zeros((offset, 6))
            self.sample_values = array(self.raw_values.ravel())
        self._dispatch_key = layout

    def _current_source_modes(self):
        return (
            self.view._source_mode_configured,
            self.view._vibrations_enabled,
            self.view._controls_enabled,
            self.view._tracking_source_mask,
        )

    def _clock_for(self, particles, time_s, reference_turns, step):
        previous = self._clocks.get(particles)
        if (
            previous is not None and float(previous[0]) == time_s
            and previous[1] == reference_turns and previous[2] == step
        ):
            return previous[0]
        return CompensatedFloat(time_s)

    def _check_state(self):
        if (
            self.manager._revision != self.revision
            or self.view.line.tracker._buffer is not self.buffer
        ):
            raise ValueError(
                'Mechanics definition changed inside a tracking call; prepare a new call'
            )
        if self._current_source_modes() != self._source_modes:
            raise ValueError(
                'Mechanics source modes changed inside a tracking call; restart tracking'
            )

    def refresh_static(self):
        """Read mutable body settings once per call, without world poses."""
        values = np.array(
            [
                [
                    body.shift_x,
                    body.shift_y,
                    body.shift_s,
                    body.rot_x_rad,
                    body.rot_y_rad,
                    body.rot_s_rad,
                ]
                for body in (self.manager[name] for name in self.names)
            ]
        )
        installation = np.array(
            [
                [
                    body.installation.shift_x,
                    body.installation.shift_y,
                    body.installation.shift_s,
                    body.installation.rot_x_rad,
                    body.installation.rot_y_rad,
                    body.installation.rot_s_rad,
                ]
                for body in (self.manager[name] for name in self.names)
            ]
        )
        if np.iscomplexobj(values) or np.iscomplexobj(installation):
            raise ValueError('Body local alignments must be real')
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(installation)):
            raise ValueError('Body local alignments must be finite')
        if self.transverse_indices and (
            np.any(values[self.transverse_indices, 2:])
            or np.any(installation[self.transverse_indices, 2:])
        ):
            raise ValueError('Relative solenoid axis supports only transverse x/y translations')
        settings = np.column_stack((values, installation))
        if np.array_equal(settings, self._static_settings):
            return
        left = rotation_increment(*values[:, 3:].T)
        right = rotation_increment(*installation[:, 3:].T)
        data = np.empty_like(self.input)
        data[:, :3] = (
            values[:, :3]
            + installation[:, :3]
            + np.einsum('bij,bj->bi', left, installation[:, :3])
        )
        data[:, 3:] = (left + right + left @ right).reshape(-1, 9)
        upload_into(self.context, self.static_local, data.ravel())
        self._static_settings = settings

    def apply(self, time_s, time_series=None):
        self._check_state()
        if not np.isscalar(time_s) or np.iscomplexobj(time_s) or not np.isfinite(time_s):
            raise ValueError('time_s must be a finite scalar in seconds')
        self._prepare_dispatch(time_series)
        if self._bounded_native:
            time_series._check_time(time_s)
        live_inputs = {}
        if self.analytic:
            self.view._preflight_applied(
                time_s,
                live_inputs,
                disabled_sources=self.disabled_sources,
                time_series=time_series,
            )
            self.context.kernels.mechanics_sources(
                n=len(self.names),
                source_starts=self.source_starts,
                source_data=self.source_data,
                time_s=time_s,
                dynamic_local=self.dynamic_local,
            )
            self._apply_kernels()
            self.view._remember_applied(
                time_s,
                live_inputs,
                disabled_sources=self.disabled_sources,
                time_series=time_series,
            )
            return
        for item in self.host_sources:
            name, binding = item.name, item.binding
            source = binding.source
            recorded = (
                not binding.live
                and time_series is not None
                and name in time_series.values
            )
            spectral = None
            exact = recorded and time_series.replay_modes[name] == 'source'
            if exact:
                time_series._check_time(time_s)
            if not binding.live and (exact or (
                not recorded and not requires_time_series(source)
            )):
                if name not in self._spectral_receivers:
                    provider = time_series._providers[name] if exact else source
                    prepared, size = prepare_spectral_receivers(
                        provider, item.positions,
                        max_bytes=_SPECTRAL_RECEIVER_BUDGET-self._spectral_receiver_bytes,
                    )
                    self._spectral_receivers[name] = prepared
                    self._spectral_receiver_bytes += size
                spectral = self._spectral_receivers[name]
            if spectral is not None:
                values = self.manager._evaluate_source(
                    binding, np.array([time_s]), positions=item.positions,
                    axes=item.axes, provider=spectral,
                )[0]
            elif recorded:
                values = time_series._sample(
                    binding, time_s, positions=item.positions, axes=item.axes,
                )
            elif requires_time_series(source):
                raise ValueError(
                    f'{name}: supply a computed motion record for this source'
                )
            elif type(source) is RigidMotion:
                values = sample_at(source.time_s, source.values, time_s)
            else:
                values = self.manager._evaluate_source(
                    binding, np.array([time_s]), positions=item.positions, axes=item.axes,
                )[0]
            if (
                values.shape != (len(item.positions), 6)
                or np.iscomplexobj(values)
                or not np.all(np.isfinite(values))
            ):
                raise ValueError(
                    f'{name}: source must return finite real motion with matching targets'
                )
            binding._check_transverse(values)
            if binding.live:
                # Providers may reuse one ndarray across bindings. Own the full
                # normalized block before another evaluation can overwrite it.
                np.copyto(self.live_values[name], values)
                live_inputs[name] = self.live_values[name]
            self.raw_values[self.row_slices[name]] = values[item.selected]
        self._check_state()
        self.view._preflight_applied(
            time_s,
            live_inputs,
            disabled_sources=self.disabled_sources,
            time_series=time_series,
        )
        upload_into(self.context, self.sample_values, self.raw_values.ravel())
        if self.native_sources:
            self.context.kernels.mechanics_sources_mixed(
                n=len(self.names),
                source_starts=self.source_starts,
                source_rows=self.source_rows,
                source_data=self.source_data,
                sample_values=self.sample_values,
                time_s=time_s,
                dynamic_local=self.dynamic_local,
            )
        else:
            self.context.kernels.mechanics_source_values(
                n=len(self.names),
                source_starts=self.source_starts,
                source_rows=self.source_rows,
                values=self.sample_values,
                dynamic_local=self.dynamic_local,
            )
        self._apply_kernels()
        self.view._remember_applied(
            time_s,
            live_inputs,
            disabled_sources=self.disabled_sources,
            time_series=time_series,
        )

    def _apply_kernels(self):
        self.context.kernels.mechanics_body_update(
            n=len(self.names),
            paths_offsets=self.paths_offsets,
            paths=self.paths,
            relations=self.relations,
            static_local=self.static_local,
            dynamic_local=self.dynamic_local,
            out=self.bodyout,
        )
        self.context.kernels.mechanics_lower(
            n=len(self.attachments),
            body_indices=self.body_indices,
            geometry=self.geometry,
            baseline=self.baseline,
            offsets=self.offsets,
            bodyout=self.bodyout,
            buffer=self.buffer.buffer,
            values=self.values,
        )
        if self.solenoid_axes:
            self.context.kernels.mechanics_lower_solenoid_axis(
                n=len(self.solenoid_axes),
                body_indices=self.axis_indices,
                projection=self.axis_projection,
                baseline=self.axis_baseline,
                offsets=self.axis_offsets,
                static_local=self.static_local,
                dynamic_local=self.dynamic_local,
                buffer=self.buffer.buffer,
                values=self.axis_values,
            )
        self.applied = True

    def finish(self):
        """Synchronize conflict-detection values once per call, in one read."""
        if self.applied:
            values = self.context.nparray_from_context_array(self.values).reshape(-1, 7)
            for attachment, value in zip(self.attachments, values):
                attachment['last_written'] = value.copy()
            if self.solenoid_axes:
                axis_values = self.context.nparray_from_context_array(self.axis_values).reshape(-1, 2)
                for attachment, value in zip(self.solenoid_axes, axis_values):
                    attachment['last_written'] = value.copy()
            self.applied = False


def can_track_prepared(tracker, particles, kwargs):
    """Conservative fixed-reference scheduler eligibility; others retain fallback."""
    allowed = {
        'ele_start',
        'ele_stop',
        'num_elements',
        'num_turns',
        'turn_by_turn_monitor',
        'freeze_longitudinal',
        'backtrack',
        'log',
        '_reset_log',
        '_force_no_end_turn_actions',
    }
    if set(kwargs) - allowed:
        return False
    monitor = kwargs.get('turn_by_turn_monitor')
    if (
        tracker.iscollective
        or tracker.line.energy_program is not None
        or tracker.line.enable_time_dependent_vars
        or tracker._sync_time_controller is not None
        or tracker.skip_end_turn_actions
        or not tracker.reset_s_at_end_turn
        or tracker.enable_pipeline_hold
        or particles.start_tracking_at_element >= 0
        or kwargs.get('ele_start') not in (None, 0)
        or kwargs.get('ele_stop') is not None
        or kwargs.get('num_elements') is not None
        or kwargs.get('backtrack', False)
        or kwargs.get('freeze_longitudinal', False)
        or kwargs.get('log') is not None
        or kwargs.get('_force_no_end_turn_actions', False)
        or monitor == 'ONE_TURN_EBE'
        or getattr(monitor, 'ebe_mode', 0) == 1
        or tracker.track_flags.XS_FLAG_BACKTRACK
        or not tracker.line.mechanics._runtime.fixed_reference
    ):
        return False
    # A shared reference clock requires a common reference speed. Reduce on
    # context and transfer one result, rather than copying the particle array.
    beta = float(particles.beta0[0])
    if not np.isfinite(beta) or beta <= 0:
        return False
    # Spare capacity uses sentinel beta0 values. Lost particles also no longer
    # consume the moving maps, so neither should force a host-scheduled fallback.
    # Keep beta0[0] as the reference, matching Tracker's shared-clock convention.
    uniform = ((particles.state <= 0) | (particles.beta0 == beta)).all()
    if hasattr(uniform, 'get'):
        uniform = uniform.get()
    return bool(uniform)


def track_prepared(
    tracker,
    particles,
    *,
    num_turns=1,
    turn_by_turn_monitor=None,
    multi_element_monitor=None,
    **kwargs,
):
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
    time_s, reference_turns, step = (
        float(getattr(particles, name))
        for name in ('time_s', 'reference_turns', 't_sim')
    )
    runtime = tracker.line.mechanics._runtime
    clock = runtime._clock_for(particles, time_s, reference_turns, step)
    context = tracker._context
    offsets = to_context(
        context,
        np.array(
            [
                particles._xobject._get_offset(name)
                for name in ('time_s', 'reference_turns', 't_sim')
            ],
            dtype=np.int64,
        ),
    )

    def publish():
        context.kernels.mechanics_publish_clock(
            buffer=particles._buffer.buffer,
            offsets=offsets,
            time_s=time_s,
            reference_turns=reference_turns,
            t_sim=step,
        )

    flag_monitor, monitor, buffer_monitor, offset_monitor = tracker._get_monitor(
        particles, turn_by_turn_monitor, num_turns
    )
    kernel, args = tracker._prepare_track_kernel_arguments(
        particles,
        flag_monitor=flag_monitor,
        buffer_monitor=buffer_monitor,
        offset_monitor=offset_monitor,
        multi_element_monitor=multi_element_monitor,
    )
    args.update(
        num_turns=1,
        ele_start=0,
        num_ele_track=tracker.num_elements,
        flag_end_turn_actions=True,
    )
    try:
        for _ in range(num_turns):
            runtime.apply(
                time_s + tracker.line.mechanics._time_origin_s,
                tracker.line.mechanics._time_series,
            )
            publish()
            kernel(**args)
            # Keep the low part across calls without changing the Particles
            # layout. An externally edited time/reference/step starts a new
            # clock epoch at the published physical time.
            clock = clock + step
            time_s = float(clock)
            reference_turns += 1.0
    finally:
        publish()
        runtime._clocks[particles] = (clock, reference_turns, step)
    tracker.record_last_track = monitor
    tracker.record_multi_element_last_track = multi_element_monitor
