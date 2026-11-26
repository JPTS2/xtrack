# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #
"""Read-only pose queries with normal Xsuite table selection."""

from weakref import ref

import numpy as np

from ..table import Table
from .motion import angles_from_matrix, checked_times


def pose_columns(nominal, reference, static, total):
    """World positions and finite relative rotations, never Euler subtraction."""
    from scipy.spatial.transform import Rotation

    data = {}
    for prefix, matrices in (
        ('nominal', nominal),
        ('reference', reference),
        ('static', static),
        ('total', total),
    ):
        for index, axis in enumerate('XYZ'):
            data[prefix + '_' + axis] = matrices[:, index, 3].copy()
        angles = angles_from_matrix(matrices[:, :3, :3])
        for index, angle in enumerate(('phi', 'theta', 'psi')):
            data[prefix + '_' + angle] = angles[:, index]
        data[prefix + '_matrix'] = matrices.copy()
    for prefix, before, after in (
        ('installation', reference, static),
        ('dynamic', static, total),
    ):
        for index, axis in enumerate('XYZ'):
            data[prefix + '_' + axis] = after[:, index, 3] - before[:, index, 3]
        relative = np.swapaxes(before[:, :3, :3], -1, -2) @ after[:, :3, :3]
        angles = angles_from_matrix(relative)
        for index, angle in enumerate(('phi', 'theta', 'psi')):
            data[prefix + '_' + angle] = angles[:, index]
        world_relative = after[:, :3, :3] @ np.swapaxes(before[:, :3, :3], -1, -2)
        data[prefix + '_rotvec_world'] = (
            Rotation.from_matrix(world_relative).as_rotvec()
            if len(total)
            else np.empty((0, 3))
        )
    return data


def deformation_columns(nominal_axes, layers, *, lever=None, mount_rotation=None):
    """Precise relative columns from local deformation data, not world poses.

    ``layers`` contains deliberate, installed and total ``(d, R-I)`` arrays.
    All are expressed about the same nominal body frames. An attached element
    additionally supplies its fixed body-local lever and orientation.
    """
    from scipy.spatial.transform import Rotation

    nominal_axes = np.asarray(nominal_axes)
    count = len(nominal_axes)
    lever = np.zeros((count, 3)) if lever is None else np.asarray(lever)
    mount = (
        np.broadcast_to(np.eye(3), (count, 3, 3))
        if mount_rotation is None
        else np.asarray(mount_rotation)
    )
    data = {}
    for prefix, before, after in (
        ('installation', layers[0], layers[1]),
        ('dynamic', layers[1], layers[2]),
    ):
        translation_before, rotation_before = before
        translation_after, rotation_after = after
        difference = rotation_after - rotation_before
        displacement = np.einsum(
            'bij,bj->bi',
            nominal_axes,
            translation_after
            - translation_before
            + np.einsum('bij,bj->bi', difference, lever),
        )
        for index, axis in enumerate('XYZ'):
            data[prefix + '_' + axis] = displacement[:, index]
        before_rotation = np.eye(3) + rotation_before
        relative_increment = (
            np.swapaxes(mount, -1, -2)
            @ np.swapaxes(before_rotation, -1, -2)
            @ difference
            @ mount
        )
        angles = angles_from_matrix(np.eye(3) + relative_increment)
        for index, angle in enumerate(('phi', 'theta', 'psi')):
            data[prefix + '_' + angle] = angles[:, index]
        world_increment = (
            nominal_axes
            @ difference
            @ np.swapaxes(before_rotation, -1, -2)
            @ np.swapaxes(nominal_axes, -1, -2)
        )
        data[prefix + '_rotvec_world'] = (
            Rotation.from_matrix(np.eye(3) + world_increment).as_rotvec()
            if count
            else np.empty((0, 3))
        )
    return data


def _deformation_arrays(poses, names):
    return (
        np.array([poses._deformations[name][0] for name in names]).reshape(-1, 3),
        np.array([poses._deformations[name][1] for name in names]).reshape(-1, 3, 3),
    )


def _attachment_mount(manager, attachment):
    from .local_deformations import attachment_reference

    body = manager[attachment['body']]
    nominal = body._nominal
    translation, rotation = attachment_reference(attachment)
    axes = attachment['ref_start'][:3, :3]
    lever = nominal[:3, :3].T @ (
        attachment['ref_start'][:3, 3] - nominal[:3, 3] + axes @ translation
    )
    mount_rotation = nominal[:3, :3].T @ attachment['base_element'][:3, :3]
    return lever, mount_rotation


def _static_inputs(body):
    """Small immutable case identity; never read lowered element/device fields."""
    from .model import _reference_values

    return tuple(_reference_values(body)) + tuple(body.installation.values())


def _table_static_case(owner, names):
    manager = owner.manager if hasattr(owner, 'manager') else owner
    case = {}
    for name in names:
        name = owner._attachments[name]['body'] if hasattr(owner, 'line') else name
        while name is not None and name not in case:
            body = manager[name]
            case[name] = _static_inputs(body)
            name = body.parent
    return case


def _table_live_inputs(owner, names):
    """Capture binding identities, not commands or a provider/history cache."""
    manager = owner.manager if hasattr(owner, 'manager') else owner
    direct = {}
    for name, binding in manager.sources.items():
        if getattr(binding, 'live', False):
            entry = (name, getattr(binding, 'role', 'disturbance'))
            for target in binding.targets:
                direct.setdefault(target, []).append(entry)
    if not direct:
        return {}
    inherited = {}
    result = {}
    for name in names:
        body_name = owner._attachments[name]['body'] if hasattr(owner, 'line') else name
        pending, current = [], body_name
        while current is not None and current not in inherited:
            pending.append(current)
            current = manager[current].parent
        for current in reversed(pending):
            parent = manager[current].parent
            entries = inherited.get(parent, ()) + tuple(direct.get(current, ()))
            inherited[current] = tuple(dict.fromkeys(entries))
        result[name] = inherited[body_name]
    return result


def mechanics_table(data, *, owner, time_s=None, time_series=None, disabled_sources=()):
    columns = list(data)
    manager = owner.manager if hasattr(owner, 'manager') else owner
    data = dict(
        data,
        _mechanics_owner=ref(owner),
        _mechanics_revision=manager._revision,
        _mechanics_record=time_series,
        _mechanics_static_case=_table_static_case(owner, data['name']),
        _mechanics_live_inputs=_table_live_inputs(owner, data['name']),
        disabled_sources=disabled_sources,
        query_time_s=time_s,
    )
    return MechanicsTable(data, col_names=columns)


def element_table(
    view,
    *,
    time_s=None,
    time_series=None,
    names=None,
    _source_values=None,
    disabled_sources=(),
):
    if time_s is None and time_series is not None:
        raise ValueError('Supply time_s when requesting a pose table')
    disabled_sources = view.manager._source_mask(disabled_sources)
    view._check()
    names = list(view._attachments) if names is None else list(names)
    attachments = [view._attachments[name] for name in names]
    data = dict(
        name=np.array(names, dtype=str),
        body=np.array([a['body'] for a in attachments], dtype=str),
        body_kind=np.array(
            [view.manager[a['body']].kind for a in attachments], dtype=str
        ),
        support_class=np.array(
            [view.manager[a['body']].support_class for a in attachments], dtype=str
        ),
        element_type=np.array(
            [type(view.line.element_dict[n]).__name__ for n in names], dtype=str
        ),
        length=np.array(
            [getattr(view.line.element_dict[n], 'length', 0.0) for n in names],
            dtype=float,
        ),
    )
    if view._solenoid_axes:
        data['solenoid_axis_body'] = np.array([
            view._solenoid_axes[name]['body'] if name in view._solenoid_axes else ''
            for name in names
        ], dtype=str)
    if time_s is not None:
        manager = view.manager
        disabled = tuple(manager.sources)
        body_names = [attachment['body'] for attachment in attachments]
        selected_bodies = list(dict.fromkeys(body_names))
        layers = [
            manager.get_body_poses(
                time_s=time_s,
                disabled_sources=disabled,
                include_installation=False,
                bodies=selected_bodies,
            ),
            manager.get_body_poses(
                time_s=time_s, disabled_sources=disabled, bodies=selected_bodies
            ),
            manager.get_body_poses(
                time_s=time_s,
                time_series=time_series,
                bodies=selected_bodies,
                _source_values=_source_values,
                disabled_sources=disabled_sources,
            ),
        ]
        nominal = np.array([a['ref_start'] for a in attachments]).reshape(-1, 4, 4)
        composed = []
        for poses in layers:
            composed.append(
                np.array(
                    [
                        (
                            poses[a['body']]
                            @ manager[a['body']].nominal_frame.inverse()
                        ).matrix
                        @ a['base_element']
                        for a in attachments
                    ]
                ).reshape(-1, 4, 4)
            )
        data.update(pose_columns(nominal, *composed))
        mounts = [_attachment_mount(manager, attachment) for attachment in attachments]
        axes = np.array(
            [manager[name]._nominal[:3, :3] for name in body_names]
        ).reshape(-1, 3, 3)
        data.update(
            deformation_columns(
                axes,
                [_deformation_arrays(pose, body_names) for pose in layers],
                lever=np.array([mount[0] for mount in mounts]).reshape(-1, 3),
                mount_rotation=np.array([mount[1] for mount in mounts]).reshape(
                    -1, 3, 3
                ),
            )
        )
    return mechanics_table(data, owner=view, time_s=time_s, time_series=time_series,
                           disabled_sources=disabled_sources)


class MechanicsTable(Table):
    """A pose/configuration snapshot, retaining context through row selection.

    Numeric columns remain readable snapshots. Live queries require the captured
    deliberate/installation inputs for the selected body and its ancestors,
    including for configuration tables made without time_s. Request a fresh
    table after changing that static case. Equal-value reapplication is allowed.
    Topology/source changes also invalidate context. No lattice history is cached.
    A point query with live inputs samples their current held state, not past
    applied commands. Inspect query_kind/live_state/live_source_names. Histories
    of such inputs require an explicit applied source recorder snapshot.
    """

    def _owner(self):
        context = self._data.get('_mechanics_owner')
        owner = None if context is None else context()
        if owner is None:
            raise ValueError(
                'No live mechanics context; request a fresh table from the model'
            )
        manager = owner.manager if hasattr(owner, 'manager') else owner
        if manager._revision != self._mechanics_revision:
            raise ValueError('Mechanics definitions changed; request a fresh table')
        return owner

    def to_dict(self, **kwargs):
        """Export a numeric snapshot without live model references/source records."""
        result = super().to_dict(**kwargs)
        for key in (
            '_mechanics_owner',
            '_mechanics_revision',
            '_mechanics_record',
            '_mechanics_static_case',
            '_mechanics_live_inputs',
        ):
            result['attrs'].pop(key, None)
        result['attrs'].update(
            query_kind=self.query_kind,
            live_state=self.live_state,
            live_inputs={name: self._live_inputs(name) for name in self.name},
        )
        return result

    def _live_inputs(self, name, *, disabled_sources=None):
        captured = self._data.get(
            '_mechanics_live_inputs', self._data.get('live_inputs', {})
        )
        if disabled_sources is None:
            disabled_sources = self._data.get('disabled_sources', ())
        return tuple(tuple(entry) for entry in captured.get(name, ())
                     if entry[0] not in disabled_sources)

    @property
    def live_source_names(self):
        """Live input bindings affecting this selection, without reading them."""
        return tuple(
            dict.fromkeys(
                source for name in self.name for source, _ in self._live_inputs(name)
            )
        )

    @property
    def live_source_roles(self):
        """Roles associated with the captured live binding identities."""
        return dict(entry for name in self.name for entry in self._live_inputs(name))

    @property
    def live_state(self):
        """Whether live inputs were unevaluated or sampled at their held state."""
        if not self.live_source_names:
            return 'not_live'
        return 'unevaluated' if self.query_time_s is None else 'current_held'

    @property
    def query_kind(self):
        """Configured point queries are not records of previously applied poses."""
        if self.query_time_s is None:
            return 'configuration'
        return 'current_held_pose' if self.live_source_names else 'prescribed_pose'

    def _selected_name(self, name):
        if name is None:
            if len(self) != 1:
                raise ValueError('Select one row or supply its exact name')
            name = self.name[0]
        if name not in self.name:
            raise KeyError(f'{name!r} is not in this table selection')
        return name

    def _check_static_case(self, owner, name):
        manager = owner.manager if hasattr(owner, 'manager') else owner
        name = owner._attachments[name]['body'] if hasattr(owner, 'line') else name
        captured = self._data.get('_mechanics_static_case', {})
        while name is not None:
            body = manager[name]
            if name not in captured or captured[name] != _static_inputs(body):
                raise ValueError(
                    'Mechanics static/reference case changed; request a fresh table'
                )
            name = body.parent

    def get_element(self, name=None):
        """Access a live element, provided this row's static case is unchanged.

        This does not apply query_time_s or return a frozen element copy. Fields
        reflect the currently applied time, and strength edits remain visible.
        This accessor does not evaluate live input providers.
        """
        owner = self._owner()
        if not hasattr(owner, 'line'):
            raise TypeError('Use line.mechanics.get_table() to select an element')
        name = self._selected_name(name)
        self._check_static_case(owner, name)
        owner._check()
        return owner.line.element_dict[name]

    def get_body(self, name=None):
        """Access a live body while the selected row's static case is unchanged."""
        owner = self._owner()
        name = self._selected_name(name)
        self._check_static_case(owner, name)
        if hasattr(owner, 'line'):
            return owner.manager[owner._attachments[name]['body']]
        return owner[name]

    def get_time_series(
        self,
        name=None,
        *,
        time_s=None,
        time_series=None,
        applied_history=None,
        disabled_sources=None,
        max_memory_bytes=128 * 1024**2,
    ):
        """Compose one selected entity over explicit times, on the host.

        If a sampled source record is bound to this table, its grid is the
        default. This includes deliberate, installation and inherited dynamic
        transforms; it differs from a source record's direct-input get_table.
        Live inputs require applied_history from record_applied_sources; current
        provider values are never used to reconstruct their past.
        """
        owner = self._owner()
        name = self._selected_name(name)
        self._check_static_case(owner, name)
        manager = owner.manager if hasattr(owner, 'manager') else owner
        body_name = owner._attachments[name]['body'] if hasattr(owner, 'line') else name
        mask = (self._data.get('disabled_sources', ()) if disabled_sources is None
                else manager._source_mask(disabled_sources))
        if applied_history is not None:
            from .applied_history import AppliedSourceHistory

            if not isinstance(applied_history, AppliedSourceHistory):
                raise TypeError('Supply an immutable applied source history snapshot')
            applied_history._check(manager, bodies=[body_name])
            if (mask or disabled_sources is not None) and set(mask) != set(applied_history.disabled_sources):
                raise ValueError('Applied history fixes the source mask')
            mask = applied_history.disabled_sources
            required = {source for source, _ in self._live_inputs(name, disabled_sources=mask)}
            if not required.issubset(applied_history.values):
                raise ValueError(
                    'Applied history does not cover all selected live inputs'
                )
            if (
                time_series is not None
                and time_series is not applied_history.prescribed_record
            ):
                raise ValueError(
                    'Supplied prescribed record differs from the applied history'
                )
            record = applied_history.prescribed_record
        else:
            record = self._mechanics_record if time_series is None else time_series
            live = self._live_inputs(name, disabled_sources=mask)
            if live:
                sources = ', '.join(source for source, _ in live)
                raise ValueError(
                    f'Cannot reconstruct history of live mechanics inputs ({sources}): '
                    'no recorded applied history is available. Inspect the numeric '
                    'snapshot or a current-held point pose instead.'
                )
        if time_s is None:
            if applied_history is not None:
                time_s = applied_history.time_s
            elif record is None:
                raise ValueError('Supply time_s or a sampled time_series record')
            else:
                time_s = record.time_s
        times = checked_times(time_s)
        if applied_history is not None:
            applied_history.at(float(times[0]))
            applied_history.at(float(times[-1]))
        if not isinstance(max_memory_bytes, int) or max_memory_bytes <= 0:
            raise ValueError('max_memory_bytes must be a positive integer')
        # Conservative bound for one output table and transient query columns.
        # Only one time slice is composed at a time; no all-body history cache.
        if 2048 * len(times) > max_memory_bytes:
            raise MemoryError('Selected pose history exceeds max_memory_bytes')
        if record is not None:
            record._check(manager)

        def replay(time):
            pose_time, values = (time, None) if applied_history is None else applied_history._sample(time)
            return dict(time_s=pose_time, _source_values=values, disabled_sources=mask)

        first = replay(float(times[0]))
        pose_times = np.empty_like(times)
        pose_times[0] = first['time_s']

        if hasattr(owner, 'line'):
            current = element_table(
                owner,
                time_series=record,
                names=[name],
                **first,
            )
            manager = owner.manager
            attachment = owner._attachments[name]
            body = manager[attachment['body']]
            mount = body.nominal_frame.inverse().matrix @ attachment['base_element']
        else:
            current = owner.survey(
                time_series=record,
                bodies=[name],
                **first,
            )
            manager = owner
        total = np.empty((len(times), 4, 4))
        total[0] = current.total_matrix[0]
        deformations = [
            manager._get_body_deformations(
                time_series=record,
                bodies=[body_name],
                **first,
            )[body_name]
        ]
        for index, time in enumerate(times[1:], 1):
            current_inputs = replay(float(time))
            pose_times[index] = current_inputs['time_s']
            poses = manager.get_body_poses(
                time_series=record,
                bodies=[body_name],
                **current_inputs,
            )
            deformations.append(poses._deformations[body_name])
            total[index] = (
                poses[attachment['body']].matrix @ mount
                if hasattr(owner, 'line')
                else poses[name].matrix
            )
        layers = [
            np.broadcast_to(current[prefix + '_matrix'][0], total.shape)
            for prefix in ('nominal', 'reference', 'static')
        ]
        output = pose_columns(*layers, total)
        disabled = tuple(manager.sources)
        deliberate = manager._get_body_deformations(
            time_s=float(times[0]),
            disabled_sources=disabled,
            include_installation=False,
            bodies=[body_name],
        )[body_name]
        installed = manager._get_body_deformations(
            time_s=float(times[0]), disabled_sources=disabled, bodies=[body_name]
        )[body_name]
        packed = [
            (
                np.broadcast_to(layer[0], (len(times), 3)),
                np.broadcast_to(layer[1], (len(times), 3, 3)),
            )
            for layer in (deliberate, installed)
        ]
        packed.append(
            (
                np.array([layer[0] for layer in deformations]),
                np.array([layer[1] for layer in deformations]),
            )
        )
        body_axes = np.broadcast_to(
            manager[body_name]._nominal[:3, :3], (len(times), 3, 3)
        )
        if hasattr(owner, 'line'):
            lever, mount_rotation = _attachment_mount(manager, attachment)
            kwargs = dict(
                lever=np.broadcast_to(lever, (len(times), 3)),
                mount_rotation=np.broadcast_to(mount_rotation, (len(times), 3, 3)),
            )
        else:
            kwargs = {}
        output.update(deformation_columns(body_axes, packed, **kwargs))
        data = dict(time_s=times.copy(), pose_time_s=pose_times, **output)
        return MotionHistoryTable(
            dict(
                data,
                entity_name=name,
                disabled_sources=mask,
                history_kind=(
                    'applied_live_inputs'
                    if applied_history is not None
                    else 'prescribed_inputs'
                ),
            ),
            col_names=list(data),
            index='time_s',
        )

    def animate(
        self,
        name=None,
        *,
        time_s,
        size_m=None,
        time_series=None,
        applied_history=None,
        disabled_sources=None,
        displacement_scale=1.0,
        rotation_scale=1.0,
        baseline='static',
        interval_ms=40,
    ):
        """Animate one rigid cuboid using its actual composed reference frames.

        size_m is local (width x, height y, length s), in metres. For an element,
        omitted size uses its length and an illustrative 0.3 m square envelope.
        The cuboid is presentation geometry, not a magnetic/aperture model.
        """
        from .plotting import animate_cuboid

        owner = self._owner()
        name = self._selected_name(name)
        if size_m is None:
            if not hasattr(owner, 'line'):
                raise ValueError('Supply size_m for a body cuboid')
            size_m = (0.3, 0.3, float(getattr(self.get_element(name), 'length', 0.0)))
        return animate_cuboid(
            self,
            name,
            time_s=time_s,
            size_m=size_m,
            time_series=time_series,
            applied_history=applied_history,
            disabled_sources=disabled_sources,
            displacement_scale=displacement_scale,
            rotation_scale=rotation_scale,
            baseline=baseline,
            interval_ms=interval_ms,
        )


class MotionHistoryTable(Table):
    """One entity's poses composed from prescribed inputs at physical seconds.

    This table does not acquire or retrospectively sample live command providers.
    """

    def __init__(self, data, col_names=None, index='time_s', **kwargs):
        super().__init__(data, col_names=col_names, index=index, **kwargs)

    def plot(self):
        """Plot dynamic world translations and rotations relative to static axes."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
        for axis in 'XYZ':
            axes[0].plot(self.time_s, self['dynamic_' + axis] * 1e9, label=axis)
        for angle, label in (('phi', 'Pitch'), ('theta', 'Yaw'), ('psi', 'Roll')):
            axes[1].plot(self.time_s, self['dynamic_' + angle] * 1e9, label=label)
        axes[0].set(ylabel='Dynamic world displacement [nm]', title=self.entity_name)
        axes[1].set(
            xlabel='Physical time [s]', ylabel='Rotation from installed axes [nrad]'
        )
        for ax in axes:
            ax.legend()
            ax.grid(alpha=0.2)
        fig.tight_layout()
        return fig, axes
