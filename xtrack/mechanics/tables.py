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
    for prefix, matrices in (('nominal', nominal), ('reference', reference),
                              ('static', static), ('total', total)):
        for index, axis in enumerate('XYZ'):
            data[prefix+'_'+axis] = matrices[:, index, 3].copy()
        angles = angles_from_matrix(matrices[:, :3, :3])
        for index, angle in enumerate(('phi', 'theta', 'psi')):
            data[prefix+'_'+angle] = angles[:, index]
        data[prefix+'_matrix'] = matrices.copy()
    for prefix, before, after in (('installation', reference, static), ('dynamic', static, total)):
        for index, axis in enumerate('XYZ'):
            data[prefix+'_'+axis] = after[:, index, 3]-before[:, index, 3]
        relative = np.swapaxes(before[:, :3, :3], -1, -2) @ after[:, :3, :3]
        angles = angles_from_matrix(relative)
        for index, angle in enumerate(('phi', 'theta', 'psi')):
            data[prefix+'_'+angle] = angles[:, index]
        world_relative = after[:, :3, :3] @ np.swapaxes(before[:, :3, :3], -1, -2)
        data[prefix+'_rotvec_world'] = (Rotation.from_matrix(world_relative).as_rotvec()
                                      if len(total) else np.empty((0, 3)))
    return data


def mechanics_table(data, *, owner, time_s=None, time_series=None):
    columns = list(data)
    manager = owner.manager if hasattr(owner, 'manager') else owner
    data = dict(data, _mechanics_owner=ref(owner), _mechanics_revision=manager._revision,
                _mechanics_record=time_series, query_time_s=time_s)
    return MechanicsTable(data, col_names=columns)


def element_table(view, *, time_s=None, time_series=None, names=None):
    if time_s is None and time_series is not None:
        raise ValueError('Supply time_s when requesting a pose table')
    view._check()
    names = list(view._attachments) if names is None else list(names)
    attachments = [view._attachments[name] for name in names]
    data = dict(name=np.array(names, dtype=str),
                body=np.array([a['body'] for a in attachments], dtype=str),
                body_kind=np.array([view.manager[a['body']].kind for a in attachments], dtype=str),
                element_type=np.array([type(view.line.element_dict[n]).__name__ for n in names], dtype=str),
                length=np.array([getattr(view.line.element_dict[n], 'length', 0.) for n in names], dtype=float))
    if time_s is not None:
        manager = view.manager
        disabled = tuple(manager.sources)
        layers = [manager.get_body_poses(time_s=time_s, disabled_sources=disabled, include_installation=False),
                  manager.get_body_poses(time_s=time_s, disabled_sources=disabled),
                  manager.get_body_poses(time_s=time_s, time_series=time_series)]
        nominal = np.array([a['ref_start'] for a in attachments]).reshape(-1, 4, 4)
        composed = []
        for poses in layers:
            composed.append(np.array([
                (poses[a['body']] @ manager[a['body']].nominal_frame.inverse()).matrix @ a['base_element']
                for a in attachments]).reshape(-1, 4, 4))
        data.update(pose_columns(nominal, *composed))
    return mechanics_table(data, owner=view, time_s=time_s, time_series=time_series)


class MechanicsTable(Table):
    """A pose/configuration snapshot, retaining context through row selection.

    History queries use the current static configuration and the supplied (or
    bound) source record. They do not cache a lattice history or write elements.
    Topology/source changes invalidate the query context; request a fresh table.
    """

    def _owner(self):
        context = self._data.get('_mechanics_owner')
        owner = None if context is None else context()
        if owner is None:
            raise ValueError('No live mechanics context; request a fresh table from the model')
        manager = owner.manager if hasattr(owner, 'manager') else owner
        if manager._revision != self._mechanics_revision:
            raise ValueError('Mechanics definitions changed; request a fresh table')
        return owner

    def to_dict(self, **kwargs):
        """Export a numeric snapshot without live model references/source records."""
        result = super().to_dict(**kwargs)
        for key in ('_mechanics_owner', '_mechanics_revision', '_mechanics_record'):
            result['attrs'].pop(key, None)
        return result

    def _selected_name(self, name):
        if name is None:
            if len(self) != 1:
                raise ValueError('Select one row or supply its exact name')
            name = self.name[0]
        if name not in self.name:
            raise KeyError(f'{name!r} is not in this table selection')
        return name

    def get_element(self, name=None):
        """Access the live Xtrack element for one selected occurrence."""
        owner = self._owner()
        if not hasattr(owner, 'line'):
            raise TypeError('Use line.mechanics.get_table() to select an element')
        name = self._selected_name(name)
        owner._check()
        return owner.line.element_dict[name]

    def get_body(self, name=None):
        """Access a selected body, or the body supporting a selected element."""
        owner = self._owner()
        name = self._selected_name(name)
        if hasattr(owner, 'line'):
            return owner.manager[owner._attachments[name]['body']]
        return owner[name]

    def get_time_series(self, name=None, *, time_s=None, time_series=None,
                        max_memory_bytes=128*1024**2):
        """Compose one selected entity over explicit times, on the host.

        If a sampled source record is bound to this table, its grid is the
        default. This includes deliberate, installation and inherited dynamic
        transforms; it differs from a source record's direct-input get_table.
        """
        owner = self._owner()
        name = self._selected_name(name)
        record = self._mechanics_record if time_series is None else time_series
        if time_s is None:
            if record is None:
                raise ValueError('Supply time_s or a sampled time_series record')
            time_s = record.time_s
        times = checked_times(time_s)
        if not isinstance(max_memory_bytes, int) or max_memory_bytes <= 0:
            raise ValueError('max_memory_bytes must be a positive integer')
        # Conservative bound for one output table and transient query columns.
        # Only one time slice is composed at a time; no all-body history cache.
        if 2048*len(times) > max_memory_bytes:
            raise MemoryError('Selected pose history exceeds max_memory_bytes')
        if record is not None:
            manager = owner.manager if hasattr(owner, 'manager') else owner
            record._check(manager)
        if hasattr(owner, 'line'):
            current = element_table(owner, time_s=float(times[0]), time_series=record, names=[name])
            manager = owner.manager
            attachment = owner._attachments[name]
            body = manager[attachment['body']]
            mount = body.nominal_frame.inverse().matrix @ attachment['base_element']
        else:
            current = owner.survey(time_s=float(times[0]), time_series=record, bodies=[name])
            manager = owner
        total = np.empty((len(times), 4, 4))
        total[0] = current.total_matrix[0]
        for index, time in enumerate(times[1:], 1):
            poses = manager.get_body_poses(time_s=float(time), time_series=record)
            total[index] = (poses[attachment['body']].matrix @ mount if hasattr(owner, 'line')
                            else poses[name].matrix)
        layers = [np.broadcast_to(current[prefix+'_matrix'][0], total.shape)
                  for prefix in ('nominal', 'reference', 'static')]
        output = pose_columns(*layers, total)
        data = dict(time_s=times.copy(), **output)
        return MotionHistoryTable(dict(data, entity_name=name), col_names=list(data), index='time_s')

    def animate(self, name=None, *, time_s, size_m=None, time_series=None,
                displacement_scale=1., rotation_scale=1., baseline='static', interval_ms=40):
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
            size_m = (.3, .3, float(getattr(self.get_element(name), 'length', 0.)))
        return animate_cuboid(self, name, time_s=time_s, size_m=size_m,
                               time_series=time_series, displacement_scale=displacement_scale,
                               rotation_scale=rotation_scale, baseline=baseline, interval_ms=interval_ms)


class MotionHistoryTable(Table):
    """One entity's composed world pose history, indexed by physical seconds."""

    def __init__(self, data, col_names=None, index='time_s', **kwargs):
        super().__init__(data, col_names=col_names, index=index, **kwargs)

    def plot(self):
        """Plot dynamic world translations and rotations relative to static axes."""
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
        for axis in 'XYZ':
            axes[0].plot(self.time_s, self['dynamic_'+axis]*1e9, label=axis)
        for angle, label in (('phi', 'Pitch'), ('theta', 'Yaw'), ('psi', 'Roll')):
            axes[1].plot(self.time_s, self['dynamic_'+angle]*1e9, label=label)
        axes[0].set(ylabel='Dynamic world displacement [nm]', title=self.entity_name)
        axes[1].set(xlabel='Physical time [s]', ylabel='Rotation from installed axes [nrad]')
        for ax in axes:
            ax.legend()
            ax.grid(alpha=.2)
        fig.tight_layout()
        return fig, axes
