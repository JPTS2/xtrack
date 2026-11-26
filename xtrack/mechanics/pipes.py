# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #
"""Installed-pipe ownership and explicit frozen-time aperture geometry.

The line aperture remains the adopted reference model. A requested aperture
view owns its placements and derived bounds; it does not change the line or
turn extracted polygons into live tracking apertures. In particular, pitch and
yaw require new intersections with the reference survey planes.
"""

from copy import deepcopy
import hashlib
import json

import numpy as np
from xobjects import JEncoder

from ..survey import Frame
from ..table import Table
from ..aperture.aperture import LineAperture
from ..beam_elements.limit_polygon import LimitPolygon
from .local_deformations import inherit


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, cls=JEncoder, sort_keys=True, separators=(',', ':')).encode()
    ).digest()


def _geometry_data(aperture):
    data = aperture.to_dict()
    data.pop('halo_params', None)
    return data


def _used_shape_definitions(environment, pipe_names):
    """Fingerprint only shapes reachable from this line's placements."""
    pipes = {name: environment.pipes[name].raw for name in pipe_names}
    profile_names = environment.profiles.keys()
    profile_slots = {
        int(position.profile_index): profile_names[position.profile_index]
        for pipe in pipes.values()
        for position in pipe.positions
    }
    return dict(
        pipes={name: pipe._to_dict() for name, pipe in pipes.items()},
        profiles={
            name: environment.profiles[name].raw._to_dict()
            for name in set(profile_slots.values())
        },
        # A profile index addresses an ordered slot in the packed environment.
        profile_slots=profile_slots,
    )


def _check_profile_coverage(aperture, survey):
    """Admit profile planes with a bracketed reference-path intersection.

    The existing aperture bounds kernel assumes finite survey intersections.
    Reject outside/unbracketed planes before entering it. This conservative
    check can reject a large curved excursion whose crossings both lie inside
    one survey segment; it does not infer an extrapolated chamber.
    """
    from .model import _frame

    path = np.column_stack((survey.X, survey.Y, survey.Z))
    indices = {name: index for index, name in enumerate(survey.name)}

    def has_bracket(points, frame):
        distance = (points - frame[:3, 3]) @ frame[:3, 2]
        left, right = distance[:-1], distance[1:]
        crosses = (left <= 1e-10) & (right >= -1e-10) | (right <= 1e-10) & (
            left >= -1e-10
        )
        return np.any(crosses & (np.abs(right - left) > 1e-12))

    for position in aperture.pipe_positions:
        _frame(Frame(position.transformation))
        index = indices[position.survey_reference_name]
        # Most installed profiles sit at or just beside their survey anchor.
        # Retain a full search for long pipes, offsets and wrapped placements.
        nearby = path[max(0, index - 2) : index + 4]
        base = survey.get_frame(index).matrix @ position.transformation
        for profile in position.pipe:
            frame = base @ profile.get_transform()
            if not has_bracket(nearby, frame) and not has_bracket(path, frame):
                raise ValueError(
                    f'{position.name}: profile plane has no bracketed survey '
                    'intersection; pipe motion outside the reference path '
                    'cannot be extrapolated'
                )


class _FrozenPipeAperture(LineAperture):
    """Frozen geometry with explicit admission of sampled tracking checks."""

    def __init__(self, line, *, context, definitions):
        from ..environment import Environment

        super().__init__(line, context=context, with_progress=False)
        self._geometry_env = Environment()
        self._geometry_env.aperture.from_dict(definitions, context=context)

    @property
    def _aperture_environment(self):
        return self._geometry_env.aperture

    def get_limit_elements_at_s(self, s_positions):
        if np.iscomplexobj(s_positions):
            raise ValueError('Aperture check positions must be real')
        positions = np.asarray(s_positions, dtype=float)
        if positions.ndim != 1 or not np.all(np.isfinite(positions)):
            raise ValueError('Use finite one-dimensional aperture check positions')
        if not len(positions):
            return {}
        candidates = list(self._pipe_candidates_at_s(positions))
        for s, pipes in zip(positions, candidates):
            if not pipes:
                raise ValueError(f'No installed pipe covers aperture check s={s:g} m')
        sections = self.get_cross_sections_at_s(positions)
        limits = {}
        for s, row, pipes in zip(positions, sections.rows, candidates):
            points = np.asarray(row.cross_section)[:-1]
            if len(points) < 3 or not np.all(np.isfinite(points)):
                raise ValueError(f'Nonfinite or degenerate aperture check at s={s:g} m')
            edges = np.roll(points, -1, axis=0) - points
            centred = points - points[0]
            area = np.sum(centred[:, 0] * edges[:, 1] - centred[:, 1] * edges[:, 0])
            if (
                not np.isfinite(area)
                or area == 0.0
                or np.any(np.all(edges == 0.0, axis=1))
            ):
                raise ValueError(f'Nonfinite or degenerate aperture check at s={s:g} m')
            limit = LimitPolygon(x_vertices=points[:, 0], y_vertices=points[:, 1])
            limit.installed_pipe_name = self._installed_pipe_name(pipes)
            limits[s] = limit
        return limits


class PipeBindings:
    """Private reference geometry for installed placements on one line."""

    def __init__(self, view):
        self.view = view
        aperture = view.line.aperture
        survey = view.line.survey()
        if aperture.is_built and any(
            not np.array_equal(
                getattr(aperture.survey, column), getattr(survey, column)
            )
            for column in ('name', 's', 'XYZ', 'E_matrix')
        ):
            raise ValueError(
                'Aperture survey is stale; discard/rebuild it before attaching pipes'
            )
        _check_profile_coverage(aperture, survey)
        # Packed placements can have been edited since the last bounds build.
        # Capture a coherent reference, including fresh derived geometry.
        aperture.build_model()
        self.aperture = aperture
        self.line_data = deepcopy(aperture.to_dict())
        self.line_digest = _digest(_geometry_data(aperture))
        self.pipe_names = frozenset(
            item['pipe'] for item in self.line_data['pipe_positions']
        )
        shapes = _used_shape_definitions(view.line.env.aperture, self.pipe_names)
        self.environment_digest = _digest(shapes)
        slots = sorted(shapes['profile_slots'])
        profile_names = [shapes['profile_slots'][slot] for slot in slots]
        indices = {slot: index for index, slot in enumerate(slots)}
        pipes = []
        pipe_names = sorted(self.pipe_names)
        for name in pipe_names:
            pipe = deepcopy(shapes['pipes'][name])
            for position in pipe['positions']:
                position['profile_index'] = indices[position['profile_index']]
            pipes.append(pipe)
        self._frozen_definitions = dict(
            pipe_names=pipe_names, pipes=pipes, profile_names=profile_names,
            profiles=deepcopy([shapes['profiles'][name] for name in profile_names]),
        )
        self.owners = {}
        self.references = {}
        for position in self.line_data['pipe_positions']:
            name = position['name']
            survey = aperture.survey.get_frame(position['survey_reference_name'])
            transform = aperture.pipe_positions[name].transformation.copy()
            world = view.reference_frame @ survey @ Frame(transform)
            self.references[name] = (transform, world.matrix)

    def validate(self, *, complete=True):
        if (
            self.view.line.aperture is not self.aperture
            or not self.aperture.is_built
            or _digest(_geometry_data(self.aperture)) != self.line_digest
            or _digest(
                _used_shape_definitions(self.view.line.env.aperture, self.pipe_names)
            )
            != self.environment_digest
        ):
            raise ValueError(
                'Adopted pipe geometry changed; clear/rebind pipe mechanics '
                'before editing aperture definitions or placements'
            )
        for name in self.view._attachments:
            if getattr(self.view.line.element_dict[name], 'installed_pipe_name', None):
                raise ValueError(
                    'Pipe-derived limits are frozen sections; do not attach '
                    'them mechanically when declaring pipe ownership. '
                    'Generate fresh limits from get_aperture(time_s=...)'
                )
        if complete and self.owners.keys() != self.references.keys():
            missing = sorted(self.references.keys() - self.owners.keys())
            raise ValueError(
                f'Declare mechanics ownership for every installed pipe, '
                f'including fixed pipes (body=None); missing: {missing}'
            )

    def attach(self, assignments):
        self.validate(complete=False)
        for name, body in assignments.items():
            if name not in self.references:
                raise KeyError(f'Unknown installed pipe placement {name!r}')
            if name in self.owners:
                raise ValueError(f'{name}: pipe ownership is already declared')
            if body is not None and len(self.aperture.pipe_positions[name].pipe) < 2:
                raise ValueError(
                    'A body-owned finite pipe requires at least two profiles; '
                    'a single projected profile is not an extruded chamber'
                )
        self.owners.update(assignments)

    @property
    def bodies(self):
        return {body for body in self.owners.values() if body is not None}

    def _deformations(self, time_s, time_series, disabled_sources, applied_history):
        if np.iscomplexobj(time_s):
            raise ValueError('Pipe pose time must be real seconds')
        needed = self.view.manager._body_closure(self.bodies)
        source_values = None
        if applied_history is not None:
            from .applied_history import AppliedSourceHistory

            if not isinstance(applied_history, AppliedSourceHistory):
                raise TypeError('Supply an immutable applied source history snapshot')
            time_s, time_series, disabled_sources, source_values = applied_history._resolve(
                self.view.manager, self.bodies, time_s, time_series, disabled_sources
            )
        required = {
            name
            for name, binding in self.view.manager.sources.items()
            if binding.live
            and name not in disabled_sources
            and needed.intersection(binding.targets)
        }
        if required.difference(source_values or {}):
            raise ValueError(
                'Frozen pipe geometry requires an applied history covering all relevant live inputs'
            )
        deformations = self.view.manager._get_body_deformations(
            time_s=time_s,
            time_series=time_series,
            disabled_sources=disabled_sources,
            bodies=self.bodies,
            _source_values=source_values,
        )
        return time_s, deformations

    def _pose(self, name, deformations):
        reference, world = self.references[name]
        body_name = self.owners[name]
        if body_name is None:
            return reference.copy(), world.copy(), np.zeros(3)
        body = self.view.manager[body_name]
        translation, increment = inherit(deformations[body_name], body._nominal, world)
        # Retain local increments; never recover nanometres by subtracting
        # two world positions. The reference placement is already included.
        moved = reference.copy()
        moved[:3, 3] += reference[:3, :3] @ translation
        moved[:3, :3] += reference[:3, :3] @ increment
        total = world.copy()
        displacement = world[:3, :3] @ translation
        total[:3, 3] += displacement
        total[:3, :3] += world[:3, :3] @ increment
        return moved, total, displacement

    def get_table(
        self,
        *,
        time_s=None,
        time_series=None,
        disabled_sources=(),
        applied_history=None,
        _resolved=None,
    ):
        names = list(self.owners)
        by_name = {item['name']: item for item in self.line_data['pipe_positions']}
        if time_s is None:
            query_kind = 'configuration'
        elif _resolved is not None:
            query_kind = _resolved[2]
        else:
            query_kind = 'prescribed_pose' if applied_history is None else 'applied_history'
        data = dict(
            name=np.array(names, dtype=str),
            pipe=np.array([by_name[name]['pipe'] for name in names], dtype=str),
            body=np.array([self.owners[name] or '' for name in names], dtype=str),
            motion=np.array(
                ['fixed' if self.owners[name] is None else 'body' for name in names]
            ),
            survey_reference=np.array(
                [by_name[name]['survey_reference_name'] for name in names], dtype=str
            ),
            query_kind=np.full(len(names), query_kind),
        )
        if time_s is None:
            if (
                time_series is not None
                or applied_history is not None
                or disabled_sources
            ):
                raise ValueError('Supply time_s for a pipe pose query')
            return Table(data)
        pose_time_s, poses = (
            self._deformations(time_s, time_series, disabled_sources, applied_history)
            if _resolved is None else _resolved[:2]
        )
        values = [self._pose(name, poses) for name in names]
        for index, coordinate in enumerate(('X', 'Y', 'Z')):
            data['reference_' + coordinate] = np.array(
                [self.references[name][1][index, 3] for name in names]
            )
            data['total_' + coordinate] = np.array(
                [value[1][index, 3] for value in values]
            )
            data['change_' + coordinate] = np.array(
                [value[2][index] for value in values]
            )
        data['pose'] = np.array([value[1] for value in values])
        data['time_s'] = np.full(len(names), time_s)
        data['pose_time_s'] = np.full(len(names), pose_time_s)
        return Table(data)

    def get_aperture(
        self, *, time_s, time_series=None, disabled_sources=(), applied_history=None
    ):
        pose_time_s, poses = self._deformations(
            time_s, time_series, disabled_sources, applied_history
        )
        return self.build_aperture(
            time_s, pose_time_s, poses,
            'prescribed_pose' if applied_history is None else 'applied_history',
        )

    def build_aperture(self, time_s, pose_time_s, poses, query_kind):
        data = deepcopy(self.line_data)
        data['halo_params'] = dict(self.aperture.halo_params)
        for item in data['pipe_positions']:
            # PipePosition stores a complete transform, not a sum of angles.
            item['transformation'] = self._pose(item['name'], poses)[0]
        aperture = _FrozenPipeAperture(
            self.view.line, context=self.aperture.context, definitions=self._frozen_definitions,
        )
        aperture.from_dict(data)
        _check_profile_coverage(aperture, self.aperture.survey)
        aperture.build_model()
        aperture.mechanics_time_s = float(time_s)
        aperture.mechanics_pose_time_s = float(pose_time_s)
        aperture.mechanics_query_kind = query_kind
        return aperture
