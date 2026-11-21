# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #
"""Environment bodies, line attachments and bounded prepared rigid updates.

Explicit application retains a host correctness reference. Automatic turn
freezes use context-native prepared kernels where supported. Geometry is
captured in bulk without taking ownership of live lattice expressions; no
per-scalar xdeps mechanics hierarchy or entire-run pose cache is installed.
"""

from dataclasses import dataclass, field
from types import MappingProxyType
from weakref import WeakKeyDictionary, ref

import numpy as np

from ..survey import Frame
from ..table import Table
from .sources import decode_source, encode_source, requires_time_series
from .motion import RigidMotion, COORDINATES, frozen_array, checked_times
from .installation import RigidOffset
from .local_deformations import (
    rotation_increment,
    compose_rotation,
    compose,
    inherit,
    static_local,
    lower,
)


_FIELDS = (
    'shift_x',
    'shift_y',
    'shift_s',
    'rot_x_rad',
    'rot_y_rad',
    'rot_s_rad_no_frame',
    'rot_shift_anchor',
)
_GEOMETRY = ('length', 'angle', 'h', 'hxl', 'rot_s_rad')


def _frame(value):
    if isinstance(value, Frame):
        matrix = value.matrix.copy()
    else:
        if np.iscomplexobj(value):
            raise ValueError('Anchor coordinates must be real')
        point = np.asarray(value, dtype=float)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError('Use an xt.Frame or a finite global XYZ anchor [m]')
        matrix = np.eye(4)
        matrix[:3, 3] = point
    rotation = matrix[:3, :3]
    if (
        not np.all(np.isfinite(matrix))
        or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-14, rtol=0)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-12, rtol=0)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-12, rtol=0)
    ):
        raise ValueError('Frame must be a finite proper rigid transform')
    return Frame(matrix)


def _values(element):
    return np.array([float(getattr(element, name)) for name in _FIELDS])


def _reference_values(body):
    values = [getattr(body, name) for name in COORDINATES]
    if np.iscomplexobj(values):
        raise ValueError(f'{body.name}: local alignment must be real')
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f'{body.name}: local alignment must be finite')
    return values


def _geometry(element):
    return tuple(float(getattr(element, name, 0.0)) for name in _GEOMETRY)


def _reference_geometry(element):
    return _geometry(element), bool(getattr(element, 'isthick', False))


def _check_alignment_consumers(manager, alignment_refs):
    if manager is None or not alignment_refs:
        return
    # Numeric lowering bypasses xdeps propagation. Inspect graph metadata once
    # at entry, rather than leaving downstream variables/fields silently stale.
    if not alignment_refs.isdisjoint(
        reference for reference, tasks in manager.deptasks.items() if tasks
    ):
        raise ValueError(
            'Mechanics alignment outputs cannot drive xdeps expressions; '
            'sample mechanics tables explicitly instead'
        )


class _BodyPoses(dict):
    """World display frames retaining their precise local deformation data."""

    def __init__(self, manager, deformations):
        self._deformations = deformations
        for name, (translation, increment) in deformations.items():
            nominal = manager[name]._nominal
            matrix = nominal.copy()
            matrix[:3, 3] += nominal[:3, :3] @ translation
            matrix[:3, :3] += nominal[:3, :3] @ increment
            self[name] = Frame(matrix)


@dataclass(slots=True)
class RigidBody:
    """Local alignment about a nominal world anchor; SI units.

    Pitch/yaw/roll follow xt.Frame.from_survey_angles(phi,theta,psi), i.e.
    Ry(yaw) Rx(-pitch) Rz(roll). Ancestor rotations compose as matrices.
    These settings extend the deliberate reference; installation is a separate
    static layer, and dynamic sources never write either set of inputs.
    Topology and anchors are fixed after creation in this first draft.
    """

    name: str
    parent: str | None
    _nominal: np.ndarray
    shift_x: float = 0.0
    shift_y: float = 0.0
    shift_s: float = 0.0
    rot_x_rad: float = 0.0
    rot_y_rad: float = 0.0
    rot_s_rad: float = 0.0
    installation: RigidOffset = field(default_factory=RigidOffset)
    _feet: object = None
    kind: str = 'body'
    support_class: str = ''

    def __setattr__(self, name, value):
        if name in (
            'name',
            'parent',
            '_nominal',
            '_feet',
            'kind',
            'support_class',
        ) and hasattr(self, name):
            raise AttributeError('Body topology is fixed; create a new binding')
        object.__setattr__(self, name, value)

    @property
    def nominal_frame(self):
        return Frame(self._nominal)

    @property
    def feet_local_m(self):
        """Read-only nominal support contacts in body coordinates, or None."""
        return self._feet

    @property
    def nominal_feet_m(self):
        """Nominal world coordinates of the contacts; no installation/dynamics."""
        if self._feet is None:
            raise ValueError('Define feet_local_m when constructing this body')
        return self._feet @ self._nominal[:3, :3].T + self._nominal[:3, 3]


@dataclass(frozen=True)
class SourceBinding:
    name: str
    source: object
    targets: tuple
    component: str
    reference: str = 'relative'
    role: str = 'disturbance'
    live: bool = False


class MechanicalTimeSeries:
    """Reusable sampled source displacements, independent of live body settings.

    Values are read-only (time, target, 6) arrays in metres and survey radians.
    Reuse this object for matched comparisons. Interpolation for apply/get_poses
    is linear, bounded by the supplied record, with no periodic replay.
    """

    def __setattr__(self, name, value):
        if hasattr(self, name):
            raise AttributeError('MechanicalTimeSeries is read-only')
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        raise AttributeError('MechanicalTimeSeries is read-only')

    def __init__(self, manager, time_s, values):
        times = checked_times(time_s)
        if any(
            name not in manager.sources or manager.sources[name].live for name in values
        ):
            raise ValueError(
                'Time series channels must name existing prescribed source bindings'
            )
        arrays = {name: frozen_array(value) for name, value in values.items()}
        if any(
            value.shape != (len(times), len(manager.sources[name].targets), 6)
            for name, value in arrays.items()
        ):
            raise ValueError(
                'Time series values must have shape (times, source targets, 6)'
            )
        self._manager = ref(manager)
        self._revision = manager._revision
        self._bodies = tuple(manager.bodies)
        self._targets = MappingProxyType(
            {name: manager.sources[name].targets for name in values}
        )
        self.references = MappingProxyType(
            {name: manager.sources[name].reference for name in values}
        )
        self.omitted_live_sources = tuple(
            name for name, binding in manager.sources.items() if binding.live
        )
        self._live_targets = {
            name: manager.sources[name].targets for name in self.omitted_live_sources
        }
        self.body_frames = MappingProxyType(
            {name: self._freeze(body._nominal) for name, body in manager.bodies.items()}
        )
        self.time_s = times
        self.values = MappingProxyType(arrays)

    @staticmethod
    def _freeze(value):
        array = np.asarray(value, dtype=float)
        return np.frombuffer(array.tobytes(), dtype=float).reshape(array.shape)

    def get_table(self, body):
        """Direct source coordinates; no static or inherited motion.

        An absolute binding reports absolute dynamic coordinates about the
        static body's anchor. Inspect references to distinguish binding modes.
        Multiple relative input rotations compose in source insertion order.
        """
        if body not in self._bodies:
            raise KeyError(body)
        if any(body in targets for targets in self._live_targets.values()):
            raise ValueError(
                'A prescribed time series does not contain live source history'
            )
        values = np.zeros((len(self.time_s), 6))
        rotation = np.broadcast_to(np.eye(3), (len(self.time_s), 3, 3)).copy()
        for name, targets in self._targets.items():
            if body in targets:
                current = self.values[name][:, targets.index(body)]
                values[:, :3] += current[:, :3]
                if np.any(current[:, 3:]):
                    from scipy.spatial.transform import Rotation

                    rotation = (
                        Rotation.from_euler(
                            'YXZ', current[:, [4, 3, 5]] * [1, -1, 1]
                        ).as_matrix()
                        @ rotation
                    )
        from .motion import angles_from_matrix

        values[:, 3:] = angles_from_matrix(rotation)
        return Table(
            dict(
                time_s=self.time_s.copy(),
                **{field: values[:, i] for i, field in enumerate(COORDINATES)},
            ),
            index='time_s',
        )

    def _check(self, manager):
        if (
            self._manager() is not manager
            or manager is None
            or manager._revision != self._revision
        ):
            raise ValueError(
                'Time series belongs to another or changed mechanics model'
            )


class MechanicsManager:
    """Physical bodies and motion sources shared by lines in one environment.

    Access as env.mechanics. line.mechanics supplies a bound line view of the
    same manager. The environment is a geometry namespace, not an implicit
    common turn clock: all motion queries/application use physical time_s.
    """

    def __init__(self, env):
        self._env_ref = ref(env)
        self._bodies, self._sources = {}, {}
        self._line_views = WeakKeyDictionary()
        self._revision = 0

    @property
    def env(self):
        return self._env_ref()

    @property
    def bodies(self):
        return MappingProxyType(self._bodies)

    @property
    def sources(self):
        return MappingProxyType(self._sources)

    @property
    def _has_configuration(self):
        """Distinguish configured mechanics from lazy accessor creation."""
        return bool(
            self._bodies
            or self._sources
            or any(view._has_configuration for view in self._line_views.values())
        )

    def __getitem__(self, name):
        return self._bodies[name]

    def for_line(self, line):
        """Return the existing bound view; never create a second physical world."""
        if isinstance(line, str):
            line = self.env.lines[line]
        if line.env is not self.env:
            raise ValueError('Line must belong to this mechanics environment')
        if line not in self._line_views:
            self._line_views[line] = LineMechanics(self, line)
        return self._line_views[line]

    def generate_installation_errors(self, **kwargs):
        """Generate an independent reproducible static case; no automatic writes."""
        from .installation import generate_installation_errors

        return generate_installation_errors(self, **kwargs)

    def _line(self, line):
        if line is None:
            if len(self.env.lines) != 1:
                raise ValueError(
                    'Specify line= or use line.mechanics when the environment has multiple lines'
                )
            line = next(iter(self.env.lines.values()))
        return self.for_line(line)

    def new_body(
        self,
        name,
        *,
        elements=None,
        line=None,
        anchor='center',
        parent=None,
        mount=None,
        feet_local_m=None,
        kind='body',
        support_class='',
    ):
        """Create a named rigid magnet, pickup, equipment or support body.

        mount is a parent-local Frame. anchor is a nominal world Frame/XYZ;
        specify one placement convention. Attachments retain their surveyed
        positions relative to the body. No force law is inferred from mounting.
        kind is an explicit table-selection label (body, magnet, pickup, ...).
        """
        anchor = self._resolve_mount(anchor=anchor, parent=parent, mount=mount)
        return self._new_body(
            name,
            elements=elements,
            line=line,
            anchor=anchor,
            parent=parent,
            feet_local_m=feet_local_m,
            kind=kind,
            support_class=support_class,
        )

    def _resolve_mount(self, *, anchor, parent, mount):
        """Resolve one parent-local mount without changing body definitions."""
        if mount is None:
            return anchor
        if parent is None or not isinstance(anchor, str) or anchor != 'center':
            raise ValueError('A mount requires a parent and no explicit world anchor')
        owner = self.bodies.get(parent) if isinstance(parent, str) else parent
        if not isinstance(owner, RigidBody) or self.bodies.get(owner.name) is not owner:
            raise ValueError('Parent must belong to this environment')
        return owner.nominal_frame @ _frame(mount)

    def new_girder(
        self,
        name,
        *,
        elements=None,
        line=None,
        anchor='center',
        parent=None,
        mount=None,
        feet_local_m=None,
        support_class='',
    ):
        """Retrofit elements, or create an explicitly anchored shared body.

        'center' is the chord midpoint of the selected entrance/exit span, with
        axes from its first nominal entrance frame. A Frame gives full world
        anchor/orientation; a three-vector gives world position with XYZ axes.
        A mount gives a parent-local Frame instead of a world anchor. Parent
        names refer to existing bodies. No sublines are required.
        """
        return self.new_body(
            name,
            elements=elements,
            line=line,
            anchor=anchor,
            parent=parent,
            mount=mount,
            feet_local_m=feet_local_m,
            kind='girder',
            support_class=support_class,
        )

    def new_girders(self, assignments, *, line=None, support_class=''):
        """Create several supports with one survey; see line.mechanics.new_girders."""
        return self._line(line).new_girders(assignments, support_class=support_class)

    def new_bodies(self, assignments, *, line=None, kind='body', support_class=''):
        """Create attached bodies with one survey; see line.mechanics.new_bodies."""
        return self._line(line).new_bodies(
            assignments, kind=kind, support_class=support_class
        )

    def _body_definition(
        self,
        name,
        *,
        anchor,
        parent,
        feet_local_m,
        kind,
        support_class='',
        candidates=(),
    ):
        """Validate/build a body without changing the manager or line."""
        if not isinstance(kind, str) or not kind:
            raise ValueError(
                'Use a nonempty body kind, e.g. magnet, pickup or equipment'
            )
        if not isinstance(support_class, str):
            raise ValueError('support_class must be a string')
        if not isinstance(name, str) or not name or name in self._bodies:
            raise ValueError('Use a new nonempty body name')
        if isinstance(parent, RigidBody):
            if self._bodies.get(parent.name) is not parent:
                raise ValueError('Parent must belong to this environment')
            parent = parent.name
        if parent is not None and parent not in self._bodies:
            raise ValueError('Create the parent body first')
        if isinstance(anchor, str):
            if anchor != 'center' or not candidates:
                raise ValueError(
                    "Use anchor='center' with attachments, or supply a world Frame/XYZ point"
                )
            nominal = Frame(candidates[0]['ref_start'])
            nominal.matrix[:3, 3] = (
                candidates[0]['ref_start'][:3, 3] + candidates[-1]['ref_end'][:3, 3]
            ) / 2
        else:
            nominal = _frame(anchor)
        frozen = np.frombuffer(nominal.matrix.tobytes(), dtype=float).reshape(4, 4)
        feet = None if feet_local_m is None else frozen_array(feet_local_m)
        if feet is not None and (feet.ndim != 2 or feet.shape[1] != 3 or not len(feet)):
            raise ValueError('feet_local_m must be nonempty local XYZ coordinates')
        return RigidBody(
            name, parent, frozen, _feet=feet, kind=kind, support_class=support_class
        )

    def _new_body(
        self,
        name,
        *,
        elements,
        line,
        anchor,
        parent,
        feet_local_m,
        kind,
        support_class='',
    ):
        # Reject invalid names before the potentially expensive candidate survey.
        if not isinstance(name, str) or not name or name in self._bodies:
            raise ValueError('Use a new nonempty body name')
        view, candidates = None, []
        if elements is not None:
            view = self._line(line)
            candidates = view._candidates(elements)
        body = self._body_definition(
            name,
            anchor=anchor,
            parent=parent,
            feet_local_m=feet_local_m,
            kind=kind,
            support_class=support_class,
            candidates=candidates,
        )
        self._bodies[name] = body
        if view is not None:
            view._attach(name, candidates)
        self._revision += 1
        return body

    def add_source(
        self,
        name,
        source,
        *,
        targets,
        component='y',
        reference='relative',
        role='disturbance',
        live=False,
    ):
        """Bind a named source model, independently of girder construction.

        Scalar inputs use nominal-body x/y/s [m] or pitch/yaw/roll [rad].
        RigidMotion supplies all six coordinates. Relative motion composes down
        the mounting tree; absolute motion replaces inherited vibration while
        retaining the static mounting/alignment. No Euler-angle subtraction.
        role='control' distinguishes applied actuator motion from disturbances.
        live=True reads the current held provider at each freeze and is never
        captured by compute_time_series. Providers must not advance plant state.
        """
        if not isinstance(name, str) or not name or name in self._sources:
            raise ValueError('Use a new nonempty source name')
        if not callable(getattr(source, 'evaluate', None)):
            raise TypeError(
                'A source must implement evaluate(time_s=..., positions_m=...)'
            )
        frame = getattr(source, 'frame', 'body_local')
        if frame not in ('body_local', 'world', 'body_local_rigid'):
            raise ValueError('Unknown source coordinate frame')
        if reference not in ('relative', 'absolute'):
            raise ValueError("reference must be 'relative' or 'absolute'")
        if role not in ('disturbance', 'control'):
            raise ValueError("role must be 'disturbance' or 'control'")
        if not isinstance(live, (bool, np.bool_)):
            raise ValueError('live must be a boolean capability flag')
        if live and requires_time_series(source):
            raise ValueError(
                'A finite FFT source requires a prescribed time series, not live evaluation'
            )
        targets = (targets,) if isinstance(targets, str) else tuple(targets)
        if (
            not targets
            or len(set(targets)) != len(targets)
            or any(n not in self._bodies for n in targets)
        ):
            raise ValueError('Select unique existing bodies')
        if component not in ('x', 'y', 's', 'pitch', 'yaw', 'roll'):
            raise ValueError('Use x/y/s or pitch/yaw/roll component')
        if frame == 'body_local' and getattr(source, 'unit', 'm') != (
            'rad' if component in ('pitch', 'yaw', 'roll') else 'm'
        ):
            raise ValueError('Source unit must match translation [m] or angle [rad]')
        if frame == 'world' and getattr(source, 'unit', 'm') != 'm':
            raise ValueError('World source must provide displacement in metres [m]')
        if (
            getattr(source, 'target_names', ())
            and tuple(source.target_names) != targets
        ):
            raise ValueError(
                'Rigid motion target names/order must match binding targets'
            )
        for binding in self.sources.values():
            if set(binding.targets).intersection(targets) and (
                'absolute' in (reference, binding.reference)
            ):
                raise ValueError(
                    'An absolute-driven body needs one explicit combined motion input'
                )
            for control_targets, absolute_targets in (
                (
                    targets if role == 'control' else (),
                    binding.targets if binding.reference == 'absolute' else (),
                ),
                (
                    binding.targets if binding.role == 'control' else (),
                    targets if reference == 'absolute' else (),
                ),
            ):
                for absolute_target in absolute_targets:
                    parent = self._bodies[absolute_target].parent
                    while parent is not None:
                        if parent in control_targets:
                            raise ValueError(
                                'An absolute descendant source would mask ancestor control; '
                                'mount the control downstream or supply one combined response'
                            )
                        parent = self._bodies[parent].parent
        for target in targets:
            parent = self._bodies[target].parent
            while parent is not None:
                if parent in targets and reference == 'relative':
                    raise ValueError(
                        'One source cannot drive both a body and its ancestor'
                    )
                parent = self._bodies[parent].parent
            if (
                frame == 'world'
                and self._bodies[target].parent is not None
                and reference != 'absolute'
            ):
                raise ValueError(
                    'World-field shortcut drives roots; child absolute drives need explicit response handling'
                )
        from .sources import validate_source_bindings

        validate_source_bindings(source, self.sources.values())
        binding = SourceBinding(
            name, source, targets, component, reference, role, bool(live)
        )
        self._sources[name] = binding
        self._revision += 1
        return binding

    def _evaluate_source(self, binding, time_s):
        positions = np.array([self._bodies[n]._nominal[:3, 3] for n in binding.targets])
        values = np.asarray(
            binding.source.evaluate(time_s=time_s, positions_m=positions)
        )
        if np.iscomplexobj(values) or not np.all(np.isfinite(values)):
            raise ValueError(
                f'{binding.name}: source must return finite real displacement'
            )
        frame = getattr(binding.source, 'frame', 'body_local')
        if frame == 'body_local_rigid':
            if values.shape != (len(time_s), len(positions), 6):
                raise ValueError('Rigid source must return shape (times, targets, 6)')
            # Providers may reuse their own buffer for the next binding.
            return np.array(values, copy=True)
        output = np.zeros((len(time_s), len(binding.targets), 6))
        if frame == 'world':
            if values.shape != (len(time_s), len(positions), 3):
                raise ValueError('World source must return shape (times, targets, 3)')
            rotations = np.array(
                [self._bodies[n]._nominal[:3, :3] for n in binding.targets]
            )
            output[..., :3] = np.einsum('bji,tbj->tbi', rotations, values)
            return output
        if values.shape == (len(time_s),):
            values = np.broadcast_to(values[:, None], (len(time_s), len(positions)))
        if values.shape != (len(time_s), len(positions)):
            raise ValueError(
                'Scalar local source must return shape (times, targets) or (times,)'
            )
        output[
            ..., ('x', 'y', 's', 'pitch', 'yaw', 'roll').index(binding.component)
        ] = values
        return output

    def compute_time_series(self, *, time_s, max_memory_bytes=128 * 1024**2):
        """Compute reusable prescribed motion at explicit times [s], on the host.

        Live bindings are omitted, never frozen into a historical record.
        Inspect omitted_live_sources; their held values are read at each freeze.
        Analytic/measured models accept nonuniform increasing times; PSD models
        require a uniform grid. This explicit short-record backend has a
        conservative allocation guard and never creates a GPU history cache.
        """
        times = checked_times(time_s)
        prescribed = {
            name: binding for name, binding in self.sources.items() if not binding.live
        }
        counts = [len(b.targets) for b in prescribed.values()]
        estimate = 96 * len(times) * sum(counts) + max(
            (
                96 * len(times) * len(b.targets)
                + getattr(b.source, 'workspace_bytes', lambda nt, nr: 0)(
                    len(times), len(b.targets)
                )
                for b in prescribed.values()
            ),
            default=0,
        )
        if not isinstance(max_memory_bytes, int) or max_memory_bytes <= 0:
            raise ValueError('max_memory_bytes must be a positive integer')
        if estimate > max_memory_bytes:
            raise ValueError(
                f'Estimated short-record workspace {estimate} bytes exceeds max_memory_bytes'
            )
        values = {
            name: self._evaluate_source(binding, times)
            for name, binding in prescribed.items()
        }
        return MechanicalTimeSeries(self, times, values)

    def _body_closure(self, bodies=None):
        requested = (
            tuple(self.bodies)
            if bodies is None
            else ((bodies,) if isinstance(bodies, str) else tuple(bodies))
        )
        if len(set(requested)) != len(requested) or not set(requested).issubset(
            self.bodies
        ):
            raise ValueError('Select unique existing mechanics bodies')
        needed = set(requested)
        for name in requested:
            parent = self.bodies[name].parent
            while parent is not None:
                needed.add(parent)
                parent = self.bodies[parent].parent
        return needed

    def _sample_live_values(self, time_s, *, bodies=None, disabled_sources=()):
        if not np.isscalar(time_s) or not np.isfinite(time_s):
            raise ValueError('time_s must be a finite scalar in seconds')
        needed = self._body_closure(bodies)
        return {
            name: self._evaluate_source(binding, np.array([time_s]))[0]
            for name, binding in self.sources.items()
            if binding.live
            and name not in disabled_sources
            and needed.intersection(binding.targets)
        }

    def get_body_poses(
        self,
        *,
        time_s=0.0,
        time_series=None,
        disabled_sources=(),
        include_installation=True,
        bodies=None,
        _source_values=None,
    ):
        """Inspect independent xt.Frame world poses, without changing the line.

        World display coordinates have ordinary float64 resolution. Tracking
        lowering retains local displacement/rotation increments independently;
        a tiny dynamic displacement is never recovered from world subtraction.
        """
        return _BodyPoses(
            self,
            self._get_body_deformations(
                time_s=time_s,
                time_series=time_series,
                disabled_sources=disabled_sources,
                include_installation=include_installation,
                bodies=bodies,
                _source_values=_source_values,
            ),
        )

    def _get_body_deformations(
        self,
        *,
        time_s=0.0,
        time_series=None,
        disabled_sources=(),
        include_installation=True,
        bodies=None,
        _source_values=None,
    ):
        """Finite affine increments about each body's nominal local frame."""
        if not np.isscalar(time_s) or not np.isfinite(time_s):
            raise ValueError('time_s must be a finite scalar in seconds')
        if not set(disabled_sources).issubset(self.sources):
            raise ValueError('Unknown disabled source')
        if time_series is not None:
            time_series._check(self)
        requested = (
            tuple(self.bodies)
            if bodies is None
            else ((bodies,) if isinstance(bodies, str) else tuple(bodies))
        )
        needed = self._body_closure(requested)
        selected = {name: body for name, body in self.bodies.items() if name in needed}
        shifts = {name: np.zeros(3) for name in selected}
        rotations = {name: np.zeros((3, 3)) for name in selected}
        absolute = {}
        for name, binding in self.sources.items():
            if name in disabled_sources:
                continue
            indices = [
                index
                for index, target in enumerate(binding.targets)
                if target in needed
            ]
            if not indices:
                continue
            targets = [binding.targets[index] for index in indices]
            if _source_values is not None and name in _source_values:
                full_values = np.asarray(_source_values[name])
                if (
                    full_values.shape != (len(binding.targets), 6)
                    or np.iscomplexobj(full_values)
                    or not np.all(np.isfinite(full_values))
                ):
                    raise ValueError(
                        'Held source values must be finite real target-by-six coordinates'
                    )
                values = full_values[indices]
            elif time_series is None or binding.live or name not in time_series.values:
                if requires_time_series(binding.source):
                    raise ValueError(
                        'Call compute_time_series and supply the prescribed source record'
                    )
                if type(binding.source) is RigidMotion:
                    source = binding.source
                    if time_s < source.time_s[0] or time_s > source.time_s[-1]:
                        raise ValueError(
                            'Requested time lies outside the rigid motion record'
                        )
                    values = np.array(
                        [
                            [
                                np.interp(
                                    time_s, source.time_s, source.values[:, index, j]
                                )
                                for j in range(6)
                            ]
                            for index in indices
                        ]
                    )
                else:
                    # Arbitrary stochastic sources may depend on the complete
                    # receiver set. Evaluate jointly, then select, preserving
                    # their seed/correlation contract.
                    values = self._evaluate_source(binding, np.array([time_s]))[0][
                        indices
                    ]
            else:
                if time_s < time_series.time_s[0] or time_s > time_series.time_s[-1]:
                    raise ValueError(
                        'Requested time lies outside the computed time series'
                    )
                record = time_series.values[name]
                values = np.array(
                    [
                        [
                            np.interp(time_s, time_series.time_s, record[:, i, j])
                            for j in range(6)
                        ]
                        for i in indices
                    ]
                )
            for target, value in zip(targets, values):
                if binding.reference == 'absolute':
                    absolute[target] = value
                else:
                    shifts[target] += value[:3]
                    if np.any(value[3:]):
                        rotations[target] = compose_rotation(
                            rotation_increment(*value[3:]), rotations[target]
                        )
        world, static = {}, {}
        for name, body in selected.items():
            local = _reference_values(body)
            install = (
                body.installation.values() if include_installation else np.zeros(6)
            )
            delta = static_local(local, install)
            if body.parent is None:
                static[name] = delta
            else:
                parent = self.bodies[body.parent]
                static[name] = compose(
                    inherit(static[parent.name], parent._nominal, body._nominal), delta
                )
            if name in absolute:
                value = absolute[name]
                world[name] = (
                    static[name][0] + value[:3],
                    compose_rotation(rotation_increment(*value[3:]), static[name][1]),
                )
                continue
            delta = (
                delta[0] + shifts[name],
                compose_rotation(rotations[name], delta[1]),
            )
            if body.parent is None:
                world[name] = delta
            else:
                parent = self.bodies[body.parent]
                world[name] = compose(
                    inherit(world[parent.name], parent._nominal, body._nominal), delta
                )
        return {name: world[name] for name in requested}

    def survey(
        self,
        *,
        time_s=0.0,
        time_series=None,
        bodies=None,
        _source_values=None,
        disabled_sources=(),
    ):
        """Read nominal, static and total world poses without writing elements."""
        from .plotting import motion_survey

        return motion_survey(
            self,
            time_s=time_s,
            time_series=time_series,
            bodies=bodies,
            _source_values=_source_values,
            disabled_sources=disabled_sources,
        )

    def plot(
        self,
        *,
        time_s=0.0,
        time_series=None,
        bodies=None,
        line=None,
        displacement_scale=1.0,
        rotation_scale=1.0,
        view='line',
        baseline='reference',
    ):
        """Plot magnified rigid geometry; gains affect display only."""
        from .plotting import plot_motion

        return plot_motion(
            self,
            time_s=time_s,
            time_series=time_series,
            bodies=bodies,
            line=line,
            displacement_scale=displacement_scale,
            rotation_scale=rotation_scale,
            view=view,
            baseline=baseline,
        )

    def animate(
        self,
        *,
        time_s,
        time_series=None,
        bodies=None,
        line=None,
        displacement_scale=1.0,
        rotation_scale=1.0,
        interval_ms=80,
        view='line',
        baseline='reference',
    ):
        """Animate selected poses with bounded history; retain the returned animation."""
        from .plotting import animate_motion

        return animate_motion(
            self,
            time_s=time_s,
            time_series=time_series,
            bodies=bodies,
            line=line,
            displacement_scale=displacement_scale,
            rotation_scale=rotation_scale,
            interval_ms=interval_ms,
            view=view,
            baseline=baseline,
        )

    def apply(self, *, time_s=0.0, time_series=None, disabled_sources=()):
        """Apply one physical time to ALL attached lines in this environment.

        Explicit host-side numeric alignment lowering. Does not start a clock,
        register a track callback, or update sources once per particle/turn.
        Unsupported bindings/conflicting edits are rejected before any writes.
        """
        self._check_aliases()
        views = [view for view in self._line_views.values() if view._attachments]
        for view in views:
            if view._applied_source_recorder is not None:
                view._applied_source_recorder.check_case()
        consumed = (
            set().union(*(view._consumed_bodies() for view in views))
            if views
            else set()
        )
        live_values = self._sample_live_values(
            time_s, bodies=consumed, disabled_sources=disabled_sources
        )
        poses = self.get_body_poses(
            time_s=time_s,
            time_series=time_series,
            disabled_sources=disabled_sources,
            bodies=consumed,
            _source_values=live_values,
        )
        updates = []
        for view in views:
            updates.extend(view._updates(poses))
        for view in views:
            view._preflight_applied(
                time_s,
                live_values,
                disabled_sources=disabled_sources,
                time_series=time_series,
            )
        for attachment, element, values in updates:
            for field, value in zip(_FIELDS, values):
                setattr(element, field, value)
            attachment['last_written'] = values.copy()
        for view in views:
            view._remember_applied(
                time_s,
                live_values,
                disabled_sources=disabled_sources,
                time_series=time_series,
            )

    def restore(self):
        """Remove the applied mechanics, restoring captured numeric local errors."""
        self._check_aliases()
        updates = []
        for view in self._line_views.values():
            view._check()
            for attachment in view._attachments.values():
                element = view.line.element_dict[attachment['element']]
                updates.append((attachment, element, attachment['baseline']))
        for attachment, element, values in updates:
            for field, value in zip(_FIELDS, values):
                setattr(element, field, value)
            attachment['last_written'] = values.copy()

    def _check_aliases(self):
        # Only attached objects consume numeric placements. Still scan every
        # physical line: an alias can be introduced under a different name or
        # on an otherwise unattached line after these bindings were prepared.
        needed = set()
        for view in self._line_views.values():
            elements = view.line.element_dict
            needed.update(id(elements[name]) for name in view._attachments)
        if not needed:
            return
        counts = dict.fromkeys(needed, 0)
        for line in self.env._lines_weakrefs:
            # Collective survey/Twiss creates shallow noncollective views in
            # distinct environments sharing this weak set. They are execution
            # views of the same occurrences, not additional physical lines.
            if line.env is not self.env:
                continue
            elements = line.element_dict
            for name in line.element_names:
                identity = id(elements[name])
                if identity in needed:
                    counts[identity] += 1
        for view in self._line_views.values():
            elements = view.line.element_dict
            for name in view._attachments:
                if counts.get(id(elements[name])) != 1:
                    raise ValueError(
                        f'{name}: shared/repeated element objects require an occurrence-buffer lowering'
                    )

    def clear(self):
        """Restore original local alignments and remove all mechanics data.

        Sources, body definitions, attachments and automatic vibration updates
        are removed. Deliberate element design geometry (including rot_s_rad)
        is never restored from an old copy or overwritten. Conflicting edits to
        effective alignment fields are rejected before any removal.
        """
        views = list(self._line_views.values())
        updates = [item for view in views for item in view._removal_updates()]
        for view in views:
            if view._attachments and view.line.aperture.is_built:
                view.line.aperture.discard_model()
        for attachment, element, values in updates:
            for field, value in zip(_FIELDS, values):
                setattr(element, field, value)
        for view in views:
            view._clear_bindings()
        self._sources.clear()
        self._bodies.clear()
        self._revision += 1
        return self

    def get_table(self, *, time_s=None, time_series=None):
        """Body inputs, or composed world poses at an explicit physical time.

        No argument returns configuration without evaluating sources. Supply
        time_s for a read-only pose table, equivalent to survey. Rows retain
        normal Xsuite selection and can query a selected body's time history.
        """
        if time_s is not None:
            return self.survey(time_s=time_s, time_series=time_series)
        if time_series is not None:
            raise ValueError('Supply time_s when requesting a pose table')
        from .tables import mechanics_table

        bodies = list(self.bodies.values())
        data = dict(
            name=np.array([b.name for b in bodies]),
            kind=np.array([b.kind for b in bodies]),
            support_class=np.array([b.support_class for b in bodies], dtype=str),
            parent=np.array([b.parent or '' for b in bodies]),
        )
        reference_values = [_reference_values(body) for body in bodies]
        installation_values = [body.installation.values() for body in bodies]
        for index, field in enumerate(COORDINATES):
            data[field] = np.array(
                [values[index] for values in reference_values], dtype=float
            )
            data['installation_' + field] = np.array(
                [values[index] for values in installation_values], dtype=float
            )
        return mechanics_table(data, owner=self)

    def get_source_table(self):
        """Inspect bindings without sampling or copying source waveform arrays."""
        bindings = list(self.sources.values())
        data = dict(
            name=np.array([b.name for b in bindings]),
            model=np.array([type(b.source).__name__ for b in bindings]),
            frame=np.array(
                [getattr(b.source, 'frame', 'body_local') for b in bindings]
            ),
            component=np.array(
                [
                    b.component
                    if getattr(b.source, 'frame', 'body_local') == 'body_local'
                    else 'XYZ'
                    if b.source.frame == 'world'
                    else 'six_coordinates'
                    for b in bindings
                ]
            ),
            unit=np.array(
                [
                    getattr(
                        b.source,
                        'unit',
                        'm, rad'
                        if getattr(b.source, 'frame', 'body_local')
                        == 'body_local_rigid'
                        else 'm',
                    )
                    for b in bindings
                ]
            ),
            reference=np.array([b.reference for b in bindings]),
            role=np.array([b.role for b in bindings]),
            live=np.array([b.live for b in bindings], dtype=bool),
            capability=np.array(['live' if b.live else 'prescribed' for b in bindings]),
            target_names=np.array([', '.join(b.targets) for b in bindings]),
            n_targets=np.array([len(b.targets) for b in bindings], dtype=int),
        )
        # Explicit object vectors preserve tuples/models as one cell per binding.
        for column, values in (
            ('targets', [b.targets for b in bindings]),
            ('seed', [getattr(b.source, 'seed', None) for b in bindings]),
            ('source', [b.source for b in bindings]),
        ):
            data[column] = np.empty(len(values), dtype=object)
            for index, value in enumerate(values):
                data[column][index] = value
        return Table(data)

    def to_dict(self):
        """Save definitions/bindings, including explicitly supplied RigidMotion.

        Computed manager records, automatic enable flags and causal filter
        state are not serialized. Supplied sampled sources retain their data.
        The line remains frozen at its currently applied pose in the saved
        environment. Conflicting local edits or changed reference geometry
        must be resolved before saving, just as before application.
        """
        self._check_aliases()
        for view in self._line_views.values():
            if view._attachments:
                # A built aperture is valid frozen snapshot data. Applying a
                # different pose still requires the normal aperture guard.
                view._check(allow_built_aperture=True)
        encoded_sources = {}
        for name, binding in self.sources.items():
            try:
                encoded_sources[name] = encode_source(binding.source)
            except TypeError as error:
                if binding.live:
                    raise TypeError(
                        f'Live source {name!r} cannot be serialized; save its controller state separately'
                    ) from error
                raise
        bodies = []
        for body in self.bodies.values():
            alignment = _reference_values(body).tolist()
            bodies.append(
                dict(
                    name=body.name,
                    kind=body.kind,
                    support_class=body.support_class,
                    parent=body.parent,
                    anchor=body._nominal.tolist(),
                    alignment=alignment,
                    installation=body.installation.values().tolist(),
                    feet_local_m=None
                    if body.feet_local_m is None
                    else body.feet_local_m.tolist(),
                )
            )
        lines = []
        for line, view in self._line_views.items():
            if not view._has_configuration:
                continue
            names = [name for name, ll in self.env.lines.items() if ll is line]
            if len(names) != 1:
                raise ValueError(
                    'Register configured lines under one environment name before serialization'
                )
            attachments = []
            for a in view._attachments.values():
                attachments.append(
                    {
                        key: value.tolist() if isinstance(value, np.ndarray) else value
                        for key, value in a.items()
                        if key not in ('last_written', 'object_id')
                    }
                )
            lines.append(
                dict(
                    line=names[0],
                    reference_frame=view._reference.matrix.tolist(),
                    reference_is_explicit=view._reference_is_explicit,
                    adopted_reference={
                        name: values.tolist()
                        for name, values in view._reference_inputs.items()
                    },
                    attachments=attachments,
                )
            )
        return dict(
            bodies=bodies,
            sources=[
                dict(
                    name=b.name,
                    targets=list(b.targets),
                    component=b.component,
                    reference=b.reference,
                    source=encoded_sources[b.name],
                    role=b.role,
                    live=b.live,
                )
                for b in self.sources.values()
            ],
            lines=lines,
        )

    def from_dict(self, data):
        """Load definitions into an empty or explicitly cleared manager.

        Merely accessing ``line.mechanics`` does not make a configured model.
        Existing attachments, adopted inputs or explicit survey placements must
        be cleared first. Automatic vibrations remain disabled after loading.
        """
        if self._has_configuration:
            raise ValueError('Load mechanics into an empty manager')
        for item in data['bodies']:
            b = self.new_body(
                item['name'],
                parent=item['parent'],
                anchor=Frame(item['anchor']),
                feet_local_m=item.get('feet_local_m'),
                kind=item.get('kind', 'body'),
                support_class=item.get('support_class', ''),
            )
            b.shift_x, b.shift_y, b.shift_s, b.rot_x_rad, b.rot_y_rad, b.rot_s_rad = (
                item['alignment']
            )
            b.installation = RigidOffset(*item.get('installation', [0.0] * 6))
        for item in data['lines']:
            view = self.for_line(item['line'])
            view.set_reference_frame(Frame(item['reference_frame']))
            stored_attachments = item['attachments']
            candidates = (
                view._candidates([a['element'] for a in stored_attachments])
                if stored_attachments
                else []
            )
            by_name = {candidate['element']: candidate for candidate in candidates}
            for stored in stored_attachments:
                candidate = by_name[stored['element']]
                for key in (
                    'ref_start',
                    'ref_end',
                    'base_element',
                    'base_element_end',
                    'baseline',
                ):
                    if key not in stored:
                        continue
                    candidate[key] = np.array(stored[key], dtype=float)
                candidate['geometry'] = tuple(stored['geometry'])
                candidate['body'] = stored['body']
            if candidates:
                view._install_candidates(candidates)
            view._reference_is_explicit = item.get('reference_is_explicit', True)
            view._reference_inputs = {
                name: np.array(values, dtype=float)
                for name, values in item.get('adopted_reference', {}).items()
            }
        for item in data['sources']:
            source = item['source']
            self.add_source(
                item['name'],
                decode_source(source),
                targets=item['targets'],
                component=item['component'],
                reference=item.get('reference', 'relative'),
                role=item.get('role', 'disturbance'),
                live=item.get('live', False),
            )
        return self


class LineMechanics:
    """Line-specific attachments/view of env.mechanics; not an independent manager."""

    def __init__(self, manager, line):
        self.manager = manager
        self._line_ref = ref(line)
        self._bound_line = None
        self._reference = Frame()
        self._reference_is_explicit = False
        self._attachments = {}
        self._element_names = None
        self._line_geometry = None
        self._vibrations_enabled = False
        self._controls_enabled = False
        self._source_mode_configured = False
        self._last_applied_time_s = None
        self._last_applied_time_series = None
        self._last_control_values = {}
        self._last_applied_controls = set()
        self._applied_source_recorder = None
        self._control_metadata_revision = None
        self._relevant_control_names = frozenset()
        self._relevant_live_control_names = frozenset()
        self._disabled_source_cache_key = None
        self._disabled_source_cache = ()
        self._time_series = None
        self._reference_inputs = {}
        self._time_origin_s = 0.0
        self._sampling = 'turn'
        self._runtime = None
        self._backend = 'auto'
        self._prepared_empty_log = None

    @property
    def _has_configuration(self):
        return bool(
            self._attachments or self._reference_inputs or self._reference_is_explicit
        )

    def adopt_reference(self, *, elements=None):
        """Adopt existing element alignments as deliberate reference inputs.

        Call before attaching bodies. The line fields are retained verbatim;
        nothing is zeroed or rewritten. Attachments adopt current inputs even
        without this call; explicit adoption freezes/checks that classification.
        Installation draws and dynamic sources remain separate body layers.
        """
        if self._attachments:
            raise ValueError(
                'Adopt the reference before creating mechanics attachments'
            )
        names = (
            self.line.element_names
            if elements is None
            else ((elements,) if isinstance(elements, str) else elements)
        )
        pending = {}
        for name in names:
            element = self.line.element_dict[name]
            if getattr(element, 'allow_rot_and_shift', False):
                pending[name] = _values(element)
        self._reference_inputs.update(pending)
        return self

    def get_reference_table(self):
        """Adopted deliberate numeric input offsets; no installation or dynamics."""
        return Table(
            dict(
                name=np.array(list(self._reference_inputs)),
                **{
                    name: np.array([v[i] for v in self._reference_inputs.values()])
                    for i, name in enumerate(_FIELDS)
                },
            )
        )

    @property
    def vibrations_enabled(self):
        return self._vibrations_enabled

    @property
    def controls_enabled(self):
        return self._controls_enabled

    @property
    def automatic_updates_enabled(self):
        return self._vibrations_enabled or self._controls_enabled

    @property
    def last_applied_time_s(self):
        return self._last_applied_time_s

    def _consumed_bodies(self):
        return {attachment['body'] for attachment in self._attachments.values()}

    def _source_enabled(self, binding):
        if not self._source_mode_configured:
            return True
        return (
            self._controls_enabled
            if binding.role == 'control'
            else self._vibrations_enabled
        )

    def _disabled_tracking_sources(self):
        key = (
            self.manager._revision,
            self._source_mode_configured,
            self._vibrations_enabled,
            self._controls_enabled,
        )
        if key != self._disabled_source_cache_key:
            self._disabled_source_cache = tuple(
                name
                for name, binding in self.manager.sources.items()
                if not self._source_enabled(binding)
            )
            self._disabled_source_cache_key = key
        return self._disabled_source_cache

    def _remember_applied(
        self, time_s, source_values, *, disabled_sources=None, time_series=None
    ):
        self._last_applied_time_s = float(time_s)
        if self._control_metadata_revision != self.manager._revision:
            needed = self.manager._body_closure(self._consumed_bodies())
            controls = {
                name: binding
                for name, binding in self.manager.sources.items()
                if binding.role == 'control' and needed.intersection(binding.targets)
            }
            self._relevant_control_names = frozenset(controls)
            self._relevant_live_control_names = frozenset(
                name for name, binding in controls.items() if binding.live
            )
            self._control_metadata_revision = self.manager._revision
        if disabled_sources is None:
            disabled_sources = self._disabled_tracking_sources()
            time_series = self._time_series
        self._last_applied_time_series = time_series
        self._last_applied_controls = self._relevant_control_names.difference(
            disabled_sources
        )
        if (
            not self._relevant_live_control_names
            and self._applied_source_recorder is None
        ):
            self._last_control_values.clear()
            return
        self._last_control_values = {
            name: np.array(values, copy=True)
            for name, values in source_values.items()
            if name in self._relevant_live_control_names
        }
        for name in disabled_sources:
            binding = self.manager.sources[name]
            if name in self._relevant_live_control_names:
                self._last_control_values[name] = np.zeros((len(binding.targets), 6))
        if self._applied_source_recorder is not None:
            self._applied_source_recorder.capture(
                time_s,
                source_values,
                disabled_sources=disabled_sources,
                time_series=time_series,
            )

    def _held_control_values(self):
        needed = self.manager._body_closure(self._consumed_bodies())
        values = {}
        time_s = self._last_applied_time_s
        for name, binding in self.manager.sources.items():
            if binding.role != 'control' or not needed.intersection(binding.targets):
                continue
            if time_s is None or name not in self._last_applied_controls:
                values[name] = np.zeros((len(binding.targets), 6))
            elif name in self._last_control_values:
                values[name] = self._last_control_values[name]
            elif (
                not binding.live
                and self._last_applied_time_series is not None
                and name in self._last_applied_time_series.values
            ):
                record = self._last_applied_time_series.values[name]
                values[name] = np.array(
                    [
                        [
                            np.interp(
                                time_s,
                                self._last_applied_time_series.time_s,
                                record[:, i, j],
                            )
                            for j in range(6)
                        ]
                        for i in range(len(binding.targets))
                    ]
                )
            else:
                values[name] = self.manager._evaluate_source(
                    binding, np.array([time_s])
                )[0]
        return values

    def _preflight_applied(
        self, time_s, source_values, *, disabled_sources=None, time_series=None
    ):
        if self._applied_source_recorder is not None:
            if disabled_sources is None:
                disabled_sources = self._disabled_tracking_sources()
                time_series = self._time_series
            self._applied_source_recorder.preflight(
                time_s,
                source_values,
                disabled_sources=disabled_sources,
                time_series=time_series,
            )

    def record_applied_sources(
        self, *, sources, max_samples=4096, max_memory_bytes=8 * 1024**2
    ):
        """Record a bounded selected live-input trace; no implicit pose history."""
        from .applied_history import AppliedSourceRecorder

        recorder = AppliedSourceRecorder(
            self,
            sources=sources,
            max_samples=max_samples,
            max_memory_bytes=max_memory_bytes,
        )
        self._applied_source_recorder = recorder
        return recorder

    def enable_vibrations(
        self, *, time_series=None, time_origin_s=None, sampling='turn', backend='auto'
    ):
        """Enable turn-boundary motion updates using Particles.time_s [s].

        Geometry and static placement freeze for each tracking call. The auto
        backend uses context-native analytic sources and hierarchy/lowering
        kernels, with batched host evaluation for sampled/custom sources.
        backend='reference' retains explicit Python
        lowering. No xdeps chain or per-particle source evaluation is installed.
        Auto retains reference validation for collective callbacks and nonempty
        or custom logs. An empty builtin Log executes no callbacks and retains
        native motion updates under the ordinary logging scheduler.
        Explicit compiled mode requires their geometry/static placement to stay
        fixed for the call; strength changes are independent of that contract.
        Shared prescribed fields require an explicit common physical
        time origin per line; there is no shared mutable filter/controller here.
        sampling='passage' prepares each occurrence at t0+s/(beta0*c), assuming
        the reference speed. SyncTime and partial-turn integration stay guarded.
        """
        self._enable_updates(
            time_series=time_series,
            time_origin_s=time_origin_s,
            sampling=sampling,
            backend=backend,
            disturbances=True,
        )

    def enable_controls(
        self, *, time_series=None, time_origin_s=None, sampling='turn', backend='auto'
    ):
        """Enable applied control motion without implicitly enabling disturbances."""
        self._enable_updates(
            time_series=time_series,
            time_origin_s=time_origin_s,
            sampling=sampling,
            backend=backend,
            disturbances=self._vibrations_enabled,
        )

    def _enable_updates(
        self, *, time_series, time_origin_s, sampling, backend, disturbances
    ):
        attached = [v for v in self.manager._line_views.values() if v._attachments]
        if not self._attachments or (len(attached) != 1 and time_origin_s is None):
            raise NotImplementedError(
                'Automatic multi-line sampling needs an explicit common time_origin_s per line'
            )
        if time_origin_s is not None and not np.isfinite(time_origin_s):
            raise ValueError('time_origin_s must be finite physical seconds')
        if sampling not in ('turn', 'passage'):
            raise ValueError('Use turn or passage sampling')
        if backend not in ('auto', 'compiled', 'reference'):
            raise ValueError('Use auto, compiled or reference backend')
        if backend == 'compiled' and sampling != 'turn':
            raise ValueError('Compiled mechanics currently freezes at turn boundaries')
        needed = self.manager._body_closure(self._consumed_bodies())
        bindings = [
            binding
            for binding in self.manager.sources.values()
            if needed.intersection(binding.targets)
            and (disturbances or binding.role == 'control')
        ]
        if sampling == 'passage' and any(binding.live for binding in bindings):
            raise ValueError(
                'Live sources support turn freezes, not predicted passage-time motion'
            )
        if time_series is not None:
            time_series._check(self.manager)
        if any(
            requires_time_series(b.source)
            and not b.live
            and (time_series is None or b.name not in time_series.values)
            for b in bindings
        ):
            raise ValueError(
                'Call compute_time_series first and supply time_series for PSD tracking'
            )
        self.manager._check_aliases()
        self._check()
        self._time_series = time_series
        self._time_origin_s = 0.0 if time_origin_s is None else float(time_origin_s)
        self._sampling = sampling
        self._backend = backend
        self._runtime = None
        self._vibrations_enabled = disturbances
        self._controls_enabled = True
        self._source_mode_configured = True

    def disable_vibrations(self):
        """Remove disturbances, retaining applied control and its automatic updates.

        Source phases/records are not reset. A later enable resumes at the
        particles' physical time, rather than restarting a random waveform.
        """
        disabled = tuple(
            name
            for name, binding in self.manager.sources.items()
            if binding.role == 'disturbance'
        )
        held = self._held_control_values() if self._sampling == 'turn' else {}
        self.apply(
            time_s=0.0
            if self._last_applied_time_s is None
            else self._last_applied_time_s,
            time_series=self._time_series,
            disabled_sources=disabled,
            sampling=self._sampling,
            _source_values=held,
        )
        self._vibrations_enabled = False
        needed = self.manager._body_closure(self._consumed_bodies())
        self._controls_enabled = self._controls_enabled and any(
            binding.role == 'control' and needed.intersection(binding.targets)
            for binding in self.manager.sources.values()
        )
        if self._time_series is not None and not any(
            binding.role == 'control'
            and name in self._time_series.values
            and needed.intersection(binding.targets)
            for name, binding in self.manager.sources.items()
        ):
            self._time_series = None
        self._source_mode_configured = True
        self._runtime = None

    def freeze_updates(self):
        """Stop automatic updates while retaining the currently applied element pose."""
        self._finish_tracking()
        self._vibrations_enabled = False
        self._controls_enabled = False
        self._source_mode_configured = True
        self._runtime = None

    def _prepare_for_tracking(self, *, log=None):
        """Validate once at call entry; geometry freezes until the call ends."""
        from ..tracker import Log

        self._prepared_empty_log = None
        if self._applied_source_recorder is not None:
            self._applied_source_recorder.check_case()
        if self._time_series is not None:
            self._time_series._check(self.manager)
        # Do not trust callbacks or subclasses overriding iteration. The exact
        # empty builtin Log has no user code; Tracker still resets its output
        # and owns the logging scheduler. Nonempty logs retain per-turn checks.
        empty_log = type(log) is Log and not log
        use_compiled = (
            self._backend != 'reference'
            and self._sampling == 'turn'
            and not self.line.enable_time_dependent_vars
            and (log is None or empty_log or self._backend == 'compiled')
            and (not self.line.tracker.iscollective or self._backend == 'compiled')
        )
        if self._backend == 'compiled' and not use_compiled:
            raise ValueError(
                'Compiled mechanics requires geometry fixed during a tracking call'
            )
        if use_compiled:
            from .runtime import PreparedMechanics

            if (
                self._runtime is None
                or self._runtime.revision != self.manager._revision
                or self._runtime.buffer is not self.line.tracker._buffer
            ):
                self.manager._check_aliases()
                self._check()
                try:
                    self._runtime = PreparedMechanics(self)
                except NotImplementedError:
                    if self._backend == 'compiled':
                        raise
                    self._runtime = None
            else:
                self._runtime.guard.check()
                self._runtime.refresh_static()
        else:
            self.manager._check_aliases()
            self._check()
            self._runtime = None
        if empty_log and self._backend == 'auto' and self._runtime is not None:
            self._prepared_empty_log = log

    def _finish_tracking(self):
        try:
            if self._runtime is not None:
                self._runtime.finish()
        finally:
            self._prepared_empty_log = None

    def _check_prepared_empty_log(self, log):
        if log is not None and log:
            raise ValueError(
                'An initially empty Log changed during prepared mechanics; '
                'restart tracking with a fixed log'
            )

    def _apply_for_tracking(self, time_s):
        admitted_log = self._prepared_empty_log
        self._check_prepared_empty_log(admitted_log)
        if self._runtime is not None:
            self._runtime.apply(time_s + self._time_origin_s, self._time_series)
        else:
            self.apply(
                time_s=time_s + self._time_origin_s,
                time_series=self._time_series,
                sampling=self._sampling,
                disabled_sources=self._disabled_tracking_sources(),
            )
        # A custom source may have added callbacks while evaluating its values.
        # Reject before Tracker executes them under a frozen-geometry contract.
        self._check_prepared_empty_log(admitted_log)

    def apply(
        self,
        *,
        time_s=0.0,
        time_series=None,
        disabled_sources=(),
        sampling='turn',
        _source_values=None,
    ):
        """Update this line only at a physical time; other beam buffers stay put.

        Passage mode uses reference-speed arrival times, not particle-specific
        zeta/delta arrival. It prepares frozen per-occurrence maps for one turn.
        This Python reference is intentionally measured separately from kernels.
        """
        self.manager._check_aliases()
        self._check()
        if self._applied_source_recorder is not None:
            if self._applied_source_recorder.active and sampling != 'turn':
                raise ValueError('Applied source recording supports turn freezes only')
            self._applied_source_recorder.check_case()
        consumed = self._consumed_bodies()
        live_values = dict(_source_values or {})
        live_values.update(
            self.manager._sample_live_values(
                time_s,
                bodies=consumed,
                disabled_sources=set(disabled_sources) | set(live_values),
            )
        )
        if sampling == 'turn':
            poses = self.manager.get_body_poses(
                time_s=time_s,
                time_series=time_series,
                disabled_sources=disabled_sources,
                bodies=consumed,
                _source_values=live_values,
            )
            updates = self._updates(poses, check=False)
        elif sampling == 'passage':
            from scipy.constants import c

            if self.line.particle_ref is None:
                raise ValueError('Set a reference particle for passage-time sampling')
            needed = self.manager._body_closure(consumed)
            if any(
                binding.live
                and name not in disabled_sources
                and needed.intersection(binding.targets)
                for name, binding in self.manager.sources.items()
            ):
                raise ValueError(
                    'Live sources support turn freezes, not predicted passage-time motion'
                )
            speed = float(self.line.particle_ref.beta0[0]) * c
            table = self.line.get_table()
            updates = []
            for name in self._attachments:
                poses = self.manager.get_body_poses(
                    time_s=time_s + float(table['s', name]) / speed,
                    time_series=time_series,
                    disabled_sources=disabled_sources,
                    bodies=[self._attachments[name]['body']],
                )
                updates.extend(self._updates(poses, names=[name], check=False))
        else:
            raise ValueError('Use turn or passage sampling')
        self._preflight_applied(
            time_s,
            live_values,
            disabled_sources=disabled_sources,
            time_series=time_series,
        )
        for attachment, element, values in updates:
            for field, value in zip(_FIELDS, values):
                setattr(element, field, value)
            attachment['last_written'] = values.copy()
        self._remember_applied(
            time_s,
            live_values,
            disabled_sources=disabled_sources,
            time_series=time_series,
        )

    @property
    def line(self):
        return self._line_ref()

    @property
    def reference_frame(self):
        return Frame(self._reference.matrix)

    def set_reference_frame(self, frame):
        """Place this line's default survey in the environment world, before attachment."""
        if self._attachments:
            raise ValueError('Set the survey placement before attaching mechanics')
        self._reference = _frame(frame)
        self._reference_is_explicit = True
        self.manager._revision += 1

    def new_girder(
        self,
        name,
        *,
        elements,
        anchor='center',
        parent=None,
        mount=None,
        feet_local_m=None,
        support_class='',
    ):
        return self.manager.new_girder(
            name,
            line=self.line,
            elements=elements,
            anchor=anchor,
            parent=parent,
            mount=mount,
            feet_local_m=feet_local_m,
            support_class=support_class,
        )

    def new_girders(self, assignments, *, support_class=''):
        """Create an inspectable batch of supports with one full-line survey.

        assignments maps a new body name to element names, or to a dictionary
        with elements and optional anchor, parent, mount, feet_local_m,
        support_class. A mount is a parent-local Frame, as in new_body.
        For example {'g1': ['q1', 's1'], 'g2': {'elements': ['q2'],
        'support_class': 'arc_quad'}}. Group members may be non-contiguous.
        All definitions are validated before any body/binding is installed.
        Existing supports may be parents; create parent levels first.
        """
        return self.new_bodies(assignments, kind='girder', support_class=support_class)

    def new_bodies(self, assignments, *, kind='body', support_class=''):
        """Create attached rigid bodies in one validated batch and one survey.

        assignments has the same element/anchor/parent/mount/contact options as
        new_girders. A parent-local mount and an explicit world anchor are
        mutually exclusive. kind labels every body, e.g. 'magnet' or 'pickup'.
        Parents must already exist. All definitions are validated before installation.
        """
        if not hasattr(assignments, 'items') or not assignments:
            raise ValueError('Use a nonempty mapping from support names to assignments')
        definitions = []
        all_names = []
        for name, value in assignments.items():
            options = dict(value) if isinstance(value, dict) else {'elements': value}
            unknown = set(options) - {
                'elements',
                'anchor',
                'parent',
                'mount',
                'feet_local_m',
                'support_class',
            }
            if unknown or 'elements' not in options:
                raise ValueError(
                    'Each assignment needs elements and supported girder options'
                )
            names = options.pop('elements')
            names = (names,) if isinstance(names, str) else tuple(names)
            if not names:
                raise ValueError('Select nonempty unique element names')
            options.setdefault('anchor', 'center')
            options.setdefault('parent', None)
            options.setdefault('feet_local_m', None)
            options.setdefault('support_class', support_class)
            options['anchor'] = self.manager._resolve_mount(
                anchor=options['anchor'],
                parent=options['parent'],
                mount=options.pop('mount', None),
            )
            definitions.append((name, names, options))
            all_names.extend(names)
        candidates = self._candidates(all_names)
        # _candidates returns line order once; do not rescan/sort the line per group.
        owner = {element: name for name, names, _ in definitions for element in names}
        groups = {name: [] for name, _, _ in definitions}
        for candidate in candidates:
            groups[owner[candidate['element']]].append(candidate)
        prepared = []
        for name, names, options in definitions:
            selected = groups[name]
            body = self.manager._body_definition(
                name, kind=kind, candidates=selected, **options
            )
            prepared.append((body, selected))
        for body, selected in prepared:
            self.manager._bodies[body.name] = body
            for candidate in selected:
                candidate['body'] = body.name
        self._install_candidates(candidates)
        self.manager._revision += 1
        return {body.name: body for body, _ in prepared}

    def attach(self, body, *, elements):
        """Attach this line's elements to an existing physical body, possibly shared."""
        name = body.name if isinstance(body, RigidBody) else body
        if name not in self.manager.bodies or (
            isinstance(body, RigidBody) and self.manager[name] is not body
        ):
            raise ValueError('Body must belong to this environment')
        candidates = self._candidates(elements)
        self._attach(name, candidates)
        self.manager._revision += 1

    def clear(self):
        """Restore this line's local alignment and detach it from mechanics.

        Shared body/source definitions are retained for other lines. Use
        env.mechanics.clear() to remove the whole mechanical configuration.
        """
        updates = self._removal_updates()
        if self._attachments and self.line.aperture.is_built:
            self.line.aperture.discard_model()
        for attachment, element, values in updates:
            for field, value in zip(_FIELDS, values):
                setattr(element, field, value)
        self._clear_bindings()
        self.manager._revision += 1
        return self

    def _removal_updates(self):
        if self._runtime is not None:
            self._runtime.finish()
        updates = []
        alignment_refs = set()
        for name, attachment in self._attachments.items():
            element = self.line.element_dict.get(name)
            if element is None or id(element) != attachment['object_id']:
                raise ValueError(
                    f'{name}: attached object changed; original alignment cannot be restored automatically'
                )
            if not np.array_equal(_values(element), attachment['last_written']):
                raise ValueError(
                    f'{name}: effective alignment was edited; clear mechanics before editing local alignment'
                )
            for field in _FIELDS:
                alignment_refs.add(getattr(self.line.env.ref[name], field))
                if (
                    getattr(getattr(self.line.env.ref[name], field), '_expr', None)
                    is not None
                ):
                    raise ValueError(
                        f'{name}.{field}: a new alignment expression must not be overwritten'
                    )
            updates.append((attachment, element, attachment['baseline']))
        _check_alignment_consumers(self.line.env.ref_manager, alignment_refs)
        return updates

    def _clear_bindings(self):
        self._attachments.clear()
        self._element_names = None
        self._line_geometry = None
        self._vibrations_enabled = False
        self._controls_enabled = False
        self._source_mode_configured = False
        self._last_applied_time_s = None
        self._last_applied_time_series = None
        self._last_applied_controls = frozenset()
        self._last_control_values.clear()
        self._applied_source_recorder = None
        self._time_series = None
        self._runtime = None
        self._reference = Frame()
        self._reference_is_explicit = False
        self._reference_inputs.clear()
        self._bound_line = None

    def _candidates(self, elements):
        if self._attachments:
            self._check()
        names = (elements,) if isinstance(elements, str) else tuple(elements)
        if not names or len(set(names)) != len(names):
            raise ValueError('Select nonempty unique element names')
        line = self.line
        if any(
            type(element).__name__ == 'Replica'
            or (hasattr(element, 'weight') and hasattr(element, '_parent'))
            for element in line._elements
        ):
            raise ValueError(
                'Mechanics bindings require an unsliced line without replicas'
            )
        if (
            not self._attachments
            and not self._reference_is_explicit
            and any(
                v is not self and v._attachments
                for v in self.manager._line_views.values()
            )
        ):
            raise ValueError(
                'Set an explicit survey reference frame before attaching a second line'
            )
        order = {}
        counts = {}
        for index, name in enumerate(line.element_names):
            order.setdefault(name, index)
            counts[name] = counts.get(name, 0) + 1
        elements_by_id = {}
        for ll in self.manager.env._lines_weakrefs:
            if ll.env is not self.manager.env:
                continue
            for name in ll.element_names:
                element = ll.element_dict[name]
                elements_by_id[id(element)] = elements_by_id.get(id(element), 0) + 1
        alignment_refs = set()
        for name in names:
            if counts.get(name, 0) != 1 or name in self._attachments:
                raise ValueError(
                    f'{name}: unknown, repeated or already attached occurrence'
                )
            element = line.element_dict[name]
            if name in self._reference_inputs and not np.array_equal(
                _values(element), self._reference_inputs[name]
            ):
                raise ValueError(
                    f'{name}: adopted reference input changed; adopt again before attaching'
                )
            if (
                not getattr(element, 'allow_rot_and_shift', False)
                or type(element).__name__ == 'Replica'
                or (hasattr(element, 'weight') and hasattr(element, '_parent'))
            ):
                raise ValueError(
                    f'{name}: first lowering supports unsliced alignment-capable elements'
                )
            if elements_by_id.get(id(element), 0) != 1:
                raise ValueError(
                    f'{name}: shared/repeated element objects require an occurrence-buffer lowering'
                )
            for field in _FIELDS:
                alignment_refs.add(getattr(line.env.ref[name], field))
                if (
                    hasattr(element, field)
                    and getattr(getattr(line.env.ref[name], field), '_expr', None)
                    is not None
                ):
                    raise ValueError(
                        f'{name}.{field}: expression-driven alignment is not supported by numeric lowering'
                    )
        _check_alignment_consumers(line.env.ref_manager, alignment_refs)
        survey = line.survey(include_element_frames=True, reverse=False)
        output = []
        selected_names = set(names)
        for name in order:
            if name not in selected_names:
                continue
            element = line.element_dict[name]
            output.append(
                dict(
                    element=name,
                    ref_start=(
                        self._reference @ survey.get_frame(name, which='ref_start')
                    ).matrix,
                    ref_end=(
                        self._reference @ survey.get_frame(name, which='ref_end')
                    ).matrix,
                    base_element=(
                        self._reference @ survey.get_frame(name, which='elem_start')
                    ).matrix,
                    base_element_end=(
                        self._reference @ survey.get_frame(name, which='elem_end')
                    ).matrix,
                    baseline=_values(element),
                    last_written=_values(element),
                    geometry=_geometry(element),
                    object_id=id(element),
                )
            )
        return output

    def _attach(self, name, candidates):
        for candidate in candidates:
            candidate['body'] = name
        self._install_candidates(candidates)

    def _install_candidates(self, candidates):
        line = self.line
        geometry = self._geometry_signature()
        self._attachments.update((c['element'], c) for c in candidates)
        self._element_names = tuple(line.element_names)
        self._line_geometry = geometry
        # Effective element fields are owned until explicit restoration/clear.
        # Losing an anonymous line must not discard the only baseline record.
        self._bound_line = line

    def _geometry_signature(self):
        return tuple(
            (
                id(self.line.element_dict[n]),
                _reference_geometry(self.line.element_dict[n]),
            )
            for n in self.line.element_names
        )

    def validate(self):
        """Validate captured line geometry and alignment ownership explicitly.

        Geometry expressions remain live and are read at attachment. Changing
        their evaluated lengths/angles invalidates the captured survey; rebuild
        the bindings. Strength-only changes need no geometry rebuild. Compiled
        tracking validates once at entry and assumes fixed reference geometry
        for that call, rather than rescanning the line at every freeze point.
        """
        self.manager._check_aliases()
        self._check()
        return self

    def _check(self, *, allow_built_aperture=False):
        # Explicit inspection inside a callback synchronizes one packed output,
        # rather than mistaking the prepared kernel's own writes for user edits.
        if self._runtime is not None:
            self._runtime.finish()
        if (
            self._attachments
            and self.line.aperture.is_built
            and not allow_built_aperture
        ):
            raise ValueError(
                'Discard/rebuild the aperture survey model when applying mechanics; pipe-body coupling is not implemented yet'
            )
        if (
            self._element_names is not None
            and tuple(self.line.element_names) != self._element_names
        ):
            raise ValueError(
                'Line sequence changed after mechanics attachment; rebuild bindings'
            )
        if (
            self._line_geometry is not None
            and self._geometry_signature() != self._line_geometry
        ):
            raise ValueError(
                'Line reference geometry changed after attachment; rebuild survey bindings'
            )
        alignment_refs = set()
        for name, a in self._attachments.items():
            element = self.line.element_dict[name]
            if id(element) != a['object_id']:
                raise ValueError(f'{name}: attached element object was replaced')
            if _geometry(element) != a['geometry']:
                raise ValueError(f'{name}: nominal geometry changed after attachment')
            if not np.array_equal(_values(element), a['last_written']):
                raise ValueError(
                    f'{name}: local alignment edited after attachment; restore/rebind explicitly'
                )
            for field in _FIELDS:
                alignment_refs.add(getattr(self.line.env.ref[name], field))
                if (
                    hasattr(element, field)
                    and getattr(getattr(self.line.env.ref[name], field), '_expr', None)
                    is not None
                ):
                    raise ValueError(
                        f'{name}.{field}: expression-driven alignment is not supported by numeric lowering'
                    )
        _check_alignment_consumers(self.line.env.ref_manager, alignment_refs)

    def _updates(self, poses, *, names=None, check=True):
        if check:
            self._check()
        updates = []
        for name in self._attachments if names is None else names:
            a = self._attachments[name]
            element = self.line.element_dict[name]
            body = self.manager[a['body']]
            if hasattr(poses, '_deformations'):
                values = lower(a, body._nominal, poses._deformations[body.name])
            else:
                # Private callers may still provide explicit world frames.
                target = (
                    poses[body.name]
                    @ Frame(body._nominal).inverse()
                    @ Frame(a['base_element'])
                )
                effective = (
                    Frame(a['ref_start']).inverse()
                    @ target
                    @ Frame().rotate_s(-element.rot_s_rad)
                )
                values = np.array(
                    [*effective.XYZ, effective.phi, effective.theta, effective.psi, 0.0]
                )
            updates.append((a, element, values))
        return updates

    def get_element_poses(self, *, time_s=0.0, time_series=None, disabled_sources=()):
        self._check()
        poses = self.manager.get_body_poses(
            time_s=time_s,
            time_series=time_series,
            disabled_sources=disabled_sources,
            bodies=self._consumed_bodies(),
        )
        return {
            name: poses[a['body']]
            @ self.manager[a['body']].nominal_frame.inverse()
            @ Frame(a['base_element'])
            for name, a in self._attachments.items()
        }

    def read_bpm(
        self, name, *, time_s=0.0, time_series=None, beam_frame=None, **coordinates
    ):
        """Project a nominal-frame beam ray into an attached pickup's moved frame."""
        self._check()
        from .observation import project_to_bpm
        from .local_deformations import entry_affine, rotation_increment

        if name not in self._attachments:
            raise ValueError('Attach the pickup Device to mechanics before reading it')
        attachment = self._attachments[name]
        body = self.manager[attachment['body']]
        deformation = self.manager._get_body_deformations(
            time_s=time_s, time_series=time_series, bodies=[body.name]
        )[body.name]
        shift, rotation = entry_affine(attachment, body._nominal, deformation)
        design_rotation = np.eye(3) + rotation_increment(psi=attachment['geometry'][-1])
        target = Frame.from_survey(shift, rotation @ design_rotation)
        if beam_frame is None:
            beam_frame = Frame()
        else:
            reference = attachment['ref_start']
            beam_frame = Frame.from_survey(
                reference[:3, :3].T @ (beam_frame.XYZ - reference[:3, 3]),
                reference[:3, :3].T @ beam_frame.matrix[:3, :3],
            )
        return project_to_bpm(beam_frame=beam_frame, bpm_frame=target, **coordinates)

    def get_table(self, *, time_s=None, time_series=None):
        """Attached element occurrences, optionally with read-only world poses.

        Each attached element has its own row, even when several share a body.
        Unattached line elements are available in line.get_table().
        """
        from .tables import element_table

        return element_table(self, time_s=time_s, time_series=time_series)

    def get_alignment_table(self):
        """Original element-local versus current effective alignment fields.

        Body settings (manager.get_table) are deliberate girder alignment;
        computed source tables are vibration inputs. These are distinct inputs,
        composed as rigid transforms. Differences of Euler angles are not an
        exact additive contribution decomposition. Design rot_s_rad is shown
        separately and is never changed by the numeric mechanics lowering.
        """
        names = list(self._attachments)
        data = dict(
            name=np.array(names),
            body=np.array([self._attachments[n]['body'] for n in names]),
        )
        for index, field in enumerate(_FIELDS):
            data['local_' + field] = np.array(
                [self._attachments[n]['baseline'][index] for n in names]
            )
            data['effective_' + field] = np.array(
                [getattr(self.line.element_dict[n], field) for n in names]
            )
        data['design_rot_s_rad'] = np.array(
            [self.line.element_dict[n].rot_s_rad for n in names]
        )
        return Table(data)
