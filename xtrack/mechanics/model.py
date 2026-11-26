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

from dataclasses import dataclass, field as dataclass_field
from types import MappingProxyType
from weakref import WeakKeyDictionary, ref

import numpy as np

from ..survey import Frame
from ..table import Table
from ..base_element import Replica
from ..beam_elements.slice_base import _SliceBase
from .sources import decode_source, encode_source, requires_time_series
from .motion import RigidMotion, COORDINATES, frozen_array, checked_times, sample_at
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
_SOLENOID_AXIS_FIELDS = ('x0', 'y0')


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


def _values(element, fields=_FIELDS):
    return np.array([float(getattr(element, name)) for name in fields])


def _attachment_fields(attachment):
    return _SOLENOID_AXIS_FIELDS if 'projection' in attachment else _FIELDS


def _check_transverse_values(body, values):
    if body._transverse_only and np.any(np.asarray(values)[..., 2:]):
        raise ValueError(
            f'{body.name}: relative solenoid axis supports only transverse x/y '
            'translations; move the common chart for rigid rotations and longitudinal motion'
        )


def _reference_values(body):
    values = [getattr(body, name) for name in COORDINATES]
    if np.iscomplexobj(values):
        raise ValueError(f'{body.name}: local alignment must be real')
    values = np.asarray(values, dtype=float)
    if values.shape != (6,):
        raise ValueError(f'{body.name}: local alignment must be six scalar coordinates')
    if not np.all(np.isfinite(values)):
        raise ValueError(f'{body.name}: local alignment must be finite')
    _check_transverse_values(body, values)
    return values


def _geometry(element):
    return tuple(float(getattr(element, name, 0.0)) for name in _GEOMETRY)


def _reference_geometry(element):
    # Survey dispatches frame-changing elements through this protocol instead
    # of the length/angle/tilt arc. Capture the same increment, so Translation,
    # Rotation (including its sequence), native slice edges and custom survey
    # elements cannot change the reference path behind an adopted binding.
    # GeometryGuard also traces the scalar storage read by this call once.
    frame_increment = None
    if hasattr(element, 'track_frame'):
        frame = Frame()
        element.track_frame(frame, backtrack=False)
        frame_increment = tuple(frame.matrix.ravel())
    return (
        _geometry(element),
        bool(getattr(element, 'isthick', False)),
        getattr(element, 'rbend_model', None),
        frame_increment,
    )


def _resolved_slice_parent(element, elements):
    """Check both the survey name and the tracking reference of a native slice."""
    name = getattr(element, 'parent_name', None)
    parent = elements.get(name)
    if parent is None or isinstance(parent, (Replica, _SliceBase)):
        raise ValueError(
            'Native slice reference geometry requires a resolved, named parent'
        )
    actual = element._parent
    if actual is not None and actual is not parent:
        raise ValueError('Native slice reference geometry changed: inconsistent parent')
    return parent


def _slice_reference_geometry(element, elements):
    parent = _resolved_slice_parent(element, elements)
    # The survey has already linked the native slices. Trace the actual
    # tracking reference as well: the cached Python parent alone cannot detect
    # a changed Xobject pointer. GeometryGuard gathers these bytes in bulk.
    pointer = element._xobject._parent
    if (
        pointer is None
        or pointer._buffer is not parent._buffer
        or pointer._offset != parent._offset
    ):
        raise ValueError('Native slice reference geometry changed: inconsistent pointer')
    return (
        element.parent_name,
        id(parent),
        float(element.weight),
        float(element.slice_offset),
        _reference_geometry(parent),
    )


def _storage_identity(element):
    xobject = getattr(element, '_xobject', None)
    if xobject is not None:
        return id(xobject._buffer), xobject._offset
    return None


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


class _BodyDeformations(dict):
    """Composed increments and the exact local inputs before inheritance."""

    def __init__(self, composed, local):
        super().__init__(composed)
        self.local = local


@dataclass(frozen=True, slots=True)
class FrozenMechanicalState:
    """One line's applied alignment and matching fixed-survey aperture cuts.

    time_s is the request; pose_time_s is the actual freeze used. static_case
    maps body names to six reference coordinates followed by six installation
    coordinates. design_alignment holds the adopted element alignment fields.
    live_values contains the normalized enabled live inputs used in this freeze.
    Element frames and aperture checks are detached outputs for inspection
    or for building a frozen tracking line.
    """

    time_s: float
    pose_time_s: float
    query_kind: str
    element_poses: object
    aperture: object
    pipe_table: object
    limits: object
    design_alignment: object
    static_case: object
    source_names: tuple
    disabled_sources: tuple
    prescribed_record: object
    live_values: object


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
    installation: RigidOffset = dataclass_field(default_factory=RigidOffset)
    _feet: object = None
    kind: str = 'body'
    support_class: str = ''
    _transverse_only: bool = False

    def __setattr__(self, name, value):
        if name in (
            'name',
            'parent',
            '_nominal',
            '_feet',
            'kind',
            'support_class',
            '_transverse_only',
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
    transverse_indices: tuple = ()

    def _check_transverse(self, values):
        if self.transverse_indices and np.any(
            values[..., list(self.transverse_indices), 2:]
        ):
            raise ValueError(
                f'{self.name}: relative solenoid axis supports only transverse x/y translations'
            )


class MechanicalTimeSeries:
    """Reusable sampled source displacements, independent of live body settings.

    Values are read-only (time, target, 6) arrays in metres and survey radians.
    Reuse this object for matched comparisons. replay_modes reports source
    evaluation, linear interpolation, or previous-sample hold for each binding.
    All replay is bounded by the supplied record.
    """

    def __setattr__(self, name, value):
        if hasattr(self, name):
            raise AttributeError('MechanicalTimeSeries is read-only')
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        raise AttributeError('MechanicalTimeSeries is read-only')

    def __init__(self, manager, time_s, values, *, interpolation='linear', _providers=None):
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
        for name, value in arrays.items():
            manager.sources[name]._check_transverse(value)
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
        if interpolation not in ('linear', 'previous'):
            raise ValueError('Use linear or previous replay for supplied samples')
        self._providers = MappingProxyType(dict(_providers or {}))
        self.replay_modes = MappingProxyType({
            name: 'source' if name in self._providers else interpolation for name in arrays
        })

    def _check_time(self, time_s):
        if time_s < self.time_s[0] or time_s > self.time_s[-1]:
            raise ValueError('Requested time lies outside the computed motion record')

    def _sample(self, binding, time_s, *, positions=None, axes=None, selection=slice(None)):
        name = binding.name
        provider = self._providers.get(name)
        if provider is None:
            return sample_at(self.time_s, self.values[name], time_s,
                             interpolation=self.replay_modes[name], selection=selection)
        self._check_time(time_s)
        if type(provider) is RigidMotion:
            return sample_at(provider.time_s, provider.values, time_s, selection=selection)
        values = self._manager()._evaluate_source(
            binding, np.array([time_s]), positions=positions, axes=axes, provider=provider,
        )
        return values[0, selection]

    @staticmethod
    def _freeze(value):
        array = np.asarray(value, dtype=float)
        return np.frombuffer(array.tobytes(), dtype=float).reshape(array.shape)

    def get_table(self, body):
        """Direct source coordinates; no static or inherited motion.

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
    Ownership checks cover physical lines in this environment. Explicitly
    sharing writable Xobject storage with another environment is unsupported;
    use env.copy() for independent mechanics scenarios.
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

    def attach_bodies(self, assignments, *, line=None):
        """Attach existing physical bodies with one line survey."""
        return self._line(line).attach_bodies(assignments)

    def new_solenoid_axis(self, name, *, elements, line=None, anchor='center'):
        """Create a transverse field-axis body relative to one attached chart."""
        return self._line(line).new_solenoid_axis(
            name, elements=elements, anchor=anchor
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
        transverse_only=False,
    ):
        """Validate/build a body without changing the manager or line."""
        if not isinstance(kind, str) or not kind:
            raise ValueError(
                'Use a nonempty body kind, e.g. magnet, pickup or equipment'
            )
        if not isinstance(support_class, str):
            raise ValueError('support_class must be a string')
        if not isinstance(transverse_only, (bool, np.bool_)):
            raise ValueError('A transverse-only body flag must be boolean')
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
            name, parent, frozen, _feet=feet, kind=kind, support_class=support_class,
            _transverse_only=bool(transverse_only),
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
        transverse_only=False,
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
            transverse_only=transverse_only,
        )
        if view is not None:
            view._attach(name, candidates)
        self._bodies[name] = body
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
        RigidMotion and ModalMotion supply all six coordinates. A mounted
        body always inherits parent motion. Supply its response relative to
        the carried port, for example using RigidMotion.relative_to().
        The legacy reference='absolute' spelling is accepted only on roots.
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
        for target in targets:
            parent = self._bodies[target].parent
            if parent is not None and reference == 'absolute':
                raise ValueError(
                    f'{target}: a mounted body inherits parent motion. '
                    'Convert a supplied total record with RigidMotion.relative_to() '
                    'using the carried motion at the same port'
                )
            while parent is not None:
                if parent in targets:
                    raise ValueError(
                        'One source cannot drive both a body and its ancestor'
                    )
                parent = self._bodies[parent].parent
            if (
                frame == 'world'
                and self._bodies[target].parent is not None
            ):
                raise ValueError(
                    'World-field shortcut drives roots. Resolve a relative response at the mounted port'
                )
        from .sources import validate_source_bindings

        validate_source_bindings(source, self.sources.values())
        transverse_indices = tuple(
            index for index, target in enumerate(targets)
            if self._bodies[target]._transverse_only
        )
        if transverse_indices and frame == 'body_local' and component not in ('x', 'y'):
            raise ValueError(
                'A relative solenoid axis accepts only transverse x/y source components'
            )
        binding = SourceBinding(
            name, source, targets, component, 'relative', role, bool(live),
            transverse_indices,
        )
        self._sources[name] = binding
        self._revision += 1
        return binding

    def _evaluate_source(self, binding, time_s, *, positions=None, axes=None, provider=None):
        if positions is None:
            positions = np.array([self._bodies[n]._nominal[:3, 3] for n in binding.targets])
        source = binding.source if provider is None else provider
        values = np.asarray(source.evaluate(time_s=time_s, positions_m=positions))
        if np.iscomplexobj(values) or not np.all(np.isfinite(values)):
            raise ValueError(
                f'{binding.name}: source must return finite real displacement'
            )
        frame = getattr(binding.source, 'frame', 'body_local')
        if frame == 'body_local_rigid':
            if values.shape != (len(time_s), len(positions), 6):
                raise ValueError('Rigid source must return shape (times, targets, 6)')
            binding._check_transverse(values)
            # Providers may reuse their own buffer for the next binding.
            return np.array(values, copy=True)
        output = np.zeros((len(time_s), len(binding.targets), 6))
        if frame == 'world':
            if values.shape != (len(time_s), len(positions), 3):
                raise ValueError('World source must return shape (times, targets, 3)')
            if axes is None:
                axes = np.array([self._bodies[n]._nominal[:3, :3] for n in binding.targets])
            output[..., :3] = np.einsum('bji,tbj->tbi', axes, values)
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

    def compute_time_series(self, *, time_s, max_memory_bytes=128 * 1024**2,
                            disabled_sources=(), interpolation='auto'):
        """Compute reusable prescribed motion at explicit times [s], on the host.

        Live bindings are omitted, never frozen into a historical record.
        Inspect omitted_live_sources; their held values are read at each freeze.
        Analytic/measured models accept nonuniform increasing times; PSD models
        require a uniform grid. This explicit short-record backend has a
        conservative allocation guard and never creates a GPU history cache.
        Pass the same source mask to subsequent pose and tracking operations.

        Auto replay evaluates built-in sources at the requested physical time,
        retaining any prepared FFT realization and the original measured knots.
        Custom prescribed providers are sampled once and replayed linearly.
        Explicit linear or previous replay uses the supplied time grid for all
        sources. These are offline resampling choices, separate from tracking
        freeze cadence.
        """
        from .sources import prepare_source_replay, retained_fourier_bytes

        if interpolation not in ('auto', 'linear', 'previous'):
            raise ValueError('Use auto, linear or previous record interpolation')
        times = checked_times(time_s)
        mask = self._source_mask(disabled_sources)
        prescribed = {
            name: binding for name, binding in self.sources.items()
            if not binding.live and name not in mask
        }
        spectral = sum(
            retained_fourier_bytes(b.source, len(times), len(b.targets))
            for b in prescribed.values() if interpolation == 'auto'
        )
        estimate = spectral + 96 * len(times) * sum(len(b.targets) for b in prescribed.values()) + max(
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
        values, providers = {}, {}
        for name, binding in prescribed.items():
            positions = np.array([self._bodies[n]._nominal[:3, 3] for n in binding.targets])
            provider = (prepare_source_replay(binding.source, times, positions)
                        if interpolation == 'auto' else None)
            if provider is not None:
                providers[name] = provider
            values[name] = self._evaluate_source(binding, times, positions=positions, provider=provider)
        return MechanicalTimeSeries(
            self, times, values, _providers=providers,
            interpolation='previous' if interpolation == 'previous' else 'linear',
        )

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

    def _source_mask(self, names):
        names = (names,) if isinstance(names, str) else tuple(names)
        for name in names:
            if not isinstance(name, str) or name not in self.sources:
                raise ValueError(f'Unknown disabled source {name!r}')
        return tuple(dict.fromkeys(names))

    def _sample_live_values(self, time_s, *, bodies=None, disabled_sources=()):
        if not np.isscalar(time_s) or np.iscomplexobj(time_s) or not np.isfinite(time_s):
            raise ValueError('time_s must be a finite scalar in seconds')
        needed = self._body_closure(bodies)
        return {
            name: self._evaluate_source(binding, np.array([time_s]))[0]
            for name, binding in self.sources.items()
            if binding.live
            and name not in disabled_sources
            and needed.intersection(binding.targets)
        }

    def relative_motion_from_poses(self, *, time_s, targets, total_poses,
                                   time_series=None, disabled_sources=()):
        """Convert total world poses at named body origins into mount motion.

        total_poses has shape (times, targets, 4, 4) and includes static
        placement. Parent motion is evaluated at the matching physical times
        with the supplied record and mask. Bind the returned motion to the
        same targets. Subsequent changes to parent motion propagate normally.
        World input translations have binary64 precision.
        """
        from .motion import angles_from_matrix, checked_frames

        times = checked_times(time_s)
        names = (targets,) if isinstance(targets, str) else tuple(targets)
        self._body_closure(names)
        bodies = [self[name] for name in names]
        parents = tuple(dict.fromkeys(body.parent for body in bodies if body.parent is not None))
        mask = self._source_mask(disabled_sources)
        needed = self._body_closure(parents)
        if any(binding.live and binding.name not in mask and needed.intersection(binding.targets)
               for binding in self.sources.values()):
            raise ValueError('Converting a pose history requires prescribed parent motion')
        matrices = np.asarray(total_poses)
        if not names or matrices.shape != (len(times), len(names), 4, 4):
            raise ValueError('Supply total poses with shape (times, targets, 4, 4)')
        matrices = checked_frames(matrices.reshape(-1, 4, 4)).reshape(matrices.shape)
        local_static = [static_local(_reference_values(body), body.installation.values())
                        for body in bodies]
        values = np.empty((len(times), len(names), 6))
        for row, time in enumerate(times):
            dynamic = self._get_body_deformations(
                time_s=time, time_series=time_series, disabled_sources=mask, bodies=parents,
            )
            for column, (body, local) in enumerate(zip(bodies, local_static)):
                parent = body.parent
                carried = local if parent is None else compose(
                    inherit(dynamic[parent], self[parent]._nominal, body._nominal), local,
                )
                basis, origin = body._nominal[:3, :3], body._nominal[:3, 3]
                position = origin + basis @ carried[0]
                carried_axes = basis @ (np.eye(3) + carried[1])
                local_axes = np.eye(3) + local[1]
                convert = local_axes @ carried_axes.T
                target = matrices[row, column]
                values[row, column, :3] = convert @ (target[:3, 3] - position)
                values[row, column, 3:] = angles_from_matrix(convert @ target[:3, :3] @ local_axes.T)
        return RigidMotion(times, values, names)

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
        if not np.isscalar(time_s) or np.iscomplexobj(time_s) or not np.isfinite(time_s):
            raise ValueError('time_s must be a finite scalar in seconds')
        disabled_sources = self._source_mask(disabled_sources)
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
                    values = sample_at(source.time_s, source.values, time_s, selection=indices)
                else:
                    # Arbitrary stochastic sources may depend on the complete
                    # receiver set. Evaluate jointly, then select, preserving
                    # their seed/correlation contract.
                    values = self._evaluate_source(binding, np.array([time_s]))[0][
                        indices
                    ]
            else:
                values = time_series._sample(binding, time_s, selection=indices)
            for target, value in zip(targets, values):
                _check_transverse_values(self[target], value)
                shifts[target] += value[:3]
                if np.any(value[3:]):
                    rotations[target] = compose_rotation(
                        rotation_increment(*value[3:]), rotations[target]
                    )
        world, local_deformations = {}, {}
        for name, body in selected.items():
            local = _reference_values(body)
            install = (
                body.installation.values() if include_installation else np.zeros(6)
            )
            _check_transverse_values(body, install)
            delta = static_local(local, install)
            delta = (
                delta[0] + shifts[name],
                compose_rotation(rotations[name], delta[1]),
            )
            if body._transverse_only:
                local_deformations[name] = delta
            if body.parent is None:
                world[name] = delta
            else:
                parent = self.bodies[body.parent]
                world[name] = compose(
                    inherit(world[parent.name], parent._nominal, body._nominal), delta
                )
        return _BodyDeformations(
            {name: world[name] for name in requested},
            {name: local_deformations[name] for name in requested
             if name in local_deformations},
        )

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
        disabled_sources=(),
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
            disabled_sources=disabled_sources,
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
        disabled_sources=(),
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
            disabled_sources=disabled_sources,
        )

    def apply(self, *, time_s=0.0, time_series=None, disabled_sources=()):
        """Apply one physical time to ALL attached lines in this environment.

        Explicit host-side numeric alignment lowering. Does not start a clock,
        register a track callback, or update sources once per particle/turn.
        Unsupported bindings/conflicting edits are rejected before any writes.
        """
        disabled_sources = self._source_mask(disabled_sources)
        self._check_aliases()
        views = [view for view in self._line_views.values()
                 if view._attachments or view._pipes is not None]
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
            for field, value in zip(_attachment_fields(attachment), values):
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
            for attachment in view._numeric_attachments():
                element = view.line.element_dict[attachment['element']]
                updates.append((attachment, element, attachment['baseline']))
        for attachment, element, values in updates:
            for field, value in zip(_attachment_fields(attachment), values):
                setattr(element, field, value)
            attachment['last_written'] = values.copy()
        for view in self._line_views.values():
            if view._applied_source_recorder is not None:
                view._applied_source_recorder.stop()
            view._last_applied_time_s = None
            view._last_applied_passage_speed = None
            view._last_applied_source_mask = ()
            view._last_applied_time_series = None
            view._last_applied_controls = frozenset()
            view._last_control_values.clear()

    def _check_aliases(self):
        # Only attached objects consume numeric placements. Still scan every
        # physical line: an alias can be introduced under a different name or
        # on an otherwise unattached line after these bindings were prepared.
        needed = []
        for view in self._line_views.values():
            elements = view.line.element_dict
            needed.extend((name, elements[name]) for name in view._attachments)
        if not needed:
            return
        counts = self._alias_counts(element for _, element in needed)
        for name, element in needed:
            if counts.get(id(element)) != 1:
                raise ValueError(
                    f'{name}: shared/repeated element objects require an occurrence-buffer lowering'
                )

    def _alias_counts(self, targets):
        """Count active occurrences that consume the target's numeric fields.

        A native slice can share parent storage without the parent's name
        appearing in any active sequence. Replicas can resolve through other
        replicas to either the target or one of its native slices.
        """
        targets = {id(element): element for element in targets}
        counts = dict.fromkeys(targets, 0)
        by_buffer = {}
        for identity, element in targets.items():
            storage = _storage_identity(element)
            if storage is not None:
                buffer_id, offset = storage
                offsets = by_buffer.setdefault(buffer_id, {})
                if offset in offsets:
                    raise ValueError(
                        'Numeric mechanics cannot own shared/repeated element storage'
                    )
                offsets[offset] = identity

        def match(element):
            identity = id(element)
            if identity in targets:
                return identity
            xobject = getattr(element, '_xobject', None)
            if xobject is not None:
                offsets = by_buffer.get(id(xobject._buffer))
                if offsets is not None:
                    return offsets.get(xobject._offset)
            return None

        for line in self.env._lines_weakrefs:
            # Collective survey/Twiss creates shallow noncollective views in
            # distinct environments sharing this weak set. They are execution
            # views of the same occurrences, not additional physical lines.
            if line.env is not self.env:
                continue
            elements = line.element_dict
            for name in line.element_names:
                element = elements[name]
                if isinstance(element, Replica):
                    # An invalid replica on another, unbound line has no
                    # resolvable target to which our numeric writes could alias.
                    try:
                        element = element.resolve(elements)
                    except (KeyError, RecursionError):
                        continue
                if isinstance(element, _SliceBase):
                    # Before building a tracker the storage pointer may be
                    # unset. Also check an existing cached parent in case the
                    # named target was subsequently changed.
                    matched = {
                        match(element),
                        match(elements.get(getattr(element, 'parent_name', None))),
                        match(element._parent),
                    }
                    matched.discard(None)
                    for identity in matched:
                        counts[identity] += 1
                else:
                    identity = match(element)
                    if identity is not None:
                        counts[identity] += 1
        return counts

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
            for field, value in zip(_attachment_fields(attachment), values):
                setattr(element, field, value)
        for view in views:
            view._clear_bindings()
        self._sources.clear()
        self._bodies.clear()
        self._revision += 1
        return self

    def get_table(self, *, time_s=None, time_series=None, disabled_sources=()):
        """Body inputs, or composed world poses at an explicit physical time.

        No argument returns configuration without evaluating sources. Supply
        time_s for a read-only pose table, equivalent to survey. Rows retain
        normal Xsuite selection and can query a selected body's time history.
        """
        disabled_sources = self._source_mask(disabled_sources)
        if time_s is not None:
            return self.survey(time_s=time_s, time_series=time_series,
                               disabled_sources=disabled_sources)
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
        return mechanics_table(data, owner=self, disabled_sources=disabled_sources)

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
            if view._attachments or view._pipes is not None:
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
            installation = body.installation.values()
            _check_transverse_values(body, installation)
            bodies.append(
                dict(
                    name=body.name,
                    kind=body.kind,
                    support_class=body.support_class,
                    transverse_only=body._transverse_only,
                    parent=body.parent,
                    anchor=body._nominal.tolist(),
                    alignment=alignment,
                    installation=installation.tolist(),
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
                    solenoid_axes=[
                        dict(element=a['element'], body=a['body'],
                             baseline=a['baseline'].tolist())
                        for a in view._solenoid_axes.values()
                    ],
                    pipes=([] if view._pipes is None else [
                        dict(name=name, body=body)
                        for name, body in view._pipes.owners.items()
                    ]),
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
        candidate = MechanicsManager(self.env)
        aperture_lines = [self.env.lines[item['line']] for item in data['lines']
                          if item.get('pipes')]
        if aperture_lines:
            aperture_env = self.env.aperture
            was_built = aperture_env.is_built
            unbuilt = [line.aperture for line in aperture_lines if not line.aperture.is_built]
        try:
            candidate._load_definitions(data)
        except Exception:
            if aperture_lines:
                if not was_built:
                    aperture_env.discard_model()
                else:
                    for aperture in unbuilt:
                        aperture.discard_model()
            raise
        self._bodies.update(candidate._bodies)
        self._sources.update(candidate._sources)
        for line, loaded in candidate._line_views.items():
            view = self.for_line(line)
            view._reference = loaded._reference
            view._reference_is_explicit = loaded._reference_is_explicit
            view._reference_inputs = loaded._reference_inputs
            view._attachments = loaded._attachments
            view._solenoid_axes = loaded._solenoid_axes
            view._element_names = loaded._element_names
            view._line_geometry = loaded._line_geometry
            view._bound_line = loaded._bound_line
            view._pipes = loaded._pipes
            if view._pipes is not None:
                view._pipes.view = view
        self._revision += 1
        return self

    def _load_definitions(self, data):
        for item in data['bodies']:
            b = self._new_body(
                item['name'],
                elements=None,
                line=None,
                parent=item['parent'],
                anchor=Frame(item['anchor']),
                feet_local_m=item.get('feet_local_m'),
                kind=item.get('kind', 'body'),
                support_class=item.get('support_class', ''),
                transverse_only=item.get('transverse_only', False),
            )
            b.shift_x, b.shift_y, b.shift_s, b.rot_x_rad, b.rot_y_rad, b.rot_s_rad = (
                item['alignment']
            )
            b.installation = RigidOffset(*item.get('installation', [0.0] * 6))
            _reference_values(b)
            _check_transverse_values(b, b.installation.values())
        for item in data['sources']:
            self.add_source(
                item['name'],
                decode_source(item['source']),
                targets=item['targets'],
                component=item['component'],
                reference=item.get('reference', 'relative'),
                role=item.get('role', 'disturbance'),
                live=item.get('live', False),
            )
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
                self[stored['body']]
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
            axes = item.get('solenoid_axes', ())
            by_body = {}
            for axis in axes:
                by_body.setdefault(axis['body'], []).append(axis['element'])
            for body, names in by_body.items():
                view.attach_solenoid_axis(body, elements=names)
            for axis in axes:
                if np.iscomplexobj(axis['baseline']):
                    raise ValueError('Stored solenoid-axis baseline must be real')
                baseline = np.array(axis['baseline'], dtype=float)
                if baseline.shape != (2,) or not np.all(np.isfinite(baseline)):
                    raise ValueError('Stored solenoid-axis baseline must contain finite x0/y0')
                view._solenoid_axes[axis['element']]['baseline'] = baseline
            view._reference_is_explicit = item.get('reference_is_explicit', True)
            view._reference_inputs = {
                name: np.array(values, dtype=float)
                for name, values in item.get('adopted_reference', {}).items()
            }
            if item.get('pipes'):
                view.attach_pipes({pipe['name']: pipe['body']
                                   for pipe in item['pipes']})
        self._check_aliases()


class LineMechanics:
    """Line-specific attachments/view of env.mechanics; not an independent manager."""

    def __init__(self, manager, line):
        self.manager = manager
        self._line_ref = ref(line)
        self._bound_line = None
        self._reference = Frame()
        self._reference_is_explicit = False
        self._attachments = {}
        self._solenoid_axes = {}
        self._pipes = None
        self._element_names = None
        self._line_geometry = None
        self._vibrations_enabled = False
        self._controls_enabled = False
        self._source_mode_configured = False
        self._tracking_source_mask = ()
        self._last_applied_source_mask = ()
        self._last_applied_time_s = None
        self._last_applied_passage_speed = None
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
            or self._pipes is not None
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
        bodies = {attachment['body'] for attachment in self._attachments.values()}
        bodies.update(axis['body'] for axis in self._solenoid_axes.values())
        if self._pipes is not None:
            bodies.update(self._pipes.bodies)
        return bodies

    def _numeric_attachments(self):
        yield from self._attachments.values()
        yield from self._solenoid_axes.values()

    def _protected_geometry_objects(self):
        """Active reference objects, including hidden named slice parents.

        Resolve only Python ownership metadata, not scalar geometry values.
        Integrations can use this to reject raw writes that would invalidate a
        captured survey, including writes to parents absent from the sequence.
        """
        elements = self.line.element_dict
        objects = {name: elements[name] for name in self.line.element_names}
        for element in tuple(objects.values()):
            if isinstance(element, _SliceBase):
                parent = _resolved_slice_parent(element, elements)
                objects[element.parent_name] = parent
        return objects

    def _occurrence_bodies(self, name):
        bodies = [self._attachments[name]['body']]
        if name in self._solenoid_axes:
            bodies.append(self._solenoid_axes[name]['body'])
        return bodies

    def _source_enabled(self, binding):
        if binding.name in self._tracking_source_mask:
            return False
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
            self._tracking_source_mask,
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
        self, time_s, source_values, *, disabled_sources=None, time_series=None,
        passage_speed=None,
    ):
        self._last_applied_time_s = float(time_s)
        self._last_applied_passage_speed = passage_speed
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
        self._last_applied_source_mask = tuple(disabled_sources)
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
                values[name] = self._last_applied_time_series._sample(binding, time_s)
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
        self, *, sources, max_samples=4096, max_memory_bytes=8 * 1024**2,
        case_scope='affected',
    ):
        """Record a bounded live-input trace; use case_scope='line' for full replay."""
        from .applied_history import AppliedSourceRecorder

        recorder = AppliedSourceRecorder(
            self,
            sources=sources,
            max_samples=max_samples,
            max_memory_bytes=max_memory_bytes,
            case_scope=case_scope,
        )
        self._applied_source_recorder = recorder
        return recorder

    def enable_vibrations(
        self, *, time_series=None, time_origin_s=None, sampling='turn', backend='auto',
        disabled_sources=None,
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
        disabled_sources selects a source mask by name. Omit it to keep the
        current tracking selection, or pass () to clear the explicit mask.
        """
        self._enable_updates(
            time_series=time_series,
            time_origin_s=time_origin_s,
            sampling=sampling,
            backend=backend,
            disturbances=True,
            disabled_sources=disabled_sources,
        )

    def enable_controls(
        self, *, time_series=None, time_origin_s=None, sampling='turn', backend='auto',
        disabled_sources=None,
    ):
        """Enable applied control motion without implicitly enabling disturbances."""
        self._enable_updates(
            time_series=time_series,
            time_origin_s=time_origin_s,
            sampling=sampling,
            backend=backend,
            disturbances=self._vibrations_enabled,
            disabled_sources=disabled_sources,
        )

    def _enable_updates(
        self, *, time_series, time_origin_s, sampling, backend, disturbances,
        disabled_sources,
    ):
        attached = [v for v in self.manager._line_views.values() if v._attachments]
        if not self._attachments or (len(attached) != 1 and time_origin_s is None):
            raise NotImplementedError(
                'Automatic multi-line sampling needs an explicit common time_origin_s per line'
            )
        if time_origin_s is not None and (
            not np.isscalar(time_origin_s) or np.iscomplexobj(time_origin_s)
            or not np.isfinite(time_origin_s)
        ):
            raise ValueError('time_origin_s must be finite physical seconds')
        if sampling not in ('turn', 'passage'):
            raise ValueError('Use turn or passage sampling')
        if backend not in ('auto', 'compiled', 'reference'):
            raise ValueError('Use auto, compiled or reference backend')
        if backend == 'compiled' and sampling != 'turn':
            raise ValueError('Compiled mechanics currently freezes at turn boundaries')
        mask = (self._tracking_source_mask if disabled_sources is None
                else self.manager._source_mask(disabled_sources))
        needed = self.manager._body_closure(self._consumed_bodies())
        bindings = [
            binding
            for binding in self.manager.sources.values()
            if needed.intersection(binding.targets)
            and binding.name not in mask
            and (disturbances or binding.role == 'control')
        ]
        self._check_automatic_pipes(bindings)
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
        self._tracking_source_mask = mask
        self._runtime = None
        self._vibrations_enabled = disturbances
        self._controls_enabled = True
        self._source_mode_configured = True

    def _check_automatic_pipes(self, bindings):
        if self._pipes is None or not self._pipes.bodies:
            return
        ancestors = self.manager._body_closure(self._pipes.bodies)
        if any(ancestors.intersection(binding.targets) for binding in bindings):
            raise NotImplementedError(
                'Automatic moving-pipe aperture updates are not implemented; '
                'use get_aperture(time_s=...) for frozen-time geometry and '
                'generate tracking limits from that view'
            )

    def disable_vibrations(self):
        """Remove disturbances, retaining applied control and its automatic updates.

        Source phases/records are not reset. A later enable resumes at the
        particles' physical time, rather than restarting a random waveform.
        """
        disturbances = tuple(
            binding.name for binding in self.manager.sources.values()
            if binding.role == 'disturbance'
        )
        disabled = self.manager._source_mask(
            (*self._tracking_source_mask, *self._last_applied_source_mask, *disturbances)
        )
        time = self._last_applied_time_s
        speed = self._last_applied_passage_speed
        sampling = self._sampling if time is None else ('turn' if speed is None else 'passage')
        held = self._held_control_values() if sampling == 'turn' else {}
        self.apply(
            time_s=0.0 if time is None else time,
            time_series=self._time_series if time is None else self._last_applied_time_series,
            disabled_sources=disabled,
            sampling=sampling,
            _source_values=held,
            _passage_speed=speed,
        )
        self._vibrations_enabled = False
        needed = self.manager._body_closure(self._consumed_bodies())
        self._controls_enabled = self._controls_enabled and any(
            binding.role == 'control' and binding.name not in self._tracking_source_mask
            and needed.intersection(binding.targets)
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
        self._check_automatic_pipes(
            binding for binding in self.manager.sources.values()
            if self._source_enabled(binding)
        )
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

    def _check_tracking_window(self, particles, *, num_turns, prepared):
        """Validate known finite replay domains before advancing any particles."""
        from scipy.constants import c
        from .runtime import has_fixed_reference
        from .sources import LocatedSource, RegionalGroundMotion, VectorGroundMotion
        from .sources import source_time_bounds

        if num_turns is None:
            num_turns = 1
        if not isinstance(num_turns, (int, np.integer)) or num_turns < 1:
            return
        tracker = self.line.tracker
        if tracker.iscollective or self.line.enable_time_dependent_vars:
            return
        record = self._time_series
        bounded = {}
        for name, binding in self.manager.sources.items():
            if not self._source_enabled(binding):
                continue
            if record is not None and not binding.live and name in record.values:
                bounds = (record.time_s[0], record.time_s[-1])
            else:
                positions = ()
                if type(binding.source) in (LocatedSource, RegionalGroundMotion, VectorGroundMotion):
                    positions = [self.manager[target]._nominal[:3, 3] for target in binding.targets]
                bounds = source_time_bounds(binding.source, positions)
            if bounds is not None:
                bounded[name] = bounds
        if not bounded:
            return
        needed = self.manager._body_closure(self._consumed_bodies())
        bounded = {name: bounds for name, bounds in bounded.items()
                   if needed.intersection(self.manager.sources[name].targets)}
        if not bounded:
            return
        fixed = self._runtime.fixed_reference if self._runtime is not None else has_fixed_reference(tracker)
        if not fixed:
            return

        first = float(particles.time_s)
        beta = float(particles._xobject.beta0[0])
        if not np.isfinite(beta) or beta <= 0:
            return
        step = tracker._tracker_data_base.line_length / (beta * c)
        if prepared:
            clock = self._runtime._clock_for(particles, first, particles.reference_turns, step)
        else:
            clock = first
        # Match each scheduler's arithmetic exactly at record boundaries.
        for _ in range(num_turns - 1):
            clock += step
        first += self._time_origin_s
        last = float(clock) + self._time_origin_s

        passage = self._sampling == 'passage'
        if passage:
            if self.line.particle_ref is None:
                raise ValueError('Set a reference particle for passage-time sampling')
            speed = float(self.line.particle_ref.beta0[0]) * c
            table = self.line.get_table()
            arrivals = [
                (self.manager._body_closure(self._occurrence_bodies(name)),
                 float(table['s', name]) / speed)
                for name in self._attachments
            ]
        for name, (start, stop) in bounded.items():
            lo, hi = first, last
            if passage:
                targets = self.manager.sources[name].targets
                delays = [delay for closure, delay in arrivals if closure.intersection(targets)]
                if not delays:
                    continue
                lo += min(delays)
                hi += max(delays)
            if lo < start or hi > stop:
                raise ValueError(
                    f'{name}: requested mechanics freeze times [{lo:.17g}, {hi:.17g}] s '
                    f'lie outside the motion record [{start:.17g}, {stop:.17g}] s'
                )

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
            self._check_automatic_pipes(
                binding for binding in self.manager.sources.values()
                if self._source_enabled(binding)
            )
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
        _passage_speed=None,
    ):
        """Update this line only at a physical time; other beam buffers stay put.

        Passage mode uses reference-speed arrival times, not particle-specific
        zeta/delta arrival. It prepares frozen per-occurrence maps for one turn.
        This Python reference is intentionally measured separately from kernels.
        """
        disabled_sources = self.manager._source_mask(disabled_sources)
        self.manager._check_aliases()
        self._check()
        if self._applied_source_recorder is not None:
            if self._applied_source_recorder.active and sampling != 'turn':
                raise ValueError('Applied source recording supports turn freezes only')
            self._applied_source_recorder.check_case()
        consumed = self._consumed_bodies()
        live_values = dict(_source_values or {})
        speed = None
        if sampling == 'turn':
            live_values.update(
                self.manager._sample_live_values(
                    time_s, bodies=consumed,
                    disabled_sources=set(disabled_sources) | set(live_values),
                )
            )
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

            if self.line.particle_ref is None and _passage_speed is None:
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
            speed = (float(self.line.particle_ref.beta0[0]) * c
                     if _passage_speed is None else _passage_speed)
            if not np.isfinite(speed) or speed <= 0:
                raise ValueError('Passage-time sampling requires a positive reference speed')
            table = self.line.get_table()
            updates = []
            for name in self._attachments:
                poses = self.manager.get_body_poses(
                    time_s=time_s + float(table['s', name]) / speed,
                    time_series=time_series,
                    disabled_sources=disabled_sources,
                    bodies=self._occurrence_bodies(name),
                )
                updates.extend(self._updates(poses, names=[name], check=False))
        else:
            raise ValueError('Use turn or passage sampling')
        self._commit_applied(
            updates, time_s, live_values, disabled_sources, time_series, passage_speed=speed,
        )

    def _commit_applied(self, updates, time_s, live_values, disabled_sources, time_series,
                        *, passage_speed=None):
        self._preflight_applied(
            time_s,
            live_values,
            disabled_sources=disabled_sources,
            time_series=time_series,
        )
        for attachment, element, values in updates:
            for field, value in zip(_attachment_fields(attachment), values):
                setattr(element, field, value)
            attachment['last_written'] = values.copy()
        self._remember_applied(
            time_s,
            live_values,
            disabled_sources=disabled_sources,
            time_series=time_series,
            passage_speed=passage_speed,
        )

    def freeze_at(
        self, *, time_s, time_series=None, disabled_sources=(), applied_history=None,
        aperture_at_s=(),
    ):
        """Apply one matched magnet and pipe state to this line at physical time [s].

        Resolve a single body state and build the independent aperture view and
        requested discrete limits before writing any magnet fields. Each limit
        uses the fixed survey plane at its s position. Insert the limits into a
        separate frozen line for loss tracking or use the returned aperture for
        inspection. The original line.aperture remains the adopted geometry.

        Stop automatic updates before applying. By default, enabled live inputs
        are sampled once. For replay, supply an applied history that covers the
        line and all relevant live inputs; it fixes the source mask, prescribed
        record, and previous-hold freeze time.
        """
        from .applied_history import AppliedSourceHistory
        from .tables import _static_inputs

        if self.automatic_updates_enabled:
            raise ValueError('Freeze automatic updates before applying a matched state')
        if not np.isscalar(time_s) or np.iscomplexobj(time_s) or not np.isfinite(time_s):
            raise ValueError('time_s must be a finite real scalar in seconds')
        self.manager._check_aliases()
        self._check()
        if self._pipes is None and np.asarray(aperture_at_s).size:
            raise ValueError('Declare installed pipe ownership before requesting limits')
        if self._applied_source_recorder is not None:
            self._applied_source_recorder.check_case()
        disabled_sources = self.manager._source_mask(disabled_sources)
        consumed = self._consumed_bodies()
        needed = self.manager._body_closure(consumed)
        pose_time_s = float(time_s)
        if applied_history is None:
            live_values = self.manager._sample_live_values(
                time_s, bodies=consumed, disabled_sources=disabled_sources
            )
        else:
            if not isinstance(applied_history, AppliedSourceHistory):
                raise TypeError('Supply an immutable applied source history snapshot')
            pose_time_s, time_series, disabled_sources, live_values = applied_history._resolve(
                self.manager, consumed, time_s, time_series, disabled_sources
            )
        sources = {
            name: binding for name, binding in self.manager.sources.items()
            if name not in disabled_sources and needed.intersection(binding.targets)
        }
        required = {name for name, binding in sources.items() if binding.live}
        if required.difference(live_values):
            raise ValueError('Applied history must cover all enabled live inputs for the line')
        live_values = {name: frozen_array(live_values[name]) for name in sources if name in required}
        deformations = self.manager._get_body_deformations(
            time_s=pose_time_s, time_series=time_series, disabled_sources=disabled_sources,
            bodies=needed, _source_values=live_values,
        )
        static_case = {name: _static_inputs(self.manager[name])
                       for name in self.manager.bodies if name in needed}
        design_alignment = {
            name: dict(zip((*_FIELDS, 'rot_s_rad'),
                           (*a['baseline'], a['geometry'][-1])))
            for name, a in self._attachments.items()
        }
        for name, axis in self._solenoid_axes.items():
            design_alignment[name].update(zip(_SOLENOID_AXIS_FIELDS, axis['baseline']))
        design_alignment = {name: MappingProxyType(values)
                            for name, values in design_alignment.items()}
        poses = _BodyPoses(self.manager, deformations)
        element_poses = self._element_poses_from_bodies(poses)
        aperture, pipe_table, limits = None, None, {}
        query_kind = 'fresh_freeze' if applied_history is None else 'applied_history'
        if self._pipes is not None:
            aperture = self._pipes.build_aperture(time_s, pose_time_s, deformations, query_kind)
            limits = aperture.get_limit_elements_at_s(aperture_at_s)
            pipe_table = self._pipes.get_table(
                time_s=time_s, _resolved=(pose_time_s, deformations, query_kind),
            )
        updates = self._updates(poses)
        self._commit_applied(updates, pose_time_s, live_values, disabled_sources, time_series)
        return FrozenMechanicalState(
            time_s=float(time_s), pose_time_s=pose_time_s, query_kind=query_kind,
            element_poses=MappingProxyType(element_poses), aperture=aperture, pipe_table=pipe_table,
            limits=MappingProxyType(limits), design_alignment=MappingProxyType(design_alignment),
            static_case=MappingProxyType(static_case),
            source_names=tuple(sources), disabled_sources=disabled_sources,
            prescribed_record=time_series, live_values=MappingProxyType(live_values),
        )

    @property
    def line(self):
        return self._line_ref()

    @property
    def reference_frame(self):
        return Frame(self._reference.matrix)

    def set_reference_frame(self, frame):
        """Place this line's default survey in the environment world, before attachment."""
        if self._attachments or self._pipes is not None:
            raise ValueError('Set the survey placement before attaching mechanics')
        self._reference = _frame(frame)
        self._reference_is_explicit = True
        self.manager._revision += 1

    def attach_pipe(self, name, *, body):
        """Declare ownership of an installed pipe placement, not its shared shape.

        body=None keeps the adopted installed placement fixed. Otherwise it
        follows a body in this environment, including installation and dynamic
        motion. Declare every placement, including fixed ones. The original
        line.aperture is retained as the reference geometry; get_aperture
        builds an independent frozen-time view with fresh intersections/bounds.
        Generated limit elements are snapshots, not automatically moving checks.
        """
        return self.attach_pipes({name: body})

    def attach_pipes(self, assignments):
        """Declare a batch of installed placement owners with one geometry capture.

        assignments maps installed placement names to bodies (or None for
        fixed pipes). Validate the whole batch before changing ownership.
        """
        from .pipes import PipeBindings

        if not hasattr(assignments, 'items') or not assignments:
            raise ValueError('Use a nonempty mapping from pipe placements to owners')
        self._check(allow_built_aperture=True)
        if not self._reference_is_explicit and any(
            other is not self and (other._attachments or other._pipes is not None)
            for other in self.manager._line_views.values()
        ):
            raise ValueError(
                'Set an explicit survey reference frame before attaching a second line'
            )
        owners = {}
        for name, body in assignments.items():
            if (not isinstance(name, str)
                    or name not in self.line.aperture.pipe_positions.keys()):
                raise KeyError(f'Unknown installed pipe placement {name!r}')
            if body is not None and not isinstance(body, (str, RigidBody)):
                raise ValueError('Pipe owner must be a body in this environment')
            name_of_body = body.name if isinstance(body, RigidBody) else body
            if body is not None and (
                name_of_body not in self.manager.bodies
                or (isinstance(body, RigidBody)
                    and self.manager[name_of_body] is not body)
            ):
                raise ValueError('Pipe owner must be a body in this environment')
            if name_of_body is not None and self.automatic_updates_enabled:
                ancestors = self.manager._body_closure([name_of_body])
                if any(self._source_enabled(binding)
                       and ancestors.intersection(binding.targets)
                       for binding in self.manager.sources.values()):
                    raise ValueError('Freeze automatic updates before binding a moving pipe')
            if (name_of_body is not None
                    and len(self.line.aperture.pipe_positions[name].pipe) < 2):
                raise ValueError('A body-owned finite pipe requires at least two profiles')
            owners[name] = name_of_body
        bindings = self._pipes or PipeBindings(self)
        bindings.attach(owners)
        self._pipes = bindings
        if self._element_names is None:
            self._element_names = tuple(self.line.element_names)
            self._line_geometry = self._geometry_signature()
        self._bound_line = self.line
        self.manager._revision += 1
        return self

    def get_pipe_table(self, *, time_s=None, time_series=None, disabled_sources=(), applied_history=None):
        """Installed ownership, or world poses at an explicit time.

        change_X/Y/Z includes owner reference adjustment, installation and
        dynamics relative to the adopted pipe placement. pose is a world Frame
        matrix. Nothing is written to the line or its reference aperture model.
        """
        if self._pipes is None:
            raise ValueError('Declare installed pipe ownership with attach_pipe first')
        self._check()
        return self._pipes.get_table(
            time_s=time_s, time_series=time_series,
            disabled_sources=self.manager._source_mask(disabled_sources),
            applied_history=applied_history,
        )

    def get_aperture(self, *, time_s, time_series=None, disabled_sources=(), applied_history=None):
        """Build a frozen-time aperture view without changing the line.

        Placements, derived geometry and copies of the adopted pipe/profile
        definitions are independent of the live environment. Fixed survey-plane
        cuts are recomputed, including pitch/yaw effects. Sample limits from
        this view for a matching frozen tracking study. The ordinary line
        aperture remains the adopted reference. This is a host preparation
        operation, not an automatic per-turn aperture updater.
        An applied history selects the latest recorded freeze at or before time_s.
        """
        if self._pipes is None:
            raise ValueError('Declare installed pipe ownership with attach_pipe first')
        self._check()
        return self._pipes.get_aperture(
            time_s=time_s, time_series=time_series,
            disabled_sources=self.manager._source_mask(disabled_sources),
            applied_history=applied_history,
        )

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
            for candidate in selected:
                candidate['body'] = body.name
        self._install_candidates(candidates)
        for body, _ in prepared:
            self.manager._bodies[body.name] = body
        self.manager._revision += 1
        return {body.name: body for body, _ in prepared}

    def attach(self, body, *, elements):
        """Attach this line's elements to an existing physical body, possibly shared."""
        name = body.name if isinstance(body, RigidBody) else body
        if name not in self.manager.bodies or (
            isinstance(body, RigidBody) and self.manager[name] is not body
        ):
            raise ValueError('Body must belong to this environment')
        self.attach_bodies({name: elements})

    def attach_bodies(self, assignments):
        """Attach existing physical bodies to this line with one full survey.

        ``assignments`` maps body names already defined in the environment to
        nonempty sets of element occurrences in this line. This is useful for
        a second, reversed traversal of a shared physical support inventory.
        Use an explicit world survey reference for the second line. Validate
        every occurrence and alias before installing any new attachment.
        """
        if not hasattr(assignments, 'items') or not assignments:
            raise ValueError('Use a nonempty mapping from existing body names to elements')
        owner = {}
        all_names = []
        for name, elements in assignments.items():
            if name not in self.manager.bodies:
                raise ValueError(f'{name}: body must belong to this environment')
            names = (elements,) if isinstance(elements, str) else tuple(elements)
            if not names:
                raise ValueError('Select nonempty unique element names')
            all_names.extend(names)
            owner.update((element, name) for element in names)
        candidates = self._candidates(all_names)
        for candidate in candidates:
            candidate['body'] = owner[candidate['element']]
        self._install_candidates(candidates)
        self.manager._revision += 1

    def new_solenoid_axis(self, name, *, elements, anchor='center'):
        """Mount one independent longitudinal field axis inside a rigid chart.

        Select previously attached VariableSolenoid elements sharing one chart
        body. The solenoid axis is a transverse-only child of that body. It
        retains each segment's adopted local x0/y0 and adds relative x/y
        displacement expressed in the local field chart. The parent carries the
        complete chart, including all multipole fields, in six degrees of
        freedom. No multipole strength or contribution is inferred or changed.

        Relative pitch/yaw/roll or longitudinal motion of the solenoid profile
        is not represented by x0/y0. Use the parent for shared rigid motion.
        The child's frame has the first selected element's adopted axes by
        default; all selected field chart axes must be parallel or antiparallel
        to its longitudinal axis.
        """
        names, parent = self._solenoid_chart(elements)
        if isinstance(anchor, str):
            if anchor != 'center':
                raise ValueError("Use anchor='center' or an explicit Frame/XYZ")
            first, last = (self._attachments[n] for n in (names[0], names[-1]))
            frame = Frame(first['base_element'])
            endpoints = []
            for attachment, key in ((first, 'base_element'),
                                    (last, 'base_element_end')):
                element = self.line.element_dict[attachment['element']]
                matrix = attachment[key]
                endpoints.append(
                    matrix[:3, 3] + matrix[:3, :2] @ _values(element, _SOLENOID_AXIS_FIELDS)
                )
            frame.matrix[:3, 3] = .5 * (endpoints[0] + endpoints[1])
            anchor = frame
        body = self.manager._body_definition(
            name, anchor=anchor, parent=parent, feet_local_m=None,
            kind='solenoid_axis', transverse_only=True,
        )
        candidates = self._solenoid_axis_candidates(body, names, parent)
        self.manager._bodies[name] = body
        self._solenoid_axes.update((a['element'], a) for a in candidates)
        self.manager._revision += 1
        return body

    def attach_solenoid_axis(self, body, *, elements):
        """Attach other chart segments to an existing transverse axis body."""
        name = body.name if isinstance(body, RigidBody) else body
        if name not in self.manager.bodies or (
            isinstance(body, RigidBody) and self.manager[name] is not body
        ):
            raise ValueError('Body must belong to this environment')
        names, parent = self._solenoid_chart(elements)
        candidates = self._solenoid_axis_candidates(self.manager[name], names, parent)
        self._solenoid_axes.update((a['element'], a) for a in candidates)
        self.manager._revision += 1

    def _solenoid_chart(self, elements):
        from ..beam_elements.variable_solenoid import VariableSolenoid

        self.manager._check_aliases()
        self._check(allow_built_aperture=True)
        names = (elements,) if isinstance(elements, str) else tuple(elements)
        if not names or len(set(names)) != len(names):
            raise ValueError('Select nonempty unique VariableSolenoid segments')
        for name in names:
            if name not in self._attachments or name in self._solenoid_axes:
                raise ValueError(
                    f'{name}: select an attached chart without an existing solenoid axis'
                )
            if not isinstance(self.line.element_dict[name], VariableSolenoid):
                raise ValueError(f'{name}: solenoid axis requires VariableSolenoid')
        parents = {self._attachments[name]['body'] for name in names}
        if len(parents) != 1:
            raise ValueError('Select segments sharing one rigid field chart body')
        selection = set(names)
        names = [name for name in self.line.element_names if name in selection]
        return names, parents.pop()

    def _solenoid_axis_candidates(self, body, names, parent):
        if not body._transverse_only or body.parent != parent:
            raise ValueError('A solenoid axis must be a transverse child of the chart body')
        candidates, refs = [], set()
        for name in names:
            chart = self._attachments[name]
            axes = chart['base_element'][:3, :3].T @ body._nominal[:3, :3]
            if not np.allclose(axes[2, :2], 0., atol=1e-12, rtol=0.):
                raise ValueError(f'{name}: select field charts with parallel solenoid axes')
            element = self.line.element_dict[name]
            baseline = _values(element, _SOLENOID_AXIS_FIELDS)
            if not np.all(np.isfinite(baseline)):
                raise ValueError(f'{name}: adopted x0/y0 must be finite')
            for field in _SOLENOID_AXIS_FIELDS:
                reference = getattr(self.line.env.ref[name], field)
                if getattr(reference, '_expr', None) is not None:
                    raise ValueError(
                        f'{name}.{field}: expression-driven axis is not supported by numeric lowering'
                    )
                refs.add(reference)
            candidates.append(dict(
                element=name, body=body.name, chart_body=parent,
                projection=axes[:2, :2].copy(), baseline=baseline,
                last_written=baseline.copy(), object_id=id(element),
            ))
        _check_alignment_consumers(self.line.env.ref_manager, refs)
        return candidates

    def clear(self):
        """Restore this line's local alignment and detach it from mechanics.

        Shared body/source definitions are retained for other lines. Use
        env.mechanics.clear() to remove the whole mechanical configuration.
        """
        updates = self._removal_updates()
        if self._attachments and self.line.aperture.is_built:
            self.line.aperture.discard_model()
        for attachment, element, values in updates:
            for field, value in zip(_attachment_fields(attachment), values):
                setattr(element, field, value)
        self._clear_bindings()
        self.manager._revision += 1
        return self

    def _removal_updates(self):
        if self._runtime is not None:
            self._runtime.finish()
        updates = []
        alignment_refs = set()
        for attachment in self._numeric_attachments():
            name = attachment['element']
            element = self.line.element_dict.get(name)
            if element is None or id(element) != attachment['object_id']:
                raise ValueError(
                    f'{name}: attached object changed; original alignment cannot be restored automatically'
                )
            fields = _attachment_fields(attachment)
            if not np.array_equal(_values(element, fields), attachment['last_written']):
                raise ValueError(
                    f'{name}: effective alignment was edited; clear mechanics before editing local alignment'
                )
            for field in fields:
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
        if self._applied_source_recorder is not None:
            self._applied_source_recorder.stop()
        self._attachments.clear()
        self._solenoid_axes.clear()
        self._pipes = None
        self._element_names = None
        self._line_geometry = None
        self._vibrations_enabled = False
        self._controls_enabled = False
        self._source_mode_configured = False
        self._tracking_source_mask = ()
        self._last_applied_source_mask = ()
        self._last_applied_time_s = None
        self._last_applied_passage_speed = None
        self._last_applied_time_series = None
        self._last_applied_controls = frozenset()
        self._last_control_values.clear()
        self._time_series = None
        self._runtime = None
        self._reference = Frame()
        self._reference_is_explicit = False
        self._reference_inputs.clear()
        self._bound_line = None

    def _candidates(self, elements):
        if self._attachments or self._pipes is not None:
            self.manager._check_aliases()
            self._check()
        names = (elements,) if isinstance(elements, str) else tuple(elements)
        if not names or len(set(names)) != len(names):
            raise ValueError('Select nonempty unique element names')
        line = self.line
        if any(isinstance(element, Replica) for element in line._elements):
            raise ValueError(
                'Mechanics reference geometry does not support replicas on the bound line'
            )
        line_elements = line.element_dict
        for element in line._elements:
            if isinstance(element, _SliceBase):
                _resolved_slice_parent(element, line_elements)
            elif hasattr(element, 'weight') and hasattr(element, '_parent'):
                raise ValueError(
                    'Mechanics reference geometry does not support custom slices'
                )
        if (
            not self._attachments
            and not self._reference_is_explicit
            and any(
                v is not self and (v._attachments or v._pipes is not None)
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
        elements_by_id = self.manager._alias_counts(
            line.element_dict[name] for name in names if name in order
        )
        alignment_refs = set()
        for name in names:
            if counts.get(name, 0) != 1 or name in self._attachments:
                raise ValueError(
                    f'{name}: unknown, repeated or already attached occurrence'
                )
            element = line.element_dict[name]
            if self._pipes is not None and getattr(element, 'installed_pipe_name', None):
                raise ValueError(
                    'Pipe-derived limits are frozen sections; generate fresh '
                    'limits from get_aperture(time_s=...) instead of attaching them'
                )
            if name in self._reference_inputs and not np.array_equal(
                _values(element), self._reference_inputs[name]
            ):
                raise ValueError(
                    f'{name}: adopted reference input changed; adopt again before attaching'
                )
            if (
                not getattr(element, 'allow_rot_and_shift', False)
                or isinstance(element, (Replica, _SliceBase))
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
        elements = self.line.element_dict
        geometry = []
        for name in self.line.element_names:
            element = elements[name]
            if isinstance(element, Replica):
                raise ValueError('Line reference geometry changed: replica introduced')
            signature = _reference_geometry(element)
            if isinstance(element, _SliceBase):
                signature += (_slice_reference_geometry(element, elements),)
            geometry.append((id(element), signature))
        return tuple(geometry)

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
        if self._pipes is not None:
            self._pipes.validate(complete=not allow_built_aperture)
        if (
            self._attachments
            and self.line.aperture.is_built
            and not allow_built_aperture
            and self._pipes is None
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
        for a in self._numeric_attachments():
            name = a['element']
            element = self.line.element_dict[name]
            if id(element) != a['object_id']:
                raise ValueError(f'{name}: attached element object was replaced')
            if 'geometry' in a and _geometry(element) != a['geometry']:
                raise ValueError(f'{name}: nominal geometry changed after attachment')
            fields = _attachment_fields(a)
            if not np.array_equal(_values(element, fields), a['last_written']):
                raise ValueError(
                    f'{name}: local alignment edited after attachment; restore/rebind explicitly'
                )
            for field in fields:
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
            values = lower(a, body._nominal, poses._deformations[body.name])
            updates.append((a, element, values))
            if name in self._solenoid_axes:
                axis = self._solenoid_axes[name]
                updates.append((axis, element, self._axis_values(axis, poses)))
        return updates

    @staticmethod
    def _axis_values(axis, poses):
        # The chart's common inherited motion cancels exactly. Work in the
        # axis body's local translation, without subtracting world frames.
        translation = poses._deformations.local[axis['body']][0][:2]
        if not np.any(translation):
            return axis['baseline'].copy()
        return axis['baseline'] + axis['projection'] @ translation

    def get_element_poses(self, *, time_s=0.0, time_series=None, disabled_sources=()):
        self._check()
        poses = self.manager.get_body_poses(
            time_s=time_s,
            time_series=time_series,
            disabled_sources=disabled_sources,
            bodies=tuple(dict.fromkeys(a['body'] for a in self._attachments.values())),
        )
        return self._element_poses_from_bodies(poses)

    def _element_poses_from_bodies(self, poses):
        return {
            name: poses[a['body']]
            @ self.manager[a['body']].nominal_frame.inverse()
            @ Frame(a['base_element'])
            for name, a in self._attachments.items()
        }

    def read_bpm(
        self, name, *, time_s=0.0, time_series=None, beam_frame=None,
        disabled_sources=(), **coordinates
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
            time_s=time_s, time_series=time_series, bodies=[body.name],
            disabled_sources=disabled_sources,
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

    def get_table(self, *, time_s=None, time_series=None, disabled_sources=()):
        """Attached element occurrences, optionally with read-only world poses.

        Each attached element has its own row, even when several share a body.
        Unattached line elements are available in line.get_table().
        """
        from .tables import element_table

        return element_table(self, time_s=time_s, time_series=time_series,
                             disabled_sources=disabled_sources)

    def get_applied_table(self):
        """Read the element-entry frames currently installed on this line.

        This reads the lowered element fields without evaluating any source.
        last_applied_time_s is the latest mechanics update time, or the turn
        start for passage sampling. pose_time_s gives the time of each map.
        World display resolution is binary64.
        """
        self._check()
        survey = self.line.survey(include_element_frames=True)
        names = list(self._attachments)
        poses = np.array([
            (self._reference @ survey.get_frame(name, which='elem_start')).matrix
            for name in names
        ]).reshape(-1, 4, 4)
        applied = self.last_applied_time_s
        times = np.full(len(names), np.nan if applied is None else applied)
        if names and self._last_applied_passage_speed is not None:
            times += survey['s', names] / self._last_applied_passage_speed
        data = dict(
            name=np.array(names, dtype=str),
            body=np.array([self._attachments[name]['body'] for name in names], dtype=str),
            X=poses[:, 0, 3], Y=poses[:, 1, 3], Z=poses[:, 2, 3], pose=poses,
            pose_time_s=times,
        )
        return Table(dict(data, last_applied_time_s=self.last_applied_time_s,
                          disabled_sources=self._last_applied_source_mask),
                     col_names=list(data))

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

    def get_solenoid_axis_table(self, *, time_s=None, time_series=None,
                                disabled_sources=()):
        """Inspect adopted and effective local x0/y0 [m] for field-axis groups.

        With no time, inspect the installed buffer values. At explicit time
        sample the relative axis sources without changing any field. The
        ordinary get_table() continues to describe the whole tracking chart.
        """
        if time_s is None and time_series is not None:
            raise ValueError('Supply time_s when requesting an axis pose table')
        disabled_sources = self.manager._source_mask(disabled_sources)
        self._check()
        names = list(self._solenoid_axes)
        axes = list(self._solenoid_axes.values())
        baseline = np.array([a['baseline'] for a in axes]).reshape(-1, 2)
        if time_s is None:
            values = np.array([a['last_written'] for a in axes]).reshape(-1, 2)
        else:
            poses = self.manager.get_body_poses(
                time_s=time_s, time_series=time_series,
                disabled_sources=disabled_sources,
                bodies=tuple(dict.fromkeys(a['body'] for a in axes)),
            )
            values = np.array([self._axis_values(a, poses) for a in axes]).reshape(-1, 2)
        return Table(dict(
            name=np.array(names, dtype=str),
            chart_body=np.array([a['chart_body'] for a in axes], dtype=str),
            axis_body=np.array([a['body'] for a in axes], dtype=str),
            local_x0=baseline[:, 0], local_y0=baseline[:, 1],
            effective_x0=values[:, 0], effective_y0=values[:, 1],
        ))
