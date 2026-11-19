# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #
"""Environment bodies, line attachments and an explicit static lowering.

The first lowering writes numeric effective alignments at an explicit physical
time. enable_vibrations opts into host reference updates at turn boundaries.
No hierarchy of xdeps expressions is installed. Occurrence buffers and girder-
boundary kernels remain alternative accelerated lowerings of this same model.
"""
from dataclasses import asdict, dataclass
from types import MappingProxyType
from weakref import WeakKeyDictionary, ref

import numpy as np

from ..survey import Frame
from ..table import Table
from .sources import SOURCE_CLASSES, DisplacementPSD, PlaneWave


_FIELDS = ('shift_x', 'shift_y', 'shift_s', 'rot_x_rad', 'rot_y_rad',
           'rot_s_rad_no_frame', 'rot_shift_anchor')
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
    if (not np.all(np.isfinite(matrix))
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-14, rtol=0)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-12, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-12, rtol=0)):
        raise ValueError('Frame must be a finite proper rigid transform')
    return Frame(matrix)


def _values(element):
    return np.array([float(getattr(element, name)) for name in _FIELDS])


def _geometry(element):
    return tuple(float(getattr(element, name, 0.)) for name in _GEOMETRY)


@dataclass(slots=True)
class RigidBody:
    """Local alignment about a nominal world anchor; SI units.

    Pitch/yaw/roll follow xt.Frame.from_survey_angles(phi,theta,psi), i.e.
    Ry(yaw) Rx(-pitch) Rz(roll). Ancestor rotations compose as matrices.
    Topology and anchors are fixed after creation in this first draft.
    """
    name: str
    parent: str | None
    _nominal: np.ndarray
    shift_x: float = 0.
    shift_y: float = 0.
    shift_s: float = 0.
    rot_x_rad: float = 0.
    rot_y_rad: float = 0.
    rot_s_rad: float = 0.

    def __setattr__(self, name, value):
        if name in ('name', 'parent', '_nominal') and hasattr(self, name):
            raise AttributeError('Body topology is fixed; create a new binding')
        object.__setattr__(self, name, value)

    @property
    def nominal_frame(self):
        return Frame(self._nominal)


@dataclass(frozen=True)
class SourceBinding:
    name: str
    source: object
    targets: tuple
    component: str


class MechanicalTimeSeries:
    """Reusable sampled source displacements, independent of live body settings.

    Values are read-only (time, target, local translation) arrays in metres.
    Reuse this object for matched comparisons. Interpolation for apply/get_poses
    is linear, bounded by the supplied record, with no periodic replay.
    """
    def __init__(self, manager, time_s, values):
        self._manager = ref(manager)
        self._revision = manager._revision
        self._bodies = tuple(manager.bodies)
        self._targets = MappingProxyType({name: binding.targets for name, binding in manager.sources.items()})
        self.body_frames = MappingProxyType({name: self._freeze(body._nominal)
                                             for name, body in manager.bodies.items()})
        self.time_s = self._freeze(time_s)
        self.values = MappingProxyType({name: self._freeze(v) for name, v in values.items()})

    @staticmethod
    def _freeze(value):
        array = np.asarray(value, dtype=float)
        return np.frombuffer(array.tobytes(), dtype=float).reshape(array.shape)

    def get_table(self, body):
        """Read source-induced local translation for one body; no parent motion."""
        if body not in self._bodies:
            raise KeyError(body)
        values = np.zeros((len(self.time_s), 3))
        for name, targets in self._targets.items():
            if body in targets:
                values += self.values[name][:, targets.index(body)]
        return Table(dict(time_s=self.time_s.copy(), shift_x=values[:, 0],
                          shift_y=values[:, 1], shift_s=values[:, 2]), index='time_s')

    def _check(self, manager):
        if self._manager() is not manager or manager is None or manager._revision != self._revision:
            raise ValueError('Time series belongs to another or changed mechanics model')


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

    def _line(self, line):
        if line is None:
            if len(self.env.lines) != 1:
                raise ValueError('Specify line= or use line.mechanics when the environment has multiple lines')
            line = next(iter(self.env.lines.values()))
        return self.for_line(line)

    def new_girder(self, name, *, elements=None, line=None, anchor='center', parent=None):
        """Retrofit elements, or create an explicitly anchored shared body.

        'center' is the chord midpoint of the selected entrance/exit span, with
        axes from its first nominal entrance frame. A Frame gives full world
        anchor/orientation; a three-vector gives world position with XYZ axes.
        Parent names refer to existing bodies. No sublines are required.
        """
        if not isinstance(name, str) or not name or name in self._bodies:
            raise ValueError('Use a new nonempty body name')
        if isinstance(parent, RigidBody):
            if self._bodies.get(parent.name) is not parent:
                raise ValueError('Parent must belong to this environment')
            parent = parent.name
        if parent is not None and parent not in self._bodies:
            raise ValueError('Create the parent body first')
        view, candidates = None, []
        if elements is not None:
            view = self._line(line)
            candidates = view._candidates(elements)
        if isinstance(anchor, str):
            if anchor != 'center' or not candidates:
                raise ValueError("Use anchor='center' with attachments, or supply a world Frame/XYZ point")
            nominal = Frame(candidates[0]['ref_start'])
            nominal.matrix[:3, 3] = (candidates[0]['ref_start'][:3, 3]
                                      + candidates[-1]['ref_end'][:3, 3]) / 2
        else:
            nominal = _frame(anchor)
        frozen = np.frombuffer(nominal.matrix.tobytes(), dtype=float).reshape(4, 4)
        body = RigidBody(name, parent, frozen)
        self._bodies[name] = body
        if view is not None:
            view._attach(name, candidates)
        self._revision += 1
        return body

    def add_source(self, name, source, *, targets, component='y'):
        """Bind a named source model, independently of girder construction.

        Scalar displacement models use nominal body-local x/y/s. PlaneWave is
        a world vector field sampled at root anchors and projected into their
        nominal axes. The first draft does not infer multi-foot rotations.
        """
        if not isinstance(name, str) or not name or name in self._sources:
            raise ValueError('Use a new nonempty source name')
        if not callable(getattr(source, 'evaluate', None)):
            raise TypeError('A source must implement evaluate(time_s=..., positions_m=...)')
        frame = getattr(source, 'frame', 'body_local')
        if frame not in ('body_local', 'world'):
            raise ValueError("Source frame must be 'body_local' or 'world'")
        targets = (targets,) if isinstance(targets, str) else tuple(targets)
        if not targets or len(set(targets)) != len(targets) or any(n not in self._bodies for n in targets):
            raise ValueError('Select unique existing bodies')
        if component not in ('x', 'y', 's'):
            raise ValueError("Displacement component must be 'x', 'y' or 's'")
        for target in targets:
            parent = self._bodies[target].parent
            while parent is not None:
                if parent in targets:
                    raise ValueError('One source cannot drive both a body and its ancestor')
                parent = self._bodies[parent].parent
            if frame == 'world' and self._bodies[target].parent is not None:
                raise ValueError('World-field shortcut drives roots; child absolute drives need explicit response handling')
        binding = SourceBinding(name, source, targets, component)
        self._sources[name] = binding
        self._revision += 1
        return binding

    def _evaluate_source(self, binding, time_s):
        positions = np.array([self._bodies[n]._nominal[:3, 3] for n in binding.targets])
        values = np.asarray(binding.source.evaluate(time_s=time_s, positions_m=positions))
        if np.iscomplexobj(values) or not np.all(np.isfinite(values)):
            raise ValueError(f'{binding.name}: source must return finite real displacement')
        if getattr(binding.source, 'frame', 'body_local') == 'world':
            if values.shape != (len(time_s), len(positions), 3):
                raise ValueError('World source must return shape (times, targets, 3)')
            rotations = np.array([self._bodies[n]._nominal[:3, :3] for n in binding.targets])
            return np.einsum('bji,tbj->tbi', rotations, values)
        if values.shape == (len(time_s),):
            values = np.broadcast_to(values[:, None], (len(time_s), len(positions)))
        if values.shape != (len(time_s), len(positions)):
            raise ValueError('Scalar local source must return shape (times, targets) or (times,)')
        output = np.zeros((len(time_s), len(binding.targets), 3))
        output[..., ('x', 'y', 's').index(binding.component)] = values
        return output

    def compute_time_series(self, *, time_s, max_memory_bytes=128*1024**2):
        """Compute reusable prescribed motion at explicit times [s], on the host.

        Analytic/measured models accept nonuniform increasing times; PSD models
        require a uniform grid. This explicit short-record backend has a
        conservative allocation guard and never creates a GPU history cache.
        """
        times = np.asarray(time_s, dtype=float)
        if times.ndim != 1 or len(times) < 2 or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
            raise ValueError('Use at least two increasing finite times in seconds')
        counts = [len(b.targets) for b in self.sources.values()]
        estimate = 24*len(times)*sum(counts) + max(
            (64*len(times)*n + 80*n*n for n in counts), default=0)
        if not isinstance(max_memory_bytes, int) or max_memory_bytes <= 0:
            raise ValueError('max_memory_bytes must be a positive integer')
        if estimate > max_memory_bytes:
            raise ValueError(f'Estimated short-record workspace {estimate} bytes exceeds max_memory_bytes')
        values = {name: self._evaluate_source(binding, times)
                  for name, binding in self.sources.items()}
        return MechanicalTimeSeries(self, times, values)

    def get_body_poses(self, *, time_s=0., time_series=None, disabled_sources=()):
        """Inspect independent xt.Frame world poses, without changing the line."""
        if not np.isscalar(time_s) or not np.isfinite(time_s):
            raise ValueError('time_s must be a finite scalar in seconds')
        if not set(disabled_sources).issubset(self.sources):
            raise ValueError('Unknown disabled source')
        if time_series is not None:
            time_series._check(self)
            if time_s < time_series.time_s[0] or time_s > time_series.time_s[-1]:
                raise ValueError('Requested time lies outside the computed time series')
        shifts = {name: np.zeros(3) for name in self.bodies}
        for name, binding in self.sources.items():
            if name in disabled_sources:
                continue
            if time_series is None:
                values = self._evaluate_source(binding, np.array([time_s]))[0]
            else:
                record = time_series.values[name]
                values = np.array([[np.interp(time_s, time_series.time_s, record[:, i, j])
                                    for j in range(3)] for i in range(len(binding.targets))])
            for target, value in zip(binding.targets, values):
                shifts[target] += value
        world = {}
        for name, body in self.bodies.items():
            local = np.array([body.shift_x, body.shift_y, body.shift_s,
                              body.rot_x_rad, body.rot_y_rad, body.rot_s_rad], dtype=float)
            if not np.all(np.isfinite(local)):
                raise ValueError(f'{name}: local alignment must be finite')
            local[:3] += shifts[name]
            delta = Frame.from_survey_angles(X=local[0], Y=local[1], Z=local[2],
                                             phi=local[3], theta=local[4], psi=local[5])
            nominal = Frame(body._nominal)
            if body.parent is None:
                world[name] = nominal @ delta
            else:
                parent = self.bodies[body.parent]
                world[name] = world[parent.name] @ Frame(parent._nominal).inverse() @ nominal @ delta
        return world

    def apply(self, *, time_s=0., time_series=None, disabled_sources=()):
        """Apply one physical time to ALL attached lines in this environment.

        Explicit host-side numeric alignment lowering. Does not start a clock,
        register a track callback, or update sources once per particle/turn.
        Unsupported bindings/conflicting edits are rejected before any writes.
        """
        self._check_aliases()
        poses = self.get_body_poses(time_s=time_s, time_series=time_series,
                                    disabled_sources=disabled_sources)
        updates = []
        for view in self._line_views.values():
            updates.extend(view._updates(poses))
        for attachment, element, values in updates:
            for field, value in zip(_FIELDS, values):
                setattr(element, field, value)
            attachment['last_written'] = values.copy()

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
        counts = {}
        for line in self.env._lines_weakrefs:
            for name in line.element_names:
                identity = id(line.element_dict[name])
                counts[identity] = counts.get(identity, 0) + 1
        for view in self._line_views.values():
            for name in view._attachments:
                if counts.get(id(view.line.element_dict[name])) != 1:
                    raise ValueError(f'{name}: shared/repeated element objects require an occurrence-buffer lowering')

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

    def get_table(self):
        """Configured local alignment; use get_body_poses for composed world frames."""
        bodies = list(self.bodies.values())
        data = dict(name=np.array([b.name for b in bodies]),
                    parent=np.array([b.parent or '' for b in bodies]))
        for field in ('shift_x', 'shift_y', 'shift_s', 'rot_x_rad', 'rot_y_rad', 'rot_s_rad'):
            data[field] = np.array([getattr(b, field) for b in bodies], dtype=float)
        return Table(data)

    def to_dict(self):
        """Save physical definitions and line bindings, not sampled source history."""
        for binding in self.sources.values():
            if SOURCE_CLASSES.get(type(binding.source).__name__) is not type(binding.source):
                raise TypeError(f'{binding.name}: custom source serialization is not defined yet')
        lines = []
        for line, view in self._line_views.items():
            if not view._attachments:
                continue
            names = [name for name, ll in self.env.lines.items() if ll is line]
            if len(names) != 1:
                raise ValueError('Register attached lines under one environment name before serialization')
            attachments = []
            for a in view._attachments.values():
                attachments.append({key: value.tolist() if isinstance(value, np.ndarray) else value
                                    for key, value in a.items() if key not in ('last_written', 'object_id')})
            lines.append(dict(line=names[0], reference_frame=view._reference.matrix.tolist(),
                              attachments=attachments))
        return dict(bodies=[dict(name=b.name, parent=b.parent, anchor=b._nominal.tolist(),
                                 alignment=[b.shift_x, b.shift_y, b.shift_s, b.rot_x_rad, b.rot_y_rad, b.rot_s_rad])
                            for b in self.bodies.values()],
                    sources=[dict(name=b.name, targets=list(b.targets), component=b.component,
                                  source=dict(type=type(b.source).__name__, parameters=asdict(b.source)))
                             for b in self.sources.values()], lines=lines)

    def from_dict(self, data):
        if self.bodies or self.sources or self._line_views:
            raise ValueError('Load mechanics into an empty manager')
        for item in data['bodies']:
            b = self.new_girder(item['name'], parent=item['parent'], anchor=Frame(item['anchor']))
            b.shift_x, b.shift_y, b.shift_s, b.rot_x_rad, b.rot_y_rad, b.rot_s_rad = item['alignment']
        for item in data['lines']:
            view = self.for_line(item['line'])
            view.set_reference_frame(Frame(item['reference_frame']))
            for stored in item['attachments']:
                candidates = view._candidates([stored['element']])
                candidate = candidates[0]
                for key in ('ref_start', 'ref_end', 'base_element', 'baseline'):
                    candidate[key] = np.array(stored[key], dtype=float)
                candidate['geometry'] = tuple(stored['geometry'])
                view._attach(stored['body'], [candidate])
        for item in data['sources']:
            source = item['source']
            self.add_source(item['name'], SOURCE_CLASSES[source['type']](**source['parameters']),
                            targets=item['targets'], component=item['component'])
        return self


class LineMechanics:
    """Line-specific attachments/view of env.mechanics; not an independent manager."""
    def __init__(self, manager, line):
        self.manager = manager
        self._line_ref = ref(line)
        self._reference = Frame()
        self._reference_is_explicit = False
        self._attachments = {}
        self._element_names = None
        self._line_geometry = None
        self._vibrations_enabled = False
        self._time_series = None

    @property
    def vibrations_enabled(self):
        return self._vibrations_enabled

    def enable_vibrations(self, *, time_series=None):
        """Enable turn-boundary motion updates using Particles.time_s [s].

        Current exploration backend: one host-side pose update per complete
        turn, including on GPU contexts. No xdeps chain or per-particle source
        evaluation. This is a correctness reference, not the final fast GPU
        scheduler. Multi-line, SyncTime and partial-turn scheduling are not yet
        defined; shared geometry remains usable with explicit manager.apply().
        """
        attached = [v for v in self.manager._line_views.values() if v._attachments]
        if len(attached) != 1 or not self._attachments:
            raise NotImplementedError('Automatic multi-line vibration clocks are not defined yet; use env.mechanics.apply(time_s=...)')
        if time_series is not None:
            time_series._check(self.manager)
        elif any(isinstance(b.source, DisplacementPSD) for b in self.manager.sources.values()):
            raise ValueError('Call compute_time_series first and supply time_series for PSD tracking')
        self.manager._check_aliases()
        self._check()
        self._time_series = time_series
        self._vibrations_enabled = True

    def disable_vibrations(self):
        """Stop automatic motion, retaining static girder and local alignment.

        Source phases/records are not reset. A later enable resumes at the
        particles' physical time, rather than restarting a random waveform.
        """
        self.manager.apply(disabled_sources=tuple(self.manager.sources))
        self._vibrations_enabled = False
        self._time_series = None

    def _apply_for_tracking(self, time_s):
        if sum(bool(v._attachments) for v in self.manager._line_views.values()) != 1:
            raise NotImplementedError('Automatic multi-line vibration clocks are not defined yet')
        self.manager.apply(time_s=time_s, time_series=self._time_series)

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

    def new_girder(self, name, *, elements, anchor='center', parent=None):
        return self.manager.new_girder(name, line=self.line, elements=elements,
                                       anchor=anchor, parent=parent)

    def attach(self, body, *, elements):
        """Attach this line's elements to an existing physical body, possibly shared."""
        name = body.name if isinstance(body, RigidBody) else body
        if name not in self.manager.bodies or (isinstance(body, RigidBody) and self.manager[name] is not body):
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
        updates = []
        for name, attachment in self._attachments.items():
            element = self.line.element_dict.get(name)
            if element is None or id(element) != attachment['object_id']:
                raise ValueError(f'{name}: attached object changed; original alignment cannot be restored automatically')
            if not np.array_equal(_values(element), attachment['last_written']):
                raise ValueError(f'{name}: effective alignment was edited; clear mechanics before editing local alignment')
            for field in _FIELDS:
                if getattr(getattr(self.line.env.ref[name], field), '_expr', None) is not None:
                    raise ValueError(f'{name}.{field}: a new alignment expression must not be overwritten')
            updates.append((attachment, element, attachment['baseline']))
        return updates

    def _clear_bindings(self):
        self._attachments.clear()
        self._element_names = None
        self._line_geometry = None
        self._vibrations_enabled = False
        self._time_series = None
        self._reference = Frame()
        self._reference_is_explicit = False

    def _candidates(self, elements):
        if self._attachments:
            self._check()
        names = (elements,) if isinstance(elements, str) else tuple(elements)
        if not names or len(set(names)) != len(names):
            raise ValueError('Select nonempty unique element names')
        line = self.line
        if not self._reference_is_explicit and any(
                v is not self and v._attachments for v in self.manager._line_views.values()):
            raise ValueError('Set an explicit survey reference frame before attaching a second line')
        order = list(line.element_names)
        elements_by_id = {}
        for ll in self.manager.env._lines_weakrefs:
            for name in ll.element_names:
                element = ll.element_dict[name]
                elements_by_id[id(element)] = elements_by_id.get(id(element), 0) + 1
        for name in names:
            if order.count(name) != 1 or name in self._attachments:
                raise ValueError(f'{name}: unknown, repeated or already attached occurrence')
            element = line.element_dict[name]
            if (not getattr(element, 'allow_rot_and_shift', False)
                    or type(element).__name__ == 'Replica'
                    or (hasattr(element, 'weight') and hasattr(element, '_parent'))):
                raise ValueError(f'{name}: first lowering supports unsliced alignment-capable elements')
            if elements_by_id.get(id(element), 0) != 1:
                raise ValueError(f'{name}: shared/repeated element objects require an occurrence-buffer lowering')
            for field in (*_FIELDS, *_GEOMETRY):
                if hasattr(element, field) and getattr(getattr(line.env.ref[name], field), '_expr', None) is not None:
                    raise ValueError(f'{name}.{field}: expression-driven geometry/alignment is not supported by numeric lowering')
        survey = line.survey(include_element_frames=True, reverse=False)
        output = []
        for name in sorted(names, key=order.index):
            element = line.element_dict[name]
            output.append(dict(element=name,
                ref_start=(self._reference @ survey.get_frame(name, which='ref_start')).matrix,
                ref_end=(self._reference @ survey.get_frame(name, which='ref_end')).matrix,
                base_element=(self._reference @ survey.get_frame(name, which='elem_start')).matrix,
                baseline=_values(element), last_written=_values(element), geometry=_geometry(element),
                object_id=id(element)))
        return output

    def _attach(self, name, candidates):
        for candidate in candidates:
            candidate['body'] = name
            self._attachments[candidate['element']] = candidate
        self._element_names = tuple(self.line.element_names)
        self._line_geometry = self._geometry_signature()

    def _geometry_signature(self):
        return tuple((id(self.line.element_dict[n]), _geometry(self.line.element_dict[n]))
                     for n in self.line.element_names)

    def _check(self):
        if self._attachments and self.line.aperture.is_built:
            raise ValueError('Discard/rebuild the aperture survey model when applying mechanics; pipe-body coupling is not implemented yet')
        if self._element_names is not None and tuple(self.line.element_names) != self._element_names:
            raise ValueError('Line sequence changed after mechanics attachment; rebuild bindings')
        if self._line_geometry is not None and self._geometry_signature() != self._line_geometry:
            raise ValueError('Line reference geometry changed after attachment; rebuild survey bindings')
        for name, a in self._attachments.items():
            element = self.line.element_dict[name]
            if id(element) != a['object_id']:
                raise ValueError(f'{name}: attached element object was replaced')
            if _geometry(element) != a['geometry']:
                raise ValueError(f'{name}: nominal geometry changed after attachment')
            if not np.array_equal(_values(element), a['last_written']):
                raise ValueError(f'{name}: local alignment edited after attachment; restore/rebind explicitly')
            for field in (*_FIELDS, *_GEOMETRY):
                if hasattr(element, field) and getattr(getattr(self.line.env.ref[name], field), '_expr', None) is not None:
                    raise ValueError(f'{name}.{field}: expression-driven geometry/alignment is not supported by numeric lowering')

    def _updates(self, poses):
        self._check()
        updates = []
        for name, a in self._attachments.items():
            element = self.line.element_dict[name]
            body = self.manager[a['body']]
            target = poses[body.name] @ Frame(body._nominal).inverse() @ Frame(a['base_element'])
            effective = Frame(a['ref_start']).inverse() @ target @ Frame().rotate_s(-element.rot_s_rad)
            values = np.array([*effective.XYZ, effective.phi, effective.theta, effective.psi, 0.])
            updates.append((a, element, values))
        return updates

    def get_element_poses(self, *, time_s=0., time_series=None, disabled_sources=()):
        poses = self.manager.get_body_poses(time_s=time_s, time_series=time_series,
                                           disabled_sources=disabled_sources)
        return {name: poses[a['body']] @ self.manager[a['body']].nominal_frame.inverse()
                @ Frame(a['base_element']) for name, a in self._attachments.items()}

    def get_table(self):
        return Table(dict(name=np.array(list(self._attachments)),
                          body=np.array([a['body'] for a in self._attachments.values()])))

    def get_alignment_table(self):
        """Original element-local versus current effective alignment fields.

        Body settings (manager.get_table) are deliberate girder alignment;
        computed source tables are vibration inputs. These are distinct inputs,
        composed as rigid transforms. Differences of Euler angles are not an
        exact additive contribution decomposition. Design rot_s_rad is shown
        separately and is never changed by the numeric mechanics lowering.
        """
        names = list(self._attachments)
        data = dict(name=np.array(names), body=np.array([self._attachments[n]['body'] for n in names]))
        for index, field in enumerate(_FIELDS):
            data['local_' + field] = np.array([self._attachments[n]['baseline'][index] for n in names])
            data['effective_' + field] = np.array([getattr(self.line.element_dict[n], field) for n in names])
        data['design_rot_s_rad'] = np.array([self.line.element_dict[n].rot_s_rad for n in names])
        return Table(data)
