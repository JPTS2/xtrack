# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #
"""Optional bounded host records of inputs actually used at mechanics freezes."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from weakref import ref

import numpy as np

from .tables import _static_inputs


def _freeze(values):
    values = np.asarray(values, dtype=float)
    return np.frombuffer(values.tobytes(), dtype=float).reshape(values.shape)


@dataclass(frozen=True)
class AppliedSourceHistory:
    """Immutable applied input samples, held until the next captured freeze.

    Samples are normalized nominal-body six-coordinates [m, rad], not world
    poses or an acquisition record. No extrapolation past the captured interval
    is permitted. A supplied prescribed record and disabled-source mask are
    retained so replay cannot silently change the accompanying disturbances.
    """

    _manager: object
    _revision: int
    _case: object
    time_s: object
    values: object
    targets: object
    references: object
    roles: object
    disabled_sources: tuple
    prescribed_record: object
    overflowed: bool
    overflow_time_s: object

    def __post_init__(self):
        times = np.asarray(self.time_s)
        if (
            times.ndim != 1
            or not len(times)
            or np.iscomplexobj(times)
            or not np.all(np.isfinite(times))
            or np.any(np.diff(times) <= 0)
        ):
            raise ValueError('Applied history requires increasing finite freeze times')
        values = {}
        for name, array in self.values.items():
            array = np.asarray(array)
            if (
                array.shape != (len(times), len(self.targets[name]), 6)
                or np.iscomplexobj(array)
                or not np.all(np.isfinite(array))
            ):
                raise ValueError(
                    'Applied history requires finite time-by-target-by-six inputs'
                )
            values[name] = _freeze(array)
        object.__setattr__(self, 'time_s', _freeze(times))
        object.__setattr__(self, 'values', MappingProxyType(values))
        for name in ('targets', 'references', 'roles', '_case'):
            data = dict(getattr(self, name))
            if name in ('targets', '_case'):
                data = {key: tuple(value) for key, value in data.items()}
            object.__setattr__(self, name, MappingProxyType(data))
        object.__setattr__(self, 'disabled_sources', tuple(self.disabled_sources))

    def _check(self, manager, *, bodies=None):
        if (
            manager is None
            or self._manager() is not manager
            or manager._revision != self._revision
        ):
            raise ValueError(
                'Applied history belongs to another or changed mechanics model'
            )
        names = tuple(self._case) if bodies is None else manager._body_closure(bodies)
        for name in names:
            if (
                name not in self._case
                or _static_inputs(manager[name]) != self._case[name]
            ):
                raise ValueError(
                    'Applied history static/reference case does not match this query'
                )
        if self.prescribed_record is not None:
            self.prescribed_record._check(manager)

    def at(self, time_s):
        """Read recorded inputs with previous-sample hold; never query providers."""
        return self._sample(time_s)[1]

    def _sample(self, time_s):
        if not np.isscalar(time_s) or np.iscomplexobj(time_s) or not np.isfinite(time_s):
            raise ValueError('Applied history time must be finite scalar seconds')
        if time_s < self.time_s[0] or time_s > self.time_s[-1]:
            raise ValueError('Requested time lies outside the captured applied history')
        index = np.searchsorted(self.time_s, time_s, side='right') - 1
        return float(self.time_s[index]), {
            name: values[index] for name, values in self.values.items()
        }

    def _resolve(self, manager, bodies, time_s, time_series, disabled_sources):
        if disabled_sources or (
            time_series is not None and time_series is not self.prescribed_record
        ):
            raise ValueError('Applied history fixes the source mask and prescribed record')
        self._check(manager, bodies=bodies)
        pose_time_s, values = self._sample(time_s)
        return pose_time_s, self.prescribed_record, self.disabled_sources, values

    def to_dict(self):
        """Export numerical evidence and provenance without a live manager."""
        prescribed = None
        if self.prescribed_record is not None:
            record = self.prescribed_record
            digest = hashlib.sha256(memoryview(record.time_s).cast('B'))
            for name, values in record.values.items():
                digest.update(name.encode())
                digest.update(record.replay_modes[name].encode())
                digest.update(memoryview(values).cast('B'))
            prescribed = dict(
                time_start_s=float(record.time_s[0]),
                time_stop_s=float(record.time_s[-1]),
                source_names=list(record.values),
                replay_modes=dict(record.replay_modes),
                sha256=digest.hexdigest(),
            )
        return dict(
            kind='AppliedSourceHistory',
            time_s=self.time_s.tolist(),
            manager_revision=self._revision,
            body_scope=list(self._case),
            values={name: value.tolist() for name, value in self.values.items()},
            targets=dict(self.targets),
            references=dict(self.references),
            roles=dict(self.roles),
            static_case={name: list(values) for name, values in self._case.items()},
            disabled_sources=self.disabled_sources,
            overflowed=self.overflowed,
            overflow_time_s=self.overflow_time_s,
            interpolation='previous',
            coordinates='nominal body [x,y,s,pitch,yaw,roll]',
            prescribed_record=prescribed,
        )

    def to_json(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + '\n')


class AppliedSourceRecorder:
    """Record explicitly selected live inputs with a fixed case/mode/record.

    Capacity exhaustion stops recording, sets overflowed, and leaves tracking
    running. snapshot() still exports the valid prefix with its exact bounds.
    Starting another numerical experiment requires a new recorder. Provider
    evaluation, plant advancement and GPU pose-history storage never occur here.
    The byte budget covers owned numeric buffers, a pending block and one
    snapshot; the already existing prescribed record is referenced, not copied.
    """

    def __init__(
        self, view, *, sources, max_samples=4096, max_memory_bytes=32 * 1024**2,
        case_scope='affected',
    ):
        manager = view.manager
        names = (sources,) if isinstance(sources, str) else tuple(sources)
        if not names or len(set(names)) != len(names):
            raise ValueError('Select unique live source names to record')
        if any(
            name not in manager.sources or not manager.sources[name].live
            for name in names
        ):
            raise ValueError(
                'Applied recording requires existing explicitly live source bindings'
            )
        if (
            isinstance(max_samples, bool)
            or not isinstance(max_samples, int)
            or max_samples < 2
            or isinstance(max_memory_bytes, bool)
            or not isinstance(max_memory_bytes, int)
            or max_memory_bytes <= 0
        ):
            raise ValueError(
                'Use max_samples >= 2 and a positive integer max_memory_bytes'
            )
        if not view._consumed_bodies():
            raise ValueError('Attach a line before recording its applied inputs')
        closure = manager._body_closure(view._consumed_bodies())
        if any(
            not closure.intersection(manager.sources[name].targets) for name in names
        ):
            raise ValueError('Recorded sources must affect this attached line')
        if case_scope not in ('affected', 'line'):
            raise ValueError("Use case_scope='affected' or 'line'")
        if case_scope == 'affected':
            selected_targets = {
                target for name in names for target in manager.sources[name].targets
            }
            affected = {}
            for name in closure:
                pending, current = [], name
                while current is not None and current not in affected:
                    pending.append(current)
                    current = manager[current].parent
                for current in reversed(pending):
                    affected[current] = current in selected_targets or affected.get(
                        manager[current].parent, False
                    )
            closure = manager._body_closure([name for name in closure if affected[name]])
        existing = getattr(view, '_applied_source_recorder', None)
        if existing is not None and existing.active:
            raise ValueError('Stop the active applied source recorder first')
        # Reserve space for both the live buffers and one immutable export.
        size = (
            8
            * max_samples
            * (1 + 6 * sum(len(manager.sources[n].targets) for n in names))
        )
        pending_bytes = 48 * sum(len(manager.sources[n].targets) for n in names)
        if 2 * size + pending_bytes + 96 * len(closure) > max_memory_bytes:
            raise MemoryError('Applied source recording exceeds max_memory_bytes')
        self._view = ref(view)
        self._manager = ref(manager)
        self._revision = manager._revision
        self._case = {name: _static_inputs(manager[name]) for name in closure}
        self._sources = names
        self._targets = {name: manager.sources[name].targets for name in names}
        self._references = {name: manager.sources[name].reference for name in names}
        self._roles = {name: manager.sources[name].role for name in names}
        self._times = np.empty(max_samples)
        self._values = {
            name: np.empty((max_samples, len(self._targets[name]), 6)) for name in names
        }
        self._count = 0
        self._disabled_sources = None
        self._prescribed_record = None
        self._invalid_reason = None
        self._pending = None
        self._preflight_complete = False
        self.active = True
        self.overflowed = False
        self.overflow_time_s = None
        view._applied_source_recorder = self

    @property
    def n_samples(self):
        return self._count

    @property
    def max_samples(self):
        return len(self._times)

    @property
    def source_names(self):
        return self._sources

    def check_case(self):
        """Validate at call entry, not once per turn in the compiled path."""
        manager = self._manager()
        if manager is None or manager._revision != self._revision:
            raise ValueError('Applied recorder belongs to a changed mechanics model')
        if any(
            _static_inputs(manager[name]) != values
            for name, values in self._case.items()
        ):
            raise ValueError(
                'Applied recorder static/reference case changed; start a new recording'
            )

    def stop(self):
        self.active = False
        self._pending = None
        self._preflight_complete = False
        view = self._view()
        if view is not None and getattr(view, '_applied_source_recorder', None) is self:
            view._applied_source_recorder = None

    def preflight(
        self, time_s, source_values, *, disabled_sources=(), time_series=None
    ):
        """Validate/copy pending inputs before any element or mode is changed."""
        if not self.active:
            return
        self._pending = None
        self._preflight_complete = False
        manager = self._manager()
        if manager is None or manager._revision != self._revision:
            raise ValueError('Applied recorder belongs to a changed mechanics model')
        if not np.isscalar(time_s) or not np.isfinite(time_s):
            raise ValueError('Applied recording requires a finite freeze time')
        disabled = tuple(sorted(disabled_sources))
        if self._disabled_sources is not None and (
            disabled != self._disabled_sources
            or time_series is not self._prescribed_record
        ):
            raise ValueError(
                'Applied source mode/record changed; start a new recording'
            )
        current = {}
        for name in self._sources:
            values = (
                np.zeros((len(self._targets[name]), 6))
                if name in disabled
                else source_values.get(name)
            )
            if values is None:
                raise ValueError(f'{name}: applied live input values were not supplied')
            values = np.asarray(values)
            if (
                values.shape != (len(self._targets[name]), 6)
                or np.iscomplexobj(values)
                or not np.all(np.isfinite(values))
            ):
                raise ValueError(
                    f'{name}: expected finite normalized target-by-six applied values'
                )
            current[name] = np.array(values, dtype=float, copy=True)
        if self._count and time_s <= self._times[self._count - 1]:
            if time_s == self._times[self._count - 1] and all(
                np.array_equal(current[name], self._values[name][self._count - 1])
                for name in self._sources
            ):
                self._preflight_complete = True
                return
            raise ValueError(
                'Applied recording received reversed or conflicting same-time freezes'
            )
        if self._count == len(self._times):
            self.overflowed, self.overflow_time_s = True, float(time_s)
            self.stop()
            return
        self._pending = (float(time_s), current, disabled, time_series)
        self._preflight_complete = True

    def capture(self, time_s, source_values, *, disabled_sources=(), time_series=None):
        """Commit a preflighted block after successful writes; never evaluate."""
        if not self.active:
            return
        if not self._preflight_complete:
            self._invalid_reason = 'Applied recording capture was not preflighted'
            self.stop()
            return
        pending = self._pending
        self._pending, self._preflight_complete = None, False
        if pending is None:  # Reapplying identical inputs at the same time.
            return
        applied_time, current, disabled, record = pending
        if applied_time != time_s:
            self._invalid_reason = (
                'Applied recording capture does not match its preflight time'
            )
            self.stop()
            return
        self._disabled_sources, self._prescribed_record = disabled, record
        self._times[self._count] = time_s
        for name, values in current.items():
            self._values[name][self._count] = values
        self._count += 1

    def snapshot(self):
        if self._invalid_reason is not None:
            raise ValueError(self._invalid_reason)
        if not self._count:
            raise ValueError('No applied source inputs have been captured')
        return AppliedSourceHistory(
            self._manager,
            self._revision,
            MappingProxyType(dict(self._case)),
            self._times[: self._count],
            {name: values[: self._count] for name, values in self._values.items()},
            MappingProxyType(dict(self._targets)),
            MappingProxyType(dict(self._references)),
            MappingProxyType(dict(self._roles)),
            self._disabled_sources,
            self._prescribed_record,
            self.overflowed,
            self.overflow_time_s,
        )
