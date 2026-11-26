# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #
"""Batched reference-geometry and alignment ownership checks at tracking entry.

The initial trace records scalar *storage* read by public geometry getters,
including property selectors and underlying fields. Subsequent validation uses
one context gather and one bounded host transfer, not scalar GPU .get calls.
No tracking-turn hook invokes this guard. Fixed geometry during a tracking call
remains a scheduler contract; explicit edits are checked at the next entry.
"""

from pathlib import Path

import numpy as np
import xobjects as xo
from .context_arrays import to_context


def _compile(context):
    if 'mechanics_guard_gather' not in context.kernels:
        context.add_kernels(
            sources=[Path(__file__).parent / 'runtime_src/geometry_guard.h'],
            kernels={
                'mechanics_guard_gather': xo.Kernel(
                    args=[
                        xo.Arg(xo.Int64, name='n'),
                        xo.Arg(xo.Int64, name='offsets', pointer=True),
                        xo.Arg(xo.Int8, name='sizes', pointer=True),
                        xo.Arg(xo.Int8, name='buffer', pointer=True),
                        xo.Arg(xo.UInt64, name='out', pointer=True),
                    ],
                    n_threads='n',
                )
            },
        )


class GeometryGuard:
    """Prepared storage-level checks, preserving numeric-lowering ownership.

    Construct only after full view.validate(). A changed value of a protected
    geometry dependency rejects reuse even if a derived geometry value happens
    to cancel. Strength changes are accepted unless geometry depends on them.
    This object is private prepared state and is never serialized.
    """

    def __init__(self, view):
        self.view = view
        self.buffer = view.line.tracker._buffer
        self.context = self.buffer.context
        self.revision = view.manager._revision
        self.names = tuple(view.line.element_names)
        elements = view.line.element_dict
        self.objects = {name: elements[name] for name in self.names}
        self.attachments = tuple(view._numeric_attachments())
        self.has_solenoid_axes = bool(view._solenoid_axes)
        from .model import (
            _attachment_fields, _reference_geometry, _resolved_slice_parent,
            _storage_identity,
        )
        from ..beam_elements.slice_base import _SliceBase

        self.slice_parents = [
            (element, element.parent_name, _resolved_slice_parent(element, elements))
            for element in self.objects.values()
            if isinstance(element, _SliceBase)
        ]
        geometry_objects = list(view._protected_geometry_objects().values())
        geometry_objects = list({id(obj): obj for obj in geometry_objects}.values())
        self.storage = [
            (element, _storage_identity(element))
            for element in geometry_objects
            if hasattr(element, '_xobject')
        ]

        self.python_geometry = [
            (element, _reference_geometry(element))
            for element in geometry_objects
            if (
                not type(element).__module__.startswith('xtrack.beam_elements')
                or getattr(element, 'iscollective', False)
            )
        ]
        buffers = {
            id(element._buffer): element._buffer
            for element in geometry_objects
            if hasattr(element, '_buffer')
        }
        for element in geometry_objects:
            if (
                type(element).__module__.startswith('xtrack.beam_elements')
                and not getattr(element, 'iscollective', False)
                and hasattr(element, '_buffer')
                and element._buffer is not self.buffer
            ):
                raise NotImplementedError(
                    'Prepared builtin geometry requires one tracker buffer'
                )
        reads = {}
        patched = []
        missing = object()

        def install(buffer, method, replacement):
            # Patch only these buffer instances, never a global buffer class.
            previous = buffer.__dict__.get(method, missing)
            patched.append((buffer, method, previous))
            setattr(buffer, method, replacement)

        def trace(buffer):
            original = buffer.to_bytearray

            def traced(offset, nbytes):
                value = original(offset, nbytes)
                if buffer is self.buffer:
                    if nbytes not in (1, 2, 4, 8):
                        raise NotImplementedError(
                            'Prepared geometry requires scalar storage dependencies'
                        )
                    reads[(int(offset), int(nbytes))] = bytes(value)
                return value

            return traced

        def reject_arrays(*args, **kwargs):
            raise NotImplementedError(
                'Prepared geometry does not support array-based geometry getters; use the reference backend'
            )

        try:
            for buffer in buffers.values():
                install(buffer, 'to_bytearray', trace(buffer))
                for method in ('to_nplike', 'to_nparray'):
                    if hasattr(buffer, method):
                        install(buffer, method, reject_arrays)
            view._geometry_signature()
        finally:
            for buffer, method, previous in reversed(patched):
                if previous is missing:
                    delattr(buffer, method)
                else:
                    setattr(buffer, method, previous)
        # Property-dependent storage is discovered once. It need not have the
        # public field name (for example Bend.length reads its private storage).
        offsets = [key[0] for key in reads]
        sizes = [key[1] for key in reads]
        expected = [
            int.from_bytes(value, byteorder='little') for value in reads.values()
        ]
        self.n_geometry = len(offsets)
        for attachment in self.attachments:
            element = self.objects[attachment['element']]
            fields = _attachment_fields(attachment)
            offsets.extend(element._xobject._get_offset(field) for field in fields)
            sizes.extend([8] * len(fields))
        self.expected_geometry = np.array(expected, dtype=np.uint64)
        self.expected = np.empty(len(offsets), dtype=np.uint64)
        self.expected[: self.n_geometry] = self.expected_geometry
        manager = view.line.env.ref_manager
        self.ref_manager = manager
        self.alignment_refs = (
            set()
            if manager is None
            else {
                getattr(view.line.env.ref[a['element']], field)
                for a in self.attachments
                for field in _attachment_fields(a)
            }
        )
        _compile(self.context)
        array = lambda value: to_context(self.context, value)
        self.offsets = array(np.array(offsets, dtype=np.int64))
        self.sizes = array(np.array(sizes, dtype=np.int8))
        self.output = self.context.zeros(len(offsets), dtype=np.uint64)
        self.n_storage_words = len(offsets)

    def check(self):
        """Validate at tracking entry, making one context-to-host array read."""
        view = self.view
        line = view.line
        # Reacquire the live mapping on every entry. Hoisting this property
        # avoids rebuilding/accessing it once per occurrence without caching
        # ownership across calls or overlooking replacement of a live element.
        elements = line.element_dict
        if view.manager._revision != self.revision:
            raise ValueError('Mechanics definitions changed; prepare bindings again')
        if line.tracker._buffer is not self.buffer:
            raise ValueError('Tracker buffer changed; prepare mechanics again')
        if tuple(line.element_names) != self.names:
            raise ValueError(
                'Line sequence changed after mechanics attachment; rebuild bindings'
            )
        if any(
            elements.get(name) is not element
            for name, element in self.objects.items()
        ):
            raise ValueError(
                'Line element object changed after mechanics attachment; rebuild bindings'
            )
        if any(
            element.parent_name != name
            or elements.get(name) is not parent
            or element._parent is not parent
            for element, name, parent in self.slice_parents
        ):
            raise ValueError(
                'Native slice reference geometry changed after attachment; rebuild bindings'
            )
        from .model import _storage_identity

        if any(_storage_identity(element) != previous
               for element, previous in self.storage):
            raise ValueError(
                'Element geometry storage changed after preparation; prepare bindings again'
            )
        if view._pipes is not None:
            view._pipes.validate()
        if line.aperture.is_built and view._pipes is None:
            raise ValueError(
                'Discard/rebuild the aperture survey model when applying mechanics; pipe-body coupling is not implemented yet'
            )
        view.manager._check_aliases()
        if line.env.ref_manager is not self.ref_manager:
            raise ValueError(
                'Lattice reference manager changed; prepare mechanics again'
            )
        from .model import _reference_geometry

        if any(
            _reference_geometry(element) != previous
            for element, previous in self.python_geometry
        ):
            raise ValueError(
                'Custom element reference geometry changed after attachment; rebuild survey bindings'
            )
        if self.ref_manager is not None and not self.alignment_refs.isdisjoint(
            self.ref_manager.tasks
        ):
            raise ValueError(
                'An attached element has expression-driven alignment; numeric mechanics cannot overwrite it'
            )
        from .model import _check_alignment_consumers

        _check_alignment_consumers(self.ref_manager, self.alignment_refs)
        self.context.kernels.mechanics_guard_gather(
            n=self.n_storage_words,
            offsets=self.offsets,
            sizes=self.sizes,
            buffer=self.buffer.buffer,
            out=self.output,
        )
        current = self.context.nparray_from_context_array(self.output)
        if not np.array_equal(current[: self.n_geometry], self.expected_geometry):
            raise ValueError(
                'Line reference geometry changed after attachment; rebuild survey bindings'
            )
        alignment_rows = [a['last_written'] for a in self.attachments]
        expected_alignment = (
            np.concatenate(alignment_rows) if self.has_solenoid_axes
            else np.array(alignment_rows, dtype=np.float64)
        )
        self.expected[self.n_geometry :] = expected_alignment.ravel().view(np.uint64)
        if not np.array_equal(
            current[self.n_geometry :], self.expected[self.n_geometry :]
        ):
            raise ValueError(
                'Local alignment edited after attachment; restore/rebind explicitly'
            )
        return self
