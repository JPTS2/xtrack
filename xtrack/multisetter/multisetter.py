import numpy as np

import xobjects as xo
import xtrack as xt


class MultiSetter(xo.HybridClass):
    _xofields = {
        'offsets': xo.Int64[:],
    }

    _extra_c_sources = [
        '#include "xtrack/multisetter/multisetter.h"',
    ]

    _kernels = {
        'get_values_at_offsets_float64': xo.Kernel(
            c_name='get_values_at_offsets_float64',
            args=[
                xo.Arg(xo.ThisClass, name='data'),
                xo.Arg(xo.Int8, pointer=True, name='buffer'),
                xo.Arg(xo.Float64, pointer=True, name='out'),
            ],
        ),
        'get_values_at_offsets_int64': xo.Kernel(
            c_name='get_values_at_offsets_int64',
            args=[
                xo.Arg(xo.ThisClass, name='data'),
                xo.Arg(xo.Int8, pointer=True, name='buffer'),
                xo.Arg(xo.Int64, pointer=True, name='out'),
            ],
        ),
        'get_values_at_offsets_int32': xo.Kernel(
            c_name='get_values_at_offsets_int32',
            args=[
                xo.Arg(xo.ThisClass, name='data'),
                xo.Arg(xo.Int8, pointer=True, name='buffer'),
                xo.Arg(xo.Int32, pointer=True, name='out'),
            ],
        ),
        'get_values_at_offsets_int8': xo.Kernel(
            c_name='get_values_at_offsets_int8',
            args=[
                xo.Arg(xo.ThisClass, name='data'),
                xo.Arg(xo.Int8, pointer=True, name='buffer'),
                xo.Arg(xo.Int8, pointer=True, name='out'),
            ],
        ),
        'set_values_at_offsets_float64': xo.Kernel(
            c_name='set_values_at_offsets_float64',
            args=[
                xo.Arg(xo.ThisClass, name='data'),
                xo.Arg(xo.Int8, pointer=True, name='buffer'),
                xo.Arg(xo.Float64, pointer=True, name='input'),
            ],
        ),
        'set_values_at_offsets_int64': xo.Kernel(
            c_name='set_values_at_offsets_int64',
            args=[
                xo.Arg(xo.ThisClass, name='data'),
                xo.Arg(xo.Int8, pointer=True, name='buffer'),
                xo.Arg(xo.Int64, pointer=True, name='input'),
            ],
        ),
        'set_values_at_offsets_int32': xo.Kernel(
            c_name='set_values_at_offsets_int32',
            args=[
                xo.Arg(xo.ThisClass, name='data'),
                xo.Arg(xo.Int8, pointer=True, name='buffer'),
                xo.Arg(xo.Int32, pointer=True, name='input'),
            ],
        ),
    }

    def __init__(self, line, elements, field, index=None, dtype=np.float64,
                skip_inconsistent_type_check=False):
        """Create object to efficiently set and get values of a specific field of
        several elements of a line.

        Parameters
        ----------
        line: xtrack.Line
            Line in which the elements are mutated
        elements: list of int or strings
            List of indices or names of the elements to be mutated.
        field: str
            Name of the field to be mutated.
        index: int or None
            If the field is an array, the index of the array to be mutated.
        dtype: numpy dtype or None
            Data type of the field (one of np.float64, np.int64, np.int32,
            np.int8). Default is np.float64. If None, it is inferred from
            the first element.
        skip_inconsistent_type_check: bool
            If True, elements whose field type does not match `dtype` are
            skipped (their offset is set to -1) instead of raising an error.
            Default is False.
        """

        if isinstance(line, xt.Tracker):
            tracker = line
        else:
            tracker = line.tracker

        if tracker.iscollective:
            tracker = tracker.line._get_non_collective_line().tracker

        context = tracker._context

        tracker_buffer = tracker._buffer
        line = tracker.line

        if len(elements) == 0:
            self._empty = True
            self.xoinitialize(_context=context, offsets=[])
            return

        self._empty = False

        # Resolve canonical owners before inspecting a RawUnion or a slice
        # parent. Temporary dressing wrappers do not own TPSA handle lifetimes.
        owners = [_canonical_element(line, name) for name in elements]
        self._targets = [_resolve_target(owner, field, line) for owner in owners]
        inner_obj, inner_name = self._targets[0]
        inferred = _field_dtype(inner_obj, inner_name, index)
        self.dtype = inferred if dtype is None else dtype
        self.xodtype = {
            np.float64: xo.Float64,
            np.int64: xo.Int64,
            np.int32: xo.Int32,
            np.int8: xo.Int8,
        }[self.dtype]

        assert self.dtype in [np.float64, np.int64, np.int32, np.int8], (
            'Only float64, int64, int32, and int8 are supported for now')

        assert np.all([owner._buffer is tracker_buffer for owner in owners])
        offsets = [_extract_offset(owner, name, index, self.dtype, self.xodtype,
                    skip_inconsistent_type_check=skip_inconsistent_type_check)
                   for owner, name in self._targets]
        self._target_index = index
        self._valid_targets = [offset >= 0 for offset in offsets]
        self.xoinitialize(_context=context, offsets=offsets)
        self.compile_kernels(only_if_needed=True)

        self.offsets = context.nparray_to_context_array(np.array(offsets))
        self.tracker = tracker
        self._tracker_buffer = tracker_buffer

        self._get_kernel = {
            np.float64: self._context.kernels.get_values_at_offsets_float64,
            np.int64: self._context.kernels.get_values_at_offsets_int64,
            np.int32: self._context.kernels.get_values_at_offsets_int32,
            np.int8: self._context.kernels.get_values_at_offsets_int8,
        }[self.dtype]

        self._set_kernel = {
            np.float64: self._context.kernels.set_values_at_offsets_float64,
            np.int64: self._context.kernels.set_values_at_offsets_int64,
            np.int32: self._context.kernels.set_values_at_offsets_int32,
        }.get(self.dtype)

    def get_values(self):
        """Get the values of the multisetter fields."""
        if self._empty:
            return self._context.zeros(0, dtype=np.float64)

        if self._needs_tpsa_access():
            values = np.zeros(len(self._targets), dtype=self.dtype)
            for index, ((owner, name), valid) in enumerate(
                    zip(self._targets, self._valid_targets)):
                if valid:
                    value = _read_target(owner, name)
                    if self._target_index is not None:
                        value = value[self._target_index]
                    values[index] = getattr(value, 'const_part', value)
            return self._context.nparray_to_context_array(values)

        out = self._context.zeros(len(self.offsets), dtype=self.dtype)
        self._get_kernel.set_n_threads(len(self.offsets))
        self._get_kernel(data=self, buffer=self._tracker_buffer.buffer, out=out)
        return out

    def set_values(self, values):
        """Set the values of the multisetter fields.

        Parameters
        ----------
        values: np.ndarray
            Array of values to be set.
        """
        if self._empty:
            return
        if self.dtype is np.int8:
            raise NotImplementedError("MultiSetter does not support setting int8 fields")

        if self._needs_tpsa_access():
            values = self._context.nparray_from_context_array(
                xt.BeamElement._arr2ctx(self, values))
            if len(values) != len(self._targets):
                raise ValueError('one value is required per MultiSetter target')
            for value, (owner, name), valid in zip(
                    values, self._targets, self._valid_targets):
                if valid:
                    backing_name = (name if name in getattr(
                        owner, '_float_or_tpsa_array_fields', ())
                        else getattr(owner, '_rename', {}).get(name, name))
                    if self._target_index is None:
                        setattr(owner, backing_name, value)
                    else:
                        getattr(owner, backing_name)[self._target_index] = value
            return

        self._set_kernel.set_n_threads(len(self.offsets))
        self._set_kernel(data=self, buffer=self._tracker_buffer.buffer,
               input=xt.BeamElement._arr2ctx(self, values))


    def _needs_tpsa_access(self):
        # Recheck on every call: an attr cache may predate parameter promotion.
        return any(getattr(owner._xobject, '_tpsa_enabled', 0)
                   and (name in getattr(owner, '_float_or_tpsa_fields', ())
                        or (self._target_index is not None and name in getattr(
                            owner, '_float_or_tpsa_array_fields', ())))
                   for owner, name in self._targets)


def _read_target(owner, name):
    if name in getattr(owner, '_float_or_tpsa_array_fields', ()):
        return getattr(owner, name)
    return getattr(owner, getattr(owner, '_rename', {}).get(name, name))


def _canonical_element(line, name):
    if isinstance(name, (int, np.integer)):
        name = line.element_names[name]
    owner = line._element_dict[name]
    if isinstance(owner, xt.Replica):
        owner = owner.resolve(line)
    return owner


def _resolve_target(owner, field_name, line=None):
    path = list(field_name) if isinstance(field_name, (list, tuple)) else [field_name]
    for name in path[:-1]:
        if name == '_parent' and line is not None and hasattr(owner, 'parent_name'):
            owner = _canonical_element(line, owner.parent_name)
        else:
            owner = getattr(owner, name)
    name = path[-1]
    # Offset access uses xofield names; descriptor access uses renamed backing
    # names. Accept either spelling, without invoking a derived public getter.
    inverse = {value: key for key, value in getattr(owner, '_rename', {}).items()}
    return owner, inverse.get(name, name)


def _field_dtype(owner, name, index):
    if name in getattr(owner, '_float_or_tpsa_fields', ()):
        return np.float64 if index is None else None
    ftype = owner._xofields.get(name)
    # Metaclass-added alignment/control entries may already be Field objects.
    ftype = getattr(ftype, 'ftype', ftype)
    if index is not None:
        ftype = getattr(ftype, '_itemtype', None)
    return {xo.Float64: np.float64, xo.Int64: np.int64,
            xo.Int32: np.int32, xo.Int8: np.int8}.get(ftype)


def _extract_offset(obj, field_name, index, dtype, xodtype, skip_inconsistent_type_check=False):
    owner, name = _resolve_target(obj, field_name)
    inconsistent = _field_dtype(owner, name, index) is not dtype
    if inconsistent:
        if skip_inconsistent_type_check:
            return -1
        raise AssertionError('Inconsistent types')
    if index is None:
        return owner._xobject._get_offset(name)
    return getattr(owner._xobject, name)._get_offset(index)
