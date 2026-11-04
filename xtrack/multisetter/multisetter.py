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

        # Get dtype from first element
        el = line[elements[0]]
        if isinstance(field, (list, tuple)):
            inner_obj = el
            inner_name = field[-1]
            for ff in field[:-1]:
                inner_obj = getattr(inner_obj, ff)
        else:
            inner_obj = el
            inner_name = field
        if dtype is None and inner_name in getattr(inner_obj, '_float_or_tpsa_fields', ()):
            dtype = np.float64
        if dtype is None:
            dd = getattr(inner_obj.copy(_context=xo.context_default)._xobject, inner_name)
            dtype = type(dd if index is None else dd[index])
        self.dtype = dtype
        self.xodtype = {
            np.float64: xo.Float64,
            np.int64: xo.Int64,
            np.int32: xo.Int32,
            np.int8: xo.Int8,
        }[self.dtype]

        assert self.dtype in [np.float64, np.int64, np.int32, np.int8], (
            'Only float64, int64, int32, and int8 are supported for now')

        assert np.all([line.get(nn)._buffer is tracker_buffer for nn in elements])
        offsets = [_extract_offset(line.get(nn), field, index, self.dtype, self.xodtype,
                                   skip_inconsistent_type_check=skip_inconsistent_type_check)
                   for nn in elements]

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

        # A promoted field holds a pointer, so raw offsets cannot be used on it. Its flag is
        # read by the same kernels, so fields that cannot hold a series check nothing.
        self._targets, self._flags = [], None
        if index is None and isinstance(context, xo.ContextCpu):
            path = list(field) if isinstance(field, (list, tuple)) else [field]
            flagged = []
            for nn, offset in zip(elements, offsets):
                owner = line.get(nn)
                for ff in path[:-1]:
                    # A slice's `_parent` is a temporary wrapper: use the element that owns the series.
                    owner = line[owner.parent_name] if ff == '_parent' else getattr(owner, ff)
                self._targets.append((owner, owner._rename.get(path[-1], path[-1]), offset >= 0))
                if path[-1] in getattr(owner, '_float_or_tpsa_fields', ()):
                    flagged.append(nn)
            if flagged:
                self._flags = MultiSetter(tracker, flagged, [*path[:-1], '_tpsa_enabled'], dtype=np.int8,
                                          skip_inconsistent_type_check=True)

    def get_values(self):
        """Get the values of the multisetter fields."""
        if self._empty:
            return self._context.zeros(0, dtype=np.float64)

        if self._flags is not None and self._flags.get_values().any():
            return np.array([getattr(getattr(owner, name), 'const_part', getattr(owner, name)) if valid else 0.
                             for owner, name, valid in self._targets], dtype=self.dtype)

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

        if self._flags is not None and self._flags.get_values().any():
            for (owner, name, valid), value in zip(self._targets, np.broadcast_to(values, len(self._targets))):
                if valid:
                    setattr(owner, name, value)
            return

        self._set_kernel.set_n_threads(len(self.offsets))
        self._set_kernel(data=self, buffer=self._tracker_buffer.buffer,
               input=xt.BeamElement._arr2ctx(self, values))


def _extract_offset(obj, field_name, index, dtype, xodtype, skip_inconsistent_type_check=False):

    if isinstance(field_name, (list, tuple)):
        inner_obj = obj
        inner_name = field_name[-1]
        for ff in field_name[:-1]:
            inner_obj = getattr(inner_obj, ff)
    else:
        inner_obj = obj
        inner_name = field_name

    if index is None and dtype is np.float64 and inner_name in getattr(inner_obj, '_float_or_tpsa_fields', ()):
        return inner_obj._xobject._get_offset(inner_name)

    if index is None:
        inconsistent_type = not isinstance(getattr(inner_obj._xobject, inner_name), dtype)
        if inconsistent_type:
            if skip_inconsistent_type_check:
                return -1
            else:
                assert not inconsistent_type, "Inconsistent types"
        return inner_obj._xobject._get_offset(inner_name)
    else:
        obj = getattr(inner_obj._xobject, inner_name)
        inconsistent_type = not hasattr(obj, "_itemtype") or obj._itemtype is not xodtype
        if inconsistent_type:
            if skip_inconsistent_type_check:
                return -1
            else:
                assert not inconsistent_type, "Inconsistent types"
        return obj._get_offset(index)
