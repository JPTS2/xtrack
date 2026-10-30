"""Indexed TPSA strengths with ordinary Float64 storage retained for kernels.

Only the owning Cavity keeps series alive. UInt64 arrays contain process-local
handles, never a serialized representation. Explicit NumPy conversion of active
series requires dtype=object; it must not silently discard parameter dependence.
"""
from __future__ import annotations

import numpy as np
from madng_tpsa import Tpsa
from madng_tpsa._cffi import ffi


class CavityArrayView(np.lib.mixins.NDArrayOperatorsMixin):
    __array_priority__ = 1000

    def __init__(self, owner, backing_name, handle_name):
        self.owner = owner
        self.backing_name = backing_name
        self.handle_name = handle_name

    @property
    def _backing(self):
        return getattr(self.owner, self.backing_name)

    @property
    def _handles(self):
        return getattr(self.owner, self.handle_name)

    @property
    def _series(self):
        return self.owner.__dict__.get('_tpsa_array_handles', {}).get(
            self.backing_name, {})

    def __len__(self):
        return len(self._backing)

    @property
    def shape(self):
        return (len(self),)

    @property
    def ndim(self):
        return 1

    @property
    def size(self):
        return len(self)

    @property
    def dtype(self):
        return np.dtype(object if self._series else np.float64)

    def __iter__(self):
        return (self[index] for index in range(len(self)))

    def _index(self, index):
        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError('cavity strength index out of bounds')
        return index

    def __getitem__(self, index):
        if np.isscalar(index):
            index = self._index(index)
            return self._series.get(index, self._backing[index])
        indices = np.arange(len(self))[index]
        values = [self[int(item)] for item in np.asarray(indices).flat]
        dtype = object if any(isinstance(item, Tpsa) for item in values) else np.float64
        return np.array(values, dtype=dtype).reshape(np.shape(indices))

    def _set_one(self, index, value):
        index = self._index(index)
        if isinstance(value, Tpsa):
            if getattr(self.owner._xobject, '_tpsa_enabled', 0):
                descriptor = self.owner.__dict__.get('_tpsa_descriptor')
                if descriptor.ptr != value.descriptor.ptr:
                    raise ValueError('cavity strengths require one TPSA descriptor')
            else:
                self.owner.enable_tpsa(value.descriptor)
            self._backing[index] = float(value.const_part)
            handles = self.owner.__dict__.setdefault('_tpsa_array_handles', {})
            handles.setdefault(self.backing_name, {})[index] = value
            self._handles[index] = int(ffi.cast('uintptr_t', value.ptr))
        else:
            self._backing[index] = float(value)
            self._handles[index] = 0
            self._series.pop(index, None)

    def __setitem__(self, index, values):
        if np.isscalar(index):
            self._set_one(index, values)
            return
        indices = np.asarray(np.arange(len(self))[index])
        if isinstance(values, Tpsa) or np.isscalar(values):
            for item in indices.flat:
                self._set_one(item, values)
        else:
            # Object dtype preserves series until the actual indexed setter.
            values = np.broadcast_to(np.asarray(values, dtype=object), indices.shape)
            for item, value in zip(indices.flat, values.flat):
                self._set_one(item, value)

    def _array(self, dtype=None):
        if self._series:
            return np.array(list(self), dtype=object if dtype is None else dtype)
        array = self.owner._context.nparray_from_context_array(self._backing)
        return np.asarray(array, dtype=dtype)

    def __array__(self, dtype=None, copy=None):
        if self._series and (dtype is None or np.dtype(dtype) != np.dtype(object)):
            raise TypeError('active cavity strengths require NumPy dtype=object')
        result = self._array(dtype=dtype)
        return result.copy() if copy else result

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        owner = inputs[0] if inputs and isinstance(inputs[0], CavityArrayView) else None
        inputs = tuple(item._array() if isinstance(item, CavityArrayView) else item
                       for item in inputs)
        if method == 'at':
            getattr(ufunc, method)(*inputs, **kwargs)
            if owner is not None:
                owner[:] = inputs[0]
            return None
        outputs = kwargs.pop('out', None)
        result = getattr(ufunc, method)(*inputs, **kwargs)
        if outputs is None:
            return result
        results = result if isinstance(result, tuple) else (result,)
        for target, value in zip(outputs, results):
            target[...] = value
        return outputs[0] if len(outputs) == 1 else outputs

    def fill(self, value):
        self[:] = value

    def put(self, indices, values, mode='raise'):
        array = self._array().copy()
        array.put(indices, values, mode=mode)
        self[:] = array

    def copy(self):
        return self._array().copy()

    def tolist(self):
        return list(self)

    def to_nparray(self):
        return self._array()

    def __getattr__(self, name):
        if self._series and name in {
                'data', 'ctypes', 'view', 'flat', 'base', 'itemset', 'byteswap',
                'resize', 'setfield', 'sort', 'partition', 'setflags'}:
            raise TypeError(f'{name} is unsupported for active cavity strengths')
        if name.startswith('__array'):
            # Let NumPy use our explicit conversion contract, rather than an
            # interface pointer belonging to a temporary object array.
            raise AttributeError(name)
        return getattr(self._array(), name)

    def __repr__(self):
        return repr(self._array())


def clear_cavity_arrays(owner):
    """Release process-local handles after refreshing their scalar mirrors."""
    arrays = owner.__dict__.get('_tpsa_array_handles', {})
    for name in getattr(owner, '_float_or_tpsa_array_fields', ()):
        backing_name = owner._rename.get(name, name)
        for index, value in arrays.get(backing_name, {}).items():
            getattr(owner, backing_name)[index] = float(value.const_part)
        getattr(owner, owner._rename.get(name+'_tpsa', name+'_tpsa'))[:] = 0
    owner.__dict__['_tpsa_array_handles'] = {}
