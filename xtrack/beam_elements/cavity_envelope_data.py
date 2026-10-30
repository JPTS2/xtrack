"""Numeric fixed Gaussian RF scalar tables for native cavity transport.

This module has no dependency on a preparation/reference implementation.
Coefficients define one scalar, not independently fitted force components.
The five RF powers are (s,f,s²,sf,f²), and invariants are
(1,X²+Y²,XU+YV,U²+V²). Arrays are C ordered: power,mode,invariant,E,Z.
Changing coefficient-defining hardware requires preparing a new table.
Interpolation domains are not physical admission/error certificates.
"""
import numpy as np
import xobjects as xo


class EnvelopeTableData(xo.HybridClass):
    """Portable numeric payload; signed Fourier modes retain their input order.

    E is actual total energy/(actual particle mass*c²); Z is dimensionless.
    Body coefficients exclude the free transverse-quadratic Hamiltonian.
    Boundary coefficients define the same globally anchored W at both planes.
    Public coefficient access returns a copy: prepare/attach a replacement to
    change hardware. Low-level xobject writes are outside this fixed-data API.
    """
    _xofields = {
        'energy_min': xo.Float64, 'energy_max': xo.Float64,
        'position_min': xo.Float64, 'position_max': xo.Float64,
        'nu': xo.Float64, 'anchor': xo.Float64,
        'profile_width': xo.Float64, 'profile_centre': xo.Float64,
        'normalization_wave_number': xo.Float64,
        'normalization_frequency': xo.Float64,
        'n_modes': xo.Int64, 'n_energy': xo.Int64, 'n_position': xo.Int64,
        'modes_m': xo.Int64[:], 'modes_n': xo.Int64[:],
        'body_real': xo.Float64[:], 'body_imag': xo.Float64[:],
        'boundary_real': xo.Float64[:], 'boundary_imag': xo.Float64[:],
    }
    _immutable_arrays = frozenset(('modes_m', 'modes_n', 'body_real', 'body_imag',
                                  'boundary_real', 'boundary_imag'))

    def __getattribute__(self, name):
        result = super().__getattribute__(name)
        if name in type(self)._immutable_arrays:
            context = super().__getattribute__('_context')
            result = context.nparray_from_context_array(result).copy()
            result.flags.writeable = False
        return result

    def __setattr__(self, name, value):
        if (name in type(self)._xofields
                and self.__dict__.get('_data_locked', False)):
            raise AttributeError('prepared table data are fixed; attach a replacement')
        super().__setattr__(name, value)

    def __init__(self, *, energy_interval=None, position_interval=None,
                 nu=None, anchor=None, profile_width=None, profile_centre=0.,
                 normalization_wave_number=0.,
                 normalization_frequency=0.,
                 modes=None, body_coefficients=None, boundary_coefficients=None,
                 _xobject=None, **kwargs):
        if _xobject is not None:
            self.xoinitialize(_xobject=_xobject, **kwargs)
            self._data_locked = True
            return
        def interval(value, name):
            array = np.asarray(value)
            if (np.iscomplexobj(array) or array.shape != (2,)
                    or not np.all(np.isfinite(array)) or array[0] >= array[1]):
                raise ValueError(name+' must be two finite increasing real values')
            return tuple(map(float, array))
        elo, ehi = interval(energy_interval, 'energy_interval')
        zlo, zhi = interval(position_interval, 'position_interval')
        if elo <= 1.:
            raise ValueError('energy_interval must lie strictly above rest')
        hardware = {}
        for name, value in dict(nu=nu, anchor=anchor,
                profile_width=profile_width, profile_centre=profile_centre).items():
            array = np.asarray(value)
            if np.iscomplexobj(array) or array.shape or not np.isfinite(array):
                raise ValueError(name+' must be a finite real scalar')
            hardware[name] = float(array)
        if hardware['nu'] <= 0 or hardware['profile_width'] <= 0:
            raise ValueError('nu and profile_width must be positive')
        if not zlo <= hardware['anchor'] <= zhi:
            raise ValueError('anchor must lie inside the position interval')
        for name, value in [('normalization_wave_number', normalization_wave_number),
                            ('normalization_frequency', normalization_frequency)]:
            scalar = np.asarray(value)
            if (np.iscomplexobj(scalar) or scalar.shape
                    or not np.isfinite(scalar) or scalar < 0):
                raise ValueError(name+' must be finite and nonnegative')
            hardware[name] = float(scalar)
        frequency = hardware['normalization_frequency']
        k = hardware['normalization_wave_number']
        if bool(frequency) != bool(k):
            raise ValueError('frequency and coordinate wave number must be bound together')
        if frequency and not np.isclose(k, 2*np.pi*frequency/299792458.,
                                       rtol=8*np.finfo(float).eps, atol=0.):
            raise ValueError('coordinate wave number must equal 2*pi*frequency/c')
        modes = np.asarray(modes)
        if (modes.ndim != 2 or modes.shape[1] != 2 or not len(modes)
                or not np.issubdtype(modes.dtype, np.integer)):
            raise ValueError('modes must be a nonempty integer (n,2) array')
        pairs = list(map(tuple, modes))
        if len(set(pairs)) != len(pairs):
            raise ValueError('Fourier modes must be unique')
        try:
            opposite = [pairs.index((-m, -n)) for m, n in pairs]
        except ValueError as error:
            raise ValueError('every mode needs its conjugate opposite') from error
        arrays = {}
        shape = None
        for name, value in [('body', body_coefficients),
                            ('boundary', boundary_coefficients)]:
            value = np.asarray(value, dtype=np.complex128)
            if (value.ndim != 5 or value.shape[:3] != (5, len(modes), 4)
                    or min(value.shape[-2:]) < 1 or not np.all(np.isfinite(value))):
                raise ValueError(name+' coefficients have invalid shape/data')
            tolerance = 128*np.finfo(float).eps*np.maximum(
                1., np.maximum(abs(value), abs(value[:, opposite])))
            if np.any(abs(value-np.conj(value[:, opposite])) > tolerance):
                raise ValueError(name+' coefficients violate Fourier conjugacy')
            if shape is not None and value.shape != shape:
                raise ValueError('body and boundary must share polynomial degrees')
            shape = value.shape
            arrays[name+'_real'] = np.ascontiguousarray(value.real).ravel()
            arrays[name+'_imag'] = np.ascontiguousarray(value.imag).ravel()
        self.xoinitialize(energy_min=elo, energy_max=ehi,
            position_min=zlo, position_max=zhi, **hardware,
            n_modes=len(modes), n_energy=shape[-2], n_position=shape[-1],
            modes_m=modes[:, 0], modes_n=modes[:, 1], **arrays, **kwargs)
        self._data_locked = True

    @classmethod
    def from_table(cls, table, **kwargs):
        """Copy a prepared owner's numeric data without importing that owner."""
        for name in ('normalization_wave_number', 'normalization_frequency'):
            if hasattr(table, name):
                kwargs.setdefault(name, getattr(table, name))
        return cls(**{name: getattr(table, name) for name in (
            'energy_interval', 'position_interval', 'nu', 'anchor',
            'profile_width', 'profile_centre', 'modes',
            'body_coefficients', 'boundary_coefficients')}, **kwargs)

    @property
    def energy_interval(self):
        return (self.energy_min, self.energy_max)

    @property
    def position_interval(self):
        return (self.position_min, self.position_max)

    @property
    def modes(self):
        return np.column_stack((self.modes_m, self.modes_n))

    def _coefficients(self, name):
        # Assign components separately: real+1j*imag can change signed zeros.
        result = np.empty((5, int(self.n_modes), 4, int(self.n_energy),
                           int(self.n_position)), dtype=np.complex128)
        result.real = np.asarray(getattr(self, name+'_real')).reshape(result.shape)
        result.imag = np.asarray(getattr(self, name+'_imag')).reshape(result.shape)
        return result

    @property
    def body_coefficients(self):
        return self._coefficients('body')

    @property
    def boundary_coefficients(self):
        return self._coefficients('boundary')

    def to_dict(self, **kwargs):
        result = {name: getattr(self, name) for name in (
            'energy_interval', 'position_interval', 'nu', 'anchor',
            'profile_width', 'profile_centre')}
        result['normalization_wave_number'] = self.normalization_wave_number
        result['normalization_frequency'] = self.normalization_frequency
        result['modes'] = self.modes.tolist()
        for name in ('body', 'boundary'):
            result[name+'_real'] = np.asarray(getattr(self, name+'_real')).tolist()
            result[name+'_imag'] = np.asarray(getattr(self, name+'_imag')).tolist()
        result['coefficient_shape'] = [5, int(self.n_modes), 4,
                                      int(self.n_energy), int(self.n_position)]
        result['__class__'] = type(self).__name__
        return result

    @classmethod
    def from_dict(cls, data, **kwargs):
        data = data.copy()
        data.pop('__class__', None)
        shape = tuple(data.pop('coefficient_shape'))
        for name in ('body', 'boundary'):
            value = np.empty(shape, dtype=np.complex128)
            value.real = np.asarray(data.pop(name+'_real')).reshape(shape)
            value.imag = np.asarray(data.pop(name+'_imag')).reshape(shape)
            data[name+'_coefficients'] = value
        return cls(**data, **kwargs)
