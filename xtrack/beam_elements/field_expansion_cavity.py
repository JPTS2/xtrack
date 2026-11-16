# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

from math import isfinite

import numpy as np
import xobjects as xo
import xtrack as xt

from ..base_element import BeamElement, FloatOrTpsa
from ._common import _NOEXPR_FIELDS
from .cavity import _CavityControls, _RADIAL_ORDER, _constant_part

# Jets 0 to 11 per table row, as CAVITY_MODEL_TABLE_JETS in cavity_models.h.
_FIELD_TABLE_JETS = 12


class FieldExpansionCavity(_CavityControls, BeamElement):

    __doc__ = '''RF cavity whose on-axis field is a sum of space harmonics.

    Over the element, the on-axis longitudinal field is
    E_z(z, t) = Re[G(z) exp(-i omega t)], G(z) = sum_m c_m exp(i k_m z), with z
    from the centre. The near-axis field follows from Maxwell's equations, and
    the element ends in hard edges. A Cavity of `n_cells` and counter-wave
    amplitude r is the case k = [k_z, -k_z], c = [1, r exp(-i counter_wave_phase)],
    k_z = pi n_cells / length. It is tracked by the experimental
    "_phase-resolved" model.

    Parameters
    ----------
    length, voltage, frequency, harmonic, phase, lag, absolute_time
        As for Cavity: `voltage` is the crest energy gain at `calibration_beta`
        and `phase` the phase of that gain, so the scale of the amplitudes is free.
    calibration_beta : float
        Fixed speed, 0 < beta <= 1, at which `voltage` is the crest energy gain:
        a field map's design speed. Default is 1. A Cavity's is the synchronous
        speed of its cells.
    wave_numbers : array of float
        Wave numbers k_m of the space harmonics, in rad/m; positive ones
        travel with the beam.
    amplitudes : array of complex
        Their relative amplitudes c_m, as many as there are wave numbers.
    integrator, num_kicks, radial_order
        As for Cavity's "_phase-resolved" model.
    field_table_points : int or None
        The field and its derivatives are tabulated over the element at this many
        points and interpolated while tracking, which makes the cost per step
        independent of the number of harmonics; quintic Hermite interpolation of
        each derivative matches the summed harmonics to about 1e-9. None
        (default) takes 20 points per shortest harmonic wavelength, at least 64;
        0 sums the harmonics at every evaluation instead. The table follows `amplitudes`; after a change
        of `length` or of `wave_numbers` the harmonics are summed directly.
    '''

    isthick = True
    has_backtrack = True
    allow_loss_refinement = True

    _rf_plan = 'track_cavity_plan'
    _xofields = {
        'length': xo.Float64,
        'voltage': FloatOrTpsa,
        'frequency': FloatOrTpsa,
        'lag': FloatOrTpsa,
        'phase': FloatOrTpsa,
        'harmonic': FloatOrTpsa,
        'lag_taper': FloatOrTpsa,
        'phase_taper': FloatOrTpsa,
        'calibration_beta': FloatOrTpsa,
        'wave_numbers': xo.Float64[:],
        'amplitudes_re': xo.Float64[:],
        'amplitudes_im': xo.Float64[:],
        'field_table': xo.Float64[:],
        'field_table_z0': xo.Float64,
        'field_table_dz': xo.Float64,
        'field_table_key': xo.Float64,
        'radial_order': xo.Field(xo.Int64, default=_RADIAL_ORDER),
        'absolute_time': xo.Int64,
        'num_kicks': xo.Int64,
        'model': xo.Int64,
        'integrator': xo.Int64,
    }

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/field_expansion_cavity.h"',
    ]

    _rename = {
        'model': '_model',
        'integrator': '_integrator',
        'frequency': '_frequency',
        'harmonic': '_harmonic',
        'lag': '_lag',
    }

    # The table is derived: it is rebuilt from its size on loading.
    _skip_in_to_dict = ['field_table', 'field_table_z0', 'field_table_dz', 'field_table_key']
    _store_in_to_dict = ['field_table_points']

    _default_integrator = 'adaptive'
    _default_calibration_beta = 1.
    _default_frequency = 0.0
    _default_harmonic = 0.0

    _noexpr_fields = _NOEXPR_FIELDS

    def __init__(self, **kwargs):
        if '_xobject' in kwargs and kwargs['_xobject'] is not None:
            self.xoinitialize(**kwargs)
            return
        model = kwargs.pop('model', '_phase-resolved')
        integrator = kwargs.pop('integrator', None)
        frequency = kwargs.pop('frequency', None)
        harmonic = kwargs.pop('harmonic', None)
        lag = kwargs.pop('lag', None)
        wave_numbers = np.atleast_1d(np.asarray(kwargs.pop('wave_numbers'), dtype=float))
        amplitudes = kwargs.pop('amplitudes', None)
        if amplitudes is None:
            amplitudes = (np.asarray(kwargs.pop('amplitudes_re'), dtype=float)
                          + 1j * np.asarray(kwargs.pop('amplitudes_im'), dtype=float))
        amplitudes = np.atleast_1d(np.asarray(amplitudes, dtype=complex))
        if wave_numbers.ndim != 1 or amplitudes.shape != wave_numbers.shape or not len(wave_numbers):
            raise ValueError('wave_numbers and amplitudes must be equal-length, '
                             'non-empty one-dimensional arrays')
        points = kwargs.pop('field_table_points', None)
        length = float(kwargs.get('length', 0.))
        if points is None:
            shortest = 2 * np.pi / max(np.max(np.abs(wave_numbers)), 1e-300)
            points = max(64, int(np.ceil(20 * length / shortest)) + 1) if length > 0. else 0
        if points == 1 or points < 0:
            raise ValueError('field_table_points must be 0 or at least 2')
        kwargs.setdefault('calibration_beta', self._default_calibration_beta)
        kwargs['wave_numbers'] = wave_numbers
        kwargs['amplitudes_re'] = amplitudes.real.copy()
        kwargs['amplitudes_im'] = amplitudes.imag.copy()
        kwargs['field_table'] = 2 * _FIELD_TABLE_JETS * points
        self.xoinitialize(**kwargs)
        self._build_field_table()
        self.model = model
        if integrator is not None:
            self.integrator = integrator
        if frequency is not None:
            self.frequency = frequency
        if harmonic is not None:
            self.harmonic = harmonic
        if lag is not None:
            self.lag = lag
        self._check_physical_model()

    @classmethod
    def from_fourier_series(cls, cos_coefficients, sin_coefficients=None, period=None,
                            **kwargs):
        """A standing wave whose on-axis profile is the Fourier series
        a_0/2 + sum_n (a_n cos(2 pi n z / P) + b_n sin(2 pi n z / P)), z from the
        centre, of period P, by default the element's `length`."""
        a = np.asarray(cos_coefficients, dtype=float)
        b = np.zeros_like(a)
        if sin_coefficients is not None:
            b[:len(sin_coefficients)] = sin_coefficients
        k = 2 * np.pi * np.arange(1, len(a)) / (kwargs['length'] if period is None else period)
        return cls(wave_numbers=np.concatenate([[0.], k, -k]),
                   amplitudes=np.concatenate([[a[0] / 2], (a[1:] - 1j * b[1:]) / 2,
                                              (a[1:] + 1j * b[1:]) / 2]), **kwargs)

    @classmethod
    def from_samples(cls, values, num_harmonics=None, **kwargs):
        """The space harmonics of a field sampled at equal steps over the element,
        both ends included: real values are a standing wave's profile, complex
        ones the phasor G. The field is read as periodic over the element, so it
        should vanish, or match, at both ends. The truncated series is corrected
        to take the samples' end value exactly: otherwise it misses it by O(1/N),
        and the hard-edge faces kick on the difference."""
        values = np.asarray(values)
        if values.ndim != 1 or len(values) < 3:
            raise ValueError('values must be at least three samples')
        if num_harmonics is None:
            num_harmonics = (len(values) - 1) // 2
        length = kwargs['length']
        z = np.linspace(-length / 2, length / 2, len(values))
        m = np.arange(-num_harmonics, num_harmonics + 1)
        k = 2 * np.pi * m / length
        from ..footprint import _trapezoid
        amplitudes = _trapezoid(np, values[None, :] * np.exp(-1j * k[:, None] * z[None, :]),
                                z, axis=1) / length
        # Both ends are one point of the period, where harmonic m is (-1)^m: the
        # least change meeting the end value is alternating and equal in size.
        sign = (-1.) ** m
        miss = sign @ amplitudes - (values[0] + values[-1]) / 2
        amplitudes = amplitudes - miss * sign / len(m)
        return cls(wave_numbers=k, amplitudes=amplitudes, **kwargs)

    def _check_physical_model(self, tracking=False):
        """Raise for settings the phase-resolved model cannot track."""
        self._check_radial_order()
        if self._model != 10:
            raise ValueError('FieldExpansionCavity is tracked by "_phase-resolved" only')
        if self.integrator == 'teapot':
            raise ValueError('_phase-resolved has no kicks to place: "teapot" '
                             'does not apply')
        if self.num_kicks < 0:
            raise ValueError('num_kicks must be non-negative')
        beta = _constant_part(self.calibration_beta)
        if not (isfinite(beta) and 0. < beta <= 1.):
            raise ValueError('calibration_beta must be in (0, 1]')
        if tracking and not (isfinite(_constant_part(self.length))
                             and _constant_part(self.length) > 0.):
            raise ValueError(f'{self.model} requires a positive finite length')
        if not (np.all(np.isfinite(self.wave_numbers))
                and np.all(np.isfinite(self.amplitudes))
                and np.any(self.amplitudes != 0.)):
            raise ValueError('The space harmonics must be finite and not all zero')

    @property
    def amplitudes(self):
        context = self._context
        return (context.nparray_from_context_array(self.amplitudes_re)
                + 1j * context.nparray_from_context_array(self.amplitudes_im))

    @amplitudes.setter
    def amplitudes(self, value):
        value = np.asarray(value, dtype=complex)
        if value.shape != (len(self.wave_numbers),):
            raise ValueError('amplitudes must have one value per wave number')
        self.amplitudes_re[:] = value.real
        self.amplitudes_im[:] = value.imag
        self._build_field_table()

    def _build_field_table(self):
        """Tabulate jets 0 to 11 of the unit-amplitude field over the element."""
        points = self.field_table_points
        if points < 2:
            return
        context = self._context
        wave_numbers = context.nparray_from_context_array(self.wave_numbers)
        amplitudes = self.amplitudes
        z = np.linspace(-self.length / 2, self.length / 2, points)
        table = np.empty((points, 2 * _FIELD_TABLE_JETS))
        for start in range(0, points, 4096):
            phasors = amplitudes * np.exp(1j * np.outer(z[start:start + 4096], wave_numbers))
            for order in range(_FIELD_TABLE_JETS):
                jet = (phasors * (1j * wave_numbers)**order).sum(axis=1)
                table[start:start + 4096, order] = jet.real
                table[start:start + 4096, _FIELD_TABLE_JETS + order] = jet.imag
        self.field_table[:] = context.nparray_to_context_array(table.ravel())
        self.field_table_z0 = -self.length / 2
        self.field_table_dz = self.length / (points - 1)
        # Matched in track_cavity_single_particle: the table applies only while
        # the harmonics and length are those it was built from.
        key = 11. * self.length
        for m, (k, a) in enumerate(zip(wave_numbers, amplitudes)):
            key += (m + 1) * (k + 3. * a.real + 7. * a.imag)
        self.field_table_key = key

    @property
    def field_table_points(self):
        return len(self.field_table) // (2 * _FIELD_TABLE_JETS)

    @property
    def _thin_slice_class(self):
        return xt.ThinSliceFieldExpansionCavity

    @property
    def _thick_slice_class(self):
        return xt.ThickSliceFieldExpansionCavity

    @property
    def _drift_slice_class(self):
        return xt.DriftSliceFieldExpansionCavity
