# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

from ..base_element import BeamElement, FloatOrTpsa
from ..general import DEPRECATION_INFO_PREP_1_0
from warnings import warn
from math import isfinite
import numpy as np
from scipy.constants import c as clight
import xobjects as xo
import xtrack as xt
from ._common import (
    _HasIntegrator,
    _HasModelRF,
    _HasKnlKsl,
    _INDEX_TO_MODEL_RF,
    _INDEX_TO_INTEGRATOR,
    _NOEXPR_FIELDS,
    _docstring_general_notes,
    _for_docstring_alignment,
)

# The physical models and their integrator are experimental, so their names start
# with an underscore. "adaptive" is the default and stays the legacy map.
_CAVITY_MODELS = dict(_INDEX_TO_MODEL_RF)
_CAVITY_MODELS.update({10: '_phase-resolved', 11: '_phase-averaged',
                       12: '_phase-resolved-paraxial', 13: '_phase-resolved-frozen'})
_CAVITY_MODEL_IDS = {value: key for key, value in _CAVITY_MODELS.items()}
_CAVITY_INTEGRATORS = dict(_INDEX_TO_INTEGRATOR)
_CAVITY_INTEGRATORS[7] = '_geometric'
_CAVITY_INTEGRATOR_IDS = {
    value: key for key, value in _CAVITY_INTEGRATORS.items()}


def _constant_part(value):
    """Inspect a knob's base point without stripping its stored dependence."""
    return getattr(value, 'const_part', value)


_NONZERO_DEFAULTS = {'edge_entry_active': 1, 'edge_exit_active': 1,
                     'radial_order': 4}
_RADIAL_ORDERS = (2, 4, 6, 8)
# Jets 0 to 11 per table row, as CAVITY_MODEL_TABLE_JETS in cavity_models.h.
_FIELD_TABLE_JETS = 12


def _default_counter_wave_amplitude(model):
    return 0. if model == '_phase-resolved-frozen' else 1.


class _CavityControls:
    """Model, integrator and RF clock controls shared by the cavity elements."""

    @property
    def model(self):
        return _CAVITY_MODELS[self._model]

    @model.setter
    def model(self, value):
        try:
            self._model = _CAVITY_MODEL_IDS[value]
        except KeyError:
            raise ValueError(f'Invalid Cavity model: {value}') from None

    @staticmethod
    def get_available_models():
        return list(_CAVITY_MODEL_IDS)

    @property
    def integrator(self):
        return _CAVITY_INTEGRATORS[self._integrator]

    @integrator.setter
    def integrator(self, value):
        if value == 'yoshida4':
            # Preserve the shared mixin's deprecated historical spelling.
            _HasIntegrator.integrator.fset(self, value)
            return
        try:
            self._integrator = _CAVITY_INTEGRATOR_IDS[value]
        except KeyError:
            raise ValueError(f'Invalid Cavity integrator: {value}') from None

    @staticmethod
    def get_available_integrators():
        return list(_CAVITY_INTEGRATOR_IDS)

    def _check_radial_order(self):
        if self.radial_order not in _RADIAL_ORDERS:
            raise ValueError(f'radial_order must be one of {_RADIAL_ORDERS}')
        if self.radial_order != 4 and self._model != 10:
            raise ValueError('radial_order applies to "_phase-resolved" only: '
                             'the other models keep their own radial order')

    def track(self, particles, *args, **kwargs):

        self._check_physical_model(tracking=True)
        if _constant_part(self.harmonic) != 0:
            raise RuntimeError("Cavity cannot be used in standalone tracking "
                               "when harmonic is not zero. Please use the "
                               "cavity within a Line or set frequency instead"
                               " of harmonic.")
        return super().track(particles, *args, **kwargs)

    @property
    def frequency(self):
        return self._frequency

    @frequency.setter
    def frequency(self, value):
        if (_constant_part(self._harmonic) != 0
                and _constant_part(value) != 0):
            raise ValueError("Cannot set non-zero frequency when harmonic is not zero.")
        self._frequency = value

    @property
    def lag(self):
        return self._lag

    @lag.setter
    def lag(self, value):
        if _constant_part(value) != 0:
            warn("`lag` (in degrees) is deprecated and will be removed in a future version. "
                 "Please use `phase` (in radians) instead. If you see this warning "
                 "while loading a saved line from a previous version of Xsuite, please "
                 "regenerate the line with the current version to use phase instead of lag. "
                 "Note that if both `lag` and `phase` are set, the effect is the sum of the two, "
                 "with `lag` converted to radians. "
                 + DEPRECATION_INFO_PREP_1_0,
                 FutureWarning, stacklevel=2)
        self._lag = value

    @property
    def harmonic(self):
        return self._harmonic

    @harmonic.setter
    def harmonic(self, value):
        if (_constant_part(self._frequency) != 0
                and _constant_part(value) != 0):
            raise ValueError("Cannot set non-zero harmonic when frequency is not zero.")
        self._harmonic = value


class Cavity(_HasModelRF, _HasIntegrator, _CavityControls, BeamElement):

    _docstring_start = \
    '''RF cavity element.

    Parameters
    ----------
    length : float
        Length of the RF cavity in meters. Default is ``0``.
    voltage : float
        Voltage of the RF cavity in Volts. Default is ``0``.
    frequency : float
        Frequency of the RF cavity in Hertz. It can be set only if harmonic is zero.
        Default is ``0``.
    harmonic : float
        Harmonic number of the RF cavity. It can be set only if frequency is zero.
        If `harmonic` is non-zero, the frequency is computed from the length of the
        beam_line and the speed of the reference particle (beta0 * clight).
        When `harmonic` is set, the cavity can only be used within a Line and not
        in standalone tracking (i.e. Cavity.track(...) will raise an error).
        Default is ``0``.
    phase : float
        Phase in radians seen at the arrival time of the reference particle (zeta = 0).
        When `absolute_time` is True, `phase` is the phase at time zero. Default is ``0``.
    lag : float
        Deprecated phase shift in degrees, added to `phase`. Default is ``0``.
    absolute_time : bool
        If True, the cavity phase is computed from the absolute time of the
        simulation, otherwise the cavity is synchronized with the arrival time of
        the reference particle (zeta=0). Default is False.

    The following parameters are used only by the experimental physical models
    "_phase-averaged", "_phase-resolved", "_phase-resolved-paraxial" and
    "_phase-resolved-frozen"; the longitudinal models ignore them. The RF field
    is uniform along the cavity and ends abruptly at its end planes, which are hard
    edges: the particle leaves field-free. For a measured field, or a single cell,
    use FieldExpansionCavity. Spin is transported as by the magnets: when the line
    is configured for spin or radiation, for particles whose spin is not zero, and
    not in TPSA tracking. A particle with spin is lost (state -99744) where the
    cavity cannot carry it: through slices, on a context other than CPU, or with
    static fields that have active faces, a misalignment or a model other than
    "_phase-resolved".

    counter_wave_amplitude : float
        Amplitude of the counter-propagating RF wave over that of the
        co-propagating one: 0 is a travelling wave, 1 a standing wave, and the
        values between are partly reflected waves. Default is 1, or 0 when
        constructed with the frozen model. It is not re-inferred when `model`
        changes.
    counter_wave_phase : float
        Phase of the counter-propagating wave in radians. Default is 0.
    n_cells : int or None
        Number of pi-mode cells, each of length `length / n_cells`: they fix
        the spatial wave number to pi n_cells / length whatever the frequency,
        and `voltage` is the crest energy gain at their synchronous speed, or
        at c if that exceeds it. None (default) has no cells: the waves travel
        at c and `voltage` is the crest gain at c. `cell_length`,
        `wave_number` and `synchronous_beta` read the result.
    nominal_energy_gain : float or None
        Reference energy programme of the frozen model, in eV. None (default)
        derives it from the RF settings.
    radial_order : int
        Power of the offset r to which "_phase-resolved" keeps the on-axis
        field's near-axis series: E_z to r^radial_order, E_r and c B_phi one
        order lower; 2, 4 (default), 6 or 8. For waves at c the interior
        series is exact at any order; slower waves gain from a higher one.
    edge_entry_active, edge_exit_active : bool
        Whether the entrance and exit faces of the static fields are applied,
        on a MagnetOverlappedCavity. Default is True.
    '''.strip()

    _docstring_models = '''model : str
        "adaptive" (default) and the drift-kick-drift models are the
        longitudinal maps, which apply RF energy kicks only. The physical models
        are experimental, so their names start with an underscore:
        "_phase-averaged" is Sagan's averaged map with finite-beta terms, the
        fastest transverse model, valid where the energy changes little per
        cell and, for a standing wave, over whole cells (a cavity ending mid-cell
        misses 3 to 5% of the kick); it takes two steps per half wavelength.
        "_phase-resolved-paraxial" keeps both waves at quadratic transverse
        order, per step as an exact second-order generator along the reference
        orbit (accurate at hard edges), for particles that the
        cavity cannot stop. "_phase-resolved" integrates the full Hamiltonian of
        the RF and static fields in Gauss collocation cells.
        "_phase-resolved-frozen" is the frozen-coefficient voltage-action model
        with canonical faces. "_phase-averaged" and "_phase-resolved-paraxial"
        keep the force linear in the offset r: exact for waves at c, while
        slower waves, from cells shorter than half a wavelength, add a relative
        (k_z^2 - k^2) r^2 / 4, 1.6e-4 at 1 mm for beta 0.5 at 704 MHz.
    integrator : str
        How the model's base step is placed. Longitudinal models: "teapot",
        "uniform", "yoshida-4/6/8". "_phase-resolved": "uniform" (Gauss cells),
        "yoshida-6/8" (higher-order compositions), "_geometric" (cells graded
        by the energy ratio). "_phase-averaged" and "_phase-resolved-paraxial":
        "uniform", "_geometric". Frozen model: "_geometric" (default, SAD's
        mesh), "uniform".
    num_kicks : int
        Number of integration cells when nonzero. With zero, "_phase-resolved"
        uses 16 per half RF wavelength and at least 64, "_phase-averaged" two
        per half RF wavelength,
        "_phase-resolved-paraxial" two per half RF wavelength for a travelling
        wave (counter_wave_amplitude = 0) and four otherwise, at least eight in all and at least
        one for each 2% of V/E, and the frozen model its own count.
    '''.strip()

    __doc__ = '\n    '.join([_docstring_start, _docstring_models,
        _for_docstring_alignment, '\n',
        _docstring_general_notes, '\n\n'])

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
        'counter_wave_amplitude': FloatOrTpsa,
        'counter_wave_phase': FloatOrTpsa,
        'n_cells': xo.Int64,
        'radial_order': xo.Int64,
        'nominal_energy_gain': FloatOrTpsa,
        'nominal_energy_gain_from_rf': xo.Int64,
        'edge_entry_active': xo.Int64,
        'edge_exit_active': xo.Int64,
        'absolute_time': xo.Int64,
        'num_kicks': xo.Int64,
        'model': xo.Int64,
        'integrator': xo.Int64,
    }

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/cavity.h"',
    ]


    _rename = {
        'model': '_model',
        'integrator': '_integrator',
        'frequency': '_frequency',
        'harmonic': '_harmonic',
        'lag': '_lag',
        'n_cells': '_n_cells',
        'nominal_energy_gain': '_nominal_energy_gain',
    }

    _default_frequency = 0.0
    _default_harmonic = 0.0

    _noexpr_fields = _NOEXPR_FIELDS

    def __init__(self, **kwargs):

        if '_xobject' in kwargs and kwargs['_xobject'] is not None:
            self.xoinitialize(**kwargs)
            return

        model = kwargs.pop('model', None)
        integrator = kwargs.pop('integrator', None)
        frequency = kwargs.pop('frequency', None)
        harmonic = kwargs.pop('harmonic', None)
        lag = kwargs.pop('lag', None)
        n_cells = kwargs.pop('n_cells', None)
        nominal_energy_gain = kwargs.pop('nominal_energy_gain', None)
        nominal_from_rf = kwargs.pop('nominal_energy_gain_from_rf',
                                     nominal_energy_gain is None)
        kwargs.setdefault('counter_wave_amplitude',
                          _default_counter_wave_amplitude(model))
        kwargs.setdefault('counter_wave_phase', 0.)
        kwargs.setdefault('edge_entry_active', 1)
        kwargs.setdefault('edge_exit_active', 1)
        kwargs.setdefault('radial_order', 4)
        kwargs['nominal_energy_gain'] = (
            0. if nominal_energy_gain is None else nominal_energy_gain)
        kwargs['nominal_energy_gain_from_rf'] = int(nominal_from_rf)

        self.xoinitialize(**kwargs)

        # Trigger properties
        if model is not None:
            self.model = model

        if integrator is not None:
            self.integrator = integrator

        if frequency is not None:
            self.frequency = frequency

        if harmonic is not None:
            self.harmonic = harmonic

        if lag is not None:
            self.lag = lag

        self.n_cells = n_cells

        self._check_physical_model()

    def _check_physical_model(self, tracking=False):
        """Raise for settings the physical models cannot track."""
        self._check_radial_order()
        if self._model < 10:
            return
        # Length is commonly assigned after construction.
        if tracking and not (isfinite(_constant_part(self.length))
                             and _constant_part(self.length) > 0.):
            raise ValueError(f'{self.model} requires a positive finite length')
        if self.num_kicks < 0:
            raise ValueError('num_kicks must be non-negative')
        if self._model == 13:
            if self.integrator not in ('adaptive', '_geometric', 'uniform'):
                raise ValueError('_phase-resolved-frozen supports the "_geometric" '
                                 'and "uniform" integrators')
            if (_constant_part(self.counter_wave_amplitude) != 0.
                    or self.n_cells is not None):
                raise ValueError('_phase-resolved-frozen requires a travelling '
                                 'wave: counter_wave_amplitude=0 and no n_cells')
        elif self._model in (11, 12):
            if self.integrator not in ('adaptive', 'uniform', '_geometric'):
                raise ValueError(f'{self.model} supports the "uniform" and '
                                 '"_geometric" integrators: its cell-tied steps '
                                 'admit no teapot placement or higher-order '
                                 'composition')
        elif self.integrator == 'teapot':
            raise ValueError('_phase-resolved has no kicks to place: "teapot" '
                             'does not apply')

    @property
    def n_cells(self):
        return int(self._n_cells) or None

    @n_cells.setter
    def n_cells(self, value):
        if value is not None and (int(value) != value or value <= 0):
            raise ValueError('n_cells must be a positive integer or None')
        self._n_cells = 0 if value is None else int(value)

    @property
    def cell_length(self):
        return None if self.n_cells is None else self.length / self.n_cells

    @property
    def wave_number(self):
        if self.n_cells is not None:
            return np.pi * self.n_cells / self.length
        return 2 * np.pi * _constant_part(self.frequency) / clight

    @property
    def synchronous_beta(self):
        if self.n_cells is None:
            return 1.
        return 2 * _constant_part(self.frequency) * self.cell_length / clight

    @property
    def nominal_energy_gain(self):
        return (None if self.nominal_energy_gain_from_rf
                else self._nominal_energy_gain)

    @nominal_energy_gain.setter
    def nominal_energy_gain(self, value):
        self.nominal_energy_gain_from_rf = int(value is None)
        self._nominal_energy_gain = 0. if value is None else value

    def to_dict(self, **kwargs):
        result = super().to_dict(**kwargs)
        # The base class omits zeros; these defaults are nonzero, so an
        # explicit zero must be written and the default itself omitted.
        defaults = dict(_NONZERO_DEFAULTS, counter_wave_amplitude=
                        _default_counter_wave_amplitude(self.model))
        for name, default in defaults.items():
            value = getattr(self, name)
            if value == default:
                result.pop(name, None)
            else:
                result[name] = value
        # An explicit zero carrier differs physically from the derived default.
        result.pop('nominal_energy_gain_from_rf', None)
        if self.nominal_energy_gain_from_rf:
            result.pop('nominal_energy_gain', None)
        else:
            result['nominal_energy_gain'] = self.nominal_energy_gain
        result.pop('n_cells', None)
        if self.n_cells is not None:
            result['n_cells'] = self.n_cells
        return result

    @property
    def _thin_slice_class(self):
        return xt.ThinSliceCavity

    @property
    def _thick_slice_class(self):
        return xt.ThickSliceCavity

    @property
    def _drift_slice_class(self):
        return xt.DriftSliceCavity


class MagnetOverlappedCavity(_HasKnlKsl, Cavity):

    _docstring_start = Cavity._docstring_start.replace(
        'RF cavity element.',
        'RF cavity overlapped with static magnetic fields: a solenoid, and normal\n'
        '    and skew multipoles. It is tracked by the experimental physical models.', 1) + '''
    ks : float
        Solenoid strength in 1/m. Normal and skew multipoles use `knl`, `ksl`.
    The static fields are uniform along the cavity.'''

    __doc__ = '\n    '.join([_docstring_start, Cavity._docstring_models,
        _HasKnlKsl._for_docstring,
        _for_docstring_alignment, '\n',
        _docstring_general_notes, '\n\n'])

    _xofields = {
        **Cavity._xofields,
        'ks': FloatOrTpsa,
        'order': xo.Int64,
        'inv_factorial_order': xo.Float64,
        'knl': xo.Float64[:],
        'ksl': xo.Float64[:],
        'knl_tpsa': xo.UInt64[:],
        'ksl_tpsa': xo.UInt64[:],
    }

    isthick = True

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/magnet_overlapped_cavity.h"',
    ]

    _skip_in_to_dict = ['_order', 'inv_factorial_order',
                        'knl_tpsa', 'ksl_tpsa', '_knl_tpsa', '_ksl_tpsa']

    _rename = {
        **Cavity._rename,
        'order': '_order',
        'knl': '_knl',
        'ksl': '_ksl',
        'knl_tpsa': '_knl_tpsa',
        'ksl_tpsa': '_ksl_tpsa',
    }

    _float_or_tpsa_array_fields = ('knl', 'ksl')

    _noexpr_fields = Cavity._noexpr_fields

    def __init__(self, **kwargs):
        if '_xobject' in kwargs and kwargs['_xobject'] is not None:
            self.xoinitialize(**kwargs)
            return
        order = kwargs.pop('order', 5)
        kwargs.update(self._prepare_multipolar_params(
            order, knl=kwargs.pop('knl', None), ksl=kwargs.pop('ksl', None)))
        for name in ('knl', 'ksl'):
            # Handles are process-local owned storage, never serialized knobs.
            handles = kwargs.pop(name + '_tpsa', None)
            if handles is not None and any(handles):
                raise ValueError('Cavity TPSA array handles cannot be imported')
            kwargs[name + '_tpsa'] = [0] * len(kwargs[name])
        # _HasKnlKsl comes first in the order and has its own __init__.
        Cavity.__init__(self, **kwargs)

    def _check_physical_model(self, tracking=False):
        super()._check_physical_model(tracking=tracking)
        if self._model < 10:
            return
        if self._model in (11, 12):
            normal, skew = np.asarray(self.knl), np.asarray(self.ksl)
            others = (np.any(normal[np.arange(len(normal)) != 1] != 0.)
                      or np.any(skew != 0.))
            solenoid = _constant_part(self.ks) != 0.
            if others or (solenoid and (np.any(normal != 0.) or self.edge_entry_active
                                        or self.edge_exit_active)):
                raise ValueError(f'{self.model} supports a uniform solenoid with '
                                 'faces off or a normal quadrupole as static field')

    @property
    def knl(self):
        from ..tpsa._cavity_array import CavityArrayView
        return CavityArrayView(self, '_knl', '_knl_tpsa')

    @knl.setter
    def knl(self, value):
        self.knl[:] = value

    @property
    def ksl(self):
        from ..tpsa._cavity_array import CavityArrayView
        return CavityArrayView(self, '_ksl', '_ksl_tpsa')

    @ksl.setter
    def ksl(self, value):
        self.ksl[:] = value

    # HybridClass inspection uses raw field names even when Python storage
    # descriptors are renamed. These are internal handles, not saved settings.
    @property
    def knl_tpsa(self):
        return self._knl_tpsa

    @property
    def ksl_tpsa(self):
        return self._ksl_tpsa

    def to_dict(self, **kwargs):
        result = super().to_dict(**kwargs)
        if self.order == 5:
            result.pop('order', None)
        else:
            result['order'] = self.order
        for name in ('knl', 'ksl'):
            values = self._context.nparray_from_context_array(getattr(self, '_' + name))
            if np.any(values != 0.):
                result[name] = values.tolist()
            else:
                result.pop(name, None)
        return result

    @classmethod
    def from_dict(cls, dct, **kwargs):
        # A later order change can deliberately leave a larger allocated
        # array. Restore the saved active order after allocating that storage.
        order = dct.get('order')
        result = super().from_dict(dct, **kwargs)
        if order is not None:
            result.order = order
        return result

    @property
    def _thin_slice_class(self):
        return xt.ThinSliceMagnetOverlappedCavity

    @property
    def _thick_slice_class(self):
        return xt.ThickSliceMagnetOverlappedCavity

    @property
    def _drift_slice_class(self):
        return xt.DriftSliceMagnetOverlappedCavity


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
        'radial_order': xo.Int64,
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
        kwargs.setdefault('calibration_beta', 1.)
        kwargs.setdefault('radial_order', 4)
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
        amplitudes = np.trapezoid(values[None, :] * np.exp(-1j * k[:, None] * z[None, :]),
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
        points = len(self.field_table) // (2 * _FIELD_TABLE_JETS)
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

    def to_dict(self, **kwargs):
        result = super().to_dict(**kwargs)
        if self.calibration_beta == 1.:
            result.pop('calibration_beta', None)
        else:
            result['calibration_beta'] = self.calibration_beta
        for name in ('wave_numbers', 'amplitudes_re', 'amplitudes_im'):
            result[name] = self._context.nparray_from_context_array(
                getattr(self, name)).tolist()
        # The table is derived; from_dict rebuilds it from its size.
        for name in ('field_table', 'field_table_z0', 'field_table_dz', 'field_table_key'):
            result.pop(name, None)
        result['field_table_points'] = len(self.field_table) // (2 * _FIELD_TABLE_JETS)
        if self.radial_order == 4:
            result.pop('radial_order', None)
        return result

    @property
    def _thin_slice_class(self):
        return xt.ThinSliceFieldExpansionCavity

    @property
    def _thick_slice_class(self):
        return xt.ThickSliceFieldExpansionCavity

    @property
    def _drift_slice_class(self):
        return xt.DriftSliceFieldExpansionCavity
