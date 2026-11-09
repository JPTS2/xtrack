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
_CAVITY_RF_PROFILES = {0: 'p16', 1: 'uniform', 2: 'gaussian'}
_CAVITY_RF_PROFILE_IDS = {value: key for key, value in _CAVITY_RF_PROFILES.items()}
_CAVITY_MAGNETIC_PROFILES = {0: 'uniform', 1: 'gaussian'}
_CAVITY_MAGNETIC_PROFILE_IDS = {
    value: key for key, value in _CAVITY_MAGNETIC_PROFILES.items()}


def _constant_part(value):
    """Inspect a knob's base point without stripping its stored dependence."""
    return getattr(value, 'const_part', value)


_NONZERO_DEFAULTS = {'edge_entry_active': 1, 'edge_exit_active': 1}


def _default_counter_wave_amplitude(model):
    return 0. if model == '_phase-resolved-frozen' else 1.


def _default_rf_profile(model):
    return 'uniform' if model == '_phase-resolved-frozen' else 'p16'


class Cavity(_HasModelRF, _HasIntegrator, BeamElement):

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
    "_phase-resolved-frozen"; the longitudinal models ignore them. A whole cavity's end planes are hard
    edges: an RF field that does not vanish there ("uniform", a truncated
    "gaussian") ends abruptly, and the particle leaves field-free.

    counter_wave_amplitude : float
        Amplitude of the counter-propagating RF wave over that of the
        co-propagating one: 0 is a travelling wave, 1 a standing wave, and the
        values between are partly reflected waves. Default is 1, or 0 when
        constructed with the frozen model. Like `rf_profile`, it is not
        re-inferred when `model` changes.
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
    rf_profile : str
        Longitudinal RF envelope: "p16", "uniform" or "gaussian". The default
        is "uniform" for the frozen model and "p16" otherwise.
    rf_profile_sigma, rf_profile_center : float
        Width (required for "gaussian") and centre of the Gaussian envelope, in
        metres relative to the cavity midpoint.
    edge_entry_active, edge_exit_active : bool
        Whether the entrance and exit faces of the static fields are applied,
        on a MagnetOverlappedCavity. Default is True.
    '''.strip()

    _docstring_models = '''model : str
        "adaptive" (default) and the drift-kick-drift models are the
        longitudinal maps, which apply RF energy kicks only. The physical models
        are experimental, so their names start with an underscore:
        "_phase-averaged" is Sagan's averaged map with finite-beta terms, the
        fastest transverse model, valid where the energy barely changes and the
        envelope is smooth over the counter-wave beat; it takes the steps its
        envelope needs, at least two per half wavelength.
        "_phase-resolved-paraxial" keeps both waves at quadratic transverse
        order, per step as an exact second-order generator along the reference
        orbit (accurate for any envelope and hard edges), for particles that the
        cavity cannot stop. "_phase-resolved" integrates the full Hamiltonian of
        the RF and static fields in Gauss collocation cells.
        "_phase-resolved-frozen" is the frozen-coefficient voltage-action model
        with canonical faces. "_phase-averaged" and "_phase-resolved-paraxial"
        drop the force cubic in the offset r, which comes only from the
        variation of the envelope: none on a uniform one, and at most
        0.56 (r/sigma)^2 of the kick for an envelope of width sigma.
    integrator : str
        How the model's base step is placed. Longitudinal models: "teapot",
        "uniform", "yoshida-4/6/8". "_phase-resolved": "uniform" (Gauss cells),
        "yoshida-6/8" (higher-order compositions), "_geometric" (cells graded
        by the energy ratio). "_phase-averaged" and "_phase-resolved-paraxial":
        "uniform", "_geometric". Frozen model: "_geometric" (default, SAD's
        mesh), "uniform".
    num_kicks : int
        Number of integration cells when nonzero. With zero, "_phase-resolved"
        uses 16 per half RF wavelength and at least 64, "_phase-averaged" the
        steps its envelope needs, at least two per half RF wavelength,
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
        'nominal_energy_gain': FloatOrTpsa,
        'nominal_energy_gain_from_rf': xo.Int64,
        'rf_profile': xo.Int64,
        'rf_profile_sigma': FloatOrTpsa,
        'rf_profile_center': FloatOrTpsa,
        'spin_tracking': xo.Int64,
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
        'rf_profile': '_rf_profile',
    }

    _default_frequency = 0.0
    _default_harmonic = 0.0

    _noexpr_fields = _NOEXPR_FIELDS | {'rf_profile'}

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
        profile = kwargs.pop('rf_profile', None)
        if profile is None:
            profile = _default_rf_profile(model)
        try:
            kwargs['rf_profile'] = _CAVITY_RF_PROFILE_IDS[profile]
        except KeyError:
            raise ValueError(f'Invalid Cavity rf_profile: {profile}') from None
        sigma = kwargs.pop('rf_profile_sigma', None)
        kwargs['rf_profile_sigma'] = 0. if sigma is None else sigma
        kwargs.setdefault('rf_profile_center', 0.)
        kwargs.setdefault('counter_wave_amplitude',
                          _default_counter_wave_amplitude(model))
        kwargs.setdefault('counter_wave_phase', 0.)
        kwargs.setdefault('edge_entry_active', 1)
        kwargs.setdefault('edge_exit_active', 1)
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
        if self._model < 10:
            return
        if self.rf_profile == 'gaussian':
            width = float(_constant_part(self.rf_profile_sigma))
            centre = float(_constant_part(self.rf_profile_center))
            if not (isfinite(width) and width > 0. and isfinite(centre)):
                raise ValueError('A Gaussian RF profile requires a positive '
                                 'finite width rf_profile_sigma and a finite '
                                 'centre rf_profile_center')
        if self._model == 13 and self.rf_profile != 'uniform':
            raise ValueError('_phase-resolved-frozen requires rf_profile="uniform"')
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
        if self.spin_tracking:
            self._check_spin_support()

    def _check_spin_support(self):
        """Raise if spin transport is not implemented for these settings."""
        if self._model in (11, 12, 13):
            return
        if (self._model != 10 or self.rf_profile == 'gaussian'
                or self.integrator == 'teapot'):
            raise NotImplementedError('Cavity spin transport requires the _phase-resolved '
                                      'model with the P16 or uniform profile')

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

    @property
    def rf_profile(self):
        return _CAVITY_RF_PROFILES[self._rf_profile]

    @rf_profile.setter
    def rf_profile(self, value):
        try:
            self._rf_profile = _CAVITY_RF_PROFILE_IDS[value]
        except KeyError:
            raise ValueError(f'Invalid Cavity rf_profile: {value}') from None

    @staticmethod
    def get_available_rf_profiles():
        return list(_CAVITY_RF_PROFILE_IDS)

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
        if self.rf_profile == _default_rf_profile(self.model):
            result.pop('rf_profile', None)
        else:
            result['rf_profile'] = self.rf_profile
        return result

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
    magnetic_profile : str
        Longitudinal profile of the static fields: "uniform" (default) or
        "gaussian"; the latter requires `magnetic_profile_sigma` and inactive
        end faces.'''

    __doc__ = '\n    '.join([_docstring_start, Cavity._docstring_models,
        _HasKnlKsl._for_docstring,
        _for_docstring_alignment, '\n',
        _docstring_general_notes, '\n\n'])

    _xofields = {
        **Cavity._xofields,
        'magnetic_profile': xo.Int64,
        'magnetic_profile_sigma': FloatOrTpsa,
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
        'magnetic_profile': '_magnetic_profile',
        'order': '_order',
        'knl': '_knl',
        'ksl': '_ksl',
        'knl_tpsa': '_knl_tpsa',
        'ksl_tpsa': '_ksl_tpsa',
    }

    _float_or_tpsa_array_fields = ('knl', 'ksl')

    _noexpr_fields = Cavity._noexpr_fields | {'magnetic_profile'}

    def __init__(self, **kwargs):
        if '_xobject' in kwargs and kwargs['_xobject'] is not None:
            self.xoinitialize(**kwargs)
            return
        magnetic_profile = kwargs.pop('magnetic_profile', 'uniform')
        try:
            kwargs['magnetic_profile'] = _CAVITY_MAGNETIC_PROFILE_IDS[magnetic_profile]
        except KeyError:
            raise ValueError(
                f'Invalid magnetic_profile: {magnetic_profile}') from None
        magnetic_sigma = kwargs.pop('magnetic_profile_sigma', None)
        kwargs['magnetic_profile_sigma'] = (
            0. if magnetic_sigma is None else magnetic_sigma)
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
        if self.magnetic_profile == 'gaussian':
            if self._model != 10:
                raise ValueError(f'{self.model} does not support a Gaussian '
                                 'magnetic profile')
            width = float(_constant_part(self.magnetic_profile_sigma))
            if not (isfinite(width) and width > 0.):
                raise ValueError('A Gaussian magnetic profile requires a '
                                 'positive finite width magnetic_profile_sigma')
            if self.edge_entry_active or self.edge_exit_active:
                raise ValueError('A Gaussian magnetic profile requires '
                                 'inactive entry/exit faces')
        if self._model in (11, 12):
            normal, skew = np.asarray(self.knl), np.asarray(self.ksl)
            others = (np.any(normal[np.arange(len(normal)) != 1] != 0.)
                      or np.any(skew != 0.) or self.magnetic_profile != 'uniform')
            solenoid = _constant_part(self.ks) != 0.
            if others or (solenoid and (np.any(normal != 0.) or self.edge_entry_active
                                        or self.edge_exit_active)):
                raise ValueError(f'{self.model} supports a uniform solenoid with '
                                 'faces off or a normal quadrupole as static field')

    def _check_spin_support(self):
        super()._check_spin_support()
        normal = np.asarray(self.knl)
        skew = np.asarray(self.ksl)
        static = _constant_part(self.ks) != 0. or np.any(normal != 0.) or np.any(skew != 0.)
        if self._model in (11, 12, 13):
            if static:
                raise NotImplementedError(f'Cavity spin transport with static fields '
                                          f'is not implemented for {self.model}')
            return
        if self._magnetic_profile != 0:
            raise NotImplementedError('Cavity spin transport requires a uniform '
                                      'magnetic profile')
        if (np.any(normal[np.arange(len(normal)) != 1] != 0.)
                or np.any(skew[np.arange(len(skew)) != 1] != 0.)
                or (self.order < 1 and (np.any(normal != 0.) or np.any(skew != 0.)))):
            raise NotImplementedError('Cavity spin transport supports only solenoid and '
                                      'normal/skew quadrupole static strengths')
        if static and (self.edge_entry_active or self.edge_exit_active):
            raise NotImplementedError('Cavity spin transport with static fields requires both faces disabled')
        if static and self.transformations_active:
            raise NotImplementedError('Cavity spin transport with static fields does not support alignment transformations')

    @property
    def magnetic_profile(self):
        return _CAVITY_MAGNETIC_PROFILES[self._magnetic_profile]

    @magnetic_profile.setter
    def magnetic_profile(self, value):
        try:
            self._magnetic_profile = _CAVITY_MAGNETIC_PROFILE_IDS[value]
        except KeyError:
            raise ValueError(f'Invalid magnetic_profile: {value}') from None

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
        if self.magnetic_profile == 'uniform':
            result.pop('magnetic_profile', None)
        else:
            result['magnetic_profile'] = self.magnetic_profile
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
