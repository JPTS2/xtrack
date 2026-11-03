# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

from ..base_element import BeamElement, FloatOrTpsa
from ..general import DEPRECATION_INFO_PREP_1_0
from warnings import warn
from math import isfinite
import numpy as np
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

# Cavity has its own physics model axis; CrabCavity and RFMultipole keep the
# shared RF table. The kernel ids 5 and 6 are the legacy kick-drift maps.
_CAVITY_MODELS = {0: 'adaptive', 6: 'longitudinal-only',
                  5: 'longitudinal-only-exact', 10: 'phase-resolved',
                  11: 'phase-averaged', 13: 'phase-resolved-frozen'}
_CAVITY_MODEL_IDS = {value: key for key, value in _CAVITY_MODELS.items()}
# For RF the rot-kick-rot maps equal drift-kick-drift-exact bit for bit.
_DEPRECATED_CAVITY_MODELS = {
    'drift-kick-drift-expanded': 'longitudinal-only',
    'drift-kick-drift-exact': 'longitudinal-only-exact',
    'rot-kick-rot-low-order': 'longitudinal-only-exact',
    'rot-kick-rot-high-order': 'longitudinal-only-exact',
}
_CAVITY_INTEGRATORS = dict(_INDEX_TO_INTEGRATOR)
_CAVITY_INTEGRATORS[7] = 'geometric'
_CAVITY_INTEGRATOR_IDS = {
    value: key for key, value in _CAVITY_INTEGRATORS.items()}
_DEPRECATED_CAVITY_INTEGRATORS = {'gauss2': 'uniform',
                                  'geometric-kick-drift': 'geometric'}
_CAVITY_RF_PROFILES = {0: 'p16', 1: 'uniform', 2: 'gaussian'}
_CAVITY_RF_PROFILE_IDS = {value: key for key, value in _CAVITY_RF_PROFILES.items()}
_CAVITY_MAGNETIC_PROFILES = {0: 'uniform', 1: 'gaussian'}
_CAVITY_MAGNETIC_PROFILE_IDS = {
    value: key for key, value in _CAVITY_MAGNETIC_PROFILES.items()}


def _constant_part(value):
    """Inspect a knob's base point without stripping its stored dependence."""
    return getattr(value, 'const_part', value)


_NONZERO_DEFAULTS = {
    'eta': 1., 'calibration_beta': 1., 'rf_field_order': 2,
    'edge_entry_active': 1, 'edge_exit_active': 1, 'order': 5,
}


def _default_eta(model):
    return 0. if model == 'phase-resolved-frozen' else 1.


def _default_rf_profile(model):
    return 'uniform' if model == 'phase-resolved-frozen' else 'p16'


class Cavity(_HasKnlKsl, _HasModelRF, _HasIntegrator, BeamElement):

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

    The following parameters are used only by the physical models
    "phase-resolved" and "phase-resolved-frozen"; the legacy models ignore
    them.

    eta : float
        Squared amplitude ratio of the counter- to the co-propagating RF wave.
        Default is 1 (standing wave), or 0 when constructed with the frozen
        model. Like `rf_profile`, it is not re-inferred when `model` changes.
    counter_phase : float
        Phase of the counter-propagating wave in radians. Default is 0.
    calibration_beta : float
        Fixed speed, 0 < beta <= 1, at which `voltage` is the crest energy
        gain. Default is 1.
    wave_number : float or None
        Longitudinal RF wave number in rad/m. None (default) follows
        2*pi*frequency/c.
    nominal_energy_gain : float or None
        Reference energy programme of the frozen model, in eV. None (default)
        derives it from the RF settings.
    rf_profile : str
        Longitudinal RF envelope: "p16", "uniform" or "gaussian". The default
        is "uniform" for the frozen model and "p16" otherwise.
    rf_profile_sigma, rf_profile_center : float
        Width (required for "gaussian") and centre of the Gaussian envelope, in
        metres relative to the cavity midpoint.
    magnetic_profile : str
        Longitudinal profile of the static fields: "uniform" (default) or
        "gaussian"; the latter requires `magnetic_profile_sigma` and inactive
        end faces.
    ks : float
        Solenoid strength in 1/m. Normal and skew multipoles use `knl`, `ksl`.
    edge_entry_active, edge_exit_active : bool
        Whether the entrance and exit faces of the static fields are applied.
        Default is True.
    '''.strip()

    _docstring_models = '''model : str
        "longitudinal-only" (default, expanded drifts) and
        "longitudinal-only-exact" (exact drifts) apply RF energy kicks only.
        "phase-resolved" integrates the full Hamiltonian of the RF and static
        fields in Gauss collocation cells; "phase-resolved-frozen" is the
        frozen-coefficient voltage-action model with canonical faces. The
        drift-kick-drift and rot-kick-rot names are deprecated aliases.
    integrator : str
        How the model's base step is placed: "uniform" (equal steps; Gauss
        cells for "phase-resolved"), "teapot", "yoshida-4/6/8" and, for the
        frozen model, "geometric" (steps graded by the energy ratio, its
        default). "gauss2" and "geometric-kick-drift" are deprecated aliases.
    num_kicks : int
        Number of integration cells when nonzero. With zero, phase-resolved
        uses 64, phase-averaged two steps per half RF wavelength, and the
        frozen model its own count.
    '''.strip()

    __doc__ = '\n    '.join([_docstring_start, _docstring_models,
        _HasKnlKsl._for_docstring,
        _for_docstring_alignment, '\n',
        _docstring_general_notes, '\n\n'])

    isthick = True
    has_backtrack = True
    allow_loss_refinement = True

    _xofields = {
        'length': xo.Float64,
        'voltage': FloatOrTpsa,
        'frequency': FloatOrTpsa,
        'lag': FloatOrTpsa,
        'phase': FloatOrTpsa,
        'harmonic': FloatOrTpsa,
        'lag_taper': FloatOrTpsa,
        'phase_taper': FloatOrTpsa,
        'eta': FloatOrTpsa,
        'counter_phase': FloatOrTpsa,
        'calibration_beta': FloatOrTpsa,
        'wave_number': FloatOrTpsa,
        'wave_number_from_frequency': xo.Int64,
        'nominal_energy_gain': FloatOrTpsa,
        'nominal_energy_gain_from_rf': xo.Int64,
        'rf_profile': xo.Int64,
        'rf_profile_sigma': FloatOrTpsa,
        'rf_profile_center': FloatOrTpsa,
        'magnetic_profile': xo.Int64,
        'magnetic_profile_sigma': FloatOrTpsa,
        'rf_field_order': xo.Int64,
        'ks': FloatOrTpsa,
        'spin_tracking': xo.Int64,
        'edge_entry_active': xo.Int64,
        'edge_exit_active': xo.Int64,
        'order': xo.Int64,
        'inv_factorial_order': xo.Float64,
        'knl': xo.Float64[:],
        'ksl': xo.Float64[:],
        'knl_tpsa': xo.UInt64[:],
        'ksl_tpsa': xo.UInt64[:],
        'absolute_time': xo.Int64,
        'num_kicks': xo.Int64,
        'model': xo.Int64,
        'integrator': xo.Int64,
    }

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/cavity.h"',
    ]

    _skip_in_to_dict = ['_order', 'inv_factorial_order',
                        'knl_tpsa', 'ksl_tpsa', '_knl_tpsa', '_ksl_tpsa']

    _rename = {
        'model': '_model',
        'integrator': '_integrator',
        'frequency': '_frequency',
        'harmonic': '_harmonic',
        'lag': '_lag',
        'wave_number': '_wave_number',
        'nominal_energy_gain': '_nominal_energy_gain',
        'rf_profile': '_rf_profile',
        'magnetic_profile': '_magnetic_profile',
        'order': '_order',
        'knl': '_knl',
        'ksl': '_ksl',
        'knl_tpsa': '_knl_tpsa',
        'ksl_tpsa': '_ksl_tpsa',
    }

    _float_or_tpsa_array_fields = ('knl', 'ksl')

    _default_frequency = 0.0
    _default_harmonic = 0.0

    _noexpr_fields = _NOEXPR_FIELDS | {'rf_profile', 'magnetic_profile'}

    def __init__(self, **kwargs):

        if '_xobject' in kwargs and kwargs['_xobject'] is not None:
            self.xoinitialize(**kwargs)
            return

        model = kwargs.pop('model', None)
        integrator = kwargs.pop('integrator', None)
        frequency = kwargs.pop('frequency', None)
        harmonic = kwargs.pop('harmonic', None)
        lag = kwargs.pop('lag', None)
        wave_number = kwargs.pop('wave_number', None)
        from_frequency = kwargs.pop('wave_number_from_frequency',
                                    wave_number is None)
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
        magnetic_profile = kwargs.pop('magnetic_profile', 'uniform')
        try:
            kwargs['magnetic_profile'] = _CAVITY_MAGNETIC_PROFILE_IDS[magnetic_profile]
        except KeyError:
            raise ValueError(
                f'Invalid Cavity magnetic_profile: {magnetic_profile}') from None
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
        kwargs.setdefault('eta', _default_eta(model))
        kwargs.setdefault('counter_phase', 0.)
        kwargs.setdefault('calibration_beta', 1.)
        kwargs.setdefault('rf_field_order', 2)
        kwargs.setdefault('edge_entry_active', 1)
        kwargs.setdefault('edge_exit_active', 1)
        kwargs['wave_number'] = 0. if wave_number is None else wave_number
        kwargs['wave_number_from_frequency'] = int(from_frequency)
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
        if self.model == 'phase-resolved-frozen' and self.rf_profile != 'uniform':
            raise ValueError('phase-resolved-frozen requires rf_profile="uniform"')
        # Length is commonly assigned after construction.
        if tracking and not (isfinite(_constant_part(self.length))
                             and _constant_part(self.length) > 0.):
            raise ValueError(f'{self.model} requires a positive finite length')
        if self.num_kicks < 0:
            raise ValueError('num_kicks must be non-negative')
        if self.rf_field_order != 2:
            raise ValueError(f'{self.model} supports only rf_field_order=2')
        if self.model == 'phase-resolved-frozen':
            if self.integrator not in ('adaptive', 'geometric', 'uniform'):
                raise ValueError('phase-resolved-frozen supports the "geometric" '
                                 'and "uniform" integrators')
            if (_constant_part(self.eta) != 0.
                    or _constant_part(self.calibration_beta) != 1.
                    or not self.wave_number_from_frequency):
                raise ValueError('phase-resolved-frozen requires eta=0, '
                                 'calibration_beta=1 and the carrier from frequency')
        elif self.model == 'phase-averaged':
            if self.integrator not in ('adaptive', 'uniform'):
                raise ValueError('phase-averaged supports the "uniform" integrator')
            if (_constant_part(self.ks) != 0. or np.any(np.asarray(self.knl) != 0.)
                    or np.any(np.asarray(self.ksl) != 0.)
                    or self.magnetic_profile != 'uniform'):
                raise ValueError('phase-averaged does not yet support static fields')
        elif self.integrator == 'teapot':
            raise ValueError('phase-resolved has no kicks to place: "teapot" '
                             'does not apply')
        if self.spin_tracking:
            self._check_spin_support()

    def _check_spin_support(self):
        """Raise if spin transport is not implemented for these settings."""
        if (self._model != 10 or self._rf_profile != 0
                or self._magnetic_profile != 0
                or self.rf_field_order != 2 or self.integrator == 'teapot'):
            raise NotImplementedError('Cavity spin transport requires the phase-resolved model '
                                      'with the P16 profile and a uniform magnetic profile')
        normal = np.asarray(self.knl)
        skew = np.asarray(self.ksl)
        if (np.any(normal[np.arange(len(normal)) != 1] != 0.)
                or np.any(skew[np.arange(len(skew)) != 1] != 0.)
                or (self.order < 1 and (np.any(normal != 0.) or np.any(skew != 0.)))):
            raise NotImplementedError('Cavity spin transport supports only solenoid and '
                                      'normal/skew quadrupole static strengths')
        static = self.ks != 0. or np.any(normal != 0.) or np.any(skew != 0.)
        if static and (self.edge_entry_active or self.edge_exit_active):
            raise NotImplementedError('Cavity spin transport with static fields requires both faces disabled')
        if static and self.transformations_active:
            raise NotImplementedError('Cavity spin transport with static fields does not support alignment transformations')

    @property
    def model(self):
        return _CAVITY_MODELS[self._model]

    @model.setter
    def model(self, value):
        if value in _DEPRECATED_CAVITY_MODELS:
            new = _DEPRECATED_CAVITY_MODELS[value]
            warn(f'Cavity model "{value}" is deprecated; use "{new}".',
                 FutureWarning)
            value = new
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
        if value in _DEPRECATED_CAVITY_INTEGRATORS:
            new = _DEPRECATED_CAVITY_INTEGRATORS[value]
            warn(f'Cavity integrator "{value}" is deprecated; use "{new}".',
                 FutureWarning)
            value = new
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
    def magnetic_profile(self):
        return _CAVITY_MAGNETIC_PROFILES[self._magnetic_profile]

    @magnetic_profile.setter
    def magnetic_profile(self, value):
        try:
            self._magnetic_profile = _CAVITY_MAGNETIC_PROFILE_IDS[value]
        except KeyError:
            raise ValueError(f'Invalid Cavity magnetic_profile: {value}') from None

    @property
    def wave_number(self):
        return None if self.wave_number_from_frequency else self._wave_number

    @wave_number.setter
    def wave_number(self, value):
        self.wave_number_from_frequency = int(value is None)
        self._wave_number = 0. if value is None else value

    @property
    def nominal_energy_gain(self):
        return (None if self.nominal_energy_gain_from_rf
                else self._nominal_energy_gain)

    @nominal_energy_gain.setter
    def nominal_energy_gain(self, value):
        self.nominal_energy_gain_from_rf = int(value is None)
        self._nominal_energy_gain = 0. if value is None else value

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
        # The base class omits zeros; these defaults are nonzero, so an
        # explicit zero must be written and the default itself omitted.
        defaults = dict(_NONZERO_DEFAULTS, eta=_default_eta(self.model))
        for name, default in defaults.items():
            value = getattr(self, name)
            if value == default:
                result.pop(name, None)
            else:
                result[name] = value
        # An explicit zero carrier differs physically from the derived default.
        for name, derived in (('wave_number', 'wave_number_from_frequency'),
                              ('nominal_energy_gain', 'nominal_energy_gain_from_rf')):
            result.pop(derived, None)
            if getattr(self, derived):
                result.pop(name, None)
            else:
                result[name] = getattr(self, name)
        for name, default in (('rf_profile', _default_rf_profile(self.model)),
                              ('magnetic_profile', 'uniform')):
            if getattr(self, name) == default:
                result.pop(name, None)
            else:
                result[name] = getattr(self, name)
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
