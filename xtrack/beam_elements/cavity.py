# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

from ..base_element import BeamElement, FloatOrTpsa
from ..general import DEPRECATION_INFO_PREP_1_0
from warnings import warn
from math import isfinite
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

# These physical models belong to Cavity. Extending the shared RF table would
# also advertise them on CrabCavity/RFMultipole, whose kernels lack them.
_CAVITY_MODELS = dict(_INDEX_TO_MODEL_RF)
_CAVITY_MODELS.update({10: 'phase-resolved', 13: 'phase-resolved-frozen'})
_CAVITY_MODEL_IDS = {value: key for key, value in _CAVITY_MODELS.items()}
_CAVITY_INTEGRATORS = dict(_INDEX_TO_INTEGRATOR)
_CAVITY_INTEGRATORS[6] = 'gauss2'
_CAVITY_INTEGRATORS[7] = 'geometric-kick-drift'
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
    them and refuse nonzero static strengths.

    eta : float
        Squared amplitude ratio of the counter- to the co-propagating RF wave.
        Default is 1 (standing wave).
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
        Legacy RF models keep their behaviour. "phase-resolved" integrates the
        full Hamiltonian of the RF and static fields with Gauss collocation;
        "phase-resolved-frozen" is the frozen-coefficient voltage-action model.
    integrator : str
        "adaptive" selects "gauss2" for the phase-resolved model and
        "geometric-kick-drift" for the frozen model.
    num_kicks : int
        Number of integration cells (phase-resolved default 64; the frozen
        model chooses its own count when zero).
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

    _track_local_particle_preflight = 'CavityModels_preflight'
    # The constrained taper pass anchors fields to its first reference lane.
    _track_transformations_transaction = (
        'GET_PARAM(el, model) >= 10 && '
        '!LocalParticle_check_track_flag(part0, XS_FLAG_SR_TAPER)')

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
        kwargs.setdefault('eta', 1.)
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

    def _check_physical_model(self):
        """Raise for settings the physical models cannot track."""
        if self._model < 10:
            if self._rf_profile == 2:
                raise ValueError('Legacy Cavity models do not support a '
                                 'Gaussian RF profile')
            return
        if self.rf_profile == 'gaussian':
            width = float(_constant_part(self.rf_profile_sigma))
            centre = float(_constant_part(self.rf_profile_center))
            if not (isfinite(width) and width > 0. and isfinite(centre)):
                raise ValueError('A Gaussian RF profile requires a positive '
                                 'finite rf_profile_sigma and finite centre')
        if self.magnetic_profile == 'gaussian':
            if self._model != 10:
                raise ValueError(f'{self.model} does not support a Gaussian '
                                 'magnetic profile')
            width = float(_constant_part(self.magnetic_profile_sigma))
            if not (isfinite(width) and width > 0.):
                raise ValueError('A Gaussian magnetic profile requires a '
                                 'positive finite magnetic_profile_sigma')
            if self.edge_entry_active or self.edge_exit_active:
                raise ValueError('A Gaussian magnetic profile requires '
                                 'inactive entry/exit faces')
        if self.model == 'phase-resolved-frozen' and self.rf_profile != 'uniform':
            raise ValueError('phase-resolved-frozen requires rf_profile="uniform"')

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
        # Zero-valued fields are normally omitted. Here an explicit zero
        # carrier differs physically from the constructor default None.
        result['wave_number'] = self.wave_number
        result['wave_number_from_frequency'] = int(self.wave_number_from_frequency)
        result['nominal_energy_gain'] = self.nominal_energy_gain
        result['nominal_energy_gain_from_rf'] = int(self.nominal_energy_gain_from_rf)
        result['rf_profile'] = self.rf_profile
        result['rf_profile_sigma'] = self.rf_profile_sigma
        result['rf_profile_center'] = self.rf_profile_center
        result['magnetic_profile'] = self.magnetic_profile
        result['magnetic_profile_sigma'] = self.magnetic_profile_sigma
        # Explicit zeros must not be replaced by nonzero constructor defaults.
        for name in ('eta', 'calibration_beta', 'rf_field_order',
                     'edge_entry_active', 'edge_exit_active', 'order'):
            result[name] = getattr(self, name)
        for name in ('knl_tpsa', 'ksl_tpsa', '_knl_tpsa', '_ksl_tpsa'):
            result.pop(name, None)
        result['knl'] = self._context.nparray_from_context_array(self._knl).tolist()
        result['ksl'] = self._context.nparray_from_context_array(self._ksl).tolist()
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
