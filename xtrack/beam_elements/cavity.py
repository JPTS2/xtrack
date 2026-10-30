# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

from ..base_element import BeamElement, FloatOrTpsa
from ..general import DEPRECATION_INFO_PREP_1_0
from warnings import warn
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
# also advertise them on CrabCavity/RFMultipole, whose kernels do not implement
# the corresponding total-field Hamiltonians.
_CAVITY_MODELS = dict(_INDEX_TO_MODEL_RF)
_CAVITY_MODELS.update({
    10: 'phase-resolved',
    11: 'phase-resolved-paraxial',
    12: 'phase-resolved-longitudinal',
    13: 'phase-resolved-frozen',
})
_CAVITY_MODEL_IDS = {value: key for key, value in _CAVITY_MODELS.items()}
_CAVITY_INTEGRATORS = dict(_INDEX_TO_INTEGRATOR)
_CAVITY_INTEGRATORS[6] = 'gauss2'
_CAVITY_INTEGRATORS[7] = 'geometric-kick-drift'
_CAVITY_INTEGRATOR_IDS = {
    value: key for key, value in _CAVITY_INTEGRATORS.items()}


def _constant_part(value):
    """Inspect a knob's base point without stripping its stored dependence."""
    return getattr(value, 'const_part', value)


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
        Arrival time is
        particles.time_s + (s - zeta) / (beta0 * c). The phase then uses one
        laboratory-time origin for all cavity locations. For a frequency ramp,
        supply phase = Phi(time_s) - 2*pi*frequency(time_s)*time_s, where Phi
        is the integrated RF phase. This linearizes Phi within each frame;
        changing frequency alone does not integrate the RF phase.
    eta : float
        Squared counter-/co-propagating RF amplitude ratio for the new physical
        models. Default is 1 (equal waves); this ideal-source default does not
        identify an installed cavity's geometry. Legacy models ignore this field.
    counter_phase : float
        Relative counter-wave phase in radians. Default is 0.
    calibration_beta : float
        Fixed electric-crest calibration ray, 0 < beta <= 1. Default is 1.
        It does not follow the tracked particle's speed.
    wave_number : float or None
        Longitudinal carrier in rad/m. None follows 2*pi*frequency/c, including
        harmonic resolution. A number is fixed; an explicit zero is distinct
        from None.
    rf_field_order : int
        Radial Maxwell-recurrence order, separate from magnetic multipole order.
        The current live-source implementation supports M=2.
    ks : float
        Uniform longitudinal magnetic field divided by signed reference
        rigidity, in inverse metres. Normal/skew integrated multipoles use
        knl/ksl, with the standard factorial and reference-rigidity conventions.

    The new phase-resolved models are experimental development interfaces.
    They do not establish physical-source, numerical-error or GPU qualification.
    '''.strip()

    _docstring_models = '''model : str
        Legacy RF selectors retain their existing behaviour. New physical
        selectors are "phase-resolved", "phase-resolved-paraxial",
        "phase-resolved-longitudinal" and "phase-resolved-frozen".
        Physical assumptions and numerical integration are separate choices.
    integrator : str
        The new continuous models use "gauss2" (two-stage Gauss collocation)
        when "adaptive" is selected. Legacy integrator indices are unchanged.
    '''.strip()

    __doc__ = '\n    '.join([_docstring_start, _docstring_models,
        _HasIntegrator._for_docstring.replace(
            'num_multipole_kicks', 'num_kicks').replace('multipole kicks', 'kicks'),
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
        'rf_field_order': xo.Int64,
        'ks': FloatOrTpsa,
        'edge_entry_active': xo.Int64,
        'edge_exit_active': xo.Int64,
        'order': xo.Int64,
        'inv_factorial_order': xo.Float64,
        'knl': xo.Float64[:],
        'ksl': xo.Float64[:],
        'absolute_time': xo.Int64,
        'num_kicks': xo.Int64,
        'num_factor_steps': xo.Int64,
        'model': xo.Int64,
        'integrator': xo.Int64,
    }

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/cavity.h"',
    ]

    _skip_in_to_dict = ['_order', 'inv_factorial_order']  # defined by knl, etc.

    _rename = {
        'model': '_model',
        'integrator': '_integrator',
        'frequency': '_frequency',
        'harmonic': '_harmonic',
        'lag': '_lag',
        'wave_number': '_wave_number',
        'order': '_order',
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
        wave_number = kwargs.pop('wave_number', None)
        from_frequency = kwargs.pop('wave_number_from_frequency',
                                    wave_number is None)
        order = kwargs.pop('order', 5)
        multipoles = self._prepare_multipolar_params(
            order, knl=kwargs.pop('knl', None), ksl=kwargs.pop('ksl', None))
        kwargs.update(multipoles)
        kwargs.setdefault('eta', 1.)
        kwargs.setdefault('counter_phase', 0.)
        kwargs.setdefault('calibration_beta', 1.)
        kwargs.setdefault('rf_field_order', 2)
        kwargs.setdefault('edge_entry_active', 1)
        kwargs.setdefault('edge_exit_active', 1)
        kwargs.setdefault('num_factor_steps', 1)
        kwargs['wave_number'] = 0. if wave_number is None else wave_number
        kwargs['wave_number_from_frequency'] = int(from_frequency)

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
    def wave_number(self):
        return None if self.wave_number_from_frequency else self._wave_number

    @wave_number.setter
    def wave_number(self, value):
        self.wave_number_from_frequency = int(value is None)
        self._wave_number = 0. if value is None else value

    def to_dict(self, **kwargs):
        result = super().to_dict(**kwargs)
        # Zero-valued fields are normally omitted. Here an explicit zero
        # carrier differs physically from the constructor default None.
        result['wave_number'] = self.wave_number
        result['wave_number_from_frequency'] = int(self.wave_number_from_frequency)
        return result

    def track(self, particles, *args, **kwargs):

        if _constant_part(self.harmonic) != 0:
            raise RuntimeError("Cavity cannot be used in standalone tracking "
                               "when harmonic is not zero. Please use the "
                               "cavity within a Line or set frequency instead"
                               " of harmonic.")
        return super().track(particles, *args, **kwargs)

    def _prepare_native_diagnostic(self, particle_ref, *, eta, radius_m,
            counter_phase=0., calibration_beta=1., num_cells=8, anchor_s=0.,
            _context=None):
        """Prepare a separate private full-root M2/P16 Gauss diagnostic.

        Existing Cavity tracking and selectors retain their meanings. This
        explicit snapshot has no numerical/source/backend admission. Passive
        BMT spin is a separate prepared-object opt-in; radiation feedback and
        noise remain unfinished. Changing settings requires preparation again.
        Only the documented fixed-alignment absolute-time subset is
        currently supported; radius_m is a runtime guard, not a qualified
        physical aperture. This is an experimental private interface.
        """
        from .cavity_native import prepare_native_diagnostic
        return prepare_native_diagnostic(self, particle_ref, eta=eta,
            radius_m=radius_m, counter_phase=counter_phase,
            calibration_beta=calibration_beta, num_cells=num_cells,
            anchor_s=anchor_s, _context=_context)

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
