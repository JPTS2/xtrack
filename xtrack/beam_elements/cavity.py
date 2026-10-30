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
from .cavity_envelope_data import EnvelopeTableData
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
    14: 'phase-averaged-second-order',
    15: 'phase-averaged-cubic',
    16: 'phase-averaged-quartic',
    17: 'phase-averaged-full-slow',
    18: 'phase-averaged-uniform-quadratic-first-boundary',
    19: 'phase-averaged-uniform-quadratic',
    20: 'phase-averaged-magnetic',
    21: 'phase-averaged-envelope',
    22: 'phase-averaged-magnetic-fringe',
    23: 'phase-averaged-dipole',
    24: 'phase-averaged-moving-reference',
})
_CAVITY_MODEL_IDS = {value: key for key, value in _CAVITY_MODELS.items()}
_CAVITY_INTEGRATORS = dict(_INDEX_TO_INTEGRATOR)
_CAVITY_INTEGRATORS[6] = 'gauss2'
_CAVITY_INTEGRATORS[7] = 'geometric-kick-drift'
_CAVITY_INTEGRATORS[8] = 'midpoint'
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
    nominal_energy_gain : float or None
        Frozen-model reference programme, in eV per reference particle,
        including its absolute reference charge. None derives the nominal
        gain from the RF settings. An explicit value, including zero, fixes
        the programme independently of the physical RF source and clock.
        Only "phase-resolved-frozen" consumes this setting.
    rf_profile : str or None
        Built-in ideal longitudinal RF envelope: "p16", "uniform" or
        "gaussian". None selects uniform for the frozen and uniform-quadratic
        models, and p16 otherwise. The selected profile is stored independently
        of the transport model: changing model never changes the profile.
        Gaussian requires an explicit positive rf_profile_sigma. Uniform and
        Gaussian intervals can have nonzero fields at their observation planes.
    rf_profile_sigma : float
        Gaussian width in metres. There is no guessed Gaussian width; it must
        be supplied explicitly when choosing that profile. The stored default
        zero is unused for other profiles.
    rf_profile_center : float
        Gaussian centre in metres relative to the parent midpoint. Default: 0.
        Width and centre remain physical lengths when RF frequency changes.
    envelope_table : EnvelopeTableData or None
        Immutable prepared scalar body/boundary data for the distinct
        phase-averaged-envelope model. Attachment copies the payload into
        this element's buffer. Its frequency, carrier, geometry and Gaussian
        profile are fixed; state and supported RF strength/phase controls
        remain live. A prepared table is not a source/error certificate.
    magnetic_profile : str
        Static magnetic envelope, "uniform" (default) or "gaussian".
        Gaussian is the original order-one donor gauge and requires an
        explicit positive magnetic_profile_sigma and inactive end faces.
        It is independent of the transport model and the RF envelope.
    magnetic_profile_sigma : float
        Gaussian magnetic width in metres, centred at the parent midpoint.
        No width is inferred. With this profile, ks is the peak normalized
        solenoid field; knl/ksl remain integrals over the finite parent.
    reference_px : float
        Fixed parent-entrance canonical reference px for the dipole and
        moving-reference models, in native Xtrack normalized momentum units.
        Default: 0. This
        prescribed programme is never retuned to the tracked particle.
    reference_x, reference_y : float
        Fixed moving-reference entrance coordinates in metres. Default: 0.
    reference_py : float
        Fixed moving-reference entrance canonical py in native Xtrack
        normalized momentum units. Default: 0.
    num_reference_steps : int
        Positive RK4 count for the fixed moving-reference programme. Default: 4.
        Independent of RF source and body/boundary numerical resolutions.
    rf_field_order : int
        Radial Maxwell-recurrence order, separate from magnetic multipole order.
        Full resolved/root sources currently support M=2. The quadratic
        constructions have the fixed RF truncation stated in their model.
    ks : float
        Uniform longitudinal magnetic field divided by signed reference
        rigidity, in inverse metres. Normal/skew integrated multipoles use
        knl/ksl, with the standard factorial and reference-rigidity conventions.
        Legacy RF selectors refuse nonzero static strengths.
    delta_taper : float
        Saved relative static-strength correction for the new source-based
        models: ks, knl and ksl are multiplied by 1+delta_taper. This physical
        field setting remains active when radiation is disabled. The dedicated
        taper-analysis pass instead samples its reference lane once per parent.
    phase_taper, lag_taper : float
        Additional phase offset, in radians/degrees respectively. In the new
        source-based models this is a physical RF phase offset even when local
        radiation is disabled. The legacy compile-gated semantics are retained
        for the old RF selectors.
    num_factor_steps : int
        Number of outer generating-factor steps for the corresponding averaged
        construction, separate from its characteristic resolution num_kicks.
        Default is 1.

    Applied mean radiation reaction and rest-spin BMT transport currently
    support only the CPU scalar full phase-resolved M2/P16 RF-only map, through
    Line.configure_radiation(model="mean") and Line.configure_spin(True).
    Mean LL is evaluated on reacted stages; it is not an emitted-energy kick.
    Other source models, static fields, quantum radiation, TPSA effects and
    sliced effects are not yet supported. Frozen-coordinate and taper-preparation
    effects analysis and mean backtracking explicitly refuse.
    The original C2/C3/C4 and connected full-slow implementations still rebuild
    state-dependent characteristic jets inside tracking; ordinary throughput
    is not yet suitable for routine production use.

    The new phase-resolved models are experimental development interfaces.
    They do not establish physical-source, numerical-error or GPU qualification.
    '''.strip()

    _docstring_models = '''model : str
        Legacy RF selectors retain their existing behaviour. New physical
        selectors are "phase-resolved", "phase-resolved-paraxial",
        "phase-resolved-longitudinal", "phase-resolved-frozen",
        "phase-averaged-second-order", "phase-averaged-cubic" and
        "phase-averaged-quartic" and "phase-averaged-full-slow". The
        averaged selectors choose the CanonicalF4 second-counter-wave-order
        product and its distinct scalar-gradient cubic/quartic corrections.
        These already retain the full slow root. The separate full-slow
        selector uses its connected global_log4 boundary/body construction.
        "phase-averaged-uniform-quadratic-first-boundary" and
        "phase-averaged-uniform-quadratic" instead use uniform nonzero-field
        intervals, quadratic transport and first- or second-boundary words,
        with live unequal-wave amplitudes and counter phase. These are
        distinct source/transport assumptions, not the localized P16 source.
        Physical assumptions and numerical integration are separate choices.
        "phase-averaged-magnetic" is the centred straight solenoid/normal
        quadrupole model at joint RF order two and transverse quadratic order;
        it requires an explicit Gaussian RF profile and width. It is distinct
        from the full-root averaged model.
        "phase-averaged-envelope" consumes a separately prepared immutable
        RF2/transverse-quadratic scalar table with its fixed hardware and
        energy/position domains. It retains live state and RF amplitudes/phases.
        "phase-averaged-magnetic-fringe" uses explicit Gaussian RF and
        magnetic profiles in the original static donor gauge. The solenoid
        strength is a peak; normal/skew quadrupole strengths are finite-parent
        integrals. Its observation planes are inside the field, without
        separate hard-edge actions.
        "phase-averaged-dipole" uses a uniform integrated horizontal dipole
        and the fixed reference_px programme. "phase-averaged-moving-reference"
        instead propagates a prescribed initial reference_x/y/px/py through
        uniform solenoid and normal/skew quadrupole fields, with a separate
        reference integration count. The reference is never fitted to the
        tracked particle. This model also uses body-only observation planes.
    integrator : str
        The continuous phase-resolved models use "gauss2" (two-stage Gauss
        collocation) when "adaptive" is selected. The frozen voltage-action
        model uses "geometric-kick-drift". Legacy indices are unchanged.
        The uniform quadratic and Gaussian averaged models use "midpoint"
        when adaptive is chosen.
    num_kicks : int
        Fixed parent integration resolution. For the continuous resolved
        models this is the number of Gauss cells (automatic: 64). The averaged
        models use it for their parent characteristic; num_factor_steps sets
        the separate outer-factor resolution. The frozen model uses the
        requested geometric word, with its model-specific automatic choice.
        Gaussian magnetic/dipole/moving-reference constructions use it for
        RF-source integration and num_factor_steps for body/boundary words.
        The prepared-envelope model instead uses these two counts for its
        body and boundary midpoint words respectively.
        Legacy selectors keep their existing automatic resolution rules.
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
        'envelope_table': xo.Ref(EnvelopeTableData),
        'magnetic_profile': xo.Int64,
        'magnetic_profile_sigma': FloatOrTpsa,
        'reference_px': FloatOrTpsa,
        'reference_x': FloatOrTpsa,
        'reference_y': FloatOrTpsa,
        'reference_py': FloatOrTpsa,
        'num_reference_steps': xo.Int64,
        'rf_field_order': xo.Int64,
        'ks': FloatOrTpsa,
        'delta_taper': FloatOrTpsa,
        'radiation_flag': xo.Int64,
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
        'num_factor_steps': xo.Int64,
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
        'envelope_table': '_envelope_table',
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

    _noexpr_fields = _NOEXPR_FIELDS | {'rf_profile', 'magnetic_profile', 'envelope_table'}

    def __init__(self, **kwargs):

        if '_xobject' in kwargs and kwargs['_xobject'] is not None:
            self.xoinitialize(**kwargs)
            return

        model = kwargs.pop('model', None)
        envelope_table = kwargs.pop('envelope_table', None)
        if isinstance(envelope_table, dict):
            envelope_table = EnvelopeTableData.from_dict(envelope_table)
        if envelope_table is not None and not isinstance(envelope_table, EnvelopeTableData):
            raise TypeError('envelope_table must be EnvelopeTableData or None')
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
            profile = ('uniform' if model in (
                'phase-resolved-frozen',
                'phase-averaged-uniform-quadratic-first-boundary',
                'phase-averaged-uniform-quadratic') else 'p16')
        try:
            profile_id = _CAVITY_RF_PROFILE_IDS[profile]
        except KeyError:
            raise ValueError(f'Invalid Cavity rf_profile: {profile}') from None
        sigma = kwargs.pop('rf_profile_sigma', None)
        center = kwargs.pop('rf_profile_center', 0.)
        requested_model = _CAVITY_MODEL_IDS.get(model, 0)
        if requested_model in (20, 21, 22, 23, 24) and profile != 'gaussian':
            raise ValueError(f'{model} requires an explicit '
                             "rf_profile='gaussian' and positive width")
        if requested_model == 21 and envelope_table is None:
            raise ValueError('phase-averaged-envelope requires envelope_table')
        if requested_model in (13, 18, 19) and profile != 'uniform':
            raise ValueError(f'{model} requires rf_profile="uniform"')
        if requested_model < 10 and profile == 'gaussian':
            raise ValueError('Legacy Cavity projections do not support a '
                             'Gaussian RF profile')
        if profile == 'gaussian':
            try:
                width = float(_constant_part(sigma))
                midpoint = float(_constant_part(center))
            except (TypeError, ValueError):
                raise ValueError('Gaussian RF profile requires an explicit '
                                 'positive finite width in metres') from None
            if not isfinite(width) or width <= 0. or not isfinite(midpoint):
                raise ValueError('Gaussian RF profile requires a positive '
                                 'finite width and finite centre in metres')
        kwargs['rf_profile'] = profile_id
        kwargs['rf_profile_sigma'] = 0. if sigma is None else sigma
        kwargs['rf_profile_center'] = center
        magnetic_profile = kwargs.pop('magnetic_profile', 'uniform')
        try:
            magnetic_profile_id = _CAVITY_MAGNETIC_PROFILE_IDS[magnetic_profile]
        except KeyError:
            raise ValueError(f'Invalid Cavity magnetic_profile: {magnetic_profile}') from None
        magnetic_sigma = kwargs.pop('magnetic_profile_sigma', None)
        if requested_model == 24:
            kwargs.setdefault('edge_entry_active', 0)
            kwargs.setdefault('edge_exit_active', 0)
            if kwargs['edge_entry_active'] or kwargs['edge_exit_active']:
                raise ValueError('The moving-reference model requires inactive end faces')
        if requested_model == 22:
            if magnetic_profile != 'gaussian':
                raise ValueError('phase-averaged-magnetic-fringe requires an explicit '
                                 'Gaussian magnetic profile and width')
            kwargs.setdefault('edge_entry_active', 0)
            kwargs.setdefault('edge_exit_active', 0)
        if magnetic_profile == 'gaussian':
            if requested_model not in (10, 11, 12, 22):
                raise ValueError(f'{model} does not support a Gaussian magnetic profile')
            try:
                magnetic_width = float(_constant_part(magnetic_sigma))
            except (TypeError, ValueError):
                raise ValueError('Gaussian magnetic profile requires an explicit '
                                 'positive finite width in metres') from None
            if not isfinite(magnetic_width) or magnetic_width <= 0.:
                raise ValueError('Gaussian magnetic profile requires a positive finite width')
            if kwargs.get('edge_entry_active', 1) or kwargs.get('edge_exit_active', 1):
                raise ValueError('Gaussian magnetic profiles require inactive entry/exit faces')
        kwargs['magnetic_profile'] = magnetic_profile_id
        kwargs['magnetic_profile_sigma'] = (
            0. if magnetic_sigma is None else magnetic_sigma)
        # A Ref must belong to this element's buffer, known after allocation.
        kwargs['envelope_table'] = None
        order = kwargs.pop('order', 5)
        multipoles = self._prepare_multipolar_params(
            order, knl=kwargs.pop('knl', None), ksl=kwargs.pop('ksl', None))
        kwargs.update(multipoles)
        for name in ('knl', 'ksl'):
            # Handles are process-local owned storage, never serialized knobs.
            handles = kwargs.pop(name+'_tpsa', None)
            if handles is not None and any(handles):
                raise ValueError('Cavity TPSA array handles cannot be imported')
            kwargs[name+'_tpsa'] = [0]*len(kwargs[name])
        kwargs.setdefault('eta', 1.)
        kwargs.setdefault('counter_phase', 0.)
        kwargs.setdefault('calibration_beta', 1.)
        kwargs.setdefault('rf_field_order', 2)
        kwargs.setdefault('edge_entry_active', 1)
        kwargs.setdefault('edge_exit_active', 1)
        kwargs.setdefault('num_factor_steps', 1)
        reference_steps = kwargs.pop('num_reference_steps', 4)
        try:
            valid_reference_steps = (
                int(reference_steps) == reference_steps and reference_steps > 0)
        except (TypeError, ValueError, OverflowError):
            valid_reference_steps = False
        if not valid_reference_steps:
            raise ValueError('num_reference_steps must be a positive integer')
        kwargs['num_reference_steps'] = int(reference_steps)
        kwargs['wave_number'] = 0. if wave_number is None else wave_number
        kwargs['wave_number_from_frequency'] = int(from_frequency)
        kwargs['nominal_energy_gain'] = (
            0. if nominal_energy_gain is None else nominal_energy_gain)
        kwargs['nominal_energy_gain_from_rf'] = int(nominal_from_rf)

        self.xoinitialize(**kwargs)
        self.envelope_table = envelope_table

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
    def envelope_table(self):
        # Ref dressing caches are not relocated by HybridClass.copy. Dress
        # the actual owned reference, rather than returning a cached wrapper
        # which can still point into the original element's buffer.
        raw = self._xobject.envelope_table
        if raw is None:
            return None
        table = EnvelopeTableData(_xobject=raw)
        table._movable = False
        return table

    @envelope_table.setter
    def envelope_table(self, value):
        if value is None:
            self._xobject.envelope_table = None
            self.__dict__.pop('_dressed_envelope_table', None)
            return
        if isinstance(value, dict):
            value = EnvelopeTableData.from_dict(value, _buffer=self._buffer)
        if not isinstance(value, EnvelopeTableData):
            raise TypeError('envelope_table must be EnvelopeTableData or None')
        if value._buffer is not self._buffer:
            value = value.copy(_buffer=self._buffer)
        self._xobject.envelope_table = value._xobject
        value._movable = False
        self.__dict__.pop('_dressed_envelope_table', None)

    def _copy_owned_xobject(self, _context=None, _buffer=None, _offset=None):
        """Relocate numeric storage plus the owned immutable table graph.

        A generic Ref-bearing Struct copy reads RawUnions as values. For
        active TPSA fields that would decode handles without their owner.
        Allocate zero union slots, then copy their eight raw bytes exactly;
        the owning Cavity retains its Python scalar/array handle lifetimes.
        """
        source = self._xobject
        values = {}
        union_names = []
        for field in self._XoStruct._fields:
            if getattr(field.ftype, '_is_float_or_tpsa', False):
                values[field.name] = 0.
                union_names.append(field.name)
            elif field.name == 'envelope_table':
                values[field.name] = None
            else:
                values[field.name] = getattr(source, field.name)
        target = self._XoStruct(_context=_context, _buffer=_buffer,
                                _offset=_offset, **values)
        for name in union_names:
            target._buffer.update_from_xbuffer(target._get_offset(name),
                source._buffer, source._get_offset(name), FloatOrTpsa._size)
        table = self.envelope_table
        if table is not None:
            # A new element owns an independent payload. Copying into the
            # selected buffer also fixes every relative reference offset.
            table = table.copy(_buffer=target._buffer)
            target.envelope_table = table._xobject
        return target

    def move(self, _context=None, _buffer=None, _offset=None):
        # Respect ordinary Env/parent ownership; only the owned payload Ref
        # is made relocatable, not arbitrary externally referenced objects.
        if not self._movable and not self._force_moveable:
            raise MemoryError('This Cavity lives within another object; make a copy.')
        target_context = _buffer.context if _buffer is not None else _context
        if (getattr(self._xobject, '_tpsa_enabled', 0)
                and target_context is not None
                and not isinstance(target_context, xo.ContextCpu)):
            raise NotImplementedError('TPSA-enabled Cavity storage requires a CPU context')
        self._xobject = self._copy_owned_xobject(
            _context=_context, _buffer=_buffer, _offset=_offset)
        self.__dict__.pop('_dressed_envelope_table', None)

    def copy(self, _context=None, _buffer=None, _offset=None):
        if getattr(self._xobject, '_tpsa_enabled', 0):
            raise NotImplementedError('Copying TPSA-enabled beam elements is not supported')
        if _context is None and _buffer is None:
            _context = self._context
        target = self._copy_owned_xobject(
            _context=_context, _buffer=_buffer, _offset=_offset)
        result = self.__class__.__new__(self.__class__)
        for name, value in self.__dict__.items():
            if name in ('_xobject', '_dressed_envelope_table'):
                continue
            result.__dict__[name] = value.copy() if hasattr(value, 'copy') else value
        result._xobject = target
        result._movable = True
        if hasattr(self, 'prototype'):
            result.prototype = self.prototype
        return result

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
        # Preserve the explicit source even when p16 has raw enum value zero.
        # Inference is a constructor convenience, never a serialization rule.
        result['rf_profile'] = self.rf_profile
        result['rf_profile_sigma'] = self.rf_profile_sigma
        result['rf_profile_center'] = self.rf_profile_center
        result['magnetic_profile'] = self.magnetic_profile
        result['magnetic_profile_sigma'] = self.magnetic_profile_sigma
        for name in ('reference_x', 'reference_y', 'reference_px',
                     'reference_py', 'num_reference_steps'):
            result[name] = getattr(self, name)
        table = self.envelope_table
        result['envelope_table'] = (None if table is None else
            table.copy(_context=xo.context_default).to_dict())
        # The xobject zero-elision rule must not replace explicit zeros by
        # nonzero Python constructor defaults on reload. This includes disabled
        # physical faces and eta=0 (a one-wave source), not just invalid inputs.
        for name in ('eta', 'calibration_beta', 'rf_field_order',
                     'edge_entry_active', 'edge_exit_active',
                     'num_factor_steps', 'order'):
            result[name] = getattr(self, name)
        for name in ('knl_tpsa', 'ksl_tpsa', '_knl_tpsa', '_ksl_tpsa'):
            result.pop(name, None)
        # Do not inherit the standard multipole mixin's approximate-zero
        # elision for these simultaneous fields. Preserve every represented
        # coefficient, and the actual allocation used by indexed controls.
        result['knl'] = self._context.nparray_from_context_array(self._knl).tolist()
        result['ksl'] = self._context.nparray_from_context_array(self._ksl).tolist()
        return result

    @classmethod
    def from_dict(cls, dct, **kwargs):
        # A later order change can deliberately leave a larger allocated
        # array. Constructor padding infers its maximum order from length;
        # restore the saved active order after allocating that same storage.
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
