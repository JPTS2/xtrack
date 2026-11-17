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
from ..random import RandomUniformAccurate, RandomExponential
from ._common import SynchrotronRadiationRecord
import xtrack as xt
from ._common import (
    _HasIntegrator,
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
_CAVITY_INTEGRATORS = dict(_INDEX_TO_INTEGRATOR)
_CAVITY_INTEGRATORS[7] = '_geometric'
_CAVITY_INTEGRATOR_IDS = {
    value: key for key, value in _CAVITY_INTEGRATORS.items()}


def _constant_part(value):
    """Inspect a knob's base point without stripping its stored dependence."""
    return getattr(value, 'const_part', value)


_RADIAL_ORDER = 6
_RADIAL_ORDERS = (2, 4, 6, 8)


class _CavityControls:
    """Model, integrator and RF clock controls shared by the cavity elements."""

    _MODELS = _CAVITY_MODELS
    # The model "adaptive" (index 0) tracks with: the element's default, which may change.
    _ADAPTIVE_MODEL = 0
    _default_model = 'adaptive'
    _default_integrator = 'adaptive'

    @property
    def _resolved_model(self):
        return self._ADAPTIVE_MODEL if self._model == 0 else self._model

    @property
    def model(self):
        return self._MODELS[self._model]

    @model.setter
    def model(self, value):
        index = next((key for key, name in self._MODELS.items() if name == value), None)
        if index is None:
            raise ValueError(f'Invalid {type(self).__name__} model: {value}')
        if (self._ADAPTIVE_MODEL if index == 0 else index) < 10 and self._integrator == 7:
            raise ValueError('"_geometric" grades the cells of the physical models; the '
                             'longitudinal models take the magnet integrators')
        self._model = index

    def __setattr__(self, name, value):
        # Settings changed after construction are checked as at construction:
        # a physical model never meets settings it cannot track.
        if name.startswith('_') or not getattr(self, '_settings_checked', False):
            return super().__setattr__(name, value)
        previous = getattr(self, name, None)
        super().__setattr__(name, value)
        try:
            self._check_physical_model()
        except ValueError:
            super().__setattr__(name, previous)
            raise

    @classmethod
    def get_available_models(cls):
        return list(cls._MODELS.values())

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
            index = _CAVITY_INTEGRATOR_IDS[value]
        except KeyError:
            raise ValueError(f'Invalid Cavity integrator: {value}') from None
        if value == '_geometric' and self._resolved_model < 10:
            raise ValueError('"_geometric" grades the cells of the physical models; the '
                             'longitudinal models take the magnet integrators')
        self._integrator = index

    @staticmethod
    def get_available_integrators():
        return list(_CAVITY_INTEGRATOR_IDS)

    def _check_radial_order(self):
        if self.radial_order not in _RADIAL_ORDERS:
            raise ValueError(f'radial_order must be one of {_RADIAL_ORDERS}')
        if self.radial_order != _RADIAL_ORDER and self._resolved_model != 10:
            raise ValueError('radial_order applies to "_phase-resolved" only: '
                             'the other models keep their own radial order')

    def track(self, particles, *args, **kwargs):

        if _constant_part(self.harmonic) != 0:
            raise RuntimeError(f"{type(self).__name__} cannot be used in standalone tracking "
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


class Cavity(_CavityControls, BeamElement):

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
    not in TPSA tracking, in every model, slice and context, with the static
    fields and their faces. The static fields radiate as the magnets do, in
    every model, when the line is configured for radiation (mean, quantum or
    quantum-kick): their field across the motion over each cell, with a
    collinear recoil. The RF field radiates each wave's Lienard power, gamma^2
    [(E + v x B)^2 - (beta.E)^2], into the weak-dipole line at the wave's seen
    frequency (photon numbers 3(1 - 2x + 2x^2)/2 up to hbar nu / (1 - beta)),
    with the photon energies raised by the quiver strength; hard edges do not
    radiate. Settings a physical model cannot track are refused at
    construction and when changed later. A particle the field stops is lost
    with state XT_LOST_STOPPED_IN_CAVITY; settings found invalid only while
    tracking (a knob's value) lose it with XT_INVALID_CAVITY_SETTINGS.

    counter_wave_amplitude : float
        Amplitude of the counter-propagating RF wave over that of the
        co-propagating one: 0 is a travelling wave, 1 a standing wave, and the
        values between are partly reflected waves. Default is 1. The frozen
        model takes a travelling wave only, so it needs 0.
    counter_wave_phase : float
        Phase of the counter-propagating wave in radians. Default is 0.
    n_cells : int
        Number of pi-mode cells, each of length `length / n_cells`: they fix
        the spatial wave number to pi n_cells / length whatever the frequency,
        and `voltage` is the crest energy gain at their synchronous speed, or
        at c if that exceeds it. 0 (default) has no cells: the waves travel
        at c and `voltage` is the crest gain at c. `cell_length`,
        `wave_number` and `synchronous_beta` read the result.
    nominal_energy_gain : float or str
        Reference energy programme of the frozen model, in eV. It can be set to
        the string 'from_rf' (default), in which case it follows the RF settings.
    radial_order : int
        Power of the offset r to which "_phase-resolved" keeps the on-axis
        field's near-axis series: E_z to r^radial_order, E_r and c B_phi one
        order lower; 2, 4, 6 (default) or 8. For waves at c the interior
        series is exact at any order; slower waves and field maps gain from a
        higher one: r^6 is 50 times closer than r^4 to the exact field for
        protons at beta 0.6 10 mm off axis, for 16% more time.
    _rf_edge_entry_active, _rf_edge_exit_active : bool
        Experimental, to test the hard-edge assumption: with a face off, the
        physical models take the RF field to continue past that end, without its
        end-plane impulse. Default is True; a real field always ends.
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
        "uniform", "_geometric", "yoshida-4/6/8" (compositions of the symmetric
        step). Frozen model: "_geometric" (default, SAD's mesh and clock),
        "uniform" (equal cells, clock at the nodes: second order),
        "yoshida-4/6/8" (compositions of the uniform cell). A step that does
        not converge is retried as two half steps, recursively.
    num_kicks : int
        Number of integration cells when nonzero. With zero, "_phase-resolved"
        uses 16 per half RF wavelength and at least 64, "_phase-averaged" two
        per half RF wavelength,
        "_phase-resolved-paraxial" two per half RF wavelength for a travelling
        wave (counter_wave_amplitude = 0) and four otherwise, at least eight in
        all and at least one for each 2% of V/E, and the frozen model its own
        count.
    '''.strip()

    __doc__ = '\n    '.join([_docstring_start, _docstring_models,
        _for_docstring_alignment, '\n',
        _docstring_general_notes, '\n\n'])

    isthick = True
    has_backtrack = True
    allow_loss_refinement = True

    # The cavity kernels share the magnets' body, with its radiation record.
    _depends_on = [RandomUniformAccurate, RandomExponential]
    _internal_record_class = SynchrotronRadiationRecord

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
        'radial_order': xo.Field(xo.Int64, default=_RADIAL_ORDER),
        'nominal_energy_gain': FloatOrTpsa,
        'nominal_energy_gain_from_rf': xo.Field(xo.Int64, default=1),
        '_rf_edge_entry_active': xo.Field(xo.Int64, default=1),
        '_rf_edge_exit_active': xo.Field(xo.Int64, default=1),
        'absolute_time': xo.Int64,
        'num_kicks': xo.Int64,
        'model': xo.Int64,
        'integrator': xo.Int64,
        'radiation_flag': xo.Int64,
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

    _skip_in_to_dict = ['nominal_energy_gain_from_rf']

    _default_frequency = 0.0
    _default_harmonic = 0.0
    _default_nominal_energy_gain = 'from_rf'
    _default_counter_wave_amplitude = 1.

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
        n_cells = kwargs.pop('n_cells', 0)
        nominal_energy_gain = kwargs.pop('nominal_energy_gain', 'from_rf')
        kwargs.setdefault('counter_wave_amplitude', self._default_counter_wave_amplitude)

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
        self.nominal_energy_gain = nominal_energy_gain

        self._check_physical_model()
        self._settings_checked = True

    def _check_physical_model(self):
        """Raise for settings the physical models cannot track."""
        self._check_radial_order()
        if self._resolved_model < 10:
            return
        if self.num_kicks < 0:
            raise ValueError('num_kicks must be non-negative')
        if self.integrator == 'teapot':
            raise ValueError(f'{self.model} has symmetric steps: "teapot" '
                             'does not apply')
        if self._resolved_model == 13 and (
                _constant_part(self.counter_wave_amplitude) != 0.
                or self.n_cells != 0):
            raise ValueError('_phase-resolved-frozen requires a travelling '
                             'wave: set counter_wave_amplitude=0 and n_cells=0')

    @property
    def n_cells(self):
        return self._n_cells

    @n_cells.setter
    def n_cells(self, value):
        if not (isinstance(value, (int, float, np.integer, np.floating)) and isfinite(value)
                and value >= 0 and int(value) == value):
            raise ValueError('n_cells must be a non-negative integer')
        self._n_cells = int(value)

    @property
    def cell_length(self):
        return self.length / self.n_cells if self.n_cells else None

    @property
    def wave_number(self):
        if self.n_cells:
            return np.pi * self.n_cells / self.length
        return 2 * np.pi * _constant_part(self.frequency) / clight

    @property
    def synchronous_beta(self):
        if not self.n_cells:
            return 1.
        return 2 * _constant_part(self.frequency) * self.cell_length / clight

    @property
    def nominal_energy_gain(self):
        if self.nominal_energy_gain_from_rf:
            return 'from_rf'
        return self._nominal_energy_gain

    @nominal_energy_gain.setter
    def nominal_energy_gain(self, value):
        if isinstance(value, str):
            if value != 'from_rf':
                raise ValueError("nominal_energy_gain can only be set to 'from_rf' as a string")
            self.nominal_energy_gain_from_rf = 1
        else:
            self.nominal_energy_gain_from_rf = 0
            self._nominal_energy_gain = value

    @property
    def _thin_slice_class(self):
        return xt.ThinSliceCavity

    @property
    def _thick_slice_class(self):
        return xt.ThickSliceCavity

    @property
    def _drift_slice_class(self):
        return xt.DriftSliceCavity
