# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

from ..base_element import BeamElement
from ..general import DEPRECATION_INFO_PREP_1_0
from warnings import warn
import xobjects as xo
import xtrack as xt
from ._common import (
    _HasIntegrator,
    _HasKnlKsl,
    _HasModelRF,
    _docstring_general_notes,
    _for_docstring_alignment,
)

class RFMultipole(_HasKnlKsl, _HasModelRF, _HasIntegrator, BeamElement):

    _docstring_start = \
    """Beam element modeling a modulated multipole, with strengths dependent
    on the z coordinate. It is a thin kick with `length` zero (default); with a
    finite length the integrated strengths and voltage are distributed in kicks
    between drifts, with no field at the ends.

    Parameters
    ----------
    frequency : float
        Frequency in Hertz. Default is ``0``.
    knl : array
        Integrated strength of the normal rf-multipole components in units of m^-n.
    ksl : array
        Integrated strength of the skew rf-multipole components in units of m^-n.
    order : int
        Order of the multipole. If not provided, it will be inferred from knl and/or ksl.
    phase_n : array
        Phase of the normal components in radians.
    phase_s : array
        Phase of the skew components in radians.
    pn : array
        Deprecated. Phase of the normal components in degrees.
    ps : array
        Deprecated. Phase of the skew components in degrees.
    voltage : float
        Longitudinal voltage. Default is ``0``.
    phase : float
        Longitudinal phase in radians seen by the reference particle. Default is ``0``.
    lag : float
        Deprecated longitudinal phase in degrees, added to `phase`. Default is ``0``.
    absolute_time : bool
        If True, the phase is computed from the absolute time of the simulation,
        otherwise the element is synchronized with the arrival time of the
        reference particle (zeta=0). Default is False.
    length : float
        Length of the element in meters. Default is ``0`` (thin).
    """.strip()

    __doc__ = '\n    '.join([_docstring_start,
        _HasModelRF._for_docstring,
        _HasIntegrator._for_docstring.replace(
            'num_multipole_kicks', 'num_kicks').replace('multipole kicks', 'kicks'),
        _for_docstring_alignment, '\n',
        _docstring_general_notes, '\n\n'])

    _rf_plan = 'track_rf_plan'
    _xofields={
        'length': xo.Float64,
        'voltage': xo.Float64,
        'frequency': xo.Float64,
        'lag': xo.Float64,
        'phase': xo.Float64,
        'order': xo.Int64,
        'inv_factorial_order': xo.Float64,
        'knl': xo.Float64[:],
        'ksl': xo.Float64[:],
        'pn': xo.Float64[:],
        'ps': xo.Float64[:],
        'phase_n': xo.Float64[:],
        'phase_s': xo.Float64[:],
        'absolute_time': xo.Int64,
        'num_kicks': xo.Int64,
        'model': xo.Int64,
        'integrator': xo.Int64,
    }

    has_backtrack = True
    allow_loss_refinement = True

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/rfmultipole.h"',
    ]

    _skip_in_to_dict = ['_order', 'inv_factorial_order']  # defined by knl, etc.

    _rename = {
        'model': '_model',
        'integrator': '_integrator',
        'order': '_order',
        'lag': '_lag',
        'pn': '_pn',
        'ps': '_ps',
    }

    def __init__(self, **kwargs):

        if '_xobject' in kwargs and kwargs['_xobject'] is not None:
            self.xoinitialize(**kwargs)
            return

        pn = kwargs.get('pn')
        ps = kwargs.get('ps')
        lag = kwargs.pop('lag', None)
        model = kwargs.pop('model', None)
        integrator = kwargs.pop('integrator', None)
        if pn is not None:
            self._warn_if_deprecated_phase_is_nonzero(pn, 'pn', 'phase_n')
        if ps is not None:
            self._warn_if_deprecated_phase_is_nonzero(ps, 'ps', 'phase_s')

        super().__init__(**kwargs)

        if lag is not None:
            self.lag = lag

        if model is not None:
            self.model = model

        if integrator is not None:
            self.integrator = integrator

    @property
    def isthick(self):
        return self.length != 0

    @property
    def _thin_slice_class(self):
        return xt.ThinSliceRFMultipole

    @property
    def _thick_slice_class(self):
        return xt.ThickSliceRFMultipole

    @property
    def _drift_slice_class(self):
        return xt.DriftSliceRFMultipole

    @property
    def lag(self):
        return self._lag

    @lag.setter
    def lag(self, value):
        if value != 0:
            warn("`lag` (in degrees) is deprecated and will be removed in a future version. "
                 "Please use `phase` (in radians) instead. "
                 "Note that if both `lag` and `phase` are set, the effect is the sum of the two,"
                 " with `lag` converted to radians. "
                 + DEPRECATION_INFO_PREP_1_0,
                 FutureWarning, stacklevel=2)
        self._lag = value

    @property
    def pn(self):
        return self._buffer.context.linked_array_type.from_array(
            self._pn,
            mode='setitem_from_container',
            container=self,
            container_setitem_name='_pn_setitem')

    @pn.setter
    def pn(self, value):
        self.pn[:] = value

    def _pn_setitem(self, index, value):
        self._warn_if_deprecated_phase_is_nonzero(value, 'pn', 'phase_n')
        self._pn[index] = value

    @property
    def ps(self):
        return self._buffer.context.linked_array_type.from_array(
            self._ps,
            mode='setitem_from_container',
            container=self,
            container_setitem_name='_ps_setitem')

    @ps.setter
    def ps(self, value):
        self.ps[:] = value

    def _ps_setitem(self, index, value):
        self._warn_if_deprecated_phase_is_nonzero(value, 'ps', 'phase_s')
        self._ps[index] = value
