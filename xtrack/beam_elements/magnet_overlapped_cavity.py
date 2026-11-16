# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

from copy import copy

import numpy as np
import xobjects as xo
import xtrack as xt

from ..base_element import FloatOrTpsa
from ._common import _HasKnlKsl, _docstring_general_notes, _for_docstring_alignment
from .cavity import Cavity, _CAVITY_MODELS, _constant_part



class MagnetOverlappedCavity(_HasKnlKsl, Cavity):

    _docstring_start = Cavity._docstring_start.replace(
        'RF cavity element.',
        'RF cavity overlapped with static magnetic fields: a solenoid, and normal\n'
        '    and skew multipoles.', 1) + '''
    ks : float
        Solenoid strength in 1/m. Normal and skew multipoles use `knl`, `ksl`.
    The static fields are uniform along the cavity.'''

    _docstring_models = '''model : str
        "adaptive" (default) is "longitudinal-kick" for now. "longitudinal-kick"
        tracks the static fields as a straight magnet with an exact drift, with
        the RF energy kick added to each of the magnet's kicks, and its faces as
        those of a Quadrupole or UniformSolenoid. The experimental physical
        models, "_phase-resolved", "_phase-averaged", "_phase-resolved-paraxial"
        and "_phase-resolved-frozen", are those of Cavity.
    integrator : str
        As for Cavity; for "longitudinal-kick" it places the magnet's kicks.
    num_kicks : int
        Number of kicks, or of integration cells for the physical models.'''

    __doc__ = '\n    '.join([_docstring_start, _docstring_models,
        _HasKnlKsl._for_docstring,
        _for_docstring_alignment, '\n',
        _docstring_general_notes, '\n\n'])

    _xofields = {
        # A Field stores its struct offset, so Cavity's cannot be shared.
        **{name: copy(field) for name, field in Cavity._xofields.items()},
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

    _skip_in_to_dict = [*Cavity._skip_in_to_dict, '_order', 'inv_factorial_order',
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

    # The longitudinal kicks with the magnet's own body and faces, and the physical models.
    _MODELS = {0: 'adaptive', 1: 'longitudinal-kick',
               **{index: name for index, name in _CAVITY_MODELS.items() if index >= 10}}
    _ADAPTIVE_MODEL = 1

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

    @Cavity.model.setter
    def model(self, value):
        index = next((key for key, name in self._MODELS.items() if name == value), None)
        if (index is not None and (self._ADAPTIVE_MODEL if index == 0 else index) < 10
                and (np.any(self._knl_tpsa) or np.any(self._ksl_tpsa))):
            raise ValueError('TPSA knobs on knl and ksl need a physical model: the '
                             'longitudinal-kick model tracks their constant parts')
        Cavity.model.fset(self, value)

    def _present(self, name):
        """Entries of knl or ksl that are set: a nonzero value or a knob's series."""
        array = self._context.nparray_from_context_array
        return ((array(getattr(self, '_' + name)) != 0.)
                | (array(getattr(self, '_' + name + '_tpsa')) != 0))

    def _check_physical_model(self, tracking=False):
        super()._check_physical_model(tracking=tracking)
        if self._resolved_model in (11, 12):
            normal, skew = self._present('knl'), self._present('ksl')
            solenoid = _constant_part(self.ks) != 0.
            others = np.any(normal[np.arange(len(normal)) != 1]) or np.any(skew)
            if others or (solenoid and (np.any(normal) or self.edge_entry_active
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

    @property
    def _thin_slice_class(self):
        return xt.ThinSliceMagnetOverlappedCavity

    @property
    def _thick_slice_class(self):
        return xt.ThickSliceMagnetOverlappedCavity

    @property
    def _drift_slice_class(self):
        return xt.DriftSliceMagnetOverlappedCavity
