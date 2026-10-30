import xobjects as xo

from .bend import Bend
from .cavity import Cavity
from .crab_cavity import CrabCavity
from .drift import Drift
from .drift_exact import DriftExact
from .multipole import Multipole
from .octupole import Octupole
from .quadrupole import Quadrupole
from .rbend import RBend
from .sextupole import Sextupole
from .slice_base import _SliceBase, COMMON_SLICE_XO_FIELDS
from ..base_element import BeamElement


class _DriftSliceElementBase(_SliceBase):
    allow_rot_and_shift = False
    rot_and_shift_from_parent = False
    allow_loss_refinement = True
    isthick = True
    _inherit_strengths = False
    # Match the historical Line cleanup classification. Specific intervals
    # carrying an active source override this property below.
    behaves_like_drift = True

class DriftSliceBend(_DriftSliceElementBase, BeamElement):

    _xofields = {'_parent': xo.Ref(Bend), **COMMON_SLICE_XO_FIELDS}

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/drift_slice_bend.h"'
    ]

    def get_equivalent_element(self):
        out = Drift(length=self._parent.length * self.weight,
                     _buffer=self._buffer)
        return out

class DriftSliceRBend(_DriftSliceElementBase, BeamElement):

    _xofields = {'_parent': xo.Ref(RBend), **COMMON_SLICE_XO_FIELDS}

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/drift_slice_rbend.h"'
    ]

    def get_equivalent_element(self):
        out = Drift(length=self._parent.length * self.weight,
                     _buffer=self._buffer)
        return out

    def track_frame(self, frame, backtrack=False):

        if self._parent.rbend_model == "straight-body":
            ll = self._parent.length_straight * self.weight
        else:
            ll = self._parent.length * self.weight

        if backtrack:
            ll *= -1

        frame.translate_s(ll)

class DriftSliceQuadrupole(_DriftSliceElementBase, BeamElement):

    _xofields = {'_parent': xo.Ref(Quadrupole), **COMMON_SLICE_XO_FIELDS}

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/drift_slice_quadrupole.h"'
    ]

    def get_equivalent_element(self):
        out = Drift(length=self._parent.length * self.weight,
                     _buffer=self._buffer)
        return out

class DriftSliceSextupole(_DriftSliceElementBase, BeamElement):

    _xofields = {'_parent': xo.Ref(Sextupole), **COMMON_SLICE_XO_FIELDS}

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/drift_slice_sextupole.h"'
    ]

    def get_equivalent_element(self):
        out = Drift(length=self._parent.length * self.weight,
                     _buffer=self._buffer)
        return out

class DriftSliceOctupole(_DriftSliceElementBase, BeamElement):

    _xofields = {'_parent': xo.Ref(Octupole), **COMMON_SLICE_XO_FIELDS}

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/drift_slice_octupole.h"'
    ]

    def get_equivalent_element(self):
        out = Drift(length=self._parent.length * self.weight,
                     _buffer=self._buffer)
        return out

class DriftSliceCavity(_DriftSliceElementBase, BeamElement):

    _track_local_particle_preflight = 'DriftSliceCavity_preflight'

    _xofields = {'_parent': xo.Ref(Cavity), **COMMON_SLICE_XO_FIELDS,
                 '_slice_offset_fraction': xo.Float64,
                 '_legacy_slice_offset': xo.Float64}
    del _xofields['slice_offset']

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/drift_slice_cavity.h"'
    ]

    def __init__(self, slice_offset=None, **kwargs):
        if slice_offset is not None:
            parent = kwargs.get('_parent')
            if parent is None:
                kwargs['_legacy_slice_offset'] = slice_offset
                kwargs['_slice_offset_fraction'] = -1.
            else:
                kwargs['_slice_offset_fraction'] = (
                    slice_offset / parent.length if parent.length else 0.)
        super().__init__(**kwargs)

    @property
    def slice_offset(self):
        if self._slice_offset_fraction < 0:
            return self._legacy_slice_offset
        return self._slice_offset_fraction * self._parent.length

    @slice_offset.setter
    def slice_offset(self, value):
        parent = self._parent
        if parent is None:
            self._legacy_slice_offset = value
            self._slice_offset_fraction = -1.
        else:
            self._slice_offset_fraction = (
                value / parent.length if parent.length else 0.)

    @property
    def behaves_like_drift(self):
        parent = self._parent
        return parent is not None and parent._model < 10

    @property
    def _inherit_strengths(self):
        parent = self._parent
        return parent is not None and parent._model >= 10

    @property
    def _rot_and_shift_from_parent_for_metadata(self):
        # New source intervals apply the parent frame manually in their
        # native helper. Do not enable the automatic legacy drift wrapper.
        parent = self._parent
        return parent is not None and parent._model >= 10

    def get_equivalent_element(self):
        if self._parent._model >= 10:
            # Retain this connected physical interval. An ordinary Drift
            # would erase the parent RF/static source during optimization.
            return self
        out = Drift(length=self._parent.length * self.weight,
                     _buffer=self._buffer)
        return out

class DriftSliceMultipole(_DriftSliceElementBase, BeamElement):

    _xofields = {'_parent': xo.Ref(Multipole), **COMMON_SLICE_XO_FIELDS}

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/drift_slice_multipole.h"'
    ]

    def get_equivalent_element(self):
        out = Drift(length=self._parent.length * self.weight,
                     _buffer=self._buffer)
        return out

class DriftSliceCrabCavity(_DriftSliceElementBase, BeamElement):

    _xofields = {'_parent': xo.Ref(CrabCavity), **COMMON_SLICE_XO_FIELDS}

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/drift_slice_crab_cavity.h"'
    ]

    def get_equivalent_element(self):
        out = Drift(length=self._parent.length * self.weight,
                     _buffer=self._buffer)
        return out

class DriftSlice(_DriftSliceElementBase, BeamElement):

    _xofields = {'_parent': xo.Ref(Drift), **COMMON_SLICE_XO_FIELDS}

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/drift_slice.h"'
    ]

    def get_equivalent_element(self):
        out = Drift(length=self._parent.length * self.weight,
                     _buffer=self._buffer)
        return out

class DriftExactSlice(_DriftSliceElementBase, BeamElement):

    _xofields = {'_parent': xo.Ref(DriftExact), **COMMON_SLICE_XO_FIELDS}

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/drift_exact_slice.h"'
    ]

    def get_equivalent_element(self):
        out = DriftExact(length=self._parent.length * self.weight,
                         _buffer=self._buffer)
        return out
