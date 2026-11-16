import xobjects as xo

from .bend import Bend
from .cavity import Cavity
from .field_expansion_cavity import FieldExpansionCavity
from .magnet_overlapped_cavity import MagnetOverlappedCavity
from .crab_cavity import CrabCavity
from .drift import Drift
from .drift_exact import DriftExact
from .multipole import Multipole
from .octupole import Octupole
from .quadrupole import Quadrupole
from .rbend import RBend
from .rf_multipole import RFMultipole
from .sextupole import Sextupole
from .slice_base import (_SliceBase, COMMON_SLICE_XO_FIELDS,
                         _CavitySliceOffset, cavity_slice_xofields)
from ..base_element import BeamElement


class _DriftSliceElementBase(_SliceBase):
    allow_rot_and_shift = False
    rot_and_shift_from_parent = False
    allow_loss_refinement = True
    isthick = True
    _inherit_strengths = False
    # A drift for line cleanup, unless the interval carries a physical model.
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

class DriftSliceCavity(_CavitySliceOffset, _DriftSliceElementBase, BeamElement):

    _xofields = cavity_slice_xofields(Cavity)

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/drift_slice_cavity.h"'
    ]

    @staticmethod
    def _carries_parent_field(parent):
        """Whether the slice tracks the parent's field, in its frame, not a drift."""
        return parent._resolved_model >= 10

    @property
    def behaves_like_drift(self):
        parent = self._parent
        return parent is not None and not self._carries_parent_field(parent)

    @property
    def _inherit_strengths(self):
        parent = self._parent
        return parent is not None and parent._resolved_model >= 10

    @property
    def _rot_and_shift_from_parent_for_metadata(self):
        parent = self._parent
        return parent is not None and self._carries_parent_field(parent)

    def get_equivalent_element(self):
        if self._carries_parent_field(self._parent):
            # Kept: a Drift would drop the parent's field.
            return self
        out = Drift(length=self._parent.length * self.weight,
                     _buffer=self._buffer)
        return out


class DriftSliceMagnetOverlappedCavity(DriftSliceCavity):

    isthick = True
    _xofields = cavity_slice_xofields(MagnetOverlappedCavity)

    @staticmethod
    def _carries_parent_field(parent):
        # Every model tracks the static fields and their faces.
        return True

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/drift_slice_magnet_overlapped_cavity.h"'
    ]


class DriftSliceFieldExpansionCavity(DriftSliceCavity):

    isthick = True
    _xofields = cavity_slice_xofields(FieldExpansionCavity)

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/drift_slice_field_expansion_cavity.h"'
    ]

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

class DriftSliceRFMultipole(_DriftSliceElementBase, BeamElement):

    _xofields = {'_parent': xo.Ref(RFMultipole), **COMMON_SLICE_XO_FIELDS}

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/drift_slice_rfmultipole.h"'
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
