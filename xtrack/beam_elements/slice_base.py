import xobjects as xo

from ..base_element import BeamElement
from ..random import RandomUniformAccurate, RandomExponential
from ._common import SynchrotronRadiationRecord

ID_RADIATION_FROM_PARENT = 10

COMMON_SLICE_XO_FIELDS = {
    'radiation_flag': xo.Field(xo.Int64, default=ID_RADIATION_FROM_PARENT),
    'delta_taper': xo.Float64,
    'weight': xo.Float64,
    'slice_offset': xo.Float64,
}

def cavity_slice_xofields(parent_class):
    """Cavity slices store their offset as a fraction of the parent length."""
    fields = {'_parent': xo.Ref(parent_class), **COMMON_SLICE_XO_FIELDS,
              '_slice_offset_fraction': xo.Float64,
              '_legacy_slice_offset': xo.Float64}
    del fields['slice_offset']
    return fields


class _CavitySliceOffset:
    """Keep a source interval fixed in the parent when its length changes."""

    def __init__(self, slice_offset=None, **kwargs):
        if slice_offset is not None:
            parent = kwargs.get('_parent')
            if parent is None:
                # Serialized slices load before Line restores the parent link.
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

    def _on_parent_linked(self):
        # A reloaded slice carries the public absolute offset until linked.
        if self._slice_offset_fraction < 0:
            self.slice_offset = self._legacy_slice_offset

    def to_dict(self, **kwargs):
        dct = super().to_dict(**kwargs)
        dct.pop('_slice_offset_fraction', None)
        dct.pop('_legacy_slice_offset', None)
        if self.slice_offset != 0.:
            dct['slice_offset'] = self.slice_offset
        return dct


def _raise_if_parent_has_transverse_rotation(parent):
    if parent.rot_x_rad != 0 or parent.rot_y_rad != 0:
        raise ValueError(
            'Equivalent elements do not support parent `rot_x_rad` or '
            '`rot_y_rad` different from zero.')


class _SliceBase:

    allow_rot_and_shift = False
    _skip_in_to_dict = ['_parent']
    has_backtrack = True
    _force_moveable = True

    _depends_on = [RandomUniformAccurate, RandomExponential]
    _internal_record_class = SynchrotronRadiationRecord

    def to_dict(self, **kwargs):
        dct = BeamElement.to_dict(self, **kwargs)
        dct['parent_name'] = self.parent_name
        return dct

    @classmethod
    def from_dict(cls, dct, **kwargs):
        obj = super().from_dict(dct, **kwargs)
        obj.parent_name = dct['parent_name']
        return obj

    def copy(self, **kwargs):
        out = BeamElement.copy(self, **kwargs)
        out._parent = None
        out.parent_name = self.parent_name
        return out
