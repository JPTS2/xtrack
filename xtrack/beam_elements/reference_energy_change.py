# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

from ..base_element import BeamElement
import xobjects as xo

class ReferenceEnergyChange(BeamElement):

    '''Beam element setting the reference momentum to an absolute value.

    Not compatible with cavities in `absolute_time` mode, as for
    `ReferenceEnergyIncrease`: for a ramp with such cavities, attach an
    `EnergyProgram` to the line.

    Parameters
    ----------
    p0c : float
        New reference momentum in eV/c. Default is ``0``.

    '''

    _xofields = {
        'p0c': xo.Float64}

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/referenceenergychange.h"',
    ]

    allow_rot_and_shift = False
