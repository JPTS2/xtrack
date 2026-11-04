# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

from ..base_element import BeamElement
import xobjects as xo

class ReferenceEnergyIncrease(BeamElement):

    '''Beam element modeling a change of reference energy (acceleration,
    deceleration).

    Conventional relative-delay updates are retained by default. Set
    ``line.use_time_based_zeta_update = True`` to preserve absolute arrival
    time during an explicit reference-momentum programme. EnergyProgram and
    SyncTime select that convention automatically when needed.

    The clock advances by the reference particle's lap time, with each segment at
    its own speed. That lap time is found at the start of a tracking call and
    kept for all its turns: track one turn per call when `time_s` matters.

    Parameters
    ----------
    Delta_p0c : float
        Change in reference momentum p0c, in eV. Default is ``0``.

    '''

    _xofields = {
        'Delta_p0c': xo.Float64}

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/referenceenergyincrease.h"',
    ]

    has_backtrack = True
    allow_rot_and_shift = False
