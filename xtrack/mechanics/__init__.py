# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #
"""Experimental rigid-body mechanics owned by an Xtrack Environment."""

from .sources import (
    Harmonic,
    PlaneWave,
    MeasuredMotion,
    DisplacementPSD,
    AngularPSD,
    SpectralGroundMotion,
    LocatedSource,
    GaussianRegion,
    RegionalGroundMotion,
    VectorGroundMotion,
)
from .motion import RigidMotion, FieldMotion, sample_field
from .responses import (
    BaseExcitedOscillator,
    LoadedTwoMass,
    TabulatedResponse,
    CausalResponse,
)
from .installation import RigidOffset, InstallationErrors
from .observation import project_to_bpm, BpmReading
from .model import MechanicsManager, LineMechanics, MechanicalTimeSeries
from .tables import MechanicsTable, MotionHistoryTable
