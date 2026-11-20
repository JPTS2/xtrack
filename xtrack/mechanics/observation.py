# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #
"""Geometric pickup readout in a moving electrical reference plane."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BpmReading:
    x: object
    y: object
    distance_m: object


def project_to_bpm(*, beam_frame, bpm_frame, x=0., y=0., px=0., py=0., delta=0.):
    """Intersect a beam ray with the pickup plane and return its local x/y.

    Frames map local (x,y,s) to world XYZ. px/py are momenta normalized to p0;
    longitudinal momentum is sqrt((1+delta)^2-px^2-py^2), with forward branch.
    This is a frozen geometric observation, not moving-boundary dynamics or
    BPM electronics. distance_m is the signed extra ray path from beam_frame.
    """
    inputs = np.asarray(np.broadcast_arrays(x, y, px, py, delta), dtype=float)
    if not np.all(np.isfinite(inputs)):
        raise ValueError('Beam ray coordinates must be finite')
    x, y, px, py, delta = inputs
    pz2 = (1+delta)**2-px**2-py**2
    if np.any(pz2 <= 0) or np.any(1+delta <= 0):
        raise ValueError('Beam ray needs positive forward longitudinal momentum')
    local_position = np.stack((x, y, np.zeros_like(x)), axis=-1)
    local_momentum = np.stack((px, py, np.sqrt(pz2)), axis=-1)
    # Transform once to pickup-local coordinates. Adding the beam's transverse
    # coordinates to a kilometre-scale world position would lose small signals.
    change = bpm_frame.matrix[:3, :3].T @ beam_frame.matrix[:3, :3]
    offset = bpm_frame.matrix[:3, :3].T @ (beam_frame.XYZ-bpm_frame.XYZ)
    origin = local_position @ change.T + offset
    direction = local_momentum @ change.T
    denominator = direction[..., 2]
    if np.any(np.abs(denominator) < 1e-14):
        raise ValueError('Beam ray is parallel to the pickup plane')
    advance = -origin[..., 2]/denominator
    local_hit = origin + advance[..., None]*direction
    return BpmReading(local_hit[..., 0], local_hit[..., 1], advance*(1+delta))
