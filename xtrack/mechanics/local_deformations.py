# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #
"""Precision-preserving rigid deformations about nominal local frames.

An increment is ``(translation, rotation_minus_identity)``. Neither small
translations nor small rotations are recovered by subtracting world poses.
The matrices remain exact finite-rotation compositions, up to binary64
arithmetic; this is not a first-order rigid-motion approximation.
"""
import numpy as np


def compose_rotation(left, right):
    """Return ``(I + left) @ (I + right) - I`` without cancellation."""
    return left + right + left @ right


def rotation_increment(phi=0., theta=0., psi=0.):
    """Stable ``Ry(theta) Rx(-phi) Rz(psi) - I``; survey radians."""
    phi, theta, psi = np.broadcast_arrays(phi, theta, psi)
    sx, sy, sz = np.sin(phi), np.sin(theta), np.sin(psi)
    cx, cy, cz = -2*np.sin(phi/2)**2, -2*np.sin(theta/2)**2, -2*np.sin(psi/2)**2
    pitch, yaw, roll = (np.zeros(phi.shape+(3,3)) for _ in range(3))
    pitch[...,1,1], pitch[...,1,2] = cx, sx
    pitch[...,2,1], pitch[...,2,2] = -sx, cx
    yaw[...,0,0], yaw[...,0,2] = cy, sy
    yaw[...,2,0], yaw[...,2,2] = -sy, cy
    roll[...,0,0], roll[...,0,1] = cz, -sz
    roll[...,1,0], roll[...,1,1] = sz, cz
    return compose_rotation(compose_rotation(yaw, pitch), roll)


def compose(left, right):
    """Compose finite affine deformations in one local coordinate system."""
    displacement, increment = left
    translation, rotation = right
    return (displacement + translation + increment @ translation,
            compose_rotation(increment, rotation))


def inherit(deformation, parent_nominal, child_nominal):
    """Express a parent's deformation about its child's nominal origin."""
    translation, increment = deformation
    parent_axes = parent_nominal[:3, :3]
    change = parent_axes.T @ child_nominal[:3, :3]
    lever = parent_axes.T @ (child_nominal[:3, 3]-parent_nominal[:3, 3])
    return (change.T @ (translation + increment @ lever),
            change.T @ increment @ change)


def static_local(values, installation):
    """Compose the deliberate body adjustment and installation placement."""
    return compose((values[:3], rotation_increment(*values[3:])),
                   (installation[:3], rotation_increment(*installation[3:])))


def attachment_reference(attachment):
    """Exact adopted entry transform, without differencing survey origins."""
    from ..survey.misalignment_survey import get_entry_transform
    fields = ('shift_x', 'shift_y', 'shift_s', 'rot_x_rad', 'rot_y_rad',
              'rot_s_rad_no_frame', 'rot_shift_anchor')
    geometry = dict(zip(('length', 'angle', 'h', 'hxl', 'rot_s_rad'),
                        attachment['geometry']))
    geometry.pop('hxl')
    parameters = get_entry_transform(**dict(zip(fields, attachment['baseline'])), **geometry)
    shift = np.array([parameters.shift_x, parameters.shift_y, parameters.shift_s])
    rotation = np.eye(3) + rotation_increment(parameters.rot_x_rad,
        parameters.rot_y_rad, parameters.rot_s_rad_no_frame)
    return shift, rotation


def entry_affine(attachment, body_nominal, deformation):
    """Moved entry transform relative to the nominal element entry frame."""
    translation, increment = deformation
    baseline_shift, baseline_rotation = attachment_reference(attachment)
    reference = attachment['ref_start']
    axes = reference[:3, :3].T @ body_nominal[:3, :3]
    lever = (body_nominal[:3, :3].T @ (reference[:3, 3]-body_nominal[:3, 3])
             + axes.T @ baseline_shift)
    shift = baseline_shift + axes @ (translation + increment @ lever)
    rotation = baseline_rotation + (axes @ increment @ axes.T) @ baseline_rotation
    return shift, rotation


def lower(attachment, body_nominal, deformation):
    """Lower to entry fields using small local increments and exact inputs."""
    translation, increment = deformation
    if not np.any(translation) and not np.any(increment):
        # Preserve intentional shifts, rotations and anchors verbatim.
        return attachment['baseline'].copy()
    shift, rotation = entry_affine(attachment, body_nominal, deformation)
    from .motion import angles_from_matrix
    phi, theta, psi = angles_from_matrix(rotation)
    return np.array([*shift, phi, theta, psi, 0.])
