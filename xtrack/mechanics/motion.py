# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #
"""Bounded host motion records and rigid support-contact reconstruction."""
from dataclasses import dataclass

import numpy as np

from ..survey import Frame
from ..table import Table


COORDINATES = ('shift_x', 'shift_y', 'shift_s', 'rot_x_rad', 'rot_y_rad', 'rot_s_rad')


def frozen_array(value):
    if np.iscomplexobj(value):
        raise ValueError('Motion samples must be real')
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError('Motion samples must be finite')
    return np.frombuffer(array.tobytes(), dtype=float).reshape(array.shape)


def checked_times(time_s, *, uniform=False):
    times = frozen_array(time_s)
    if times.ndim != 1 or len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError('Use at least two increasing finite times')
    if uniform:
        dt = (times[-1] - times[0]) / (len(times) - 1)
        tolerance = 64*np.finfo(float).eps*max(abs(times[0]), abs(times[-1]), dt)
        if tolerance > .01*dt or not np.allclose(np.diff(times), dt, rtol=0, atol=tolerance):
            raise ValueError('Filtering requires uniformly spaced times')
    return times


def angles_from_matrix(matrix):
    """Principal survey pitch/yaw/roll, retaining the rigid map near lock.

    Close to pitch = +/-pi/2, independently recovering yaw and roll from tiny
    matrix entries can destroy their well-determined combination. Recover yaw
    from the dominant first column once roll has been chosen. At exact lock
    individual yaw/roll remain nonunique; their reconstructed rigid map does not.
    """
    matrix = np.asarray(matrix)
    cosine_pitch = np.hypot(matrix[...,1,0],matrix[...,1,1])
    pitch = np.arctan2(matrix[...,1,2],cosine_pitch)
    yaw = np.arctan2(matrix[...,0,2],matrix[...,2,2])
    roll = np.arctan2(matrix[...,1,0],matrix[...,1,1])
    near_lock = cosine_pitch < .01
    if np.any(near_lock):
        coupled = (np.arctan2(-matrix[...,2,0],matrix[...,0,0])
                   - np.arctan2(np.sin(pitch)*np.sin(roll),np.cos(roll)))
        yaw = np.where(near_lock,np.arctan2(np.sin(coupled),np.cos(coupled)),yaw)
    return np.stack((pitch,yaw,roll),axis=-1)


def checked_frames(frames):
    matrices = np.array([f.matrix if isinstance(f, Frame) else f for f in frames], dtype=float)
    if matrices.ndim != 3 or matrices.shape[1:] != (4,4) or not len(matrices) or not np.all(np.isfinite(matrices)):
        raise ValueError('Use nonempty finite proper body frames')
    rotation = matrices[:,:3,:3]
    if (not np.allclose(matrices[:,3,:], [0,0,0,1], atol=1e-14, rtol=0)
            or not np.allclose(np.swapaxes(rotation,-1,-2) @ rotation, np.eye(3), atol=1e-12,rtol=0)
            or not np.allclose(np.linalg.det(rotation),1.,atol=1e-12,rtol=0)):
        raise ValueError('Use proper rigid body frames')
    return matrices


@dataclass(frozen=True)
class RigidMotion:
    """Prescribed body-local rigid coordinates, metres and survey radians.

    values has shape (times, targets, 6), or (times, 6) for one target. Rotations
    are survey pitch/yaw/roll, Ry(yaw) Rx(-pitch) Rz(roll), not a rotation vector.
    Replay interpolates these coordinates linearly without extrapolation. Use
    adequate sampling for angular motion; finite composition is done by Frames.
    A record does not contain deliberate static alignment or filter state.
    """
    time_s: object
    values: object
    target_names: tuple = ()
    frame = 'body_local_rigid'

    def __post_init__(self):
        times = checked_times(self.time_s)
        values = frozen_array(self.values)
        if values.ndim == 2:
            values = values[:, None, :]
        if values.ndim != 3 or values.shape[0] != len(times) or values.shape[2] != 6:
            raise ValueError('Rigid values must have shape (times, targets, 6)')
        names = (self.target_names,) if isinstance(self.target_names,str) else tuple(self.target_names)
        if names and (len(names) != values.shape[1] or len(set(names)) != len(names)):
            raise ValueError('Target names must uniquely match rigid channels')
        object.__setattr__(self, 'time_s', times)
        object.__setattr__(self, 'values', values)
        object.__setattr__(self, 'target_names', names)

    @property
    def nbytes(self):
        return self.time_s.nbytes + self.values.nbytes

    def evaluate(self, time_s, positions_m):
        times = np.asarray(time_s)
        if np.any(times < self.time_s[0]) or np.any(times > self.time_s[-1]):
            raise ValueError('Requested time lies outside the rigid motion record')
        if len(positions_m) != self.values.shape[1]:
            raise ValueError('Rigid motion channels must match source targets')
        flat = self.values.reshape(len(self.time_s), -1)
        return np.column_stack([np.interp(times, self.time_s, col) for col in flat.T]).reshape(
            len(times), self.values.shape[1], 6)

    def get_table(self, target=0):
        index = self.target_names.index(target) if isinstance(target, str) else target
        return Table(dict(time_s=self.time_s.copy(), **{
            name: self.values[:, index, i].copy() for i, name in enumerate(COORDINATES)
        }), index='time_s')

    def through(self, transfer, *, target_names=None):
        """Apply an offline periodic LTI response; no causal startup is implied."""
        return RigidMotion(self.time_s, transfer.apply(self.values, self.time_s),
                           self.target_names if target_names is None else target_names)

    def transport(self, *, from_frames, to_frames, target_names=None):
        """Transport prescribed rigid motion between nominal ports, exactly.

        Includes reference-point lever arms and axes. A subsequent transfer is
        defined at the destination port. No small-angle Euler subtraction; the
        response law itself remains a small-signal model if it is linear LTI.
        """
        from .local_deformations import rotation_increment
        origin = checked_frames(from_frames)
        target = checked_frames(to_frames)
        count = self.values.shape[1]
        if origin.shape != (count,4,4) or target.shape != origin.shape:
            raise ValueError('One source and destination frame is required per channel')
        local_increment = rotation_increment(*np.moveaxis(self.values[...,3:],-1,0))
        world_increment = origin[None,:,:3,:3] @ local_increment @ np.swapaxes(origin[None,:,:3,:3],-1,-2)
        offset = target[:,:3,3]-origin[:,:3,3]
        world_translation = (np.einsum('bij,tbj->tbi',origin[:,:3,:3],self.values[...,:3])
            + np.einsum('tbij,bj->tbi',world_increment,offset))
        values = np.zeros_like(self.values)
        values[...,:3] = np.einsum('bji,tbj->tbi',target[:,:3,:3],world_translation)
        output_increment = np.swapaxes(target[None,:,:3,:3],-1,-2) @ world_increment @ target[None,:,:3,:3]
        values[...,3:] = angles_from_matrix(np.eye(3)+output_increment)
        return RigidMotion(self.time_s,values,self.target_names if target_names is None else target_names)

    def select(self, targets, *, target_names=None):
        """Select named channels, preserving their order and physical samples."""
        if not self.target_names:
            raise ValueError('Named selection requires target_names')
        names = (targets,) if isinstance(targets,str) else tuple(targets)
        if not names:
            raise ValueError('Select named motion channels')
        indices = [self.target_names.index(name) for name in names]
        return RigidMotion(self.time_s,self.values[:,indices],names if target_names is None else target_names)

    def superpose(self, other):
        """Combine coframed prescribed motions: sum translation, compose rotation.

        Both records must use the same grid, target order and nominal ports.
        The other rotation left-composes this rotation. This is prescribed
        kinematic superposition, not a solution of coupled nonlinear mechanics.
        """
        from scipy.spatial.transform import Rotation
        if (not isinstance(other,RigidMotion) or self.values.shape != other.values.shape
                or self.target_names != other.target_names or not np.array_equal(self.time_s,other.time_s)):
            raise ValueError('Superpose motion at matching times, targets and nominal ports')
        rotations = []
        for record in (self,other):
            angles = record.values[...,[4,3,5]]*[1,-1,1]
            rotations.append(Rotation.from_euler('YXZ',angles.reshape(-1,3)).as_matrix().reshape(
                (*record.values.shape[:2],3,3)))
        values = self.values.copy()
        values[...,:3] += other.values[...,:3]
        values[...,3:] = angles_from_matrix(rotations[1] @ rotations[0])
        return RigidMotion(self.time_s,values,self.target_names)

    def to_dict(self):
        return dict(time_s=self.time_s.tolist(), values=self.values.tolist(),
                    target_names=list(self.target_names))


@dataclass(frozen=True)
class FieldMotion:
    """World XYZ displacement at fixed nominal receivers, in metres."""
    time_s: object
    positions_m: object
    values: object

    def __post_init__(self):
        times, positions, values = checked_times(self.time_s), frozen_array(self.positions_m), frozen_array(self.values)
        if positions.ndim != 2 or positions.shape[1] != 3 or values.shape != (len(times), len(positions), 3):
            raise ValueError('Use receiver XYZ positions and (times, receivers, 3) displacements')
        object.__setattr__(self, 'time_s', times)
        object.__setattr__(self, 'positions_m', positions)
        object.__setattr__(self, 'values', values)

    @property
    def nbytes(self):
        return self.time_s.nbytes + self.positions_m.nbytes + self.values.nbytes

    def to_rigid_motion(self, *, frames, target_names=()):
        """Project anchor translations into nominal body axes; infer no rotation."""
        matrices = checked_frames(frames)
        if matrices.shape != (len(self.positions_m), 4, 4) or not np.allclose(
                matrices[:, :3, 3], self.positions_m, atol=1e-10, rtol=0):
            raise ValueError('Body anchors must match sampled receivers')
        values = np.zeros((len(self.time_s), len(frames), 6))
        values[..., :3] = np.einsum('bji,tbj->tbi', matrices[:, :3, :3], self.values)
        return RigidMotion(self.time_s, values, target_names)

    def __add__(self, other):
        """Superpose co-sampled displacement fields, retaining their coherence."""
        if (not isinstance(other,FieldMotion) or not np.array_equal(self.time_s,other.time_s)
                or not np.array_equal(self.positions_m,other.positions_m)):
            raise ValueError('Superpose fields at identical times and receiver positions')
        return FieldMotion(self.time_s,self.positions_m,self.values+other.values)

    def fit_rigid_bodies(self, *, frames, feet_local_m, target_names=(), weights=None):
        """Weighted Kabsch fit of sampled contacts, not a force/dynamics model.

        Receivers are ordered body-major then foot-major. At least three
        noncollinear feet per body are needed to determine all rotations.
        Returns (RigidMotion, residual_rms_m), including deformation residuals.
        """
        frames = checked_frames(frames)
        feet = np.asarray(feet_local_m, dtype=float)
        if feet.ndim == 2:
            feet = np.broadcast_to(feet, (len(frames), *feet.shape))
        if (feet.ndim != 3 or feet.shape[0] != len(frames) or feet.shape[2] != 3
                or feet.shape[1] < 3 or frames.shape != (len(feet), 4, 4)
                or not np.all(np.isfinite(feet))):
            raise ValueError('Use proper body frames and at least three finite feet per body')
        count = feet.shape[1]
        weight = np.ones(count) if weights is None else np.asarray(weights, dtype=float)
        if weight.shape != (count,) or np.any(weight <= 0) or not np.all(np.isfinite(weight)):
            raise ValueError('Foot weights must be finite and positive')
        weight = weight / weight.sum()
        center = np.einsum('j,bjk->bk', weight, feet)
        centered = feet - center[:, None, :]
        if np.any(np.linalg.matrix_rank(centered) < 2):
            raise ValueError('Collinear feet do not determine all rigid rotations')
        receivers = np.einsum('bij,bfj->bfi', frames[:, :3, :3], feet) + frames[:, None, :3, 3]
        if self.positions_m.shape != (len(frames)*count, 3) or not np.allclose(
                receivers.reshape(-1, 3), self.positions_m, atol=1e-10, rtol=0):
            raise ValueError('Field receivers must match body-major nominal foot positions')
        displacements = self.values.reshape(len(self.time_s), len(frames), count, 3)
        local = feet[None] + np.einsum('bji,tbfj->tbfi', frames[:, :3, :3], displacements)
        moved_center = np.einsum('f,tbfj->tbj', weight, local)
        covariance = np.einsum('bfi,tbfj,f->tbij', centered, local - moved_center[:, :, None], weight)
        uu, _, vh = np.linalg.svd(covariance)
        correction = np.broadcast_to(np.eye(3), covariance.shape).copy()
        correction[..., 2, 2] = np.linalg.det(np.swapaxes(vh, -1, -2) @ np.swapaxes(uu, -1, -2))
        rotation = np.swapaxes(vh, -1, -2) @ correction @ np.swapaxes(uu, -1, -2)
        translation = moved_center - np.einsum('tbij,bj->tbi', rotation, center)
        fitted = np.einsum('tbij,bfj->tbfi', rotation, feet) + translation[:, :, None]
        residual = np.sqrt(np.einsum('tbfi,tbfi,f->tb', fitted-local, fitted-local, weight))
        return RigidMotion(self.time_s, np.concatenate((translation, angles_from_matrix(rotation)), axis=-1),
                           target_names), residual


def sample_field(source, *, time_s, positions_m, polarization=(0., 1., 0.), max_memory_bytes=128*1024**2):
    """Sample one joint field at all receivers; scalar sources use polarization.

    This bounded host reference does not allocate a GPU time-by-body history.
    DisplacementPSD has dense spatial cost; account for that workspace too.
    """
    times, positions = checked_times(time_s), frozen_array(positions_m)
    if positions.ndim != 2 or positions.shape[1] != 3 or not len(positions):
        raise ValueError('Use nonempty receiver XYZ positions')
    channels = len(positions)
    estimate = 112*len(times)*channels + getattr(
        source, 'workspace_bytes', lambda nt, nr: 0)(len(times), channels)
    if not isinstance(max_memory_bytes, int) or max_memory_bytes <= 0 or estimate > max_memory_bytes:
        raise ValueError(f'Estimated field workspace {estimate} bytes exceeds max_memory_bytes')
    values = np.asarray(source.evaluate(time_s=times, positions_m=positions))
    if values.shape == (len(times), channels):
        if getattr(source,'unit','m') != 'm':
            raise ValueError('A ground field requires displacement [m], not angular input')
        vector = np.asarray(polarization, dtype=float)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)) or np.linalg.norm(vector) == 0:
            raise ValueError('Use a finite nonzero world polarization')
        values = values[..., None] * (vector / np.linalg.norm(vector))
    elif getattr(source, 'frame', None) != 'world':
        raise ValueError('A vector displacement source must declare frame=world')
    return FieldMotion(times, positions, values)
