# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #
"""Static installation errors, independent of dynamic source seeds."""
from dataclasses import dataclass
import hashlib

import numpy as np

from ..table import Table
from .motion import COORDINATES, frozen_array


@dataclass
class RigidOffset:
    """Static parent-local displacement and survey angles [m, rad]."""
    shift_x: float = 0.
    shift_y: float = 0.
    shift_s: float = 0.
    rot_x_rad: float = 0.
    rot_y_rad: float = 0.
    rot_s_rad: float = 0.

    def values(self):
        values = np.array([getattr(self, name) for name in COORDINATES], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError('Static installation offsets must be finite')
        return values


@dataclass(frozen=True)
class InstallationErrors:
    """One immutable installation/correction case; not a dynamic motion record."""
    names: tuple
    values: object
    seed: int | None = None

    def __post_init__(self):
        names = (self.names,) if isinstance(self.names,str) else tuple(self.names)
        values = frozen_array(self.values)
        if values.shape != (len(names), 6) or len(set(names)) != len(names):
            raise ValueError('Unique body names must match six installation coordinates')
        object.__setattr__(self, 'names', names)
        object.__setattr__(self, 'values', values)

    def apply(self, manager):
        """Set the installation layer only; call manager.apply for tracking fields."""
        if not set(self.names).issubset(manager.bodies):
            raise ValueError('Installation case contains unknown bodies')
        for name, values in zip(self.names, self.values):
            for field, value in zip(COORDINATES, values):
                setattr(manager[name].installation, field, value)

    def corrected(self, correction):
        """Return a new static case without changing the as-installed record.

        correction is (bodies,6) in the same local small-error coordinates. This
        operation represents alignment trims, not arbitrary finite SE(3) addition.
        """
        correction = frozen_array(correction)
        if correction.shape != self.values.shape:
            raise ValueError('Correction must match installation body coordinates')
        return InstallationErrors(self.names, self.values+correction, self.seed)

    def get_table(self):
        return Table(dict(name=np.array(self.names), **{
            field: self.values[:, i].copy() for i, field in enumerate(COORDINATES)}))


def generate_installation_errors(manager, *, seed, bodies=None,
        translation_sigma_m=(0., 0., 0.), rotation_sigma_rad=(0., 0., 0.),
        distribution='gaussian', truncate_sigma=3., groups=None):
    """Generate static local installation errors with stable body/group identities.

    By default select root bodies, so child errors are not accidentally added at
    every hierarchy level. Gaussian sigma is the untruncated parent distribution;
    rejection truncation reduces the realized RMS. Uniform uses ±sqrt(3)*sigma.
    A named group shares one six-coordinate draw in each body's local axes.
    Reordering or selecting bodies leaves their draws unchanged. Grouping is a
    stated engineering rule, not an inferred spatial ground correlation.
    """
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('Use a nonnegative integer installation seed')
    names = tuple(name for name, body in manager.bodies.items() if body.parent is None) if bodies is None else (
        (bodies,) if isinstance(bodies,str) else tuple(bodies))
    if not names or len(set(names)) != len(names) or not set(names).issubset(manager.bodies):
        raise ValueError('Select unique existing installation bodies')
    sigma = np.array([*translation_sigma_m, *rotation_sigma_rad], dtype=float)
    if sigma.shape != (6,) or not np.all(np.isfinite(sigma)) or np.any(sigma < 0):
        raise ValueError('Use three nonnegative finite translation and rotation sigmas')
    if distribution not in ('gaussian', 'uniform'):
        raise ValueError('Use gaussian or uniform installation distribution')
    if truncate_sigma is not None and (not np.isfinite(truncate_sigma) or truncate_sigma < .1):
        raise ValueError('Use truncate_sigma >= 0.1, or None')
    owners = {}
    for group, members in (groups or {}).items():
        for name in members:
            if name not in names or name in owners:
                raise ValueError('Installation groups must select disjoint included bodies')
            owners[name] = 'group:'+str(group)
    result = []
    for name in names:
        key = owners.get(name, 'body:'+name)
        identity = np.frombuffer(hashlib.sha256(key.encode()).digest()[:16], dtype='<u4').tolist()
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), *identity]))
        if distribution == 'uniform':
            draw = rng.uniform(-np.sqrt(3), np.sqrt(3), 6)
        else:
            draw = rng.standard_normal(6)
            if truncate_sigma is not None:
                mask = np.abs(draw) > truncate_sigma
                while mask.any():
                    draw[mask] = rng.standard_normal(mask.sum())
                    mask = np.abs(draw) > truncate_sigma
        result.append(sigma*draw)
    return InstallationErrors(names, result, int(seed))
