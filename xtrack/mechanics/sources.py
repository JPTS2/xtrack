# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #
"""Prescribed displacement sources for the experimental mechanics interface."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Harmonic:
    """Scalar peak displacement [m], using sin(2 pi f t + phase)."""
    amplitude: float
    frequency_hz: float
    phase_rad: float = 0.0
    frame = 'body_local'

    def __post_init__(self):
        if np.iscomplexobj([self.amplitude, self.frequency_hz, self.phase_rad]):
            raise ValueError('Use real amplitude/frequency and explicit phase_rad')
        if not np.all(np.isfinite([self.amplitude, self.frequency_hz, self.phase_rad])):
            raise ValueError('Harmonic parameters must be finite')
        if self.frequency_hz < 0:
            raise ValueError('frequency_hz must be nonnegative')

    def evaluate(self, time_s, positions_m):
        values = self.amplitude * np.sin(2*np.pi*self.frequency_hz*time_s + self.phase_rad)
        return np.broadcast_to(values[:, None], (len(time_s), len(positions_m))).copy()


@dataclass(frozen=True)
class PlaneWave:
    """World displacement field; direction and polarization are independent.

    Positions/amplitude are metres, frequency Hz, speed m/s. Positive phase
    travels along +direction. Sampled at nominal body anchors; no rotation or
    multi-foot response is inferred. This is prescribed motion, not elasticity.
    """
    amplitude: float
    frequency_hz: float
    direction: tuple
    speed_m_s: float
    polarization: tuple = (0., 1., 0.)
    origin_m: tuple = (0., 0., 0.)
    phase_rad: float = 0.
    frame = 'world'

    def __post_init__(self):
        if any(np.iscomplexobj(value) for value in (self.amplitude, self.frequency_hz,
                self.speed_m_s, self.phase_rad, self.direction, self.polarization, self.origin_m)):
            raise ValueError('Wave parameters must be real')
        if not np.all(np.isfinite([self.amplitude, self.frequency_hz,
                                  self.speed_m_s, self.phase_rad])):
            raise ValueError('Wave parameters must be finite')
        if self.frequency_hz < 0 or self.speed_m_s <= 0:
            raise ValueError('Use nonnegative frequency and positive speed')
        for name in ('direction', 'polarization', 'origin_m'):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                raise ValueError(f'{name} must be a finite XYZ vector')
            if name != 'origin_m':
                norm = np.linalg.norm(value)
                if norm == 0:
                    raise ValueError(f'{name} must be nonzero')
                value = value / norm
            object.__setattr__(self, name, tuple(value))

    def evaluate(self, time_s, positions_m):
        delay = (positions_m - self.origin_m) @ np.asarray(self.direction) / self.speed_m_s
        values = self.amplitude * np.sin(
            2*np.pi*self.frequency_hz*(time_s[:, None] - delay) + self.phase_rad)
        return values[..., None] * np.asarray(self.polarization)


@dataclass(frozen=True)
class MeasuredMotion:
    """Scalar displacement samples [m]; explicit linear or previous-sample replay.

    A single channel is shared by all targets; (times, targets) values give
    explicitly ordered channels. No extrapolation or periodic wrap is allowed.
    """
    time_s: tuple
    displacement: tuple
    interpolation: str = 'linear'
    frame = 'body_local'

    def __post_init__(self):
        if np.iscomplexobj(self.time_s) or np.iscomplexobj(self.displacement):
            raise ValueError('Measured displacement and time must be real')
        times = np.asarray(self.time_s, dtype=float)
        values = np.asarray(self.displacement, dtype=float)
        if (times.ndim != 1 or len(times) < 2 or np.any(np.diff(times) <= 0)
                or not np.all(np.isfinite(times)) or values.ndim not in (1, 2)
                or values.shape[0] != len(times) or not np.all(np.isfinite(values))):
            raise ValueError('Use increasing finite times and matching displacement samples')
        if self.interpolation not in ('linear', 'previous'):
            raise ValueError("interpolation must be 'linear' or 'previous'")
        object.__setattr__(self, 'time_s', tuple(times))
        object.__setattr__(self, 'displacement', tuple(values) if values.ndim == 1
                           else tuple(tuple(row) for row in values))

    def evaluate(self, time_s, positions_m):
        times, values = np.asarray(self.time_s), np.asarray(self.displacement)
        if np.any(time_s < times[0]) or np.any(time_s > times[-1]):
            raise ValueError('Requested time lies outside the measured motion record')
        if values.ndim == 1:
            values = np.broadcast_to(values[:, None], (len(times), len(positions_m)))
        if values.shape[1] != len(positions_m):
            raise ValueError('Measured channels must match source targets')
        if self.interpolation == 'previous':
            return values[np.searchsorted(times, time_s, side='right') - 1].copy()
        return np.column_stack([np.interp(time_s, times, col) for col in values.T])


@dataclass(frozen=True)
class DisplacementPSD:
    """One-sided displacement PSD [m²/Hz] with a reproducible Gaussian record.

    Frequency samples [Hz] are linearly interpolated onto the requested uniform
    time grid's rFFT bins, zero outside the supplied band. DC is zero; the band
    must lie below Nyquist. Discrete expected mean square is df*sum(bin_density).
    Correlation, when requested, is exp(-Euclidean anchor distance/length),
    a real amplitude coherency, not magnitude-squared coherence or a wave.
    Dense spatial factorization is a small-system reference backend.
    For fixed N/dt/receivers/seed, Fourier phases use physical time origin zero;
    shifting the requested window does not restart the same waveform. Changing
    N or dt changes the finite periodic model. Reuse one computed record for
    tracking comparisons; compute_time_series never implicitly wraps a record.
    """
    frequency_hz: tuple
    density: tuple
    seed: int
    correlation_length_m: float | None = None
    frame = 'body_local'

    def __post_init__(self):
        if np.iscomplexobj(self.frequency_hz) or np.iscomplexobj(self.density):
            raise ValueError('PSD frequency and density must be real')
        f, s = np.asarray(self.frequency_hz, dtype=float), np.asarray(self.density, dtype=float)
        if (f.ndim != 1 or len(f) < 2 or s.shape != f.shape or f[0] < 0
                or np.any(np.diff(f) <= 0) or not np.all(np.isfinite(f))
                or not np.all(np.isfinite(s)) or np.any(s < 0)):
            raise ValueError('Use increasing frequencies and finite nonnegative PSD density')
        if isinstance(self.seed, (bool, np.bool_)) or not isinstance(self.seed, (int, np.integer)) or self.seed < 0:
            raise ValueError('seed must be a nonnegative integer')
        length = self.correlation_length_m
        if length is not None and (not np.isfinite(length) or length <= 0):
            raise ValueError('correlation_length_m must be finite and positive')
        object.__setattr__(self, 'frequency_hz', tuple(f))
        object.__setattr__(self, 'density', tuple(s))
        object.__setattr__(self, 'seed', int(self.seed))

    def evaluate(self, time_s, positions_m):
        count, channels = len(time_s), len(positions_m)
        if count < 2:
            raise ValueError('A PSD requires compute_time_series(time_s=...) before applying it')
        dt = (time_s[-1] - time_s[0]) / (count - 1)
        tolerance = 64*np.finfo(float).eps*max(abs(time_s[0]), abs(time_s[-1]), dt)
        if (dt <= 0 or tolerance >= .01*dt or not np.allclose(
                time_s, time_s[0] + dt*np.arange(count), rtol=0, atol=tolerance)):
            raise ValueError('PSD sampling requires uniformly spaced increasing times')
        f = np.fft.rfftfreq(count, dt)
        input_f, input_s = np.asarray(self.frequency_hz), np.asarray(self.density)
        if np.any(input_s[input_f >= f[-1]] > 0):
            raise ValueError('PSD band must lie below Nyquist; reduce the time step')
        s = np.interp(f, input_f, input_s, left=0., right=0.)
        s[0] = 0.
        # Require no power at the unpaired Nyquist endpoint in an even record.
        if count % 2 == 0:
            s[-1] = 0.
        if self.correlation_length_m is None:
            factor = np.eye(channels)
        else:
            distances = np.linalg.norm(positions_m[:, None] - positions_m[None, :], axis=-1)
            correlation = np.exp(-distances / self.correlation_length_m)
            eigenvalues, eigenvectors = np.linalg.eigh(correlation)
            tolerance = 128*np.finfo(float).eps*channels*max(1., eigenvalues[-1])
            eigenvalues[np.abs(eigenvalues) < tolerance] = 0.
            factor = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.))[None, :]
        rng = np.random.default_rng(self.seed)
        quadratures = rng.standard_normal((2, len(f), channels)) @ factor.T
        coefficients = (quadratures[0] + 1j*quadratures[1]) * np.sqrt(count/dt*s[:, None]/4)
        coefficients *= np.exp(2j*np.pi*f*time_s[0])[:, None]
        return np.fft.irfft(coefficients, n=count, axis=0)


SOURCE_CLASSES = {cls.__name__: cls for cls in (Harmonic, PlaneWave, MeasuredMotion, DisplacementPSD)}
