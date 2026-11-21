# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #
"""Prescribed motion sources for the experimental mechanics interface."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Harmonic:
    """Scalar peak displacement [m] or angle [rad], sin(2 pi f t + phase).

    unit must match the binding's translation or angular component.
    """

    amplitude: float
    frequency_hz: float
    phase_rad: float = 0.0
    unit: str = 'm'
    frame = 'body_local'

    def __post_init__(self):
        if self.unit not in ('m', 'rad'):
            raise ValueError("Harmonic unit must be 'm' or 'rad'")
        if np.iscomplexobj([self.amplitude, self.frequency_hz, self.phase_rad]):
            raise ValueError('Use real amplitude/frequency and explicit phase_rad')
        if not np.all(np.isfinite([self.amplitude, self.frequency_hz, self.phase_rad])):
            raise ValueError('Harmonic parameters must be finite')
        if self.frequency_hz < 0:
            raise ValueError('frequency_hz must be nonnegative')

    def evaluate(self, time_s, positions_m):
        values = self.amplitude * np.sin(
            2 * np.pi * self.frequency_hz * time_s + self.phase_rad
        )
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
    polarization: tuple = (0.0, 1.0, 0.0)
    origin_m: tuple = (0.0, 0.0, 0.0)
    phase_rad: float = 0.0
    frame = 'world'

    def __post_init__(self):
        if any(
            np.iscomplexobj(value)
            for value in (
                self.amplitude,
                self.frequency_hz,
                self.speed_m_s,
                self.phase_rad,
                self.direction,
                self.polarization,
                self.origin_m,
            )
        ):
            raise ValueError('Wave parameters must be real')
        if not np.all(
            np.isfinite(
                [self.amplitude, self.frequency_hz, self.speed_m_s, self.phase_rad]
            )
        ):
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
        delay = (
            (positions_m - self.origin_m) @ np.asarray(self.direction) / self.speed_m_s
        )
        values = self.amplitude * np.sin(
            2 * np.pi * self.frequency_hz * (time_s[:, None] - delay) + self.phase_rad
        )
        return values[..., None] * np.asarray(self.polarization)


@dataclass(frozen=True)
class MeasuredMotion:
    """Scalar samples [m or rad]; explicit linear or previous-sample replay.

    A single channel is shared by all targets; (times, targets) values give
    explicitly ordered channels. No extrapolation or periodic wrap is allowed.
    """

    time_s: tuple
    displacement: tuple
    interpolation: str = 'linear'
    unit: str = 'm'
    frame = 'body_local'

    def __post_init__(self):
        if self.unit not in ('m', 'rad'):
            raise ValueError("Measured motion unit must be 'm' or 'rad'")
        if np.iscomplexobj(self.time_s) or np.iscomplexobj(self.displacement):
            raise ValueError('Measured displacement and time must be real')
        times = np.asarray(self.time_s, dtype=float)
        values = np.asarray(self.displacement, dtype=float)
        if (
            times.ndim != 1
            or len(times) < 2
            or np.any(np.diff(times) <= 0)
            or not np.all(np.isfinite(times))
            or values.ndim not in (1, 2)
            or values.shape[0] != len(times)
            or not np.all(np.isfinite(values))
        ):
            raise ValueError(
                'Use increasing finite times and matching displacement samples'
            )
        if self.interpolation not in ('linear', 'previous'):
            raise ValueError("interpolation must be 'linear' or 'previous'")
        object.__setattr__(self, 'time_s', tuple(times))
        object.__setattr__(
            self,
            'displacement',
            tuple(values) if values.ndim == 1 else tuple(tuple(row) for row in values),
        )

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
    Bind one receiver set jointly. Separate bindings of the same FFT field are
    rejected because restarting its random receiver basis changes coherency.
    SpectralGroundMotion instead defines one field across receiver subsets.
    """

    frequency_hz: tuple
    density: tuple
    seed: int
    correlation_length_m: float | None = None
    frame = 'body_local'
    unit = 'm'

    def __post_init__(self):
        if np.iscomplexobj(self.frequency_hz) or np.iscomplexobj(self.density):
            raise ValueError('PSD frequency and density must be real')
        f, s = (
            np.asarray(self.frequency_hz, dtype=float),
            np.asarray(self.density, dtype=float),
        )
        if (
            f.ndim != 1
            or len(f) < 2
            or s.shape != f.shape
            or f[0] < 0
            or np.any(np.diff(f) <= 0)
            or not np.all(np.isfinite(f))
            or not np.all(np.isfinite(s))
            or np.any(s < 0)
        ):
            raise ValueError(
                'Use increasing frequencies and finite nonnegative PSD density'
            )
        if (
            isinstance(self.seed, (bool, np.bool_))
            or not isinstance(self.seed, (int, np.integer))
            or self.seed < 0
        ):
            raise ValueError('seed must be a nonnegative integer')
        length = self.correlation_length_m
        if length is not None and (not np.isfinite(length) or length <= 0):
            raise ValueError('correlation_length_m must be finite and positive')
        object.__setattr__(self, 'frequency_hz', tuple(f))
        object.__setattr__(self, 'density', tuple(s))
        object.__setattr__(self, 'seed', int(self.seed))

    def workspace_bytes(self, n_times, n_receivers):
        # FFT quadratures/coefficients plus the dense spatial reference path.
        # An uncorrelated record needs no identity covariance/factorization.
        return 80 * n_times * n_receivers + (
            80 * n_receivers**2 if self.correlation_length_m is not None else 0
        )

    def evaluate(self, time_s, positions_m):
        count, channels = len(time_s), len(positions_m)
        if count < 2:
            raise ValueError(
                'A PSD requires compute_time_series(time_s=...) before applying it'
            )
        dt = (time_s[-1] - time_s[0]) / (count - 1)
        tolerance = 64 * np.finfo(float).eps * max(abs(time_s[0]), abs(time_s[-1]), dt)
        if (
            dt <= 0
            or tolerance >= 0.01 * dt
            or not np.allclose(
                time_s, time_s[0] + dt * np.arange(count), rtol=0, atol=tolerance
            )
        ):
            raise ValueError('PSD sampling requires uniformly spaced increasing times')
        f = np.fft.rfftfreq(count, dt)
        input_f, input_s = np.asarray(self.frequency_hz), np.asarray(self.density)
        active_intervals = (input_s[:-1] + input_s[1:]) > 0
        band_end = np.max(input_f[1:][active_intervals], initial=0.0)
        nyquist = 0.5 / dt
        if (
            band_end > nyquist
            or np.interp(nyquist, input_f, input_s, left=0, right=0) > 0
        ):
            raise ValueError('PSD band must lie below Nyquist; reduce the time step')
        s = np.interp(f, input_f, input_s, left=0.0, right=0.0)
        s[0] = 0.0
        # Require no power at the unpaired Nyquist endpoint in an even record.
        if count % 2 == 0:
            s[-1] = 0.0
        if self.correlation_length_m is None:
            factor = None
        else:
            distances = np.linalg.norm(
                positions_m[:, None] - positions_m[None, :], axis=-1
            )
            correlation = np.exp(-distances / self.correlation_length_m)
            eigenvalues, eigenvectors = np.linalg.eigh(correlation)
            tolerance = 128 * np.finfo(float).eps * channels * max(1.0, eigenvalues[-1])
            eigenvalues[np.abs(eigenvalues) < tolerance] = 0.0
            factor = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0))[None, :]
        rng = np.random.default_rng(self.seed)
        quadratures = rng.standard_normal((2, len(f), channels))
        if factor is not None:
            quadratures = quadratures @ factor.T
        coefficients = (quadratures[0] + 1j * quadratures[1]) * np.sqrt(
            count / dt * s[:, None] / 4
        )
        coefficients *= np.exp(2j * np.pi * f * time_s[0])[:, None]
        return np.fft.irfft(coefficients, n=count, axis=0)


@dataclass(frozen=True)
class AngularPSD(DisplacementPSD):
    """Same one-sided convention as DisplacementPSD, density in rad²/Hz."""

    unit = 'rad'


@dataclass(frozen=True)
class SpectralGroundMotion:
    """Continuous stationary approximate field with bounded random modes.

    One-sided displacement PSD [m²/Hz]; exp(-Euclidean d/length) amplitude
    coherency in expectation. Frequencies are sampled from the normalized PSD;
    isotropic Cauchy wavevectors and shared phases define one field everywhere.
    Time chunks, receiver subsets/order and selected diagnostics do not change
    this field. Finite modes approximate the spectrum/covariance (not exact
    measured ground); mode rank and sampling error must be checked explicitly.
    This random feature model is prescribed displacement, not an elastic medium.
    """

    frequency_hz: tuple
    density: tuple
    seed: int
    correlation_length_m: float
    n_modes: int = 2048
    frame = 'body_local'
    unit = 'm'

    def __post_init__(self):
        specification = DisplacementPSD(
            self.frequency_hz, self.density, self.seed, self.correlation_length_m
        )
        if (
            not isinstance(self.n_modes, int)
            or isinstance(self.n_modes, bool)
            or self.n_modes < 16
        ):
            raise ValueError('Use at least 16 random modes')
        object.__setattr__(self, 'frequency_hz', specification.frequency_hz)
        object.__setattr__(self, 'density', specification.density)
        object.__setattr__(self, 'seed', specification.seed)
        rng = np.random.default_rng(self.seed)
        f, s = np.array(self.frequency_hz), np.array(self.density)
        area = 0.5 * (s[1:] + s[:-1]) * np.diff(f)
        variance = area.sum()
        if variance <= 0:
            raise ValueError('Spectral ground motion needs positive integrated PSD')
        cumulative = np.concatenate(([0.0], np.cumsum(area)))
        draw = rng.random(self.n_modes) * variance
        index = np.minimum(
            np.searchsorted(cumulative, draw, side='right') - 1, len(area) - 1
        )
        slope = (s[index + 1] - s[index]) / (f[index + 1] - f[index])
        local_area = draw - cumulative[index]
        root = np.sqrt(np.maximum(0.0, s[index] ** 2 + 2 * slope * local_area))
        denominator = s[index] + root
        offset = np.divide(
            2 * local_area, denominator, out=np.zeros_like(draw), where=denominator > 0
        )
        frequencies = f[index] + offset
        wavevectors = rng.standard_normal((self.n_modes, 3)) / (
            np.maximum(np.abs(rng.standard_normal(self.n_modes)), np.finfo(float).tiny)[
                :, None
            ]
            * self.correlation_length_m
        )
        for name, value in (
            ('_frequencies', frequencies),
            ('_wavevectors', wavevectors),
            ('_phases', rng.uniform(0.0, 2 * np.pi, self.n_modes)),
        ):
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        object.__setattr__(self, '_amplitude', np.sqrt(2 * variance / self.n_modes))

    def workspace_bytes(self, n_times, n_receivers):
        return 32 * self.n_modes * (n_times + n_receivers) + 16 * n_times * n_receivers

    def evaluate(self, time_s, positions_m):
        phase = np.asarray(positions_m) @ self._wavevectors.T + self._phases
        time_phase = 2 * np.pi * np.asarray(time_s)[:, None] * self._frequencies
        return self._amplitude * (
            np.sin(time_phase) @ np.cos(phase).T + np.cos(time_phase) @ np.sin(phase).T
        )


@dataclass(frozen=True)
class LocatedSource:
    """One latent displacement source with prescribed gain and travel delay.

    gain = exp(-distance/decay_length_m); speed [m/s] gives distance/speed delay.
    This is an explicit illustrative coupling law, not inferred pump/floor
    elasticity. The waveform is a scalar displacement, not an applied force.
    Measured input must include all required delayed prehistory.
    """

    waveform: object
    position_m: tuple
    decay_length_m: float
    speed_m_s: float
    polarization: tuple = (0.0, 1.0, 0.0)
    frame = 'world'

    def __post_init__(self):
        if any(
            np.iscomplexobj(value)
            for value in (
                self.position_m,
                self.polarization,
                self.decay_length_m,
                self.speed_m_s,
            )
        ):
            raise ValueError('Source geometry and propagation parameters must be real')
        position, polarization = (
            np.asarray(self.position_m, dtype=float),
            np.asarray(self.polarization, dtype=float),
        )
        if (
            position.shape != (3,)
            or polarization.shape != (3,)
            or not np.all(np.isfinite(position))
            or not np.all(np.isfinite(polarization))
            or np.linalg.norm(polarization) == 0
            or not np.all(np.isfinite([self.decay_length_m, self.speed_m_s]))
            or min(self.decay_length_m, self.speed_m_s) <= 0
            or getattr(self.waveform, 'frame', None) != 'body_local'
            or getattr(self.waveform, 'unit', 'm') != 'm'
        ):
            raise ValueError(
                'Use finite position, displacement waveform, polarization, positive length/speed'
            )
        object.__setattr__(self, 'position_m', tuple(position))
        object.__setattr__(
            self, 'polarization', tuple(polarization / np.linalg.norm(polarization))
        )

    def evaluate(self, time_s, positions_m):
        distance = np.linalg.norm(np.asarray(positions_m) - self.position_m, axis=1)
        signals = [
            self.waveform.evaluate(
                np.asarray(time_s) - d / self.speed_m_s,
                np.asarray(self.position_m)[None, :],
            )[:, 0]
            * np.exp(-d / self.decay_length_m)
            for d in distance
        ]
        return np.column_stack(signals)[..., None] * np.asarray(self.polarization)

    def to_dict(self):
        return dict(
            waveform=encode_source(self.waveform),
            position_m=self.position_m,
            decay_length_m=self.decay_length_m,
            speed_m_s=self.speed_m_s,
            polarization=self.polarization,
        )


@dataclass(frozen=True)
class GaussianRegion:
    """An independent excess displacement field with Gaussian amplitude gain."""

    source: object
    center_m: tuple
    sigma_m: float

    def __post_init__(self):
        if np.iscomplexobj(self.center_m) or np.iscomplexobj(self.sigma_m):
            raise ValueError('Regional geometry must be real')
        center = np.asarray(self.center_m, dtype=float)
        if (
            center.shape != (3,)
            or not np.all(np.isfinite(center))
            or not np.isfinite(self.sigma_m)
            or self.sigma_m <= 0
        ):
            raise ValueError('Use finite regional centers and positive spatial widths')
        object.__setattr__(self, 'center_m', tuple(center))


@dataclass(frozen=True)
class RegionalGroundMotion:
    """Background plus independent excess spectra with smooth spatial gains.

    Regional gain is exp(-r²/(2 sigma²)); its PSD contribution is gain²*S.
    Cross terms remain for shared background. Equal seeds are rejected for
    supposedly independent components. This is not arbitrary PSD interpolation.
    """

    background: object
    regions: tuple
    frame = 'body_local'
    unit = 'm'

    def __post_init__(self):
        sources = [self.background] + [r.source for r in self.regions]
        if any(
            getattr(s, 'frame', None) != 'body_local' or getattr(s, 'unit', 'm') != 'm'
            for s in sources
        ):
            raise ValueError('Regional components must be scalar displacement fields')
        seeds = [seed for source in sources for seed in _source_seeds(source)]
        if len(seeds) != len(set(seeds)):
            raise ValueError('Independent regional components require distinct seeds')
        for region in self.regions:
            if (
                np.asarray(region.center_m).shape != (3,)
                or not np.all(np.isfinite(region.center_m))
                or not np.isfinite(region.sigma_m)
                or region.sigma_m <= 0
            ):
                raise ValueError(
                    'Use finite regional centers and positive spatial widths'
                )
        object.__setattr__(self, 'regions', tuple(self.regions))

    def workspace_bytes(self, n_times, n_receivers):
        sources = [self.background] + [r.source for r in self.regions]
        return (
            max(
                (
                    getattr(s, 'workspace_bytes', lambda nt, nr: 0)(
                        n_times, n_receivers
                    )
                    for s in sources
                ),
                default=0,
            )
            + 24 * n_times * n_receivers
        )

    def evaluate(self, time_s, positions_m):
        background = np.asarray(self.background.evaluate(time_s, positions_m))
        # Providers may return cached or read-only samples. Own the accumulation
        # buffer and promote integer inputs without discarding complex values;
        # the sampling/binding boundary still validates real displacement.
        output = np.array(
            background, dtype=np.result_type(background.dtype, float), copy=True
        )
        del background
        for region in self.regions:
            radius2 = np.sum((positions_m - region.center_m) ** 2, axis=1)
            output += np.exp(-radius2 / (2 * region.sigma_m**2))[
                None, :
            ] * region.source.evaluate(time_s, positions_m)
        return output

    def to_dict(self):
        return dict(
            background=encode_source(self.background),
            regions=[
                dict(
                    source=encode_source(r.source),
                    center_m=r.center_m,
                    sigma_m=r.sigma_m,
                )
                for r in self.regions
            ],
        )


@dataclass(frozen=True)
class VectorGroundMotion:
    """Independent scalar XYZ latent fields, optionally mixed by a real matrix.

    Each component is a displacement field or None. mixing maps latent XYZ to
    world XYZ and creates specified component cross spectra. PSD amplitudes do
    not define this mixing automatically. Independent seeded inputs need distinct
    seeds; components themselves may contain their own background/region model.
    """

    components: tuple
    mixing: tuple = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    frame = 'world'

    def __post_init__(self):
        components = tuple(self.components)
        if np.iscomplexobj(self.mixing):
            raise ValueError('World mixing matrix must be real')
        mixing = np.asarray(self.mixing, dtype=float)
        if (
            len(components) != 3
            or all(c is None for c in components)
            or mixing.shape != (3, 3)
            or not np.all(np.isfinite(mixing))
            or any(
                c is not None
                and (
                    getattr(c, 'frame', None) != 'body_local'
                    or getattr(c, 'unit', 'm') != 'm'
                )
                for c in components
            )
        ):
            raise ValueError(
                'Use three scalar displacement components and a finite real mixing matrix'
            )
        seeds = [
            seed
            for component in components
            if component is not None
            for seed in _source_seeds(component)
        ]
        if len(seeds) != len(set(seeds)):
            raise ValueError('Independent vector components require distinct seeds')
        object.__setattr__(self, 'components', components)
        object.__setattr__(self, 'mixing', tuple(tuple(row) for row in mixing))

    def workspace_bytes(self, n_times, n_receivers):
        return 48 * n_times * n_receivers + max(
            (
                getattr(c, 'workspace_bytes', lambda nt, nr: 0)(n_times, n_receivers)
                for c in self.components
                if c is not None
            ),
            default=0,
        )

    def evaluate(self, time_s, positions_m):
        values = np.zeros((len(time_s), len(positions_m), 3))
        for axis, component in enumerate(self.components):
            if component is not None:
                values[..., axis] = component.evaluate(time_s, positions_m)
        return values @ np.asarray(self.mixing).T

    def to_dict(self):
        return dict(
            components=[
                encode_source(c) if c is not None else None for c in self.components
            ],
            mixing=self.mixing,
        )


SOURCE_CLASSES = {
    cls.__name__: cls
    for cls in (
        Harmonic,
        PlaneWave,
        MeasuredMotion,
        DisplacementPSD,
        AngularPSD,
        SpectralGroundMotion,
        LocatedSource,
        RegionalGroundMotion,
        VectorGroundMotion,
    )
}


def _source_seeds(source):
    if hasattr(source, 'seed'):
        return [source.seed]
    if isinstance(source, RegionalGroundMotion):
        return _source_seeds(source.background) + [
            seed for region in source.regions for seed in _source_seeds(region.source)
        ]
    if isinstance(source, VectorGroundMotion):
        return [
            seed
            for component in source.components
            if component is not None
            for seed in _source_seeds(component)
        ]
    return []


def _fft_field_keys(source):
    """Finite FFT fields whose receiver basis must be sampled jointly.

    LocatedSource samples its waveform at the one excitation port, independently
    of the receivers. Reusing that temporal excitation is intentional and does
    not restart a receiver-dependent spatial factorization.
    """
    if isinstance(source, DisplacementPSD):
        return {
            (
                type(source).__name__,
                source.frequency_hz,
                source.density,
                source.seed,
                source.correlation_length_m,
            )
        }
    if isinstance(source, RegionalGroundMotion):
        return set().union(
            _fft_field_keys(source.background),
            *(_fft_field_keys(region.source) for region in source.regions),
        )
    if isinstance(source, VectorGroundMotion):
        return set().union(
            *(_fft_field_keys(component) for component in source.components)
        )
    return set()


def validate_source_bindings(source, existing_bindings):
    """Reject separate bindings that restart one finite FFT receiver model.

    A DisplacementPSD/AngularPSD record is defined jointly for a receiver set.
    Repeating the specification in another binding restarts its random basis;
    equal seeds alone do not preserve the requested spatial coherency. Bind the
    receivers together, or use a continuous SpectralGroundMotion field. This
    check also survives copy/JSON, where shared Python identity is not retained.
    """
    incoming = _fft_field_keys(source)
    if incoming and any(
        incoming.intersection(_fft_field_keys(binding.source))
        for binding in existing_bindings
    ):
        raise ValueError(
            'One finite FFT field needs one joint receiver binding; select all '
            'targets together or explicitly replay a jointly computed record. '
            'For displacement fields, SpectralGroundMotion supports receiver '
            'subsets'
        )


def requires_time_series(source):
    """Whether a source tree needs a uniform joint record before point queries."""
    if isinstance(source, DisplacementPSD):
        return True
    if isinstance(source, LocatedSource):
        return requires_time_series(source.waveform)
    if isinstance(source, RegionalGroundMotion):
        return requires_time_series(source.background) or any(
            requires_time_series(r.source) for r in source.regions
        )
    if isinstance(source, VectorGroundMotion):
        return any(requires_time_series(c) for c in source.components if c is not None)
    return False


def encode_source(source):
    from dataclasses import asdict
    from .motion import RigidMotion

    if type(source) is RigidMotion:
        return dict(type='RigidMotion', parameters=source.to_dict())
    if SOURCE_CLASSES.get(type(source).__name__) is not type(source):
        raise TypeError('custom source serialization is not defined yet')
    return dict(
        type=type(source).__name__,
        parameters=source.to_dict() if hasattr(source, 'to_dict') else asdict(source),
    )


def decode_source(stored):
    from .motion import RigidMotion

    cls = (
        RigidMotion
        if stored['type'] == 'RigidMotion'
        else SOURCE_CLASSES[stored['type']]
    )
    parameters = dict(stored['parameters'])
    if cls is LocatedSource:
        parameters['waveform'] = decode_source(parameters['waveform'])
    elif cls is RegionalGroundMotion:
        parameters['background'] = decode_source(parameters['background'])
        parameters['regions'] = tuple(
            GaussianRegion(decode_source(r['source']), r['center_m'], r['sigma_m'])
            for r in parameters['regions']
        )
    elif cls is VectorGroundMotion:
        parameters['components'] = tuple(
            decode_source(c) if c is not None else None
            for c in parameters['components']
        )
    return cls(**parameters)
