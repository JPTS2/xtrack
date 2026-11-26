# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #
"""Small linear response models; complex phase uses exp(+i omega t)."""

from dataclasses import dataclass

import numpy as np

from .motion import checked_scalar, checked_times


class FrequencyResponse:
    def apply(self, values, time_s):
        """Offline periodic FFT response. No padding, warm-up or causal transient.

        Scalar transfer is dimensionless. A matrix maps the final input-channel
        axis to its output-channel axis. Its shape is (frequencies, outputs,
        inputs); leading receiver/batch axes are retained. The caller specifies
        channel order, units and input/output ports. No frame/lever conversion
        or physical force response is inferred from an array's dimensions.
        """
        times = checked_times(time_s, uniform=True)
        values = np.asarray(values)
        if (
            np.iscomplexobj(values)
            or values.ndim < 1
            or not np.all(np.isfinite(values))
            or values.shape[0] != len(times)
        ):
            raise ValueError('Response input must be finite real samples matching time')
        dt = (times[-1] - times[0]) / (len(times) - 1)
        frequency = np.fft.rfftfreq(len(times), dt)
        transfer = np.asarray(self.evaluate(frequency), dtype=complex)
        if not np.all(np.isfinite(transfer)):
            raise ValueError('Nonfinite frequency response')
        spectrum = np.fft.rfft(values, axis=0)
        if transfer.shape == frequency.shape:
            spectrum *= transfer.reshape((-1,) + (1,) * (values.ndim - 1))
        elif (
            values.ndim >= 2
            and transfer.ndim == 3
            and transfer.shape[0] == len(frequency)
            and transfer.shape[1] > 0
            and transfer.shape[2] > 0
            and transfer.shape[2] == values.shape[-1]
        ):
            spectrum = np.einsum('fij,f...j->f...i', transfer, spectrum)
        else:
            raise ValueError(
                'Response must return (frequencies,) or '
                '(frequencies, output_channels, input_channels) matching the input'
            )
        # A real record has real self-conjugate endpoints. Reject lost phase if
        # a nonzero endpoint cannot represent the requested complex response.
        endpoints = [0, -1] if len(times) % 2 == 0 else [0]
        scale = np.maximum(1e-300, np.max(np.abs(spectrum), axis=0))
        for index in endpoints:
            if np.any(np.abs(spectrum[index].imag) > 1e-12 * scale):
                raise ValueError(
                    'Complex DC/Nyquist output is not representable; remove endpoint input power'
                )
        return np.fft.irfft(spectrum, n=len(times), axis=0)


@dataclass(frozen=True)
class BaseExcitedOscillator(FrequencyResponse):
    """Absolute displacement/base displacement, including viscous base force."""

    frequency_hz: float
    damping_ratio: float

    def __post_init__(self):
        for name in ('frequency_hz', 'damping_ratio'):
            object.__setattr__(self, name, checked_scalar(getattr(self, name), name))
        if (
            self.frequency_hz <= 0
            or self.damping_ratio <= 0
        ):
            raise ValueError(
                'Use positive finite resonance frequency and damping ratio'
            )

    def evaluate(self, frequency_hz):
        ratio = np.asarray(frequency_hz) / self.frequency_hz
        return (1 + 2j * self.damping_ratio * ratio) / (
            1 - ratio**2 + 2j * self.damping_ratio * ratio
        )


@dataclass(frozen=True)
class LoadedTwoMass(FrequencyResponse):
    """Loaded floor/girder/magnet model, both bodies rigid, one translation.

    stage is floor_to_girder, girder_to_magnet (conditional), or floor_to_magnet.
    Masses [kg], spring constants [N/m], viscous coefficients [N s/m].
    """

    girder_mass_kg: float
    magnet_mass_kg: float
    floor_stiffness_n_m: float
    magnet_stiffness_n_m: float
    floor_damping_ns_m: float
    magnet_damping_ns_m: float
    stage: str = 'floor_to_girder'

    def __post_init__(self):
        for name in (
            'girder_mass_kg', 'magnet_mass_kg', 'floor_stiffness_n_m',
            'magnet_stiffness_n_m', 'floor_damping_ns_m', 'magnet_damping_ns_m',
        ):
            value = checked_scalar(getattr(self, name), name)
            if value <= 0:
                raise ValueError('Use positive finite masses, stiffnesses and damping')
            object.__setattr__(self, name, value)
        if self.stage not in ('floor_to_girder', 'girder_to_magnet', 'floor_to_magnet'):
            raise ValueError('Unknown loaded-response stage')

    def evaluate(self, frequency_hz):
        omega = 2 * np.pi * np.asarray(frequency_hz)
        zg = self.floor_stiffness_n_m + 1j * omega * self.floor_damping_ns_m
        zm = self.magnet_stiffness_n_m + 1j * omega * self.magnet_damping_ns_m
        d11 = zg + zm - self.girder_mass_kg * omega**2
        d22 = zm - self.magnet_mass_kg * omega**2
        determinant = d11 * d22 - zm**2
        if self.stage == 'girder_to_magnet':
            return zm / d22
        return zg * (d22 if self.stage == 'floor_to_girder' else zm) / determinant


@dataclass(frozen=True)
class TabulatedResponse(FrequencyResponse):
    """Complex scalar or matrix response, interpolated in real/imaginary parts.

    Matrices have shape (frequencies, output_channels, input_channels). For a
    rigid-to-rigid response both axes use [x,y,s,pitch,yaw,roll], with SI cross
    units such as m/rad. A contact-to-body map may instead have 3*N contact
    displacement inputs and six rigid outputs. Specify all port frames, channel
    order and units explicitly. No implicit fitting or lever-arm transport.
    """

    frequency_hz: object
    response: object

    def __post_init__(self):
        if np.iscomplexobj(self.frequency_hz):
            raise ValueError('Response frequencies must be real')
        frequency, response = (
            np.array(self.frequency_hz, dtype=float),
            np.array(self.response, dtype=complex),
        )
        if (
            frequency.ndim != 1
            or len(frequency) < 2
            or frequency[0] < 0
            or np.any(np.diff(frequency) <= 0)
            or response.ndim not in (1, 3)
            or response.shape[0] != len(frequency)
            or not np.all(np.isfinite(frequency))
            or not np.all(np.isfinite(response))
            or (response.ndim == 3 and 0 in response.shape[1:])
        ):
            raise ValueError(
                'Use increasing frequency and finite scalar or '
                '(frequencies, output_channels, input_channels) response'
            )
        frequency.flags.writeable = response.flags.writeable = False
        object.__setattr__(self, 'frequency_hz', frequency)
        object.__setattr__(self, 'response', response)

    def evaluate(self, frequency_hz):
        frequency = np.asarray(frequency_hz)
        if (frequency.ndim > 1 or frequency.dtype.kind not in 'fiu'
                or not np.all(np.isfinite(frequency))):
            raise ValueError('Use finite real scalar or vector query frequencies')
        scalar = frequency.ndim == 0
        frequency = np.atleast_1d(frequency)
        if np.any(frequency < self.frequency_hz[0]) or np.any(
            frequency > self.frequency_hz[-1]
        ):
            raise ValueError('Requested frequency lies outside the response table')
        flat = self.response.reshape(len(self.frequency_hz), -1)
        values = np.column_stack(
            [
                np.interp(frequency, self.frequency_hz, col.real)
                + 1j * np.interp(frequency, self.frequency_hz, col.imag)
                for col in flat.T
            ]
        )
        output = values.reshape((len(frequency),) + self.response.shape[1:])
        return output[0] if scalar else output


class CausalResponse:
    """Causal sampled response with bounded SOS delay state, initially at rest.

    Default ZOH discretization is exact at sample times for held input and the
    stated continuous LTI model. Optional bilinear discretization has no
    prewarping. The first output is at the first sample time, initially at rest.
    """

    def __init__(self, *, oscillator, sample_step_s, method='zoh'):
        from scipy.signal import cont2discrete

        sample_step_s = checked_scalar(sample_step_s, 'sample_step_s')
        if (
            not isinstance(oscillator, BaseExcitedOscillator)
            or sample_step_s <= 0
        ):
            raise ValueError('Use a BaseExcitedOscillator and a positive sample step')
        if method not in ('zoh', 'bilinear'):
            raise ValueError('Use zoh or bilinear discretization')
        omega = 2 * np.pi * oscillator.frequency_hz
        numerator, denominator, _ = cont2discrete(
            (
                [2 * oscillator.damping_ratio * omega, omega**2],
                [1.0, 2 * oscillator.damping_ratio * omega, omega**2],
            ),
            sample_step_s,
            method=method,
        )
        # Retain the leading zero of the strictly proper ZOH numerator: dropping
        # it while balancing poles/zeros would advance the output by one tick.
        self.sos = np.array(
            [
                [
                    *np.pad(numerator[0], (3 - len(numerator[0]), 0)),
                    *np.pad(denominator, (3 - len(denominator), 0)),
                ]
            ]
        )
        self.method = method
        self.sample_step_s = sample_step_s
        self._state = None

    def reset(self):
        self._state = None

    def get_state(self):
        """Export JSON-compatible filter delays for continuing at the next sample."""
        return dict(
            sample_step_s=self.sample_step_s,
            sos=self.sos.tolist(),
            delay=None if self._state is None else self._state.tolist(),
        )

    def set_state(self, state):
        """Restore delays for the same discretized response and ordered channels.

        The caller supplies the next consecutive input sample on the same
        time grid. The filter has no absolute clock or input/source state.
        """
        if state['sample_step_s'] != self.sample_step_s or not np.array_equal(
            state['sos'], self.sos
        ):
            raise ValueError('Response state requires the same discrete filter and sample step')
        delay = state['delay']
        if delay is not None:
            if np.iscomplexobj(delay):
                raise ValueError('Response delays must be real')
            delay = np.array(delay, dtype=float, copy=True)
            if (delay.shape[:2] != (len(self.sos), 2) or not delay.size
                    or not np.all(np.isfinite(delay))):
                raise ValueError('Invalid causal response delay state')
        self._state = delay

    def process(self, values):
        from scipy.signal import sosfilt

        values = np.asarray(values)
        if (
            np.iscomplexobj(values)
            or values.ndim < 1
            or not values.size
            or not np.all(np.isfinite(values))
        ):
            raise ValueError('Causal response requires nonempty finite real samples')
        shape = (len(self.sos), 2) + values.shape[1:]
        state = np.zeros(shape) if self._state is None else self._state
        if state.shape != shape:
            raise ValueError('Response channel layout changed; reset explicitly')
        output, new_state = sosfilt(self.sos, values, axis=0, zi=state)
        self._state = new_state
        return output
