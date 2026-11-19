# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #
"""Small linear response models; complex phase uses exp(+i omega t)."""
from dataclasses import dataclass

import numpy as np

from .motion import checked_times


class FrequencyResponse:
    def apply(self, values, time_s):
        """Offline periodic FFT response. No padding, warm-up or causal transient.

        Scalar transfer is dimensionless. A matrix acts on the final coordinate
        axis; mixed m/rad channels require corresponding SI cross-coupling units.
        Input/output ports must have the same specified reference frames/points.
        """
        times = checked_times(time_s, uniform=True)
        values = np.asarray(values)
        if np.iscomplexobj(values) or values.ndim < 1 or not np.all(np.isfinite(values)) or values.shape[0] != len(times):
            raise ValueError('Response input must be finite real samples matching time')
        dt = (times[-1]-times[0])/(len(times)-1)
        frequency = np.fft.rfftfreq(len(times), dt)
        transfer = np.asarray(self.evaluate(frequency), dtype=complex)
        if not np.all(np.isfinite(transfer)):
            raise ValueError('Nonfinite frequency response')
        spectrum = np.fft.rfft(values, axis=0)
        if transfer.shape == frequency.shape:
            spectrum *= transfer.reshape((-1,) + (1,)*(values.ndim-1))
        elif transfer.shape == (len(frequency), values.shape[-1], values.shape[-1]):
            spectrum = np.einsum('fij,f...j->f...i', transfer, spectrum)
        else:
            raise ValueError('Response must return (frequencies,) or (frequencies, channels, channels)')
        # A real record has real self-conjugate endpoints. Reject lost phase if
        # a nonzero endpoint cannot represent the requested complex response.
        endpoints = [0, -1] if len(times) % 2 == 0 else [0]
        for index in endpoints:
            scale = max(1e-300, np.max(np.abs(spectrum)))
            if np.max(np.abs(spectrum[index].imag)) > 1e-12*scale:
                raise ValueError('Complex DC/Nyquist output is not representable; remove endpoint input power')
        return np.fft.irfft(spectrum, n=len(times), axis=0)


@dataclass(frozen=True)
class BaseExcitedOscillator(FrequencyResponse):
    """Absolute displacement/base displacement, including viscous base force."""
    frequency_hz: float
    damping_ratio: float

    def __post_init__(self):
        if not np.all(np.isfinite([self.frequency_hz, self.damping_ratio])) or self.frequency_hz <= 0 or self.damping_ratio <= 0:
            raise ValueError('Use positive finite resonance frequency and damping ratio')

    def evaluate(self, frequency_hz):
        ratio = np.asarray(frequency_hz) / self.frequency_hz
        return (1+2j*self.damping_ratio*ratio) / (1-ratio**2+2j*self.damping_ratio*ratio)


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
        values = [self.girder_mass_kg, self.magnet_mass_kg, self.floor_stiffness_n_m,
                  self.magnet_stiffness_n_m, self.floor_damping_ns_m, self.magnet_damping_ns_m]
        if not np.all(np.isfinite(values)) or min(values) <= 0:
            raise ValueError('Use positive finite masses, stiffnesses and damping')
        if self.stage not in ('floor_to_girder', 'girder_to_magnet', 'floor_to_magnet'):
            raise ValueError('Unknown loaded-response stage')

    def evaluate(self, frequency_hz):
        omega = 2*np.pi*np.asarray(frequency_hz)
        zg = self.floor_stiffness_n_m + 1j*omega*self.floor_damping_ns_m
        zm = self.magnet_stiffness_n_m + 1j*omega*self.magnet_damping_ns_m
        d11 = zg + zm - self.girder_mass_kg*omega**2
        d22 = zm - self.magnet_mass_kg*omega**2
        determinant = d11*d22-zm**2
        if self.stage == 'girder_to_magnet':
            return zm/d22
        return zg*(d22 if self.stage == 'floor_to_girder' else zm)/determinant


@dataclass(frozen=True)
class TabulatedResponse(FrequencyResponse):
    """Complex scalar or matrix response, interpolated in real/imaginary parts.

    Matrices use SI coordinates [x,y,s,pitch,yaw,roll] at stated matching ports.
    Cross terms therefore carry m/rad or rad/m. No implicit lever-arm transport.
    """
    frequency_hz: object
    response: object

    def __post_init__(self):
        if np.iscomplexobj(self.frequency_hz):
            raise ValueError('Response frequencies must be real')
        frequency, response = np.array(self.frequency_hz, dtype=float), np.array(self.response, dtype=complex)
        if (frequency.ndim != 1 or len(frequency) < 2 or frequency[0] < 0 or np.any(np.diff(frequency) <= 0)
                or response.shape[0] != len(frequency) or not np.all(np.isfinite(frequency))
                or not np.all(np.isfinite(response)) or response.ndim not in (1, 3)
                or (response.ndim == 3 and response.shape[1:] != (6, 6))):
            raise ValueError('Use increasing frequency and finite scalar or 6x6 response')
        frequency.flags.writeable = response.flags.writeable = False
        object.__setattr__(self, 'frequency_hz', frequency)
        object.__setattr__(self, 'response', response)

    def evaluate(self, frequency_hz):
        frequency = np.asarray(frequency_hz)
        scalar = frequency.ndim == 0
        frequency = np.atleast_1d(frequency)
        if np.any(frequency < self.frequency_hz[0]) or np.any(frequency > self.frequency_hz[-1]):
            raise ValueError('Requested frequency lies outside the response table')
        flat = self.response.reshape(len(self.frequency_hz), -1)
        values = np.column_stack([np.interp(frequency, self.frequency_hz, col.real)
                                  + 1j*np.interp(frequency, self.frequency_hz, col.imag) for col in flat.T])
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
        if not isinstance(oscillator, BaseExcitedOscillator) or not np.isfinite(sample_step_s) or sample_step_s <= 0:
            raise ValueError('Use a BaseExcitedOscillator and a positive sample step')
        if method not in ('zoh','bilinear'):
            raise ValueError('Use zoh or bilinear discretization')
        omega = 2*np.pi*oscillator.frequency_hz
        numerator, denominator, _ = cont2discrete(
            ([2*oscillator.damping_ratio*omega, omega**2],
             [1., 2*oscillator.damping_ratio*omega, omega**2]), sample_step_s, method=method)
        # Retain the leading zero of the strictly proper ZOH numerator: dropping
        # it while balancing poles/zeros would advance the output by one tick.
        self.sos = np.array([[*np.pad(numerator[0],(3-len(numerator[0]),0)),
                              *np.pad(denominator,(3-len(denominator),0))]])
        self.method = method
        self.sample_step_s = sample_step_s
        self._state = None

    def reset(self):
        self._state = None

    def process(self, values):
        from scipy.signal import sosfilt
        values = np.asarray(values)
        if np.iscomplexobj(values) or values.ndim < 1 or not np.all(np.isfinite(values)):
            raise ValueError('Causal response requires finite real samples')
        shape = (len(self.sos), 2) + values.shape[1:]
        if self._state is None:
            self._state = np.zeros(shape)
        if self._state.shape != shape:
            raise ValueError('Response channel layout changed; reset explicitly')
        output, self._state = sosfilt(self.sos, values, axis=0, zi=self._state)
        return output
