# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #
"""Reference orbit through a prescribed axisymmetric RF field.

The Lorentz force in a field written directly from Maxwell's equations,
integrated along z by an adaptive eighth-order Runge-Kutta method (DOP853) for
many particles at once. It is standalone: it uses no Xtrack element, map,
gauge or calibration, so it can check them.

The field is a sum of space harmonics with an absolute on-axis amplitude,

    E_z(r=0, z, t) = Re[G(z) exp(-i omega t)],  G(z) = sum_m c_m exp(i k_m z),

for |z| <= length / 2, field-free outside. Off axis each harmonic is a TM0 wave
with kappa_m^2 = k^2 - k_m^2: exactly, E_z = c_m J0(kappa r) and E_r, c B_phi
from J1 (I0, I1 for slow harmonics), or the near-axis series of those Bessel
functions to E_z ~ r^order, E_r and c B_phi ~ r^(order - 1).

Particles enter and leave in Xsuite coordinates (x, px, y, py, zeta, pzeta)
relative to a reference momentum p0c and mass0. The clock is centred: the
reference particle passes z = 0 at t = 0, and zeta = z - beta0 c t. Inside, the
state is mechanical, (x, y, c t) in metres and (P_x c, P_y c, E) in eV.
"""
from functools import lru_cache
from math import factorial
from pathlib import Path

import numpy as np
import xobjects as xo
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.rk import DOP853
from scipy.special import i0, i1, j0, j1

CLIGHT = 299792458.0


class HarmonicField:
    """The on-axis field G(z) = sum_m c_m exp(i k_m z), c_m in V/m, of an RF
    field at `frequency` over |z| <= length / 2."""

    def __init__(self, wave_numbers, amplitudes, frequency, length):
        self.wave_numbers = np.atleast_1d(np.asarray(wave_numbers, float))
        self.amplitudes = np.atleast_1d(np.asarray(amplitudes, complex))
        if self.wave_numbers.shape != self.amplitudes.shape or self.wave_numbers.ndim != 1:
            raise ValueError('wave_numbers and amplitudes must be equal-length 1D arrays')
        self.frequency = float(frequency)
        self.length = float(length)
        if not (self.frequency > 0. and self.length > 0.):
            raise ValueError('frequency and length must be positive')

    @property
    def k(self):
        return 2 * np.pi * self.frequency / CLIGHT

    def rating(self, beta):
        """Energy gain phasor [eV per unit charge] of a particle at constant
        speed beta c crossing z = 0 at t = 0: its gain is Re of the result."""
        half = self.length / 2
        detuning = self.wave_numbers - self.k / beta
        return self.amplitudes @ (2 * half * np.sinc(detuning * half / np.pi))

    @classmethod
    def from_voltage(cls, wave_numbers, amplitudes, frequency, length, voltage, phase,
                     calibration_beta=1.):
        """Scale relative amplitudes so that a particle at calibration_beta
        crossing z = 0 at t = 0 gains voltage * sin(phase): crest at pi / 2.
        As in Xtrack, the phase grows with arrival time: crossing at c t, the
        particle gains voltage * sin(phase + k c t)."""
        field = cls(wave_numbers, amplitudes, frequency, length)
        scale = voltage * np.exp(-1j * (phase - np.pi / 2)) / field.rating(calibration_beta)
        return cls(wave_numbers, field.amplitudes * scale, frequency, length)

    def radial_factors(self, r, order=None):
        """Per particle and harmonic, A with E_z = Re[c A e^(i(k_m z - omega t))]
        and B with E_r = r Re[i k_m c B e^(...)], c B_phi = r Re[i k c B e^(...)]:
        A = J0(kappa r), B = -J1(kappa r) / (kappa r) (I0 and -I1 / x for slow
        harmonics, the same series in -kappa^2 r^2 / 4), or their series to
        `order`; r has shape (N,), the result (N, harmonics)."""
        kappa2 = self.k**2 - self.wave_numbers**2
        if order is not None:
            if order < 2 or order % 2:
                raise ValueError('order must be an even integer of at least 2')
            u = -np.outer(r * r, kappa2) / 4
            A = sum(u**n / factorial(n)**2 for n in range(order // 2 + 1))
            B = -sum(u**n / (factorial(n) * factorial(n + 1))
                     for n in range(order // 2)) / 2
            return A, B
        s = np.sqrt(np.abs(kappa2))
        argument = np.outer(r, s)
        A = np.empty_like(argument)
        B = np.empty_like(argument)
        for fast, zero, one in ((kappa2 > 0, j0, j1), (kappa2 < 0, i0, i1)):
            if fast.any():
                A[:, fast] = zero(argument[:, fast])
                small = argument[:, fast] == 0.
                B[:, fast] = np.where(small, -.5, -one(argument[:, fast])
                                      / np.where(small, 1., argument[:, fast]))
        synchronous = kappa2 == 0
        A[:, synchronous], B[:, synchronous] = 1., -.5
        return A, B

    def fields(self, x, y, z, ct, order=None):
        """E_x, E_y, E_z [V/m] and c B_x, c B_y [V/m] for particles at (x, y,
        c t), arrays of shape (N,), on the plane z; c B_z = 0. Zero outside
        |z| <= length / 2."""
        r = np.hypot(x, y)
        A, B = self.radial_factors(r, order)
        wave = np.outer(np.exp(-1j * self.k * ct),
                        self.amplitudes * np.exp(1j * self.wave_numbers * z))
        Ez = np.real(np.sum(wave * A, axis=1))
        radial = np.real(np.sum(1j * self.wave_numbers * wave * B, axis=1))
        azimuthal = np.real(np.sum(1j * self.k * wave * B, axis=1))
        inside = float(abs(z) <= self.length / 2)
        Ex, Ey = radial * x * inside, radial * y * inside
        return Ex, Ey, Ez * inside, -azimuthal * y * inside, azimuthal * x * inside


class _WorstParticleDOP853(DOP853):
    """DOP853 whose error norm is that of the worst particle, not the RMS over
    the whole bunch, so each particle keeps the requested tolerance. The
    state holds `components` blocks of one value per particle."""

    def __init__(self, *args, components=6, **options):
        super().__init__(*args, **options)
        self.components = components

    def _estimate_error_norm(self, K, h, scale):
        err5 = (np.dot(K.T, self.E5) / scale).reshape(self.components, -1)
        err3 = (np.dot(K.T, self.E3) / scale).reshape(self.components, -1)
        err5_norm_2 = np.sum(err5**2, axis=0)
        err3_norm_2 = np.sum(err3**2, axis=0)
        denom = err5_norm_2 + 0.01 * err3_norm_2
        norms = np.where(denom > 0., np.abs(h) * err5_norm_2
                         / np.sqrt(np.where(denom > 0., denom, 1.) * self.components), 0.)
        return np.max(norms)


if not hasattr(DOP853, '_estimate_error_norm'):  # pragma: no cover
    raise ImportError('scipy DOP853 no longer exposes _estimate_error_norm')


@lru_cache(maxsize=1)
def _c_context():
    """A CPU context holding the C engine, lorentz_reference_src/lorentz_reference.h."""
    context = xo.ContextCpu()
    pointer = dict(pointer=True)
    kernel = xo.Kernel(c_name='lorentz_track', args=[
        xo.Arg(xo.Int64, name='count'), xo.Arg(xo.Float64, name='state', **pointer),
        xo.Arg(xo.Int64, name='status', **pointer), xo.Arg(xo.Int64, name='harmonics'),
        xo.Arg(xo.Float64, name='wave_number', const=True, **pointer),
        xo.Arg(xo.Float64, name='re', const=True, **pointer),
        xo.Arg(xo.Float64, name='im', const=True, **pointer),
        *(xo.Arg(xo.Float64, name=name) for name in ('k', 'half', 'q0', 'mass0', 'anomaly')),
        xo.Arg(xo.Int64, name='order'), xo.Arg(xo.Float64, name='ramp'),
        xo.Arg(xo.Float64, name='rtol'), xo.Arg(xo.Float64, name='atol', const=True, **pointer),
        xo.Arg(xo.Float64, name='max_step')])
    context.add_kernels(kernels={'lorentz_track': kernel},
                        sources=[Path(__file__).parent / 'lorentz_reference_src' / 'lorentz_reference.h'])
    return context


def track(field, x, px, y, py, zeta, pzeta, p0c, mass0, q0=1., order=None,
          rtol=1e-12, atol_position=1e-15, atol_momentum=1e-15, max_step=None,
          ramps=None, anomaly=None, spin=None, backend='python'):
    """Exit Xsuite coordinates of particles launched at z = -length / 2.

    order=None uses the exact Bessel field, an even integer the near-axis
    series. rtol is relative; atol_position [m] and atol_momentum (in units of
    p0c for momenta and energy, and for spin) are absolute. max_step [m]
    defaults to a sixteenth of the RF wavelength; the error control sets the
    steps within. Returns a dict of numpy arrays x, px, y, py, zeta, pzeta.

    By default the field is cut abruptly at the ends. `ramps`, a sequence of
    ramp lengths, instead tracks the field times a sin^2 envelope h(z) rising
    over each ramp inside the ends, each ramp its own segment, and returns the
    polynomial extrapolation of the exits to zero ramp: the hard edge, linear
    for two ramps and quadratic for three. With W = Re[G exp(-i k ct)], the
    envelope adds -h' W r / 2 to E_r and, integrated by parts along the orbit
    to avoid the 1 / ramp^2 spike of h'', the r^2 term of E_z,
    h' / 4 [2 (x P_x + y P_y) / P_z W + r^2 (dW/dct E / P_z - dW/dz)]; the rest
    of the field, higher radial terms and exact Bessel factors included, is
    multiplied by h only, as in the maps.

    With `anomaly`, the spin's three components `spin` (spin_x, spin_y,
    spin_z) follow the Thomas-BMT equation in the same fields, edge terms
    included, and are returned as spin_x, spin_y, spin_z.

    backend='python' integrates the bunch as one SciPy system with the error
    of its worst particle; backend='c' runs the same equations in C
    (lorentz_reference_src/), each particle with its own DOP853 steps."""
    with_spin = anomaly is not None
    if with_spin != (spin is not None):
        raise ValueError('anomaly and spin go together')
    coordinates = np.broadcast_arrays(*(np.atleast_1d(np.asarray(v, float)) for v in
                                        (x, px, y, py, zeta, pzeta, *(spin if with_spin else ()))))
    x, px, y, py, zeta, pzeta = coordinates[:6]
    count = x.size
    energy0 = np.hypot(p0c, mass0)
    beta0 = p0c / energy0
    half = field.length / 2
    if ramps is not None and not (all(0. < ramp < half for ramp in ramps)
                                  and len(set(ramps)) == len(ramps)):
        raise ValueError('ramps must be distinct, positive and below length / 2')
    energy = energy0 + pzeta * beta0 * p0c
    ct = (-half - zeta) / beta0
    state = np.concatenate([x, y, ct, px * p0c, py * p0c, energy, *coordinates[6:]])

    def rhs(z, flat, ramp):
        X, Y, CT, PX, PY, E = flat[:6 * count].reshape(6, count)
        pz = np.sqrt(E**2 - mass0**2 - PX**2 - PY**2)
        Ex, Ey, Ez, cBx, cBy = field.fields(X, Y, z, CT, order)
        if ramp is not None:
            a = min(max((half - abs(z)) / ramp, 0.), 1.)
            h, dh = np.sin(np.pi * a / 2)**2, -np.sign(z) * np.pi / (2 * ramp) * np.sin(np.pi * a)
            wave = np.outer(np.exp(-1j * field.k * CT),
                            field.amplitudes * np.exp(1j * field.wave_numbers * z))
            W = np.sum(wave.real, axis=1)
            W_z = -wave.imag @ field.wave_numbers
            W_ct = field.k * np.sum(wave.imag, axis=1)
            Ex, Ey = h * Ex - dh * W / 2 * X, h * Ey - dh * W / 2 * Y
            Ez = h * Ez + dh / 4 * (2 * (X * PX + Y * PY) / pz * W
                                    + (X * X + Y * Y) * (W_ct * E / pz - W_z))
            cBx, cBy = h * cBx, h * cBy
        fx = q0 * (Ex - pz * cBy / E)
        fy = q0 * (Ey + pz * cBx / E)
        rates = [PX / pz, PY / pz, E / pz, fx * E / pz, fy * E / pz,
                 q0 * (Ex * PX + Ey * PY + Ez * pz) / pz]
        if with_spin:
            # Thomas-BMT in eV units, per unit z: dS/dz = Omega x S E / (P_z c).
            gamma = E / mass0
            beta = np.array([PX, PY, pz]) / E
            cB = np.array([cBx, cBy, np.zeros(count)])
            Omega = -q0 / mass0 * ((anomaly + 1 / gamma) * cB
                - anomaly * gamma / (gamma + 1) * np.sum(beta * cB, axis=0) * beta
                - (anomaly + 1 / (gamma + 1)) * np.cross(beta, [Ex, Ey, Ez], axis=0))
            rates.extend(np.cross(Omega, flat[6 * count:].reshape(3, count), axis=0) * E / pz)
        return np.concatenate(rates)

    components = len(coordinates)
    atol = np.repeat([atol_position] * 3 + [atol_momentum * p0c] * 3
                     + [atol_momentum] * (components - 6), count)
    step = max_step or np.pi / field.k / 8
    ends = []
    if backend not in ('python', 'c'):
        raise ValueError('backend must be "python" or "c"')
    for ramp in ramps or (None,):
        planes = (-half, half) if ramp is None else (-half, -half + ramp, half - ramp, half)
        flat = state
        if backend == 'c':
            flat = np.array(state)
            status = np.zeros(count, np.int64)
            _c_context().kernels.lorentz_track(
                count=count, state=flat, status=status, harmonics=len(field.wave_numbers),
                wave_number=field.wave_numbers, re=np.ascontiguousarray(field.amplitudes.real),
                im=np.ascontiguousarray(field.amplitudes.imag), k=field.k, half=half, q0=q0,
                mass0=mass0, anomaly=anomaly if with_spin else np.nan,
                order=0 if order is None else order, ramp=ramp or 0., rtol=rtol,
                atol=np.array([atol_position] * 3 + [atol_momentum * p0c] * 3
                              + [atol_momentum] * 3), max_step=step)
            if not status.all():
                raise RuntimeError(f'{np.sum(status == 0)} particles failed to cross the field')
            planes = ()
        for a, b in zip(planes[:-1], planes[1:]):
            solution = solve_ivp(rhs, (a, b), flat, method=_WorstParticleDOP853, rtol=rtol,
                                 atol=atol, args=(ramp,), components=components,
                                 max_step=step if ramp is None else min(step, (b - a) / 16))
            if not solution.success:
                raise RuntimeError(solution.message)
            flat = solution.y[:, -1]
        ends.append(flat)
    weights = [np.prod([r / (r - s) for r in ramps if r != s]) for s in ramps] if ramps else [1.]
    final = sum(w * end for w, end in zip(weights, ends))
    X, Y, CT, PX, PY, E = final[:6 * count].reshape(6, count)
    end = dict(x=X, px=PX / p0c, y=Y, py=PY / p0c, zeta=half - beta0 * CT,
               pzeta=(E - energy0) / (beta0 * p0c))
    if with_spin:
        end.update(zip(('spin_x', 'spin_y', 'spin_z'), final[6 * count:].reshape(3, count)))
    return end
