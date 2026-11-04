"""The Lorentz orbit of a particle in the near-axis RF field of a Cavity.

An independent reference for the cavity maps: the field is written directly from
the wave equation at quadratic transverse order and integrated with a general
ODE solver, using none of the maps' generators, gauges or factors.
"""
import numpy as np
from scipy.constants import c as clight
from scipy.integrate import quad, solve_ivp

COORDINATES = ('x', 'px', 'y', 'py', 'zeta', 'delta')


def p16(length):
    half = length / 2

    def envelope(z):
        u = np.clip(z / half, -1, 1)
        w = 1 - u * u
        return w**16, -32 * u * w**15 / half, (-32 * w**15 + 960 * u * u * w**14) / half**2
    return envelope


def ramped(length, ramp):
    """Uniform field with sin^2 ramps inside its ends: a hard edge as the ramp vanishes."""
    half = length / 2

    def envelope(z):
        a = np.clip((half - abs(z)) / ramp, 0., 1.)
        inside = (a > 0.) & (a < 1.)
        return (np.sin(np.pi * a / 2)**2,
                -np.sign(z) * np.pi / (2 * ramp) * np.sin(np.pi * a) if inside else 0.,
                (np.pi / ramp)**2 / 2 * np.cos(np.pi * a) if inside else 0.)
    return envelope


def gaussian(sigma):
    def envelope(z):
        h = np.exp(-z * z / (2 * sigma**2))
        return h, -z / sigma**2 * h, (z * z / sigma**4 - 1 / sigma**2) * h
    return envelope


def hard_edge(z):
    return 1., 0., 0.


def calibrated_waves(voltage, frequency, length, phase, eta, envelope,
                     counter_phase=0., calibration_beta=1., q0=1.):
    """Peak field [V/m] of the two waves and the phase of the forward one at T = Z = 0,
    such that a particle at calibration_beta gains voltage * sin(phase)."""
    k = 2 * np.pi * frequency / clight
    half = k * length / 2
    ratio = np.sqrt(eta)
    forward, backward = 1 / (1 + ratio), ratio / (1 + ratio)

    def integrand(Z):
        return envelope(Z / k)[0] * (forward * np.exp(1j * (1 / calibration_beta - 1) * Z)
            + backward * np.exp(1j * (counter_phase + (1 / calibration_beta + 1) * Z)))
    limit = 64 + int(8 * half)
    real, _ = quad(lambda Z: integrand(Z).real, -half, half, epsabs=1e-13, epsrel=1e-13, limit=limit)
    imag, _ = quad(lambda Z: integrand(Z).imag, -half, half, epsabs=1e-13, epsrel=1e-13, limit=limit)
    phasor = real + 1j * imag
    amplitude = k * voltage / abs(phasor)
    return dict(k=k, forward=amplitude * forward, backward=amplitude * backward,
                counter_phase=counter_phase,
                phase=(half * (1 - 1 / calibration_beta) + phase - np.copysign(np.pi / 2, q0)
                       - np.angle(phasor)))


def lorentz_end(launch, waves, length, envelope, p0c, mass0, charge=1., anomaly=None, tolerance=1e-12,
                magnet=None):
    """Coordinates at the exit plane, from those at the entry plane, for a field-free
    outside: Xtrack's x, px, y, py, zeta, delta relative to p0c. With `anomaly`, the
    launch's spin_x, spin_y, spin_z follow the Thomas-BMT equation in the same fields.
    `magnet(z)` adds static fields: c B_z, its derivative, and the quadrupole gradient
    c dB_y/dx, in eV/m and eV/m^2 per unit charge."""
    k, phase = waves['k'], waves['phase']
    F, B, chi = waves['forward'], waves['backward'], waves['counter_phase']
    beta0 = p0c / np.hypot(p0c, mass0)
    momentum = p0c * (1 + launch['delta'])

    def rhs(z, state):
        x, y, ct, px, py, energy = state[:6]
        pz = np.sqrt(energy**2 - mass0**2 - px**2 - py**2)
        h, dh, d2h = envelope(z)
        u, w = k * (ct - z) + phase, k * (ct + z) + phase + chi
        W = F * np.cos(u) + B * np.cos(w)
        W_z = k * (F * np.sin(u) - B * np.sin(w))
        W_ct = -k * (F * np.sin(u) + B * np.sin(w))
        r2 = x * x + y * y
        E_z = h * W - r2 / 4 * (d2h * W + 2 * dh * W_z)
        E_r_over_r = -(dh * W + h * W_z) / 2
        cB_over_r = h * W_ct / 2
        v_z = pz / energy
        fx = charge * (E_r_over_r - v_z * cB_over_r) * x
        fy = charge * (E_r_over_r - v_z * cB_over_r) * y
        work = charge * (E_r_over_r * (x * px + y * py) + E_z * pz)
        rates = [px / pz, py / pz, energy / pz, fx * energy / pz, fy * energy / pz, work / pz]
        if magnet is not None:
            cB_z, dcB_z, gradient = magnet(z)
            cB_x, cB_y = -x / 2 * dcB_z + gradient * y, -y / 2 * dcB_z + gradient * x
            rates[3] += charge * (py / pz * cB_z - cB_y)
            rates[4] += charge * (cB_x - px / pz * cB_z)
        if anomaly is None:
            return rates
        # Thomas-BMT in eV units, per unit z: dS/dz = Omega x S * energy / (pz c)
        field_E = np.array([E_r_over_r * x, E_r_over_r * y, E_z])
        field_cB = np.array([-cB_over_r * y, cB_over_r * x, 0.])
        p = np.array([px, py, pz])
        gamma = energy / mass0
        beta = p / energy
        Omega = -charge / mass0 * ((anomaly + 1 / gamma) * field_cB
            - anomaly * gamma / (gamma + 1) * np.dot(beta, field_cB) * beta
            - (anomaly + 1 / (gamma + 1)) * np.cross(beta, field_E))
        spin = np.cross(Omega, state[6:9]) * energy / pz
        return rates + list(spin)

    start = [launch['x'], launch['y'], -length / 2 - launch['zeta'] / beta0,
             launch['px'] * p0c, launch['py'] * p0c, np.hypot(momentum, mass0)]
    if anomaly is not None:
        start += [launch['spin_x'], launch['spin_y'], launch['spin_z']]
    solution = solve_ivp(rhs, (-length / 2, length / 2), start, method='DOP853', rtol=tolerance,
                         atol=1e-14, max_step=2 * np.pi / k / 64)
    assert solution.success, solution.message
    x, y, ct, px, py, energy = solution.y[:6, -1]
    end = dict(x=x, px=px / p0c, y=y, py=py / p0c, zeta=length - beta0 * (ct + length / 2),
               delta=np.sqrt(energy**2 - mass0**2) / p0c - 1)
    if anomaly is not None:
        end.update(spin_x=solution.y[6, -1], spin_y=solution.y[7, -1], spin_z=solution.y[8, -1])
    return end
