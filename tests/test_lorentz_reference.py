# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #
"""The standalone Lorentz reference: its field, its integrator, and both against Xtrack.

The field is checked against Maxwell's equations and its own closed forms, the
integrator against the Lorentz reference of the cavity tests and against the
FieldExpansionCavity map. Tolerances state what each comparison promises.
"""
import numpy as np
import pytest
import xtrack as xt
from _cavity_lorentz import calibrated_waves, hard_edge, lorentz_end
from scipy.constants import c as clight
from xtrack._temp.lorentz_reference import HarmonicField, track

_FREQUENCY = 801.58e6
_K = 2 * np.pi * _FREQUENCY / clight
_PROTON = xt.PROTON_MASS_EV
_COORDINATES = ('x', 'px', 'y', 'py', 'zeta', 'pzeta')


@pytest.mark.parametrize('wave_number', [.6 * _K, 1.4 * _K, _K], ids=['fast', 'slow', 'synchronous'])
def test_radial_series_converge_onto_the_bessel_factors(wave_number):
    """Each two orders of the near-axis series gain a factor (kappa r / 2)^2 / n^2 on
    the exact J0, J1 (I0, I1) factors; a synchronous harmonic is exactly (1, -1/2)."""
    field = HarmonicField([wave_number], [1.], _FREQUENCY, 1.)
    kappa = np.sqrt(abs(_K**2 - wave_number**2))
    r = np.array([.1, .3]) / max(kappa, 1.)
    exact = field.radial_factors(r)
    errors = [max(np.max(abs(series - truth)) for series, truth in zip(field.radial_factors(r, order), exact))
              for order in (2, 4, 6, 8)]
    if wave_number == _K:
        assert np.all(exact[0] == 1.) and np.all(exact[1] == -.5) and max(errors) == 0.
    else:
        assert errors[0] < 1e-2 and all(fine < 1e-2 * coarse for coarse, fine in zip(errors, errors[1:]))


def test_exact_field_satisfies_maxwell():
    """Gauss's law for E and B, Faraday's and the displacement-current law, by central
    differences at an off-axis point of a field of fast, slow and synchronous harmonics.
    The differences are good to (k h)^2 of the field's derivatives, about 1e-7."""
    rng = np.random.default_rng(3)
    wave_numbers = _K * np.array([.4, -1., 1., 1.7, -2.3])
    field = HarmonicField(wave_numbers, rng.normal(size=5) + 1j * rng.normal(size=5), _FREQUENCY, 2.)
    x, y, z, ct, h = 1.5e-2, -2e-2, .13, .41, 1e-5

    def at(dx=0., dy=0., dz=0., dct=0.):
        return np.array([value[0] for value in field.fields(np.array([x + dx]), np.array([y + dy]), z + dz, ct + dct)])
    d = {name: (at(**{name: h}) - at(**{name: -h})) / (2 * h) for name in ('dx', 'dy', 'dz', 'dct')}
    (Ex_x, Ey_x, Ez_x, cBx_x, cBy_x), (Ex_y, Ey_y, Ez_y, cBx_y, cBy_y) = d['dx'], d['dy']
    (Ex_z, Ey_z, Ez_z, cBx_z, cBy_z), (Ex_t, Ey_t, Ez_t, cBx_t, cBy_t) = d['dz'], d['dct']
    scale = np.max(abs(field.amplitudes)) * max(abs(wave_numbers))
    laws = [Ex_x + Ey_y + Ez_z, cBx_x + cBy_y,                       # div E, div B (c B_z = 0)
            cBx_t + (Ez_y - Ey_z), cBy_t + (Ex_z - Ez_x),            # d(cB)/d(ct) = -curl E
            Ex_t + cBy_z, Ey_t - cBx_z, Ez_t - (cBy_x - cBx_y)]      # dE/d(ct) = curl cB
    assert abs(Ex_x) > 1e-3 * scale and abs(cBy_t) > 1e-3 * scale
    assert max(abs(law) for law in laws) < 1e-7 * scale


@pytest.mark.parametrize('beta', [.55, .9, 1.])
def test_rating_is_the_constant_speed_gain_and_from_voltage_calibrates_it(beta):
    """rating(beta) = int G(z) exp(-i k z / beta) dz over the length, the gain phasor
    of a particle at constant speed crossing z = 0 at t = 0."""
    wave_numbers = _K * np.array([1.8, -1.8, .7, 0.])
    field = HarmonicField(wave_numbers, [1., .4 - .3j, .2j, -.1], _FREQUENCY, .9)
    z = np.linspace(-.45, .45, 200001)
    integrand = (field.amplitudes @ np.exp(1j * np.outer(wave_numbers, z))) * np.exp(-1j * _K * z / beta)
    trapezoid = np.trapezoid(integrand.real, z) + 1j * np.trapezoid(integrand.imag, z)
    assert abs(field.rating(beta) - trapezoid) < 1e-9 * np.sum(abs(field.amplitudes)) * .9
    for phase in (np.pi / 2, 1.2, -.4, 2.8):
        calibrated = HarmonicField.from_voltage(wave_numbers, field.amplitudes, _FREQUENCY, .9, 2e6, phase, beta)
        assert abs(abs(calibrated.rating(beta)) - 2e6) < 1e-8
        assert abs(calibrated.rating(beta).real - 2e6 * np.sin(phase)) < 1e-8


def test_a_bunch_tracks_as_its_particles_alone():
    """The worst-particle error norm keeps each particle at the tolerance it has alone."""
    rng = np.random.default_rng(7)
    gamma = 1.5
    p0c = _PROTON * np.sqrt(gamma**2 - 1)
    beta0 = np.sqrt(1 - 1 / gamma**2)
    length = 5 * beta0 * np.pi / _K
    field = HarmonicField.from_voltage([np.pi * 5 / length, -np.pi * 5 / length], [1., .5 * np.exp(-.7j)],
                                       _FREQUENCY, length, 4e6, 1.2, beta0)
    launch = dict(x=rng.uniform(-1e-2, 1e-2, 50), px=rng.uniform(-1e-3, 1e-3, 50),
                  y=rng.uniform(-1e-2, 1e-2, 50), py=rng.uniform(-1e-3, 1e-3, 50),
                  zeta=rng.uniform(-2e-2, 2e-2, 50), pzeta=rng.uniform(-3e-3, 3e-3, 50))
    bunch = track(field, **launch, p0c=p0c, mass0=_PROTON)
    for i in range(50):
        alone = track(field, **{name: value[i] for name, value in launch.items()}, p0c=p0c, mass0=_PROTON)
        for name in _COORDINATES:
            # The floor is the integrator's absolute scale, 1e-15 per step.
            assert abs(bunch[name][i] - alone[name][0]) <= 1e-10 * abs(alone[name][0] - launch[name][i]) + 1e-14


@pytest.mark.parametrize('backend', ['python', 'c'])
@pytest.mark.parametrize('radial_order', [2, 4])
@pytest.mark.parametrize('mass0, gamma, n_cells', [(_PROTON, 1.5, 5), (xt.ELECTRON_MASS_EV, 40., None)],
                         ids=['slow protons', 'electrons at c'])
def test_matches_the_cavity_tests_lorentz_reference(mass0, gamma, n_cells, radial_order, backend):
    """The two waves W = F cos(k ct - kz z + phi) + B cos(k ct + kz z + phi + chi) of
    `lorentz_end`, whose clock starts at the entrance at c t = -L/2 - zeta / beta0, are
    the harmonics +kz, F exp(-i (phi + k Delta)) and -kz, B exp(-i (phi + chi + k Delta)),
    with Delta = (L/2) (1/beta0 - 1) the clock offset to the centred one."""
    beta0 = np.sqrt(1 - 1 / gamma**2)
    p0c = mass0 * np.sqrt(gamma**2 - 1)
    energy0 = np.hypot(p0c, mass0)
    length = 5 * beta0 * np.pi / _K
    waves = calibrated_waves(5e6 * length, _FREQUENCY, length, 1.2, .5, hard_edge, .7, n_cells)
    kz = waves['wave_number']
    shift = _K * length / 2 * (1 / beta0 - 1)
    field = HarmonicField([kz, -kz], [waves['forward'] * np.exp(-1j * (waves['phase'] + shift)),
                                      waves['backward'] * np.exp(-1j * (waves['phase'] + .7 + shift))],
                          _FREQUENCY, length)
    for launch in (dict(x=3e-3, px=1e-4, y=-2e-3, py=-5e-5, zeta=-2e-3, delta=1e-3),
                   dict(x=-1e-3, px=-2e-4, y=4e-3, py=1e-4, zeta=3e-3, delta=-2e-3)):
        truth = lorentz_end(launch, waves, length, hard_edge, p0c, mass0, radial_order=radial_order)
        pzeta = (np.hypot(p0c * (1 + launch['delta']), mass0) - energy0) / (beta0 * p0c)
        end = track(field, launch['x'], launch['px'], launch['y'], launch['py'], launch['zeta'], pzeta, p0c, mass0,
                    order=radial_order, backend=backend)
        end = {name: end[name][0] for name in _COORDINATES}
        end['delta'] = np.sqrt((energy0 + end['pzeta'] * beta0 * p0c)**2 - mass0**2) / p0c - 1
        for name in ('x', 'px', 'y', 'py', 'zeta', 'delta'):
            assert abs(end[name] - truth[name]) < 1e-9 * abs(truth[name] - launch[name])


@pytest.mark.parametrize('backend', ['python', 'c'])
@pytest.mark.parametrize('radial_order', [4, 6])
@pytest.mark.parametrize('calibration', [1., .8], ids=['calibrated at beta0', 'calibrated at 0.8 beta0'])
def test_matches_the_field_expansion_cavity(calibration, radial_order, backend):
    """Five beta lambda/2 cells times the window (1 + cos(2 pi z / L))^3, so the field
    and its first six derivatives vanish at the ends, where the map's hard-edge faces
    then agree with the reference's abrupt cut to r^8. The map's calibration particle
    enters with the reference one, from_voltage's crosses the centre with it, so the
    centred clock moves by (L/2) (1/beta_c - 1/beta0)."""
    gamma, voltage, phase = 1.5, 3e6, 1.2
    beta0 = np.sqrt(1 - 1 / gamma**2)
    beta_c = calibration * beta0
    p0c = _PROTON * np.sqrt(gamma**2 - 1)
    length = 5 * beta0 * np.pi / _K
    kz, q = np.pi * 5 / length, 2 * np.pi / length
    # cos(kz z) (5/2 + 15/4 cos(q z) + 3/2 cos(2 q z) + 1/4 cos(3 q z)) as exponentials.
    wave_numbers = [s * kz + n * q for s in (1, -1) for n in (0, 1, -1, 2, -2, 3, -3)]
    amplitudes = [1.25, .9375, .9375, .375, .375, .0625, .0625] * 2
    rf = xt.FieldExpansionCavity(model='_phase-resolved', wave_numbers=wave_numbers, amplitudes=amplitudes,
                                 length=length, frequency=_FREQUENCY, voltage=voltage, phase=phase,
                                 calibration_beta=beta_c, num_kicks=4096, radial_order=radial_order,
                                 field_table_points=0)
    launch = dict(x=np.array([5e-3, -8e-3, 7e-3]), px=np.array([1e-4, -2e-4, 0.]),
                  y=np.array([-6e-3, 4e-3, 7e-3]), py=np.array([0., 1e-4, 0.]),
                  zeta=np.array([1e-3, -2e-3, 0.]), delta=np.array([1e-3, -1e-3, 0.]))
    particles = xt.Particles(p0c=p0c, mass0=_PROTON, **launch)
    start = {name: getattr(particles, name).copy() for name in _COORDINATES}
    rf.track(particles)
    assert np.all(particles.state == 1)
    field = HarmonicField.from_voltage(wave_numbers, amplitudes, _FREQUENCY, length, voltage, phase, beta_c)
    field = HarmonicField(wave_numbers, field.amplitudes * np.exp(1j * _K * length / 2 * (1 / beta_c - 1 / beta0)),
                          _FREQUENCY, length)
    truth = track(field, *(start[name] for name in _COORDINATES), p0c, _PROTON, order=radial_order,
                  backend=backend)
    for name in _COORDINATES:
        assert np.all(abs(getattr(particles, name) - truth[name]) < 1e-9 * abs(truth[name] - start[name]))


@pytest.mark.parametrize('order', [None, 6])
def test_c_engine_follows_the_python_engine(order):
    """Hard edges and spin included; each C particle takes its own steps, the Python bunch
    shared ones, so they agree to the integrators' tolerance, not to roundoff."""
    rng = np.random.default_rng(11)
    count, gamma = 30, 1.25
    p0c = _PROTON * np.sqrt(gamma**2 - 1)
    field = HarmonicField.from_voltage(_K * np.array([2., -2., .7, 3.1]), [1., .4, .2j, .05], _FREQUENCY, 1.1,
                                       5e6, 1.2, .5)
    launch = dict(x=rng.uniform(-1e-2, 1e-2, count), px=rng.normal(0, 1e-4, count),
                  y=rng.uniform(-1e-2, 1e-2, count), py=rng.normal(0, 1e-4, count),
                  zeta=rng.normal(0, 1e-2, count), pzeta=rng.normal(0, 1e-3, count))
    spin = np.array([np.full(count, .6), np.zeros(count), np.full(count, .8)])
    ends = [track(field, **launch, p0c=p0c, mass0=_PROTON, order=order, ramps=(1e-3, 5e-4), anomaly=1.79,
                  spin=spin, backend=backend) for backend in ('python', 'c')]
    for name in ends[0]:
        assert np.max(abs(ends[1][name] - ends[0][name])) < 1e-12

