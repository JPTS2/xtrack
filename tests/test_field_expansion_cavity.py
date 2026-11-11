# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #
"""FieldExpansionCavity: Cavity as its special case, and a real cavity's field."""
import numpy as np
import pytest
import xobjects as xo
import xtrack as xt
from _cavity_lorentz import calibrated_waves, lorentz_end, ramped
from scipy.constants import c as clight

# TESLA 9-cell on-axis field, Fourier cosine series over 1.31879807 m (PRST-AB 3, 092001).
TESLA = np.array([
    0.1644024074311037, -0.1324009958969339, 4.3443060026047219e-2, 8.5602654094946495e-2,
    -0.2433578169042885, 0.5297150596779437, 0.7164884680963959, -5.2579522442877296e-3,
    -5.5025369142193678e-2, 4.6845673335028933e-2, -2.3279346335638568e-2, 4.0800777539657775e-3,
    4.1378326533752169e-3, -2.5040533340490805e-3, -4.0654981400000964e-3, 9.6630592067498289e-3,
    -8.5275895985990214e-3, -5.8078747006425020e-2, -2.4044337836660403e-2, 1.0968240064697212e-2,
    -3.4461179858301418e-3, -8.1201564869443749e-4, 2.1438992904959380e-3, -1.4997753525697276e-3,
    1.8685171825676386e-4])
TESLA_LENGTH = 1.31879807
COORDINATES = ('x', 'px', 'y', 'py', 'zeta', 'delta', 'spin_x', 'spin_y', 'spin_z')


@pytest.mark.parametrize('mass0, gamma, n_cells, ratio, counter_phase, phase', [
    (xt.PROTON_MASS_EV, 1.5, 5, .5, .7, 1.2),
    (xt.ELECTRON_MASS_EV, 40., 9, 1., 0., .3),
    (xt.ELECTRON_MASS_EV, 40., 7, .3, -2., 2.1)])
def test_cavity_is_the_two_harmonic_case(mass0, gamma, n_cells, ratio, counter_phase, phase):
    frequency = 704.4e6
    length = n_cells * np.sqrt(1 - 1 / gamma**2) * clight / frequency / 2
    wave_number = np.pi * n_cells / length
    common = dict(length=length, voltage=3e6 * length, frequency=frequency, phase=phase,
                  num_kicks=16 * n_cells, shift_x=1e-4, rot_s_rad=.1)
    cavity = xt.Cavity(model='_phase-resolved', n_cells=n_cells,
                       counter_wave_amplitude=ratio, counter_wave_phase=counter_phase, **common)
    expansion = xt.FieldExpansionCavity(
        wave_numbers=[wave_number, -wave_number],
        amplitudes=[1., ratio * np.exp(-1j * counter_phase)],
        calibration_beta=cavity.synchronous_beta, field_table_points=0, **common)
    p0c = mass0 * np.sqrt(gamma**2 - 1)
    ends = []
    for element in (cavity, expansion):
        line = xt.Line(elements={'rf': element})
        line.particle_ref = xt.Particles(p0c=p0c, mass0=mass0)
        line.build_tracker()
        line.configure_spin(True)
        particles = xt.Particles(p0c=p0c, mass0=mass0, x=[1e-3, -4e-4], px=[2e-4, 1e-4],
                                 y=[-5e-4, 3e-4], py=[1e-4, -2e-4], zeta=[1e-3, -2e-3],
                                 delta=[1e-3, -2e-3], spin_x=.6, spin_z=.8,
                                 anomalous_magnetic_moment=1.79)
        line.track(particles)
        assert np.all(particles.state == 1)
        ends.append(np.array([getattr(particles, name) for name in COORDINATES]))
    np.testing.assert_allclose(ends[1], ends[0], rtol=0, atol=2e-15)


@pytest.mark.parametrize('gamma', [10., 40.])
@pytest.mark.parametrize('offset', [0., -.6])
def test_tesla_field_follows_the_orbit(gamma, offset):
    """The reference resolves the field's small value at the ends as a 1 mm ramp."""
    frequency, voltage, mass0 = 1.3e9, 20e6, xt.ELECTRON_MASS_EV
    p0c = mass0 * np.sqrt(gamma**2 - 1)
    edges = ramped(TESLA_LENGTH, 1e-3)
    harmonics = 2 * np.pi * np.arange(len(TESLA)) / TESLA_LENGTH
    weights = TESLA * np.where(np.arange(len(TESLA)) == 0, .5, 1.)

    def field(z):
        e = np.sum(weights * np.cos(harmonics * z))
        de = -np.sum(weights * harmonics * np.sin(harmonics * z))
        dde = -np.sum(weights * harmonics**2 * np.cos(harmonics * z))
        r, dr, ddr = edges(z)
        return e * r, de * r + e * dr, dde * r + 2 * de * dr + e * ddr
    waves = calibrated_waves(voltage, frequency, TESLA_LENGTH, np.pi / 2 + offset, 1., field,
                             wave_number=0.)
    launch = dict(x=1e-3, px=0., y=0., py=0., zeta=0., delta=0.)
    truth = lorentz_end(launch, waves, TESLA_LENGTH, field, p0c, mass0, ramp=1e-3)
    rf = xt.FieldExpansionCavity.from_fourier_series(
        TESLA, length=TESLA_LENGTH, frequency=frequency, voltage=voltage,
        phase=np.pi / 2 + offset, num_kicks=400)
    particles = xt.Particles(p0c=p0c, mass0=mass0, **launch)
    rf.track(particles)
    assert particles.state[0] == 1
    assert abs(particles.px[0] - truth['px']) < 1e-4 * abs(truth['px'])
    assert abs(particles.delta[0] - truth['delta']) < 1e-6 * abs(truth['delta'])


def test_samples_of_a_fourier_series_give_its_field():
    # The same field from two sets of harmonics, summed directly: equal to round-off.
    controls = dict(length=TESLA_LENGTH, frequency=1.3e9, voltage=20e6, phase=1.2,
                    num_kicks=200, field_table_points=0)
    series = xt.FieldExpansionCavity.from_fourier_series(TESLA, **controls)
    z = np.linspace(-TESLA_LENGTH / 2, TESLA_LENGTH / 2, 101)
    n = np.arange(len(TESLA))
    samples = (TESLA * np.where(n == 0, .5, 1.)
               * np.cos(2 * np.pi * n * z[:, None] / TESLA_LENGTH)).sum(axis=1)
    sampled = xt.FieldExpansionCavity.from_samples(samples, **controls)
    ends = []
    for rf in (series, sampled):
        particles = xt.Particles(p0c=20e6, mass0=xt.ELECTRON_MASS_EV, x=1e-3, px=1e-4)
        rf.track(particles)
        ends.append([particles.x[0], particles.px[0], particles.delta[0]])
    np.testing.assert_allclose(ends[1], ends[0], rtol=1e-12, atol=1e-16)


@pytest.mark.parametrize('mode', ['thick', 'thin'])
def test_slices_compose_and_serialize(mode):
    rf = xt.FieldExpansionCavity.from_fourier_series(
        TESLA, length=TESLA_LENGTH, frequency=1.3e9, voltage=20e6, phase=1.2,
        num_kicks=48, calibration_beta=.99)
    reference = xt.Particles(p0c=20e6, mass0=xt.ELECTRON_MASS_EV)
    whole = xt.Line(elements={'rf': rf})
    whole.particle_ref = reference
    sliced = xt.Line.from_dict(whole.to_dict())
    sliced.slice_thick_elements([xt.Strategy(slicing=xt.Teapot(2, mode=mode))])
    sliced = xt.Line.from_dict(sliced.to_dict())
    assert sliced['rf'].calibration_beta == .99
    np.testing.assert_array_equal(sliced['rf'].amplitudes, rf.amplitudes)
    ends = []
    for line in (whole, sliced):
        line.particle_ref = reference
        line.build_tracker(_context=xo.ContextCpu())
        particles = xt.Particles(p0c=20e6, mass0=xt.ELECTRON_MASS_EV, x=1e-3, px=1e-4,
                                 zeta=1e-3)
        line.track(particles)
        assert particles.state[0] == 1
        ends.append([particles.x[0], particles.px[0], particles.zeta[0], particles.delta[0]])
    np.testing.assert_allclose(ends[1], ends[0], rtol=0, atol=1e-13)


def _spin_end(element, gamma, phase_offset=.5):
    p0c = xt.ELECTRON_MASS_EV * np.sqrt(gamma**2 - 1)
    k = 2 * np.pi * element.frequency / clight
    line = xt.Line(elements={'c': element})
    line.particle_ref = xt.Particles(p0c=p0c, mass0=xt.ELECTRON_MASS_EV)
    line.build_tracker()
    line.configure_spin(spin_model=True)
    particles = xt.Particles(p0c=p0c, mass0=xt.ELECTRON_MASS_EV, x=1e-3, y=-5e-4, px=1e-4,
                             zeta=-phase_offset * np.sqrt(1 - 1 / gamma**2) / k, spin_x=.6, spin_z=.8,
                             anomalous_magnetic_moment=0.00115965218)
    line.track(particles)
    assert particles.state[0] == 1
    return np.array([getattr(particles, name)[0] for name in COORDINATES])


@pytest.mark.parametrize('seed', [1, 2, 3])
def test_field_table_follows_the_harmonic_sum(seed):
    """Any field: the tabulated jets give the map of the summed harmonics."""
    rng = np.random.default_rng(seed)
    count = int(rng.integers(5, 40))
    frequency = rng.uniform(.5e9, 3e9)
    # Space harmonics of a real structure lie within a few k of the vacuum wave number,
    # around a synchronous one that carries the acceleration.
    k = 2 * np.pi * frequency / clight
    wave_numbers = np.append(rng.uniform(-3., 3., count) * k, k)
    amplitudes = np.append(.3 * (rng.normal(size=count) + 1j * rng.normal(size=count)), 3.)
    common = dict(wave_numbers=wave_numbers, amplitudes=amplitudes, length=rng.uniform(.3, 1.5),
                  frequency=frequency, voltage=5e6, phase=np.pi / 2, num_kicks=256)
    table = _spin_end(xt.FieldExpansionCavity(**common), 40.)
    direct = _spin_end(xt.FieldExpansionCavity(field_table_points=0, **common), 40.)
    launch = np.array([1e-3, 1e-4, -5e-4, 0., 0., 0., .6, 0., .8])
    change = np.abs(direct - launch)
    # Quintic interpolation at 20 points per shortest wavelength, relative to each change.
    assert np.all(np.abs(table - direct) <= 2e-6 * change + 1e-15)


def test_field_table_follows_the_amplitudes_and_not_a_changed_length():
    common = dict(wave_numbers=[-60., 25., 80.], amplitudes=[1., .4j, -.3], frequency=1e9,
                  voltage=5e6, phase=np.pi / 2, num_kicks=256)
    element = xt.FieldExpansionCavity(length=.8, **common)
    element.amplitudes = [.7, .2j, .5]
    direct = xt.FieldExpansionCavity(length=.8, field_table_points=0,
                                     **(common | dict(amplitudes=[.7, .2j, .5])))
    np.testing.assert_allclose(_spin_end(element, 20.), _spin_end(direct, 20.), rtol=1e-6, atol=1e-12)
    # A changed length leaves the table behind: the harmonics are summed directly.
    element.length = .9
    direct.length = .9
    np.testing.assert_array_equal(_spin_end(element, 20.), _spin_end(direct, 20.))
    rebuilt = xt.FieldExpansionCavity.from_dict(element.to_dict())
    assert len(rebuilt.field_table) == len(element.field_table)
