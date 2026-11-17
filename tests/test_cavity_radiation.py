"""The static fields of a MagnetOverlappedCavity radiate as the magnets' do, in every
model: mean, quantum and quantum-kick, with slices."""
import numpy as np
import pytest
import xtrack as xt

MODELS = ('longitudinal-kick', '_phase-resolved', '_phase-averaged',
          '_phase-resolved-paraxial', '_phase-resolved-frozen')
P0C, LENGTH, K1, X, Y = 45.6e9, 2.0, 0.1, 3e-3, -1e-3


def tracked(model, radiation, n=1, slices=0, voltage=0.):
    cavity = xt.MagnetOverlappedCavity(length=LENGTH, voltage=voltage, frequency=400e6,
                                       knl=[0., K1 * LENGTH], model=model, num_kicks=24,
                                       counter_wave_amplitude=0., edge_entry_active=0,
                                       edge_exit_active=0)
    line = xt.Line(elements={'rf': cavity})
    line.particle_ref = xt.Particles(p0c=P0C, mass0=xt.ELECTRON_MASS_EV)
    if slices:
        line.slice_thick_elements([xt.Strategy(xt.Teapot(slices, mode='thick'))])
    line.build_tracker()
    line.configure_radiation(model=radiation)
    particles = xt.Particles(p0c=P0C, mass0=xt.ELECTRON_MASS_EV, x=np.full(n, X),
                             y=np.full(n, Y))
    if radiation != 'mean':
        particles._init_random_number_generator(seeds=np.arange(n) + 7)
    line.track(particles)
    assert np.all(particles.state == 1)
    return particles


@pytest.mark.parametrize('model', MODELS)
def test_mean_radiation_follows_the_quadrupole_orbit(model):
    """dE/E = (2/3) r_e gamma^3 k1^2 int (x^2 + y^2) ds along the focusing orbit."""
    gamma = P0C / xt.ELECTRON_MASS_EV
    root = np.sqrt(K1)
    along = .5 * (LENGTH + np.sin(2 * root * LENGTH) / (2 * root))
    across = .5 * (LENGTH + np.sinh(2 * root * LENGTH) / (2 * root))
    expected = -2 / 3 * 2.8179403262e-15 * gamma**3 * K1**2 * (X**2 * along + Y**2 * across)
    np.testing.assert_allclose(tracked(model, 'mean').delta[0], expected, rtol=1e-3)


@pytest.mark.parametrize('radiation', ['quantum', 'quantum-kick'])
@pytest.mark.parametrize('model', MODELS[1:])
def test_quantum_radiation_matches_the_magnet_body(model, radiation):
    n = 20000
    reference, particles = (tracked(m, radiation, n).delta for m in ('longitudinal-kick', model))
    error = np.hypot(reference.std(), particles.std()) / np.sqrt(n)
    assert abs(particles.mean() - reference.mean()) < 5 * error
    np.testing.assert_allclose(particles.std(), reference.std(), rtol=.05)


@pytest.mark.parametrize('model', MODELS[1:])
def test_slices_radiate_as_the_whole_cavity(model):
    whole, sliced = (tracked(model, 'mean', slices=n, voltage=1e6).delta[0] for n in (0, 3))
    np.testing.assert_allclose(sliced, whole, rtol=0, atol=1e-15)


def test_radiation_is_off_until_configured():
    cavity = xt.MagnetOverlappedCavity(length=LENGTH, frequency=400e6, knl=[0., K1 * LENGTH],
                                       model='_phase-resolved', edge_entry_active=0,
                                       edge_exit_active=0)
    line = xt.Line(elements={'rf': cavity})
    line.particle_ref = xt.Particles(p0c=P0C, mass0=xt.ELECTRON_MASS_EV)
    particles = line.build_particles(x=X)
    line.track(particles)
    assert particles.delta[0] == 0.
    line.configure_radiation(model='mean')
    assert cavity.radiation_flag == 1


@pytest.mark.parametrize('model', MODELS)
def test_kick_as_first_particle_gives_every_particle_the_first_ones_kick(model):
    """As in the magnets, the others take the first particle's kick of its last
    step at every step, whatever their own orbit."""
    deltas = {}
    for name in ('longitudinal-kick', model):
        cavity = xt.MagnetOverlappedCavity(length=LENGTH, frequency=400e6,
                                           knl=[0., K1 * LENGTH], model=name, num_kicks=24,
                                           counter_wave_amplitude=0., edge_entry_active=0,
                                           edge_exit_active=0)
        line = xt.Line(elements={'rf': cavity})
        line.particle_ref = xt.Particles(p0c=P0C, mass0=xt.ELECTRON_MASS_EV)
        line.build_tracker()
        line.configure_radiation(model='mean')
        line.tracker.track_flags.XS_FLAG_SR_KICK_SAME_AS_FIRST = True
        particles = line.build_particles(x=[X, 2 * X, -3 * X])
        line.track(particles)
        deltas[name] = particles.delta.copy()
    assert deltas[model][1] < 0.
    np.testing.assert_allclose(deltas[model][2], deltas[model][1], rtol=1e-12)
    np.testing.assert_allclose(deltas[model], deltas['longitudinal-kick'], rtol=1e-3)
