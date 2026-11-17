"""Cavities radiate in every model: the static fields as the magnets' do, the RF field
its Lienard power into the weak-dipole line; mean, quantum and quantum-kick, with
slices."""
import numpy as np
import pytest
import xobjects as xo
import xtrack as xt
from scipy.constants import c as clight

from _cavity_lorentz import calibrated_waves, hard_edge, lorentz_end, ramped
from xtrack.beam_elements._common import SynchrotronRadiationRecord
from xtrack.random import RandomExponential, RandomUniformAccurate

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


GAMMA, FREQUENCY, GRADIENT = 2e4, 1.3e9, 30e6
CELLS = 9 * clight / FREQUENCY / 2


def rf_loss(element, radiation, x, n=1):
    """Energy radiated through the element, from the same launch without radiation."""
    line = xt.Line(elements={'rf': element})
    line.particle_ref = xt.Particles(p0c=xt.ELECTRON_MASS_EV * np.sqrt(GAMMA**2 - 1),
                                     mass0=xt.ELECTRON_MASS_EV)
    line.build_tracker()
    ends = []
    for model in (None, radiation):
        line.configure_radiation(model=model)
        particles = line.build_particles(x=np.full(n, x), y=np.full(n, -x / 2))
        if model not in (None, 'mean'):
            particles._init_random_number_generator(seeds=np.arange(n) + 11)
        line.track(particles)
        ends.append(particles.energy)
    return ends[0] - ends[1]


def standing_wave(model):
    return xt.Cavity(length=CELLS, voltage=GRADIENT * CELLS, frequency=FREQUENCY, phase=1.2,
                     model=model, counter_wave_amplitude=1.)


@pytest.mark.parametrize('model', ['_phase-resolved', '_phase-averaged',
                                   '_phase-resolved-paraxial'])
def test_rf_field_radiates_the_lienard_power_of_the_orbit(model):
    """Against (2/3) r_e gamma^2 [(E + v x B)^2 - (beta.E)^2] / (m c^2) along the
    Lorentz orbit's body, the hard edges extrapolated from two ramps."""
    launch = dict(x=1e-3, px=0., y=-5e-4, py=0., zeta=0., delta=0.)
    p0c = xt.ELECTRON_MASS_EV * np.sqrt(GAMMA**2 - 1)
    waves = calibrated_waves(GRADIENT * CELLS, FREQUENCY, CELLS, 1.2, 1., hard_edge, 0., 0, 1.)
    ends = [lorentz_end(launch, waves, CELLS, ramped(CELLS, ramp), p0c, xt.ELECTRON_MASS_EV,
                        1., ramp=ramp, tolerance=1e-11, lienard=True)['radiated']
            for ramp in (1e-3, 5e-4)]
    np.testing.assert_allclose(rf_loss(standing_wave(model), 'mean', 1e-3)[0],
                               2 * ends[1] - ends[0], rtol=1e-3)


def test_space_harmonics_radiate_as_their_waves():
    k = 2 * np.pi * FREQUENCY / clight
    harmonics = xt.FieldExpansionCavity(length=CELLS, voltage=GRADIENT * CELLS,
                                        frequency=FREQUENCY, phase=1.2,
                                        wave_numbers=[k, -k], amplitudes=[.5, .5])
    np.testing.assert_allclose(rf_loss(harmonics, 'mean', 1e-3),
                               rf_loss(standing_wave('_phase-resolved'), 'mean', 1e-3),
                               rtol=1e-5)


@pytest.mark.parametrize('radiation', ['quantum', 'quantum-kick'])
def test_rf_photons_carry_the_mean_loss(radiation):
    n = 100000
    mean = rf_loss(standing_wave('_phase-resolved'), 'mean', 1e-2)[0]
    loss = rf_loss(standing_wave('_phase-resolved'), radiation, 1e-2, n)
    assert loss.min() >= 0.
    assert abs(loss.mean() - mean) < 5 * loss.std() / np.sqrt(n)


class _PhotonSumProbe(xt.BeamElement):
    """Draw the energy of `count` photons of the line, in units of the largest."""
    allow_rot_and_shift = False
    _xofields = {'count': xo.Int64}
    _depends_on = [RandomUniformAccurate, RandomExponential]
    _internal_record_class = SynchrotronRadiationRecord
    _extra_c_sources = [r'''
        #include "xtrack/beam_elements/elements_src/cavity_models_radiation.h"
        GPUFUN
        void _PhotonSumProbe_track_local_particle(
                _PhotonSumProbeData el, LocalParticle* part0) {
            START_PER_PARTICLE_BLOCK(part0, part)
                LocalParticle_set_x(part, CavityEffects_photons_energy(
                    part, _PhotonSumProbeData_get_count(el)));
            END_PER_PARTICLE_BLOCK;
        }
    ''']


@pytest.mark.parametrize('count', [1, 2, 3, 5, 12, 20])
def test_photon_sums_have_the_lines_moments(count):
    """n photons of the line: mean n/2, variance n/10, within [0, n]."""
    n = 200000
    particles = xt.Particles(p0c=1e9, x=np.zeros(n))
    particles._init_random_number_generator(seeds=np.arange(n))
    _PhotonSumProbe(count=count).track(particles)
    energy = particles.x
    assert energy.min() >= 0. and energy.max() <= count
    np.testing.assert_allclose(energy.mean(), count / 2, rtol=5e-3)
    np.testing.assert_allclose(energy.var(), count / 10, rtol=2e-2)
