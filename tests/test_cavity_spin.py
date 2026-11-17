"""Cavity spin follows the magnets: every model turns it, with the static fields,
their faces, slices, misalignments and the analysis flags."""
from math import factorial

import numpy as np
import pytest
import xobjects as xo
import xtrack as xt
from scipy.constants import c as clight

from _cavity_lorentz import calibrated_waves, hard_edge, lorentz_end, ramped

# Each model's spin error with static fields, relative to the spin's turn.
MODELS = {'_phase-resolved': 1e-6, '_phase-resolved-paraxial': 1e-4,
          '_phase-averaged': 1e-3, '_phase-resolved-frozen': 5e-3}
MASS, ANOMALY, FREQUENCY = xt.ELECTRON_MASS_EV, 0.00115965218, 801.58e6
LENGTH = 6 * clight / FREQUENCY / 2
P0C = MASS * np.sqrt(40.**2 - 1)
LAUNCH = dict(x=1e-3, px=1e-4, y=-5e-4, py=-5e-5, zeta=-2e-3, delta=1e-3,
              spin_x=0.6, spin_y=0., spin_z=0.8)
STATIC = {'multipoles': dict(order=3, knl=[2e-4, .05, 2., 0.], ksl=[0., .05, 0., 200.],
                             edge_entry_active=0, edge_exit_active=0),
          'solenoid': dict(ks=.5)}


@pytest.fixture(scope='module')
def lorentz_spin():
    """The Lorentz and Thomas-BMT spin of each static field through a travelling wave,
    with the hard edges extrapolated from two ramps."""
    waves = calibrated_waves(10e6 * LENGTH, FREQUENCY, LENGTH, 1.2, 0., hard_edge, 0., 0, 1.)
    knl, ksl = STATIC['multipoles']['knl'], STATIC['multipoles']['ksl']
    series = [P0C * (knl[n] + 1j * ksl[n]) / (factorial(n) * LENGTH) for n in range(4)]
    truth = {}
    for name in STATIC:
        ends = []
        for ramp in (1e-3, 5e-4):
            edges = ramped(LENGTH, ramp)
            magnet = ((lambda z: (0., 0., series)) if name == 'multipoles' else
                      (lambda z, e=edges: (.5 * P0C * e(z)[0], .5 * P0C * e(z)[1], [0j])))
            end = lorentz_end(LAUNCH, waves, LENGTH, ramped(LENGTH, ramp), P0C, MASS, 1.,
                              anomaly=ANOMALY, magnet=magnet, ramp=ramp, tolerance=1e-12)
            ends.append(np.array([end['spin_x'], end['spin_y'], end['spin_z']]))
        truth[name] = 2 * ends[1] - ends[0]
    return truth


def tracked(element, context=None, slices=0, **launch):
    line = xt.Line(elements={'rf': element})
    line.particle_ref = xt.Particles(p0c=P0C, mass0=MASS)
    if slices:
        line.slice_thick_elements([xt.Strategy(xt.Teapot(slices, mode='thick'))])
    line.build_tracker(_context=context or xo.ContextCpu())
    line.configure_spin(True)
    particles = xt.Particles(_context=line._context, p0c=P0C, mass0=MASS,
                             anomalous_magnetic_moment=ANOMALY, **{**LAUNCH, **launch})
    line.track(particles)
    particles.move(_context=xo.context_default)
    assert particles.state[0] == 1
    return particles


def cavity(model, **changes):
    return xt.MagnetOverlappedCavity(frequency=FREQUENCY, length=LENGTH, voltage=10e6 * LENGTH,
                                     phase=1.2, counter_wave_amplitude=0., model=model,
                                     **changes)


@pytest.mark.parametrize('static', STATIC)
@pytest.mark.parametrize('model', MODELS)
def test_static_fields_turn_the_spin_as_the_lorentz_reference(lorentz_spin, model, static):
    p = tracked(cavity(model, **STATIC[static]))
    spin = np.array([p.spin_x[0], p.spin_y[0], p.spin_z[0]])
    turn = np.linalg.norm(lorentz_spin[static] - [.6, 0., .8])
    assert np.linalg.norm(spin - lorentz_spin[static]) < MODELS[model] * turn


@pytest.mark.parametrize('integrator', ['uniform', 'yoshida-4'])
@pytest.mark.parametrize('model', MODELS)
def test_slices_turn_the_spin_as_the_whole_cavity(model, integrator):
    if model == '_phase-resolved' and integrator != 'uniform':
        pytest.skip('the Gauss cell is already fourth order')
    # Slices share the parent's cells when their count is given.
    ends = [tracked(cavity(model, integrator=integrator, num_kicks=12,
                           **STATIC['multipoles']), slices=n)
            for n in (0, 3)]
    orbit, spin = [[np.array([getattr(p, n)[0] for n in names]) for p in ends]
                   for names in (('x', 'px', 'y', 'py', 'zeta', 'delta'),
                                 ('spin_x', 'spin_y', 'spin_z'))]
    np.testing.assert_allclose(orbit[1], orbit[0], rtol=0, atol=1e-12)
    # The frozen model's slices invert its spin rotations to first order.
    np.testing.assert_allclose(spin[1], spin[0], rtol=0,
                               atol=1e-9 if model.endswith('frozen') else 1e-14)


@pytest.mark.parametrize('model', MODELS)
def test_misaligned_cavity_turns_the_spin_as_the_translated_launch(model):
    shifted = tracked(cavity(model, ks=.5, shift_x=2e-4))
    translated = tracked(cavity(model, ks=.5), x=LAUNCH['x'] - 2e-4)
    for name in ('px', 'py', 'delta', 'spin_x', 'spin_y', 'spin_z'):
        np.testing.assert_allclose(getattr(shifted, name), getattr(translated, name),
                                   rtol=0, atol=1e-14)


@pytest.mark.parametrize('flag', ['XS_FLAG_KILL_CAVITY_KICK', 'XS_FLAG_SR_KICK_SAME_AS_FIRST'])
@pytest.mark.parametrize('model', MODELS)
def test_analysis_flags_keep_the_spin(model, flag):
    line = xt.Line(elements={'rf': cavity(model, ks=.5)})
    line.particle_ref = xt.Particles(p0c=P0C, mass0=MASS)
    line.build_tracker()
    line.configure_spin(True)
    setattr(line.tracker.track_flags, flag, True)
    particles = xt.Particles(p0c=P0C, mass0=MASS, anomalous_magnetic_moment=ANOMALY, **LAUNCH)
    line.track(particles)
    assert particles.state[0] == 1
    assert particles.spin_x[0] != LAUNCH['spin_x']


def test_radiation_alone_also_turns_the_spin():
    line = xt.Line(elements={'rf': cavity('_phase-resolved')})
    line.particle_ref = xt.Particles(p0c=P0C, mass0=MASS)
    line.build_tracker()
    line.configure_radiation(model='mean')
    particles = xt.Particles(p0c=P0C, mass0=MASS, anomalous_magnetic_moment=ANOMALY, **LAUNCH)
    line.track(particles)
    assert particles.state[0] == 1
    assert particles.spin_x[0] != LAUNCH['spin_x']
