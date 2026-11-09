"""Every physical cavity map against the Lorentz orbit in the same fields.

The orbit is integrated by a general ODE solver (`_cavity_lorentz.py`) with none of
the maps' gauges, generators or factors. Tolerances state what each map promises,
relative to the size of the effect it models, and leave room for roundoff only.
"""
from functools import lru_cache

import numpy as np
import pytest
import xobjects as xo
import xtrack as xt
from _cavity_lorentz import COORDINATES, calibrated_waves, hard_edge, lorentz_end, ramped
from scipy.constants import c as clight

_FREQUENCY = 801.58e6
_HALF_WAVE = clight / _FREQUENCY / 2
_ELECTRON = xt.ELECTRON_MASS_EV
_LAUNCHES = (dict(x=1e-3, px=1e-4, y=-5e-4, py=-5e-5, zeta=-2e-3, delta=1e-3),
             dict(x=-5e-4, px=-2e-4, y=1e-3, py=1e-4, zeta=2e-3, delta=-1e-3),
             dict(x=2e-4, px=0., y=2e-4, py=0., zeta=0., delta=0.))


@pytest.fixture(scope='module')
def context():
    return xo.ContextCpu()


def _track(context, controls, p0c, mass0=_ELECTRON, q0=1.):
    line = xt.Line(elements={'rf': xt.MagnetOverlappedCavity(frequency=_FREQUENCY, **controls)})
    line.particle_ref = xt.Particles(p0c=p0c, mass0=mass0, q0=q0)
    line.build_tracker(_context=context)
    ends = []
    for launch in _LAUNCHES:
        particles = xt.Particles(_context=context, p0c=p0c, mass0=mass0, q0=q0, **launch)
        line.track(particles)
        assert particles.state[0] == 1
        ends.append({name: getattr(particles, name)[0] for name in COORDINATES})
    return ends


@lru_cache(maxsize=None)
def _truth(gamma, half_waves, gradient, amplitude, phase, counter_phase=0.,
           mass0=_ELECTRON, q0=1., n_cells=None, ks=0., k1=0.):
    p0c = mass0 * np.sqrt(gamma**2 - 1)
    length = half_waves * _HALF_WAVE
    voltage = gradient * length
    waves = calibrated_waves(voltage, _FREQUENCY, length, phase, amplitude, hard_edge, counter_phase,
                             n_cells, q0)
    def magnet(ramp):
        if ks == 0. and k1 == 0.:
            return None
        ends = ramped(length, ramp)

        # In this helper the solenoid ramps with the RF; the quadrupole has no fringe.
        def fields(z):
            h, dh, _ = ends(z)
            return ks * p0c / q0 * h, ks * p0c / q0 * dh, k1 * p0c / q0
        return fields

    # A hard edge is the limit of vanishing ramps, linear in the ramp's length.
    ramps = [[lorentz_end(launch, waves, length, ramped(length, ramp), p0c, mass0, q0,
                          magnet=magnet(ramp), ramp=ramp)
              for launch in _LAUNCHES] for ramp in (1e-3, 5e-4)]
    return tuple({name: 2 * fine[name] - coarse[name] for name in COORDINATES}
                 for coarse, fine in zip(*ramps))


def _errors(ends, truth, p0c, voltage):
    """Worst position error, kick error relative to the kick, energy error relative to the gain."""
    position = max(abs(e[n] - t[n]) for e, t in zip(ends, truth) for n in ('x', 'y'))
    kick = max(abs(t[n] - launch[n]) for t, launch in zip(truth, _LAUNCHES) for n in ('px', 'py'))
    momentum = max(abs(e[n] - t[n]) for e, t in zip(ends, truth) for n in ('px', 'py'))
    energy = max(abs(e['delta'] - t['delta']) for e, t in zip(ends, truth)) * p0c / voltage
    return position, momentum / kick, energy


# The ramp-resolved reference pins the energy through hard edges to about 1e-6 of the gain.
_REFERENCE_ENERGY = 5e-6

_GRID = [(gamma, gradient, half_waves, amplitude, phase)
         for gamma, gradient in ((20., 5e6), (400., 20e6), (4000., 20e6))
         for half_waves in (2, 5, 15)
         for amplitude, phase in ((0., np.pi / 2), (1., 1.2))]


@pytest.mark.parametrize('gamma, gradient, half_waves, amplitude, phase', _GRID)
def test_phase_resolved_follows_the_orbit(context, gamma, gradient, half_waves, amplitude, phase):
    length = half_waves * _HALF_WAVE
    p0c = _ELECTRON * np.sqrt(gamma**2 - 1)
    ends = _track(context, dict(model='_phase-resolved', length=length, voltage=gradient * length,
                                phase=phase, counter_wave_amplitude=amplitude, num_kicks=16 * half_waves), p0c)
    position, kick, energy = _errors(ends, _truth(gamma, half_waves, gradient, amplitude, phase), p0c,
                                     gradient * length)
    assert position < 1e-8 and kick < 1e-3 and energy < _REFERENCE_ENERGY


@pytest.mark.parametrize('gamma, gradient, half_waves, amplitude, phase', _GRID)
def test_phase_resolved_paraxial_follows_the_orbit(context, gamma, gradient, half_waves, amplitude, phase):
    length = half_waves * _HALF_WAVE
    p0c = _ELECTRON * np.sqrt(gamma**2 - 1)
    ends = _track(context, dict(model='_phase-resolved-paraxial', length=length,
                                voltage=gradient * length, phase=phase, counter_wave_amplitude=amplitude), p0c)
    position, kick, energy = _errors(ends, _truth(gamma, half_waves, gradient, amplitude, phase), p0c,
                                     gradient * length)
    # The map truncates at second order in the RF strength g/p0.
    strength = gradient / (2 * np.pi * _FREQUENCY / clight * p0c)
    assert position < 2e-7 and kick < 1e-3 + 3 * strength**2 and energy < _REFERENCE_ENERGY


@pytest.mark.parametrize('gamma, gradient', [(20., 5e6), (400., 20e6), (4000., 20e6)])
@pytest.mark.parametrize('half_waves', [2, 3, 4, 5, 6, 8, 10, 15])
def test_phase_averaged_resolves_short_travelling_waves(context, gamma, gradient, half_waves):
    # Two steps per half wavelength alone miss 69 % of the kick for two half wavelengths.
    length = half_waves * _HALF_WAVE
    p0c = _ELECTRON * np.sqrt(gamma**2 - 1)
    ends = _track(context, dict(model='_phase-averaged', length=length, voltage=gradient * length,
                                phase=np.pi / 2, counter_wave_amplitude=0.), p0c)
    position, kick, energy = _errors(ends, _truth(gamma, half_waves, gradient, 0., np.pi / 2), p0c,
                                     gradient * length)
    assert kick < (5e-3 if gamma < 100 else 3e-4)
    assert energy < _REFERENCE_ENERGY


@pytest.mark.parametrize('gamma, half_waves', [(400., 15), (4000., 15), (400., 20), (4000., 20)])
def test_phase_averaged_follows_the_orbit_where_the_envelope_beat_is_small(context, gamma, half_waves):
    length = half_waves * _HALF_WAVE
    p0c = _ELECTRON * np.sqrt(gamma**2 - 1)
    ends = _track(context, dict(model='_phase-averaged', length=length, voltage=20e6 * length,
                                phase=np.pi / 2, counter_wave_amplitude=1.), p0c)
    _, kick, _ = _errors(ends, _truth(gamma, half_waves, 20e6, 1., np.pi / 2), p0c, 20e6 * length)
    assert kick < 1e-3


@pytest.mark.parametrize('gamma, half_waves, amplitude, phase', [
    (400., 2, 0., np.pi / 2), (400., 5.3, 1., 1.2), (40., 5.5, .6, 1.2), (4000., 5.3, 0., 1.2)])
@pytest.mark.parametrize('model', ['_phase-resolved', '_phase-resolved-paraxial', '_phase-averaged',
                                   '_phase-resolved-frozen'])
def test_hard_edges_follow_the_orbit(context, model, gamma, half_waves, amplitude, phase):
    if model == '_phase-resolved-frozen' and amplitude != 0.:
        pytest.skip('The frozen map has no counter wave.')
    if model == '_phase-averaged' and amplitude != 0.:
        pytest.skip('These standing waves end mid-cell, where the average misses the edge beat.')
    gradient = 20e6 if gamma > 100 else 5e6
    length = half_waves * _HALF_WAVE
    p0c = _ELECTRON * np.sqrt(gamma**2 - 1)
    controls = dict(model=model, length=length, voltage=gradient * length, phase=phase, counter_wave_amplitude=amplitude)
    if model == '_phase-resolved':
        controls['num_kicks'] = int(16 * half_waves)
    ends = _track(context, controls, p0c)
    truth = _truth(gamma, half_waves, gradient, amplitude, phase)
    assert _errors(ends, truth, p0c, gradient * length)[1] < 3e-4


@pytest.mark.parametrize('model', ['_phase-resolved', '_phase-resolved-paraxial'])
@pytest.mark.parametrize('counter_phase', [.7, -2.])
def test_counter_wave_phase_follows_the_orbit(context, model, counter_phase):
    gamma, half_waves, gradient, amplitude, phase = 400., 5, 20e6, .8, 1.2
    length = half_waves * _HALF_WAVE
    p0c = _ELECTRON * np.sqrt(gamma**2 - 1)
    controls = dict(model=model, length=length, voltage=gradient * length, phase=phase, counter_wave_amplitude=amplitude,
                    counter_wave_phase=counter_phase)
    if model == '_phase-resolved':
        controls['num_kicks'] = 16 * half_waves
    ends = _track(context, controls, p0c)
    truth = _truth(gamma, half_waves, gradient, amplitude, phase, counter_phase)
    position, kick, energy = _errors(ends, truth, p0c, gradient * length)
    assert kick < 2e-3 and energy < _REFERENCE_ENERGY


@pytest.mark.parametrize('q0', [-1., 2.])
@pytest.mark.parametrize('model', ['_phase-resolved', '_phase-resolved-paraxial'])
def test_charge_sign_and_size_follow_the_orbit(context, model, q0):
    gamma, half_waves, gradient, amplitude, phase = 400., 5, 20e6, .5, 1.2
    length = half_waves * _HALF_WAVE
    p0c = _ELECTRON * np.sqrt(gamma**2 - 1)
    controls = dict(model=model, length=length, voltage=gradient * length, phase=phase, counter_wave_amplitude=amplitude)
    if model == '_phase-resolved':
        controls['num_kicks'] = 16 * half_waves
    ends = _track(context, controls, p0c, q0=q0)
    truth = _truth(gamma, half_waves, gradient, amplitude, phase, q0=q0)
    assert _errors(ends, truth, p0c, abs(q0) * gradient * length)[1] < 1e-3


@pytest.mark.parametrize('gamma', [1.5, 2., 5.])
@pytest.mark.parametrize('model', ['_phase-resolved', '_phase-resolved-paraxial'])
def test_low_beta_protons_follow_the_orbit_through_synchronous_cells(context, model, gamma):
    """Five cells of beta lambda/2: the waves are slower than c and not linear in r."""
    n_cells, gradient, amplitude, phase = 5, 5e6, .5, 1.2
    beta = np.sqrt(1 - 1 / gamma**2)
    half_waves = n_cells * beta
    length = half_waves * _HALF_WAVE
    mass0 = xt.PROTON_MASS_EV
    p0c = mass0 * np.sqrt(gamma**2 - 1)
    controls = dict(model=model, length=length, voltage=gradient * length, phase=phase,
                    counter_wave_amplitude=amplitude, n_cells=n_cells)
    if model == '_phase-resolved':
        controls['num_kicks'] = 16 * n_cells
    ends = _track(context, controls, p0c, mass0=mass0)
    truth = _truth(gamma, half_waves, gradient, amplitude, phase, mass0=mass0, n_cells=n_cells)
    assert _errors(ends, truth, p0c, gradient * length)[1] < 1e-3


def test_paraxial_takes_a_step_for_each_two_percent_of_energy_change(context):
    # V/E is 0.62, so 31 steps where the wave alone asks for 4.
    gamma, gradient, half_waves, phase = 5.9095, 5e6, 2, -np.pi / 2
    length = half_waves * _HALF_WAVE
    p0c = _ELECTRON * np.sqrt(gamma**2 - 1)
    common = dict(model='_phase-resolved-paraxial', length=length, voltage=gradient * length,
                  phase=phase, counter_wave_amplitude=0.)
    assert _track(context, dict(common), p0c) == _track(context, dict(num_kicks=31, **common), p0c)


def test_paraxial_steps_follow_each_particles_own_reference(context):
    # The plan is prepared once per call, and this reference differs from the line's.
    gamma, gradient, half_waves, phase = 5.9095, 5e6, 2, -np.pi / 2
    length = half_waves * _HALF_WAVE
    p0c = _ELECTRON * np.sqrt(gamma**2 - 1)
    line = xt.Line(elements={'rf': xt.Cavity(model='_phase-resolved-paraxial', frequency=_FREQUENCY, length=length,
                                             voltage=gradient * length, phase=phase, counter_wave_amplitude=0.)})
    line.particle_ref = xt.Particles(p0c=100 * p0c, mass0=_ELECTRON)
    line.build_tracker(_context=context)
    particles = xt.Particles(_context=context, p0c=p0c, mass0=_ELECTRON, **_LAUNCHES[0])
    line.track(particles)
    ends = _track(context, dict(model='_phase-resolved-paraxial', num_kicks=31, length=length,
                                voltage=gradient * length, phase=phase, counter_wave_amplitude=0.), p0c)
    assert particles.px[0] == ends[0]['px']


def test_paraxial_keeps_its_accuracy_where_it_needs_more_steps_than_the_wave(context):
    gamma, gradient, half_waves, phase = 5.9095, 5e6, 2, -np.pi / 2
    length = half_waves * _HALF_WAVE
    p0c = _ELECTRON * np.sqrt(gamma**2 - 1)
    ends = _track(context, dict(model='_phase-resolved-paraxial', length=length, voltage=gradient * length,
                                phase=phase, counter_wave_amplitude=0.), p0c)
    assert _errors(ends, _truth(gamma, half_waves, gradient, 0., phase), p0c, gradient * length)[1] < 1e-2


def test_phase_resolved_takes_sixteen_steps_per_half_wavelength(context):
    length = 30 * _HALF_WAVE
    p0c = _ELECTRON * 10.
    common = dict(model='_phase-resolved', length=length, voltage=1e6, phase=.6, counter_wave_amplitude=0.)
    assert _track(context, dict(common), p0c) == _track(context, dict(num_kicks=480, **common), p0c)


@pytest.mark.parametrize('frequency, length, kinetic_energy', [
    (2e8, .1, .5e6), (2e8, .1, 2e6), (5e7, .3, .5e6), (1e6, 1., 5e4)])
def test_phase_resolved_resolves_the_envelope_of_a_short_cavity(context, frequency, length, kinetic_energy):
    # Far under a half wavelength the wave asks for one to three steps, but the envelope does not.
    p0c = np.sqrt((_ELECTRON + kinetic_energy)**2 - _ELECTRON**2)
    ends = []
    for steps in (dict(), dict(num_kicks=2048)):
        rf = xt.Cavity(model='_phase-resolved', frequency=frequency, length=length, voltage=1e5,
                       phase=1.2, counter_wave_amplitude=0., **steps)
        particles = xt.Particles(_context=context, p0c=p0c, mass0=_ELECTRON, x=1e-3, px=1e-4)
        rf.track(particles)
        assert particles.state[0] == 1
        ends.append(particles.px[0])
    assert abs(ends[0] - ends[1]) < 1e-3 * abs(ends[1] - 1e-4)


def test_paraxial_resolves_the_envelope_of_a_short_cavity(context):
    # One step misses this kick by 10%. V/E alone asks for one.
    p0c = np.sqrt((_ELECTRON + 50e6)**2 - _ELECTRON**2)
    ends = []
    for model, steps in (('_phase-resolved-paraxial', 0), ('_phase-resolved', 4096)):
        rf = xt.Cavity(model=model, num_kicks=steps, frequency=2e8, length=.1, voltage=2e4,
                       phase=1.2, counter_wave_amplitude=0.)
        particles = xt.Particles(_context=context, p0c=p0c, mass0=_ELECTRON, x=1e-3, px=1e-4)
        rf.track(particles)
        ends.append(particles.px[0])
    assert abs(ends[0] - ends[1]) < 1e-3 * abs(ends[1] - 1e-4)


def test_phase_resolved_keeps_its_accuracy_on_a_long_cavity_that_could_stop_the_particle(context):
    gamma, gradient, half_waves, amplitude, phase = 17.5015, 30e6, 30, 1., .6
    length = half_waves * _HALF_WAVE
    p0c = _ELECTRON * np.sqrt(gamma**2 - 1)
    ends = _track(context, dict(model='_phase-resolved', length=length, voltage=gradient * length,
                                phase=phase, counter_wave_amplitude=amplitude), p0c)
    assert _errors(ends, _truth(gamma, half_waves, gradient, amplitude, phase), p0c, gradient * length)[1] < 1e-2


@pytest.mark.parametrize('gamma, voltage_over_energy', [(5., .1), (10., .05)])
def test_phase_averaged_reads_its_radial_kick_at_the_mean_energy_of_the_step(context, gamma, voltage_over_energy):
    # The error of a radial kick read at one end of the step is (dE/E per step) / 2: here 0.2 to 0.5%.
    half_waves = 6.4
    length = half_waves * _HALF_WAVE
    p0c = _ELECTRON * np.sqrt(gamma**2 - 1)
    gradient = voltage_over_energy * gamma * _ELECTRON / length
    for phase in (.6, 1.2, -.6, -1.2):
        ends = _track(context, dict(model='_phase-averaged', length=length, voltage=gradient * length,
                                    phase=phase, counter_wave_amplitude=0.), p0c)
        truth = _truth(gamma, half_waves, gradient, 0., phase)
        assert _errors(ends, truth, p0c, gradient * length)[1] < 1e-3


def _spin_end(context, model, gamma, half_waves, gradient, amplitude, phase, anomaly, spin, **controls):
    length = half_waves * _HALF_WAVE
    p0c = _ELECTRON * np.sqrt(gamma**2 - 1)
    line = xt.Line(elements={'rf': xt.Cavity(model=model, frequency=_FREQUENCY, length=length,
                   voltage=gradient * length, phase=phase, counter_wave_amplitude=amplitude, **controls)})
    line.particle_ref = xt.Particles(p0c=p0c, mass0=_ELECTRON)
    line.build_tracker(_context=context)
    line.configure_spin(spin_model=True)
    particles = xt.Particles(_context=context, p0c=p0c, mass0=_ELECTRON, spin_x=spin[0], spin_y=spin[1],
                             spin_z=spin[2], anomalous_magnetic_moment=anomaly, **_LAUNCHES[0])
    line.track(particles)
    assert particles.state[0] == 1
    return np.array([particles.spin_x[0], particles.spin_y[0], particles.spin_z[0]])


@pytest.mark.parametrize('gamma, gradient', [(40., 5e6), (400., 20e6)])
@pytest.mark.parametrize('amplitude', [0., .5, 1.])
def test_spin_through_hard_edges_follows_thomas_bmt(context, gamma, gradient, amplitude):
    """A uniform field ends at its faces: the spin turns there too. The reference
    resolves each edge as a 0.3 mm ramp, the physical limit of a hard edge."""
    half_waves, phase, anomaly = 15, 1.2, 0.00115965218
    n_cells = half_waves if amplitude > 0. else None
    spin = np.array([.6, 0., .8])
    length = half_waves * _HALF_WAVE
    p0c = _ELECTRON * np.sqrt(gamma**2 - 1)
    end = _spin_end(context, '_phase-resolved', gamma, half_waves, gradient, amplitude, phase, anomaly,
                    spin, num_kicks=16 * half_waves, n_cells=n_cells)
    waves = calibrated_waves(gradient * length, _FREQUENCY, length, phase, amplitude, hard_edge,
                             n_cells=n_cells)
    launch = dict(_LAUNCHES[0], spin_x=spin[0], spin_y=spin[1], spin_z=spin[2])
    truth = lorentz_end(launch, waves, length, ramped(length, 3e-4), p0c, _ELECTRON, anomaly=anomaly, ramp=3e-4)
    truth = np.array([truth['spin_x'], truth['spin_y'], truth['spin_z']])
    assert np.linalg.norm(end - truth) < 1e-3 * np.linalg.norm(truth - spin)


@pytest.mark.parametrize('model, gamma, gradient, tolerance', [
    ('_phase-resolved', 40., 5e6, 2e-3), ('_phase-resolved', 400., 20e6, 2e-3),
    # Sagan's edge and energy-kick spin rules, shared with Bmad, have this known error.
    ('_phase-averaged', 40., 5e6, .1), ('_phase-averaged', 400., 20e6, 5e-2)])
def test_spin_rotation_follows_thomas_bmt(context, model, gamma, gradient, tolerance):
    half_waves, amplitude, phase, anomaly = 15, 0., 1.2, 0.00115965218
    spin = np.array([.6, 0., .8])
    length = half_waves * _HALF_WAVE
    p0c = _ELECTRON * np.sqrt(gamma**2 - 1)
    controls = dict(num_kicks=16 * half_waves) if model == '_phase-resolved' else {}
    end = _spin_end(context, model, gamma, half_waves, gradient, amplitude, phase, anomaly, spin, **controls)
    waves = calibrated_waves(gradient * length, _FREQUENCY, length, phase, amplitude, hard_edge)
    launch = dict(_LAUNCHES[0], spin_x=spin[0], spin_y=spin[1], spin_z=spin[2])
    truth = lorentz_end(launch, waves, length, ramped(length, 3e-4), p0c, _ELECTRON, anomaly=anomaly, ramp=3e-4)
    truth = np.array([truth['spin_x'], truth['spin_y'], truth['spin_z']])
    rotation = np.linalg.norm(truth - spin)
    assert rotation > 0.
    assert np.linalg.norm(end - truth) < tolerance * rotation


@pytest.mark.parametrize('ks, k1', [(.3, 0.), (-.8, 0.), (0., .4), (0., -1.), (.3, .4)])
@pytest.mark.parametrize('phase', [np.pi / 2, 1.2])
def test_static_fields_inside_the_cavity_follow_the_orbit(context, ks, k1, phase):
    gamma, half_waves, gradient, amplitude = 400., 5, 20e6, .5
    length = half_waves * _HALF_WAVE
    p0c = _ELECTRON * np.sqrt(gamma**2 - 1)
    ends = _track(context, dict(model='_phase-resolved', length=length, voltage=gradient * length,
                                phase=phase, counter_wave_amplitude=amplitude, num_kicks=32 * half_waves, ks=ks,
                                knl=[0., k1 * length], edge_entry_active=ks != 0.,
                                edge_exit_active=ks != 0.), p0c)
    truth = _truth(gamma, half_waves, gradient, amplitude, phase, ks=ks, k1=k1)
    position, kick, energy = _errors(ends, truth, p0c, gradient * length)
    assert position < 2e-7 and kick < 1e-4 and energy < _REFERENCE_ENERGY


@pytest.mark.parametrize('misalignment', [dict(shift_x=1e-3), dict(shift_y=-2e-3), dict(rot_s_rad=.3),
                                          dict(shift_x=5e-4, shift_y=3e-4, rot_s_rad=-.2)],
                         ids=['shift x', 'shift y', 'roll', 'combined'])
@pytest.mark.parametrize('model', ['_phase-resolved', '_phase-resolved-paraxial', '_phase-averaged'])
def test_misalignment_moves_the_field_seen_by_the_orbit(context, model, misalignment):
    # In the cavity's frame the launch is displaced and rolled; back again at the exit.
    gamma, half_waves, gradient, amplitude, phase = 400., 15, 20e6, 0., 1.2
    length = half_waves * _HALF_WAVE
    p0c = _ELECTRON * np.sqrt(gamma**2 - 1)
    controls = dict(model=model, length=length, voltage=gradient * length, phase=phase, counter_wave_amplitude=amplitude,
                    rot_shift_anchor=length / 2, **misalignment)
    if model == '_phase-resolved':
        controls['num_kicks'] = 16 * half_waves
    ends = _track(context, controls, p0c)
    dx, dy = misalignment.get('shift_x', 0.), misalignment.get('shift_y', 0.)
    cos, sin = np.cos(misalignment.get('rot_s_rad', 0.)), np.sin(misalignment.get('rot_s_rad', 0.))
    waves = calibrated_waves(gradient * length, _FREQUENCY, length, phase, amplitude, hard_edge)
    truth = []
    for launch in _LAUNCHES:
        local = dict(launch, x=cos * (launch['x'] - dx) + sin * (launch['y'] - dy),
                     y=-sin * (launch['x'] - dx) + cos * (launch['y'] - dy),
                     px=cos * launch['px'] + sin * launch['py'],
                     py=-sin * launch['px'] + cos * launch['py'])
        end = lorentz_end(local, waves, length, ramped(length, 3e-4), p0c, _ELECTRON, ramp=3e-4)
        truth.append(dict(end, x=cos * end['x'] - sin * end['y'] + dx, y=sin * end['x'] + cos * end['y'] + dy,
                          px=cos * end['px'] - sin * end['py'], py=sin * end['px'] + cos * end['py']))
    assert _errors(ends, truth, p0c, gradient * length)[1] < 1e-3


@pytest.mark.parametrize('gamma, half_waves, phase', [(400., 40, np.pi / 2), (4000., 40, np.pi / 2), (4000., 80, 1.2)])
def test_lorentz_reference_reproduces_the_published_standing_wave_matrix(gamma, half_waves, phase):
    # The ultrarelativistic matrix of a pi-mode standing wave with hard edges, known in
    # closed form: this anchors the reference itself to a result from outside this code.
    gradient, length = 20e6, half_waves * _HALF_WAVE
    p0c = _ELECTRON * np.sqrt(gamma**2 - 1)
    waves = calibrated_waves(gradient * length, _FREQUENCY, length, phase, 1., hard_edge)
    gamma_rate = gradient * np.sin(phase) / _ELECTRON
    gamma_end = gamma + gamma_rate * length
    alpha = np.log(gamma_end / gamma) / np.sqrt(8) / np.sin(phase)
    expected = np.array([
        [np.cos(alpha) - np.sqrt(2) * np.sin(phase) * np.sin(alpha),
         np.sqrt(8) * gamma / gamma_rate * np.sin(phase) * np.sin(alpha)],
        [-(1 / np.sin(phase) + 2 * np.sin(phase)) * np.sin(alpha) * gamma_rate / (np.sqrt(8) * gamma_end),
         gamma / gamma_end * (np.cos(alpha) + np.sqrt(2) * np.sin(phase) * np.sin(alpha))]])

    def end(x, slope, ramp):
        launch = dict(x=x, px=slope, y=0., py=0., zeta=0., delta=0.)
        result = lorentz_end(launch, waves, length, ramped(length, ramp), p0c, _ELECTRON)
        return np.array([result['x'], result['px'] * p0c / (p0c * (1 + result['delta']))])

    step = 1e-6
    def matrix(ramp):
        column_x = (end(step, 0., ramp) - end(-step, 0., ramp)) / (2 * step)
        column_slope = (end(0., step, ramp) - end(0., -step, ramp)) / (2 * step)
        return np.column_stack([column_x, column_slope])
    measured = 2 * matrix(5e-4) - matrix(1e-3)
    np.testing.assert_allclose(measured, expected, rtol=2e-4, atol=0)


@pytest.mark.parametrize('model', ['_phase-resolved', '_phase-resolved-paraxial', '_phase-averaged'])
def test_tpsa_map_matches_the_derivatives_of_the_orbit(model):
    # First-order map of the tracked series against finite differences of the orbit.
    import xtrack.tpsa as xtpsa
    gamma, half_waves, gradient, amplitude, phase = 400., 15, 20e6, 0., 1.2
    length = half_waves * _HALF_WAVE
    p0c = _ELECTRON * np.sqrt(gamma**2 - 1)
    controls = dict(model=model, length=length, voltage=gradient * length, phase=phase, counter_wave_amplitude=amplitude,
                    frequency=_FREQUENCY)
    if model == '_phase-resolved':
        controls['num_kicks'] = 16 * half_waves
    line = xt.Line(elements={'rf': xt.Cavity(**controls)})
    line.particle_ref = xt.Particles(p0c=p0c, mass0=_ELECTRON)
    line.build_tracker()
    launch = _LAUNCHES[0]
    series = xtpsa.ParticlesTpsa(order=1, p0c=p0c, mass0=_ELECTRON, **launch)
    line.track(series)
    jacobian = np.asarray(series.jacobian())
    waves = calibrated_waves(gradient * length, _FREQUENCY, length, phase, amplitude, hard_edge)
    steps = dict(x=1e-5, px=1e-6, y=1e-5, py=1e-6, zeta=1e-5, delta=1e-5)
    truth = np.zeros((6, 6))
    for column, name in enumerate(COORDINATES):
        plus = lorentz_end(dict(launch, **{name: launch[name] + steps[name]}), waves, length,
                           ramped(length, 3e-4), p0c, _ELECTRON, ramp=3e-4)
        minus = lorentz_end(dict(launch, **{name: launch[name] - steps[name]}), waves, length,
                            ramped(length, 3e-4), p0c, _ELECTRON, ramp=3e-4)
        truth[:, column] = [(plus[row] - minus[row]) / (2 * steps[name]) for row in COORDINATES]
    # Relative to each element's own size, with a floor for the structural zeros.
    np.testing.assert_allclose(jacobian, truth, rtol=2e-3, atol=2e-5)
