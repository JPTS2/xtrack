"""Normal Line consumers of the actual Gaussian-dipole RF2/P2 map.

The analytic zero-RF ray is independent of the implementation/reference
software. Wiring, finite-prefix and restored-map taper checks do not supply
an omitted-physics or numerical-error qualification.
"""
import json

import numpy as np
import pytest
import xobjects as xo
import xtrack as xt

from test_cavity_native import native_context

NAMES = ('x', 'px', 'y', 'py', 'zeta', 'delta')


@pytest.fixture(scope='module')
def context(tmp_path_factory):
    with native_context(tmp_path_factory.mktemp('actual-dipole')) as ctx:
        yield ctx


def controls(**updates):
    values = dict(model='phase-averaged-dipole', length=.08,
        frequency=400e6, voltage=2e3, phase=.4, eta=.25, counter_phase=.3,
        calibration_beta=.9, rf_profile='gaussian', rf_profile_sigma=.02,
        rf_profile_center=.003, knl=[.01], reference_px=.02,
        num_kicks=4, num_factor_steps=2,
        edge_entry_active=0, edge_exit_active=0, absolute_time=True)
    values.update(updates)
    return values


def make_line(ctx, **updates):
    line = xt.Line(elements=[xt.Cavity(**controls(**updates)),
        xt.ReferenceEnergyIncrease(Delta_p0c=0.), xt.Marker()],
        element_names=['rf', 'reference_update', 'end'])
    line.particle_ref = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1.,
        p0c=xt.ELECTRON_MASS_EV*np.sqrt(2.4**2-1.))
    line.reset_s_at_end_turn = False
    line.use_time_based_zeta_update = True
    return line


def launch(line, **updates):
    result = line.particle_ref.copy()
    values = dict(x=3e-4, px=.0202, y=-2e-4, py=-1e-4,
                  zeta=1e-4, delta=.002)
    values.update(updates)
    for name, value in values.items():
        setattr(result, name, value)
    return result


def coordinates(particles):
    assert particles.state[0] == 1
    return np.array([float(getattr(particles, name)[0]) for name in NAMES])


def track_rf(line, particles, *, reverse=False):
    line.track(particles, ele_start=0, ele_stop='reference_update',
               backtrack=reverse)
    return coordinates(particles)


@pytest.mark.parametrize('reference_px', (0., .02))
def test_zero_rf_follows_actual_analytic_dipole_reference(context, reference_px):
    line = make_line(context, voltage=0., reference_px=reference_px)
    line.build_tracker(_context=context)
    probe = launch(line, x=0., y=0., px=reference_px, py=0., zeta=0., delta=0.)
    expected_initial = coordinates(probe)
    rf = line['rf']
    k = 2*np.pi*rf.frequency/299792458.
    kappa = float(probe.p0c[0]/probe.mass0)
    beta = float(probe.beta0[0])
    energy = np.sqrt(1.+kappa*kappa)
    span = k*rf.length
    bend = kappa*rf.knl[0]/(rf.length*k)
    pa = kappa*reference_px
    pb = pa-bend*span
    sa, sb = (np.sqrt(energy*energy-1.-p*p) for p in (pa, pb))
    orbit = span*(pa+pb)/(sa+sb)
    clock = energy/bend*np.arctan2(bend*(span*sb+pb*orbit), sa*sb+pa*pb)
    expected = np.array([orbit/k, pb/kappa, 0., 0.,
                         rf.length-beta*clock/k, 0.])
    np.testing.assert_allclose(track_rf(line, probe), expected,
                               atol=4e-13, rtol=4e-13)
    # reference_px=0 still bends: the actual orbit is never replaced by zero.
    assert abs(probe.x[0]) > 1e-5
    np.testing.assert_allclose(track_rf(line, probe, reverse=True), expected_initial,
                               atol=4e-13, rtol=4e-13)


def test_normal_set_reference_programme_twiss_copy_and_json(context):
    line = make_line(context)
    line.set('programme_px', .02)
    line.set('rf_phase', .4)
    line.set('rf', reference_px='programme_px', phase='rf_phase')
    line.build_tracker(_context=context)
    line.set('programme_px', .03)
    line.set('rf_phase', .55)
    original = launch(line)
    endpoint = track_rf(line, original)
    fresh = make_line(context, reference_px=.03, phase=.55)
    fresh.build_tracker(_context=context)
    np.testing.assert_allclose(endpoint, track_rf(fresh, launch(fresh)), atol=4e-14, rtol=0.)
    for result in (line.copy(), xt.Line.from_dict(
            json.loads(json.dumps(line.to_dict(), cls=xo.JEncoder)))):
        result.build_tracker(_context=context)
        assert result['rf'].reference_px == .03
        np.testing.assert_allclose(track_rf(result, launch(result)), endpoint,
                                   atol=4e-14, rtol=0.)
    line['reference_update'].Delta_p0c = line.particle_ref.p0c[0]*endpoint[5]
    twiss = line.twiss(method='6d', start=xt.START, end=xt.END,
        **dict(zip(NAMES, coordinates(launch(line)))),
        betx=2., alfx=.1, bety=3., alfy=-.2)
    line['reference_update'].Delta_p0c = 0.
    assert np.all(np.isfinite(twiss.betx)) and np.all(twiss.betx > 0.)
    assert np.all(np.isfinite(twiss.bety)) and np.all(twiss.bety > 0.)


@pytest.mark.parametrize('mode', ('thick', 'thin'))
def test_slices_retain_fixed_parent_prefix_programme(context, mode):
    whole = make_line(context)
    sliced = make_line(context)
    sliced.slice_thick_elements([xt.Strategy(
        slicing=xt.Teapot(3, mode=mode), element_type=xt.Cavity)])
    whole.build_tracker(_context=context)
    sliced.build_tracker(_context=context)
    initial = coordinates(launch(whole))
    expected = track_rf(whole, launch(whole))
    probe = launch(sliced)
    np.testing.assert_allclose(track_rf(sliced, probe), expected, atol=3e-12, rtol=0.)
    np.testing.assert_allclose(track_rf(sliced, probe, reverse=True), initial,
                               atol=3e-12, rtol=0.)
    assert sliced['rf'].reference_px == .02


@pytest.mark.parametrize('updates', ({'ks':.01}, {'ksl':[.001]},
                                    {'knl':[.01,.001]}))
def test_actual_dipole_refuses_other_static_sources(context, updates):
    line = make_line(context, **updates)
    line.build_tracker(_context=context)
    probe = launch(line)
    initial = coordinates(probe)
    line.track(probe, ele_start=0, ele_stop='reference_update')
    assert probe.state[0] < 0
    np.testing.assert_array_equal(np.array([getattr(probe, name)[0]
                                          for name in NAMES]), initial)


def test_dipole_taper_closes_actual_rf_and_saved_profile(context, monkeypatch):
    """A known energy sink exercises taper orchestration, not radiation law."""
    rf = xt.Cavity(**controls(voltage=2e4, knl=[.001], reference_px=.002))
    line = xt.Line(elements=[
        xt.LineSegmentMap(length=1., qx=.23, qy=.31, betx=2., bety=3.,
            longitudinal_mode='frozen', energy_increment=-100.),
        xt.Multipole(knl=[0.]), rf], element_names=['sink', 'taper_slot', 'rf'])
    line.particle_ref = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1.,
        p0c=xt.ELECTRON_MASS_EV*np.sqrt(2.4**2-1.))
    line.build_tracker(_context=context)
    hardware = {name:getattr(rf, name) for name in
                ('voltage', 'frequency', 'harmonic', 'phase', 'lag', 'reference_px')}
    # The preparation follows its closed-orbit launch, which differs from
    # particle_ref in a lattice containing a real transverse dipole field.
    # Capture that exact returned particle without changing the search/map.
    prepared_launches = []
    find_closed_orbit = line.find_closed_orbit
    def capture_launch(*args, **kwargs):
        result = find_closed_orbit(*args, **kwargs)
        prepared_launches.append(result.copy())
        return result
    monkeypatch.setattr(line, 'find_closed_orbit', capture_launch)
    line.compensate_radiation_energy_loss(delta0=0., rtol_eneloss=1e-12,
                                         verbose=False)
    for name, value in hardware.items():
        assert getattr(rf, name) == value
    assert prepared_launches
    probe = prepared_launches[-1].copy()
    probe.delta = 0.
    energy0 = float(probe.energy[0])
    line.track(probe, turn_by_turn_monitor='ONE_TURN_EBE')
    assert probe.state[0] == 1
    means = .5*(line.record_last_track.delta[0, :-1]
                +line.record_last_track.delta[0, 1:])
    item = line.attr._cache['delta_taper']
    tolerance = 1e-12*energy0/float(probe.beta0[0]*probe.p0c[0])
    np.testing.assert_allclose(item.multisetter.get_values(), means[item.mask],
                               rtol=0., atol=tolerance)
    assert abs(float(probe.energy[0])-energy0) <= 1e-12*energy0
