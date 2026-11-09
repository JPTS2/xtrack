"""Properties of the cavity maps that hold without a reference orbit: symplecticity,
reversal, the order of their integrators, spin rules, and what adaptive selects.
Their accuracy against the Lorentz orbit is in test_cavity_lorentz.py.
"""
import numpy as np
import pytest
import xobjects as xo
import xtrack as xt
from scipy.constants import c as clight
from _cavity_context import kernel_context


COORDS = ('x', 'px', 'y', 'py', 'zeta', 'delta')


@pytest.fixture(scope='module')
def model_context(tmp_path_factory):
    directory = tmp_path_factory.mktemp('cavity-model-wiring')
    with kernel_context(directory) as context:
        yield context


def cavity(model='_phase-resolved', **kwargs):
    controls = dict(length=.05, voltage=20e3, frequency=400e6, phase=.7,
                    eta=.25, counter_phase=.3, calibration_beta=1.,
                    model=model, integrator='uniform', num_kicks=16)
    if model == '_phase-resolved-frozen':
        controls.update(eta=0., integrator='_geometric')
    controls.update(kwargs)
    return xt.MagnetOverlappedCavity(**controls)


def particle(context=None):
    return xt.Particles(_context=context, mass0=xt.ELECTRON_MASS_EV,
                        q0=1., p0c=2e6, x=.0004, px=.002,
                        y=-.0003, py=-.001, zeta=.0002, delta=.003)


def coordinates(particles):
    return np.array([float(getattr(particles, name)[0]) for name in COORDS])


def test_thin_layout_converges_to_parent_map(model_context):
    """Thin nodes are markers; drift slices carry the cells on another mesh."""
    def endpoint(cells, sliced):
        line = xt.Line(elements={'rf': cavity(_context=model_context,
            num_kicks=cells, edge_entry_active=0), 'end': xt.Marker()})
        line.particle_ref = particle(model_context)
        if sliced:
            line.slice_thick_elements([xt.Strategy(xt.Teapot(2, mode='thin'))])
        line.build_tracker(_context=model_context)
        line.reset_s_at_end_turn = False
        probe = particle(model_context)
        line.track(probe, ele_start=0, ele_stop=line.element_names.index('end'))
        assert probe.state[0] == 1
        return coordinates(probe)
    errors = [np.abs(endpoint(cells, True) - endpoint(cells, False)).max()
              for cells in (16, 64)]
    # Fourth-order Gauss cells: a fourfold refinement gains about 4**4.
    assert errors[1] < errors[0] / 200
    assert errors[1] < 1e-9


@pytest.mark.parametrize('faces', [0, 1])
def test_frozen_model_is_symplectic_with_canonical_faces(model_context, faces):
    """The SAD face rescales momenta by p_after/p; the canonical face does not."""
    names = ('x', 'px', 'y', 'py', 'zeta', 'pzeta')
    base = dict(x=1e-3, px=2e-4, y=-1e-3, py=1e-4, zeta=1e-3, pzeta=1e-4)
    p0c = xt.ELECTRON_MASS_EV * np.sqrt(3.2**2 - 1)
    rf = xt.Cavity(_context=model_context, length=1., voltage=2e6,
                   frequency=1.3e9, phase=.4, model='_phase-resolved-frozen',
                   edge_entry_active=faces, edge_exit_active=faces)
    step = 1e-6
    matrix = np.zeros((6, 6))
    for column, name in enumerate(names):
        ends = []
        for sign in (1, -1):
            launch = dict(base)
            launch[name] += sign * step
            probe = xt.Particles(_context=model_context, p0c=p0c,
                                 mass0=xt.ELECTRON_MASS_EV, **launch)
            rf.track(probe)
            assert probe.state[0] == 1
            ends.append(np.array([getattr(probe, n)[0] for n in names]))
        matrix[:, column] = (ends[0] - ends[1]) / (2 * step)
    form = np.zeros((6, 6))
    for i in range(3):
        form[2*i, 2*i+1], form[2*i+1, 2*i] = 1, -1
    # The SAD face fails this at 4e-6; finite differences give about 1e-9.
    assert np.abs(matrix.T @ form @ matrix - form).max() < 1e-7


def _resolved_endpoint(integrator, cells):
    rf = xt.Cavity(length=.5, voltage=5e6, frequency=1.3e9, phase=.6, eta=.3,
                   model='_phase-resolved', integrator=integrator, num_kicks=cells)
    probe = xt.Particles(p0c=xt.ELECTRON_MASS_EV*np.sqrt(20.**2 - 1),
        mass0=xt.ELECTRON_MASS_EV, x=1e-3, px=2e-3, y=-5e-4, zeta=1e-3,
        delta=1e-3)
    rf.track(probe)
    assert probe.state[0] == 1
    return np.array([probe.x[0], probe.px[0], probe.zeta[0], probe.delta[0]])


@pytest.mark.parametrize('integrator,cells,order', [
    ('uniform', 32, 4), ('_geometric', 32, 4), ('yoshida-6', 16, 6),
    ('yoshida-8', 8, 7.5)])
def test_resolved_integrators_reach_their_order(integrator, cells, order):
    """Gauss cells are fourth order; Yoshida compositions raise the order."""
    reference = _resolved_endpoint('yoshida-8', 512)
    coarse, fine = (np.abs(_resolved_endpoint(integrator, n) - reference).max()
                    for n in (cells, 2*cells))
    assert np.log2(coarse/fine) > order - .5


def test_yoshida_4_equals_uniform_for_the_fourth_order_cell():
    np.testing.assert_array_equal(_resolved_endpoint('yoshida-4', 16),
                                  _resolved_endpoint('uniform', 16))


def test_frozen_uniform_and_geometric_meshes_share_their_limit():
    def endpoint(integrator, cells):
        rf = xt.Cavity(length=.5, voltage=5e6, frequency=1.3e9, phase=.6,
                       model='_phase-resolved-frozen', integrator=integrator,
                       num_kicks=cells)
        probe = xt.Particles(p0c=xt.ELECTRON_MASS_EV*np.sqrt(20.**2 - 1),
            mass0=xt.ELECTRON_MASS_EV, x=1e-3, px=2e-3, zeta=1e-3)
        rf.track(probe)
        return np.array([probe.x[0], probe.px[0], probe.zeta[0], probe.delta[0]])
    gaps = [np.abs(endpoint('uniform', n) - endpoint('_geometric', n)).max()
            for n in (32, 128)]
    assert gaps[1] < gaps[0]/3


def _averaged_cavity(model='_phase-averaged', **changes):
    controls = dict(length=1.15, voltage=20e6, frequency=1.3e9, phase=1.2,
                    model=model, num_kicks=32)
    controls.update(changes)
    return xt.Cavity(**controls)


def _launch(gamma, **changes):
    values = dict(x=1e-3, px=2e-4, y=-1e-3, py=1e-4, zeta=1e-3, delta=1e-3)
    values.update(changes)
    return xt.Particles(p0c=xt.ELECTRON_MASS_EV*np.sqrt(gamma**2 - 1),
                        mass0=xt.ELECTRON_MASS_EV, **values)


@pytest.mark.parametrize('model', ['_phase-averaged', '_phase-resolved-paraxial'])
@pytest.mark.parametrize('eta', [0., 1.])
def test_phase_averaged_is_symplectic(model, eta):
    names = ('x', 'px', 'y', 'py', 'zeta', 'pzeta')
    rf = _averaged_cavity(model, eta=eta)
    step = 1e-6
    matrix = np.zeros((6, 6))
    for column, name in enumerate(names):
        ends = []
        for sign in (1, -1):
            probe = xt.Particles(p0c=xt.ELECTRON_MASS_EV*np.sqrt(20.**2 - 1),
                mass0=xt.ELECTRON_MASS_EV, x=1e-3, px=2e-4, y=-1e-3, py=1e-4,
                zeta=1e-3, pzeta=1e-3)
            setattr(probe, name, getattr(probe, name) + sign*step)
            rf.track(probe)
            ends.append(np.array([getattr(probe, n)[0] for n in names]))
        matrix[:, column] = (ends[0] - ends[1]) / (2*step)
    form = np.zeros((6, 6))
    for i in range(3):
        form[2*i, 2*i+1], form[2*i+1, 2*i] = 1, -1
    assert np.abs(matrix.T @ form @ matrix - form).max() < 1e-7


@pytest.mark.parametrize('model', ['_phase-averaged', '_phase-resolved-paraxial'])
def test_phase_averaged_reverses_and_slices_exactly(model):
    names = ('x', 'px', 'y', 'py', 'zeta', 'delta')
    def line(sliced):
        out = xt.Line(elements={'rf': _averaged_cavity(model), 'end': xt.Marker()})
        out.particle_ref = _launch(20., x=0., px=0., y=0., py=0., zeta=0., delta=0.)
        if sliced:
            # Two slices split the 32 parent steps exactly; an uneven split
            # (ceil per slice) gives a different staircase.
            out.slice_thick_elements([xt.Strategy(xt.Teapot(2, mode='thick'))])
        out.build_tracker()
        out.reset_s_at_end_turn = False
        return out
    whole, sliced = line(False), line(True)
    start = _launch(20.)
    a, b = start.copy(), start.copy()
    whole.track(a, ele_start=0, ele_stop=1)
    sliced.track(b, ele_start=0, ele_stop=sliced.element_names.index('end'))
    for name in names:
        assert getattr(a, name)[0] == getattr(b, name)[0]
    whole.track(a, ele_start=0, ele_stop=1, backtrack=True)
    for name in names:
        np.testing.assert_allclose(getattr(a, name)[0], getattr(start, name)[0],
                                   rtol=0, atol=1e-14)


@pytest.mark.parametrize('model', ['_phase-averaged', '_phase-resolved-paraxial'])
def test_phase_averaged_tpsa_matches_finite_differences(model):
    import xtrack.tpsa as xtpsa
    gamma = 20.
    p0c = xt.ELECTRON_MASS_EV*np.sqrt(gamma**2 - 1)
    base = dict(x=1e-3, px=2e-4, y=-1e-3, py=1e-4, zeta=1e-3, delta=1e-3)
    names = ('x', 'px', 'y', 'py', 'zeta', 'delta')
    line = xt.Line(elements={'rf': _averaged_cavity(model)})
    line.particle_ref = xt.Particles(p0c=p0c, mass0=xt.ELECTRON_MASS_EV)
    line.build_tracker()
    line.reset_s_at_end_turn = False
    series = xtpsa.ParticlesTpsa(order=2, p0c=p0c, mass0=xt.ELECTRON_MASS_EV, **base)
    line.track(series)
    finite = np.zeros((6, 6))
    for column, name in enumerate(names):
        ends = []
        for sign in (1, -1):
            launch = dict(base)
            launch[name] += sign*1e-6
            probe = xt.Particles(p0c=p0c, mass0=xt.ELECTRON_MASS_EV, **launch)
            line.track(probe)
            ends.append(np.array([getattr(probe, n)[0] for n in names]))
        finite[:, column] = (ends[0] - ends[1]) / 2e-6
    np.testing.assert_allclose(series.jacobian(), finite, rtol=0, atol=2e-8)


def test_phase_averaged_geometric_and_uniform_converge_together():
    def endpoint(integrator, cells):
        probe = _launch(20.)
        _averaged_cavity(integrator=integrator, num_kicks=cells,
                         rf_profile='uniform', eta=0.).track(probe)
        return np.array([probe.x[0], probe.px[0], probe.zeta[0], probe.delta[0]])
    gaps = [np.abs(endpoint('_geometric', n) - endpoint('uniform', n)).max()
            for n in (16, 64)]
    assert gaps[1] < gaps[0]/10


@pytest.mark.parametrize('static', [dict(knl=[0., .05]),
    dict(ks=.3, edge_entry_active=0, edge_exit_active=0)])
@pytest.mark.parametrize('voltage', [0., 20e6])
@pytest.mark.parametrize('model', ['_phase-averaged', '_phase-resolved-paraxial'])
def test_phase_averaged_static_fields_follow_phase_resolved(static, voltage, model):
    ends = []
    for chosen in (model, '_phase-resolved'):
        probe = _launch(2000.)
        xt.MagnetOverlappedCavity(length=1.15, voltage=voltage, frequency=1.3e9,
            phase=1.2, model=chosen, num_kicks=64, **static).track(probe)
        assert probe.state[0] == 1
        ends.append(np.array([getattr(probe, n)[0]
                              for n in ('x', 'px', 'y', 'py', 'zeta', 'delta')]))
    np.testing.assert_allclose(ends[0], ends[1], rtol=0, atol=5e-10)


@pytest.mark.parametrize('phase', [np.pi/2, 1.])
@pytest.mark.parametrize('model', ['_phase-averaged', '_phase-resolved-paraxial'])
def test_phase_averaged_spin_follows_phase_resolved(phase, model):
    """Thomas-BMT: averaged factor rules and paraxial Magnus rotation."""
    gamma, anomaly = 200., 0.00115965218
    p0c = xt.ELECTRON_MASS_EV*np.sqrt(gamma**2 - 1)
    turned = {}
    for chosen in (model, '_phase-resolved'):
        line = xt.Line(elements={'rf': _averaged_cavity(chosen, phase=phase,
                                                        num_kicks=64)})
        line.particle_ref = xt.Particles(p0c=p0c, mass0=xt.ELECTRON_MASS_EV)
        line.build_tracker()
        line.configure_spin(True)
        probe = xt.Particles(p0c=p0c, mass0=xt.ELECTRON_MASS_EV, x=2e-3, y=-1e-3,
            px=1e-4, anomalous_magnetic_moment=anomaly, spin_z=1.)
        line.track(probe)
        assert probe.state[0] == 1
        turned[chosen] = np.array([probe.spin_x[0], probe.spin_y[0],
                                   probe.spin_z[0] - 1.])
    gap = np.linalg.norm(turned[model] - turned['_phase-resolved'])
    assert gap < .03*np.linalg.norm(turned['_phase-resolved'])


@pytest.mark.parametrize('eta,counter_phase', [(1., 0.), (.25, .7)])
@pytest.mark.parametrize('model', ['_phase-averaged', '_phase-resolved-paraxial'])
def test_phase_averaged_energy_gain_includes_the_counter_wave(eta, counter_phase, model):
    """The shared calibration includes the counter wave's direct gain, which
    does not average away in a short cavity (2.7 RF wavelengths here)."""
    gains = []
    for chosen in (model, '_phase-resolved'):
        rf = xt.Cavity(length=1., voltage=350e6, frequency=801.58e6, phase=2.6,
                       model=chosen, eta=eta, counter_phase=counter_phase)
        probe = xt.Particles(p0c=182.5e9, mass0=xt.ELECTRON_MASS_EV)
        rf.track(probe)
        gains.append(probe.delta[0])
    np.testing.assert_allclose(gains[0], gains[1], rtol=1e-6)


def _focusing(rf, gamma=400., step=1e-6):
    """R11 - 1, R21 and R22 - 1 of the kinetic (x, px) map about the axis."""
    probe = xt.Particles(p0c=xt.ELECTRON_MASS_EV*np.sqrt(gamma**2 - 1),
        mass0=xt.ELECTRON_MASS_EV, x=[step, -step, 0., 0.], px=[0., 0., step, -step])
    rf.track(probe)
    assert np.all(probe.state == 1)
    x, px = np.asarray(probe.x), np.asarray(probe.px - probe.ax)
    return np.array([(x[0] - x[1])/(2*step) - 1, (px[0] - px[1])/(2*step),
                     (px[2] - px[3])/(2*step) - 1])


def _short_cavity(model, cells, **changes):
    length = cells*clight/801.58e6/2
    controls = dict(length=length, voltage=20e6*length, frequency=801.58e6,
                    phase=np.pi/2 - .7, model=model)
    controls.update(changes)
    return xt.Cavity(**controls)


@pytest.mark.parametrize('profile,extra', [('uniform', {}),
                                           ('gaussian', dict(rf_profile_sigma=.13))])
def test_phase_resolved_truncates_end_fields_as_hard_edges(profile, extra):
    """A whole cavity's end planes are hard edges of both waves: the
    particle leaves field-free, as in the paraxial map, and slices compose."""
    cells = 5.3
    resolved = _short_cavity('_phase-resolved', cells, eta=.6, rf_profile=profile,
                             num_kicks=530, **extra)
    paraxial = _short_cavity('_phase-resolved-paraxial', cells, eta=.6, rf_profile=profile,
                             num_kicks=170, **extra)
    np.testing.assert_allclose(_focusing(resolved), _focusing(paraxial), rtol=1e-5)
    p0c = xt.ELECTRON_MASS_EV*np.sqrt(400.**2 - 1)
    lines = {}
    for sliced in (False, True):
        line = xt.Line(elements={'rf': resolved.copy(), 'end': xt.Marker()})
        line.particle_ref = xt.Particles(p0c=p0c, mass0=xt.ELECTRON_MASS_EV)
        if sliced:
            line.slice_thick_elements([xt.Strategy(xt.Teapot(2, mode='thick'))])
        line.build_tracker()
        lines[sliced] = line
    ends = []
    for sliced, line in lines.items():
        probe = xt.Particles(p0c=p0c, mass0=xt.ELECTRON_MASS_EV, x=1e-3, px=1e-4,
                             y=-5e-4, zeta=1e-3, delta=1e-3)
        line.track(probe, ele_stop=line.element_names.index('end'))
        assert probe.ax[0] == 0. and probe.ay[0] == 0.
        ends.append(np.array([getattr(probe, n)[0]
                              for n in ('x', 'px', 'y', 'py', 'zeta', 'delta')]))
    np.testing.assert_allclose(ends[0], ends[1], rtol=0, atol=1e-13)


@pytest.mark.parametrize('model', ['_phase-averaged', '_phase-resolved',
                                   '_phase-resolved-frozen'])
def test_taper_pass_tracks_physical_cavities_longitudinally(model):
    """The taper pass shares the energy loss at zero frequency (tapering.py)."""
    results = []
    for chosen in (model, 'drift-kick-drift-expanded'):
        line = xt.Line(elements=[xt.Cavity(length=1., voltage=10e6,
                                           frequency=0., phase=np.pi/2,
                                           model=chosen)])
        line.particle_ref = xt.Particles(p0c=1e10, mass0=xt.ELECTRON_MASS_EV)
        line.build_tracker()
        line.tracker.track_flags.XS_FLAG_SR_TAPER = True
        probe = xt.Particles(p0c=1e10, mass0=xt.ELECTRON_MASS_EV, x=1e-3)
        line.track(probe)
        results.append([probe.x[0], probe.px[0], probe.delta[0]])
    np.testing.assert_array_equal(results[0], results[1])


_PI_MODE = 9*clight/1.3e9/2


@pytest.mark.parametrize('phase', [np.pi/2, 1.])
def test_frozen_spin_uses_the_shared_factor_rules(phase):
    """The frozen factors are the compact eta = 0 factors."""
    gamma, anomaly = 200., 0.00115965218
    p0c = xt.ELECTRON_MASS_EV*np.sqrt(gamma**2 - 1)
    turned = {}
    for model in ('_phase-resolved-frozen', '_phase-averaged'):
        line = xt.Line(elements={'rf': _averaged_cavity(model, phase=phase,
            num_kicks=64, rf_profile='uniform', eta=0.)})
        line.particle_ref = xt.Particles(p0c=p0c, mass0=xt.ELECTRON_MASS_EV)
        line.build_tracker()
        line.configure_spin(True)
        probe = xt.Particles(p0c=p0c, mass0=xt.ELECTRON_MASS_EV, x=2e-3, y=-1e-3,
            px=1e-4, anomalous_magnetic_moment=anomaly, spin_z=1.)
        line.track(probe)
        assert probe.state[0] == 1
        turned[model] = np.array([probe.spin_x[0], probe.spin_y[0],
                                  probe.spin_z[0] - 1.])
    gap = np.linalg.norm(turned['_phase-resolved-frozen'] - turned['_phase-averaged'])
    assert gap < .01*np.linalg.norm(turned['_phase-averaged'])


@pytest.mark.parametrize('model', ['_phase-averaged', '_phase-resolved'])
def test_geometric_grading_ignores_the_voltage_sign(model):
    """A decelerating cavity grades its cells too (mesh policy on |V|)."""
    def endpoint(voltage, integrator):
        probe = _launch(20.)
        _averaged_cavity(model, voltage=voltage, phase=-1.2, num_kicks=8,
                         integrator=integrator).track(probe)
        return np.array([probe.x[0], probe.delta[0]])
    graded_gap = np.abs(endpoint(-20e6, '_geometric') - endpoint(-20e6, 'uniform')).max()
    assert graded_gap > 0.


def _adaptive_line(model_context, **controls):
    line = xt.Line(elements={'rf': xt.Cavity(**controls)})
    line.particle_ref = xt.Particles(p0c=1e9)
    line.build_tracker(_context=model_context)
    return line


def test_harmonic_resolves_steps_from_the_frequency(model_context):
    # A harmonic-locked cavity is the same cavity at the frequency it locks to.
    circumference, harmonic, p0c = 100., 300, 1e10
    def end(model, **rf):
        line = xt.Line(elements={'rf': xt.Cavity(length=1., voltage=2e7, phase=1.4,
                                                 model=model, **rf),
                                 'rest': xt.Drift(length=circumference - 1.)})
        line.particle_ref = xt.Particles(p0c=p0c, mass0=xt.ELECTRON_MASS_EV)
        line.build_tracker(_context=model_context)
        probe = line.build_particles(x=1e-3, px=1e-4, zeta=1e-3, delta=1e-4)
        line.track(probe, ele_start=0, ele_stop=1)
        probe.move(_context=xo.context_default)
        return np.array([getattr(probe, name)[0] for name in COORDS]), probe.beta0[0]
    for model in ('_phase-averaged', '_phase-resolved-paraxial'):
        locked, beta0 = end(model, harmonic=harmonic)
        frequency = harmonic * beta0 * clight / circumference
        direct, _ = end(model, frequency=frequency)
        np.testing.assert_allclose(locked, direct, rtol=0, atol=1e-13)


def _picks_the_same_as(model, half_waves, **controls):
    frequency, p0c = 1.3e9, 1e9
    length = half_waves * clight / frequency / 2
    def end(selector):
        rf = xt.Cavity(length=length, voltage=2e7 * length, frequency=frequency, phase=np.pi / 2,
                       model=selector, **controls)
        probe = xt.Particles(p0c=p0c, mass0=xt.ELECTRON_MASS_EV, x=1e-3, zeta=1e-3)
        rf.track(probe)
        return [getattr(probe, name)[0] for name in COORDS]
    return end('adaptive') == end(model)

