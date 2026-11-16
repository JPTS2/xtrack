"""Properties of the cavity maps that hold without a reference orbit: symplecticity,
reversal, slicing, the order of their integrators and their spin rules. Their
accuracy against the Lorentz orbit is in test_cavity_lorentz.py.
"""
import numpy as np
import pytest
import xobjects as xo
import xtrack as xt
from scipy.constants import c as clight
from xobjects.test_helpers import for_all_test_contexts
from _symplectic import assert_symplectic, fd_jacobian

COORDS = ('x', 'px', 'y', 'py', 'zeta', 'delta')
ELECTRON = xt.ELECTRON_MASS_EV
ANOMALY = 0.00115965218
# A short, partly standing cavity at low energy, and a 1.3 GHz one at gamma 20.
SHORT_CAVITY = dict(length=.05, voltage=20e3, frequency=400e6, phase=.7,
                    counter_wave_amplitude=.5, counter_wave_phase=.3,
                    model='_phase-resolved', integrator='uniform')
L_BAND_CAVITY = dict(length=1.15, voltage=20e6, frequency=1.3e9, phase=1.2, num_kicks=32)
OFF_AXIS = dict(x=1e-3, px=2e-4, y=-1e-3, py=1e-4, zeta=1e-3, delta=1e-3)
STANDING_L_BAND = dict(length=.5, voltage=5e6, frequency=1.3e9, phase=.6,
                       counter_wave_amplitude=.55, model='_phase-resolved')
GAMMA_20_PROBE = dict(p0c=ELECTRON * np.sqrt(20.**2 - 1), mass0=ELECTRON,
                      x=1e-3, px=2e-3, y=-5e-4, zeta=1e-3, delta=1e-3)


@for_all_test_contexts
def test_thin_slices_converge_to_the_parent_map(test_context):
    """Thin nodes are markers; the drift slices carry the cells on another mesh."""
    end = {}
    for num_kicks in (16, 64):
        for sliced in (False, True):
            rf = xt.MagnetOverlappedCavity(**SHORT_CAVITY, num_kicks=num_kicks,
                                           edge_entry_active=0)
            line = xt.Line(elements={'rf': rf, 'end': xt.Marker()})
            line.particle_ref = xt.Particles(mass0=ELECTRON, p0c=2e6)
            if sliced:
                line.slice_thick_elements([xt.Strategy(xt.Teapot(2, mode='thin'))])
            line.build_tracker(_context=test_context)
            line.reset_s_at_end_turn = False

            probe = line.build_particles(x=4e-4, px=2e-3, y=-3e-4, py=-1e-3,
                                         zeta=2e-4, delta=3e-3)
            line.track(probe, ele_start=0, ele_stop=line.element_names.index('end'))
            probe.move(_context=xo.ContextCpu())

            assert probe.state[0] == 1
            end[num_kicks, sliced] = np.array([getattr(probe, n)[0] for n in COORDS])

    error_16 = np.max(np.abs(end[16, True] - end[16, False]))
    error_64 = np.max(np.abs(end[64, True] - end[64, False]))
    # Fourth-order Gauss cells: four times the cells cuts the error by about 4**4.
    assert error_64 < error_16 / 200
    assert error_64 < 1e-9


@for_all_test_contexts
@pytest.mark.parametrize('faces', [0, 1])
def test_frozen_model_is_symplectic_with_canonical_faces(test_context, faces):
    """The SAD face scales the momenta by p_after/p; the canonical face does not."""
    rf = xt.Cavity(length=1., voltage=2e6, frequency=1.3e9, phase=.4,
                   model='_phase-resolved-frozen', counter_wave_amplitude=0.,
                   edge_entry_active=faces, edge_exit_active=faces)
    line = xt.Line(elements=[rf])
    line.particle_ref = xt.Particles(mass0=ELECTRON, p0c=ELECTRON * np.sqrt(3.2**2 - 1))
    line.build_tracker(_context=test_context)

    jacobian = fd_jacobian(line, [1e-3, 2e-4, -1e-3, 1e-4, 1e-3, 1e-4], steps=1e-6)

    # The SAD face fails this at 4e-6.
    assert_symplectic(jacobian, atol=1e-7)


@for_all_test_contexts
@pytest.mark.parametrize('model', ['_phase-averaged', '_phase-resolved-paraxial'])
@pytest.mark.parametrize('amplitude', [0., 1.])
def test_averaged_and_paraxial_maps_are_symplectic(test_context, model, amplitude):
    rf = xt.Cavity(**L_BAND_CAVITY, model=model, counter_wave_amplitude=amplitude)
    line = xt.Line(elements=[rf])
    line.particle_ref = xt.Particles(mass0=ELECTRON, p0c=ELECTRON * np.sqrt(20.**2 - 1))
    line.build_tracker(_context=test_context)

    jacobian = fd_jacobian(line, [1e-3, 2e-4, -1e-3, 1e-4, 1e-3, 1e-3], steps=1e-6)

    assert_symplectic(jacobian, atol=1e-7)


@for_all_test_contexts
@pytest.mark.parametrize('radial_order', [2, 4, 6, 8])
def test_phase_resolved_is_symplectic_at_each_radial_order(test_context, radial_order):
    """Protons in five slow standing-wave cells, 10 mm off axis, where every order counts."""
    gamma, n_cells, frequency = 1.25, 5, 704.42e6
    length = n_cells * np.sqrt(1 - 1 / gamma**2) * clight / frequency / 2
    rf = xt.Cavity(model='_phase-resolved', length=length, voltage=5e6 * length,
                   frequency=frequency, phase=1.2, n_cells=n_cells, num_kicks=160,
                   radial_order=radial_order)
    line = xt.Line(elements=[rf])
    line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV,
                                     p0c=xt.PROTON_MASS_EV * np.sqrt(gamma**2 - 1))
    line.build_tracker(_context=test_context)

    jacobian = fd_jacobian(line, [1e-2, 2e-4, -6e-3, 1e-4, 1e-3, 1e-4], steps=1e-7)

    assert_symplectic(jacobian, atol=1e-7)


@pytest.mark.parametrize('integrator,num_kicks,order', [
    ('uniform', 32, 4), ('_geometric', 32, 4), ('yoshida-6', 16, 6),
    ('yoshida-8', 8, 7.5), ('yoshida-4', 32, 4)])
def test_resolved_integrators_reach_their_order(integrator, num_kicks, order):
    """Gauss cells are fourth order, and yoshida-4 is the Gauss cell itself;
    yoshida-6 and yoshida-8 compositions raise the order."""
    end = {}
    for chosen, cells in ((integrator, num_kicks), (integrator, 2 * num_kicks),
                          ('yoshida-8', 512), ('uniform', num_kicks)):
        rf = xt.Cavity(**STANDING_L_BAND, integrator=chosen, num_kicks=cells)
        probe = xt.Particles(**GAMMA_20_PROBE)
        rf.track(probe)
        assert probe.state[0] == 1
        end[chosen, cells] = np.array([probe.x[0], probe.px[0], probe.zeta[0], probe.delta[0]])
    reference = end['yoshida-8', 512]
    coarse = np.max(np.abs(end[integrator, num_kicks] - reference))
    fine = np.max(np.abs(end[integrator, 2 * num_kicks] - reference))
    assert np.log2(coarse / fine) > order - .5
    if integrator == 'yoshida-4':
        np.testing.assert_array_equal(end['yoshida-4', num_kicks], end['uniform', num_kicks])


@for_all_test_contexts
def test_frozen_uniform_and_geometric_meshes_share_their_limit(test_context):
    gap = {}
    for num_kicks in (32, 128):
        end = {}
        for integrator in ('uniform', '_geometric'):
            rf = xt.Cavity(_context=test_context, length=.5, voltage=5e6, frequency=1.3e9,
                           phase=.6, model='_phase-resolved-frozen', integrator=integrator,
                           counter_wave_amplitude=0., num_kicks=num_kicks)
            probe = xt.Particles(_context=test_context, p0c=ELECTRON * np.sqrt(20.**2 - 1),
                                 mass0=ELECTRON, x=1e-3, px=2e-3, zeta=1e-3)
            rf.track(probe)
            probe.move(_context=xo.ContextCpu())
            end[integrator] = np.array([probe.x[0], probe.px[0], probe.zeta[0], probe.delta[0]])
        gap[num_kicks] = np.max(np.abs(end['uniform'] - end['_geometric']))
    assert gap[128] < gap[32] / 3


@for_all_test_contexts
def test_averaged_geometric_and_uniform_meshes_converge_together(test_context):
    gap = {}
    for num_kicks in (16, 64):
        end = {}
        for integrator in ('uniform', '_geometric'):
            rf = xt.Cavity(**dict(L_BAND_CAVITY, num_kicks=num_kicks), _context=test_context,
                           model='_phase-averaged', integrator=integrator,
                           counter_wave_amplitude=0.)
            probe = xt.Particles(_context=test_context, p0c=ELECTRON * np.sqrt(20.**2 - 1),
                                 mass0=ELECTRON, **OFF_AXIS)
            rf.track(probe)
            probe.move(_context=xo.ContextCpu())
            end[integrator] = np.array([probe.x[0], probe.px[0], probe.zeta[0], probe.delta[0]])
        gap[num_kicks] = np.max(np.abs(end['uniform'] - end['_geometric']))
    assert gap[64] < gap[16] / 10


@for_all_test_contexts
@pytest.mark.parametrize('model', ['_phase-averaged', '_phase-resolved-paraxial'])
def test_averaged_and_paraxial_maps_reverse_and_slice_exactly(test_context, model):
    """Two slices split the 32 parent steps exactly; an uneven split would give a
    different staircase."""
    whole = xt.Line(elements={'rf': xt.Cavity(**L_BAND_CAVITY, model=model),
                              'end': xt.Marker()})
    sliced = whole.copy()
    sliced.slice_thick_elements([xt.Strategy(xt.Teapot(2, mode='thick'))])
    for line in (whole, sliced):
        line.particle_ref = xt.Particles(mass0=ELECTRON, p0c=ELECTRON * np.sqrt(20.**2 - 1))
        line.build_tracker(_context=test_context)
        line.reset_s_at_end_turn = False

    start = whole.build_particles(**OFF_AXIS)
    through_whole, through_slices = start.copy(), start.copy()
    whole.track(through_whole, ele_start=0, ele_stop=1)
    sliced.track(through_slices, ele_start=0, ele_stop=sliced.element_names.index('end'))
    back = through_whole.copy()
    whole.track(back, ele_start=0, ele_stop=1, backtrack=True)
    for particles in (start, through_whole, through_slices, back):
        particles.move(_context=xo.ContextCpu())

    for name in COORDS:
        assert getattr(through_whole, name)[0] == getattr(through_slices, name)[0]
        np.testing.assert_allclose(getattr(back, name)[0], getattr(start, name)[0],
                                   rtol=0, atol=1e-14)


@pytest.mark.parametrize('model', ['_phase-averaged', '_phase-resolved-paraxial'])
def test_averaged_and_paraxial_tpsa_match_finite_differences(model):
    import xtrack.tpsa as xtpsa
    line = xt.Line(elements={'rf': xt.Cavity(**L_BAND_CAVITY, model=model)})
    line.particle_ref = xt.Particles(mass0=ELECTRON, p0c=ELECTRON * np.sqrt(20.**2 - 1))
    line.build_tracker()
    line.reset_s_at_end_turn = False

    series = xtpsa.ParticlesTpsa(order=2, p0c=line.particle_ref.p0c[0], mass0=ELECTRON,
                                 **OFF_AXIS)
    line.track(series)
    finite = np.zeros((6, 6))
    for column, name in enumerate(COORDS):
        plus = line.build_particles(**dict(OFF_AXIS, **{name: OFF_AXIS[name] + 1e-6}))
        minus = line.build_particles(**dict(OFF_AXIS, **{name: OFF_AXIS[name] - 1e-6}))
        line.track(plus)
        line.track(minus)
        finite[:, column] = [(getattr(plus, n)[0] - getattr(minus, n)[0]) / 2e-6
                             for n in COORDS]

    np.testing.assert_allclose(series.jacobian(), finite, rtol=0, atol=2e-8)


@for_all_test_contexts
@pytest.mark.parametrize('static', [dict(knl=[0., .05]),
                                    dict(ks=.3, edge_entry_active=0, edge_exit_active=0)])
@pytest.mark.parametrize('voltage', [0., 20e6])
@pytest.mark.parametrize('model', ['_phase-averaged', '_phase-resolved-paraxial'])
def test_averaged_and_paraxial_static_fields_follow_phase_resolved(test_context, static,
                                                                   voltage, model):
    end = {}
    for chosen in (model, '_phase-resolved'):
        rf = xt.MagnetOverlappedCavity(_context=test_context, length=10 * clight / 1.3e9 / 2,
                                       n_cells=10, voltage=voltage, frequency=1.3e9,
                                       phase=1.2, model=chosen, num_kicks=64, **static)
        probe = xt.Particles(_context=test_context, p0c=ELECTRON * np.sqrt(2000.**2 - 1),
                             mass0=ELECTRON, **OFF_AXIS)
        rf.track(probe)
        probe.move(_context=xo.ContextCpu())
        assert probe.state[0] == 1
        end[chosen] = np.array([getattr(probe, n)[0] for n in COORDS])

    # The averaged map's own error over these ten cells at V/E 0.02 is 2.5e-8 in x.
    tolerance = 5e-8 if model == '_phase-averaged' and voltage else 5e-10
    np.testing.assert_allclose(end[model], end['_phase-resolved'], rtol=0, atol=tolerance)


@pytest.mark.parametrize('phase', [np.pi / 2, 1.])
@pytest.mark.parametrize('model,reference,agreement', [
    ('_phase-averaged', '_phase-resolved', .03),
    ('_phase-resolved-paraxial', '_phase-resolved', .03),
    ('_phase-resolved-frozen', '_phase-averaged', .01)])
def test_spin_rules_follow_the_finer_model(phase, model, reference, agreement):
    """Thomas-BMT through the averaged factor rules, the paraxial Magnus rotation
    and the frozen factors, which are the compact travelling-wave ones."""
    p0c = ELECTRON * np.sqrt(200.**2 - 1)
    wave = 0. if '_phase-resolved-frozen' in (model, reference) else 1.
    turned = {}
    for chosen in (model, reference):
        rf = xt.Cavity(**dict(L_BAND_CAVITY, num_kicks=64, phase=phase), model=chosen,
                       counter_wave_amplitude=wave)
        line = xt.Line(elements={'rf': rf})
        line.particle_ref = xt.Particles(p0c=p0c, mass0=ELECTRON)
        line.build_tracker()
        line.configure_spin(True)

        probe = xt.Particles(p0c=p0c, mass0=ELECTRON, x=2e-3, y=-1e-3, px=1e-4,
                             anomalous_magnetic_moment=ANOMALY, spin_z=1.)
        line.track(probe)

        assert probe.state[0] == 1
        turned[chosen] = np.array([probe.spin_x[0], probe.spin_y[0], probe.spin_z[0] - 1.])

    gap = np.linalg.norm(turned[model] - turned[reference])
    assert gap < agreement * np.linalg.norm(turned[reference])


@for_all_test_contexts
@pytest.mark.parametrize('amplitude,counter_phase', [(1., 0.), (.5, .7)])
@pytest.mark.parametrize('model', ['_phase-averaged', '_phase-resolved-paraxial'])
def test_energy_gain_includes_the_counter_wave(test_context, amplitude, counter_phase, model):
    """The shared calibration includes the counter wave's direct gain, which does not
    average away in a short cavity, 2.7 RF wavelengths here."""
    gain = {}
    for chosen in (model, '_phase-resolved'):
        rf = xt.Cavity(_context=test_context, length=1., voltage=350e6, frequency=801.58e6,
                       phase=2.6, model=chosen, counter_wave_amplitude=amplitude,
                       counter_wave_phase=counter_phase)
        probe = xt.Particles(_context=test_context, p0c=182.5e9, mass0=ELECTRON)
        rf.track(probe)
        probe.move(_context=xo.ContextCpu())
        gain[chosen] = probe.delta[0]
    np.testing.assert_allclose(gain[model], gain['_phase-resolved'], rtol=1e-6)


@for_all_test_contexts
def test_phase_resolved_ends_in_hard_edges(test_context):
    """At the end planes both waves stop: the particle leaves field-free, as in the
    paraxial map, and two slices compose to the whole cavity."""
    length = 5.3 * clight / 801.58e6 / 2
    p0c, step = ELECTRON * np.sqrt(400.**2 - 1), 1e-6
    focusing = {}
    for model, num_kicks in (('_phase-resolved', 530), ('_phase-resolved-paraxial', 170)):
        rf = xt.Cavity(_context=test_context, length=length, voltage=20e6 * length,
                       frequency=801.58e6, phase=np.pi / 2 - .7, model=model,
                       counter_wave_amplitude=.8, num_kicks=num_kicks)
        probe = xt.Particles(_context=test_context, p0c=p0c, mass0=ELECTRON,
                             x=[step, -step, 0., 0.], px=[0., 0., step, -step])
        rf.track(probe)
        probe.move(_context=xo.ContextCpu())
        assert np.all(probe.state == 1)
        x, kinetic_px = probe.x, probe.px - probe.ax
        # R11 - 1, R21 and R22 - 1 of the kinetic (x, px) map about the axis.
        focusing[model] = np.array([(x[0] - x[1]) / (2 * step) - 1,
                                    (kinetic_px[0] - kinetic_px[1]) / (2 * step),
                                    (kinetic_px[2] - kinetic_px[3]) / (2 * step) - 1])
    np.testing.assert_allclose(focusing['_phase-resolved'],
                               focusing['_phase-resolved-paraxial'], rtol=1e-5)

    end = {}
    for sliced in (False, True):
        rf = xt.Cavity(length=length, voltage=20e6 * length, frequency=801.58e6,
                       phase=np.pi / 2 - .7, model='_phase-resolved',
                       counter_wave_amplitude=.8, num_kicks=530)
        line = xt.Line(elements={'rf': rf, 'end': xt.Marker()})
        line.particle_ref = xt.Particles(p0c=p0c, mass0=ELECTRON)
        if sliced:
            line.slice_thick_elements([xt.Strategy(xt.Teapot(2, mode='thick'))])
        line.build_tracker(_context=test_context)

        probe = line.build_particles(x=1e-3, px=1e-4, y=-5e-4, zeta=1e-3, delta=1e-3)
        line.track(probe, ele_stop=line.element_names.index('end'))
        probe.move(_context=xo.ContextCpu())

        assert probe.ax[0] == 0. and probe.ay[0] == 0.
        end[sliced] = np.array([getattr(probe, n)[0] for n in COORDS])
    np.testing.assert_allclose(end[False], end[True], rtol=0, atol=1e-13)


@for_all_test_contexts
@pytest.mark.parametrize('model', ['_phase-averaged', '_phase-resolved',
                                   '_phase-resolved-frozen'])
def test_taper_pass_tracks_physical_cavities_longitudinally(test_context, model):
    """The taper pass shares the energy loss at zero frequency."""
    result = {}
    for chosen in (model, 'drift-kick-drift-expanded'):
        wave = 0. if chosen == '_phase-resolved-frozen' else 1.
        rf = xt.Cavity(length=1., voltage=10e6, frequency=0., phase=np.pi / 2,
                       model=chosen, counter_wave_amplitude=wave)
        line = xt.Line(elements=[rf])
        line.particle_ref = xt.Particles(p0c=1e10, mass0=ELECTRON)
        line.build_tracker(_context=test_context)
        line.tracker.track_flags.XS_FLAG_SR_TAPER = True

        probe = line.build_particles(x=1e-3)
        line.track(probe)
        probe.move(_context=xo.ContextCpu())
        result[chosen] = [probe.x[0], probe.px[0], probe.delta[0]]
    np.testing.assert_array_equal(result[model], result['drift-kick-drift-expanded'])


@for_all_test_contexts
@pytest.mark.parametrize('model', ['_phase-averaged', '_phase-resolved'])
def test_geometric_grading_follows_the_voltage_magnitude(test_context, model):
    """A decelerating cavity grades its cells too."""
    end = {}
    for integrator in ('_geometric', 'uniform'):
        rf = xt.Cavity(**dict(L_BAND_CAVITY, voltage=-20e6, phase=-1.2, num_kicks=8),
                       _context=test_context, model=model, integrator=integrator)
        probe = xt.Particles(_context=test_context, mass0=ELECTRON,
                             p0c=ELECTRON * np.sqrt(20.**2 - 1), **OFF_AXIS)
        rf.track(probe)
        probe.move(_context=xo.ContextCpu())
        end[integrator] = np.array([probe.x[0], probe.delta[0]])
    assert np.max(np.abs(end['_geometric'] - end['uniform'])) > 0.


@for_all_test_contexts
@pytest.mark.parametrize('model', ['_phase-averaged', '_phase-resolved-paraxial'])
def test_harmonic_resolves_steps_from_the_frequency(test_context, model):
    """A harmonic-locked cavity is the same cavity at the frequency it locks to."""
    circumference, harmonic, p0c = 100., 300, 1e10
    end = {}
    for lock in ('harmonic', 'frequency'):
        beta0 = p0c / np.hypot(p0c, ELECTRON)
        rf_clock = (dict(harmonic=harmonic) if lock == 'harmonic'
                    else dict(frequency=harmonic * beta0 * clight / circumference))
        rf = xt.Cavity(length=1., voltage=2e7, phase=1.4, model=model, **rf_clock)
        line = xt.Line(elements={'rf': rf, 'rest': xt.Drift(length=circumference - 1.)})
        line.particle_ref = xt.Particles(p0c=p0c, mass0=ELECTRON)
        line.build_tracker(_context=test_context)

        probe = line.build_particles(x=1e-3, px=1e-4, zeta=1e-3, delta=1e-4)
        line.track(probe, ele_start=0, ele_stop=1)
        probe.move(_context=xo.ContextCpu())
        end[lock] = np.array([getattr(probe, n)[0] for n in COORDS])
    np.testing.assert_allclose(end['harmonic'], end['frequency'], rtol=0, atol=1e-13)
