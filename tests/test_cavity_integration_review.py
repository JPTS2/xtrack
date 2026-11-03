"""Independent normal-Line checks for the public physical cavity models.

These checks compare live mutation with fresh construction, and invariances
of the laboratory RF clock. They do not import the scientific Python maps or
infer physical/source qualification from agreement between native runs.
"""
import numpy as np
import pytest
import xobjects as xo

import xtrack as xt

from _cavity_context import kernel_context


MODELS = ('phase-resolved', 'phase-resolved-frozen')
EVOLVING_MODELS = MODELS[:1]
COORDINATES = ('x', 'px', 'y', 'py', 'zeta', 'delta')
MASS_EV = 510998.95
P0C_EV = MASS_EV*np.sqrt(2.4**2-1.)


@pytest.fixture(scope='module')
def cpu_context(tmp_path_factory):
    """Keep JIT output context-local."""
    with kernel_context(tmp_path_factory.mktemp('cavity-integration-review')) as context:
        yield context


def hardware(model):
    """Small declared hardware, with nonzero counter-wave where retained."""
    return dict(model=model, length=.05, voltage=2e4, frequency=400e6,
        phase=.35, eta=0. if model == 'phase-resolved-frozen' else .25,
        counter_phase=.3, calibration_beta=1., num_kicks=16,
        absolute_time=True)


def particle(context, **overrides):
    """Use a finite off-axis launch, independently of any reference donor."""
    values = dict(mass0=MASS_EV, p0c=P0C_EV, q0=1., x=3e-5,
        px=4e-4, y=-2e-5, py=-3e-4, zeta=2e-4, delta=1e-3)
    values.update(overrides)
    return xt.Particles(_context=context, **values)


def values(particles):
    assert particles.state[0] == 1
    return np.array([getattr(particles, name)[0] for name in COORDINATES])


def line_for(context, controls):
    line = xt.Line(elements=[xt.Cavity(_context=context, **controls)],
                   element_names=['rf'])
    line.particle_ref = xt.Particles(_context=context, mass0=MASS_EV,
                                    p0c=P0C_EV, q0=1.)
    line.reset_s_at_end_turn = False
    line.build_tracker(_context=context)
    return line


def endpoint(line, context):
    probe = particle(context)
    line.track(probe)
    return values(probe)


@pytest.mark.parametrize('model', EVOLVING_MODELS)
def test_live_expression_updates_equal_fresh_construction(cpu_context, model):
    """A source cached at construction must not mask ordinary variable edits."""
    controls = hardware(model)
    line = line_for(cpu_context, controls)
    line.set('rf_phase', controls['phase'])
    line.set('rf_eta', controls['eta'])
    line.set('rf', phase='rf_phase', eta='rf_eta')
    first = endpoint(line, cpu_context)
    for phase, eta in ((.75, .49), (-.2, .09)):
        line.set('rf_phase', phase)
        line.set('rf_eta', eta)
        actual = endpoint(line, cpu_context)
        fresh = line_for(cpu_context, dict(controls, phase=phase, eta=eta))
        np.testing.assert_allclose(actual, endpoint(fresh, cpu_context),
                                   rtol=0., atol=2e-13)
        assert np.linalg.norm(actual-first) > 1e-6


@pytest.mark.parametrize('model', MODELS)
def test_copy_json_rebuild_and_reverse_preserve_public_map(cpu_context, model):
    """Copy/load is tested for ordinary fields, without active TPSA handles."""
    line = line_for(cpu_context, hardware(model))
    expected = endpoint(line, cpu_context)
    for copy in (line.copy(_context=cpu_context),
                 xt.Line.from_dict(line.to_dict())):
        if copy.tracker is None:
            copy.build_tracker(_context=cpu_context)
        np.testing.assert_allclose(endpoint(copy, cpu_context), expected,
                                   rtol=0., atol=2e-13)
    initial = particle(cpu_context)
    tracked = initial.copy()
    line.track(tracked)
    line.track(tracked, backtrack=True)
    np.testing.assert_allclose(values(tracked), values(initial),
                               rtol=0., atol=2e-11)


@pytest.mark.parametrize('model', MODELS)
def test_absolute_clock_origin_is_one_shared_physical_event(cpu_context, model):
    """A clock-origin change compensated in RF phase leaves the map invariant."""
    controls = hardware(model)
    first = xt.Cavity(_context=cpu_context, **controls)
    delay = .73e-9
    second = xt.Cavity(_context=cpu_context,
        **dict(controls, phase=controls['phase']-2*np.pi*controls['frequency']*delay))
    before = particle(cpu_context, time_s=1.9e-9, s=.17)
    after = particle(cpu_context, time_s=1.9e-9+delay, s=.17)
    first.track(before)
    second.track(after)
    np.testing.assert_allclose(values(before), values(after),
                               rtol=0., atol=2e-12)


@pytest.mark.parametrize('model', EVOLVING_MODELS)
def test_normal_match_uses_selected_tracking_map(cpu_context, model):
    """Recover an energy target produced by tracking at another RF phase."""
    controls = hardware(model)
    target_controls = dict(controls, phase=.7)
    reference = xt.Particles(_context=cpu_context, mass0=MASS_EV,
                             p0c=P0C_EV, q0=1.)
    pilot = reference.copy()
    xt.Cavity(_context=cpu_context, **target_controls).track(pilot)
    assert pilot.state[0] == 1
    target_delta = float(pilot.delta[0])
    line = line_for(cpu_context, controls)
    line.set('phase_knob', controls['phase'])
    line.set('rf', phase='phase_knob')
    line.match(method='6d', start=xt.START, end=xt.END,
        betx=2., alfx=.3, bety=3., alfy=-.2,
        vary=[xt.Vary('phase_knob', step=1e-4)],
        targets=[xt.Target('delta', at=xt.END, value=target_delta, tol=1e-11)])
    matched = reference.copy()
    line.track(matched)
    assert matched.state[0] == 1
    assert abs(float(matched.delta[0])-target_delta) < 1e-11


@pytest.mark.parametrize('model,field', [('phase-resolved', 'eta'),
    ('phase-resolved', 'edge_entry_active'), ('phase-resolved', 'edge_exit_active'),
    ('phase-resolved', 'order'), ('phase-resolved-frozen', 'order')])
def test_zero_controls_survive_json_roundtrip(model, field):
    """Zero hardware/face settings must not turn into constructor defaults."""
    original = xt.Cavity(**dict(hardware(model), **{field: 0}))
    restored = xt.Cavity.from_dict(original.to_dict())
    assert getattr(restored, field) == 0
    assert restored.model == original.model


@pytest.mark.parametrize('field', ('calibration_beta', 'rf_field_order'))
def test_inadmissible_zero_controls_refuse_at_construction(field):
    with pytest.raises(ValueError):
        xt.Cavity(**dict(hardware('phase-resolved-frozen'), **{field: 0}))


def test_failed_knob_apply_restores_original_series_and_new_promotions():
    """A late descriptor refusal must roll back earlier variable propagation."""
    madng_tpsa = pytest.importorskip('madng_tpsa')
    from xtrack.tpsa import KnobParameters

    line = xt.Line(elements={name:xt.Quadrupole(length=.1)
        for name in ('same', 'fresh', 'other')},
        element_names=['same', 'fresh', 'other'])
    for name, value in (('a', .2), ('c', .3), ('b', .4)):
        line.set(name, value)
    line.set('same', k1='a')
    line.set('fresh', k1='c')
    line.set('other', k1='b')
    descriptor = madng_tpsa.Descriptor(6, 1, num_params=3, param_order=1)
    other_descriptor = madng_tpsa.Descriptor(6, 1, num_params=1, param_order=1)
    line.vars['a'] = descriptor.param(3, .2)
    original_a = line['a']
    same, fresh, other = (line.element_dict[name]
                          for name in ('same', 'fresh', 'other'))
    other.k1s = other_descriptor.param(1, .7)
    assert same._tpsa_enabled == other._tpsa_enabled == 1
    assert fresh._tpsa_enabled == 0
    parameters = KnobParameters(line, ['a', 'c', 'b'], descriptor)
    try:
        with pytest.raises(ValueError, match='descriptor'):
            parameters.apply()
        assert not parameters.applied
        assert line['a'] is original_a
        assert line['a'].param_grad() == [0., 0., 1.]
        assert line['c'] == .3 and not isinstance(line['c'], madng_tpsa.Tpsa)
        assert line['b'] == .4 and not isinstance(line['b'], madng_tpsa.Tpsa)
        assert fresh._tpsa_enabled == 0
        assert fresh.k1 == .3
        assert same._tpsa_enabled == other._tpsa_enabled == 1
        assert same.k1.param_grad() == [0., 0., 1.]
        assert other.k1s.param_grad() == [1.]
        assert other.k1.const_part == .4
        parameters.teardown()
        assert line['a'] is original_a
        # Restoration must preserve the existing expression dependencies too.
        line.set('c', .31)
        line.set('b', .42)
        assert fresh.k1 == .31
        assert other.k1.const_part == .42
        assert other.k1s.param_grad() == [1.]
    finally:
        for owner in (same, fresh, other):
            owner.disable_tpsa()


C_LIGHT = 299792458.
CLOCK_BETA = float(P0C_EV/np.sqrt(P0C_EV**2+MASS_EV**2))
CLOCK_LAUNCH = dict(s=.17, time_s=1.9e-9, zeta=.02, delta=.03,
                    px=0., py=0.)


def clock_line(context, ramp, mode):
    """A reference-only momentum change between two physical drifts."""
    line = xt.Line(elements=[xt.Drift(length=.2),
        xt.ReferenceEnergyIncrease(Delta_p0c=ramp), xt.Drift(length=.3),
        xt.Marker()], element_names=['d0', 'reference', 'd1', 'end'])
    line.particle_ref = xt.Particles(mass0=MASS_EV, p0c=P0C_EV, q0=1.)
    line.reset_s_at_end_turn = False
    line.use_time_based_zeta_update = mode
    line.build_tracker(_context=context, compile=False)
    return line


def physical_clock_chart(particles):
    """Read scalar coordinates in one fixed momentum and physical-ct chart."""
    scale = float(particles.p0c[0])/P0C_EV
    ct = C_LIGHT*float(particles.time_s)+(particles.s[0]-particles.zeta[0])/particles.beta0[0]
    return np.array([particles.x[0], scale*particles.px[0],
        particles.y[0], scale*particles.py[0], CLOCK_BETA*ct,
        scale*(1.+particles.delta[0])-1.])


def tpsa_physical_clock_chart(particles):
    """Same chart, retaining every represented first-order coefficient."""
    scale = float(particles.p0c/P0C_EV)
    ct = C_LIGHT*particles.time_s+(particles._local_series['s']-particles.zeta)/particles.beta0
    series = (particles.x, scale*particles.px, particles.y,
        scale*particles.py, CLOCK_BETA*ct, scale*(1.+particles.delta)-1.)
    return (np.array([value.const_part for value in series]),
            np.array([value.grad() for value in series]))


@pytest.mark.parametrize('mode', (False, True))
def test_time_based_clock_option_survives_copy_json(cpu_context, mode):
    line = clock_line(cpu_context, .4*MASS_EV, mode)
    assert xt.Line(elements=[xt.Drift(length=1.)]).use_time_based_zeta_update is False
    for restored in (line.copy(), xt.Line.from_dict(line.to_dict())):
        assert restored.use_time_based_zeta_update is mode
    with pytest.raises(TypeError, match='boolean'):
        line.use_time_based_zeta_update = 'True'
    assert line.use_time_based_zeta_update is mode


def test_reference_ramp_preserves_physical_clock_across_calls_and_reverse(cpu_context):
    ramped = clock_line(cpu_context, .4*MASS_EV, True)
    control = clock_line(cpu_context, 0., True)
    initial = particle(cpu_context, **CLOCK_LAUNCH)
    start = physical_clock_chart(initial)
    split, baseline = initial.copy(), initial.copy()
    for index in range(4):
        before = physical_clock_chart(split)
        ramped.tracker.track_flags.XS_FLAG_USE_TIME_BASED_ZETA_UPDATE = False
        ramped.track(split, ele_start=index, ele_stop=index+1)
        control.track(baseline, ele_start=index, ele_stop=index+1)
        assert ramped.tracker.track_flags.XS_FLAG_USE_TIME_BASED_ZETA_UPDATE
        np.testing.assert_allclose(physical_clock_chart(split),
            physical_clock_chart(baseline), rtol=0., atol=3e-13)
        if index == 1:
            np.testing.assert_allclose(physical_clock_chart(split), before,
                                       rtol=0., atol=3e-13)
    full = initial.copy()
    ramped.track(full)
    np.testing.assert_allclose(physical_clock_chart(full),
                               physical_clock_chart(split), rtol=0., atol=3e-13)
    momentum = P0C_EV*(1.+CLOCK_LAUNCH['delta'])
    beta = momentum/np.sqrt(momentum**2+MASS_EV**2)
    assert abs(physical_clock_chart(full)[4]-start[4]-.5*CLOCK_BETA/beta) < 3e-13
    ramped.track(full, backtrack=True)
    np.testing.assert_allclose(physical_clock_chart(full), start,
                               rtol=0., atol=3e-13)


def test_default_reference_ramp_retains_relative_delay_convention(cpu_context):
    line = clock_line(cpu_context, .4*MASS_EV, False)
    tracked = particle(cpu_context, **CLOCK_LAUNCH)
    line.track(tracked, ele_start=0, ele_stop=1)
    before = physical_clock_chart(tracked)[4]/CLOCK_BETA
    old_beta = float(tracked.beta0[0])
    delay = float(tracked.zeta[0])/old_beta
    position = float(tracked.s[0])
    line.tracker.track_flags.XS_FLAG_USE_TIME_BASED_ZETA_UPDATE = True
    line.track(tracked, ele_start=1, ele_stop=2)
    assert not line.tracker.track_flags.XS_FLAG_USE_TIME_BASED_ZETA_UPDATE
    new_beta = float(tracked.beta0[0])
    assert abs(float(tracked.zeta[0])/new_beta-delay) < 3e-13
    jump = physical_clock_chart(tracked)[4]/CLOCK_BETA-before
    assert abs(jump-position*(1./new_beta-1./old_beta)) < 3e-13
    assert abs(jump) > 1e-4


@pytest.fixture
def reference_clock_tpsa_case(cpu_context, cleanup):
    from xtrack.tpsa import ParticlesTpsa
    ramped = clock_line(cpu_context, .4*MASS_EV, True)
    control = clock_line(cpu_context, 0., True)
    launch = dict(mass0=MASS_EV, p0c=P0C_EV, q0=1., **CLOCK_LAUNCH)
    full, split, baseline = (ParticlesTpsa(order=1, **launch) for _ in range(3))
    ramped.track(full)
    for index in range(4):
        ramped.track(split, ele_start=index, ele_stop=index+1)
    control.track(baseline)
    try:
        yield full, split, baseline
    finally:
        # Pytest retains fixture argument objects through other finalizers.
        # Release their ABI/reference buffers before the repository's explicit
        # default-context ownership check, rather than bypassing that check.
        for tracked in (full, split, baseline):
            tracked._xobject = None
            tracked._ref_particle = None


def test_reference_ramp_tpsa_clock_constant_and_jacobian(reference_clock_tpsa_case):
    full, split, baseline = reference_clock_tpsa_case
    expected = tpsa_physical_clock_chart(baseline)
    for tracked in (full, split):
        assert tracked._xobject.state == 1
        actual = tpsa_physical_clock_chart(tracked)
        np.testing.assert_allclose(actual[0], expected[0], rtol=0., atol=3e-13)
        np.testing.assert_allclose(actual[1], expected[1], rtol=0., atol=3e-12)


def test_reference_ramp_tpsa_converter_uses_current_reference(reference_clock_tpsa_case):
    """A scalar validation particle must carry the tracked reference momentum."""
    tracked, _, _ = reference_clock_tpsa_case
    converted = tracked.to_particles()
    assert converted.p0c[0] == tracked.p0c
    assert converted.beta0[0] == tracked.beta0
    np.testing.assert_allclose(physical_clock_chart(converted),
        tpsa_physical_clock_chart(tracked)[0], rtol=0., atol=3e-13)


def reference_only_line(context, mode, kind='increase'):
    reference = (xt.ReferenceEnergyIncrease(Delta_p0c=.4*MASS_EV)
        if kind == 'increase' else xt.ReferenceEnergyChange(p0c=P0C_EV+.4*MASS_EV))
    line = xt.Line(elements=[reference,
                             xt.Marker()], element_names=['reference', 'end'])
    line.particle_ref = xt.Particles(mass0=MASS_EV, p0c=P0C_EV, q0=1.)
    line.reset_s_at_end_turn = False
    line.use_time_based_zeta_update = mode
    line.build_tracker(_context=context, compile=False)
    return line


@pytest.mark.parametrize('mode', (False, True))
@pytest.mark.parametrize('kind', ('increase', 'change'))
def test_reference_rebase_preserves_scalar_nonzero_potential(tmp_path, monkeypatch, mode, kind):
    """p0*px and p0*A, hence mechanical transverse momentum, are invariant."""
    monkeypatch.chdir(tmp_path)
    context = xo.ContextCpu(omp_num_threads=0)
    line = reference_only_line(context, mode, kind)
    launch = dict(CLOCK_LAUNCH, px=.13, py=-.07, ax=.025, ay=-.017)
    tracked = particle(context, **launch)

    def chart():
        scale = float(tracked.p0c[0])/float(P0C_EV)
        px, py, ax, ay = (float(getattr(tracked, name)[0])
                          for name in ('px', 'py', 'ax', 'ay'))
        return scale*np.array([px, py, ax, ay, px-ax, py-ay])

    expected = chart()
    line.track(tracked)
    np.testing.assert_allclose(chart(), expected, rtol=0., atol=3e-14)
    # The absolute setter has no inverse reference history; the increment
    # element supplies an actual signed inverse.
    if kind == 'increase':
        line.track(tracked, backtrack=True)
        np.testing.assert_allclose(chart(), expected, rtol=0., atol=3e-14)


def test_reference_rebase_preserves_tpsa_nonzero_potential(cpu_context):
    from xtrack.tpsa import ParticlesTpsa
    line = reference_only_line(cpu_context, True)
    launch = dict(CLOCK_LAUNCH, mass0=MASS_EV, p0c=P0C_EV, q0=1.,
                  px=.13, py=-.07, ax=.025, ay=-.017)
    tracked = ParticlesTpsa(order=1, **launch)
    ax, ay = (tracked._local_series[name] for name in ('ax', 'ay'))
    assert ax.const_part == launch['ax']
    assert ay.const_part == launch['ay']
    # The metadata itself can carry canonical derivatives from a field map.
    ax.set((1, 0, 0, 0, 0, 0), .2)
    ay.set((0, 0, 1, 0, 0, 0), -.3)

    def chart():
        scale = float(tracked.p0c/P0C_EV)
        series = (tracked.px, tracked.py, ax, ay,
                  tracked.px-ax, tracked.py-ay)
        return (np.array([scale*item.const_part for item in series]),
                np.array([scale*np.asarray(item.grad()) for item in series]))

    expected = chart()
    line.track(tracked)
    actual = chart()
    np.testing.assert_allclose(actual[0], expected[0], rtol=0., atol=3e-14)
    np.testing.assert_allclose(actual[1], expected[1], rtol=0., atol=3e-14)


def test_gaussian_readback_does_not_revisit_non_cavity_elements():
    """Repeated strength readback must not scan an ordinary lattice in Python.

    Exercise the correction independently of tracker construction or timing:
    after structural discovery no ordinary element needs to be read again.
    """
    from types import SimpleNamespace

    class NoElementReads(dict):
        def __getitem__(self, key):
            pytest.fail('Gaussian correction revisited an ordinary element')

    elements = {'drift': xt.Drift(length=.3),
                'quad': xt.Quadrupole(length=.2, k1=.4)}
    line = SimpleNamespace(element_names=list(elements), _element_dict=elements)
    attributes = xt.line.LineAttr(line, fields={})
    line._element_dict = NoElementReads(elements)
    original = np.array([.3, .08])
    for key in ('ksoll', '_k1l_no_rel', '_k1sl_no_rel'):
        assert attributes._gaussian_cavity_integrals(key, original) is original


def test_gaussian_readback_keeps_live_parent_slice_and_replica_controls():
    """Structural caching must not freeze a profile, width or interval.

    The independent reference integrates the physical finite Gaussian parent;
    ks is peak strength, while knl/ksl are finite-parent integrals.
    """
    from math import erf, sqrt, pi
    from types import SimpleNamespace

    parent = xt.Cavity(model='phase-resolved', length=.12, voltage=0.,
        frequency=400e6, ks=.3, knl=[0., .08], ksl=[0., -.02],
        edge_entry_active=0, edge_exit_active=0)
    interval = xt.DriftSliceCavity(_parent=parent, _buffer=parent._buffer,
        weight=.25, slice_offset=.015)
    elements = {'drift': xt.Drift(length=.3), 'rf': parent,
                'interval': interval,
                'rf_replica': xt.Replica(parent_name='rf'),
                'interval_replica': xt.Replica(parent_name='interval')}
    line = SimpleNamespace(element_names=list(elements), _element_dict=elements)
    attributes = xt.line.LineAttr(line, fields={})

    for profile, sigma, length, strength, weight, start in (
            ('uniform', .02, .12, .3, .25, .125),
            ('gaussian', .02, .12, .3, .25, .125),
            ('gaussian', .035, .18, -.2, .4, .2),
            ('uniform', .035, .18, -.2, .4, .2)):
        parent.magnetic_profile = profile
        parent.magnetic_profile_sigma = sigma
        parent.length = length
        parent.ks = strength
        parent.knl[1], parent.ksl[1] = 2*strength, -strength/3
        interval.weight = weight
        interval.slice_offset = start*length

        def integral(a, b):
            scale = sqrt(2.)*sigma
            return sqrt(pi/2.)*sigma*(erf(b/scale)-erf(a/scale))

        full = integral(-length/2, length/2)
        partial = integral(length*(start-.5), length*(start+weight-.5))
        for key, coefficient in (('ksoll', strength),
                ('_k1l_no_rel', 2*strength), ('_k1sl_no_rel', -strength/3)):
            nominal_full = coefficient*length if key == 'ksoll' else coefficient
            nominal = np.array([0., nominal_full, nominal_full*weight,
                                nominal_full, nominal_full*weight])
            expected = nominal.copy()
            if profile == 'gaussian':
                if key == 'ksoll':
                    full_value, partial_value = coefficient*full, coefficient*partial
                else:
                    full_value, partial_value = coefficient, coefficient*partial/full
                expected[1:] = full_value, partial_value, full_value, partial_value
            actual = attributes._gaussian_cavity_integrals(key, nominal)
            np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-16)
            if profile == 'uniform':
                assert actual is nominal


NUMERIC_FIELDS = COORDINATES + (
    'pzeta', 'rvv', 'rpp', 's', 'ax', 'ay', 'spin_x', 'spin_y', 'spin_z',
    'p0c', 'gamma0', 'beta0', 'chi', 'charge_ratio', 'weight',
    'anomalous_magnetic_moment')


def numeric_rows_by_id(particles):
    """A failed ordinary kernel can compact losses; identity owns each row."""
    return {int(identifier): np.array([getattr(particles, name)[index]
            for name in NUMERIC_FIELDS])
            for index, identifier in enumerate(particles.particle_id)}


@pytest.fixture(scope='module', params=(0, 1), ids=('serial', 'openmp_one_thread'))
def thread_context(request, tmp_path_factory):
    with kernel_context(tmp_path_factory.mktemp('cavity-thread-context')) as serial:
        yield serial if request.param == 0 else xo.ContextCpu(omp_num_threads=1)


@pytest.mark.parametrize('model', ('phase-resolved',))
@pytest.mark.parametrize('cells', (3, 64))
def test_compact_p16_face_has_exactly_zero_potential_readback(cpu_context, model, cells):
    """Observe the declared support face, independently of accumulated z+=h.

    The synthetic 64-cell mesh previously finished a few ulps inside support,
    manufacturing tiny nonzero caches at an exactly field-free plane.
    """
    line = xt.Line(elements=[xt.Cavity(_context=cpu_context, model=model,
        rf_profile='p16', length=.56, frequency=712e6, voltage=2000.,
        phase=.4, eta=.25, counter_phase=.3, calibration_beta=.9,
        num_kicks=cells, absolute_time=True), xt.Marker(_context=cpu_context)],
        element_names=['rf', 'end'])
    line.particle_ref = xt.Particles(_context=cpu_context, mass0=MASS_EV,
                                    p0c=P0C_EV, q0=1.)
    line.reset_s_at_end_turn = False
    line.build_tracker(_context=cpu_context)
    rays = particle(cpu_context, x=[.0004, -.0003], y=[-.0003, .0002],
        px=[.002, -.001], py=[-.001, .0005], s=.17, time_s=1.9e-9)
    for reverse in (False, True):
        line.track(rays, ele_start=0, ele_stop=1, backtrack=reverse)
        assert np.all(rays.state == 1)
        np.testing.assert_array_equal(rays.ax, np.zeros(2))
        np.testing.assert_array_equal(rays.ay, np.zeros(2))


@pytest.mark.parametrize('profile', ('uniform', 'gaussian'))
def test_finite_rf_face_keeps_nonzero_potential_readback(cpu_context, profile):
    """Finite-profile observation planes must not acquire a false zero gauge."""
    line = xt.Line(elements=[xt.Cavity(_context=cpu_context,
        model='phase-resolved', rf_profile=profile, rf_profile_sigma=.12,
        rf_profile_center=.02, length=.56, frequency=712e6, voltage=2000.,
        phase=.4, eta=.25, counter_phase=.3, calibration_beta=.9,
        num_kicks=64, absolute_time=True), xt.Marker(_context=cpu_context)],
        element_names=['rf', 'end'])
    line.particle_ref = xt.Particles(_context=cpu_context, mass0=MASS_EV,
                                    p0c=P0C_EV, q0=1.)
    line.reset_s_at_end_turn = False
    line.build_tracker(_context=cpu_context)
    ray = particle(cpu_context, x=.0004, y=-.0003, px=.002, py=-.001,
                   s=.17, time_s=1.9e-9)
    for reverse in (False, True):
        line.track(ray, ele_start=0, ele_stop=1, backtrack=reverse)
        assert ray.state[0] == 1
        caches = np.array([ray.ax[0], ray.ay[0]])
        assert np.all(np.isfinite(caches))
        assert np.all(caches != 0.)


def test_p16_face_frequency_tpsa_has_zero_cache_coefficients(cpu_context):
    """The support face moves with k; retain that same boundary in TPSA."""
    import madng_tpsa
    import xtrack.tpsa as xtpsa

    frequency = 712e6
    line = xt.Line(elements=[xt.Cavity(_context=cpu_context,
        model='phase-resolved', rf_profile='p16', length=.56,
        frequency=frequency, voltage=2000., phase=.4, eta=.25,
        counter_phase=.3, calibration_beta=.9, num_kicks=3,
        absolute_time=True), xt.Marker(_context=cpu_context)],
        element_names=['rf', 'end'])
    line.particle_ref = xt.Particles(_context=cpu_context, mass0=MASS_EV,
                                    p0c=P0C_EV, q0=1.)
    line.reset_s_at_end_turn = False
    line.build_tracker(_context=cpu_context)
    descriptor = madng_tpsa.Descriptor(6, 2, num_params=1, param_order=2)
    line['rf'].frequency = frequency*(1.+descriptor.param(1, 0.))
    series = xtpsa.ParticlesTpsa(order=2, descriptor=descriptor,
        mass0=MASS_EV, p0c=P0C_EV, q0=1., x=.0004, y=-.0003,
        px=.002, py=-.001, zeta=.0002, delta=.001, s=.17, time_s=1.9e-9)
    # High-level TPSA backtracking is independently unsupported by Tracker;
    # the scalar cases above check both directions of the cavity word.
    line.track(series, ele_start=0, ele_stop=1)
    assert series._xobject.state == 1
    assert series._local_series['ax'].is_zero()
    assert series._local_series['ay'].is_zero()


@pytest.mark.parametrize('mode', (None, 'thick', 'thin'),
                         ids=('direct', 'thick', 'thin'))
def test_accepted_transformed_species_and_reference_rows_are_independent(
        thread_context, mode):
    """Transformed sources must preserve every lane's reference and RF clock.

    Check accepted maps: each ray has different
    reference momentum, physical mass and charge. Reference updates surround
    the transformed source, and the partial Line excludes shared turn-clock
    advancement. Thin slicing also exercises manual source-bearing drifts.
    The reverse tolerance is a selected finite-word regression, not a domain
    or physical inverse-error bound.
    """
    context = thread_context
    controls = dict(hardware('phase-resolved'), num_kicks=24,
        shift_x=.001, shift_y=-.0003, shift_s=.001,
        rot_x_rad=-.02, rot_y_rad=.03, rot_s_rad=.12)
    line = xt.Line(elements=[
        xt.ReferenceEnergyIncrease(Delta_p0c=.2*MASS_EV),
        xt.Cavity(**controls),
        xt.ReferenceEnergyIncrease(Delta_p0c=-.2*MASS_EV), xt.Marker()],
        element_names=['reference_in', 'rf', 'reference_out', 'end'])
    line.particle_ref = xt.Particles(mass0=MASS_EV, p0c=P0C_EV, q0=1.)
    line.reset_s_at_end_turn = False
    line.use_time_based_zeta_update = True
    if mode:
        # N24 aligns both thick halves and the thin 1/6, 2/3, 1/6 intervals.
        line.slice_thick_elements([xt.Strategy(xt.Teapot(2, mode=mode),
                                              element_type=xt.Cavity)])
    line.build_tracker(_context=context)
    stop = line.element_names.index('end')
    launch = dict(x=[.0002, -.0003, .0004], px=[.0001, -.0002, .0003],
        y=[-.0001, .0002, -.0003], py=[.0003, -.0001, .0002],
        p0c=P0C_EV*np.array([.93, 1.1, 1.4]), mass_ratio=[1., 1.5, 3.],
        charge_ratio=[1., -1., .5], particle_id=[7, 3, 11],
        zeta=[.0002, -.0003, .0001], delta=[.001, -.002, .003],
        spin_x=.1, spin_y=.2, spin_z=.3, s=.17, time_s=1.9e-9)
    rays = particle(context, **launch)
    before = numeric_rows_by_id(rays)
    expected = {}
    for index, identifier in enumerate(launch['particle_id']):
        one_launch = {name: (value[index]
            if isinstance(value, (list, np.ndarray)) else value)
            for name, value in launch.items()}
        one = particle(context, **one_launch)
        line.track(one, ele_start=0, ele_stop=stop)
        assert one.state[0] == 1
        expected[identifier] = numeric_rows_by_id(one)[identifier]
    line.track(rays, ele_start=0, ele_stop=stop)
    assert np.all(rays.state == 1)
    for identifier, row in numeric_rows_by_id(rays).items():
        np.testing.assert_array_equal(row, expected[identifier])
    assert rays.time_s == launch['time_s']

    line.track(rays, ele_start=0, ele_stop=stop, backtrack=True)
    assert np.all(rays.state == 1)
    coordinate_slots = [NUMERIC_FIELDS.index(name) for name in
        COORDINATES+('s', 'rpp', 'rvv', 'pzeta', 'spin_x', 'spin_y', 'spin_z')]
    reference_slots = [NUMERIC_FIELDS.index(name) for name in
        ('p0c', 'gamma0', 'beta0', 'chi', 'charge_ratio')]
    for identifier, row in numeric_rows_by_id(rays).items():
        np.testing.assert_allclose(row[coordinate_slots],
            before[identifier][coordinate_slots], rtol=0., atol=2e-13)
        np.testing.assert_allclose(row[reference_slots],
            before[identifier][reference_slots], rtol=3e-15, atol=0.)
    assert rays.time_s == launch['time_s']
