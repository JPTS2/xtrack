"""Independent normal-Line checks for the public physical cavity models.

These checks compare live mutation with fresh construction, and invariances
of the laboratory RF clock. They do not import the scientific Python maps or
infer physical/source qualification from agreement between native runs.
"""
import numpy as np
import pytest
import xobjects as xo

import xtrack as xt

from test_cavity_native import native_context


MODELS = ('phase-resolved', 'phase-resolved-paraxial',
          'phase-resolved-longitudinal', 'phase-resolved-frozen')
EVOLVING_MODELS = MODELS[:3]
COORDINATES = ('x', 'px', 'y', 'py', 'zeta', 'delta')
MASS_EV = 510998.95
P0C_EV = MASS_EV*np.sqrt(2.4**2-1.)


@pytest.fixture(scope='module')
def cpu_context(tmp_path_factory):
    """Keep JIT output context-local, using the diagnosed linker preparation."""
    with native_context(tmp_path_factory.mktemp('cavity-integration-review')) as context:
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


def taper_ring(context, model, *, strong=False, unavailable_voltage=False):
    """A known energy sink tests orchestration, not a radiation prescription.

    The strong case is an independently chosen synthetic regime with rounded
    inputs. Its reference has a material phase-slip correction relative to
    a fixed-beta crest calibration. The weak legacy case is a control.
    """
    if strong:
        controls = dict(length=1.25, voltage=14e6,
            frequency=3e9, phase=.2, eta=0., calibration_beta=1.,
            num_kicks=256, model=model)
        p0c, loss = 14e6, 1e5
    else:
        controls = dict(length=.05, voltage=1e5, frequency=1e8,
            phase=.2, eta=0., calibration_beta=1., num_kicks=32, model=model)
        p0c, loss = 1e9, 1000.
    if unavailable_voltage:
        loss = 2*controls['voltage']
    line = xt.Line(elements=[
        xt.LineSegmentMap(_context=context, length=1., qx=.23, qy=.31, betx=2., bety=3.,
            longitudinal_mode='frozen', energy_increment=-loss),
        xt.Multipole(_context=context, knl=[0.]), xt.Cavity(_context=context, **controls)],
        element_names=['sink', 'taper_slot', 'rf'])
    line.particle_ref = xt.Particles(_context=context, mass0=MASS_EV, q0=1., p0c=p0c)
    line.build_tracker(_context=context)
    return line


def taper_controls(line):
    return {name:float(getattr(line['rf'], name)) for name in
            ('voltage', 'frequency', 'harmonic', 'phase', 'lag',
             'phase_taper', 'lag_taper')}


@pytest.mark.parametrize('model', ('drift-kick-drift-expanded', *EVOLVING_MODELS))
@pytest.mark.parametrize('strong', (False, True), ids=('weak', 'synthetic-strong'))
def test_taper_restored_physical_map_closes_energy(cpu_context, model, strong):
    """Check the real restored RF map, not only its temporary DC analysis."""
    line = taper_ring(cpu_context, model, strong=strong)
    before = taper_controls(line)
    rtol = 1e-12  # The existing compensation method's numerical request.
    line.compensate_radiation_energy_loss(delta0=0, max_iter=10,
                                         rtol_eneloss=rtol, verbose=False)
    after = taper_controls(line)
    for name in ('voltage', 'frequency', 'harmonic', 'phase', 'lag'):
        assert after[name] == before[name]
    probe = line.particle_ref.copy(_context=cpu_context)
    energy0 = float(probe.energy[0])
    line.track(probe)
    assert probe.state[0] == 1
    # Ten times the requested threshold accommodates readback rounding and
    # termination on the immediately preceding pilot. This is numerical
    # solver convergence, not a proposed physical accuracy specification.
    assert abs(float(probe.energy[0])-energy0) <= 10*rtol*energy0


@pytest.mark.parametrize('model', ('drift-kick-drift-expanded', *EVOLVING_MODELS))
def test_failed_taper_restores_caller_controls(cpu_context, model):
    """A deliberately insufficient voltage must not leave f=0 or trial V."""
    line = taper_ring(cpu_context, model, unavailable_voltage=True)
    before = taper_controls(line)
    delta_taper = float(line['taper_slot'].delta_taper)
    with pytest.raises((AssertionError, ValueError, RuntimeError)):
        line.compensate_radiation_energy_loss(delta0=0, max_iter=3,
                                             verbose=False)
    assert taper_controls(line) == before
    assert line['taper_slot'].delta_taper == delta_taper


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


def effect_configuration(line):
    """Only state whose mutation would misleadingly enable an absent effect."""
    return (line.tracker, getattr(line, '_radiation_model', None),
            getattr(line, '_spin_model', None),
            getattr(line, '_needs_rng', False), dict(line.config))


@pytest.mark.parametrize('model,radiation', [
    (model,radiation) for model in MODELS
    for radiation in ('mean','quantum','quantum-kick')
    if not (model=='phase-resolved' and radiation=='mean')])
def test_unimplemented_radiation_request_is_atomic(model, radiation):
    """Refusing an unfinished effect must precede changing Line state."""
    line = xt.Line(elements={'rf':xt.Cavity(**hardware(model))})
    before = effect_configuration(line)
    with pytest.raises(NotImplementedError, match='[Cc]avity'):
        line.configure_radiation(model=radiation)
    assert effect_configuration(line) == before
    assert line.tracker is None
    assert line['rf'].radiation_flag == 0


@pytest.mark.parametrize('model', MODELS[1:])
@pytest.mark.parametrize('spin', (True, 'auto'))
def test_unimplemented_spin_request_is_atomic(model, spin):
    line = xt.Line(elements={'rf':xt.Cavity(**hardware(model))})
    before = effect_configuration(line)
    with pytest.raises(NotImplementedError, match='[Cc]avity'):
        line.configure_spin(spin)
    assert effect_configuration(line) == before
    assert line['rf'].spin_tracking == 0


@pytest.mark.parametrize('layout', ('slices', 'replica'))
def test_effect_guard_reaches_used_physical_parent(layout):
    line = xt.Line(elements={'rf':xt.Cavity(**hardware('phase-resolved-paraxial'))},
                   element_names=['rf'])
    if layout == 'slices':
        line.slice_thick_elements([xt.Strategy(
            slicing=xt.Teapot(2, mode='thick'), element_type=xt.Cavity)])
    else:
        line.env.new('alias', 'rf', mode='replica')
        line.element_names = ['alias']
    assert 'rf' not in line.element_names
    with pytest.raises(NotImplementedError, match='[Cc]avity'):
        line.configure_radiation(model='mean')
    with pytest.raises(NotImplementedError, match='[Cc]avity'):
        line.configure_spin(True)
    assert line['rf'].radiation_flag == 0
    assert line['rf'].spin_tracking == 0


def test_unused_source_does_not_block_or_inherit_other_line_effects(cpu_context):
    line = xt.Line(elements={'used':xt.Cavity(),
        'other':xt.Cavity(**hardware('phase-resolved'))},
        element_names=['used'])
    line.particle_ref = xt.Particles(mass0=MASS_EV, p0c=P0C_EV, q0=1.)
    line.build_tracker(_context=cpu_context, compile=False)
    line.configure_radiation(model='mean')
    line.configure_spin(True)
    assert line['used'].radiation_flag == 1
    assert line['used'].spin_tracking == 1
    assert line['other'].radiation_flag == 0
    assert line['other'].spin_tracking == 0


@pytest.mark.parametrize('model', MODELS)
@pytest.mark.parametrize('effect', ('radiation_flag', 'spin_tracking'))
def test_runtime_guard_catches_effect_then_model_switch(cpu_context, model, effect):
    """A later model change must not silently inherit unsupported physics."""
    line = line_for(cpu_context, dict(hardware(model), model='adaptive'))
    if effect == 'radiation_flag':
        line.configure_radiation(model='mean')
    else:
        line.configure_spin(True)
    assert getattr(line['rf'], effect) == 1
    try:
        line.set('rf', model=model)
    except (NotImplementedError, ValueError):
        # A friendly setter can refuse before changing the model. The direct
        # flag test below independently checks the native guard itself.
        assert line['rf'].model == 'adaptive'
        return
    tracked = particle(cpu_context)
    initial = values(tracked)
    line['rf'].track(tracked)
    if model=='phase-resolved':
        # This exact RF-only P16 case now has scalar mean/spin transport.
        assert tracked.state[0]==1
        return
    assert tracked.state[0] < 0
    np.testing.assert_array_equal(
        np.array([getattr(tracked, name)[0] for name in COORDINATES]), initial)


@pytest.mark.parametrize('model', MODELS)
@pytest.mark.parametrize('effect', ('radiation_flag', 'spin_tracking'))
def test_native_effect_flag_refuses_without_partial_map(cpu_context, model, effect):
    controls = hardware(model)
    element = xt.Cavity(_context=cpu_context, **controls)
    baseline = particle(cpu_context)
    element.track(baseline)
    assert baseline.state[0] == 1
    setattr(element, effect, 1)
    tracked = particle(cpu_context)
    initial = values(tracked)
    element.track(tracked)
    if model=='phase-resolved':
        assert tracked.state[0]==1
        return
    assert tracked.state[0] < 0
    np.testing.assert_array_equal(
        np.array([getattr(tracked, name)[0] for name in COORDINATES]), initial)


@pytest.mark.parametrize('model', ('drift-kick-drift-expanded', 'phase-resolved'))
def test_default_context_build_uses_local_cavity_abi(tmp_path, monkeypatch, model):
    """A same-version installed binary must not consume changed Cavity data.

    Use a fresh default CPU context, without the test fixture's JIT or linker
    overrides. The checkout must arrange its own class-scoped compilation.
    """
    xsuite = pytest.importorskip('xsuite')
    monkeypatch.chdir(tmp_path)
    original_lookup = xsuite.get_suitable_kernel

    def checked_lookup(*args, **kwargs):
        requested = list(kwargs.get('tracker_element_classes', ()))
        requested += list(kwargs.get('classes', ()))
        names = [getattr(cls, '_DressingClass', cls).__name__
                 for cls in requested]
        assert not any('Cavity' in name for name in names), names
        return original_lookup(*args, **kwargs)

    monkeypatch.setattr(xsuite, 'get_suitable_kernel', checked_lookup)
    context = xo.ContextCpu(omp_num_threads=0)
    before = (context.allow_prebuilt_kernels, context.allow_kernel_compilation)
    controls = dict(hardware('phase-resolved'), model=model)
    line = line_for(context, controls)
    actual = endpoint(line, context)

    # A complete Line call advances its longitudinal reference origin at the
    # turn boundary. Compare each public route with the same route, retaining
    # its clock convention, rather than equating raw Line/standalone zeta.
    jit_context = xo.ContextCpu(omp_num_threads=0)
    jit_context.allow_prebuilt_kernels = False
    jit_context.allow_kernel_compilation = True
    expected_line = line_for(jit_context, controls)
    np.testing.assert_allclose(actual, endpoint(expected_line, jit_context),
                               rtol=0., atol=2e-13)
    direct = particle(context)
    xt.Cavity(_context=context, **controls).track(direct)
    expected_direct = particle(jit_context)
    xt.Cavity(_context=jit_context, **controls).track(expected_direct)
    np.testing.assert_allclose(values(direct), values(expected_direct),
                               rtol=0., atol=2e-13)
    assert (context.allow_prebuilt_kernels, context.allow_kernel_compilation) == before


def test_unused_cavity_does_not_disable_prebuilt_lookup(monkeypatch):
    """A shared environment's unused Cavity must not change a Quad-only line."""
    xsuite = pytest.importorskip('xsuite')
    context = xo.ContextCpu(omp_num_threads=0)
    context.allow_prebuilt_kernels = True
    line = xt.Line(elements={'q':xt.Quadrupole(length=.02, k1=.2),
        'unused':xt.Cavity(**hardware('phase-resolved'))}, element_names=['q'])
    line.build_tracker(_context=context, compile=False)
    looked_up = []

    def lookup(**kwargs):
        looked_up.append([cls._DressingClass.__name__
                          for cls in kwargs['tracker_element_classes']])
        return None

    sentinel = object()
    monkeypatch.setattr(xsuite, 'get_suitable_kernel', lookup)
    monkeypatch.setattr(xt.Tracker, '_compile_kernel_from_classes',
        classmethod(lambda cls, **kwargs: {'kernel':sentinel}))
    assert line.tracker._build_kernel(compile=True) is sentinel
    assert len(looked_up) == 1
    assert 'Quadrupole' in looked_up[0]
    assert not any('Cavity' in name for name in looked_up[0])


def test_local_linker_repair_is_reversible_and_class_scoped(monkeypatch):
    """Only identical conda rpaths are removed, including on compile failure."""
    from pathlib import Path
    import shlex
    import sys
    import sysconfig
    from xtrack._local_kernel import local_kernel_compilation

    config = sysconfig.get_config_vars()
    token = '-Wl,-rpath,'+str(Path(sys.prefix)/'lib')
    original = shlex.join(('cc', token, '-Wl,-rpath,/other/lib', token, '-shared'))
    keys = ('LDSHARED', 'LDCXXSHARED', 'BLDSHARED', 'LDFLAGS')
    context = xo.ContextCpu(omp_num_threads=0)
    previous_prebuilt = context.allow_prebuilt_kernels
    for key in keys:
        monkeypatch.setitem(config, key, original)
    with local_kernel_compilation([xt.Quadrupole], context):
        assert all(config[key] == original for key in keys)
        assert context.allow_prebuilt_kernels == previous_prebuilt
    with pytest.raises(RuntimeError, match='simulated compiler failure'):
        with local_kernel_compilation([xt.Cavity._XoStruct], context):
            assert context.allow_prebuilt_kernels is False
            for key in keys:
                tokens = shlex.split(config[key])
                assert tokens.count(token) == 1
                assert '-Wl,-rpath,/other/lib' in tokens
                assert tokens[-1] == '-shared'
            raise RuntimeError('simulated compiler failure')
    assert all(config[key] == original for key in keys)
    assert context.allow_prebuilt_kernels == previous_prebuilt


@pytest.mark.parametrize('field', ('eta', 'calibration_beta', 'rf_field_order',
    'edge_entry_active', 'edge_exit_active', 'num_factor_steps', 'order'))
def test_zero_controls_survive_json_roundtrip(field):
    """Zero hardware/face settings must not turn into constructor defaults.

    Some zero controls are deliberately inadmissible for tracking. A roundtrip
    must preserve that rejection rather than silently make them admissible.
    """
    controls = dict(hardware('phase-resolved-frozen'), **{field:0})
    original = xt.Cavity(**controls)
    restored = xt.Cavity.from_dict(original.to_dict())
    assert getattr(restored, field) == 0
    assert restored.model == original.model


def test_taper_refuses_active_generic_slice_parent_before_attr_cache():
    """A sliced Quad's parameter storage is owned by its physical parent."""
    madng_tpsa = pytest.importorskip('madng_tpsa')
    context = xo.ContextCpu(omp_num_threads=0)
    line = xt.Line(elements={
        'q':xt.Quadrupole(length=.1, k1=.2),
        'rf':xt.Cavity(length=.05, frequency=400e6, voltage=2e4,
                      model='drift-kick-drift-expanded')},
        element_names=['q', 'rf'])
    line.particle_ref = xt.Particles(p0c=1e9)
    line.slice_thick_elements([xt.Strategy(None), xt.Strategy(
        slicing=xt.Teapot(2, mode='thick'), element_type=xt.Quadrupole)])
    line.build_tracker(_context=context, compile=False)
    descriptor = madng_tpsa.Descriptor(6, 1, num_params=1, param_order=1)
    parent = line.element_dict['q']
    parent.k1 = descriptor.param(1, .2)
    assert 'q' not in line.element_names
    assert 'attr' not in line.tracker._tracker_data_base.cache
    before = taper_controls(line)
    try:
        with pytest.raises(NotImplementedError, match='scalar element storage'):
            line.compensate_radiation_energy_loss(verbose=False)
        assert parent.k1.const_part == .2
        assert parent.k1.param_grad() == [1.]
        assert taper_controls(line) == before
        assert 'attr' not in line.tracker._tracker_data_base.cache
    finally:
        parent.disable_tpsa()


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
    policy = (context.allow_prebuilt_kernels, context.allow_kernel_compilation)
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
    assert (context.allow_prebuilt_kernels, context.allow_kernel_compilation) == policy


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


def test_local_abi_respects_existing_cffi_compilation_hard_stop(tmp_path, monkeypatch):
    """An ABI fallback must respect Xobjects' actual compilation prohibition."""
    import sysconfig
    monkeypatch.chdir(tmp_path)
    context = xo.ContextCpu(omp_num_threads=0)
    policy = (context.allow_prebuilt_kernels, context.allow_kernel_compilation)
    line = reference_only_line(context, True)
    tracked = particle(context, **CLOCK_LAUNCH)
    keys = ('LDSHARED', 'LDCXXSHARED', 'BLDSHARED', 'LDFLAGS')
    before = {key:sysconfig.get_config_vars().get(key) for key in keys}

    def forbidden_compiler(*args, **kwargs):
        pytest.fail('the compiler ran despite the CFFI hard stop')

    monkeypatch.setattr(context, 'compile_kernel', forbidden_compiler)
    monkeypatch.setattr(xo.settings, 'cffi_forbid_compile', True)
    with pytest.raises(RuntimeError, match='CFFI compilation is forbidden'):
        line.track(tracked)
    assert (context.allow_prebuilt_kernels, context.allow_kernel_compilation) == policy
    assert {key:sysconfig.get_config_vars().get(key) for key in keys} == before


@pytest.fixture(scope='module')
def compact_taper_line(cpu_context):
    """One compiled Line exercises live family selection and RF restoration."""
    return taper_ring(cpu_context, 'phase-resolved-frozen')


@pytest.mark.parametrize('model', (
    'phase-resolved-frozen',
    'phase-averaged-second-order',
    'phase-averaged-cubic',
    'phase-averaged-quartic',
    'phase-averaged-full-slow',
    'phase-averaged-uniform-quadratic-first-boundary',
    'phase-averaged-uniform-quadratic',
    'phase-averaged-magnetic'))
def test_compact_taper_restores_actual_rf_and_saved_profile(compact_taper_line, model):
    """The temporary uniform-DC predictor must close on each actual RF map."""
    line = compact_taper_line
    cavity = line['rf']
    cavity.model = model
    cavity.rf_profile = ('gaussian' if model == 'phase-averaged-magnetic'
        else 'p16' if model in ('phase-averaged-second-order',
            'phase-averaged-cubic', 'phase-averaged-quartic',
            'phase-averaged-full-slow') else 'uniform')
    cavity.rf_profile_sigma = .0125
    cavity.rf_profile_center = .003
    cavity.num_factor_steps = 4
    cavity.phase = .2
    cavity.phase_taper = cavity.lag_taper = 0.
    cavity.delta_taper = line['taper_slot'].delta_taper = 0.
    before = taper_controls(line)
    line.compensate_radiation_energy_loss(delta0=0.,
        rtol_eneloss=1e-12, verbose=False)
    after = taper_controls(line)
    for name in ('voltage', 'frequency', 'harmonic', 'phase', 'lag'):
        assert after[name] == before[name]
    probe = line.particle_ref.copy()
    energy0 = float(probe.energy[0])
    line.track(probe, turn_by_turn_monitor='ONE_TURN_EBE')
    assert probe.state[0] == 1
    monitor = line.record_last_track
    means = .5*(monitor.delta[0, :-1]+monitor.delta[0, 1:])
    item = line.attr._cache['delta_taper']
    tolerance = 1e-12*energy0/float(probe.beta0[0]*probe.p0c[0])
    np.testing.assert_allclose(item.multisetter.get_values(), means[item.mask],
                               rtol=0., atol=tolerance)
    assert abs(float(probe.energy[0])-energy0) <= 1e-12*energy0


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


TRANSACTION_NUMERIC_FIELDS = COORDINATES + (
    'pzeta', 'rvv', 'rpp', 's', 'ax', 'ay', 'spin_x', 'spin_y', 'spin_z',
    'p0c', 'gamma0', 'beta0', 'chi', 'charge_ratio', 'weight',
    'anomalous_magnetic_moment')


def numeric_rows_by_id(particles):
    """A failed ordinary kernel can compact losses; identity owns each row."""
    return {int(identifier): np.array([getattr(particles, name)[index]
            for name in TRANSACTION_NUMERIC_FIELDS])
            for index, identifier in enumerate(particles.particle_id)}


@pytest.fixture(scope='module', params=(0, 1), ids=('serial', 'openmp_one_thread'))
def transaction_context(request, tmp_path_factory):
    with native_context(tmp_path_factory.mktemp('cavity-frame-transaction')) as serial:
        yield serial if request.param == 0 else xo.ContextCpu(omp_num_threads=1)


@pytest.mark.parametrize('invalid', (
    dict(eta=-.1), dict(calibration_beta=0.),
    dict(rf_profile='uniform', wave_number=0., eta=1., counter_phase=np.pi)))
def test_transformed_source_refusal_restores_each_numeric_lane(transaction_context, invalid):
    """Invalid hardware/calibration must leave each laboratory row intact."""
    context = transaction_context
    controls = dict(hardware('phase-resolved'), shift_x=.001, rot_y_rad=.03)
    controls.update(invalid)
    cavity = xt.Cavity(_context=context, **controls)
    rays = particle(context, x=[.0002, -.0003, .0004], px=[.0001, -.0002, .0003],
        particle_id=[7, 3, 11], spin_x=.1, spin_y=.2, spin_z=.3,
        ax=2e-5, ay=-3e-5, s=.17)
    before = numeric_rows_by_id(rays)
    cavity.track(rays)
    assert np.all(rays.state <= 0)
    for identifier, row in numeric_rows_by_id(rays).items():
        np.testing.assert_array_equal(row.view(np.uint64), before[identifier].view(np.uint64))


def test_transformed_mixed_domain_failure_restores_only_failed_lanes(transaction_context):
    """One inadmissible off-axis map must not disturb a different valid ray."""
    context = transaction_context
    controls = dict(hardware('phase-resolved'), shift_x=.001, rot_y_rad=.03,
                    num_kicks=8)
    cavity = xt.Cavity(_context=context, **controls)
    rays = particle(context, x=[.0002, 5., -.0003, 1.],
        px=[.0001, .0001, -.0002, .0001], particle_id=[7, 3, 11, 5],
        spin_x=.1, spin_y=.2, spin_z=.3, ax=2e-5, ay=-3e-5)
    before = numeric_rows_by_id(rays)
    expected = {}
    for identifier, x, px in ((7, .0002, .0001), (11, -.0003, -.0002)):
        one = particle(context, x=x, px=px, particle_id=identifier,
            spin_x=.1, spin_y=.2, spin_z=.3, ax=2e-5, ay=-3e-5)
        cavity.track(one)
        assert one.state[0] == 1
        expected[identifier] = numeric_rows_by_id(one)[identifier]
    cavity.track(rays)
    rows = numeric_rows_by_id(rays)
    states = dict(zip(map(int, rays.particle_id), map(int, rays.state)))
    for identifier in (3, 5):
        assert states[identifier] <= 0
        np.testing.assert_array_equal(rows[identifier].view(np.uint64),
                                      before[identifier].view(np.uint64))
    for identifier in (7, 11):
        assert states[identifier] == 1
        np.testing.assert_allclose(rows[identifier], expected[identifier], rtol=0., atol=2e-13)


@pytest.mark.parametrize('slice_class', (xt.DriftSliceCavity, xt.ThickSliceCavity,
                                       xt.ThinSliceCavity))
def test_transformed_slice_refusal_restores_numeric_lanes(transaction_context, slice_class):
    """Cover both manual parent frames and the generated slice wrappers."""
    context = transaction_context
    parent = xt.Cavity(_context=context, **dict(hardware('phase-resolved'),
        shift_x=.001, rot_y_rad=.03, eta=-.1))
    if slice_class is xt.ThinSliceCavity:
        # New-model thin nodes have no source action; an invalid model selector
        # exercises their own actual refusal rather than an unused RF setting.
        parent._model = 25
    interval = slice_class(_parent=parent, _buffer=parent._buffer, weight=.5,
        **({} if slice_class is xt.ThinSliceCavity else dict(slice_offset=0.)))
    rays = particle(context, x=[.0002, -.0003], particle_id=[7, 3], s=.17,
        spin_x=.1, spin_y=.2, spin_z=.3, ax=2e-5, ay=-3e-5)
    before = numeric_rows_by_id(rays)
    interval.track(rays)
    assert np.all(rays.state <= 0)
    for identifier, row in numeric_rows_by_id(rays).items():
        np.testing.assert_array_equal(row.view(np.uint64), before[identifier].view(np.uint64))


def test_partial_line_transformed_refusal_keeps_numeric_launch(transaction_context):
    """Exercise the actual Line route, ending before any turn-clock action."""
    context = transaction_context
    cavity = xt.Cavity(_context=context, **dict(hardware('phase-resolved'),
        shift_x=.001, rot_y_rad=.03))
    line = xt.Line(elements=[cavity, xt.Marker(_context=context)],
                   element_names=['rf', 'end'])
    line.particle_ref = xt.Particles(_context=context, mass0=MASS_EV,
                                    q0=1., p0c=P0C_EV)
    line.reset_s_at_end_turn = False
    line.build_tracker(_context=context)
    line.set('rf', eta=-.1)
    rays = particle(context, x=[.0002, -.0003], particle_id=[7, 3], s=.17,
        spin_x=.1, spin_y=.2, spin_z=.3, ax=2e-5, ay=-3e-5)
    before = numeric_rows_by_id(rays)
    line.track(rays, ele_start=0, ele_stop=1)
    assert np.all(rays.state <= 0)
    for identifier, row in numeric_rows_by_id(rays).items():
        np.testing.assert_array_equal(row.view(np.uint64), before[identifier].view(np.uint64))


@pytest.mark.parametrize('failure', ('source_refusal', 'descriptor_exception'))
def test_transformed_tpsa_failure_restores_all_coefficients(cpu_context, failure):
    """Rollback includes series coefficients, even when field access throws."""
    import madng_tpsa
    import xtrack.tpsa as xtpsa

    descriptor = madng_tpsa.Descriptor(6, 2, num_params=1, param_order=1)
    controls = dict(hardware('phase-resolved'), shift_x=.001, rot_y_rad=.03)
    if failure == 'source_refusal':
        controls['eta'] = -.1
    cavity = xt.Cavity(_context=cpu_context, **controls)
    if failure == 'descriptor_exception':
        foreign = madng_tpsa.Descriptor(6, 2, num_params=2, param_order=1)
        cavity.phase = foreign.param(1, controls['phase'])
    series = xtpsa.ParticlesTpsa(order=2, descriptor=descriptor,
        mass0=MASS_EV, p0c=P0C_EV, q0=1., x=.0002, px=.0001,
        y=-.0003, py=.0002, zeta=.0004, delta=.002, s=.17,
        ax=2e-5, ay=-3e-5, spin_x=.1, spin_y=.2, spin_z=.3)

    def coefficients():
        from itertools import combinations_with_replacement
        items = dict(zip(COORDINATES, series.coords))
        items.update(series._local_series)
        result = {}
        for name, value in items.items():
            coefficients = []
            dimension = descriptor.monomial_length
            for degree in range(value.order+1):
                for indices in combinations_with_replacement(range(dimension), degree):
                    powers = [indices.count(axis) for axis in range(dimension)]
                    if descriptor.is_valid_monomial(powers):
                        coefficients.append(value.get(powers))
            result[name] = np.asarray(coefficients)
        return result

    before = coefficients()
    references = {name: getattr(series, name) for name in (
        'p0c', 'gamma0', 'beta0', 'chi', 'charge_ratio', 'weight',
        'anomalous_magnetic_moment')}
    with pytest.raises(RuntimeError, match='TPSA map lost'):
        cavity.track(series)
    assert series._xobject.state <= 0
    for name, values in coefficients().items():
        np.testing.assert_array_equal(values.view(np.uint64), before[name].view(np.uint64))
    assert {name: getattr(series, name) for name in references} == references
