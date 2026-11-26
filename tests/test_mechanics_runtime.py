"""Portable prepared-kernel checks against independent/reference transforms."""
import numpy as np
import pytest
import xobjects as xo
from xobjects.test_helpers import for_all_test_contexts

import xtrack as xt
from xtrack.mechanics import Harmonic, MeasuredMotion, PlaneWave, RigidMotion


FIELDS = ('shift_x', 'shift_y', 'shift_s', 'rot_x_rad', 'rot_y_rad',
          'rot_s_rad_no_frame', 'rot_shift_anchor')


@pytest.mark.parametrize('backend', ['compiled', 'reference'])
def test_prepared_held_measurements_and_continuous_sources_keep_their_laws(backend):
    env = xt.Environment()
    env.new('q', xt.Quadrupole, length=.5, k1=.1)
    line = env.new_line(components=['q'])
    line.set_particle_ref('proton', p0c=7e12)
    line.mechanics.new_girder('g', elements=['q'])
    env.mechanics.add_source('held', MeasuredMotion([0., .4, 1.], [0., 2e-6, 2e-6],
                                                   interpolation='previous'), targets='g')
    env.mechanics.add_source('continuous', Harmonic(1e-6, 1.), targets='g', component='x')
    record = env.mechanics.compute_time_series(time_s=[0., 1.])
    assert dict(record.replay_modes) == {'held': 'source', 'continuous': 'source'}
    line.mechanics.enable_vibrations(backend=backend, time_series=record)
    particles = line.build_particles()
    particles.time_s = .25
    line.track(particles)
    assert line['q'].shift_y == 0.
    assert line['q'].shift_x == pytest.approx(1e-6)
    particles.time_s = .45
    line.track(particles)
    assert line['q'].shift_y == pytest.approx(2e-6)
    line.mechanics.freeze_updates()
    sampled = env.mechanics.compute_time_series(time_s=[0., 1.], interpolation='linear')
    line.mechanics.apply(time_s=.25, time_series=sampled)
    assert line['q'].shift_y == pytest.approx(.5e-6)


@pytest.mark.parametrize('backend', ['compiled', 'reference'])
def test_fft_psd_replay_at_off_grid_phase(backend):
    env = xt.Environment()
    env.new('q', xt.Quadrupole, length=.5, k1=.1)
    line = env.new_line(components=['q'])
    line.set_particle_ref('proton', p0c=7e12)
    line.mechanics.new_girder('g', elements=['q'])
    source = xt.mechanics.DisplacementPSD([0., 3., 6., 7.], [0., 1e-12, 1e-12, 0.], 956)
    env.mechanics.add_source('floor', source, targets='g')
    times = np.arange(64) / 32
    record = env.mechanics.compute_time_series(time_s=times)
    shifted = times + 1 / 64
    expected = source.evaluate(shifted, np.zeros((1, 3)))[:, 0]
    assert abs(expected[0] - (record.values['floor'][0, 0, 1]
                              + record.values['floor'][1, 0, 1]) / 2) > 1e-9
    line.mechanics.enable_vibrations(backend=backend, time_series=record)
    particles = line.build_particles()
    for index in (0, 7, 32):
        particles.time_s = shifted[index]
        line.track(particles)
        assert line['q'].shift_y == pytest.approx(expected[index], abs=2e-20)


def test_compound_fft_replay_keeps_regional_mix_and_local_delay():
    env = xt.Environment()
    for index, point in enumerate(((0., 0., 0.), (3., 0., 0.))):
        env.mechanics.new_girder(f'g{index}', anchor=point)
    points = np.array([[0., 0., 0.], [3., 0., 0.]])
    psd_args = ([0., 3., 6., 7.], [0., 1e-12, 1e-12, 0.])
    regional = xt.mechanics.RegionalGroundMotion(
        xt.mechanics.DisplacementPSD(*psd_args, seed=4),
        (xt.mechanics.GaussianRegion(
            xt.mechanics.DisplacementPSD(*psd_args, seed=5), (0., 0., 0.), 2.,
        ),),
    )
    field = xt.mechanics.VectorGroundMotion((regional, None, None))
    pump = xt.mechanics.LocatedSource(
        xt.mechanics.DisplacementPSD(*psd_args, seed=6), (0., 0., 0.), 4., 60.,
    )
    env.mechanics.add_source('floor', field, targets=('g0', 'g1'))
    env.mechanics.add_source('pump', pump, targets=('g0', 'g1'))
    times = np.arange(64) / 32
    record = env.mechanics.compute_time_series(time_s=times)
    shifted = times + 1 / 64
    expected = field.evaluate(shifted, points) + pump.evaluate(shifted, points)
    for index in (0, 7, 32):
        poses = env.mechanics.get_body_poses(time_s=shifted[index], time_series=record)
        actual = np.array([poses[name].XYZ for name in ('g0', 'g1')]) - points
        np.testing.assert_allclose(actual, expected[index], atol=3e-16, rtol=0.)


@pytest.mark.parametrize('budget', [0, 32 * 1024**2])
def test_spectral_field_exact_replay_with_or_without_receiver_cache(monkeypatch, budget):
    from xtrack.mechanics import runtime

    monkeypatch.setattr(runtime, '_SPECTRAL_RECEIVER_BUDGET', budget)
    env = xt.Environment()
    env.new('q', xt.Quadrupole, length=.5, k1=.1)
    line = env.new_line(components=['q'])
    line.set_particle_ref('proton', p0c=7e12)
    line.mechanics.new_girder('g', elements=['q'])
    source = xt.mechanics.SpectralGroundMotion(
        [0., 3., 6., 7.], [0., 1e-12, 1e-12, 0.],
        seed=956, correlation_length_m=30., n_modes=16,
    )
    env.mechanics.add_source('floor', source, targets='g')
    record = env.mechanics.compute_time_series(time_s=[0., 1.])
    line.mechanics.enable_vibrations(backend='compiled', time_series=record)
    particles = line.build_particles()
    particles.time_s = .37
    line.track(particles)
    expected = env.mechanics.get_body_poses(time_s=.37)['g'].Y
    assert line['q'].shift_y == pytest.approx(expected, abs=1e-20)
    assert bool(line.mechanics._runtime._spectral_receiver_bytes) == bool(budget)


@pytest.mark.parametrize('backend, sampling', [
    ('compiled', 'turn'), ('reference', 'turn'), ('reference', 'passage'),
])
@pytest.mark.parametrize('use_record', [False, True])
def test_finite_replay_preflights_complete_call_before_advancing(backend, sampling, use_record):
    from scipy.constants import c

    env = xt.Environment()
    env.new('q', xt.Quadrupole, length=.5, k1=.1)
    env.new('d', xt.Drift, length=1.5)
    line = env.new_line(components=['d', 'q'])
    line.set_particle_ref('proton', p0c=7e12)
    line.mechanics.new_girder('g', elements=['q'])
    period = line.get_length() / (float(line.particle_ref.beta0[0]) * c)
    delay = .75 * period if sampling == 'passage' else 0.
    duration = period + delay
    env.mechanics.add_source('motion', MeasuredMotion([0., duration], [1e-6, 2e-6]), targets='g')
    record = (env.mechanics.compute_time_series(time_s=[0., duration]) if use_record else None)
    line.mechanics.enable_vibrations(backend=backend, sampling=sampling, time_series=record)
    particles = line.build_particles()
    with pytest.raises(ValueError, match='motion: requested mechanics freeze times'):
        line.track(particles, num_turns=5, with_progress=1)
    assert particles.at_turn[0] == 0
    assert particles.time_s == 0.
    assert line['q'].shift_y == 0.
    assert line.mechanics.last_applied_time_s is None
    line.track(particles, num_turns=2)
    assert particles.at_turn[0] == 2
    assert line['q'].shift_y == pytest.approx(2e-6)


def test_delayed_record_preflight_includes_propagation_time():
    env = xt.Environment()
    env.new('q', xt.Quadrupole, length=.5, k1=.1)
    line = env.new_line(components=['q'])
    line.set_particle_ref('proton', p0c=7e12)
    line.mechanics.new_girder('g', elements=['q'])
    waveform = MeasuredMotion([0., .1], [1e-6, 2e-6])
    pump = xt.mechanics.LocatedSource(waveform, (0., 0., 0.), 2., 100.)
    env.mechanics.add_source('pump', pump, targets='g')
    line.mechanics.enable_vibrations(backend='compiled')
    particles = line.build_particles()
    with pytest.raises(ValueError, match='pump: requested mechanics freeze times'):
        line.track(particles)
    assert particles.at_turn[0] == 0
    particles.time_s = .0025
    line.track(particles)
    assert particles.at_turn[0] == 1


def make_case(context, translated_parent=False):
    env = xt.Environment()
    env.new('q', xt.Quadrupole, length=.5, k1=.1, shift_x=.003,
            rot_x_rad=.02, rot_s_rad=.03, rot_shift_anchor=.2)
    env.new('b', xt.Bend, length=1.5, angle=.1, rot_s_rad=.07,
            shift_y=.001, rot_y_rad=.01, rot_shift_anchor=.8)
    line = env.new_line(components=['q', 'b'])
    line.set_particle_ref('proton', p0c=7e12)
    line.build_tracker(_context=context)
    line.mechanics.set_reference_frame(xt.Frame.from_survey_angles(X=20000., Z=-30000., theta=.3))
    floor = env.mechanics.new_girder('floor', anchor=xt.Frame.from_survey_angles(
        X=20000., Y=-1., Z=-30000., theta=.2, psi=.1))
    girder = line.mechanics.new_girder('girder', elements=['q', 'b'], parent=floor)
    floor.shift_y = 3e-6
    if translated_parent:
        floor.installation.shift_x = .04
    floor.installation.rot_s_rad = .002
    girder.installation.rot_x_rad = .003
    env.mechanics.add_source('floor', Harmonic(1e-7, 4.), targets='floor', component='x')
    env.mechanics.add_source('motion', RigidMotion([0., 1.],
        [[1e-9, 2e-9, 3e-9, 1e-8, 2e-8, 3e-8],
         [2e-9, 3e-9, 4e-9, 2e-8, 3e-8, 4e-8]], target_names=('girder',)),
        targets='girder')
    return env, line


@for_all_test_contexts
@pytest.mark.parametrize('translated_parent', [False, True])
def test_prepared_matches_finite_hierarchy(test_context, translated_parent):
    from xtrack.mechanics.runtime import PreparedMechanics
    env, line = make_case(test_context, translated_parent)
    expected = []
    for t in (0., .37, .91):
        env.mechanics.apply(time_s=t)
        expected.append(np.array([[getattr(line[n], f) for f in FIELDS] for n in ('q', 'b')]))
    runtime = PreparedMechanics(line.mechanics)
    for t, values in zip((0., .37, .91), expected):
        runtime.apply(t)
        runtime.finish()
        actual = [[getattr(line[n], f) for f in FIELDS] for n in ('q', 'b')]
        np.testing.assert_allclose(actual, values, atol=3e-17, rtol=2e-14)
    line.mechanics.validate()


@for_all_test_contexts
def test_tracking_and_split_calls_keep_same_clock(test_context):
    _, line = make_case(test_context)
    line.mechanics.enable_vibrations()
    first = line.build_particles(x=1e-7, y=2e-8)
    line.track(first, num_turns=5)
    second = line.build_particles(x=1e-7, y=2e-8)
    line.track(second, num_turns=2)
    line.track(second, num_turns=3)
    for name in ('x', 'px', 'y', 'py', 'zeta', 'delta'):
        np.testing.assert_array_equal(test_context.nparray_from_context_array(getattr(first, name)),
                                      test_context.nparray_from_context_array(getattr(second, name)))
    line.mechanics.disable_vibrations()
    assert not line.mechanics.vibrations_enabled
    # Installation remains, while deliberate element values are recovered on clear.
    line.mechanics.clear()
    assert line['q'].shift_x == .003
    assert line['q'].rot_shift_anchor == .2


@for_all_test_contexts
def test_prepared_tiny_motion_and_exact_zero(test_context):
    env = xt.Environment()
    env.new('q', xt.Quadrupole, length=.5, k1=.1, rot_shift_anchor=.2)
    line = env.new_line(components=['q'])
    line.build_tracker(_context=test_context)
    line.mechanics.set_reference_frame(xt.Frame.from_survey_angles(X=50000., Z=-30000.))
    line.mechanics.new_girder('g', elements=['q'])
    env.mechanics.add_source('h', Harmonic(1e-13, 0., np.pi/2), targets='g', component='x')
    from xtrack.mechanics.runtime import PreparedMechanics
    runtime = PreparedMechanics(line.mechanics)
    runtime.apply(0.)
    runtime.finish()
    assert line['q'].shift_x == pytest.approx(1e-13, rel=1e-15, abs=0.)
    env.mechanics.restore()
    assert line['q'].rot_shift_anchor == .2


@for_all_test_contexts
def test_prepared_record_and_world_source_parity(test_context):
    env = xt.Environment()
    env.new('q', xt.Quadrupole, length=.5, k1=.1)
    line = env.new_line(components=['q'])
    line.build_tracker(_context=test_context)
    line.mechanics.new_girder('g', elements=['q'], anchor=xt.Frame.from_survey_angles(psi=.3))
    env.mechanics.add_source('wave', PlaneWave(1e-9, 3., (1., 0., 0.), 100.), targets='g')
    env.mechanics.add_source('measured', MeasuredMotion([0., .5, 1.], [0., 1e-9, 0.]), targets='g', component='s')
    series = env.mechanics.compute_time_series(time_s=np.array([0., .5, 1.]))
    from xtrack.mechanics.runtime import PreparedMechanics
    runtime = PreparedMechanics(line.mechanics)
    for record in (None, series):
        env.mechanics.apply(time_s=.27, time_series=record)
        expected = np.array([getattr(line['q'], f) for f in FIELDS])
        runtime.apply(.27, record)
        runtime.finish()
        np.testing.assert_allclose([getattr(line['q'], f) for f in FIELDS], expected,
                                   atol=1e-24, rtol=1e-14)


@pytest.mark.parametrize('mixed', [False, True])
@for_all_test_contexts
def test_prepared_plane_wave_retains_phase_after_long_propagation(test_context, mixed):
    """Subtract propagation delay before converting the local time to phase."""
    from xtrack.mechanics.runtime import PreparedMechanics

    line = xt.Line(elements={'q': xt.Quadrupole(length=1.)})
    line.build_tracker(_context=test_context)
    line.mechanics.new_girder('g', elements=['q'], anchor=[0., 0., 0.])
    wave = PlaneWave(2e-9, 1000., (1., 0., 0.), .001,
                     origin_m=(-1e9, 0., 0.), phase_rad=.4)
    line.env.mechanics.add_source('wave', wave, targets='g')
    arrival = 1e12
    if mixed:
        line.env.mechanics.add_source(
            'sample', MeasuredMotion([arrival-1., arrival+1.], [0., 0.]), targets='g',
        )
    line.mechanics.apply(time_s=arrival)
    expected = line['q'].shift_y
    assert expected == pytest.approx(2e-9*np.sin(.4), rel=1e-14, abs=0.)

    runtime = PreparedMechanics(line.mechanics)
    assert runtime.analytic is (not mixed)
    runtime.apply(arrival)
    runtime.finish()
    assert line['q'].shift_y == pytest.approx(expected, rel=1e-14, abs=0.)


def test_failed_record_keeps_conflict_guards_and_existing_fields():
    import xobjects as xo
    _, line = make_case(xo.ContextCpu())
    line.mechanics.enable_vibrations()
    particles = line.build_particles()
    with pytest.raises(ValueError, match='outside'):
        line.mechanics._apply_for_tracking(2.)
    # Outside tracking, full validation still detects edits.
    line['q'].shift_x += .001
    with pytest.raises(ValueError, match='edited'):
        line.track(particles, num_turns=1)


def test_builtin_subclass_keeps_custom_evaluate_and_rejects_complex():
    import xobjects as xo
    from xtrack.mechanics.runtime import PreparedMechanics
    class CustomHarmonic(Harmonic):
        def evaluate(self, time_s, positions_m):
            return np.full((len(time_s), len(positions_m)), 7e-9)
    env = xt.Environment()
    env.new('q', xt.Quadrupole, length=.5, k1=.1)
    line = env.new_line(components=['q'])
    line.build_tracker(_context=xo.ContextCpu())
    line.mechanics.new_girder('g', elements=['q'])
    env.mechanics.add_source('custom', CustomHarmonic(1e-9, 3.), targets='g')
    runtime = PreparedMechanics(line.mechanics)
    assert not runtime.analytic
    runtime.apply(.1)
    runtime.finish()
    assert line['q'].shift_y == 7e-9
    class ComplexSource:
        frame = 'body_local'
        def evaluate(self, time_s, positions_m):
            return np.full((len(time_s), len(positions_m)), 1e-9+1e-9j)
    env.mechanics.add_source('bad', ComplexSource(), targets='g')
    runtime = PreparedMechanics(line.mechanics)
    with pytest.raises(ValueError, match='real'):
        runtime.apply(.1)
    assert line['q'].shift_y == 7e-9


def test_cached_runtime_refreshes_installation_and_allows_explicit_inspection():
    import xobjects as xo
    _, line = make_case(xo.ContextCpu())
    view = line.mechanics
    view.enable_vibrations(backend='compiled')
    seen = []
    def inspect(line, particles):
        seen.append(line.mechanics.get_table())
        return float(line['q'].shift_y)
    line.track(line.build_particles(), num_turns=2, log=xt.Log(inspect=inspect))
    runtime = view._runtime
    assert len(seen) == 2
    view.manager['girder'].installation.shift_y += 1e-6
    line.track(line.build_particles(), num_turns=1)
    assert view._runtime is runtime
    expected = view._updates(view.manager.get_body_poses(time_s=0.), check=False)
    for attachment, element, values in expected:
        np.testing.assert_allclose([getattr(element, f) for f in FIELDS], values,
                                   atol=3e-17, rtol=2e-14)


@for_all_test_contexts
def test_fixed_reference_clock_and_monitor_match_reference_scheduler(test_context):
    _, line = make_case(test_context)
    records = []
    # An empty log selects the established collective scheduler while retaining
    # identical compiled mechanics arithmetic, isolating scheduling and clocks.
    for log in (None, xt.Log()):
        line.mechanics.enable_vibrations(backend='compiled')
        particle = line.build_particles(x=1e-7, y=2e-8)
        particle.time_s = .2
        line.track(particle, num_turns=7, turn_by_turn_monitor=True, log=log)
        monitor = line.record_last_track
        records.append((particle.copy(_context=xo.ContextCpu()), {
            name:test_context.nparray_from_context_array(getattr(monitor, name))
            for name in ('x', 'px', 'y', 'py', 'zeta', 'delta', 'at_turn', 'at_element', 's')}))
        line.mechanics.disable_vibrations()
    for name in records[0][1]:
        np.testing.assert_array_equal(records[0][1][name], records[1][1][name])
    for name in ('reference_turns', 't_sim'):
        assert getattr(records[0][0], name) == getattr(records[1][0], name)
    from math import fsum
    assert records[0][0].time_s == fsum([.2]+[float(records[0][0].t_sim)]*7)
    assert abs(records[0][0].time_s-records[1][0].time_s) < 1e-16


def test_fixed_scheduler_falls_back_for_reference_energy_changes():
    import xobjects as xo
    from xtrack.mechanics.runtime import can_track_prepared
    line = xt.Line(elements=[xt.Quadrupole(length=.5, k1=.1), xt.ReferenceEnergyIncrease(Delta_p0c=1e6)])
    line.set_particle_ref('proton', p0c=7e12)
    line.build_tracker(_context=xo.ContextCpu())
    line.mechanics.new_girder('g', elements=['e0'])
    line.mechanics.enable_vibrations()
    line.mechanics._prepare_for_tracking()
    assert not can_track_prepared(line.tracker, line.build_particles(), {})


@for_all_test_contexts
def test_prepared_scheduler_accepts_spare_particle_capacity(test_context):
    from xtrack.mechanics.runtime import can_track_prepared

    _, line = make_case(test_context)
    line.mechanics.enable_vibrations(backend='compiled')
    line.mechanics._prepare_for_tracking()
    particles = xt.Particles(_context=test_context, p0c=7e12,
        x=[1e-7, -2e-7], y=[2e-8, -1e-8], _capacity=8)
    reference = particles.copy()
    assert can_track_prepared(line.tracker, particles, {})
    line.track(particles, num_turns=3)
    assert particles in line.mechanics._runtime._clocks
    line.track(reference, num_turns=3, log=xt.Log())
    for name in ('x', 'px', 'y', 'py', 'zeta', 'delta', 'at_turn', 'state'):
        np.testing.assert_array_equal(
            test_context.nparray_from_context_array(getattr(particles, name)),
            test_context.nparray_from_context_array(getattr(reference, name)))
    assert particles.reference_turns == reference.reference_turns == 3.
    assert particles.time_s == reference.time_s

    # A real active-particle speed mismatch still requires the conservative
    # scheduler. Inactive slots do not change its reference-beta convention.
    particles.beta0[1] = .5
    assert not can_track_prepared(line.tracker, particles, {})
    particles.state[1] = -1
    particles.reorganize()
    assert can_track_prepared(line.tracker, particles, {})


@pytest.mark.parametrize('layer', ['reference', 'installation'])
@pytest.mark.parametrize('warm', [False, True])
def test_prepared_static_inputs_reject_complex_before_any_write(layer, warm):
    _, line = make_case(xo.ContextCpu())
    view = line.mechanics
    view.enable_vibrations(backend='compiled')
    particles = line.build_particles(x=1e-7)
    if warm:
        line.track(particles, num_turns=1)
    before = np.array([[getattr(line[name], field) for field in FIELDS]
                       for name in ('q', 'b')])
    previous_turn = particles.at_turn.copy()
    body = view.manager['girder']
    offset = body if layer == 'reference' else body.installation
    offset.shift_x = np.complex128(1e-6j)
    with pytest.raises(ValueError, match='real'):
        line.track(particles, num_turns=1)
    np.testing.assert_array_equal(particles.at_turn, previous_turn)
    np.testing.assert_array_equal(
        [[getattr(line[name], field) for field in FIELDS] for name in ('q', 'b')], before)
    assert view.vibrations_enabled
