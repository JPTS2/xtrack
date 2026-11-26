"""Independent parallel solenoid axes preserve the adopted multipole charts."""

import numpy as np
import pytest
import xobjects as xo
import xtrack as xt
from xobjects.test_helpers import for_all_test_contexts


def make_line(context=None, *, rolls=(.2, -.1, .3)):
    env = xt.Environment()
    names = [f'sol.{i}' for i in range(3)]
    for i, (name, profile, roll) in enumerate(zip(
        names, ((0., .12), (.12, .12), (.12, 0.)), rolls,
    )):
        env.new(
            name, xt.VariableSolenoid, length=.25, ks_profile=profile,
            x0=(i+1)*3e-6, y0=-(i+1)*2e-6,
            shift_x=(i+1)*1e-6, rot_s_rad_no_frame=roll,
            knl=[.0001, .02, .08], ksl=[.0003, -.004, .06],
        )
    line = env.new_line(name='ring', components=names)
    line.set_particle_ref('positron', p0c=45.6e9)
    line.build_tracker(_context=context)
    chart = line.mechanics.new_girder('chart', elements=names)
    return env, line, chart


def offsets(line):
    return np.array([[line[name].x0, line[name].y0] for name in line.element_names])


def field_strengths(line):
    return [
        tuple(line._context.nparray_from_context_array(getattr(line[name], field))
              for field in ('knl', 'ksl', 'ks_profile'))
        for name in line.element_names
    ]


def expected_offsets(adopted, dx, dy, rolls=(.2, -.1, .3)):
    # Axis coordinates follow the first adopted element, while each x0/y0
    # lives in that element's chart, potentially with a different roll.
    angle = np.array(rolls) - rolls[0]
    return adopted + np.column_stack((
        np.cos(angle)*dx + np.sin(angle)*dy,
        -np.sin(angle)*dx + np.cos(angle)*dy,
    ))


@for_all_test_contexts
def test_axis_group_tracking_and_restoration(test_context):
    env, line, chart = make_line(test_context)
    mech = env.mechanics
    view = line.mechanics
    adopted = offsets(line)
    chart_baseline = line['sol.0'].shift_x
    axis = view.new_solenoid_axis('field', elements=line.element_names)
    assert axis.parent == 'chart'
    assert axis.kind == 'solenoid_axis'
    assert view.get_table()['solenoid_axis_body', 'sol.2'] == axis.name
    chart.shift_y = 2e-5
    chart.rot_x_rad = 3e-4
    chart.rot_y_rad = -2e-4
    chart.rot_s_rad = 4e-4
    chart.installation.shift_s = 3e-5
    axis.shift_x, axis.shift_y = 2e-6, -3e-6
    axis.installation.shift_x, axis.installation.shift_y = 1e-6, 2e-6
    mech.add_source('axis_x', xt.mechanics.Harmonic(2e-6, 150., .3),
                    targets='field', component='x')
    times = np.linspace(0, .02, 21)
    displacement = np.zeros((len(times), 1, 6))
    displacement[:, 0, 1] = 1e-6*np.cos(2*np.pi*50*times + .4)
    mech.add_source(
        'axis_y', xt.mechanics.RigidMotion(time_s=times, values=displacement,
                                          target_names=['field']),
        targets='field',
    )
    record = mech.compute_time_series(time_s=times)
    # A magnet powering trim remains live; no axis update rewrites strengths.
    env['sol.1'].knl[1] = .023
    strengths = field_strengths(line)
    t = times[5]
    dx = 3e-6 + 2e-6*np.sin(2*np.pi*150*t+.3)
    dy = -1e-6 + displacement[5, 0, 1]
    expected = expected_offsets(adopted, dx, dy)
    table = view.get_solenoid_axis_table(time_s=t, time_series=record)
    np.testing.assert_allclose(np.column_stack((table.effective_x0, table.effective_y0)),
                               expected, atol=2e-19, rtol=0.)
    np.testing.assert_array_equal(offsets(line), adopted)
    frozen = view.freeze_at(time_s=t, time_series=record)
    assert frozen.design_alignment['sol.2']['x0'] == adopted[-1, 0]
    np.testing.assert_allclose(offsets(line), expected, atol=2e-19, rtol=0.)

    initial = line.build_particles(x=[1e-5, -1e-5], y=[2e-5, -3e-5])
    tracked = []
    for backend in ('reference', 'compiled'):
        view.enable_vibrations(backend=backend, time_series=record)
        particles = initial.copy()
        line.track(particles, num_turns=4)
        assert np.all(test_context.nparray_from_context_array(particles.state) > 0)
        tracked.append([
            test_context.nparray_from_context_array(getattr(particles, name))
            for name in ('x', 'px', 'y', 'py', 'zeta', 'delta', 'ax', 'ay')
        ])
        current = view.get_solenoid_axis_table(time_s=view.last_applied_time_s,
                                               time_series=record)
        np.testing.assert_allclose(
            offsets(line), np.column_stack((current.effective_x0, current.effective_y0)),
            atol=2e-19, rtol=0.,
        )
        view.disable_vibrations()
        np.testing.assert_allclose(
            offsets(line), expected_offsets(adopted, 3e-6, -1e-6),
            atol=2e-19, rtol=0.,
        )
    np.testing.assert_allclose(tracked[0], tracked[1], atol=2e-18, rtol=2e-13)
    for actual, stored in zip(field_strengths(line), strengths):
        for a, b in zip(actual, stored):
            np.testing.assert_array_equal(a, b)

    # The currently applied fields and original baselines survive a copy.
    copy = env.copy(with_progress=False)
    np.testing.assert_array_equal(offsets(copy.ring), offsets(line))
    assert not copy.ring.mechanics.automatic_updates_enabled
    copy.mechanics.restore()
    np.testing.assert_array_equal(offsets(copy.ring), adopted)
    copy.ring.mechanics.clear()
    np.testing.assert_array_equal(offsets(copy.ring), adopted)
    assert copy.ring['sol.0'].shift_x == chart_baseline
    assert not copy.ring.mechanics._solenoid_axes
    mech.clear()
    np.testing.assert_array_equal(offsets(line), adopted)
    assert line['sol.0'].shift_x == chart_baseline
    assert not mech.bodies


def test_axis_native_analytic_motion_and_geometry_guard(monkeypatch):
    env, line, chart = make_line(xo.ContextCpu(), rolls=(0.,)*3)
    axis = line.mechanics.new_solenoid_axis('field', elements=line.element_names)
    adopted = offsets(line)
    env.mechanics.add_source('wave', xt.mechanics.Harmonic(1e-6, 20., .6),
                             targets=axis.name, component='x')
    particles = line.build_particles()
    line.mechanics.enable_vibrations(backend='compiled')
    line.track(particles, num_turns=2)
    runtime = line.mechanics._runtime
    assert runtime.analytic
    assert runtime.solenoid_axes
    guard = runtime.guard
    line['sol.1'].knl[0] += 1e-6
    with monkeypatch.context() as patch:
        def forbidden(*args, **kwargs):
            raise AssertionError('Axis guard made a scalar context read')
        patch.setattr(type(guard.buffer), 'to_bytearray', forbidden)
        guard.check()
    baseline = offsets(line)
    line['sol.1'].x0 += 1e-7
    with pytest.raises(ValueError, match='alignment edited'):
        line.track(particles, num_turns=1)
    assert particles.at_turn[0] == 2
    line['sol.1'].x0 = baseline[1, 0]
    # A matching-value expression is still a conflicting numeric owner.
    env['offset'] = baseline[1, 0]
    env.ref['sol.1'].x0 = env.ref['offset']
    with pytest.raises(ValueError, match='expression-driven alignment'):
        line.track(particles, num_turns=1)
    env.ref['sol.1'].x0 = baseline[1, 0]
    line.mechanics.clear()
    np.testing.assert_array_equal(offsets(line), adopted)


@pytest.mark.parametrize('backend', ['explicit', 'reference', 'compiled'])
@pytest.mark.parametrize('invalid', ['setting', 'installation', 'sampled', 'live'])
def test_axis_rejects_unrepresentable_relative_motion(backend, invalid):
    env, line, chart = make_line(xo.ContextCpu(), rolls=(0.,)*3)
    axis = line.mechanics.new_solenoid_axis('field', elements=line.element_names)
    chart.shift_x = 4e-6
    adopted = offsets(line)
    if invalid == 'setting':
        axis.shift_s = 1e-6
    elif invalid == 'installation':
        axis.installation.rot_y_rad = 1e-6
    elif invalid == 'sampled':
        data = np.zeros((2, 1, 6))
        data[:, 0, 3] = 2e-6
        env.mechanics.add_source(
            'bad', xt.mechanics.RigidMotion(time_s=[0., .1], values=data,
                                          target_names=[axis.name]),
            targets=axis.name,
        )
        with pytest.raises(ValueError, match='only transverse'):
            env.mechanics.compute_time_series(time_s=[0., .1])
    else:
        class HeldMotion:
            frame = 'body_local_rigid'
            def evaluate(self, time_s, positions_m):
                out = np.zeros((len(time_s), len(positions_m), 6))
                out[..., 5] = 2e-6
                return out
        env.mechanics.add_source('bad', HeldMotion(), targets=axis.name, live=True)
    if backend == 'explicit':
        with pytest.raises(ValueError, match='only transverse'):
            env.mechanics.apply()
    else:
        line.mechanics.enable_vibrations(backend=backend)
        particles = line.build_particles()
        with pytest.raises(ValueError, match='only transverse'):
            line.track(particles, num_turns=1)
        assert particles.at_turn[0] == 0
    assert line['sol.0'].shift_x == 1e-6
    np.testing.assert_array_equal(offsets(line), adopted)


def test_axis_rejects_invalid_source_component_and_installation_case():
    env, line, chart = make_line(xo.ContextCpu())
    axis = line.mechanics.new_solenoid_axis('field', elements=line.element_names)
    manager = env.mechanics
    with pytest.raises(ValueError, match='only transverse'):
        manager.add_source('bad', xt.mechanics.Harmonic(1e-6, 3., unit='rad'),
                           targets=axis.name, component='roll')
    with pytest.raises(ValueError, match='only transverse'):
        manager.add_source('bad', xt.mechanics.MeasuredMotion([0., 1.], [0., 1e-6]),
                           targets=axis.name, component='s')
    with pytest.raises(ValueError, match='World-field shortcut drives roots'):
        manager.add_source('bad', xt.mechanics.PlaneWave(1e-6, 3., (1, 0, 0), 100.),
                           targets=axis.name)
    assert not manager.sources
    manager.add_source('x', xt.mechanics.Harmonic(1e-6, 3.),
                       targets=axis.name, component='x')
    forged = np.ones((2, 1, 6))
    with pytest.raises(ValueError, match='only transverse'):
        xt.mechanics.MechanicalTimeSeries(manager, [0., 1.], {'x': forged})
    with pytest.raises(ValueError, match='only transverse'):
        manager.generate_installation_errors(
            seed=4, bodies=[axis.name], translation_sigma_m=(1e-6, 0., 1e-6),
        )
    case = xt.mechanics.InstallationErrors(
        [chart.name, axis.name], [[1e-6, 0., 0., 0., 0., 0.],
                                 [1e-6, 0., 1e-6, 0., 0., 0.]],
    )
    with pytest.raises(ValueError, match='only transverse'):
        case.apply(manager)
    assert chart.installation.shift_x == 0.
    valid = manager.generate_installation_errors(
        seed=4, bodies=[axis.name], translation_sigma_m=(1e-6, 2e-6, 0.),
    )
    valid.apply(manager)
    manager.apply()
    np.testing.assert_allclose(
        offsets(line), expected_offsets(np.array([[3e-6, -2e-6], [6e-6, -4e-6],
                                                  [9e-6, -6e-6]]),
                                         *valid.values[0, :2]),
        atol=2e-19, rtol=0.,
    )


def test_axis_rejects_conflicting_field_ownership_before_binding():
    env, line, chart = make_line(xo.ContextCpu())
    env['offset'] = 6e-6
    env.ref['sol.1'].x0 = env.ref['offset']
    with pytest.raises(ValueError, match='expression-driven axis'):
        line.mechanics.new_solenoid_axis('bad', elements=line.element_names)
    assert 'bad' not in env.mechanics.bodies
    assert not line.mechanics._solenoid_axes
    env.ref['sol.1'].x0 = 6e-6
    axis = line.mechanics.new_solenoid_axis('field', elements=['sol.0', 'sol.1'])
    with pytest.raises(ValueError, match='existing solenoid axis'):
        line.mechanics.attach_solenoid_axis(axis, elements=['sol.1', 'sol.2'])
    assert 'sol.2' not in line.mechanics._solenoid_axes
    line.mechanics.attach_solenoid_axis(axis, elements=['sol.2'])
    assert line.mechanics.get_solenoid_axis_table().name.tolist() == list(line.element_names)


def test_reverse_line_projects_shared_relative_axis_with_the_correct_sign():
    env = xt.Environment()
    for name in ('forward', 'reverse'):
        env.new(name, xt.VariableSolenoid, length=.5, ks_profile=[0., .1],
                x0=1e-6, y0=2e-6)
    line = env.new_line(name='b1', components=['forward'])
    reverse = env.new_line(name='b2', components=['reverse'])
    chart = line.mechanics.new_girder('chart', elements=['forward'])
    axis = line.mechanics.new_solenoid_axis('field', elements=['forward'])
    reverse.mechanics.set_reference_frame(xt.Frame().translate_s(.5).rotate_y(np.pi))
    reverse.mechanics.attach(chart, elements=['reverse'])
    reverse.mechanics.attach_solenoid_axis(axis, elements=['reverse'])
    axis.shift_x, axis.shift_y = 3e-6, 4e-6
    env.mechanics.apply()
    np.testing.assert_allclose(offsets(line), [[4e-6, 6e-6]], atol=2e-20)
    np.testing.assert_allclose(offsets(reverse), [[-2e-6, 6e-6]], atol=2e-20)
    env.mechanics.restore()
    np.testing.assert_array_equal(offsets(line), [[1e-6, 2e-6]])
    np.testing.assert_array_equal(offsets(reverse), [[1e-6, 2e-6]])


def test_axis_rejects_nonparallel_charts():
    env, line, chart = make_line(xo.ContextCpu())
    line.mechanics.clear()
    line['sol.1'].rot_y_rad = .01
    line.mechanics.attach(chart, elements=line.element_names)
    with pytest.raises(ValueError, match='parallel solenoid axes'):
        line.mechanics.new_solenoid_axis('bad', elements=line.element_names)
    assert 'bad' not in env.mechanics.bodies
    assert not line.mechanics._solenoid_axes


def test_axis_prescribed_psd_and_passage_sampling():
    env, line, chart = make_line(xo.ContextCpu(), rolls=(0.,)*3)
    axis = line.mechanics.new_solenoid_axis('field', elements=line.element_names)
    manager = env.mechanics
    psd = xt.mechanics.DisplacementPSD([0., 10., 100., 120.],
                                      [0., 1e-16, 1e-16, 0.], seed=4)
    with pytest.raises(ValueError, match='only transverse'):
        manager.add_source('bad', psd, targets=axis.name, component='s')
    manager.add_source('psd', psd, targets=axis.name, component='y')
    record = manager.compute_time_series(time_s=np.linspace(0., .1, 201))
    adopted = offsets(line)
    time_s = .02
    table = line.get_table()
    speed = float(line.particle_ref.beta0[0]) * 299792458.
    expected = adopted.copy()
    for i, name in enumerate(line.element_names):
        expected[i, 1] += record._sample(
            manager.sources['psd'], time_s + table['s', name]/speed,
        )[0, 1]
    line.mechanics.apply(time_s=time_s, sampling='passage', time_series=record)
    np.testing.assert_allclose(offsets(line), expected, atol=2e-20, rtol=0.)
    # Near the upper bound the first pose is valid but later segment passages
    # are not: validate the field-axis source window before advancing particles.
    line.mechanics.enable_vibrations(
        sampling='passage', backend='reference', time_series=record,
        time_origin_s=record.time_s[-1],
    )
    p = line.build_particles()
    with pytest.raises(ValueError, match='lie outside the motion record'):
        line.track(p, num_turns=1)
    assert p.at_turn[0] == 0
    np.testing.assert_allclose(offsets(line), expected, atol=2e-20, rtol=0.)
