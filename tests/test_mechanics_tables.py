import numpy as np
import pytest

import xtrack as xt


def model():
    env = xt.Environment()
    env.new('q1', xt.Quadrupole, length=.4, k1=.2, shift_x=1e-3, rot_s_rad=.12)
    env.new('d', xt.Drift, length=1.2)
    env.new('q2', xt.Quadrupole, length=.4, k1=-.2)
    line = env.new_line(name='line', components=['q1', 'd', 'q2'])
    line.mechanics.adopt_reference()
    girder = line.mechanics.new_girder('support', elements=['q1', 'q2'])
    girder.installation.shift_y = 20e-6
    girder.installation.rot_y_rad = .03
    mech = env.mechanics
    mech.add_source('pitch', xt.mechanics.Harmonic(.02, 2., .4, unit='rad'),
                    targets=['support'], component='pitch')
    mech.add_source('y', xt.mechanics.Harmonic(50e-9, 3.), targets=['support'])
    return env, line


def test_table_selection_preserves_context_and_individual_element_poses():
    env, line = model()
    original = [(line[n].shift_x, line[n].shift_y, line[n].rot_s_rad) for n in ('q1', 'q2')]
    bodies = env.mechanics.get_table(time_s=.125)
    assert list(bodies.rows.match(kind='girder').name) == ['support']
    table = line.mechanics.get_table(time_s=.125)
    quads = table.rows.match(element_type='Quadrupole').cols['name body total_X total_Y']
    assert list(quads.name) == ['q1', 'q2']
    assert list(quads.body) == ['support', 'support']
    assert quads.rows['q1'].get_element() is line.element_dict['q1']
    assert quads.rows['q1'].get_body() is env.mechanics['support']
    assert [(line[n].shift_x, line[n].shift_y, line[n].rot_s_rad) for n in ('q1', 'q2')] == original
    # Independent check through the actual alignment lowering and native survey.
    env.mechanics.apply(time_s=.125)
    actual = line.survey(include_element_frames=True)
    for name in quads.name:
        np.testing.assert_allclose(table['total_matrix', name],
                                   actual.get_frame(name, which='elem_start').matrix, atol=2e-14)


def test_selected_history_has_inherited_lever_arm_and_finite_relative_rotations():
    env, line = model()
    times = np.linspace(0, .5, 31)
    record = env.mechanics.compute_time_series(time_s=times)
    table = line.mechanics.get_table(time_s=0., time_series=record).rows['q1']
    history = table.get_time_series()
    np.testing.assert_array_equal(history.time_s, times)
    # A rotating shared support moves its two magnet entrances differently.
    assert np.max(np.abs(history.dynamic_Y-record.get_table('support').shift_y)) > .01
    for i in (0, 13, 30):
        env.mechanics.apply(time_s=times[i], time_series=record)
        actual = line.survey(include_element_frames=True).get_frame('q1', which='elem_start')
        np.testing.assert_allclose(history.total_matrix[i], actual.matrix, atol=2e-14)
        for prefix, before, after in (('installation', 'reference', 'static'), ('dynamic', 'static', 'total')):
            relative = xt.Frame.from_survey_angles(**{
                angle: history[prefix+'_'+angle][i] for angle in ('phi', 'theta', 'psi')})
            np.testing.assert_allclose(history[before+'_matrix'][i, :3, :3] @ relative.matrix[:3, :3],
                                       history[after+'_matrix'][i, :3, :3], atol=2e-15)
    assert env['q1'].rot_s_rad == .12


def test_body_history_queries_preserve_parent_motion_and_selected_rows():
    env = xt.Environment()
    root = env.mechanics.new_girder('root', anchor=xt.Frame())
    env.mechanics.new_body('magnet', kind='magnet', parent=root, mount=xt.Frame().translate_s(2))
    env.mechanics.add_source('pitch', xt.mechanics.Harmonic(.1, 0, np.pi/2, unit='rad'),
                             targets=['root'], component='pitch')
    selected = env.mechanics.get_table().rows.match(kind='magnet')
    history = selected.get_time_series(time_s=[0., 1.])
    np.testing.assert_allclose(history.dynamic_Y, 2*np.sin(.1), atol=2e-15)
    assert selected.get_body().name == 'magnet'
    with pytest.raises(KeyError, match='selection'):
        selected.get_time_series('root', time_s=[0., 1.])
    with pytest.raises(TypeError, match='line.mechanics'):
        selected.get_element()


def test_source_inventory_is_unsampled_and_keeps_models_and_target_tuples():
    class Unsampled:
        frame = 'world'
        def evaluate(self, **kwargs):
            raise AssertionError('Inventory must not sample')
    env = xt.Environment()
    env.mechanics.new_girder('one', anchor=xt.Frame())
    env.mechanics.new_girder('two', anchor=xt.Frame())
    source = Unsampled()
    env.mechanics.add_source('floor', source, targets=['one', 'two'])
    table = env.mechanics.get_source_table().rows.match(model='Unsampled')
    assert table['source', 'floor'] is source
    assert table['targets', 'floor'] == ('one', 'two')
    assert table['target_names', 'floor'] == 'one, two'
    assert table['component', 'floor'] == 'XYZ'
    assert table['n_targets', 'floor'] == 2
    assert table['unit', 'floor'] == 'm'
    assert table['seed', 'floor'] is None
    assert len(xt.Environment().mechanics.get_source_table()) == 0


def test_source_inventory_supports_minimal_scalar_protocol_and_psd_seed():
    class Scalar:
        def evaluate(self, **kwargs):
            raise AssertionError('Inventory must not sample')
    env = xt.Environment()
    env.mechanics.new_girder('g', anchor=xt.Frame())
    env.mechanics.add_source('custom', Scalar(), targets='g', component='x')
    env.mechanics.add_source('ground', xt.mechanics.DisplacementPSD(
        [0, 1, 2, 10], [0, 0, 1e-18, 0], 956), targets='g')
    table = env.mechanics.get_source_table()
    assert table['frame', 'custom'] == 'body_local'
    assert table['unit', 'custom'] == 'm'
    assert table['seed', 'ground'] == 956


def test_kind_survives_environment_copy_and_cannot_change_silently():
    env, line = model()
    env.mechanics.new_body('pump', kind='equipment', parent='support', mount=xt.Frame())
    copied = env.copy(with_progress=False)
    assert copied.mechanics['support'].kind == 'girder'
    assert copied.mechanics['pump'].kind == 'equipment'
    with pytest.raises(AttributeError, match='topology'):
        env.mechanics['pump'].kind = 'girder'
    with pytest.raises(ValueError, match='kind'):
        env.mechanics.new_body('bad', anchor=xt.Frame(), kind='')


def test_history_grid_bounds_memory_guard_and_stale_definitions():
    env, line = model()
    table = line.mechanics.get_table().rows['q1']
    with pytest.raises(ValueError, match='Supply time_s'):
        table.get_time_series()
    with pytest.raises(MemoryError, match='max_memory_bytes'):
        table.get_time_series(time_s=[0., 1.], max_memory_bytes=1)
    with pytest.raises(ValueError, match='increasing'):
        table.get_time_series(time_s=[1., 0.])
    record = env.mechanics.compute_time_series(time_s=np.linspace(0, 1, 10))
    table = line.mechanics.get_table(time_s=0., time_series=record)
    with pytest.raises(ValueError, match='outside'):
        table.get_time_series('q1', time_s=[0., 2.])
    env.mechanics.add_source('new', xt.mechanics.Harmonic(1e-9, 1.), targets=['support'])
    with pytest.raises(ValueError, match='fresh table'):
        table.get_time_series('q1')


def test_empty_pose_tables_are_selectable():
    env = xt.Environment()
    line = env.new_line(name='empty', components=[])
    assert len(env.mechanics.get_table(time_s=0.)) == 0
    assert len(line.mechanics.get_table(time_s=0.)) == 0
    assert env.mechanics.get_table(time_s=0.).total_matrix.shape == (0, 4, 4)


def test_numeric_pose_and_history_exports_do_not_serialize_live_context(tmp_path):
    env, line = model()
    record = env.mechanics.compute_time_series(time_s=np.linspace(0., 1., 11))
    table = line.mechanics.get_table(time_s=.125, time_series=record).rows['q1']
    destination = tmp_path/'pose.json'
    table.to_json(destination)
    assert '_mechanics_record' not in destination.read_text()
    restored = xt.mechanics.MechanicsTable.from_json(destination)
    np.testing.assert_allclose(restored.total_matrix, table.total_matrix)
    with pytest.raises(ValueError, match='live mechanics context'):
        restored.get_element()
    history = table.get_time_series()
    history.to_json(tmp_path/'history.json')
    loaded = xt.mechanics.MotionHistoryTable.from_json(tmp_path/'history.json')
    np.testing.assert_array_equal(loaded.time_s, history.time_s)
    np.testing.assert_allclose(loaded.total_matrix, history.total_matrix)


def test_selected_quad_animation_renders_without_applying_motion(tmp_path):
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    Image = pytest.importorskip('PIL.Image')
    env, line = model()
    selected = line.mechanics.get_table().rows['q1']
    before = line['q1'].shift_x, line['q1'].shift_y
    history = selected.get_time_series(time_s=np.linspace(0., .5, 5))
    figure, _ = history.plot()
    figure.canvas.draw()
    animation = selected.animate(time_s=np.linspace(0., .5, 4), size_m=(.3, .3, .4),
                                  displacement_scale=3., rotation_scale=4.)
    destination = tmp_path/'quad.gif'
    animation.save(destination, writer='pillow', fps=4, dpi=50)
    with Image.open(destination) as gif:
        assert gif.n_frames == 4
        gif.seek(0)
        first = np.array(gif.convert('RGB'))
        gif.seek(2)
        assert not np.array_equal(first, np.array(gif.convert('RGB')))
    assert (line['q1'].shift_x, line['q1'].shift_y) == before
    plt.close('all')
