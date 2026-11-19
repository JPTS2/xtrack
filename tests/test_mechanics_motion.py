import json
from io import StringIO

import numpy as np
import pytest

import xtrack as xt


def test_absolute_motion_does_not_double_count_parent_or_erase_installation():
    env = xt.Environment()
    parent = env.mechanics.new_girder('parent', anchor=xt.Frame())
    child = env.mechanics.new_body('child', parent=parent, mount=xt.Frame().translate_s(2))
    parent.installation.shift_y = 3e-6
    times = np.array([0., 1.])
    values = np.zeros((2, 1, 6))
    values[..., 1] = 1e-6
    values[..., 4] = .2
    env.mechanics.add_source('moving_parent', xt.mechanics.RigidMotion(times, values), targets=['parent'])
    env.mechanics.add_source('fixed_child', xt.mechanics.RigidMotion(times, np.zeros((2, 1, 6))),
                              targets=['child'], reference='absolute')
    poses = env.mechanics.get_body_poses(time_s=.5)
    np.testing.assert_allclose(poses['child'].XYZ, [0, 3e-6, 2], atol=1e-15)
    np.testing.assert_allclose(poses['child'].matrix[:3, :3], np.eye(3), atol=1e-15)
    assert poses['parent'].theta == pytest.approx(.2)
    with pytest.raises(ValueError, match='combined'):
        env.mechanics.add_source('extra', xt.mechanics.Harmonic(1e-6, 1), targets=['child'])


def test_angular_units_and_noncommuting_source_composition():
    env = xt.Environment()
    env.mechanics.new_girder('body', anchor=xt.Frame())
    with pytest.raises(ValueError, match='unit'):
        env.mechanics.add_source('bad', xt.mechanics.Harmonic(.2, 0, np.pi/2), targets=['body'], component='yaw')
    for name, component, amplitude in [('pitch', 'pitch', .2), ('yaw', 'yaw', .3)]:
        env.mechanics.add_source(name, xt.mechanics.Harmonic(amplitude, 0, np.pi/2, unit='rad'),
                                  targets=['body'], component=component)
    expected = xt.Frame.from_survey_angles(theta=.3) @ xt.Frame.from_survey_angles(phi=.2)
    actual = env.mechanics.get_body_poses()['body']
    np.testing.assert_allclose(actual.matrix, expected.matrix, atol=1e-15)
    record = env.mechanics.compute_time_series(time_s=[0., 1.])
    table = record.get_table('body')
    np.testing.assert_allclose(xt.Frame.from_survey_angles(phi=table.rot_x_rad[0],
        theta=table.rot_y_rad[0], psi=table.rot_s_rad[0]).matrix, expected.matrix, atol=1e-15)


def test_rigid_fit_recovers_known_finite_transform_and_reports_deformation():
    feet = np.array([[-.3, -.5, -1], [.3, -.5, -1], [-.3, -.5, 1], [.3, -.5, 1.]])
    frame = xt.Frame.from_survey_angles(X=12, Y=2, Z=-3, theta=.7, psi=.1)
    delta = xt.Frame.from_survey_angles(X=.002, Y=.003, Z=-.001, phi=.02, theta=.03, psi=-.01)
    positions = feet @ frame.matrix[:3, :3].T + frame.XYZ
    moved = feet @ (frame @ delta).matrix[:3, :3].T + (frame @ delta).XYZ
    values = np.broadcast_to(moved-positions, (2, 4, 3)).copy()
    field = xt.mechanics.FieldMotion([0., 1.], positions, values)
    motion, residual = field.fit_rigid_bodies(frames=[frame], feet_local_m=feet)
    np.testing.assert_allclose(motion.values[0, 0], [.002, .003, -.001, .02, .03, -.01], atol=2e-14)
    assert residual.max() < 1e-14
    values[0, 0, 1] += 1e-3
    distorted = xt.mechanics.FieldMotion([0., 1.], positions, values)
    _, residual = distorted.fit_rigid_bodies(frames=[frame], feet_local_m=feet)
    assert residual[0, 0] > 1e-5
    with pytest.raises(ValueError, match='Collinear'):
        field.fit_rigid_bodies(frames=[frame], feet_local_m=[[0, 0, -1], [0, 0, 0], [0, 0, 1]])


def test_loaded_response_matches_force_equilibrium_and_exact_cascade():
    args = dict(girder_mass_kg=500., magnet_mass_kg=150., floor_stiffness_n_m=3e6,
                magnet_stiffness_n_m=7e6, floor_damping_ns_m=1e4, magnet_damping_ns_m=2e4)
    frequency = np.array([0., 1., 10., 30., 100.])
    hg = xt.mechanics.LoadedTwoMass(**args).evaluate(frequency)
    hm = xt.mechanics.LoadedTwoMass(**args, stage='girder_to_magnet').evaluate(frequency)
    direct = xt.mechanics.LoadedTwoMass(**args, stage='floor_to_magnet').evaluate(frequency)
    np.testing.assert_allclose(hg*hm, direct, atol=1e-15)
    omega = 2*np.pi*frequency
    zg = args['floor_stiffness_n_m']+1j*omega*args['floor_damping_ns_m']
    zm = args['magnet_stiffness_n_m']+1j*omega*args['magnet_damping_ns_m']
    np.testing.assert_allclose((zg+zm-args['girder_mass_kg']*omega**2)*hg-zm*direct, zg, rtol=1e-14)
    np.testing.assert_allclose((zm-args['magnet_mass_kg']*omega**2)*direct-zm*hg, 0, atol=1e-8)


def test_matrix_filter_phase_and_causal_block_state():
    times = np.arange(2048)/256
    values = np.zeros((len(times), 1, 6))
    values[:, 0, 1] = np.cos(2*np.pi*8*times)
    matrix = np.broadcast_to(np.eye(6), (2, 6, 6)).copy()
    matrix[:, 3, 1] = .02  # rad/m, at the same reference point
    output = xt.mechanics.RigidMotion(times, values).through(xt.mechanics.TabulatedResponse([0, 128], matrix))
    np.testing.assert_allclose(output.values[:, 0, 3], .02*values[:, 0, 1], atol=1e-15)
    oscillator = xt.mechanics.BaseExcitedOscillator(12., .2)
    scalar = oscillator.apply(values[:, 0, 1], times)
    expected = np.real(oscillator.evaluate(np.array([8.]))[0]*np.exp(2j*np.pi*8*times))
    np.testing.assert_allclose(scalar, expected, atol=5e-14)
    causal = xt.mechanics.CausalResponse(oscillator=oscillator, sample_step_s=1/256)
    whole = causal.process(values)
    causal.reset()
    split = np.concatenate([causal.process(values[:701]), causal.process(values[701:])])
    np.testing.assert_array_equal(split, whole)
    from scipy.signal import lsim
    omega = 2*np.pi*oscillator.frequency_hz
    step = np.ones(256)*1e-8
    causal.reset()
    actual = causal.process(step)
    _, expected, _ = lsim(([2*oscillator.damping_ratio*omega,omega**2],
        [1,2*oscillator.damping_ratio*omega,omega**2]),step,np.arange(len(step))/256,interp=False)
    np.testing.assert_allclose(actual,expected,atol=1e-20,rtol=1e-12)
    assert actual[0] == 0


def test_continuous_ground_field_subset_and_chunks_are_the_same_process():
    source = xt.mechanics.SpectralGroundMotion([0, 1, 2, 10, 12], [0, 0, 1e-18, 1e-18, 0],
                                               seed=956, correlation_length_m=20, n_modes=128)
    positions = np.array([[0., 0, 0], [2., 0, 0], [40., 0, 0]])
    times = np.arange(100)/128
    whole = source.evaluate(times, positions)
    chunks = np.concatenate([source.evaluate(times[:31], positions), source.evaluate(times[31:], positions)])
    np.testing.assert_allclose(chunks, whole, atol=1e-22, rtol=0)
    np.testing.assert_allclose(source.evaluate(times, positions[[2, 0]]), whole[:, [2, 0]], atol=1e-22, rtol=0)
    field = xt.mechanics.sample_field(source, time_s=times, positions_m=positions)
    np.testing.assert_allclose(field.values[..., 1], whole, atol=1e-22)
    with pytest.raises(ValueError, match='workspace'):
        xt.mechanics.sample_field(source, time_s=times, positions_m=positions, max_memory_bytes=1000)


def test_located_source_delay_gain_and_regional_seed_contract():
    source = xt.mechanics.LocatedSource(xt.mechanics.Harmonic(1e-8, 5), (0, 0, 0), 10, 100)
    positions = np.array([[0., 0, 0], [10., 0, 0]])
    actual = source.evaluate(np.array([.05]), positions)
    np.testing.assert_allclose(actual[0, :, 1], [1e-8, -1e-8*np.exp(-1)], atol=1e-22)
    background = xt.mechanics.SpectralGroundMotion([0, 1, 2], [0, 1e-18, 0], 5, 10, 32)
    with pytest.raises(ValueError, match='distinct seeds'):
        xt.mechanics.RegionalGroundMotion(background, (xt.mechanics.GaussianRegion(background, (0,0,0), 10),))


def test_adopted_reference_installation_and_dynamic_seeds_are_independent():
    env = xt.Environment()
    env.new('q', xt.Quadrupole, length=.5, k1=.2, shift_x=.001, rot_s_rad=.1)
    line = env.new_line(name='line', components=['q'])
    line.mechanics.adopt_reference()
    body = line.mechanics.new_girder('g', elements=['q'])
    for dynamic_seed in (5, 6):
        installation = env.mechanics.generate_installation_errors(seed=956, translation_sigma_m=(1e-5,2e-5,0))
        installation.apply(env.mechanics)
        expected = installation.values.copy()
        # Dynamic source seeds do not enter the static generator.
        source = xt.mechanics.DisplacementPSD([0,1,2], [0,1e-18,0], dynamic_seed)
        source.evaluate(np.arange(128)/64, np.zeros((1,3)))
        np.testing.assert_array_equal(env.mechanics.generate_installation_errors(
            seed=956, translation_sigma_m=(1e-5,2e-5,0)).values, expected)
    record = xt.mechanics.InstallationErrors(('g',), [[0, 2e-5, 0, 0, 0, 0]], 5)
    corrected = record.corrected([[0,-1e-5,0,0,0,0]])
    assert record.values[0,1] == 2e-5 and corrected.values[0,1] == 1e-5
    record.apply(env.mechanics)
    env.mechanics.add_source('tone', xt.mechanics.Harmonic(1e-8,5), targets=['g'])
    env.mechanics.apply(time_s=.05)
    stream = StringIO()
    env.to_json(stream)
    copied = xt.Environment.from_dict(json.loads(stream.getvalue()))
    copied.mechanics.apply(time_s=.05)
    assert copied['q'].shift_x == pytest.approx(env['q'].shift_x)
    assert copied.mechanics['g'].installation.shift_y == 2e-5
    line.mechanics.disable_vibrations()
    assert env['q'].shift_y == pytest.approx(2e-5)
    env.mechanics.clear()
    assert env['q'].shift_x == .001 and env['q'].shift_y == 0 and env['q'].rot_s_rad == .1


def test_installation_draw_identity_groups_and_subset_stability():
    env = xt.Environment()
    for name in ('a','b','c'):
        env.mechanics.new_girder(name, anchor=xt.Frame())
    args = dict(seed=956, translation_sigma_m=(1e-4,1e-4,1e-4), rotation_sigma_rad=(1e-6,1e-6,1e-6))
    all_errors = env.mechanics.generate_installation_errors(**args)
    subset = env.mechanics.generate_installation_errors(**args, bodies=['c','a'])
    np.testing.assert_array_equal(subset.values, all_errors.values[[2,0]])
    common = env.mechanics.generate_installation_errors(**args, groups={'support':['a','b']})
    np.testing.assert_array_equal(common.values[0], common.values[1])
    assert np.all(np.abs(common.values) <= 3*np.array([1e-4]*3+[1e-6]*3))


def test_pickup_projection_is_rigid_covariant_with_offset_roll_and_pitch():
    beam = xt.Frame()
    pickup = xt.Frame.from_survey_angles(X=1e-3,Y=2e-3,Z=.1,phi=.01,psi=.1)
    coordinates = dict(x=1e-4,y=-1e-4,px=.001,py=-.002,delta=.01)
    reference = xt.mechanics.project_to_bpm(beam_frame=beam,bpm_frame=pickup,**coordinates)
    common = xt.Frame.from_survey_angles(X=3,Y=2,Z=-1,phi=.2,theta=-.3,psi=.1)
    moved = xt.mechanics.project_to_bpm(beam_frame=common@beam,bpm_frame=common@pickup,**coordinates)
    np.testing.assert_allclose([moved.x,moved.y,moved.distance_m],
                               [reference.x,reference.y,reference.distance_m],atol=2e-15)


def test_survey_separates_three_position_classes_without_writes():
    env = xt.Environment()
    body = env.mechanics.new_girder('g',anchor=xt.Frame())
    body.shift_y = 1e-3
    body.installation.shift_y = 2e-5
    env.mechanics.add_source('tone',xt.mechanics.Harmonic(1e-8,0,np.pi/2),targets=['g'])
    survey = env.mechanics.survey()
    assert survey.reference_Y[0] == 1e-3
    assert survey.installation_Y[0] == pytest.approx(2e-5)
    assert survey.dynamic_Y[0] == pytest.approx(1e-8)


def test_line_local_shared_clock_updates_do_not_write_other_beam_fields():
    env = xt.Environment()
    for name in ('a', 'b'):
        env.new(name, xt.Quadrupole, length=1, k1=0)
    aa = env.new_line(name='aa', components=['a'])
    bb = env.new_line(name='bb', components=['b'])
    bb.mechanics.set_reference_frame(xt.Frame.from_survey_angles(Z=1,theta=np.pi))
    body = aa.mechanics.new_girder('shared',elements=['a'])
    bb.mechanics.attach(body,elements=['b'])
    env.mechanics.add_source('wave', xt.mechanics.PlaneWave(1e-8,5,(0,0,1),100,
        polarization=(1,0,0)),targets=['shared'])
    aa.mechanics.apply(time_s=.05)
    first_value = env['a'].shift_x
    assert env['b'].shift_x == 0
    bb.mechanics.apply(time_s=.02)
    assert env['a'].shift_x == first_value
    for line in (aa,bb):
        line.set_particle_ref('proton',p0c=7e12)
        line.build_tracker()
        line.mechanics.enable_vibrations(time_origin_s=.01)
        line.track(line.build_particles(),num_turns=2)
        line.mechanics.disable_vibrations()


def test_passage_sampling_uses_reference_arrival_and_is_transactional():
    from scipy.constants import c
    env = xt.Environment()
    env.new('a',xt.Quadrupole,length=1,k1=0)
    env.new('d',xt.Drift,length=100)
    env.new('b',xt.Quadrupole,length=1,k1=0)
    line = env.new_line(name='line',components=['a','d','b'])
    line.set_particle_ref('proton',p0c=7e12)
    for name in ('a','b'):
        line.mechanics.new_girder('g.'+name,elements=[name])
    env.mechanics.add_source('tone',xt.mechanics.Harmonic(1e-8,1e5),targets=['g.a','g.b'])
    line.mechanics.apply(time_s=0,sampling='passage')
    assert env['a'].shift_y == 0
    expected = 1e-8*np.sin(2*np.pi*1e5*101/(float(line.particle_ref.beta0[0])*c))
    assert env['b'].shift_y == pytest.approx(expected)
    record = env.mechanics.compute_time_series(time_s=[0.,1e-8])
    before = env['a'].shift_y
    with pytest.raises(ValueError,match='outside'):
        line.mechanics.apply(time_s=0,time_series=record,sampling='passage')
    assert env['a'].shift_y == before


def test_finite_rigid_port_transport_includes_lever_arm_and_axes():
    origin = xt.Frame.from_survey_angles(X=3,Y=2,Z=-1,theta=.4,psi=.2)
    target = origin @ xt.Frame.from_survey_angles(X=.2,Y=.3,Z=2,theta=-.1)
    values = np.array([[[.01,.02,.03,.1,-.2,.15]]]*2)
    record = xt.mechanics.RigidMotion([0.,1.],values)
    moved = record.transport(from_frames=[origin],to_frames=[target])
    delta = xt.Frame.from_survey_angles(X=.01,Y=.02,Z=.03,phi=.1,theta=-.2,psi=.15)
    expected = origin @ delta @ origin.inverse() @ target
    v = moved.values[0,0]
    actual = target @ xt.Frame.from_survey_angles(X=v[0],Y=v[1],Z=v[2],phi=v[3],theta=v[4],psi=v[5])
    np.testing.assert_allclose(actual.matrix,expected.matrix,atol=2e-15)


def test_vector_field_seed_ownership_projection_and_roundtrip():
    from xtrack.mechanics.sources import encode_source,decode_source
    background = xt.mechanics.SpectralGroundMotion([0,1,2],[0,1e-18,0],5,10,32)
    regional = xt.mechanics.RegionalGroundMotion(background,())
    with pytest.raises(ValueError,match='distinct seeds'):
        xt.mechanics.VectorGroundMotion((background,regional,None))
    source = xt.mechanics.VectorGroundMotion((None,regional,None))
    copied = decode_source(encode_source(source))
    times = np.arange(100)/128
    frame = xt.Frame.from_survey_angles(theta=.3,psi=.1)
    field = xt.mechanics.sample_field(source,time_s=times,positions_m=[frame.XYZ])
    motion = field.to_rigid_motion(frames=[frame])
    np.testing.assert_allclose(np.einsum('ij,tbj->tbi',frame.matrix[:3,:3],motion.values[...,:3]),field.values,atol=1e-23)
    np.testing.assert_array_equal(copied.evaluate(times,np.array([frame.XYZ])),field.values)
    bad = xt.Frame(np.diag([1.00000001,1,1,1]))
    with pytest.raises(ValueError,match='proper'):
        field.to_rigid_motion(frames=[bad])


def test_rigid_record_serializes_and_psd_band_edge_is_not_silently_truncated():
    from xtrack.mechanics.sources import encode_source,decode_source
    record = xt.mechanics.RigidMotion([0,1],np.ones((2,1,6))*1e-8,('body',))
    copied = decode_source(json.loads(json.dumps(encode_source(record))))
    np.testing.assert_array_equal(copied.values,record.values)
    assert copied.target_names == record.target_names
    source = xt.mechanics.DisplacementPSD([0,1,40,60],[0,1e-18,1e-18,0],5)
    with pytest.raises(ValueError,match='Nyquist'):
        source.evaluate(np.arange(1024)/100,np.zeros((1,3)))


def test_support_contacts_are_owned_immutable_geometry_and_survive_copy():
    env = xt.Environment()
    feet = np.array([[-.3,-.5,-1],[.3,-.5,-1],[-.3,-.5,1],[.3,-.5,1]])
    body = env.mechanics.new_girder('g',anchor=xt.Frame.from_survey_angles(X=2,Z=3,theta=.2),feet_local_m=feet)
    before = body.nominal_feet_m.copy()
    feet[:] = 99
    np.testing.assert_array_equal(body.nominal_feet_m,before)
    with pytest.raises(ValueError):
        body.feet_local_m[0,0] = 10
    copied = env.copy(with_progress=False)
    np.testing.assert_array_equal(copied.mechanics['g'].nominal_feet_m,before)


def test_located_psd_requires_a_joint_record_before_automatic_tracking():
    env = xt.Environment()
    env.new('q',xt.Quadrupole,length=1,k1=0)
    line = env.new_line(name='line',components=['q'])
    line.mechanics.new_girder('g',elements=['q'])
    source = xt.mechanics.LocatedSource(
        xt.mechanics.DisplacementPSD([0,1,2,10,12],[0,0,1e-18,1e-18,0],956),
        (0,0,0),10,100)
    env.mechanics.add_source('pump',source,targets=['g'])
    with pytest.raises(ValueError,match='compute_time_series'):
        line.mechanics.enable_vibrations()
    record = env.mechanics.compute_time_series(time_s=np.arange(1024)/128)
    line.mechanics.enable_vibrations(time_series=record)
    line.mechanics.disable_vibrations()


def test_plot_body_view_focuses_contacts_instead_of_full_line():
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    env = xt.Environment()
    env.new('q',xt.Quadrupole,length=1,k1=0)
    env.new('d',xt.Drift,length=100)
    line = env.new_line(name='line',components=['q','d'])
    line.mechanics.new_girder('g',elements=['q'],feet_local_m=[[-.3,-.5,-1],[.3,-.5,1]])
    fig,(layout,_) = env.mechanics.plot(line=line,view='bodies')
    assert np.diff(layout.get_ylim())[0] < 5
    assert len(fig._mechanics_geometry) == 3  # contacts and physical element span
    assert all(np.max(np.abs(artist.get_data_3d())) < 5 for artist in layout.lines)
    fig.canvas.draw()
    plt.close(fig)


def test_composed_fields_and_named_motion_ports_preserve_geometry():
    time = np.array([0.,1.])
    positions = np.zeros((1,3))
    aa = xt.mechanics.FieldMotion(time,positions,np.ones((2,1,3))*1e-9)
    bb = xt.mechanics.FieldMotion(time,positions,np.ones((2,1,3))*2e-9)
    np.testing.assert_allclose((aa+bb).values,3e-9)
    with pytest.raises(ValueError,match='identical'):
        _ = aa+xt.mechanics.FieldMotion(time,positions+1,bb.values)
    va = np.zeros((2,1,6)); va[...,3] = .2
    vb = np.zeros_like(va); vb[...,4] = .3
    motion = xt.mechanics.RigidMotion(time,va,'parent').superpose(xt.mechanics.RigidMotion(time,vb,'parent'))
    expected = xt.Frame.from_survey_angles(theta=.3) @ xt.Frame.from_survey_angles(phi=.2)
    v = motion.values[0,0]
    actual = xt.Frame.from_survey_angles(phi=v[3],theta=v[4],psi=v[5])
    np.testing.assert_allclose(actual.matrix,expected.matrix,atol=1e-15)
    ports = motion.select(['parent','parent'],target_names=['pickup_a','pickup_b'])
    np.testing.assert_array_equal(ports.values[:,0],ports.values[:,1])
    assert motion.select('parent').target_names == ('parent',)


def test_dynamic_plot_baseline_excludes_installation():
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    env = xt.Environment()
    body = env.mechanics.new_girder('girder',anchor=xt.Frame())
    body.installation.shift_y = 3e-6
    env.mechanics.add_source('tone',xt.mechanics.Harmonic(2e-8,0,np.pi/2),targets='girder')
    fig,(axis,_) = env.mechanics.plot(bodies='girder',baseline='static',displacement_scale=1e6,view='bodies')
    reference = axis.collections[0]._offsets3d[2][0]
    total = axis.collections[1]._offsets3d[2][0]
    assert total-reference == pytest.approx(.02)
    plt.close(fig)
