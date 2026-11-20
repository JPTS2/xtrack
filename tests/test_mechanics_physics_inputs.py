"""Physical units and real geometry must survive validation without casts."""
import numpy as np
import pytest

import xtrack as xt


def _field():
    feet = np.array([[-.3, -.5, -1.], [.3, -.5, -1.],
                     [-.3, -.5, 1.], [.3, -.5, 1.]])
    return xt.mechanics.FieldMotion([0., 1.], feet, np.zeros((2, 4, 3))), feet


@pytest.mark.parametrize('vector', [False, True])
def test_angular_source_is_not_relabelled_as_ground_displacement(vector):
    class AngularField:
        frame = 'world'
        unit = 'rad'

        def evaluate(self, time_s, positions_m):
            shape = (len(time_s), len(positions_m)) + ((3,) if vector else ())
            return np.ones(shape)

    with pytest.raises(ValueError, match='displacement'):
        xt.mechanics.sample_field(AngularField(), time_s=[0., 1.], positions_m=[[0., 0., 0.]])


@pytest.mark.parametrize('parameter', ['position_m', 'polarization'])
def test_located_source_rejects_complex_geometry_without_discarding_it(parameter):
    options = dict(position_m=(0., 0., 0.), polarization=(0., 1., 0.))
    options[parameter] = np.array([1j, 1., 0.])
    with pytest.raises(ValueError, match='real'):
        xt.mechanics.LocatedSource(xt.mechanics.Harmonic(1e-9, 5.),
            decay_length_m=10., speed_m_s=100., **options)


def test_vector_field_and_polarization_reject_complex_coordinates():
    scalar = xt.mechanics.Harmonic(1e-9, 5.)
    with pytest.raises(ValueError, match='real'):
        xt.mechanics.VectorGroundMotion((None, scalar, None), mixing=np.eye(3)*(1+1j))
    with pytest.raises(ValueError, match='real'):
        xt.mechanics.sample_field(scalar, time_s=[0., 1.], positions_m=[[0., 0., 0.]],
                                 polarization=np.array([1j, 1., 0.]))


def test_support_fit_rejects_complex_contacts_weights_and_frames():
    field, feet = _field()
    with pytest.raises(ValueError, match='real'):
        field.fit_rigid_bodies(frames=[xt.Frame()], feet_local_m=feet*(1+1j))
    with pytest.raises(ValueError, match='real'):
        field.fit_rigid_bodies(frames=[xt.Frame()], feet_local_m=feet,
                               weights=np.ones(4)*(1+1j))
    motion = xt.mechanics.RigidMotion([0., 1.], np.zeros((2, 1, 6)))
    with pytest.raises(ValueError, match='real'):
        motion.transport(from_frames=[np.eye(4)*(1+1j)], to_frames=[xt.Frame()])


def test_regional_geometry_is_real_and_owned_by_the_definition():
    scalar = xt.mechanics.Harmonic(1e-9, 5.)
    with pytest.raises(ValueError, match='real'):
        xt.mechanics.GaussianRegion(scalar, np.array([1j, 0., 0.]), 10.)
    center = np.array([1., 2., 3.])
    region = xt.mechanics.GaussianRegion(scalar, center, 10.)
    center[:] = 99.
    assert region.center_m == (1., 2., 3.)


def test_installation_rejects_complex_offsets_and_distribution_widths():
    with pytest.raises(ValueError, match='real'):
        xt.mechanics.RigidOffset(shift_y=np.complex128(1e-6+1e-6j)).values()
    env = xt.Environment()
    env.mechanics.new_girder('g', anchor=xt.Frame())
    for parameter in ('translation_sigma_m', 'rotation_sigma_rad'):
        with pytest.raises(ValueError, match='real'):
            env.mechanics.generate_installation_errors(
                seed=956, **{parameter: np.array([1e-6+1e-6j, 0., 0.])})


def test_pickup_requires_real_ray_and_proper_rigid_frames():
    with pytest.raises(ValueError, match='real'):
        xt.mechanics.project_to_bpm(beam_frame=xt.Frame(), bpm_frame=xt.Frame(),
                                    x=np.array(1e-3+1e-3j))
    with pytest.raises(ValueError, match='proper'):
        xt.mechanics.project_to_bpm(beam_frame=xt.Frame(),
                                    bpm_frame=xt.Frame(np.diag([1.1, 1., 1., 1.])))


def test_pickup_parallel_test_uses_direction_not_absolute_momentum():
    reading = xt.mechanics.project_to_bpm(beam_frame=xt.Frame(),
        bpm_frame=xt.Frame.from_survey_angles(Z=.25), x=1e-4,
        delta=-1+8*np.finfo(float).eps)
    assert reading.x == pytest.approx(1e-4)
    assert reading.distance_m == pytest.approx(.25)


def test_mixed_unit_channel_cannot_hide_an_unrepresentable_nyquist_phase():
    # Endpoint validity is independent of unrelated channels' units/scales.
    time = np.arange(16, dtype=float)
    values = np.zeros((len(time), 1, 6))
    values[:, 0, 0] = 1.
    values[:, 0, 3] = 1e-15*(-1.)**np.arange(len(time))
    matrix = np.broadcast_to(np.eye(6), (2, 6, 6)).astype(complex).copy()
    matrix[-1, 3, 3] = 1j
    response = xt.mechanics.TabulatedResponse([0., .5], matrix)
    with pytest.raises(ValueError, match='Nyquist'):
        response.apply(values, time)


def test_invalid_response_dimensions_give_a_validation_error():
    with pytest.raises(ValueError, match='response'):
        xt.mechanics.TabulatedResponse([0., 1.], 1.)
