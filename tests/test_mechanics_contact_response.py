"""Loaded contact response retains inputs discarded by a geometric fit."""

import numpy as np
import pytest

import xtrack as xt


def test_rectangular_contact_response_matches_rigid_force_balance_and_replays():
    times = np.arange(256) / 256
    frequency = np.fft.rfftfreq(len(times), times[1] - times[0])
    feet = np.array([[-.3, -.5, -.5], [.3, -.5, -.5],
                     [-.3, -.5, .5], [.3, -.5, .5]])
    pattern = np.array([1., -1., -1., 1.]) * 1e-9
    field_values = np.zeros((len(times), 4, 3))
    field_values[..., 1] = np.cos(2 * np.pi * 10 * times)[:, None] * pattern
    field = xt.mechanics.FieldMotion(times, feet, field_values)
    fitted, residual = field.fit_rigid_bodies(frames=[xt.Frame()], feet_local_m=feet)
    # Adding nm displacements to the metre-scale contact geometry introduces
    # ordinary fit roundoff. This is still six orders below the loaded output.
    assert np.max(np.abs(fitted.values)) < 1e-16
    assert residual.max() > .9e-9

    # Three permitted rigid DOFs [Y, survey pitch, roll]; the other DOFs are
    # constrained in this stated model. No girder deformation is introduced.
    contact = np.column_stack([np.ones(4), feet[:, 2], feet[:, 0]])
    stiffness = np.array([2., 1., 1., 1.]) * 1e6
    damping = np.array([1., 2., 1., 1.]) * 2000
    mass = np.diag([500., 100., 20.])
    omega = 2 * np.pi * frequency
    impedance = stiffness + 1j * omega[:, None] * damping
    dynamic = (np.einsum('ji,fj,jk->fik', contact, impedance, contact)
               - omega[:, None, None]**2 * mass)
    loaded = np.linalg.solve(dynamic, contact.T[None] * impedance[:, None])
    matrix = np.zeros((len(frequency), 6, 12), dtype=complex)
    for index, coordinate in enumerate((1, 3, 5)):
        matrix[:, coordinate, 1::3] = loaded[:, index]
    transfer = xt.mechanics.TabulatedResponse(frequency, matrix)

    # Inputs are foot-major XYZ displacements, in the declared common axes.
    output = transfer.apply(field.values.reshape(len(times), 1, 12), times)
    expected_q = np.linalg.solve(dynamic[10], contact.T @ (impedance[10] * pattern))
    expected = np.zeros((len(times), 1, 6))
    for index, coordinate in enumerate((1, 3, 5)):
        expected[:, 0, coordinate] = np.real(
            expected_q[index] * np.exp(2j * np.pi * 10 * times))
    np.testing.assert_allclose(output, expected, rtol=2e-13, atol=1e-23)
    assert np.max(np.abs(output)) > 1e-10

    env = xt.Environment()
    env.new('q', xt.Quadrupole, length=.5, k1=.2, shift_x=.001)
    line = env.new_line(name='line', components=['q'])
    line.mechanics.new_girder('g', elements=['q'], anchor=xt.Frame())
    env.mechanics.add_source('loaded_contacts', xt.mechanics.RigidMotion(
        times, output, target_names=['g']), targets=['g'], reference='absolute')
    # At an exact record sample the supplied loaded pose is preserved.
    table = env.mechanics.get_table(time_s=times[7])
    assert table['dynamic_Y', 'g'] == pytest.approx(expected[7, 0, 1], abs=1e-23)
    env.mechanics.apply(time_s=times[7])
    env.mechanics.clear()
    assert line['q'].shift_x == .001


def test_rectangular_response_retains_leading_batch_axes_and_output_order():
    times = np.arange(32) / 32
    values = np.zeros((len(times), 2, 3, 4))
    values[..., 0] = np.cos(2 * np.pi * 3 * times)[:, None, None]
    values[..., 3] = np.sin(2 * np.pi * 5 * times)[:, None, None]
    matrix = np.array([[2., 0., 0., -1.], [0., 0., 0., 3.]])
    transfer = xt.mechanics.TabulatedResponse([0., 16.], [matrix, matrix])
    actual = transfer.apply(values, times)
    assert actual.shape == (len(times), 2, 3, 2)
    np.testing.assert_allclose(actual, np.einsum('ij,t...j->t...i', matrix, values), atol=1e-14)


def test_rectangular_response_checks_input_channels_and_real_endpoints():
    times = np.arange(16, dtype=float)
    transfer = xt.mechanics.TabulatedResponse([0., .5], np.ones((2, 6, 12)))
    with pytest.raises(ValueError, match='matching the input'):
        transfer.apply(np.zeros((16, 1, 6)), times)
    with pytest.raises(ValueError, match='matching the input'):
        transfer.apply(np.zeros(16), times)
    values = np.ones((16, 1, 12))
    matrix = np.ones((2, 6, 12), dtype=complex)
    matrix[0] *= 1j
    with pytest.raises(ValueError, match='DC/Nyquist'):
        xt.mechanics.TabulatedResponse([0., .5], matrix).apply(values, times)

    class EmptyInputResponse(xt.mechanics.responses.FrequencyResponse):
        def evaluate(self, frequency_hz):
            return np.zeros((len(frequency_hz), 6, 0))

    with pytest.raises(ValueError, match='matching the input'):
        EmptyInputResponse().apply(np.zeros((16, 0)), times)


def test_world_contact_channels_need_an_explicit_body_basis_response():
    times = np.arange(64) / 64
    feet = np.array([[-.3, -.5, -.5], [.3, -.5, -.5],
                     [-.3, -.5, .5], [.3, -.5, .5]])
    frame = xt.Frame.from_survey_angles(X=12., Y=-4., Z=8., phi=.2, theta=.7, psi=-.3)
    local = np.zeros((len(times), 4, 3))
    local[..., 1] = np.cos(2 * np.pi * 5 * times)[:, None] * np.array([1., 2., 4., 8.]) * 1e-9
    local[..., 0] = np.sin(2 * np.pi * 3 * times)[:, None] * 2e-9
    world = xt.mechanics.FieldMotion(times, feet @ frame.E_matrix.T + frame.XYZ,
                                     local @ frame.E_matrix.T)
    local_map = np.zeros((6, 12))
    local_map[1, 1::3] = .25
    local_map[3, 1::3] = feet[:, 2] / np.sum(feet[:, 2]**2)
    local_map[5, 1::3] = feet[:, 0] / np.sum(feet[:, 0]**2)
    # FieldMotion stores WORLD XYZ. These supplied coefficients produce local
    # rigid coordinates, so each ordered contact needs a world-to-body change.
    world_map = local_map @ np.kron(np.eye(4), frame.E_matrix.T)
    transfer = xt.mechanics.TabulatedResponse([0., 32.], [world_map, world_map])
    actual = transfer.apply(world.values.reshape(len(times), 1, 12), times)
    expected = np.einsum('ij,tj->ti', local_map, local.reshape(len(times), 12))
    np.testing.assert_allclose(actual[:, 0], expected, atol=1e-23, rtol=3e-14)
    wrong_basis = xt.mechanics.TabulatedResponse([0., 32.], [local_map, local_map])
    wrong = wrong_basis.apply(world.values.reshape(len(times), 1, 12), times)
    assert np.max(np.abs(wrong[:, 0] - expected)) > 1e-10


@pytest.mark.parametrize('shape', [(2, 0, 12), (2, 6, 0), (2, 6), (2, 6, 12, 1)])
def test_empty_or_ambiguous_response_axes_are_rejected(shape):
    with pytest.raises(ValueError, match='response'):
        xt.mechanics.TabulatedResponse([0., 1.], np.zeros(shape))
