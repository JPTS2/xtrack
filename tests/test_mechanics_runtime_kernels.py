"""Context-portable conformance tests for the prepared mechanics kernels."""
from pathlib import Path

import numpy as np
import pytest
import xobjects as xo
from xobjects.test_helpers import allow_kernel_compilation, for_all_test_contexts

from xtrack.mechanics.local_deformations import compose, inherit, rotation_increment


@pytest.mark.parametrize('context_class', [xo.ContextCupy, xo.ContextPyopencl])
def test_prepared_kernel_registration_uses_portable_context_api(context_class):
    """Exercise the GPU registration signature without requiring a device."""
    from xtrack.mechanics.geometry_guard import _compile as compile_guard
    from xtrack.mechanics.runtime import _compile as compile_runtime

    class RegistrationContext(context_class):
        def __init__(self):
            self._kernels = {}
            self.registrations = []

        def build_kernels(self, kernel_descriptions, **kwargs):
            self.registrations.append(tuple(kernel_descriptions))
            return dict.fromkeys(kernel_descriptions)

    context = RegistrationContext()
    compile_runtime(context)
    compile_guard(context)
    compile_runtime(context)
    compile_guard(context)
    assert len(context.registrations) == 2
    assert 'mechanics_sources_mixed' in context.kernels
    assert 'mechanics_guard_gather' in context.kernels


def compile_kernels(context):
    def pointer(name, dtype=xo.Float64):
        return xo.Arg(dtype, name=name, pointer=True)
    kernels = {
        'mechanics_source_values': xo.Kernel(args=[xo.Arg(xo.Int64, name='n'),
            pointer('source_starts', xo.Int64), pointer('source_rows', xo.Int64),
            pointer('values'), pointer('dynamic_local')], n_threads='n'),
        'mechanics_sources': xo.Kernel(args=[xo.Arg(xo.Int64, name='n'),
            pointer('source_starts', xo.Int64), pointer('source_data'),
            xo.Arg(xo.Float64, name='time_s'), pointer('dynamic_local')], n_threads='n'),
        'mechanics_body_update': xo.Kernel(args=[xo.Arg(xo.Int64, name='n'),
            pointer('paths_offsets', xo.Int64), pointer('paths', xo.Int64),
            pointer('relations'), pointer('static_local'), pointer('dynamic_local'),
            pointer('out')], n_threads='n'),
        'mechanics_lower': xo.Kernel(args=[xo.Arg(xo.Int64, name='n'),
            pointer('body_indices', xo.Int64), pointer('geometry'), pointer('baseline'),
            pointer('offsets', xo.Int64), pointer('bodyout'),
            pointer('buffer', xo.Int8), pointer('values')], n_threads='n')}
    context.add_kernels(sources=[Path(__file__).resolve().parents[1]
        /'xtrack/mechanics/runtime_src/update.h'], kernels=kernels)


@for_all_test_contexts
@allow_kernel_compilation
def test_prepared_kernel_hierarchy_and_lowering(test_context):
    compile_kernels(test_context)
    to_context = test_context.nparray_to_context_array
    to_numpy = test_context.nparray_from_context_array
    rng = np.random.default_rng(956)
    n = 19
    parents = np.array([-1]+[rng.integers(i) for i in range(1, n)])
    nominal = np.tile(np.eye(4), (n, 1, 1))
    for matrix in nominal:
        matrix[:3, :3] += rotation_increment(*rng.normal(0, .2, 3))
        matrix[:3, 3] = rng.normal(0, 2e4, 3)
    relations, static, dynamic = (np.zeros((n, 12)) for _ in range(3))
    paths, starts = [], [0]
    for node, parent in enumerate(parents):
        relations[node, :9] = np.eye(3).ravel()
        if parent >= 0:
            relations[node, :9] = (nominal[parent, :3, :3].T @ nominal[node, :3, :3]).ravel()
            relations[node, 9:] = nominal[parent, :3, :3].T @ (
                nominal[node, :3, 3]-nominal[parent, :3, 3])
        chain = [node]
        while parents[chain[-1]] >= 0:
            chain.append(parents[chain[-1]])
        paths.extend(reversed(chain))
        starts.append(len(paths))
    for data, scale in ((static, .01), (dynamic, 1e-10)):
        data[:, :3] = rng.normal(0, scale, (n, 3))
        for row in data:
            row[3:] = rotation_increment(*rng.normal(0, scale, 3)).ravel()
    actual_device = test_context.zeros(n*12, dtype=np.float64)
    test_context.kernels.mechanics_body_update(n=n,
        paths_offsets=to_context(np.array(starts, dtype='i8')),
        paths=to_context(np.array(paths, dtype='i8')),
        relations=to_context(relations.ravel()), static_local=to_context(static.ravel()),
        dynamic_local=to_context(dynamic.ravel()), out=actual_device)
    actual = to_numpy(actual_device).reshape(n, 12)
    expected = np.zeros_like(actual)
    def unpack(row):
        return row[:3], row[3:].reshape(3, 3)
    for node, parent in enumerate(parents):
        zero = (np.zeros(3), np.zeros((3, 3)))
        wi = inherit(unpack(expected[parent]), nominal[parent], nominal[node]) if parent >= 0 else zero
        d, e = unpack(dynamic[node])
        ls, le = unpack(static[node])
        wd, we = compose(wi, (ls+d, e+le+e@le))
        expected[node, :3], expected[node, 3:] = wd, we.ravel()
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=2e-14)

    # Lower random local deformations and verify both diagnostic rows and all
    # writes to an independently allocated, aligned scratch buffer.
    a = 11
    geometry, baseline = np.zeros((a, 24)), np.zeros((a, 7))
    for row in geometry:
        row[:9] = (np.eye(3)+rotation_increment(*rng.normal(0, .1, 3))).ravel()
        row[9:12] = rng.normal(0, 2, 3)
        row[12:15] = rng.normal(0, 1e-4, 3)
        row[15:] = (np.eye(3)+rotation_increment(*rng.normal(0, .1, 3))).ravel()
    actual[0] = 0.
    baseline[0] = rng.normal(size=7)
    values = test_context.zeros(a*7, dtype=np.float64)
    buffer = test_context.zeros(a*7*8, dtype=np.int8)
    test_context.kernels.mechanics_lower(n=a,
        body_indices=to_context(np.arange(a, dtype='i8')), geometry=to_context(geometry.ravel()),
        baseline=to_context(baseline.ravel()), offsets=to_context(np.arange(a*7, dtype='i8')*8),
        bodyout=to_context(actual.ravel()), buffer=buffer, values=values)
    expected_values = np.zeros((a, 7))
    for i in range(a):
        if not np.any(actual[i]):
            expected_values[i] = baseline[i]
            continue
        d, e = unpack(actual[i])
        g = geometry[i]
        axes, r0 = g[:9].reshape(3, 3), g[15:].reshape(3, 3)
        shift = g[12:15]+axes@(d+e@g[9:12])
        rotation = r0+(axes@e@axes.T)@r0
        psi = np.arctan2(rotation[1, 0], rotation[1, 1])
        theta = np.arctan2(rotation[0, 2], rotation[2, 2])
        phi = np.arctan2(rotation[1, 2], rotation[1, 1]/np.cos(psi))
        expected_values[i] = [*shift, phi, theta, psi, 0.]
    actual_values = to_numpy(values).reshape(a, 7)
    np.testing.assert_allclose(actual_values, expected_values, atol=2e-12, rtol=2e-14)
    np.testing.assert_array_equal(to_numpy(buffer).view('f8').reshape(a, 7), actual_values)
    np.testing.assert_array_equal(actual_values[0], baseline[0])


@for_all_test_contexts
@allow_kernel_compilation
def test_prepared_analytic_source_order_and_small_rotations(test_context):
    compile_kernels(test_context)
    to_context = test_context.nparray_to_context_array
    # A projected plane wave, two scalar translations, then ordered finite
    # pitch/yaw/roll and a picoradian pitch. Empty bodies remain exactly zero.
    rows = np.array([[-1, 1e-9, 12., .3, .125, .2, -.4, .8],
        [0, 2e-9, 3., .1, 0., 0., 0., 0.], [2, -4e-9, 7., -.2, 0., 0., 0., 0.],
        [3, .1, 5., .8, 0., 0., 0., 0.], [4, -.2, 1., .6, 0., 0., 0., 0.],
        [5, .3, 9., .1, 0., 0., 0., 0.], [3, 1e-12, 4., .5, 0., 0., 0., 0.]])
    starts = np.array([0, 3, 6, 7, 7], dtype='i8')
    times = (-.1, 0., 3.)
    for time_s in times:
        output = test_context.zeros(4*12, dtype=np.float64)
        test_context.kernels.mechanics_sources(n=4, source_starts=to_context(starts),
            source_data=to_context(rows.ravel()), time_s=time_s, dynamic_local=output)
        expected = np.zeros((4, 12))
        for body in range(4):
            for component, amplitude, omega, phase, delay, px, py, pz in rows[starts[body]:starts[body+1]]:
                component = int(component)
                value = amplitude*np.sin(omega*(time_s-delay)+phase)
                if component == -1:
                    expected[body, :3] += value*np.array([px, py, pz])
                elif component < 3:
                    expected[body, component] += value
                else:
                    angles = np.zeros(3)
                    angles[component-3] = value
                    left = rotation_increment(*angles)
                    right = expected[body, 3:].reshape(3, 3)
                    expected[body, 3:] = (left+right+left@right).ravel()
        actual = test_context.nparray_from_context_array(output).reshape(4, 12)
        np.testing.assert_allclose(actual, expected, atol=1e-25, rtol=2e-14)
        np.testing.assert_array_equal(actual[3], np.zeros(12))


@for_all_test_contexts
@allow_kernel_compilation
def test_prepared_lowering_preserves_rotation_near_pitch_lock(test_context):
    from xtrack import Frame
    compile_kernels(test_context)
    to_context = test_context.nparray_to_context_array
    pitches = np.array([np.pi/2, -np.pi/2, np.pi/2-1e-12,
                        -np.pi/2+1e-12, np.pi/2+1e-5, -np.pi/2-1e-5])
    n = len(pitches)
    body = np.zeros((n, 12))
    geometry = np.zeros((n, 24))
    expected = []
    for i, phi in enumerate(pitches):
        body[i, 3:] = rotation_increment(phi, .3, -.7).ravel()
        geometry[i, :9] = geometry[i, 15:] = np.eye(3).ravel()
        expected.append(Frame.from_survey_angles(phi=phi, theta=.3, psi=-.7).matrix[:3, :3])
    values = test_context.zeros(n*7, dtype=np.float64)
    buffer = test_context.zeros(n*7*8, dtype=np.int8)
    test_context.kernels.mechanics_lower(n=n,
        body_indices=to_context(np.arange(n, dtype='i8')),
        geometry=to_context(geometry.ravel()), baseline=to_context(np.zeros(n*7)),
        offsets=to_context(np.arange(n*7, dtype='i8')*8), bodyout=to_context(body.ravel()),
        buffer=buffer, values=values)
    actual = test_context.nparray_from_context_array(values).reshape(n, 7)
    for row, matrix in zip(actual, expected):
        rebuilt = Frame.from_survey_angles(phi=row[3], theta=row[4], psi=row[5]).matrix[:3, :3]
        np.testing.assert_allclose(rebuilt, matrix, atol=2e-15, rtol=0.)


@for_all_test_contexts
@allow_kernel_compilation
def test_prepared_general_source_rows_compose_in_declared_order(test_context):
    compile_kernels(test_context)
    to_context = test_context.nparray_to_context_array
    values = np.array([[1e-9, -2e-9, 3e-9, .3, -.2, .1],
        [2e-10, 1e-10, -3e-10, -.1, .2, -.3],
        [2e-10, -1e-9, 0., 1e-12, -2e-12, 3e-12],
        [0., 0., 0., np.pi/2, .3, -.7],
        [1e-13, 2e-13, -3e-13, 0., 0., 0.],
        [0., 0., 0., 1e-12, -2e-12, 3e-12]])
    starts = np.array([0, 3, 4, 5, 6, 6], dtype='i8')
    rows = np.array([2, 0, 1, 3, 4, 5], dtype='i8')
    output = test_context.zeros(5*12, dtype=np.float64)
    test_context.kernels.mechanics_source_values(n=5, source_starts=to_context(starts),
        source_rows=to_context(rows), values=to_context(values.ravel()), dynamic_local=output)
    expected = np.zeros((5, 12))
    for body in range(5):
        for row in rows[starts[body]:starts[body+1]]:
            expected[body, :3] += values[row, :3]
            left = rotation_increment(*values[row, 3:])
            right = expected[body, 3:].reshape(3, 3)
            expected[body, 3:] = (left+right+left@right).ravel()
    actual = test_context.nparray_from_context_array(output).reshape(5, 12)
    np.testing.assert_allclose(actual[:, :3], expected[:, :3], atol=1e-25, rtol=2e-14)
    # Finite rotations near pitch lock contain cancellation-sized entries;
    # test their absolute matrix error rather than relative error to zero.
    np.testing.assert_allclose(actual[:, 3:], expected[:, 3:], atol=5e-16, rtol=2e-14)
    np.testing.assert_allclose(actual[3, 3:], expected[3, 3:], atol=1e-35, rtol=2e-14)
    np.testing.assert_array_equal(actual[4], np.zeros(12))
