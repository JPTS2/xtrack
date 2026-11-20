"""Host staging ownership at the context upload boundary."""
import numpy as np
import xobjects as xo

from xtrack.mechanics.context_arrays import to_context


class FakeCupyContext(xo.ContextCupy):
    """Exercise the upload policy without pretending to execute CUDA."""
    def __init__(self):
        self.blocking_arguments = []

    @property
    def nplike_lib(self):
        return self

    def array(self, value, *, blocking=False):
        self.blocking_arguments.append(blocking)
        # A nonblocking implementation may still read the caller's staging
        # array later. The helper must explicitly request finished consumption.
        return value.copy() if blocking else value

    def nparray_to_context_array(self, value, copy=False):
        raise AssertionError('The generic CuPy upload has an unsafe nonblocking default')


def test_cupy_upload_explicitly_finishes_consuming_reusable_staging():
    context = FakeCupyContext()
    staging = np.arange(24., dtype=np.float64).reshape(4, 6)[:, ::2]
    expected = staging.copy()
    uploaded = to_context(context, staging)
    staging[:] = -1.
    assert context.blocking_arguments == [True]
    assert uploaded.flags.c_contiguous
    np.testing.assert_array_equal(uploaded, expected)


def test_cpu_upload_keeps_dtype_shape_and_contiguous_values():
    values = np.arange(30, dtype=np.int64).reshape(5, 6)[:, ::2]
    uploaded = to_context(xo.ContextCpu(), values)
    assert uploaded.flags.c_contiguous
    assert uploaded.dtype == values.dtype
    np.testing.assert_array_equal(uploaded, values)
