"""Reusable context payloads must not allocate temporary device arrays."""

import numpy as np
import pytest
import xobjects as xo

from xtrack.mechanics.context_arrays import upload_into


def test_cpu_inplace_upload_keeps_destination_and_consumes_noncontiguous_input():
    context = xo.ContextCpu()
    destination = context.zeros(12, dtype=np.float64)
    staging = np.arange(24.0, dtype=np.float64)[::2]
    expected = staging.copy()
    upload_into(context, destination, staging)
    staging[:] = -1.0
    np.testing.assert_array_equal(destination, expected)
    with pytest.raises(ValueError, match='shape and dtype'):
        upload_into(context, destination, np.zeros(12, dtype=np.int64))


def test_cupy_inplace_upload_waits_for_current_stream_without_device_temporary():
    class Stream:
        def __init__(self):
            self.pending = None
            self.waits = 0

        def synchronize(self):
            self.waits += 1
            self.pending()

    class FakeContext(xo.ContextCupy):
        def __init__(self):
            self.stream = Stream()

        @property
        def nplike_lib(self):
            return self

        @property
        def cuda(self):
            return self

        def get_current_stream(self):
            return self.stream

        def nparray_to_context_array(self, value, **kwargs):
            raise AssertionError('A device temporary was allocated')

    class Destination:
        shape = (6,)
        dtype = np.dtype('float64')

        def __init__(self):
            self.values = np.zeros(6)

        def set(self, values, *, stream):
            stream.pending = lambda: np.copyto(self.values, values)

    context, destination = FakeContext(), Destination()
    staging = np.arange(6.0)
    upload_into(context, destination, staging)
    staging[:] = -1.0
    assert context.stream.waits == 1
    np.testing.assert_array_equal(destination.values, np.arange(6.0))


def test_opencl_inplace_upload_explicitly_consumes_host_staging():
    class FakeContext(xo.ContextPyopencl):
        def __init__(self):
            self.queue = object()

        def nparray_to_context_array(self, value, **kwargs):
            raise AssertionError('A device temporary was allocated')

    class Destination:
        shape = (6,)
        dtype = np.dtype('float64')

        def __init__(self):
            self.values = np.zeros(6)

        def set(self, values, *, queue, async_):
            assert queue is context.queue
            assert async_ is False
            self.values[:] = values

    context, destination = FakeContext(), Destination()
    staging = np.arange(6.0)
    upload_into(context, destination, staging)
    staging[:] = -1.0
    np.testing.assert_array_equal(destination.values, np.arange(6.0))
