"""Bounded host-array uploads with explicit host staging ownership."""
import numpy as np
import xobjects as xo


def to_context(context, value):
    """Finish consuming a reusable host array before returning to its owner.

    CuPy's default nonblocking array construction can race the next host write.
    One explicit blocking batch transfer is the reference sampled-input policy;
    analytic motion has no per-freeze host payload. OpenCL's Xobjects upload
    uses its blocking to_device default, and CPU execution is synchronous.
    """
    value = np.ascontiguousarray(value)
    if isinstance(context, xo.ContextCupy):
        return context.nplike_lib.array(value, blocking=True)
    return context.nparray_to_context_array(value)
