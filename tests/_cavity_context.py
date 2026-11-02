"""Shared CPU context for the physical-model Cavity tests."""
from contextlib import contextmanager
import os

import xobjects as xo


@contextmanager
def native_context(directory):
    """Compile locally in `directory`; restore the working directory."""
    old_cwd = os.getcwd()
    os.chdir(directory)
    try:
        context = xo.ContextCpu(omp_num_threads=0)
        context.allow_kernel_compilation = True
        context.allow_prebuilt_kernels = False
        yield context
    finally:
        os.chdir(old_cwd)
