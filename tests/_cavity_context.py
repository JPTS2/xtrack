"""Shared CPU context for the physical-model Cavity tests."""
from contextlib import contextmanager
import os

import xobjects as xo


@contextmanager
def kernel_context(directory):
    """Compile in `directory`, so build files stay out of the checkout."""
    old_cwd = os.getcwd()
    os.chdir(directory)
    try:
        yield xo.ContextCpu(omp_num_threads=0)
    finally:
        os.chdir(old_cwd)
