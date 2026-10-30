"""Context-local developer JIT, with a reversible conda linker workaround."""
from contextlib import contextmanager
import os
from pathlib import Path
import shlex
import sys
import sysconfig

import xobjects as xo


@contextmanager
def cpu_context(directory):
    """Keep builds local; remove only duplicate identical conda rpath tokens.

    No installed environment or configuration file is changed. Ordinary
    compiler configurations pass through unchanged, and process-local flags
    are restored on exit.
    """
    directory=Path(directory).resolve()
    directory.mkdir(parents=True,exist_ok=True)
    config=sysconfig.get_config_vars()
    names=('LDSHARED','LDCXXSHARED','BLDSHARED','LDFLAGS')
    previous={name:config.get(name) for name in names}
    token='-Wl,-rpath,'+str(Path(sys.prefix)/'lib')
    old_cwd=Path.cwd()
    try:
        for name,value in previous.items():
            if not isinstance(value,str):continue
            tokens=shlex.split(value)
            if tokens.count(token)>1:
                seen=False;clean=[]
                for part in tokens:
                    if part==token:
                        if seen:continue
                        seen=True
                    clean.append(part)
                config[name]=shlex.join(clean)
        os.chdir(directory)
        context=xo.ContextCpu(omp_num_threads=0)
        context.allow_kernel_compilation=True
        context.allow_prebuilt_kernels=False
        yield context
    finally:
        os.chdir(old_cwd)
        config.update(previous)
