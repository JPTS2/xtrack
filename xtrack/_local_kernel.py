"""Local compilation for element classes whose layout differs from prebuilt ABI."""
from contextlib import contextmanager
from pathlib import Path
import shlex
import sys
import sysconfig
import threading


_compile_lock = threading.RLock()


# Same-version installed kernels predate these checkout data/entry contracts.
# Keep the development ABI boundary here, out of physical element definitions.
_LOCAL_ABI_CLASSES = {
    ('xtrack.beam_elements.cavity', 'Cavity'),
    ('xtrack.beam_elements.slice_elements_thin', 'ThinSliceCavity'),
    ('xtrack.beam_elements.slice_elements_thick', 'ThickSliceCavity'),
    ('xtrack.beam_elements.slice_elements_drift', 'DriftSliceCavity'),
    ('xtrack.tpsa.particles', 'TpsaParticleData'),
    ('xtrack.beam_elements.reference_energy_increase', 'ReferenceEnergyIncrease'),
    ('xtrack.beam_elements.reference_energy_change', 'ReferenceEnergyChange'),
}


def prebuilt_compatible(element_class):
    """A source/layout change must explicitly opt out of installed binaries."""
    owner = getattr(element_class, '_DressingClass', element_class)
    if any((base.__module__, base.__name__) in _LOCAL_ABI_CLASSES
           for base in getattr(owner, '__mro__', (owner,))):
        return False
    return getattr(owner, '_prebuilt_kernel_compatible', True)


@contextmanager
def local_kernel_compilation(classes, context=None):
    """Repair duplicate identical conda rpaths only for changed local classes.

    Some conda Python builds repeat their own library rpath. macOS rejects
    the duplicate when linking a new extension. No configuration file or
    installed package is changed. The transient process-wide sysconfig repair
    is serialized across these scopes and restored even after compile failure.
    """
    if all(prebuilt_compatible(cls) for cls in classes):
        yield
        return
    with _compile_lock:
        config = sysconfig.get_config_vars()
        keys = ('LDSHARED', 'LDCXXSHARED', 'BLDSHARED', 'LDFLAGS')
        present = {key for key in keys if key in config}
        previous = {key: config.get(key) for key in keys}
        token = '-Wl,-rpath,' + str(Path(sys.prefix) / 'lib')
        previous_prebuilt = (context.allow_prebuilt_kernels
                             if context is not None else None)
        try:
            # HybridClass delegates to Struct.compile_class_kernels, whose
            # independent lookup has no per-class optout. Scope its policy to
            # this compilation only; restore the caller's context afterwards.
            if context is not None:
                context.allow_prebuilt_kernels = False
            for key, value in previous.items():
                if not isinstance(value, str):
                    continue
                parts = shlex.split(value)
                if parts.count(token) > 1:
                    seen = False
                    clean = []
                    for part in parts:
                        if part == token:
                            if seen:
                                continue
                            seen = True
                        clean.append(part)
                    config[key] = shlex.join(clean)
            yield
        finally:
            if context is not None:
                context.allow_prebuilt_kernels = previous_prebuilt
            for key in keys:
                if key in present:
                    config[key] = previous[key]
                else:
                    config.pop(key, None)
