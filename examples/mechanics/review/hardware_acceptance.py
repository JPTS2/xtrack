"""Run mechanics conformance on one explicitly selected real context.

    PYTHONPATH=. python examples/mechanics/review/hardware_acceptance.py \
        --context cuda --output /tmp/mechanics-cuda

This reuses native regression tests rather than a second implementation. It
never falls back to CPU. Exit codes: 0 passed, 1 failed, 2 context unavailable,
3 incomplete (including skipped tests). Kernel compilation must be enabled in
the normal Xsuite way on a source checkout. Throughput is a separate FCC command
in ``fcc_prepared_runtime.py``; a small smoke test cannot establish a 2x budget.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
import traceback
import xml.etree.ElementTree as ET

import numpy as np
import xobjects as xo
from xobjects.context import get_context_from_string
import xtrack as xt


ROOT = Path(__file__).resolve().parents[3]
TESTS = (
    'tests/test_mechanics_runtime_kernels.py',
    'tests/test_mechanics_runtime_frames.py',
    'tests/test_mechanics_runtime.py::test_prepared_matches_finite_hierarchy',
    'tests/test_mechanics_runtime.py::test_tracking_and_split_calls_keep_same_clock',
    'tests/test_mechanics_runtime.py::test_prepared_tiny_motion_and_exact_zero',
    'tests/test_mechanics_runtime.py::test_prepared_record_and_world_source_parity',
    'tests/test_mechanics_runtime.py::test_fixed_reference_clock_and_monitor_match_reference_scheduler',
    'tests/test_mechanics_geometry_guard.py::test_guard_accepts_strength_changes_and_tracks_current_alignment',
    'tests/test_mechanics_geometry_guard.py::test_guard_catches_geometry_and_alignment_changes',
    'tests/test_mechanics_geometry_guard.py::test_guard_rejects_new_equal_value_alignment_expression',
    'tests/test_mechanics_geometry_guard.py::test_guard_rejects_object_replacement_and_added_alias',
)
# Deliberately fixed for this acceptance selection and one requested context.
# Update this with TESTS/parametrization changes; a partial run cannot pass.
EXPECTED_TEST_COUNT = 19


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context', choices=('cpu', 'openmp', 'cuda', 'opencl'), required=True)
    parser.add_argument('--device', help='CUDA index or OpenCL platform.device; no automatic CPU fallback')
    parser.add_argument('--threads', type=int, default=2, help='OpenMP threads')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.threads <= 0 or (args.device is not None and args.context in ('cpu', 'openmp')):
        parser.error('Use positive threads; --device applies only to CUDA/OpenCL')
    context_name = {'cpu': 'ContextCpu', 'openmp': f'ContextCpu:{args.threads}',
                    'cuda': 'ContextCupy', 'opencl': 'ContextPyopencl'}[args.context]
    if args.device is not None:
        context_name += ':' + args.device
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_paths = sorted((ROOT/'xtrack/mechanics').rglob('*.py'))
    source_paths += sorted((ROOT/'xtrack/mechanics').rglob('*.h'))
    source_paths += [ROOT/'xtrack'/name for name in
                     ('__init__.py', 'environment.py', 'line.py', 'tracker.py')]
    source_paths += [Path(__file__).resolve()]
    source_paths += sorted({ROOT/test.split('::')[0] for test in TESTS})
    result = dict(requested_context=context_name, status='not_run',
        python=sys.version, platform=platform.platform(), xtrack=xt.__version__,
        xobjects=xo.__version__, numpy=np.__version__,
        imported_xtrack=str(Path(xt.__file__).resolve()),
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in source_paths},
        environment={key: os.environ.get(key) for key in (
            'CC', 'LDSHARED', 'XSUITE_ALLOW_KERNEL_COMPILATION',
            'XSUITE_CFFI_FORBID_COMPILE', 'OMP_NUM_THREADS',
            'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'PYTEST_ADDOPTS')},
        expected_test_count=EXPECTED_TEST_COUNT,
        throughput_validated=False)
    started = perf_counter()
    code = None
    if Path(xt.__file__).resolve() != (ROOT/'xtrack/__init__.py').resolve():
        result.update(status='incomplete', stage='source_import',
            reason='Imported another Xtrack installation; rerun from this checkout with PYTHONPATH=.')
        code = 3
    else:
        stage = 'context_dependency'
        try:
            # ContextCupy can construct lazily even when CuPy is absent.
            # Establish dependency/device availability before testing an actual
            # allocation, so allocation bugs cannot masquerade as missing GPUs.
            if args.context == 'cuda':
                cupy = importlib.import_module('cupy')
                result['context_dependency_version'] = cupy.__version__
                stage = 'device_discovery'
                if cupy.cuda.runtime.getDeviceCount() == 0:
                    raise RuntimeError('No CUDA devices found')
            elif args.context == 'opencl':
                opencl = importlib.import_module('pyopencl')
                result['context_dependency_version'] = opencl.VERSION_TEXT
            stage = 'context_construction'
            context = get_context_from_string(context_name)
        except Exception as exc:
            result.update(status='unavailable', stage=stage,
                          reason=f'{type(exc).__name__}: {exc}')
            (output/'preflight.txt').write_text(traceback.format_exc())
            code = 2
    if code is None:
        # Actual allocation, transfer and synchronization, not an import check.
        try:
            expected = np.array([1., -2., np.pi], dtype=np.float64)
            actual = context.nparray_from_context_array(context.nparray_to_context_array(expected))
            context.synchronize()
            np.testing.assert_array_equal(actual, expected)
            result['actual_context'] = str(context)
            if args.context == 'cuda':
                device = context.nplike_lib.cuda.Device()
                properties = context.nplike_lib.cuda.runtime.getDeviceProperties(device.id)
                name = properties['name']
                result['device_name'] = name.decode() if isinstance(name, bytes) else str(name)
            elif args.context == 'opencl':
                result['device_name'] = context.device.name
                result['device_platform'] = context.platform.name
        except Exception as exc:
            result.update(status='failed', stage='device_allocation_and_transfer',
                          reason=f'{type(exc).__name__}: {exc}')
            (output/'preflight.txt').write_text(traceback.format_exc())
            code = 1
    if code is None:
        junit = output/'pytest.xml'
        command = [sys.executable, '-m', 'pytest', '-q', '--tb=short', '-o', 'addopts=',
                   '--junitxml', str(junit), *TESTS]
        env = dict(os.environ, XOBJECTS_TEST_CONTEXTS=context_name)
        # Acceptance must not inherit an accidental -k/-m/--deselect filter.
        env.pop('PYTEST_ADDOPTS', None)
        env['PYTHONPATH'] = str(ROOT)+os.pathsep+env.get('PYTHONPATH', '')
        result.update(command=command, test_context_environment=context_name)
        with (output/'tests.txt').open('w') as log:
            process = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                     stderr=subprocess.STDOUT)
        counts = dict(tests=0, failures=0, errors=0, skipped=0)
        if junit.exists():
            for suite in ET.parse(junit).getroot().iter('testsuite'):
                for key in counts:
                    counts[key] += int(suite.get(key, 0))
        result.update(pytest_exit_code=process.returncode, counts=counts)
        if process.returncode or counts['failures'] or counts['errors']:
            result['status'], code = 'failed', 1
        elif counts['tests'] != EXPECTED_TEST_COUNT or counts['skipped']:
            result['status'], code = 'incomplete', 3
        else:
            result['status'], code = 'passed', 0
    changed = [str(p.relative_to(ROOT)) for p in source_paths
               if hashlib.sha256(p.read_bytes()).hexdigest()
               != result['source_sha256'][str(p.relative_to(ROOT))]]
    if changed:
        result['source_changed_during_run'] = changed
        if code == 0:
            result['status'], code = 'incomplete', 3
    result['elapsed_s'] = perf_counter()-started
    result['exit_code'] = code
    (output/'acceptance.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key: value for key, value in result.items()
                      if key != 'source_sha256'}, indent=2))
    raise SystemExit(code)


if __name__ == '__main__':
    main()
