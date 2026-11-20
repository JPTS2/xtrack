"""Project the exploratory branch into a reproducible local draft patch.

No branch, index, history or remote is changed. The selected implementation,
tests and teaching examples are taken verbatim from a clean committed state.
Historical review captures and design studies stay on the exploration branch.
"""
import argparse
import fnmatch
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile


ROOT = Path(__file__).resolve().parents[4]
PATTERNS = (
    'xtrack/mechanics/*',
    'xtrack/__init__.py',
    'xtrack/environment.py',
    'xtrack/line.py',
    'xtrack/tracker.py',
    'tests/test_mechanics*.py',
    'test_data/example_lattices/demo_ring/*',
    'examples/mechanics/.gitignore',
    'examples/mechanics/[0-9][0-9][0-9]_*.py',
    'examples/mechanics/README.md',
    'examples/mechanics/API.md',
    'examples/mechanics/DELIVERY.md',
    'examples/mechanics/review/HARDWARE_ACCEPTANCE.md',
    'examples/mechanics/review/FCC_INPUT.md',
    'examples/mechanics/review/hardware_acceptance.py',
    'examples/mechanics/review/fcc_prepared_runtime.py',
    'examples/mechanics/review/run_examples.py',
)


def git(*args, cwd=ROOT):
    return subprocess.check_output(['git', *args], cwd=cwd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='main', help='Resolve its merge base with HEAD')
    parser.add_argument('--output', type=Path, required=True, help='New directory outside the checkout')
    parser.add_argument('--materialize', action='store_true', help='Replay the patch onto an archived baseline')
    args = parser.parse_args()
    output = args.output.resolve()
    if output == ROOT or ROOT in output.parents:
        parser.error('Choose an output directory outside the source checkout')
    if output.exists():
        parser.error('Choose a new output directory to preserve previous evidence')
    base = git('merge-base', args.base, 'HEAD').decode().strip()
    tracked = git('ls-files', '-z').decode().split('\0')
    baseline_paths = git('ls-tree', '-r', '--name-only', '-z', base).decode().split('\0')
    paths = sorted(path for path in set(tracked + baseline_paths) if path and any(
        fnmatch.fnmatchcase(path, pattern) for pattern in PATTERNS))
    untracked = git('ls-files', '--others', '--exclude-standard', '-z').decode().split('\0')
    missing = [path for path in untracked if path and any(
        fnmatch.fnmatchcase(path, pattern) for pattern in PATTERNS)]
    dirty = git('status', '--porcelain', '--', *paths).decode()
    if missing or dirty:
        parser.error('Commit the selected delivery files first; projection never omits pending source edits')
    head = git('rev-parse', 'HEAD').decode().strip()
    patch = git('diff', '--binary', '--full-index', base, head, '--', *paths)
    if not patch:
        parser.error('No selected changes relative to the resolved baseline')
    changed = git('diff', '--name-only', base, head, '--', *paths).decode().splitlines()
    output.mkdir(parents=True)
    patch_file = output/'draft.patch'
    patch_file.write_bytes(patch)
    manifest = dict(base_commit=base, source_commit=head,
        source_branch=git('branch', '--show-current').decode().strip(),
        projection='verbatim selected files; no history rewrite or publication',
        patch_sha256=hashlib.sha256(patch).hexdigest(),
        selection_patterns=PATTERNS,
        files={name: (dict(bytes=(ROOT/name).stat().st_size,
                         sha256=hashlib.sha256((ROOT/name).read_bytes()).hexdigest())
                      if (ROOT/name).exists() else dict(deleted=True))
               for name in changed},
        diff_stat=git('diff', '--stat', base, head, '--', *paths).decode())
    if args.materialize:
        checkout = output/'checkout'
        checkout.mkdir()
        process = subprocess.Popen(['git', 'archive', '--format=tar', base],
                                   cwd=ROOT, stdout=subprocess.PIPE)
        with tarfile.open(fileobj=process.stdout, mode='r|') as archive:
            archive.extractall(checkout, filter='data')
        process.stdout.close()
        if process.wait():
            raise RuntimeError('Baseline archive failed')
        subprocess.run(['git', 'apply', '--check', str(patch_file)], cwd=checkout, check=True)
        subprocess.run(['git', 'apply', str(patch_file)], cwd=checkout, check=True)
        mismatch = []
        for name, info in manifest['files'].items():
            path = checkout/name
            if info.get('deleted'):
                if path.exists():
                    mismatch.append(name)
            elif not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != info['sha256']:
                mismatch.append(name)
        if mismatch:
            raise RuntimeError(f'Projected files differ: {mismatch}')
        manifest['replay'] = dict(status='passed', checkout=str(checkout),
                                selected_files_match=True)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(manifest['diff_stat'])
    print(f'Patch: {patch_file}')
    print(f'Manifest: {output / "manifest.json"}')


if __name__ == '__main__':
    main()
