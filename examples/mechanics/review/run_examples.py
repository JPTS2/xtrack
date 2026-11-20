"""Capture each lesson in a fresh process; does not change the lesson interface.

    python examples/mechanics/review/run_examples.py --examples 001 006

Results contain terminal.txt, stderr.txt, figures, timings and source hashes.
Run sequentially when comparing timing results. Kernel compilation and plotting
are distinct from each lesson's explicitly instrumented steady costs.
"""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import runpy
import shutil
import subprocess
import sys
from time import perf_counter


HERE = Path(__file__).resolve().parent
EXAMPLES = HERE.parent
ROOT = EXAMPLES.parents[1]


def worker(script, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import xtrack as xt
    destination.mkdir(parents=True, exist_ok=True)
    figures = []

    def capture(*args, **kwargs):
        for number in plt.get_fignums():
            figure = plt.figure(number)
            stem = f'figure_{len(figures)+1:02d}'
            figure.savefig(destination / (stem+'.png'), dpi=140)
            svg = destination / (stem+'.svg')
            figure.savefig(svg)
            svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
            figures.append(stem)
        plt.close('all')

    plt.show = capture
    started = perf_counter()
    with (destination/'terminal.txt').open('w') as stdout, (destination/'stderr.txt').open('w') as stderr:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            scope = runpy.run_path(str(script), run_name='__main__')
            if plt.get_fignums():
                capture()
    artifacts = []
    if scope.get('SAVE_GIF') and scope.get('GIF_FILE'):
        gif = Path(scope['GIF_FILE'])
        shutil.copy2(gif, destination/'motion.gif')
        artifacts.append('motion.gif')
    metadata = dict(example=script.name, elapsed_s=perf_counter()-started,
        timings=scope.get('timings', {}), figures=figures, python=sys.version,
        artifacts=artifacts,
        xtrack=xt.__version__, numpy=np.__version__, platform=platform.platform(),
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/
            (1024**2 if sys.platform=='darwin' else 1024),
        script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
        mechanics_source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT/'xtrack/mechanics').glob('*.py'))})
    (destination/'result.json').write_text(json.dumps(metadata, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--examples', nargs='+', default=None)
    parser.add_argument('--output', type=Path, default=HERE/'results')
    parser.add_argument('--worker', type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker.resolve(), args.output.resolve())
        return
    scripts = sorted(EXAMPLES.glob('[0-9][0-9][0-9]_*.py'))
    if args.examples:
        scripts = [p for p in scripts if p.name[:3] in args.examples]
        if len(scripts) != len(args.examples):
            parser.error('Select existing three-digit lesson numbers')
    env = dict(os.environ, MPLBACKEND='Agg')
    env['PYTHONPATH'] = str(ROOT)+os.pathsep+env.get('PYTHONPATH','')
    results = []
    for script in scripts:
        print(f'Running {script.name}', flush=True)
        destination = args.output.resolve()/script.stem
        process = subprocess.run([sys.executable, str(Path(__file__).resolve()),
            '--worker', str(script), '--output', str(destination)], cwd=ROOT, env=env,
            capture_output=True, text=True)
        result = dict(example=script.name, exit_code=process.returncode)
        if process.returncode:
            destination.mkdir(parents=True,exist_ok=True)
            (destination/'worker_error.txt').write_text(process.stdout+process.stderr)
            print(process.stderr, flush=True)
        else:
            result.update(json.loads((destination/'result.json').read_text()))
            print(f'  {result["elapsed_s"]:.2f} s; {len(result["figures"])} figure(s)',flush=True)
        results.append(result)
    args.output.mkdir(parents=True,exist_ok=True)
    if args.examples and (args.output/'manifest.json').exists():
        previous = {r['example']:r for r in json.loads((args.output/'manifest.json').read_text())}
        previous.update({r['example']:r for r in results})
        results = [previous[name] for name in sorted(previous)]
    (args.output/'manifest.json').write_text(json.dumps(results,indent=2)+'\n')
    if any(r['exit_code'] for r in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
