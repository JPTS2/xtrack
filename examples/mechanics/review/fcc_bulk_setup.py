"""Measure public bulk support preparation on the unmodified FCC dev input.

One provisional support per quadrupole isolates setup scaling. This is not an
engineering girder map. No lattice expressions are removed or input files saved.
"""
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import xtrack as xt


ROOT = Path(__file__).resolve().parents[3]
LATTICE = ROOT.parent/'inputs/acc-models-fcc-ee-108.1.0/lattices/z/fccee_z_beam_1_positrons.json'
OUTPUT = Path(__file__).resolve().parent/'results/fcc_bulk_setup'


def timed(function):
    start = perf_counter()
    value = function()
    return value, perf_counter()-start


def main():
    env, load_s = timed(lambda: xt.load(LATTICE))
    line = env.fccee_p_ring
    table, inspect_s = timed(line.get_table)
    names = list(table.rows.match(element_type='Quadrupole').name)
    original_lengths = [float(line[n].length) for n in names]
    original_strengths = [float(line[n].k1) for n in names]
    original_exprs = [env.ref[n].length._expr for n in names]
    _, adopt_s = timed(line.mechanics.adopt_reference)
    bodies, definition_s = timed(lambda: line.mechanics.new_girders(
        {f'g.{name}': [name] for name in names}, support_class='provisional_quad_site'))
    _, validation_s = timed(line.mechanics.validate)
    assert len(bodies) == len(names) == 2663
    assert original_lengths == [float(line[n].length) for n in names]
    assert original_strengths == [float(line[n].k1) for n in names]
    assert all(before is env.ref[n].length._expr for n,before in zip(names, original_exprs))
    metadata = dict(lattice_sha256=hashlib.sha256(LATTICE.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        mechanics_source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted((ROOT/'xtrack/mechanics').glob('*.py'))},
        platform=platform.platform(), xtrack=xt.__version__,
        public_api=True, detached_numeric_snapshot=False, engineering_assignment=False,
        n_supports=len(bodies), n_members=len(names),
        n_expression_lengths=sum(expr is not None for expr in original_exprs),
        load_s=load_s, table_inspection_s=inspect_s, reference_adoption_s=adopt_s,
        bulk_definition_s=definition_s, explicit_validation_s=validation_s,
        live_geometry_and_strengths_preserved=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT/'result.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
