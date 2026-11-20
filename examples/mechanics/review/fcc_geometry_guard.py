"""Entry-validation cost on the original FCC model, separately from tracking."""
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import xtrack as xt
from xtrack.mechanics.geometry_guard import GeometryGuard

ROOT = Path(__file__).resolve().parents[3]
LATTICE = ROOT.parent/'inputs/acc-models-fcc-ee-108.1.0/lattices/z/fccee_z_beam_1_positrons.json'
OUTPUT = Path(__file__).resolve().parent/'results/fcc_geometry_guard'


def measure(call, repeats=1):
    times = []
    for _ in range(repeats):
        start = perf_counter()
        result = call()
        times.append(perf_counter()-start)
    return result, dict(median_s=float(np.median(times)), samples_s=times)


def main():
    env, load = measure(lambda:xt.load(LATTICE))
    line = env.fccee_p_ring
    _, build = measure(line.build_tracker)
    names = list(line.get_table().rows.match(element_type='Quadrupole').name)
    _, definition = measure(lambda:line.mechanics.new_girders(
        {f'g.{name}':[name] for name in names}, support_class='provisional_quad_site'))
    _, full = measure(line.mechanics.validate, repeats=3)
    guard, prepare = measure(lambda:GeometryGuard(line.mechanics))
    _, packed = measure(guard.check, repeats=7)
    buffer_type = type(guard.buffer)
    original = buffer_type.to_bytearray
    scalar_reads = 0
    def counted(*args, **kwargs):
        nonlocal scalar_reads
        scalar_reads += 1
        return original(*args, **kwargs)
    try:
        buffer_type.to_bytearray = counted
        guard.check()
    finally:
        buffer_type.to_bytearray = original
    assert scalar_reads == 0
    record = dict(lattice_sha256=hashlib.sha256(LATTICE.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        mechanics_source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted((ROOT/'xtrack/mechanics').glob('*.py'))},
        n_supports=len(names), n_occurrences=len(line.element_names),
        engineering_assignment=False, detached_numeric_snapshot=False,
        load=load, tracker_build=build, public_bulk_definition=definition,
        full_validation=full, guard_preparation=prepare, packed_entry_validation=packed,
        protected_geometry_storage_words=guard.n_geometry,
        total_protected_storage_words=guard.n_storage_words,
        gathered_bytes_per_entry=guard.n_storage_words*8,
        scalar_buffer_reads_per_entry=scalar_reads,
        context_to_host_arrays_per_entry=1,
        cpu_only=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT/'result.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__ == '__main__':
    main()
