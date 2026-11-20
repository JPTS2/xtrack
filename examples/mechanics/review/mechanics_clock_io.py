"""Count particle-clock scalar I/O in otherwise prepared turn-held tracking.

CPU buffer method counts establish access structure, not measured CUDA latency.
No mechanics/geometry setup is inside the instrumented region.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import xtrack as xt


def main():
    line = xt.Line(elements=[xt.Quadrupole(length=.5, k1=.1), xt.Drift(length=100.)])
    line.set_particle_ref('proton', p0c=7e12)
    line.build_tracker()
    line.mechanics.new_girder('g', elements=[line.element_names[0]])
    line.env.mechanics.add_source('h', xt.mechanics.Harmonic(1e-9, 3.), targets='g')
    line.mechanics.enable_vibrations(backend='compiled')
    line.track(line.build_particles(), num_turns=1)
    measurements = []
    for turns in (1, 4, 16):
        particles = line.build_particles(reference_turns=0.)
        target = particles._buffer
        buffer_type = type(target)
        read, write = buffer_type.to_bytearray, buffer_type.update_from_buffer
        names = {particles._xobject._get_offset(name): name
                 for name in ('time_s', 't_sim', 'reference_turns')}
        names[particles._xobject.beta0._get_offset(0)] = 'beta0[0]'
        counts = Counter()
        def counted_read(buffer, offset, nbytes):
            if buffer is target:
                counts['read:'+names.get(offset, 'other')] += 1
                counts['read_bytes'] += nbytes
            return read(buffer, offset, nbytes)
        def counted_write(buffer, offset, source):
            if buffer is target:
                counts['write:'+names.get(offset, 'other')] += 1
                counts['write_bytes'] += len(source)
            return write(buffer, offset, source)
        try:
            buffer_type.to_bytearray, buffer_type.update_from_buffer = counted_read, counted_write
            line.track(particles, num_turns=turns)
        finally:
            buffer_type.to_bytearray, buffer_type.update_from_buffer = read, write
        assert np.all(particles.state > 0)
        measurements.append(dict(turns=turns, counts=dict(counts)))
    result = dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        xtrack_version=xt.__version__, context='serial CPU instrumentation',
        meaning='Host accesses to the particle buffer during prepared mechanics tracking; no GPU timing',
        measurements=measurements)
    output = Path(__file__).parent/'results/fcc_freeze'
    output.mkdir(parents=True, exist_ok=True)
    (output/'clock_io.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
