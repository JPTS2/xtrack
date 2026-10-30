"""Track a prepared finite Gaussian RF scalar with Xtrack alone.

    python examples/cavity_models/005_prepared_envelope.py --tpsa

The accompanying numeric input was prepared offline from the retained Gaussian
RF²/transverse² scalar, degree four/Gauss sixteen. Its source hashes, physical
hardware and approximation OPENs are inside the gzip JSON. Tracking performs
neither preparation nor imports from cavity-modelling. This is a different
physical/numerical recipe from phase-averaged-magnetic, not its Gaussian alias.

The coefficients fix frequency, carrier ratio, width, centre and physical
anchor. Voltage, eta, oscillator phase, counterphase and calibration beta may
vary. To change coefficient-defining hardware, prepare and attach a new table.
The energy rectangle covers actual evolving total energy/(actual mass*c²).
Nonzero-field observation planes use native canonical optics. No uniform
interpolation, omitted-order or release qualification follows from this case.
"""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import xtrack as xt

try:
    from ._context import cpu_context
except ImportError:
    from _context import cpu_context


COORDS = ('x', 'px', 'y', 'py', 'zeta', 'delta')


def values(particles):
    return np.array([float(getattr(particles, name)[0]) for name in COORDS])


def make_line(context, steps=8):
    path = Path(__file__).with_suffix('.json.gz')
    with gzip.open(path, 'rt') as stream:
        record = json.load(stream)
    table = xt.EnvelopeTableData.from_dict(record['table'])
    k = table.normalization_wave_number
    rf = xt.Cavity(_context=context, model='phase-averaged-envelope',
        envelope_table=table, integrator='midpoint',
        frequency=table.normalization_frequency,
        length=(table.position_max-table.position_min)/k,
        rf_profile='gaussian', rf_profile_sigma=table.profile_width/k,
        rf_profile_center=table.profile_centre/k,
        wave_number=table.nu*k, voltage=2e3, eta=.25,
        phase=.4, counter_phase=.3, calibration_beta=.9,
        num_kicks=steps, num_factor_steps=2)
    line = xt.Line(elements=[rf, xt.ReferenceEnergyIncrease(Delta_p0c=0.),
                             xt.Marker()],
                   element_names=['rf', 'reference_update', 'end'])
    line.particle_ref = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1.,
        p0c=xt.ELECTRON_MASS_EV*np.sqrt(2.4**2-1.))
    line.use_time_based_zeta_update = True
    line.reset_s_at_end_turn = False
    for name, field, value in [('rf_phase', 'phase', .4), ('rf_eta', 'eta', .25)]:
        line.vars[name] = value
        setattr(line.element_refs['rf'], field, line.vars[name])
    line.build_tracker(_context=context)
    return line, record['provenance']


def run(line, provenance, with_tpsa=False):
    initial = line.particle_ref.copy()
    for name, value in dict(x=3e-4, px=2e-4, y=-2e-4, py=-1e-4,
                            zeta=1e-4, delta=.002).items():
        setattr(initial, name, value)
    tracked = initial.copy()
    line.track(tracked, ele_start='rf', ele_stop='reference_update')
    if tracked.state[0] != 1:
        raise RuntimeError('Prepared map refused: '+str(tracked.state[0]))
    endpoint = values(tracked)
    result = dict(model=line['rf'].model, preparation=provenance,
        coordinates=list(COORDS), endpoint=endpoint.tolist(),
        energy_gain_eV=float(tracked.energy[0]-initial.energy[0]))
    line.track(tracked, ele_start='rf', ele_stop='reference_update', backtrack=True)
    if tracked.state[0] != 1:
        raise RuntimeError('Prepared inverse refused')
    result['reverse_coordinate_gaps'] = (values(tracked)-values(initial)).tolist()
    line['reference_update'].Delta_p0c = initial.p0c[0]*endpoint[5]
    twiss = line.twiss(method='6d', start=xt.START, end=xt.END,
        **dict(zip(COORDS, values(initial))),
        betx=2., alfx=.1, bety=3., alfy=-.2)
    line['reference_update'].Delta_p0c = 0.
    result['end_native_canonical_beta_m'] = [float(twiss.betx[-1]),
                                            float(twiss.bety[-1])]
    copied = xt.Cavity.from_dict(line['rf'].to_dict())
    assert copied.envelope_table.body_coefficients.tobytes() == \
           line['rf'].envelope_table.body_coefficients.tobytes()
    result['copy_serialization_preserves_table'] = True
    line.vars['rf_eta'] = .36
    line.vars['rf_phase'] = .55
    changed = initial.copy()
    line.track(changed, ele_start='rf', ele_stop='reference_update')
    if changed.state[0] != 1:
        raise RuntimeError('Live source update refused')
    result['live_eta_phase_coordinate_change'] = (values(changed)-endpoint).tolist()
    if with_tpsa:
        import madng_tpsa
        import xtrack.tpsa as xtpsa
        descriptor = madng_tpsa.Descriptor(6, 2, num_params=2, param_order=2)
        knobs = xtpsa.KnobParameters(line, ['rf_eta', 'rf_phase'], descriptor)
        knobs.apply()
        try:
            series = xtpsa.ParticlesTpsa(order=2, descriptor=descriptor,
                mass0=initial.mass0, q0=initial.q0, p0c=initial.p0c[0],
                **dict(zip(COORDS, values(initial))))
            line.track(series, ele_start='rf', ele_stop='reference_update')
            if series._xobject.state != 1:
                raise RuntimeError('Prepared TPSA map refused')
            result['tpsa'] = dict(order=2, parameter_names=['eta', 'phase_rad'],
                constant_part=series.const_part.tolist(),
                state_jacobian=series.jacobian().tolist(),
                parameter_jacobian=series.param_jacobian().tolist(),
                d2x_dpx_deta=float(series.coefficient('x',
                    (0, 1, 0, 0, 0, 0, 1, 0))),
                maximum_scalar_constant_difference=float(np.max(
                    abs(series.const_part-values(changed)))))
        finally:
            knobs.teardown()
    result['production_qualified'] = False
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--tpsa', action='store_true')
    parser.add_argument('--json', type=Path)
    parser.add_argument('--build-dir', type=Path,
        default=Path(__file__).resolve().parents[2]/'build'/'cavity-models-envelope')
    args = parser.parse_args(argv)
    if args.steps < 1:
        parser.error('steps must be positive')
    destination = args.json.resolve() if args.json else None
    with cpu_context(args.build_dir) as context:
        line, provenance = make_line(context, args.steps)
        result = run(line, provenance, args.tpsa)
    rendered = json.dumps(result, indent=2)
    if destination:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered+'\n')
    print(rendered)


if __name__ == '__main__':
    main()
