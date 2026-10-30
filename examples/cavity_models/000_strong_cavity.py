"""Native strong-cavity model comparison using normal Xtrack operations.

Run, for example::

    python examples/cavity_models/000_strong_cavity.py --models phase-resolved
    python examples/cavity_models/000_strong_cavity.py --models all --json result.json
    python examples/cavity_models/000_strong_cavity.py --species all --plot betas.png

ACA112 hardware and entrance Twiss come from the supplied SuperKEKB lattice.
The continuous source uses an explicitly assumed ideal traveling-wave P16
profile. The frozen voltage-action model instead uses its stated uniform body
and face approximation. Neither is a measured installed-cavity distribution.
All examples here
are conservative and radiation off. They do not establish source, numerical
or backend qualification. Production tracking imports Xtrack only.

For the continuous P16 source the four species see the same external RF field
and entrance gamma. Xtrack's reference-polarity convention requires a public
phase shift of minus pi for negative q0 to retain that same field; electrons
consequently decelerate. The frozen construction uses the same nominal inputs
but its reference-dependent body/face approximation is a distinct effective
source, rather than an assertion of identical P16 fields.
Using the same public accelerating phase for both signs would change the
physical oscillator phase by pi. The example is RF-only; holding normalized
magnetic strengths fixed across species would not hold the physical B fixed.
Geometric
Twiss at the exit uses an explicit on-axis reference-momentum update. The
off-axis backtrack case then traverses the same deterministic line in reverse.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import xtrack as xt
from xtrack.particles.masses import Pb208_MASS_EV

from _context import cpu_context


HARDWARE = dict(length=1.270948, voltage=13696827.259102523,
                frequency=2.856e9, phase=np.deg2rad(50.), eta=0.,
                counter_phase=0., calibration_beta=1.)
PIN_OVER_MC = 27.805838507235492
ENTRANCE_TWISS = dict(betx=9.587671823, alfx=5.475230932,
                      bety=8.974262874, alfy=4.726744870)
COORDS = ('x', 'px', 'y', 'py', 'zeta', 'delta')


def species_data():
    return {'e-': (xt.ELECTRON_MASS_EV, -1.),
            'e+': (xt.ELECTRON_MASS_EV, 1.),
            'p+': (xt.PROTON_MASS_EV, 1.),
            'Pb208': (Pb208_MASS_EV, 82.)}


def state(particle):
    return np.array([float(getattr(particle, name)[0]) for name in COORDS])


def run_case(model, species, steps, eta=0., context=None):
    mass, charge = species_data()[species]
    reference = xt.Particles(_context=context, mass0=mass, q0=charge,
                             p0c=mass*PIN_OVER_MC)
    controls = dict(HARDWARE, eta=eta)
    if charge < 0:
        controls['phase'] -= np.pi
    cavity = xt.Cavity(_context=context, model=model, num_kicks=steps, **controls)
    pilot = reference.copy()
    cavity.track(pilot)
    if pilot.state[0] != 1:
        raise RuntimeError(f'{model}/{species}: source status {pilot.state[0]}')
    p_exit = reference.p0c[0]*(1+pilot.delta[0])
    line = xt.Line(elements={
        'rf': cavity,
        'reference_update': xt.ReferenceEnergyIncrease(
            _context=context, Delta_p0c=p_exit-reference.p0c[0]),
    })
    line.particle_ref = reference
    line.reset_s_at_end_turn = False
    line.use_time_based_zeta_update = True
    line.build_tracker(_context=context)
    twiss = line.twiss(method='6d', start=xt.START, end=xt.END,
                       **ENTRANCE_TWISS)
    matrix = line.get_R_matrix(particle_on_co=reference)['R_matrix']
    test = reference.copy()
    test.x = 1e-4
    test.px = 2e-5
    test.y = -5e-5
    test.py = -1e-5
    test.delta = 1e-3
    before = state(test)
    line.track(test)
    if test.state[0] != 1:
        raise RuntimeError(f'{model}/{species}: forward status {test.state[0]}')
    endpoint = state(test)
    endpoint_event = dict(s_m=float(test.s[0]), time_s=float(test.time_s),
        beta0=float(test.beta0[0]),
        ct_m=float((test.s[0]-test.zeta[0])/test.beta0[0]
                   + 299792458.*test.time_s))
    line.track(test, backtrack=True)
    if test.state[0] != 1:
        raise RuntimeError(f'{model}/{species}: reverse status {test.state[0]}')
    error = state(test)-before
    # A normal variable change drives the live tracked source. Recompute the
    # declared reference programme when hardware/phase settings are changed.
    line.vars['rf_phase'] = controls['phase']
    line.element_refs['rf'].phase = line.vars['rf_phase']
    line.vars['rf_phase'] = controls['phase'] + .05
    moved = reference.copy()
    line['rf'].track(moved)
    if moved.state[0] != 1:
        raise RuntimeError(f'{model}/{species}: variable update failed')
    return dict(model=model, species=species, mass_eV=mass, charge_e=charge,
                public_phase_rad=controls['phase'], eta=eta,
                source=('uniform-body frozen voltage action and faces'
                        if model == 'phase-resolved-frozen'
                        else 'ideal M2/P16 two-wave source'),
                p_in_eV=float(reference.p0c[0]), p_out_eV=float(p_exit),
                gain_eV=float(pilot.energy[0]-reference.energy[0]),
                beta_x_out_m=float(twiss.betx[-1]),
                beta_y_out_m=float(twiss.bety[-1]),
                R21_per_m=float(matrix[1, 0]),
                R43_per_m=float(matrix[3, 2]),
                endpoint=dict(zip(COORDS, endpoint.tolist())),
                endpoint_event=endpoint_event,
                reverse_difference=dict(zip(COORDS, error.tolist())),
                phase_knob_gain_change_eV=float(moved.energy[0]-pilot.energy[0]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', default=['phase-resolved',
                        'phase-resolved-paraxial', 'phase-resolved-longitudinal'])
    parser.add_argument('--species', nargs='+', default=['e+'])
    parser.add_argument('--steps', type=int, default=256)
    parser.add_argument('--eta', type=float, default=0.,
                        help='Squared counter/co-wave amplitude ratio (default0).')
    parser.add_argument('--json', type=Path)
    parser.add_argument('--plot', type=Path)
    parser.add_argument('--build-dir', type=Path,
                        default=Path(__file__).resolve().parents[2]/'build'/'cavity-models')
    options = parser.parse_args(argv)
    if options.steps < 0:
        parser.error('--steps must be nonnegative (0 selects model defaults)')
    models = (list(xt.Cavity.get_available_models()) if options.models == ['all']
              else options.models)
    if options.models == ['all']:
        models = [name for name in models if name.startswith('phase-')
                  and not name.startswith('phase-averaged-uniform-')
                  and name!='phase-averaged-magnetic']
        print('Uniform and Gaussian magnetic models use separate source cohorts: '
              'run 002_averaged.py --cohort uniform, or --cohort gaussian-magnetic '
              '--sigma WIDTH_IN_METRES.',flush=True)
    species = list(species_data()) if options.species == ['all'] else options.species
    results, failures = [], []
    for name in species:
        if name not in species_data():
            parser.error('unknown species '+name)
    with cpu_context(options.build_dir) as context:
        for name in species:
            for model in models:
                try:
                    result = run_case(model, name, options.steps, options.eta,
                                      context=context)
                    results.append(result)
                    print(f"{name:6s} {model:30s} gain={result['gain_eV']:.8g} eV "
                          f"beta=({result['beta_x_out_m']:.8g},"
                          f"{result['beta_y_out_m']:.8g}) m "
                          f"R21={result['R21_per_m']:.8g} /m", flush=True)
                except Exception as error:
                    failures.append(dict(model=model, species=name, error=repr(error)))
                    print(f'{name} {model}: FAILED {error}', flush=True)
    payload = dict(hardware=dict(HARDWARE, eta=options.eta),
                   entrance_twiss=ENTRANCE_TWISS,
                   radiation='off', source='see per-model source assumptions',
                   results=results, failures=failures)
    if options.json:
        options.json.write_text(json.dumps(payload, indent=2))
    if options.plot and results:
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
        labels = [item['species']+'\n'+item['model'] for item in results]
        axes[0].bar(labels, [-item['R21_per_m'] for item in results])
        axes[1].bar(labels, [item['beta_x_out_m'] for item in results])
        axes[0].set_ylabel(r'$-R_{21}$ [m$^{-1}$], physical slope coordinates')
        axes[1].set_ylabel(r'Exit geometric $\beta_x$ [m]')
        for axis in axes:
            axis.tick_params(axis='x', labelrotation=35, labelsize=8)
            axis.grid(axis='y', alpha=.2)
        title = 'ACA112 hardware, assumed ideal P16 traveling wave'
        if any(item['model'] == 'phase-resolved-frozen' for item in results):
            title += '\nFrozen model: uniform-body and prescribed-face approximation'
        figure.suptitle(title)
        figure.savefig(options.plot, dpi=160)
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
