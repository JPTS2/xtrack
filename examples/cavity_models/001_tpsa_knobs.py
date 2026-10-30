"""Native cavity tracking with state and physical-parameter Taylor maps.

For example, from the Xtrack checkout::

    python examples/cavity_models/001_tpsa_knobs.py --model phase-resolved
    python examples/cavity_models/001_tpsa_knobs.py --order 3 --json build/tpsa.json

The positive eta base point deliberately contains both waves. Parameter
derivatives include the fixed-beta voltage calibration and oscillator phase,
not just the explicit force coefficients. At eta=0, sqrt(eta) is generally
nonanalytic: ordinary state/phase maps remain meaningful, while an eta Taylor
parameter there is explicitly refused. This example does not change eta's
squared-amplitude-ratio meaning or claim stochastic/radiation differentiation.

The incoming reference momentum is fixed. Matrices and mixed coefficients
refer to native canonical coordinates, not a retuned reference programme or
an implicitly normalized geometric beta. All models here are experimental
conservative maps; no source/numerical/backend admission follows from this
example. Tracking requires Xtrack and its existing madng_tpsa backend only.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import madng_tpsa
import xtrack as xt
import xtrack.tpsa as xtpsa
try:
    from ._context import cpu_context
except ImportError:
    from _context import cpu_context


MODELS = ('phase-resolved', 'phase-resolved-paraxial',
          'phase-resolved-longitudinal')
COORDS = ('x', 'px', 'y', 'py', 'zeta', 'delta')


def state(particles):
    return np.array([float(getattr(particles, name)[0]) for name in COORDS])


def run(context, model='phase-resolved', order=2, steps=8):
    mass=xt.ELECTRON_MASS_EV
    momentum=np.sqrt(2.4**2-1)
    launch=dict(mass0=mass,q0=1.,p0c=mass*momentum,
        x=4e-4,px=.005/momentum,y=-3e-4,py=-.004/momentum,
        zeta=3e-4,delta=.002)
    line=xt.Line(elements={'rf':xt.Cavity(length=.05,voltage=2e4,
        frequency=400e6,model=model,num_kicks=steps,
        counter_phase=.3,calibration_beta=.9)})
    line.particle_ref=xt.Particles(mass0=mass,q0=1.,p0c=mass*momentum)
    line.reset_s_at_end_turn=False
    line.vars['rf_eta']=.25
    line.vars['rf_phase']=.4
    line.element_refs['rf'].eta=line.vars['rf_eta']
    line.element_refs['rf'].phase=line.vars['rf_phase']
    line.build_tracker(_context=context)
    scalar=xt.Particles(_context=context,**launch)
    line.track(scalar)
    if scalar.state[0]!=1:
        raise RuntimeError(f'scalar cavity source status {scalar.state[0]}')

    descriptor=madng_tpsa.Descriptor(6,order,num_params=2,param_order=1)
    knobs=xtpsa.KnobParameters(line,['rf_eta','rf_phase'],descriptor)
    knobs.apply()
    try:
        series=xtpsa.ParticlesTpsa(order=order,descriptor=descriptor,**launch)
        line.track(series)
        first=dict(constant_part=series.const_part.tolist(),
            state_jacobian=series.jacobian().tolist(),
            parameter_names=['eta','phase_rad'],
            parameter_jacobian=series.param_jacobian().tolist(),
            d2x_dpx_deta=float(series.coefficient('x',(0,1,0,0,0,0,1,0))),
            maximum_scalar_constant_difference=float(np.max(
                abs(series.const_part-state(scalar)))))
        knobs.refresh([.36,.55])
        refreshed=xtpsa.ParticlesTpsa(order=order,descriptor=descriptor,**launch)
        line.track(refreshed)
        second=dict(eta=.36,phase_rad=.55,
            constant_part=refreshed.const_part.tolist(),
            parameter_jacobian=refreshed.param_jacobian().tolist())
    finally:
        knobs.teardown()

    # The parameter context is over: ordinary variables, set, copying and
    # serialization operate on the same physical inputs again.
    line.vars['rf_phase']=.65
    line.set('rf',counter_phase=.35)
    copied=line.copy()
    restored=xt.Line.from_dict(copied.to_dict())
    restored.build_tracker(_context=context)
    original=xt.Particles(_context=context,**launch)
    roundtrip=xt.Particles(_context=context,**launch)
    line.track(original)
    restored.track(roundtrip)
    return dict(model=model,order=order,num_kicks=steps,
        reference_p0c_eV=launch['p0c'],coordinates=list(COORDS),
        initial=first,refreshed=second,
        scalar_storage_restored=not bool(line['rf']._tpsa_enabled),
        derived_carrier_preserved=line['rf'].wave_number is None,
        copy_serialization_maximum_difference=float(np.max(
            abs(state(original)-state(roundtrip)))))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=MODELS,default=MODELS[0])
    parser.add_argument('--order',type=int,default=2)
    parser.add_argument('--steps',type=int,default=8)
    parser.add_argument('--json',type=Path)
    parser.add_argument('--build-dir',type=Path,
        default=Path(__file__).resolve().parents[2]/'build'/'cavity_models_tpsa')
    options=parser.parse_args(argv)
    if options.order<2 or options.steps<1:
        parser.error('order must be at least 2 and steps must be positive')
    destination=options.json.resolve() if options.json else None
    build=options.build_dir.resolve()
    with cpu_context(build) as context:
        result=run(context,options.model,options.order,options.steps)
    rendered=json.dumps(result,indent=2)
    if destination:
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_text(rendered+'\n')
    print(rendered)


if __name__=='__main__':
    main()
