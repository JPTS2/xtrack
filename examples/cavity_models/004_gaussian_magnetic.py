"""Gaussian RF with a centred straight solenoid and normal quadrupole.

    python examples/cavity_models/004_gaussian_magnetic.py --sigma .012 --tpsa

The RF width and centre are physical metres, relative to the parent midpoint.
The width is required; no installed-cavity shape or default width is inferred.
This joint RF-order-two/transverse-quadratic model observes nonzero-field
canonical planes (both edge flags are off). Its beta functions are native
canonical-plane optics, not field-free mechanical Twiss functions.

One Line/context handles tracking, Twiss, reversal, live controls and state/
parameter Taylor maps. Only Xtrack and its existing TPSA backend are used.
This working example supplies no physical or numerical error certificate.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import xtrack as xt

try:
    from ._context import cpu_context
except ImportError:
    from _context import cpu_context


COORDS=('x','px','y','py','zeta','delta')


def values(p):
    return np.array([float(getattr(p,name)[0]) for name in COORDS])


def make_line(context,sigma,center=0.,steps=4):
    rf=xt.Cavity(_context=context,length=.05,frequency=400e6,voltage=2e3,
        phase=.4,eta=.25,counter_phase=.3,calibration_beta=1.,
        model='phase-averaged-magnetic',integrator='midpoint',
        rf_profile='gaussian',rf_profile_sigma=sigma,rf_profile_center=center,
        ks=.2,knl=[0.,.01],num_kicks=steps,num_factor_steps=1,
        edge_entry_active=0,edge_exit_active=0)
    line=xt.Line(elements=[rf,xt.ReferenceEnergyIncrease(_context=context,
        Delta_p0c=0.),xt.Marker(_context=context)],
        element_names=['rf','reference_update','end'])
    line.particle_ref=xt.Particles(_context=context,mass0=xt.ELECTRON_MASS_EV,
        q0=1.,p0c=xt.ELECTRON_MASS_EV*np.sqrt(2.4**2-1.))
    for variable,field,value in [('phase','phase',.4),('mix','eta',.25),
            ('sigma_m','rf_profile_sigma',sigma),
            ('center_m','rf_profile_center',center)]:
        line.vars[variable]=value
        setattr(line.element_refs['rf'],field,line.vars[variable])
    line.build_tracker(_context=context)
    line.tracker.track_flags.XS_FLAG_USE_TIME_BASED_ZETA_UPDATE=True
    return line


def launch(line):
    p=line.particle_ref.copy()
    for name,value in dict(x=3e-4,px=2e-4,y=-2e-4,py=-1e-4,
                           zeta=1e-4,delta=.002).items():
        setattr(p,name,value)
    return p


def run(line,with_tpsa=False,order=2):
    initial=launch(line);particle=initial.copy()
    line.track(particle,ele_start=0,ele_stop=1)
    if particle.state[0]!=1:raise RuntimeError('Gaussian map refused: '+str(particle.state[0]))
    endpoint=values(particle)
    result=dict(model=line['rf'].model,rf_profile=line['rf'].rf_profile,
        sigma_m=float(line['rf'].rf_profile_sigma),
        center_m=float(line['rf'].rf_profile_center),
        energy_gain_eV=float(particle.energy[0]-initial.energy[0]),
        endpoint=dict(zip(COORDS,endpoint.tolist())))
    line.track(particle,ele_start=0,ele_stop=1,backtrack=True)
    if particle.state[0]!=1:raise RuntimeError('Gaussian inverse refused')
    result['reverse_coordinate_gaps']=dict(zip(COORDS,(values(particle)-values(initial)).tolist()))
    line['reference_update'].Delta_p0c=initial.p0c[0]*endpoint[5]
    tw=line.twiss(method='6d',start=xt.START,end=xt.END,
        **dict(zip(COORDS,values(initial))),betx=2.,alfx=.1,bety=3.,alfy=-.2)
    line['reference_update'].Delta_p0c=0.
    result['end_native_canonical_beta_m']=[float(tw.betx[-1]),float(tw.bety[-1])]

    # Ordinary copy/serialization retain the source independently of the model.
    for copied in (line['rf'].copy(),xt.Cavity.from_dict(line['rf'].to_dict())):
        assert copied.rf_profile=='gaussian'
        assert copied.rf_profile_sigma==line['rf'].rf_profile_sigma
        assert copied.rf_profile_center==line['rf'].rf_profile_center
        np.testing.assert_array_equal(copied.knl,line['rf'].knl)
    result['copy_and_serialization_preserve_source']=True
    sigma=float(line['rf'].rf_profile_sigma)
    line.vars['sigma_m']=1.1*sigma
    changed=initial.copy();line.track(changed,ele_start=0,ele_stop=1)
    if changed.state[0]!=1:raise RuntimeError('Live width update refused')
    result['live_width_endpoint_change']=(values(changed)-endpoint).tolist()
    line.vars['sigma_m']=sigma
    line.set('rf',num_factor_steps=1)

    if with_tpsa:
        import madng_tpsa
        import xtrack.tpsa as xtpsa
        descriptor=madng_tpsa.Descriptor(6,order,num_params=4,param_order=1)
        knobs=xtpsa.KnobParameters(line,['phase','mix','sigma_m','center_m'],descriptor)
        knobs.apply()
        try:
            series=xtpsa.ParticlesTpsa(order=order,descriptor=descriptor,
                mass0=initial.mass0,q0=initial.q0,p0c=initial.p0c[0],
                **dict(zip(COORDS,values(initial))))
            line.track(series,ele_start=0,ele_stop=1)
            result['tpsa']=dict(order=order,coordinates=list(COORDS),
                parameter_names=['phase_rad','eta','sigma_m','center_m'],
                constant_part=series.const_part.tolist(),
                state_jacobian=series.jacobian().tolist(),
                parameter_jacobian=series.param_jacobian().tolist(),
                d2x_dpx_dsigma=float(series.coefficient('x',
                    (0,1,0,0,0,0,0,0,1,0))),
                max_native_constant_difference=float(np.max(abs(series.const_part-endpoint))))
        finally:
            knobs.teardown()
    result['production_qualified']=False
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sigma',type=float,required=True,help='Gaussian width in metres')
    parser.add_argument('--center',type=float,default=0.,help='Centre from parent midpoint, metres')
    parser.add_argument('--steps',type=int,default=4)
    parser.add_argument('--tpsa',action='store_true')
    parser.add_argument('--order',type=int,default=2)
    parser.add_argument('--json',type=Path)
    parser.add_argument('--build-dir',type=Path,
        default=Path(__file__).resolve().parents[2]/'build'/'cavity-models')
    args=parser.parse_args(argv)
    if not np.isfinite(args.sigma) or args.sigma<=0. or args.steps<=0 or args.order<2:
        parser.error('sigma and steps must be positive; mixed TPSA needs order>=2')
    destination=args.json.resolve() if args.json else None
    with cpu_context(args.build_dir) as context:
        result=run(make_line(context,args.sigma,args.center,args.steps),args.tpsa,args.order)
    rendered=json.dumps(result,indent=2)
    if destination:
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_text(rendered+'\n')
    print(rendered)


if __name__=='__main__':
    main()
