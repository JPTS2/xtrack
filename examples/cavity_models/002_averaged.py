"""Compare native canonical averaged and phase-resolved tracking.

Run from the Xtrack checkout, with the candidate package selected::

    python examples/cavity_models/002_averaged.py --refine
    python examples/cavity_models/002_averaged.py --cohort uniform --tpsa
    python examples/cavity_models/002_averaged.py --cohort gaussian-magnetic --sigma .012 --tpsa

All tracking and Twiss calls use Xtrack alone. Each cohort shares its declared
RF source. The localized cohort uses ideal P16 fields; the uniform cohort uses
equal waves on nonzero-field intervals and reports native canonical-plane
optics. The Gaussian magnetic cohort uses an explicitly supplied width in
metres, nonzero solenoid/normal-quadrupole fields and body-only observation
planes. These are different source assumptions, not one hardware comparison.
Numerical mesh differences and physical model
differences are distinct; these development returns are not error certificates.
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


COORDINATES=('x','px','y','py','zeta','delta')
HARDWARE=dict(length=.05,frequency=400e6,voltage=20e3,phase=.4,
    eta=.25,counter_phase=.3,calibration_beta=1.,integrator='gauss2')
UNIFORM_MODELS=('phase-averaged-uniform-quadratic-first-boundary',
                'phase-averaged-uniform-quadratic')
UNIFORM_HARDWARE=dict(HARDWARE,eta=1.,counter_phase=0.,integrator='midpoint')
MAGNETIC_MODEL='phase-averaged-magnetic'


def cohort_hardware(cohort,sigma=None,center=0.):
    if cohort=='uniform':return dict(UNIFORM_HARDWARE,rf_profile='uniform')
    if cohort=='gaussian-magnetic':
        if sigma is None:
            raise ValueError('The Gaussian cohort requires an explicit --sigma in metres')
        return dict(HARDWARE,rf_profile='gaussian',rf_profile_sigma=sigma,
            rf_profile_center=center,integrator='midpoint',ks=.2,knl=[0.,.01],
            edge_entry_active=0,edge_exit_active=0)
    return dict(HARDWARE,rf_profile='p16')


def coordinates(p):
    return np.array([float(getattr(p,name)[0]) for name in COORDINATES])


def make_line(context,cohort,model,hardware=None):
    """One stable buffer/tracker for model, mesh and Taylor comparisons."""
    hardware=cohort_hardware(cohort) if hardware is None else hardware
    mass=xt.ELECTRON_MASS_EV
    line=xt.Line(elements=[xt.Cavity(_context=context,**hardware,model=model),
        xt.ReferenceEnergyIncrease(_context=context,Delta_p0c=0.),
        xt.Marker(_context=context)],
        element_names=['cavity','reference_update','end'])
    line.particle_ref=xt.Particles(_context=context,mass0=mass,q0=1.,
        p0c=mass*np.sqrt(2.4**2-1.))
    line.vars['rf_phase']=hardware['phase']
    line.vars['rf_voltage']=hardware['voltage']
    line.vars['rf_eta']=hardware['eta']
    line.element_refs['cavity'].phase=line.vars['rf_phase']
    line.element_refs['cavity'].voltage=line.vars['rf_voltage']
    line.element_refs['cavity'].eta=line.vars['rf_eta']
    if cohort=='gaussian-magnetic':
        line.vars['sigma_m']=hardware['rf_profile_sigma']
        line.vars['center_m']=hardware['rf_profile_center']
        line.element_refs['cavity'].rf_profile_sigma=line.vars['sigma_m']
        line.element_refs['cavity'].rf_profile_center=line.vars['center_m']
    line.build_tracker(_context=context)
    line.use_time_based_zeta_update=True
    return line


def run_case(context,model,cells,cohort='localized',with_tpsa=False,line=None,
             hardware=None):
    hardware=cohort_hardware(cohort) if hardware is None else hardware
    if line is None:line=make_line(context,cohort,model,hardware)
    controls={key:value for key,value in hardware.items()
              if key not in ('length','phase','voltage','eta',
                             'rf_profile_sigma','rf_profile_center','knl','ksl')}
    line.set('cavity',**controls,model=model,num_kicks=cells,num_factor_steps=1)
    line.vars['rf_phase']=hardware['phase']
    line.vars['rf_voltage']=hardware['voltage']
    line.vars['rf_eta']=hardware['eta']
    if cohort=='gaussian-magnetic':
        line.vars['sigma_m']=hardware['rf_profile_sigma']
        line.vars['center_m']=hardware['rf_profile_center']
    for name in ('knl','ksl'):
        if name in hardware:
            getattr(line['cavity'],name)[:]=0.
            getattr(line['cavity'],name)[:len(hardware[name])]=hardware[name]
    line['reference_update'].Delta_p0c=0.
    mass=xt.ELECTRON_MASS_EV
    momentum=np.sqrt(2.4**2-1.)
    reference=line.particle_ref.copy(_context=context)
    initial=reference.copy(_context=context)
    initial.x=.0004;initial.y=-.0003
    initial.px=.005/momentum;initial.py=-.004/momentum
    launch=coordinates(initial)
    # Observe before the reference update and turn-frame bookkeeping.
    particle=initial.copy(_context=context)
    line.track(particle,ele_start=0,ele_stop=1)
    if particle.state[0]!=1:
        raise RuntimeError(model+' tracking refused with state '+str(particle.state[0]))
    endpoint=coordinates(particle)
    gain=float(particle.ptau[0]*particle.p0c[0])
    physical_time=float(particle.time_s
        +(particle.s[0]-particle.zeta[0])/(particle.beta0[0]*299792458.))
    kinetic_px=float(particle.kin_px[0]);kinetic_py=float(particle.kin_py[0])
    slope=float(kinetic_px/np.sqrt((1+particle.delta[0])**2
        -kinetic_px**2-kinetic_py**2))
    line.track(particle,ele_start=0,ele_stop=1,backtrack=True)
    if particle.state[0]!=1:
        raise RuntimeError(model+' inverse refused with state '+str(particle.state[0]))
    inverse_gap=coordinates(particle)-launch

    # The localized field-free output admits the usual geometric beta. The
    # uniform interval retains a nonzero vector potential at its output:
    # its normal Twiss result is explicitly a native canonical-plane beta.
    # This momentum rebase does not retime the RF source.
    line['reference_update'].Delta_p0c=reference.p0c[0]*endpoint[5]
    tw=line.twiss(method='6d',start=xt.START,end=xt.END,
        x=launch[0],px=launch[1],y=launch[2],py=launch[3],
        zeta=launch[4],delta=launch[5],betx=2.,alfx=.1,bety=3.,alfy=-.2)
    line['reference_update'].Delta_p0c=0.
    result=dict(model=model,cohort=cohort,num_kicks=cells,num_factor_steps=1,
        source_assumptions=(
            'Gaussian RF + centred solenoid/normal quadrupole; nonzero-field canonical endpoints'
            if cohort=='gaussian-magnetic' else
            'uniform equal waves; nonzero-field canonical endpoints'
            if cohort=='uniform' else 'ideal localized M2/P16 two-wave source'),
        physical_hardware=hardware,endpoint=dict(zip(COORDINATES,endpoint.tolist())),
        energy_gain_eV=gain,physical_arrival_time_s=physical_time,
        exit_kinetic_xprime=slope,
        inverse_coordinate_gaps=dict(zip(COORDINATES,inverse_gap.tolist())),
        numerical_error_certified=False,production_qualified=False)
    label='geometric' if cohort=='localized' else 'native_canonical'
    result['end_'+label+'_beta_x_m']=float(tw.betx[-1])
    result['end_'+label+'_beta_y_m']=float(tw.bety[-1])
    if cohort=='localized':result['field_free_exit_xprime']=slope
    if with_tpsa:
        import madng_tpsa
        import xtrack.tpsa as xtpsa
        names=(['rf_phase','rf_eta','sigma_m','center_m']
               if cohort=='gaussian-magnetic' else ['rf_phase','rf_voltage'])
        labels=(['phase_rad','eta','sigma_m','center_m']
                if cohort=='gaussian-magnetic' else ['phase_rad','voltage_V'])
        descriptor=madng_tpsa.Descriptor(6,1,num_params=len(names),param_order=1)
        knobs=xtpsa.KnobParameters(line,names,descriptor)
        knobs.apply()
        try:
            series=xtpsa.ParticlesTpsa(order=1,descriptor=descriptor,
                mass0=mass,q0=1.,p0c=mass*momentum,**dict(zip(COORDINATES,launch)))
            line.track(series,ele_start=0,ele_stop=1)
            result['tpsa']=dict(order=1,coordinates=list(COORDINATES),
                parameter_names=labels,
                constant_part=series.const_part.tolist(),
                state_jacobian=series.jacobian().tolist(),
                parameter_jacobian=series.param_jacobian().tolist(),
                max_native_constant_difference=float(np.max(
                    abs(series.const_part-endpoint))))
        finally:
            knobs.teardown()
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort',choices=['localized','uniform','gaussian-magnetic'],
                        default='localized')
    parser.add_argument('--sigma',type=float,
                        help='Required Gaussian RF width in metres; no default width.')
    parser.add_argument('--center',type=float,default=0.,
                        help='Gaussian RF centre in metres from the parent midpoint.')
    parser.add_argument('--models',nargs='+')
    parser.add_argument('--tpsa',action='store_true',
        help='Also compute state, phase and voltage Taylor derivatives.')
    parser.add_argument('--cells',type=int,default=16)
    parser.add_argument('--refine',action='store_true')
    parser.add_argument('--build-dir',type=Path,
        default=Path(__file__).resolve().parents[2]/'build'/'cavity-models')
    parser.add_argument('--json',type=Path)
    options=parser.parse_args(argv)
    if options.cells<=0:parser.error('--cells must be positive')
    if options.cohort=='gaussian-magnetic' and (options.sigma is None
            or not np.isfinite(options.sigma) or options.sigma<=0.):
        parser.error('--cohort gaussian-magnetic requires an explicit positive --sigma')
    models=options.models or ([MAGNETIC_MODEL] if options.cohort=='gaussian-magnetic'
        else list(UNIFORM_MODELS) if options.cohort=='uniform'
        else ['phase-resolved','phase-averaged-second-order'])
    for name in models:
        required=('gaussian-magnetic' if name==MAGNETIC_MODEL else
                  'uniform' if name in UNIFORM_MODELS else 'localized')
        if required!=options.cohort:
            parser.error(name+' requires --cohort '+required+' in this example')
    destination=options.json.resolve() if options.json else None
    available=xt.Cavity.get_available_models()
    missing=[name for name in models if name not in available]
    if missing:raise RuntimeError('Candidate Xtrack models unavailable: '+repr(missing)
                                 +'; imported '+str(xt.__file__))
    records=[]
    with cpu_context(options.build_dir) as context:
        hardware=cohort_hardware(options.cohort,options.sigma,options.center)
        line=make_line(context,options.cohort,models[0],hardware)
        for model in models:
            coarse=run_case(context,model,options.cells,options.cohort,
                            options.tpsa,line=line,hardware=hardware)
            if options.refine:
                fine=run_case(context,model,2*options.cells,options.cohort,line=line,
                              hardware=hardware)
                coarse['mesh_differences']={name:fine['endpoint'][name]
                    -coarse['endpoint'][name] for name in COORDINATES}
                coarse['refined_case']=fine
            records.append(coarse)
    result=dict(xtrack=str(xt.__file__),cases=records)
    if destination:
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
