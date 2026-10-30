"""Independent uniform magnetic-overlap LL/BMT checks for model10 M2/P16.

Static and RF fields are added before forming the nonlinear LL law. This
finite-case CPU evidence does not qualify other profiles, higher multipoles,
quantum recoil, TPSA effects or GPU execution.
"""
import numpy as np
import pytest
from scipy.integrate import solve_ivp
import xobjects as xo
import xtrack as xt

from test_cavity_native import native_context
from test_cavity_native_review import point_oracle
from test_cavity_effects import (
    potential as rf_potential, fields as rf_fields, derivative, physical_case)


def total_potential(point,settings,magnet):
    _,x,y,_=point;b,gn,gs=magnet
    return rf_potential(point,settings)+np.array(
        [-b*y/2,b*x/2,-gn*(x*x-y*y)/2+gs*x*y])


def total_fields(point,settings,magnet):
    _,x,y,_=point;b,gn,gs=magnet
    return rf_fields(point,settings)+np.array([0.,0.,0.,gs*x+gn*y,gn*x-gs*y,b])


def material(point,settings,magnet,beta,step=1e-4):
    return derivative(lambda p:total_fields(p,settings,magnet),
                      point,np.r_[1.,beta],step)


@pytest.fixture(scope='module')
def overlap_context(tmp_path_factory):
    from xtrack.track_flags import c_header_flag_mapping
    with native_context(tmp_path_factory.mktemp('effects-overlap')) as context:
        context.add_kernels(sources=[r'''
        #include "xobjects/headers/common.h"
        #include "xtrack/beam_elements/elements_src/cavity_models.h"
        GPUKERN void review_overlap(double *q,double z,double *settings,double *magnet,
            double *u,double charge,double anomaly,double *out,int64_t *status){
          CavityModelSource source={0};source.half_span=settings[0];source.carrier=settings[1];
          source.forward=settings[2];source.backward=settings[3];source.counter_phase=settings[4];
          source.rf_profile=0;source.k=1.;source.length=2*source.half_span;
          source.magnetic_scale=1.;source.ks=magnet[0];source.order=1;
          double knl[2]={0.,magnet[1]*source.length},ksl[2]={0.,magnet[2]*source.length};
          source.knl=knl;source.ksl=ksl;
          CavityModelEffectsField field;
          int code=CavityModel_effects_fields(q,z,&source,&field);
          if(code==CAVITY_MODEL_OK){
            for(int j=0;j<3;++j){out[j]=field.E[j];out[3+j]=field.B[j];}
            for(int i=0;i<4;++i)for(int j=0;j<3;++j){
              out[6+3*i+j]=field.dE[i][j];out[18+3*i+j]=field.dB[i][j];}
            code=CavityModel_effects_law(u,charge,anomaly,&field,out+30,out+36,out+39);
          }
          if(code==CAVITY_MODEL_OK)code=CavityModel_effects(q,z,charge,&source,anomaly,out+42,out+48,out+51);
          status[0]=code;
        }
        GPUKERN void review_overlap_cells(double *q,double *spin,double *settings,
            double *magnet,double charge,double anomaly,double rho,int64_t count,
            int64_t mean,double *out,int64_t *status){
          CavityModelSource source={0};source.half_span=settings[0];source.carrier=settings[1];
          source.forward=settings[2];source.backward=settings[3];source.counter_phase=settings[4];
          source.rf_profile=0;source.k=1.;source.length=2*source.half_span;
          source.magnetic_scale=1.;source.ks=magnet[0];source.order=1;
          double knl[2]={0.,magnet[1]*source.length},ksl[2]={0.,magnet[2]*source.length};
          source.knl=knl;source.ksl=ksl;
          double current[6],s[3];for(int j=0;j<6;++j)current[j]=q[j];
          for(int j=0;j<3;++j)s[j]=spin[j];
          const double h=2*source.half_span/count;int code=CAVITY_MODEL_OK;
          for(int64_t i=0;i<count && code==CAVITY_MODEL_OK;++i){double next[6],sn[3];
            code=CavityModel_effects_cell(current,s,-source.half_span+i*h,h,charge,
                &source,anomaly,rho,mean,1,next,sn);
            if(code==CAVITY_MODEL_OK){for(int j=0;j<6;++j)current[j]=next[j];
                for(int j=0;j<3;++j)s[j]=sn[j];}}
          for(int j=0;j<6;++j)out[j]=current[j];for(int j=0;j<3;++j)out[6+j]=s[j];
          status[0]=code;
        }
        '''],kernels={'review_overlap':xo.Kernel(args=[
            xo.Arg(xo.Float64,pointer=True,name='q'),xo.Arg(xo.Float64,name='z'),
            xo.Arg(xo.Float64,pointer=True,name='settings'),xo.Arg(xo.Float64,pointer=True,name='magnet'),
            xo.Arg(xo.Float64,pointer=True,name='u'),xo.Arg(xo.Float64,name='charge'),
            xo.Arg(xo.Float64,name='anomaly'),xo.Arg(xo.Float64,pointer=True,name='out'),
            xo.Arg(xo.Int64,pointer=True,name='status')],n_threads=1),
          'review_overlap_cells':xo.Kernel(args=[
            xo.Arg(xo.Float64,pointer=True,name='q'),xo.Arg(xo.Float64,pointer=True,name='spin'),
            xo.Arg(xo.Float64,pointer=True,name='settings'),xo.Arg(xo.Float64,pointer=True,name='magnet'),
            xo.Arg(xo.Float64,name='charge'),xo.Arg(xo.Float64,name='anomaly'),
            xo.Arg(xo.Float64,name='rho'),xo.Arg(xo.Int64,name='count'),xo.Arg(xo.Int64,name='mean'),
            xo.Arg(xo.Float64,pointer=True,name='out'),xo.Arg(xo.Int64,pointer=True,name='status')],n_threads=1)},
            extra_classes=[xt.Particles._XoStruct],extra_headers=[c_header_flag_mapping])
        yield context


def test_total_potential_fields_material_and_signed_ll_bmt(overlap_context):
    settings=np.array([.4,1.1,.07,.03,.3,0.]);magnet=np.array([.35,.6,-.25])
    for point,u,charge in (
            (np.array([.4,.013,-.009,.035]),np.array([.2,-.1,2.]),1.),
            (np.array([-.7,-.008,.011,-.17]),np.array([-.13,.07,3.5]),-.5)):
        gamma=np.hypot(1.,np.linalg.norm(u));beta=u/gamma
        target=total_fields(point,settings,magnet)
        jac=np.array([derivative(lambda p:total_potential(p,settings,magnet),
                                 point,np.eye(4)[j],1e-5) for j in range(4)])
        curl=np.r_[-jac[0],jac[2,2]-jac[3,1],jac[3,0]-jac[1,2],jac[1,1]-jac[2,0]]
        np.testing.assert_allclose(target,curl,atol=2e-10,rtol=2e-8)
        gradients=np.array([derivative(lambda p:total_fields(p,settings,magnet),
                                       point,np.eye(4)[j],1e-4) for j in range(4)])
        dm=material(point,settings,magnet,beta)
        np.testing.assert_allclose(dm,material(point,settings,magnet,beta,5e-5),
                                   atol=2e-9,rtol=2e-7)
        canonical=np.r_[point[1:3],point[0],u[:2]+charge*total_potential(point,settings,magnet)[:2],gamma]
        out=np.full(54,np.nan);status=np.zeros(1,dtype=np.int64)
        overlap_context.kernels.review_overlap(q=canonical,z=point[3],settings=settings,
            magnet=magnet,u=u,charge=charge,anomaly=.17,out=out,status=status)
        assert status[0]==1
        np.testing.assert_allclose(out[:6],target,atol=2e-13,rtol=2e-12)
        np.testing.assert_allclose(out[6:18].reshape(4,3),gradients[:,:3],atol=2e-9,rtol=2e-7)
        np.testing.assert_allclose(out[18:30].reshape(4,3),gradients[:,3:],atol=2e-9,rtol=2e-7)
        expected,_=point_oracle(u,charge,1.,.17,*np.split(target,2),*np.split(dm,2))
        rf_dm=material(point,settings,np.zeros(3),beta)
        rf_only,_=point_oracle(u,charge,1.,.17,*np.split(rf_fields(point,settings),2),*np.split(rf_dm,2))
        assert np.max(abs(expected[:6]-rf_only[:6]))>1e-3
        np.testing.assert_allclose(out[30:36],expected[:6],atol=3e-8,rtol=3e-7)
        np.testing.assert_allclose(out[36:39],expected[6:9],atol=3e-13,rtol=3e-12)
        np.testing.assert_allclose(out[39:42],expected[9:12],atol=3e-8,rtol=3e-7)
        np.testing.assert_allclose(out[42:],out[30:42],atol=3e-12,rtol=3e-12)


def mechanical_rates(z,state,case,rho,magnet):
    u=state[3:6];gamma=np.hypot(1.,np.linalg.norm(u));beta=u/gamma
    point=np.array([state[2],state[0],state[1],z]);charge=case['charge']
    electric,magnetic=np.split(total_fields(point,case['settings'],magnet),2)
    de,db=np.split(material(point,case['settings'],magnet,beta),2)
    g=electric+np.cross(beta,magnetic)
    emitted=charge**2*gamma**2*(g@g-(beta@electric)**2)
    reaction=(charge*gamma*(de+np.cross(beta,db))
        +charge**2*(np.cross(electric,magnetic)+np.cross(magnetic,np.cross(magnetic,beta))
                    +electric*(beta@electric))-emitted*beta)
    a=case['anomaly']
    omega=-charge*((a+1/gamma)*magnetic-a*gamma/(gamma+1)*(beta@magnetic)*beta
                   -(a+1/(gamma+1))*np.cross(beta,electric))
    omega+=rho*np.cross(reaction,u)/(gamma+1)
    clock=gamma/u[2]
    return np.r_[u[:2]/u[2],clock,clock*(charge*g+rho*reaction),
                 clock*np.cross(omega,state[6:])]


def dense(case,rho,magnet):
    solution=solve_ivp(lambda z,q:mechanical_rates(z,q,case,rho,magnet),
        (-case['settings'][0],case['settings'][0]),case['initial'],method='DOP853',
        rtol=3e-13,atol=np.r_[[2e-13]*3,[3e-11]*3,[2e-13]*3],max_step=.08)
    assert solution.success,solution.message
    return solution.y[:,-1]


def test_actual_species_overlap_reacted_cells_against_mechanical_ode(overlap_context):
    """Applied-law proof independent of the public configuration/adapter route."""
    case=physical_case();case.update(mass_ratio=2.,charge_ratio=-1.,charge=-.5,rho=case['rho']/2)
    magnet=np.array([.35,.6,-.25]);initial=case['initial']
    point=np.array([initial[2],initial[0],initial[1],-case['settings'][0]])
    q=np.r_[initial[:3],initial[3:5]+case['charge']*total_potential(point,case['settings'],magnet)[:2],case['gamma']]
    references=[dense(case,rho,magnet) for rho in (0.,case['rho'])]
    response=references[1]-references[0]
    assert np.max(abs(response[3:6]))>1e-8
    errors=[];orbit_errors=[];spin_errors=[]
    for count in (64,128):
        results=[]
        for mean in (0,1):
            out=np.full(9,np.nan);status=np.zeros(1,dtype=np.int64)
            overlap_context.kernels.review_overlap_cells(q=q,spin=initial[6:].copy(),
                settings=case['settings'],magnet=magnet,charge=case['charge'],
                anomaly=case['anomaly'],rho=case['rho'],count=count,mean=mean,
                out=out,status=status)
            assert status[0]==1
            point=np.r_[out[2],out[:2],case['settings'][0]]
            uxy=out[3:5]-case['charge']*total_potential(point,case['settings'],magnet)[:2]
            uz=np.sqrt(out[5]**2-1-uxy@uxy)
            results.append(np.r_[out[:3],uxy,uz,out[6:]])
        errors.append(float(np.max(abs((results[1]-results[0])[3:6]-response[3:6]))))
        orbit_errors.append(float(np.max(abs(results[0][:6]-references[0][:6]))))
        spin_errors.append(float(np.max(abs(results[1][6:]-references[1][6:]))))
    assert errors[-1]<.02*np.max(abs(response[3:6]))
    assert orbit_errors[-1]<2e-7
    assert orbit_errors[-1]<=max(orbit_errors[0]/4,2e-10)
    assert spin_errors[-1]<2e-8
    assert spin_errors[-1]<=max(spin_errors[0]/4,2e-11)
    print('direct reacted overlap cells rho',case['rho'],'response',response,
          'reaction errors',errors,'orbit errors',orbit_errors,'spin errors',spin_errors)


def native(context,case,magnet,count,mean):
    b,gn,gs=magnet;k=case['k'];kappa=case['kappa'];initial=case['initial']
    rf=xt.Cavity(_context=context,model='phase-resolved',rf_profile='p16',
        length=case['length'],frequency=case['frequency'],voltage=case['voltage'],
        phase=case['phase'],eta=case['eta'],counter_phase=case['counter'],
        calibration_beta=case['calibration_beta'],num_kicks=count,absolute_time=True,
        ks=b*k/kappa,knl=[0.,gn*case['length']*k*k/kappa],
        ksl=[0.,gs*case['length']*k*k/kappa],edge_entry_active=0,edge_exit_active=0)
    line=xt.Line(elements=[rf,xt.Marker(_context=context)],element_names=['rf','exit'])
    line.particle_ref=xt.Particles(_context=context,mass0=case['mass'],q0=1.,p0c=case['mass']*kappa)
    line.reset_s_at_end_turn=False;line.use_time_based_zeta_update=True
    line.build_tracker(_context=context);line.configure_spin(True)
    if mean:
        line.configure_radiation(model='mean')
    entrance=np.array([initial[2],initial[0],initial[1],-case['settings'][0]])
    a=total_potential(entrance,case['settings'],magnet)[:2]
    canonical=initial[3:5]+case['charge']*a
    particle=xt.Particles(_context=context,mass0=case['mass'],q0=1.,p0c=case['mass']*kappa,
        mass_ratio=case['mass_ratio'],charge_ratio=case['charge_ratio'],
        x=initial[0]/k,y=initial[1]/k,px=canonical[0]/kappa,py=canonical[1]/kappa,
        delta=0.,zeta=0.,
        ax=case['charge']*a[0]/kappa,ay=case['charge']*a[1]/kappa,
        spin_x=initial[6],spin_y=initial[7],spin_z=initial[8],
        anomalous_magnetic_moment=case['anomaly'])
    # Canonical P and source A define the inside-field state. Input ax/ay are
    # cached readbacks, not an extra independent momentum specification.
    stale_cache=particle.copy();stale_cache.ax=0.;stale_cache.ay=0.
    line.track(particle,ele_start=0,ele_stop=1)
    line.track(stale_cache,ele_start=0,ele_stop=1)
    assert particle.state[0]==1
    assert stale_cache.state[0]==1
    for name in ('x','px','y','py','zeta','delta','ax','ay','spin_x','spin_y','spin_z'):
        np.testing.assert_allclose(getattr(stale_cache,name),getattr(particle,name),
                                   atol=2e-13,rtol=0.)
    x=k*float(particle.x[0]);y=k*float(particle.y[0])
    time=case['clock']+k*case['length']/case['beta']-k*float(particle.zeta[0])/case['beta']
    a=total_potential(np.array([time,x,y,case['settings'][0]]),case['settings'],magnet)[:2]
    ux=kappa*float(particle.px[0])-case['charge']*a[0]
    uy=kappa*float(particle.py[0])-case['charge']*a[1]
    momentum=kappa*(1+float(particle.delta[0]));uz=np.sqrt(momentum**2-ux**2-uy**2)
    np.testing.assert_allclose(kappa*np.array([particle.ax[0],particle.ay[0]]),
                               case['charge']*a,atol=2e-12,rtol=0.)
    return np.r_[x,y,time,ux,uy,uz,particle.spin_x[0],particle.spin_y[0],particle.spin_z[0]]


def test_actual_species_overlap_mean_spin_and_refinement(overlap_context):
    case=physical_case();case.update(mass_ratio=2.,charge_ratio=-1.,charge=-.5,rho=case['rho']/2)
    magnet=np.array([.35,.6,-.25])
    baseline=dense(case,0.,magnet);reacted=dense(case,case['rho'],magnet)
    rf_only=dense(case,0.,np.zeros(3))
    assert np.max(abs(baseline[3:6]-rf_only[3:6]))>1e-3
    response=reacted-baseline
    assert np.max(abs(response[3:6]))>1e-8
    errors=[];spin_errors=[];orbit_errors=[]
    for count in (64,128):
        plain=native(overlap_context,case,magnet,count,False)
        mean=native(overlap_context,case,magnet,count,True)
        errors.append(float(np.max(abs((mean-plain)[3:6]-response[3:6]))))
        spin_errors.append(float(np.max(abs(mean[6:]-reacted[6:]))))
        orbit_errors.append(float(np.max(abs(plain[:6]-baseline[:6]))))
        assert abs(np.linalg.norm(mean[6:])-1.)<2e-10
    assert errors[-1]<.02*np.max(abs(response[3:6]))
    assert errors[-1]<=max(errors[0]/4,2e-10)
    assert spin_errors[-1]<2e-8
    assert spin_errors[-1]<=max(spin_errors[0]/4,2e-11)
    assert orbit_errors[-1]<2e-7
    assert orbit_errors[-1]<=max(orbit_errors[0]/4,2e-10)
    print('physical overlap rho',case['rho'],'resolved reaction',response,
          'reaction errors',errors,'spin errors',spin_errors,
          'mechanical orbit errors',orbit_errors)
