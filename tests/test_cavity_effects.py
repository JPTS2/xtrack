"""Independent scalar full-M2 RF fields, classical LL and rest-spin laws.

Point coefficients divide out rho: tiny physical damping cannot make an
incorrect all-zero implementation pass. Applied cases use actual species
rho, a mechanical-coordinate ODE and explicit mesh refinement. These are
finite-case checks, not physical/numerical/backend qualification.
"""
import math

import numpy as np
from numpy.polynomial import Polynomial
import pytest
from scipy.constants import c, elementary_charge, epsilon_0
from scipy.integrate import solve_ivp

import xobjects as xo
import xtrack as xt
from test_cavity_native import native_context
from test_cavity_native_review import point_oracle


_PROFILE_POLYNOMIALS = [Polynomial([1.])]
for _order in range(5):
    _p = _PROFILE_POLYNOMIALS[-1]
    _PROFILE_POLYNOMIALS.append(Polynomial([1., 0., -1.])*_p.deriv()
        - 2*(16-_order)*Polynomial([0., 1.])*_p)


def profile_jets(z, half_span, maximum=5):
    """Stable factored polynomial recurrence, independently of C Leibniz."""
    if abs(z) >= half_span:
        return np.zeros(maximum+1)
    u = z/half_span
    return np.array([_PROFILE_POLYNOMIALS[n](u)*(1-u*u)**(16-n)/half_span**n
                     for n in range(maximum+1)])


def wave_jets(point, settings, maximum=5):
    """Physical derivatives of Q=A(s sin(T-nu Z)+f sin(T+nu Z+phi))."""
    time, _, _, z = point
    half_span, nu, slow, fast, counter, _ = settings
    profile = profile_jets(z, half_span, maximum)
    values = np.zeros(maximum+1, complex)
    for strength, axial, phase in ((slow, -nu, 0.), (fast, nu, counter)):
        for n in range(maximum+1):
            values[n] += strength*np.exp(1j*(time+axial*z+phase))*sum(
                math.comb(n, j)*profile[j]*(1j*axial)**(n-j)
                for j in range(n+1))
    return values.imag, values.real


def potential(point, settings):
    """Independent compact temporal-gauge M2 potential, before field FD."""
    _, x, y, _ = point
    s, _ = wave_jets(point, settings, 4)
    r2 = x*x+y*y
    radial = s[1]/2-r2*(s[1]+s[3])/16
    return np.array([x*radial, y*radial,
        -s[0]+r2*(s[0]+s[2])/4-r2*r2*(s[0]+2*s[2]+s[4])/64])


def fields(point, settings):
    """Closed curls/time derivatives; cross-checked against potential FD."""
    _, x, y, _ = point
    s, cosine = wave_jets(point, settings)
    r2 = x*x+y*y
    radial_e = cosine[1]/2-r2*(cosine[1]+cosine[3])/16
    radial_b = s[0]/2-r2*(s[0]+s[2])/16
    electric = np.array([-x*radial_e, -y*radial_e,
        cosine[0]-r2*(cosine[0]+cosine[2])/4
        +r2*r2*(cosine[0]+2*cosine[2]+cosine[4])/64+settings[5]])
    magnetic = np.array([y*radial_b, -x*radial_b, 0.])
    return np.r_[electric, magnetic]


def derivative(function, point, direction, step):
    return (function(point-2*step*direction)-8*function(point-step*direction)
            +8*function(point+step*direction)-function(point+2*step*direction))/(12*step)


def material(point, settings, beta, step=1e-4):
    direction = np.r_[1., beta]
    return derivative(lambda p: fields(p, settings), point, direction, step)


@pytest.fixture(scope='module')
def effects_context(tmp_path_factory):
    with native_context(tmp_path_factory.mktemp('cavity-effects')) as context:
        yield context


def compile_point_queries(context):
    from xtrack.track_flags import c_header_flag_mapping
    context.add_kernels(sources=[r'''
      #include "xobjects/headers/common.h"
      #include "xtrack/beam_elements/elements_src/cavity_models.h"
      GPUKERN void cavity_effects_review(double *q,double z,double *settings,
          double *u,double charge,double anomaly,double *out,int64_t *status){
       CavityModelSource source={0};source.half_span=settings[0];source.carrier=settings[1];
       source.forward=settings[2];source.backward=settings[3];source.counter_phase=settings[4];
       source.dc_gradient=settings[5];source.rf_profile=0;
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
      GPUKERN void cavity_effects_direct_law_review(double *u,double charge,double anomaly,
          double *raw,double *out,int64_t *status){
       CavityModelEffectsField field;
       for(int j=0;j<3;++j){field.E[j]=raw[j];field.B[j]=raw[3+j];}
       for(int i=0;i<4;++i)for(int j=0;j<3;++j){
             field.dE[i][j]=raw[6+3*i+j];field.dB[i][j]=raw[18+3*i+j];}
       status[0]=CavityModel_effects_law(u,charge,anomaly,&field,out,out+6,out+9);
      }
      GPUKERN void cavity_effects_species_review(double omega,double mass,double charge,
          double *out,int64_t *status){
       status[0]=CavityModel_species_rho(omega,mass,charge,out);
      }
      GPUKERN void cavity_effects_formal_review(double *q,double *spin,double *settings,
          double rho,double anomaly,int64_t count,double *out,int64_t *status){
       CavityModelSource source={0};source.half_span=settings[0];source.carrier=settings[1];
       source.forward=settings[2];source.backward=settings[3];source.counter_phase=settings[4];
       source.dc_gradient=settings[5];source.rf_profile=0;
       source.k=1.;source.length=2*source.half_span;source.order=-1;
       double current[6],s[3];for(int j=0;j<6;++j)current[j]=q[j];
       for(int j=0;j<3;++j)s[j]=spin[j];
       const double h=2*source.half_span/count;int code=CAVITY_MODEL_OK;
       for(int64_t i=0;i<count && code==CAVITY_MODEL_OK;++i){double next[6],sn[3];
        code=CavityModel_effects_cell(current,s,-source.half_span+i*h,h,1.,&source,
             anomaly,rho,1,1,next,sn);
        if(code==CAVITY_MODEL_OK){for(int j=0;j<6;++j)current[j]=next[j];
             for(int j=0;j<3;++j)s[j]=sn[j];}}
       for(int j=0;j<6;++j)out[j]=current[j];for(int j=0;j<3;++j)out[6+j]=s[j];
       status[0]=code;
      }
    '''], kernels={
        'cavity_effects_review':xo.Kernel(args=[
            xo.Arg(xo.Float64,pointer=True,name='q'),xo.Arg(xo.Float64,name='z'),
            xo.Arg(xo.Float64,pointer=True,name='settings'),xo.Arg(xo.Float64,pointer=True,name='u'),
            xo.Arg(xo.Float64,name='charge'),xo.Arg(xo.Float64,name='anomaly'),
            xo.Arg(xo.Float64,pointer=True,name='out'),xo.Arg(xo.Int64,pointer=True,name='status')],n_threads=1),
        'cavity_effects_direct_law_review':xo.Kernel(args=[
            xo.Arg(xo.Float64,pointer=True,name='u'),xo.Arg(xo.Float64,name='charge'),
            xo.Arg(xo.Float64,name='anomaly'),xo.Arg(xo.Float64,pointer=True,name='raw'),
            xo.Arg(xo.Float64,pointer=True,name='out'),xo.Arg(xo.Int64,pointer=True,name='status')],n_threads=1),
        'cavity_effects_species_review':xo.Kernel(args=[
            xo.Arg(xo.Float64,name='omega'),xo.Arg(xo.Float64,name='mass'),
            xo.Arg(xo.Float64,name='charge'),xo.Arg(xo.Float64,pointer=True,name='out'),
            xo.Arg(xo.Int64,pointer=True,name='status')],n_threads=1),
        'cavity_effects_formal_review':xo.Kernel(args=[
            xo.Arg(xo.Float64,pointer=True,name='q'),xo.Arg(xo.Float64,pointer=True,name='spin'),
            xo.Arg(xo.Float64,pointer=True,name='settings'),xo.Arg(xo.Float64,name='rho'),
            xo.Arg(xo.Float64,name='anomaly'),xo.Arg(xo.Int64,name='count'),
            xo.Arg(xo.Float64,pointer=True,name='out'),xo.Arg(xo.Int64,pointer=True,name='status')],n_threads=1)},
        extra_classes=[xt.Particles._XoStruct],extra_headers=[c_header_flag_mapping])


def test_m2_fields_material_and_rho_free_ll_bmt_thomas(effects_context):
    context = effects_context
    compile_point_queries(context)
    settings = np.array([.4, 1.1, .07, .03, .3, 0.])
    for point, u, charge, anomaly in (
            (np.array([.4,.013,-.009,.035]),np.array([.2,-.1,2.]),1.,.001),
            (np.array([-.7,-.008,.011,-.17]),np.array([-.13,.07,3.5]),-.5,.17)):
        gamma = np.hypot(1.,np.linalg.norm(u));beta = u/gamma
        target = fields(point,settings)
        jac = np.array([derivative(lambda p:potential(p,settings),point,
                                   np.eye(4)[j],1e-5) for j in range(4)])
        field_fd = np.r_[-jac[0],jac[2,2]-jac[3,1],
                          jac[3,0]-jac[1,2],jac[1,1]-jac[2,0]]
        np.testing.assert_allclose(target,field_fd,atol=2e-10,rtol=2e-8)
        derivatives = np.array([derivative(lambda p:fields(p,settings),point,
                                           np.eye(4)[j],1e-4) for j in range(4)])
        dm = material(point,settings,beta)
        np.testing.assert_allclose(dm,material(point,settings,beta,5e-5),atol=2e-9,rtol=2e-7)
        canonical = np.array([point[1],point[2],point[0],0.,0.,gamma])
        canonical[3:5] = u[:2]+charge*potential(point,settings)[:2]
        out=np.full(54,np.nan);status=np.zeros(1,dtype=np.int64)
        context.kernels.cavity_effects_review(q=canonical,z=point[3],settings=settings,
            u=u,charge=charge,anomaly=anomaly,out=out,status=status)
        assert status[0]==1
        np.testing.assert_allclose(out[:6],target,atol=2e-13,rtol=2e-12)
        np.testing.assert_allclose(out[6:18].reshape(4,3),derivatives[:,:3],atol=2e-9,rtol=2e-7)
        np.testing.assert_allclose(out[18:30].reshape(4,3),derivatives[:,3:],atol=2e-9,rtol=2e-7)
        expected,covariant=point_oracle(u,charge,1.,anomaly,*np.split(target,2),*np.split(dm,2))
        assert np.linalg.norm(expected[:6])>1e-3
        assert np.linalg.norm(expected[9:12])>1e-4
        np.testing.assert_allclose(out[30:36],expected[:6],atol=3e-8,rtol=3e-7)
        np.testing.assert_allclose(out[36:39],expected[6:9],atol=2e-13,rtol=2e-12)
        np.testing.assert_allclose(out[39:42],expected[9:12],atol=3e-8,rtol=3e-7)
        np.testing.assert_allclose(out[36:39]+out[39:42],covariant,atol=3e-8,rtol=3e-7)
        np.testing.assert_allclose(out[42:],out[30:42],atol=3e-12,rtol=3e-12)
    raw=np.zeros(30);raw[2]=.7;u=np.array([0.,0.,2.])
    out=np.full(12,np.nan);status=np.zeros(1,dtype=np.int64)
    context.kernels.cavity_effects_direct_law_review(u=u,charge=-1.,anomaly=.001,
        raw=raw,out=out,status=status)
    assert status[0]==1
    np.testing.assert_allclose(out[:6],0.,atol=2e-13)
    np.testing.assert_allclose(out[9:12],0.,atol=2e-13)
    # The LL cancellation does not mean that emitted power vanishes.
    positive_emission=point_oracle(u,-1.,1.,.001,raw[:3],raw[3:6],np.zeros(3),np.zeros(3))[0][20]
    assert positive_emission==pytest.approx(.49,rel=3e-15)


def physical_case():
    """Synthetic classical positron case with resolved physical LL response."""
    mass=xt.ELECTRON_MASS_EV
    frequency=3e9;length=.1;voltage=20e6;gamma=1000.
    k=2*np.pi*frequency/c;half=k*length/2
    eta=.25;counter=.3;phase=.7;calibration_beta=1.
    ratio=np.sqrt(eta);weights=np.array([1.,ratio])/(1+ratio)
    nodes,quadrature=np.polynomial.legendre.leggauss(96)
    z=half*nodes;envelope=(1-nodes*nodes)**16
    integrals=np.array([half*np.sum(quadrature*envelope*np.cos(detuning*z))
        for detuning in (1/calibration_beta-1.,1/calibration_beta+1.)])
    phasor=np.exp(1j*half/calibration_beta)*(
        weights[0]*integrals[0]+weights[1]*integrals[1]*np.exp(1j*counter))
    amplitude=voltage/(mass*abs(phasor))
    clock=phase-np.pi/2-np.angle(phasor)
    settings=np.array([half,1.,amplitude*weights[0],amplitude*weights[1],counter,0.])
    kappa=np.sqrt(gamma*gamma-1.)
    momentum=np.array([2.,-1.,np.sqrt(kappa*kappa-5.)])
    spin=np.array([.3,.4,np.sqrt(.75)])
    initial=np.r_[.1,-.07,clock,momentum,spin]
    rho=elementary_charge*(2*np.pi*frequency)/(6*np.pi*epsilon_0*c)/mass
    return dict(mass=mass,frequency=frequency,length=length,voltage=voltage,
        gamma=gamma,k=k,kappa=kappa,beta=kappa/gamma,eta=eta,counter=counter,
        phase=phase,calibration_beta=calibration_beta,clock=clock,settings=settings,
        initial=initial,rho=rho,anomaly=.00115965218128)


def mechanical_rates(z,state,settings,charge,rho,anomaly,spin=True,thomas=True):
    """Independent total mechanical LL ODE in Z, with fixed-lab rest spin."""
    u=state[3:6];gamma=np.hypot(1.,np.linalg.norm(u));beta=u/gamma
    point=np.array([state[2],state[0],state[1],z])
    electric,magnetic=np.split(fields(point,settings),2)
    de,db=np.split(material(point,settings,beta),2)
    force=charge*(electric+np.cross(beta,magnetic))
    g=electric+np.cross(beta,magnetic)
    emitted=charge**2*gamma**2*(g@g-(beta@electric)**2)
    coefficient=(charge*gamma*(de+np.cross(beta,db))
        +charge**2*(np.cross(electric,magnetic)
            +np.cross(magnetic,np.cross(magnetic,beta))+electric*(beta@electric))
        -emitted*beta)
    omega=-charge*((anomaly+1/gamma)*magnetic
        -anomaly*gamma/(gamma+1)*(beta@magnetic)*beta
        -(anomaly+1/(gamma+1))*np.cross(beta,electric))
    if thomas:
        omega+=rho*np.cross(coefficient,u)/(gamma+1)
    clock=gamma/u[2]
    result=np.r_[u[0]/u[2],u[1]/u[2],clock,clock*(force+rho*coefficient),
                 clock*np.cross(omega,state[6:9]) if spin else np.zeros(3)]
    assert np.all(np.isfinite(result))
    return result


def dense_reference(case,rho,thomas=True):
    result=solve_ivp(lambda z,q:mechanical_rates(z,q,case['settings'],case.get('charge',1.),rho,
            case['anomaly'],thomas=thomas),(-case['settings'][0],case['settings'][0]),
        case['initial'],method='DOP853',rtol=3e-13,
        atol=np.array([2e-13]*3+[3e-11]*3+[2e-13]*3),max_step=.08)
    assert result.success,result.message
    return result.y[:,-1]


def native_physical_result(particle,case):
    """P16 faces have A=0, so native momentum lifts mechanically directly."""
    assert particle.state[0]==1
    kappa=case['kappa'];ux=kappa*float(particle.px[0]);uy=kappa*float(particle.py[0])
    total=kappa*(1+float(particle.delta[0]))
    uz=np.sqrt(total*total-ux*ux-uy*uy)
    time=case['clock']+case['k']*case['length']/case['beta']
    time-=case['k']*float(particle.zeta[0])/case['beta']
    return np.array([case['k']*particle.x[0],case['k']*particle.y[0],time,
        ux,uy,uz,particle.spin_x[0],particle.spin_y[0],particle.spin_z[0]])


def native_applied(context,case,count,mean):
    rf=xt.Cavity(_context=context,model='phase-resolved',rf_profile='p16',
        length=case['length'],frequency=case['frequency'],voltage=case['voltage'],
        phase=case['phase'],eta=case['eta'],counter_phase=case['counter'],
        calibration_beta=case['calibration_beta'],num_kicks=count,absolute_time=True)
    line=xt.Line(elements=[rf,xt.Marker(_context=context)],element_names=['rf','exit'])
    line.particle_ref=xt.Particles(_context=context,mass0=case['mass'],q0=1.,
                                 p0c=case['mass']*case['kappa'])
    line.reset_s_at_end_turn=False;line.use_time_based_zeta_update=True
    line.build_tracker(_context=context)
    line.configure_spin(True)
    if mean:
        line.configure_radiation(model='mean')
    initial=case['initial']
    particle=xt.Particles(_context=context,mass0=case['mass'],q0=1.,
        p0c=case['mass']*case['kappa'],x=initial[0]/case['k'],y=initial[1]/case['k'],
        px=initial[3]/case['kappa'],py=initial[4]/case['kappa'],zeta=0.,delta=0.,
        spin_x=initial[6],spin_y=initial[7],spin_z=initial[8],
        mass_ratio=case.get('mass_ratio',1.),charge_ratio=case.get('charge_ratio',1.),
        anomalous_magnetic_moment=case['anomaly'])
    line.track(particle,ele_start=0,ele_stop=1)
    result=native_physical_result(particle,case)
    if not mean:
        line.track(particle,ele_start=0,ele_stop=1,backtrack=True)
        returned=np.array([particle.x[0],particle.px[0],particle.y[0],particle.py[0],
                           particle.zeta[0],particle.delta[0]])
        np.testing.assert_allclose(returned,
            [initial[0]/case['k'],initial[3]/case['kappa'],initial[1]/case['k'],
             initial[4]/case['kappa'],0.,0.],atol=5e-12,rtol=0.)
        np.testing.assert_allclose([particle.spin_x[0],particle.spin_y[0],particle.spin_z[0]],
                                   initial[6:],atol=5e-12,rtol=0.)
    return result


def test_applied_physical_mean_and_spin_against_dense_mechanical_ode(effects_context):
    case=physical_case()
    assert 1e-14<case['rho']<1e-12
    baseline=dense_reference(case,0.)
    reacted=dense_reference(case,case['rho'])
    target_response=reacted-baseline
    assert np.max(abs(baseline[6:]-case['initial'][6:]))>1e-5
    # A physical, resolved momentum response; a zero-effect implementation
    # cannot pass via a tolerance covering the actual species rho.
    assert np.max(abs(target_response[3:6]))>1e-8
    response_errors=[];spin_errors=[]
    for count in (32,64,128):
        plain=native_applied(effects_context,case,count,False)
        mean=native_applied(effects_context,case,count,True)
        response=mean-plain
        response_errors.append(float(np.max(abs(response[3:6]-target_response[3:6]))))
        spin_errors.append(float(np.max(abs(plain[6:]-baseline[6:]))))
        assert abs(np.linalg.norm(mean[6:])-1.)<2e-10
    print('actual rho',case['rho'],'resolved response',target_response,
          'mean response errors',response_errors,'spin errors',spin_errors)
    assert response_errors[-1]<.02*np.max(abs(target_response[3:6]))
    assert response_errors[-1]<=max(response_errors[0]/4,2e-10)
    assert spin_errors[-1]<2e-8
    assert spin_errors[-1]<=max(spin_errors[0]/4,2e-11)
    np.testing.assert_allclose(mean[6:],reacted[6:],atol=2e-8,rtol=0.)


def test_actual_alternate_species_and_rho_scaling(effects_context):
    """Actual non-reference mass/charge, not an artificial radiation knob."""
    compile_point_queries(effects_context)
    omega=2*np.pi*3e9
    for mass,charge in ((xt.ELECTRON_MASS_EV,-1.),(xt.PROTON_MASS_EV,1.),
                        (208*xt.PROTON_MASS_EV,82.)):
        out=np.full(1,np.nan);status=np.zeros(1,dtype=np.int64)
        effects_context.kernels.cavity_effects_species_review(omega=omega,mass=mass,
            charge=charge,out=out,status=status)
        expected=elementary_charge*omega/(6*np.pi*epsilon_0*c)*charge**2/mass
        assert status[0]==1 and out[0]>0.
        # Production uses CODATA2018 epsilon0; scipy can ship newer CODATA.
        np.testing.assert_allclose(out[0],expected,rtol=2e-9,atol=0.)
    case=physical_case()
    case.update(mass_ratio=2.,charge_ratio=-1.,charge=-.5,rho=case['rho']/2)
    baseline=dense_reference(case,0.);reacted=dense_reference(case,case['rho'])
    plain=native_applied(effects_context,case,128,False)
    mean=native_applied(effects_context,case,128,True)
    response=reacted-baseline
    assert np.max(abs(response[3:6]))>1e-8
    np.testing.assert_allclose((mean-plain)[3:6],response[3:6],
                               atol=.02*np.max(abs(response[3:6])),rtol=0.)
    np.testing.assert_allclose(mean[6:],reacted[6:],atol=2e-8,rtol=0.)
    print('alternate mass_ratio=2 charge_ratio=-1 rho',case['rho'],
          'resolved response',response,'native response',mean-plain)


def test_formal_rho_reacted_feedback_and_thomas(effects_context):
    """FORMAL rho stress; not a physical-species admission or production knob."""
    compile_point_queries(effects_context)
    case=physical_case();formal_rho=1e-6
    baseline=dense_reference(case,0.)
    reference=dense_reference(case,formal_rho)
    without_thomas=dense_reference(case,formal_rho,thomas=False)
    assert np.max(abs(reference[6:]-without_thomas[6:]))>1e-7
    half=dense_reference(case,formal_rho/2)
    # An endpoint linear first-response correction fails this resolved test.
    nonlinear=reference-2*half+baseline
    assert np.max(abs(nonlinear[3:6]))>1e-5
    q=np.r_[case['initial'][:3],case['initial'][3:5],case['gamma']]
    spin=case['initial'][6:].copy();errors=[]
    for count in (64,128):
        out=np.full(9,np.nan);status=np.zeros(1,dtype=np.int64)
        effects_context.kernels.cavity_effects_formal_review(q=q,spin=spin,
            settings=case['settings'],rho=formal_rho,anomaly=case['anomaly'],
            count=count,out=out,status=status)
        assert status[0]==1
        mechanical=np.r_[out[:5],np.sqrt(out[5]**2-1-out[3]**2-out[4]**2),out[6:]]
        errors.append(float(np.max(abs(mechanical[3:6]-reference[3:6]))))
        np.testing.assert_allclose(mechanical[6:],reference[6:],atol=3e-8,rtol=0.)
    assert errors[-1]<2e-6
    assert errors[-1]<errors[0]/4
    print('FORMAL rho',formal_rho,'resolved nonlinear feedback',nonlinear,
          'momentum errors',errors)
