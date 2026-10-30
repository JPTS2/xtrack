"""Independent physical checks of public conservative cavity maps.

Expected vacuum trajectories and linear magnetic matrices are analytic, not
transcribed from the cavity source or generated Python expressions. These
checks establish the stated limits and clock/chart consistency, not numerical
admission, measured-cavity accuracy or radiation support.
"""

import ctypes
from math import comb, factorial
from pathlib import Path
import subprocess

import numpy as np
import pytest
import xtrack as xt

from test_cavity_native import native_context


MODELS = ('phase-resolved', 'phase-resolved-paraxial',
          'phase-resolved-longitudinal')
C_LIGHT = 299792458.
MASS0 = 510998.95069
COORDINATES = ('x', 'px', 'y', 'py', 'zeta', 'delta')


def values(particle):
    """Copy physical particle slots without relying on a map's own tangent."""
    return np.array([np.asarray(getattr(particle, name))[0]
                     for name in COORDINATES])


def particle(context, gamma, charge=1., **changes):
    """Use the ordinary Xtrack canonical field-free entrance convention."""
    launch = dict(mass0=MASS0, q0=charge,
                  p0c=MASS0*np.sqrt((gamma-1.)*(gamma+1.)),
                  x=.0004, px=.12, y=-.0003, py=-.07,
                  zeta=.002, delta=0., s=0., time_s=0.)
    launch.update(changes)
    return xt.Particles(_context=context, **launch)


def test_public_cavity_independent_vacuum_limits(tmp_path):
    """H2 keeps its own finite-angle drift, rather than aliasing full drift."""
    length = .37
    with native_context(tmp_path) as context:
        for model in MODELS:
            element = xt.Cavity(_context=context, length=length, voltage=0.,
                frequency=0., eta=.3, model=model, integrator='gauss2',
                num_kicks=7, absolute_time=True)
            line = xt.Line(elements=[element,xt.Marker(_context=context)],
                           element_names=['cavity','physical_exit'])
            line.particle_ref = particle(context, 2.4, x=0., px=0., y=0.,
                                         py=0., zeta=0.)
            line.reset_s_at_end_turn = False
            for gamma in (2.4, 20.):
                for charge in (-1., 1.):
                    incoming = particle(context, gamma, charge)
                    initial = values(incoming)
                    expected = initial.copy()
                    angle2 = initial[1]**2+initial[3]**2
                    if model == 'phase-resolved-paraxial':
                        expected[0] += length*initial[1]
                        expected[2] += length*initial[3]
                        expected[4] -= length*angle2/2
                    else:
                        longitudinal = np.sqrt(1-angle2)
                        expected[0] += length*initial[1]/longitudinal
                        expected[2] += length*initial[3]/longitudinal
                        expected[4] += length*(1-1/longitudinal)
                    # Observe the physical exit before the usual end-turn
                    # time_s/zeta reference-frame change. DriftExact has that
                    # same bookkeeping; it is separate from the cavity map.
                    line.track(incoming,ele_start=0,ele_stop=1)
                    assert incoming.state[0] == 1, model
                    np.testing.assert_allclose(values(incoming), expected,
                        rtol=0., atol=2e-13, err_msg=model)
                    np.testing.assert_allclose(incoming.s[0], length,
                                               rtol=0., atol=2e-14)
                    line.track(incoming,ele_start=0,ele_stop=1,backtrack=True)
                    assert incoming.state[0] == 1, model
                    np.testing.assert_allclose(values(incoming), initial,
                        rtol=0., atol=3e-13, err_msg=model+' signed inverse')


def test_public_cavity_independent_quadrupole_limit(tmp_path):
    """Ordinary positive normal strength focuses x, with integrated units."""
    length, k1, epsilon = .4, .3, 1e-6
    wave = np.sqrt(k1)
    angle = wave*length
    expected_x = epsilon*np.array([np.cos(angle), -wave*np.sin(angle)])
    expected_y = epsilon*np.array([np.cosh(angle), wave*np.sinh(angle)])
    with native_context(tmp_path) as context:
        for model in MODELS:
            element = xt.Cavity(_context=context, length=length, voltage=0.,
                frequency=0., eta=0., knl=[0., k1*length], ks=0.,
                model=model, integrator='gauss2', num_kicks=64)
            for charge in (-1., 1.):
                incoming = particle(context, 2.4, charge,
                    x=epsilon, px=0., y=epsilon, py=0., zeta=0.)
                element.track(incoming)
                assert incoming.state[0] == 1, model
                result = values(incoming)
                # Finite launch and collocation errors are below this linear
                # limit check; the test is not a finite-amplitude exact map.
                np.testing.assert_allclose(result[:2], expected_x,
                    rtol=2e-8, atol=3e-14, err_msg=model+' normal x')
                np.testing.assert_allclose(result[2:4], expected_y,
                    rtol=2e-8, atol=3e-14, err_msg=model+' normal y')
                np.testing.assert_allclose(result[5], 0., rtol=0., atol=2e-13)


def test_public_cavity_physical_clock_frame_invariance(tmp_path):
    """A coordinated time_s/zeta frame change leaves the RF event fixed."""
    delay = .37e-9
    gamma = 2.4
    beta0 = np.sqrt((gamma-1.)*(gamma+1.))/gamma
    shift_zeta = beta0*C_LIGHT*delay
    with native_context(tmp_path) as context:
        for model in MODELS:
            element = xt.Cavity(_context=context, length=.05, voltage=20e3,
                frequency=400e6, phase=.23, eta=.25, counter_phase=.3,
                calibration_beta=1., model=model, integrator='gauss2',
                num_kicks=16, absolute_time=True)
            first = particle(context, gamma, px=.005, py=-.004,
                             zeta=.001, time_s=0.)
            second = particle(context, gamma, px=.005, py=-.004,
                zeta=.001+shift_zeta, time_s=delay)
            element.track(first)
            element.track(second)
            assert first.state[0] == second.state[0] == 1, model
            translated = values(second)
            translated[4] -= shift_zeta
            np.testing.assert_allclose(translated, values(first),
                rtol=0., atol=2e-12, err_msg=model+' physical RF clock')


def test_public_cavity_rejects_nonpositive_native_momentum(tmp_path):
    """Squaring p cannot turn a backward native ray into a forward one."""
    checked=0
    names=(*COORDINATES,'ptau','pzeta','rpp','rvv','s','ax','ay')
    with native_context(tmp_path) as context:
        for model in MODELS:
            element=xt.Cavity(_context=context,length=.05,voltage=0.,
                frequency=0.,eta=.3,model=model,integrator='gauss2',
                num_kicks=4)
            for delta in (-1.1,-1.):
                try:
                    incoming=particle(context,2.4,px=0.,py=0.,delta=delta)
                except ValueError:
                    # Some particle constructors may already reject this
                    # direction; do not corrupt raw buffers to bypass them.
                    continue
                before={name:np.asarray(getattr(incoming,name)).copy()
                        for name in names}
                time_before=float(incoming.time_s)
                element.track(incoming)
                assert incoming.state[0]<0,(model,delta)
                for name,expected in before.items():
                    np.testing.assert_array_equal(getattr(incoming,name),
                        expected,err_msg=repr((model,delta,name)))
                assert float(incoming.time_s)==time_before
                checked+=1
    if not checked:
        pytest.skip('Particles constructor already rejects both directions')


@pytest.fixture(scope='module')
def scalar_source_library(tmp_path_factory):
    """Compile the shipped scalar source without its Python exporter.

    These two point-law probes do not use a generated reference force, a
    prepared source cache or the public native-coordinate adapter.
    """
    directory=tmp_path_factory.mktemp('cavity-models-physical-source')
    source=directory/'source.c'
    source.write_text(r'''
#define GPUFUN static inline
#define GPUGLMEM
#define XTRACK_TRACK_H
typedef double xt_float_or_tpsa;
typedef const double xt_float_or_tpsa_arg;
static inline double xt_float_or_tpsa_const_part(double x){return x;}
static inline int xt_float_or_tpsa_is_zero(double x){return x==0.;}
#include "xtrack/beam_elements/elements_src/cavity_models.h"
int static_case(const double *q,double Q,double scale,int model,double *out){
 CavityModelSource s={0};
 const double normal[4]={.017,.23,-.19,.13};
 const double skew[4]={-.01,.11,.07,-.17};
 s.k=.9;s.omega=299792458.*s.k;s.length=.43;s.half_span=.7;
 s.carrier=1.1;s.counter_phase=.3;s.forward=0.;s.backward=0.;
 s.magnetic_scale=scale;s.ks=.17;s.order=3;s.knl=normal;s.ksl=skew;
 int status=CavityModel_potential(q,.02,&s,model,out);
 if(status!=1)return status;
 return CavityModel_flow(q,.02,Q,&s,model,out+4);
}
int shell_case(double energy,double *out){
 CavityModelSource s={0};
 s.k=1.1;s.omega=299792458.*s.k;s.half_span=.7;
 s.length=2.*s.half_span/s.k;s.carrier=1.2;s.counter_phase=.3;
 s.forward=.1;s.backward=.03;s.order=-1;
 const double q[6]={.05,.02,1.57079632679489661923,0.,0.,energy};
 int status=CavityModel_physical_potential(q,.175,&s,12,out);
 if(status!=1)return status;
 status=CavityModel_potential(q,.175,&s,12,out+4);
 if(status!=1)return status;
 for(int i=8;i<14;++i)out[i]=123456.;
 return CavityModel_flow(q,.175,1.3,&s,12,out+8);
}
''')
    library=directory/'physical_source.so'
    checkout=Path(xt.__file__).resolve().parents[1]
    subprocess.run(['cc','-shared','-O2','-fPIC','-I'+str(checkout),
        str(source),'-o',str(library)],check=True,timeout=30)
    result=ctypes.CDLL(str(library))
    pointer=np.ctypeslib.ndpointer(dtype=np.float64,ndim=1,
                                  flags='C_CONTIGUOUS')
    result.static_case.argtypes=[pointer,ctypes.c_double,ctypes.c_double,
                                 ctypes.c_int,pointer]
    result.static_case.restype=ctypes.c_int
    result.shell_case.argtypes=[ctypes.c_double,pointer]
    result.shell_case.restype=ctypes.c_int
    return result


def independent_static_point(state,charge,scale,model):
    """Polynomial B_y+iB_x and canonical Lorentz connection in Z=k*s."""
    k,length,ks=.9,.43,.17
    normal=np.array([.017,.23,-.19,.13])
    skew=np.array([-.01,.11,.07,-.17])
    x,y=state[0]/k,state[1]/k
    coordinate=complex(x,y)
    field=sum(complex(n,s)*coordinate**j/factorial(j)
              for j,(n,s) in enumerate(zip(normal,skew)))
    primitive=sum(complex(n,s)*coordinate**(j+1)/factorial(j+1)
                  for j,(n,s) in enumerate(zip(normal,skew)))
    longitudinal_field=scale*ks
    potential=np.array([-longitudinal_field*y/2,
        longitudinal_field*x/2,-scale*primitive.real/length,0.])
    kinetic=state[3:5]-charge*potential[:2]
    momentum=np.sqrt((state[5]-1)*(state[5]+1))
    longitudinal=np.sqrt(momentum*momentum-kinetic@kinetic)
    denominator=momentum if model==11 else longitudinal
    flow=np.zeros(6)
    flow[:2]=kinetic/denominator
    flow[2]=state[5]/denominator
    if model==11:flow[2]*=1+(kinetic@kinetic)/(2*momentum*momentum)
    by=scale*field.real/length
    bx=scale*field.imag/length
    mechanical=np.array([charge*(-by+flow[1]*longitudinal_field)/k,
                          charge*(bx-flow[0]*longitudinal_field)/k])
    connection=charge*np.array([-longitudinal_field*flow[1]/(2*k),
                                 longitudinal_field*flow[0]/(2*k)])
    flow[3:5]=mechanical+connection
    return np.r_[potential,flow]


def test_source_static_independent_magnetic_laws(scalar_source_library):
    """Challenge charges, near-rest roots, normal/skew factorials and H2."""
    for model in (10,11,12):
        for scale in (.7,-1.2):
            for charge in (-1.3,.37,1.8):
                for energy in (1.03,2.4,20.):
                    for x,y in ((.012,-.009),(-.023,.017)):
                        state=np.array([x,y,.71,.12,-.08,energy])
                        actual=np.zeros(10)
                        status=scalar_source_library.static_case(
                            state,charge,scale,model,actual)
                        assert status==1,(model,scale,charge,energy,status)
                        expected=independent_static_point(
                            state,charge,scale,model)
                        np.testing.assert_allclose(actual,expected,
                            rtol=2e-14,atol=2e-14)


def test_projected_source_retains_full_physical_shell(scalar_source_library):
    """An auxiliary positive shell cannot hide a failed physical RF lift."""
    span,position=.7,.175
    u=position/span
    base=1-u*u
    envelope=np.array([base**16,-32*u*base**15/span,
        32*(31*u*u-1)*base**14/span**2,
        960*u*(3-31*u*u)*base**13/span**3])
    profile=np.array([sum(weight*np.exp(1j*(nu*position+phase))*sum(
        comb(j,l)*envelope[l]*(1j*nu)**(j-l) for l in range(j+1))
        for weight,nu,phase in ((.1,1.2,0.),(.03,-1.2,-.3)))
        for j in range(4)])
    time=np.pi/2
    sine=profile.real*np.sin(time)-profile.imag*np.cos(time)
    cosine=profile.real*np.cos(time)+profile.imag*np.sin(time)
    radius2=.05**2+.02**2
    radial=sine[1]/2-radius2*(sine[1]+sine[3])/16
    physical_potential=np.array([.05*radial,.02*radial])
    for energy,expected_status in ((1.00001,-1),(2.4,1)):
        actual=np.zeros(14)
        status=scalar_source_library.shell_case(energy,actual)
        assert status==expected_status
        np.testing.assert_allclose(actual[:2],physical_potential,
                                   rtol=2e-14,atol=2e-15)
        np.testing.assert_array_equal(actual[4:6],0.)
        auxiliary=(energy-1)*(energy+1)
        physical=auxiliary-1.3**2*(physical_potential@physical_potential)
        assert auxiliary>0
        if status==-1:
            assert physical<0
            np.testing.assert_array_equal(actual[8:],123456.)
        else:
            assert physical>0
            expected=np.array([0.,0.,energy/np.sqrt(auxiliary),
                               0.,0.,1.3*cosine[0]])
            np.testing.assert_allclose(actual[8:],expected,
                                       rtol=2e-14,atol=2e-14)
