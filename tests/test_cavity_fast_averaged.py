"""Independent original rational packets for uniform quadratic RF2 tracking.

The production optimization shares powers in the same twelve coefficient
rows. The original P(p)+E Q(p) representation below is evaluated with NumPy
polynomials, independently of the expanded production expressions. These
checks neither qualify general C2/C4 nor establish a performance target.
"""
import numpy as np
from numpy.polynomial import Polynomial
import pytest
import xobjects as xo
import xtrack as xt

from test_cavity_native import native_context


def original_rational_packets(energy,nu):
    """Three harmonics, four invariants, columns(value, fixed-hardware dE)."""
    n2,n3,n4=nu**2,nu**3,nu**4
    numerators=(
        [-1.],[2-n2,0.,3*n2-1],[0.,-7*n3+6*nu,0.,-n3-3*nu],
        [-3*n2-2,0.,3*n4-18*n2-1,0.,n4+6*n2+1],
        [-1.],[2-3*n2,0.,1-3*n2],[0.,n2+2,0.,-n2-1],
        [0.,3*n2-14,0.,n2+1],[-1.],[2-n2,0.,n2+1],
        [-4*nu,0.,-3*n3-6*nu,0.,-n3-3*nu],
        [0.,13*n2-14,0.,3*n4+18*n2-13,0.,n4+6*n2+1])
    companions=(
        [0.],[0.,2*n3],[2-n2,0.,-3*n2-1],
        [0.,-12*n3-8*nu,0.,4*n3+4*nu],
        [0.],[0.,-2*n3],[4*nu,0.,-2*nu],[-12*nu,0.,2*nu],
        [0.],[0.,2*nu],[0.,2-9*n2,0.,-3*n2-1],
        [-8*nu,0.,12*n3-8*nu,0.,4*n3+4*nu])
    denominators=((16,1,1,0,2),(64,1,1,0,2),(64,2,2,0,3),(128,3,3,0,4),
        (32,0,0,0,3),(128,0,0,0,3),(128,1,0,0,4),(256,1,0,0,5),
        (16,0,0,1,2),(64,0,0,1,2),(64,1,0,2,3),(128,1,0,3,4))
    p=np.sqrt((energy-1)*(energy+1));rate=energy/p;d=energy+nu*p
    result=[]
    for numerator,companion,(scale,pp,nn,ee,dd) in zip(
            numerators,companions,denominators):
        poly=Polynomial(numerator);other=Polynomial(companion)
        value=poly(p)+energy*other(p)
        derivative=rate*poly.deriv()(p)+other(p)+energy*rate*other.deriv()(p)
        factor=1/(scale*p**pp*nu**nn*energy**ee*d**dd)
        log_derivative=-pp*energy/p**2-ee/energy-dd*(1+nu*rate)/d
        result.append([factor*value,factor*(derivative+log_derivative*value)])
    return np.asarray(result).reshape(3,4,2)


@pytest.fixture(scope='module')
def packet_context(tmp_path_factory):
    from xtrack.track_flags import c_header_flag_mapping
    with native_context(tmp_path_factory.mktemp('fast-quadratic-packets')) as context:
        context.add_kernels(sources=[r'''
        #include "xobjects/headers/common.h"
        #include "xtrack/beam_elements/elements_src/cavity_models_averaged.h"
        GPUKERN void review_quadratic_packets(double energy,double nu,double *out){
            double packet[3][4][2],p=sqrt((energy-1.)*(energy+1.));
            CavityQuadratic_boundary_coefficients(energy,p,nu,packet);
            for(int i=0;i<3;++i)for(int j=0;j<4;++j)for(int k=0;k<2;++k)
                out[8*i+2*j+k]=packet[i][j][k];
        }
        GPUKERN void review_averaged_slow(double *q,double *settings,double *out,
                int64_t *status){
            CavityAveragedSource source={0};source.half_span=settings[0];
            source.slow=settings[1];source.carrier=settings[2];
            source.dc_gradient=settings[7];
            CavityModelSource physical={0};physical.k=1.;
            physical.length=2*source.half_span;physical.half_span=source.half_span;
            physical.rf_profile=(int)settings[6];physical.profile_sigma=settings[4];
            physical.profile_center=settings[5];physical.order=-1;
            source.profile_source=&physical;
            CavityFullJet state[6],flow[6];
            for(int axis=0;axis<6;++axis){state[axis]=CavityFull_constant(q[axis],1);
                state[axis].c[CavityFull_unit(axis)]=1.;}
            status[0]=CavityFull_slow_flow(state,settings[3],&source,flow,1);
            if(status[0]==CAVITY_MODEL_OK)for(int axis=0;axis<6;++axis){
                out[7*axis]=flow[axis].c[0];
                for(int column=0;column<6;++column)
                    out[7*axis+1+column]=flow[axis].c[CavityFull_unit(column)];}
        }
        '''],kernels={'review_quadratic_packets':xo.Kernel(args=[
            xo.Arg(xo.Float64,name='energy'),xo.Arg(xo.Float64,name='nu'),
            xo.Arg(xo.Float64,pointer=True,name='out')],n_threads=1),
          'review_averaged_slow':xo.Kernel(args=[
            xo.Arg(xo.Float64,pointer=True,name='q'),
            xo.Arg(xo.Float64,pointer=True,name='settings'),
            xo.Arg(xo.Float64,pointer=True,name='out'),
            xo.Arg(xo.Int64,pointer=True,name='status')],n_threads=1)},
            extra_classes=[xt.Particles._XoStruct],extra_headers=[c_header_flag_mapping])
        yield context


def native_packets(context,energy,nu):
    out=np.full(24,np.nan)
    context.kernels.review_quadratic_packets(energy=energy,nu=nu,out=out)
    assert np.all(np.isfinite(out))
    return out.reshape(3,4,2)


def test_shared_powers_against_original_rational_rows(packet_context):
    maximum=0.
    for energy in (1.05,1.4,2.4,7.1):
        for nu in (.2,.7,1.,1.8):
            actual=native_packets(packet_context,energy,nu)
            expected=original_rational_packets(energy,nu)
            maximum=max(maximum,float(np.max(abs(actual-expected))))
            np.testing.assert_allclose(actual,expected,atol=3e-13,rtol=3e-13)
    print('shared-power vs original twelve rational rows maximum',maximum)


def test_fixed_hardware_energy_derivative_from_scalar_packet(packet_context):
    maximum=0.
    for energy,nu in ((1.1,.3),(2.4,1.),(5.,1.7)):
        step=2e-5
        derivative=(native_packets(packet_context,energy-2*step,nu)[:,:,0]
            -8*native_packets(packet_context,energy-step,nu)[:,:,0]
            +8*native_packets(packet_context,energy+step,nu)[:,:,0]
            -native_packets(packet_context,energy+2*step,nu)[:,:,0])/(12*step)
        actual=native_packets(packet_context,energy,nu)[:,:,1]
        maximum=max(maximum,float(np.max(abs(actual-derivative))))
        np.testing.assert_allclose(actual,derivative,atol=2e-8,rtol=3e-7)
    print('independent fixed-hardware five-point dE maximum',maximum)


def slow_hamiltonian(q,settings):
    """Independent M2 potential, then the unexpanded longitudinal mass shell.

    All quantities use (X,Y,T,Px,Py,E), with the negative T--E block.
    No production flow or potential-gradient implementation is imported.
    """
    from math import comb
    half,amplitude,nu,z,sigma,centre,profile,dc=settings
    if profile==0:
        envelope=Polynomial([1.,0.,-1/half**2])**16
        jets=[envelope.deriv(order)(z) for order in range(5)]
    elif profile==1:
        jets=[1.,0.,0.,0.,0.]
    else:
        u=(z-centre)/sigma;g=np.exp(-u*u/2)
        jets=[g,-u*g/sigma,(u*u-1)*g/sigma**2,
              u*(3-u*u)*g/sigma**3,(u**4-6*u*u+3)*g/sigma**4]
    field=[amplitude*np.exp(1j*nu*z)*sum(comb(order,k)*jets[k]
           *(1j*nu)**(order-k) for k in range(order+1)) for order in range(5)]
    x,y,t,px,py,energy=q
    sine=np.sin(t);cosine=np.cos(t)
    s=[value.real*sine-value.imag*cosine for value in field]
    radius2=x*x+y*y
    radial=s[1]/2-radius2*(s[1]+s[3])/16
    ax=x*radial;ay=y*radial
    az=-s[0]+radius2*(s[0]+s[2])/4-radius2**2*(s[0]+2*s[2]+s[4])/64-dc*t
    return -az-np.sqrt(energy**2-1-(px-ax)**2-(py-ay)**2)


def slow_flow_from_hamiltonian(q,settings):
    gradient=np.empty(6)
    for axis in range(6):
        shifted=np.asarray(q,dtype=complex)
        shifted[axis]+=1e-24j
        gradient[axis]=slow_hamiltonian(shifted,settings).imag/1e-24
    return gradient[[3,4,5,0,1,2]]*np.array([1.,1.,-1.,-1.,-1.,1.])


@pytest.mark.parametrize('profile',[0,1,2])
def test_averaged_slow_flow_and_jet_from_total_hamiltonian(packet_context,profile):
    """Check the RF derivative contraction and its actual first state jet."""
    settings=np.array([.7,.035,1.3,.11,.23,.04,float(profile),.001])
    q=np.array([.008,-.006,.7,.012,-.009,1.4])
    out=np.full(42,np.nan);status=np.zeros(1,dtype=np.int64)
    packet_context.kernels.review_averaged_slow(q=q,settings=settings,out=out,status=status)
    assert status[0]==1
    actual=out.reshape(6,7)
    expected=slow_flow_from_hamiltonian(q,settings)
    np.testing.assert_allclose(actual[:,0],expected,atol=3e-13,rtol=3e-12)
    finite=np.zeros((6,6));step=2e-5
    for column in range(6):
        for offset,weight in ((-2,1.),(-1,-8.),(1,8.),(2,-1.)):
            shifted=q.copy();shifted[column]+=offset*step
            finite[:,column]+=weight*slow_flow_from_hamiltonian(shifted,settings)/(12*step)
    np.testing.assert_allclose(actual[:,1:],finite,atol=5e-9,rtol=2e-7)
