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
        '''],kernels={'review_quadratic_packets':xo.Kernel(args=[
            xo.Arg(xo.Float64,name='energy'),xo.Arg(xo.Float64,name='nu'),
            xo.Arg(xo.Float64,pointer=True,name='out')],n_threads=1)},
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
