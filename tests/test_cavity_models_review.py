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


def test_public_cavity_mixed_species_reference_equivalence(tmp_path):
    """Changing the reference description preserves the same actual fields.

    Native px/py and delta use mass_ratio*p0c, so these two descriptions
    have identical normalized coordinates and beta0. For the rebuilt
    negative-charge reference, phase-pi preserves the physical RF clock;
    static strengths are rescaled by q0/p0c to preserve the SI magnetic
    fields. Frozen-coefficient models have a separate nominal programme
    and are deliberately outside this reference-invariance statement.
    """
    gamma = 2.4
    base_static = dict(ks=.17, knl=[.001, .004], ksl=[-.0007, .002])
    actual_static = {
        name: np.asarray(value)*(-.5) for name, value in base_static.items()
    }
    controls = dict(length=.05, voltage=20e3, frequency=400e6, phase=.23,
                    eta=.25, counter_phase=.3, calibration_beta=1.,
                    integrator='gauss2', num_kicks=16, absolute_time=True)
    metadata = ('ptau', 'pzeta', 'rpp', 'rvv', 'ax', 'ay', 's')
    with native_context(tmp_path) as context:
        for model in MODELS:
            fixed = xt.Cavity(_context=context, model=model,
                              **controls, **base_static)
            rebuilt_controls = {**controls, 'phase': controls['phase']-np.pi}
            rebuilt = xt.Cavity(_context=context, model=model,
                                **rebuilt_controls, **actual_static)
            mixed = particle(context, gamma, charge_ratio=-1., mass_ratio=2.,
                             px=.005, py=-.004, zeta=.001, delta=.03)
            actual = particle(context, gamma, charge=-1., mass0=2*MASS0,
                p0c=2*MASS0*np.sqrt((gamma-1.)*(gamma+1.)),
                px=.005, py=-.004, zeta=.001, delta=.03)
            fixed.track(mixed)
            rebuilt.track(actual)
            assert mixed.state[0] == actual.state[0] == 1, model
            np.testing.assert_allclose(values(mixed), values(actual),
                rtol=0., atol=8e-13, err_msg=model+' actual-species chart')
            for name in metadata:
                np.testing.assert_allclose(getattr(mixed, name),
                    getattr(actual, name), rtol=0., atol=8e-13,
                    err_msg=model+' '+name)
            # Physical momenta/energies also agree, despite the different
            # reference masses, charges and momentum normalizations.
            np.testing.assert_allclose(mixed.energy, actual.energy,
                rtol=3e-13, atol=2e-7, err_msg=model+' physical energy')


def test_public_full_slow_static_uses_same_hamiltonian(tmp_path):
    """The zero-RF full-slow limit is the complete total-static root.

    This exercises the public adapter's physical source pointer and signed
    species factor, independently of the high-rank source exporter. Both
    models use exactly the same G2 nodes; all normal/skew orders and the
    solenoid coexist inside the root. Endpoints lie in that magnetic gauge.
    """
    controls = dict(length=.37, voltage=0., frequency=0., eta=.3,
        ks=.41, knl=[.013, .12, -.07, .04, 2.3, -4.1],
        ksl=[-.009, .04, .09, -.03, -.7, 3.2],
        integrator='gauss2', num_kicks=11,
        edge_entry_active=False, edge_exit_active=False)
    chi, x, y = -.5, .035, -.027
    metadata = ('ptau', 'pzeta', 'rpp', 'rvv', 'ax', 'ay', 's')
    results = []
    with native_context(tmp_path) as context:
        for model in ('phase-resolved', 'phase-averaged-full-slow'):
            element = xt.Cavity(_context=context, model=model, **controls)
            line = xt.Line(elements=[element, xt.Marker(_context=context)],
                           element_names=['cavity', 'physical_exit'])
            line.particle_ref = particle(context, 2.4, x=0., px=0., y=0.,
                                         py=0., zeta=0.)
            line.reset_s_at_end_turn = False
            incoming = particle(context, 2.4, charge_ratio=-1., mass_ratio=2.,
                x=x, y=y, px=.04, py=-.03, zeta=.001, delta=.03,
                ax=-chi*controls['ks']*y/2, ay=chi*controls['ks']*x/2)
            initial = values(incoming)
            initial_metadata = {name: np.asarray(getattr(incoming, name)).copy()
                                for name in metadata}
            initial_energy = np.asarray(incoming.energy).copy()
            line.track(incoming, ele_start=0, ele_stop=1)
            assert incoming.state[0] == 1, model
            results.append((values(incoming),
                {name: np.asarray(getattr(incoming, name)).copy()
                 for name in metadata}))
            np.testing.assert_allclose(incoming.energy, initial_energy,
                rtol=3e-13, atol=2e-7, err_msg=model+' static energy')
            line.track(incoming, ele_start=0, ele_stop=1, backtrack=True)
            assert incoming.state[0] == 1, model
            np.testing.assert_allclose(values(incoming), initial,
                rtol=0., atol=2e-12, err_msg=model+' static signed inverse')
            for name, expected in initial_metadata.items():
                np.testing.assert_allclose(getattr(incoming, name), expected,
                    rtol=0., atol=2e-12, err_msg=model+' inverse '+name)
    np.testing.assert_allclose(results[1][0], results[0][0],
        rtol=0., atol=2e-12, err_msg='same complete static Hamiltonian')
    for name in metadata:
        np.testing.assert_allclose(results[1][1][name], results[0][1][name],
            rtol=0., atol=2e-12, err_msg='same static readback '+name)


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
int profile_case(const double *a,int profile,double fraction,double *out){
 CavityModelSource s={0};
 const double k=a[0]>0.?2.*3.14159265358979323846*a[0]/299792458.:1.;
 int status=CavityModel_prepare_profile(a[0],a[1],a[2],a[3],a[4],a[5],
  a[6],a[7]*k,510998.95069,a[10],profile,a[8],a[9],&s);
 if(status!=CAVITY_MODEL_OK)return status;
 out[0]=s.k;out[1]=s.half_span;out[2]=s.forward;out[3]=s.backward;
 out[4]=s.clock_origin;out[5]=s.profile_sigma;out[6]=s.profile_center;
 CavityModel_source_envelope(fraction*s.half_span,&s,out+7);
 CavityModel_profile(fraction*s.half_span,&s,out+12,out+17);
 CavityModel_profile_integral(s.half_span,1./a[6]-s.carrier,&s,out+22);
 CavityModel_profile_integral(s.half_span,1./a[6]+s.carrier,&s,out+24);
 return status;
}
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
int mixed_case(const double *q,double Q,int model,double *out){
 CavityModelSource s={0};
 const double normal[4]={.017,.23,-.19,.13};
 const double skew[4]={-.01,.11,.07,-.17};
 s.k=.9;s.omega=299792458.*s.k;s.length=.43;
 s.half_span=s.k*s.length*.5;s.carrier=1.1;s.counter_phase=.3;
 s.forward=.04;s.backward=.017;s.magnetic_scale=.7;s.ks=.17;
 s.order=3;s.knl=normal;s.ksl=skew;
 const double z=.17*s.half_span;
 int status=CavityModel_physical_potential(q,z,&s,model,out);
 if(status!=1)return status;
 status=CavityModel_potential(q,z,&s,model,out+4);
 if(status!=1)return status;
 return CavityModel_flow(q,z,Q,&s,model,out+8);
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
    result.mixed_case.argtypes=[pointer,ctypes.c_double,ctypes.c_int,pointer]
    result.mixed_case.restype=ctypes.c_int
    result.profile_case.argtypes=[pointer,ctypes.c_int,ctypes.c_double,pointer]
    result.profile_case.restype=ctypes.c_int
    return result


def gaussian_fourier_integral(half_span, sigma, center, detuning):
    """Exact finite Gaussian Fourier integral, independent of source Gauss64."""
    from scipy.special import erf
    shift = 1j*detuning*sigma*sigma
    upper = (half_span-center-shift)/(np.sqrt(2)*sigma)
    lower = (-half_span-center-shift)/(np.sqrt(2)*sigma)
    return (sigma*np.sqrt(np.pi/2)
        *np.exp(1j*detuning*center-(detuning*sigma)**2/2)
        *(erf(upper)-erf(lower)))


def test_builtin_gaussian_source_and_crest_calibration(scalar_source_library):
    """Hermite jets and complex crest calibration retain live source knobs.

    The calibration ray has fixed beta; signed reference charge controls
    the accelerating-phase convention. A displaced Gaussian is not even,
    so its Fourier integral has an essential imaginary part. Its tails
    remain present beyond the finite observation interval.
    """
    # f, V, L, phase, eta, counterphase, beta_cal, nu, sigma_m, center_m, q0.
    base = np.array([400e6, 20e3, .23, .37, .25, .43,
                     .83, 1.17, .035, .012, 1.])
    cases = [base.copy()]
    for index, value in ((3, -.61), (4, 0.), (4, np.inf), (4, 1.3),
                         (5, -.8), (6, .71), (6, 1.), (8, .052),
                         (9, -.027), (10, -1.)):
        changed = base.copy()
        changed[index] = value
        cases.append(changed)
    for settings in cases:
        frequency, voltage, length, phase, eta, counterphase, beta, nu, \
            sigma_m, center_m, charge = settings
        k = 2*np.pi*frequency/C_LIGHT
        half_span, sigma, center = k*length/2, k*sigma_m, k*center_m
        if np.isinf(eta):
            forward, backward = 0., 1.
        else:
            forward = 1/(1+np.sqrt(eta))
            backward = np.sqrt(eta)/(1+np.sqrt(eta))
        slow_integral = gaussian_fourier_integral(
            half_span, sigma, center, 1/beta-nu)
        fast_integral = gaussian_fourier_integral(
            half_span, sigma, center, 1/beta+nu)
        crest = np.exp(1j*half_span/beta)*(forward*slow_integral
            +backward*np.exp(1j*counterphase)*fast_integral)
        amplitude = voltage/(MASS0*abs(crest))
        weights = amplitude*np.array([forward, backward])
        clock = phase-np.copysign(np.pi/2, charge)-np.angle(crest)
        for fraction in (-1.3, -.55, .17, .8, 1.2):
            observed = np.full(26, np.nan)
            status = scalar_source_library.profile_case(
                settings, 2, fraction, observed)
            assert status == 1, settings
            np.testing.assert_allclose(observed[:2], [k, half_span],
                                       rtol=2e-15, atol=0.)
            np.testing.assert_allclose(observed[2:4], weights,
                                       rtol=3e-13, atol=2e-15)
            np.testing.assert_allclose(np.exp(1j*observed[4]), np.exp(1j*clock),
                                       rtol=0., atol=3e-13)
            np.testing.assert_allclose(observed[5:7], [sigma, center],
                                       rtol=2e-15, atol=0.)
            z = fraction*half_span
            u = (z-center)/sigma
            envelope = np.exp(-u*u/2)
            derivatives = []
            for order in range(5):
                coefficient = np.zeros(order+1)
                coefficient[-1] = 1.
                derivatives.append((-1)**order*envelope
                    *np.polynomial.hermite_e.hermeval(u, coefficient)
                    /sigma**order)
            np.testing.assert_allclose(observed[7:12], derivatives,
                                       rtol=5e-13, atol=3e-14)
            # The source stores R with S=Re(R)*sin(T)-Im(R)*cos(T),
            # hence conjugate carrier signs relative to the physical
            # sin(T-nu*Z) and sin(T+nu*Z+counterphase) waves.
            expected = []
            for order in range(5):
                expected.append(sum(comb(order, power)*derivatives[order-power]
                    *(weights[0]*(1j*nu)**power*np.exp(1j*nu*z)
                      +weights[1]*(-1j*nu)**power
                      *np.exp(-1j*(nu*z+counterphase)))
                    for power in range(order+1)))
            np.testing.assert_allclose(observed[12:17]+1j*observed[17:22],
                expected, rtol=7e-13, atol=3e-14)
            np.testing.assert_allclose(observed[22]+1j*observed[23],
                slow_integral, rtol=3e-13, atol=3e-15)
            np.testing.assert_allclose(observed[24]+1j*observed[25],
                fast_integral, rtol=3e-13, atol=3e-15)
        # Independent fixed-beta longitudinal work, using the analytic
        # integral rather than the calibrated amplitude's own quadrature.
        gain = charge*np.real(np.exp(1j*(clock+half_span/beta))
            *(weights[0]*slow_integral
              +weights[1]*np.exp(1j*counterphase)*fast_integral))
        np.testing.assert_allclose(gain, abs(charge)*voltage/MASS0*np.sin(phase),
                                   rtol=4e-13, atol=3e-15)


def test_gaussian_profile_selection_is_stored_independently():
    """A model edit cannot silently replace the explicitly selected source."""
    default = xt.Cavity(length=.2, model='phase-resolved')
    assert default.rf_profile == 'p16'
    default.model = 'phase-averaged-full-slow'
    assert default.rf_profile == 'p16'
    gaussian = xt.Cavity(length=.2, model='phase-resolved',
        rf_profile='gaussian', rf_profile_sigma=.04, rf_profile_center=.01)
    gaussian.model = 'phase-resolved-longitudinal'
    assert gaussian.rf_profile == 'gaussian'
    restored = xt.Cavity.from_dict(gaussian.to_dict())
    assert restored.rf_profile == 'gaussian'
    assert restored.rf_profile_sigma == .04
    assert restored.rf_profile_center == .01
    with pytest.raises(ValueError, match='width'):
        xt.Cavity(length=.2, model='phase-resolved', rf_profile='gaussian')


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


def mixed_source_profile():
    """Differentiate the defining P16 polynomial, not the C recurrence."""
    span=.9*.43/2
    position=.17*span
    polynomial=np.polynomial.Polynomial([1.,0.,-1.])**16
    envelope=[polynomial.deriv(j)(position/span)/span**j for j in range(5)]
    return np.array([sum(weight*np.exp(1j*(nu*position+phase))*sum(
        comb(j,l)*envelope[l]*(1j*nu)**(j-l) for l in range(j+1))
        for weight,nu,phase in ((.04,1.1,0.),(.017,-1.1,-.3)))
        for j in range(5)])


def independent_mixed_potential(state,profile,model):
    """Analytic potential suitable for complex-step scalar differentiation.

    Static multipoles are explicit Cartesian polynomials, avoiding both the
    C factorial recurrence and real/imag operations on perturbed coordinates.
    """
    X,Y,time=state[:3]
    x,y=X/.9,Y/.9
    re=(x,x*x-y*y,x**3-3*x*y*y,x**4-6*x*x*y*y+y**4)
    im=(y,2*x*y,3*x*x*y-y**3,4*x*y*(x*x-y*y))
    normal=(.017,.23,-.19,.13)
    skew=(-.01,.11,.07,-.17)
    dtype=np.result_type(state,float)
    static=np.array([-.7*.17*y/2,.7*.17*x/2,
        -.7/.43*sum((n*r-s*i)/factorial(j+1)
            for j,(n,s,r,i) in enumerate(zip(normal,skew,re,im))),0.],
        dtype=dtype)
    sine=profile.real*np.sin(time)-profile.imag*np.cos(time)
    if model==12:
        return static+np.array([0.,0.,-sine[0],0.],dtype=dtype)
    radius2=X*X+Y*Y
    radial=sine[1]/2-radius2*(sine[1]+sine[3])/16
    rf=np.array([X*radial,Y*radial,
        -sine[0]+radius2*(sine[0]+sine[2])/4
        -radius2*radius2*(sine[0]+2*sine[2]+sine[4])/64,0.],dtype=dtype)
    return static+rf


def independent_mixed_hamiltonian(state,charge,profile,model):
    """Select the physical scalar before differentiating its actual root."""
    potential=independent_mixed_potential(state,profile,model)
    kinetic=state[3:5]-charge*potential[:2]
    square=state[5]*state[5]-1
    if model==11:
        momentum=np.sqrt(square)
        return -charge*potential[2]-momentum+(kinetic@kinetic)/(2*momentum)
    return -charge*potential[2]-np.sqrt(square-kinetic@kinetic)


def test_source_mixed_fields_independent_hamiltonian(scalar_source_library):
    """RF/static coupling comes from the summed potential inside each root."""
    profile=mixed_source_profile()
    for model in (10,11,12):
        for charge in (-1.3,.37,1.8):
            for x,y,time,energy in ((.012,-.009,.71,1.03),
                                    (-.023,.017,1.13,2.4)):
                state=np.array([x,y,time,.12,-.08,energy])
                actual=np.zeros(14)
                status=scalar_source_library.mixed_case(state,charge,model,actual)
                assert status==1,(model,charge,state,status)
                np.testing.assert_allclose(actual[:4],
                    independent_mixed_potential(state,profile,10),
                    rtol=3e-12,atol=2e-13)
                np.testing.assert_allclose(actual[4:8],
                    independent_mixed_potential(state,profile,model),
                    rtol=3e-12,atol=2e-13)
                gradient=np.empty(6)
                for j in range(6):
                    perturbed=state.astype(complex)
                    perturbed[j]+=1j*1e-28
                    gradient[j]=independent_mixed_hamiltonian(
                        perturbed,charge,profile,model).imag/1e-28
                flow=gradient[[3,4,5,0,1,2]]*np.array([1.,1.,-1.,-1.,-1.,1.])
                np.testing.assert_allclose(actual[8:],flow,
                    rtol=3e-11,atol=2e-12,
                    err_msg=repr((model,charge,state)))
