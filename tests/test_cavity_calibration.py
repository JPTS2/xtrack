"""Native RF crest calibration against independent finite-profile integrals.

The scalar oracle uses the analytic P16 Fourier transform and finite Gaussian
complex-erf integral, not the native quadrature or a tracked calibration ray.
These tests check source normalisation and hardware derivatives; they do not
qualify the retained radial field or a particle transport error tolerance.
"""
import numpy as np
import pytest
from scipy.special import beta, erf, erfc, hyp0f1
import madng_tpsa
import xobjects as xo
import xtrack as xt
import xtrack.tpsa as xtpsa
from xtrack.base_element import FloatOrTpsa

from test_cavity_native import native_context


MASS_EV = 510998.95069
LIGHT_SPEED = 299792458.


class _CalibrationProbe(xt.BeamElement):
    """Read the real source factory through scalar and actual TPSA kernels."""
    allow_rot_and_shift = False
    _xofields = {name: FloatOrTpsa for name in (
        'frequency', 'voltage', 'length', 'phase', 'eta', 'counter_phase',
        'calibration_beta', 'wave_number', 'sigma', 'center')}
    _xofields.update(profile=xo.Int64, reference_charge=xo.Float64)
    _extra_c_sources = [r'''
        #include "xtrack/beam_elements/elements_src/cavity_models.h"
        GPUFUN
        void _CalibrationProbe_track_local_particle(
                _CalibrationProbeData el, LocalParticle* part0) {
            CavityModelSource source={0};
            int status=CavityModel_prepare_profile(
                _CalibrationProbeData_get_frequency(el),
                _CalibrationProbeData_get_voltage(el),
                _CalibrationProbeData_get_length(el),
                _CalibrationProbeData_get_phase(el),
                _CalibrationProbeData_get_eta(el),
                _CalibrationProbeData_get_counter_phase(el),
                _CalibrationProbeData_get_calibration_beta(el),
                _CalibrationProbeData_get_wave_number(el),
                510998.95069, _CalibrationProbeData_get_reference_charge(el),
                _CalibrationProbeData_get_profile(el),
                _CalibrationProbeData_get_sigma(el),
                _CalibrationProbeData_get_center(el), &source);
            START_PER_PARTICLE_BLOCK(part0, part)
                if(status==CAVITY_MODEL_OK) {
                    LocalParticle_set_x(part,source.forward);
                    LocalParticle_set_px(part,source.backward);
                    LocalParticle_set_y(part,cos(source.clock_origin));
                    LocalParticle_set_py(part,sin(source.clock_origin));
                } else LocalParticle_set_state(part,-99740+status);
            END_PER_PARTICLE_BLOCK;
        }
    ''']


def hardware(profile=0, **overrides):
    result = dict(frequency=400e6, voltage=2000., length=.05, phase=.4,
        eta=.25, counter_phase=.3, calibration_beta=.9, sigma=.0125,
        center=.003, profile=profile, reference_charge=1.)
    result.update(overrides)
    result.setdefault('wave_number', 2*np.pi*result['frequency']/LIGHT_SPEED)
    return result


def analytic_integral(half_span, detuning, profile, sigma, center):
    """Integral envelope(Z) exp(+i detuning Z) over the finite parent."""
    if profile == 0:
        return half_span*beta(.5, 17)*hyp0f1(
            17.5, -.25*(detuning*half_span)**2)+0j
    if profile == 1:
        return 2*half_span*np.sinc(detuning*half_span/np.pi)+0j
    scale = np.sqrt(2)*sigma
    upper = (half_span-center-1j*detuning*sigma**2)/scale
    lower = (-half_span-center-1j*detuning*sigma**2)/scale
    # Same-sign tails must not subtract two erf values rounded to +/-1.
    if upper.real <= 0.:
        difference=erfc(-upper)-erfc(-lower)
    elif lower.real >= 0.:
        difference=erfc(lower)-erfc(upper)
    else:
        difference=erf(upper)-erf(lower)
    return (sigma*np.sqrt(np.pi/2)*np.exp(
        -.5*(detuning*sigma)**2+1j*detuning*center)*difference)


def analytic_source(settings):
    """Independent amplitude and clock, with signed public voltage/charge."""
    s = settings
    k = 2*np.pi*s['frequency']/LIGHT_SPEED
    half = k*s['length']/2
    nu = s['wave_number']/k
    ratio = np.sqrt(s['eta'])
    forward = 0. if np.isinf(ratio) else 1/(1+ratio)
    backward = 1. if np.isinf(ratio) else ratio/(1+ratio)
    slow = analytic_integral(half, 1/s['calibration_beta']-nu,
        s['profile'], k*s['sigma'], k*s['center'])
    fast = analytic_integral(half, 1/s['calibration_beta']+nu,
        s['profile'], k*s['sigma'], k*s['center'])
    phasor = np.exp(1j*half/s['calibration_beta'])*(
        forward*slow+backward*np.exp(1j*s['counter_phase'])*fast)
    amplitude = s['voltage']/(MASS_EV*abs(phasor))
    clock = s['phase']-np.copysign(np.pi/2,s['reference_charge'])-np.angle(phasor)
    return np.array([amplitude*forward, amplitude*backward,
                     np.cos(clock), np.sin(clock)])


@pytest.fixture(scope='module')
def calibration_context(tmp_path_factory):
    with native_context(tmp_path_factory.mktemp('independent-calibration')) as context:
        yield context


def native_source(context, settings):
    element = _CalibrationProbe(_context=context, **settings)
    particle = xt.Particles(_context=context, p0c=2e6, mass0=MASS_EV, q0=1.)
    element.track(particle)
    assert particle.state[0] == 1
    return np.array([getattr(particle,name)[0] for name in ('x','px','y','py')])


@pytest.mark.parametrize('settings', [
    hardware(),
    hardware(reference_charge=-1., voltage=-2000.),
    hardware(frequency=2.856e9, length=3.05, eta=1., calibration_beta=1.),
    hardware(frequency=2.856e9, length=5., eta=1., calibration_beta=1.),
    hardware(frequency=2.856e9, length=3.05, eta=.7, calibration_beta=.9),
    hardware(profile=2, length=1., sigma=.005, center=0., eta=0., calibration_beta=1.),
    hardware(profile=2, length=1., sigma=.01, center=0., eta=0., calibration_beta=1.),
    hardware(profile=2, length=1., sigma=.005, center=.023, eta=.25),
    hardware(profile=2, length=.2, sigma=.025, center=-.013, eta=1.),
    hardware(profile=2, length=.2, sigma=.025, center=.11, eta=.4),
    hardware(profile=2, length=1., sigma=.005, center=.55, eta=.25),
    hardware(profile=1, length=5., eta=.3),
    # Pure counter-wave makes the Bessel/0F1 switch visible in normalisation.
    *[hardware(length=x*LIGHT_SPEED/(2*np.pi*400e6),eta=np.inf,
               calibration_beta=1.) for x in (15.999,16.,16.001)],
])
def test_finite_profile_crest_and_clock_against_analytic_integral(
        calibration_context, settings):
    observed = native_source(calibration_context, settings)
    expected = analytic_source(settings)
    np.testing.assert_allclose(observed[:2],expected[:2],rtol=2e-11,atol=2e-14)
    np.testing.assert_allclose(observed[2:],expected[2:],rtol=0.,atol=2e-11)


def test_nearly_cancelling_phasor_retains_relative_normalisation(calibration_context):
    settings=hardware(calibration_beta=1.)
    k=2*np.pi*settings['frequency']/LIGHT_SPEED
    half=k*settings['length']/2
    slow=analytic_integral(half,0.,0,0.,0.).real
    fast=analytic_integral(half,2.,0,0.,0.).real
    settings.update(eta=(slow/fast)**2,counter_phase=np.pi-1e-3)
    np.testing.assert_allclose(native_source(calibration_context,settings),
                               analytic_source(settings),rtol=2e-11,atol=2e-11)


@pytest.mark.parametrize('profile,calibration_beta', [(0,.9),(2,.9),(2,1.)])
def test_source_hardware_tpsa_derivatives_against_analytic_integral(
        calibration_context, profile, calibration_beta):
    settings = hardware(profile=profile, length=.2, sigma=.008, center=.017,
                        calibration_beta=calibration_beta)
    names = ('frequency','voltage','length','phase','eta','counter_phase',
             'calibration_beta','wave_number','sigma','center')
    for name in names:
        if profile == 0 and name in ('sigma','center'):
            continue
        scale = abs(settings[name]) or .01
        element = _CalibrationProbe(_context=calibration_context, **settings)
        desc = madng_tpsa.Descriptor(6,2,num_params=1,param_order=2)
        setattr(element,name, settings[name]+scale*desc.param(1,0.))
        series = xtpsa.ParticlesTpsa(order=2,descriptor=desc,
                                    p0c=2e6,mass0=MASS_EV,q0=1.)
        element.track(series)
        assert series._xobject.state == 1
        np.testing.assert_allclose(series.const_part[:4],analytic_source(settings),
                                   rtol=2e-11,atol=2e-11)
        h = 2e-4
        finite = sum(weight*analytic_source(dict(settings,**{
            name:settings[name]+offset*h*scale})) for offset,weight in
            ((-2,1.),(-1,-8.),(1,8.),(2,-1.)))/(12*h)
        np.testing.assert_allclose(series.param_jacobian()[:4,0],finite,
            rtol=3e-7,atol=2e-9,err_msg=name)


def test_gaussian_width_center_mixed_tpsa_derivative(calibration_context):
    settings = hardware(profile=2,length=.2,sigma=.008,center=.017)
    element = _CalibrationProbe(_context=calibration_context,**settings)
    desc = madng_tpsa.Descriptor(6,2,num_params=2,param_order=2)
    scales = (settings['sigma'],settings['center'])
    element.sigma=settings['sigma']+scales[0]*desc.param(1,0.)
    element.center=settings['center']+scales[1]*desc.param(2,0.)
    series=xtpsa.ParticlesTpsa(order=2,descriptor=desc,p0c=2e6,mass0=MASS_EV,q0=1.)
    element.track(series)
    assert series._xobject.state == 1
    powers=(0,0,0,0,0,0,1,1)
    actual=np.array([series.coefficient(name,powers) for name in ('x','px','y','py')])
    h=1e-3
    finite=sum(a*b*analytic_source(dict(settings,
        sigma=settings['sigma']+a*h*scales[0],
        center=settings['center']+b*h*scales[1]))
        for a in (-1,1) for b in (-1,1))/(4*h*h)
    np.testing.assert_allclose(actual,finite,rtol=2e-5,atol=2e-7)


def test_seeded_synchronous_gaussian_detuning_keeps_derivatives(calibration_context):
    settings=hardware(profile=2,length=1.,sigma=.005,center=.023,
                      eta=0.,calibration_beta=1.)
    element=_CalibrationProbe(_context=calibration_context,**settings)
    desc=madng_tpsa.Descriptor(6,2,num_params=1,param_order=2)
    scale=settings['wave_number']
    element.wave_number=settings['wave_number']+scale*desc.param(1,0.)
    series=xtpsa.ParticlesTpsa(order=2,descriptor=desc,p0c=2e6,mass0=MASS_EV,q0=1.)
    element.track(series)
    assert series._xobject.state == 1
    h=1e-3
    values={n:analytic_source(dict(settings,
        wave_number=settings['wave_number']+n*h*scale)) for n in (-2,-1,0,1,2)}
    first=(values[-2]-8*values[-1]+8*values[1]-values[2])/(12*h)
    second=(-values[-2]+16*values[-1]-30*values[0]+16*values[1]-values[2])/(12*h*h)
    np.testing.assert_allclose(series.param_jacobian()[:4,0],first,rtol=3e-7,atol=2e-9)
    actual=np.array([series.coefficient(name,(0,0,0,0,0,0,2))
                     for name in ('x','px','y','py')])
    np.testing.assert_allclose(actual,second/2,rtol=3e-5,atol=2e-8)


def test_seeded_zero_p16_detuning_retains_fourth_order(calibration_context):
    settings=hardware(length=.2,eta=0.,calibration_beta=1.)
    element=_CalibrationProbe(_context=calibration_context,**settings)
    desc=madng_tpsa.Descriptor(6,4,num_params=1,param_order=4)
    element.wave_number=settings['wave_number']*(1.+desc.param(1,0.))
    series=xtpsa.ParticlesTpsa(order=4,descriptor=desc,p0c=2e6,mass0=MASS_EV,q0=1.)
    element.track(series)
    assert series._xobject.state == 1
    half=settings['wave_number']*settings['length']/2
    constant=analytic_source(settings)[0]
    # Reciprocal of 0F1(;b;-(H*parameter)^2/4), b=35/2.
    quadratic=half**2/70
    quartic=quadratic**2-half**4/(32*17.5*18.5)
    assert series.coefficient('x',(0,0,0,0,0,0,2)) == pytest.approx(
        constant*quadratic,rel=2e-11,abs=2e-14)
    assert series.coefficient('x',(0,0,0,0,0,0,4)) == pytest.approx(
        constant*quartic,rel=2e-10,abs=2e-14)


@pytest.mark.parametrize('argument',[15.999,16.,16.001])
def test_p16_switch_actual_tpsa_derivatives(calibration_context,argument):
    settings=hardware(length=argument*LIGHT_SPEED/(2*np.pi*400e6),
                      eta=np.inf,calibration_beta=1.)
    element=_CalibrationProbe(_context=calibration_context,**settings)
    desc=madng_tpsa.Descriptor(6,2,num_params=1,param_order=2)
    scale=settings['wave_number']
    element.wave_number=scale*(1.+desc.param(1,0.))
    series=xtpsa.ParticlesTpsa(order=2,descriptor=desc,p0c=2e6,mass0=MASS_EV,q0=1.)
    element.track(series)
    assert series._xobject.state == 1
    np.testing.assert_allclose(series.const_part[:4],analytic_source(settings),
                               rtol=2e-11,atol=2e-11)
    h=1e-3
    values={n:analytic_source(dict(settings,wave_number=scale*(1+n*h)))
            for n in (-2,-1,0,1,2)}
    first=(values[-2]-8*values[-1]+8*values[1]-values[2])/(12*h)
    second=(-values[-2]+16*values[-1]-30*values[0]+16*values[1]-values[2])/(12*h*h)
    np.testing.assert_allclose(series.param_jacobian()[:4,0],first,rtol=2e-6,atol=2e-9)
    actual=np.array([series.coefficient(name,(0,0,0,0,0,0,2))
                     for name in ('x','px','y','py')])
    np.testing.assert_allclose(actual,second/2,rtol=2e-5,atol=3e-8)


def test_broad_offset_gaussian_uses_well_conditioned_normalisation(calibration_context):
    settings=hardware(profile=2,length=1.,sigma=1e6,center=1e6,
                      eta=0.,calibration_beta=1.)
    observed=native_source(calibration_context,settings)
    k=2*np.pi*settings['frequency']/LIGHT_SPEED
    half=k*settings['length']/2
    # With C/S=1, exp(-(Z-C)^2/(2S²)) has zero quadratic coefficient.
    # Its finite symmetric integral is 2H exp(-1/2)*(1-H^4/(60S^4)+...).
    # The omitted relative correction here is below1e-28, unlike erfc
    # subtraction of two almost-equal finite numbers in a binary64 oracle.
    integral=2*half*np.exp(-.5)
    amplitude=settings['voltage']/(MASS_EV*integral)
    clock=settings['phase']-np.pi/2-half
    expected=np.array([amplitude,0.,np.cos(clock),np.sin(clock)])
    np.testing.assert_allclose(observed,expected,rtol=2e-13,atol=2e-14)


def test_unresolved_gaussian_calibration_refuses_without_coordinate_writes(
        calibration_context):
    settings=hardware(profile=2,length=1.,sigma=.1,wave_number=1e12)
    element=_CalibrationProbe(_context=calibration_context,**settings)
    particle=xt.Particles(_context=calibration_context,p0c=2e6,mass0=MASS_EV,q0=1.,
        x=.001,px=.002,y=-.003,py=-.004,zeta=.005,delta=.006)
    names=('x','px','y','py','zeta','delta')
    before=np.array([getattr(particle,name)[0] for name in names])
    element.track(particle)
    assert particle.state[0] == -99742
    np.testing.assert_array_equal(
        np.array([getattr(particle,name)[0] for name in names]),before)
