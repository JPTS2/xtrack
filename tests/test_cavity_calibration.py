"""Native RF crest calibration against the independent uniform-field integral.

The scalar oracle uses the analytic Fourier integral of the uniform envelope, not
a tracked calibration ray.
These tests check source normalisation and hardware derivatives; they do not
qualify the retained radial field or a particle transport error tolerance.
"""
import numpy as np
import pytest
import madng_tpsa
import xobjects as xo
import xtrack as xt
import xtrack.tpsa as xtpsa
from xtrack.base_element import FloatOrTpsa

from _cavity_context import kernel_context


MASS_EV = 510998.95069
LIGHT_SPEED = 299792458.


class _CalibrationProbe(xt.BeamElement):
    """Read the real source factory through scalar and actual TPSA kernels."""
    allow_rot_and_shift = False
    _xofields = {name: FloatOrTpsa for name in (
        'frequency', 'voltage', 'length', 'phase', 'counter_amplitude', 'counter_phase',
        'calibration_beta', 'wave_number')}
    _xofields.update(reference_charge=xo.Float64)
    _extra_c_sources = [r'''
        #include "xtrack/beam_elements/elements_src/cavity_models.h"
        GPUFUN
        void _CalibrationProbe_track_local_particle(
                _CalibrationProbeData el, LocalParticle* part0) {
            CavityModelSource source={0};
            int status=CavityModel_prepare(
                _CalibrationProbeData_get_frequency(el),
                _CalibrationProbeData_get_voltage(el),
                _CalibrationProbeData_get_length(el),
                _CalibrationProbeData_get_phase(el),
                _CalibrationProbeData_get_counter_amplitude(el),
                _CalibrationProbeData_get_counter_phase(el),
                _CalibrationProbeData_get_calibration_beta(el),
                _CalibrationProbeData_get_wave_number(el),
                510998.95069, _CalibrationProbeData_get_reference_charge(el),
                &source);
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


def hardware(**overrides):
    result = dict(frequency=400e6, voltage=2000., length=.05, phase=.4,
        counter_amplitude=.5, counter_phase=.3, calibration_beta=.9,
        reference_charge=1.)
    result.update(overrides)
    result.setdefault('wave_number', 2*np.pi*result['frequency']/LIGHT_SPEED)
    return result


def analytic_integral(half_span, detuning):
    """Integral of the uniform envelope times exp(+i detuning Z) over the parent."""
    return 2*half_span*np.sinc(detuning*half_span/np.pi)+0j


def analytic_source(settings):
    """Independent amplitude and clock, with signed public voltage/charge."""
    s = settings
    k = 2*np.pi*s['frequency']/LIGHT_SPEED
    half = k*s['length']/2
    nu = s['wave_number']/k
    ratio = s['counter_amplitude']
    forward = 0. if np.isinf(ratio) else 1/(1+ratio)
    backward = 1. if np.isinf(ratio) else ratio/(1+ratio)
    slow = analytic_integral(half, 1/s['calibration_beta']-nu)
    fast = analytic_integral(half, 1/s['calibration_beta']+nu)
    phasor = np.exp(1j*half/s['calibration_beta'])*(
        forward*slow+backward*np.exp(1j*s['counter_phase'])*fast)
    amplitude = s['voltage']/(MASS_EV*abs(phasor))
    clock = s['phase']-np.copysign(np.pi/2,s['reference_charge'])-np.angle(phasor)
    return np.array([amplitude*forward, amplitude*backward,
                     np.cos(clock), np.sin(clock)])


@pytest.fixture(scope='module')
def calibration_context(tmp_path_factory):
    with kernel_context(tmp_path_factory.mktemp('independent-calibration')) as context:
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
    hardware(frequency=2.856e9, length=3.05, counter_amplitude=1., calibration_beta=1.),
    hardware(frequency=2.856e9, length=5., counter_amplitude=1., calibration_beta=1.),
    hardware(frequency=2.856e9, length=3.05, counter_amplitude=.84, calibration_beta=.9),
    hardware(length=5., counter_amplitude=.55),
    hardware(length=1., counter_amplitude=0., calibration_beta=1.),
    hardware(length=.3, counter_amplitude=np.inf, calibration_beta=1.),
])
def test_uniform_crest_and_clock_against_analytic_integral(
        calibration_context, settings):
    observed = native_source(calibration_context, settings)
    expected = analytic_source(settings)
    np.testing.assert_allclose(observed[:2],expected[:2],rtol=2e-11,atol=2e-14)
    np.testing.assert_allclose(observed[2:],expected[2:],rtol=0.,atol=2e-11)


def test_nearly_cancelling_phasor_retains_relative_normalisation(calibration_context):
    settings=hardware(calibration_beta=1.)
    k=2*np.pi*settings['frequency']/LIGHT_SPEED
    half=k*settings['length']/2
    slow=analytic_integral(half,0.).real
    fast=analytic_integral(half,2.).real
    settings.update(counter_amplitude=slow/fast,counter_phase=np.pi-1e-3)
    np.testing.assert_allclose(native_source(calibration_context,settings),
                               analytic_source(settings),rtol=2e-11,atol=2e-11)


@pytest.mark.parametrize('calibration_beta', [.9, 1.])
def test_source_hardware_tpsa_derivatives_against_analytic_integral(
        calibration_context, calibration_beta):
    settings = hardware(length=.2, calibration_beta=calibration_beta)
    names = ('frequency','voltage','length','phase','counter_amplitude','counter_phase',
             'calibration_beta','wave_number')
    for name in names:
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

