"""Cavity tracking and source-parameter derivatives in the existing TPSA backend.

Finite differences below independently check the executed maps. They are not
the implementation of TPSA tracking. The generic-array test checks the active
descriptor lifecycle needed by cavity source/stage aggregate storage.
"""
import numpy as np
import pytest

import madng_tpsa
import xtrack as xt
import xtrack.tpsa as xtpsa
from xtrack.base_element import FloatOrTpsa
from xtrack.tpsa._knobs import KnobParameters
from test_cavity_native import native_context


COORDS = ('x', 'px', 'y', 'py', 'zeta', 'delta')
LAUNCH = dict(p0c=14.208754281066904e6, mass0=xt.ELECTRON_MASS_EV,
              q0=1., x=2e-4, px=3e-4, y=-1e-4, py=-2e-4,
              zeta=4e-4, delta=2e-3, time_s=1.9e-9)


class _DefaultSeriesAggregate(xt.BeamElement):
    """A polynomial with known mixed coefficients, without a cavity donor."""
    _xofields = {'strength': FloatOrTpsa}
    _extra_c_sources = [r'''
        #include "xtrack/headers/track.h"
        #include "xtrack/beam_elements/elements_src/cavity_models_compat.h"
        GPUFUN
        void _DefaultSeriesAggregate_track_local_particle(
                _DefaultSeriesAggregateData el, LocalParticle* part0) {
            START_PER_PARTICLE_BLOCK(part0, part)
                #ifdef XTRACK_TPSA_TRACK
                struct Stage { xt_float_or_tpsa value; xt_float_or_tpsa zero; };
                struct Stage stage;
                xt_float_or_tpsa storage[2];
                storage[0] = LocalParticle_get_x(part);
                stage.value = _DefaultSeriesAggregateData_get_strength(el)
                    * storage[0] * storage[0];
                LocalParticle_set_x(part,stage.value+stage.zero+storage[1]);
                if (!(CavityModel_scalar_finite(stage.value)
                        && CavityModel_scalar_norm(storage[0])>0.))
                    LocalParticle_set_state(part,-1);
                #endif
            END_PER_PARTICLE_BLOCK;
        }
    ''']


def test_generic_series_arrays_follow_active_descriptor(tmp_path):
    with native_context(tmp_path) as context:
        for order, param_order in ((1, 1), (3, 2)):
            desc = madng_tpsa.Descriptor(6, order, num_params=1,
                                         param_order=param_order)
            element = _DefaultSeriesAggregate(_context=context, strength=2.)
            element.strength = desc.param(1, 2.)
            particles = xtpsa.ParticlesTpsa(order=order, descriptor=desc,
                                           p0c=7e12, x=.25)
            element.track(particles)
            assert particles._xobject.state == 1
            assert particles.const_part[0] == pytest.approx(.125)
            assert particles.jacobian()[0, 0] == pytest.approx(1.)
            assert particles.param_jacobian()[0, 0] == pytest.approx(.0625)
            if order >= 2:
                assert particles.coefficient('x', (2,0,0,0,0,0,0)) == 2.
                assert particles.coefficient('x', (1,0,0,0,0,0,1)) == .5


def _values(particles):
    return np.array([getattr(particles, name)[0] for name in COORDS])


@pytest.mark.parametrize('absolute_time', [False, True])
def test_legacy_cavity_tpsa_clock_and_phase_knob(tmp_path, absolute_time):
    """Same physical clock, including a nonzero cavity position and time_s."""
    with native_context(tmp_path) as context:
        line = xt.Line(elements=[xt.Drift(length=.17),
            xt.Cavity(voltage=2e5, frequency=400e6, phase=.7,
                      absolute_time=absolute_time)],
            element_names=['d', 'c'])
        line.particle_ref = xt.Particles(p0c=LAUNCH['p0c'],
            mass0=LAUNCH['mass0'], q0=1.)
        line.reset_s_at_end_turn = False
        line.build_tracker(_context=context)
        scalar = xt.Particles(_context=context, **LAUNCH)
        line.track(scalar)
        desc = madng_tpsa.Descriptor(6, 3, num_params=1, param_order=2)
        line['c'].phase = desc.param(1, .7)
        particles = xtpsa.ParticlesTpsa(order=3, descriptor=desc, **LAUNCH)
        line.track(particles)
        np.testing.assert_allclose(particles.const_part, _values(scalar),
                                   rtol=0., atol=3e-14)
        gradient = particles.param_jacobian()[:, 0]
        line.disable_tpsa_elements()
        endpoints = []
        for phase in (.7-1e-5, .7+1e-5):
            line['c'].phase = phase
            probe = xt.Particles(_context=context, **LAUNCH)
            line.track(probe)
            endpoints.append(_values(probe))
        derivative = (endpoints[1]-endpoints[0])/(2e-5)
        np.testing.assert_allclose(gradient, derivative,
                                   rtol=2e-7, atol=2e-10)


def test_legacy_cavity_tpsa_knob_refresh_and_teardown(tmp_path):
    with native_context(tmp_path) as context:
        line = xt.Line(elements=[xt.Cavity(voltage=2e5, frequency=400e6)],
                       element_names=['c'])
        line.particle_ref = xt.Particles(p0c=LAUNCH['p0c'],
            mass0=LAUNCH['mass0'], q0=1.)
        line['rf_phase'] = .4
        line['c'].phase = line.ref['rf_phase']
        line.build_tracker(_context=context)
        desc = madng_tpsa.Descriptor(6, 2, num_params=1, param_order=1)
        knobs = KnobParameters(line, ['rf_phase'], desc)
        knobs.apply()
        first = xtpsa.ParticlesTpsa(order=2, descriptor=desc, **LAUNCH)
        line.track(first)
        knobs.refresh([.8])
        second = xtpsa.ParticlesTpsa(order=2, descriptor=desc, **LAUNCH)
        line.track(second)
        assert second.const_part[5] != first.const_part[5]
        assert second.param_jacobian()[5, 0] != 0.
        knobs.teardown()
        assert not line['c']._tpsa_enabled
        restored = xt.Line.from_dict(line.to_dict())
        assert restored['c'].phase == pytest.approx(.8)


@pytest.mark.parametrize('carrier', [None, 0.])
def test_cavity_tpsa_renamed_carrier_storage(carrier):
    """Promoting backing slots must preserve a derived/explicit public input."""
    line = xt.Line(elements=[xt.Cavity(wave_number=carrier)],
                   element_names=['c'])
    line['phase_knob'] = .4
    line['c'].phase = line.ref['phase_knob']
    desc = madng_tpsa.Descriptor(6, 2, num_params=1, param_order=1)
    knobs = KnobParameters(line, ['phase_knob'], desc)
    knobs.apply()
    assert line['c'].wave_number_from_frequency == (carrier is None)
    if carrier is None:
        assert line['c'].wave_number is None
    else:
        assert line['c'].wave_number.const_part == 0.
    with pytest.raises(ValueError, match='one value'):
        knobs.refresh([.8, .9])
    assert line['c'].phase.const_part == .4
    knobs.refresh([.8])
    knobs.teardown()
    if carrier is None:
        assert line['c'].wave_number is None
    else:
        assert line['c'].wave_number == 0.
    assert line['c'].phase == .8


def test_disable_tpsa_only_used_environment_owners():
    line = xt.Line(elements={'used':xt.Cavity(), 'other':xt.Cavity()},
                   element_names=['used', 'used'])
    desc = madng_tpsa.Descriptor(6, 1, num_params=1, param_order=1)
    line['used'].phase=desc.param(1,.3)
    line['other'].phase=desc.param(1,.6)
    line.disable_tpsa_elements()
    assert not line['used']._tpsa_enabled
    assert line['other']._tpsa_enabled
    assert line['used'].phase == .3
    assert line.tracker is None
    line['other'].disable_tpsa()


def test_disable_tpsa_reaches_used_slice_parent():
    line = xt.Line(elements={'rf':xt.Cavity(length=.05),
                             'other':xt.Cavity()}, element_names=['rf'])
    line.slice_thick_elements([xt.Strategy(
        slicing=xt.Teapot(2,mode='thick'),element_type=xt.Cavity)])
    desc = madng_tpsa.Descriptor(6, 1, num_params=1, param_order=1)
    line['rf'].phase=desc.param(1,.3)
    line['other'].phase=desc.param(1,.6)
    assert 'rf' not in line.element_names
    line.disable_tpsa_elements()
    assert not line['rf']._tpsa_enabled
    assert line['other']._tpsa_enabled
    line.disable_tpsa_elements()  # idempotent with repeated parent links
    line['other'].disable_tpsa()


def test_legacy_standalone_cavity_nonzero_s_tpsa(tmp_path):
    with native_context(tmp_path) as context:
        element = xt.Cavity(_context=context, voltage=2e5, frequency=400e6,
                            phase=.7, absolute_time=True)
        launch = dict(LAUNCH, s=.17)
        scalar = xt.Particles(_context=context, **launch)
        series = xtpsa.ParticlesTpsa(order=1, **launch)
        element.track(scalar)
        element.track(series)
        np.testing.assert_allclose(series.const_part, _values(scalar),
                                   rtol=0., atol=3e-14)


@pytest.mark.parametrize('model', ['phase-resolved',
    'phase-resolved-paraxial', 'phase-resolved-longitudinal'])
def test_source_cavity_tpsa_eta_and_phase(tmp_path, model):
    """Nontrivial calibration and two waves expose hidden frozen preparation."""
    mass = xt.ELECTRON_MASS_EV
    launch = dict(LAUNCH, p0c=mass*np.sqrt(2.4**2-1),
                  x=4e-4, y=-3e-4, px=.005/np.sqrt(2.4**2-1),
                  py=-.004/np.sqrt(2.4**2-1))
    with native_context(tmp_path) as context:
        line = xt.Line(elements=[xt.Cavity(length=.05, voltage=2e4,
            frequency=400e6, phase=.4, eta=.25, counter_phase=.3,
            calibration_beta=.9, model=model, integrator='gauss2',
            num_kicks=8)], element_names=['c'])
        line.particle_ref = xt.Particles(p0c=launch['p0c'], mass0=mass, q0=1.)
        line.reset_s_at_end_turn=False
        line.build_tracker(_context=context)
        scalar = xt.Particles(_context=context, **launch)
        line.track(scalar)
        assert scalar.state[0] == 1
        desc = madng_tpsa.Descriptor(6, 2, num_params=2, param_order=1)
        line['c'].eta = desc.param(1, .25)
        line['c'].phase = desc.param(2, .4)
        series = xtpsa.ParticlesTpsa(order=2, descriptor=desc, **launch)
        line.track(series)
        assert series._xobject.state == 1
        np.testing.assert_allclose(series.const_part, _values(scalar),
                                   rtol=0., atol=3e-13)
        gradients = series.param_jacobian().copy()
        line.disable_tpsa_elements()
        for index,(name,value,step) in enumerate([
                ('eta',.25,1e-5), ('phase',.4,1e-5)]):
            endpoints=[]
            for sign in (-1,1):
                setattr(line['c'],name,value+sign*step)
                particle=xt.Particles(_context=context,**launch)
                line.track(particle)
                assert particle.state[0] == 1
                endpoints.append(_values(particle))
            setattr(line['c'],name,value)
            fd=(endpoints[1]-endpoints[0])/(2*step)
            np.testing.assert_allclose(gradients[:,index],fd,
                                       rtol=2e-6,atol=3e-10)
