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
# Independently chosen synthetic launch; no machine momentum is embedded.
LAUNCH = dict(p0c=14e6, mass0=xt.ELECTRON_MASS_EV,
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


@pytest.mark.parametrize('order',[2,3])
@pytest.mark.parametrize('delta',[-.2,.002])
def test_initial_energy_metadata_series(order,delta):
    """Input energy auxiliaries satisfy their physical delta constraints."""
    series=xtpsa.ParticlesTpsa(order=order,p0c=1e9,delta=delta,s=.17,ax=.025,ay=-.017)
    beta0=float(series._xobject.beta0)
    root=np.sqrt(beta0**2*(1.+delta)**2+1.-beta0**2)
    rvv=(1.+delta)/root
    np.testing.assert_allclose(series._local_series['rpp'].grad()[5],-1./(1.+delta)**2,rtol=2e-14)
    np.testing.assert_allclose(series._local_series['pzeta'].grad()[5],rvv,rtol=2e-14)
    np.testing.assert_allclose(series._local_series['rvv'].grad()[5],(1.-beta0**2)/root**3,rtol=2e-14)
    powers=[0]*6;powers[5]=2
    np.testing.assert_allclose(series._local_series['rpp'].get(powers),1./(1.+delta)**3,rtol=2e-14)
    np.testing.assert_allclose(series._local_series['pzeta'].get(powers),.5*(1.-beta0**2)/root**3,rtol=2e-14)
    if order==3:
        powers[5]=3
        np.testing.assert_allclose(series._local_series['rpp'].get(powers),-1./(1.+delta)**4,rtol=2e-14)
        np.testing.assert_allclose(series._local_series['pzeta'].get(powers),
            -.5*beta0**2*(1.-beta0**2)*(1.+delta)/root**5,rtol=2e-14)
    assert series._local_series['s'].const_part==.17
    assert series._local_series['ax'].const_part==.025
    assert series._local_series['ay'].const_part==-.017


def test_frozen_energy_drift_uses_input_delta_series(tmp_path):
    with native_context(tmp_path) as context:
        line=xt.Line(elements={'d':xt.Drift(length=.17)})
        line.particle_ref=xt.Particles(p0c=1e9)
        line.reset_s_at_end_turn=False
        line.freeze_energy(force=True)
        line.build_tracker(_context=context)
        launch=dict(p0c=1e9,x=.001,px=.01,y=-.002,py=-.008,zeta=0.,delta=.002)
        desc=madng_tpsa.Descriptor(6,2)
        series=xtpsa.ParticlesTpsa(order=2,descriptor=desc,**launch)
        line.track(series)
        matrix=series.jacobian()
        finite=np.zeros((6,6))
        for column,name in enumerate(COORDS):
            for offset,weight in {-2:1.,-1:-8.,1:8.,2:-1.}.items():
                particle=xt.Particles(_context=context,**dict(launch,**{name:launch[name]+offset*1e-5}))
                line.track(particle)
                finite[:,column]+=weight*_values(particle)/(12e-5)
        np.testing.assert_allclose(matrix,finite,rtol=3e-6,atol=3e-8)


def test_field_getter_preserves_orders_and_rejects_incompatible_handles(tmp_path):
    """Exercise the real generated FoT getter, including its no-truncation guard."""
    with native_context(tmp_path) as context:
        desc = madng_tpsa.Descriptor(6, 3, num_params=1, param_order=3)
        element = _DefaultSeriesAggregate(_context=context, strength=2.)
        parameter = desc.param(1, 0.)
        element.strength = 2. + parameter + .3*parameter**2 + .7*parameter**3
        particles = xtpsa.ParticlesTpsa(order=3, descriptor=desc, p0c=7e12, x=.25)
        element.track(particles)
        assert particles.coefficient('x', (0,0,0,0,0,0,3)) == pytest.approx(.7*.25**2)
        assert particles.coefficient('x', (1,0,0,0,0,0,2)) == pytest.approx(.3*.5)
        # Inject a lower-storage-order active prototype deliberately. Normal
        # ParticlesTpsa construction prevents this shape; the native getter must
        # still refuse it rather than silently discard the source's high terms.
        low = madng_tpsa.Tpsa(desc, order=1)
        particles = xtpsa.ParticlesTpsa(order=3, descriptor=desc, p0c=7e12, x=.25)
        particles.coords[0] = low
        particles._xobject.x = int(madng_tpsa.ffi.cast('uintptr_t', low.ptr))
        # The tracking boundary intentionally translates native tracking_error
        # exceptions into the standard lost-map status and Python message.
        with pytest.raises(RuntimeError, match='TPSA map lost'):
            element.track(particles)
        assert particles._xobject.state == -50
        # MAD interns equivalent descriptors; change the actual schema rather
        # than merely constructing a second Python wrapper for the same one.
        other = madng_tpsa.Descriptor(6, 3, num_params=2, param_order=3)
        particles = xtpsa.ParticlesTpsa(order=3, descriptor=other, p0c=7e12, x=.25)
        with pytest.raises(RuntimeError, match='TPSA map lost'):
            element.track(particles)
        assert particles._xobject.state == -50


@pytest.mark.parametrize('order,model', [(2,'phase-averaged-full-slow'),
                                      (3,'phase-resolved')])
def test_calibration_order_one_handle_on_higher_order_map(cavity_magnetic_context, order, model):
    """Descriptor.param's direct order-one handle must not alter source constants."""
    # Full-slow order-three convergence is a separate recorded limitation;
    # its N2/N4 finite implicit solves currently refuse with -99742. This test
    # isolates the common field bridge at orders two and three without claiming
    # that every model/control/mesh combination is thereby qualified.
    context = cavity_magnetic_context
    launch = dict(LAUNCH, p0c=xt.ELECTRON_MASS_EV*np.sqrt(2.4**2-1))
    line = xt.Line(elements={'c':xt.Cavity(length=.167, voltage=2e4,
        frequency=400e6, phase=.4, eta=.25, counter_phase=.3,
        calibration_beta=.9, model=model,
        num_kicks=2 if model=='phase-averaged-full-slow' else 8)})
    line.particle_ref = xt.Particles(p0c=launch['p0c'], mass0=launch['mass0'], q0=1.)
    line.reset_s_at_end_turn = False
    _build_shared_cavity_tracker(line, context)
    scalar = xt.Particles(_context=context, **launch)
    line.track(scalar)
    desc = madng_tpsa.Descriptor(6, order, num_params=1, param_order=1)
    source = desc.param(1, .9)
    assert madng_tpsa.lib.mad_tpsa_ord(source.ptr, 0) == 1
    line['c'].calibration_beta = source
    series = xtpsa.ParticlesTpsa(order=order, descriptor=desc, **launch)
    try:
        line.track(series)
    except RuntimeError as error:
        raise AssertionError(f'calibration order {order}, native status {series._xobject.state}') from error
    np.testing.assert_allclose(series.const_part, _values(scalar), rtol=0., atol=3e-12)
    gradient = series.param_jacobian()[:,0].copy()
    mixed = series.coefficient('x', (0,1,0,0,0,0,1))
    line.disable_tpsa_elements()
    finite = np.zeros(6)
    finite_mixed = 0.
    weights = {-2:1., -1:-8., 1:8., 2:-1.}
    for ib, wb in weights.items():
        line['c'].calibration_beta = .9 + ib*1e-4
        particle = xt.Particles(_context=context, **launch)
        line.track(particle)
        finite += wb*_values(particle)/(12e-4)
        for ip, wp in weights.items():
            particle = xt.Particles(_context=context, **dict(launch,px=launch['px']+ip*1e-3))
            line.track(particle)
            finite_mixed += wb*wp*float(particle.x[0])/(144*1e-4*1e-3)
    np.testing.assert_allclose(gradient, finite, rtol=3e-6, atol=3e-8)
    np.testing.assert_allclose(mixed, finite_mixed, rtol=3e-5, atol=3e-8)


def test_frozen_fixed_nominal_gain_with_absolute_clock(cavity_magnetic_context):
    """RF phase and an independent eV/reference programme are separate knobs."""
    context=cavity_magnetic_context
    launch=dict(LAUNCH,p0c=xt.ELECTRON_MASS_EV*np.sqrt(2.4**2-1),s=.17)
    line=xt.Line(elements={'c':xt.Cavity(
        length=.167, voltage=2e4,frequency=400e6,phase=.4,eta=0.,
        model='phase-resolved-frozen',absolute_time=True,num_kicks=4,
        nominal_energy_gain=8e3)})
    line.particle_ref=xt.Particles(p0c=launch['p0c'],mass0=launch['mass0'],q0=1.)
    line.reset_s_at_end_turn=False
    line.use_time_based_zeta_update=True
    _build_shared_cavity_tracker(line,context)
    scalar=xt.Particles(_context=context,**launch);line.track(scalar)
    assert scalar.state[0]==1
    desc=madng_tpsa.Descriptor(6,2,num_params=2,param_order=1)
    line['c'].phase=desc.param(1,.4)
    line['c'].nominal_energy_gain=desc.param(2,8e3)
    series=xtpsa.ParticlesTpsa(order=2,descriptor=desc,**launch)
    line.track(series)
    np.testing.assert_allclose(series.const_part,_values(scalar),rtol=0.,atol=3e-12)
    gradients=series.param_jacobian().copy()
    line.disable_tpsa_elements()
    for column,(name,value,scale) in enumerate((('phase',.4,1.),
                                              ('nominal_energy_gain',8e3,8e3))):
        finite=np.zeros(6)
        for offset,weight in {-2:1.,-1:-8.,1:8.,2:-1.}.items():
            setattr(line['c'],name,value+offset*scale*1e-4)
            particle=xt.Particles(_context=context,**launch);line.track(particle)
            finite+=weight*_values(particle)/(12e-4)
        setattr(line['c'],name,value)
        np.testing.assert_allclose(gradients[:,column]*scale,finite,rtol=3e-6,atol=3e-8)
    restored=xt.Line.from_dict(line.to_dict())
    assert restored['c'].nominal_energy_gain==8e3
    assert not restored['c'].nominal_energy_gain_from_rf


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


def test_knob_teardown_preserves_preexisting_unrelated_map():
    line=xt.Line(elements={'used':xt.Cavity(),'other':xt.Cavity()},
                 element_names=['used'])
    line['phase_knob']=.3
    line['used'].phase=line.ref['phase_knob']
    other_desc=madng_tpsa.Descriptor(6,1,num_params=2,param_order=1)
    line['other'].phase=other_desc.param(2,.6)
    before=line['other'].phase.param_grad().copy()
    desc=madng_tpsa.Descriptor(6,1,num_params=1,param_order=1)
    knobs=xtpsa.KnobParameters(line,['phase_knob'],desc)
    knobs.apply()
    assert all(name=='used' for name,attr in knobs.driven_elements())
    assert all(name=='used' for name,attr in knobs.strength_jacobian())
    knobs.refresh([.5])
    knobs.teardown()
    assert not line['used']._tpsa_enabled
    assert line['other']._tpsa_enabled
    np.testing.assert_array_equal(line['other'].phase.param_grad(),before)
    line['other'].disable_tpsa()


def test_knob_teardown_cleans_its_shared_variable_promotions():
    line=xt.Line(elements={'used':xt.Cavity(),'other':xt.Cavity()},
                 element_names=['used'])
    line['common_phase']=.3
    line['used'].phase=line.ref['common_phase']
    line['other'].phase=line.ref['common_phase']
    desc=madng_tpsa.Descriptor(6,1,num_params=1,param_order=1)
    knobs=xtpsa.KnobParameters(line,['common_phase'],desc)
    for value in (.5,.7):
        knobs.apply([value])
        assert line['used']._tpsa_enabled
        assert line['other']._tpsa_enabled
        knobs.teardown()
        assert not line['used']._tpsa_enabled
        assert not line['other']._tpsa_enabled
        assert line['other'].phase==value


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
@pytest.mark.parametrize('order',[2,3])
def test_source_cavity_tpsa_eta_and_phase(cavity_magnetic_context, model, order):
    """Nontrivial calibration and two waves expose hidden frozen preparation."""
    mass = xt.ELECTRON_MASS_EV
    launch = dict(LAUNCH, p0c=mass*np.sqrt(2.4**2-1),
                  x=4e-4, y=-3e-4, px=.005/np.sqrt(2.4**2-1),
                  py=-.004/np.sqrt(2.4**2-1))
    context=cavity_magnetic_context
    line = xt.Line(elements=[xt.Cavity(length=.05, voltage=2e4,
        frequency=400e6, phase=.4, eta=.25, counter_phase=.3,
        calibration_beta=.9, model=model, integrator='gauss2',
        num_kicks=8)], element_names=['c'])
    line.particle_ref = xt.Particles(p0c=launch['p0c'], mass0=mass, q0=1.)
    line.reset_s_at_end_turn=False
    _build_shared_cavity_tracker(line,context)
    scalar = xt.Particles(_context=context, **launch)
    line.track(scalar)
    assert scalar.state[0] == 1
    desc = madng_tpsa.Descriptor(6, order, num_params=2, param_order=1)
    line['c'].eta = desc.param(1, .25)
    line['c'].phase = desc.param(2, .4)
    series = xtpsa.ParticlesTpsa(order=order, descriptor=desc, **launch)
    line.track(series)
    assert series._xobject.state == 1
    np.testing.assert_allclose(series.const_part, _values(scalar),
                               rtol=0., atol=3e-13)
    gradients = series.param_jacobian().copy()
    state_jacobian = series.jacobian().copy()
    mixed = series.coefficient('x',(0,1,0,0,0,0,1,0))
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
    # Five-point scalar stencils check the state map, independently of
    # the TPSA propagation and without requiring an analytic donor tangent.
    weights={-2:1.,-1:-8.,1:8.,2:-1.}
    fd_matrix=np.zeros((6,6))
    for column,name in enumerate(COORDS):
        step=2e-6
        for offset,coefficient in weights.items():
            shifted=dict(launch)
            shifted[name]+=offset*step
            particle=xt.Particles(_context=context,**shifted)
            line.track(particle)
            assert particle.state[0] == 1
            fd_matrix[:,column]+=coefficient*_values(particle)/(12*step)
    np.testing.assert_allclose(state_jacobian,fd_matrix,
                               rtol=2e-6,atol=2e-8)
    # Actual cavity mixed coefficient, not the generic polynomial above.
    # A tensor product of fourth-order scalar differences checks dR12/deta.
    fd_mixed=0.
    for ip,wp in weights.items():
        shifted=dict(launch,px=launch['px']+ip*1e-4)
        for ie,we in weights.items():
            line['c'].eta=.25+ie*.005
            particle=xt.Particles(_context=context,**shifted)
            line.track(particle)
            assert particle.state[0] == 1
            fd_mixed+=wp*we*float(particle.x[0])/(144*1e-4*.005)
    line['c'].eta=.25
    np.testing.assert_allclose(mixed,fd_mixed,rtol=2e-4,atol=3e-8)


@pytest.mark.parametrize('model',['phase-resolved','phase-resolved-paraxial',
                                 'phase-resolved-longitudinal'])
@pytest.mark.parametrize('explicit_carrier',[False,True])
def test_source_cavity_tpsa_continuous_controls(cavity_magnetic_context,model,explicit_carrier):
    mass=xt.ELECTRON_MASS_EV
    launch=dict(LAUNCH,p0c=mass*np.sqrt(2.4**2-1))
    values=dict(voltage=2e4,frequency=400e6,counter_phase=.3,
        calibration_beta=.9,ks=.03,delta_taper=.02,phase_taper=.07,lag_taper=1.2)
    if explicit_carrier:
        values['wave_number']=1.1*2*np.pi*400e6/299792458.
    context=cavity_magnetic_context
    line=xt.Line(elements={'c':xt.Cavity(length=.05,phase=.4,eta=.25,
        model=model,num_kicks=8,absolute_time=True,**values)})
    line.particle_ref=xt.Particles(p0c=launch['p0c'],mass0=mass,q0=1.)
    _build_shared_cavity_tracker(line,context)
    scalar=xt.Particles(_context=context,**launch)
    line.track(scalar)
    assert scalar.state[0] == 1
    desc=madng_tpsa.Descriptor(6,1,num_params=len(values),param_order=1)
    for index,(name,value) in enumerate(values.items(),start=1):
        setattr(line['c'],name,desc.param(index,value))
    series=xtpsa.ParticlesTpsa(order=1,descriptor=desc,**launch)
    line.track(series)
    np.testing.assert_allclose(series.const_part,_values(scalar),
                               rtol=0.,atol=3e-13)
    gradients=series.param_jacobian().copy()
    line.disable_tpsa_elements()
    for index,(name,value) in enumerate(values.items()):
        scale=max(1.,abs(value))
        step=scale*1e-5
        endpoints=[]
        for sign in (-1,1):
            setattr(line['c'],name,value+sign*step)
            particle=xt.Particles(_context=context,**launch)
            line.track(particle)
            assert particle.state[0] == 1
            endpoints.append(_values(particle))
        setattr(line['c'],name,value)
        fd=(endpoints[1]-endpoints[0])/(2*step)
        np.testing.assert_allclose(gradients[:,index]*scale,fd*scale,
                                   rtol=3e-6,atol=3e-9,
                                   err_msg=name)


@pytest.mark.parametrize('absolute_time',[False,True])
def test_frozen_cavity_tpsa_phase_and_clock(tmp_path,absolute_time):
    with native_context(tmp_path) as context:
        line=xt.Line(elements=[xt.Drift(length=.17),xt.Cavity(length=.05,
            voltage=2e4,frequency=400e6,phase=.4,eta=0.,
            model='phase-resolved-frozen',num_kicks=8,
            absolute_time=absolute_time)],element_names=['d','c'])
        line.particle_ref=xt.Particles(p0c=LAUNCH['p0c'],
            mass0=LAUNCH['mass0'],q0=1.)
        line.reset_s_at_end_turn=False
        line.build_tracker(_context=context)
        scalar=xt.Particles(_context=context,**LAUNCH)
        line.track(scalar)
        assert scalar.state[0] == 1
        desc=madng_tpsa.Descriptor(6,2,num_params=1,param_order=1)
        line['c'].phase=desc.param(1,.4)
        series=xtpsa.ParticlesTpsa(order=2,descriptor=desc,**LAUNCH)
        line.track(series)
        np.testing.assert_allclose(series.const_part,_values(scalar),
                                   rtol=0.,atol=3e-13)
        derivative=series.param_jacobian()[:,0]
        line.disable_tpsa_elements()
        endpoints=[]
        for phase in (.4-1e-5,.4+1e-5):
            line['c'].phase=phase
            particle=xt.Particles(_context=context,**LAUNCH)
            line.track(particle)
            endpoints.append(_values(particle))
        np.testing.assert_allclose(derivative,
            (endpoints[1]-endpoints[0])/(2e-5),rtol=3e-6,atol=3e-10)


@pytest.mark.parametrize('order',[1,2])
def test_second_order_averaged_cavity_tpsa(cavity_magnetic_context,order):
    """Differentiate the actual retained characteristic/product, not a proxy."""
    mass=xt.ELECTRON_MASS_EV
    launch=dict(LAUNCH,p0c=mass*np.sqrt(2.4**2-1))
    context=cavity_magnetic_context
    line=xt.Line(elements={'c':xt.Cavity(length=.05,voltage=2e3,
        frequency=400e6,phase=.4,eta=.25,counter_phase=.3,
        model='phase-averaged-second-order',num_kicks=4,num_factor_steps=1)})
    line.particle_ref=xt.Particles(p0c=launch['p0c'],mass0=mass,q0=1.)
    line.reset_s_at_end_turn=False
    _build_shared_cavity_tracker(line,context)
    scalar=xt.Particles(_context=context,**launch)
    line.track(scalar)
    assert scalar.state[0]==1
    desc=madng_tpsa.Descriptor(6,order,num_params=2,param_order=1)
    line['c'].eta=desc.param(1,.25)
    line['c'].phase=desc.param(2,.4)
    series=xtpsa.ParticlesTpsa(order=order,descriptor=desc,**launch)
    try:
        line.track(series)
    except RuntimeError as error:
        raise AssertionError(f"native TPSA status {series._xobject.state}") from error
    assert series._xobject.state==1
    np.testing.assert_allclose(series.const_part,_values(scalar),
                               rtol=0.,atol=3e-12)
    gradients=series.param_jacobian().copy()
    jacobian=series.jacobian().copy()
    mixed=series.coefficient('x',(0,1,0,0,0,0,1,0)) if order>=2 else None
    line.disable_tpsa_elements()
    for index,(name,value) in enumerate([('eta',.25),('phase',.4)]):
        endpoints=[]
        for sign in (-1,1):
            setattr(line['c'],name,value+sign*3e-4)
            particle=xt.Particles(_context=context,**launch)
            line.track(particle)
            assert particle.state[0]==1
            endpoints.append(_values(particle))
        setattr(line['c'],name,value)
        fd=(endpoints[1]-endpoints[0])/(6e-4)
        np.testing.assert_allclose(gradients[:,index],fd,
                                   rtol=3e-5,atol=2e-8)
        if index==0:
            coarse_eta_fd=fd.copy()


    if order==2:
        weights={-2:1.,-1:-8.,1:8.,2:-1.}
        finite_jacobian=np.zeros((6,6))
        for column,name in enumerate(COORDS):
            for offset,weight in weights.items():
                particle=xt.Particles(_context=context,**dict(
                    launch,**{name:launch[name]+offset*2e-6}))
                line.track(particle)
                assert particle.state[0]==1
                finite_jacobian[:,column]+=weight*_values(particle)/(24e-6)
        np.testing.assert_allclose(jacobian,finite_jacobian,
                                   rtol=3e-6,atol=3e-8)
        finite_mixed=0.
        for ip,wp in weights.items():
            for ie,we in weights.items():
                line['c'].eta=.25+ie*.005
                particle=xt.Particles(_context=context,**dict(
                    launch,px=launch['px']+ip*1e-4))
                line.track(particle)
                assert particle.state[0]==1
                finite_mixed+=wp*we*float(particle.x[0])/(144*1e-4*.005)
        line['c'].eta=.25
        np.testing.assert_allclose(mixed,finite_mixed,rtol=3e-4,atol=3e-8)
        # Refine the parent's characteristic mesh, keeping the selected
        # retained product and outer factor mesh unchanged. This is a
        # local consistency check, not a trajectory-error certificate.
        refinements=[_values(scalar)]
        # Use the same centred scalar stencil at every mesh, so this comparison
        # measures mesh changes without mixing TPSA and finite-difference bias.
        derivatives=[coarse_eta_fd]
        for count in (8,16):
            line['c'].num_kicks=count
            particle=xt.Particles(_context=context,**launch)
            line.track(particle)
            assert particle.state[0]==1
            refinements.append(_values(particle))
            endpoints=[]
            for eta in (.25-3e-4,.25+3e-4):
                line['c'].eta=eta
                particle=xt.Particles(_context=context,**launch)
                line.track(particle)
                assert particle.state[0]==1
                endpoints.append(_values(particle))
            derivatives.append((endpoints[1]-endpoints[0])/(6e-4))
            line['c'].eta=.25
        map_steps=[np.max(abs(refinements[i+1]-refinements[i])) for i in (0,1)]
        gradient_steps=[np.max(abs(derivatives[i+1]-derivatives[i])) for i in (0,1)]
        print('C2 parent mesh N4,N8,N16 map changes',map_steps,
              'eta gradient changes',gradient_steps)
        assert map_steps[1]<map_steps[0]+3e-13
        assert gradient_steps[1]<gradient_steps[0]+3e-10

def test_frozen_solenoid_zero_strength_parameter_tpsa(tmp_path):
    """The exact frozen SOL body uses an entire chart at zero field."""
    mass=xt.ELECTRON_MASS_EV
    launch=dict(LAUNCH,p0c=mass*np.sqrt(2.4**2-1))
    with native_context(tmp_path) as context:
        line=xt.Line(elements={'c':xt.Cavity(length=.05,frequency=400e6,
            voltage=0.,eta=0.,ks=0.,edge_entry_active=False,
            edge_exit_active=False,model='phase-resolved-frozen',num_kicks=4)})
        line.particle_ref=xt.Particles(p0c=launch['p0c'],mass0=mass,q0=1.)
        line.reset_s_at_end_turn=False
        line.build_tracker(_context=context)
        scalar=xt.Particles(_context=context,**launch)
        line.track(scalar)
        assert scalar.state[0]==1
        desc=madng_tpsa.Descriptor(6,2,num_params=1,param_order=1)
        line['c'].ks=desc.param(1,0.)
        series=xtpsa.ParticlesTpsa(order=2,descriptor=desc,**launch)
        line.track(series)
        np.testing.assert_allclose(series.const_part,_values(scalar),
                                   rtol=0.,atol=3e-13)
        derivative=series.param_jacobian()[:,0]
        line.disable_tpsa_elements()
        endpoints=[]
        for value in (-1e-5,1e-5):
            line['c'].ks=value
            particle=xt.Particles(_context=context,**launch)
            line.track(particle)
            assert particle.state[0]==1
            endpoints.append(_values(particle))
        finite_difference=(endpoints[1]-endpoints[0])/(2e-5)
        assert np.max(abs(derivative))>1e-6
        np.testing.assert_allclose(derivative,finite_difference,
                                   rtol=3e-6,atol=3e-10)


def test_frozen_cavity_tpsa_voltage_and_frequency(tmp_path):
    """Both the actual kick and fixed nominal programme carry RF knobs."""
    values=dict(voltage=2e4,frequency=400e6,phase_taper=.07,lag_taper=1.2)
    with native_context(tmp_path) as context:
        line=xt.Line(elements=[xt.Drift(length=.17),xt.Cavity(length=.05,
            phase=.4,eta=0.,model='phase-resolved-frozen',num_kicks=8,
            absolute_time=True,**values)],element_names=['d','c'])
        line.particle_ref=xt.Particles(p0c=LAUNCH['p0c'],
            mass0=LAUNCH['mass0'],q0=1.)
        line.reset_s_at_end_turn=False
        line.build_tracker(_context=context)
        scalar=xt.Particles(_context=context,**LAUNCH)
        line.track(scalar)
        assert scalar.state[0]==1
        desc=madng_tpsa.Descriptor(6,2,num_params=len(values),param_order=1)
        for index,(name,value) in enumerate(values.items(),start=1):
            setattr(line['c'],name,desc.param(index,value))
        series=xtpsa.ParticlesTpsa(order=2,descriptor=desc,**LAUNCH)
        line.track(series)
        np.testing.assert_allclose(series.const_part,_values(scalar),
                                   rtol=0.,atol=3e-13)
        derivatives=series.param_jacobian().copy()
        line.disable_tpsa_elements()
        for index,(name,value) in enumerate(values.items()):
            scale=max(1.,abs(value));step=scale*1e-5
            endpoints=[]
            for sign in (-1,1):
                setattr(line['c'],name,value+sign*step)
                particle=xt.Particles(_context=context,**LAUNCH)
                line.track(particle)
                assert particle.state[0]==1
                endpoints.append(_values(particle))
            setattr(line['c'],name,value)
            finite_difference=(endpoints[1]-endpoints[0])/(2*step)
            np.testing.assert_allclose(derivatives[:,index]*scale,
                finite_difference*scale,rtol=3e-6,atol=3e-9,err_msg=name)


class _FieldMetadataWriter(xt.BeamElement):
    _xofields={}
    _extra_c_sources=[r'''
    #include "xtrack/headers/track.h"
    GPUFUN
    void _FieldMetadataWriter_track_local_particle(
            _FieldMetadataWriterData el, LocalParticle* part0) {
        START_PER_PARTICLE_BLOCK(part0,part)
            LocalParticle_set_ax(part,.2*LocalParticle_get_x(part));
            LocalParticle_set_ay(part,-.3*LocalParticle_get_y(part));
        END_PER_PARTICLE_BLOCK;
    }
    ''']


class _FieldMetadataReader(xt.BeamElement):
    _xofields={}
    _extra_c_sources=[r'''
    #include "xtrack/headers/track.h"
    GPUFUN
    void _FieldMetadataReader_track_local_particle(
            _FieldMetadataReaderData el, LocalParticle* part0) {
        START_PER_PARTICLE_BLOCK(part0,part)
            LocalParticle_add_to_x(part,LocalParticle_get_ax(part));
            LocalParticle_add_to_y(part,LocalParticle_get_ay(part));
        END_PER_PARTICLE_BLOCK;
    }
    ''']


@pytest.mark.parametrize('default_context',[False,True])
def test_tpsa_field_metadata_survives_continuation(tmp_path,monkeypatch,default_context):
    """An exact algebraic witness for local chart metadata across track calls."""
    from contextlib import nullcontext
    import xobjects as xo
    monkeypatch.chdir(tmp_path)
    scope=nullcontext(xo.ContextCpu()) if default_context else native_context(tmp_path)
    with scope as context:
        line=xt.Line(elements={'write':_FieldMetadataWriter(),
                              'read':_FieldMetadataReader()})
        # The default-context witness exercises the TPSA kernel route, rather
        # than requesting an unrelated ordinary prebuilt for custom fixtures.
        line.build_tracker(_context=context,compile=not default_context)
        maps=[]
        for route in ('whole','partial','standalone'):
            particles=xtpsa.ParticlesTpsa(order=2,**LAUNCH)
            if route=='whole':
                line.track(particles)
            elif route=='partial':
                line.track(particles,ele_start=0,ele_stop=1)
                line.track(particles,ele_start=1,ele_stop=2)
            else:
                line['write'].track(particles)
                line['read'].track(particles)
            jacobian=particles.jacobian()
            np.testing.assert_allclose(jacobian[[0,2],[0,2]],[1.2,.7],
                                       rtol=0.,atol=2e-15,err_msg=route)
            np.testing.assert_allclose(particles.const_part[[0,2]],
                [1.2*LAUNCH['x'],.7*LAUNCH['y']],rtol=0.,atol=2e-15,
                err_msg=route)
            maps.append(particles.const_part)
        np.testing.assert_allclose(maps[0],maps[1],rtol=0.,atol=2e-15)
        np.testing.assert_allclose(maps[0],maps[2],rtol=0.,atol=2e-15)


def test_multisetter_preserves_active_cavity_parameters(tmp_path):
    with native_context(tmp_path) as context:
        line=xt.Line(elements={'c':xt.Cavity(length=.05,phase=.4,eta=.25),
                              'd':xt.Drift(length=.2)})
        line.build_tracker(_context=context)
        before=xt.MultiSetter(line,['c'],'phase',dtype=None)
        before_length=xt.MultiSetter(line,['c','d'],'length',dtype=None)
        line['c'].shift_x=.001
        alignment=xt.MultiSetter(line,['c'],'shift_x',dtype=None)
        np.testing.assert_array_equal(alignment.get_values(),[.001])
        desc=madng_tpsa.Descriptor(6,2,num_params=2,param_order=1)
        line['c'].phase=desc.param(1,.4)
        line['c'].eta=desc.param(2,.25)
        after=xt.MultiSetter(line,['c'],'phase',dtype=None)
        after_length=xt.MultiSetter(line,['c','d'],'length',dtype=None)
        np.testing.assert_array_equal(before.get_values(),[.4])
        np.testing.assert_array_equal(after.get_values(),[.4])
        np.testing.assert_array_equal(before_length.get_values(),[.05,.2])
        np.testing.assert_array_equal(after_length.get_values(),[.05,.2])
        before.set_values(np.array([.7]))
        np.testing.assert_array_equal(after.get_values(),[.7])
        np.testing.assert_array_equal(line['c'].phase.param_grad(),[0.,0.])
        np.testing.assert_array_equal(line['c'].eta.param_grad(),[0.,1.])
        assert line['c']._tpsa_enabled
        np.testing.assert_array_equal(alignment.get_values(),[.001])
        np.testing.assert_array_equal(line.attr['_tpsa_enabled'],[1.,0.])
        line.disable_tpsa_elements()
        assert not before._needs_tpsa_access()
        after.set_values(np.array([.8]))
        np.testing.assert_array_equal(before.get_values(),[.8])
        assert line['c'].phase==.8


def test_multisetter_resolves_canonical_slice_parent(tmp_path):
    with native_context(tmp_path) as context:
        line=xt.Line(elements={'c':xt.Cavity(length=.05,phase=.4,eta=.25)})
        line.slice_thick_elements([xt.Strategy(
            slicing=xt.Teapot(2,mode='thick'),element_type=xt.Cavity)])
        line.build_tracker(_context=context)
        slices=[name for name in line.element_names
                if isinstance(line._element_dict[name],xt.ThickSliceCavity)]
        setter=xt.MultiSetter(line,slices,('_parent','phase'),dtype=None)
        desc=madng_tpsa.Descriptor(6,2,num_params=1,param_order=1)
        line['c'].eta=desc.param(1,.25)
        np.testing.assert_array_equal(setter.get_values(),[.4,.4])
        setter.set_values(np.array([.7,.7]))
        assert line['c'].phase.const_part==.7
        np.testing.assert_array_equal(line['c'].eta.param_grad(),[1.])
        np.testing.assert_array_equal(line.attr['length'],[0.,.025,.025,0.])
        line.disable_tpsa_elements()
        assert not setter._needs_tpsa_access()
        np.testing.assert_array_equal(setter.get_values(),[.7,.7])


def test_indexed_cavity_knob_lifecycle(tmp_path):
    with native_context(tmp_path) as context:
        line=xt.Line(elements={'c':xt.Cavity(length=.05,knl=[0.,.2],ksl=[0.,.3])})
        line.vars['normal']=.2;line.vars['skew']=.3
        line.element_refs['c'].knl[1]=line.vars['normal']
        line.element_refs['c'].ksl[1]=line.vars['skew']
        line.build_tracker(_context=context,compile=False)
        owner=line._element_dict['c']
        np.testing.assert_array_equal(np.asarray(owner.knl)[:2],[0.,.2])
        cached=xt.MultiSetter(line,['c'],'knl',index=1,dtype=None)
        desc=madng_tpsa.Descriptor(6,2,num_params=2,param_order=1)
        knobs=KnobParameters(line,['normal','skew'],desc)
        knobs.apply()
        assert owner._tpsa_enabled
        assert owner._knl_tpsa[1]!=0
        np.testing.assert_array_equal(
            knobs.strength_jacobian()[('c',('knl',1))],[1.,0.])
        with pytest.raises(TypeError,match='dtype=object'):
            np.asarray(owner.knl)
        assert isinstance(np.asarray(owner.knl,dtype=object)[1],madng_tpsa.Tpsa)
        np.testing.assert_array_equal(cached.get_values(),[.2])
        current=xt.MultiSetter(line,['c'],'knl',index=1,dtype=None)
        current.set_values(np.array([.4]))
        assert owner.knl[1]==.4
        np.testing.assert_array_equal(owner.ksl[1].param_grad(),[0.,1.])
        knobs.refresh([.5,.6])
        np.testing.assert_array_equal(cached.get_values(),[.5])
        np.testing.assert_array_equal(owner.knl[1].param_grad(),[1.,0.])
        np.add.at(owner.knl,[1],.1)
        assert owner.knl[1].const_part==pytest.approx(.6)
        np.testing.assert_array_equal(owner.knl[1].param_grad(),[1.,0.])
        owner.knl.put([1],[owner.knl[1]+.1])
        assert owner.knl[1].const_part==pytest.approx(.7)
        np.testing.assert_array_equal(owner.knl[1].param_grad(),[1.,0.])
        with pytest.raises(TypeError,match='unsupported'):
            owner.knl.sort()
        owner.knl.fill(0.)
        np.testing.assert_array_equal(np.asarray(owner.knl),np.zeros(len(owner.knl)))
        assert not owner.__dict__['_tpsa_array_handles']['_knl']
        knobs.refresh([.7,.8])
        knobs.teardown()
        assert not owner._tpsa_enabled
        assert not np.any(owner._knl_tpsa)
        assert not np.any(owner._ksl_tpsa)
        assert not cached._needs_tpsa_access()
        np.testing.assert_array_equal(cached.get_values(),[.7])
        copied=line.copy()
        restored=xt.Line.from_dict(copied.to_dict())
        np.testing.assert_array_equal(restored._element_dict['c'].knl,owner.knl)
        np.testing.assert_array_equal(restored._element_dict['c'].ksl,owner.ksl)
        assert not np.any(restored._element_dict['c']._knl_tpsa)
        assert not np.any(restored._element_dict['c']._ksl_tpsa)


def _build_shared_cavity_tracker(line, context):
    # Kernel ownership is per Tracker, not per Context. Reuse the supported
    # track_kernel dictionary across identical one-Cavity type/ABI sets.
    previous = getattr(context, '_cavity_test_track_kernels', None)
    line.build_tracker(_context=context, track_kernel=previous)
    context._cavity_test_track_kernels = line.tracker.track_kernel


@pytest.fixture(scope='module')
def cavity_magnetic_context(tmp_path_factory):
    # All model values share one public data/kernel ABI. Reuse the actual JIT
    # kernel instead of recompiling the same heavy scalar/TPSA sources per case.
    with native_context(tmp_path_factory.mktemp('cavity_magnetic_tpsa')) as context:
        yield context


@pytest.mark.parametrize('model',['phase-resolved','phase-resolved-paraxial',
                                  'phase-resolved-longitudinal','phase-resolved-frozen',
                                  'phase-averaged-full-slow'])
def test_native_indexed_magnetic_parameter_tpsa(cavity_magnetic_context,model):
    mass=xt.ELECTRON_MASS_EV
    launch=dict(LAUNCH,p0c=mass*np.sqrt(2.4**2-1),x=4e-4,y=-3e-4)
    frozen=model=='phase-resolved-frozen'
    values={'normal':0. if frozen else .01}
    if not frozen: values['skew']=.005
    context=cavity_magnetic_context
    full_slow=model=='phase-averaged-full-slow'
    line=xt.Line(elements={'c':xt.Cavity(length=.05,voltage=2e3 if full_slow else 2e4,
        frequency=400e6,phase=.4,eta=0. if frozen else .25,
        model=model,num_kicks=4 if full_slow else 8)})
    for name,value in values.items():line.vars[name]=value
    line.element_refs['c'].knl[1]=line.vars['normal']
    if not frozen:line.element_refs['c'].ksl[2]=line.vars['skew']
    line.particle_ref=xt.Particles(p0c=launch['p0c'],mass0=mass,q0=1.)
    line.reset_s_at_end_turn=False
    _build_shared_cavity_tracker(line,context)
    scalar=xt.Particles(_context=context,**launch)
    line.track(scalar)
    assert scalar.state[0]==1
    desc=madng_tpsa.Descriptor(6,2,num_params=len(values),param_order=1)
    knobs=KnobParameters(line,list(values),desc)
    knobs.apply()
    series=xtpsa.ParticlesTpsa(order=2,descriptor=desc,**launch)
    line.track(series)
    np.testing.assert_allclose(series.const_part,_values(scalar),rtol=0.,atol=3e-13)
    gradients=series.param_jacobian().copy()
    powers=(1,0,0,0,0,0,1)+(0,)*(len(values)-1)
    mixed=series.coefficient('px',powers)
    knobs.teardown()
    for index,(name,value) in enumerate(values.items()):
        endpoints=[]
        for sign in (-1,1):
            line.vars[name]=value+sign*1e-5
            particle=xt.Particles(_context=context,**launch)
            line.track(particle)
            assert particle.state[0]==1
            endpoints.append(_values(particle))
        line.vars[name]=value
        fd=(endpoints[1]-endpoints[0])/(2e-5)
        np.testing.assert_allclose(gradients[:,index],fd,rtol=3e-6,atol=3e-10)
    difference=0.
    for sx in (-1,1):
        for sg in (-1,1):
            line.vars['normal']=values['normal']+sg*1e-4
            particle=xt.Particles(_context=context,**dict(launch,x=launch['x']+sx*1e-5))
            line.track(particle)
            assert particle.state[0]==1
            difference+=sx*sg*float(particle.px[0])/(4e-9)
    line.vars['normal']=values['normal']
    assert abs(mixed)>1e-4
    np.testing.assert_allclose(mixed,difference,rtol=3e-5,atol=3e-8)


def test_cavity_descriptor_change_is_atomic():
    owner=xt.Cavity(length=.05)
    first=madng_tpsa.Descriptor(6,2,num_params=1,param_order=1)
    second=madng_tpsa.Descriptor(6,3,num_params=1,param_order=1)
    owner.phase=first.param(1,.4)
    owner.knl[1]=first.param(1,.2)
    pointer=int(owner._knl_tpsa[1])
    with pytest.raises(ValueError,match='descriptor'):
        owner.enable_tpsa(second)
    with pytest.raises(ValueError,match='descriptor'):
        owner.voltage=second.param(1,2e4)
    with pytest.raises(ValueError,match='descriptor'):
        owner.knl[1]=second.param(1,.3)
    assert owner._tpsa_descriptor.ptr==first.ptr
    assert owner.phase.const_part==.4
    assert owner.knl[1].const_part==.2
    assert int(owner._knl_tpsa[1])==pointer
    np.testing.assert_array_equal(owner.phase.param_grad(),[1.])
    owner.enable_tpsa(first)
    np.testing.assert_array_equal(owner.knl[1].param_grad(),[1.])
    owner.disable_tpsa()
    assert owner.phase==.4
    assert owner.knl[1]==.2
    assert not np.any(owner._knl_tpsa)


def test_compact_averaged_public_tpsa(cavity_magnetic_context):
    """Public C2/C3/C4 maps and jets use their actual compact tracking word."""
    context=cavity_magnetic_context
    mass=xt.ELECTRON_MASS_EV
    launch=dict(LAUNCH,p0c=mass*np.sqrt(2.4**2-1))
    line=xt.Line(elements={'c':xt.Cavity(length=.167,voltage=2e4,
        frequency=400e6,phase=.4,eta=.25,counter_phase=.3,
        calibration_beta=.9,model='phase-averaged-second-order',
        num_kicks=1,num_factor_steps=1)})
    line.particle_ref=xt.Particles(p0c=launch['p0c'],mass0=mass,q0=1.)
    line.reset_s_at_end_turn=False
    _build_shared_cavity_tracker(line,context)
    weights={-2:1.,-1:-8.,1:8.,2:-1.}
    for model in ('phase-averaged-second-order','phase-averaged-cubic',
                  'phase-averaged-quartic'):
        line['c'].model=model
        scalar=xt.Particles(_context=context,**launch)
        line.track(scalar)
        assert scalar.state[0]==1,model
        desc=madng_tpsa.Descriptor(6,2,num_params=2,param_order=1)
        line['c'].phase=desc.param(1,.4)
        line['c'].eta=desc.param(2,.25)
        series=xtpsa.ParticlesTpsa(order=2,descriptor=desc,**launch)
        line.track(series)
        np.testing.assert_allclose(series.const_part,_values(scalar),
                                   rtol=0.,atol=3e-12,err_msg=model)
        jacobian=series.jacobian().copy()
        gradients=series.param_jacobian().copy()
        mixed=series.coefficient('x',(0,1,0,0,0,0,0,1))
        line.disable_tpsa_elements()
        finite_jacobian=np.zeros((6,6))
        for column,name in enumerate(COORDS):
            for offset,weight in weights.items():
                particle=xt.Particles(_context=context,**dict(
                    launch,**{name:launch[name]+offset*1e-4}))
                line.track(particle)
                assert particle.state[0]==1,model
                finite_jacobian[:,column]+=weight*_values(particle)/(12e-4)
        np.testing.assert_allclose(jacobian,finite_jacobian,
                                   rtol=3e-6,atol=3e-9,err_msg=model)
        for column,(name,value) in enumerate((('phase',.4),('eta',.25))):
            derivative=np.zeros(6)
            for offset,weight in weights.items():
                setattr(line['c'],name,value+offset*1e-3)
                particle=xt.Particles(_context=context,**launch)
                line.track(particle)
                assert particle.state[0]==1,model
                derivative+=weight*_values(particle)/(12e-3)
            setattr(line['c'],name,value)
            np.testing.assert_allclose(gradients[:,column],derivative,
                                       rtol=3e-6,atol=3e-9,err_msg=model+name)
        finite_mixed=0.
        for ip,wp in weights.items():
            for ie,we in weights.items():
                line['c'].eta=.25+ie*.01
                particle=xt.Particles(_context=context,**dict(
                    launch,px=launch['px']+ip*1e-3))
                line.track(particle)
                assert particle.state[0]==1,model
                finite_mixed+=wp*we*float(particle.x[0])/(144*1e-3*.01)
        line['c'].eta=.25
        assert abs(mixed)>1e-5,model
        np.testing.assert_allclose(mixed,finite_mixed,
                                   rtol=3e-5,atol=3e-8,err_msg=model)


@pytest.mark.parametrize('model',[
    'phase-averaged-uniform-quadratic-first-boundary',
    'phase-averaged-uniform-quadratic','phase-averaged-full-slow'])
def test_full_slow_and_uniform_public_tpsa(cavity_magnetic_context,model):
    """Executed 17/18/19 maps retain state and calibrated hardware derivatives."""
    context=cavity_magnetic_context
    mass=xt.ELECTRON_MASS_EV
    launch=dict(LAUNCH,p0c=mass*np.sqrt(2.4**2-1))
    values=dict(phase=.4,eta=.25,counter_phase=.3,voltage=2e4,
                frequency=400e6,calibration_beta=.9)
    line=xt.Line(elements={'c':xt.Cavity(length=.167,
        model='phase-averaged-full-slow',num_kicks=2,num_factor_steps=1,**values)})
    line.particle_ref=xt.Particles(p0c=launch['p0c'],mass0=mass,q0=1.)
    line.reset_s_at_end_turn=False
    _build_shared_cavity_tracker(line,context)
    weights={-2:1.,-1:-8.,1:8.,2:-1.}
    line['c'].model=model
    # The profile is a stored physical input, not retargeted implicitly
    # when selecting another model on the same parent element.
    line['c'].rf_profile='p16' if model=='phase-averaged-full-slow' else 'uniform'
    line['c'].num_kicks=2 if model=='phase-averaged-full-slow' else 8
    scalar=xt.Particles(_context=context,**launch)
    line.track(scalar)
    assert scalar.state[0]==1,model
    desc=madng_tpsa.Descriptor(6,2,num_params=len(values),param_order=1)
    for index,(name,value) in enumerate(values.items(),start=1):
        setattr(line['c'],name,desc.param(index,value))
    series=xtpsa.ParticlesTpsa(order=2,descriptor=desc,**launch)
    line.track(series)
    np.testing.assert_allclose(series.const_part,_values(scalar),
                               rtol=0.,atol=3e-12,err_msg=model)
    jacobian=series.jacobian().copy()
    gradients=series.param_jacobian().copy()
    mixed=series.coefficient('x',(0,1,0,0,0,0,0,1,0,0,0,0))
    line.disable_tpsa_elements()
    finite_jacobian=np.zeros((6,6))
    for column,name in enumerate(COORDS):
        for offset,weight in weights.items():
            particle=xt.Particles(_context=context,**dict(
                launch,**{name:launch[name]+offset*1e-4}))
            line.track(particle)
            assert particle.state[0]==1,model
            finite_jacobian[:,column]+=weight*_values(particle)/(12e-4)
    np.testing.assert_allclose(jacobian,finite_jacobian,
                               rtol=3e-6,atol=3e-9,err_msg=model)
    for column,(name,value) in enumerate(values.items()):
        scale=max(1.,abs(value));step=scale*1e-4
        derivative=np.zeros(6)
        for offset,weight in weights.items():
            setattr(line['c'],name,value+offset*step)
            particle=xt.Particles(_context=context,**launch)
            line.track(particle)
            assert particle.state[0]==1,model
            derivative+=weight*_values(particle)/(12*step)
        setattr(line['c'],name,value)
        np.testing.assert_allclose(gradients[:,column]*scale,derivative*scale,
                                   rtol=3e-6,atol=3e-8,err_msg=model+name)
    finite_mixed=0.
    for ip,wp in weights.items():
        for ie,we in weights.items():
            line['c'].eta=.25+ie*.01
            particle=xt.Particles(_context=context,**dict(
                launch,px=launch['px']+ip*1e-3))
            line.track(particle)
            assert particle.state[0]==1,model
            finite_mixed+=wp*we*float(particle.x[0])/(144*1e-3*.01)
    line['c'].eta=.25
    np.testing.assert_allclose(mixed,finite_mixed,
                               rtol=3e-5,atol=3e-8,err_msg=model)
    print(model,'mixed dR12/deta',mixed,'FD',finite_mixed)


def test_full_slow_zero_fast_and_zero_voltage_series(cavity_magnetic_context):
    """Only entire-series zero fast fields may skip the counter-wave factors."""
    context=cavity_magnetic_context
    mass=xt.ELECTRON_MASS_EV
    launch=dict(LAUNCH,p0c=mass*np.sqrt(2.4**2-1))
    line=xt.Line(elements={'c':xt.Cavity(length=.167,voltage=2e4,
        frequency=400e6,phase=.4,eta=0.,calibration_beta=.9,
        model='phase-averaged-full-slow',num_kicks=2,num_factor_steps=1)})
    line.particle_ref=xt.Particles(p0c=launch['p0c'],mass0=mass,q0=1.)
    line.reset_s_at_end_turn=False
    _build_shared_cavity_tracker(line,context)
    scalar=xt.Particles(_context=context,**launch)
    line.track(scalar)
    assert scalar.state[0]==1
    desc=madng_tpsa.Descriptor(6,2,num_params=1,param_order=1)
    line['c'].phase=desc.param(1,.4)
    series=xtpsa.ParticlesTpsa(order=2,descriptor=desc,**launch)
    line.track(series)
    np.testing.assert_allclose(series.const_part,_values(scalar),rtol=0.,atol=3e-12)
    derivative=series.param_jacobian()[:,0]
    line.disable_tpsa_elements()
    endpoints=[]
    for phase in (.4-1e-4,.4+1e-4):
        line['c'].phase=phase
        particle=xt.Particles(_context=context,**launch);line.track(particle)
        assert particle.state[0]==1
        endpoints.append(_values(particle))
    np.testing.assert_allclose(derivative,(endpoints[1]-endpoints[0])/2e-4,
                               rtol=3e-6,atol=3e-9)
    line['c'].phase=.4;line['c'].eta=.25
    second=madng_tpsa.Descriptor(6,2,num_params=1,param_order=2)
    line['c'].voltage=2e4*second.param(1,0.)
    series=xtpsa.ParticlesTpsa(order=2,descriptor=second,**launch)
    line.track(series)
    quadratic=np.array([series.coefficient(name,(0,0,0,0,0,0,2)) for name in COORDS])
    line.disable_tpsa_elements()
    h=.02;finite=np.zeros(6)
    for offset,weight in {0:-30.,-1:16.,1:16.,-2:-1.,2:-1.}.items():
        line['c'].voltage=2e4*offset*h
        particle=xt.Particles(_context=context,**launch);line.track(particle)
        assert particle.state[0]==1
        finite+=weight*_values(particle)/(24*h*h)
    assert np.max(abs(quadratic))>1e-10
    np.testing.assert_allclose(quadratic,finite,rtol=3e-5,atol=3e-9)


def test_gaussian_magnetic_profile_tpsa(cavity_magnetic_context):
    """Differentiate the native Gaussian M1 lift, including source calibration."""
    context=cavity_magnetic_context
    launch=dict(LAUNCH,p0c=xt.ELECTRON_MASS_EV*np.sqrt(2.4**2-1))
    values=dict(phase=.4,eta=.25,counter_phase=.3,voltage=2e4,
        frequency=400e6,calibration_beta=.9,rf_profile_sigma=.035,
        rf_profile_center=.012)
    line=xt.Line(elements={'c':xt.Cavity(length=.167,
        model='phase-averaged-magnetic',rf_profile='gaussian',
        num_kicks=8,num_factor_steps=2,**values)})
    line.particle_ref=xt.Particles(p0c=launch['p0c'],mass0=launch['mass0'],q0=1.)
    line.reset_s_at_end_turn=False
    _build_shared_cavity_tracker(line,context)
    scalar=xt.Particles(_context=context,**launch);line.track(scalar)
    assert scalar.state[0]==1
    desc=madng_tpsa.Descriptor(6,2,num_params=len(values),param_order=1)
    for index,(name,value) in enumerate(values.items(),1):
        setattr(line['c'],name,desc.param(index,value))
    series=xtpsa.ParticlesTpsa(order=2,descriptor=desc,**launch)
    line.track(series)
    np.testing.assert_allclose(series.const_part,_values(scalar),rtol=0.,atol=3e-12)
    gradients=series.param_jacobian().copy()
    jacobian=series.jacobian().copy()
    mixed=series.coefficient('x',(0,1,0,0,0,0,0,0,0,0,0,0,1,0))
    line.disable_tpsa_elements()
    weights={-2:1.,-1:-8.,1:8.,2:-1.}
    for column,(name,value) in enumerate(values.items()):
        scale=abs(value) if value else 1.
        step=scale*1e-4
        finite=np.zeros(6)
        for offset,weight in weights.items():
            setattr(line['c'],name,value+offset*step)
            particle=xt.Particles(_context=context,**launch);line.track(particle)
            assert particle.state[0]==1
            finite+=weight*_values(particle)/(12*step)
        setattr(line['c'],name,value)
        np.testing.assert_allclose(gradients[:,column]*scale,finite*scale,
                                   rtol=3e-6,atol=3e-8,err_msg=name)
    finite_jacobian=np.zeros((6,6))
    for column,name in enumerate(COORDS):
        for offset,weight in weights.items():
            particle=xt.Particles(_context=context,**dict(launch,**{name:launch[name]+offset*1e-4}))
            line.track(particle)
            finite_jacobian[:,column]+=weight*_values(particle)/(12e-4)
    np.testing.assert_allclose(jacobian,finite_jacobian,rtol=3e-6,atol=3e-8)
    finite_mixed=0.
    for ip,wp in weights.items():
        for iw,ww in weights.items():
            line['c'].rf_profile_sigma=values['rf_profile_sigma']+iw*1e-4
            particle=xt.Particles(_context=context,**dict(launch,px=launch['px']+ip*1e-3))
            line.track(particle)
            finite_mixed+=wp*ww*float(particle.x[0])/(144*1e-3*1e-4)
    np.testing.assert_allclose(mixed,finite_mixed,rtol=3e-5,atol=3e-8)
    print('Gaussian20 mixed dR12/dsigma',mixed,'FD',finite_mixed)


def test_gaussian_fringe_static_parameter_tpsa(cavity_magnetic_context):
    """Native localized static fields retain their width and indexed knobs."""
    context=cavity_magnetic_context
    launch=dict(LAUNCH,p0c=xt.ELECTRON_MASS_EV*np.sqrt(2.4**2-1),
                x=4e-4,y=-3e-4)
    values=dict(phase=.4,eta=.25,magnetic_profile_sigma=.04,ks=.3,
                normal=.004,skew=.001)
    line=xt.Line(elements={'c':xt.Cavity(length=.167,voltage=2e4,
        frequency=400e6,counter_phase=.3,calibration_beta=.9,
        model='phase-averaged-magnetic-fringe',rf_profile='gaussian',
        rf_profile_sigma=.035,rf_profile_center=.012,
        magnetic_profile='gaussian',magnetic_profile_sigma=.04,
        edge_entry_active=False,edge_exit_active=False,
        num_kicks=8,num_factor_steps=2)})
    for name,value in values.items():
        line.vars[name]=value
        if name=='normal':line.element_refs['c'].knl[1]=line.vars[name]
        elif name=='skew':line.element_refs['c'].ksl[1]=line.vars[name]
        else:setattr(line.element_refs['c'],name,line.vars[name])
    line.particle_ref=xt.Particles(p0c=launch['p0c'],mass0=launch['mass0'],q0=1.)
    line.reset_s_at_end_turn=False
    _build_shared_cavity_tracker(line,context)
    scalar=xt.Particles(_context=context,**launch);line.track(scalar)
    assert scalar.state[0]==1
    desc=madng_tpsa.Descriptor(6,2,num_params=len(values),param_order=1)
    knobs=KnobParameters(line,list(values),desc)
    knobs.apply()
    series=xtpsa.ParticlesTpsa(order=2,descriptor=desc,**launch)
    line.track(series)
    np.testing.assert_allclose(series.const_part,_values(scalar),rtol=0.,atol=3e-12)
    potential_gradients=np.array([series._local_series[name].param_grad()
                                  for name in ('ax','ay')])
    np.testing.assert_allclose([series._local_series[name].const_part for name in ('ax','ay')],
                               [float(scalar.ax[0]),float(scalar.ay[0])],rtol=0.,atol=3e-12)
    gradients=series.param_jacobian().copy()
    jacobian=series.jacobian().copy()
    mixed=series.coefficient('x',(0,1,0,0,0,0,0,0,1,0,0,0))
    knobs.teardown()
    weights={-2:1.,-1:-8.,1:8.,2:-1.}
    for column,(name,value) in enumerate(values.items()):
        step=abs(value)*1e-4
        finite=np.zeros(6)
        finite_potential=np.zeros(2)
        for offset,weight in weights.items():
            line.vars[name]=value+offset*step
            particle=xt.Particles(_context=context,**launch);line.track(particle)
            assert particle.state[0]==1
            finite+=weight*_values(particle)/(12*step)
            finite_potential+=weight*np.array([float(particle.ax[0]),float(particle.ay[0])])/(12*step)
        line.vars[name]=value
        np.testing.assert_allclose(gradients[:,column]*abs(value),finite*abs(value),
                                   rtol=3e-6,atol=3e-8,err_msg=name)
        np.testing.assert_allclose(potential_gradients[:,column]*abs(value),finite_potential*abs(value),
                                   rtol=3e-6,atol=3e-8,err_msg=name+' potential metadata')
        assert np.max(abs(gradients[:,column]))>0.,name
    finite_jacobian=np.zeros((6,6))
    for column,name in enumerate(COORDS):
        for offset,weight in weights.items():
            particle=xt.Particles(_context=context,**dict(launch,**{name:launch[name]+offset*1e-4}))
            line.track(particle)
            finite_jacobian[:,column]+=weight*_values(particle)/(12e-4)
    np.testing.assert_allclose(jacobian,finite_jacobian,rtol=3e-6,atol=3e-8)
    finite_mixed=0.
    for ip,wp in weights.items():
        for iw,ww in weights.items():
            line.vars['magnetic_profile_sigma']=values['magnetic_profile_sigma']+iw*1e-4
            particle=xt.Particles(_context=context,**dict(launch,px=launch['px']+ip*1e-3))
            line.track(particle)
            finite_mixed+=wp*ww*float(particle.x[0])/(144*1e-3*1e-4)
    np.testing.assert_allclose(mixed,finite_mixed,rtol=3e-5,atol=3e-8)
    print('Gaussian22 mixed dR12/dmagnetic_sigma',mixed,'FD',finite_mixed)


def test_gaussian_dipole_public_parameter_tpsa(cavity_magnetic_context):
    """The fixed reference orbit is a hardware/reduction input, not tracked px."""
    context=cavity_magnetic_context
    launch=dict(LAUNCH,p0c=xt.ELECTRON_MASS_EV*np.sqrt(2.4**2-1),x=4e-4,y=-3e-4)
    values=dict(phase=.4,eta=.25,calibration_beta=.9,frequency=400e6,
                reference_px=.003,dipole=.002)
    line=xt.Line(elements={'c':xt.Cavity(length=.167,voltage=2e4,
        model='phase-averaged-dipole',rf_profile='gaussian',
        frequency=400e6,
        rf_profile_sigma=.035,rf_profile_center=.012,
        counter_phase=.3,num_kicks=8,num_factor_steps=2)})
    for name,value in values.items():
        line.vars[name]=value
        if name=='dipole':line.element_refs['c'].knl[0]=line.vars[name]
        else:setattr(line.element_refs['c'],name,line.vars[name])
    line.particle_ref=xt.Particles(p0c=launch['p0c'],mass0=launch['mass0'],q0=1.)
    line.reset_s_at_end_turn=False
    _build_shared_cavity_tracker(line,context)
    scalar=xt.Particles(_context=context,**launch);line.track(scalar)
    assert scalar.state[0]==1
    desc=madng_tpsa.Descriptor(6,2,num_params=len(values),param_order=1)
    knobs=KnobParameters(line,list(values),desc);knobs.apply()
    series=xtpsa.ParticlesTpsa(order=2,descriptor=desc,**launch);line.track(series)
    np.testing.assert_allclose(series.const_part,_values(scalar),rtol=0.,atol=3e-12)
    gradients=series.param_jacobian().copy();jacobian=series.jacobian().copy()
    mixed=series.coefficient('x',(0,1,0,0,0,0,0,0,0,0,1,0))
    knobs.teardown()
    weights={-2:1.,-1:-8.,1:8.,2:-1.}
    for column,(name,value) in enumerate(values.items()):
        step=abs(value)*1e-4;finite=np.zeros(6)
        for offset,weight in weights.items():
            line.vars[name]=value+offset*step
            particle=xt.Particles(_context=context,**launch);line.track(particle)
            assert particle.state[0]==1
            finite+=weight*_values(particle)/(12*step)
        line.vars[name]=value
        np.testing.assert_allclose(gradients[:,column]*abs(value),finite*abs(value),
                                   rtol=3e-6,atol=3e-8,err_msg=name)
    finite_jacobian=np.zeros((6,6))
    for column,name in enumerate(COORDS):
        for offset,weight in weights.items():
            particle=xt.Particles(_context=context,**dict(launch,**{name:launch[name]+offset*1e-4}))
            line.track(particle)
            finite_jacobian[:,column]+=weight*_values(particle)/(12e-4)
    np.testing.assert_allclose(jacobian,finite_jacobian,rtol=3e-6,atol=3e-8)
    finite_mixed=0.
    for ip,wp in weights.items():
        for ir,wr in weights.items():
            line.vars['reference_px']=values['reference_px']+ir*1e-4
            particle=xt.Particles(_context=context,**dict(launch,px=launch['px']+ip*1e-3))
            line.track(particle)
            finite_mixed+=wp*wr*float(particle.x[0])/(144*1e-3*1e-4)
    np.testing.assert_allclose(mixed,finite_mixed,rtol=3e-5,atol=3e-8)
    print('Gaussian23 mixed dR12/dreference_px',mixed,'FD',finite_mixed)


def test_gaussian_dipole_zero_strength_order_three(cavity_magnetic_context):
    """A seeded zero bend must retain the analytic nilpotent source clock."""
    context=cavity_magnetic_context
    launch=dict(LAUNCH,p0c=xt.ELECTRON_MASS_EV*np.sqrt(2.4**2-1))
    line=xt.Line(elements={'c':xt.Cavity(length=.167,voltage=2e4,frequency=400e6,
        phase=.4,eta=.25,counter_phase=.3,calibration_beta=.9,reference_px=.003,
        model='phase-averaged-dipole',rf_profile='gaussian',
        rf_profile_sigma=.035,rf_profile_center=.012,num_kicks=8,num_factor_steps=2)})
    line.particle_ref=xt.Particles(p0c=launch['p0c'],mass0=launch['mass0'],q0=1.)
    line.reset_s_at_end_turn=False
    _build_shared_cavity_tracker(line,context)
    scalar=xt.Particles(_context=context,**launch);line.track(scalar)
    desc=madng_tpsa.Descriptor(6,3,num_params=1,param_order=3)
    line['c'].knl[0]=desc.param(1,0.)
    series=xtpsa.ParticlesTpsa(order=3,descriptor=desc,**launch);line.track(series)
    np.testing.assert_allclose(series.const_part,_values(scalar),rtol=0.,atol=3e-12)
    cubic=np.array([series.coefficient(name,(0,0,0,0,0,0,3)) for name in COORDS])
    line.disable_tpsa_elements()
    offsets=np.arange(-3,4,dtype=float)
    # Seven scalar samples isolate the cubic Taylor coefficient while
    # cancelling monomials through degree six; TPSA itself uses no stencil.
    moments=np.array([offsets**power for power in range(7)])
    target=np.zeros(7);target[3]=1.
    weights=np.linalg.solve(moments,target)
    finite=np.zeros(6);step=.002
    for offset,weight in zip(offsets,weights):
        line['c'].knl[0]=offset*step
        particle=xt.Particles(_context=context,**launch);line.track(particle)
        assert particle.state[0]==1
        finite+=weight*_values(particle)/step**3
    np.testing.assert_allclose(cubic,finite,rtol=3e-4,atol=3e-7)
    print('Gaussian23 zero-bend cubic',cubic.tolist(),'FD',finite.tolist())


def test_gaussian_moving_reference_public_tpsa(cavity_magnetic_context):
    """Public SI/strength wiring of the full-static moving-reference word.

    The source2/factor1/reference2 mesh is an implementation check. It is not
    a numerical-accuracy recommendation or a physical/source qualification.
    """
    context=cavity_magnetic_context
    launch=dict(LAUNCH,p0c=xt.ELECTRON_MASS_EV*np.sqrt(2.4**2-1),x=4e-4,y=-3e-4)
    values=dict(phase=.4,eta=.25,frequency=400e6,calibration_beta=.9,
        reference_x=.001,reference_y=-.0008,reference_px=.003,reference_py=-.0025,
        ks=.3,normal=.004,skew=.001)
    line=xt.Line(elements={'c':xt.Cavity(length=.167,voltage=2e3,frequency=400e6,
        model='phase-averaged-moving-reference',rf_profile='gaussian',
        rf_profile_sigma=.035,rf_profile_center=.012,counter_phase=.3,
        num_kicks=2,num_factor_steps=1,num_reference_steps=2,
        edge_entry_active=False,edge_exit_active=False)})
    for name,value in values.items():
        line.vars[name]=value
        if name=='normal':line.element_refs['c'].knl[1]=line.vars[name]
        elif name=='skew':line.element_refs['c'].ksl[1]=line.vars[name]
        else:setattr(line.element_refs['c'],name,line.vars[name])
    line.particle_ref=xt.Particles(p0c=launch['p0c'],mass0=launch['mass0'],q0=1.)
    line.reset_s_at_end_turn=False
    _build_shared_cavity_tracker(line,context)
    scalar=xt.Particles(_context=context,**launch);line.track(scalar)
    assert scalar.state[0]==1
    desc=madng_tpsa.Descriptor(6,2,num_params=len(values),param_order=1)
    knobs=KnobParameters(line,list(values),desc);knobs.apply()
    series=xtpsa.ParticlesTpsa(order=2,descriptor=desc,**launch);line.track(series)
    np.testing.assert_allclose(series.const_part,_values(scalar),rtol=0.,atol=3e-12)
    gradients=series.param_jacobian().copy();jacobian=series.jacobian().copy()
    potential_gradients=np.array([series._local_series[name].param_grad() for name in ('ax','ay')])
    powers=[0]*(6+len(values));powers[1]=1;powers[6+list(values).index('reference_px')]=1
    mixed=series.coefficient('x',tuple(powers))
    knobs.teardown()
    weights={-2:1.,-1:-8.,1:8.,2:-1.}
    for column,(name,value) in enumerate(values.items()):
        step=abs(value)*1e-4;finite=np.zeros(6);finite_potential=np.zeros(2)
        for offset,weight in weights.items():
            line.vars[name]=value+offset*step
            particle=xt.Particles(_context=context,**launch);line.track(particle)
            assert particle.state[0]==1,name
            finite+=weight*_values(particle)/(12*step)
            finite_potential+=weight*np.array([float(particle.ax[0]),float(particle.ay[0])])/(12*step)
        line.vars[name]=value
        np.testing.assert_allclose(gradients[:,column]*abs(value),finite*abs(value),
                                   rtol=3e-6,atol=3e-8,err_msg=name)
        np.testing.assert_allclose(potential_gradients[:,column]*abs(value),finite_potential*abs(value),
                                   rtol=3e-6,atol=3e-8,err_msg=name+' potential metadata')
    finite_jacobian=np.zeros((6,6))
    for column,name in enumerate(COORDS):
        for offset,weight in weights.items():
            particle=xt.Particles(_context=context,**dict(launch,**{name:launch[name]+offset*1e-4}))
            line.track(particle)
            finite_jacobian[:,column]+=weight*_values(particle)/(12e-4)
    np.testing.assert_allclose(jacobian,finite_jacobian,rtol=3e-6,atol=3e-8)
    finite_mixed=0.
    for ip,wp in weights.items():
        for ir,wr in weights.items():
            line.vars['reference_px']=values['reference_px']+ir*1e-4
            particle=xt.Particles(_context=context,**dict(launch,px=launch['px']+ip*1e-3))
            line.track(particle)
            finite_mixed+=wp*wr*float(particle.x[0])/(144*1e-3*1e-4)
    line.vars['reference_px']=values['reference_px']
    np.testing.assert_allclose(mixed,finite_mixed,rtol=3e-5,atol=3e-8)
    print('Gaussian24 mixed dR12/dreference_px',mixed,'FD',finite_mixed)


def test_moving_reference_energy_frozen_analysis_metadata(cavity_magnetic_context):
    """Exercise the separate energy-constrained analysis word through Line.

    These end-state/derivative checks complement the independent physical
    stage-mask tests. They do not identify this constrained word with the
    unconstrained six-dimensional map or establish its physical accuracy.
    """
    context=cavity_magnetic_context
    launch=dict(LAUNCH,p0c=xt.ELECTRON_MASS_EV*np.sqrt(2.4**2-1))
    line=xt.Line(elements={'c':xt.Cavity(length=.167,voltage=2e3,frequency=400e6,
        phase=.4,eta=.25,counter_phase=.3,calibration_beta=.9,ks=.3,
        reference_x=.001,reference_y=-.0008,reference_px=.003,reference_py=-.0025,
        model='phase-averaged-moving-reference',rf_profile='gaussian',
        rf_profile_sigma=.035,rf_profile_center=.012,num_kicks=2,
        num_factor_steps=1,num_reference_steps=2,edge_entry_active=False,edge_exit_active=False)})
    line['c'].knl[1]=.004;line['c'].ksl[1]=.001
    line.particle_ref=xt.Particles(p0c=launch['p0c'],mass0=launch['mass0'],q0=1.)
    line.reset_s_at_end_turn=False
    line.freeze_energy(force=True)
    _build_shared_cavity_tracker(line,context)
    scalar=xt.Particles(_context=context,**launch)
    before={name:float(getattr(scalar,name)[0]) for name in ('delta','ptau','pzeta','rpp','rvv')}
    line.track(scalar)
    for name,value in before.items():
        assert float(getattr(scalar,name)[0])==pytest.approx(value,abs=3e-14,rel=0.)
    desc=madng_tpsa.Descriptor(6,1,num_params=1,param_order=1)
    line['c'].phase=desc.param(1,.4)
    series=xtpsa.ParticlesTpsa(order=1,descriptor=desc,**launch)
    derived_before={name:series._local_series[name].grad()[5]
                    for name in ('pzeta','rpp','rvv')}
    line.track(series)
    np.testing.assert_allclose(series.const_part,_values(scalar),rtol=0.,atol=3e-12)
    np.testing.assert_allclose(series.jacobian()[5],[0.,0.,0.,0.,0.,1.],rtol=0.,atol=3e-12)
    assert series.param_jacobian()[5,0]==0.
    for name,value in derived_before.items():
        assert series._local_series[name].const_part==pytest.approx(float(getattr(scalar,name)[0]),abs=3e-14,rel=0.)
        assert series._local_series[name].grad()[5]==pytest.approx(value,abs=3e-12,rel=0.)
    gradient=series.param_jacobian()[:,0].copy()
    line.disable_tpsa_elements()
    endpoints=[]
    for phase in (.4-1e-4,.4+1e-4):
        line['c'].phase=phase
        particle=xt.Particles(_context=context,**launch);line.track(particle)
        endpoints.append(_values(particle))
    np.testing.assert_allclose(gradient,(endpoints[1]-endpoints[0])/2e-4,rtol=3e-5,atol=3e-8)
