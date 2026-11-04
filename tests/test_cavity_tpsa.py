"""Derivatives of the cavity maps: tracking and source-parameter series against finite
differences and analytic results.
"""
import numpy as np
import pytest
import madng_tpsa
import xtrack as xt
import xtrack.tpsa as xtpsa
from xtrack.tpsa._knobs import KnobParameters
from _cavity_context import kernel_context


COORDS = ('x', 'px', 'y', 'py', 'zeta', 'delta')


# Independently chosen synthetic launch; no machine momentum is embedded.
LAUNCH = dict(p0c=14e6, mass0=xt.ELECTRON_MASS_EV,
              q0=1., x=2e-4, px=3e-4, y=-1e-4, py=-2e-4,
              zeta=4e-4, delta=2e-3, time_s=1.9e-9)


def _values(particles):
    return np.array([getattr(particles, name)[0] for name in COORDS])


def test_frozen_energy_drift_uses_input_delta_series(tmp_path):
    with kernel_context(tmp_path) as context:
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


@pytest.mark.parametrize('order,model,static_fields', [
    (2,'phase-resolved',False),
    (3,'phase-resolved',False),
    (3,'phase-resolved',True)])
def test_calibration_order_one_handle_on_higher_order_map(
        cavity_magnetic_context, order, model, static_fields):
    """Descriptor.param's direct order-one handle must not alter source constants."""
    # A weak straight-static case also checks the retained generic static RHS.
    context = cavity_magnetic_context
    launch = dict(LAUNCH, p0c=xt.ELECTRON_MASS_EV*np.sqrt(2.4**2-1))
    fields = dict(ks=.2, knl=[0.,.01], ksl=[0.,0.,.005]) if static_fields else {}
    line = xt.Line(elements={'c':xt.Cavity(
        length=.05 if static_fields else .167,
        voltage=2e3 if static_fields else 2e4,
        frequency=400e6, phase=.4, eta=.25, counter_phase=.3,
        calibration_beta=.9, model=model,
        num_kicks=8, **fields)})
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
    # Order three also checks that the fixed stage sweeps after primal
    # convergence settle the cubic coefficients.
    check_cubic = order == 3
    if check_cubic:
        cubic = np.array([series.coefficient('x', powers) for powers in (
            (0,1,2,0,0,0,0), (0,3,0,0,0,0,0))])
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
    if check_cubic:
        # Differentiate independently tracked ORDER-TWO coefficient maps.
        # d/dpx [x]_(y^2) = [x]_(px*y^2), whereas d/dpx [x]_(px^2)
        # is three times [x]_(px^3), in the divided-coefficient convention.
        line['c'].calibration_beta = .9
        lower = madng_tpsa.Descriptor(6, 2)
        finite_cubic = np.zeros(2)
        for offset, weight in weights.items():
            ray = xtpsa.ParticlesTpsa(order=2, descriptor=lower,
                **dict(launch, px=launch['px']+offset*1e-3))
            line.track(ray)
            assert ray._xobject.state == 1
            finite_cubic += weight*np.array([
                ray.coefficient('x', (0,0,2,0,0,0)),
                ray.coefficient('x', (0,2,0,0,0,0))/3.])/(12e-3)
        assert np.all(abs(cubic) > .01)
        np.testing.assert_allclose(cubic, finite_cubic, rtol=3e-5, atol=3e-7)


def test_zero_primal_static_controls_keep_derivatives(cavity_magnetic_context):
    """Zero hardware values with live static jets must retain the static flow."""
    context = cavity_magnetic_context
    mass = xt.ELECTRON_MASS_EV
    launch = dict(LAUNCH, p0c=mass*np.sqrt(2.4**2-1), x=4e-4, y=-3e-4)
    line = xt.Line(elements={'c':xt.Cavity(length=.05, voltage=2e3,
        frequency=400e6, phase=.4, eta=.25, counter_phase=.3,
        calibration_beta=1., model='phase-resolved', num_kicks=4)})
    controls = ('solenoid', 'normal', 'skew')
    for name in controls:
        line.vars[name] = 0.
    line.element_refs['c'].ks = line.vars['solenoid']
    line.element_refs['c'].knl[1] = line.vars['normal']
    line.element_refs['c'].ksl[2] = line.vars['skew']
    line.particle_ref = xt.Particles(p0c=launch['p0c'], mass0=mass, q0=1.)
    line.reset_s_at_end_turn = False
    _build_shared_cavity_tracker(line, context)
    scalar = xt.Particles(_context=context, **launch)
    line.track(scalar)
    assert scalar.state[0] == 1
    descriptor = madng_tpsa.Descriptor(6, 2, num_params=3, param_order=1)
    knobs = KnobParameters(line, controls, descriptor)
    knobs.apply()
    series = xtpsa.ParticlesTpsa(order=2, descriptor=descriptor, **launch)
    line.track(series)
    assert series._xobject.state == 1
    np.testing.assert_allclose(series.const_part, _values(scalar), rtol=0., atol=3e-13)
    gradient = series.param_jacobian().copy()
    mixed = series.coefficient('px', (1,0,0,0,0,0,0,1,0))
    knobs.teardown()
    for column, name in enumerate(controls):
        endpoints = []
        for sign in (-1, 1):
            line.vars[name] = sign*1e-5
            ray = xt.Particles(_context=context, **launch)
            line.track(ray)
            assert ray.state[0] == 1
            endpoints.append(_values(ray))
        line.vars[name] = 0.
        finite = (endpoints[1]-endpoints[0])/(2e-5)
        assert np.max(abs(gradient[:, column])) > 1e-7
        np.testing.assert_allclose(gradient[:, column], finite, rtol=3e-6, atol=3e-10)
    finite_mixed = 0.
    for sx in (-1, 1):
        for sg in (-1, 1):
            line.vars['normal'] = sg*1e-4
            ray = xt.Particles(_context=context,
                **dict(launch, x=launch['x']+sx*1e-5))
            line.track(ray)
            assert ray.state[0] == 1
            finite_mixed += sx*sg*float(ray.px[0])/(4e-9)
    line.vars['normal'] = 0.
    assert abs(mixed) > 1e-4
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
    with kernel_context(tmp_path) as context:
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
    with kernel_context(tmp_path) as context:
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


def test_legacy_standalone_cavity_nonzero_s_tpsa(tmp_path):
    with kernel_context(tmp_path) as context:
        element = xt.Cavity(_context=context, voltage=2e5, frequency=400e6,
                            phase=.7, absolute_time=True)
        launch = dict(LAUNCH, s=.17)
        scalar = xt.Particles(_context=context, **launch)
        series = xtpsa.ParticlesTpsa(order=1, **launch)
        element.track(scalar)
        element.track(series)
        np.testing.assert_allclose(series.const_part, _values(scalar),
                                   rtol=0., atol=3e-14)


@pytest.mark.parametrize('model', ['phase-resolved'])
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
        calibration_beta=.9, model=model, integrator='uniform',
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


@pytest.mark.parametrize('model',['phase-resolved'])
@pytest.mark.parametrize('explicit_carrier',[False,True])
def test_source_cavity_tpsa_continuous_controls(cavity_magnetic_context,model,explicit_carrier):
    mass=xt.ELECTRON_MASS_EV
    launch=dict(LAUNCH,p0c=mass*np.sqrt(2.4**2-1))
    values=dict(voltage=2e4,frequency=400e6,counter_phase=.3,
        calibration_beta=.9,ks=.03,phase_taper=.07,lag_taper=1.2)
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
    with kernel_context(tmp_path) as context:
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


def test_frozen_solenoid_zero_strength_parameter_tpsa(tmp_path):
    """The exact frozen SOL body uses an entire chart at zero field."""
    mass=xt.ELECTRON_MASS_EV
    launch=dict(LAUNCH,p0c=mass*np.sqrt(2.4**2-1))
    with kernel_context(tmp_path) as context:
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
    with kernel_context(tmp_path) as context:
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
    with kernel_context(tmp_path_factory.mktemp('cavity_magnetic_tpsa')) as context:
        yield context


@pytest.mark.parametrize('model',['phase-resolved','phase-resolved-frozen'])
def test_native_indexed_magnetic_parameter_tpsa(cavity_magnetic_context,model):
    mass=xt.ELECTRON_MASS_EV
    launch=dict(LAUNCH,p0c=mass*np.sqrt(2.4**2-1),x=4e-4,y=-3e-4)
    frozen=model=='phase-resolved-frozen'
    values={'normal':0. if frozen else .01}
    if not frozen: values['skew']=.005
    context=cavity_magnetic_context
    line=xt.Line(elements={'c':xt.Cavity(length=.05,voltage=2e4,
        frequency=400e6,phase=.4,eta=0. if frozen else .25,
        model=model,num_kicks=8)})
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


def test_gaussian_magnetic_profile_tpsa(cavity_magnetic_context):
    """Differentiate the native Gaussian M1 lift, including source calibration."""
    context=cavity_magnetic_context
    launch=dict(LAUNCH,p0c=xt.ELECTRON_MASS_EV*np.sqrt(2.4**2-1))
    values=dict(phase=.4,eta=.25,counter_phase=.3,voltage=2e4,
        frequency=400e6,calibration_beta=.9,rf_profile_sigma=.035,
        rf_profile_center=.012)
    line=xt.Line(elements={'c':xt.Cavity(length=.167,
        model='phase-resolved',rf_profile='gaussian',
        num_kicks=8,**values)})
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

