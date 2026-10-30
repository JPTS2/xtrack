"""Public Gaussian magnetic model: ordinary Line and Taylor workflows.

Independent physical/source oracles belong to the native and review tests.
These checks exercise the declared finite map, not release error bounds.
"""
import numpy as np
import pytest
import xtrack as xt

from test_cavity_native import native_context


COORDS=('x','px','y','py','zeta','delta')
MASS=xt.ELECTRON_MASS_EV
LAUNCH=dict(mass0=MASS,q0=1.,p0c=MASS*np.sqrt(2.4**2-1.),
    x=3e-4,px=2e-4,y=-2e-4,py=-1e-4,zeta=1e-4,delta=.002)
HARDWARE=dict(length=.05,frequency=400e6,voltage=2e3,phase=.4,eta=.25,
    counter_phase=.3,calibration_beta=1.,rf_profile='gaussian',
    rf_profile_sigma=.012,rf_profile_center=.002,ks=.2,knl=[0.,.01],
    model='phase-averaged-magnetic',integrator='midpoint',num_kicks=4,
    num_factor_steps=1,edge_entry_active=0,edge_exit_active=0)


def values(p):
    assert p.state[0]==1
    return np.array([float(getattr(p,name)[0]) for name in COORDS])


@pytest.fixture(scope='module')
def shared_line(tmp_path_factory):
    with native_context(tmp_path_factory.mktemp('gaussian-magnetic-public')) as context:
        line=xt.Line(elements=[xt.Cavity(_context=context,**HARDWARE),
            xt.ReferenceEnergyIncrease(_context=context,Delta_p0c=0.),
            xt.Marker(_context=context)],element_names=['rf','reference_update','end'])
        line.particle_ref=xt.Particles(_context=context,mass0=MASS,q0=1.,p0c=LAUNCH['p0c'])
        for variable,field,value in [('phase','phase',.4),('mix','eta',.25),
            ('sigma_m','rf_profile_sigma',.012),('center_m','rf_profile_center',.002)]:
            line.vars[variable]=value
            setattr(line.element_refs['rf'],field,line.vars[variable])
        line.build_tracker(_context=context)
        line.tracker.track_flags.XS_FLAG_USE_TIME_BASED_ZETA_UPDATE=True
        yield line


@pytest.fixture
def magnetic_line(shared_line):
    line=shared_line
    for name,value in [('phase',.4),('mix',.25),('sigma_m',.012),('center_m',.002)]:
        line.vars[name]=value
    line['reference_update'].Delta_p0c=0.
    yield line
    if line['rf']._tpsa_enabled:line.disable_tpsa_elements()


def endpoint(line,**overrides):
    p=xt.Particles(_context=line._context,**dict(LAUNCH,**overrides))
    line.track(p,ele_start=0,ele_stop=1)
    return values(p)


def test_gaussian_model_line_twiss_reverse_copy_serialization(magnetic_line):
    line=magnetic_line
    initial=xt.Particles(_context=line._context,**LAUNCH)
    p=initial.copy();line.track(p,ele_start=0,ele_stop=1)
    expected=values(p)
    line.track(p,ele_start=0,ele_stop=1,backtrack=True)
    np.testing.assert_allclose(values(p),values(initial),rtol=0.,atol=3e-11)
    line['reference_update'].Delta_p0c=LAUNCH['p0c']*expected[5]
    tw=line.twiss(method='6d',start=xt.START,end=xt.END,
        **{name:LAUNCH[name] for name in COORDS},betx=2.,alfx=.1,bety=3.,alfy=-.2)
    assert np.all(np.isfinite(tw.betx)) and np.all(tw.betx>0.)
    assert np.all(np.isfinite(tw.bety)) and np.all(tw.bety>0.)
    line['reference_update'].Delta_p0c=0.
    kernels=line.tracker.track_kernel
    for copied in (line.copy(_context=line._context),xt.Line.from_dict(line.to_dict())):
        if copied.tracker is not None:copied.discard_tracker()
        copied.build_tracker(_context=line._context,track_kernel=kernels)
        assert copied['rf'].rf_profile=='gaussian'
        assert copied['rf'].rf_profile_sigma==.012
        assert copied['rf'].rf_profile_center==.002
        np.testing.assert_allclose(endpoint(copied),expected,rtol=0.,atol=3e-12)
    line.vars['sigma_m']=.015
    line.vars['center_m']=-.001
    changed=endpoint(line)
    assert np.max(abs(changed-expected))>1e-10
    assert line['rf'].rf_profile_sigma==.015
    assert line['rf'].rf_profile_center==-.001


def test_gaussian_model_tpsa_state_parameters_and_mixed(magnetic_line):
    import madng_tpsa
    import xtrack.tpsa as xtpsa

    line=magnetic_line
    scalar=endpoint(line)
    descriptor=madng_tpsa.Descriptor(6,2,num_params=4,param_order=1)
    knobs=xtpsa.KnobParameters(line,['phase','mix','sigma_m','center_m'],descriptor)
    knobs.apply()
    try:
        series=xtpsa.ParticlesTpsa(order=2,descriptor=descriptor,**LAUNCH)
        line.track(series,ele_start=0,ele_stop=1)
        assert series._xobject.state==1
        np.testing.assert_allclose(series.const_part,scalar,rtol=0.,atol=3e-12)
        jac=series.jacobian().copy();derivatives=series.param_jacobian().copy()
        mixed=series.coefficient('x',(0,1,0,0,0,0,0,0,1,0))
    finally:
        knobs.teardown()
    assert not line['rf']._tpsa_enabled
    assert np.all(np.isfinite(jac)) and np.all(np.isfinite(derivatives))
    for column,(name,value,step) in enumerate([('phase',.4,1e-5),
            ('mix',.25,3e-5),('sigma_m',.012,2e-6),('center_m',.002,2e-6)]):
        samples=[]
        for sign in (-1,1):
            line.vars[name]=value+sign*step;samples.append(endpoint(line))
        line.vars[name]=value
        finite=(samples[1]-samples[0])/(2*step)
        assert np.max(abs(derivatives[:,column]))>1e-9
        np.testing.assert_allclose(derivatives[:,column],finite,rtol=3e-5,atol=2e-8)
    hpx=2e-6
    finite=(endpoint(line,px=LAUNCH['px']+hpx)
            -endpoint(line,px=LAUNCH['px']-hpx))/(2*hpx)
    np.testing.assert_allclose(jac[:,1],finite,rtol=3e-6,atol=3e-9)
    mixed_fd=0.;hpx=2e-4;hsigma=5e-5
    for ip in (-1,1):
        for iw in (-1,1):
            line.vars['sigma_m']=.012+iw*hsigma
            mixed_fd+=ip*iw*endpoint(line,px=LAUNCH['px']+ip*hpx)[0]/(4*hpx*hsigma)
    line.vars['sigma_m']=.012
    np.testing.assert_allclose(mixed,mixed_fd,rtol=3e-3,atol=3e-8)
