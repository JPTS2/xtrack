"""Public configuration boundaries of physical-model Cavity spin transport."""
import numpy as np
import pytest
import xtrack as xt
from _cavity_context import kernel_context


def eligible_line(**changes):
    controls=dict(model='phase-resolved',length=.05,frequency=400e6,
                  voltage=2e4,phase=.4,num_kicks=8)
    controls.update(changes)
    if controls['model']=='phase-resolved':
        controls.setdefault('eta',.25)
    line=xt.Line(elements={'rf':xt.Cavity(**controls)})
    line.particle_ref=xt.Particles(p0c=14e6,mass0=xt.ELECTRON_MASS_EV)
    return line


def snapshot(line):
    return (line._has_valid_tracker(),getattr(line,'_spin_model',None),
            dict(line.config),line['rf'].spin_tracking)


@pytest.mark.parametrize('changes',[
    dict(model='phase-resolved-frozen'),
    dict(rf_profile='uniform'),
    dict(rf_profile='gaussian',rf_profile_sigma=.01),
    dict(ks=.1),
    dict(knl=[0.,0.,.1],order=2),
    dict(ks=.1,edge_entry_active=0,edge_exit_active=0,shift_x=1e-4),
])
def test_unsupported_spin_configuration_refuses_atomically(changes):
    line=eligible_line(**changes)
    before=snapshot(line)
    with pytest.raises(NotImplementedError,match='Cavity spin'):
        line.configure_spin(True)
    assert snapshot(line)==before


@pytest.mark.parametrize('flag',['FREEZE_VAR_x','XTRACK_TPSA_TRACK'])
def test_frozen_or_tpsa_line_refuses_spin(flag):
    line=eligible_line();line.config[flag]=True;before=snapshot(line)
    with pytest.raises(NotImplementedError,match='Cavity spin'):
        line.configure_spin(True)
    assert snapshot(line)==before


def test_sliced_cavity_refuses_spin():
    line=eligible_line()
    line.slice_thick_elements([xt.Strategy(xt.Teapot(2))])
    with pytest.raises(NotImplementedError,match='through slices'):
        line.configure_spin(True)


@pytest.mark.parametrize('strength',[{},dict(ks=.1),dict(knl=[0.,.001]),dict(ksl=[0.,.001])])
def test_supported_spin_configuration_and_disable(strength):
    faces={} if not strength else dict(edge_entry_active=0,edge_exit_active=0)
    line=eligible_line(**strength,**faces)
    line.configure_spin(True)
    assert line['rf'].spin_tracking==1
    line.configure_spin(False)
    assert line['rf'].spin_tracking==0


def test_later_unsupported_mutation_refuses_before_tracking(tmp_path):
    with kernel_context(tmp_path) as context:
        line=eligible_line()
        line.build_tracker(_context=context)
        line.configure_spin(True)
        line['rf'].rf_profile='uniform'
        particle=xt.Particles(_context=context,p0c=14e6,mass0=xt.ELECTRON_MASS_EV,
            x=2e-4,px=3e-4,zeta=4e-4,delta=.002,spin_x=.6,spin_z=.8)
        with pytest.raises(NotImplementedError,match='Cavity spin'):
            line.track(particle)
        assert particle.state[0]==1
        np.testing.assert_array_equal([particle.x[0],particle.spin_x[0]],[2e-4,.6])


@pytest.mark.parametrize('changes',[dict(integrator='teapot'),
                                    dict(rf_field_order=1),dict(eta=.3)])
def test_later_invalid_configuration_raises_before_tracking(tmp_path,changes):
    with kernel_context(tmp_path) as context:
        line=eligible_line(model='phase-resolved-frozen')
        line.build_tracker(_context=context)
        for name,value in changes.items():
            setattr(line['rf'],name,value)
        particle=xt.Particles(_context=context,p0c=14e6,mass0=xt.ELECTRON_MASS_EV,x=2e-4)
        with pytest.raises(ValueError):
            line.track(particle)
        assert particle.state[0]==1


def test_slicing_after_spin_configuration_refuses_before_tracking(tmp_path):
    with kernel_context(tmp_path) as context:
        line=eligible_line()
        line.configure_spin(True)
        line.slice_thick_elements([xt.Strategy(xt.Teapot(2))])
        line.build_tracker(_context=context)
        particle=xt.Particles(_context=context,p0c=14e6,mass0=xt.ELECTRON_MASS_EV,
                              x=2e-4,spin_z=1.)
        with pytest.raises(NotImplementedError,match='through slices'):
            line.track(particle)
        assert particle.state[0]==1


def test_tpsa_tracking_with_spin_refuses_before_tracking(tmp_path):
    import xtrack.tpsa as xtpsa
    with kernel_context(tmp_path) as context:
        line=eligible_line()
        line.build_tracker(_context=context)
        line.configure_spin(True)
        with pytest.raises(NotImplementedError,match='not TPSA'):
            line.track(xtpsa.ParticlesTpsa(order=1,p0c=14e6,
                                           mass0=xt.ELECTRON_MASS_EV,x=2e-4))


@pytest.mark.parametrize('flag',['XS_FLAG_KILL_CAVITY_KICK','XS_FLAG_SR_KICK_SAME_AS_FIRST'])
def test_spin_with_closed_orbit_analysis_flags_refuses(tmp_path,flag):
    with kernel_context(tmp_path) as context:
        line=eligible_line()
        line.build_tracker(_context=context)
        line.configure_spin(True)
        setattr(line.tracker.track_flags,flag,True)
        particle=xt.Particles(_context=context,p0c=14e6,mass0=xt.ELECTRON_MASS_EV,
                              x=2e-4,spin_z=1.)
        with pytest.raises(NotImplementedError,match='4D'):
            line.track(particle)
        assert particle.state[0]==1
