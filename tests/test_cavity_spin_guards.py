"""Cavity spin follows the magnets: turned where supported, otherwise the particle is lost."""
import numpy as np
import pytest
import xobjects as xo
import xtrack as xt


def eligible_line(**changes):
    controls=dict(model='_phase-resolved',length=.05,frequency=400e6,
                  voltage=2e4,phase=.4,num_kicks=8)
    controls.update(changes)
    if controls['model']=='_phase-resolved':
        controls.setdefault('counter_wave_amplitude',.5)
    line=xt.Line(elements={'rf':xt.MagnetOverlappedCavity(**controls)})
    line.particle_ref=xt.Particles(p0c=14e6,mass0=xt.ELECTRON_MASS_EV)
    return line


def spinning_and_still(context):
    """One particle with spin and one without, otherwise alike."""
    return xt.Particles(_context=context,p0c=14e6,mass0=xt.ELECTRON_MASS_EV,
                        x=[2e-4,2e-4],spin_x=[.6,0.],spin_z=[.8,0.],
                        anomalous_magnetic_moment=.00116)


@pytest.mark.parametrize('strength',[{},dict(ks=.1),dict(knl=[0.,.001]),dict(ksl=[0.,.001])])
def test_supported_cavity_turns_the_spin(tmp_path,strength):
    faces={} if not strength else dict(edge_entry_active=0,edge_exit_active=0)
    context = xo.ContextCpu()
    line=eligible_line(**strength,**faces)
    line.build_tracker(_context=context)
    line.configure_spin(True)
    particles=spinning_and_still(context)
    line.track(particles)
    np.testing.assert_array_equal(particles.state,[1,1])
    assert particles.spin_x[0]!=.6
    assert particles.spin_x[1]==particles.spin_z[1]==0.


def test_radiation_alone_also_turns_the_spin(tmp_path):
    context = xo.ContextCpu()
    line=eligible_line()
    line.build_tracker(_context=context)
    line.configure_radiation(model='mean')
    particles=spinning_and_still(context)
    line.track(particles)
    np.testing.assert_array_equal(particles.state,[1,1])
    assert particles.spin_x[0]!=.6


@pytest.mark.parametrize('changes',[
    dict(model='_phase-resolved-frozen', counter_wave_amplitude=0., knl=[0., .05]),
    dict(ks=.1),
    dict(knl=[0.,0.,.1],order=2),
    dict(ks=.1,edge_entry_active=0,edge_exit_active=0,shift_x=1e-4),
])
def test_unsupported_cavity_loses_only_the_spinning_particle(tmp_path,changes):
    context = xo.ContextCpu()
    line=eligible_line(**changes)
    line.build_tracker(_context=context)
    line.configure_spin(True)
    particles=spinning_and_still(context)
    line.track(particles)
    np.testing.assert_array_equal(particles.state[np.argsort(particles.particle_id)],[-99744,1])


def test_sliced_cavity_loses_only_the_spinning_particle(tmp_path):
    context = xo.ContextCpu()
    line=eligible_line()
    line.slice_thick_elements([xt.Strategy(xt.Teapot(2))])
    line.build_tracker(_context=context)
    line.configure_spin(True)
    particles=spinning_and_still(context)
    line.track(particles)
    np.testing.assert_array_equal(particles.state[np.argsort(particles.particle_id)],[-99744,1])


@pytest.mark.parametrize('flag',['XS_FLAG_KILL_CAVITY_KICK','XS_FLAG_SR_KICK_SAME_AS_FIRST'])
def test_closed_orbit_analysis_loses_only_the_spinning_particle(tmp_path,flag):
    context = xo.ContextCpu()
    line=eligible_line()
    line.build_tracker(_context=context)
    line.configure_spin(True)
    setattr(line.tracker.track_flags,flag,True)
    particles=spinning_and_still(context)
    line.track(particles)
    np.testing.assert_array_equal(particles.state[np.argsort(particles.particle_id)],[-99744,1])


@pytest.mark.parametrize('changes',[dict(integrator='teapot'),
                                    dict(n_cells=2),dict(counter_wave_amplitude=.55)])
def test_later_invalid_configuration_loses_the_particle(tmp_path,changes):
    """Settings changed after construction are refused by the kernel (-99744)."""
    context = xo.ContextCpu()
    line=eligible_line(model='_phase-resolved-frozen',counter_wave_amplitude=0.)
    line.build_tracker(_context=context)
    for name,value in changes.items():
        setattr(line['rf'],name,value)
    particle=xt.Particles(_context=context,p0c=14e6,mass0=xt.ELECTRON_MASS_EV,x=2e-4)
    line.track(particle)
    assert particle.state[0]==-99744
