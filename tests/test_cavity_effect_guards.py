"""Atomic public configuration boundaries for the first cavity effects route.

Field, force-law and applied-transport oracles live in test_cavity_effects.py.
These tests exercise eligibility and configuration, without inferring physics
qualification from a flag being accepted.
"""
import pytest
import xtrack as xt
import madng_tpsa
from xtrack.beam_elements.cavity import _CAVITY_MODELS
from test_cavity_native import native_context


def eligible_line(**changes):
    controls=dict(model='phase-resolved',length=.05,frequency=400e6,
                  voltage=2e4,phase=.4,eta=.25,num_kicks=8)
    controls.update(changes)
    line=xt.Line(elements={'rf':xt.Cavity(**controls)})
    line.particle_ref=xt.Particles(p0c=14e6,mass0=xt.ELECTRON_MASS_EV)
    return line


def snapshot(line):
    flags=dict(line.tracker.track_flags.flags) if line._has_valid_tracker() else None
    return (line.tracker,getattr(line,'_radiation_model',None),
        getattr(line,'_spin_model',None),getattr(line,'_needs_rng',False),
        dict(line.config),flags,
        tuple((id(ee),ee.radiation_flag,ee.spin_tracking) for ee in line._used_cavity_owners()))


def request(line,effect):
    if effect=='mean':line.configure_radiation(model='mean')
    else:line.configure_spin(True)


def draft_request(line,effect):
    line._validate_cavity_first_effect('radiation' if effect=='mean' else 'spin',
                                      radiation_model='mean' if effect=='mean' else None)


@pytest.mark.parametrize('effect',['mean','spin'])
def test_draft_validator_is_read_only(effect):
    line=eligible_line();before=snapshot(line)
    draft_request(line,effect)
    assert snapshot(line)==before
    assert line.tracker is None


@pytest.mark.parametrize('model',range(11,25))
@pytest.mark.parametrize('effect',['mean','spin'])
def test_other_cavity_models_refuse_atomically(model,effect):
    line=eligible_line();line['rf'].model=_CAVITY_MODELS[model];before=snapshot(line)
    with pytest.raises(NotImplementedError,match='[Cc]avity'):
        request(line,effect)
    assert snapshot(line)==before


@pytest.mark.parametrize('profile',['uniform','gaussian'])
@pytest.mark.parametrize('effect',['mean','spin'])
def test_other_rf_profiles_refuse_atomically(profile,effect):
    line=eligible_line(rf_profile=profile,rf_profile_sigma=.01);before=snapshot(line)
    with pytest.raises(NotImplementedError,match='[Cc]avity'):
        request(line,effect)
    assert snapshot(line)==before


@pytest.mark.parametrize('effect',['mean','spin'])
def test_zero_strength_gaussian_magnetic_profile_refuses_atomically(effect):
    line=eligible_line(magnetic_profile='gaussian',magnetic_profile_sigma=.01,
                       edge_entry_active=False,edge_exit_active=False)
    before=snapshot(line)
    with pytest.raises(NotImplementedError,match='[Cc]avity'):
        request(line,effect)
    assert snapshot(line)==before


@pytest.mark.parametrize('field',['ks','normal','skew','stored_above_order'])
@pytest.mark.parametrize('effect',['mean','spin'])
def test_all_stored_static_strengths_refuse_atomically(field,effect):
    line=eligible_line()
    if field=='ks':line['rf'].ks=.1
    elif field=='normal':line['rf'].knl[1]=.001
    elif field=='skew':line['rf'].ksl[1]=.001
    else:
        line['rf'].order=0
        line['rf'].knl[5]=.001
    before=snapshot(line)
    with pytest.raises(NotImplementedError,match='[Cc]avity'):
        request(line,effect)
    assert snapshot(line)==before


@pytest.mark.parametrize('flag',['FREEZE_VAR_x','FREEZE_VAR_delta','FREEZE_VAR_spin_x','XTRACK_TPSA_TRACK'])
@pytest.mark.parametrize('effect',['mean','spin'])
def test_frozen_or_tpsa_configuration_refuses_atomically(flag,effect):
    line=eligible_line();line.config[flag]=True;before=snapshot(line)
    with pytest.raises(NotImplementedError,match='[Cc]avity'):
        request(line,effect)
    assert snapshot(line)==before


@pytest.mark.parametrize('owner',['cavity','zero_array','other_used'])
@pytest.mark.parametrize('effect',['mean','spin'])
def test_active_series_refuse_without_constant_projection(owner,effect):
    line=eligible_line();desc=madng_tpsa.Descriptor(6,2,num_params=1,param_order=1)
    if owner=='cavity':line['rf'].phase=desc.param(1,.4)
    elif owner=='zero_array':line['rf'].knl[0]=desc.param(1,0.)
    else:
        line.env.new('q',xt.Quadrupole,length=.01,k1=.1)
        line.element_names=['q','rf'];line['q'].k1=desc.param(1,.1)
    before=snapshot(line)
    with pytest.raises(NotImplementedError,match='[Cc]avity'):
        request(line,effect)
    assert snapshot(line)==before
    assert any(getattr(ee._xobject,'_tpsa_enabled',0) for ee in line._element_dict.values())
    line.disable_tpsa_elements()


@pytest.mark.parametrize('effect',['mean','spin'])
def test_non_cpu_configuration_refuses_before_tracker_creation(monkeypatch,effect):
    line=eligible_line();before=snapshot(line)
    monkeypatch.setattr(type(line),'_context',property(lambda self:object()))
    with pytest.raises(NotImplementedError,match='[Cc]avity'):
        request(line,effect)
    assert snapshot(line)==before


@pytest.mark.parametrize('effect',['mean','spin'])
def test_first_effects_slice_request_refuses_atomically(effect):
    line=eligible_line()
    line.slice_thick_elements([xt.Strategy(slicing=xt.Teapot(2,mode='thick'),element_type=xt.Cavity)])
    before=snapshot(line)
    with pytest.raises(NotImplementedError,match='[Cc]avity'):
        request(line,effect)
    assert snapshot(line)==before


@pytest.mark.parametrize('flag',['XS_FLAG_KILL_CAVITY_KICK','XS_FLAG_SR_TAPER','XS_FLAG_SR_KICK_SAME_AS_FIRST'])
@pytest.mark.parametrize('effect',['mean','spin'])
def test_analysis_track_flags_refuse_atomically(flag,effect):
    line=eligible_line();line.build_tracker(compile=False)
    setattr(line.tracker.track_flags,flag,True);before=snapshot(line)
    with pytest.raises(NotImplementedError,match='[Cc]avity'):
        request(line,effect)
    assert snapshot(line)==before


@pytest.mark.parametrize('radiation',['quantum','quantum-kick'])
def test_quantum_request_refuses_atomically(radiation):
    line=eligible_line();before=snapshot(line)
    with pytest.raises(NotImplementedError,match='[Cc]avity'):
        line.configure_radiation(model=radiation)
    assert snapshot(line)==before


@pytest.mark.parametrize('changes',[dict(rf_field_order=1),dict(integrator='midpoint')])
@pytest.mark.parametrize('effect',['mean','spin'])
def test_other_retained_order_or_integrator_refuses_atomically(changes,effect):
    line=eligible_line(**changes);before=snapshot(line)
    with pytest.raises(NotImplementedError,match='[Cc]avity'):
        request(line,effect)
    assert snapshot(line)==before


@pytest.mark.parametrize('kwargs',[dict(model='invalid'),
    dict(model='mean',model_beamstrahlung='invalid'),dict(model='mean',model_bhabha='invalid')])
def test_invalid_radiation_arguments_do_not_create_tracker(kwargs):
    line=eligible_line();before=snapshot(line)
    with pytest.raises(AssertionError):line.configure_radiation(**kwargs)
    assert snapshot(line)==before


def test_supported_mean_and_spin_configuration_and_disable():
    line=eligible_line()
    line.configure_radiation(model='mean')
    line.configure_spin(True)
    assert line['rf'].radiation_flag==1
    assert line['rf'].spin_tracking==1
    assert line.config.get('XTRACK_MULTIPOLE_NO_SYNRAD',False) is False
    line.configure_radiation(model=None);line.configure_spin(False)
    assert line['rf'].radiation_flag==line['rf'].spin_tracking==0
    assert line.config.get('XTRACK_MULTIPOLE_NO_SYNRAD',False) is True


def test_supported_replica_and_unused_series_are_scoped():
    line=eligible_line()
    line.env.new('alias','rf',mode='replica');line.element_names=['alias']
    line.env.new('unused',xt.Cavity)
    desc=madng_tpsa.Descriptor(6,1,num_params=1,param_order=1)
    line['unused'].phase=desc.param(1,.1)
    line.configure_spin(True)
    assert line['rf'].spin_tracking==1
    assert line['unused'].spin_tracking==0
    assert line['unused']._tpsa_enabled
    line['unused'].disable_tpsa()


def test_mean_reverse_refuses_but_spin_only_configuration_allows_signed_reverse():
    line=eligible_line();line.build_tracker(compile=False)
    line.tracker.track_flags.XS_FLAG_BACKTRACK=True;before=snapshot(line)
    with pytest.raises(NotImplementedError,match='[Cc]avity'):
        line.configure_radiation(model='mean')
    assert snapshot(line)==before
    line.configure_spin(True)
    assert line['rf'].spin_tracking==1


@pytest.fixture(scope='module')
def runtime_context(tmp_path_factory):
    with native_context(tmp_path_factory.mktemp('cavity-effects-runtime-policy')) as context:
        yield context


def runtime_line(context):
    line=eligible_line()
    line.build_tracker(_context=context,compile=False,
        track_kernel=getattr(context,'_cavity_effect_guard_kernels',None))
    context._cavity_effect_guard_kernels=line.tracker.track_kernel
    line.configure_radiation(model='mean')
    return line


def scalar_launch(context):
    return xt.Particles(_context=context,p0c=14e6,mass0=xt.ELECTRON_MASS_EV,
        x=2e-4,px=3e-4,y=-1e-4,py=-2e-4,zeta=4e-4,delta=.002,
        spin_x=.2,spin_y=.3,spin_z=.8)


def physical_state(particle):
    return [float(getattr(particle,name)[0]) for name in
            ('x','px','y','py','zeta','delta','spin_x','spin_y','spin_z')]


@pytest.mark.parametrize('change',['profile','static','stored_above_order','quantum','active_series'])
def test_later_unsupported_mutation_refuses_native_transport(runtime_context,change):
    line=runtime_line(runtime_context)
    if change=='profile':line['rf'].rf_profile='uniform'
    elif change=='static':line['rf'].ks=.1
    elif change=='stored_above_order':
        line['rf'].order=0;line['rf'].knl[5]=.001
    elif change=='quantum':line['rf'].radiation_flag=2
    else:
        desc=madng_tpsa.Descriptor(6,1,num_params=1,param_order=1)
        line['rf'].phase=desc.param(1,.4)
    particle=scalar_launch(runtime_context);before=physical_state(particle)
    line.track(particle)
    assert particle.state[0]<0
    assert physical_state(particle)==before
    if change=='active_series':line.disable_tpsa_elements()


@pytest.mark.parametrize('flag',['FREEZE_VAR_rpp','FREEZE_VAR_spin_x','XTRACK_MULTIPOLE_NO_SYNRAD'])
def test_later_analysis_flags_refuse_native_transport(runtime_context,flag):
    line=runtime_line(runtime_context);line.config[flag]=True
    particle=scalar_launch(runtime_context);before=physical_state(particle)
    line.track(particle)
    assert particle.state[0]<0
    assert physical_state(particle)==before


def test_state_tpsa_cannot_bypass_runtime_effect_guard(runtime_context):
    import xtrack.tpsa as xtpsa
    line=runtime_line(runtime_context)
    # Even bypassing the normal Line NO_SYNRAD precheck cannot silently turn
    # a requested mean map into conservative/Taylor or constant-part transport.
    line.config.XTRACK_MULTIPOLE_NO_SYNRAD=True
    particle=xtpsa.ParticlesTpsa(order=1,p0c=14e6,mass0=xt.ELECTRON_MASS_EV,
        x=2e-4,px=3e-4,y=-1e-4,py=-2e-4,zeta=4e-4,delta=.002)
    before=particle.const_part.copy()
    with pytest.raises((RuntimeError,NotImplementedError)):
        line.track(particle)
    assert (particle.const_part==before).all()
