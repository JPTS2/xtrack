"""How TPSA controls live on elements: storage, knobs, array operations and teardown,
independently of the physics of a map.
"""
import numpy as np
import pytest
import xobjects as xo
import madng_tpsa
import xtrack as xt
import xtrack.tpsa as xtpsa
from xtrack.base_element import FloatOrTpsa
from xtrack.tpsa._knobs import KnobParameters
from _cavity_context import kernel_context


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
    with kernel_context(tmp_path) as context:
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


def test_field_getter_preserves_orders_and_rejects_incompatible_handles(tmp_path):
    """Exercise the real generated FoT getter, including its no-truncation guard."""
    with kernel_context(tmp_path) as context:
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


@pytest.mark.parametrize('programme', [None, 0.])
def test_cavity_tpsa_renamed_programme_storage(programme):
    """Promoting backing slots must preserve a derived/explicit public input."""
    line = xt.Line(elements=[xt.Cavity(nominal_energy_gain=programme)],
                   element_names=['c'])
    line['phase_knob'] = .4
    line['c'].phase = line.ref['phase_knob']
    desc = madng_tpsa.Descriptor(6, 2, num_params=1, param_order=1)
    knobs = KnobParameters(line, ['phase_knob'], desc)
    knobs.apply()
    assert line['c'].nominal_energy_gain_from_rf == (programme is None)
    if programme is None:
        assert line['c'].nominal_energy_gain is None
    else:
        assert line['c'].nominal_energy_gain.const_part == 0.
    with pytest.raises(ValueError, match='one value'):
        knobs.refresh([.8, .9])
    assert line['c'].phase.const_part == .4
    knobs.refresh([.8])
    knobs.teardown()
    if programme is None:
        assert line['c'].nominal_energy_gain is None
    else:
        assert line['c'].nominal_energy_gain == 0.
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
    scope=nullcontext(xo.ContextCpu()) if default_context else kernel_context(tmp_path)
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
    with kernel_context(tmp_path) as context:
        line=xt.Line(elements={'c':xt.Cavity(length=.05,phase=.4,counter_wave_amplitude=.5),
                              'd':xt.Drift(length=.2)})
        line.build_tracker(_context=context)
        before=xt.MultiSetter(line,['c'],'phase',dtype=None)
        before_length=xt.MultiSetter(line,['c','d'],'length',dtype=None)
        line['c'].shift_x=.001
        alignment=xt.MultiSetter(line,['c'],'shift_x',dtype=None)
        np.testing.assert_array_equal(alignment.get_values(),[.001])
        desc=madng_tpsa.Descriptor(6,2,num_params=2,param_order=1)
        line['c'].phase=desc.param(1,.4)
        line['c'].counter_wave_amplitude=desc.param(2,.5)
        after=xt.MultiSetter(line,['c'],'phase',dtype=None)
        after_length=xt.MultiSetter(line,['c','d'],'length',dtype=None)
        np.testing.assert_array_equal(before.get_values(),[.4])
        np.testing.assert_array_equal(after.get_values(),[.4])
        np.testing.assert_array_equal(before_length.get_values(),[.05,.2])
        np.testing.assert_array_equal(after_length.get_values(),[.05,.2])
        before.set_values(np.array([.7]))
        np.testing.assert_array_equal(after.get_values(),[.7])
        np.testing.assert_array_equal(line['c'].phase.param_grad(),[0.,0.])
        np.testing.assert_array_equal(line['c'].counter_wave_amplitude.param_grad(),[0.,1.])
        assert line['c']._tpsa_enabled
        np.testing.assert_array_equal(alignment.get_values(),[.001])
        np.testing.assert_array_equal(line.attr['_tpsa_enabled'],[1.,0.])
        line.disable_tpsa_elements()
        assert not before._needs_tpsa_access()
        after.set_values(np.array([.8]))
        np.testing.assert_array_equal(before.get_values(),[.8])
        assert line['c'].phase==.8


def test_multisetter_resolves_canonical_slice_parent(tmp_path):
    with kernel_context(tmp_path) as context:
        line=xt.Line(elements={'c':xt.Cavity(length=.05,phase=.4,counter_wave_amplitude=.5)})
        line.slice_thick_elements([xt.Strategy(
            slicing=xt.Teapot(2,mode='thick'),element_type=xt.Cavity)])
        line.build_tracker(_context=context)
        slices=[name for name in line.element_names
                if isinstance(line._element_dict[name],xt.ThickSliceCavity)]
        setter=xt.MultiSetter(line,slices,('_parent','phase'),dtype=None)
        desc=madng_tpsa.Descriptor(6,2,num_params=1,param_order=1)
        line['c'].counter_wave_amplitude=desc.param(1,.5)
        np.testing.assert_array_equal(setter.get_values(),[.4,.4])
        setter.set_values(np.array([.7,.7]))
        assert line['c'].phase.const_part==.7
        np.testing.assert_array_equal(line['c'].counter_wave_amplitude.param_grad(),[1.])
        np.testing.assert_array_equal(line.attr['length'],[0.,.025,.025,0.])
        line.disable_tpsa_elements()
        assert not setter._needs_tpsa_access()
        np.testing.assert_array_equal(setter.get_values(),[.7,.7])


def test_indexed_cavity_knob_lifecycle(tmp_path):
    with kernel_context(tmp_path) as context:
        line=xt.Line(elements={'c':xt.MagnetOverlappedCavity(model='_phase-resolved',length=.05,knl=[0.,.2],ksl=[0.,.3])})
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


@pytest.mark.parametrize('active', [False, True])
def test_cavity_array_masked_ufunc_preserves_untouched_storage(active):
    """Masked outputs preserve scalar mirrors and the actual owned series."""
    context = xo.ContextCpu(omp_num_threads=0)
    owner = xt.MagnetOverlappedCavity(model='_phase-resolved',_context=context, order=2,
                     knl=[1.,2.,3.], ksl=[91.,82.,73.])
    mask = np.array([True,False,True])
    if active:
        descriptor = madng_tpsa.Descriptor(6,2,num_params=1,param_order=1)
        owner.knl[:] = [descriptor.param(1,value) for value in (1.,2.,3.)]
        owner.ksl[:] = [descriptor.param(1,value) for value in (91.,82.,73.)]
    normal, skew = owner.knl, owner.ksl
    untouched = (normal[1], skew[1])
    handles = (int(owner._knl_tpsa[1]), int(owner._ksl_tpsa[1]))
    mirrors = (np.asarray(owner._knl).copy(), np.asarray(owner._ksl).copy())
    try:
        assert np.add(normal,1.,out=skew,where=mask) is skew
        np.testing.assert_array_equal(
            [getattr(value,'const_part',value) for value in skew], [2.,82.,4.])
        assert np.multiply(normal,2.,out=normal,where=mask) is normal
        np.testing.assert_array_equal(
            [getattr(value,'const_part',value) for value in normal], [2.,2.,6.])
        for actual, before in zip((owner._knl, owner._ksl), mirrors):
            np.testing.assert_array_equal(np.asarray(actual)[~mask].view(np.uint64),
                                          before[~mask].view(np.uint64))
        assert (int(owner._knl_tpsa[1]), int(owner._ksl_tpsa[1])) == handles
        if active:
            assert normal[1] is untouched[0] and skew[1] is untouched[1]
            np.testing.assert_array_equal([value.param_grad()[0] for value in normal],
                                          [2.,1.,2.])
            np.testing.assert_array_equal([value.param_grad()[0] for value in skew],
                                          [1.,1.,1.])
    finally:
        owner.disable_tpsa()


def test_cavity_array_ufunc_allocated_and_aliased_multiple_outputs():
    context = xo.ContextCpu(omp_num_threads=0)
    owner = xt.MagnetOverlappedCavity(model='_phase-resolved',_context=context, order=2,
                     knl=[1.25,2.5,3.75], ksl=[91.,82.,73.])
    allocated = np.add(owner.knl,1.,out=(None,))
    assert isinstance(allocated,np.ndarray)
    np.testing.assert_array_equal(allocated,[2.25,3.5,4.75])
    mask = np.array([True,False,True])
    fraction, integer = owner.ksl, owner.knl
    result = np.modf(owner.knl,out=(fraction,integer),where=mask)
    assert result[0] is fraction and result[1] is integer
    np.testing.assert_array_equal(np.asarray(fraction),[.25,82.,.75])
    np.testing.assert_array_equal(np.asarray(integer),[1.,2.5,3.])
    owner.knl[:] = [1.25,2.5,3.75]
    result = np.modf(owner.knl,out=(None,fraction),where=mask)
    assert result[1] is fraction
    np.testing.assert_array_equal(result[0][mask],[.25,.75])
    np.testing.assert_array_equal(np.asarray(fraction),[1.,82.,3.])
    # Different view objects can still identify the same physical output.
    first, second = owner.ksl, owner.ksl
    reference = np.asarray(first).copy()
    np.modf(np.asarray(owner.knl),out=(reference,reference),where=mask)
    result = np.modf(owner.knl,out=(first,second),where=mask)
    assert result[0] is first and result[1] is second
    np.testing.assert_array_equal(np.asarray(first),reference)


def test_cavity_array_ufunc_numpy_outputs_keep_casting_and_mask_rules():
    context = xo.ContextCpu(omp_num_threads=0)
    owner = xt.MagnetOverlappedCavity(model='_phase-resolved',_context=context, order=2,knl=[1.25,2.5,3.75])
    integer_target = np.array([91,82,73],dtype=np.int64)
    with pytest.raises(TypeError,match='cast'):
        np.add(owner.knl,.5,out=integer_target)
    np.testing.assert_array_equal(integer_target,[91,82,73])
    np.testing.assert_array_equal(np.asarray(owner.knl),[1.25,2.5,3.75])
    descriptor = madng_tpsa.Descriptor(6,2,num_params=1,param_order=1)
    owner.knl[:] = [descriptor.param(1,value) for value in (1.,2.,3.)]
    target = np.array([descriptor.param(1,value) for value in (91.,82.,73.)],
                      dtype=object)
    untouched = target[1]
    try:
        assert np.add(owner.knl,1.,out=target,where=[True,False,True]) is target
        assert target[1] is untouched
        np.testing.assert_array_equal([value.const_part for value in target],
                                      [2.,82.,4.])
        np.testing.assert_array_equal([value.param_grad()[0] for value in target],
                                      [1.,1.,1.])
    finally:
        owner.disable_tpsa()


@pytest.mark.parametrize('same_owner', [False, True])
def test_cavity_array_ufunc_promotes_scalar_output_without_losing_mask(same_owner):
    context = xo.ContextCpu(omp_num_threads=0)
    source = xt.MagnetOverlappedCavity(model='_phase-resolved',_context=context, order=2,ksl=[91.,82.,73.])
    target = source if same_owner else xt.MagnetOverlappedCavity(model='_phase-resolved',
        _context=context, order=2,ksl=[91.,82.,73.])
    descriptor = madng_tpsa.Descriptor(6,2,num_params=1,param_order=1)
    source.knl[:] = [descriptor.param(1,value) for value in (1.,2.,3.)]
    output = target.ksl
    try:
        assert np.add(source.knl,1.,out=output,where=[True,False,True]) is output
        assert target._tpsa_enabled
        np.testing.assert_array_equal(
            [getattr(value,'const_part',value) for value in output], [2.,82.,4.])
        np.testing.assert_array_equal([
            value.param_grad()[0] if isinstance(value,madng_tpsa.Tpsa) else 0.
            for value in output], [1.,0.,1.])
        assert target._ksl_tpsa[1] == 0 and target._ksl[1] == 82.
    finally:
        source.disable_tpsa()
        if target is not source:
            target.disable_tpsa()


@pytest.mark.parametrize('active', [False, True])
def test_cavity_array_ufunc_rejects_invalid_results_before_target_mutation(active):
    """A late invalid result must not alter earlier coefficients or activation."""
    context = xo.ContextCpu(omp_num_threads=0)
    owner = xt.MagnetOverlappedCavity(model='_phase-resolved',_context=context, order=2,
                     knl=[1.,2.,3.], ksl=[91.,82.,73.])
    descriptor = madng_tpsa.Descriptor(6,2,num_params=1,param_order=1)
    foreign = madng_tpsa.Descriptor(6,2,num_params=2,param_order=1)
    if active:
        owner.ksl[:] = [descriptor.param(1,value) for value in (91.,82.,73.)]
        foreign_rhs = np.array([1.,foreign.param(1,10.),1.],dtype=object)
    else:
        foreign_rhs = np.array([descriptor.param(1,10.),
                                foreign.param(1,20.),1.],dtype=object)
    fields = ('_knl','_ksl','_knl_tpsa','_ksl_tpsa')
    before = {name:np.asarray(getattr(owner,name)).copy() for name in fields}
    untouched = list(owner.ksl)
    try:
        for rhs,error in ((foreign_rhs,ValueError),
                (np.array([1.,1j,1.],dtype=object),TypeError)):
            with pytest.raises(error):
                np.add(owner.knl,rhs,out=owner.ksl)
            assert bool(owner._tpsa_enabled) is active
            for name in fields:
                np.testing.assert_array_equal(
                    np.asarray(getattr(owner,name)).view(np.uint64),
                    before[name].view(np.uint64))
            if active:
                assert all(value is saved for value,saved in zip(owner.ksl,untouched))
    finally:
        owner.disable_tpsa()


def test_cavity_descriptor_change_is_atomic():
    owner=xt.MagnetOverlappedCavity(length=.05)
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

