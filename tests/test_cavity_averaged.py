"""Native canonical averaged maps retain their own physical product.

Ordinary workflows use Xtrack alone. The optional source-reference test
consumes cavity-modelling only as an independent development comparator.
"""
import numpy as np
import pytest
import xtrack as xt

from test_cavity_native import native_context


HARDWARE=dict(length=.05,frequency=400e6,voltage=20e3,phase=.4,
    eta=.25,counter_phase=.3,calibration_beta=1.,num_kicks=8,
    num_factor_steps=1,model='phase-averaged-second-order',integrator='gauss2')
NAMES=('x','px','y','py','zeta','delta')


def reference_particle(context):
    return xt.Particles(_context=context,mass0=510998.95069,q0=1.,
                        p0c=510998.95069*np.sqrt(2.4**2-1.))


def launch(context):
    p=reference_particle(context)
    kappa=np.sqrt(2.4**2-1.)
    p.x=.0004;p.y=-.0003;p.px=.005/kappa;p.py=-.004/kappa
    return p


def values(p):
    return np.array([getattr(p,name)[0] for name in NAMES])


def test_second_order_canonical_native_python_product(tmp_path):
    setup=pytest.importorskip('verification.maps.phase_resolved.rf_setup')
    owner=pytest.importorskip('verification.maps.averaged.finite_map')
    with native_context(tmp_path) as context:
        element=xt.Cavity(_context=context,**HARDWARE)
        p=launch(context)
        prepared=setup.LocalizedRFSettings(frequency_hz=400e6,
            voltage_v=20e3,length_m=.05,phase_rad=.4,eta=.25,
            counter_phase_rad=.3,calibration_beta=1.).prepare(
                mass_eV=510998.95069,charge_e=1.)
        model=owner.CanonicalF4(**prepared.canonical_arguments(2),
            characteristic_steps=8,flow_steps=1,tolerance=2e-13)
        k=prepared.setup.k;kappa=np.sqrt(2.4**2-1.)
        incoming=np.array([k*.0004,-k*.0003,
            prepared.entrance_clock,.005,-.004,2.4])
        q=model.map(incoming)
        expected=np.array([q[0]/k,q[3]/kappa,q[1]/k,q[4]/kappa,
            p.beta0[0]/k*(prepared.entrance_clock
                +2*prepared.setup.interval[1]/p.beta0[0]-q[2]),
            np.sqrt((q[5]-1)*(q[5]+1))/kappa-1.])
        element.track(p)
        assert p.state[0]==1
        np.testing.assert_allclose(values(p),expected,rtol=2e-10,atol=2e-12)


def test_second_order_canonical_line_twiss_reverse_and_partition(tmp_path):
    with native_context(tmp_path) as context:
        element=xt.Cavity(_context=context,**HARDWARE)
        line=xt.Line(elements=[element,xt.Marker(_context=context)],
                     element_names=['cavity','end'])
        line.particle_ref=reference_particle(context)
        p=launch(context);initial=values(p)
        line.build_tracker(_context=context)
        line.track(p,ele_start=0,ele_stop=1)
        assert p.state[0]==1
        endpoint=values(p)
        line.track(p,ele_start=0,ele_stop=1,backtrack=True)
        assert p.state[0]==1
        np.testing.assert_allclose(values(p),initial,rtol=0.,atol=3e-11)
        tw=line.twiss(method='6d',start=xt.START,end=xt.END,
            betx=2.,alfx=.1,bety=3.,alfy=-.2)
        assert np.all(np.isfinite(tw.betx)) and np.all(tw.betx>0.)
        assert np.all(np.isfinite(tw.bety)) and np.all(tw.bety>0.)
        for mode in ('thick','thin'):
            split=xt.Line(elements={'cavity':element.copy(_context=context),
                'end':xt.Marker(_context=context)},element_names=['cavity','end'])
            split.particle_ref=reference_particle(context)
            split.slice_thick_elements([xt.Strategy(xt.Teapot(2,mode=mode))])
            split.build_tracker(_context=context)
            q=launch(context)
            stop=split.element_names.index('end')
            split.track(q,ele_start=0,ele_stop=stop)
            assert q.state[0]==1,(mode,q.state)
            np.testing.assert_allclose(values(q),endpoint,rtol=0.,atol=3e-11,
                err_msg=mode+' parent-anchored averaged prefix composition')


def test_second_order_canonical_live_variables(tmp_path):
    with native_context(tmp_path) as context:
        line=xt.Line(elements=[xt.Cavity(_context=context,**HARDWARE),
                              xt.Marker(_context=context)],
                     element_names=['cavity','end'])
        line.particle_ref=reference_particle(context)
        line['mix']=.25;line['rf_phase']=.4
        line.element_refs['cavity'].eta=line.ref['mix']
        line.element_refs['cavity'].phase=line.ref['rf_phase']
        line.build_tracker(_context=context)
        before=launch(context);line.track(before,ele_start=0,ele_stop=1)
        line.set('mix',.16);line.set('rf_phase',.53)
        after=launch(context);line.track(after,ele_start=0,ele_stop=1)
        fresh=launch(context)
        xt.Cavity(_context=context,**dict(HARDWARE,eta=.16,phase=.53)).track(fresh)
        assert before.state[0]==after.state[0]==fresh.state[0]==1
        assert np.max(np.abs(values(after)-values(before)))>1e-7
        np.testing.assert_allclose(values(after),values(fresh),rtol=0.,atol=2e-12)


def test_second_order_source_mesh_refinement_and_signed_inverse(tmp_path):
    """Own-map numerical refinement, expressed in dimensionless source units.

    This is a selected convergence case, not an RF-truncation error bound.
    Source and generating-factor meshes refine together; no resolved-model
    or frozen-model endpoint is substituted for the canonical C2 product.
    """
    with native_context(tmp_path) as context:
        controls=dict(length=.16,frequency=400e6,voltage=2e3,phase=.4,
            eta=.25,counter_phase=.3,calibration_beta=.9,
            model='phase-averaged-second-order',integrator='gauss2')
        element=xt.Cavity(_context=context,**controls)
        line=xt.Line(elements=[element,xt.Marker(_context=context)],
            element_names=['cavity','end'])
        line.particle_ref=reference_particle(context)
        line.build_tracker(_context=context)
        scale=2*np.pi*controls['frequency']/299792458.
        kappa=float(line.particle_ref.p0c[0]/line.particle_ref.mass0)
        beta=float(line.particle_ref.beta0[0])

        def normalized(p):
            momentum=kappa*(1.+p.delta[0])
            # The omitted common clock/plane offset cancels at each shared
            # observation plane; momentum and energy are in actual mc units.
            return np.array([scale*p.x[0],scale*p.y[0],
                -scale*p.zeta[0]/beta,kappa*p.px[0],kappa*p.py[0],
                np.sqrt(1.+momentum*momentum)])

        initial=launch(context)
        before=normalized(initial)
        endpoints=[]
        for cells in (8,16,32,64):
            element.num_kicks=cells
            element.num_factor_steps=cells//4
            p=initial.copy()
            line.track(p,ele_start=0,ele_stop=1)
            assert p.state[0]==1,cells
            endpoints.append(normalized(p))
            line.track(p,ele_start=0,ele_stop=1,backtrack=True)
            assert p.state[0]==1,cells
            np.testing.assert_allclose(normalized(p),before,atol=3e-12,rtol=0.,
                err_msg='C2 same-word signed inverse at '+str(cells)+' cells')
        errors=[np.max(abs(endpoint-endpoints[-1])) for endpoint in endpoints[:-1]]
        assert all(after<before/8 for before,after in zip(errors,errors[1:])),errors
