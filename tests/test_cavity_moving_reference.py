"""Independent physical static-limit check of the full-reference RF2/P2 map.

The target is the exact uniform-solenoid helix and clock. Reference-mesh
refinement checks the declared finite RK recipe, not a uniform error bound.
There is no external map/reference implementation or retained data asset.
"""
import numpy as np
import xobjects as xo
import xtrack as xt


def test_full_static_reference_matches_exact_solenoid_with_refinement(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    context = xo.ContextCpu(omp_num_threads=0)
    policy = (context.allow_prebuilt_kernels, context.allow_kernel_compilation)
    length, ks, mass, energy = .15, 4., xt.ELECTRON_MASS_EV, 2.4
    kappa = np.sqrt(energy*energy-1.)
    frequency = 400e6
    k = 2*np.pi*frequency/299792458.
    initial = dict(x=.001, y=-.0007, px=.02, py=-.01)
    rf = xt.Cavity(model='phase-averaged-moving-reference', length=length,
        frequency=frequency, voltage=0., eta=.25, phase=.4, counter_phase=.3,
        calibration_beta=.9, rf_profile='gaussian', rf_profile_sigma=.04,
        ks=ks, reference_x=initial['x'], reference_y=initial['y'],
        reference_px=initial['px'], reference_py=initial['py'],
        num_reference_steps=2, num_kicks=1, num_factor_steps=1,
        edge_entry_active=0, edge_exit_active=0, absolute_time=True)
    line = xt.Line(elements=[rf, xt.Marker()], element_names=['rf', 'exit'])
    line.particle_ref = xt.Particles(mass0=mass, q0=1., p0c=kappa*mass)
    line.reset_s_at_end_turn = False
    line.use_time_based_zeta_update = True
    line.build_tracker(_context=context)

    # All four reference inputs are nonzero. The incoming particle coincides
    # with that reference, so its moving deviations remain exactly zero.
    bend, span = kappa*ks/k, k*length
    x, y = k*initial['x'], k*initial['y']
    u, v = kappa*initial['px'], kappa*initial['py']
    pix, piy = u+bend*y/2, v-bend*x/2
    pz = np.sqrt(energy*energy-1.-pix*pix-piy*piy)
    angle = bend*span/pz
    pxf = np.cos(angle)*pix+np.sin(angle)*piy
    pyf = -np.sin(angle)*pix+np.cos(angle)*piy
    xf = x+(pix*np.sin(angle)+piy*(1-np.cos(angle)))/bend
    yf = y+(piy*np.sin(angle)-pix*(1-np.cos(angle)))/bend
    expected = np.array([xf/k, (pxf-bend*yf/2)/kappa, yf/k,
        (pyf+bend*xf/2)/kappa, length-kappa*length/pz, 0.])

    errors = []
    for count in (2, 4, 8, 16):
        rf.num_reference_steps = count
        particle = xt.Particles(_context=context, mass0=mass, q0=1., p0c=kappa*mass,
            **initial, zeta=0., delta=0., ax=-ks*initial['y']/2, ay=ks*initial['x']/2)
        line.track(particle, ele_start=0, ele_stop=1)
        assert particle.state[0] == 1
        observed = np.array([getattr(particle, name)[0] for name in
                            ('x', 'px', 'y', 'py', 'zeta', 'delta')])
        errors.append(float(np.max(abs(observed-expected))))
        np.testing.assert_allclose([particle.ax[0], particle.ay[0]],
            [-ks*particle.y[0]/2, ks*particle.x[0]/2], atol=2e-14, rtol=2e-14)
    # Expected fourth-order pointwise convergence, with room for higher terms.
    assert all(after < before/10 for before, after in zip(errors, errors[1:])), errors
    assert errors[-1] < 2e-10, errors
    assert (context.allow_prebuilt_kernels, context.allow_kernel_compilation) == policy
