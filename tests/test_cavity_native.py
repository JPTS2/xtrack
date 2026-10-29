"""Private native full-root/Gauss diagnostic, separate from public Cavity.

The optional reference checks consume the cavity-modelling checkout only in
this test module. Shipped Xtrack source has no dependency on that project.
Point agreement and finite differences do not establish numerical admission,
physical adequacy, an analytic optics capability, or GPU acceptance.
"""
from contextlib import contextmanager
import os
from pathlib import Path
import shlex
import sysconfig

import numpy as np
import pytest
import xobjects as xo
import xtrack as xt


class NativeCavityProbe(xt.BeamElement):
    """Test-owned opt-in consumer of the shared native conservative kernel."""

    _xofields = {
        'length': xo.Float64, 'k': xo.Float64, 'omega': xo.Float64,
        'anchor_s': xo.Float64, 'clock_origin': xo.Float64,
        'source_mass_eV': xo.Float64, 'aperture': xo.Float64,
        'tolerance': xo.Float64, 'maximum_iterations': xo.Int64,
        'cells': xo.Int64, 'mesh': xo.Float64[:], 'profiles': xo.Float64[:],
    }
    isthick = True
    has_backtrack = True
    allow_rot_and_shift = False
    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/cavity_native_adapter.h"']

    def __init__(self, *, mesh, profiles, k, omega, source_mass_eV,
                 anchor_s=0., clock_origin=0., aperture=1., tolerance=2e-13,
                 maximum_iterations=80, **kwargs):
        mesh = np.asarray(mesh, dtype=float)
        profiles = np.asarray(profiles, dtype=float)
        if (mesh.ndim != 1 or len(mesh)<2 or not np.all(np.isfinite(mesh))
                or not np.all(np.diff(mesh)>0)):
            raise ValueError('finite strictly increasing mesh required')
        cells = len(mesh)-1
        if profiles.shape != (cells, 4, 10) or not np.all(np.isfinite(profiles)):
            raise ValueError('finite cell profiles require shape(cells,4,10)')
        for name, value in (('k', k), ('omega', omega),
                            ('source_mass_eV', source_mass_eV),
                            ('aperture', aperture), ('tolerance', tolerance)):
            if not np.isscalar(value) or not np.isfinite(value) or value<=0:
                raise ValueError(name+' must be finite and positive')
        if (isinstance(maximum_iterations, (bool, np.bool_))
                or not isinstance(maximum_iterations, (int, np.integer))
                or maximum_iterations<=0):
            raise ValueError('maximum_iterations must be a positive integer')
        if not np.isfinite(anchor_s) or not np.isfinite(clock_origin):
            raise ValueError('finite physical anchor and clock required')
        if not np.isclose(omega/k, 299792458., rtol=8*np.finfo(float).eps):
            raise ValueError('k and omega must describe the same physical clock')
        super().__init__(length=(mesh[-1]-mesh[0])/k, mesh=mesh,
            profiles=profiles.ravel(), cells=cells, k=k, omega=omega,
            source_mass_eV=source_mass_eV, anchor_s=anchor_s,
            clock_origin=clock_origin, aperture=aperture,
            tolerance=tolerance, maximum_iterations=maximum_iterations,
            **kwargs)


@contextmanager
def native_context(directory):
    """Use diagnosed process-local linker repair and context-local JIT only."""
    config = sysconfig.get_config_vars()
    previous = {name: config.get(name) for name in
                ('LDSHARED', 'LDCXXSHARED', 'BLDSHARED', 'LDFLAGS')}
    for name, value in previous.items():
        if not isinstance(value, str):
            continue
        tokens, clean, seen = shlex.split(value), [], set()
        for token in tokens:
            if token.startswith('-Wl,-rpath,'):
                if token in seen:
                    continue
                seen.add(token)
            clean.append(token)
        if clean != tokens:
            config[name] = shlex.join(clean)
    old_cwd = Path.cwd()
    os.chdir(directory)
    try:
        context = xo.ContextCpu(omp_num_threads=0)
        context.allow_kernel_compilation = True
        context.allow_prebuilt_kernels = False
        yield context
    finally:
        os.chdir(old_cwd)
        config.update(previous)


def prepared_case(cells=1, *, anchor_s=0., time_s=0.):
    """The existing nu=1 hardware/launch; no q0-driven recalibration."""
    setup_owner = pytest.importorskip(
        'verification.maps.phase_resolved.rf_setup')
    provider = pytest.importorskip(
        'verification.maps.phase_resolved.conservative_orbit')
    settings = setup_owner.LocalizedRFSettings(frequency_hz=400e6,
        voltage_v=20e3, length_m=.05, eta=.25, calibration_beta=1.,
        counter_phase_rad=.3, reference_charge_e=1.)
    prepared = settings.prepare(mass_eV=510998.95069, charge_e=-1.)
    fixture = prepared.fixture(2)
    full_mesh = np.linspace(*prepared.setup.interval, 9)
    mesh = full_mesh[:cells+1]
    nodes = np.array([.5-np.sqrt(3)/6, .5+np.sqrt(3)/6])
    profiles = []
    for left, right in zip(mesh[:-1], mesh[1:]):
        planes = [left, *(left+nodes*(right-left)), right]
        values = [provider.profile_values(z, fixture['profile_half_span'],
            fixture['slow_strength'], fixture['fast_amplitude_interval'][1],
            fixture['carrier'], fixture['counter_phase'], 4) for z in planes]
        profiles.append([np.r_[value.real, value.imag] for value in values])
    kappa0 = np.sqrt(2.4**2-1)
    beta0 = kappa0/2.4
    clock = prepared.setup.omega*(time_s+anchor_s/(beta0*299792458.))
    clock += prepared.clock_origin
    # Preserve the physical T=.4 launch under nonzero laboratory anchors.
    zeta = beta0/prepared.setup.k*(clock-.4)
    particle = dict(mass0=510998.95069, p0c=510998.95069*kappa0,
        q0=-1., x=.0004, y=-.0003, px=.005/kappa0, py=-.004/kappa0,
        delta=0., zeta=zeta, s=anchor_s, time_s=time_s)
    element = dict(mesh=mesh, profiles=np.asarray(profiles),
        k=prepared.setup.k, omega=prepared.setup.omega,
        source_mass_eV=510998.95069, anchor_s=anchor_s,
        clock_origin=prepared.clock_origin, aperture=prepared.setup.k*.001)
    canonical = np.array([prepared.setup.k*.0004,
        -prepared.setup.k*.0003, .4, .005, -.004, 2.4])
    return prepared, fixture, element, particle, canonical


PRIMARY_AND_AUX = ('x', 'px', 'y', 'py', 'zeta', 'delta', 's',
                   'pzeta', 'rpp', 'rvv')


def particle_values(particle):
    """Read by particle_id so loss reordering cannot hide rollback failures."""
    order = np.argsort(np.asarray(particle.particle_id))
    return {name: np.asarray(getattr(particle, name))[order].copy()
            for name in PRIMARY_AND_AUX}


def reference_endpoint(fixture, canonical, mesh):
    """Same finite Gauss word with the existing conservative Python owner."""
    from verification.maps.phase_resolved.providers import TotalFieldReference
    from verification.maps.phase_resolved.conservative_orbit import ConservativeOrbit
    from verification.support.gauss import gauss_step
    reference = TotalFieldReference(fixture, (0., 0., 0.),
        gauge_amplitude=0., conservative=True)
    orbit = ConservativeOrbit(reference)
    q = canonical.copy()
    for left, right in zip(mesh[:-1], mesh[1:]):
        q, residual = gauss_step(orbit,
            left, q, right-left, tolerance=2e-13, maximum_iterations=80)
    return q, reference


@pytest.mark.parametrize('cells', [1, 8])
def test_native_cavity_same_word(tmp_path, cells):
    coordinates = pytest.importorskip('validation.xsuite.coordinates')
    BoundaryPotentials = coordinates.BoundaryPotentials
    FixedReferenceCoordinates = coordinates.FixedReferenceCoordinates
    prepared, fixture, data, launch, canonical = prepared_case(cells)
    expected, reference = reference_endpoint(fixture, canonical, data['mesh'])
    kappa0 = np.sqrt(2.4**2-1)
    chart = FixedReferenceCoordinates(k=data['k'], kappa0=kappa0,
        beta0=kappa0/2.4, anchor=data['mesh'][0],
        reference_time=prepared.clock_origin)

    def potentials(z, point):
        rf, magnetic = reference.jets(z, point)
        # Actual signed local potentials; native gauge is explicitly zero.
        local = fixture['charge']*(rf+magnetic)[[0, 1, 3], 0]
        return BoundaryPotentials(np.zeros(2), local, np.zeros((2, 3)),
                                  np.zeros((3, 3)))

    native_expected = chart.to_xtrack(expected, data['mesh'][-1], potentials)
    with native_context(tmp_path) as context:
        element = NativeCavityProbe(_context=context, **data)
        particle = xt.Particles(_context=context, **launch)
        element.track(particle)
        assert int(particle.state[0]) == 1
        observed = np.array([float(getattr(particle, name)[0]) for name in
                            ('x', 'px', 'y', 'py', 'zeta', 'delta')])
        np.testing.assert_allclose(observed, native_expected.state,
                                   rtol=0., atol=3e-13)
        assert particle.s[0] == pytest.approx(data['anchor_s']+element.length)
        rebuilt = xt.Particles(_context=context, **dict(launch,
            x=observed[0], px=observed[1], y=observed[2], py=observed[3],
            zeta=observed[4], delta=observed[5]))
        for name in ('pzeta', 'rpp', 'rvv'):
            np.testing.assert_allclose(getattr(particle, name),
                getattr(rebuilt, name), rtol=0., atol=8*np.finfo(float).eps)


def test_native_cavity_exhaustion_rolls_back(tmp_path):
    _, _, data, launch, _ = prepared_case()
    with native_context(tmp_path) as context:
        element = NativeCavityProbe(_context=context, maximum_iterations=1,
                                    **data)
        particle = xt.Particles(_context=context, **launch)
        before = particle_values(particle)
        element.track(particle)
        assert int(particle.state[0]) == -99702
        for name, value in before.items():
            np.testing.assert_array_equal(particle_values(particle)[name], value)


def test_native_cavity_line_reverse(tmp_path):
    _, _, data, launch, _ = prepared_case(8)
    with native_context(tmp_path) as context:
        element = NativeCavityProbe(_context=context, **data)
        line = xt.Line(elements=[element])
        line.reset_s_at_end_turn = False
        line.build_tracker(_context=context)
        particle = xt.Particles(_context=context, **launch)
        before = particle_values(particle)
        line.track(particle, num_turns=1)
        line.track(particle, num_turns=1, backtrack=True)
        assert int(particle.state[0]) == 1
        for name, value in before.items():
            np.testing.assert_allclose(particle_values(particle)[name], value,
                                       rtol=0., atol=2e-12, err_msg=name)


def test_native_cavity_mutable_controls_refuse_safely(tmp_path):
    _, _, data, launch, _ = prepared_case()
    with native_context(tmp_path) as context:
        for field, value, status in (
                ('cells', 0, -99704), ('cells', 2**63-1, -99704),
                ('k', 0., -99704), ('source_mass_eV', -1., -99704),
                ('maximum_iterations', 0, -99704),
                ('anchor_s', .1, -99704)):
            element = NativeCavityProbe(_context=context, **data)
            setattr(element, field, value)
            particle = xt.Particles(_context=context, **launch)
            before = particle_values(particle)
            element.track(particle)
            assert int(particle.state[0]) == status, field
            for name, original in before.items():
                np.testing.assert_array_equal(particle_values(particle)[name],
                                              original, err_msg=field+':'+name)
        element = NativeCavityProbe(_context=context, **data)
        element.profiles[0] = np.nan
        particle = xt.Particles(_context=context, **launch)
        before = particle_values(particle)
        element.track(particle)
        assert int(particle.state[0]) == -99703
        for name, original in before.items():
            np.testing.assert_array_equal(particle_values(particle)[name],
                                          original)


def test_native_cavity_physical_clock_anchor(tmp_path):
    _, _, baseline, launch, _ = prepared_case()
    _, _, shifted, shifted_launch, _ = prepared_case(
        anchor_s=.125, time_s=3e-10)
    with native_context(tmp_path) as context:
        first = xt.Particles(_context=context, **launch)
        second = xt.Particles(_context=context, **shifted_launch)
        NativeCavityProbe(_context=context, **baseline).track(first)
        NativeCavityProbe(_context=context, **shifted).track(second)
        assert first.state[0] == second.state[0] == 1
        for name in ('x', 'px', 'y', 'py', 'delta', 'pzeta', 'rpp', 'rvv'):
            np.testing.assert_allclose(getattr(first, name),
                getattr(second, name), rtol=0., atol=2e-14, err_msg=name)
        np.testing.assert_allclose(second.zeta-first.zeta,
            shifted_launch['zeta']-launch['zeta'], rtol=0., atol=2e-14)
        np.testing.assert_allclose(second.s-first.s, .125, rtol=0., atol=2e-14)


def drift_case():
    """Standalone exact positive-root limit with no external reference code."""
    kappa0 = np.sqrt(2.4**2-1)
    data = dict(mesh=np.array([-.01, .01]), profiles=np.zeros((1, 4, 10)),
        k=1., omega=299792458., source_mass_eV=510998.95069,
        anchor_s=.125, clock_origin=.4, aperture=.01)
    particle = dict(mass0=510998.95069, p0c=510998.95069*kappa0, q0=-1.,
        x=.0004, y=-.0003, px=.005/kappa0, py=-.004/kappa0,
        delta=.001, zeta=.002, s=.125, time_s=3e-10)
    return data, particle


def test_native_cavity_zero_source_exact_root(tmp_path):
    data, launch = drift_case()
    kappa0 = launch['p0c']/launch['mass0']
    beta0 = kappa0/np.hypot(1., kappa0)
    gamma = np.hypot(1., kappa0*(1.+launch['delta']))
    ux, uy = kappa0*launch['px'], kappa0*launch['py']
    uz = np.sqrt((gamma-1.)*(gamma+1.)-ux*ux-uy*uy)
    length = (data['mesh'][-1]-data['mesh'][0])/data['k']
    expected = dict(launch, x=launch['x']+length*ux/uz,
        y=launch['y']+length*uy/uz,
        zeta=launch['zeta']+length*(1.-beta0*gamma/uz),
        s=launch['s']+length)
    with native_context(tmp_path) as context:
        particle = xt.Particles(_context=context, **launch)
        NativeCavityProbe(_context=context, **data).track(particle)
        assert particle.state[0] == 1
        for name in ('x', 'px', 'y', 'py', 'zeta', 'delta', 's'):
            np.testing.assert_allclose(getattr(particle, name), expected[name],
                rtol=0., atol=3e-15, err_msg=name)
        # Independent native exact drift is another consumer, with its shared
        # positive-root premise identified rather than called field validation.
        native = xt.Particles(_context=context, **launch)
        xt.DriftExact(_context=context, length=length).track(native)
        for name in PRIMARY_AND_AUX:
            np.testing.assert_allclose(getattr(particle, name),
                getattr(native, name), rtol=0., atol=3e-15, err_msg=name)


def test_native_cavity_standalone_domain_and_gauge_refusal(tmp_path):
    data, launch = drift_case()
    with native_context(tmp_path) as context:
        element = NativeCavityProbe(_context=context, **data)
        for changed, status in ((dict(x=.02), -99701),
                                (dict(ax=.001), -99704),
                                (dict(ay=-.001), -99704),
                                (dict(s=.126), -99704)):
            particle = xt.Particles(_context=context, **dict(launch, **changed))
            before = particle_values(particle)
            element.track(particle)
            assert particle.state[0] == status, changed
            for name, original in before.items():
                np.testing.assert_array_equal(particle_values(particle)[name],
                                              original, err_msg=name)
