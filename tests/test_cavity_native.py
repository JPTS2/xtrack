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
import sys

import numpy as np
import pytest
import xobjects as xo
import xtrack as xt
from xtrack.beam_elements.cavity_native import (
    CavityNativeDiagnostic as NativeCavityProbe,
)


@contextmanager
def native_context(directory):
    """Use diagnosed process-local linker repair and context-local JIT only."""
    checkout = Path(__file__).resolve().parents[1]
    assert Path(xt.__file__).resolve() == checkout/'xtrack'/'__init__.py'
    config = sysconfig.get_config_vars()
    previous = {name: config.get(name) for name in
                ('LDSHARED', 'LDCXXSHARED', 'BLDSHARED', 'LDFLAGS')}
    for name, value in previous.items():
        if not isinstance(value, str):
            continue
        tokens, clean, seen = shlex.split(value), [], set()
        for token in tokens:
            if token == '-Wl,-rpath,'+str(Path(sys.prefix)/'lib'):
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
            setattr(element._xobject, field, value)
            particle = xt.Particles(_context=context, **launch)
            before = particle_values(particle)
            element.track(particle)
            assert int(particle.state[0]) == status, field
            for name, original in before.items():
                np.testing.assert_array_equal(particle_values(particle)[name],
                                              original, err_msg=field+':'+name)
        element = NativeCavityProbe(_context=context, **data)
        element._xobject.profiles[0] = np.nan
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


def tangent_output(element, particles, *, reverse=False):
    """Run the observational kernel; buffers use physical particle slots."""
    context = particles._context
    output = context.nparray_to_context_array(
        np.full((particles._capacity, 44), np.nan))
    statuses = context.nparray_to_context_array(
        np.zeros(particles._capacity, dtype=np.int64))
    element.native_tangent(particles, reverse=int(reverse), output=output,
                           statuses=statuses)
    return np.asarray(output).copy(), np.asarray(statuses).copy()


def test_native_cavity_own_finite_root_derivative(tmp_path):
    _, _, data, launch, _ = prepared_case()
    scales = np.array([1/data['k'], 1/np.sqrt(2.4**2-1),
        1/data['k'], 1/np.sqrt(2.4**2-1),
        (np.sqrt(2.4**2-1)/2.4)/data['k'], 1.])
    names = ('x', 'px', 'y', 'py', 'zeta', 'delta')
    with native_context(tmp_path) as context:
        element = NativeCavityProbe(_context=context, **data)
        particle = xt.Particles(_context=context, **launch)
        original = particle_values(particle)
        output, statuses = tangent_output(element, particle)
        assert statuses[0] == 1
        for name, value in original.items():
            np.testing.assert_array_equal(particle_values(particle)[name], value)
        element.track(particle)
        endpoint = np.array([getattr(particle, name)[0] for name in names])
        np.testing.assert_array_equal(output[0, :6], endpoint)
        assert output[0, 42] < element.tolerance
        assert output[0, 43] < element.derivative_tolerance
        matrix = output[0, 6:42].reshape(6, 6)
        normalized = matrix*scales[np.newaxis, :]/scales[:, np.newaxis]
        stencils = []
        for h in (2e-6, 1e-6):
            columns = []
            for column, name in enumerate(names):
                plus = xt.Particles(_context=context, **dict(launch,
                    **{name: launch[name]+h*scales[column]}))
                minus = xt.Particles(_context=context, **dict(launch,
                    **{name: launch[name]-h*scales[column]}))
                element.track(plus)
                element.track(minus)
                assert plus.state[0] == minus.state[0] == 1
                columns.append(np.array([getattr(plus, row)[0]
                    -getattr(minus, row)[0] for row in names])/(2*h*scales))
            stencils.append(np.column_stack(columns))
        extrapolated = (4*stencils[1]-stencils[0])/3
        # A dimensionless finite-difference diagnostic, not a derivative error
        # certificate or differentiation of the floating stopping program.
        np.testing.assert_allclose(normalized, extrapolated,
                                   rtol=2e-7, atol=2e-8)
        backward, backward_status = tangent_output(element, particle,
                                                    reverse=True)
        assert backward_status[0] == 1
        reverse_matrix = backward[0, 6:42].reshape(6, 6)
        product = reverse_matrix@matrix
        product = product*scales[np.newaxis, :]/scales[:, np.newaxis]
        np.testing.assert_allclose(product, np.eye(6), rtol=0., atol=3e-11)


def test_native_cavity_derivative_failure_is_observational(tmp_path):
    data, launch = drift_case()
    with native_context(tmp_path) as context:
        element = NativeCavityProbe(_context=context, **data)
        element.derivative_tolerance = 0.
        particle = xt.Particles(_context=context, **launch)
        before = particle_values(particle)
        output, statuses = tangent_output(element, particle)
        assert statuses[0] == -4
        assert np.all(np.isnan(output))
        assert particle.state[0] == 1
        for name, value in before.items():
            np.testing.assert_array_equal(particle_values(particle)[name], value)


def test_native_cavity_variational_exhaustion_is_observational(tmp_path):
    _, _, data, launch, _ = prepared_case()
    with native_context(tmp_path) as context:
        element = NativeCavityProbe(_context=context,
            derivative_tolerance=1e-300, **data)
        # The nominal stages use the ordinary positive tolerance/80-iteration
        # contract. This separate diagnostic requests an unattainably tight
        # variational residual on a nontrivial interior RF cell.
        particle = xt.Particles(_context=context, **launch)
        before = particle_values(particle)
        output, statuses = tangent_output(element, particle)
        assert statuses[0] == -2
        assert np.all(np.isnan(output))
        assert particle.state[0] == 1
        for name, value in before.items():
            np.testing.assert_array_equal(particle_values(particle)[name], value)
        element.track(particle)
        assert particle.state[0] == 1


def test_native_cavity_actual_species_shared_hardware(tmp_path):
    _, fixture, data, launch, canonical = prepared_case()
    species_owner = pytest.importorskip('validation.species')
    # A heterogeneous batch under the same electron reference; all species
    # have the same normalized launch, not the same absolute momentum.
    species = [species_owner.SPECIES_BY_NAME[name]
               for name in ('e-', 'e+', 'p+', 'Pb208')]
    mass_eV = np.array([item.mass_eV for item in species])
    charge_e = np.array([item.charge_e for item in species])
    mass_ratio = mass_eV/launch['mass0']
    charge_ratio = charge_e/launch['q0']
    expected = []
    for mass, charge in zip(mass_eV, charge_e):
        selected = dict(fixture, charge=charge*launch['mass0']/mass)
        endpoint, reference = reference_endpoint(selected, canonical,
                                                 data['mesh'])
        rf, magnetic = reference.jets(data['mesh'][-1], endpoint[:3])
        potential = selected['charge']*(rf+magnetic)[:, 0]
        gamma = endpoint[5]-potential[3]
        kappa = launch['p0c']/launch['mass0']
        beta = kappa/2.4
        expected.append([endpoint[0]/data['k'],
            (endpoint[3]-potential[0])/kappa,
            endpoint[1]/data['k'], (endpoint[4]-potential[1])/kappa,
            beta/data['k']*(data['clock_origin']
                +(data['mesh'][-1]-data['mesh'][0])/beta-endpoint[2]),
            np.sqrt((gamma-1.)*(gamma+1.))/kappa-1.])
    with native_context(tmp_path) as context:
        element = NativeCavityProbe(_context=context, **data)
        particle = xt.Particles(_context=context, **launch,
            mass_ratio=mass_ratio, charge_ratio=charge_ratio,
            particle_id=np.arange(4))
        element.track(particle)
        assert np.all(particle.state == 1)
        actual = np.column_stack([getattr(particle, name) for name in
                                  ('x', 'px', 'y', 'py', 'zeta', 'delta')])
        np.testing.assert_allclose(actual, expected, rtol=0., atol=3e-13)


def factory_reference(context=None, *, q0=-1., mass0=510998.95069):
    return xt.Particles(_context=context, mass0=mass0,
        p0c=mass0*np.sqrt(2.4**2-1), q0=q0)


def test_cavity_native_factory_same_hardware(tmp_path):
    _, _, reference_data, launch, _ = prepared_case(8)
    with native_context(tmp_path) as context:
        reference = factory_reference(context)
        # Historical preparation used calibration q0=+1,phase=0. With native
        # electron q0=-1 this explicit phase conversion preserves its FIELD.
        cavity = xt.Cavity(length=.05, voltage=20e3, frequency=400e6,
                            phase=-np.pi, absolute_time=True)
        element = cavity._prepare_native_diagnostic(reference, eta=.25,
            counter_phase=.3, calibration_beta=1., num_cells=8,
            radius_m=.001, _context=context)
        for name in ('k', 'omega', 'source_mass_eV', 'clock_origin', 'aperture'):
            assert getattr(element, name) == reference_data[name], name
        np.testing.assert_array_equal(element.mesh, reference_data['mesh'])
        np.testing.assert_array_equal(element.profiles, reference_data['profiles'])
        assert not element.accepted_production_return
        assert not element.preparation['accepted_production_return']
        assert element.preparation['reference_charge_e'] == -1.
        actual = xt.Particles(_context=context, **launch)
        original = actual.copy()
        element.track(actual)
        NativeCavityProbe(_context=context, **reference_data).track(original)
        for name in PRIMARY_AND_AUX:
            np.testing.assert_array_equal(getattr(actual, name),
                                          getattr(original, name))


@pytest.mark.parametrize('q0,voltage,eta,beta_cal', [
    (-1., 20e3, .25, 1.), (1., -20e3, 1., .8),
    (-1., -20e3, 0., .9), (1., 20e3, np.inf, 1.),
])
def test_cavity_native_factory_electric_crest(q0, voltage, eta, beta_cal):
    from scipy.integrate import quad
    phase, counter_phase = .37, .3
    reference = factory_reference(q0=q0)
    cavity = xt.Cavity(length=.05, voltage=voltage, frequency=400e6,
                       phase=phase, absolute_time=True)
    element = cavity._prepare_native_diagnostic(reference, eta=eta,
        counter_phase=counter_phase, calibration_beta=beta_cal,
        radius_m=.001)
    # Independent real-axis quadrature of the PREPARED electric field on
    # the prescribed calibration ray, not an accelerated-particle gain.
    H = element.mesh[-1]
    ratio = np.sqrt(eta)
    weights = (0., 1.) if np.isinf(eta) else (1/(1+ratio), ratio/(1+ratio))
    # Recover the actual prepared on-axis amplitudes without using its
    # Fourier formula: evaluate at Z=0 and solve the two known wave weights.
    # The mesh has a centre boundary at cell4, whose profile is prepared data.
    phasor0 = element.profiles[4, 0, 0]+1j*element.profiles[4, 0, 5]
    denominator = weights[0]+weights[1]*np.exp(-1j*counter_phase)
    amplitude = phasor0/denominator
    assert abs(amplitude.imag) < 2e-15*max(1., abs(amplitude.real))

    def electric(z):
        envelope = (1-(z/H)**2)**16
        phasor = amplitude.real*envelope*(weights[0]*np.exp(1j*z)
            +weights[1]*np.exp(-1j*(z+counter_phase)))
        return float(np.real(phasor*np.exp(-1j*(element.clock_origin
            +(z+H)/beta_cal))))*reference.mass0

    integral, quadrature_error = quad(electric, -H, H, epsabs=1e-9,
                                      epsrel=2e-13)
    expected = np.sign(q0)*voltage*np.sin(phase)
    np.testing.assert_allclose(integral, expected, rtol=2e-12, atol=1e-8)
    assert quadrature_error < 1e-7


def test_cavity_native_factory_snapshot_copy_and_serialization():
    reference = factory_reference()
    cavity = xt.Cavity(length=.05, voltage=20e3, frequency=400e6,
                       phase=-np.pi, absolute_time=True)
    native = cavity._prepare_native_diagnostic(reference, eta=.25,
        counter_phase=.3, radius_m=.001)
    mesh, profiles = native.mesh, native.profiles
    with pytest.raises(ValueError):
        mesh[0] = 0.
    with pytest.raises(ValueError):
        profiles[0, 0, 0] = 0.
    with pytest.raises(AttributeError):
        native.k = 1.
    with pytest.raises(TypeError):
        native.preparation['frequency_hz'] = 1.
    cavity.voltage = 40e3
    newer = cavity._prepare_native_diagnostic(reference, eta=.25,
        counter_phase=.3, radius_m=.001)
    np.testing.assert_array_equal(native.profiles, profiles)
    np.testing.assert_array_equal(newer.profiles, 2*profiles)
    copied = native.copy()
    restored = type(native).from_dict(native.to_dict())
    for candidate in (copied, restored):
        np.testing.assert_array_equal(candidate.mesh, mesh)
        np.testing.assert_array_equal(candidate.profiles, profiles)
        assert dict(candidate.preparation) == dict(native.preparation)
        with pytest.raises(TypeError):
            candidate.preparation['voltage_v'] = 1.


@pytest.mark.parametrize('changes', [
    dict(absolute_time=False), dict(length=0.), dict(frequency=0.),
    dict(frequency=0., harmonic=4.), dict(phase_taper=.1),
    dict(lag_taper=.1), dict(shift_x=.001),
    dict(model='rot-kick-rot-low-order'), dict(integrator='uniform'),
])
def test_cavity_native_factory_refuses_unsupported_settings(changes):
    settings = dict(length=.05, voltage=20e3, frequency=400e6,
                     phase=-np.pi, absolute_time=True)
    settings.update(changes)
    cavity = xt.Cavity(**settings)
    with pytest.raises(ValueError):
        cavity._prepare_native_diagnostic(factory_reference(), eta=.25,
                                          radius_m=.001)


def test_cavity_native_factory_legacy_compiled_isolation(tmp_path):
    with native_context(tmp_path) as context:
        with pytest.warns(FutureWarning, match='lag'):
            cavity = xt.Cavity(_context=context, length=.05, voltage=20e3,
                frequency=400e6, phase=.37, lag=17., absolute_time=True)
        original = cavity.copy()
        offset, size = cavity._xobject._offset, cavity._xobject._size
        before_bytes = cavity._buffer.buffer[offset:offset+size].copy()
        reference = factory_reference(context)
        native = cavity._prepare_native_diagnostic(reference, eta=.25,
            counter_phase=.3, radius_m=.001, _context=context)
        np.testing.assert_array_equal(
            cavity._buffer.buffer[offset:offset+size], before_bytes)
        without_lag = xt.Cavity(_context=context, length=.05, voltage=20e3,
            frequency=400e6, phase=.37+np.pi/180*17., absolute_time=True)
        combined = without_lag._prepare_native_diagnostic(reference, eta=.25,
            counter_phase=.3, radius_m=.001, _context=context)
        assert native.clock_origin == combined.clock_origin
        np.testing.assert_array_equal(native.profiles, combined.profiles)
        source_path = tmp_path/'legacy-cavity.c'
        cavity.compile_kernels(save_source_as=str(source_path))
        source = source_path.read_text()
        assert 'cavity_native_' not in source
        assert 'CavityNative' not in source
        values = dict(mass0=510998.95069,
            p0c=510998.95069*np.sqrt(2.4**2-1), q0=-1.,
            x=.0004, px=.001, y=-.0003, py=-.002, delta=.001,
            zeta=.01, s=0., time_s=3.1e-10)
        actual = xt.Particles(_context=context, **values)
        expected = xt.Particles(_context=context, **values)
        cavity.track(actual)
        original.track(expected)
        assert actual.state[0] == expected.state[0] == 1
        for name in PRIMARY_AND_AUX:
            np.testing.assert_array_equal(getattr(actual, name),
                                          getattr(expected, name))


def test_cavity_native_explicit_class_line_roundtrip():
    reference = factory_reference()
    cavity = xt.Cavity(length=.05, voltage=20e3, frequency=400e6,
                       phase=-np.pi, absolute_time=True)
    native = cavity._prepare_native_diagnostic(reference, eta=.25,
        counter_phase=.3, radius_m=.001)
    line = xt.Line(elements=[native], element_names=['rf'], particle_ref=reference)
    # This is explicit custom-class lookup, not an assertion of automatic
    # public registration or fresh-process default Line deserialization.
    restored = xt.Line.from_dict(line.to_dict(), classes=(type(native),),
                                verbose=False, with_progress=False)
    np.testing.assert_array_equal(restored.element_dict['rf'].mesh, native.mesh)
    np.testing.assert_array_equal(restored.element_dict['rf'].profiles,
                                  native.profiles)
    assert dict(restored.element_dict['rf'].preparation) == dict(native.preparation)


def secondary_case(context):
    reference = factory_reference(context)
    cavity = xt.Cavity(_context=context, length=.05, voltage=20e3,
        frequency=400e6, phase=-np.pi, absolute_time=True)
    element = cavity._prepare_native_diagnostic(reference, eta=.25,
        counter_phase=.3, radius_m=.001, _context=context)
    kappa = reference.p0c[0]/reference.mass0
    beta0 = reference.beta0[0]
    position = 0.
    physical_s = element.anchor_s+(position-element.mesh[0])/element.k
    zeta = beta0/element.k*(element.clock_origin
        +(position-element.mesh[0])/beta0-.4)
    particle = xt.Particles(_context=context, mass0=reference.mass0,
        p0c=reference.p0c[0], q0=-1., x=.0004, y=-.0003,
        px=.005/kappa, py=-.004/kappa, delta=0., zeta=zeta, s=physical_s,
        anomalous_magnetic_moment=.00115965218059,
        spin_x=.2, spin_y=-.1, spin_z=.9)
    return element,particle,position


def raw_secondary_output(element, particle, position, profile, rho_unit):
    context = particle._context
    output = context.nparray_to_context_array(
        np.full((particle._capacity,25),np.nan))
    statuses = context.nparray_to_context_array(
        np.zeros(particle._capacity,dtype=np.int64))
    element.native_secondary(particle,position=position,
        profile=context.nparray_to_context_array(profile),
        rho_unit_eV=rho_unit,output=output,statuses=statuses)
    return np.asarray(output).copy(),np.asarray(statuses).copy()


def test_native_cavity_secondary_factory_same_event(tmp_path):
    with native_context(tmp_path) as context:
        element,particle,position = secondary_case(context)
        profile4 = element.profile_at(position)
        profile5 = element.profile_at(position,maximum=5)
        np.testing.assert_array_equal(profile4[:5],profile5[:5])
        np.testing.assert_array_equal(profile4[5:],profile5[6:11])
        np.testing.assert_array_equal(profile4,element.profiles[4,0])
        assert not profile5.flags.writeable
        before = particle_values(particle)
        extra = {name: np.asarray(getattr(particle,name)).copy() for name in
                 ('spin_x','spin_y','spin_z','ax','ay')}
        result = element.observe_secondary(particle,position)
        assert result['status'][0] == 1
        assert result['particle_id'][0] == particle.particle_id[0]
        assert np.all(np.isfinite(result['values']))
        np.testing.assert_array_equal(result['values'][0,:3],np.zeros(3))
        assert np.linalg.norm(result['values'][0,12:18])>0
        assert np.linalg.norm(result['values'][0,:6])>0
        assert result['values'][0,20]>0
        for name,value in before.items():
            np.testing.assert_array_equal(particle_values(particle)[name],value)
        for name,value in extra.items():
            np.testing.assert_array_equal(getattr(particle,name),value)
        zero,status = raw_secondary_output(element,particle,position,
            profile5,0.)
        assert status[0] == 1
        np.testing.assert_array_equal(zero[0,:6],np.zeros(6))
        np.testing.assert_array_equal(zero[0,6:9],result['values'][0,6:9])
        np.testing.assert_array_equal(zero[0,9:12],np.zeros(3))


def test_native_cavity_secondary_refusal_preserves_particle(tmp_path):
    with native_context(tmp_path) as context:
        element,original,position = secondary_case(context)
        profile = element.profile_at(position,maximum=5)
        rho_unit = element.observe_secondary(original,position)['rho_unit_eV']
        for fault,status in (('plane',-4),('profile5',-3),('rho',-4),
                             ('radius',-1),('mass0',-3)):
            particle = original.copy()
            data = profile.copy()
            query,rho = position,rho_unit
            if fault=='plane':
                query += .001
            elif fault=='profile5':
                data[5] = np.nan
            elif fault=='rho':
                rho = -1.
            elif fault=='radius':
                particle.x = .002
            elif fault=='mass0':
                # Deliberate private backend fault: ratio remains positive,
                # but a negative reference mass is physically inadmissible.
                particle._xobject.mass0 = -particle.mass0
                particle._xobject.p0c[0] = -particle.p0c[0]
            before = particle_values(particle)
            values,statuses = raw_secondary_output(element,particle,query,data,rho)
            assert statuses[0] == status, fault
            assert np.all(np.isnan(values)), fault
            assert particle.state[0] == 1
            for name,value in before.items():
                np.testing.assert_array_equal(particle_values(particle)[name],
                                              value,err_msg=fault+':'+name)


SPIN_AND_RNG = ('spin_x','spin_y','spin_z',
                '_rng_s1','_rng_s2','_rng_s3','_rng_s4')


def spin_rng_values(particle):
    order = np.argsort(np.asarray(particle.particle_id))
    values = particle_values(particle)
    values.update({name: np.asarray(getattr(particle,name))[order].copy()
                   for name in (*SPIN_AND_RNG,'anomalous_magnetic_moment','ax','ay')})
    return values


def set_spin_rng(particle):
    particle.spin_x = 2.
    particle.spin_y = -1.
    particle.spin_z = .5
    particle.anomalous_magnetic_moment = .00115965218059
    for j,name in enumerate(SPIN_AND_RNG[3:]):
        getattr(particle,name)[:] = np.arange(particle._capacity,dtype=np.uint32)+23+j


@pytest.mark.parametrize('cells',[1,8])
def test_native_cavity_passive_spin_actual_line(tmp_path,cells):
    data,launch = native_p16_case(cells)
    with native_context(tmp_path) as context:
        source = NativeCavityProbe(_context=context,**data)
        spin_source = NativeCavityProbe(_context=context,**data).enable_passive_spin()
        particle = xt.Particles(_context=context,**launch)
        set_spin_rng(particle)
        initial = particle.copy()
        before = spin_rng_values(particle)
        standalone = particle.copy()
        source.track(standalone)
        conservative = particle.copy()
        orbit_line = xt.Line(elements=[source])
        orbit_line.reset_s_at_end_turn = False
        orbit_line.build_tracker(_context=context)
        orbit_line.track(conservative,num_turns=1)
        line = xt.Line(elements=[spin_source])
        line.reset_s_at_end_turn = False
        line.build_tracker(_context=context)
        line.track(particle,num_turns=1)
        assert particle.state[0] == 1
        assert particle.time_s == conservative.time_s
        # End-turn actions advance time_s and rebase zeta even when s is
        # unwrapped. Compare identical Line frames, and separately verify the
        # physical event against standalone tracking in its original frame.
        physical_time = particle.time_s+(particle.s[0]-particle.zeta[0])/(\
            particle.beta0[0]*299792458.)
        standalone_time = standalone.time_s+(standalone.s[0]-standalone.zeta[0])/(\
            standalone.beta0[0]*299792458.)
        np.testing.assert_allclose(physical_time,standalone_time,
            rtol=32*np.finfo(float).eps,atol=0.)
        # Passive spin cannot change the nominal conservative word/energy.
        for name in PRIMARY_AND_AUX:
            np.testing.assert_array_equal(getattr(particle,name),
                                          getattr(conservative,name))
        for name in SPIN_AND_RNG[:3]:
            np.testing.assert_array_equal(getattr(conservative,name),
                                          getattr(initial,name))
        spin_before = np.array([getattr(initial,name)[0] for name in SPIN_AND_RNG[:3]])
        spin_after = np.array([getattr(particle,name)[0] for name in SPIN_AND_RNG[:3]])
        assert np.max(abs(spin_after-spin_before))>1e-14
        # These are finite residual/tableau diagnostics, not an admitted spin
        # accuracy or an exact norm theorem for represented arithmetic.
        scale = np.linalg.norm(spin_before)
        np.testing.assert_allclose(np.linalg.norm(spin_after),scale,
            rtol=0.,atol=32*np.finfo(float).eps*scale)
        for name in SPIN_AND_RNG[3:]:
            np.testing.assert_array_equal(spin_rng_values(particle)[name],before[name])
        line.track(particle,num_turns=1,backtrack=True)
        assert particle.state[0] == 1
        after = spin_rng_values(particle)
        for name in PRIMARY_AND_AUX:
            np.testing.assert_allclose(after[name],before[name],rtol=0.,atol=2e-12,
                                       err_msg=name)
        for name in SPIN_AND_RNG[:3]:
            np.testing.assert_allclose(after[name],before[name],rtol=0.,
                atol=5e-13*scale,err_msg=name)
        for name in SPIN_AND_RNG[3:]:
            np.testing.assert_array_equal(after[name],before[name])


def test_native_cavity_passive_spin_transactional_failure(tmp_path):
    data,launch = native_p16_case(8)
    with native_context(tmp_path) as context:
        for fault,expected in (('spin_iterations',-99702),
                ('spin_nonfinite',-99703),('anomaly',-99704),
                ('mode',-99704),('tolerance',-99704),('radius',-99701)):
            element = NativeCavityProbe(_context=context,**data).enable_passive_spin()
            particle = xt.Particles(_context=context,**launch)
            set_spin_rng(particle)
            if fault=='spin_iterations':
                element.spin_tolerance = 1e-300
                assert element.maximum_iterations == 80
                # The same valid positive nominal solve still succeeds; only
                # the independently requested spin matrix residual exhausts.
                baseline = particle.copy()
                NativeCavityProbe(_context=context,**data).track(baseline)
                assert baseline.state[0] == 1
            elif fault=='spin_nonfinite':
                particle.spin_y = np.nan
            elif fault=='anomaly':
                particle.anomalous_magnetic_moment = np.nan
            elif fault=='mode':
                element.spin_tracking = 2
            elif fault=='tolerance':
                element.spin_tolerance = 0.
            elif fault=='radius':
                particle.x = .002
            before = spin_rng_values(particle)
            element.track(particle)
            assert particle.state[0] == expected,fault
            after = spin_rng_values(particle)
            for name,value in before.items():
                np.testing.assert_array_equal(after[name],value,err_msg=fault+':'+name)


def test_native_cavity_passive_spin_mixed_loss_by_id(tmp_path):
    data,launch = native_p16_case()
    with native_context(tmp_path) as context:
        element = NativeCavityProbe(_context=context,**data).enable_passive_spin()
        particle = xt.Particles(_context=context,particle_id=[17,3],**launch)
        set_spin_rng(particle)
        particle.spin_x[0] = np.nan
        before = spin_rng_values(particle)
        element.track(particle)
        order = np.argsort(np.asarray(particle.particle_id))
        np.testing.assert_array_equal(np.asarray(particle.state)[order],[1,-99703])
        after = spin_rng_values(particle)
        for name,value in before.items():
            np.testing.assert_array_equal(after[name][1],value[1],err_msg=name)
        for name in SPIN_AND_RNG[3:]:
            np.testing.assert_array_equal(after[name],before[name])


def native_p16_case(cells=1):
    """The original hardware/launch using only the shipped private factory."""
    reference = factory_reference()
    cavity = xt.Cavity(length=.05,voltage=20e3,frequency=400e6,
                       phase=-np.pi,absolute_time=True)
    native = cavity._prepare_native_diagnostic(reference,eta=.25,
        counter_phase=.3,radius_m=.001,num_cells=8)
    data = dict(mesh=native.mesh[:cells+1],profiles=native.profiles[:cells],
        k=native.k,omega=native.omega,source_mass_eV=native.source_mass_eV,
        anchor_s=native.anchor_s,clock_origin=native.clock_origin,
        aperture=native.aperture)
    kappa = np.sqrt(2.4**2-1)
    beta = kappa/2.4
    launch = dict(mass0=510998.95069,p0c=510998.95069*kappa,q0=-1.,
        x=.0004,y=-.0003,px=.005/kappa,py=-.004/kappa,delta=0.,s=0.,time_s=0.,
        zeta=beta/native.k*(native.clock_origin-.4))
    return data,launch


def test_native_cavity_first_cell_audit_actual_candidate(tmp_path):
    data,launch = drift_case()
    with native_context(tmp_path) as context:
        element = NativeCavityProbe(_context=context,**data)
        particle = xt.Particles(_context=context,**launch)
        set_spin_rng(particle)
        before = spin_rng_values(particle)
        audit = element.audit_first_cell(particle)
        assert audit['status'][0] == 1
        assert audit['cell_index'][0] == 0
        assert audit['particle_id'][0] == particle.particle_id[0]
        assert audit['start'][0] == data['mesh'][0]
        assert audit['step'][0] == data['mesh'][1]-data['mesh'][0]
        assert audit['effective_Q'][0] == -1.
        assert audit['residual'][0] < element.tolerance
        for name,value in before.items():
            np.testing.assert_array_equal(spin_rng_values(particle)[name],value)
        q = audit['qin'][0]
        assert abs(q[2]-.4)>.01  # Actual native clock, not the ideal legacy launch.
        assert abs(q[5]-2.4)>.001
        uz = np.sqrt((q[5]-1)*(q[5]+1)-q[3]**2-q[4]**2)
        flow = np.r_[q[3:5]/uz,q[5]/uz,np.zeros(3)]
        h = audit['step'][0]
        np.testing.assert_allclose(audit['qout'][0],q+h*flow,rtol=0.,atol=3e-15)
        np.testing.assert_allclose(audit['stages'][0],
            q+audit['tableau']['C'][:,None]*h*flow,rtol=0.,atol=3e-15)
        tracked = particle.copy()
        element.track(tracked)
        out = audit['qout'][0]
        kappa = launch['p0c']/launch['mass0']
        beta = particle.beta0[0]
        clock = data['omega']*(launch['time_s']
            +data['anchor_s']/(beta*299792458.))+data['clock_origin']
        expected_native = np.array([out[0]/data['k'],out[3]/kappa,
            out[1]/data['k'],out[4]/kappa,
            beta/data['k']*(clock+h/beta-out[2]),
            np.sqrt((out[5]-1)*(out[5]+1))/kappa-1])
        actual = np.array([getattr(tracked,name)[0] for name in
                           ('x','px','y','py','zeta','delta')])
        np.testing.assert_allclose(actual,expected_native,rtol=0.,atol=3e-15)
        snapshot = audit['source_snapshot']
        np.testing.assert_array_equal(snapshot['mesh'],data['mesh'])
        np.testing.assert_array_equal(snapshot['profiles'],data['profiles'])
        np.testing.assert_array_equal(snapshot['cached_row_ids'],np.arange(4))
        np.testing.assert_array_equal(snapshot['cached_plane_positions'],
            np.r_[data['mesh'][0],data['mesh'][0]+audit['tableau']['C']*h,
                  data['mesh'][1]])
        assert snapshot['cached_plane_position_convention'] == (
            'forward_numpy_float64_left_plus_stored_C_times_h_replay')
        assert not snapshot['profiles'].flags.writeable
        assert not element.accepted_production_return


def test_native_cavity_first_cell_audit_failure_and_ids(tmp_path):
    data,launch = native_p16_case()
    with native_context(tmp_path) as context:
        element = NativeCavityProbe(_context=context,**data,maximum_iterations=1)
        particle = xt.Particles(_context=context,**launch)
        set_spin_rng(particle)
        before = spin_rng_values(particle)
        audit = element.audit_first_cell(particle)
        assert audit['status'][0] == -2
        assert np.all(np.isnan(audit['qin']))
        assert np.all(np.isnan(audit['stages']))
        assert np.all(np.isnan(audit['qout']))
        assert np.all(np.isnan(audit['start']))
        assert particle.state[0] == 1
        for name,value in before.items():
            np.testing.assert_array_equal(spin_rng_values(particle)[name],value)
        element.maximum_iterations = 80
        batch = xt.Particles(_context=context,particle_id=[17,3],**launch)
        set_spin_rng(batch)
        batch.state[0] = -7
        before = spin_rng_values(batch)
        audit = element.audit_first_cell(batch)
        by_id = {int(identity):index for index,identity in enumerate(audit['particle_id'])}
        assert audit['status'][by_id[3]] == 1
        assert audit['status'][by_id[17]] == 0
        assert np.all(np.isnan(audit['qout'][by_id[17]]))
        for name,value in before.items():
            np.testing.assert_array_equal(spin_rng_values(batch)[name],value)


def test_native_cavity_first_cell_audit_fresh_snapshot(tmp_path):
    data,launch = native_p16_case()
    with native_context(tmp_path) as context:
        element = NativeCavityProbe(_context=context,**data)
        particle = xt.Particles(_context=context,**launch)
        first = element.audit_first_cell(particle)
        assert first['status'][0] == 1
        old_input = first['qin'].copy()
        old_profiles = first['source_snapshot']['profiles'].copy()
        particle.x += 1e-5
        second = element.audit_first_cell(particle)
        assert second['status'][0] == 1
        assert not np.array_equal(first['qin'],second['qin'])
        np.testing.assert_array_equal(first['qin'],old_input)
        changed = NativeCavityProbe(_context=context,
            **dict(data,profiles=2*data['profiles']))
        third = changed.audit_first_cell(particle)
        assert third['status'][0] == 1
        np.testing.assert_array_equal(third['source_snapshot']['profiles'],2*old_profiles)
        np.testing.assert_array_equal(first['source_snapshot']['profiles'],old_profiles)
        assert not np.array_equal(third['qout'],second['qout'])
