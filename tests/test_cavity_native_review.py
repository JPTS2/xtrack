"""Independent point-law checks of the optional native LL/BMT observer.

This standalone test uses an analytic complex longitudinal profile, not P16
or a qualified cavity. Field probes share the compiled field algebra; their
material derivatives and the vector LL/covariant spin laws are independent
of the generated LL/BMT expressions. No cavity-modelling import is needed.
Point agreement establishes neither transport, noise, nor numerical admission.
"""
import numpy as np
import xobjects as xo
import xtrack as xt

from test_cavity_native import native_context, particle_values
from xtrack.beam_elements.cavity_native import CavityNativeDiagnostic


MASS0 = 510998.95069
KAPPA0 = np.sqrt(2.4**2 - 1)
BETA0 = KAPPA0 / 2.4
C_LIGHT = 299792458.


def profile_at(z, maximum=5):
    """An explicit unwindowed local M2 source, with analytic shifted jets."""
    value = 0.07 * np.exp(1j * (0.7*z + 0.3))
    jets = value * (0.7j)**np.arange(maximum + 1)
    return np.r_[jets.real, jets.imag]


def observer_element(context):
    """Supply finite raw data; no P16 provenance or admission is implied."""
    mesh = np.array((-.2, .2))
    nodes = np.array((.5 - np.sqrt(3)/6, .5 + np.sqrt(3)/6))
    planes = np.r_[mesh[0], mesh[0] + np.diff(mesh)[0]*nodes, mesh[1]]
    omega = 2*np.pi*400e6
    return CavityNativeDiagnostic(_context=context, mesh=mesh,
        profiles=np.array([[profile_at(z, 4) for z in planes]]),
        k=omega/C_LIGHT, omega=omega, source_mass_eV=MASS0,
        anchor_s=.25, clock_origin=.37, aperture=.1)


def event_particles(context, element, z, coordinates, *, masses=MASS0,
                    charges=-1., momentum=(.2, -.1, 2.), ids=None, states=1,
                    anomaly=.001):
    """Encode the same physical events in the zero-native-A chart.

    Z shifts change s and zeta together at fixed T. beta0 is the frozen
    coordinate reference; it is never the actual particle beta or calibration.
    """
    points = np.atleast_2d(np.asarray(coordinates, dtype=float))
    count = len(points)
    u = np.broadcast_to(momentum, (count, 3))
    p = np.linalg.norm(u, axis=1)
    clock = element.omega*element.anchor_s/(BETA0*C_LIGHT)
    clock += element.clock_origin
    zeta = BETA0/element.k*(clock + (z-element.mesh[0])/BETA0 - points[:, 2])
    return xt.Particles(_context=context, mass0=MASS0, p0c=MASS0*KAPPA0,
        q0=-1., mass_ratio=np.broadcast_to(masses, count)/MASS0,
        charge_ratio=-np.broadcast_to(charges, count),
        x=points[:, 0]/element.k, y=points[:, 1]/element.k,
        px=u[:, 0]/KAPPA0, py=u[:, 1]/KAPPA0, delta=p/KAPPA0-1,
        zeta=zeta, s=element.anchor_s+(z-element.mesh[0])/element.k,
        anomalous_magnetic_moment=anomaly,
        particle_id=np.arange(count) if ids is None else ids, state=states)


def observe(element, particles, z, profile, rho_unit):
    """Use raw query jets; return physical particle IDs after any reordering."""
    context = particles._context
    output = context.nparray_to_context_array(
        np.full((particles._capacity, 25), np.nan))
    statuses = context.nparray_to_context_array(
        np.zeros(particles._capacity, dtype=np.int64))
    jets = context.nparray_to_context_array(np.asarray(profile, dtype=float))
    element.native_secondary(particles, position=z, profile=jets,
        rho_unit_eV=rho_unit, output=output, statuses=statuses)
    return (np.asarray(output).copy(), np.asarray(statuses).copy(),
            np.asarray(particles.particle_id).copy())


def particle_snapshot(particles):
    """Include runtime species/spin metadata and state, always by ID."""
    answer = particle_values(particles)
    order = np.argsort(np.asarray(particles.particle_id))
    for name in ('state', 'particle_id', 'chi', 'charge_ratio',
                 'anomalous_magnetic_moment'):
        answer[name] = np.asarray(getattr(particles, name))[order].copy()
    return answer


def material_field_probes(context, element, z, point, u):
    """Differentiate physical fields using two central stencils/Richardson.

    At each shifted Z the actual analytic profile jets, native s and zeta
    change consistently. A fixed query profile would omit spatial derivatives;
    shifting s alone would contaminate them with the RF time derivative.
    """
    material = []
    center = None
    beta = np.asarray(u)/np.hypot(1., np.linalg.norm(u))
    for h in (1e-4, 5e-5):
        points = np.tile(point, (7, 1))
        for axis in range(3):
            points[1+2*axis, axis] += h
            points[2+2*axis, axis] -= h
        particles = event_particles(context, element, z, points, momentum=u)
        values, status, ids = observe(element, particles, z, profile_at(z), 0.)
        assert np.all(status == 1)
        fields = values[np.argsort(ids), 12:18]
        center = fields[0]
        gradient = np.array([(fields[1+2*j]-fields[2+2*j])/(2*h)
                             for j in range(3)])
        longitudinal = []
        for shifted in (z+h, z-h):
            particles = event_particles(context, element, shifted, point,
                                         momentum=u)
            values, status, _ = observe(element, particles, shifted,
                                        profile_at(shifted), 0.)
            assert status[0] == 1
            longitudinal.append(values[0, 12:18])
        dz = (longitudinal[0]-longitudinal[1])/(2*h)
        material.append(gradient[2] + beta[0]*gradient[0]
                        + beta[1]*gradient[1] + beta[2]*dz)
    richardson = (4*material[1]-material[0])/3
    np.testing.assert_allclose(material[1], richardson, rtol=2e-7, atol=2e-10)
    assert np.linalg.norm(richardson) > 1e-3
    return center[:3], center[3:], richardson[:3], richardson[3:]


def cross_matrix(vector):
    """The fixed-laboratory-axis cross product operator."""
    x, y, z = vector
    return np.array(((0., -z, y), (z, 0., -x), (-y, x, 0.)))


def point_oracle(u, Q, rho, anomaly, E, B, DE, DB):
    """Independent vector LL, covariant BMT/Fermi-Walker and accounting.

    Field tensor convention is (+---), F*U=(E.u, gamma*E+u cross B).
    The reaction acceleration below is dU/dT; its proper-time value is gamma
    times this, so division by gamma leaves the displayed extra spin term.
    """
    u = np.asarray(u)
    gamma = np.hypot(1., np.linalg.norm(u))
    beta = u/gamma
    clock = gamma/u[2]
    G = E+np.cross(beta, B)
    force = Q*G
    emitted = Q**2*gamma**2*(G@G-(beta@E)**2)
    coefficient = (Q*gamma*(DE+np.cross(beta, DB))
        + Q**2*(np.cross(E, B)+np.cross(B, np.cross(B, beta))+E*(beta@E))
        - emitted*beta)
    reaction = rho*coefficient
    correction = gamma**2*(2*(force@reaction-(beta@force)*(beta@reaction))
        + reaction@reaction-(beta@reaction)**2)
    omega = -Q*((anomaly+1/gamma)*B
        - anomaly*gamma/(gamma+1)*(beta@B)*beta
        - (anomaly+1/(gamma+1))*np.cross(beta, E))
    expected = np.r_[np.zeros(3), clock*reaction[:2], clock*(beta@reaction),
        omega, np.cross(reaction, u)/(gamma+1), E, B,
        Q*(beta@E), beta@reaction, rho*emitted, rho*Q*(u@E),
        -rho*Q*(E@reaction), rho*(emitted+correction),
        rho*(-Q*(E@reaction)+correction)]
    U = np.r_[gamma, u]
    metric = np.diag((1., -1., -1., -1.))
    tensor = np.zeros((4, 4))
    tensor[0, 1:] = tensor[1:, 0] = E
    tensor[1:, 1:] = -cross_matrix(B)
    spin = np.vstack((u, np.eye(3)+np.outer(u, u)/(gamma+1)))
    spin_dot = Q/gamma*((1+anomaly)*(tensor@spin)
        + anomaly*np.outer(U, spin.T@metric@(tensor@U)))
    spin_dot -= np.outer(U, spin.T@metric@np.r_[beta@reaction, reaction])
    udot = force+reaction
    rest_dot = (spin_dot[1:]-np.outer(udot, spin[0])/(gamma+1)
        - np.outer(u, spin_dot[0])/(gamma+1)
        + np.outer(u, spin[0])*(beta@udot)/(gamma+1)**2)
    np.testing.assert_allclose(rest_dot+rest_dot.T, 0., atol=2e-13)
    covariant_omega = np.array((rest_dot[2, 1], rest_dot[0, 2], rest_dot[1, 0]))
    return expected, covariant_omega


def test_native_secondary_independent_point_laws(tmp_path):
    """Compile once, then check physical probes, species and local limits."""
    with native_context(tmp_path) as context:
        element = observer_element(context)
        z = .03
        point = np.array((.003, -.002, .4))
        u = np.array((.2, -.1, 2.))
        E, B, DE, DB = material_field_probes(context, element, z, point, u)
        masses = np.array((MASS0, MASS0, 938272088.16, 193687272900., MASS0))
        charges = np.array((-1., 1., 1., 82., -1.))
        ids = np.array((41, 7, 99, 8, 5))
        anomalies = np.array((.001, .002, .3, 0., .004))
        # The first slot is lost, making a slot-indexed rho array unsafe if
        # the native wrapper reorganizes particles before the callback.
        states = np.array((0, 1, 1, 1, 1))
        by_id = {int(pid): (float(mass), float(charge), int(state), float(a))
                 for pid, mass, charge, state, a in zip(
                     ids, masses, charges, states, anomalies)}
        particles = event_particles(context, element, z, np.tile(point, (5, 1)),
            masses=masses, charges=charges, momentum=u, ids=ids, states=states,
            anomaly=anomalies)
        before = particle_snapshot(particles)
        # Algebraic stress, not a physical LL domain claim.
        rho_unit = MASS0*1e-8
        values, status, after_ids = observe(element, particles, z,
                                            profile_at(z), rho_unit)
        for row, code, pid in zip(values, status, after_ids):
            mass, charge, state, anomaly = by_id[int(pid)]
            if state <= 0:
                assert code == 0 and np.all(np.isnan(row))
                continue
            assert code == 1
            Q = charge*MASS0/mass
            rho = rho_unit*charge**2/mass
            expected, covariant = point_oracle(u, Q, rho, anomaly, E, B, DE, DB)
            assert np.max(abs(expected[:6]/rho)) > 1e-7
            assert np.max(abs(row[:6])) > 0
            np.testing.assert_allclose(row[:6]/rho, expected[:6]/rho,
                                       rtol=2e-7, atol=2e-9)
            np.testing.assert_allclose(row[9:12]/rho, expected[9:12]/rho,
                                       rtol=2e-7, atol=2e-9)
            np.testing.assert_allclose(row[6:9], expected[6:9],
                                       rtol=2e-12, atol=2e-13)
            np.testing.assert_allclose(row[6:9]+row[9:12], covariant,
                                       rtol=2e-10, atol=3e-12)
            np.testing.assert_allclose(row[18], expected[18], atol=2e-13)
            np.testing.assert_allclose(row[19:]/rho, expected[19:]/rho,
                                       rtol=2e-7, atol=2e-9)
        for name, saved in before.items():
            np.testing.assert_array_equal(
                particle_snapshot(particles)[name], saved)
        zero, status, _ = observe(element, particles, z, profile_at(z), 0.)
        np.testing.assert_array_equal(zero[status == 1, :6], 0.)
        np.testing.assert_array_equal(zero[status == 1, 9:12], 0.)
        # A local uniform axial electric crest: derivative LL, quadratic LL
        # and drag cancel, while emitted energy is strictly positive.
        axial_profile = np.zeros(12)
        axial_profile[0] = 1.
        particles = event_particles(context, element, z,
            np.zeros((2, 3)), charges=np.array((-1., 1.)),
            momentum=(0., 0., KAPPA0))
        values, status, _ = observe(
            element, particles, z, axial_profile, rho_unit)
        assert np.all(status == 1)
        np.testing.assert_allclose(values[:, :6]/1e-8, 0., atol=2e-12)
        np.testing.assert_allclose(values[:, 9:12]/1e-8, 0., atol=2e-12)
        assert np.all(values[:, 20] > 0)
        np.testing.assert_allclose(values[:, 20]/1e-8, 1., atol=2e-12)


def compile_schott_query(context):
    """A thin direct C-source consumer, without a new tracking element."""
    context.add_kernels(sources=[r'''
        #include "xobjects/headers/common.h"
        #include "xtrack/beam_elements/elements_src/cavity_native_secondary.h"
        GPUKERN
        void cavity_schott_source_review(double* q, double charge,
                double* profile, double aperture, double* output,
                int64_t* status) {
            status[0] = CavityNative_schott_source(
                q, charge, profile, aperture, output);
        }
        GPUKERN
        void cavity_spin_response_source_review(double* q, double charge,
                double anomaly, double* profile, double aperture,
                double* output, int64_t* status) {
            status[0] = CavityNative_spin_response_source(
                q, charge, anomaly, profile, aperture, output);
        }
        GPUKERN
        void cavity_spin_schott_source_review(double* q, double charge,
                double anomaly, double* profile, double aperture,
                double* output, int64_t* status) {
            status[0] = CavityNative_spin_schott_source(
                q, charge, anomaly, profile, aperture, output);
        }
    '''], kernels={'cavity_schott_source_review': xo.Kernel(args=[
        xo.Arg(xo.Float64, pointer=True, name='q'),
        xo.Arg(xo.Float64, name='charge'),
        xo.Arg(xo.Float64, pointer=True, name='profile'),
        xo.Arg(xo.Float64, name='aperture'),
        xo.Arg(xo.Float64, pointer=True, name='output'),
        xo.Arg(xo.Int64, pointer=True, name='status')], n_threads=1),
        'cavity_spin_response_source_review': xo.Kernel(args=[
            xo.Arg(xo.Float64, pointer=True, name='q'),
            xo.Arg(xo.Float64, name='charge'),
            xo.Arg(xo.Float64, name='anomaly'),
            xo.Arg(xo.Float64, pointer=True, name='profile'),
            xo.Arg(xo.Float64, name='aperture'),
            xo.Arg(xo.Float64, pointer=True, name='output'),
            xo.Arg(xo.Int64, pointer=True, name='status')], n_threads=1),
        'cavity_spin_schott_source_review': xo.Kernel(args=[
            xo.Arg(xo.Float64, pointer=True, name='q'),
            xo.Arg(xo.Float64, name='charge'),
            xo.Arg(xo.Float64, name='anomaly'),
            xo.Arg(xo.Float64, pointer=True, name='profile'),
            xo.Arg(xo.Float64, name='aperture'),
            xo.Arg(xo.Float64, pointer=True, name='output'),
            xo.Arg(xo.Int64, pointer=True, name='status')], n_threads=1)})


def schott_query(context, q, charge, profile, *, aperture=.1):
    """Retain finite sentinels to expose any partial source write on refusal."""
    state = context.nparray_to_context_array(np.asarray(q, dtype=float).copy())
    jets = context.nparray_to_context_array(
        np.asarray(profile, dtype=float).copy())
    sentinel = np.arange(12, dtype=float) + 81.25
    output = context.nparray_to_context_array(sentinel.copy())
    status = context.nparray_to_context_array(np.zeros(1, dtype=np.int64))
    context.kernels.cavity_schott_source_review(q=state, charge=charge,
        profile=jets, aperture=aperture, output=output, status=status)
    np.testing.assert_array_equal(np.asarray(state), q)
    np.testing.assert_array_equal(np.asarray(jets), profile)
    return np.asarray(output).copy(), int(np.asarray(status)[0]), sentinel


def canonical_event(point, momentum, charge, profile):
    """Encode the declared temporal M2 potential, independently in Python.

    Only the common source/gauge/chart is shared here. The Schott and emitted
    flux formula below is independent of the generated source expressions.
    """
    values = np.asarray(profile[:5]) + 1j*np.asarray(profile[5:])
    imaginary = np.imag(values*np.exp(-1j*point[2]))
    radial = point[0]**2 + point[1]**2
    transverse = point[:2]*(
        -imaginary[1]/2 + radial*(imaginary[1]+imaginary[3])/16)
    gamma = np.hypot(1., np.linalg.norm(momentum))
    return np.r_[point, momentum[:2]+charge*transverse, gamma]


def spin_source_query(context, q, charge, anomaly, profile, *, aperture=.1):
    """Query all 24 rho-free coefficients; protect outputs on failure."""
    state = context.nparray_to_context_array(np.asarray(q, dtype=float).copy())
    jets = context.nparray_to_context_array(
        np.asarray(profile, dtype=float).copy())
    sentinel = np.arange(24, dtype=float) + 117.25
    output = context.nparray_to_context_array(sentinel.copy())
    status = context.nparray_to_context_array(np.zeros(1, dtype=np.int64))
    context.kernels.cavity_spin_response_source_review(q=state, charge=charge,
        anomaly=anomaly, profile=jets, aperture=aperture, output=output,
        status=status)
    np.testing.assert_array_equal(np.asarray(state), q)
    np.testing.assert_array_equal(np.asarray(jets), profile)
    return np.asarray(output).copy(), int(np.asarray(status)[0]), sentinel


def mechanical_event(q, charge, profile):
    """Invert the declared temporal M2 chart at fixed source and fixed Z."""
    profile10 = np.r_[profile[:5], profile[6:11]]
    offset = canonical_event(q[:3], np.zeros(3), charge, profile10)[3:5]
    transverse = q[3:5]-offset
    uz = np.sqrt((q[5]-1)*(q[5]+1)-transverse@transverse)
    return np.r_[transverse, uz]


def covariant_spin_generator(context, element, z, q, charge, anomaly, rho):
    """Decode each physical event, boost its spin basis and include its clock.

    The field algebra is shared with the observer. Differentiation, covariant
    BMT/Fermi-Walker and the rest-spin boost use independent vector operations.
    Positive formal rho is solely an algebraic directional-derivative probe.
    """
    u = mechanical_event(q, charge, profile_at(z))
    if rho:
        E, B, DE, DB = material_field_probes(
            context, element, z, q[:3], u)
    else:
        particles = event_particles(context, element, z, q[:3], momentum=u)
        values, statuses, _ = observe(
            element, particles, z, profile_at(z), 0.)
        assert statuses[0] == 1
        E, B = values[0, 12:15], values[0, 15:18]
        DE = DB = np.zeros(3)
    _, omega = point_oracle(u, charge, rho, anomaly, E, B, DE, DB)
    clock = q[5]/u[2]
    return clock*omega, clock, u, E, B


def test_native_schott_source_independent_coefficients(tmp_path):
    """Check rho-free full-six s/ell, nonzero faces and transactional refusal.

    These are local impressed-source points, not admitted cavity boundaries.
    The source's classical emitted momentum is W*beta; replacing it by an
    on-shell photon energy decrement gives a different vector and must fail.
    """
    with native_context(tmp_path) as context:
        compile_schott_query(context)
        element = observer_element(context)
        # This local law fixture excites transverse Schott momentum above
        # its stated nonzero guard; it is not the gamma=2.4 cavity launch.
        u = np.array((.2, -.1, 10.))
        gamma = np.hypot(1., np.linalg.norm(u))
        beta = u/gamma
        boundary_s = []
        for z, point in ((-.08, np.array((.003, -.002, .4))),
                         (.09, np.array((.004, -.001, .7)))):
            profile = profile_at(z, 4)
            particles = event_particles(context, element, z, point, momentum=u)
            fields, statuses, _ = observe(
                element, particles, z, profile_at(z), 0.)
            assert statuses[0] == 1
            raw_e, raw_b = fields[0, 12:15], fields[0, 15:18]
            signed = []
            for charge in (-1., 1.):
                e, b = charge*raw_e, charge*raw_b
                force = e+np.cross(beta, b)
                direction = u/np.linalg.norm(u)
                parallel = direction@force
                perpendicular = force-parallel*direction
                power = parallel**2 + gamma**2*(perpendicular@perpendicular)
                s = np.r_[np.zeros(3), gamma*force[:2], gamma*(beta@e)]
                ell = np.r_[np.zeros(3), power*beta[:2], power]/beta[2]
                q = canonical_event(point, u, charge, profile)
                value, status, _ = schott_query(context, q, charge, profile)
                assert status == 1 and power > 0
                assert np.linalg.norm(s[3:5]) > 1e-5
                assert abs(s[5]) > 1e-3
                np.testing.assert_allclose(value[:6], s,
                                           rtol=2e-12, atol=2e-13)
                np.testing.assert_allclose(value[6:], ell,
                                           rtol=2e-12, atol=2e-13)
                np.testing.assert_allclose(value[9:11]/value[11], beta[:2],
                                           rtol=2e-13, atol=2e-15)
                signed.append(value)
            np.testing.assert_allclose(signed[0][:6], -signed[1][:6],
                                       rtol=2e-12, atol=2e-13)
            np.testing.assert_allclose(signed[0][6:], signed[1][6:],
                                       rtol=2e-12, atol=2e-13)
            boundary_s.append(signed[0][:6])
        assert np.max(abs(boundary_s[1]-boundary_s[0])) > 1e-3
        for charge, data in ((0., profile), (1., np.zeros(10))):
            q = canonical_event(point, u, charge, data)
            value, status, _ = schott_query(context, q, charge, data)
            assert status == 1
            np.testing.assert_array_equal(value, 0.)
        q = canonical_event(point, u, -1., profile)
        low_energy = q.copy()
        low_energy[5] = .9
        invalid_profile = profile.copy()
        invalid_profile[4] = np.nan
        for state, data, aperture, expected in (
            (low_energy, profile, .1, -1),
            (q, profile, .001, -1),
            (q, invalid_profile, .1, -3),
            (q, profile, -1., -4),
        ):
            value, status, sentinel = schott_query(
                context, state, -1., data, aperture=aperture)
            assert status == expected
            np.testing.assert_array_equal(value, sentinel)


def test_native_spin_response_independent_directional_law(tmp_path):
    """Check full clock/physical-response coupling against covariant spin.

    This is a rho-free local source check, not a transported spin-response
    word or physical finite-rho acceptance. The two formal steps resolve the
    derivative without subtracting tiny actual-species radiation increments.
    """
    with native_context(tmp_path) as context:
        compile_schott_query(context)
        element = observer_element(context)
        z = .03
        point = np.array((.003, -.002, .4))
        u = np.array((.2, -.1, 2.))
        profile = profile_at(z)
        profile10 = np.r_[profile[:5], profile[6:11]]
        masses = (MASS0, MASS0, 938272088.16, 193687272900.)
        charges = (-1., 1., 1., 82.)
        anomalies = (.001, .002, .3, 0.)
        xi = np.array((.03, -.02, .01, .02, -.01, .025))
        steps = (2e-4, 1e-4)
        for mass, physical_charge, anomaly in zip(masses, charges, anomalies):
            charge = physical_charge*MASS0/mass
            scale = abs(charge)
            q = canonical_event(point, u, charge, profile10)
            source, status, _ = spin_source_query(
                context, q, charge, anomaly, profile)
            assert status == 1
            g0, clock, actual_u, E, B = covariant_spin_generator(
                context, element, z, q, charge, anomaly, 0.)
            np.testing.assert_allclose(source[:3]/scale, g0/scale,
                                       rtol=2e-11, atol=2e-13)
            np.testing.assert_allclose(actual_u, u, atol=2e-15)
            jacobians, clock_jacobians = [], []
            for h in steps:
                columns, clock_columns = [], []
                for column in range(6):
                    direction = np.eye(6)[column]
                    plus = covariant_spin_generator(context, element, z,
                        q+h*direction, charge, anomaly, 0.)
                    minus = covariant_spin_generator(context, element, z,
                        q-h*direction, charge, anomaly, 0.)
                    columns.append((plus[0]-minus[0])/(2*h))
                    clock_columns.append((plus[1]-minus[1])/(2*h))
                jacobians.append(np.array(columns).T)
                clock_jacobians.append(np.array(clock_columns))
            jacobian = (4*jacobians[1]-jacobians[0])/3
            clock_jacobian = (4*clock_jacobians[1]-clock_jacobians[0])/3
            np.testing.assert_allclose(jacobians[1]/scale, jacobian/scale,
                                       rtol=3e-7, atol=3e-10)
            native_jacobian = source[3:21].reshape(3, 6)
            np.testing.assert_allclose(native_jacobian/scale, jacobian/scale,
                                       rtol=3e-7, atol=3e-10)
            assert np.linalg.norm(jacobian/scale) > 1e-3
            beta = u/q[5]
            force = charge*(E+np.cross(beta, B))
            # This independent physical Schott event vector restores all six
            # canonical coordinates; xi alone is not the physical response.
            s = np.r_[np.zeros(3), q[5]*force[:2], q[5]*charge*(beta@E)]
            response = xi+s
            g1_steps, full_steps = [], []
            for h in steps:
                g1 = covariant_spin_generator(context, element, z, q,
                    charge, anomaly, h)[0]
                full = covariant_spin_generator(context, element, z,
                    q+h*response, charge, anomaly, h)[0]
                g1_steps.append((g1-g0)/h)
                full_steps.append((full-g0)/h)
            g1_reference = 2*g1_steps[1]-g1_steps[0]
            full_reference = 2*full_steps[1]-full_steps[0]
            np.testing.assert_allclose(source[21:]/scale,
                g1_reference/scale, rtol=3e-7, atol=3e-10)
            full_native = native_jacobian@response+source[21:]
            np.testing.assert_allclose(full_native/scale,
                full_reference/scale, rtol=3e-7, atol=3e-10)
            changed_anomaly, status, _ = spin_source_query(
                context, q, charge, anomaly+.137, profile)
            assert status == 1
            np.testing.assert_array_equal(changed_anomaly[21:], source[21:])
            if abs(charge) == 1:
                error = np.linalg.norm(full_reference-full_steps[1])
                assert np.linalg.norm(native_jacobian@s) > 100*(error+1e-12)
                clock_fault = (g0/clock)*(clock_jacobian@response)
                assert np.linalg.norm(clock_fault) > 100*(error+1e-12)
        for charge, data in ((0., profile), (1., np.zeros(12))):
            q = canonical_event(point, u, charge,
                                np.r_[data[:5], data[6:11]])
            source, status, _ = spin_source_query(
                context, q, charge, .001, data)
            assert status == 1
            np.testing.assert_array_equal(source, 0.)
        q = canonical_event(point, u, -1., profile10)
        low_energy = q.copy()
        low_energy[5] = .9
        invalid_profile = profile.copy()
        invalid_profile[5] = np.nan
        for state, data, anomaly, aperture, expected in (
            (low_energy, profile, .001, .1, -1),
            (q, invalid_profile, .001, .1, -3),
            (q, profile, np.nan, .1, -4),
            (q, profile, .001, -1., -4),
        ):
            value, status, sentinel = spin_source_query(
                context, state, -1., anomaly, data, aperture=aperture)
            assert status == expected
            np.testing.assert_array_equal(value, sentinel)


def compile_spin_word_query(context):
    """Expose the actual word and its existing accepted-stage consumers."""
    context.add_kernels(sources=[r'''
        #include "xobjects/headers/common.h"
        #include "xtrack/beam_elements/elements_src/cavity_native_response.h"
        GPUKERN
        void cavity_spin_word_review(double* q, double* xi, double h,
                double charge, double anomaly, double* profiles,
                double response_tol, int64_t stages, double* output,
                double* qs, double* Rs, double* rs, int64_t* status) {
            if (!stages) {
                status[0] = CavityNative_gauss_schott_spin_response(
                    q, xi, h, charge, anomaly, profiles, .1, 80,
                    2e-13, 2e-13, 2e-13, response_tol, output, output+6,
                    output+12, output+18, output+27, output+36, output+39);
                return;
            }
            double p[40], meanq[6], meanqs[12], nominal_residual;
            for (int plane=0; plane<4; ++plane)
                for (int j=0; j<5; ++j) {
                    p[plane*10+j] = profiles[plane*12+j];
                    p[plane*10+j+5] = profiles[plane*12+j+6];
                }
            status[0] = CavityNative_gauss_spin_stages(q, h, charge,
                anomaly, p, .1, 80, 2e-13, 2e-13, output, output+18,
                output+39, output+41, qs, Rs);
            if (status[0] != 1) return;
            status[0] = CavityNative_gauss_schott_response_stages(q, xi,
                h, charge, p, .1, 80, 2e-13, 2e-13, meanq, output+6,
                output+12, output+36, &nominal_residual, output+40,
                meanqs, rs);
            if (status[0] != 1) return;
            for (int j=0; j<6; ++j)
                if (output[j] != meanq[j]
                        || signbit(output[j]) != signbit(meanq[j]))
                    status[0] = -4;
            for (int j=0; j<12; ++j)
                if (qs[j] != meanqs[j]
                        || signbit(qs[j]) != signbit(meanqs[j]))
                    status[0] = -4;
        }
    '''], kernels={'cavity_spin_word_review': xo.Kernel(args=[
        xo.Arg(xo.Float64, pointer=True, name='q'),
        xo.Arg(xo.Float64, pointer=True, name='xi'),
        xo.Arg(xo.Float64, name='h'),
        xo.Arg(xo.Float64, name='charge'),
        xo.Arg(xo.Float64, name='anomaly'),
        xo.Arg(xo.Float64, pointer=True, name='profiles'),
        xo.Arg(xo.Float64, name='response_tol'),
        xo.Arg(xo.Int64, name='stages'),
        xo.Arg(xo.Float64, pointer=True, name='output'),
        xo.Arg(xo.Float64, pointer=True, name='qs'),
        xo.Arg(xo.Float64, pointer=True, name='Rs'),
        xo.Arg(xo.Float64, pointer=True, name='rs'),
        xo.Arg(xo.Int64, pointer=True, name='status')], n_threads=1)})


def spin_word_query(context, q, xi, h, charge, anomaly, profiles, *,
                    stages=False, response_tol=2e-13):
    """Keep every public output sentinel, including unused stage buffers."""
    inputs = [np.asarray(value, dtype=float).copy()
              for value in (q, xi, profiles)]
    buffers = [context.nparray_to_context_array(value) for value in inputs]
    sentinels = [np.arange(size, dtype=float)+211.25+i
                 for i, size in enumerate((43, 12, 18, 12))]
    outputs = [context.nparray_to_context_array(value.copy())
               for value in sentinels]
    status = context.nparray_to_context_array(np.zeros(1, dtype=np.int64))
    context.kernels.cavity_spin_word_review(q=buffers[0], xi=buffers[1], h=h,
        charge=charge, anomaly=anomaly, profiles=buffers[2],
        response_tol=response_tol, stages=int(stages), output=outputs[0],
        qs=outputs[1], Rs=outputs[2], rs=outputs[3], status=status)
    for before, after in zip(inputs, buffers):
        np.testing.assert_array_equal(before, np.asarray(after))
    return ([np.asarray(value).copy() for value in outputs],
            int(np.asarray(status)[0]), sentinels)


def dense_spin_word(generators, forcing, h, incoming, incoming_response):
    """Solve each vector column with the defining two-stage Gauss matrix."""
    A = np.array(((.25, .25-np.sqrt(3)/6),
                  (.25+np.sqrt(3)/6, .25)))
    identity = np.eye(3)
    system = np.block([[identity-h*A[0, 0]*generators[0],
                        -h*A[0, 1]*generators[1]],
                       [-h*A[1, 0]*generators[0],
                        identity-h*A[1, 1]*generators[1]]])
    rotation = np.linalg.solve(system, np.vstack((incoming, incoming)))
    rotation = rotation.reshape(2, 3, 3)
    source = [forcing[j]@rotation[j] for j in range(2)]
    rhs = np.vstack([incoming_response+h*(A[i, 0]*source[0]
                        + A[i, 1]*source[1]) for i in range(2)])
    response = np.linalg.solve(system, rhs).reshape(2, 3, 3)
    out_rotation = incoming+h/2*sum(
        generators[j]@rotation[j] for j in range(2))
    out_response = incoming_response+h/2*sum(
        generators[j]@response[j]+source[j] for j in range(2))
    return out_rotation, out_response, rotation


def independent_schott_event(context, element, z, q, charge):
    """The physical event vector fixes xi=r-s without a generated split."""
    _, _, u, E, B = covariant_spin_generator(
        context, element, z, q, charge, .001, 0.)
    force = charge*(E+np.cross(u/q[5], B))
    return np.r_[np.zeros(3), q[5]*force[:2], charge*(u@E)]


def test_native_schott_spin_word_independent_dense_carry(tmp_path):
    """Check actual accepted stages, dense covariant spin and global carry.

    Matrix coefficients below are formal gyro diagnostics for the supplied
    anomalies. In particular the Pb lane has zero physical input spin; its
    zero-spin readback remains zero. No physical finite-rho claim is made.
    """
    with native_context(tmp_path) as context:
        compile_spin_word_query(context)
        element = observer_element(context)
        point = np.array((.003, -.002, .4))
        momentum = np.array((.2, -.1, 2.))
        mesh = np.array((-.04, .02, .08))
        nodes = np.array((.5-np.sqrt(3)/6, .5+np.sqrt(3)/6))
        masses = (MASS0, MASS0, 938272088.16, 193687272900.)
        charges = (-1., 1., 1., 82.)
        anomalies = (.001, .002, .3, 0.)
        angle = .31
        axis = np.array((1., 2., -1.))/np.sqrt(6)
        skew = cross_matrix(axis)
        initial_rotation = (np.eye(3)+np.sin(angle)*skew
                            +(1-np.cos(angle))*(skew@skew))
        tangent = cross_matrix(np.array((.017, -.011, .023)))
        initial_response = tangent@initial_rotation
        strict_case = None
        for mass, physical_charge, anomaly in zip(masses, charges, anomalies):
            charge = physical_charge*MASS0/mass
            scale = abs(charge)
            p0 = profile_at(mesh[0], 4)
            q = canonical_event(point, momentum, charge, p0)
            incoming_r = np.array((.03, -.02, .01, .02, -.01, .025))
            xi = incoming_r-independent_schott_event(
                context, element, mesh[0], q, charge)
            rotation = initial_rotation.copy()
            response = initial_response.copy()
            for start, stop in zip(mesh[:-1], mesh[1:]):
                h = stop-start
                planes = np.r_[start, start+h*nodes, stop]
                profiles = np.concatenate([profile_at(z) for z in planes])
                inputs = (q.copy(), xi.copy(), h, charge, anomaly,
                          profiles.copy())
                outputs, status, _ = spin_word_query(context, *inputs)
                assert status == 1
                data = outputs[0]
                accepted, status, _ = spin_word_query(
                    context, *inputs, stages=True)
                assert status == 1
                nominal, qs, Rs, rs = accepted
                for left, right in ((data[:27], nominal[:27]),
                                    (data[36:42], nominal[36:42])):
                    np.testing.assert_array_equal(
                        left.view(np.uint64), right.view(np.uint64))
                states = qs.reshape(2, 6)
                physical = rs.reshape(2, 6)
                generators, forcing, auxiliary_forcing = [], [], []
                for z, state, r in zip(planes[1:3], states, physical):
                    g0 = covariant_spin_generator(context, element, z,
                        state, charge, anomaly, 0.)[0]
                    derivatives = []
                    for step in (2e-4, 1e-4):
                        altered = covariant_spin_generator(context, element,
                            z, state+step*r, charge, anomaly, step)[0]
                        derivatives.append((altered-g0)/step)
                    H1 = 2*derivatives[1]-derivatives[0]
                    generators.append(cross_matrix(g0))
                    forcing.append(cross_matrix(H1))
                    if abs(charge) == 1:
                        s = independent_schott_event(
                            context, element, z, state, charge)
                        derivatives = []
                        for step in (2e-4, 1e-4):
                            plus = covariant_spin_generator(context, element,
                                z, state+step*s, charge, anomaly, 0.)[0]
                            minus = covariant_spin_generator(
                                context, element, z, state-step*s,
                                charge, anomaly, 0.)[0]
                            derivatives.append((plus-minus)/(2*step))
                        schott_direction = (4*derivatives[1]
                                            - derivatives[0])/3
                        auxiliary_forcing.append(
                            cross_matrix(H1-schott_direction))
                U0, U1, local_stages = dense_spin_word(generators, forcing,
                    h, np.eye(3), np.zeros((3, 3)))
                np.testing.assert_allclose(Rs.reshape(2, 3, 3), local_stages,
                                           rtol=2e-11, atol=2e-13)
                np.testing.assert_allclose(data[18:27].reshape(3, 3), U0,
                                           rtol=2e-11, atol=2e-13)
                np.testing.assert_allclose(data[27:36].reshape(3, 3)/scale,
                    U1/scale, rtol=3e-6, atol=3e-10)
                expected_R, expected_P, _ = dense_spin_word(generators,
                    forcing, h, rotation, response)
                native_U0 = data[18:27].reshape(3, 3)
                native_U1 = data[27:36].reshape(3, 3)
                carried_R = native_U0@rotation
                carried_P = native_U1@rotation+native_U0@response
                np.testing.assert_allclose(carried_R, expected_R,
                                           rtol=2e-11, atol=2e-13)
                np.testing.assert_allclose(carried_P, expected_P,
                                           rtol=3e-8, atol=3e-11)
                if abs(charge) == 1:
                    # Wrong multiplication order or resetting incoming P
                    # cannot masquerade as the global defining-word result.
                    wrong = rotation@native_U1+response@native_U0
                    assert np.max(abs(carried_P-wrong)) > 1e-9
                    assert np.max(abs(native_U1)) > 1e-8
                    _, wrong_auxiliary, _ = dense_spin_word(generators,
                        auxiliary_forcing, h, np.eye(3), np.zeros((3, 3)))
                    assert np.max(abs(U1-wrong_auxiliary)) > 1e-10
                np.testing.assert_allclose(carried_R.T@carried_P
                    + carried_P.T@carried_R, 0., atol=2e-11)
                np.testing.assert_array_equal(carried_P@np.zeros(3), 0.)
                q, xi = data[:6].copy(), data[6:12].copy()
                rotation, response = carried_R, carried_P
                if strict_case is None:
                    strict_case = inputs
            assert np.max(abs(response-initial_response)) > 1e-10*scale
        failed, status, sentinels = spin_word_query(context, *strict_case,
                                                   response_tol=1e-300)
        assert status == -2
        for output, sentinel in zip(failed, sentinels):
            np.testing.assert_array_equal(output, sentinel)


def spin_schott_source_query(context, q, charge, anomaly, profile, *,
                             aperture=.1):
    """Retain all 27 output sentinels on any source refusal."""
    state = context.nparray_to_context_array(np.asarray(q, dtype=float).copy())
    jets = context.nparray_to_context_array(
        np.asarray(profile, dtype=float).copy())
    sentinel = np.arange(27, dtype=float)+331.25
    output = context.nparray_to_context_array(sentinel.copy())
    status = context.nparray_to_context_array(np.zeros(1, dtype=np.int64))
    context.kernels.cavity_spin_schott_source_review(q=state, charge=charge,
        anomaly=anomaly, profile=jets, aperture=aperture, output=output,
        status=status)
    np.testing.assert_array_equal(np.asarray(state), q)
    np.testing.assert_array_equal(np.asarray(jets), profile)
    return np.asarray(output).copy(), int(np.asarray(status)[0]), sentinel


def covariant_field_generator(u, charge, anomaly, E, B):
    """The independent rest-spin generator with its actual per-Z clock."""
    _, omega = point_oracle(u, charge, 0., anomaly, E, B,
                            np.zeros(3), np.zeros(3))
    return np.hypot(1., np.linalg.norm(u))/u[2]*omega


def unit_spin_schott_oracle(u, anomaly, E, B):
    """Resolve the quadratic coefficient using a unit-charge direction.

    At fixed mechanical event k is linear and B is quadratic in signed Q.
    Differentiating unit-charge generators avoids subtracting O(Q) numbers
    along O(Q) directions for ions. No actual small radiation is divided out.
    """
    gamma = np.hypot(1., np.linalg.norm(u))
    beta = u/gamma
    force = E+np.cross(beta, B)
    schott_momentum = gamma*force
    k = np.cross(schott_momentum, u)/(gamma+1)
    g0 = covariant_field_generator(u, 1., anomaly, E, B)
    directions = []
    for h in (2e-4, 1e-4):
        plus = covariant_field_generator(
            u+h*schott_momentum, 1., anomaly, E, B)
        minus = covariant_field_generator(
            u-h*schott_momentum, 1., anomaly, E, B)
        directions.append((plus-minus)/(2*h))
    derivative = (4*directions[1]-directions[0])/3
    clock = gamma/u[2]
    residue = (derivative+clock*k*(beta@E)/(gamma+1)
               + np.cross(g0, k))
    return k, residue, derivative, g0


def test_native_spin_schott_source_independent_cancellation(tmp_path):
    """Check new k/B coefficients, charge scaling and full D0k cancellation.

    The shifted events form a Lorentz tangent curve with actual source jets,
    rather than holding profiles fixed when Z changes. This tests a local
    coordinate identity, not the accuracy of a new finite response word.
    """
    with native_context(tmp_path) as context:
        compile_schott_query(context)
        element = observer_element(context)
        z = .03
        point = np.array((.003, -.002, .4))
        u = np.array((.2, -.1, 2.))
        gamma = np.hypot(1., np.linalg.norm(u))
        beta = u/gamma
        clock = gamma/u[2]
        profile = profile_at(z, 4)
        E, B, DE, DB = material_field_probes(
            context, element, z, point, u)
        for anomaly in (.001, .3):
            k, residue, direction, g0 = unit_spin_schott_oracle(
                u, anomaly, E, B)
            assert np.linalg.norm(k) > 1e-5
            assert np.linalg.norm(residue) > 1e-7
            for charge in (-1., 1., MASS0/938272088.16,
                           82*MASS0/193687272900.):
                q = canonical_event(point, u, charge, profile)
                source, status, _ = spin_schott_source_query(
                    context, q, charge, anomaly, profile)
                assert status == 1
                np.testing.assert_allclose(source[:3]/charge, g0,
                                           rtol=2e-11, atol=2e-13)
                np.testing.assert_allclose(source[21:24]/charge, k,
                                           rtol=2e-11, atol=2e-13)
                np.testing.assert_allclose(source[24:]/charge**2, residue,
                                           rtol=3e-7, atol=3e-10)
                schott = np.r_[np.zeros(3), gamma*charge*(
                    E+np.cross(beta, B))[:2], gamma*charge*(beta@E)]
                native_direction = source[3:21].reshape(3, 6)@schott
                np.testing.assert_allclose(native_direction/charge**2,
                                           direction, rtol=3e-7, atol=3e-10)
            # Signed total derivatives include the material field derivative
            # that the new residue has eliminated analytically, not omitted.
            for charge in (-1., 1.):
                force = charge*(E+np.cross(beta, B))
                velocity = np.r_[u[:2]/u[2], clock]
                derivatives = []
                for h in (2e-4, 1e-4):
                    shifted_k = []
                    for sign in (1., -1.):
                        shifted_z = z+sign*h
                        shifted_point = point+sign*h*velocity
                        shifted_u = u+sign*h*clock*force
                        particles = event_particles(context, element,
                            shifted_z, shifted_point, momentum=shifted_u)
                        values, status, _ = observe(element, particles,
                            shifted_z, profile_at(shifted_z), 0.)
                        assert status[0] == 1
                        raw_E, raw_B = values[0, 12:15], values[0, 15:18]
                        shifted_gamma = np.hypot(
                            1., np.linalg.norm(shifted_u))
                        shifted_force = charge*(raw_E+np.cross(
                            shifted_u/shifted_gamma, raw_B))
                        shifted_k.append(np.cross(
                            shifted_gamma*shifted_force, shifted_u)
                            /(shifted_gamma+1))
                    derivatives.append((shifted_k[0]-shifted_k[1])/(2*h))
                D0k = (4*derivatives[1]-derivatives[0])/3
                np.testing.assert_allclose(derivatives[1], D0k,
                                           rtol=3e-7, atol=3e-10)
                _, omega0 = point_oracle(
                    u, charge, 0., anomaly, E, B, DE, DB)
                # Unit formal rho extracts an exactly linear covariant spin
                # coefficient; it does not represent a physical LL launch.
                _, omega1 = point_oracle(
                    u, charge, 1., anomaly, E, B, DE, DB)
                expected = (clock*(omega1-omega0)
                    - clock*(charge*k)*(beta@(charge*E))/(gamma+1))
                np.testing.assert_allclose(D0k, expected,
                                           rtol=3e-7, atol=3e-10)
        for charge, data in ((0., profile), (1., np.zeros(10))):
            q = canonical_event(point, u, charge, data)
            value, status, _ = spin_schott_source_query(
                context, q, charge, .001, data)
            assert status == 1
            np.testing.assert_array_equal(value, 0.)
        q = canonical_event(point, u, -1., profile)
        low_energy = q.copy()
        low_energy[5] = .9
        invalid_profile = profile.copy()
        invalid_profile[4] = np.nan
        for state, data, anomaly, aperture, expected in (
            (low_energy, profile, .001, .1, -1),
            (q, invalid_profile, .001, .1, -3),
            (q, profile, np.nan, .1, -4),
            (q, profile, .001, -1., -4),
        ):
            value, status, sentinel = spin_schott_source_query(
                context, state, -1., anomaly, data, aperture=aperture)
            assert status == expected
            np.testing.assert_array_equal(value, sentinel)
