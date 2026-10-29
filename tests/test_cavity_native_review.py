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
    '''], kernels={'cavity_schott_source_review': xo.Kernel(args=[
        xo.Arg(xo.Float64, pointer=True, name='q'),
        xo.Arg(xo.Float64, name='charge'),
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
