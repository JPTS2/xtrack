"""The shared clock and everything that reads it.

Ground truths are independent of the clock's own formulas: the arrival time of a
constant-speed particle, a reference particle returning to zeta = 0 on every lap,
analytic fixed points of a detuned ring, and one tracking call against many.
"""
import numpy as np
import pytest
import xobjects as xo
import xtrack as xt
from scipy.constants import c as clight
from scipy.integrate import quad
from xobjects.test_helpers import for_all_test_contexts

_COORDS = ('x', 'px', 'y', 'py', 'zeta', 'delta')
_FREQUENCY = 4e8
_RF = dict(frequency=_FREQUENCY, absolute_time=True)
_MULTIPOLE = dict(knl=[1e-3, 2e-2], ksl=[1e-4, 3e-3], phase_n=[.1, .2], phase_s=[.3, .4],
                  voltage=1e5, phase=.2)
_FAMILIES = {
    'cavity': (xt.Cavity, dict(_RF, voltage=1e6, phase=.2, model='drift-kick-drift-expanded')),
    'crab': (xt.CrabCavity, dict(_RF, crab_voltage=1e6, phase=.2)),
    'multipole': (xt.RFMultipole, dict(_RF, **_MULTIPOLE)),
}
_PHYSICAL = ('_phase-resolved', '_phase-averaged', '_phase-resolved-paraxial')


def _launch(context, time_s=0., s=.3):
    return xt.Particles(_context=context, p0c=1e10, x=1e-3, px=1e-4, y=-2e-3, py=2e-5,
                        zeta=.01, delta=1e-4, s=s, time_s=time_s)


def _line(context, element, slicing=None):
    line = xt.Line(elements={'rf': element})
    line.particle_ref = xt.Particles(p0c=1e10)
    if slicing is not None:
        line.slice_thick_elements([xt.Strategy(slicing)])
    line.build_tracker(_context=context)
    return line


def _end(line, context, time_s=0., s=.3):
    probe = _launch(context, time_s, s)
    line.track(probe)
    probe.move(_context=xo.context_default)
    assert probe.state[0] == 1
    return np.array([getattr(probe, name)[0] for name in _COORDS])


def _assert_follows_the_clock(line, context):
    start = _end(line, context, time_s=1e-6)
    # A whole number of periods is the same phase, a quarter period is not.
    xo.assert_allclose(_end(line, context, time_s=1e-6 + 1000 / _FREQUENCY), start,
                       rtol=0, atol=2e-8)
    assert np.abs(_end(line, context, time_s=1e-6 + .25 / _FREQUENCY) - start).max() > 1e-7


@for_all_test_contexts
@pytest.mark.parametrize('layout', ['whole thin', 'whole thick', 'thin slice', 'thick slice'])
@pytest.mark.parametrize('family', _FAMILIES)
def test_legacy_elements_and_slices_follow_the_clock(test_context, family, layout):
    cls, controls = _FAMILIES[family]
    controls = dict(controls)
    if layout != 'whole thin':
        controls.update(length=.5, num_kicks=1)
    slicing = {'thin slice': xt.Teapot(1), 'thick slice': xt.Teapot(1, mode='thick')}.get(layout)
    line = _line(test_context, cls(**controls), slicing)
    _assert_follows_the_clock(line, test_context)
    if slicing is not None:
        # One kick at the centre, with the phase of that position.
        whole = _line(test_context, cls(**controls))
        xo.assert_allclose(_end(line, test_context, 1e-6), _end(whole, test_context, 1e-6),
                           rtol=0, atol=1e-13)


@for_all_test_contexts
@pytest.mark.parametrize('layout', ['whole', 'thick slices', 'thin layout'])
@pytest.mark.parametrize('model', _PHYSICAL)
def test_physical_models_and_slices_follow_the_clock(test_context, model, layout):
    controls = dict(_RF, voltage=1e6, phase=.2, length=.5, model=model, num_kicks=12)
    slicing = {'thick slices': xt.Teapot(3, mode='thick'), 'thin layout': xt.Teapot(3)}.get(layout)
    line = _line(test_context, xt.Cavity(**controls), slicing)
    _assert_follows_the_clock(line, test_context)
    if slicing is not None:
        whole = _line(test_context, xt.Cavity(**controls))
        xo.assert_allclose(_end(line, test_context, 1e-6), _end(whole, test_context, 1e-6),
                           rtol=0, atol=1e-9)


@for_all_test_contexts
def test_multipole_thin_slice_equivalent_element_keeps_the_clock(test_context):
    parent = xt.RFMultipole(**dict(_RF, **_MULTIPOLE, length=.5, num_kicks=1))
    line = _line(test_context, parent, xt.Teapot(1))
    thin = [line[nn] for nn in line.element_names if isinstance(line[nn], xt.ThinSliceRFMultipole)]
    assert thin[0].get_equivalent_element().absolute_time == 1


def _reference_phase(line, cavity, particles, zeta, s, start, stop):
    energies = []
    for phase in (0., np.pi / 2):
        cavity.phase = phase
        probe = particles.copy()
        probe.zeta = zeta
        probe.s = s
        before = probe.copy(_context=xo.context_default).energy[0]
        line.track(probe, ele_start=start, ele_stop=stop)
        probe.move(_context=xo.context_default)
        energies.append(probe.energy[0] - before)
    return np.arctan2(energies[0], energies[1])


@for_all_test_contexts
@pytest.mark.parametrize('model', ['drift-kick-drift-expanded', '_phase-resolved', '_phase-averaged'])
def test_locked_phase_is_the_integral_of_the_frequency_through_an_energy_program(test_context, model):
    # The truth is 2 pi h int beta c / L dt from the program, not the clock's own count.
    # The physical maps use a cell short enough that their transit is negligible.
    circumference, harmonic, s_cavity, length = 628., 8, 100., 1e-3
    cavity = xt.Cavity(voltage=1., harmonic=harmonic, absolute_time=True, model=model, length=length)
    line = xt.Line(elements={'before': xt.Drift(length=s_cavity), 'rf': cavity, 'end': xt.Marker(),
                             'after': xt.Drift(length=circumference - s_cavity - length)})
    line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, kinetic_energy0=1.4e9)
    program = xt.EnergyProgram(t_s=[0., .01], kinetic_energy0=[1.4e9, 2e9])
    line.energy_program = program
    line.enable_time_dependent_vars = True
    line.build_tracker(_context=test_context)
    particles = line.build_particles(zeta=[0.])
    errors = []
    for _ in range(3):
        cavity.voltage = 0.
        line.track(particles, num_turns=500)
        cavity.voltage = 1.
        line.enable_time_dependent_vars = False
        state = particles.copy(_context=xo.context_default)
        for delay in (0., 1.2e-8):
            arrival = program.get_t_s_at_turn(state.reference_turns + s_cavity / circumference) + delay
            turns, _ = quad(lambda t: program.get_beta0_at_t_s(t) * clight / circumference,
                            0., arrival, limit=200, epsabs=1e-11, epsrel=1e-13)
            zeta = s_cavity - state.beta0[0] * clight * (arrival - state.time_s)
            phase = _reference_phase(line, cavity, particles, zeta, s_cavity, 'rf', 'end')
            errors.append(np.angle(np.exp(1j * (phase - 2 * np.pi * harmonic * turns))))
        line.enable_time_dependent_vars = True
    # What remains is the speed's change within one lap, 2 pi h (s/L) dbeta/beta.
    assert np.max(np.abs(errors)) < 2e-4


def _constant_speed_line(context, elements, time_based=True):
    line = xt.Line(elements=elements)
    line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, p0c=2e9)
    line.use_time_based_zeta_update = time_based
    line.build_tracker(_context=context)
    return line


def _arrival(particles):
    probe = particles.copy(_context=xo.context_default)
    return probe.time_s + (probe.s[0] - probe.zeta[0]) / (probe.beta0[0] * clight)


@for_all_test_contexts
@pytest.mark.parametrize('cls', [xt.TimeDelay, xt.ZetaShift])
def test_delay_elements_delay_the_arrival_and_not_the_clock(test_context, cls):
    shift = 1e-3
    field = 'shift_zeta' if cls is xt.TimeDelay else 'dzeta'
    line = _constant_speed_line(test_context, {
        'before': xt.Drift(length=137.), 'delay': cls(**{field: shift}), 'after': xt.Drift(length=491.)})
    particles = xt.Particles(_context=test_context, mass0=xt.PROTON_MASS_EV, p0c=2e9, zeta=0.)
    line.track(particles, num_turns=50)
    particles.move(_context=xo.context_default)
    beta0 = particles.beta0[0]
    xo.assert_allclose(_arrival(particles), 50 * (628. + shift) / (beta0 * clight), rtol=1e-11, atol=0)
    assert particles.reference_turns == 50.


@for_all_test_contexts
def test_delay_moves_a_clocked_and_a_locked_cavity_the_same_way(test_context):
    shift, voltage = 2e-3, 1e6
    for rf, rate in ((dict(frequency=_FREQUENCY), _FREQUENCY),
                     (dict(harmonic=8), None)):
        def gain(*before):
            line = xt.Line(elements=[*before, xt.Cavity(voltage=voltage, phase=.3, absolute_time=True, **rf),
                                     xt.Drift(length=628.)])
            line.particle_ref = xt.Particles(p0c=2e9)
            line.build_tracker(_context=test_context)
            probe = xt.Particles(_context=test_context, p0c=2e9, time_s=1e-6)
            start = probe.copy(_context=xo.context_default).energy[0]
            line.track(probe, ele_stop=len(before) + 1)
            probe.move(_context=xo.context_default)
            return probe.energy[0] - start, probe.beta0[0], probe.reference_turns
        undelayed = gain(xt.Drift(length=0.))
        delayed = gain(xt.TimeDelay(shift_zeta=shift))
        frequency = rate or 8 * undelayed[1] * clight / 628.
        # sin(a + d) from sin(a) and the phase a, known from the clock.
        turns = undelayed[2]
        phase = .3 + 2 * np.pi * (frequency * 1e-6 if rate else 8 * turns)
        xo.assert_allclose(undelayed[0], voltage * np.sin(phase), rtol=0, atol=1e-5)
        advance = 2 * np.pi * frequency * shift / (undelayed[1] * clight)
        xo.assert_allclose(delayed[0], voltage * np.sin(phase + advance), rtol=0, atol=1e-5)


def _ring_fixed_point(context, rf_elements, zeta=0.):
    alpha, length, p0c = 1e-3, 628., 2e9
    arc = xt.LineSegmentMap(length=length, qx=.1, qy=.2, momentum_compaction_factor=alpha,
                            longitudinal_mode='nonlinear', voltage_rf=[0.], frequency_rf=[0.],
                            phase_rf=[0.])
    line = xt.Line(elements=[arc, *rf_elements])
    line.particle_ref = xt.Particles(p0c=p0c, mass0=xt.PROTON_MASS_EV)
    line.build_tracker(_context=context)
    reference = line.particle_ref.copy()
    beta0, gamma0 = reference.beta0[0], reference.gamma0[0]
    eta = alpha - 1 / gamma0**2
    return line, beta0 * clight / length, eta


@for_all_test_contexts
@pytest.mark.parametrize('layout', ['absolute detuned', 'locked with time delay'])
def test_detuned_ring_reaches_the_analytic_momentum(test_context, layout):
    # A detuned RF and the equivalent per-turn delay both move the ring to dp = -df/(f eta).
    harmonic, df = 8, 20.
    _, frev, eta = _ring_fixed_point(test_context, [])
    frequency = harmonic * frev
    expected = -df / frequency / eta
    if layout == 'absolute detuned':
        rf = [xt.Cavity(voltage=1e5, frequency=frequency + df, absolute_time=True)]
    else:
        rf = [xt.Cavity(voltage=1e5, harmonic=harmonic, absolute_time=True),
              xt.TimeDelay(shift_zeta=628. * df / frequency)]
    line, _, _ = _ring_fixed_point(test_context, rf)
    particles = line.build_particles(zeta=[0.], delta=[expected])
    line.track(particles, num_turns=4000, turn_by_turn_monitor=True)
    record = line.record_last_track
    delta = line._context.nparray_from_context_array(record.delta)[0]
    xo.assert_allclose(delta.mean(), expected, rtol=.01, atol=0)
    assert particles.copy(_context=xo.context_default).state[0] == 1


def test_mixed_species_share_a_reference_and_a_clock():
    line = xt.Line(elements=[xt.Cavity(voltage=1e5, frequency=_FREQUENCY, absolute_time=True, phase=.3),
                             xt.Drift(length=1.)])
    line.build_tracker()
    particles = xt.Particles(mass0=xt.PROTON_MASS_EV, q0=1., p0c=2e9, mass_ratio=[1., 2.],
                             charge_ratio=[1., 1.], zeta=0., delta=[0., 0.])
    start = particles.energy.copy()
    line.track(particles, ele_stop=1)
    xo.assert_allclose((particles.energy - start)[0], (particles.energy - start)[1], rtol=1e-12)


@for_all_test_contexts(excluding='ContextPyopencl')
def test_synctime_clock_counts_the_same_revolutions_as_the_program(test_context):
    from xtrack import synctime as st
    line = xt.Line(elements={'d0': xt.Drift(length=25.), 'rf': xt.Cavity(voltage=0., harmonic=2,
                   absolute_time=True), 'd1': xt.Drift(length=75.)})
    line.particle_ref = xt.Particles(mass0=56 * xt.PROTON_MASS_EV, q0=25, p0c=20e9)
    times = np.linspace(0, 2e-4, 2001)
    line.energy_program = xt.EnergyProgram(t_s=times, p0c=20e9 * (1 + .15 * times / times[-1]))
    line.enable_time_dependent_vars = True
    st.install_sync_time_at_collective_elements(line)
    line.build_tracker(_context=test_context)
    particles = xt.Particles(_context=test_context, mass0=line.particle_ref.mass0,
                             q0=line.particle_ref.q0, p0c=20e9, zeta=[0.])
    line.track(particles, num_turns=30)
    xo.assert_allclose(line.energy_program.get_turn_at_t_s(particles.time_s),
                       particles.reference_turns, rtol=0, atol=1e-9)


@for_all_test_contexts
def test_line_segment_map_recipe_for_a_detuned_rf(test_context):
    # The map itself ignores the detuning; the documented recipe reaches the fixed point.
    _, frev, eta = _ring_fixed_point(test_context, [])
    df, frequency = 20., 8 * frev
    expected = -df / frequency / eta
    recipe, _, _ = _ring_fixed_point(test_context,
        [xt.Cavity(voltage=1e5, frequency=frequency + df, absolute_time=True)])
    native = xt.Line(elements=[xt.LineSegmentMap(length=628., qx=.1, qy=.2,
        momentum_compaction_factor=1e-3, longitudinal_mode='nonlinear', voltage_rf=[1e5],
        frequency_rf=[frequency + df], phase_rf=[0.])])
    native.particle_ref = xt.Particles(p0c=2e9, mass0=xt.PROTON_MASS_EV)
    native.build_tracker(_context=test_context)
    means = []
    for line in (recipe, native):
        particles = line.build_particles(zeta=[0.], delta=[expected])
        line.track(particles, num_turns=4000, turn_by_turn_monitor=True)
        means.append(line._context.nparray_from_context_array(line.record_last_track.delta)[0].mean())
    xo.assert_allclose(means[0], expected, rtol=.01, atol=0)
    assert abs(means[1]) < .1 * abs(expected)


def _exciter_line(context, knl=1e-6, **controls):
    element = xt.Exciter(_context=context, knl=[knl], samples=np.arange(4000.), **controls)
    return element


def _picked_sample(element, context, knl=1e-6, **launch):
    probe = xt.Particles(_context=context, p0c=1e10, **launch)
    element.track(probe)
    probe.move(_context=xo.context_default)
    return -probe.px[0] / knl, probe.beta0[0]


@for_all_test_contexts
def test_exciter_absolute_time_reads_the_arrival_time(test_context):
    fs, frev, start_turn = 1e9, 1e6, 2
    time_s, s0, zeta = start_turn / frev + 2e-7, 15., .3
    element = _exciter_line(test_context, sampling_frequency=fs, frev=frev, start_turn=start_turn,
                            absolute_time=True)
    index, beta0 = _picked_sample(element, test_context, s=s0, zeta=zeta, time_s=time_s,
                                  at_turn=start_turn)
    arrival = time_s - start_turn / frev + (s0 - zeta) / (beta0 * clight)
    assert np.round(index) == np.floor(fs * arrival)
    # With no start turn it needs no revolution frequency.
    free = _exciter_line(test_context, sampling_frequency=fs, absolute_time=True)
    assert np.round(_picked_sample(free, test_context, time_s=1e-6)[0]) == 1000


@for_all_test_contexts
def test_exciter_turn_clock_is_unchanged_in_its_own_domain(test_context):
    fs, frev = 1e9, 1e6
    element = _exciter_line(test_context, sampling_frequency=fs, frev=frev, start_turn=1)
    # Samples follow the turn and zeta, whatever time_s and s say.
    index, beta0 = _picked_sample(element, test_context, at_turn=2, zeta=-.03, time_s=5., s=9.)
    assert np.round(index) == np.floor(fs * (1 / frev + .03 / (beta0 * clight)))


@for_all_test_contexts
@pytest.mark.parametrize('absolute_time', [False, True])
def test_exciter_gives_no_kick_before_its_first_sample(test_context, absolute_time):
    fs = 1e9
    element = xt.Exciter(_context=test_context, knl=[1e-6], samples=np.ones(10), sampling_frequency=fs,
                         frev=1e6, absolute_time=absolute_time)
    beta0 = xt.Particles(p0c=1e10).beta0[0]
    # Arrives 0.8 of a sample before the start, and just after it.
    early = xt.Particles(_context=test_context, p0c=1e10, zeta=.8 / fs * beta0 * clight)
    element.track(early)
    late = xt.Particles(_context=test_context, p0c=1e10, zeta=-.2 / fs * beta0 * clight)
    element.track(late)
    early.move(_context=xo.context_default)
    late.move(_context=xo.context_default)
    assert early.px[0] == 0.
    assert late.px[0] != 0.


def test_exciter_turn_clock_needs_a_revolution_frequency():
    with pytest.raises(ValueError, match='frev'):
        xt.Exciter(knl=[1e-6], samples=np.ones(10), sampling_frequency=1e6)
    # The time clock needs none.
    xt.Exciter(knl=[1e-6], samples=np.ones(10), sampling_frequency=1e6, absolute_time=True)


@for_all_test_contexts
def test_exciter_time_clock_keeps_lab_time_through_a_ramp(test_context):
    fs, turns = 1e7, 2000
    kinetic = dict(t_s=[0., .01], kinetic_energy0=[1.4e9, 2e9])
    frev = xt.Particles(mass0=xt.PROTON_MASS_EV, kinetic_energy0=1.4e9).beta0[0] * clight / 628.
    picks = {}
    for absolute_time in (False, True):
        element = xt.Exciter(knl=[0.], samples=np.arange(int(1e6), dtype=float),
                             sampling_frequency=fs, frev=frev, absolute_time=absolute_time)
        line = xt.Line(elements={'a': xt.Drift(length=100.), 'ex': element, 'b': xt.Drift(length=528.)})
        line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, kinetic_energy0=1.4e9)
        line.energy_program = xt.EnergyProgram(**kinetic)
        line.enable_time_dependent_vars = True
        line.build_tracker(_context=test_context)
        particles = line.build_particles(zeta=[0.])
        line.track(particles, num_turns=turns)
        line.enable_time_dependent_vars = False
        element.knl[0] = 1e-9
        probe = particles.copy()
        probe.zeta = 0.
        probe.s = 100.
        px = probe.copy(_context=xo.context_default).px[0]
        line.track(probe, ele_start='ex', ele_stop='b')
        probe.move(_context=xo.context_default)
        picks[absolute_time] = (px - probe.px[0]) / 1e-9
        lab = fs * (probe.time_s + 100. / (probe.beta0[0] * clight))
    assert abs(picks[True] - lab) <= 1.
    # One frev cannot describe the ramp: the turn clock is many samples away.
    assert abs(picks[False] - lab) > 100.
