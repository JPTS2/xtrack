"""Mechanical source phase, restart and long-run clock precision."""
import math

import numpy as np
import pytest
import xobjects as xo

import xtrack as xt
from xtrack.composer.positions import CompensatedFloat


def make_clock_line():
    env = xt.Environment()
    env.new('d',xt.Drift,length=90639.69986590669)
    env.new('q',xt.Quadrupole,length=.5,k1=0.)
    line = env.new_line(components=['d','q'])
    line.set_particle_ref('positron',p0c=45.6e9)
    line.build_tracker(_context=xo.ContextCpu())
    line.mechanics.new_girder('g',elements=['q'])
    env.mechanics.add_source('clock_probe',xt.mechanics.Harmonic(1e-7,1000.,.4),
                            targets='g',component='x')
    line.mechanics.enable_vibrations()
    return line


def test_compensated_fcc_clock_preserves_kilohertz_phase_over_one_million_turns():
    step = .000302343162587492
    count = 1000000
    clock, ordinary = CompensatedFloat(), 0.
    for _ in range(count):
        clock = clock+step
        ordinary += step
    expected = count*step
    phase_error = 2*np.pi*1000.*(float(clock)-expected)
    ordinary_phase_error = 2*np.pi*1000.*(ordinary-expected)
    assert abs(phase_error) < 1e-10
    assert abs(ordinary_phase_error) > 1e-5
    assert float(clock) == expected


def test_prepared_clock_and_mechanical_phase_are_chunk_invariant():
    line = make_clock_line()
    whole = line.build_particles()
    line.track(whole,num_turns=97)
    whole_value = line['q'].shift_x
    divided = line.build_particles()
    for count in (17,31,49):
        line.track(divided,num_turns=count)
    step = whole.t_sim
    assert whole.time_s == divided.time_s == 97*step
    assert whole.reference_turns == divided.reference_turns == 97.
    assert line['q'].shift_x == whole_value
    expected_source = 1e-7*math.sin(2*np.pi*1000.*(96*step)+.4)
    assert whole_value == pytest.approx(expected_source,rel=1e-14,abs=1e-23)
    runtime = line.mechanics._runtime
    assert isinstance(runtime._clocks[whole][0],CompensatedFloat)
    assert runtime._clocks[whole][0].lo != 0.
    assert runtime._clocks[whole][0] == runtime._clocks[divided][0]


def test_prepared_clock_external_time_edit_starts_a_new_epoch():
    line = make_clock_line()
    particle = line.build_particles()
    line.track(particle,num_turns=7)
    origin = .123456789
    particle.time_s = origin
    line.track(particle,num_turns=3)
    step = particle.t_sim
    expected = CompensatedFloat(origin)
    for _ in range(3):
        expected = expected+step
    assert particle.time_s == float(expected)
    expected_source = 1e-7*math.sin(2*np.pi*1000.*float(CompensatedFloat(origin)+step+step)+.4)
    assert line['q'].shift_x == pytest.approx(expected_source,rel=1e-14,abs=1e-23)


def test_prepared_clock_period_change_keeps_published_physical_origin():
    line = make_clock_line()
    particle = line.build_particles()
    line.track(particle,num_turns=7)
    origin, previous_step = particle.time_s, particle.t_sim
    particle.update_p0c(np.array([3e9]))
    line.track(particle,num_turns=3)
    assert particle.t_sim != previous_step
    expected = CompensatedFloat(origin)
    for _ in range(3):
        expected = expected+particle.t_sim
    assert particle.time_s == float(expected)
    assert line.mechanics._runtime._clocks[particle][2] == particle.t_sim
