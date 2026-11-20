"""Distinguish physical ownership from collective execution/survey views."""
import numpy as np
import pytest
import xtrack as xt


class CollectiveCallback:
    iscollective = True

    def __init__(self, change_geometry=False):
        self.change_geometry = change_geometry

    def track(self, particles):
        if self.change_geometry:
            self.line['q'].length += .01


def make_line(change_geometry=False):
    callback = CollectiveCallback(change_geometry)
    line = xt.Line(elements={'q': xt.Quadrupole(length=.5, k1=.1), 'callback': callback})
    callback.line = line
    line.set_particle_ref('proton', p0c=7e12)
    line.build_tracker()
    return line


def test_collective_survey_views_do_not_duplicate_physical_ownership():
    line = make_line()
    # Retain views deliberately: correctness cannot depend on garbage collection.
    views = [line._get_non_collective_line() for _ in range(3)]
    assert all(view.env is not line.env for view in views)
    line.mechanics.new_girder('g', elements=['q'])
    line.mechanics.enable_vibrations()
    line.track(line.build_particles(), num_turns=2)
    assert line.mechanics._runtime is None
    other = line.env.new_line(name='another_physical_line', components=['q'])
    assert other.env is line.env
    with pytest.raises(ValueError, match='shared/repeated'):
        line.mechanics.apply()


def test_auto_collective_callbacks_keep_per_turn_geometry_validation():
    line = make_line(change_geometry=True)
    line.mechanics.new_girder('g', elements=['q'])
    line.mechanics.enable_vibrations()
    with pytest.raises(ValueError, match='reference geometry changed'):
        line.track(line.build_particles(), num_turns=2)
    assert line.mechanics._runtime is None


def test_explicit_compiled_collective_mode_accepts_fixed_geometry():
    line = make_line()
    line.mechanics.new_girder('g', elements=['q'])
    line.env.mechanics.add_source('h', xt.mechanics.Harmonic(1e-9, 3., np.pi/2), targets='g')
    line.mechanics.enable_vibrations(backend='compiled')
    line.track(line.build_particles(), num_turns=2)
    assert line.mechanics._runtime is not None
    assert line['q'].shift_y == pytest.approx(1e-9, rel=1e-12)
    line.mechanics.disable_vibrations()
    assert line['q'].shift_y == 0.
