"""Presentation must follow the selected mechanical history without applying it."""

import numpy as np
import pytest

import xtrack as xt


class HeldInput:
    frame = 'body_local'
    unit = 'm'

    def __init__(self):
        self.calls = 0

    def evaluate(self, time_s, positions_m):
        self.calls += 1
        return np.full((len(time_s), len(positions_m)), float(self.calls))


@pytest.fixture
def model():
    plt = pytest.importorskip('matplotlib.pyplot')
    env = xt.Environment()
    mech = env.mechanics
    root = mech.new_girder('root', anchor=xt.Frame())
    mech.new_body('child', parent=root, mount=xt.Frame().translate_s(2.))
    mech.new_girder('quiet', anchor=xt.Frame())
    held = HeldInput()
    mech.add_source('mover', held, targets='root', live=True, component='y')
    mech.add_source('tone', xt.mechanics.Harmonic(.1, 1.), targets='quiet',
                    component='x')
    yield mech, held
    plt.close('all')


def test_live_pose_is_labelled_and_animation_rejects_live_ancestor(model):
    mech, held = model
    with pytest.raises(ValueError, match='history of live mechanics inputs: mover'):
        mech.animate(bodies=['child'], time_s=[0., .25])
    assert held.calls == 0

    fig, (layout, _) = mech.plot(bodies=['child'], time_s=.25)
    assert 'Current held live inputs' in layout.get_title()
    assert held.calls == 1


@pytest.mark.parametrize('bodies, disabled_sources', [
    (['quiet'], ()),
    (['child'], ('mover',)),
])
def test_animation_selects_bodies_once_without_reading_unneeded_live_inputs(
    model, bodies, disabled_sources,
):
    mech, held = model
    animation = mech.animate(bodies=iter(bodies), time_s=[0., .25],
                             disabled_sources=disabled_sources)
    animation._init_draw()
    animation._draw_frame(1)
    displayed = np.array(animation._fig.axes[0].collections[-1]._offsets3d).T
    expected = mech.survey(bodies=bodies, time_s=.25,
                           disabled_sources=disabled_sources).total_matrix[:, :3, 3]
    np.testing.assert_allclose(displayed, expected[:, [0, 2, 1]])
    assert held.calls == 0


@pytest.mark.parametrize('kwargs', [
    {'time_s': [0., .25j]},
    {'interval_ms': 0.},
    {'interval_ms': np.nan},
])
def test_animation_rejects_invalid_presentation_inputs(model, kwargs):
    mech, held = model
    with pytest.raises(ValueError, match='animation frame times|frame interval'):
        mech.animate(**({'bodies': ['quiet'], 'time_s': [0., .25]} | kwargs))
    assert held.calls == 0
