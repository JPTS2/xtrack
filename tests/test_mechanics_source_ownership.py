"""Regional composition owns its output, independent of provider buffer policy."""

import numpy as np
import pytest

import xtrack as xt


class CachedScalarSource:
    frame = 'body_local'
    unit = 'm'

    def __init__(self, values, *, readonly=False):
        self.values = np.array(values, copy=True)
        self.values.flags.writeable = not readonly

    def evaluate(self, time_s, positions_m):
        return self.values


@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.int64, np.uint64])
@pytest.mark.parametrize('readonly', [False, True])
def test_regional_accumulation_does_not_mutate_cached_background(dtype, readonly):
    times = np.array([0., 1.])
    positions = np.array([[0., 0., 0.], [1., 0., 0.]])
    values = np.arange(4).reshape(2, 2).astype(dtype)
    background = CachedScalarSource(values, readonly=readonly)
    excess = CachedScalarSource(np.full((2, 2), .25), readonly=True)
    region = xt.mechanics.GaussianRegion(excess, (0., 0., 0.), 1.)
    source = xt.mechanics.RegionalGroundMotion(background, (region,))
    expected = values.astype(float) + .25*np.exp(-np.array([0., 1.])/2)

    first = source.evaluate(times, positions)
    second = source.evaluate(times, positions)
    np.testing.assert_allclose(first, expected, rtol=0, atol=1e-15)
    np.testing.assert_array_equal(second, first)
    np.testing.assert_array_equal(background.values, values)
    np.testing.assert_array_equal(excess.values, np.full((2, 2), .25))
    assert not np.shares_memory(first, background.values)
    assert not np.shares_memory(first, second)

    # Mutating a consumer's result must not poison either provider or next query.
    first[:] = 99.
    np.testing.assert_array_equal(source.evaluate(times, positions), second)


def test_regional_without_excess_still_returns_owned_samples():
    background = CachedScalarSource(np.ones((2, 1)), readonly=True)
    source = xt.mechanics.RegionalGroundMotion(background, ())
    output = source.evaluate(np.array([0., 1.]), np.zeros((1, 3)))
    output[:] = 0.
    np.testing.assert_array_equal(background.values, 1.)


@pytest.mark.parametrize('invalid, message', [
    (np.ones((2, 1), dtype=complex)*(1+1j), 'real'),
    (np.full((2, 1), np.nan), 'finite'),
    (np.ones((2, 1, 1)), 'shape'),
])
def test_regional_copy_does_not_hide_invalid_provider_output(invalid, message):
    source = xt.mechanics.RegionalGroundMotion(CachedScalarSource(invalid), ())
    with pytest.raises(ValueError, match=message):
        xt.mechanics.sample_field(source, time_s=[0., 1.], positions_m=[[0., 0., 0.]])


def test_nested_builtin_regions_preserve_background_and_field_queries():
    times = np.arange(33)/128
    positions = np.array([[0., 0., 0.], [1., 2., 3.], [10., 0., 0.]])
    background = xt.mechanics.SpectralGroundMotion(
        [0., 1., 2., 10., 12.], [0., 0., 1e-18, 1e-18, 0.],
        seed=956, correlation_length_m=20., n_modes=32)
    inner = xt.mechanics.RegionalGroundMotion(background, (
        xt.mechanics.GaussianRegion(xt.mechanics.Harmonic(2e-9, 3.), (0., 0., 0.), 2.),))
    outer = xt.mechanics.RegionalGroundMotion(inner, (
        xt.mechanics.GaussianRegion(xt.mechanics.Harmonic(1e-9, 5.), (2., 0., 0.), 3.),))
    background_before = background.evaluate(times, positions)
    inner_before = inner.evaluate(times, positions)
    whole = outer.evaluate(times, positions)
    np.testing.assert_array_equal(outer.evaluate(times, positions), whole)
    np.testing.assert_array_equal(background.evaluate(times, positions), background_before)
    np.testing.assert_array_equal(inner.evaluate(times, positions), inner_before)
    chunks = np.concatenate([outer.evaluate(times[:13], positions),
                             outer.evaluate(times[13:], positions)])
    np.testing.assert_allclose(chunks, whole, rtol=0, atol=1e-23)
    subset = outer.evaluate(times, positions[[2, 0]])
    np.testing.assert_allclose(subset, whole[:, [2, 0]], rtol=0, atol=1e-23)


def test_regional_definition_consumes_iterables_only_once():
    """Building regions from an iterator must not silently remove their motion."""
    background = xt.mechanics.Harmonic(1., 0., phase_rad=np.pi/2)
    excess = xt.mechanics.Harmonic(2., 0., phase_rad=np.pi/2)
    region = xt.mechanics.GaussianRegion(excess, (0., 0., 0.), 1.)
    source = xt.mechanics.RegionalGroundMotion(background, iter([region]))
    assert source.regions == (region,)
    np.testing.assert_array_equal(source.evaluate(np.array([0., 1.]), np.zeros((1, 3))), 3.)


class SharedScalarSource:
    frame = 'body_local'
    unit = 'm'

    def evaluate(self, *, time_s, positions_m):
        return 1. + 2.*time_s


@pytest.mark.parametrize('n_times', [2, 3])
def test_shared_scalar_waveform_preserves_time_axis_in_source_compositions(n_times):
    """A shared time history must never broadcast along the receiver axis."""
    times = np.arange(n_times, dtype=float)
    positions = np.array([[0., 0., 0.], [1., 0., 0.]])
    waveform = SharedScalarSource()
    shared = np.broadcast_to((1. + 2.*times)[:, None], (n_times, 2))

    field = xt.mechanics.sample_field(waveform, time_s=times, positions_m=positions)
    vector = xt.mechanics.VectorGroundMotion((None, waveform, None))
    np.testing.assert_array_equal(field.values[..., 1], shared)
    np.testing.assert_array_equal(vector.evaluate(times, positions)[..., 1], shared)

    region = xt.mechanics.GaussianRegion(waveform, (0., 0., 0.), 1.)
    regional = xt.mechanics.RegionalGroundMotion(waveform, (region,))
    expected = shared * (1. + np.exp(-np.array([0., 1.])/2))
    np.testing.assert_allclose(regional.evaluate(times, positions), expected, rtol=1e-15)

    located = xt.mechanics.LocatedSource(waveform, (0., 0., 0.), 2., 4.)
    distance = np.array([0., 1.])
    expected = (1. + 2.*(times[:, None] - distance/4.)) * np.exp(-distance/2.)
    np.testing.assert_allclose(located.evaluate(times, positions)[..., 1], expected, rtol=1e-15)


@pytest.mark.parametrize('wrapper', ['vector', 'regional_excess', 'located'])
@pytest.mark.parametrize('invalid, message', [
    (np.array([[1.+1j]]), 'real'),
    (np.array([[np.inf]]), 'finite'),
    (np.ones((1, 1)), 'shape'),
])
def test_scalar_source_wrappers_reject_invalid_samples_before_composition(wrapper, invalid, message):
    """Neither a zero mixing weight nor broadcasting can hide invalid input."""
    provider = CachedScalarSource(invalid)
    if wrapper == 'vector':
        source = xt.mechanics.VectorGroundMotion((provider, None, None), mixing=np.zeros((3, 3)))
    elif wrapper == 'regional_excess':
        region = xt.mechanics.GaussianRegion(provider, (0., 0., 0.), 1.)
        source = xt.mechanics.RegionalGroundMotion(SharedScalarSource(), (region,))
    else:
        source = xt.mechanics.LocatedSource(provider, (0., 0., 0.), 2., 4.)
    with pytest.raises(ValueError, match=message):
        source.evaluate(np.array([0., 1.]), np.zeros((2, 3)))
