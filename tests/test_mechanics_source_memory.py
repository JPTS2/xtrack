"""Source-specific budgets must distinguish finite modes from dense PSDs."""
import numpy as np
import pytest
import xtrack as xt


def test_large_harmonic_record_has_no_quadratic_covariance_charge():
    env = xt.Environment()
    for i in range(3000):
        env.mechanics.new_girder(f'g{i}', anchor=(0., 0., float(i)))
    env.mechanics.add_source('h', xt.mechanics.Harmonic(1e-9, 3.),
                             targets=tuple(env.mechanics.bodies))
    record = env.mechanics.compute_time_series(time_s=[0., .01], max_memory_bytes=2*1024**2)
    assert record.values['h'].shape == (2, 3000, 6)


def test_uncorrelated_psd_is_linear_but_correlated_reference_is_budgeted():
    positions = np.zeros((500, 3))
    times = np.arange(8)*.1
    independent = xt.mechanics.DisplacementPSD([0., 1., 2.], [0., 1e-18, 0.], seed=956)
    field = xt.mechanics.sample_field(independent, time_s=times, positions_m=positions,
                                      max_memory_bytes=1024**2)
    assert field.values.shape == (8, 500, 3)
    dense = xt.mechanics.DisplacementPSD([0., 1., 2.], [0., 1e-18, 0.], seed=956,
                                        correlation_length_m=30.)
    with pytest.raises(ValueError, match='workspace'):
        xt.mechanics.sample_field(dense, time_s=times, positions_m=positions,
                                  max_memory_bytes=1024**2)


def test_finite_modes_use_their_own_workspace_estimate():
    source = xt.mechanics.SpectralGroundMotion([0., 1., 2.], [0., 1e-18, 0.],
        seed=956, correlation_length_m=30., n_modes=16)
    field = xt.mechanics.sample_field(source, time_s=[0., .01],
        positions_m=np.zeros((1000, 3)), max_memory_bytes=1024**2)
    assert field.values.shape == (2, 1000, 3)


def test_uncached_spectral_receivers_use_bounded_blocks_without_changing_field(monkeypatch):
    from xtrack.mechanics import sources

    source = xt.mechanics.SpectralGroundMotion(
        [0., 1., 20., 30.], [0., 1e-18, 1e-18, 0.],
        seed=956, correlation_length_m=30., n_modes=64,
    )
    positions = np.random.default_rng(2).normal(0., 2e4, (103, 3))
    times = np.array([0., .013, .57])
    expected = source._prepare_receivers(positions).evaluate(times, positions)
    # Enough for seven receivers and the frequency vector.
    scratch_bytes = 8*source.n_modes*(2*7+1)
    monkeypatch.setattr(sources, '_SPECTRAL_RECEIVER_SCRATCH_BYTES', scratch_bytes)
    prepared = []
    original = xt.mechanics.SpectralGroundMotion._prepare_receivers
    def capture(self, receivers):
        assert 8*self.n_modes*(2*len(receivers)+1) <= scratch_bytes
        prepared.append(receivers.copy())
        return original(self, receivers)
    monkeypatch.setattr(xt.mechanics.SpectralGroundMotion, '_prepare_receivers', capture)
    actual = source.evaluate(times, positions)
    np.testing.assert_array_equal(np.concatenate(prepared), positions)
    np.testing.assert_allclose(actual, expected, atol=1e-22, rtol=2e-14)
    # Continuous latent phase is tied to receiver position, not chunk size,
    # selection or ordering. A different slice of receivers gives the same field.
    indices = np.arange(2, 103, 3)[::-1]
    np.testing.assert_allclose(source.evaluate(times, positions[indices]),
                               expected[:, indices], atol=1e-22, rtol=2e-14)
