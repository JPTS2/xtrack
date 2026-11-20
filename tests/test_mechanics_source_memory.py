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
