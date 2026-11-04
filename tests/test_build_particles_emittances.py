import numpy as np
import pytest

import xtrack as xt


@pytest.mark.parametrize('at_element', [None, 'd1'])
def test_build_particles_geometric_emittances_with_twiss(at_element):
    line = xt.Line(elements=[xt.Drift(length=1), xt.Drift(length=2)],
                   element_names=['d0', 'd1'])
    line.particle_ref = xt.Particles(p0c=2e9)
    beta_gamma = line.particle_ref.beta0[0] * line.particle_ref.gamma0[0]
    coordinates = ['x', 'px', 'y', 'py', 'zeta', 'pzeta']
    normalized = {f'{name}_norm': [0.01, -0.02] for name in coordinates}
    gemitt = dict(gemitt_x=4e-6, gemitt_y=9e-6, gemitt_zeta=1e-4)
    nemitt = {key.replace('gemitt', 'nemitt'): value * beta_gamma
              for key, value in gemitt.items()}
    options = dict(method='4d', betx=2, bety=3, at_element=at_element)
    geometric = line.build_particles(**normalized, **gemitt, **options)
    equivalent = line.build_particles(**normalized, **nemitt, **options)

    for name in coordinates:
        np.testing.assert_allclose(getattr(geometric, name), getattr(equivalent, name),
                                   rtol=1e-12, atol=1e-14)
    recovered = line.twiss4d(betx=2, bety=3).get_normalized_coordinates(
        geometric, **nemitt)
    for name, values in normalized.items():
        np.testing.assert_allclose(recovered[name], values, rtol=1e-10, atol=1e-12)


def test_unscaled_courant_snyder_coordinates():
    line = xt.Line(elements = [xt.Drift(length = 1.)])
    line.set_particle_ref('proton', p0c = 2e9)
    launch = dict(
        x_norm  = np.array([1e-4, -2e-4]),
        px_norm = np.array([2e-4, -1e-4]),
        y_norm  = np.array([3e-4, -4e-4]),
        py_norm = np.array([0., 5e-5]),
    )
    optics = dict(method = '4d', betx = 4., bety = 9., alfx = 1.2, alfy = -0.5)
    implicit = line.build_particles(**launch, **optics)
    explicit = line.build_particles(
        **launch,
        **optics,
        gemitt_x    = 1.,
        gemitt_y    = 1.,
    )

    # Unit geometric scaling gives the unscaled Courant-Snyder transform,
    # independently of the reference beta*gamma. Radians are dimensionless.
    np.testing.assert_allclose(
        implicit.x, 2 * launch['x_norm'],
        rtol    = 1e-12,
        atol    = 1e-14,
    )
    np.testing.assert_allclose(
        implicit.px, (launch['px_norm'] - 1.2 * launch['x_norm']) / 2,
        rtol    = 1e-12,
        atol    = 1e-14,
    )
    np.testing.assert_allclose(
        implicit.y, 3 * launch['y_norm'],
        rtol    = 1e-12,
        atol    = 1e-14,
    )
    np.testing.assert_allclose(
        implicit.py, (launch['py_norm'] + 0.5 * launch['y_norm']) / 3,
        rtol    = 1e-12,
        atol    = 1e-14,
    )
    for coordinate in ['x', 'px', 'y', 'py']:
        np.testing.assert_allclose(
            getattr(implicit, coordinate), getattr(explicit, coordinate),
            rtol    = 1e-12,
            atol    = 1e-14,
        )
    recovered = line.twiss(**optics).get_normalized_coordinates(implicit)
    for coordinate, expected in launch.items():
        np.testing.assert_allclose(
            recovered[coordinate], expected,
            rtol    = 1e-12,
            atol    = 1e-14,
        )
