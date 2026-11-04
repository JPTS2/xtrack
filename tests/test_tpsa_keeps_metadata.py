"""TPSA tracking keeps a particle's metadata like scalar tracking, and refuses an element
field on another descriptor."""
import madng_tpsa
import numpy as np
import pytest
import xtrack as xt
import xtrack.tpsa as xtpsa


@pytest.mark.parametrize('calls', [1, 2], ids=['one call', 'split into two calls'])
def test_map_snapshot_matches_scalar_tracking_through_a_reference_change(calls):
    line = xt.Line(elements=[xt.Drift(length=.2), xt.ReferenceEnergyIncrease(Delta_p0c=1e8), xt.Drift(length=.2)])
    line.particle_ref = xt.Particles(p0c=1e9)
    line.reset_s_at_end_turn = False
    line.build_tracker()
    launch = dict(p0c=1e9, x=1e-4, px=2e-5, delta=1e-3, s=.3, ax=1e-5, ay=-2e-5)
    scalar = xt.Particles(**launch)
    line.track(scalar)
    series = xtpsa.ParticlesTpsa(order=1, **launch)
    if calls == 1:
        line.track(series)
    else:
        line.track(series, ele_start=0, ele_stop=1)
        line.track(series, ele_start=1)
    snapshot = series.to_particles()
    for name in ('p0c', 'delta', 'px', 'x', 's', 'ax', 'ay'):
        np.testing.assert_allclose(getattr(snapshot, name)[0], getattr(scalar, name)[0], rtol=1e-14)


def test_field_on_another_descriptor_is_refused():
    element = xt.Quadrupole(length=1., k1=.1)
    element.k1 = madng_tpsa.Descriptor(6, 2, num_params=1, param_order=1).param(1, .1)
    particles = xtpsa.ParticlesTpsa(order=2, descriptor=madng_tpsa.Descriptor(6, 2, num_params=2, param_order=1),
                                    p0c=1e9, x=1e-3)
    with pytest.raises(RuntimeError, match='TPSA map lost'):
        element.track(particles)
    element.disable_tpsa()
