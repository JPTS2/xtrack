"""Particle metadata conventions shared by scalar and TPSA tracking."""
import numpy as np
import xtrack as xt
import xtrack.tpsa as xtpsa


def test_p0c_change_scales_vector_potential_with_momenta():
    """ax/ay are q*A/p0, normalized like px/py, so kinetic px-ax is kept."""
    p0c = 10e9
    def launch():
        return xt.Particles(p0c=p0c, px=1e-3, py=-2e-3, ax=4e-4, ay=-5e-4)
    kernel = launch()
    xt.ReferenceEnergyIncrease(Delta_p0c=p0c).track(kernel)
    python = launch()
    python.update_p0c_and_energy_deviations(p0c=2*p0c,
                                            update_pxpy=True)
    for particles in (kernel, python):
        np.testing.assert_allclose(
            [particles.px[0], particles.py[0], particles.ax[0], particles.ay[0]],
            [5e-4, -1e-3, 2e-4, -2.5e-4], rtol=1e-15)


def test_tpsa_tracking_keeps_s_and_vector_potential_like_scalar():
    line = xt.Line(elements=[xt.Drift(length=.4)])
    line.particle_ref = xt.Particles(p0c=1e9)
    line.reset_s_at_end_turn = False
    line.build_tracker()
    launch = dict(p0c=1e9, x=1e-4, px=2e-5, s=.3, ax=1e-5, ay=-2e-5)
    scalar = xt.Particles(**launch)
    line.track(scalar)
    series = xtpsa.ParticlesTpsa(order=1, **launch)
    line.track(series)
    snapshot = series.to_particles()
    for name in ('s', 'ax', 'ay', 'x'):
        np.testing.assert_allclose(getattr(snapshot, name)[0],
                                   getattr(scalar, name)[0], rtol=0, atol=1e-16)
