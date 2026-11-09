# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

from pathlib import Path

import numpy as np
import xtrack as xt


model_root = Path(__file__).resolve().parents[1] / 'test_data/example_lattices'


def test_teaching_ring_geometry_and_optics():
    env = xt.load(model_root / 'ring/lattice.py')
    line = env['ring']
    assert line.particle_ref.q0 == -1
    np.testing.assert_allclose(line.particle_ref.mass0, xt.ELECTRON_MASS_EV)
    np.testing.assert_allclose(line.particle_ref.p0c, 1e9)
    np.testing.assert_allclose(line.get_length(), 40., rtol = 0, atol = 1e-12)

    table = line.get_table()
    np.testing.assert_allclose(table['s', 'obs.0'], 0., atol = 1e-12)
    np.testing.assert_allclose(table['s', 'obs.4'], 20., atol = 1e-12)
    survey = line.survey()
    np.testing.assert_allclose(
        [survey.X[-1], survey.Y[-1], survey.Z[-1]],
        [survey.X[0], survey.Y[0], survey.Z[0]], atol = 1e-11,
    )
    tw = line.twiss(method = '4d')
    np.testing.assert_allclose(
        [tw.qx, tw.qy], [1.62, 1.58], rtol = 0, atol = 1e-8,
    )
    np.testing.assert_allclose(tw.betx[-1], tw.betx[0], rtol = 1e-8)
    np.testing.assert_allclose(tw.bety[-1], tw.bety[0], rtol = 1e-8)


def test_teaching_transport_open_optics():
    env = xt.load(model_root / 'transport/lattice.py')
    line = env['transport']
    assert line.particle_ref.q0 == -1
    np.testing.assert_allclose(line.particle_ref.mass0, xt.ELECTRON_MASS_EV)
    np.testing.assert_allclose(line.particle_ref.p0c, 1e9)
    np.testing.assert_allclose(line.get_length(), 7.1, rtol = 0, atol = 1e-12)
    tw = line.twiss(method = '4d', betx = 1., bety = 1.)
    # Qualified entrance conditions, independently retained from the preview.
    np.testing.assert_allclose(
        [tw.betx[-1], tw.bety[-1]],
        [55.01759394372164, 46.16305551464726], rtol = 1e-8,
    )


def test_teaching_model_load_and_json_independence(tmp_path):
    env = xt.load(model_root / 'ring/lattice.py')
    independent = xt.load(model_root / 'ring/lattice.py')
    path = tmp_path / 'ring.json'
    env.to_json(path)
    loaded = xt.load(path)
    baseline = env['ring']['qf.1'].k1
    loaded['kqf'] *= 1.01
    independent['kqf'] *= 0.99
    for name in ['qf.1', 'qf.2', 'qf.8']:
        np.testing.assert_allclose(loaded['ring'][name].k1, baseline * 1.01)
        np.testing.assert_allclose(env['ring'][name].k1, baseline)
        np.testing.assert_allclose(
            independent['ring'][name].k1, baseline * 0.99,
        )


def test_teaching_rf_ring_synchronous_orbit_and_modes():
    from scipy.constants import c

    env = xt.load(model_root / 'rf_ring/lattice.py')
    line = env['rf_ring']
    np.testing.assert_allclose(line.get_length(), 40., atol = 1e-12)
    assert 'rf' not in env['ring'].element_names
    np.testing.assert_allclose(
        env['rf'].frequency,
        20 * line.particle_ref.beta0[0] * c / line.get_length(),
        rtol = 1e-14)
    np.testing.assert_allclose(env['rf'].phase, np.pi, atol = 1e-14)

    tw = line.twiss6d()
    assert tw.slip_factor > 0
    assert tw.muzeta[-1] < tw.muzeta[0]
    np.testing.assert_allclose(tw.qs, 0.011393221247, atol = 1e-8)
    np.testing.assert_allclose(
        np.abs(np.linalg.eigvals(tw.R_matrix)), 1., atol = 1e-7)
    tracked = tw.particle_on_co.copy()
    line.track(tracked, num_turns = 1)
    assert tracked.state[0] == 1
    for coordinate in ['x', 'px', 'y', 'py', 'zeta', 'pzeta']:
        np.testing.assert_allclose(
            getattr(tracked, coordinate),
            getattr(tw.particle_on_co, coordinate),
            rtol = 0, atol = 1e-9)

    bare = env['ring'].twiss4d()
    np.testing.assert_allclose([bare.qx, bare.qy], [1.62, 1.58], atol = 1e-8)
