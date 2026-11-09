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


demo_recipe = model_root / 'demo_ring/lattice.py'


def test_demo_ring_geometry_ip_and_sextupole_pair_maps():
    env = xt.load(demo_recipe)
    line = env['demo']
    np.testing.assert_allclose(line.particle_ref.p0c, 1e9)
    assert line.particle_ref.q0 == -1
    np.testing.assert_allclose(line.get_length(), 1500, atol = 1e-10)
    survey = line.survey()
    np.testing.assert_allclose(survey.XYZ[-1], survey.XYZ[0], atol = 1e-9)
    for angle in ['theta', 'phi', 'psi']:
        difference = survey[angle][-1] - survey[angle][0]
        np.testing.assert_allclose(np.exp(1j * difference), 1, atol = 1e-10)
    tw = line.twiss4d()
    np.testing.assert_allclose(
        [tw['betx', 'ip.1'], tw['bety', 'ip.1']], [1, .01],
        rtol = 1e-6, atol = 1e-9)
    np.testing.assert_allclose(
        [tw[key, 'ip.1'] for key in ['alfx', 'alfy', 'dx', 'dpx']],
        0, atol = 1e-7)
    np.testing.assert_allclose(
        abs(np.linalg.eigvals(tw.R_matrix[:4, :4])), 1, atol = 1e-7)
    for part in ['sf', 'sd']:
        matrix = tw.get_R_matrix(f'arc.s1.c1.{part}', f'arc.s1.c3.{part}')
        np.testing.assert_allclose(matrix[:4, :4], -np.eye(4), atol = 1e-6)
    bpms = line.steering_monitors_x
    assert len(bpms) == 14
    assert all(isinstance(line[name], xt.Device)
               for name in bpms)
    particles = line.build_particles(x = [1e-6, -1e-6], y = [1e-7, -1e-7])
    line.track(particles, num_turns = 16)
    np.testing.assert_array_equal(particles.state, 1)


def test_demo_ring_independent_controls_and_rf_mode():
    env = xt.load(demo_recipe)
    independent = xt.load(demo_recipe)
    quad = 'arc.s1.c1.qf.in'
    before = independent['demo'][quad].k1
    env['dkqf'] = 1e-4
    np.testing.assert_allclose(env['demo'][quad].k1, before + 1e-4)
    np.testing.assert_allclose(independent['demo'][quad].k1, before)
    env['dkqf'] = 0
    env['h_cor_s1'], env['v_cor_s1'] = 2e-6, -3e-6
    probe = xt.Particles(p0c = 1e9, mass0 = xt.ELECTRON_MASS_EV, q0 = -1)
    env['demo']['cor.s1'].track(probe)
    np.testing.assert_allclose([probe.px[0], probe.py[0]], [2e-6, -3e-6])
    assert independent['h_cor_s1'] == independent['v_cor_s1'] == 0

    independent['voltage_rf'] = 1e5
    line = independent['demo']
    tw = line.twiss6d()
    assert tw.slip_factor > 0 and tw.qs > 0
    np.testing.assert_allclose(abs(np.linalg.eigvals(tw.R_matrix)), 1, atol = 1e-7)
    synchronous = tw.particle_on_co.copy()
    line.track(synchronous)
    for coordinate in ['x', 'px', 'y', 'py', 'zeta', 'pzeta']:
        np.testing.assert_allclose(
            getattr(synchronous, coordinate),
            getattr(tw.particle_on_co, coordinate), atol = 1e-9, rtol = 0)
    assert env['voltage_rf'] == 0
