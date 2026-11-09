# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

from pathlib import Path

import numpy as np
import xtrack as xt


model_root = Path(__file__).resolve().parents[1] / 'test_data/example_lattices'


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
    env = xt.load(model_root / 'demo_ring/lattice.py')
    independent = xt.load(model_root / 'demo_ring/lattice.py')
    path = tmp_path / 'demo.json'
    env.to_json(path)
    loaded = xt.load(path)
    baseline = env['demo']['arc.s1.c1.qf.in'].k1
    loaded['kqf'] *= 1.001
    independent['kqf'] *= 0.999
    for name in ['arc.s1.c1.qf.in', 'arc.s2.c2.qf.in', 'arc.s6.c4.qf.out']:
        np.testing.assert_allclose(loaded['demo'][name].k1, baseline * 1.001)
        np.testing.assert_allclose(env['demo'][name].k1, baseline)
        np.testing.assert_allclose(
            independent['demo'][name].k1, baseline * 0.999,
        )


def test_demo_ring_synchronous_orbit_and_modes():
    from scipy.constants import c

    env = xt.load(model_root / 'demo_ring/lattice.py')
    line = env['demo']
    assert env['voltage_rf'] == 0
    bare = line.twiss4d()
    env['voltage_rf'] = 1e5
    np.testing.assert_allclose(line.get_length(), 1500., atol = 1e-10)
    np.testing.assert_allclose(
        env['rf.1'].frequency,
        200 * line.particle_ref.beta0[0] * c / line.get_length(),
        rtol = 1e-14)
    np.testing.assert_allclose(env['rf.1'].phase, np.pi, atol = 1e-14)

    tw = line.twiss6d()
    assert tw.slip_factor > 0
    assert tw.muzeta[-1] < tw.muzeta[0]
    np.testing.assert_allclose(tw.qs, 0.00792518424, rtol = 0, atol = 1e-8)
    np.testing.assert_allclose(
        np.abs(np.linalg.eigvals(tw.R_matrix)), 1., rtol = 0, atol = 1e-7)
    tracked = tw.particle_on_co.copy()
    line.track(tracked, num_turns = 1)
    assert tracked.state[0] == 1
    for coordinate in ['x', 'px', 'y', 'py', 'zeta', 'pzeta']:
        np.testing.assert_allclose(
            getattr(tracked, coordinate),
            getattr(tw.particle_on_co, coordinate),
            rtol = 0, atol = 1e-9)

    env['voltage_rf'] = 0
    restored = line.twiss4d()
    np.testing.assert_allclose([restored.qx, restored.qy],
                               [bare.qx, bare.qy], rtol = 0, atol = 1e-10)


demo_recipe = model_root / 'demo_ring/lattice.py'


def test_demo_ring_geometry_ip_and_sextupole_pair_maps():
    env = xt.load(demo_recipe)
    line = env['demo']
    np.testing.assert_allclose(line.particle_ref.p0c, 1e9)
    np.testing.assert_allclose(line.particle_ref.mass0, xt.ELECTRON_MASS_EV)
    assert line.particle_ref.q0 == -1
    np.testing.assert_allclose(line.get_length(), 1500, atol = 1e-10)
    survey = line.survey()
    np.testing.assert_allclose(survey.XYZ[-1], survey.XYZ[0], atol = 1e-9)
    for angle in ['theta', 'phi', 'psi']:
        difference = survey[angle][-1] - survey[angle][0]
        np.testing.assert_allclose(np.exp(1j * difference), 1, atol = 1e-10)
    tw = line.twiss4d()
    np.testing.assert_allclose([tw.qx, tw.qy],
                               [8.7548324869, 10.0174524224],
                               rtol = 0, atol = 1e-8)
    np.testing.assert_allclose([tw.betx[-1], tw.bety[-1]],
                               [tw.betx[0], tw.bety[0]], rtol = 1e-8)
    np.testing.assert_allclose(
        [tw['betx', 'ip.1'], tw['bety', 'ip.1']], [1, .01],
        rtol = 1e-6, atol = 1e-9)
    np.testing.assert_allclose(
        [tw[key, 'ip.1'] for key in ['alfx', 'alfy', 'dx', 'dpx']],
        0, atol = 1e-7)
    np.testing.assert_allclose(
        abs(np.linalg.eigvals(tw.R_matrix[:4, :4])), 1,
        rtol = 0, atol = 1e-7)
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


def test_demo_ring_independent_controls():
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

    assert env['voltage_rf'] == 0
