import numpy as np
import pytest

import xpart as xp
import xtrack as xt


def test_cavity_model_properties_and_serialization():
    cavity = xt.Cavity(
        model="sagan",
        fringe_model="sagan",
        cavity_type="traveling-wave",
        edge_entry_active=False,
        edge_exit_active=True,
        drift_model="drift-kick-drift-exact",
        num_kicks=7,
    )

    assert cavity.model == "sagan"
    assert cavity.fringe_model == "sagan"
    assert cavity.cavity_type == "traveling-wave"
    assert cavity.edge_entry_active == 0
    assert cavity.edge_exit_active == 1
    assert cavity.drift_model == "drift-kick-drift-exact"

    serialized = cavity.to_dict()
    assert serialized["model"] == "sagan"
    assert serialized["fringe_model"] == "sagan"
    assert serialized["cavity_type"] == "traveling-wave"
    assert serialized["drift_model"] == "drift-kick-drift-exact"

    restored = xt.Cavity.from_dict(serialized)
    assert restored.model == cavity.model
    assert restored.fringe_model == cavity.fringe_model
    assert restored.cavity_type == cavity.cavity_type
    assert restored.drift_model == cavity.drift_model
    assert restored.num_kicks == cavity.num_kicks


def test_cavity_type_accepts_uk_spelling_alias():
    cavity = xt.Cavity(model="sagan", cavity_type="travelling-wave")
    assert cavity.cavity_type == "traveling-wave"

    cavity.cavity_type = "standing-wave"
    cavity.cavity_type = "travelling-wave"
    assert cavity.cavity_type == "traveling-wave"

    line = xt.Line(elements=[xt.Cavity(length=1.0, voltage=1e6, frequency=1e9)])
    line.configure_cavity_model(model="sagan", cavity_type="travelling-wave")
    assert line.elements[0].cavity_type == "traveling-wave"

    with pytest.raises(ValueError):
        xt.Cavity(
            model="rosenzweig-serafini", cavity_type="travelling-wave")


def test_sad_track_trpt_public_name_and_serialization():
    cavity = xt.Cavity(
        model="sad-track-trpt",
        fringe_model="sad-track-trpt",
    )

    assert cavity.model == "sad-track-trpt"
    assert cavity.fringe_model == "sad-track-trpt"
    restored = xt.Cavity.from_dict(cavity.to_dict())
    assert restored.model == "sad-track-trpt"
    assert restored.fringe_model == "sad-track-trpt"

    twiss_cavity = xt.Cavity(
        model="sad-twiss-trpt",
        fringe_model="sad-twiss-trpt",
    )
    assert twiss_cavity.model == "sad-twiss-trpt"
    assert twiss_cavity.fringe_model == "sad-twiss-trpt"
    restored = xt.Cavity.from_dict(twiss_cavity.to_dict())
    assert restored.model == "sad-twiss-trpt"
    assert restored.fringe_model == "sad-twiss-trpt"


def test_legacy_cavity_model_is_migrated_to_drift_model():
    with pytest.warns(FutureWarning, match="drift_model"):
        cavity = xt.Cavity(model="drift-kick-drift-exact")

    assert cavity.model == "longitudinal-only"
    assert cavity.drift_model == "drift-kick-drift-exact"
    assert cavity.to_dict()["drift_model"] == "drift-kick-drift-exact"


def test_legacy_cavity_model_assignment_is_migrated():
    cavity = xt.Cavity(model="sagan")

    with pytest.warns(FutureWarning, match="drift_model"):
        cavity.model = "drift-kick-drift-exact"

    assert cavity.model == "longitudinal-only"
    assert cavity.drift_model == "drift-kick-drift-exact"


def test_line_configure_cavity_model():
    line = xt.Line(
        elements={
            "cavity_1": xt.Cavity(),
            "drift": xt.Drift(length=1),
            "cavity_2": xt.Cavity(),
        },
        element_names=["cavity_1", "drift", "cavity_2"],
    )

    line.configure_cavity_model(
        model="sagan",
        fringe_model="sagan",
        cavity_type="standing-wave",
        edge_entry_active=False,
        edge_exit_active=True,
        drift_model="drift-kick-drift-exact",
        num_kicks=11,
        integrator="uniform",
    )

    for name in ("cavity_1", "cavity_2"):
        cavity = line[name]
        assert cavity.model == "sagan"
        assert cavity.fringe_model == "sagan"
        assert cavity.cavity_type == "standing-wave"
        assert cavity.edge_entry_active == 0
        assert cavity.edge_exit_active == 1
        assert cavity.drift_model == "drift-kick-drift-exact"
        assert cavity.num_kicks == 11
        assert cavity.integrator == "uniform"


@pytest.mark.parametrize(
    "keyword, value",
    (
        ("model", "unknown"),
        ("fringe_model", "unknown"),
        ("cavity_type", "unknown"),
        ("drift_model", "unknown"),
    ),
)
def test_invalid_cavity_configuration(keyword, value):
    with pytest.raises(ValueError):
        xt.Cavity(**{keyword: value})


def test_rs_options_are_exposed():
    assert "rosenzweig-serafini" in xt.Cavity.get_available_models()
    assert "rosenzweig-serafini" in xt.Cavity.get_available_fringe_models()


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "model": "sagan",
            "fringe_model": "rosenzweig-serafini",
        },
        {
            "model": "rosenzweig-serafini",
            "fringe_model": "sagan",
        },
        {
            "model": "rosenzweig-serafini",
            "cavity_type": "traveling-wave",
        },
        {
            "model": "sad-track-trpt",
            "cavity_type": "traveling-wave",
        },
        {
            "model": "sad-twiss-trpt",
            "cavity_type": "traveling-wave",
        },
    ],
)
def test_unsupported_cavity_model_combinations_are_rejected(kwargs):
    with pytest.raises(ValueError):
        xt.Cavity(**kwargs)


def test_configure_cavity_model_validates_before_mutating_line():
    first = xt.Cavity(model="sagan", fringe_model="auto")
    second = xt.Cavity(
        model="sagan", fringe_model="sagan", cavity_type="standing-wave")
    line = xt.Line(elements=[first, second])

    with pytest.raises(ValueError):
        line.configure_cavity_model(model="rosenzweig-serafini")

    assert first.model == "sagan"
    assert second.model == "sagan"

    line.configure_cavity_model(
        model="rosenzweig-serafini",
        fringe_model="rosenzweig-serafini",
    )
    assert first.model == "rosenzweig-serafini"
    assert second.model == "rosenzweig-serafini"


def test_sagan_native_map_and_backtracking():
    mass0 = xp.ELECTRON_MASS_EV
    energy0 = 100e6
    p0c = np.sqrt(energy0**2 - mass0**2)
    coordinates = np.array((
        0.3e-3, 2e-6, -0.4e-3, 1e-6, 0.2e-3, 5e-7,
    ))
    expected = np.array((
        3.005253183517888343e-04,
        1.998839460443983058e-06,
        -3.970516217951933985e-04,
        1.019406791423381079e-06,
        2.001267555448297913e-04,
        9.802253811174161223e-03,
    ))

    line = xt.Line(elements=[xt.Cavity(
        length=1.0,
        voltage=1e6,
        frequency=100e6,
        phase=0.5 * np.pi + 0.2,
        model="sagan",
        fringe_model="sagan",
        cavity_type="standing-wave",
        num_kicks=4,
    )])
    particles = xp.Particles(
        p0c=p0c,
        mass0=mass0,
        x=coordinates[0],
        px=coordinates[1],
        y=coordinates[2],
        py=coordinates[3],
        zeta=coordinates[4],
        pzeta=coordinates[5],
    )
    initial = particles.copy()

    line.track(particles)
    actual = np.array((
        particles.x[0], particles.px[0],
        particles.y[0], particles.py[0],
        particles.zeta[0], particles.pzeta[0],
    ))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=5e-16)

    np.testing.assert_allclose(particles.p0c, initial.p0c, rtol=0, atol=0)

    line.track(particles, backtrack=True)
    for coordinate in ("x", "px", "y", "py", "zeta", "pzeta"):
        np.testing.assert_allclose(
            getattr(particles, coordinate),
            getattr(initial, coordinate),
            rtol=0,
            atol=1e-14,
        )
    np.testing.assert_allclose(particles.p0c, initial.p0c, rtol=2e-16)


def test_rs_native_transverse_map_and_backtracking():
    mass0 = xp.ELECTRON_MASS_EV
    energy0 = 100e6
    p0c = np.sqrt(energy0**2 - mass0**2)
    expected = np.array((
        (9.517862301650036327e-01, 9.529214331654027870e-01),
        (-3.248595794882056399e-03, 9.518894304377670812e-01),
    ))
    line = xt.Line(elements=[xt.Cavity(
        length=1.0,
        voltage=10e6,
        frequency=100e6,
        phase=0.5 * np.pi,
        model="rosenzweig-serafini",
        fringe_model="auto",
    )])
    particles = xp.Particles(
        p0c=p0c,
        mass0=mass0,
        x=[1e-6, 0],
        px=[0, 1e-6],
        y=0,
        py=0,
        zeta=0,
        delta=0,
    )
    initial = particles.copy()

    line.track(particles)
    momentum_ratio = np.sqrt((energy0 + 10e6)**2 - mass0**2) / p0c
    expected[1] *= momentum_ratio
    actual = np.array((particles.x, particles.px)) / 1e-6
    np.testing.assert_allclose(actual, expected, rtol=0, atol=5e-15)
    np.testing.assert_allclose(particles.p0c, initial.p0c, rtol=0, atol=0)

    line.track(particles, backtrack=True)
    np.testing.assert_allclose(particles.x, initial.x, rtol=0, atol=1e-20)
    np.testing.assert_allclose(particles.px, initial.px, rtol=0, atol=1e-20)
    np.testing.assert_allclose(particles.p0c, initial.p0c, rtol=2e-16)


def test_rs_longitudinal_map_matches_live_ocelot_reference():
    # Frozen from a native computation cross-checked against a live OCELOT
    # oracle to 1.64e-3 worst-case over a broad energy/voltage/phase grid.
    mass0 = xp.ELECTRON_MASS_EV
    energy0 = 100e6
    p0c = np.sqrt(energy0**2 - mass0**2)
    line = xt.Line(elements=[xt.Cavity(
        length=1.0,
        voltage=5e6,
        frequency=100e6,
        phase=0.5 * np.pi - 0.3,
        model="rosenzweig-serafini",
        fringe_model="auto",
    )])
    particles = xp.Particles(
        p0c=p0c,
        mass0=mass0,
        x=1e-6,
        px=0.5e-6,
        y=-0.5e-6,
        py=0.3e-6,
        zeta=2e-4,
        delta=1e-4,
    )

    line.track(particles)

    np.testing.assert_allclose(
        particles.zeta, 0.0002000023617475422, rtol=0, atol=1e-16)
    np.testing.assert_allclose(
        particles.delta, 0.04786184514933023, rtol=0, atol=1e-16)


def test_rs_longitudinal_forward_backward_closure():
    # Closure degrades at extreme single-element gain; see track_cavity_rs.h.
    mass0 = xp.ELECTRON_MASS_EV
    energy0 = 100e6
    p0c = np.sqrt(energy0**2 - mass0**2)
    line = xt.Line(elements=[xt.Cavity(
        length=1.0,
        voltage=5e6,
        frequency=100e6,
        phase=0.5 * np.pi - 0.3,
        model="rosenzweig-serafini",
        fringe_model="auto",
    )])
    particles = xp.Particles(
        p0c=p0c,
        mass0=mass0,
        x=1e-6,
        px=0.5e-6,
        y=-0.5e-6,
        py=0.3e-6,
        zeta=2e-4,
        delta=1e-4,
    )
    initial = particles.copy()

    line.track(particles)
    line.track(particles, backtrack=True)

    np.testing.assert_allclose(particles.zeta, initial.zeta, rtol=0, atol=2e-6)
    np.testing.assert_allclose(particles.delta, initial.delta, rtol=0, atol=1e-9)


def test_sad_track_trpt_body_matches_frozen_sad_particle_map():
    mass0 = xp.ELECTRON_MASS_EV
    p0c_in = 14208754.28106691
    length = 1.270948
    voltage = 13696827.259102523
    frequency = 2.856e9
    phase = -2.2689280275926285 + np.pi
    energy_out = np.hypot(p0c_in, mass0) + voltage * np.sin(phase)
    p0c_out = np.sqrt(energy_out**2 - mass0**2)
    coordinate_steps = np.full(6, 1e-4)
    coordinates = np.zeros((13, 6))
    for index, step in enumerate(coordinate_steps):
        coordinates[2 * index + 1, index] = step
        coordinates[2 * index + 2, index] = -step

    expected_output = np.array([
        [0, 0, 0, 0, -1.1765722868099114e-05, 1.4986815795592698e-04],
        [1.0064322780027493e-04, 8.5720973866838797e-07, 0, 0,
         -1.1765721626751303e-05, 1.4987121388281561e-04],
        [-1.0064322780027493e-04, -8.5720973866838797e-07, 0, 0,
         -1.1765721626751303e-05, 1.4987121388281561e-04],
        [9.5394651721824472e-05, 5.7958525649123122e-05, 0, 0,
         -1.1769379751864490e-05, 1.4991534506663727e-04],
        [-9.5394651721824472e-05, -5.7958525649123122e-05, 0, 0,
         -1.1769379751864490e-05, 1.4991534506663727e-04],
        [0, 0, 1.0064322780027493e-04, 8.5720973866838797e-07,
         -1.1765721626751303e-05, 1.4987121388281561e-04],
        [0, 0, -1.0064322780027493e-04, -8.5720973866838797e-07,
         -1.1765721626751303e-05, 1.4987121388281561e-04],
        [0, 0, 9.5394651721824472e-05, 5.7958525649123122e-05,
         -1.1769379751864490e-05, 1.4991534506663727e-04],
        [0, 0, -9.5394651721824472e-05, -5.7958525649123122e-05,
         -1.1769379751864490e-05, 1.4991534506663727e-04],
        [0, 0, 0, 0, 8.7266074992502728e-05, -1.9823217234477089e-03],
        [0, 0, 0, 0, -1.1080913027717908e-04, 2.2670389080647890e-03],
        [0, 0, 0, 0, -1.1691606004474519e-05, 2.0635051721510945e-04],
        [0, 0, 0, 0, -1.1839857523200194e-05, 9.3386058830418554e-05],
    ])

    line = xt.Line(elements=[
        xt.Cavity(
            length=length,
            voltage=voltage,
            frequency=frequency,
            phase=phase,
            model="sad-track-trpt",
            fringe_model="suppressed",
            num_kicks=0,
        ),
        xt.ReferenceEnergyIncrease(Delta_p0c=p0c_out - p0c_in),
    ])
    particles = xp.Particles(
        "positron",
        p0c=p0c_in,
        x=coordinates[:, 0],
        px=coordinates[:, 1],
        y=coordinates[:, 2],
        py=coordinates[:, 3],
        zeta=coordinates[:, 4],
        delta=coordinates[:, 5],
    )

    line.track(particles)
    output = np.column_stack([
        particles.x,
        particles.px,
        particles.y,
        particles.py,
        particles.zeta,
        particles.delta,
    ])
    actual_matrix = np.column_stack([
        (output[2 * index + 1] - output[2 * index + 2])
        / (2.0 * coordinate_steps[index])
        for index in range(6)
    ])
    expected_matrix = np.column_stack([
        (expected_output[2 * index + 1] - expected_output[2 * index + 2])
        / (2.0 * coordinate_steps[index])
        for index in range(6)
    ])

    np.testing.assert_allclose(
        output, expected_output, rtol=0, atol=2e-15,
    )
    np.testing.assert_allclose(
        actual_matrix, expected_matrix, rtol=0, atol=1e-11,
    )


def test_sad_track_trpt_full_fringe_matches_frozen_sad_particle_map():
    mass0 = xp.ELECTRON_MASS_EV
    p0c_in = 14208754.28106691
    length = 1.270948
    voltage = 13696827.259102523
    frequency = 2.856e9
    phase = -2.2689280275926285 + np.pi
    energy_out = np.hypot(p0c_in, mass0) + voltage * np.sin(phase)
    p0c_out = np.sqrt(energy_out**2 - mass0**2)
    coordinates = np.zeros((13, 6))
    for index in range(6):
        coordinates[2 * index + 1, index] = 1e-4
        coordinates[2 * index + 2, index] = -1e-4

    expected_output = np.array([
        [0, 0, 0, 0, -1.1765722868099116e-05, 1.4986815795592698e-04],
        [7.2930185571343023e-05, -3.7877368582321046e-06, 0, 0,
         -1.1765963465776307e-05, 1.4989309320754887e-04],
        [-7.2930185571343023e-05, 3.7877368582321046e-06, 0, 0,
         -1.1765963465776307e-05, 1.4989309320754887e-04],
        [9.5394651721824472e-05, 7.3906710249304621e-05, 0, 0,
         -1.1769379751672511e-05, 1.4987718513763144e-04],
        [-9.5394651721824472e-05, -7.3906710249304621e-05, 0, 0,
         -1.1769379751672511e-05, 1.4987718513763144e-04],
        [0, 0, 7.2930185571343023e-05, -3.7877368582321046e-06,
         -1.1765963465776307e-05, 1.4989309320754887e-04],
        [0, 0, -7.2930185571343023e-05, 3.7877368582321046e-06,
         -1.1765963465776307e-05, 1.4989309320754887e-04],
        [0, 0, 9.5394651721824472e-05, 7.3906710249304621e-05,
         -1.1769379751672511e-05, 1.4987718513763144e-04],
        [0, 0, -9.5394651721824472e-05, -7.3906710249304621e-05,
         -1.1769379751672511e-05, 1.4987718513763144e-04],
        [0, 0, 0, 0, 8.7266074992502714e-05, -1.9823217234477089e-03],
        [0, 0, 0, 0, -1.1080913027717908e-04, 2.2670389080647890e-03],
        [0, 0, 0, 0, -1.1691606004474517e-05, 2.0635051721510945e-04],
        [0, 0, 0, 0, -1.1839857523200192e-05, 9.3386058830418554e-05],
    ])

    line = xt.Line(elements=[
        xt.Cavity(
            length=length,
            voltage=voltage,
            frequency=frequency,
            phase=phase,
            model="sad-track-trpt",
            fringe_model="auto",
            num_kicks=0,
        ),
        xt.ReferenceEnergyIncrease(Delta_p0c=p0c_out - p0c_in),
    ])
    particles = xp.Particles(
        "positron",
        p0c=p0c_in,
        x=coordinates[:, 0],
        px=coordinates[:, 1],
        y=coordinates[:, 2],
        py=coordinates[:, 3],
        zeta=coordinates[:, 4],
        delta=coordinates[:, 5],
    )

    line.track(particles)
    output = np.column_stack([
        particles.x,
        particles.px,
        particles.y,
        particles.py,
        particles.zeta,
        particles.delta,
    ])
    np.testing.assert_allclose(output, expected_output, rtol=0, atol=2e-15)


@pytest.mark.parametrize(
    "fringe_model,num_kicks,phase",
    [
        ("suppressed", 0, 0.8726646259971647),
        ("auto", 0, 0.8726646259971647),
        ("auto", 7, 0.35),
        ("auto", 7, -0.35),
    ],
)
def test_sad_track_trpt_forward_backward_closure(
        fringe_model, num_kicks, phase):
    mass0 = xp.ELECTRON_MASS_EV
    p0c_in = 14208754.28106691
    voltage = 13696827.259102523
    energy_out = np.hypot(p0c_in, mass0) + voltage * np.sin(phase)
    p0c_out = np.sqrt(energy_out**2 - mass0**2)
    coordinates = np.array([
        [0, 0, 0, 0, 0, 0],
        [1e-4, 2e-5, -3e-4, 4e-5, 5e-4, 6e-4],
        [-2e-3, 3e-4, 1e-3, -2e-4, -7e-4, -5e-4],
    ])
    line = xt.Line(elements=[
        xt.Cavity(
            length=1.270948,
            voltage=voltage,
            frequency=2.856e9,
            phase=phase,
            model="sad-track-trpt",
            fringe_model=fringe_model,
            num_kicks=num_kicks,
        ),
        xt.ReferenceEnergyIncrease(Delta_p0c=p0c_out - p0c_in),
    ])
    particles = xp.Particles(
        "positron",
        p0c=p0c_in,
        x=coordinates[:, 0],
        px=coordinates[:, 1],
        y=coordinates[:, 2],
        py=coordinates[:, 3],
        zeta=coordinates[:, 4],
        delta=coordinates[:, 5],
    )
    initial = particles.copy()

    line.track(particles)
    line.track(particles, backtrack=True)

    for coordinate in ("x", "px", "y", "py", "zeta", "delta", "s"):
        np.testing.assert_allclose(
            getattr(particles, coordinate),
            getattr(initial, coordinate),
            rtol=0,
            atol=2e-15,
        )
    np.testing.assert_allclose(particles.p0c, initial.p0c, rtol=0, atol=0)


@pytest.mark.parametrize(
    "fringe_model,expected_matrix",
    [
        (
            "suppressed",
            np.array([
                [1.006424041236356, 0.954562558086129, 0, 0, 0, 0],
                [0.008576256367914512, 0.5795991924273787, 0, 0, 0, 0],
                [0, 0, 1.0064240412363565, 0.9545625580861292, 0, 0],
                [0, 0, 0.008576256367914077, 0.5795991924273779, 0, 0],
                [0, 0, 0, 0, 0.9904274997156705, 0.0007441607417683668],
                [0, 0, 0, 0, -21.259123540241955, 0.5647215806712221],
            ]),
        ),
        (
            "auto",
            np.array([
                [0.7291160660227245, 0.954562558086129, 0, 0, 0, 0],
                [-0.03798031948205952, 0.739088634187651, 0, 0, 0, 0],
                [0, 0, 0.7291160660227249, 0.9545625580861289, 0, 0],
                [0, 0, -0.03798031948205984, 0.7390886341876499, 0, 0],
                [0, 0, 0, 0, 0.9904274997156705, 0.0007441607417683668],
                [0, 0, 0, 0, -21.259123540241955, 0.5647215806712221],
            ]),
        ),
    ],
)
def test_sad_twiss_trpt_matches_frozen_sad_tcave_matrix(
        fringe_model, expected_matrix):
    momentum_in = 14208754.281066904
    momentum_out = 24705034.234605756
    entrance_coordinates = np.array([
        0, 0, 0, 0, 1.00759114e-7, -5.56361160e-7,
    ])
    coordinate_steps = np.full(6, 1e-8)
    coordinates = np.repeat(entrance_coordinates[None, :], 12, axis=0)
    for index, step in enumerate(coordinate_steps):
        coordinates[2 * index, index] += step
        coordinates[2 * index + 1, index] -= step

    line = xt.Line(elements=[
        xt.Cavity(
            length=1.270948,
            voltage=13696827.259102523,
            frequency=2.856e9,
            phase=0.8726646259971647,
            model="sad-twiss-trpt",
            fringe_model=fringe_model,
            num_kicks=0,
        ),
        xt.ReferenceEnergyIncrease(Delta_p0c=momentum_out - momentum_in),
    ])
    particles = xp.Particles(
        "positron",
        p0c=momentum_in,
        x=coordinates[:, 0],
        px=coordinates[:, 1],
        y=coordinates[:, 2],
        py=coordinates[:, 3],
        zeta=coordinates[:, 4],
        delta=coordinates[:, 5],
    )
    line.track(particles)
    output = np.column_stack([
        particles.x, particles.px, particles.y, particles.py,
        particles.zeta, particles.delta,
    ])
    actual_matrix = np.column_stack([
        (output[2 * index] - output[2 * index + 1])
        / (2.0 * coordinate_steps[index])
        for index in range(6)
    ])

    np.testing.assert_allclose(
        actual_matrix[:4, :4], expected_matrix[:4, :4],
        rtol=0, atol=5e-14,
    )
    np.testing.assert_allclose(
        actual_matrix[4:, 4:], expected_matrix[4:, 4:],
        rtol=0, atol=7e-8,
    )


@pytest.mark.parametrize("fringe_model", ["suppressed", "auto"])
def test_sad_twiss_trpt_forward_backward_closure(fringe_model):
    momentum_in = 14208754.281066904
    momentum_out = 24705034.234605756
    line = xt.Line(elements=[
        xt.Cavity(
            length=1.270948,
            voltage=13696827.259102523,
            frequency=2.856e9,
            phase=0.8726646259971647,
            model="sad-twiss-trpt",
            fringe_model=fringe_model,
            num_kicks=4,
        ),
        xt.ReferenceEnergyIncrease(Delta_p0c=momentum_out - momentum_in),
    ])
    particles = xp.Particles(
        "positron",
        p0c=momentum_in,
        x=[0, 1e-4, -2e-3],
        px=[0, 2e-5, 3e-4],
        y=[0, -3e-4, 1e-3],
        py=[0, 4e-5, -2e-4],
        zeta=[1.00759114e-7, 5e-4, -7e-4],
        delta=[-5.56361160e-7, 6e-4, -5e-4],
    )
    initial = particles.copy()

    line.track(particles)
    line.track(particles, backtrack=True)

    for coordinate in ("x", "px", "y", "py", "zeta", "delta", "s"):
        np.testing.assert_allclose(
            getattr(particles, coordinate),
            getattr(initial, coordinate),
            rtol=0,
            atol=3e-15,
        )
    np.testing.assert_allclose(particles.p0c, initial.p0c, rtol=0, atol=0)


def test_zero_length_sagan_is_longitudinal_only():
    common = dict(
        length=0,
        voltage=5e6,
        frequency=400e6,
        phase=0.3,
    )
    sagan_line = xt.Line(elements=[xt.Cavity(
        **common,
        model="sagan",
        fringe_model="sagan",
        cavity_type="standing-wave",
    )])
    particles = xp.Particles(
        p0c=10e9,
        mass0=xp.PROTON_MASS_EV,
        x=[1e-3, -2e-3],
        px=[2e-4, -3e-4],
        y=[-1.5e-3, 0.5e-3],
        py=[-1e-4, 4e-4],
        zeta=[0, 2e-3],
        delta=[0, 1e-3],
    )
    initial = particles.copy()
    initial_energy = particles.energy.copy()
    initial_tau = particles.zeta / particles.beta0

    sagan_line.track(particles)

    rf_phase = (
        common["phase"]
        - 2 * np.pi * common["frequency"] * initial_tau / 299792458.0
    )
    np.testing.assert_allclose(
        particles.energy - initial_energy,
        common["voltage"] * np.sin(rf_phase),
        rtol=2e-14,
        atol=2e-6,
    )
    np.testing.assert_allclose(particles.p0c, initial.p0c, rtol=0, atol=0)
    np.testing.assert_allclose(particles.x, initial.x, rtol=0, atol=0)
    np.testing.assert_allclose(particles.y, initial.y, rtol=0, atol=0)
    np.testing.assert_allclose(particles.px, initial.px, rtol=0, atol=0)
    np.testing.assert_allclose(particles.py, initial.py, rtol=0, atol=0)
    np.testing.assert_allclose(
        particles.zeta / particles.beta0, initial_tau, rtol=2e-15,
    )


def test_zero_length_sagan_scales_multispecies_energy():
    voltage = 3e6
    phase = 0.4
    line = xt.Line(elements=[xt.Cavity(
        voltage=voltage,
        phase=phase,
        model="sagan",
    )])
    particles = xp.Particles(
        mass0=xp.PROTON_MASS_EV,
        q0=1,
        p0c=1.4e9,
        zeta=0,
        delta=[0, 1e-4],
        mass_ratio=[0.2, 1.7],
        charge_ratio=[0.5, 2.0],
    )
    initial_energy = particles.energy.copy()

    line.track(particles)

    expected_gain = particles.charge_ratio * voltage * np.sin(phase)
    np.testing.assert_allclose(
        particles.energy - initial_energy,
        expected_gain,
        rtol=2e-13,
        atol=2e-6,
    )


@pytest.mark.parametrize("model", ["sagan", "rosenzweig-serafini"])
def test_reference_energy_change_is_explicit(model):
    mass0 = xp.ELECTRON_MASS_EV
    energy_in = 100e6
    p0c_in = np.sqrt(energy_in**2 - mass0**2)
    voltage = 1e6
    energy_out = energy_in + voltage
    p0c_out = np.sqrt(energy_out**2 - mass0**2)

    line = xt.Line(elements=[
        xt.Cavity(
            length=0,
            voltage=voltage,
            phase=0.5 * np.pi,
            model=model,
        ),
        xt.ReferenceEnergyIncrease(Delta_p0c=p0c_out - p0c_in),
    ])
    particles = xp.Particles(p0c=p0c_in, mass0=mass0)

    line.track(particles)

    np.testing.assert_allclose(particles.p0c, p0c_out, rtol=2e-16)
    np.testing.assert_allclose(particles.delta, 0, rtol=0, atol=2e-16)
    np.testing.assert_allclose(particles.energy, energy_out, rtol=2e-16)


def test_sagan_thick_slicing_preserves_map():
    cavity = xt.Cavity(
        length=1.0,
        voltage=1e6,
        frequency=100e6,
        phase=0.5 * np.pi + 0.2,
        model="sagan",
        fringe_model="sagan",
        cavity_type="standing-wave",
        num_kicks=4,
    )
    unsliced = xt.Line(elements={"cavity": cavity}, element_names=["cavity"])
    sliced = unsliced.copy()
    sliced.slice_thick_elements([
        xt.Strategy(xt.Uniform(2, mode="thick")),
    ])
    particles = xp.Particles(
        p0c=100e6,
        mass0=xp.ELECTRON_MASS_EV,
        x=0.3e-3,
        px=2e-6,
        y=-0.4e-3,
        py=1e-6,
        zeta=0.2e-3,
        pzeta=5e-7,
    )
    expected = particles.copy()

    unsliced.track(expected)
    sliced.track(particles)

    for coordinate in ("x", "px", "y", "py", "zeta", "pzeta", "p0c"):
        np.testing.assert_allclose(
            getattr(particles, coordinate),
            getattr(expected, coordinate),
            rtol=0,
            atol=5e-15,
        )


@pytest.mark.parametrize("model", ["sad-track-trpt", "sad-twiss-trpt"])
def test_sad_thick_slicing_preserves_map(model):
    # Both SAD models' reference-phase progression is a property of the
    # whole (unsliced) cavity, not of each external slice in isolation:
    # the focusing coefficient uses the cavity's nominal (un-weight-scaled)
    # voltage, and the reference-clock accumulator that tracks how far the
    # synchronous particle has advanced starts from slice_offset/length,
    # not zero, at every slice. See physics.md "Length and RF-step
    # semantics". sad-twiss-trpt composes to a slightly looser floor
    # (~1e-9) than sad-track-trpt (~1e-10) even at fine internal
    # resolution -- both are of the same order as R&S's own accepted
    # composition residual, not a remaining bug (see DEVLOG.md).
    cavity = xt.Cavity(
        length=1.270948,
        voltage=13696827.259102523,
        frequency=2.856e9,
        phase=0.8726646259971647,
        model=model,
        fringe_model=model,
        num_kicks=20,
    )
    unsliced = xt.Line(elements={"cavity": cavity}, element_names=["cavity"])
    sliced = unsliced.copy()
    sliced.slice_thick_elements([
        xt.Strategy(xt.Uniform(2, mode="thick")),
    ])
    particles = xp.Particles(
        p0c=100e6,
        mass0=xp.ELECTRON_MASS_EV,
        x=0.3e-3,
        px=2e-6,
        y=-0.4e-3,
        py=1e-6,
        zeta=0.2e-3,
        pzeta=5e-7,
    )
    expected = particles.copy()

    unsliced.track(expected)
    sliced.track(particles)

    for coordinate in ("x", "px", "y", "py", "zeta", "pzeta", "p0c"):
        np.testing.assert_allclose(
            getattr(particles, coordinate),
            getattr(expected, coordinate),
            rtol=0,
            atol=1e-8,
        )


@pytest.mark.parametrize("model", ["sad-track-trpt", "sad-twiss-trpt"])
def test_sad_thick_slicing_backward_closure(model):
    cavity = xt.Cavity(
        length=1.270948,
        voltage=13696827.259102523,
        frequency=2.856e9,
        phase=0.8726646259971647,
        model=model,
        fringe_model=model,
        num_kicks=20,
    )
    line = xt.Line(elements={"cavity": cavity}, element_names=["cavity"])
    line.slice_thick_elements([xt.Strategy(xt.Uniform(2, mode="thick"))])

    original = xp.Particles(
        p0c=100e6,
        mass0=xp.ELECTRON_MASS_EV,
        x=0.3e-3,
        px=2e-6,
        y=-0.4e-3,
        py=1e-6,
        zeta=0.2e-3,
        pzeta=5e-7,
    )
    particles = original.copy()

    line.track(particles)
    line.track(particles, backtrack=True)

    for coordinate in ("x", "px", "y", "py", "zeta", "pzeta", "p0c"):
        np.testing.assert_allclose(
            getattr(particles, coordinate),
            getattr(original, coordinate),
            rtol=0,
            atol=1e-12,
        )


@pytest.mark.parametrize("model", [
    "longitudinal-only", "rosenzweig-serafini", "sagan",
    "sad-track-trpt", "sad-twiss-trpt",
])
def test_slicing_structure_independent_of_cavity_model(model):
    # Slicing behavior must not depend on which model is set: slice a
    # generic cavity, then switch its model, and confirm the slice
    # structure (names, count) is unaffected.
    cavity = xt.Cavity(length=1.0, voltage=1e6, frequency=100e6, phase=0.5)
    line = xt.Line(elements={"cavity": cavity}, element_names=["cavity"])
    line.slice_thick_elements([xt.Strategy(xt.Uniform(3, mode="thick"))])
    structure_before = list(line.element_names)

    line.configure_cavity_model(
        model=model,
        fringe_model=(None if model == "longitudinal-only" else model),
    )

    assert line.element_names == structure_before
    line.build_tracker()  # must not raise for any model


def test_reference_energy_increase_between_sad_slices_is_rejected():
    cavity = xt.Cavity(
        length=1.270948, voltage=13696827.259102523, frequency=2.856e9,
        phase=0.8726646259971647, model="sad-track-trpt",
        fringe_model="sad-track-trpt", num_kicks=4,
    )
    line = xt.Line(elements={"cavity": cavity}, element_names=["cavity"])
    line.slice_thick_elements([xt.Strategy(xt.Uniform(2, mode="thick"))])
    # ['cavity_entry', 'cavity..0', 'cavity..1', 'cavity_exit']
    assert line.element_dict["cavity..0"].parent_name == "cavity"
    assert line.element_dict["cavity..1"].parent_name == "cavity"

    # Insert a ReferenceEnergyIncrease between the cavity's own two slices.
    line.element_dict["bad_energy_update"] = xt.ReferenceEnergyIncrease(
        Delta_p0c=1e5)
    names = list(line.element_names)
    split_at = names.index("cavity..1")
    line.element_names = (
        names[:split_at] + ["bad_energy_update"] + names[split_at:])

    with pytest.raises(ValueError, match="ReferenceEnergyIncrease"):
        line.build_tracker()


def test_reference_energy_increase_elsewhere_is_allowed():
    # A ReferenceEnergyIncrease after a fully-sliced cavity (the normal,
    # legitimate usage, e.g. sad2xs's own reference-energy bookkeeping)
    # must not be flagged.
    cavity = xt.Cavity(
        length=1.270948, voltage=13696827.259102523, frequency=2.856e9,
        phase=0.8726646259971647, model="sad-track-trpt",
        fringe_model="sad-track-trpt", num_kicks=4,
    )
    line = xt.Line(elements={"cavity": cavity}, element_names=["cavity"])
    line.slice_thick_elements([xt.Strategy(xt.Uniform(2, mode="thick"))])
    line.element_dict["energy_update"] = xt.ReferenceEnergyIncrease(
        Delta_p0c=1e5)
    line.element_names = list(line.element_names) + ["energy_update"]

    line.build_tracker()  # must not raise


def test_rs_thick_slicing_preserves_map():
    # Composition is not exact (R&S recomputes its matrix per slice from the
    # true entry energy) but the residual is ordinary discretization noise,
    # not the reference-bookkeeping bug guarded against for the SAD models.
    cavity = xt.Cavity(
        length=1.0,
        voltage=1e6,
        frequency=100e6,
        phase=0.5 * np.pi + 0.2,
        model="rosenzweig-serafini",
        fringe_model="rosenzweig-serafini",
        cavity_type="standing-wave",
    )
    unsliced = xt.Line(elements={"cavity": cavity}, element_names=["cavity"])
    sliced = unsliced.copy()
    sliced.slice_thick_elements([
        xt.Strategy(xt.Uniform(2, mode="thick")),
    ])
    particles = xp.Particles(
        p0c=100e6,
        mass0=xp.ELECTRON_MASS_EV,
        x=0.3e-3,
        px=2e-6,
        y=-0.4e-3,
        py=1e-6,
        zeta=0.2e-3,
        pzeta=5e-7,
    )
    expected = particles.copy()

    unsliced.track(expected)
    sliced.track(particles)

    for coordinate in ("x", "px", "y", "py", "zeta", "pzeta", "p0c"):
        np.testing.assert_allclose(
            getattr(particles, coordinate),
            getattr(expected, coordinate),
            rtol=0,
            atol=1e-7,
        )


def test_sagan_is_preserved_during_thin_slicing():
    line = xt.Line(
        elements={"cavity": xt.Cavity(length=1.0, model="sagan")},
        element_names=["cavity"],
    )

    line.slice_thick_elements([xt.Strategy(xt.Teapot(2))])

    assert line.element_names == ["cavity"]
    assert line["cavity"].model == "sagan"
