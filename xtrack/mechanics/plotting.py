# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #
"""Read-only survey and presentation of rigid poses."""

import numpy as np


def _display_matrices(
    reference,
    total,
    displacement_scale,
    rotation_scale,
    *,
    displacement=None,
    world_rotvec=None,
):
    from scipy.spatial.transform import Rotation

    shown = reference.copy()
    displacement = (
        total[:, :3, 3] - reference[:, :3, 3] if displacement is None else displacement
    )
    shown[:, :3, 3] += displacement_scale * displacement
    if world_rotvec is None:
        relative = np.swapaxes(reference[:, :3, :3], -1, -2) @ total[:, :3, :3]
        shown[:, :3, :3] = (
            reference[:, :3, :3]
            @ Rotation.from_rotvec(
                rotation_scale * Rotation.from_matrix(relative).as_rotvec()
            ).as_matrix()
        )
    else:
        shown[:, :3, :3] = (
            Rotation.from_rotvec(rotation_scale * world_rotvec).as_matrix()
            @ reference[:, :3, :3]
        )
    return shown


def _table_display_motion(table, baseline):
    """Scale resolved motion before adding it to large world display origins."""
    from scipy.spatial.transform import Rotation

    if baseline == 'nominal' and np.array_equal(
        table.nominal_matrix, table.reference_matrix
    ):
        baseline = 'reference'
    if baseline not in ('reference', 'static'):
        return {}
    displacement = np.column_stack([table['dynamic_' + axis] for axis in 'XYZ'])
    rotation = Rotation.from_rotvec(table.dynamic_rotvec_world)
    if baseline == 'reference':
        displacement += np.column_stack(
            [table['installation_' + axis] for axis in 'XYZ']
        )
        rotation = rotation * Rotation.from_rotvec(table.installation_rotvec_world)
    return dict(displacement=displacement, world_rotvec=rotation.as_rotvec())


def _world_points(matrix, points):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def motion_survey(
    manager,
    *,
    time_s,
    time_series=None,
    bodies=None,
    _source_values=None,
    disabled_sources=(),
):
    from .tables import (
        mechanics_table,
        pose_columns,
        deformation_columns,
        _deformation_arrays,
    )

    names = (
        list(manager.bodies)
        if bodies is None
        else ([bodies] if isinstance(bodies, str) else list(bodies))
    )
    if len(set(names)) != len(names) or not set(names).issubset(manager.bodies):
        raise ValueError('Select existing mechanics bodies')
    total = manager.get_body_poses(
        time_s=time_s,
        time_series=time_series,
        bodies=names,
        _source_values=_source_values,
        disabled_sources=disabled_sources,
    )
    disabled = tuple(manager.sources)
    static = manager.get_body_poses(
        time_s=time_s, disabled_sources=disabled, bodies=names
    )
    deliberate = manager.get_body_poses(
        time_s=time_s,
        disabled_sources=disabled,
        include_installation=False,
        bodies=names,
    )
    data = dict(
        name=np.array(names, dtype=str),
        kind=np.array([manager[n].kind for n in names], dtype=str),
        support_class=np.array([manager[n].support_class for n in names], dtype=str),
        parent=np.array([manager[n].parent or '' for n in names], dtype=str),
    )
    layers = [
        np.array([frames[n].matrix for n in names]).reshape(-1, 4, 4)
        for frames in (
            {n: manager[n].nominal_frame for n in names},
            deliberate,
            static,
            total,
        )
    ]
    data.update(pose_columns(*layers))
    data.update(
        deformation_columns(
            layers[0][:, :3, :3],
            [
                _deformation_arrays(poses, names)
                for poses in (deliberate, static, total)
            ],
        )
    )
    return mechanics_table(data, owner=manager, time_s=time_s, time_series=time_series)


def plot_motion(
    manager,
    *,
    time_s,
    time_series=None,
    bodies=None,
    line=None,
    displacement_scale=1.0,
    rotation_scale=1.0,
    view='line',
    baseline='reference',
):
    """Plot total-minus-reference translation and proper relative rotation.

    Axes are independently scaled for visibility. Rotation gain acts on the
    rotation vector, preserving orthonormal frames, never on Euler differences.
    """
    import matplotlib.pyplot as plt
    from scipy.spatial.transform import Rotation

    if (
        not np.all(np.isfinite([displacement_scale, rotation_scale]))
        or min(displacement_scale, rotation_scale) <= 0
    ):
        raise ValueError('Use finite positive display gains')
    if view not in ('line', 'bodies'):
        raise ValueError('Use line or bodies plot view')
    if baseline not in ('nominal', 'reference', 'static'):
        raise ValueError('Use nominal, reference or static plot baseline')
    survey = motion_survey(
        manager, time_s=time_s, time_series=time_series, bodies=bodies
    )
    if not len(survey):
        raise ValueError('Select at least one body to plot')
    reference = survey[baseline + '_matrix']
    total = survey.total_matrix
    display = _display_matrices(
        reference,
        total,
        displacement_scale,
        rotation_scale,
        **_table_display_motion(survey, baseline),
    )
    shown = display[:, :3, 3]
    fig = plt.figure(figsize=(10, 4.8))
    layout = fig.add_subplot(121, projection='3d')
    if line is not None and view == 'line':
        path = line.survey()
        world = manager.for_line(line).reference_frame.matrix
        xyz = np.column_stack((path.X, path.Y, path.Z)) @ world[:3, :3].T + world[:3, 3]
        layout.plot(*xyz[:, [0, 2, 1]].T, color='0.7', label='Nominal path')
    layout.scatter(
        *reference[:, :3, 3][:, [0, 2, 1]].T,
        s=10,
        color='0.6',
        label=baseline.capitalize(),
    )
    layout.scatter(*shown[:, [0, 2, 1]].T, s=14, label='Moved anchors')
    geometry = []
    index_by_name = {name: i for i, name in enumerate(survey.name)}
    for name, index in index_by_name.items():
        body = manager[name]
        if body.feet_local_m is not None:
            for foot in body.feet_local_m:
                geometry.append((index, np.array([np.zeros(3), foot])))
    if line is not None:
        line_view = manager.for_line(line)
        for attachment in line_view._attachments.values():
            if attachment['body'] not in index_by_name:
                continue
            index = index_by_name[attachment['body']]
            body = manager[attachment['body']]
            ends = np.array(
                [
                    attachment['base_element'][:3, 3],
                    attachment.get('base_element_end', attachment['base_element'])[
                        :3, 3
                    ],
                ]
            )
            local = (ends - body.nominal_frame.XYZ) @ body.nominal_frame.matrix[:3, :3]
            geometry.append((index, local))
    artists = []
    for index, local in geometry:
        original = _world_points(reference[index], local)
        moved = _world_points(display[index], local)
        layout.plot(*original[:, [0, 2, 1]].T, color='0.65', lw=1)
        (artist,) = layout.plot(*moved[:, [0, 2, 1]].T, color='tab:blue', lw=1.5)
        artists.append((index, local, artist))
    fig._mechanics_geometry = artists
    if view == 'bodies':
        vertices = [reference[:, :3, 3], shown]
        for index, local, _ in artists:
            vertices.extend(
                [
                    _world_points(reference[index], local),
                    _world_points(display[index], local),
                ]
            )
        _set_bounds(layout, np.concatenate(vertices))
    layout.set(
        xlabel='World X [m]',
        ylabel='World Z [m]',
        zlabel='World Y [m]',
        title=f'Displacement ×{displacement_scale:g}, t={time_s:g} s\nAxes independently scaled',
    )
    layout.legend(fontsize=8)
    orientation = fig.add_subplot(122, projection='3d')
    rotation = reference[0, :3, :3].T @ display[0, :3, :3]
    for i, (name, color) in enumerate(
        zip(('x', 'y', 's'), ('tab:red', 'tab:green', 'tab:blue'))
    ):
        for vector, style in ((np.eye(3)[:, i], '--'), (rotation[:, i], '-')):
            orientation.plot(
                *np.array([np.zeros(3), vector])[:, [0, 2, 1]].T, style, color=color
            )
        orientation.text(*rotation[:, i][[0, 2, 1]], name, color=color)
    orientation.set(
        xlabel='Local x',
        ylabel='Local s',
        zlabel='Local y',
        title=f'{survey.name[0]}: orientation ×{rotation_scale:g}\nDashed {baseline}; solid moved',
    )
    orientation.set_box_aspect((1, 1, 1))
    fig.tight_layout()
    return fig, (layout, orientation)


def animate_motion(
    manager,
    *,
    time_s,
    time_series=None,
    bodies=None,
    line=None,
    displacement_scale=1.0,
    rotation_scale=1.0,
    interval_ms=80,
    view='line',
    baseline='reference',
):
    from matplotlib.animation import FuncAnimation
    from scipy.spatial.transform import Rotation

    times = np.asarray(time_s, dtype=float)
    if times.ndim != 1 or not len(times) or not np.all(np.isfinite(times)):
        raise ValueError('Use finite animation frame times')
    fig, (axis, orientation) = plot_motion(
        manager,
        time_s=times[0],
        time_series=time_series,
        bodies=bodies,
        line=line,
        displacement_scale=displacement_scale,
        rotation_scale=rotation_scale,
        view=view,
        baseline=baseline,
    )
    points = axis.collections[-1]
    survey = motion_survey(
        manager, time_s=times[0], time_series=time_series, bodies=bodies
    )
    reference_frames = survey[baseline + '_matrix']
    reference = reference_frames[:, :3, 3]
    names = list(survey.name)
    # Prepass current frames for fixed display limits; never retain a pose history.
    lower, upper = reference.min(axis=0), reference.max(axis=0)
    for time in times:
        current = motion_survey(
            manager, time_s=time, time_series=time_series, bodies=names
        )
        display = _display_matrices(
            reference_frames,
            current.total_matrix,
            displacement_scale,
            rotation_scale,
            **_table_display_motion(current, baseline),
        )
        shown = display[:, :3, 3]
        lower, upper = (
            np.minimum(lower, shown.min(axis=0)),
            np.maximum(upper, shown.max(axis=0)),
        )
        for index, local, _ in fig._mechanics_geometry:
            xyz = _world_points(display[index], local)
            lower, upper = (
                np.minimum(lower, xyz.min(axis=0)),
                np.maximum(upper, xyz.max(axis=0)),
            )
    for extent in axis.lines:
        if view == 'bodies' and extent.get_label() == 'Nominal path':
            continue
        xyz = np.array(extent.get_data_3d())[[0, 2, 1]].T
        if len(xyz):
            lower, upper = (
                np.minimum(lower, xyz.min(axis=0)),
                np.maximum(upper, xyz.max(axis=0)),
            )
    padding = np.maximum(0.1 * (upper - lower), 1e-3)
    axis.set(
        xlim=(lower[0] - padding[0], upper[0] + padding[0]),
        ylim=(lower[2] - padding[2], upper[2] + padding[2]),
        zlim=(lower[1] - padding[1], upper[1] + padding[1]),
    )
    orientation.set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1), zlim=(-1.1, 1.1))

    def update(index):
        current = motion_survey(
            manager, time_s=times[index], time_series=time_series, bodies=names
        )
        display = _display_matrices(
            reference_frames,
            current.total_matrix,
            displacement_scale,
            rotation_scale,
            **_table_display_motion(current, baseline),
        )
        shown = display[:, :3, 3]
        points._offsets3d = tuple(shown[:, [0, 2, 1]].T)
        for body_index, local, artist in fig._mechanics_geometry:
            artist.set_data_3d(
                *_world_points(display[body_index], local)[:, [0, 2, 1]].T
            )
        rotation = reference_frames[0, :3, :3].T @ display[0, :3, :3]
        for coordinate in range(3):
            xyz = rotation[:, coordinate][[0, 2, 1]]
            orientation.lines[2 * coordinate + 1].set_data_3d(
                *np.array([np.zeros(3), xyz]).T
            )
            orientation.texts[coordinate].set_position_3d(xyz)
        axis.set_title(
            f'Displacement ×{displacement_scale:g}, t={times[index]:g} s\nAxes independently scaled'
        )
        return (points, *orientation.lines)

    return FuncAnimation(
        fig,
        update,
        frames=len(times),
        interval=interval_ms,
        blit=False,
        cache_frame_data=False,
    )


def _set_bounds(axis, vertices):
    lower, upper = vertices.min(axis=0), vertices.max(axis=0)
    padding = np.maximum(0.1 * (upper - lower), 1e-3)
    axis.set(
        xlim=(lower[0] - padding[0], upper[0] + padding[0]),
        ylim=(lower[2] - padding[2], upper[2] + padding[2]),
        zlim=(lower[1] - padding[1], upper[1] + padding[1]),
    )


def animate_cuboid(
    table,
    name,
    *,
    time_s,
    size_m,
    time_series=None,
    applied_history=None,
    displacement_scale=1.0,
    rotation_scale=1.0,
    baseline='static',
    interval_ms=40,
):
    """One illustrative hollow cuboid; all surfaces share one rigid transform."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    size = np.asarray(size_m)
    if (
        np.iscomplexobj(size)
        or size.shape != (3,)
        or not np.all(np.isfinite(size))
        or np.any(size <= 0)
    ):
        raise ValueError('size_m must be three positive finite dimensions in metres')
    if (
        not np.all(np.isfinite([displacement_scale, rotation_scale, interval_ms]))
        or min(displacement_scale, rotation_scale, interval_ms) <= 0
    ):
        raise ValueError('Use positive finite display gains and frame interval')
    if baseline not in ('nominal', 'reference', 'static'):
        raise ValueError('Use nominal, reference or static as display baseline')
    history = table.get_time_series(
        name, time_s=time_s, time_series=time_series, applied_history=applied_history
    )
    reference = history[baseline + '_matrix']
    shown = _display_matrices(
        reference,
        history.total_matrix,
        displacement_scale,
        rotation_scale,
        **_table_display_motion(history, baseline),
    )
    # Elements are anchored at their entrance; body presentation boxes are centered.
    owner = table._owner()
    width, height, length = size
    z0 = 0.0 if hasattr(owner, 'line') else -length / 2
    corners = np.array(
        [
            (x, y, z)
            for x in (-width / 2, width / 2)
            for y in (-height / 2, height / 2)
            for z in (z0, z0 + length)
        ]
    )
    faces = np.array(
        [
            [0, 1, 3, 2],
            [4, 6, 7, 5],
            [0, 4, 5, 1],
            [2, 3, 7, 6],
            [0, 2, 6, 4],
            [1, 5, 7, 3],
        ]
    )
    origin = reference[0, :3, 3]

    def displayed_points(matrix, local):
        return (_world_points(matrix, local) - origin)[:, [0, 2, 1]]

    # Four bars leave a rectangular beam opening and make roll unambiguous.
    blocks = []
    for low, high in (
        ((-width / 2, -height / 2), (-width * 0.23, height / 2)),
        ((width * 0.23, -height / 2), (width / 2, height / 2)),
        ((-width * 0.23, -height / 2), (width * 0.23, -height * 0.23)),
        ((-width * 0.23, height * 0.23), (width * 0.23, height / 2)),
    ):
        blocks.append(
            np.array(
                [
                    (x, y, z)
                    for x in (low[0], high[0])
                    for y in (low[1], high[1])
                    for z in (z0, z0 + length)
                ]
            )
        )
    fig = plt.figure(figsize=(8, 6.4))
    ax = fig.add_subplot(111, projection='3d')
    artists = []
    for block, color in zip(blocks, ('#236ca2', '#236ca2', '#d99035', '#d99035')):
        vertices = displayed_points(shown[0], block)
        artist = Poly3DCollection(
            vertices[faces],
            facecolor=color,
            edgecolor='#19314a',
            linewidth=0.65,
            alpha=0.93,
        )
        ax.add_collection3d(artist)
        artists.append((block, artist))
    ghost = displayed_points(reference[0], corners)
    outline = Poly3DCollection(
        ghost[faces],
        facecolor=(0, 0, 0, 0),
        edgecolor='0.65',
        linewidth=0.7,
        linestyle='--',
    )
    ax.add_collection3d(outline)
    triad = []
    triad_origin = np.array([0.0, 0.0, z0 + length / 2])
    for index, color in enumerate(('tab:red', 'tab:green', 'tab:purple')):
        local = np.array(
            [triad_origin, triad_origin + np.eye(3)[index] * max(size) * 0.7]
        )
        points = displayed_points(shown[0], local)
        (artist,) = ax.plot(
            *points.T,
            color=color,
            lw=2.2,
            label=('Local x', 'Local y', 'Local s')[index],
        )
        triad.append((local, artist))
    bounds = np.concatenate([displayed_points(matrix, corners) for matrix in shown])
    for local, _ in triad:
        bounds = np.concatenate(
            (bounds, *[displayed_points(matrix, local) for matrix in shown])
        )
    bounds = np.concatenate((bounds, ghost))
    low, high = bounds.min(axis=0), bounds.max(axis=0)
    span = max(high - low) * 1.15
    center = (low + high) / 2
    ax.set(
        xlim=(center[0] - span / 2, center[0] + span / 2),
        ylim=(center[1] - span / 2, center[1] + span / 2),
        zlim=(center[2] - span / 2, center[2] + span / 2),
        xlabel='World X relative to baseline [m]',
        ylabel='World Z relative to baseline [m]',
        zlabel='World Y relative to baseline [m]',
    )
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=22, azim=-58)
    ax.legend(loc='upper left', fontsize=8)
    fig.text(
        0.5,
        0.025,
        f'Translations ×{displacement_scale:g}; rotations ×{rotation_scale:g}; '
        f'dashed = {baseline}\nIllustrative envelope; display gains do not affect tracking',
        ha='center',
        fontsize=9,
    )
    fig.subplots_adjust(bottom=0.13, top=0.9)

    def update(index):
        for local, artist in artists:
            artist.set_verts(displayed_points(shown[index], local)[faces])
        for local, artist in triad:
            artist.set_data_3d(*displayed_points(shown[index], local).T)
        ax.set_title(
            f'{name} — six-coordinate rigid motion\nPhysical time {history.time_s[index]:.4f} s'
        )
        return tuple(artist for _, artist in artists + triad)

    update(0)
    return FuncAnimation(
        fig,
        update,
        frames=len(history),
        interval=interval_ms,
        blit=False,
        cache_frame_data=False,
    )
