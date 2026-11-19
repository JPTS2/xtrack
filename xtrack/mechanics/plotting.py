# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #
"""Read-only survey and presentation of rigid poses."""
import numpy as np

from ..table import Table


def _display_matrices(reference, total, displacement_scale, rotation_scale):
    from scipy.spatial.transform import Rotation
    shown = reference.copy()
    shown[:,:3,3] += displacement_scale*(total[:,:3,3]-reference[:,:3,3])
    relative = np.swapaxes(reference[:,:3,:3],-1,-2) @ total[:,:3,:3]
    shown[:,:3,:3] = reference[:,:3,:3] @ Rotation.from_rotvec(
        rotation_scale*Rotation.from_matrix(relative).as_rotvec()).as_matrix()
    return shown


def _world_points(matrix, points):
    return points @ matrix[:3,:3].T + matrix[:3,3]


def motion_survey(manager, *, time_s, time_series=None, bodies=None):
    from scipy.spatial.transform import Rotation
    names = list(manager.bodies) if bodies is None else ([bodies] if isinstance(bodies,str) else list(bodies))
    if not names or not set(names).issubset(manager.bodies):
        raise ValueError('Select existing mechanics bodies')
    total = manager.get_body_poses(time_s=time_s, time_series=time_series)
    disabled = tuple(manager.sources)
    static = manager.get_body_poses(time_s=time_s, disabled_sources=disabled)
    deliberate = manager.get_body_poses(time_s=time_s, disabled_sources=disabled, include_installation=False)
    data = dict(name=np.array(names), parent=np.array([manager[n].parent or '' for n in names]))
    for prefix, frames in (('nominal', {n: manager[n].nominal_frame for n in names}),
                           ('reference', deliberate), ('static', static), ('total', total)):
        for index, axis in enumerate('XYZ'):
            data[prefix+'_'+axis] = np.array([frames[n].XYZ[index] for n in names])
        for attr in ('phi', 'theta', 'psi'):
            data[prefix+'_'+attr] = np.array([getattr(frames[n], attr) for n in names])
        data[prefix+'_matrix'] = np.array([frames[n].matrix for n in names])
    for index, axis in enumerate('XYZ'):
        data['installation_'+axis] = data['static_'+axis]-data['reference_'+axis]
        data['dynamic_'+axis] = data['total_'+axis]-data['static_'+axis]
    data['dynamic_rotvec_world'] = np.array([Rotation.from_matrix(
        total[n].matrix[:3, :3] @ static[n].matrix[:3, :3].T).as_rotvec() for n in names])
    return Table(data)


def plot_motion(manager, *, time_s, time_series=None, bodies=None, line=None,
                displacement_scale=1., rotation_scale=1.,view='line',baseline='reference'):
    """Plot total-minus-reference translation and proper relative rotation.

    Axes are independently scaled for visibility. Rotation gain acts on the
    rotation vector, preserving orthonormal frames, never on Euler differences.
    """
    import matplotlib.pyplot as plt
    from scipy.spatial.transform import Rotation
    if not np.all(np.isfinite([displacement_scale, rotation_scale])) or min(displacement_scale, rotation_scale) <= 0:
        raise ValueError('Use finite positive display gains')
    if view not in ('line','bodies'):
        raise ValueError('Use line or bodies plot view')
    if baseline not in ('nominal','reference','static'):
        raise ValueError('Use nominal, reference or static plot baseline')
    survey = motion_survey(manager, time_s=time_s, time_series=time_series, bodies=bodies)
    reference = survey[baseline+'_matrix']
    total = survey.total_matrix
    display = _display_matrices(reference,total,displacement_scale,rotation_scale)
    shown = display[:,:3,3]
    fig = plt.figure(figsize=(10, 4.8))
    layout = fig.add_subplot(121, projection='3d')
    if line is not None and view == 'line':
        path = line.survey()
        world = manager.for_line(line).reference_frame.matrix
        xyz = np.column_stack((path.X, path.Y, path.Z)) @ world[:3, :3].T + world[:3, 3]
        layout.plot(*xyz[:, [0, 2, 1]].T, color='0.7', label='Nominal path')
    layout.scatter(*reference[:, :3, 3][:, [0, 2, 1]].T, s=10, color='0.6', label=baseline.capitalize())
    layout.scatter(*shown[:, [0, 2, 1]].T, s=14, label='Moved anchors')
    geometry = []
    index_by_name = {name:i for i,name in enumerate(survey.name)}
    for name, index in index_by_name.items():
        body = manager[name]
        if body.feet_local_m is not None:
            for foot in body.feet_local_m:
                geometry.append((index,np.array([np.zeros(3),foot])))
    if line is not None:
        line_view = manager.for_line(line)
        for attachment in line_view._attachments.values():
            if attachment['body'] not in index_by_name:
                continue
            index = index_by_name[attachment['body']]
            body = manager[attachment['body']]
            ends = np.array([attachment['base_element'][:3,3],
                attachment.get('base_element_end',attachment['base_element'])[:3,3]])
            local = (ends-body.nominal_frame.XYZ) @ body.nominal_frame.matrix[:3,:3]
            geometry.append((index,local))
    artists = []
    for index,local in geometry:
        original = _world_points(reference[index],local)
        moved = _world_points(display[index],local)
        layout.plot(*original[:,[0,2,1]].T,color='0.65',lw=1)
        artist, = layout.plot(*moved[:,[0,2,1]].T,color='tab:blue',lw=1.5)
        artists.append((index,local,artist))
    fig._mechanics_geometry = artists
    if view == 'bodies':
        vertices = [reference[:,:3,3],shown]
        for index,local,_ in artists:
            vertices.extend([_world_points(reference[index],local),_world_points(display[index],local)])
        _set_bounds(layout,np.concatenate(vertices))
    layout.set(xlabel='World X [m]', ylabel='World Z [m]', zlabel='World Y [m]',
               title=f'Displacement ×{displacement_scale:g}, t={time_s:g} s\nAxes independently scaled')
    layout.legend(fontsize=8)
    orientation = fig.add_subplot(122, projection='3d')
    relative_rotation = reference[0, :3, :3].T @ total[0, :3, :3]
    rotation = Rotation.from_rotvec(rotation_scale*Rotation.from_matrix(relative_rotation).as_rotvec()).as_matrix()
    for i, (name, color) in enumerate(zip(('x', 'y', 's'), ('tab:red', 'tab:green', 'tab:blue'))):
        for vector, style in ((np.eye(3)[:, i], '--'), (rotation[:, i], '-')):
            orientation.plot(*np.array([np.zeros(3), vector])[:, [0, 2, 1]].T, style, color=color)
        orientation.text(*rotation[:, i][[0, 2, 1]], name, color=color)
    orientation.set(xlabel='Local x', ylabel='Local s', zlabel='Local y',
                    title=f'{survey.name[0]}: orientation ×{rotation_scale:g}\nDashed {baseline}; solid moved')
    orientation.set_box_aspect((1, 1, 1))
    fig.tight_layout()
    return fig, (layout, orientation)


def animate_motion(manager, *, time_s, time_series=None, bodies=None, line=None,
                   displacement_scale=1., rotation_scale=1., interval_ms=80,view='line',baseline='reference'):
    from matplotlib.animation import FuncAnimation
    from scipy.spatial.transform import Rotation
    times = np.asarray(time_s, dtype=float)
    if times.ndim != 1 or not len(times) or not np.all(np.isfinite(times)):
        raise ValueError('Use finite animation frame times')
    fig, (axis, orientation) = plot_motion(manager, time_s=times[0], time_series=time_series,
        bodies=bodies, line=line, displacement_scale=displacement_scale, rotation_scale=rotation_scale,view=view,baseline=baseline)
    points = axis.collections[-1]
    survey = motion_survey(manager, time_s=times[0], time_series=time_series, bodies=bodies)
    reference_frames = survey[baseline+'_matrix']
    reference = reference_frames[:, :3, 3]
    names = list(survey.name)
    # Prepass current frames for fixed display limits; never retain a pose history.
    lower, upper = reference.min(axis=0), reference.max(axis=0)
    for time in times:
        frames = manager.get_body_poses(time_s=time, time_series=time_series)
        positions = np.array([frames[n].XYZ for n in names])
        shown = reference+displacement_scale*(positions-reference)
        lower, upper = np.minimum(lower,shown.min(axis=0)), np.maximum(upper,shown.max(axis=0))
        display = _display_matrices(reference_frames,np.array([frames[n].matrix for n in names]),
                                    displacement_scale,rotation_scale)
        for index,local,_ in fig._mechanics_geometry:
            xyz = _world_points(display[index],local)
            lower,upper = np.minimum(lower,xyz.min(axis=0)),np.maximum(upper,xyz.max(axis=0))
    for extent in axis.lines:
        if view == 'bodies' and extent.get_label() == 'Nominal path':
            continue
        xyz = np.array(extent.get_data_3d())[[0,2,1]].T
        if len(xyz):
            lower, upper = np.minimum(lower,xyz.min(axis=0)), np.maximum(upper,xyz.max(axis=0))
    padding = np.maximum(.1*(upper-lower), 1e-3)
    axis.set(xlim=(lower[0]-padding[0],upper[0]+padding[0]),
             ylim=(lower[2]-padding[2],upper[2]+padding[2]),
             zlim=(lower[1]-padding[1],upper[1]+padding[1]))
    orientation.set(xlim=(-1.1,1.1),ylim=(-1.1,1.1),zlim=(-1.1,1.1))

    def update(index):
        frames = manager.get_body_poses(time_s=times[index], time_series=time_series)
        positions = np.array([frames[n].XYZ for n in names])
        shown = reference+displacement_scale*(positions-reference)
        points._offsets3d = tuple(shown[:, [0, 2, 1]].T)
        display = _display_matrices(reference_frames,np.array([frames[n].matrix for n in names]),
                                    displacement_scale,rotation_scale)
        for body_index,local,artist in fig._mechanics_geometry:
            artist.set_data_3d(*_world_points(display[body_index],local)[:,[0,2,1]].T)
        relative = reference_frames[0, :3, :3].T @ frames[names[0]].matrix[:3, :3]
        rotation = Rotation.from_rotvec(rotation_scale*Rotation.from_matrix(relative).as_rotvec()).as_matrix()
        for index in range(3):
            xyz = rotation[:, index][[0,2,1]]
            orientation.lines[2*index+1].set_data_3d(*np.array([np.zeros(3),xyz]).T)
            orientation.texts[index].set_position_3d(xyz)
        axis.set_title(f'Displacement ×{displacement_scale:g}, t={times[index]:g} s\nAxes independently scaled')
        return (points, *orientation.lines)

    return FuncAnimation(fig, update, frames=len(times), interval=interval_ms, blit=False, cache_frame_data=False)


def _set_bounds(axis,vertices):
    lower,upper = vertices.min(axis=0),vertices.max(axis=0)
    padding = np.maximum(.1*(upper-lower),1e-3)
    axis.set(xlim=(lower[0]-padding[0],upper[0]+padding[0]),
             ylim=(lower[2]-padding[2],upper[2]+padding[2]),
             zlim=(lower[1]-padding[1],upper[1]+padding[1]))
