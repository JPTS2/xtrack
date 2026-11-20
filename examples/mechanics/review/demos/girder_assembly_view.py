"""Presentation helpers for the local assembly review; no tracking writes."""
import numpy as np
from scipy.spatial.transform import Rotation


def magnify(reference, total, translation_scale, rotation_scale):
    shown = reference.copy()
    shown[:, :3, 3] += translation_scale * (total[:, :3, 3] - reference[:, :3, 3])
    relative = np.swapaxes(reference[:, :3, :3], -1, -2) @ total[:, :3, :3]
    shown[:, :3, :3] = reference[:, :3, :3] @ Rotation.from_rotvec(
        rotation_scale * Rotation.from_matrix(relative).as_rotvec()).as_matrix()
    return shown


def display_hierarchy(histories, *, girder_name, girder_translation_scale,
                      girder_rotation_scale, magnet_translation_scale, magnet_rotation_scale):
    """Magnify the parent and each relative child transform before composition.

    Independently magnifying world positions/angles would separate rigid mounts
    even when the magnets had no relative motion. This preserves the hierarchy.
    """
    parent = histories[girder_name]
    displayed = {girder_name: magnify(parent.static_matrix, parent.total_matrix,
                                     girder_translation_scale, girder_rotation_scale)}
    parent_static_inverse = np.linalg.inv(parent.static_matrix)
    parent_total_inverse = np.linalg.inv(parent.total_matrix)
    for name, history in histories.items():
        if name == girder_name:
            continue
        reference = parent_static_inverse @ history.static_matrix
        actual = parent_total_inverse @ history.total_matrix
        relative = magnify(reference, actual, magnet_translation_scale, magnet_rotation_scale)
        displayed[name] = displayed[girder_name] @ relative
    return displayed


def vertices(size, center=(0., 0., 0.)):
    half = np.asarray(size) / 2
    return np.array([(x, y, z) for x in (-half[0], half[0])
                     for y in (-half[1], half[1]) for z in (-half[2], half[2])]) + center


FACES = np.array([[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1],
                   [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]])


def animate_assembly(histories, full_histories, *, girder_name, elements, girder_size_m,
                      feet_local_m, scales, fps, title):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d import proj3d
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    shown = display_hierarchy(histories, girder_name=girder_name, **scales)
    times = histories[girder_name].time_s
    origin = histories[girder_name].static_matrix[0, :3, 3].copy()
    origin[1] = 0.

    def points(matrix, local):
        world = local @ matrix[:3, :3].T + matrix[:3, 3] - origin
        return world[:, [2, 0, 1]]  # Longitudinal world Z across the screen

    fig = plt.figure(figsize=(12.8, 6.8), layout='constrained')
    grid = fig.add_gridspec(2, 2, height_ratios=(4.8, 1.), width_ratios=(4., 1.))
    ax = fig.add_subplot(grid[0, :], projection='3d')
    plot = fig.add_subplot(grid[1, 0])
    status = fig.add_subplot(grid[1, 1])
    status.set_axis_off()
    fig.suptitle(title, fontsize=13, fontweight='bold')
    # Axes3D normally forces a square viewport, wasting a landscape canvas.
    # Keep a rectangular viewport and compensate its projection so physical
    # world scales stay equal. This is local presentation code only.
    ax.apply_aspect = lambda position=None: None
    original_projection = ax.get_proj
    projection_scale = 1.
    projection_offset = np.zeros(2)

    def wide_projection():
        projection = original_projection()
        position = ax.get_position(original=True)
        ratio = position.width * fig.get_figwidth() / (position.height * fig.get_figheight())
        projection[0] /= ratio
        projection[0] = projection_scale * projection[0] + projection_offset[0] * projection[3]
        projection[1] = projection_scale * projection[1] + projection_offset[1] * projection[3]
        return projection

    ax.get_proj = wide_projection
    surfaces, lines, labels = [], [], []
    bounds = []

    def box(name, local, color, alpha=1.):
        first = points(shown[name][0], local)
        artist = Poly3DCollection(first[FACES], facecolor=color, edgecolor='#243344',
                                  linewidth=.55, alpha=alpha, zsort='average')
        ax.add_collection3d(artist)
        surfaces.append((name, local, artist))
        bounds.extend(points(matrix, local) for matrix in shown[name])

    def moving_line(name, local, **kwargs):
        artist, = ax.plot(*points(shown[name][0], local).T, **kwargs)
        lines.append((name, local, artist))
        return artist

    ########################################
    # Draw the rigid girder and its legs
    ########################################
    box(girder_name, vertices(girder_size_m), '#637180')
    for foot in feet_local_m:
        top = np.array([foot[0], -girder_size_m[1] / 2, foot[2]])
        center = (top + foot) / 2
        box(girder_name, vertices((.12, abs(top[1] - foot[1]), .15), center), '#8795a3')
        box(girder_name, vertices((.24, .045, .25), foot), '#475868')

    ########################################
    # Draw fixed envelopes for the installed baseline
    ########################################
    for name, info in elements.items():
        corners = vertices(info['size_m'])
        ghost = points(histories[name].static_matrix[0], corners)
        artist = Poly3DCollection(ghost[FACES], facecolor=(0, 0, 0, 0), edgecolor='0.72',
                                  linewidth=.65, linestyle='--')
        ax.add_collection3d(artist)

    ########################################
    # Give the quadrupoles an opening and the dipole a two-pole gap
    ########################################
    for name, info in elements.items():
        width, height, length = info['size_m']
        if info['type'] == 'Quadrupole':
            for center_x in (-width * .365, width * .365):
                box(name, vertices((width * .27, height, length), (center_x, 0., 0.)), info['color'])
            for center_y in (-height * .365, height * .365):
                box(name, vertices((width * .46, height * .27, length), (0., center_y, 0.)), '#dc9f46')
        else:
            for center_y in (-height * .34, height * .34):
                box(name, vertices((width, height * .32, length), (0., center_y, 0.)), info['color'])
            box(name, vertices((width * .18, height * .36, length), (width * .41, 0., 0.)), '#338974')
        moving_line(name, info['beam_local_m'], color='#e5b648', lw=1.6, alpha=.8)
        label = ax.text2D(0., 0., info['label'], ha='center', color=info['color'], fontsize=10,
                           fontweight='bold', zorder=1000, transform=ax.transData,
                           bbox=dict(facecolor='white', edgecolor='none', alpha=.7, pad=1.))
        labels.append((name, np.array([[0., height / 2 + .1, 0.]]), label))
        bounds.extend(points(matrix, np.array([[0., height / 2 + .1, 0.]])) for matrix in shown[name])

    ########################################
    # Show two compliant mount links below each independent magnet
    ########################################
    parent_reference = np.linalg.inv(histories[girder_name].static_matrix[0])
    mounts = []
    for name, info in elements.items():
        width, height, length = info['size_m']
        for longitudinal in (-length * .3, length * .3):
            upper = np.array([0., -height / 2, longitudinal])
            world = histories[name].static_matrix[0] @ np.r_[upper, 1.]
            anchor = (parent_reference @ world)[:3]
            anchor[1] = girder_size_m[1] / 2
            box(girder_name, vertices((.22, .035, .22), anchor), '#b6c0c9')
            artist, = ax.plot([], [], [], color='#d47d29', lw=1.4)
            mounts.append((name, anchor, upper, artist))

    ########################################
    # Add a static floor and readable world dimensions
    ########################################
    floor_y = -.6
    floor = np.array([[-1.95, -.55, floor_y], [1.95, -.55, floor_y],
                      [1.95, .55, floor_y], [-1.95, .55, floor_y]])
    floor[:, 2] += histories[girder_name].static_matrix[0, 1, 3]
    ax.add_collection3d(Poly3DCollection([floor], facecolor='#edf0f3', edgecolor='#c2cbd2', alpha=.55))
    ground_links = []
    for foot in feet_local_m:
        anchor = points(histories[girder_name].static_matrix[0], foot[None, :])[0]
        anchor[2] = floor[0, 2]
        artist, = ax.plot([], [], [], color='#b77a3b', lw=1.1)
        ground_links.append((anchor, foot, artist))
    bounds = np.concatenate(bounds + [floor])
    low, high = bounds.min(axis=0), bounds.max(axis=0)
    padding = np.array([.04, .04, .06])
    low -= padding
    high += padding
    ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), zlim=(low[2], high[2]),
           xlabel='World Z relative to assembly centre [m]', ylabel='World X [m]', zlabel='World Y [m]')
    ax.set_box_aspect(high - low)
    ax.view_init(elev=18, azim=-67)
    ax.set_axis_off()

    ########################################
    # Compare each magnet with its moving girder in physical units
    ########################################
    full_parent = full_histories[girder_name]
    parent_actual_inverse = np.linalg.inv(full_parent.total_matrix)
    parent_static_inverse = np.linalg.inv(full_parent.static_matrix)
    relative_y = {}
    for name, info in elements.items():
        history = full_histories[name]
        actual = parent_actual_inverse @ history.total_matrix
        reference = parent_static_inverse @ history.static_matrix
        relative_y[name] = (actual[:, 1, 3] - reference[:, 1, 3]) * 1e9
        plot.plot(history.time_s, relative_y[name], color=info['color'], label=info['short_label'], lw=1.1)
    cursor = plot.axvline(times[0], color='0.2', lw=1.)
    plot.set(xlim=(full_parent.time_s[0], full_parent.time_s[-1]),
             xlabel='Physical time [s]', ylabel='Relative y [nm]')
    plot.tick_params(labelsize=7)
    plot.xaxis.label.set_size(8)
    plot.yaxis.label.set_size(8)
    plot.grid(alpha=.2)
    plot.legend(loc='upper right', ncol=3, fontsize=7)
    clock = status.text(.02, .98, '', fontsize=9, family='monospace', va='top', transform=status.transAxes)
    status.text(.02, .13, f"Display T/R: girder ×{scales['girder_translation_scale']:g}/{scales['girder_rotation_scale']:g}\n"
                f"Magnet/girder ×{scales['magnet_translation_scale']:g}/{scales['magnet_rotation_scale']:g}",
                fontsize=6.5, va='top', transform=status.transAxes)

    def update(index):
        for name, local, artist in surfaces:
            artist.set_verts(points(shown[name][index], local)[FACES])
        for name, local, artist in lines:
            artist.set_data_3d(*points(shown[name][index], local).T)
        for name, local, label in labels:
            position = points(shown[name][index], local)[0]
            projected_x, projected_y, _ = proj3d.proj_transform(*position, ax.get_proj())
            label.set_position((projected_x, projected_y))
        for name, lower, upper, artist in mounts:
            endpoints = np.vstack((points(shown[girder_name][index], lower[None, :]),
                                    points(shown[name][index], upper[None, :])))
            # A zigzag is a visual link, not an inferred spring-force model.
            fraction = np.linspace(0., 1., 13)
            zigzag = endpoints[0] + fraction[:, None] * (endpoints[1] - endpoints[0])
            zigzag[1:-1, 1] += .018 * (-1.) ** np.arange(11)
            artist.set_data_3d(*zigzag.T)
        for anchor, foot, artist in ground_links:
            upper = points(shown[girder_name][index], foot[None, :])[0]
            fraction = np.linspace(0., 1., 13)
            zigzag = anchor + fraction[:, None] * (upper - anchor)
            zigzag[1:-1, 0] += .025 * (-1.) ** np.arange(11)
            artist.set_data_3d(*zigzag.T)
        cursor.set_xdata([times[index], times[index]])
        text = [f't = {times[index]:5.2f} s; y [nm]']
        for name, info in elements.items():
            value = np.interp(times[index], full_parent.time_s, relative_y[name])
            text.append(f"{info['short_label']:<5} {value:7.2f}")
        clock.set_text('\n'.join(text))
        return tuple(artist for _, _, artist in surfaces + lines) + (cursor, clock)

    update(0)
    # Lay out genuine subplots, then fit the complete swept assembly once into
    # its own panel. A fixed fit prevents clipping or camera changes per frame.
    fig.canvas.draw()
    bbox = ax.get_window_extent()
    target = ax.transData.inverted().transform(np.array([
        [bbox.x0 + .035*bbox.width, bbox.y0 + .07*bbox.height],
        [bbox.x1 - .035*bbox.width, bbox.y1 - .09*bbox.height],
    ]))
    xx, yy, _ = proj3d.proj_transform(*bounds.T, ax.get_proj())
    projected = np.column_stack((xx, yy))
    projected_low, projected_high = projected.min(axis=0), projected.max(axis=0)
    projection_scale = np.min((target[1]-target[0]) / (projected_high-projected_low))
    projection_offset = (target[0]+target[1])/2 - projection_scale*(projected_low+projected_high)/2
    fig.set_layout_engine(None)
    xx, yy, _ = proj3d.proj_transform(*bounds.T, ax.get_proj())
    fitted_pixels = ax.transData.transform(np.column_stack((xx, yy)))
    assert np.all(fitted_pixels.min(axis=0) >= np.array([bbox.x0, bbox.y0]) - 1e-6)
    assert np.all(fitted_pixels.max(axis=0) <= np.array([bbox.x1, bbox.y1]) + 1e-6)
    animation = FuncAnimation(fig, update, frames=len(times), interval=1000/fps,
                               cache_frame_data=False, blit=False)
    return animation, fig, shown
