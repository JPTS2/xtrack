# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

# Local visual review: a rigid girder, two quadrupoles and an intervening bend.
# Magnify the girder and the independent magnet/girder motion hierarchically.

################################################################################
# Required packages
################################################################################
from pathlib import Path
from time import perf_counter
import hashlib
import json
import resource
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import xtrack as xt

from girder_assembly_view import animate_assembly

################################################################################
# User parameters
################################################################################

########################################
# Exact line layout and illustrative bend angle
########################################
QUAD_LENGTH_M          = .5
DRIFT_LENGTH_M         = .5
BEND_LENGTH_M          = 1.5
BEND_ANGLE_RAD         = .06          # Reference bend angle; not a girder flexure
DELIBERATE_Q1_X_M      = 1e-3
DELIBERATE_Q1_ROLL_RAD = .08

########################################
# Rigid girder and magnet display dimensions
########################################
GIRDER_SIZE_M   = (.7, .22, 3.7)                             # Width, height, length; includes end overhang
GIRDER_HEIGHT_M = -.6                                        # Girder anchor below the beam [m]
QUAD_SIZE_M     = (.44, .44, QUAD_LENGTH_M)
BEND_SIZE_M     = (.58, .50, BEND_LENGTH_M)
FEET_LOCAL_M    = [(-.25, -.35, -1.65), (.25, -.35, -1.65),
                (-.25, -.35, 1.65), (.25, -.35, 1.65)]

########################################
# Independent installation placement
########################################
GIRDER_INSTALLATION_Y_M     = 20e-6
GIRDER_INSTALLATION_YAW_RAD = 2e-6
MAGNET_INSTALLATION_Y_M     = (3e-6, -5e-6, 4e-6)

########################################
# Dominant slow common motion: one girder cycle over the four-second record
########################################
GIRDER_DRIVES = {
    'x': (40e-9, .25, .3), 'y': (180e-9, .25, 0.), 's': (15e-9, .25, .7),
    'pitch': (60e-9, .25, .4), 'yaw': (45e-9, .25, .8), 'roll': (70e-9, .25, .2),
}

########################################
# Smaller fast relative motion: eight to twelve cycles over the same record
########################################
MAGNET_DRIVES = {
    'q1': {'y': (14e-9, 3., .2), 'x': (3e-9, 2., .5),
           'pitch': (8e-9, 2., .1), 'roll': (10e-9, 3., .6)},
    'bend': {'y': (20e-9, 2., 1.2), 'x': (4e-9, 2.5, .3),
             'pitch': (10e-9, 2.5, .8), 'yaw': (10e-9, 2., .4)},
    'q2': {'y': (16e-9, 2.5, 2.1), 'x': (3e-9, 3., .9),
           'pitch': (12e-9, 2.5, .7), 'roll': (12e-9, 2., 1.1)},
}

########################################
# Physical time and display magnification
########################################
DURATION_S     = 4.
SAMPLE_STEP_S  = 1 / 128
N_FRAMES       = 100
GIF_FPS        = 25
DISPLAY_SCALES = dict(
    # The same gain preserves the common/relative amplitude hierarchy.
    girder_translation_scale = 1e6, girder_rotation_scale = 1e6,
    magnet_translation_scale = 1e6, magnet_rotation_scale = 1e6,
)

########################################
# Local review output
########################################
OUTPUT_DIR   = Path(__file__).resolve().parents[1] / 'results' / '015_girder_assembly'
SHOW_FIGURES = False

################################################################################
# Build the ordinary line and its reference survey
################################################################################

########################################
# Set up timing and local artifact output
########################################
timings = {}
OUTPUT_DIR.mkdir(parents = True, exist_ok = True)
started = perf_counter()

########################################
# Construct the requested quad / drift / bend / drift / quad sequence
########################################
env = xt.Environment()
env.new('q1', xt.Quadrupole, length = QUAD_LENGTH_M, k1 = .2,
        shift_x = DELIBERATE_Q1_X_M, rot_s_rad = DELIBERATE_Q1_ROLL_RAD)
env.new('d1', xt.Drift, length = DRIFT_LENGTH_M)
env.new('bend', xt.Bend, length = BEND_LENGTH_M, angle = BEND_ANGLE_RAD,
        k0 = BEND_ANGLE_RAD / BEND_LENGTH_M)
env.new('d2', xt.Drift, length = DRIFT_LENGTH_M)
env.new('q2', xt.Quadrupole, length = QUAD_LENGTH_M, k1 = -.2)
line = env.new_line(name = 'assembly', components = ['q1', 'd1', 'bend', 'd2', 'q2'])
line.set_particle_ref('proton', p0c = 7e12)
timings['line_construction_s'] = perf_counter() - started

########################################
# Prepare the tracker explicitly
########################################
started = perf_counter()
line.build_tracker()
timings['tracker_build_s'] = perf_counter() - started

########################################
# Adopt the deliberate magnetic inputs
########################################
started = perf_counter()
line.mechanics.adopt_reference()
timings['reference_adoption_s'] = perf_counter() - started

########################################
# Inspect the static reference ports
########################################
started = perf_counter()
survey = line.survey(include_element_frames = True)
timings['reference_survey_s'] = perf_counter() - started
mech = env.mechanics

################################################################################
# Define a rigid girder and three independently moving magnet bodies
################################################################################

########################################
# Locate the girder along the reference chord, below the beam
########################################
started = perf_counter()
entrance = survey.get_frame('q1', which = 'ref_start')
exit = survey.get_frame('q2', which = 'ref_end')
matrix = entrance.matrix.copy()
half_rotation = Rotation.from_matrix(entrance.matrix[:3, :3].T @ exit.matrix[:3, :3]).as_rotvec() / 2
matrix[:3, :3] = entrance.matrix[:3, :3] @ Rotation.from_rotvec(half_rotation).as_matrix()
matrix[:3, 3] = (entrance.XYZ + exit.XYZ) / 2
matrix[1, 3] = GIRDER_HEIGHT_M
girder = mech.new_girder('girder', anchor = xt.Frame(matrix), feet_local_m = FEET_LOCAL_M)

########################################
# Anchor each magnet at its surveyed rigid centre
########################################
magnets = {}
for name in ('q1', 'bend', 'q2'):
    first = survey.get_frame(name, which = 'elem_start')
    last = survey.get_frame(name, which = 'elem_end')
    matrix = first.matrix.copy()
    matrix[:3, 3] = (first.XYZ + last.XYZ) / 2
    half_rotation = Rotation.from_matrix(first.matrix[:3, :3].T @ last.matrix[:3, :3]).as_rotvec() / 2
    matrix[:3, :3] = first.matrix[:3, :3] @ Rotation.from_rotvec(half_rotation).as_matrix()
    magnets[name] = mech.new_body(name, kind = 'magnet', parent = girder,
                                  anchor = xt.Frame(matrix), elements = [name], line = line)
timings['mechanics_definition_s'] = perf_counter() - started

########################################
# Keep installation separate from both deliberate and dynamic placement
########################################
girder.installation.shift_y = GIRDER_INSTALLATION_Y_M
girder.installation.rot_y_rad = GIRDER_INSTALLATION_YAW_RAD
for name, offset in zip(magnets, MAGNET_INSTALLATION_Y_M):
    magnets[name].installation.shift_y = offset

################################################################################
# Prescribe common girder motion and additional relative magnet motion
################################################################################

########################################
# Register the six-coordinate girder inputs
########################################
started = perf_counter()
for component, (amplitude, frequency, phase) in GIRDER_DRIVES.items():
    mech.add_source('girder.' + component, xt.mechanics.Harmonic(
        amplitude, frequency, phase, unit = 'm' if component in ('x', 'y', 's') else 'rad',
    ), targets = ['girder'], component = component)

########################################
# Add each magnet's own inputs with relative inheritance
########################################
for name, drives in MAGNET_DRIVES.items():
    for component, (amplitude, frequency, phase) in drives.items():
        mech.add_source(name + '.' + component, xt.mechanics.Harmonic(
            amplitude, frequency, phase, unit = 'm' if component in ('x', 'y', 's') else 'rad',
        ), targets = [name], component = component, reference = 'relative')
timings['source_definition_s'] = perf_counter() - started

########################################
# Compute one common physical-time source record
########################################
started = perf_counter()
times = np.arange(round(DURATION_S / SAMPLE_STEP_S) + 1) * SAMPLE_STEP_S
motion = mech.compute_time_series(time_s = times)
timings['sample_motion_s'] = perf_counter() - started

################################################################################
# Query native tables and check the actual element lowering
################################################################################

########################################
# Read each body's composed history through its selected table row
########################################
started = perf_counter()
bodies = mech.get_table(time_s = 0., time_series = motion)
full_histories = {name: bodies.rows[name].get_time_series() for name in bodies.name}
frame_times = np.linspace(0., DURATION_S, N_FRAMES, endpoint = False)
histories = {name: bodies.rows[name].get_time_series(time_s = frame_times) for name in bodies.name}
timings['selected_histories_s'] = perf_counter() - started

########################################
# Compare requested and actual frames at an interior time
########################################
check_time = .375
requested = line.mechanics.get_table(time_s = check_time, time_series = motion)
mech.apply(time_s = check_time, time_series = motion)
actual = line.survey(include_element_frames = True)
for name in magnets:
    np.testing.assert_allclose(requested['total_matrix', name],
        actual.get_frame(name, which = 'elem_start').matrix, rtol = 0, atol = 2e-13)
mech.restore()

################################################################################
# Supply illustrative geometry to the local presentation helper
################################################################################

########################################
# Keep every magnet envelope rigid, including the dipole
########################################
elements = {}
for name, color, label in (('q1', '#2878b5', 'Q1 · 0.5 m'),
                            ('bend', '#269479', 'Bend · 1.5 m'), ('q2', '#b85a43', 'Q2 · 0.5 m')):
    first = survey.get_frame(name, which = 'elem_start')
    if name == 'bend':
        angle = np.linspace(0., BEND_ANGLE_RAD, 41)
        radius = BEND_LENGTH_M / BEND_ANGLE_RAD
        path = np.column_stack((radius * (np.cos(angle) - 1), np.zeros(len(angle)), radius * np.sin(angle)))
    else:
        path = np.array([[0., 0., 0.], [0., 0., QUAD_LENGTH_M]])
    world_path = path @ first.matrix[:3, :3].T + first.XYZ
    local_path = (world_path - magnets[name].nominal_frame.XYZ) @ magnets[name].nominal_frame.matrix[:3, :3]
    elements[name] = dict(size_m = BEND_SIZE_M if name == 'bend' else QUAD_SIZE_M,
        color = color, label = label, short_label = 'Bend' if name == 'bend' else name.upper(),
        type = 'Bend' if name == 'bend' else 'Quadrupole', beam_local_m = local_path)
    np.testing.assert_allclose(world_path[-1], survey.get_frame(name, which = 'elem_end').XYZ, atol = 1e-12)

########################################
# Compose magnified parent and relative child motion before drawing
########################################
started = perf_counter()
animation, figure, shown = animate_assembly(
    histories, full_histories, girder_name = 'girder', elements = elements,
    girder_size_m = GIRDER_SIZE_M, feet_local_m = np.array(FEET_LOCAL_M),
    scales = DISPLAY_SCALES, fps = GIF_FPS,
    title = 'Slow common girder sway · small fast motion of the mounted magnets',
)
timings['animation_preparation_s'] = perf_counter() - started

########################################
# Check that display magnification kept every body rigid
########################################
for matrices in shown.values():
    rotation = matrices[:, :3, :3]
    np.testing.assert_allclose(np.swapaxes(rotation, -1, -2) @ rotation,
                               np.broadcast_to(np.eye(3), rotation.shape), atol = 2e-14)

########################################
# Verify that common inherited motion dominates the displayed residual
########################################
hierarchy_ratios = {}
parent_static_inverse = np.linalg.inv(histories['girder'].static_matrix)
for name in magnets:
    carried = shown['girder'] @ parent_static_inverse @ histories[name].static_matrix
    common = np.linalg.norm(carried[:, :3, 3] - histories[name].static_matrix[:, :3, 3], axis = 1)
    relative = np.linalg.norm(shown[name][:, :3, 3] - carried[:, :3, 3], axis = 1)
    hierarchy_ratios[name] = float(np.max(common) / np.max(relative))
    assert hierarchy_ratios[name] > 4.

################################################################################
# Render the local review artifacts
################################################################################

########################################
# Measure GIF rendering and encoding separately
########################################
started = perf_counter()
gif_file = OUTPUT_DIR / 'girder_assembly.gif'
animation.save(gif_file, writer = 'pillow', fps = GIF_FPS, dpi = 90)
timings['gif_render_and_encode_s'] = perf_counter() - started
figure.savefig(OUTPUT_DIR / 'assembly.png', dpi = 140)
svg_file = OUTPUT_DIR / 'assembly.svg'
figure.savefig(svg_file)
svg_file.write_text('\n'.join(row.rstrip() for row in svg_file.read_text().splitlines()) + '\n')
with Image.open(gif_file) as gif:
    assert gif.n_frames == N_FRAMES
    resolution = gif.size
    playback_duration_s = 0.
    for index in range(gif.n_frames):
        gif.seek(index)
        playback_duration_s += gif.info['duration'] / 1000

########################################
# Report the native model, costs and unchanged reference inputs
########################################
assert line.get_length() == 3.5
np.testing.assert_equal(env['q1'].shift_x, DELIBERATE_Q1_X_M)
np.testing.assert_equal(env['q1'].rot_s_rad, DELIBERATE_Q1_ROLL_RAD)
print('Local rigid-girder assembly review\n----------------------------------')
print('Layout: Q1 0.5 m | drift 0.5 m | bend 1.5 m | drift 0.5 m | Q2 0.5 m')
print(f'Line: {line.get_length():g} m; girder: {GIRDER_SIZE_M[2]:g} m including overhang; bend: {BEND_ANGLE_RAD:g} rad')
print(f'Native bodies: {len(mech.bodies)}; registered inputs: {len(mech.sources)}')
print(f'Physical duration: {DURATION_S:g} s; source samples: {len(times)}; GIF: {N_FRAMES} frames at {GIF_FPS} fps')
print('Girder and all magnets remain rigid; each magnet has additional relative motion.')
print('Common girder: 0.25 Hz, 180 nm peak vertical input; relative magnets: 2–3 Hz, 14–20 nm peak y.')
print('All translation and rotation display gains are 1,000,000; relative motion is not enlarged separately.')
print('Peak common / relative displacement ratios at magnet anchors:', hierarchy_ratios)
print('Moving wire outlines show each magnet carried rigidly by the girder, before its relative motion.')
print('Orange links are presentation geometry; no force/elasticity model is inferred.')
print(f'GIF: {gif_file}')
print(f'GIF dimensions: {resolution[0]} × {resolution[1]}; playback duration: {playback_duration_s:g} s')
print('\nMeasured operation costs\n------------------------')
for name, seconds in timings.items():
    print(f'{name:<32} {seconds:10.6f} s')
peak_rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2 if sys.platform == 'darwin' else 1024)
print(f'Peak process RSS including GIF encoder [MiB]: {peak_rss_mib:.1f}')
(OUTPUT_DIR / 'result.json').write_text(json.dumps(dict(timings = timings, peak_rss_mib = peak_rss_mib,
    frames = N_FRAMES, duration_s = DURATION_S, display_scales = DISPLAY_SCALES,
    peak_common_to_relative_displacement = hierarchy_ratios,
    playback_duration_s = playback_duration_s, gif_resolution = resolution, xtrack = xt.__version__,
    script_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    helper_sha256 = hashlib.sha256(Path(__file__).with_name('girder_assembly_view.py').read_bytes()).hexdigest(),
    mechanics_source_sha256 = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (Path(xt.__file__).parent / 'mechanics').glob('*.py')},
    layout_m = [QUAD_LENGTH_M, DRIFT_LENGTH_M, BEND_LENGTH_M, DRIFT_LENGTH_M, QUAD_LENGTH_M]), indent = 2) + '\n')

########################################
# Keep interactive display optional for this heavier local review
########################################
if SHOW_FIGURES:
    plt.show()
else:
    plt.close('all')
