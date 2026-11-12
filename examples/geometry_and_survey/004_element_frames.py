# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt
from xtrack.survey import SurveyTable

################################################################################
# User parameters
################################################################################

########################################
# Reference geometry
########################################
LINE_LENGTH     = 4.0                    # [m]
QUAD_START      = 1.0                    # Entrance s [m]
QUAD_LENGTH     = 2.0                    # [m]
N_SLICES        = 4

########################################
# Element alignment and local point
########################################
ANCHOR_S        = 0.75                   # Pivot distance from entry [m]
SHIFT_X         = 0.10                   # Horizontal translation [m]
YAW             = 0.20                   # Rotation about local y [rad]
LOCAL_POINT     = np.array([0.02, 0.0, 0.5])  # Local (x, y, s) [m]

########################################
# Output file
########################################
SAVE_FILE       = 'outputs/element_frames.json'

################################################################################
# Build a straight reference line with a displaced, yawed quadrupole
################################################################################
env = xt.Environment()
line = env.new_line(
    length      = LINE_LENGTH,
    refer       = 'start',
    components  = [
        env.new(
            'q', xt.Quadrupole,
            at                  = QUAD_START,
            length              = QUAD_LENGTH,
            rot_shift_anchor    = ANCHOR_S,
            shift_x             = SHIFT_X,
            rot_y_rad           = YAW,
        ),
    ],
)
# rot_shift_anchor is a distance in metres from the entrance, not a fraction.
# For this straight element the shifted pivot stays at reference pivot + shift;
# yaw rotates the actual element about that pivot. The reference line stays put.

################################################################################
# Inspect reference and actual element frames
################################################################################

########################################
# Extract all four frames
########################################
survey      = line.survey(include_element_frames = True)
frames      = survey.get_all_frames('q')
ref_start   = frames['ref_start']
ref_end     = frames['ref_end']
elem_start  = frames['elem_start']
elem_end    = frames['elem_end']

########################################
# Compare frame origins and misalignment pivots
########################################
print('Frame origins in global survey (X, Y, Z) [m]:')
for name, frame in frames.items():
    print(f'{name:12s}', frame.XYZ)
reference_pivot = ref_start.XYZ + ANCHOR_S * ref_start.es
actual_pivot    = elem_start.XYZ + ANCHOR_S * elem_start.es
print('Reference pivot [m]:', reference_pivot)
print('Actual pivot [m]:   ', actual_pivot)

################################################################################
# Transform a local point and convert an actual frame to CCS
################################################################################

########################################
# Map the same local point to global coordinates and back
########################################
# Frame axes are orthonormal global vectors: ex, ey, es map local (x, y, s).
basis           = np.column_stack([elem_start.ex, elem_start.ey, elem_start.es])
global_point    = elem_start.XYZ + basis @ LOCAL_POINT
recovered_local = basis.T @ (global_point - elem_start.XYZ)
print('Local point (x, y, s) [m]:', LOCAL_POINT)
print('Global point (X, Y, Z) [m]:', global_point)
print('Recovered local point [m]:', recovered_local)

########################################
# Express the actual entrance frame in CCS
########################################
# CCS positions use (x, y, z) = (-X, Z, Y): initial reference-forward y, up z;
# the CCS horizontal survey angle is in gon (400 gon = 2*pi). Its phi and psi
# angles remain in radians.
elem_start_ccs = elem_start.to_ccs()
print('Actual entrance in CCS:', elem_start_ccs)

################################################################################
# Preserve the element frames through thick slicing
################################################################################

########################################
# Slice a separate copy of the same aligned element
########################################
sliced = line.copy()
sliced.slice_thick_elements([
    xt.Strategy(xt.Uniform(N_SLICES, mode = 'thick'), name = 'q'),
])
sliced_survey = sliced.survey(include_element_frames = True)
slice_names = [name for name in sliced.element_names
               if isinstance(sliced.get(name), xt.ThickSliceQuadrupole)]
first_slice = sliced_survey.get_all_frames(slice_names[0])
last_slice  = sliced_survey.get_all_frames(slice_names[-1])
sliced_survey.rows[slice_names].cols[
    'name s X_elem_start Z_elem_start X_elem_end Z_elem_end'].show()

########################################
# Compare the physical entrance and exit frames
########################################
for label, parent_frame, slice_frame in [
    ('Entrance', elem_start, first_slice['elem_start']),
    ('Exit', elem_end, last_slice['elem_end']),
]:
    print(label, 'sliced minus parent position [m]:',
          slice_frame.XYZ - parent_frame.XYZ)
    print(label, 'largest basis difference:',
          np.max(np.abs(slice_frame.E_matrix - parent_frame.E_matrix)))

################################################################################
# Save and reload the surveyed frames
################################################################################
# Native JSON preserves frame arrays and names; save the model separately.
save_path = Path(__file__).resolve().parent / SAVE_FILE
save_path.parent.mkdir(parents = True, exist_ok = True)
survey.to_json(save_path, exclude = 'line')
loaded = SurveyTable.from_json(save_path)
loaded_frame = loaded.get_all_frames('q')['elem_start']
print('Saved survey:', save_path)
print('Reloaded actual entrance minus original [m]:',
      loaded_frame.XYZ - elem_start.XYZ)
print('Reloaded entrance basis difference:',
      np.max(np.abs(loaded_frame.E_matrix - elem_start.E_matrix)))

################################################################################
# Plot the reference placement, actual element and pivots
################################################################################
fig, ax = plt.subplots(figsize = (8, 5))
ax.plot(survey.Z, survey.X, '.-', label = 'Reference path')
for start, end, label, color in [
    (ref_start, ref_end, 'Reference quadrupole', 'C0'),
    (elem_start, elem_end, 'Actual quadrupole', 'C1'),
]:
    ax.plot(
        [start.Z, end.Z], [start.X, end.X], '.-',
        linewidth   = 4,
        color       = color,
        label       = label,
    )
ax.plot(reference_pivot[2], reference_pivot[0], 'o', label = 'Reference pivot')
ax.plot(actual_pivot[2], actual_pivot[0], 'x', label = 'Shifted pivot')
ax.plot(global_point[2], global_point[0], 'k+', label = 'Local point in global')
slice_boundaries = np.array([
    sliced_survey.get_all_frames(name)['elem_start'].XYZ
    for name in slice_names] + [last_slice['elem_end'].XYZ])
ax.plot(slice_boundaries[:, 2], slice_boundaries[:, 0], 'k|',
        markersize = 12, label = 'Thick-slice boundaries')
ax.set_xlabel('Z [m]')
ax.set_ylabel('X [m]')
ax.axis('equal')
ax.grid()
ax.legend()
fig.tight_layout()
plt.show()
