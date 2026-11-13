# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Rectangular magnet and common field-free launch
########################################
P0C             = 1e9                   # Proton reference p0*c [eV]
CHORD_LENGTH    = 2.0                   # Magnet chord [m]
ANGLE           = 0.2                   # Reference deflection [rad]
DRIFT_LENGTH    = 1.0                   # Each adjacent drift [m]
X_START         = 1e-3                  # [m]
Y_START         = 2e-3                  # [m]
DELTA           = 1e-3                  # Relative momentum offset
N_SLICES        = 8

################################################################################
# Compare horizontal and vertical rectangular-bend representations
################################################################################

########################################
# Keep the physical magnet and field setting fixed
########################################
fig, axes = plt.subplots(2, 2, figsize = (11, 8), layout = 'constrained')
results = {}
for row, (plane, roll) in enumerate([('Horizontal', 0.0),
                                     ('Vertical', np.pi / 2)]):
    for body in ['curved-body', 'straight-body']:
        magnet = xt.RBend(
            length_straight     = CHORD_LENGTH,
            angle               = ANGLE,
            rot_s_rad           = roll,
            rbend_model         = body,
            model               = 'bend-kick-bend',
            edge_entry_model    = 'full',
            edge_exit_model     = 'full',
        )
        line = xt.Line(elements = {
            'before': xt.Drift(length = DRIFT_LENGTH),
            'bend': magnet,
            'after': xt.Drift(length = DRIFT_LENGTH),
        })

        ########################################
        # Slice and reconstruct the same entrance ray
        ########################################
        line.set_particle_ref('proton', p0c = P0C)
        unsliced = line.copy()
        line.slice_thick_elements(slicing_strategies = [
            xt.Strategy(slicing = xt.Uniform(N_SLICES, mode = 'thick'),
                        element_type = xt.RBend),
        ])
        survey = line.survey(include_element_frames = True)
        tw = line.twiss4d(betx = 1.0, bety = 1.0,
                         x = X_START, y = Y_START, delta = DELTA)
        trajectory = (survey.XYZ + tw.x[:, None] * survey.ex
                       + tw.y[:, None] * survey.ey)
        results[plane, body] = (line, survey, tw, trajectory)
        transverse_index = 0 if row == 0 else 1
        axes[row, 0].plot(survey.Z, survey.XYZ[:, transverse_index],
                          '.-', label = body + ' reference')
        axes[row, 0].plot(trajectory[:, 2], trajectory[:, transverse_index],
                          '--', label = body + ' ray')
        axes[row, 1].plot(tw.s, tw.betx, label = body + ' beta_x')
        axes[row, 1].plot(tw.s, tw.bety, '--', label = body + ' beta_y')

        ########################################
        # Inspect face frames and distributed reference angles
        ########################################
        print(plane, body, 'arc length / chord [m]:',
              magnet.length, magnet.length_straight)
        print('  Reference exit XYZ [m]:', survey.XYZ[-1])
        print('  Physical ray exit XYZ [m]:', trajectory[-1])
        unsliced_survey = unsliced.survey(include_element_frames = True)
        frames = unsliced_survey.get_all_frames('bend')
        for name, frame in frames.items():
            print('  Face frame', name, 'XYZ/es:', frame.XYZ, frame.es)
        print('  Sliced minus unsliced reference exit [m]:',
              survey.XYZ[-1] - unsliced_survey.XYZ[-1])
        # Straight-body RBend puts its face rotations at the boundaries;
        # curved-body RBend distributes the angle through the body slices.
        # Reference-frame origin offsets at faces are bookkeeping changes;
        # the reconstructed physical ray remains continuous.
        line.get_table(attr = True).cols['name s element_type angle'].show()
    axes[row, 0].set(xlabel = 'Global Z [m]',
                     ylabel = 'Global ' + ('X' if row == 0 else 'Y') + ' [m]',
                     title = plane + ' bending plane')
    axes[row, 1].set(xlabel = 'Reference s [m]', ylabel = 'Beta [m]')
for axis in axes.ravel():
    axis.legend(fontsize = 'small')
    axis.grid()
# Finite slicing/body-model differences are not a proof of equivalent finite-
# amplitude maps. Compare physical rays and face frames, not local x/y alone.
plt.show()
