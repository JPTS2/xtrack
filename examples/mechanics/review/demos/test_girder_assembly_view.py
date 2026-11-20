"""Checks for the local presentation helper, outside the shipped test suite."""
from types import SimpleNamespace

import numpy as np
import xtrack as xt

from girder_assembly_view import display_hierarchy, vertices


def test_unequal_display_gains_do_not_separate_rigid_mounts():
    baseline = xt.Frame.from_survey_angles(Y=-.6, psi=.1).matrix
    parent = np.array([(xt.Frame(baseline) @ xt.Frame.from_survey_angles(
        Y=value*1e-5, phi=value*2e-5, theta=value*3e-5)).matrix for value in (0., .5, 1.)])
    static_parent = np.broadcast_to(baseline, parent.shape)
    mount = xt.Frame.from_survey_angles(X=.1, Y=.6, Z=1.5, theta=.03).matrix
    history = dict(girder=SimpleNamespace(static_matrix=static_parent, total_matrix=parent),
                   magnet=SimpleNamespace(static_matrix=static_parent @ mount, total_matrix=parent @ mount))
    shown = display_hierarchy(history, girder_name='girder', girder_translation_scale=100.,
        girder_rotation_scale=300., magnet_translation_scale=1700., magnet_rotation_scale=900.)
    np.testing.assert_allclose(shown['magnet'], shown['girder'] @ mount, atol=2e-12)
    np.testing.assert_array_equal(history['girder'].total_matrix, parent)


def test_relative_motion_is_visible_and_cuboid_dimensions_stay_rigid():
    baseline = xt.Frame().matrix
    parent = np.broadcast_to(baseline, (3, 4, 4))
    mount = xt.Frame.from_survey_angles(Y=.6, Z=1.).matrix
    relative = np.array([(xt.Frame(mount) @ xt.Frame.from_survey_angles(
        Y=value*1e-5, psi=value*1e-4)).matrix for value in (0., .5, 1.)])
    history = dict(girder=SimpleNamespace(static_matrix=parent, total_matrix=parent),
                   magnet=SimpleNamespace(static_matrix=parent @ mount, total_matrix=relative))
    shown = display_hierarchy(history, girder_name='girder', girder_translation_scale=100.,
        girder_rotation_scale=300., magnet_translation_scale=1000., magnet_rotation_scale=900.)
    np.testing.assert_allclose(shown['magnet'][:, 1, 3]-.6, [0., .005, .01], atol=1e-12)
    corners = vertices((.44, .44, 1.5))
    distance = np.linalg.norm(corners[0]-corners[-1])
    world = np.einsum('tij,vj->tvi', shown['magnet'][:, :3, :3], corners) + shown['magnet'][:, None, :3, 3]
    np.testing.assert_allclose(np.linalg.norm(world[:, 0]-world[:, -1], axis=1), distance, atol=1e-14)
