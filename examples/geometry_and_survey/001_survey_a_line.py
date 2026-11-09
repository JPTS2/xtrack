# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
from pathlib import Path

import matplotlib.pyplot as plt
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Native transport line
########################################
LATTICE_FILE    = 'test_data/example_lattices/transport/lattice.py'
LINE_NAME       = 'transport'

########################################
# Floor plot
########################################
PLOT_LABELS     = ['qf.1', 'b1.1', 'b2.1', 'qf.2']
ELEMENT_WIDTH   = 0.1                    # Drawing width [m]
FIGURE_SIZE     = (8, 5)                 # [inch]

################################################################################
# Load the line and compute its reference geometry
################################################################################
repo_root   = Path(__file__).resolve().parents[2]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env[LINE_NAME]
# Survey uses lengths and geometric bend angles, without tracking a particle
# or solving for an orbit. The default initial frame has its tangent along +Z.
survey      = line.survey()

################################################################################
# Inspect positions and orientation
################################################################################

########################################
# Read the geometric bend angles
########################################
table = line.get_table(attr = True)
table.rows[['b1.1', 'b2.1']].cols['name s_start s_end angle'].show()

########################################
# Read element entrances and the final exit
########################################
# s is accumulated reference-path length [m], not the global Z coordinate.
# Global X, Y, Z are in metres; theta, phi, psi are in radians. Positive bends
# in this horizontal line decrease theta; quadrupole strengths do not enter.
survey.cols['name s X Y Z theta phi psi'].show(maxwidth = 'full')
print('Final reference-path length [m]:', survey.s[-1])
print('Final global Z [m]:', survey.Z[-1])

################################################################################
# Draw the reference line in the horizontal plane
################################################################################
survey.plot(
    projection      = 'ZX',
    labels          = PLOT_LABELS,
    element_width   = ELEMENT_WIDTH,
    figsize         = FIGURE_SIZE)
plt.title('Transport-line reference geometry')
plt.tight_layout()
plt.show()
