# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

# Legacy data input only. Prefer native Python definitions for new models.

################################################################################
# Required packages
################################################################################
from pathlib import Path

import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Legacy data
########################################
# A small existing-style data definition, embedded here to make the input
# visible. This is parsed by Xtrack; no MAD-X process or macro is executed.
LEGACY_DATA = """
kq = 0.2;
qf: quadrupole, l=0.4, k1:=kq;
qd: quadrupole, l=0.4, k1:=-kq;
transport: sequence, l=3.0, refer=centre;
    qf, at=0.7;
    qd, at=2.3;
endsequence;
"""
LINE_NAME   = 'transport'
KQ_AFTER    = 0.3                       # [m^-2]

########################################
# Native output file
########################################
# Relative to this script; the legacy input is never overwritten.
SAVE_FILE   = 'outputs/converted_legacy_environment.json'

################################################################################
# Read legacy data with the native parser
################################################################################

########################################
# Load the definition
########################################
env     = xt.load(string = LEGACY_DATA, format = 'madx')
line    = env[LINE_NAME]

########################################
# Inspect geometry and expressions
########################################
# Legacy sequence positions refer to element centres here.
line.get_table().cols['name s_start s_center s_end element_type'].show()
print('Imported length [m]:', line.get_length())
print('Imported qf and qd k1 [m^-2]:', line['qf'].k1, line['qd'].k1)
env.info('qf')

################################################################################
# Continue with native Xtrack data
################################################################################

########################################
# Change an imported variable
########################################
env['kq'] = KQ_AFTER
print('After changing kq [m^-2]:', line['qf'].k1, line['qd'].k1)

########################################
# Save and reload native JSON
########################################
save_path = Path(__file__).resolve().parent / SAVE_FILE
save_path.parent.mkdir(parents = True, exist_ok = True)

env.to_json(save_path)
env_loaded      = xt.load(save_path)
line_loaded     = env_loaded[LINE_NAME]
print('Saved native environment:', save_path)
print('Native reload length [m]:', line_loaded.get_length())
print('Native reload qf and qd k1 [m^-2]:',
      line_loaded['qf'].k1, line_loaded['qd'].k1)
