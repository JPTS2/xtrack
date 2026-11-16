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
# Keep the small legacy inputs visible here. The example writes local copies
# before loading them; no MAD-X process or macro is executed.
LEGACY_DATA = """
chamber: marker, apertype="polygon",
    aper_vx={-0.012, 0.012, 0.012, -0.012},
    aper_vy={-0.008, -0.008, 0.008, 0.008};
qf: quadrupole, l=0.4, k1:=kq;
qd: quadrupole, l=0.4, k1:=-kq;
transport: sequence, l=3.0, refer=centre;
    chamber, at=0.1;
    qf, at=0.7;
    qd, at=2.3;
endsequence;
return_path: sequence, l=3.0, refer=centre;
    qf, at=0.7;
    qd, at=2.3;
endsequence;
"""
LEGACY_STRENGTHS    = """
kq = 0.2;
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
# Load separate lattice and strength files
########################################
output_dir      = Path(__file__).resolve().parent / 'outputs'
output_dir.mkdir(parents = True, exist_ok = True)
lattice_path    = output_dir / 'legacy_transport.madx'
strengths_path  = output_dir / 'legacy_strengths.madx'
lattice_path.write_text(LEGACY_DATA)
strengths_path.write_text(LEGACY_STRENGTHS)
print('Read legacy lattice and strength data'
      '\n-------------------------------------')
print('Legacy input only: prefer native Python definitions for new models.')
print('Xtrack parses these files directly; no MAD-X process or macros run.')
env         = xt.load(
    file            = [lattice_path, strengths_path],
    format          = 'madx',
    reverse_lines   = ['return_path'],
)
line        = env[LINE_NAME]

########################################
# Inspect geometry and expressions
########################################
# Legacy sequence positions refer to element centres here. Reversed beams
# receive separate definitions; shared names are qualified by the line name.
print('\nImported transport geometry\n---------------------------')
print('Positions [m]. The legacy sequences specify element centres.')
line.get_table().cols[
    'name s_start s_center s_end element_type'].show()
print(f'Imported length [m]: {line.get_length():.3f}')
print(f'qf/transport k1 [m^-2]: {line["qf/transport"].k1:.3f}')
print(f'qd/transport k1 [m^-2]: {line["qd/transport"].k1:.3f}')
print('Imported qf strength and its deferred expression:')
env['qf/transport'].get_table().rows['k1'].show()

########################################
# Inspect the polygon limit created from stored aperture data
########################################
print('\nStored aperture data becomes a native tracking limit'
      '\n----------------------------------------------------')
limits      = line.get_table().rows['.*_aper.*']
limits.cols['name s element_type'].show()
aperture    = line.get(limits.name[0])
print('Polygon x vertices [m]:', aperture.x_vertices)
print('Polygon y vertices [m]:', aperture.y_vertices)
probes      = xt.Particles('electron', p0c = 1e9, x = [0.0, 0.020])
aperture.track(probes)
probes.sort(interleave_lost_particles = True)
print('Probe x [m]:', probes.x)
print('Probe states (1 = alive, 0 = lost):', probes.state)
# These are native tracking limits; loading stored geometry does not run MAD-X.

########################################
# Inspect the reversed stored beam definition
########################################
# The return path now runs from qd to qf in its own beam coordinate convention.
print('\nConvert the return path to its reversed beam convention'
      '\n-------------------------------------------------------')
print('reverse_lines reverses element order and transforms magnet parameters.')
print('Here the normal quadrupole k1 signs change; unary -line only reorders.')
print('Positions [m]:')
env['return_path'].get_table().cols[
    'name s_start s_center s_end element_type'].show()
print(f'Return-path length [m]: {env["return_path"].get_length():.3f}')
print(f'qd/return_path k1 [m^-2]: {env["qd/return_path"].k1:.3f}')
print(f'qf/return_path k1 [m^-2]: {env["qf/return_path"].k1:.3f}')

################################################################################
# Continue with native Xtrack data
################################################################################

########################################
# Change an imported variable
########################################
print('\nChange the imported strength variable'
      '\n-------------------------------------')
env['kq'] = KQ_AFTER
print(f'kq [m^-2]: {env["kq"]:.3f}')
print(f'qf/transport k1 [m^-2]: {line["qf/transport"].k1:.3f}')
print(f'qd/transport k1 [m^-2]: {line["qd/transport"].k1:.3f}')
print(f'qd/return_path k1 [m^-2]: {env["qd/return_path"].k1:.3f}')
print(f'qf/return_path k1 [m^-2]: {env["qf/return_path"].k1:.3f}')

########################################
# Save and reload native JSON
########################################
save_path       = Path(__file__).resolve().parent / SAVE_FILE
save_path.parent.mkdir(parents = True, exist_ok = True)

print('\nSave and reload the current setting as native JSON'
      '\n--------------------------------------------------')
env.to_json(save_path)
print('Saved native environment:', save_path)
env_loaded      = xt.load(save_path)
line_loaded     = env_loaded[LINE_NAME]
print(f'Reloaded transport length [m]: {line_loaded.get_length():.3f}')
print(f'Reloaded qf/transport k1 [m^-2]: '
      f'{line_loaded["qf/transport"].k1:.3f}')
print(f'Reloaded qd/transport k1 [m^-2]: '
      f'{line_loaded["qd/transport"].k1:.3f}')
