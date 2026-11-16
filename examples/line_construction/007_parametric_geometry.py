# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
import xtrack as xt

################################################################################
# User parameters
################################################################################
SCALE_START = 1.0
SCALE_AFTER = 1.5

################################################################################
# Define geometry expressions
################################################################################

########################################
# Define element lengths
########################################
env = xt.Environment()
env['scale'] = SCALE_START

env.new('qf', xt.Quadrupole, length = '0.4 * scale', k1 = 0.2)
env.new('qd', xt.Quadrupole, length = '0.4 * scale', k1 = -0.2)

########################################
# Add placements before assembling the line
########################################
# compose=True defers assembly, allowing placements to be added below.
# Ordinary env.new_line(...) also retains its placement composer.
line = env.new_line(
    name    = 'parametric_line',
    length  = '5 * scale',
    refer   = 'start',
    compose = True,
)
line.place('qf', at = 'scale')
line.place('qd', at = '2 * scale', from_ = 'qf@end')

########################################
# Assemble and inspect the starting geometry
########################################
line.end_compose()

table_before = line.get_table()
print('Initial geometry [m]\n--------------------')
print(f'Geometry scale: {env["scale"]:g}')
table_before.cols['name s_start s_end'].show()

################################################################################
# Regenerate the geometry
################################################################################

########################################
# Change the scale without resolving the placements
########################################
# Magnet lengths update immediately, but generated drift lengths do not.
# k1 stays fixed: changing the geometry does not preserve its optics.
env['scale'] = SCALE_AFTER
print('\nScale changed; generated drifts unchanged'
      '\n-----------------------------------------')
print('Magnet lengths have updated, but the placement expressions')
print('have not been reapplied. The requested layout is not yet satisfied.')
print(f'Geometry scale: {env["scale"]:g}')
line.get_table().cols['name s_start s_end'].show()
print(f'Current length [m]: {line.get_length():.3f}')
print(f'Requested length, 5 * scale [m]: {env.eval("5 * scale"):.3f}')

########################################
# Resolve the placements and recalculate the drifts
########################################
# Changing a variable does not rerun placement resolution in an assembled line.
# Regeneration re-enters compose mode; end_compose resolves the placements.
line.regenerate_from_composer()
line.end_compose()

########################################
# Inspect the regenerated positions
########################################
table = line.get_table()
print('\nRebuild the layout and recalculate drifts'
      '\n-----------------------------------------')
print('Placement expressions are resolved again; the drifts are recalculated.')
print(f'Geometry scale: {env["scale"]:g}')
table.cols['name s_start s_end'].show()
print(f'Length [m]: {table_before.s[-1]:.3f} -> {table.s[-1]:.3f}')

################################################################################
# Reuse the scaled section in a longer beamline
################################################################################

########################################
# Choose the order and number of subline occurrences
########################################
# line is one 7.5 m section containing qf followed by qd at these settings.
# -line reverses its element order, while 2 * line repeats it twice.
# This reuses the line algebra introduced in 005_line_algebra.py.
composite = env.new_line(
    name        = 'composite',
    compose     = True,
    components  = [line, -line, 2 * line],
)
composite.end_compose()
print('\nReuse the section in a longer beamline'
      '\n--------------------------------------')
print('components = [line, -line, 2 * line]')
print('Four sections: original order, reversed order, then two original-order')
print('sections. The magnet sequence is qf,qd | qd,qf | qf,qd | qf,qd.')
print('Reversal changes element order, not magnet strengths or beam direction.')
print('Magnet positions [m]:')
# Show the magnets only; drift rows would repeat the preceding geometry.
composite.get_table().rows['qf.*|qd.*'].cols[
    'name s_start s_end'].show()
print(f'Combined length [m]: {composite.get_length():.3f}')

################################################################################
# Restore and regenerate the placed subline
################################################################################

########################################
# Reconstruct the current state from a dictionary
########################################
print('\nReconstruct from a dictionary\n-----------------------------')
restored = xt.Line.from_dict(line.to_dict())
print(f'Original line length [m]: {line.get_length():.3f}')
print(f'Restored line length [m]: {restored.get_length():.3f}')

########################################
# Change the scale in the restored copy only
########################################
restored['scale'] = SCALE_START
restored.regenerate_from_composer()
restored.end_compose()
print('\nReset the scale in the restored copy'
      '\n------------------------------------')
print(f'Restored geometry scale: {restored["scale"]:g}')
restored.get_table().cols['name s_start s_end'].show()
print(f'Unchanged original length [m]: {line.get_length():.3f}')
print(f'Modified restored length [m]: {restored.get_length():.3f}')
