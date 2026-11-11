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

################################################################################
# User parameters
################################################################################

########################################
# Reference and field geometry
########################################
LATTICE_FILE    = 'test_data/fcc_ee/fccee_t.seq'
FIELD_FILE      = 'test_data/fcc_ee/Bz_closed_before_quads.dat'
P0C             = 182.5e9                # Positron reference momentum [eV/c]
TILT            = 15e-3                  # Detector/beam half-angle [rad]
BEAM_LENGTH     = 4.4                    # Replaced beam-axis interval [m]
N_SLICES        = 1000                   # Even number of field intervals
IP              = 'ip.1'
LEFT            = 'pqc2le.4'
RIGHT           = 'pqc2re.1'

########################################
# Correction resolution
########################################
ORBIT_STEP      = 1e-8                   # Integrated dipole strength step [rad]
SKEW_STEP       = 1e-7                   # Skew gradient step [m^-2]
NORMAL_STEP     = 1e-6                   # Normal gradient step [m^-2]
N_PASSES        = 2                      # Alternate the three correction stages

################################################################################
# Load the stored lattice and sample the tabulated axial field
################################################################################

########################################
# Read the legacy sequence with the native parser
########################################
repo_root           = Path(__file__).resolve().parents[3]
env                 = xt.load(repo_root / LATTICE_FILE)
line                = env['fccee_p_ring']
line.particle_ref   = xt.Particles('positron', p0c = P0C)
# The sequence is a stored machine definition; no MAD-X executable is used.
# Freeze RF focusing out of this transverse, radiation-free comparison.
line['voltca1']         = 0.0
line['voltca2']         = 0.0
line.cycle('ip.4')

########################################
# Convert physical Bz to the signed normalized solenoid strength
########################################
field               = np.loadtxt(repo_root / FIELD_FILE, skiprows = 1)
solenoid_length     = BEAM_LENGTH * np.cos(TILT)
z_edges             = np.linspace(-solenoid_length / 2,
                                 solenoid_length / 2, N_SLICES + 1)
bz_edges            = np.interp(z_edges, field[:, 0], field[:, 1])
# The supplied profile already contains screening fields of opposite polarity.
# Outside its sampled range, close the field to zero at the two boundaries.
bz_edges[[0, -1]]       = 0.0
rigidity            = line.particle_ref.rigidity0[0]  # Signed B rho [T m]
ks_edges            = bz_edges / rigidity            # [m^-1]
print('Reference momentum [GeV/c]:', P0C * 1e-9)
print('Signed rigidity [T m]:', rigidity)
print('Sampled integral Bz dz [T m]:', np.trapezoid(bz_edges, z_edges))

################################################################################
# Replace the IP drift by the tilted detector-field section
################################################################################

########################################
# Fix the beam/detector frame transformation
########################################
ip_s                                = line.get_table()['s', IP]
line.insert('detector.begin', xt.Marker(), at = ip_s - BEAM_LENGTH / 2)
line.insert('detector.end', xt.Marker(), at = ip_s + BEAM_LENGTH / 2)
env.elements['detector.rotate.in']      = xt.Rotation(rot_y_rad = -TILT)
env.elements['detector.rotate.out']     = xt.Rotation(rot_y_rad = TILT)
shift_x                             = solenoid_length / 2 * np.tan(TILT)
env.elements['detector.shift.in']       = xt.Translation(shift_x = shift_x)
env.elements['detector.shift.out']      = xt.Translation(shift_x = shift_x)
# Inside the rotated coordinates the field-free trajectory has
# kin_xp=tan(TILT) at the IP. Compare physical slopes, not canonical px alone.

########################################
# Keep each field slice connected to one live strength knob
########################################
env['on_detector']  = 0.0
slice_names         = []
for index, length in enumerate(np.diff(z_edges)):
    name    = f'detector.slice.{index}'
    env.new(
        name, xt.VariableSolenoid,
        length      = length,
        ks_profile  = [ks_edges[index] * env.ref['on_detector'],
                       ks_edges[index + 1] * env.ref['on_detector']],
    )
    slice_names.append(name)

table               = line.get_table()
before              = list(table.rows[:'detector.begin'].env_name)
after               = list(table.rows['detector.end':].env_name[:-1])
line.element_names  = (
    before + ['detector.rotate.in', 'detector.shift.in']
    + slice_names[:N_SLICES // 2] + [IP] + slice_names[N_SLICES // 2:]
    + ['detector.shift.out', 'detector.rotate.out'] + after)

################################################################################
# Prepare independent orbit, skew and normal-quadrupole controls
################################################################################

########################################
# Attach gradient corrections to the four doublet sections on each side
########################################
quads   = {
    'l1': ['qc1l1.4', 'qc2l1.4', 'qc2l2.4', 'qc1l2.4'],
    'r1': ['qc1r1.1', 'qc2r1.1', 'qc2r2.1', 'qc1r2.1'],
}
for side, names in quads.items():
    for index, name in enumerate(names):
        skew                = f'detector.skew.{side}.{index}'
        normal              = f'detector.normal.{side}.{index}'
        env[skew]           = 0.0
        env[normal]         = 0.0
        env.ref[name].k1s   = env.ref[skew]
        env.ref[name].k1 += env.ref[normal]

########################################
# Resolve thick magnets for coupled-optics correction
########################################
strategies  = [
    xt.Strategy(slicing = None),
    xt.Strategy(slicing = xt.Teapot(3), element_type = xt.Bend),
]
for pattern, count in [
    (r'^qf.*', 5), (r'^qd.*', 5), (r'^qfg.*', 5), (r'^qdg.*', 5),
    (r'^ql.*', 5), (r'^qs.*', 5), (r'^qb.*', 10), (r'^qg.*', 10),
    (r'^qh.*', 10), (r'^qi.*', 10), (r'^qr.*', 10), (r'^qu.*', 10),
    (r'^qy.*', 10), (r'^qa.*', 50), (r'^qc.*', 50),
    (r'^sy\..*', 20), (r'^mwi\..*', 30),
]:
    strategies.append(xt.Strategy(slicing = xt.Teapot(count), name = pattern))
line.slice_thick_elements(slicing_strategies = strategies)

########################################
# Install two horizontal/vertical correctors on each side
########################################
corrector_sites = {
    'l1': ['qc1l1.4_entry', 'qc1l2.4_entry'],
    'r1': ['qc1r1.1_exit', 'qc1r2.1_exit'],
}
for side, sites in corrector_sites.items():
    for index, site in enumerate(sites):
        name                = f'detector.corrector.{side}.{index}'
        h_knob              = f'detector.h.{side}.{index}'
        v_knob              = f'detector.v.{side}.{index}'
        env[h_knob]         = 0.0
        env[v_knob]         = 0.0
        line.insert(name, xt.Multipole(knl = [0.0], ksl = [0.0]), at = site)
        env.ref[name].knl[0]    = env.ref[h_knob]
        env.ref[name].ksl[0]    = env.ref[v_knob]

################################################################################
# Establish the field-free reference and the uncorrected perturbation
################################################################################
reference           = line.twiss4d()
env['on_detector']  = 1.0
uncorrected         = line.twiss4d()
print('Field-free IP physical horizontal slope:', reference['kin_xp', IP])
print('Expected tan(TILT):', np.tan(TILT))
print('Uncorrected coupling c_minus:', uncorrected.c_minus)

################################################################################
# Restore orbit, decoupling and optics on the two sides of the IP
################################################################################

########################################
# Define each half-insertion from its unperturbed external boundary
########################################
optimizers  = []
for side, start, end, init_at in [
    ('l1', LEFT, IP, xt.START),
    ('r1', IP, RIGHT, xt.END),
]:
    orbit_knobs     = [f'detector.{plane}.{side}.{index}'
                       for index in range(2) for plane in ['h', 'v']]
    skew_knobs      = [f'detector.skew.{side}.{index}' for index in range(4)]
    normal_knobs    = [f'detector.normal.{side}.{index}' for index in range(4)]
    opt             = line.match(
        solve       = False,
        method      = '4d',
        n_steps_max = 30,
        start       = start,
        end         = end,
        init        = reference,
        init_at     = init_at,
        vary        = [
            xt.VaryList(orbit_knobs, step = ORBIT_STEP, tag = 'orbit'),
            xt.VaryList(skew_knobs, step = SKEW_STEP, tag = 'skew'),
            xt.VaryList(normal_knobs, step = NORMAL_STEP, tag = 'normal'),
        ],
        targets     = [
            xt.TargetSet(
                ['x', 'y'],
                value   = reference,
                at      = IP,
                tol     = 1e-7,
                tag     = 'orbit',
            ),
            xt.TargetSet(
                ['px', 'py'],
                value   = reference,
                at      = IP,
                tol     = 1e-10,
                tag     = 'orbit',
            ),
            xt.TargetRmatrix(
                r13 = 0, r14 = 0, r23 = 0, r24 = 0,
                r31 = 0, r32 = 0, r41 = 0, r42 = 0,
                start   = start,
                end     = end,
                tol     = 1e-6,
                tag     = 'coupling',
            ),
            xt.TargetSet(
                ['mux', 'muy'],
                value   = reference,
                at      = IP,
                tol     = 1e-6,
                weight  = 0.1,
                tag     = 'phase',
            ),
            xt.Target(
                'betx',
                value   = reference,
                at      = IP,
                tol     = 1e-5,
                tag     = 'beta',
            ),
            xt.Target(
                'bety',
                value   = reference,
                at      = IP,
                tol     = 1e-6,
                weight  = 10,
                tag     = 'beta',
            ),
            xt.TargetSet(
                ['alfx', 'alfy'],
                value   = reference,
                at      = IP,
                tol     = 1e-4,
                weight  = 0.1,
                tag     = 'alpha',
            ),
        ],
    )

    ########################################
    # Alternate orbit, skew and normal-gradient correction, then solve together
    ########################################
    for iteration in range(N_PASSES):
        opt.disable(target = True)
        opt.disable(vary = True)
        opt.enable(target = 'orbit')
        opt.enable(vary = 'orbit')
        opt.solve()

        opt.disable(target = True)
        opt.disable(vary = True)
        opt.enable(target = 'coupling')
        opt.enable(vary = 'skew')
        opt.solve()

        opt.disable(target = True)
        opt.disable(vary = True)
        opt.enable(vary = 'normal')
        for tag in ['phase', 'beta', 'alpha']:
            opt.enable(target = tag)
            opt.solve()

    opt.enable(target = True)
    opt.enable(vary = True)
    opt.solve()
    opt.target_status()
    optimizers.append(opt)

################################################################################
# Inspect the periodic machine after the local corrections
################################################################################

########################################
# Compare independent periodic optics and physical orbit quantities
########################################
corrected       = line.twiss4d()
for label, tw in [('Field off', reference), ('Field on', uncorrected),
                  ('Corrected', corrected)]:
    print(label, 'c_minus:', tw.c_minus)
    summary = tw.rows[[LEFT, IP, RIGHT]]
    summary.cols['name x y kin_xp kin_yp'].show(maxwidth = 'full')
    summary.cols['name betx bety betx2 bety1 dy'].show(maxwidth = 'full')
# Orbit, beta and dispersion columns are in metres; slopes are dimensionless.
print('Corrected/reference IP beta ratios:',
      corrected['betx', IP] / reference['betx', IP],
      corrected['bety', IP] / reference['bety', IP])
for start, end in [(LEFT, IP), (IP, RIGHT)]:
    matrix  = corrected.get_R_matrix(start, end)
    print('Cross-plane blocks:', start, 'to', end)
    print(matrix[:2, 2:4])
    print(matrix[2:4, :2])
print('Block units by row: [[1,m],[m^-1,1]]')

########################################
# Plot the field, vertical orbit and coupled-mode leakage
########################################
# Correction closes the orbit at the external boundaries and IP; the excursion
# inside the detector remains. A small bety1 measures mode-1 leakage into y.
fig, axes   = plt.subplots(3, 1, figsize = (9, 9), constrained_layout = True)
axes[0].plot(z_edges, bz_edges)
axes[0].set(xlabel = 'Detector axial coordinate z [m]', ylabel = 'Bz [T]')
for label, tw in [('Field off', reference), ('Field on', uncorrected),
                  ('Corrected', corrected)]:
    region  = tw.rows[LEFT:RIGHT]
    s_ip    = region.s - tw['s', IP]
    axes[1].plot(s_ip, region.y * 1e6, label = label)
    axes[2].plot(s_ip, region.bety1, label = label)
axes[1].set(xlabel = 's - s(IP) [m]', ylabel = 'Vertical orbit [um]')
axes[2].set(xlabel = 's - s(IP) [m]', ylabel = r'$\beta_{y,1}$ [m]')
for axis in axes:
    axis.grid(alpha = 0.3)
axes[1].legend()
axes[2].legend()
plt.show()
