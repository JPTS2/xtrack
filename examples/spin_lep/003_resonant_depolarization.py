import xtrack as xt
import xpart as xp
import xobjects as xo
import numpy as np

num_turns = 500

line = xt.Line.from_json('lep_sol.json')
line.particle_ref.anomalous_magnetic_moment=0.00115965218128
line.particle_ref.gamma0 = 89207.78287659843 # to have a spin tune of 103.45
spin_tune = line.particle_ref.anomalous_magnetic_moment[0]*line.particle_ref.gamma0[0]


# All off
line['on_solenoids'] = 0
line['on_spin_bumps'] = 0

# Match tunes
opt = line.match(
    method='4d',
    solve=False,
    vary=xt.VaryList(['kqf', 'kqd'], step=1e-4),
    targets=xt.TargetSet(qx=65.10, qy=71.20, tol=1e-4)
)
opt.solve()

tw = line.twiss4d(spin=True, radiation_integrals=True)

line.configure_spin(spin_model='auto')

p0 = tw.particle_on_co.copy()
p0.spin_x = 1e-4
line.track(p0, num_turns=1000, turn_by_turn_monitor=True,
              with_progress=10)
mon0 = line.record_last_track

tt = line.get_table()
tt_bend = tt.rows[(tt.element_type == 'Bend') | (tt.element_type == 'RBend')]
tt_quad = tt.rows[(tt.element_type == 'Quadrupole')]

line.set(tt_bend, model='drift-kick-drift-expanded', integrator='uniform',
        num_multipole_kicks=3)

class Chirper(xt.BeamElement):

    _xofields = {
        'k0sl': xo.Float64,
        'q_start': xo.Float64,
        'q_end': xo.Float64,
        'num_turns': xo.Float64,
    }

    _extra_c_sources =['''
        /*gpufun*/
        void Chirper_track_local_particle(
                ChirperData el, LocalParticle* part0){

            double const k0sl = ChirperData_get_k0sl(el);
            double const q_start = ChirperData_get_q_start(el);
            double const q_end = ChirperData_get_q_end(el);
            double const num_turns = ChirperData_get_num_turns(el);

            //start_per_particle_block (part0->part)
                double const at_turn = LocalParticle_get_at_turn(part);
                if (at_turn < num_turns){
                    double const qq = q_start + (q_end - q_start) * ((double) at_turn) / ((double) num_turns);
                    double const dpy = k0sl * sin(2 * PI * qq * at_turn);
                    LocalParticle_add_to_py(part, dpy);
                }
            //end_per_particle_block
        }
        ''']

chirper = Chirper(
    k0sl=0,
    q_start=0,
    q_end=0,
    num_turns=0,
)
line.insert('chirper', obj=chirper, at='bfkv1.qs18.r2@start')

dq_sweep = 0.003

q_start_tests = np.linspace(0.425, 0.455, 5)[1:]

q_start_tests = [0.44, 0.448, 0.453]


import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)

for iii, qqq in enumerate(q_start_tests):
    num_turns = 15000
    q_start_excitation = qqq
    q_end_excitation = q_start_excitation + dq_sweep
    k0sl_peak = 5e-6

    chirper.k0sl = k0sl_peak
    chirper.q_start = q_start_excitation
    chirper.q_end = q_end_excitation
    chirper.num_turns = num_turns

    p = tw.particle_on_co.copy()

    line.track(p, num_turns=num_turns, turn_by_turn_monitor=True,
            with_progress=1000)
    mon = line.record_last_track

    import nafflib
    spin_tune_freq_analysis = nafflib.get_tune(mon0.spin_x[0, :])

    freq_axis = np.linspace(q_start_excitation, q_end_excitation, num_turns)


    plt.plot(freq_axis, mon.spin_y.T)
    plt.xlabel('Excitation tune')
    plt.ylabel('Spin y')
    plt.axvline(spin_tune_freq_analysis)

    plt.show()
    plt.pause(1)

