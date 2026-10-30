import numpy as np
from scipy.constants import c as clight

from .general import _print

import xtrack as xt
import xobjects as xo

def _compensate_radiation_energy_loss(line, delta0='zero_mean', rtol_eneloss=1e-12,
                                      max_iter=100, verbose=True,
                                      co_search_at=None,
                                      **kwargs):

    assert isinstance(line._context, xo.ContextCpu), "Only CPU context is supported"
    assert line.particle_ref is not None, "Particle reference is not set"
    assert np.abs(line.particle_ref.q0) == 1, "Only |q0| = 1 is supported (for now)"

    enames_no_autodrift = [nn for nn in line.element_names if not nn.startswith('||drift_')]
    if len(set(enames_no_autodrift)) != len(enames_no_autodrift):
        raise ValueError("Line must not contain repeated elements to use "
                         "`compensate_radiation_energy_loss(...)`. ")

    ele_type = line.tracker._tracker_data_base._line_table.element_type
    ele_type_set = set(ele_type)
    for et in ele_type_set:
        if 'SliceCavity' in et:
            raise ValueError(
                f"Element type '{et}' is not supported for radiation energy "
                "loss compensation.")

    if 'record_iterations' in kwargs:
        record_iterations = kwargs['record_iterations']
        kwargs.pop('record_iterations')
        line._tapering_iterations = []
    else:
        record_iterations = False

    if verbose: _print("Compensating energy loss.")

    line.config.XTRACK_MULTIPOLE_NO_SYNRAD = True
    with xt.line._preserve_track_flags(line):
        line.tracker.track_flags.XS_FLAG_KILL_CAVITY_KICK = True
        particle_on_co = line.find_closed_orbit(co_search_at=co_search_at,
                                                delta0=0 # 4d search
                                                )
    line.config.XTRACK_MULTIPOLE_NO_SYNRAD = False

    beta0 = float(particle_on_co._xobject.beta0[0])

    # Check whether compensation is needed
    p_test = particle_on_co.copy()
    p_test.delta = line.attr._cache['delta_taper'].multisetter.get_values()[0]
    line.track(p_test, turn_by_turn_monitor='ONE_TURN_EBE')
    mon = line.record_last_track
    eloss = -(mon.ptau[0, -1] - mon.ptau[0, 0]) * p_test.p0c[0]
    if p_test.state[0] > 0 and abs(eloss) < p_test.energy0[0] * rtol_eneloss:
        if verbose: _print("  - No compensation needed")
        return

    # save voltages
    # Can do "own" as cavity slices are not supported here (see assert above)
    v_setter = line.attr._cache['_own_voltage'].multisetter
    f_setter = line.attr._cache['_own_frequency'].multisetter
    h_setter = line.attr._cache['_own_harmonic'].multisetter
    lag_setter = line.attr._cache['_own_lag'].multisetter
    phase_setter = line.attr._cache['_own_phase'].multisetter
    phase_taper_setter = line.attr._cache['_own_phase_taper'].multisetter

    v0 = v_setter.get_values()
    f0 = f_setter.get_values()
    h0 = h_setter.get_values()
    lag_zero = lag_setter.get_values()
    phase_zero = phase_setter.get_values()

    # Legacy, deprecated, clean up if used
    lag_taper_setter = line.attr._cache['_own_lag_taper'].multisetter
    lag_taper_setter.set_values(0 * phase_zero)

    f0_all = f0 + h0 / (line.tracker._tracker_data_base.line_length / beta0 / clight)
    eneloss_partitioning = v0 / v0.sum()

    # Put all cavities on crest and at zero frequency
    phase_taper_setter.set_values(np.pi/2 - np.deg2rad(lag_zero) - phase_zero)
    v_setter.set_values(np.zeros_like(v_setter.get_values()))
    f_setter.set_values(np.zeros_like(f_setter.get_values()))
    h_setter.set_values(np.zeros_like(h_setter.get_values()))

    if verbose: _print("Share energy loss among cavities (repeat until energy loss is zero)")
    with xt.line._preserve_track_flags(line):
        line.tracker.track_flags.XS_FLAG_SR_TAPER = True

        if delta0 == 'zero_mean':
            num_rounds = 2
            delta_start = 0
        else:
            num_rounds = 1
            delta_start = delta0

        for round in range(num_rounds):
            i_iter = 0

            while True:

                if round == 1 and delta0 == 'zero_mean':
                    delta_ave = xt.twiss.trapz(delta, ss) / ss[-1]
                    delta_start -= delta_ave

                p_test = particle_on_co.copy()
                p_test.delta = delta_start
                line.track(p_test, turn_by_turn_monitor='ONE_TURN_EBE')
                mon = line.record_last_track

                if record_iterations:
                    line._tapering_iterations.append(mon)

                eloss = -(mon.ptau[0, -1] - mon.ptau[0, 0]) * p_test.p0c[0]
                if verbose: _print(f"Energy loss: {eloss:_.3f} eV             ")#, end='\r', flush=True)

                if np.abs(eloss) < p_test.energy0[0] * rtol_eneloss:
                    break

                v_setter.set_values(v_setter.get_values()
                                    + eloss * eneloss_partitioning)

                delta = mon.delta[0, :-1]
                ss = mon.s[0, :-1]

                i_iter += 1
                if i_iter > max_iter:
                    raise RuntimeError("Maximum number of iterations reached")
    if verbose: _print()
    delta_taper_full = 0.5*(mon.delta[0, :-1] + mon.delta[0, 1:]) # last point is added by the monitor

    if verbose: _print("  - Set delta_taper")
    delta_taper_mask = line.attr._cache['delta_taper'].mask
    delta_taper_setter = line.attr._cache['delta_taper'].multisetter
    delta_taper_setter.set_values(delta_taper_full[delta_taper_mask])

    if verbose: _print("  - Restore cavity voltage and frequency. Set cavity lag")
    v_synchronous = v_setter.get_values()

    mask_cav = line.attr._cache['_own_voltage'].mask
    zeta_at_cav = np.atleast_1d(np.squeeze(mon.zeta[0, :-1]))[mask_cav]
    mask_active_cav = np.abs(v0) > 0
    v_ratio = v0 * 0
    v_ratio[mask_active_cav] = v_synchronous[mask_active_cav] / v0[mask_active_cav]
    assert np.all(np.abs(v_ratio[mask_active_cav]) < 1)
    inst_phase = np.arcsin(v_ratio)

    total_phase = (inst_phase - (2 * np.pi) * f0_all * zeta_at_cav / beta0 / clight)
    total_phase = np.pi - total_phase # we are above transition
    phase_taper = total_phase - np.deg2rad(lag_zero) - phase_zero
    phase_taper[~mask_active_cav] = 0

    v_setter.set_values(v0)
    f_setter.set_values(f0)
    h_setter.set_values(h0)
    lag_taper_setter.set_values(0 * phase_taper)
    phase_taper_setter.set_values(phase_taper)

    # The sine inversion is exact for an isolated voltage kick. A finite
    # transit map samples its own evolving arrival phase, even for a legacy
    # cavity with several kicks. Preserve the existing result when it already
    # closes; otherwise solve the actual restored tracking map.
    _refine_tracked_cavity_phases(line, particle_on_co, delta_start,
                                  v_synchronous, rtol_eneloss, max_iter)
    _refresh_tracked_taper_profile(line, particle_on_co, delta_start,
                                  v_synchronous, rtol_eneloss, max_iter,
                                  zero_mean=(delta0 == 'zero_mean'))


def _refresh_tracked_taper_profile(line, particle_on_co, delta_start,
                                  synchronous_voltages, rtol, max_iter,
                                  zero_mean=False):
    """Prepare fixed hardware strengths from the restored actual trajectory.

    Preserve the standard endpoint-mean delta convention. This is a saved
    constant per parent, rather than a field following the particle inside
    an integration stage. The DC pilot is only the first estimate. Convergence
    here checks preparation consistency and RF work, not radiative accuracy.
    """
    initial = particle_on_co.copy()
    initial.delta = delta_start
    # Geometric endpoints avoid turn-end resetting of the monitor's s chart.
    # Use the same endpoint-mean quadrature convention as the saved hardware.
    positions = np.r_[line._get_s_elements(), line.get_length()]
    if zero_mean and positions[-1] <= 0.:
        raise ValueError('A zero-mean taper requires positive line length')
    item = line.attr._cache['delta_taper']
    for _ in range(max_iter):
        energy0 = float(initial.energy[0])
        energy_tolerance = max(rtol*energy0, 32*np.finfo(float).eps*energy0)
        # dE/d(delta)=p0c*beta for the current entrance state. This translates
        # the requested energy tolerance to its relative-momentum chart.
        delta_tolerance = energy_tolerance/float(
            initial.beta0[0]*initial.rvv[0]*initial.p0c[0])
        probe = initial.copy()
        line.track(probe, turn_by_turn_monitor='ONE_TURN_EBE')
        if probe.state[0] <= 0:
            raise RuntimeError('Restored taper map lost the reference particle')
        monitor = line.record_last_track
        means = .5*(monitor.delta[0, :-1]+monitor.delta[0, 1:])
        target = means[item.mask]
        current = item.multisetter.get_values()
        error = float(probe.energy[0])-energy0
        mean_delta = (xt.twiss.trapz(monitor.delta[0], positions)/positions[-1]
                      if zero_mean else 0.)
        if (np.all(np.abs(target-current) <= delta_tolerance)
                and abs(error) <= energy_tolerance
                and abs(mean_delta) <= delta_tolerance):
            return
        item.multisetter.set_values(target)
        if zero_mean:
            delta_start -= mean_delta
            initial.delta = delta_start
        # If changing the prescribed hardware alters RF work, solve the
        # restored maps again. Failure remains transactional in the wrapper.
        _refine_tracked_cavity_phases(line, particle_on_co, delta_start,
                                      synchronous_voltages, rtol, max_iter)
    raise RuntimeError('Restored taper endpoint-mean/zero-mean preparation did not '
                       'converge within max_iter')


def _refine_tracked_cavity_phases(line, particle_on_co, delta_start,
                                 synchronous_voltages, rtol, max_iter):
    """Keep the requested DC work while restoring the finite-frequency maps.

    The DC taper pass determines each cavity's work partition. Its arcsine
    phase is an initial estimate. Tracking evaluates the actual selected map,
    including the entrance momentum and arrival time; no RF reference model
    or V*sin(phase) substitution is used in this correction.
    """
    from scipy.optimize import root_scalar

    initial = particle_on_co.copy()
    initial.delta = delta_start
    energy0 = float(initial.energy[0])
    tolerance = max(rtol*energy0, 32*np.finfo(float).eps*energy0)
    check = initial.copy()
    line.track(check)
    if check.state[0] <= 0:
        raise RuntimeError('Restored RF map lost the taper reference particle')
    if abs(float(check.energy[0])-energy0) <= tolerance:
        return

    # Match the mask/order used by the standard voltage multisetter. Elements
    # without a longitudinal Cavity map retain their existing behaviour.
    voltage_mask = line.attr._cache['_own_voltage'].mask
    voltage_indices = np.flatnonzero(voltage_mask)
    targets = {int(index): abs(float(initial.q0))*float(voltage)
               for index, voltage in zip(voltage_indices, synchronous_voltages)}
    pilot = initial.copy()
    for index, name in enumerate(line.element_names):
        element = line[name]
        if (index in targets and isinstance(element, xt.Cavity)
                and float(element.voltage) != 0.):
            target = targets[index]
            phase0 = float(element.phase_taper)

            def residual(phase):
                element.phase_taper = phase
                trial = pilot.copy()
                before = float(trial.energy[0])
                line.track(trial, ele_start=index, ele_stop=index+1)
                if trial.state[0] <= 0:
                    return None
                difference = float(trial.energy[0])-before-target
                return difference if np.isfinite(difference) else None

            value = residual(phase0)
            if value is None or abs(value) > tolerance:
                samples = {} if value is None else {phase0: value}
                bracket = None
                for width in (.025, .05, .1, .2, .4, .8, np.pi/2):
                    for phase in (phase0-width, phase0+width):
                        value = residual(phase)
                        if value is not None:
                            samples[phase] = value
                    ordered = sorted(samples)
                    candidates = [(left, right) for left, right in
                        zip(ordered[:-1], ordered[1:])
                        if samples[left]*samples[right] <= 0.]
                    if candidates:
                        bracket = min(candidates,
                                      key=lambda pair: abs(sum(pair)/2-phase0))
                        break
                if bracket is None:
                    raise RuntimeError(
                        f'No admissible tracked taper phase bracket for {name!r}')

                def valid_residual(phase):
                    value = residual(phase)
                    if value is None:
                        raise RuntimeError(
                            f'Taper phase search left the tracking domain of {name!r}')
                    return value

                solution = root_scalar(valid_residual, bracket=bracket,
                    method='brentq', xtol=4*np.finfo(float).eps,
                    rtol=8*np.finfo(float).eps, maxiter=max_iter)
                if not solution.converged:
                    raise RuntimeError(f'Tracked taper phase did not converge for {name!r}')
                value = valid_residual(solution.root)
                if abs(value) > tolerance:
                    raise RuntimeError(f'Tracked taper work did not close for {name!r}: '
                                       f'{value:g} eV')
            else:
                element.phase_taper = phase0
        line.track(pilot, ele_start=index, ele_stop=index+1)
        if pilot.state[0] <= 0:
            raise RuntimeError(f'Taper reference particle lost at {name!r}')

    # Check one uninterrupted call too: call-boundary/reference-clock effects
    # must not be hidden by the element-local phase evaluations above.
    check = initial.copy()
    line.track(check)
    if check.state[0] <= 0:
        raise RuntimeError('Corrected RF map lost the taper reference particle')
    error = float(check.energy[0])-energy0
    if abs(error) > tolerance:
        raise RuntimeError('Restored physical taper map does not close energy: '
                           f'{error:g} eV')


def compensate_radiation_energy_loss(line, delta0='zero_mean', rtol_eneloss=1e-12,
                                     max_iter=100, verbose=True,
                                     co_search_at=None, **kwargs):
    """Prepare radiation tapering transactionally and check restored RF work.

    Parameterized TPSA storage must first be returned to ordinary scalars:
    this preparation mutates numeric strengths through multisetter buffers.
    TPSA tracking of the prepared element settings is a separate operation.
    """
    if not isinstance(line._context, xo.ContextCpu):
        # Preserve the existing precondition/error before constructing caches.
        return _compensate_radiation_energy_loss(line, delta0, rtol_eneloss,
            max_iter, verbose, co_search_at, **kwargs)
    # Numeric caches probe parent storage as well as the used slice objects.
    # Visit all physical owners, including magnets; a TPSA-enabled quadrupole
    # parent is just as unsafe for a numeric multisetter as a Cavity parent.
    pending = list(line.element_names)
    used = []
    seen = set()
    while pending:
        element = line._element_dict[pending.pop()]
        if isinstance(element, xt.Replica):
            element = element.resolve(line)
        if id(element) in seen:
            continue
        seen.add(id(element))
        used.append(element)
        parent_name = getattr(element, 'parent_name', None)
        if parent_name is not None:
            pending.append(parent_name)
    if any(isinstance(element, (xt.ThinSliceCavity, xt.ThickSliceCavity,
                                 xt.DriftSliceCavity)) for element in used):
        raise ValueError('Cavity slices are not supported for radiation energy '
                         'loss compensation.')
    if any(getattr(getattr(element, '_xobject', None), '_tpsa_enabled', 0)
           for element in used):
        raise NotImplementedError(
            'Taper preparation requires scalar element storage. Restore TPSA '
            'knobs to doubles before tapering; then apply them again for tracking.')

    attributes = line.attr
    names = ('_own_voltage', '_own_frequency', '_own_harmonic', '_own_phase',
             '_own_lag', '_own_phase_taper', '_own_lag_taper', 'delta_taper')
    saved = [(attributes._cache[name].multisetter,
              attributes._cache[name].multisetter.get_values().copy())
             for name in names]
    config = dict(line.config)
    try:
        return _compensate_radiation_energy_loss(line, delta0, rtol_eneloss,
            max_iter, verbose, co_search_at, **kwargs)
    except BaseException:
        for setter, values in saved:
            setter.set_values(values)
        line.config.clear()
        line.config.update(config)
        raise
