"""Private opt-in native P16 preparation and conservative diagnostic.

The Fourier-integral formula is analytic; SciPy evaluates it in binary64.
Neither those values nor the prepared finite stage data are error-certified.
There is no runtime dependency on the scientific cavity-modelling project.
The supported source API is a read-only snapshot; private backend mutation
is unsupported and does not acquire numerical or physical qualification.
"""
from math import comb
from operator import index
from types import MappingProxyType

import numpy as np
from scipy.special import beta, hyp0f1
import xobjects as xo

from ..base_element import BeamElement

C_LIGHT = 299792458.
GAUSS_NODES = np.array([0.21132486540518713, 0.7886751345948129])
_STORED_GAUSS_A = ((.25,-.038675134594812866),(.5386751345948129,.25))
# Retained diagnostic SI context, matching the scientific reference. These
# represented values do not update Xtrack's global/native constants or supply
# coherent-SI or physical preparation qualification.
_DIAGNOSTIC_ELEMENTARY_CHARGE_C = 1.602176634e-19
_DIAGNOSTIC_EPSILON0_F_M = 8.8541878128e-12


def _scalar(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or np.iscomplexobj(value):
        raise ValueError(name+' requires a finite real scalar')
    array = np.asarray(value)
    if array.shape != () or not np.isfinite(float(array)):
        raise ValueError(name+' requires a finite real scalar')
    value = float(array)
    if positive and value<=0:
        raise ValueError(name+' requires a positive scalar')
    return value


def _p16_derivatives():
    polynomials = [(1,)]
    for order in range(5):
        following = [0]*(len(polynomials[-1])+1)
        for power, coefficient in enumerate(polynomials[-1]):
            if power:
                following[power-1] += power*coefficient
            following[power+1] -= (power+2*(16-order))*coefficient
        polynomials.append(tuple(following))
    return tuple(polynomials)


P16_DERIVATIVES = _p16_derivatives()


def _envelope_derivatives(z, H, maximum=4):
    if type(maximum) is not int or not 0<=maximum<=5:
        raise ValueError('private P16 preparation supports derivative orders0..5')
    if abs(z)>=H:
        return np.zeros(maximum+1)
    u = z/H
    return np.array([(1-u*u)**(16-j)
        *np.polynomial.polynomial.polyval(u,P16_DERIVATIVES[j])/H**j
        for j in range(maximum+1)])


def _profile(z, H, forward, backward, counter_phase, maximum=4):
    envelope = _envelope_derivatives(z,H,maximum)
    derivatives = np.array([sum(amplitude*np.exp(1j*(carrier*z+phase))
        *sum(comb(j,ell)*envelope[ell]*(1j*carrier)**(j-ell)
             for ell in range(j+1))
        for amplitude,carrier,phase in ((forward,1.,0.),
                                       (backward,-1.,-counter_phase)))
        for j in range(maximum+1)])
    return np.r_[derivatives.real,derivatives.imag]


def _integral(H, w):
    return H*beta(.5,17)*hyp0f1(17.5,-(w*H)**2/4)


def prepare_p16_native_data(cavity, particle_ref, *, eta,
        radius_m, counter_phase=0., calibration_beta=1., num_cells=8,
        anchor_s=0.):
    """Freeze one diagnostic M2/nu1 source from actual Cavity controls.

    This explicit opt-in supplies a different finite full-root/Gauss map;
    it does not reinterpret any existing model/integrator selector. Existing
    Cavity tracking remains unchanged. Later changes require a fresh call.
    """
    if getattr(cavity._xobject, '_tpsa_enabled', 0):
        raise ValueError('private native diagnostic requires scalar controls')
    if (cavity.ks != 0. or np.any(np.asarray(cavity.knl) != 0.)
            or np.any(np.asarray(cavity.ksl) != 0.)):
        raise ValueError('private native diagnostic does not implement static fields')
    if (cavity.wave_number is not None or cavity.rf_field_order != 2
            or cavity.rf_profile != 'p16'):
        raise ValueError('private native diagnostic requires its fixed M2/nu1 source')
    if cavity.radiation_flag or cavity.spin_tracking:
        raise ValueError('private native diagnostic does not consume Cavity effect flags')
    if cavity.harmonic!=0:
        raise ValueError('private native diagnostic requires explicit frequency')
    if not cavity.absolute_time:
        raise ValueError('private native diagnostic requires absolute_time=True')
    if cavity.phase_taper!=0 or cavity.lag_taper!=0:
        raise ValueError('private native diagnostic does not implement tapers')
    if cavity.transformations_active:
        raise ValueError('private native diagnostic requires fixed alignment')
    if cavity.model!='adaptive' or cavity.integrator!='adaptive':
        raise ValueError('do not reinterpret existing model/integrator selectors')
    length = _scalar(cavity.length,'length',positive=True)
    frequency = _scalar(cavity.frequency,'frequency',positive=True)
    voltage = _scalar(cavity.voltage,'voltage')
    phase = _scalar(cavity.phase,'phase')+np.pi/180*_scalar(cavity.lag,'lag')
    radius_m = _scalar(radius_m,'runtime radius guard',positive=True)
    calibration_beta = _scalar(calibration_beta,'calibration_beta',positive=True)
    if calibration_beta>1:
        raise ValueError('calibration_beta must be at most one')
    counter_phase = _scalar(counter_phase,'counter_phase')
    anchor_s = _scalar(anchor_s,'physical anchor_s')
    if isinstance(eta,(bool,np.bool_)) or np.ndim(eta)!=0 or np.iscomplexobj(eta):
        raise ValueError('eta must be a nonnegative real scalar')
    eta = float(eta)
    if np.isnan(eta) or eta<0:
        raise ValueError('eta must be nonnegative, allowing infinity')
    if isinstance(num_cells,(bool,np.bool_)):
        raise ValueError('num_cells must be a positive integer')
    num_cells = index(num_cells)
    if num_cells<=0:
        raise ValueError('num_cells must be a positive integer')
    mass0 = _scalar(particle_ref.mass0,'fixed reference mass0',positive=True)
    q0 = _scalar(particle_ref.q0,'fixed reference q0')
    p0c = np.asarray(particle_ref.p0c)
    beta0 = np.asarray(particle_ref.beta0)
    if (q0==0 or p0c.shape!=(1,) or beta0.shape!=(1,)
            or not np.isfinite(p0c[0]) or p0c[0]<=0
            or not np.isfinite(beta0[0]) or not 0<beta0[0]<1
            or not np.all(np.asarray(particle_ref.mass_ratio)==1.)
            or not np.all(np.asarray(particle_ref.charge_ratio)==1.)):
        raise ValueError('one charged fixed unit-ratio reference is required')
    omega = 2*np.pi*frequency
    k = omega/C_LIGHT
    H = k*length/2
    ratio = np.sqrt(eta)
    slow_weight = 0. if np.isinf(eta) else 1/(1+ratio)
    fast_weight = 1. if np.isinf(eta) else ratio/(1+ratio)
    slow = slow_weight*_integral(H,1/calibration_beta-1.)
    fast = fast_weight*np.exp(1j*counter_phase)*_integral(
        H,1/calibration_beta+1.)
    phasor = np.exp(1j*H/calibration_beta)*(slow+fast)
    if voltage and abs(phasor)<=64*np.finfo(float).eps*(abs(slow)+abs(fast)):
        raise ValueError('electric crest calibration phasor cancels')
    amplitude = abs(voltage)/mass0/abs(phasor) if voltage else 0.
    sign = -1. if voltage<0 else 1.
    forward = sign*amplitude*slow_weight
    backward = sign*amplitude*fast_weight
    clock_origin = phase-np.copysign(np.pi/2,q0)-np.angle(phasor)
    mesh = np.linspace(-H,H,num_cells+1)
    profiles = np.array([[_profile(z,H,forward,backward,counter_phase)
        for z in (left,*(left+GAUSS_NODES*(right-left)),right)]
        for left,right in zip(mesh[:-1],mesh[1:])])
    if not np.all(np.isfinite(profiles)):
        raise ValueError('finite prepared profile data required')
    data = dict(mesh=mesh,profiles=profiles,k=k,omega=omega,
        source_mass_eV=mass0,anchor_s=anchor_s,
        clock_origin=float(clock_origin),aperture=radius_m*k)
    metadata = MappingProxyType(dict(frequency_hz=frequency,voltage_v=voltage,
        phase_rad=phase,length_m=length,eta=eta,counter_phase=counter_phase,
        calibration_beta=calibration_beta,reference_charge_e=q0,
        source_mass_eV=mass0,reference_p0c_eV=float(p0c[0]),
        runtime_radius_guard_m=radius_m,radial_order=2,carrier_nu=1.,
        profile_half_span=float(H),forward_strength=float(forward),
        backward_strength=float(backward),
        absolute_time=True,num_cells=num_cells,
        source_qualified=False,numerical_error_certified=False,
        backend_arithmetic_qualified=False,accepted_production_return=False))
    return data,metadata


def prepare_native_diagnostic(cavity, particle_ref, *, eta, radius_m,
        counter_phase=0., calibration_beta=1., num_cells=8, anchor_s=0.,
        _context=None):
    """Prepare a fresh independent source; no lazy cache follows Cavity edits.

    Calibration uses particle_ref.q0 for the established public phase polarity.
    For the historical hardware calibrated with q0=+1,phase=0, an electron
    native reference q0=-1 requires phase=-pi to preserve that same field and
    clock. Recalibrating the electron at phase=0 would reverse the laboratory
    field. Actual tracked signed charge/mass still enters the kernel once.
    """
    data, metadata = prepare_p16_native_data(cavity, particle_ref, eta=eta,
        radius_m=radius_m, counter_phase=counter_phase,
        calibration_beta=calibration_beta, num_cells=num_cells,
        anchor_s=anchor_s)
    if _context is None:
        _context = particle_ref._context
    return CavityNativeDiagnostic(_context=_context,
        preparation=dict(metadata), **data)


class CavityNativeDiagnostic(BeamElement):
    """Private full-root orbit and observational finite-map derivative.

    Source controls/readbacks are immutable through the supported Python API.
    This is not an enforced memory boundary: direct private _xobject mutation
    is unsupported. Runtime guards reject malformed backend data, not every
    coherent alternate source or an intentional bypass of preparation.
    Interior native px/py are mechanical in the declared zero-native-A chart;
    the derivative is not standard canonical interior native Twiss.
    High-level read-only observations copy Particles in the same context:
    their cost includes the full particle capacity. CPU callbacks may compact
    that copy; returned IDs/readbacks follow its order. Caller storage, order,
    counters and clock remain untouched. Raw native_* per-particle kernels
    retain their ordinary CPU reorganization behavior.
    """

    _xofields = {
        'length': xo.Float64, 'k': xo.Float64, 'omega': xo.Float64,
        'anchor_s': xo.Float64, 'clock_origin': xo.Float64,
        'source_mass_eV': xo.Float64, 'aperture': xo.Float64,
        'tolerance': xo.Float64, 'maximum_iterations': xo.Int64,
        'derivative_tolerance': xo.Float64,
        'spin_tracking': xo.Int64, 'spin_tolerance': xo.Float64,
        'cells': xo.Int64, 'mesh': xo.Float64[:], 'profiles': xo.Float64[:],
    }
    _source_fields = ('length', 'k', 'omega', 'anchor_s', 'clock_origin',
        'source_mass_eV', 'aperture', 'cells', 'mesh', 'profiles')
    _rename = {name: '_source_'+name for name in _source_fields}
    isthick = True
    has_backtrack = True
    allow_rot_and_shift = False
    accepted_production_return = False
    numerical_error_certified = False
    physical_source_qualified = False
    backend_arithmetic_qualified = False
    _repr_fields = ('length', 'cells', 'tolerance')
    _extra_c_sources = [
        '#define XTRACK_CAVITY_NATIVE_TANGENT\n'
        '#define XTRACK_CAVITY_NATIVE_SECONDARY\n'
        '#define XTRACK_CAVITY_NATIVE_SPIN\n'
        '#define XTRACK_CAVITY_NATIVE_AUDIT\n'
        '#define XTRACK_CAVITY_NATIVE_RESPONSE\n'
        '#include "xtrack/beam_elements/elements_src/cavity_native_adapter.h"']
    _per_particle_kernels = {
        'native_tangent': xo.Kernel(
            c_name='CavityNativeDiagnostic_tangent_local_particle', args=[
                xo.Arg(xo.Int64, name='reverse'),
                xo.Arg(xo.Float64, pointer=True, name='output'),
                xo.Arg(xo.Int64, pointer=True, name='statuses')]),
        'native_secondary': xo.Kernel(
            c_name='CavityNativeDiagnostic_secondary_local_particle', args=[
                xo.Arg(xo.Float64, name='position'),
                xo.Arg(xo.Float64, pointer=True, name='profile'),
                xo.Arg(xo.Float64, name='rho_unit_eV'),
                xo.Arg(xo.Float64, pointer=True, name='output'),
                xo.Arg(xo.Int64, pointer=True, name='statuses')]),
        'native_audit_first_cell': xo.Kernel(
            c_name='CavityNativeDiagnostic_audit_first_cell_local_particle',args=[
                xo.Arg(xo.Float64,pointer=True,name='output'),
                xo.Arg(xo.Float64,pointer=True,name='word'),
                xo.Arg(xo.Int64,pointer=True,name='statuses')]),
        'native_schott_response': xo.Kernel(
            c_name='CavityNativeDiagnostic_schott_response_local_particle',args=[
                xo.Arg(xo.Float64,name='rho_unit_eV'),
                xo.Arg(xo.Float64,name='response_tolerance'),
                xo.Arg(xo.Float64,pointer=True,name='output'),
                xo.Arg(xo.Int64,pointer=True,name='statuses')]),
        'native_schott_spin_response': xo.Kernel(
            c_name='CavityNativeDiagnostic_schott_spin_response_local_particle',args=[
                xo.Arg(xo.Float64,name='rho_unit_eV'),
                xo.Arg(xo.Float64,name='response_tolerance'),
                xo.Arg(xo.Float64,name='spin_tolerance'),
                xo.Arg(xo.Float64,name='shifted_spin_tolerance'),
                xo.Arg(xo.Float64,pointer=True,name='output'),
                xo.Arg(xo.Int64,pointer=True,name='statuses')]),
    }

    def __init__(self, *, mesh=None, profiles=None, k=None, omega=None,
            source_mass_eV=None, anchor_s=0., clock_origin=0., aperture=1.,
            tolerance=2e-13, maximum_iterations=80,
            derivative_tolerance=2e-13, spin_tracking=0, spin_tolerance=2e-13,
            preparation=None, **kwargs):
        metadata = dict(preparation or {})
        for flag in ('source_qualified', 'numerical_error_certified',
                     'backend_arithmetic_qualified', 'accepted_production_return'):
            if metadata.get(flag, False) is not False:
                raise ValueError('private diagnostic has no '+flag+' admission')
        self._preparation = tuple(sorted(metadata.items()))
        if kwargs.get('_xobject') is not None:
            self.xoinitialize(**kwargs)
            return
        mesh = np.asarray(mesh, dtype=float)
        profiles = np.asarray(profiles, dtype=float)
        if (mesh.ndim!=1 or len(mesh)<2 or not np.all(np.isfinite(mesh))
                or not np.all(np.diff(mesh)>0)):
            raise ValueError('finite strictly increasing mesh required')
        cells = len(mesh)-1
        if profiles.shape==(cells*40,):
            profiles = profiles.reshape(cells,4,10)
        if profiles.shape!=(cells,4,10) or not np.all(np.isfinite(profiles)):
            raise ValueError('finite cell profiles require shape(cells,4,10)')
        for name,value in (('k',k), ('omega',omega),
                ('source_mass_eV',source_mass_eV), ('aperture',aperture),
                ('tolerance',tolerance), ('derivative_tolerance',derivative_tolerance),
                ('spin_tolerance',spin_tolerance)):
            _scalar(value,name,positive=True)
        if (not isinstance(spin_tracking,(int,np.integer,bool,np.bool_))
                or int(spin_tracking) not in (0,1)):
            raise ValueError('spin_tracking must be zero or one')
        if (isinstance(maximum_iterations,(bool,np.bool_))
                or not isinstance(maximum_iterations,(int,np.integer))
                or maximum_iterations<=0):
            raise ValueError('maximum_iterations must be a positive integer')
        _scalar(anchor_s,'physical anchor_s')
        _scalar(clock_origin,'clock_origin')
        if abs(omega/k-C_LIGHT)>8*np.finfo(float).eps*C_LIGHT:
            raise ValueError('k and omega must describe the same physical clock')
        length = (mesh[-1]-mesh[0])/k
        supplied_length = kwargs.pop('length',length)
        supplied_cells = kwargs.pop('cells',cells)
        if (supplied_cells!=cells or not np.isfinite(supplied_length)
                or abs(supplied_length-length)>16*np.finfo(float).eps*length):
            raise ValueError('stored length/cells must match the prepared mesh')
        super().__init__(length=length, mesh=mesh, profiles=profiles.ravel(),
            cells=cells, k=k, omega=omega, source_mass_eV=source_mass_eV,
            anchor_s=anchor_s, clock_origin=clock_origin, aperture=aperture,
            tolerance=tolerance, maximum_iterations=maximum_iterations,
            derivative_tolerance=derivative_tolerance,
            spin_tracking=int(spin_tracking),spin_tolerance=spin_tolerance,**kwargs)

    @property
    def preparation(self):
        """Return immutable semantic inputs and explicit qualification limits."""
        return MappingProxyType(dict(self._preparation))

    def to_dict(self, **kwargs):
        data = super().to_dict(**kwargs)
        data['preparation'] = dict(self.preparation)
        return data

    def profile_at(self, position, *, maximum=4):
        """Return immutable same-source spatial jets at a declared plane.

        Explicit-data construction alone supplies no analytic source binding;
        this method requires the factory's frozen P16 preparation metadata.
        Higher requested slots are not invented by extending a cached table.
        """
        position = _scalar(position,'normalized source plane')
        source = self.preparation
        required = ('profile_half_span','forward_strength',
                    'backward_strength','counter_phase')
        if any(name not in source for name in required):
            raise ValueError('factory P16 source metadata required for query jets')
        result = _profile(position,source['profile_half_span'],
            source['forward_strength'],source['backward_strength'],
            source['counter_phase'],maximum)
        if not np.all(np.isfinite(result)):
            raise ValueError('finite query profile jets required')
        return np.frombuffer(result.tobytes(),dtype=float)

    def observe_secondary(self, particles, position):
        """Observe classical LL/BMT/accounting at one same-event source point.

        This does not advance orbit or spin and supplies no photon law. Rows
        0:6 are canonical LL source per Z; 6:9/9:12 are Lorentz/reacted BMT
        omega per T; 12:18 are uncharged SOURCE-mass-normalized E/B. Rows
        18:25 contain external/reaction/Lorentz-emission powers, Schott ENERGY,
        Lorentz remainder, actual-LL emission and actual remainder. Powers
        are per T; Schott energy is not a rate. Only per-T omega/power rows
        need gamma/uz once in a spatial history quadrature. Canonical LL0:6
        is already per Z; Schott row21 is an endpoint energy. This point
        observer applies no additional clock multiplier to any row.

        Results are indexed by the returned post-callback particle_id array
        of the working copy; caller storage is unchanged. Status0
        means inactive/unvisited. Failed outputs retain NaNs and no particle
        coordinates or spin are changed by identity. Raw private-kernel calls
        may stress rho_unit; this factory route uses actual tau0*storedomega.
        """
        position = _scalar(position,'normalized source plane')
        if particles._context is not self._context:
            raise ValueError('observer particles and source require one context')
        context = self._context
        particles = particles.copy(_context=context)
        profile = context.nparray_to_context_array(
            self.profile_at(position,maximum=5))
        rho_unit = (_DIAGNOSTIC_ELEMENTARY_CHARGE_C*self.omega
            /(6*np.pi*_DIAGNOSTIC_EPSILON0_F_M*C_LIGHT))
        output = context.nparray_to_context_array(
            np.full((particles._capacity,25),np.nan))
        statuses = context.nparray_to_context_array(
            np.zeros(particles._capacity,dtype=np.int64))
        self.native_secondary(particles,position=position,profile=profile,
            rho_unit_eV=rho_unit,output=output,statuses=statuses)
        return dict(values=context.nparray_from_context_array(output).copy(),
            status=context.nparray_from_context_array(statuses).copy(),
            particle_id=context.nparray_from_context_array(
                particles.particle_id).copy(),rho_unit_eV=rho_unit)

    def observe_schott_response(self, particles, *, response_tolerance=2e-13):
        """Observe the new first-rho Schott-coordinate forward Gauss word.

        Physical canonical response starts at zero. Its Schott coordinate
        xi=-s_in is initialized once and carried across the whole word;
        the returned canonical response is xi+s_out, not xi. native_response
        applies the full final physical-boundary Jacobian in the declared
        zero-native-A mechanical chart. rho_native_response remains separate
        from the nominal endpoint to avoid losing a small response in addition.

        work_coefficients columns are changed RF work, forward emitted energy
        and GLOBAL Schott endpoint change; work_eV scales each by actualmass*rho.
        The latter is s_finalE-s_initialE, not a sum of per-cell face jumps:
        raw tables may have inconsistent shared faces and have no qualified
        continuous physical source interpretation.
        Their defining energy identity is r_E=RF-emission+Schott for physical
        initial r=0, up to finite solve/arithmetic defects of THIS new word.
        This does not assert the old direct-LL-word quadrature identity, a
        nonlinear LL map, feedback, reaction spin or numerical/physical
        admission. Residuals are diagnostics, not error bounds.

        No particle state, orbit, spin or RNG is changed. Results follow the
        returned post-callback IDs of the copy; caller storage is unchanged.
        Inactive status0 and failed outputs retain
        NaNs. This CPU observation ignores optional passive spin and requires
        the same no-concurrent-mutation assumption as audit_first_cell.
        """
        response_tolerance = _scalar(response_tolerance,
                                     'response tolerance',positive=True)
        context = self._context
        if not isinstance(context,xo.ContextCpu):
            raise NotImplementedError('Schott response currently supports CPU')
        if particles._context is not context:
            raise ValueError('response particles and source require one context')
        particles = particles.copy(_context=context)
        rho_unit = (_DIAGNOSTIC_ELEMENTARY_CHARGE_C*self.omega
            /(6*np.pi*_DIAGNOSTIC_EPSILON0_F_M*C_LIGHT))
        output = context.nparray_to_context_array(
            np.full((particles._capacity,33),np.nan))
        statuses = context.nparray_to_context_array(
            np.zeros(particles._capacity,dtype=np.int64))
        self.native_schott_response(particles,rho_unit_eV=rho_unit,
            response_tolerance=response_tolerance,output=output,statuses=statuses)
        values = context.nparray_from_context_array(output).copy()
        return self._response_readback(values,statuses,particles,rho_unit)

    def _response_readback(self,values,statuses,particles,rho_unit):
        """Common value readback for the two explicit response observers."""
        context = self._context
        return dict(nominal_endpoint=values[:,:6],
            canonical_response=values[:,6:12],native_response=values[:,12:18],
            rho_native_response=values[:,18:24],
            work_coefficients=values[:,24:27],work_eV=values[:,27:30],
            actual_rho=values[:,30],residual=values[:,31],
            response_residual=values[:,32],
            status=context.nparray_from_context_array(statuses).copy(),
            particle_id=context.nparray_from_context_array(
                particles.particle_id).copy(),rho_unit_eV=rho_unit)

    def observe_schott_spin_response(self, particles, *, response_tolerance=2e-13,
            spin_tolerance=None, shifted_spin_tolerance=1e-20):
        """Observe the GLOBAL Y=P-KR first-rho spin/mean forward word.

        Physical r and P start at zero, R at identity: xi=-s_in and Y=-K_in
        are initialized ONCE and carried across all cached faces. P=Y+K_out R
        is reconstructed from the actual final face; raw inconsistent faces
        never trigger an implicit Y reset. This is a different finite word
        from the older per-cell physical-P response. Mean outputs follow
        observe_schott_response including the full physical boundary Jacobian.

        nominal_rotation and nominal_spin=R*S_in are separate from physical
        P, rho*P and rho*P*S_in. No tiny response is added to a nominal state.
        Actual per-particle anomaly and arbitrary finite supplied spin are
        used, without normalization, species inference or RNG. A formal
        coefficient for a spinless nucleus is not a physical spin observable.

        The explicit default Y tolerance1e-20 is for this zero-entry,
        field-free target. Interior/nonzero-Y initializations require an
        explicit realizable tolerance; no tolerance is silently relaxed.
        Residuals and sampled accuracy comparisons are diagnostics, not error
        certificates. No finite-rho mean/spin feedback, stochastic law, spin
        optics, physical-source qualification or map acceptance is supplied.
        All output/status is external, CPU-only, forward-only and indexed by
        post-callback particle IDs of the copy. Caller storage is unchanged.
        The no-concurrent-mutation assumption of audit_first_cell also applies.
        """
        context = self._context
        if not isinstance(context,xo.ContextCpu):
            raise NotImplementedError('Schott spin response currently supports CPU')
        if particles._context is not context:
            raise ValueError('spin response particles and source require one context')
        particles = particles.copy(_context=context)
        response_tolerance = _scalar(response_tolerance,'response tolerance',positive=True)
        spin_tolerance = _scalar(self.spin_tolerance if spin_tolerance is None
            else spin_tolerance,'spin tolerance',positive=True)
        shifted_spin_tolerance = _scalar(shifted_spin_tolerance,
                                         'shifted spin tolerance',positive=True)
        rho_unit = (_DIAGNOSTIC_ELEMENTARY_CHARGE_C*self.omega
            /(6*np.pi*_DIAGNOSTIC_EPSILON0_F_M*C_LIGHT))
        output = context.nparray_to_context_array(
            np.full((particles._capacity,69),np.nan))
        statuses = context.nparray_to_context_array(
            np.zeros(particles._capacity,dtype=np.int64))
        self.native_schott_spin_response(particles,rho_unit_eV=rho_unit,
            response_tolerance=response_tolerance,spin_tolerance=spin_tolerance,
            shifted_spin_tolerance=shifted_spin_tolerance,output=output,statuses=statuses)
        values = context.nparray_from_context_array(output).copy()
        result = self._response_readback(values,statuses,particles,rho_unit)
        result.update(nominal_rotation=values[:,33:42].reshape(-1,3,3),
            nominal_spin=values[:,42:45],
            physical_spin_response=values[:,45:54].reshape(-1,3,3),
            rho_spin_response=values[:,54:63].reshape(-1,3,3),
            rho_spin_vector_response=values[:,63:66],
            local_spin_residual=values[:,66],global_spin_residual=values[:,67],
            shifted_spin_residual=values[:,68])
        return result

    def enable_passive_spin(self, *, tolerance=None):
        """Opt into same-Gauss Lorentz BMT transport on the conservative orbit.

        This rotates the supplied finite dimensionless spin vector in the
        fixed straight-element frame, using the actual particle anomaly.
        It does not normalize spin or infer an anomaly from species metadata.
        Spin and orbit are committed together only after the whole word passes.
        LL/Thomas reaction, mean orbit feedback and photon noise are absent.
        The tolerance is an operator-stage residual, not a physical spin-error
        bound. native_tangent still returns the six-dimensional orbit derivative
        only; spin launch derivatives are not supplied by that observer.
        """
        if tolerance is not None:
            self.spin_tolerance = _scalar(tolerance,'spin tolerance',positive=True)
        else:
            _scalar(self.spin_tolerance,'spin tolerance',positive=True)
        self.spin_tracking = 1
        return self

    def audit_first_cell(self, particles):
        """Return one actual native orbit candidate without changing particles.

        qin, accepted stages and qout are copied in C before boundary decode;
        no ideal launch, host solve or reconstructed native endpoint replaces
        them. Only the FIRST FORWARD cell is observed, regardless of optional
        spin tracking. This is not a spin, boundary, ODE, physical-source or
        global backend-arithmetic certificate and sets no acceptance flag.

        The actual source/control data are frozen in the returned snapshot.
        Cached-plane positions replay the exact forward NumPy preparation
        convention from that snapshot and the stored binary C; they are not
        independent device position measurements or reverse affine nodes.
        CPU operation and no concurrent source/control/private-buffer mutation
        are supported assumptions, not memory or permission enforcement.
        Failed output and word buffers retain NaNs; status0 is unvisited.
        All arrays use the working copy's returned POST-callback particle IDs;
        caller storage/order/counters/clock are unchanged.
        native_input contains post-callback physical/reference readbacks from
        that same event, under the nonmutation and no-concurrency assumptions
        above. x, y, zeta and s are metres; p0c and mass0 are eV; time_s is
        seconds; px/py retain native reference-momentum normalization. This
        does not qualify the SI clock or force embedding.
        """
        context = self._context
        if not isinstance(context,xo.ContextCpu):
            raise NotImplementedError('native candidate audit currently supports CPU')
        if particles._context is not context:
            raise ValueError('audit particles and source require one context')
        particles = particles.copy(_context=context)

        def frozen(value, dtype=float):
            array = np.asarray(value,dtype=dtype)
            return np.frombuffer(array.tobytes(),dtype=dtype).reshape(array.shape)

        mesh = frozen(context.nparray_from_context_array(self._source_mesh))
        raw = frozen(context.nparray_from_context_array(self._source_profiles))
        cells = int(self.cells)
        profiles = (raw.reshape(cells,4,10)
            if cells>0 and cells==len(mesh)-1 and raw.size%40==0
                and cells==raw.size//40 else raw)
        planes = np.full(4,np.nan)
        if len(mesh)>1:
            h = mesh[1]-mesh[0]
            planes = np.r_[mesh[0],mesh[0]+GAUSS_NODES*h,mesh[1]]
        snapshot = dict(mesh=mesh,profiles=profiles,cells=cells,
            k=self.k,omega=self.omega,anchor_s=self.anchor_s,
            clock_origin=self.clock_origin,source_mass_eV=self.source_mass_eV,
            aperture=self.aperture,tolerance=self.tolerance,
            maximum_iterations=int(self.maximum_iterations),
            spin_tracking=int(self.spin_tracking),preparation=dict(self.preparation),
            cached_row_ids=frozen(np.arange(4),dtype=np.int64),
            cached_plane_positions=frozen(planes),
            cached_plane_position_convention=
                'forward_numpy_float64_left_plus_stored_C_times_h_replay')
        output = context.nparray_to_context_array(
            np.full((particles._capacity,24),np.nan))
        word = context.nparray_to_context_array(
            np.full((particles._capacity,5),np.nan))
        statuses = context.nparray_to_context_array(
            np.zeros(particles._capacity,dtype=np.int64))
        self.native_audit_first_cell(particles,output=output,word=word,
                                     statuses=statuses)
        values = context.nparray_from_context_array(output).copy()
        words = context.nparray_from_context_array(word).copy()
        status = context.nparray_from_context_array(statuses).copy()
        native_input = {name: frozen(context.nparray_from_context_array(
            getattr(particles,name)),dtype=(np.int64 if name=='state' else float))
            for name in ('x','px','y','py','zeta','delta','s','p0c','beta0',
                         'gamma0','charge_ratio','chi','state')}
        native_input.update({name: float(getattr(particles,name))
                             for name in ('mass0','q0','time_s')})
        cell_index = np.full(particles._capacity,-1,dtype=np.int64)
        cell_index[status==1] = words[status==1,0].astype(np.int64)
        return dict(qin=values[:,:6].copy(),
            stages=values[:,6:18].reshape(-1,2,6).copy(),
            qout=values[:,18:24].copy(),status=status,
            particle_id=context.nparray_from_context_array(particles.particle_id).copy(),
            cell_index=cell_index,start=words[:,1].copy(),step=words[:,2].copy(),
            effective_Q=words[:,3].copy(),residual=words[:,4].copy(),
            tableau=dict(A=frozen(_STORED_GAUSS_A),C=frozen(GAUSS_NODES),
                         b=frozen((.5,.5))),source_snapshot=snapshot,
            native_input=native_input)


def _source_readback(name):
    """Expose source arrays as immutable independent bytes, not linked views."""
    def read(self):
        value = getattr(self,'_source_'+name)
        if name not in ('mesh','profiles'):
            return value
        array = self._context.nparray_from_context_array(value)
        result = np.frombuffer(array.tobytes(),dtype=float)
        if name=='profiles':
            result = result.reshape(self.cells,4,10)
        return result
    return property(read)


for _name in CavityNativeDiagnostic._source_fields:
    setattr(CavityNativeDiagnostic,_name,_source_readback(_name))
