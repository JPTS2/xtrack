"""Track and Twiss the supplied SuperKEKB conversion with whole RF parents.

Example from the Xtrack checkout (input files are read only)::

    python examples/cavity_models/003_superkekb.py \
      --lattice ../cavity-modelling/validation/inputs/rfg_linac_bte.py \
      --optics ../cavity-modelling/validation/inputs/rfg_linac_bte_import_optics.py \
      --stop-s 30 --json build/linac.json

The ideal P16 profile is an ASSUMPTION, not a measured SuperKEKB field. The
frozen model has its separate uniform voltage-action/face assumptions. The
literal conversion's split RF/MULT pairs are first consolidated into one
physical RF parent: no repeated P16 calibration or internal artificial faces.

All models consume ONE fixed nominal momentum and RF-clock programme derived
from the input voltage/phase settings. The clock anchor advances L/beta_nom at
each original parent; actual arrival residuals are carried, not reset by a
model-dependent pilot. ReferenceEnergyIncrease uses time-based zeta updating
so its coordinate change preserves the physical event. This declared clock
programme is not a newly measured/extracted SAD laboratory TOF.

The example depends on Xtrack only. Optional saved SAD optics are comparison
INPUTS; source-algorithm agreement or whole-line release qualification does
not follow from plotting them. Existing converter/magnet/face differences
must be separated from RF model differences.
"""
import argparse
import json
from pathlib import Path
import re
from time import perf_counter

import numpy as np
import xtrack as xt
from _context import cpu_context


COORDS = ('x', 'px', 'y', 'py', 'zeta', 'delta')
MODELS = ('phase-resolved', 'phase-resolved-paraxial',
          'phase-resolved-longitudinal', 'phase-resolved-frozen')
PART = re.compile(r'(.+)_(mult|cavi)_(\d+)(?:\.\d+)?$')


def _element(line, name):
    element = line._element_dict[name]
    if isinstance(element, xt.Replica):
        element = element.resolve(line)
    return element


def _sum_arrays(elements, field):
    arrays = [np.asarray(getattr(element, field), dtype=float) for element in elements]
    count = max((len(array) for array in arrays), default=0)
    result = np.zeros(count)
    for array in arrays:
        result[:len(array)] += array
    return result


def whole_parents(line):
    """Consolidate only explicit contiguous RF/MULT groups; refuse ambiguity."""
    result = []
    names = list(line.element_names)
    i = 0
    while i < len(names):
        name = names[i]
        element = _element(line, name)
        match = PART.fullmatch(name)
        if match and match[2] == 'mult' and int(match[3]) == 0:
            prefix = match[1]
            bodies, cavities, consumed = [], [], []
            j = i
            index = 0
            while j+1 < len(names):
                first, second = PART.fullmatch(names[j]), PART.fullmatch(names[j+1])
                if not (first and second and first[1] == second[1] == prefix
                        and first[2] == 'mult' and second[2] == 'cavi'
                        and int(first[3]) == int(second[3]) == index):
                    break
                body, cavity = _element(line, names[j]), _element(line, names[j+1])
                if not isinstance(cavity, xt.Cavity) or float(cavity.length) != 0.:
                    raise ValueError('Expected thin converted RF member '+names[j+1])
                if any(float(getattr(body, key, 0.)) != 0. for key in
                       ('hxl', 'shift_x', 'shift_y', 'shift_s', 'rot_s_rad')):
                    raise NotImplementedError('Unreviewed converted overlap control '+names[j])
                bodies.append(body); cavities.append(cavity)
                consumed.extend(names[j:j+2]);j += 2;index += 1
            if not cavities:
                raise ValueError('Empty RF/MULT group '+name)
            for field in ('frequency', 'phase', 'lag', 'harmonic'):
                if any(float(getattr(cavity, field)) != float(getattr(cavities[0], field))
                       for cavity in cavities):
                    raise ValueError('Nonuniform parent RF control '+prefix+' '+field)
            if any(float(cavity.harmonic) != 0. for cavity in cavities):
                raise NotImplementedError('This fixed programme requires explicit RF frequency')
            result.append(dict(name=prefix, members=consumed, kind='rf',
                length=sum(float(body.length) for body in bodies),
                voltage=sum(float(cavity.voltage) for cavity in cavities),
                frequency=float(cavities[0].frequency),
                phase=float(cavities[0].phase)+np.deg2rad(float(cavities[0].lag)),
                knl=_sum_arrays(bodies, 'knl').tolist(),
                ksl=_sum_arrays(bodies, 'ksl').tolist(),
                ks=float(getattr(bodies[0], 'ks', 0.))))
            if any(float(getattr(body, 'ks', 0.)) != result[-1]['ks'] for body in bodies):
                raise ValueError('Nonuniform overlap solenoid '+prefix)
            i = j
            continue
        if isinstance(element, xt.Cavity):
            length, voltage = float(element.length), float(element.voltage)
            if length <= 0.:
                if voltage != 0.:
                    raise NotImplementedError('Unpaired thin RF parent '+name)
                result.append(dict(name=name, kind='other', element=xt.Marker(), length=0.))
            else:
                result.append(dict(name=name, members=[name], kind='rf',
                    length=length, voltage=voltage, frequency=float(element.frequency),
                    phase=float(element.phase)+np.deg2rad(float(element.lag)),
                    knl=np.asarray(element.knl).tolist(),ksl=np.asarray(element.ksl).tolist(),
                    ks=float(element.ks)))
        else:
            result.append(dict(name=name,kind='other',element=element,
                length=float(getattr(element, 'length', 0.))))
        i += 1
    return result


def nominal_programme(parents, particle_ref):
    """Use input hardware once, independently of the model being compared."""
    mass = float(particle_ref.mass0)
    charge = abs(float(particle_ref.q0))
    momentum = float(particle_ref.p0c[0])
    anchor, s = 0., 0.
    result = []
    for parent in parents:
        entry = dict(parent, p0c_in=momentum, s_in=s, clock_anchor_s=anchor)
        beta = momentum/np.hypot(momentum, mass)
        anchor += parent['length']/(beta*299792458.)
        s += parent['length']
        if parent['kind'] == 'rf':
            entry['nominal_energy_gain_eV'] = charge*parent['voltage']*np.sin(parent['phase'])
            energy = np.hypot(momentum, mass)+entry['nominal_energy_gain_eV']
            if energy <= mass:
                raise ValueError('Nonphysical nominal reference at '+parent['name'])
            momentum = np.sqrt((energy-mass)*(energy+mass))
        entry.update(p0c_out=momentum,s_out=s)
        result.append(entry)
    return result


def build_line(programme, reference, model, steps, context, *, stop_s=None, parent=None):
    elements, names, selected = [], [], []
    for index, record in enumerate(programme):
        if parent is not None and record['name'].lower() != parent.lower():
            continue
        if stop_s is not None and record['s_out'] > stop_s:
            break
        name = record['name']+'__occ_'+str(index)
        if record['kind'] == 'rf':
            phase = record['phase']
            # A single-parent case chooses its declared entrance event as t=0.
            if parent is None:
                phase -= 2*np.pi*record['frequency']*record['clock_anchor_s']
            element = xt.Cavity(_context=context, length=record['length'],
                voltage=record['voltage'], frequency=record['frequency'],phase=phase,
                absolute_time=parent is None, eta=0., calibration_beta=1.,
                model=model,num_kicks=steps,knl=record['knl'],ksl=record['ksl'],ks=record['ks'],
                nominal_energy_gain=record['nominal_energy_gain_eV'])
        else:
            element = record['element'].copy(_context=context)
        elements.append(element);names.append(name);selected.append(record)
        if record['kind'] == 'rf':
            elements.append(xt.ReferenceEnergyIncrease(_context=context,
                Delta_p0c=record['p0c_out']-record['p0c_in']))
            names.append(name+'__reference')
    if not selected:
        raise ValueError('No selected source interval')
    elements.append(xt.Marker(_context=context));names.append('comparison_end')
    line = xt.Line(elements=elements,element_names=names)
    initial = reference.copy(_context=context)
    if parent is not None:
        initial.p0c = selected[0]['p0c_in']
    line.particle_ref = initial
    line.reset_s_at_end_turn = False
    line.build_tracker(_context=context)
    line.use_time_based_zeta_update = True
    return line, selected


def finite_difference_transport(line, initial):
    """Raw coordinate Jacobian at two steps, independent of Twiss mode labels."""
    steps = np.full(6, 1e-6)
    offsets = [np.zeros(6)]
    for scale in (1., .5):
        for axis in range(6):
            for sign in (1., -1.):
                offset = np.zeros(6)
                offset[axis] = sign*scale*steps[axis]
                offsets.append(offset)
    offsets = np.asarray(offsets)
    rays = xt.Particles(_context=line._buffer.context,
        mass0=float(initial.mass0), q0=float(initial.q0),
        p0c=float(initial.p0c[0]),
        **{key:float(getattr(initial,key)[0])+offsets[:,axis]
           for axis,key in enumerate(COORDS)})
    line.track(rays, ele_stop=len(line.element_names)-1)
    if not np.all(np.asarray(rays.state) == 1):
        raise RuntimeError('A finite-difference transport ray was lost')
    values = np.asarray([getattr(rays,key) for key in COORDS],dtype=float).T
    matrices = []
    for group,scale in enumerate((1., .5)):
        matrix = np.empty((6,6))
        for axis in range(6):
            first = 1+group*12+axis*2
            matrix[:,axis] = (values[first]-values[first+1])/(2*scale*steps[axis])
        matrices.append(matrix)
    return dict(coordinates=list(COORDS), steps=steps.tolist(),
        matrix=matrices[1].tolist(), matrix_coarse=matrices[0].tolist(),
        matrix_step_max_difference=float(np.max(np.abs(matrices[0]-matrices[1]))),
        pilot=values[0].tolist(), pilot_ax=float(rays.ax[0]), pilot_ay=float(rays.ay[0]))


def projected_optics(transport, init):
    """Equal-input-emittance covariance diagnostic, at a field-free endpoint.

    This is distinct from the individual mode functions returned by Twiss.
    An endpoint inside a magnetic field needs the local potential derivatives
    in the canonical-to-kinetic transformation and is outside this diagnostic.
    """
    if transport['pilot_ax'] != 0. or transport['pilot_ay'] != 0.:
        return dict(available=False,reason='Endpoint has nonzero vector potential')
    covariance = np.zeros((6,6))
    for offset,plane in ((0,'x'),(2,'y')):
        beta,alpha = init['bet'+plane],init['alf'+plane]
        covariance[offset:offset+2,offset:offset+2] = (
            (beta,-alpha),(-alpha,(1.+alpha*alpha)/beta))
    matrix = np.asarray(transport['matrix'])
    _,px,_,py,_,delta = transport['pilot']
    d = 1.+delta
    w = np.sqrt(d*d-px*px-py*py)
    jacobian = np.zeros((4,6))
    jacobian[0,0] = jacobian[2,2] = 1.
    jacobian[1,1] = 1./w+px*px/w**3
    jacobian[3,3] = 1./w+py*py/w**3
    jacobian[1,3] = jacobian[3,1] = px*py/w**3
    jacobian[1,5] = -px*d/w**3
    jacobian[3,5] = -py*d/w**3
    transported = jacobian@matrix@covariance@matrix.T@jacobian.T
    result = dict(available=True,
        definition='Projected physical-slope covariance; equal unit entrance emittances, zero longitudinal spread',
        endpoint_requirement='Field-free; zero pilot vector potential alone does not establish zero field',
        covariance=transported.tolist())
    for offset,plane in ((0,'x'),(2,'y')):
        emit = np.sqrt(np.linalg.det(transported[offset:offset+2,offset:offset+2]))
        result['bet'+plane] = float(transported[offset,offset]/emit)
        result['alf'+plane] = float(-transported[offset,offset+1]/emit)
        result['emittance_ratio_'+plane] = float(emit)
    return result


def run_case(line, init):
    pilot = line.particle_ref.copy()
    line.track(pilot,ele_stop=len(line.element_names)-1)
    if pilot.state[0] != 1:
        return dict(completed=False,state=int(pilot.state[0]),
            at_element=int(pilot.at_element[0]),
            element=line.element_names[min(int(pilot.at_element[0]),len(line.element_names)-1)])
    tw = line.twiss(method='6d',start=xt.START,end=xt.END,**init)
    probe = line.particle_ref.copy();probe.x=1e-5;probe.px=2e-6
    probe.y=-1e-5;probe.py=-1e-6;probe.delta=1e-5
    before=np.array([getattr(probe,key)[0] for key in COORDS])
    line.track(probe,ele_stop=len(line.element_names)-1)
    line.track(probe,ele_start=0,ele_stop=len(line.element_names)-1,backtrack=True)
    if probe.state[0] != 1:
        raise RuntimeError('Backtrack lost the finite-amplitude probe')
    transport = finite_difference_transport(line,line.particle_ref)
    return dict(completed=True,pilot=dict(zip(COORDS,
        [float(getattr(pilot,key)[0]) for key in COORDS])),
        p0c_out_eV=float(pilot.p0c[0]),physical_arrival_time_s=float(pilot.time_s
            +(pilot.s[0]-pilot.zeta[0])/(pilot.beta0[0]*299792458.)),
        reverse_difference=dict(zip(COORDS,
            (np.array([getattr(probe,key)[0] for key in COORDS])-before).tolist())),
        finite_difference_transport=transport,
        projected_optics=projected_optics(transport,init),
        twiss={key:np.asarray(tw[key]).tolist() for key in
            ('name','s','x','px','y','py','zeta','delta','betx','bety','alfx','alfy')})


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lattice',type=Path)
    parser.add_argument('--optics',type=Path)
    parser.add_argument('--environment-json',type=Path)
    parser.add_argument('--line',default='line')
    parser.add_argument('--sad-optics-json',type=Path)
    parser.add_argument('--models',nargs='+',choices=MODELS,default=list(MODELS[:3]))
    parser.add_argument('--steps',type=int,default=128)
    parser.add_argument('--stop-s',type=float)
    parser.add_argument('--parent')
    parser.add_argument('--json',type=Path,required=True)
    parser.add_argument('--build-dir',type=Path,
        default=Path(__file__).resolve().parents[2]/'build'/'superkekb-cavity-models')
    options=parser.parse_args(argv)
    if options.environment_json:
        env=xt.Environment.from_json(options.environment_json)
    elif options.lattice:
        env=xt.Environment();env.call(str(options.lattice.resolve()))
        if options.optics:env.call(str(options.optics.resolve()))
    else:
        parser.error('--lattice or --environment-json is required')
    original=env.lines[options.line]
    programme=nominal_programme(whole_parents(original),original.particle_ref)
    sad=json.loads(options.sad_optics_json.read_text()) if options.sad_optics_json else None
    init=dict(betx=1.,bety=1.,alfx=0.,alfy=0.)
    sad_index=0
    if sad:
        if options.parent:
            sad_index=next(i for i,name in enumerate(sad['name'])
                if name.lower()==options.parent.lower())
        init={key:sad[key][sad_index] for key in
              ('x','px','y','py','betx','bety','alfx','alfy','dx','dy','dpx','dpy')}
    records=[]
    destination=options.json.resolve();destination.parent.mkdir(parents=True,exist_ok=True)
    with cpu_context(options.build_dir) as context:
        line = None
        for model in options.models:
            started=perf_counter()
            try:
                if line is None:
                    line,selected=build_line(programme,original.particle_ref,model,
                        options.steps,context,stop_s=options.stop_s,parent=options.parent)
                else:
                    # Public controls select the map on ONE compiled Line.
                    # Changing the source assumption is explicit here.
                    for element in line.elements:
                        if isinstance(element,xt.Cavity):
                            element.model = model
                            element.rf_profile = ('uniform' if model ==
                                'phase-resolved-frozen' else 'p16')
                result=run_case(line,init)
                result.update(model=model,seconds=perf_counter()-started,
                    selected_parents=len(selected),rf_parents=sum(r['kind']=='rf' for r in selected),
                    physical_length_m=sum(r['length'] for r in selected),
                    input_reference_p0c_eV=float(line.particle_ref.p0c[0]))
                if sad:
                    end_s=selected[-1]['s_out']
                    candidates=np.flatnonzero(np.isclose(np.asarray(sad['s']),end_s,rtol=0.,atol=2e-4))
                    result['sad_comparison']=dict(matching_plane_indices=candidates.tolist(),
                        end_s_m=end_s, entrance_index=sad_index,
                        warning='Conversion/magnet/clock discrepancies are not attributed solely to RF')
                    if len(candidates):
                        j=int(candidates[-1])
                        result['sad_comparison'].update(index=j,name=sad['name'][j],
                            betx_m=sad['betx'][j],bety_m=sad['bety'][j])
            except Exception as error:
                result=dict(model=model,completed=False,error=repr(error),seconds=perf_counter()-started)
            records.append(result)
            payload=dict(xtrack=str(xt.__file__),models=records,
                source='whole-parent ideal P16; frozen has distinct uniform voltage-action assumptions',
                reference_programme='fixed input nominal V*sin(phase), time-preserving momentum rebases',
                clock='fixed sum L/(beta_nom*c); physical arrival residual carried',
                eta=0.,radiation='off',
                twiss_frame='Prescribed nominal reference momentum; no model-dependent pilot retuning',
                source_or_numerical_release_qualified=False)
            destination.write_text(json.dumps(payload,indent=2)+'\n')
            summary={key:value for key,value in result.items() if key!='twiss'}
            if result.get('completed'):
                summary['exit_beta_m']=[result['twiss']['betx'][-1],result['twiss']['bety'][-1]]
            print(json.dumps(summary),flush=True)
    return int(any(not record.get('completed') for record in records))


if __name__=='__main__':
    raise SystemExit(main())
