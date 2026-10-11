from __future__ import annotations
from pathlib import Path
import importlib
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
REQUIRED_SOURCES = (
    'vehicles/remus100.py', 'vehicles/__init__.py',
    'lib/gnc.py', 'lib/control.py', 'lib/guidance.py', 'lib/__init__.py',
)

def source_root():
    """The existing repository is the sole REMUS implementation."""
    return ROOT/'PythonVehicleSimulator/src/python_vehicle_simulator'

def verify_sources(root=None):
    """Compatibility name for ordinary file availability checks.

    Local source edits and Windows line endings are accepted. Runtime interface,
    dimensions and numerical parameters are checked separately by load_model.
    """
    root=(Path(root) if root else source_root()).resolve();checks={}
    for rel in REQUIRED_SOURCES:
        path=root/rel
        checks[rel]={'exists':path.is_file()}
        if not checks[rel]['exists']:
            raise FileNotFoundError(f'Existing PythonVehicleSimulator source missing: {path}')
    return {'selected_root':str(root),'validation':'file_availability_and_runtime_interface',
            'files':checks}

def validate_vehicle_interface(vehicle):
    """Check the actual model contract without modifying model parameters."""
    for method in ('dynamics','depthHeadingAutopilot'):
        if not callable(getattr(vehicle,method,None)):
            raise TypeError(f'REMUS interface missing callable {method}')
    for name,shape in (('nu',(6,)),('u_actual',(3,)),('M',(6,6))):
        value=np.asarray(getattr(vehicle,name,None),dtype=float)
        if value.shape!=shape or not np.isfinite(value).all():
            raise ValueError(f'REMUS {name} must have shape {shape} and finite values')
    if getattr(vehicle,'dimU',None)!=3:
        raise ValueError('REMUS must expose three actuator commands')
    if not np.allclose(vehicle.M,vehicle.M.T,atol=1e-10) or np.linalg.eigvalsh(vehicle.M).min()<=0:
        raise ValueError('REMUS mass matrix must be symmetric positive definite')
    for name in ('L','diam','W','rho','deltaMax_r','deltaMax_s','nMax','T_delta','T_n','r_max'):
        value=getattr(vehicle,name,None)
        if value is None or not np.isfinite(value) or value<=0:
            raise ValueError(f'REMUS {name} must be finite and positive')
    return {'state_dimension':6,'velocity_dimension':6,'actuator_dimension':3,
            'positive_definite_mass_matrix':True,'finite_positive_parameters':True}

def load_model():
    root=source_root().resolve();verify_sources(root)
    # Use standard imports, including the repository's own package initializers.
    # A different installed copy is rejected by its path, never by file content.
    src=str(root.parent)
    if src not in sys.path:sys.path.insert(0,src)
    modules={}
    for relative in ('lib.gnc','lib.control','lib.guidance','vehicles.remus100'):
        name='python_vehicle_simulator.'+relative
        module=importlib.import_module(name)
        actual=Path(module.__file__).resolve()
        expected=root/Path(*relative.split('.')).with_suffix('.py')
        if actual!=expected:
            raise RuntimeError(f'{name} loaded from {actual}; expected repository source {expected}')
        modules[relative]=module
    cls=modules['vehicles.remus100'].remus100
    validate_vehicle_interface(cls())
    gnc=modules['lib.gnc']
    for name in ('Rzyx','Tzyx','attitudeEuler'):
        if not callable(getattr(gnc,name,None)):
            raise TypeError(f'REMUS coordinate interface missing {name}')
    return cls,gnc

class RemusPlant:
    """Thin wrapper around the repository's existing 6-DOF source.

    nu is the source model's body-frame absolute velocity. Do NOT add current
    a second time in position integration. Only source-supported horizontal
    constant current is accepted. This wrapper refuses 100-200m depth targets.
    """
    def __init__(self,depth=30.,heading_deg=50.,rpm=1525.,current_speed=.5,current_direction_deg=170.,
                 eta0=None,initialize_references=False):
        if not 0<=depth<=100:raise ValueError('REMUS target depth must remain in [0,100] m')
        if not 0<=rpm<=1525:raise ValueError('RPM must remain in [0,1525]')
        if current_speed<0 or not np.isfinite([depth,heading_deg,rpm,current_speed,current_direction_deg]).all():
            raise ValueError('invalid reference/current')
        cls,self.gnc=load_model();self.vehicle=cls('depthHeadingAutopilot',depth,heading_deg,rpm,current_speed,current_direction_deg)
        self.eta=np.zeros(6) if eta0 is None else np.asarray(eta0,dtype=float).copy()
        if self.eta.shape!=(6,) or not np.isfinite(self.eta).all():raise ValueError('eta0 must be 6 finite values')
        self.nu=self.vehicle.nu.copy();self.actual=self.vehicle.u_actual.copy();self.time_s=0.
        if initialize_references:
            self.vehicle.z_d=float(self.eta[2]);self.vehicle.psi_d=float(self.eta[5])
    def set_reference(self,depth,heading_deg,rpm):
        if not 0<=depth<=100 or not 0<=rpm<=1525:raise ValueError('reference exceeds REMUS operating envelope')
        if not np.isfinite([depth,heading_deg,rpm]).all():raise ValueError('nonfinite reference')
        self.vehicle.ref_z=float(depth);self.vehicle.ref_psi=float(heading_deg);self.vehicle.ref_n=float(rpm)
    def step(self,dt=.02,command=None,clip_autopilot_commands=False):
        if not 0<dt<=.02:raise ValueError('use REMUS integration dt <=0.02s')
        if command is None:
            command=self.vehicle.depthHeadingAutopilot(self.eta,self.nu,dt)
            if clip_autopilot_commands:
                lim=np.array([self.vehicle.deltaMax_r,self.vehicle.deltaMax_s,self.vehicle.nMax])
                command=np.clip(command,-lim,lim)
        else:
            command=np.asarray(command,dtype=float)
            lim=np.array([self.vehicle.deltaMax_r,self.vehicle.deltaMax_s,self.vehicle.nMax])
            if command.shape!=(3,) or not np.isfinite(command).all() or np.any(abs(command)>lim+1e-12):
                raise ValueError('invalid direct actuator command')
        command=np.asarray(command,dtype=float)
        if command.shape!=(3,) or not np.isfinite(command).all():
            raise ValueError('actuator command must contain three finite values')
        self.nu,self.actual=self.vehicle.dynamics(self.eta,self.nu,self.actual,command,dt)
        self.eta=self.gnc.attitudeEuler(self.eta,self.nu,dt)
        self.time_s+=dt
        if not np.isfinite(np.r_[self.eta,self.nu,self.actual]).all():raise FloatingPointError('REMUS became nonfinite')
        if abs(self.eta[4])>np.deg2rad(85):raise FloatingPointError('REMUS near Euler singularity')
        return np.asarray(command).copy()
    def relative_velocity(self):
        u_c=self.vehicle.V_c*np.cos(self.vehicle.beta_c-self.eta[5])
        v_c=self.vehicle.V_c*np.sin(self.vehicle.beta_c-self.eta[5])
        return self.nu[:3]-np.array([u_c,v_c,0.])
    def ground_velocity(self):return self.gnc.Rzyx(*self.eta[3:])@self.nu[:3]

def regression(duration=200.,dt=.02):
    if not np.isfinite(duration) or duration<=0 or not 0<dt<=.02:
        raise ValueError('regression requires finite duration>0 and 0<dt<=0.02s')
    plant=RemusPlant();states=[];tic=time.perf_counter()
    for i in range(round(duration/dt)+1):
        states.append([i*dt,*plant.eta,*plant.nu,*plant.actual])
        if i<round(duration/dt):plant.step(dt)
    states=np.asarray(states)
    result={'task':'existing_repository_regression','source':verify_sources(),'duration_s':duration,'dt_s':dt,
            'n_records':len(states),'final_depth_m':float(plant.eta[2]),'final_heading_deg':float(np.rad2deg(plant.eta[5])),
            'path_length_m':float(np.linalg.norm(np.diff(states[:,1:4],axis=0),axis=1).sum()),
            'all_finite':bool(np.isfinite(states).all()),'wall_clock_s':time.perf_counter()-tic,
            'raw_original_autopilot':True,'hydrodynamic_parameters_modified':False}
    if abs(duration-200)<1e-9:
        expected={'final_depth_m':29.74975,'final_heading_deg':49.99890,'path_length_m':453.55089}
        result['readme_absolute_errors']={k:abs(result[k]-v) for k,v in expected.items()}
        result['matches_readme_rounding']=all(e<=1e-5 for e in result['readme_absolute_errors'].values())
    return result,states

def parameter_audit():
    cls,_=load_model();m=cls()
    return {'source':verify_sources(),'unchanged':True,
       'p25_model':'Eq.2 simplified 3D kinematics; not REMUS full dynamics',
       'p25_not_provided':['AUV hull length','AUV mass','AUV hydrodynamic coefficient matrix',
                          'AUV actuator time constants','AUV DWA exact command limits','physical observation delta_t'],
       'remus':{'length_m':m.L,'diameter_m':m.diam,'mass_computed_from_code_kg':float(m.W/9.81),
                'mass_in_upstream_docstring_kg':31.9,'mass_note':'do not change computed mass solely to match the docstring',
                'rho_kg_m3':m.rho,'max_rpm':m.nMax,'rudder_limit_deg':float(np.rad2deg(m.deltaMax_r)),
                'stern_plane_limit_deg':float(np.rad2deg(m.deltaMax_s)),'fin_time_constant_s':m.T_delta,
                'propeller_time_constant_s':m.T_n,'heading_reference_rate_limit_deg_s':float(np.rad2deg(m.r_max)),
                'depth_reference_limit_m':[0,100],'mass_matrix_eigenvalues':np.linalg.eigvalsh(m.M).tolist(),
                'positive_definite_mass_matrix':bool(np.linalg.eigvalsh(m.M).min()>0)},
       'do_not_copy':{'P25_Table2':'dynamic obstacle limits, not AUV parameters',
                      'P25_map_depth_200m':'environment extent, not permission to remove REMUS 100m reference guard'}}

def route_demo(max_time=500.,dt=.02):
    waypoints=np.array([[270.,80.,35.],[460.,210.,45.],[690.,150.,35.]])
    plant=RemusPlant(depth=35.,heading_deg=0.,rpm=1200.,current_speed=.2,current_direction_deg=90.,
                      eta0=np.array([60.,80.,30.,0.,0.,0.]),initialize_references=True)
    states=[];refs=[];index=0;visited=[];tic=time.perf_counter()
    for i in range(round(max_time/dt)+1):
        target=waypoints[index];err=target-plant.eta[:3]
        if np.linalg.norm(err[:2])<12 and abs(err[2])<4:
            visited.append({'index':index,'time_s':i*dt,'error_m':float(np.linalg.norm(err))})
            index+=1
            if index==len(waypoints):break
            target=waypoints[index];err=target-plant.eta[:3]
        desired=np.arctan2(err[1],err[0]);current_yaw=plant.eta[5]
        desired=current_yaw+float((desired-current_yaw+np.pi)%(2*np.pi)-np.pi)
        plant.set_reference(target[2],np.rad2deg(desired),1200.)
        states.append([i*dt,*plant.eta,*plant.nu,*plant.actual]);refs.append([i*dt,index,*target])
        if i<round(max_time/dt):plant.step(dt,clip_autopilot_commands=True)
    report={'task':'6dof_3d_waypoint_tracking_not_obstacle_avoidance','status':'success' if index==len(waypoints) else 'timeout',
            'waypoints':waypoints.tolist(),'visited':visited,'duration_s':plant.time_s,'final_position':plant.eta[:3].tolist(),
            'all_finite':bool(np.isfinite(states).all()),'depth_min_m':float(np.min(np.asarray(states)[:,3])),
            'depth_max_m':float(np.max(np.asarray(states)[:,3])),'wall_clock_s':time.perf_counter()-tic,
            'hydrodynamic_parameters_modified':False,'adapter_changes':['reference initialization at initial pose','nearest-equivalent yaw reference','clipped autopilot actuator requests'],
            'obstacle_avoidance_claim':False}
    return report,np.asarray(states),np.asarray(refs)


def characterize(duration=100.0, dt=.02):
    """Operating-point tests on the UNMODIFIED model; not hardware calibration.

    Straight runs inform a future speed-to-RPM adapter. The other runs expose
    transient tracking errors. None of these estimates is a physical maximum.
    """
    if duration < 40 or not 0 < dt <= .02:
        raise ValueError('characterization requires duration>=40s and dt<=.02s')
    cases=[(f'rpm_{rpm}',float(rpm),0.,30.,0.) for rpm in (600,900,1200,1525)]
    cases += [('heading_plus30',1200.,30.,30.,0.),
              ('heading_minus30',1200.,-30.,30.,0.),
              ('depth_plus5',1200.,0.,35.,0.),
              ('cross_current',1200.,0.,30.,.3)]
    records=[];series={};tic=time.perf_counter()
    for name,rpm,heading,depth,current in cases:
        plant=RemusPlant(depth=depth,heading_deg=heading,rpm=rpm,
                         current_speed=current,current_direction_deg=90.,
                         eta0=np.array([0.,0.,30.,0.,0.,0.]),initialize_references=True)
        states=[];rel=[]
        for i in range(round(duration/dt)+1):
            states.append([i*dt,*plant.eta,*plant.nu,*plant.actual])
            rel.append(plant.relative_velocity())
            if i < round(duration/dt):
                plant.step(dt,clip_autopilot_commands=True)
        data=np.asarray(states);relative=np.asarray(rel);tail=data[:,0]>=duration-20
        series[name]=data
        records.append({'case':name,'rpm':rpm,'current_speed_mps':current,
            'target_heading_deg':heading,'target_depth_m':depth,
            'mean_final20s_relative_surge_mps':float(relative[tail,0].mean()),
            'mean_final20s_absolute_body_surge_mps':float(data[tail,7].mean()),
            'final_depth_error_m':float(plant.eta[2]-depth),
            'final_heading_error_deg':float(np.rad2deg(plant.gnc.ssa(plant.eta[5]-np.deg2rad(heading)))),
            'observed_max_abs_pitch_rate_deg_s':float(np.rad2deg(abs(data[:,11])).max()),
            'observed_max_abs_yaw_rate_deg_s':float(np.rad2deg(abs(data[:,12])).max()),
            'all_finite':bool(np.isfinite(data).all())})
    return {'claim':'model operating-point characterization, NOT physical parameter identification or maximum maneuverability',
            'duration_per_case_s':duration,'dt_s':dt,'source':verify_sources(),
            'hydrodynamic_parameters_modified':False,'wall_clock_s':time.perf_counter()-tic,
            'cases':records},series
