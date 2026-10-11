from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from .sensors import Track

PPLNM_ARCHITECTURE = 'lstm_2x256_layernorm_3_linear_heads'


def pplnm_contract(dt, epochs_completed, training_label):
    """Physical/model compatibility fields; no file identity claim is made."""
    return {
        'predictor_name': 'pplnm_reimplementation',
        'architecture': PPLNM_ARCHITECTURE,
        'input_dim': 9,
        'output_dim': 3,
        'history_length': 6,
        'prediction_dt_s': float(dt),
        'horizon_steps': 3,
        'epochs_completed': int(epochs_completed),
        'training_label': str(training_label),
    }

@dataclass(frozen=True)
class History:
    features: np.ndarray
    real_mask: np.ndarray
    latest_time_s: float

def make_history(times,positions,dt=1.0,length=6) -> History:
    """Causal resampling and P25 Eq.11 backward CONSTANT-VELOCITY completion.

    Although the paper diagram says backward CA, its printed Eq.11 contains
    only p_t-v_t*k*dt. One real sample has unknown velocity: use zero explicitly.
    The real/pseudo mask is metadata, not a tenth LSTM input channel.
    """
    t=np.asarray(times,dtype=float);p=np.asarray(positions,dtype=float)
    if t.ndim!=1 or len(t)==0 or p.shape!=(len(t),3):raise ValueError('empty/invalid history')
    if not np.isfinite(t).all() or not np.isfinite(p).all():raise ValueError('history must be finite')
    if not np.isfinite(dt) or dt<=0 or length<3 or np.any(np.diff(t)<=0):raise ValueError('invalid times/dt/history length')
    grid=t[-1]-dt*np.arange(length-1,-1,-1)
    v0=np.zeros(3) if len(t)==1 else (p[-1]-p[-2])/(t[-1]-t[-2])
    hist=np.column_stack([np.interp(grid,t,p[:,j]) for j in range(3)])
    left=grid<t[0]-1e-9;hist[left]=p[-1]+(grid[left]-t[-1])[:,None]*v0
    v=np.empty_like(hist);v[1:]=np.diff(hist,axis=0)/dt;v[0]=v[1]
    a=np.zeros_like(hist);a[1:]=np.diff(v,axis=0)/dt;a[0]=a[1]
    real=np.array([np.any(np.isclose(g,t,atol=1e-7,rtol=0)) for g in grid])
    return History(np.concatenate([hist,v,a],axis=1),real,float(t[-1]))

def constant_acceleration(features,offsets):
    f=np.asarray(features);h=np.asarray(offsets,dtype=float)
    if np.any(h<0):raise ValueError('negative forecast horizon')
    return f[-1,:3]+h[:,None]*f[-1,3:6]+.5*h[:,None]**2*f[-1,6:9]

class CAPredictor:
    name='constant_acceleration_debug_baseline'
    def __init__(self,dt=1.):self.dt=dt
    def predict(self,track:Track,absolute_times):
        hist=make_history(track.times,track.positions,self.dt)
        return constant_acceleration(hist.features,np.asarray(absolute_times)-hist.latest_time_s)

class SphericalCalibration:
    """P25-style speed-bin empirical quantiles, not a task safety guarantee."""
    def __init__(self,bins=(0.,.75,1.5,2.5,float('inf')),alpha=.05,min_count=30):
        if not 0<alpha<1:raise ValueError('alpha must be in (0,1)')
        self.bins=np.asarray(bins,dtype=float);self.alpha=alpha;self.min_count=min_count
        if self.bins.ndim!=1 or len(self.bins)<2 or not np.isfinite(self.bins[:-1]).all() or np.isnan(self.bins[-1]) or np.any(np.diff(self.bins)<=0):
            raise ValueError('calibration speed bins must be ordered and finite except for the last upper bound')
        if not isinstance(min_count,int) or min_count<1:raise ValueError('min_count must be positive')
        self.quantiles=None;self.global_q=None;self.counts=None;self.metadata={}
    def fit(self,predictions,targets,speeds):
        predictions=np.asarray(predictions);targets=np.asarray(targets)
        if predictions.shape!=targets.shape or predictions.ndim!=3 or predictions.shape[-1]!=3:
            raise ValueError('calibration expects matching [N,horizon,3] predictions and targets')
        errors=np.linalg.norm(predictions-targets,axis=-1)
        speeds=np.asarray(speeds)
        if errors.ndim!=2 or len(errors)<self.min_count:raise ValueError('insufficient calibration data')
        if speeds.shape!=(len(errors),) or not np.isfinite(speeds).all() or np.any(speeds<0) or not np.isfinite(errors).all():
            raise ValueError('calibration values must be finite and speeds nonnegative')
        self.global_q=np.quantile(errors,1-self.alpha,axis=0,method='higher')
        self.quantiles=[];self.counts=[]
        for lo,hi in zip(self.bins[:-1],self.bins[1:]):
            idx=(speeds>=lo)&(speeds<hi);self.counts.append(int(idx.sum()))
            self.quantiles.append(np.quantile(errors[idx],1-self.alpha,axis=0,method='higher') if idx.sum()>=self.min_count else self.global_q.copy())
        self.quantiles=np.asarray(self.quantiles);return self
    def radius(self,speed,offsets,dt=1.):
        if self.quantiles is None:raise RuntimeError('calibrator is not fitted')
        if not np.isfinite(speed) or speed<0 or not np.isfinite(dt) or dt<=0:raise ValueError('invalid calibration speed/dt')
        idx=np.clip(np.searchsorted(self.bins,speed,side='right')-1,0,len(self.quantiles)-1)
        offsets=np.asarray(offsets)
        if not np.isfinite(offsets).all() or np.any(offsets<0):raise ValueError('invalid calibration horizon')
        if np.any(offsets>len(self.global_q)*dt+1e-9):raise ValueError('do not silently extrapolate calibrated horizon')
        return np.interp(offsets,np.arange(len(self.global_q)+1)*dt,np.r_[0.,self.quantiles[idx]])
    def validate_for(self,predictor,dt,horizon_steps):
        """Validate useful parameters, without asserting identical model files."""
        meta=self.metadata
        if meta.get('predictor_name')!=predictor.name:
            raise ValueError('calibration/predictor identity mismatch; do not mix CA and learned calibration')
        if not np.isfinite(getattr(predictor,'dt',float('nan'))) or abs(predictor.dt-dt)>1e-9:
            raise ValueError('predictor physical time step mismatch')
        meta_dt=meta.get('prediction_dt_s')
        if not isinstance(meta_dt,(int,float)) or not np.isfinite(meta_dt) or abs(meta_dt-dt)>1e-9:
            raise ValueError('calibration physical time step mismatch')
        if self.global_q is None or self.global_q.shape!=(horizon_steps,):
            raise ValueError('calibration horizon dimension mismatch')
        if self.quantiles is None or self.quantiles.shape!=(len(self.bins)-1,horizon_steps):
            raise ValueError('calibration speed-bin/horizon dimension mismatch')
        if not np.isfinite(self.global_q).all() or not np.isfinite(self.quantiles).all() or np.any(self.global_q<0) or np.any(self.quantiles<0):
            raise ValueError('calibration radii must be finite and nonnegative')
        if hasattr(predictor,'checkpoint_meta'):
            for field,expected in predictor.checkpoint_meta.items():
                actual=meta.get(field)
                if field=='prediction_dt_s':continue  # Tolerant physical comparison above.
                if actual!=expected:
                    raise ValueError(f'calibration model parameter mismatch: {field}')
    def save(self,path):
        Path(path).write_text(json.dumps({'bins':[float(x) if np.isfinite(x) else None for x in self.bins],
            'alpha':self.alpha,'min_count':self.min_count,'quantiles':self.quantiles.tolist(),
            'global_q':self.global_q.tolist(),'counts':self.counts,'claim':'empirical marginal only','metadata':self.metadata},indent=2))
    @classmethod
    def load(cls,path):
        d=json.loads(Path(path).read_text());obj=cls([float('inf') if x is None else x for x in d['bins']],d['alpha'],d['min_count'])
        obj.quantiles=np.array(d['quantiles']);obj.global_q=np.array(d['global_q']);obj.counts=d['counts'];obj.metadata=d.get('metadata',{});return obj

class PPLNMPredictor:
    name='pplnm_reimplementation'
    def __init__(self,checkpoint,device='cpu',threads=2):
        import torch
        from .learning import PPLNM
        if threads < 1: raise ValueError('threads must be positive')
        if device == 'cpu': torch.set_num_threads(threads)
        self.torch=torch;self.device=device
        checkpoint=Path(checkpoint)
        if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
        ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
        self.dt=float(ck['dt'])
        if not np.isfinite(self.dt) or self.dt<=0:raise ValueError('checkpoint dt must be finite and positive')
        if not isinstance(ck.get('epochs_completed'),int) or ck['epochs_completed']<1:
            raise ValueError('checkpoint requires positive epochs_completed')
        if not isinstance(ck.get('training_label'),str) or not ck['training_label'].strip():
            raise ValueError('checkpoint requires a training_label')
        self.checkpoint_meta=pplnm_contract(self.dt,ck['epochs_completed'],ck['training_label'])
        # Existing stage-1 weights predate explicit architecture metadata. Their
        # tensor shapes are checked by strict load_state_dict below; the known
        # fixed model supplies dimensions absent from those original files.
        if 'model_config' in ck and ck['model_config']!=self.checkpoint_meta:
            raise ValueError('checkpoint model configuration mismatch')
        self.xmean=np.asarray(ck['xmean'],dtype=float);self.xstd=np.asarray(ck['xstd'],dtype=float);self.yscale=np.asarray(ck['yscale'],dtype=float)
        if self.xmean.shape!=(9,) or self.xstd.shape!=(9,) or self.yscale.shape!=(3,):
            raise ValueError('checkpoint normalization dimension mismatch')
        if any(not np.isfinite(a).all() for a in (self.xmean,self.xstd,self.yscale)) or np.any(self.xstd<=0) or np.any(self.yscale<=0):
            raise ValueError('checkpoint normalization must be finite with positive scales')
        if any(not torch.isfinite(value).all() for value in ck['model'].values()):
            raise ValueError('checkpoint contains nonfinite model parameters')
        self.model=PPLNM().to(device);self.model.load_state_dict(ck['model'],strict=True);self.model.eval()
        self.max_horizon_s=self.checkpoint_meta['horizon_steps']*self.dt
    def predict(self,track,absolute_times):
        h=make_history(track.times,track.positions,self.dt);offsets=np.asarray(absolute_times)-h.latest_time_s
        if offsets.ndim!=1 or not np.isfinite(offsets).all() or np.any(offsets<0) or np.any(offsets>self.max_horizon_s+1e-8):
            raise ValueError('PPLNM supports only its trained 3-step horizon; stale track needs explicit fallback')
        x=(h.features-self.xmean)/self.xstd
        with self.torch.inference_mode():
            residual=self.model(self.torch.tensor(x[None],dtype=self.torch.float32,device=self.device)).cpu().numpy()[0]*self.yscale
        if residual.shape!=(3,3) or not np.isfinite(residual).all():
            raise FloatingPointError('PPLNM prediction must be finite with shape (3,3)')
        pred=constant_acceleration(h.features,np.arange(1,4)*self.dt)+residual
        grid=np.vstack([h.features[-1,:3],pred])
        return np.column_stack([np.interp(offsets,np.arange(4)*self.dt,grid[:,j]) for j in range(3)])
