from __future__ import annotations
import numpy as np

def wrap_pi(angle):
    return (np.asarray(angle) + np.pi) % (2*np.pi) - np.pi

def rotation_zyx(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cp,sp=np.cos(roll),np.sin(roll); ct,st=np.cos(pitch),np.sin(pitch); cy,sy=np.cos(yaw),np.sin(yaw)
    return np.array([[cy*ct,cy*st*sp-sy*cp,cy*st*cp+sy*sp],
                     [sy*ct,sy*st*sp+cy*cp,sy*st*cp-cy*sp],[-st,ct*sp,ct*cp]])

def segment_point_distance(a, b, points):
    a,b,points=map(np.asarray,(a,b,points)); delta=b-a
    u=np.clip(np.sum((points-a)*delta,axis=-1)/(np.sum(delta*delta,axis=-1)+1e-30),0,1)
    return np.linalg.norm(points-(a+u[...,None]*delta),axis=-1)

def swept_sphere_clearance(a0,a1,b0,b1,total_radius):
    """Exact for linearly moving enclosing spheres over the SAME time interval."""
    d0=np.asarray(a0)-np.asarray(b0);d1=np.asarray(a1)-np.asarray(b1)
    return segment_point_distance(d0,d1,np.zeros_like(d0))-np.asarray(total_radius)

def kinematic_derivative(state, command, current=(0.,0.,0.)):
    """P25 Eq.2: state=(N,E,D,u,theta,psi); command=(u,q,r).

    r is BODY yaw rate; Euler yaw rate is r/cos(theta), NOT r.
    Optional additive NED current is an explicit extension, disabled by default.
    """
    x=np.asarray(state,dtype=float); a=np.asarray(command,dtype=float)
    theta,psi=x[...,4],x[...,5];u,q,r=a[...,0],a[...,1],a[...,2]
    if np.any(np.abs(np.cos(theta)) < 0.05): raise ValueError('Euler pitch singularity')
    out=np.zeros_like(x)
    out[...,0]=u*np.cos(psi)*np.cos(theta)+current[0]
    out[...,1]=u*np.sin(psi)*np.cos(theta)+current[1]
    out[...,2]=-u*np.sin(theta)+current[2]
    out[...,4]=q;out[...,5]=r/np.cos(theta)
    return out

def kinematic_step(state,command,dt,current=(0.,0.,0.)):
    """RK4 integration of Eq.2; rates and speed fixed for this short interval."""
    if dt <= 0 or not np.isfinite(dt): raise ValueError('dt must be positive')
    x=np.asarray(state,dtype=float).copy();cmd=np.asarray(command,dtype=float)
    x[...,3]=cmd[...,0]
    f=lambda z:kinematic_derivative(z,cmd,current)
    k1=f(x);k2=f(x+dt*k1/2);k3=f(x+dt*k2/2);k4=f(x+dt*k3)
    y=x+dt*(k1+2*k2+2*k3+k4)/6
    y[...,5]=wrap_pi(y[...,5]);y[...,3]=cmd[...,0]
    if not np.isfinite(y).all(): raise FloatingPointError('non-finite kinematic state')
    return y
