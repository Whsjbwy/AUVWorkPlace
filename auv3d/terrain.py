from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .config import TerrainConfig

@dataclass
class HeightField:
    xs: np.ndarray
    ys: np.ndarray
    depths: np.ndarray  # [y,x], NED positive down
    slope_bound: float

    @classmethod
    def flat(cls,width=1000.,height=1000.,depth=200.):
        return cls(np.array([0.,width]),np.array([0.,height]),np.full((2,2),depth),0.)

    @classmethod
    def perlin(cls,cfg: TerrainConfig,seed: int):
        """Seeded gradient Perlin, P25 Eq.4-5 normalization.

        Gradient generator, octave persistence=0.5 and grid resolution are
        disclosed reimplementation choices; not bit-identical to authors' map.
        """
        rng=np.random.default_rng(seed)
        xs=np.linspace(0,cfg.width,int(np.ceil(cfg.width/cfg.grid_spacing))+1)
        ys=np.linspace(0,cfg.height,int(np.ceil(cfg.height/cfg.grid_spacing))+1)
        X,Y=np.meshgrid(xs/cfg.scale,ys/cfg.scale)
        field=np.zeros_like(X); total=0.
        def fade(t):return t**3*(t*(t*6-15)+10)
        for k in range(cfg.octaves):
            x,y=X*2**k,Y*2**k
            nx,ny=int(np.floor(x.max()))+2,int(np.floor(y.max()))+2
            angles=rng.uniform(0,2*np.pi,(ny,nx));grad=np.stack([np.cos(angles),np.sin(angles)],-1)
            xi,yi=np.floor(x).astype(int),np.floor(y).astype(int)
            xf,yf=x-xi,y-yi
            def dot(dx,dy):
                g=grad[yi+dy,xi+dx]
                return g[...,0]*(xf-dx)+g[...,1]*(yf-dy)
            a,b=dot(0,0),dot(1,0);c,d=dot(0,1),dot(1,1)
            v=(a+(b-a)*fade(xf))*(1-fade(yf))+(c+(d-c)*fade(xf))*fade(yf)
            weight=0.5**k;field+=weight*v;total+=weight
        h=((field/total+1)/2)**cfg.exponent*cfg.amplitude
        span=float(np.ptp(h))
        if span < 1e-12: raise ValueError('degenerate terrain')
        depth=cfg.max_depth-(h-h.min())/span*cfg.amplitude
        gx=np.diff(depth,axis=1)/np.diff(xs)[None,:]
        gy=np.diff(depth,axis=0)/np.diff(ys)[:,None]
        L=float(np.hypot(np.abs(gx).max(),np.abs(gy).max()))
        return cls(xs,ys,depth,L)

    def depth_at(self, xy):
        xy=np.asarray(xy,dtype=float);x,y=xy[...,0],xy[...,1]
        ix=np.clip(np.searchsorted(self.xs,x,side='right')-1,0,len(self.xs)-2)
        iy=np.clip(np.searchsorted(self.ys,y,side='right')-1,0,len(self.ys)-2)
        tx=np.clip((x-self.xs[ix])/(self.xs[ix+1]-self.xs[ix]),0,1)
        ty=np.clip((y-self.ys[iy])/(self.ys[iy+1]-self.ys[iy]),0,1)
        a,b=self.depths[iy,ix],self.depths[iy,ix+1]
        c,d=self.depths[iy+1,ix],self.depths[iy+1,ix+1]
        return (a*(1-tx)+b*tx)*(1-ty)+(c*(1-tx)+d*tx)*ty

    def clearance(self, xyz, radius=0.):
        """Conservative sphere-heightfield clearance using a global slope bound.

        This is a sufficient vertical clearance bound, not exact signed distance.
        Domain and surface are included; positive means within this model.
        """
        p=np.asarray(xyz); x,y,z=p[...,0],p[...,1],p[...,2]
        bottom=self.depth_at(p[...,:2])-z-radius*np.sqrt(1+self.slope_bound**2)
        return np.minimum.reduce([bottom,z-radius,x-self.xs[0]-radius,
            self.xs[-1]-x-radius,y-self.ys[0]-radius,self.ys[-1]-y-radius])

    def segment_clearance(self,a,b,radius=0.,spacing=.25):
        n=max(2,int(np.ceil(np.linalg.norm(np.asarray(b)-a)/spacing))+1)
        p=np.linspace(a,b,n)
        # Between samples, subtract a Lipschitz bound. This deliberately errs safe.
        gap=np.linalg.norm(np.asarray(b)-a)/(n-1)
        return float(self.clearance(p,radius).min()-0.5*gap*np.sqrt(1+self.slope_bound**2))

    def visible_line(self,a,b):
        n=max(2,int(np.ceil(np.linalg.norm(np.asarray(b)-a)/2))+1)
        line=np.linspace(a,b,n)
        return bool(np.all(self.depth_at(line[:,:2])>line[:,2]))
