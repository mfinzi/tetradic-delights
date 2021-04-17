import torch
import torch.nn as nn
import numpy as np
from oil.utils.mytqdm import tqdm
from oil.logging.lazyLogger import LazyLogger
from torch.autograd import grad
from oil.utils.utils import export
#from pdes import NeuralPDE
from .utils import batch_jacobian
from .base import NeuralPDE
import scipy as sp
import scipy.special
# # def grad(fn):
# #     return lambda x: torch.autograd.grad(fn(x),x,create_graph=True)

# # Equivalent of the trainer abstraction
# """ Computes the Ricci scalar given the cholesky decomposition of g=LLT.
#         cholesky_g is the lower trianglular matrix of size (bs,d,d) """
#     bs,d,_ = cholesky_g.shape
#     g = cholesky_g@cholesky_g.permute(0,2,1)
#     g_inv = torch.cholesky_inverse(cholesky_g)
#     dg = batch_jacobian(g.reshape(bs,-1))

ES = torch.einsum

def christoffel_symbols(dg,ginv):
    """ given partial derivatives of g (bs,d,d,d) and ginv (bs,d,d)
        compute the Christoffel symbols Γ (bs,d,d,d) """
    print(f"Γ bs {dg.shape[0]}")
    # (Γ_abc) = (a,b,c + b,c,a - c,a,b)/2
    Γ_lower = (dg + dg.permute(0,2,3,1)-dg.permute(0,3,1,2))/2
    Γ = Γ_lower@ginv[:,None,:,:]
    return Γ # Γ_ab^c with ab indices first and c last

def d_g(g,x):
    _,d = x.shape
    g_func = lambda x: g(x).reshape(-1,d*d)
    dg = batch_jacobian(g_func,x,d*d).reshape(-1,d,d,d)
    return dg

def Γ_vec(gfunc,x):
    # cholesky_g = g_L_func(x)
    # bs,d,_ = cholesky_g.shape
    # g = cholesky_g@cholesky_g.permute(0,2,1)
    # ginv = torch.cholesky_inverse(cholesky_g)
    # dg = batch_jacobian(g.reshape(bs,d*d)).reshape(bs,d,d,d)
    dg = d_g(gfunc,x)
    g = gfunc(x)
    ginv = torch.inverse(g)
    Γ = christoffel_symbols(dg,ginv)
    Γ_vec = ES('nab,nabc->nc',ginv,Γ)-ES('nac,nabb->nc',ginv,Γ)
    return Γ_vec #(bs,d)
    
@export
def scalar_curvature(gfunc,x):
    """ given partial derivatives of g (bs,d,d,d) and ginv (bs,d,d)
        compute the Ricci scalar """
    bs,d = x.shape
    dg = d_g(gfunc,x)
    print(dg.shape)
    g = gfunc(x)
    ginv = torch.inverse(g)
    Γ = christoffel_symbols(dg,ginv)
    #print(Γ)
    ibp = ES('nad,neb,ncde,nabc->n',ginv,ginv,dg,Γ)-ES('nad,neb,nbde,nacc->n',ginv,ginv,dg,Γ)
    Γcomm = ES('nab,nabd,ncdc->n',ginv,Γ,Γ)-ES('nab,nacd,nbdc->n',ginv,Γ,Γ)
    
    dΓ_vec = batch_jacobian(lambda x: Γ_vec(gfunc,x),x,d) # (bs,d,d)
    print(dΓ_vec.shape)
    I = torch.eye(d,device=x.device)
    divΓ_vec = (dΓ_vec*I).sum(-1).sum(-1)

    return divΓ_vec+ibp+Γcomm


@export
def ricci_curvature(gfunc,x):
    bs,d = x.shape
    Γfunc = lambda x: christoffel_symbols(d_g(gfunc,x),torch.inverse(gfunc(x)))
    dΓ = batch_jacobian(lambda x: Γfunc(x).reshape(-1,d**3),x,d**3).reshape(bs,d,d,d,d)
    dΓ_terms = ES('ncabc->nab',dΓ)-ES('nacbc->nab',dΓ)
    Γ = Γfunc(x)
    ΓΓ_terms = ES('nabd,ncdc->nab',Γ,Γ)-ES('nacd,nbdc->nab',Γ,Γ)
    return dΓ_terms+ΓΓ_terms

def sqrt_det_g(gfunc,x):
    return (-torch.det(gfunc(x))).sqrt()
    # cholesky_g = g_L_func(x)
    # return torch.prod(torch.diagonal(cholesky_g,dim1=1,dim2=2),dim=-1)
    
    

def loguniform(lower,upper,*shape):
    return torch.exp(np.log(lower)+np.log(upper/lower)*torch.rand(*shape))

class ImportanceDist(object):
    def density(self,x):
        raise NotImplementedError
    def sample(self,N):
        raise NotImplementedError

class RadialLogUniform(ImportanceDist):
    def __init__(self,lower,upper):
        super().__init__()
        self.lower = lower
        self.upper=upper
    def density(self,x):
        r = (x[:,1:]**2).sum(-1).sqrt()
        return 1/(np.log(self.upper/self.lower)*r**3)
    def sample(self,N):
        z = torch.randn(N,4)
        r = (z[:,1:]**2).sum(-1).sqrt()
        new_r =loguniform(self.lower,self.upper,N,1)
        z[:,1:]*=new_r/r[:,None]
        #print(z.shape)
        #out = z[(r>1)&(r<20)]
        #print(out.shape)
        return z.cuda()

@export
class EinsteinPDE(NeuralPDE):
    # def empty_space_lagrangian(self,X):
    #     return scalar_curvature(self.model,X)*sqrt_det_g(self.model,X)
    importance_dist = RadialLogUniform(1,20)
    def action(self,X):
        # R = scalar_curvature(self.model,X)
        # corrected_density = sqrt_det_g(self.model,X)/self.importance_dist.density(X)
        # return (R*corrected_density).mean()
        Rab = ricci_curvature(self.model,X)
        corrected_density = sqrt_det_g(self.model,X).detach()/self.importance_dist.density(X)
        return ((Rab**2).sum(-1).sum(-1)*corrected_density).mean()
    def sample_domain(self,N):
        return self.importance_dist.sample(N)
    def sample_inner_boundary(self,N):
        raise NotImplementedError
    def solve(self,iterations):
        for i in tqdm(range(iterations),desc='fitting pde'):
            X_samples = self.sample_domain(self.bs)
            X_samples += torch.zeros_like(X_samples,requires_grad=True)
            # dX_samples = self.sample_boundary(self.bs)
            # dX_samples += torch.zeros_like(dX_samples,requires_grad=True)
            # iX_samples = self.sample_inner_boundary(self.bs)
            # iX_samples += torch.zeros_like(iX_samples,requires_grad=True)
            self.optimizer.zero_grad()
            loss = self.action(X_samples)#,dX_samples,iX_samples)
            loss.backward()
            self.optimizer.step()
            with self.logger as do_log:
                if do_log: self.log(loss,i)

from .networks import Swish

# @export
# def schwarzchild_metric(x,M=1):
#     """ Computes the schwarzchild metric in cartesian like coordinates"""
#     bs,d = x.shape
#     r = (x[:,1:]**2).sum(-1).sqrt()
#     rhat = x[:,1:]/r[:,None]
#     rs = 2*M
#     f = rs/r
#     ημ𝜈 = torch.diag(torch.tensor([-1.,1.,1.,1.]))[None].repeat(bs,1,1).to(x.device)
#     one = torch.ones(bs,1,device=x.device)
#     kμ = torch.cat([one,rhat],dim=-1)
#     g = ημ𝜈+f[:,None,None]*kμ[:,None,:]*kμ[:,:,None]
#     return g#,ginv,sqrt_det_g

@export
def schwarzchild_metric(x,M=1):
    """ Computes the schwarzchild metric in cartesian like coordinates"""
    bs,d = x.shape
    r = (x[:,1:]**2).sum(-1).sqrt()
    rhat = x[:,1:]/r[:,None]
    rs = 2*M
    a = (1-rs/r)[:,None]
    g00 = -a
    g0i = torch.zeros_like(x[:,1:])
    # I +(1/a -1)rrT
    gij = torch.eye(3,dtype=x.dtype).to(x.device)+rhat[:,:,None]*rhat[:,None,:]*(1/a[...,None]-1)
    g0μ = torch.cat([g00,g0i],dim=1)
    giμ = torch.cat([g0i[:,None,:],gij],dim=1)
    g = torch.cat([g0μ[:,:,None],giμ],dim=2)
    # sqrt_det_g = torch.ones_like(x[:,0]) #1
    # ginv00 = -torch.ones_like(x[:,:1])/a
    # # I + (a-1)rrT
    # ginvij = torch.eye(3,dtype=x.dtype) +rhat[:,:,None]*rhat[:,None,:]*(a[...,None]-1)
    # ginv0μ = torch.cat([ginv00,g0i],dim=1)
    # ginviμ = torch.cat([g0i[:,None,:],ginvij],dim=1)
    # ginv = torch.cat([ginv0μ[:,:,None],ginviμ],dim=2)
    return g#,ginv,sqrt_det_g


@export
def spherical_schwarzchild_metric(x,M=1):
    """ Computes the schwarzchild metric in cartesian like coordinates"""
    bs,d = x.shape
    t,r,theta,phi = x.T
    rs = 2*M
    a = (1-rs/r)
    gdiag = torch.stack([-a,1/a,r**2,r**2*theta.sin()**2],dim=-1)
    g = torch.diag_embed(gdiag)
    print(g.shape)
    return g

# @export
# def radial_schwarzchild_metric(x,M=1):
#     """ Computes the schwarzchild metric in cartesian like coordinates"""
#     bs,d = x.shape
#     r = (x[:,1:]**2).sum(-1).sqrt()[:,None]
#     rhat = x[:,1:]/r
#     rs = 2*M
#     a = (1-rs/r)
#     g00 = -a
#     gdiag = torch.cat([-a,1/a,r**2,r*sin**2])
#     g0i = torch.zeros_like(x[:,1:])
#     # I +(1/a -1)rrT
#     gij = torch.eye(3,dtype=x.dtype).to(x.device)+rhat[:,:,None]*rhat[:,None,:]*(1/a[...,None]-1)
#     g0μ = torch.cat([g00,g0i],dim=1)
#     giμ = torch.cat([g0i[:,None,:],gij],dim=1)
#     g = torch.cat([g0μ[:,:,None],giμ],dim=2)
#     # sqrt_det_g = torch.ones_like(x[:,0]) #1
#     # ginv00 = -torch.ones_like(x[:,:1])/a
#     # # I + (a-1)rrT
#     # ginvij = torch.eye(3,dtype=x.dtype) +rhat[:,:,None]*rhat[:,None,:]*(a[...,None]-1)
#     # ginv0μ = torch.cat([ginv00,g0i],dim=1)
#     # ginviμ = torch.cat([g0i[:,None,:],ginvij],dim=1)
#     # ginv = torch.cat([ginv0μ[:,:,None],ginviμ],dim=2)
#     return g#,ginv,sqrt_det_g

# def schwarzchild_metric_GP(x,M=1):
#     """ Computes the schwarzchild metric in Gullstrand–Painlevé coordinates"""
#     r = (x**2).sum(-1).sqrt()
#     rhat = x/r[:,None]
#     g00 = -(1-2*M/r)
#     g0i = (2*M/r).sqrt()[:,None]*rhat
#     gij = torch.eye(3,dtype=x.dtype)
#     g0μ = torch.cat([g00[:,None],g0i],dim=-1)
#     giμ = torch.cat([g0i,gij],dim=-1)
#     g = torch.cat([g0μ,giμ],dim=-1)
#     sqrt_det_g = torch.ones_like(x[:,0])
#     ginv00 = -torch.ones_like(x[:,:1])
#     ginvij = torch.eye(3,dtype=x.dtype) - g0i[:,:,None]*g0i[:,None,:]
#     ginv0μ = torch.cat([ginv00,g0i],dim=-1)
#     ginviμ = torch.cat([g0i,ginvij],dim=-1)
#     ginv = torch.cat([ginv0μ,ginviμ],dim=-1)
#     return g,ginv,sqrt_det_g

def cubic_interp(x0,xf,tt):
    t = torch.clamp(tt,min=0,max=1)
    return x0*(1-t)**3 + 3*(1-t)**2*t*x0+3*(1-t)*t**2*xf+xf*t**3

def bezier_interp(x0,xf,tt,n=5):
    t = torch.clamp(tt,min=0,max=1)
    ck = [sp.special.binom(n,k)*t**k*(1-t)**(n-k) for k in range(n+1)]
    return sum([ck[k]*x0 if k<n/2 else ck[k]*xf for k in range(n+1)])

@export
class gMLP(nn.Module):
    def __init__(self,d=4,k=256,L=3):
        super().__init__()
        channels = [d]+(L-1)*[k]#+[1]
        self.network = nn.Sequential(
            *[nn.Sequential(nn.Linear(kin,kout),Swish()) for kin,kout in zip(channels,channels[1:])],
            nn.Linear(k,d*d))
    def forward(self,x):
        bs,d = x.shape
        E = torch.eye(d,device=x.device)+.1*self.network(x).reshape(bs,d,d)
        minkowski = torch.diag(torch.tensor([-1.,1.,1.,1.]).to(x.device))[None]
        g = E@minkowski@E.permute(0,2,1)
        rs = 1
        g_s = schwarzchild_metric(x,M=rs/2)
        r = (x[:,1:]**2).sum(-1).sqrt()
        # # Linear transition from gs to g from 1rs - 1.5rs
        # blend_near = torch.clamp((r-rs)/(2*rs-rs),min=0,max=1)
        # # Linear transition from g to gs from 15rs to 20rs
        # blend_far = 1-torch.clamp((r-15*rs)/(20*rs-15*rs),min=0,max=1)
        # blend = (blend_near*blend_far)[:,None,None]
        # return (blend*g+(1-blend)*g_s)
        g_with_inner = bezier_interp(g_s,g,(r[:,None,None]/rs-1.5)/(4.5-1.5))
        g_with_both = bezier_interp(g_with_inner,g_s,(r[:,None,None]/rs-15)/(20-15))
        return g_with_both