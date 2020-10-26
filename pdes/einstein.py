import torch
import torch.nn as nn
import numpy as np
from oil.utils.mytqdm import tqdm
from oil.logging.lazyLogger import LazyLogger
from torch.autograd import grad
from oil.utils.utils import export
#from pdes import NeuralPDE
from .utils import batch_jacobian
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
    # (Γ_abc) = (a,b,c + b,c,a - c,a,b)/2
    Γ_lower = (dg + dg.permute(1,2,0)-dg.permute(2,0,1))/2
    Γ = Γ_lower@ginv
    return Γ # Γ_ab^c with ab indices first and c last

def d_g(g,x):
    bs,d = x.shape
    g_func = lambda x: g(x).reshape(bs,d*d)
    dg = batch_jacobian(g_func,x,d*d).reshape(bs,d,d,d)
    return dg

def Γ_vec(g_L_func,x):
    # cholesky_g = g_L_func(x)
    # bs,d,_ = cholesky_g.shape
    # g = cholesky_g@cholesky_g.permute(0,2,1)
    # ginv = torch.cholesky_inverse(cholesky_g)
    # dg = batch_jacobian(g.reshape(bs,d*d)).reshape(bs,d,d,d)
    dg = d_g(g_L_func,x)
    cholesky_g = g_L_func(x)
    ginv = torch.cholesky_inverse(cholesky_g) #TODO: ammend for Lorentzian metric
    Γ = christoffel_symbols(dg,ginv)
    Γ_vec = ES('nab,nabc->nc',ginv,Γ)-ES('nac,nabb->nc',ginv,Γ)
    return Γ_vec
    

def scalar_curvature(g_L_func,x):
    """ given partial derivatives of g (bs,d,d,d) and ginv (bs,d,d)
        compute the Ricci scalar """
    bs,d = x.shape
    dΓ_vec = batch_jacobian(lambda x: Γ_vec(g_L_func,x),x,d) # (bs,d,d)
    I = torch.eye(d,dtype=x.dtype)
    divΓ_vec = (dΓ_vec*I).sum(-1).sum(-1)

    dg = d_g(g_L_func,x)
    cholesky_g = g_L_func(x)
    ginv = torch.cholesky_inverse(cholesky_g)
    Γ = christoffel_symbols(dg,ginv)
    
    ibp = -ES('nad,neb,ncde,nabc->n',ginv,ginv,dg,Γ)+ES('nad,neb,nbde,nacc->n',ginv,ginv,dg,Γ)
    Γcomm = ES('nab,nabd,ncdc->n',ginv,Γ,Γ)-ES('nab,nacd,nbdc->n',ginv,Γ,Γ)
    
    return divΓ_vec+ibp+Γcomm

def sqrt_det_g(g_L_func,x):
    #TODO: ammend for Lorentzian metric with - signature
    cholesky_g = g_L_func(x)
    return torch.prod(torch.diagonal(cholesky_g,dim1=1,dim2=2),dim=-1)
    
    
    

@export
class EinsteinPDE(NeuralPDE):
    def empty_space_lagrangian(self,X):
        return scalar_curvature(self.model,X)*sqrt_det_g(self.model,X)
    def outer_boundary_residual()
    def action(self,X,B):

    def sample_inner_boundary(self,N):
        raise NotImplementedError
    def solve(self,iterations):
        for i in tqdm(range(iterations),desc='fitting pde'):
            X_samples = self.sample_domain(self.bs)
            X_samples += torch.zeros_like(X_samples,requires_grad=True)
            dX_samples = self.sample_boundary(self.bs)
            dX_samples += torch.zeros_like(dX_samples,requires_grad=True)
            iX_samples = self.sample_inner_boundary(self.bs)
            iX_samples += torch.zeros_like(iX_samples,requires_grad=True)
            self.optimizer.zero_grad()
            loss = self.action(X_samples,dX_samples,iX_samples)
            loss.backward()
            self.optimizer.step()
            with self.logger as do_log:
                if do_log: self.log(loss,i)

from .networks import Swish

def schwarzchild_metric(x,M=1):
    """ Computes the schwarzchild metric in Gullstrand–Painlevé coordinates"""
    r = (x**2).sum(-1).sqrt()
    rhat = x/r[:,None]
    g00 = -(1-2*M/r)
    g0i = (2*M/r).sqrt()[:,None]*rhat
    gij = torch.eye(3,dtype=x.dtype)
    g0μ = torch.cat([g00[:,None],g0i],dim=-1)
    giμ = torch.cat([g0i,gij],dim=-1)
    g = torch.cat([g0μ,giμ],dim=-1)
    sqrt_det_g = torch.ones_like(x[:,0])
    ginv00 = -torch.ones_like(x[:,:1])
    ginvij = torch.eye(3,dtype=x.dtype) - g0i[:,:,None]*g0i[:,None,:]
    ginv0μ = torch.cat([ginv00,g0i],dim=-1)
    ginviμ = torch.cat([g0i,ginvij],dim=-1)
    ginv = torch.cat([ginv0μ,ginviμ],dim=-1)
    return g,ginv,sqrt_det_g

    

@export
class gMLP(nn.Module):
    def __init__(self,d=2,k=1024,L=5):
        super().__init__()
        channels = [d]+(L-1)*[k]#+[1]
        self.network = nn.Sequential(
            *[nn.Sequential(nn.Linear(kin,kout),Swish()) for kin,kout in zip(channels,channels[1:])],
            nn.Linear(k,d*d))
    def forward(self,x):
        flat_L = self.network(x)
        return 
def g(g_L_func,x):
    cholesky_g = g_L_func(x)
    bs,d,_ = cholesky_g.shape
    g = cholesky_g@cholesky_g.permute(0,2,1) #TODO: ammend for Lorentzian metric
    return g