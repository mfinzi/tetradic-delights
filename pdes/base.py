import torch
import torch.nn as nn
import numpy as np
from oil.utils.mytqdm import tqdm
from oil.logging.lazyLogger import LazyLogger
from torch.autograd import grad
from oil.utils.utils import export

# def grad(fn):
#     return lambda x: torch.autograd.grad(fn(x),x,create_graph=True)

# Equivalent of the trainer abstraction
@export
class NeuralPDE(object):

    def __init__(self,model,opt = torch.optim.Adam,lr_sched=lambda e:1,bs=1000,
                    boundary_conditions=None,coordinate_transform=None,log_kwargs={'timeFrac':1/10,'minPeriod':0.03}):
        super().__init__()
        self.model = model
        self.optimizer = opt(model.parameters())
        self.bs = bs
        self.logger = LazyLogger(**log_kwargs)
        try: self.lr_scheduler = lr_sched(optimizer=self.optimizer)
        except TypeError: self.lr_schedulers = torch.optim.lr_scheduler.LambdaLR(self.optimizer,lr_sched)

    def action(self,domain_points,boundary_points):
        raise NotImplementedError
    def sample_domain(self,N):
        raise NotImplementedError
    def sample_boundary(self,N):
        raise NotImplementedError
    def solve(self,iterations):
        for i in tqdm(range(iterations),desc='fitting pde'):
            X_samples = self.sample_domain(self.bs)
            X_samples += torch.zeros_like(X_samples,requires_grad=True)
            dX_samples = self.sample_boundary(self.bs)
            dX_samples += torch.zeros_like(dX_samples,requires_grad=True)
            self.optimizer.zero_grad()
            loss = self.action(X_samples,dX_samples)
            loss.backward()
            self.optimizer.step()
            with self.logger as do_log:
                if do_log: self.log(loss,i)
    def log(self,loss,i):
        self.logger.add_scalars('metrics',{'loss':loss.cpu().item()},i)
        self.logger.report()
    def visualize(self):
        raise NotImplementedError

@export
class InnerNeuralPDE(NeuralPDE):
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

# @export
# class Poisson(NeuralPDE):
#     def sample_domain(self,N):
#         return torch.rand(N,2)
#     def sample_boundary(self,N):
#         side = torch.randint(low=0,high=4,size=(N,))
#         walls = torch.stack([torch.rand(N)
#     def action(self,X,B):
#         phi = self.model(X)
#         dphi = grad(phi,X,create_graph=True)
#         phi_B = self.model(B)
#         lagrangian = (dphi**2).sum(-1)/2 + phi*self.rho(X)
#         boundary = (phi_B**2)
#         return lagrangian.mean()+boundary.mean()

#%% print("hello")
# %%
#print("hello")
# %%