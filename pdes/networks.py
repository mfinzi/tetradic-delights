import torch
import torch.nn as nn
from oil.utils.utils import Expression,export

import numpy as np

class Sine(nn.Module):
    def forward(self,x):
        return (30*x).sin()

def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)


# def Siren(k=1024,L=5):
#     return nn.Sequential(
#         *[nn.Sequential]
#     )

@export
class Siren(nn.Module):
    def __init__(self,d=2,k=1024,L=5):
        super().__init__()
        channels = [d]+(L-1)*[k]#+[1]
        self.network = nn.Sequential(
            *[nn.Sequential(nn.Linear(kin,kout),Sine()) for kin,kout in zip(channels,channels[1:])],
            nn.Linear(k,1),
            Expression(lambda x: x.squeeze(-1)))
        self.network.apply(sine_init)
        self.network[0].apply(first_layer_sine_init)

    def forward(self,x):
        return self.network(x)
@export
class Swish(nn.Module):
    def forward(self,x):
        return x.sigmoid()*x     

@export
class NN(nn.Module):
    def __init__(self,d=2,k=1024,L=5):
        super().__init__()
        channels = [d]+(L-1)*[k]#+[1]
        self.network = nn.Sequential(
            *[nn.Sequential(nn.Linear(kin,kout),Swish()) for kin,kout in zip(channels,channels[1:])],
            nn.Linear(k,1),
            Expression(lambda x: x.squeeze(-1)))
    def forward(self,x):
        return self.network(x)