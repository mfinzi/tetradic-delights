import numpy as np#
import copy
from pdes import batch_jacobian
import unittest
import torch

class Jacobian(unittest.TestCase):
    def test_batch(self):
        vf = lambda x: torch.stack([-x[...,-1],x[...,0]],dim=-1)
        inp = torch.randn(3,2,requires_grad=True)
        #print(batch_jacobian(vf,inp,2))
        #self.assertTrue(torch.autograd.gradcheck(vf,inp))
    # def test_composite_representation(self):
    #     vf = lambda x: torch.stack([x[...,0]],dim=-1)
    #     inp = torch.randn(3,2,requires_grad=True)
    #     print(batch_jacobian(vf,inp,1).shape)
    def test_batch2(self):
        vf = lambda x: torch.stack([-x[...,-1],x[...,0]],dim=-1)
        inp = torch.randn(3,2,2,requires_grad=True)
        print(batch_jacobian(vf,inp,2)[:,:,0,1])
        print(batch_jacobian(vf,inp,2).shape)



if __name__ == '__main__':
    unittest.main()