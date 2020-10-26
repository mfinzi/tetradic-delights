import torch
from oil.utils.utils import export


@export
def batch_jacobian(vf,x,n_outputs,retain_graph=True):
    n_outputs = int(n_outputs)
    repear_arg = (n_outputs,) + (1,) * len(x.size())
    xr = x.repeat(*repear_arg)
    xr.requires_grad_(True)
    y = vf(xr)#.view(n_outputs, -1)
    I = torch.eye(n_outputs, device=xr.device)
    for i in range(1,len(y.shape)-1):
        I = I.unsqueeze(i)
    I = I.repeat(1,*y.shape[1:-1],1)
    #print(y.shape,I.shape,xr.shape)
    J = torch.autograd.grad(y, xr,
                      grad_outputs=I,
                      retain_graph=retain_graph,
                      create_graph=True,  # for higher order derivatives
                      )[0]

    return J.permute(*list(range(1,len(y.shape)-1)),0,-1)


@export
def jacobian(fxn, x, n_outputs, retain_graph):
    """
    the basic idea is to create N copies of the input
    and then ask for each of the N dimensions of the
    output... this allows us to compute J with pytorch's
    jacobian-vector engine
    """

    # expand the input, one copy per output dimension
    n_outputs = int(n_outputs)
    repear_arg = (n_outputs,) + (1,) * len(x.size())
    xr = x.repeat(*repear_arg)
    xr.requires_grad_(True)

    # both y and I are shape (n_outputs, n_outputs)
    #  checking y shape lets us report something meaningful
    y = fxn(xr).view(n_outputs, -1)

    if y.size(1) != n_outputs: 
        raise ValueError('Function `fxn` does not give output '
                         'compatible with `n_outputs`=%d, size '
                         'of fxn(x) : %s' 
                         '' % (n_outputs, y.size(1)))
    I = torch.eye(n_outputs, device=xr.device)

    J = torch.autograd.grad(y, xr,
                      grad_outputs=I,
                      retain_graph=retain_graph,
                      create_graph=True,  # for higher order derivatives
                      )

    return J[0]