import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax.ops import index, index_add, index_update
import numpy as np
from functools import partial


@jit
def spherical_s_metric(x,M=1/2):
    """ Computes the schwarzchild metric in cartesian like coordinates"""
    t,r,theta,phi = x.T
    rs = 2*M
    a = (1-rs/r)
    gdiag = jnp.stack([-a,1/a,r**2,r**2*jnp.sin(theta)**2],-1)
    g = jnp.diag(gdiag)
    return g

@jit
def cartesian_s_metric(x,M=1/2):
    """ Computes the schwarzchild metric in cartesian like coordinates"""
    r = jnp.linalg.norm(x[1:])
    rhat = x[1:]/r
    rs = 2*M
    a = (1-rs/r)
    g = jnp.zeros((4,4),dtype=r.dtype)
    g = index_update(g, index[0, 0], -a)
    g = index_update(g, index[1:, 1:], jnp.eye(3)+(1/a-1)*jnp.outer(rhat,rhat))
#     g[0,0] += -a
#     g[1:,1:] += jnp.eye(3)+(1/a-1)*jnp.outer(rhat,rhat)
    return g

D = jacfwd
@jit
def christoffel_symbols(dg,ginv):
    Γ_lower = (dg.transpose((2,0,1)) + dg.transpose((1,2,0))-dg.transpose((0,1,2)))/2
    return jnp.einsum('abd,cd->abc',Γ_lower,ginv)#jnp.dot(Γ_lower.transpose((1,2,0)),ginv)

@partial(jit, static_argnums=(0,))
def ricci_curvature(gfunc,x):
    Γfunc = lambda x: christoffel_symbols(D(gfunc)(x),jnp.linalg.inv(gfunc(x)))
    DΓ = D(Γfunc)(x)
    dΓ_terms = jnp.einsum('abcc',DΓ)-jnp.einsum('cbca',DΓ)
    Γ = Γfunc(x)
    ΓΓ_terms = jnp.einsum('abd,cdc',Γ,Γ)-jnp.einsum('acd,bdc',Γ,Γ)
    return dΓ_terms+ΓΓ_terms