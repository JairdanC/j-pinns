from typing import Callable
from jax import vmap, grad
import jax.numpy as jnp

from .methods import build_n_soliton

def gen_loss_fn(soliton_params: dict, domain_numeric, expect_cons: list[int] = None) -> Callable:
    """
    Generates a JIT-compatible PINN loss function for the KdV eq.
    - Will have to change this to add a batched version when domain size increases
    - Need to implement device state management (GPU)
    - Replace separate model and grad(model) calls with value_and_grad to compute u and u_x simultaneously and halve compute time.
    - Ensure jnp.trapezoid axes are explicitly mapped when integrating conserved quantities across multiple time slices to prevent cross-slice bugs.
    - Extract hardcoded coefficients (e.g., KdV's 6) into soliton_params to support general non-linear dispersive wave sweeps.
    """
    #Calculate predicted info such that they are included in the closure of the func

    if expect_cons is None:
        expect_cons = [0]
    
    #Precompute exact arrays, these are added to the closure as constants
    u = build_n_soliton(soliton_params['k_vec'], d_vec=soliton_params['d_vec'])[0]

    u_ic = vmap(u)(domain_numeric.x_ic, domain_numeric.t_ic)
    u_bc = jnp.zeros_like(domain_numeric.x_bc) #BC of 0

    #Initialize to 0.0 to avoid unbound variable errors in the closure
    momentum_quantity, energy_quantity, hamilt_quantity = 0.0, 0.0, 0.0

    if 1 in expect_cons:
        momentum_quantity = jnp.trapezoid(u_ic, domain_numeric.x_ic, axis=0)
    if 2 in expect_cons:
        energy_quantity = jnp.trapezoid(u_ic**2, domain_numeric.x_ic, axis=0)
    if 3 in expect_cons:
        u_x_ic = vmap(grad(u, argnums=0))(domain_numeric.x_ic, domain_numeric.t_ic)
        hamilt_quantity = jnp.trapezoid(u_ic**3 - 0.5*(u_x_ic**2), domain_numeric.x_ic, axis=0)

    #Closure func
    def loss_fn(model, weights, domain):
        #IC
        u_pred_ic =  vmap(model)(domain.x_ic, domain.t_ic)
        loss_ic = jnp.mean((u_pred_ic - u_ic)**2) #L2 / MSE Loss

        #BC
        u_pred_bc = vmap(model)(domain.x_bc, domain.t_bc)
        loss_bc = jnp.mean((u_pred_bc - u_bc)**2) #L2 / MSE Loss

        #PDE
        
        u_pred_pde = vmap(model)(domain.x_coll, domain.t_coll)
        u_t_pde = vmap(grad(model, argnums=1))(domain.x_coll, domain.t_coll) 
        u_x_pde = vmap(grad(model, argnums=0))(domain.x_coll, domain.t_coll)
        u_xxx_pde = vmap(grad(grad(grad(model, argnums=0), argnums=0), argnums=0))(domain.x_coll, domain.t_coll)

        resid = u_t_pde + 6*u_pred_pde*u_x_pde + u_xxx_pde
        loss_pde = jnp.mean((resid)**2) #L2 / MSE Loss
        
        #CONS - using the momentum of the system
        if 1 in expect_cons:
            u_pred_momentum = vmap(model)(domain.x_ic, domain.t_momentum)
            momentum_pred = jnp.trapezoid(u_pred_momentum, domain.x_ic, axis=0)
            loss_momentum = jnp.mean((momentum_pred - momentum_quantity)**2)
        else: 
            loss_momentum = jnp.array(0.0)

        #CONS - using the energy of the system
        if 2 in expect_cons:
            u_pred_energy = vmap(model)(domain.x_ic, domain.t_energy)
            energy_pred = jnp.trapezoid(u_pred_energy**2, domain.x_ic, axis=0)
            loss_energy = jnp.mean((energy_pred - energy_quantity)**2)
        else:
            loss_energy = jnp.array(0.0)

        #CONS - using the hamiltonian of the system
        if 3 in expect_cons:
            u_pred_hamilt = vmap(model)(domain.x_ic, domain.t_hamilt)
            u_x_hamilt = vmap(grad(model, argnums=0))(domain.x_ic, domain.t_hamilt)
            hamilt_pred = jnp.trapezoid(u_pred_hamilt**3 - 0.5*(u_x_hamilt**2), domain.x_ic, axis=0)
            loss_hamilt = jnp.mean((hamilt_pred - hamilt_quantity)**2)
        else:
            loss_hamilt = jnp.array(0.0)

        loss_comps = jnp.stack([loss_pde, loss_ic, loss_bc, loss_momentum, loss_energy, loss_hamilt])
        total_loss = jnp.dot(weights, loss_comps)
        return total_loss

    return loss_fn


        

