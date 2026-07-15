from typing import Callable
import jax
from jax import vmap, grad, value_and_grad
from jax.tree_util import Partial
import jax.numpy as jnp
from jaxtyping import Float
import equinox as eqx
import optax.losses as optl


from .kdv_methods import build_n_soliton
from ..types import KDV_DOMAIN

def gen_loss_fn(soliton_params: dict[str, jax.Array],
                domain_init,
                ) -> Callable:
    """
    Generates a JIT-compatible PINN loss function for the KdV eq.
    - Will have to change this to add a batched version when domain size increases
    - Need to implement device state management (GPU)
    - Replace separate model and grad(model) calls with value_and_grad to compute u and u_x simultaneously and halve compute time.
    - Ensure jnp.trapezoid axes are explicitly mapped when integrating conserved quantities across multiple time slices to prevent cross-slice bugs.
    """
    #Calculate predicted info such that they are included in the closure of the func
    
    #Precompute exact arrays, these are added to the closure as constants
    u = build_n_soliton(soliton_params['k_vec'], d_vec=soliton_params['d_vec'])[0]
    u_ic = vmap(u)(domain_init.x_ic, domain_init.t_ic)

    #Initialize to 0.0 to avoid unbound variable errors in the closure
    momentum_quantity, energy_quantity, hamilt_quantity = jnp.array(0.0), jnp.array(0.0), jnp.array(0.0)

    if domain_init.t_momentum is not None:
        momentum_quantity = jnp.trapezoid(u_ic, domain_init.x_ic, axis=0)
    if domain_init.t_energy is not None:
        energy_quantity = jnp.trapezoid(u_ic**2, domain_init.x_ic, axis=0)
    if domain_init.t_hamilt is not None:
        u_x_ic = vmap(grad(u, argnums=0))(domain_init.x_ic, domain_init.t_ic)
        hamilt_quantity = jnp.trapezoid(u_ic**3 - 0.5*(u_x_ic**2), domain_init.x_ic, axis=0)

    #Closure func
    def loss_fn(model: eqx.Module, weights: Float[jax.Array, "6"], domain: KDV_DOMAIN):
        #IC
        u_pred_ic =  vmap(model)(domain.x_ic, domain.t_ic)
        loss_ic = jnp.mean(optl.l2_loss(u_pred_ic, u_ic)) 

        #BC
        u_pred_bc = vmap(model)(domain.x_bc, domain.t_bc)
        loss_bc = jnp.mean(optl.l2_loss(u_pred_bc, jnp.zeros_like(u_pred_bc))) #Dirichlet boundary condition 

        #PDE
        u_pred_pde, grads_pde = vmap(value_and_grad(model, argnums=(0,1)))(domain.x_coll, domain.t_coll)
        u_x_pde, u_t_pde = grads_pde
        u_xxx_pde = vmap(grad(grad(grad(model, argnums=0), argnums=0), argnums=0))(domain.x_coll, domain.t_coll)

        resid = u_t_pde + 6*u_pred_pde*u_x_pde + u_xxx_pde
        loss_pde = jnp.mean(optl.l2_loss(resid, jnp.zeros_like(resid)))
        
        #CONS - using the momentum of the system
        if domain.t_momentum is not None:
            u_pred_momentum = vmap(model)(domain.x_ic, domain.t_momentum)
            momentum_pred = jnp.trapezoid(u_pred_momentum, domain.x_ic, axis=0)
            loss_momentum = optl.l2_loss(momentum_pred, momentum_quantity)
        else: 
            loss_momentum = jnp.array(0.0)

        #CONS - using the energy of the system
        if domain.t_energy is not None:
            u_pred_energy = vmap(model)(domain.x_ic, domain.t_energy)
            energy_pred = jnp.trapezoid(u_pred_energy**2, domain.x_ic, axis=0)
            loss_energy = optl.l2_loss(energy_pred, energy_quantity)
        else:
            loss_energy = jnp.array(0.0)

        #CONS - using the hamiltonian of the system
        if domain.t_hamilt is not None:
            u_pred_hamilt, u_x_hamilt = vmap(value_and_grad(model, argnums=0))(domain.x_ic, domain.t_hamilt)
            hamilt_pred = jnp.trapezoid(u_pred_hamilt**3 - 0.5*(u_x_hamilt**2), domain.x_ic, axis=0)
            loss_hamilt = optl.l2_loss(hamilt_pred, hamilt_quantity)
        else:
            loss_hamilt = jnp.array(0.0)

        loss_comps = jnp.stack([loss_pde, loss_ic, loss_bc, loss_momentum, loss_energy, loss_hamilt])
        total_loss = jnp.dot(weights, loss_comps)
        return total_loss

    return loss_fn


        

