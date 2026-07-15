import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PyTree
import optax as opt
import equinox as eqx
from typing import Callable

def adam_train(model: eqx.Module, 
               loss_fn: Callable, 
               epochs: Int, 
               lr: Float
               ) -> eqx.Module:
    """
    Train the module passed as a parameter using ADAM optimization on the given loss
    function

    - Performance improvement, use JAX loop instead of compiled make step function once 
    verified working as intended
    - Add loss logging
    """
    optim = opt.adam(learning_rate=lr)
    optim_state = optim.init(eqx.filter(model, eqx.is_array))

    #start with uncompiled epoch section and compiled make_step function
    #I don't think that make step should take a batch since the loss should evaluate the entire domain for stability
    @eqx.filter_jit
    def make_step(model, optim_state):
        loss_val, grad = eqx.filter_value_and_grad(loss_fn)(model)
        updates, optim_state = optim.update(grad, optim_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model, optim_state, loss_val
    
    for epoch in range(epochs):
        model, optim_state, training_loss = make_step(model, optim_state)
        if (epoch % 200) == 0 or (epoch == epochs - 1):
            print(f'EPOCH: {epoch} | MODEL LOSS: {training_loss}')
    
    return model

    
def adam_train(model: eqx.Module, loss_fn: Callable, epochs, schedule):
    NotImplemented
    
def lbfgs_train(model: eqx.Module, 
               loss_fn: Callable, 
               epochs: Int, 
               lr: Float,
               mem_size: Int,
               tolerance: Float = 1e-9
               ) -> eqx.Module:
    
    """
    Train the module passed as a parameter using LBFGS optimization on the given loss
    function

    - Performance improvement, use JAX loop instead of compiled make step function once 
    verified working as intended
    - Add loss logging
    """
    
    optim = opt.lbfgs(learning_rate=lr, memory_size=mem_size)
    optim_state = optim.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def make_step(model, optim_state):
        loss_val, grad = eqx.filter_value_and_grad(loss_fn)(model)
        updates, optim_state = optim.update(grad, optim_state, params=eqx.filter(model, eqx.is_array),
                                            value=loss_val, grad=grad, value_fn=loss_fn)
        model = eqx.apply_updates(model, updates)
        return model, optim_state, loss_val, grad
    
    for epoch in range(epochs):
        model, optim_state, training_loss, grad = make_step(model, optim_state)
        if (epoch % 200) == 0 or (epoch == epochs - 1):
            print(f'EPOCH: {epoch} | MODEL LOSS: {training_loss}')

        grad_norm = jnp.linalg.norm(jax.tree_util.tree_leaves(grad)[0])
        if grad_norm < tolerance:
            print(f'Converged at epoch {epoch}')
            break

    return model

