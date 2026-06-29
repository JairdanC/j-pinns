import jax
import jax.numpy as jnp
import equinox as eqx

class PINN(eqx.Module):
    """
    A PINN focused multilayer perceptron class used to replicate the basic network
    used in M. Raissi et. all (2019) in which tanh is the activation function between
    a series of linear layers
    """
    layers: tuple

    def __init__(self, key,
                 input: int = 2, 
                 output: int = 1, 
                 n_hidden: int  = 4, 
                 n_neurons: int = 32
                 ) -> None:
        """
        Initialization for a multilayer perceptron using a sequence of nn.Linear,
        activation and layers to build the network 
        """
        #build network
        keys = jax.random.split(key, (n_hidden + 2))
        temp_layers = [eqx.nn.Linear(input, n_neurons, key=keys[0])]
        temp_layers.append(jax.nn.tanh)
        for i in range(1, (n_hidden + 1)):
            temp_layers.append(eqx.nn.Linear(n_neurons, n_neurons, key=keys[i]))
            temp_layers.append(jax.nn.tanh)
        temp_layers.append(eqx.nn.Linear(n_neurons, output, key=keys[-1]))
        self.layers = tuple(temp_layers)


    def __call__(self, *args) -> jax.Array:
        """
        Forward pass of the network

        Uses *args to allow scalar inputs of different dims (ex: (x,t) -> u for KdV
        and (x, y, t) -> u for KP) without changing the function signature, it is 
        reccommended that __call__ is wrapped in equation specific PINN class
        """
        x = jnp.stack(args)
        for layer in self.layers:
            x = layer(x)
        return x
    
class SIREN(eqx.Module):
    NotImplemented