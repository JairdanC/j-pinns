import equinox as eqx
import jax
import jax.numpy as jnp

#FLOAT DEF
_FLOAT = jnp.float32
class KDV_DOMAIN(eqx.Module):

    x_coll: jax.Array
    t_coll: jax.Array
    x_ic: jax.Array
    t_ic: jax.Array
    x_bc: jax.Array
    t_bc: jax.Array
    t_momentum: jax.Array | None
    t_energy: jax.Array | None
    t_hamilt: jax.Array | None
    
    x_lims: tuple = eqx.field(static=True)
    t_lims: tuple = eqx.field(static=True)

    def __init__(self, key, x_lims: tuple[int], t_lims: tuple[int], 
                 n_coll: int, n_ic: int, n_bc: int,
                 n_momentum: int | None = None, n_energy: int | None = None, 
                 n_hamilt: int | None = None
                 ) -> None:
        
        self.x_lims = x_lims
        self.t_lims = t_lims
    
        key_x, key_t = jax.random.split(key)
        self.x_coll = jax.random.uniform(key_x, shape=(n_coll,), dtype=_FLOAT, minval=x_lims[0], maxval=x_lims[1])
        self.t_coll = jax.random.uniform(key_t, shape=(n_coll,), dtype=_FLOAT, minval=t_lims[0], maxval=t_lims[1])

        t_bound_left = jnp.linspace(t_lims[0], t_lims[1], num=n_bc//2, dtype=_FLOAT)
        t_bound_right = jnp.linspace(t_lims[0], t_lims[1], num=n_bc//2, dtype=_FLOAT)
        
        x_bound_left = jnp.full_like(t_bound_left, x_lims[0], dtype=_FLOAT)
        x_bound_right = jnp.full_like(t_bound_right, x_lims[1], dtype=_FLOAT)
        
        self.x_bc = jnp.concatenate([x_bound_left, x_bound_right])
        self.t_bc = jnp.concatenate([t_bound_left, t_bound_right])

        self.x_ic = jnp.linspace(x_lims[0], x_lims[1], num=n_ic, dtype=_FLOAT)
        self.t_ic = jnp.ones_like(self.x_ic, dtype=_FLOAT) * t_lims[0]


        if n_momentum is not None:
            self.t_momentum = jnp.linspace(t_lims[0], t_lims[1], num=n_momentum, dtype=_FLOAT)
        else: 
            self.t_momentum = None
            
        if n_energy is not None:
            self.t_energy = jnp.linspace(t_lims[0], t_lims[1], num=n_energy, dtype=_FLOAT)
        else: 
            self.t_energy = None
            
        if n_hamilt is not None:
            self.t_hamilt = jnp.linspace(t_lims[0], t_lims[1], num=n_hamilt, dtype=_FLOAT)
        else: 
            self.t_hamilt = None