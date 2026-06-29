import jax
import jax.numpy as jnp

def build_n_soliton(k_vec: jax.Array,
                    d_vec: jax.Array
                    ) -> tuple[function]:
    """
    Using Karnav's work on the n_soliton (see PyTorch sister lib) and Dorey, P. E. (2026)
    In Solitons 2025-26 (Chapter 7) Durham University. A general builder for the functions f, f_x, f_xx used 
    in the Cole-Hopf Transform from the original n_soliton function.
    """
    N = k_vec.size
    k_row = k_vec[None, :]
    k_col = k_vec[:, None]
    C = 2.0 * k_col / (k_row + k_col)
    I = jnp.eye(N)

    def f(x: float, t: float) -> float:
        eta = k_vec * (x - k_vec**2 * t) + d_vec
        S = I + C * jnp.exp(eta)[None, :]
        return jnp.linalg.det(S)
    
    f_x = jax.grad(f, argnums=0)
    f_xx = jax.grad(f_x, argnums=0)

    def u(x: float, t: float) -> float: #this corresponds to the old version of n_soliton, returned as a func
        fv = f(x, t)
        fxv = f_x(x, t)
        fxxv = f_xx(x, t)
        return 2.0 * (fv * fxxv - fxv**2) / (fv**2)

    return u, f, f_x, f_xx

def phase_shifts(k_vec: jax.Array) -> jax.Array:
    """
    Compute the phase shifts for the linear combination of single-soliton solutions.
    """
    NotImplemented

def linear_combination(x: jax.Array,
                       t: jax.Array, 
                       k_vec: jax.Array, 
                       phi_vec: jax.Array
                       ) -> jax.Array:
    """
    Computes the linear solution of n soliton by creating a linear combination of single
    solitons, excluding the nonlinearity
    """
    NotImplemented