

from __future__ import annotations

import importlib.util
from typing import Callable

import numpy as np

_JAX_SPEC = importlib.util.find_spec("jax")
if _JAX_SPEC is None:
    raise RuntimeError("jax is required for carleman_coeffs.py")

import jax
import jax.numpy as jnp


LossFn = Callable[[jnp.ndarray, jnp.ndarray, object], jnp.ndarray]


def build_vector_field(loss_fn: LossFn, w_dim: int) -> Callable[[jnp.ndarray, object], jnp.ndarray]:
    

    @jax.jit
    def vector_field(v: jnp.ndarray, batch: object) -> jnp.ndarray:
        w = v[:w_dim]
        u = v[w_dim:]
        grad_w, grad_u = jax.grad(loss_fn, argnums=(0, 1))(w, u, batch)
        return jnp.concatenate([grad_w, -grad_u])

    return vector_field


def compute_coeffs(
    loss_fn: LossFn,
    v: jnp.ndarray,
    w_dim: int,
    batch: object,
    max_order: int = 2,
) -> dict[str, np.ndarray]:
    
    vector_field = build_vector_field(loss_fn, w_dim)
    vf_v = lambda v_in: vector_field(v_in, batch)

    F0 = vf_v(v)
    F1 = jax.jacfwd(vf_v)(v)
    coeffs = {
        "F0": np.asarray(F0),
        "F1": np.asarray(F1),
    }

    if max_order >= 2:
        F2 = 0.5 * jax.jacfwd(jax.jacfwd(vf_v))(v)
        coeffs["F2"] = np.asarray(F2)

    return coeffs


def verify_linear_approx(
    loss_fn: LossFn,
    v: jnp.ndarray,
    w_dim: int,
    batch: object,
    tol: float = 1e-2,
    seed: int = 0,
) -> None:
    
    rng = np.random.default_rng(seed)
    delta = jnp.asarray(rng.normal(scale=1e-2, size=v.shape))

    coeffs = compute_coeffs(loss_fn, v, w_dim, batch, max_order=1)
    F0 = coeffs["F0"]
    F1 = coeffs["F1"]

    vector_field = build_vector_field(loss_fn, w_dim)
    approx = F0 + F1 @ np.asarray(delta)
    true_val = np.asarray(vector_field(v + delta, batch))
    rel_error = np.linalg.norm(true_val - approx) / (np.linalg.norm(true_val) + 1e-8)
    if rel_error > tol:
        raise AssertionError(f"Linear approximation error too high: {rel_error:.4e}")
