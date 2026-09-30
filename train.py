"""Train CHOQOLATE on a synthetic two-latent concept task."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from choqolate import Choqolate
from choqolate.core import CE_TEMPERATURE, softmax_cross_entropy


N_CONCEPTS = 6
N_NODES = 3
N_CLASSES = 2
BATCH = 64
LR = 1e-2
NOISE = 0.02
L1 = 1e-3


def synthetic_batch(key, batch_size: int = BATCH, noise: float = NOISE):
    """Concepts 0,1,2 are noisy copies of one latent. 3,4,5 are another.

    The label is 1 when the first latent is above 0.5.
    """
    k1, k2, k3, k4 = jax.random.split(key, 4)
    z1 = jax.random.uniform(k1, (batch_size,))
    z2 = jax.random.uniform(k2, (batch_size,))
    n1 = jax.random.normal(k3, (batch_size, 3)) * noise
    n2 = jax.random.normal(k4, (batch_size, 3)) * noise
    concepts = jnp.concatenate(
        [
            jnp.clip(z1[:, None] + n1, 0.0, 1.0),
            jnp.clip(z2[:, None] + n2, 0.0, 1.0),
        ],
        axis=1,
    )
    labels = (z1 > 0.5).astype(jnp.int32)
    return concepts, labels


def train_synthetic(steps: int, seed: int = 0, eval_size: int = 1024):
    """Adam on resampled batches. Losses are measured on a fixed eval batch."""
    key = jax.random.PRNGKey(seed)
    k_model, k_eval, k_train = jax.random.split(key, 3)
    model = Choqolate(N_CONCEPTS, N_NODES, N_CLASSES, key=k_model)
    eval_x, eval_y = synthetic_batch(k_eval, eval_size)
    optimizer = optax.adam(LR)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def step(model, opt_state, x, y):
        def loss_fn(m):
            return m.loss(x, y, l1=L1, temperature=CE_TEMPERATURE)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    def metrics(m):
        ce = softmax_cross_entropy(m(eval_x), eval_y, CE_TEMPERATURE)
        total = ce + L1 * m.interaction_penalty()
        return float(ce), float(total)

    initial_ce, initial_loss = metrics(model)
    for _ in range(steps):
        k_train, kb = jax.random.split(k_train)
        x, y = synthetic_batch(kb, BATCH)
        model, opt_state, _ = step(model, opt_state, x, y)
    final_ce, final_loss = metrics(model)
    return {
        "model": model,
        "initial_ce": initial_ce,
        "final_ce": final_ce,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "key": k_train,
    }


def main() -> int:
    result = train_synthetic(steps=60, seed=0)
    print(f"initial loss {result['initial_loss']:.6f}")
    print(f"final loss {result['final_loss']:.6f}")
    return 0 if result["final_loss"] < result["initial_loss"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
