"""Kernel of the 2-additive Choquet integral used by CHOQOLATE.

Weight order in theta is singletons a_0..a_{p-1}, then pairwise min
coefficients b_{j,l} for j < l in row-major order, then pairwise max
coefficients c_{j,l} in the same order. The weight softmax temperature is 1.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx


# Paper setting. Integral outputs lie in [0, 1], so an untempered softmax
# stays near uniform. Tce = 0.005 restores a usual logit range.
CE_TEMPERATURE = 0.005


def n_choquet_params(p: int) -> int:
    """Number of unconstrained logits for a p-input 2-additive integral."""
    if p < 1:
        raise ValueError(f"p must be positive, got {p}")
    return p * p


def _split_p(theta) -> int:
    q = int(theta.shape[-1])
    p = int(round(q**0.5))
    if p * p != q:
        raise ValueError(f"theta length {q} is not a square")
    return p


def _pairs_to_matrix(values, p: int):
    j, l = jnp.triu_indices(p, k=1)
    mat = jnp.zeros((p, p), dtype=values.dtype)
    return mat.at[j, l].set(values)


def weights_from_logits(theta) -> dict:
    """Softmax reparametrization. Upper triangle of b and c is used; diagonal is 0."""
    theta = jnp.asarray(theta)
    p = _split_p(theta)
    w = jax.nn.softmax(theta)
    n_pairs = p * (p - 1) // 2
    a = w[:p]
    b = _pairs_to_matrix(w[p : p + n_pairs], p)
    c = _pairs_to_matrix(w[p + n_pairs :], p)
    return {"a": a, "b": b, "c": c}


def features(u):
    """Lift scores to (u_j, min pairs, max pairs). Accepts (p,) or (B, p)."""
    u = jnp.asarray(u)
    single = u.ndim == 1
    if single:
        u = u[None, :]
    p = int(u.shape[-1])
    j, l = jnp.triu_indices(p, k=1)
    mins = jnp.minimum(u[:, j], u[:, l])
    maxs = jnp.maximum(u[:, j], u[:, l])
    phi = jnp.concatenate([u, mins, maxs], axis=-1)
    return phi[0] if single else phi


def _flat_weights(weights) -> jax.Array:
    a = weights["a"]
    b = weights["b"]
    c = weights["c"]
    p = int(a.shape[0])
    j, l = jnp.triu_indices(p, k=1)
    return jnp.concatenate([a, b[j, l], c[j, l]])


def choquet(weights, u):
    """2-additive Choquet integral. u is (p,) or (B, p)."""
    phi = features(u)
    flat = _flat_weights(weights)
    if phi.ndim == 1:
        return jnp.dot(flat, phi)
    return phi @ flat


def shapley(weights) -> jax.Array:
    """Closed-form Shapley values. Sums to 1 when the weights do."""
    a = weights["a"]
    b = weights["b"]
    c = weights["c"]
    p = int(a.shape[0])
    j, l = jnp.triu_indices(p, k=1)
    pair = 0.5 * (b[j, l] + c[j, l])
    shap = a.at[j].add(pair)
    shap = shap.at[l].add(pair)
    return shap


def choquet_logit_grad(theta, u, residual) -> jax.Array:
    """Eq. 7: d(residual * C)/d theta_i = residual * w_i * (phi_i - C).

    This is jax.grad of residual * C(softmax(theta), u). No extra loss term.
    """
    theta = jnp.asarray(theta)
    u = jnp.asarray(u)
    residual = jnp.asarray(residual)
    w = jax.nn.softmax(theta)
    phi = features(u)
    c_val = jnp.dot(w, phi)
    return residual * w * (phi - c_val)


def _touch_mask(p: int, concepts) -> jax.Array:
    concepts = jnp.asarray(concepts, dtype=jnp.int32).reshape(-1)
    q = n_choquet_params(p)
    mask = jnp.zeros((q,), dtype=bool)
    mask = mask.at[concepts].set(True)
    j, l = jnp.triu_indices(p, k=1)
    touches = jnp.isin(j, concepts) | jnp.isin(l, concepts)
    n_pairs = p * (p - 1) // 2
    mask = mask.at[p : p + n_pairs].set(touches)
    mask = mask.at[p + n_pairs :].set(touches)
    return mask


def suppress(theta, concepts, p: int) -> jax.Array:
    """CHOQ-PCR on logits: every coordinate that touches `concepts` becomes -1e9.

    Other logits are unchanged. Softmax then renormalizes the surviving weights.
    """
    theta = jnp.asarray(theta)
    mask = _touch_mask(p, concepts)
    return jnp.where(mask, jnp.asarray(-1e9, dtype=theta.dtype), theta)


def softmax_cross_entropy(logits, labels, temperature: float = CE_TEMPERATURE):
    """Mean softmax cross-entropy. labels are class indices or one-hot."""
    logits = jnp.asarray(logits) / temperature
    labels = jnp.asarray(labels)
    log_probs = jax.nn.log_softmax(logits)
    if labels.ndim == 1:
        n = logits.shape[0]
        return -jnp.mean(log_probs[jnp.arange(n), labels])
    return -jnp.mean(jnp.sum(labels * log_probs, axis=-1))


class Choqolate(eqx.Module):
    """Two Choquet layers: M concepts -> N nodes -> C class scores.

    Node outputs are clipped to [0, 1] before the second layer so a tiny
    numeric overshoot does not leave the domain of the next integral.
    Stored fields are unconstrained logits. Weights are softmax(theta).
    """

    node_theta: jax.Array
    class_theta: jax.Array

    def __init__(
        self,
        n_concepts: int,
        n_nodes: int,
        n_classes: int,
        *,
        key,
        init_scale: float = 0.05,
    ):
        k1, k2 = jax.random.split(key)
        self.node_theta = (
            jax.random.normal(k1, (n_nodes, n_choquet_params(n_concepts))) * init_scale
        )
        self.class_theta = (
            jax.random.normal(k2, (n_classes, n_choquet_params(n_nodes))) * init_scale
        )

    @property
    def n_nodes(self) -> int:
        return int(self.node_theta.shape[0])

    @property
    def n_classes(self) -> int:
        return int(self.class_theta.shape[0])

    @property
    def n_concepts(self) -> int:
        return int(round(self.node_theta.shape[-1] ** 0.5))

    def __call__(self, concepts):
        concepts = jnp.asarray(concepts)

        def node_out(theta):
            return choquet(weights_from_logits(theta), concepts)

        nodes = jax.vmap(node_out)(self.node_theta)
        nodes = jnp.clip(nodes, 0.0, 1.0).T

        def class_out(theta):
            return choquet(weights_from_logits(theta), nodes)

        logits = jax.vmap(class_out)(self.class_theta)
        return jnp.transpose(logits)

    def node_shapley(self):
        def one(theta):
            return shapley(weights_from_logits(theta))

        return jax.vmap(one)(self.node_theta)

    def interaction_penalty(self):
        """Sum of first-layer min and max weights. Singletons are excluded."""

        def one(theta):
            w = weights_from_logits(theta)
            return jnp.sum(w["b"]) + jnp.sum(w["c"])

        return jnp.sum(jax.vmap(one)(self.node_theta))

    def loss(self, concepts, labels, *, l1: float = 1e-3, temperature: float = CE_TEMPERATURE):
        ce = softmax_cross_entropy(self(concepts), labels, temperature)
        return ce + l1 * self.interaction_penalty()

    def with_suppressed(self, concepts):
        """CHOQ-PCR on the first layer. Other logits are unchanged."""
        p = self.n_concepts
        new_theta = jax.vmap(lambda th: suppress(th, concepts, p))(self.node_theta)
        return eqx.tree_at(lambda m: m.node_theta, self, new_theta)
