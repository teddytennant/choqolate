"""CPU checks for the 2-additive Choquet kernel and the synthetic task."""

import jax
import jax.numpy as jnp
import pytest

from choqolate import (
    Choqolate,
    choquet,
    choquet_logit_grad,
    n_choquet_params,
    shapley,
    suppress,
    weights_from_logits,
)
import train


def _central_fd(weights, u, index, eps=1e-3):
    e = jnp.zeros_like(u).at[index].set(eps)
    return (choquet(weights, u + e) - choquet(weights, u - e)) / (2.0 * eps)


def test_softmax_weights_singleton_and_shapley():
    key = jax.random.PRNGKey(0)
    p = 5
    theta = jax.random.normal(key, (n_choquet_params(p),))
    w = weights_from_logits(theta)
    total = jnp.sum(w["a"]) + jnp.sum(w["b"]) + jnp.sum(w["c"])
    assert jnp.all(w["a"] >= 0)
    assert jnp.all(w["b"] >= 0)
    assert jnp.all(w["c"] >= 0)
    assert jnp.allclose(total, 1.0, atol=1e-6)
    assert jnp.allclose(jnp.diag(w["b"]), 0.0)
    assert jnp.allclose(jnp.diag(w["c"]), 0.0)
    assert jnp.allclose(w["b"], jnp.triu(w["b"], k=1))
    assert jnp.allclose(w["c"], jnp.triu(w["c"], k=1))
    assert jnp.allclose(jnp.sum(shapley(w)), 1.0, atol=1e-5)

    a = jnp.zeros((4,)).at[0].set(1.0)
    zeros = jnp.zeros((4, 4))
    pure = {"a": a, "b": zeros, "c": zeros}
    u = jnp.array([0.2, 0.5, 0.9, 0.1])
    assert jnp.allclose(choquet(pure, u), u[0])
    assert jnp.allclose(shapley(pure), jnp.array([1.0, 0.0, 0.0, 0.0]))

    # Pair terms: min and max, and the 1/2 split in Shapley.
    b = jnp.zeros((3, 3)).at[0, 1].set(1.0)
    c = jnp.zeros((3, 3)).at[0, 2].set(1.0)
    a3 = jnp.zeros((3,))
    u3 = jnp.array([0.2, 0.7, 0.4])
    assert jnp.allclose(choquet({"a": a3, "b": b, "c": zeros[:3, :3]}, u3), 0.2)
    assert jnp.allclose(
        shapley({"a": a3, "b": b, "c": jnp.zeros((3, 3))}),
        jnp.array([0.5, 0.5, 0.0]),
    )
    assert jnp.allclose(choquet({"a": a3, "b": jnp.zeros((3, 3)), "c": c}, u3), 0.4)
    assert jnp.allclose(
        shapley({"a": a3, "b": jnp.zeros((3, 3)), "c": c}),
        jnp.array([0.5, 0.0, 0.5]),
    )

    batch = jax.random.uniform(jax.random.PRNGKey(1), (8, p))
    out = choquet(w, batch)
    assert out.shape == (8,)
    assert jnp.allclose(out[0], choquet(w, batch[0]), atol=1e-6)


def test_monotone_in_each_coordinate():
    p = 5
    theta = jax.random.normal(jax.random.PRNGKey(3), (n_choquet_params(p),))
    w = weights_from_logits(theta)
    u = jax.random.uniform(jax.random.PRNGKey(4), (p,)) * 0.5
    key = jax.random.PRNGKey(5)
    base = choquet(w, u)
    for _ in range(20):
        key, kc, ks = jax.random.split(key, 3)
        coord = int(jax.random.randint(kc, (), 0, p))
        step = float(jax.random.uniform(ks, (), minval=0.05, maxval=0.45))
        moved = choquet(w, u.at[coord].add(step))
        assert float(moved) >= float(base) - 1e-5


def test_closed_form_grad_matches_jax():
    p = 4
    k1, k2 = jax.random.split(jax.random.PRNGKey(7))
    theta = jax.random.normal(k1, (n_choquet_params(p),))
    u = jax.random.uniform(k2, (p,))
    residual = 0.3
    g = choquet_logit_grad(theta, u, residual)

    def f(th):
        return residual * choquet(weights_from_logits(th), u)

    g_ref = jax.grad(f)(theta)
    assert jnp.linalg.norm(g) > 1e-6
    assert jnp.allclose(g, g_ref, atol=1e-5, rtol=1e-5)


def test_suppress_nullifies_concept():
    p = 4
    theta = jnp.zeros((n_choquet_params(p),))
    theta = theta.at[0].set(4.0)
    theta = theta.at[2].set(1.0)
    # b_{0,2} is the second upper-triangle pair and does not touch concept 1.
    theta = theta.at[p + 1].set(0.5)
    sup = suppress(theta, [1], p)
    assert float(sup[0]) == 4.0
    assert float(sup[2]) == 1.0
    assert float(sup[p + 1]) == 0.5
    assert float(sup[1]) == -1e9
    # b_{0,1} is the first pair and must be killed.
    assert float(sup[p]) == -1e9

    w = weights_from_logits(sup)
    assert float(shapley(w)[1]) < 1e-6
    u = jnp.array([0.3, 0.4, 0.5, 0.6])
    assert abs(float(_central_fd(w, u, 1))) < 1e-6
    # Concept 0 kept a positive singleton, so C still moves with u_0.
    assert abs(float(_central_fd(w, u, 0))) > 1e-2


def test_forward_shape():
    model = Choqolate(6, 4, 3, key=jax.random.PRNGKey(0))
    x = jax.random.uniform(jax.random.PRNGKey(1), (8, 6))
    y = model(x)
    assert y.shape == (8, 3)
    shap = model.node_shapley()
    assert shap.shape == (4, 6)
    assert jnp.allclose(jnp.sum(shap, axis=1), 1.0, atol=1e-5)


@pytest.fixture(scope="module")
def trained_run():
    return train.train_synthetic(steps=40, seed=0, eval_size=2048)


def test_cross_entropy_falls(trained_run):
    initial = trained_run["initial_ce"]
    final = trained_run["final_ce"]
    assert final < initial
    assert final < 0.9 * initial


def test_choq_pcr_intervention(trained_run):
    model = trained_run["model"]
    key = jax.random.fold_in(trained_run["key"], 7)
    x, y = train.synthetic_batch(key, 4096)
    acc_signal = _accuracy(model.with_suppressed([0, 1, 2]), x, y)
    acc_spurious = _accuracy(model.with_suppressed([3, 4, 5]), x, y)
    assert acc_spurious >= acc_signal - 0.05
    assert acc_signal < 0.7


def _accuracy(model, x, y):
    pred = jnp.argmax(model(x), axis=-1)
    return float(jnp.mean(pred == y))
