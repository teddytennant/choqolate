"""2-additive Choquet layer (CHOQOLATE)."""

from choqolate.core import (
    Choqolate,
    choquet,
    choquet_logit_grad,
    n_choquet_params,
    shapley,
    suppress,
    weights_from_logits,
)

__all__ = [
    "Choqolate",
    "choquet",
    "choquet_logit_grad",
    "n_choquet_params",
    "shapley",
    "suppress",
    "weights_from_logits",
]
