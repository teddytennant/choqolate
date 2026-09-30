JAX kernel of CHOQOLATE (arXiv:2609.37786): a 2-additive Choquet layer that merges concept scores, with closed-form Shapley values and post-hoc concept suppression.

PYTHONPATH=src python -m pytest
PYTHONPATH=src python train.py

Does not reproduce the CLIP concept-bottleneck tables. Those need frozen ViT-L/14 embeddings and the four image datasets. This checks the integral, the Shapley formula, the closed-form softmax gradient, and that suppressing the concepts a synthetic label does not use beats suppressing the ones it does.

Weights are a softmax over singletons and pairwise min and max terms. Node outputs are clipped to [0, 1] before the second layer. The l1 penalty hits interaction weights only. Training adds that penalty to softmax cross-entropy at temperature 0.005, the paper's setting for scores that lie in [0, 1]. No CLIP encoder is included.
