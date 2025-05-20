import jax
import torch


def typespec(d):
    return jax.tree.map(lambda x: type(x), d)


def spec(thing):
    return jax.tree.map(lambda x: x.shape if isinstance(x, torch.Tensor) else type(x), thing)
