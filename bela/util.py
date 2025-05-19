import jax
import torch


def spec(thing):
    return jax.tree.map(lambda x: x.shape if isinstance(x, torch.Tensor) else type(x), thing)
