import jax
import torch


def typespec(d):
    return jax.tree.map(lambda x: type(x), d)


def spec(thing):
    return jax.tree.map(lambda x: x.shape if isinstance(x, torch.Tensor) else type(x), thing)


def find_torch_unstable(pytree):
    """Returns PyTree whose leaves are isnan."""

    def check_nan(x):
        unstable = lambda _x: torch.isinf(_x).any() or torch.isnan(_x).any()
        return unstable(x) if isinstance(x, torch.Tensor) else False

    return jax.tree.map(check_nan, pytree)
