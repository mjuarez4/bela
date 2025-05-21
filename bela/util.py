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


def resize_to_480x640(img: torch.Tensor) -> torch.Tensor:
    """Resize (C, 224, 224) image to (C, 480, 640) with padding."""
    squeeze = img.ndim == 3
    if squeeze:
        img = img.unsqueeze(0)
    if img.ndim != 4:
        raise ValueError(f"unexpected shape {img.shape}")

    img = torch.nn.functional.interpolate(img, size=(480, 480), mode="bilinear", align_corners=False)
    img = torch.nn.functional.pad(img, (80, 80, 0, 0))
    return img.squeeze(0) if squeeze else img
