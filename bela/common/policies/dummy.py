from lerobot.configs.types import FeatureType
import numpy as np

from bela.typ import PolicyFeature


def generate_stats(x: PolicyFeature):
    """Generate fake stats for the dataset.
    Args:
        x: The input feature.
    """
    return {
        "count": np.array(55),
        "max": np.random.random(x.shape),
        "min": np.random.random(x.shape),
        "mean": np.random.random(x.shape),
        "std": np.random.random(x.shape),
    }


def generate_feat(x, batch_size, chunk=1):
    shape = list(x.shape)  # these should be flat

    match x.type:
        case FeatureType.STATE:
            return np.zeros(([batch_size] + shape))
        case FeatureType.VISUAL:
            return np.zeros(([batch_size] + shape))
        case FeatureType.ACTION:
            return np.zeros(([batch_size, chunk] + shape))
        case _:
            raise ValueError(f"Unknown feature type: {x.type}")
