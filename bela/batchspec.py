from dataclasses import dataclass, field

from flax.traverse_util import flatten_dict
from lerobot.configs.types import FeatureType
import numpy as np

from .typ import PolicyFeature


@dataclass

class BatchSpec:
    """Container for batch features."""

    robot: dict[str, PolicyFeature] = field(default_factory=dict)
    human: dict[str, PolicyFeature] = field(default_factory=dict)
    shared: dict[str, PolicyFeature] = field(default_factory=dict)

    def flat(self) -> dict[str, PolicyFeature]:
        spec = {
            "observation": {
                "robot": self.robot,
                "human": self.human,
                "shared": self.shared,
            }
        }
        return flatten_dict(spec, sep=".")

    def _filter(self, typ: FeatureType, *, invert=False) -> "BatchSpec":
        pred = (lambda pf: pf.type == typ) if not invert else (lambda pf: pf.type != typ)
        return BatchSpec(
            robot={k: v for k, v in self.robot.items() if pred(v)},
            human={k: v for k, v in self.human.items() if pred(v)},
            shared={k: v for k, v in self.shared.items() if pred(v)},
        )

    def flat_vector(self, typ: FeatureType | None = None, *, invert=False) -> np.ndarray:
        spec = self if typ is None else self._filter(typ, invert=invert)
        total = sum(np.prod(f.shape) for f in spec.flat().values())
        return np.zeros(total)

    def update(self, other: "BatchSpec") -> None:
        self.robot.update(other.robot)
        self.human.update(other.human)
        self.shared.update(other.shared)

    def diff(self, other: "BatchSpec") -> dict[str, tuple[PolicyFeature, PolicyFeature | None]]:
        diff = {}
        mine, theirs = self.flat(), other.flat()
        for k, v in mine.items():
            if k not in theirs or theirs[k] != v:
                diff[k] = (v, theirs.get(k))
        for k, v in theirs.items():
            if k not in mine:
                diff[k] = (mine.get(k), v)
        return diff

    def __eq__(self, other: object) -> bool:
        return isinstance(other, BatchSpec) and self.flat() == other.flat()

    def __or__(self, other: "BatchSpec") -> "BatchSpec":
        return BatchSpec(
            robot={**self.robot, **other.robot},
            human={**self.human, **other.human},
            shared={**self.shared, **other.shared},
        )

    def __ror__(self, other: "BatchSpec") -> "BatchSpec":
        return self.__or__(other)

    def __ior__(self, other: "BatchSpec") -> "BatchSpec":
        self.update(other)
        return self


DEFAULT_SPEC = BatchSpec(
    robot={
        "joints": PolicyFeature(FeatureType.STATE, (7,)),
        "image.side": PolicyFeature(FeatureType.VISUAL, (3, 480, 640)),
        "image.wrist": PolicyFeature(FeatureType.VISUAL, (3, 480, 640)),
        "gripper": PolicyFeature(FeatureType.STATE, (1,)),
    },
    human={
        "mano.hand_pose": PolicyFeature(FeatureType.STATE, (15, 9)),
    },
    shared={
        "image.low": PolicyFeature(FeatureType.VISUAL, (3, 480, 640)),
        "cam.pose": PolicyFeature(FeatureType.STATE, (6,)),
    },
)

SIMPLE_SPEC = BatchSpec(robot={"joints": PolicyFeature(FeatureType.STATE, (7,))})
