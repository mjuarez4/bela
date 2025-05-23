from dataclasses import dataclass

import jax
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import math
from pytorch3d.transforms import matrix_to_rotation_6d
import torch
from tqdm import tqdm
from xgym import BASE
from xgym.calibrate.urdf.robot import RobotTree

import bela
from bela.util import resize_to_480x640

_ROBOT_TREE = RobotTree()
cam2base = torch.Tensor(np.load(str(BASE / "base.npz"))["base"])

if not (statdir := bela.ROOT / "stats").exists():
    statdir.mkdir(parents=True, exist_ok=True)


@dataclass
class DataStats:
    head: str
    repo_id: str  # hf repo ie : "lerobot/robot"
    dataset: LeRobotDataset

    stats: dict[str, dict[str, torch.Tensor]] | None = None
    quick: bool = False  # quick compute stats ?

    @property
    def loc(self):
        return bela.ROOT / "stats"

    @property
    def path(self):
        repo = self.repo_id.replace("/", "_")
        return self.loc / f"stats-{self.head}-{repo}.pt"

    def load(self):
        stats = torch.load(self.path) if self.path.exists() else {}
        self.stats = stats

    def save(self):
        torch.save(self.stats, self.path)

    def compute(self, batchspec):
        if not self.stats:
            self.stats = compute_stats(self.dataset, batchspec, head=self.head, quick=self.quick)
            self.save()

    def maybe_compute(self, batchspec):
        if not self.stats:
            self.load()
        if not self.stats:
            self.compute(batchspec)
        return self.stats


def join_jaxpath(path):
    return ".".join([str(x.key) for x in path])


def maybe_squeeze_image(path, mim):
    """some images are accidentally 4d, so we need to squeeze them
    this means the batch might be 5d
    """
    path = join_jaxpath(path)
    if "image" not in path:
        return mim

    if mim.ndim == 4:
        return mim
    if mim.ndim == 5:
        return mim.squeeze(1)
    raise ValueError(f"Image is not 4d: {path} {mim.shape}")


def maybe_resize_robot_image(path, img):
    """Resize robot images from 224×224 to 480×640."""
    path = join_jaxpath(path)
    if "observation.robot" not in path or "image" not in path:
        return img
    if img.shape[-2:] == (224, 224):
        return resize_to_480x640(img)
    return img


def matx_to_9d(mat: torch.Tensor) -> torch.Tensor:
    assert mat.shape[-2:] == (4, 4), "Input must end in (4, 4)"

    translation = mat[..., :3, 3]
    rotation = mat[..., :3, :3]
    rot6d = matrix_to_rotation_6d(rotation)
    return torch.cat([translation, rot6d], dim=-1)


def compute_robot_campose(batch):
    bs, t, *_ = batch["observation.robot.joints"].shape
    kin = _ROBOT_TREE.set_pose(batch["observation.robot.joints"].reshape(-1, 7))  # , end_only=True)
    kin = jax.tree.map(lambda x: x.get_matrix(), kin)
    kin = jax.tree.map(lambda x: x.reshape((bs, t, *x.shape[1:])), kin)

    kin = jax.tree.map(lambda x: cam2base @ x, kin)  # now in camera space
    kin = jax.tree.map(lambda x: matx_to_9d(x), kin)

    return kin["link_eef"]  # b,t,9


def postprocess(batch, batchspec, head, flat=True):
    if "task" in batch:
        batch.pop("task")  # it is annoying to pprint
    batch = {k.replace("observation.", f"observation.{head}."): v for k, v in batch.items()}
    batch = {k.replace(".state.", "."): v for k, v in batch.items()}

    # move shared features
    # if the key would be in sharespec, if named properly, then rename
    sharespec = {k: v for k, v in batchspec.items() if k.startswith("observation.shared")}

    def canshare(k):
        sharekey = k.replace(f"observation.{head}.", "observation.shared.")
        return sharekey in sharespec, sharekey

    newbatch = {}
    for k, v in batch.items():
        can, key = canshare(k)
        # print(k, key)
        if can:
            newbatch[key] = v
        else:
            newbatch[k] = v
    batch = newbatch

    # create shared features that don't exist
    if head == "human":
        kp3d = batch["observation.human.kp3d"]
        bs, t, n, *_ = kp3d.shape
        palm = kp3d[:, :, 0]  # b,t,n,3 -> b,t,3
        kp3d = kp3d[:, :, 1:]  # should be 20 now
        kp3d = kp3d.reshape(bs, t, -1) if flat else kp3d
        batch["observation.human.kp3d"] = kp3d

        bs, t, *_ = kp3d.shape
        rot = batch["observation.human.mano.global_orient"]
        rot = matrix_to_rotation_6d(rot.reshape(-1, 3, 3))
        rot = rot.reshape(bs, t, -1)
        batch["observation.shared.cam.pose"] = torch.cat([palm, rot], dim=-1)

        # convert mano.hand_pose to rotation_6d
        manopose = batch["observation.human.mano.hand_pose"]  # b,t,m,3,3
        selected = [0, 1, 2, 3, 6, 9, 12]  # palm
        manopose = manopose[:, :, selected]
        manopose = matrix_to_rotation_6d(manopose.reshape(-1, 3, 3))
        manopose = manopose.reshape(bs, t, -1, manopose.shape[-1])
        manopose = manopose.reshape(bs, t, -1) if flat else manopose
        batch["observation.human.mano.hand_pose"] = manopose

        # all the actions will have the same pad
        batch["action_is_pad"] = batch["observation.human.kp3d_is_pad"]

    if head == "robot":
        bs, t, *_ = batch["observation.robot.joints"].shape

        batch["observation.robot.gripper"] = batch["observation.robot.gripper"].reshape(bs, t, -1)
        # zeros bs,t,9
        batch["observation.shared.cam.pose"] = compute_robot_campose(batch)
        # all the actions will have the same pad
        batch["action_is_pad"] = batch["observation.robot.joints_is_pad"]

    batch = {k: v for k, v in batch.items() if k in batchspec or k == "action_is_pad"}

    # design action
    # everything is a state variable if image is not in the name

    batch = jax.tree.map_with_path(maybe_squeeze_image, batch)
    batch = jax.tree.map_with_path(maybe_resize_robot_image, batch)

    isstate = lambda x: "image" not in x and "action" not in x
    actions = {}
    for k, v in batch.items():
        if isstate(k):
            a = batch[k]
            # first one is qpos
            q = batch[k][:, 0]  # b,0 not 0,t
            actions[k.replace("observation.", "action.")] = a
            actions[k] = q
    batch = batch | actions

    batch["heads"] = [head, "shared"]
    return batch


def compute_stats(dataset, batchspec, head, quick=False, batch_size=64, num_workers=4):
    """Compute statistics using a dataloader for speed."""

    limit = min(len(dataset), 8000) if quick else len(dataset)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
    )

    data, seen = [], 0
    total_batches = math.ceil(limit / batch_size)
    for batch in tqdm(loader, total=total_batches, desc="Computing stats", leave=False):
        if seen >= limit:
            break
        batch.pop("task", None)
        batch = postprocess(batch, batchspec, head=head)
        batch.pop("heads")
        keep = min(limit - seen, next(iter(batch.values())).shape[0])
        if keep != next(iter(batch.values())).shape[0]:
            batch = jax.tree.map(lambda x: x[:keep], batch)
        seen += keep
        data.append(batch)
    data = jax.tree.map(lambda *x: torch.concatenate(x), *data)
    data.pop("action_is_pad")  # doesnt need stats

    def make_stat(stack):
        return {
            "mean": stack.mean(dim=0),
            "std": stack.std(dim=0),
            "max": stack.max(dim=0)[0],
            "min": stack.min(dim=0)[0],
            "count": stack.shape[0],
        }

    def take_act(k, v):
        y = k[0].key.startswith("action") and isinstance(v, torch.Tensor)
        return v[0] if y else v

    stats = jax.tree.map(lambda x: make_stat(x), data)
    stats = jax.tree.map_with_path(take_act, stats)
    return stats
