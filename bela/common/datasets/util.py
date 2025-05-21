from dataclasses import dataclass, field
import logging
from pathlib import Path

import jax
from pytorch3d.transforms import matrix_to_rotation_6d
import torch
from tqdm import tqdm

import bela


@dataclass
class DataStats:
    head: str
    stats: dict[str, dict[str, torch.Tensor]] = field(default_factory=lambda: {})

    @property
    def path(self):
        return Path(bela.ROOT) / f"stats_{self.head}.pt"

    def load(self):
        stats = torch.load(self.path) if self.path.exists() else {}
        self.stats = stats

    def save(self):
        torch.save(self.stats, self.path)

    def compute(self, dataset, batchspec):
        if not self.stats:
            self.stats = compute_stats(dataset, batchspec, head=self.head)
            self.save()

    def maybe_compute(self, dataset, batchspec):
        if not self.stats:
            self.load()
        if not self.stats:
            self.compute(dataset, batchspec)
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
        manopose = matrix_to_rotation_6d(manopose.reshape(-1, 3, 3))
        manopose = manopose.reshape(bs, t, -1, manopose.shape[-1])
        manopose = manopose.reshape(bs, t, -1) if flat else manopose
        batch["observation.human.mano.hand_pose"] = manopose

        # all the actions will have the same pad
        batch["action_is_pad"] = batch["observation.human.kp3d_is_pad"]

    if head == "robot":
        bs, t, *_ = batch["observation.robot.joints"].shape

        batch["observation.robot.gripper"] = batch["observation.robot.gripper"].reshape(bs, t, -1)
        logging.warning("cam.pose is not implemented")
        # zeros bs,t,9
        batch["observation.shared.cam.pose"] = torch.zeros((bs, t, 9), device=batch["observation.robot.joints"].device)
        # all the actions will have the same pad
        batch["action_is_pad"] = batch["observation.robot.joints_is_pad"]

    batch = {k: v for k, v in batch.items() if k in batchspec or k == "action_is_pad"}

    # design action
    # everything is a state variable if image is not in the name

    batch = jax.tree.map_with_path(maybe_squeeze_image, batch)

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


def compute_stats(dataset, batchspec, head):
    """Compute statistics for normalization
    you need separate stats for actions even with shared heads
    """

    stats, data = {}, []
    samples = list(range(len(dataset)))[:100]
    for i in tqdm(samples, total=len(samples), desc="Computing stats", leave=False):
        d = dataset[i]
        d.pop("task")
        d = jax.tree.map(lambda x: x.unsqueeze(0), d)
        d = postprocess(d, batchspec, head=head)
        d.pop("heads")
        data.append(d)
    data = jax.tree.map(lambda *x: torch.concatenate((x)), data[0], *data[1:])
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
