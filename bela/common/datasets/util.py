from dataclasses import dataclass, field
from pathlib import Path

import jax
from pytorch3d.transforms import matrix_to_rotation_6d
import torch
from tqdm import tqdm


@dataclass
class DataStats:
    path: Path
    stats: dict[str, dict[str, torch.Tensor]] = field(default_factory=lambda: dict)

    def load(self):
        stats = torch.load(self.path) if self.path.exists() else {}
        self.stats = stats

    def save(self):
        torch.save(self.stats, self.path)

    def compute(self, dataset, batchspec, h):
        if not self.stats:
            self.stats = compute_stats(dataset, batchspec, h)
            self.save()


def postprocess(batch, batchspec, h, flat=True):
    if "task" in batch:
        batch.pop("task")  # it is annoying to pprint
    batch = {k.replace("observation.", f"observation.{h}."): v for k, v in batch.items()}
    batch = {k.replace(".state.", "."): v for k, v in batch.items()}

    # move shared features
    # if the key would be in sharespec, if named properly, then rename
    sharespec = {k: v for k, v in batchspec.items() if k.startswith("observation.shared")}

    def canshare(k):
        sharekey = k.replace(f"observation.{h}.", "observation.shared.")
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
    myfeat = "observation.shared.cam.pose"
    if h == "human":
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
        batch[myfeat] = torch.cat([palm, rot], dim=-1)

        # convert mano.hand_pose to rotation_6d
        manopose = batch["observation.human.mano.hand_pose"]  # b,t,m,3,3
        manopose = matrix_to_rotation_6d(manopose.reshape(-1, 3, 3))
        manopose = manopose.reshape(bs, t, -1, manopose.shape[-1])
        manopose = manopose.reshape(bs, t, -1) if flat else manopose
        batch["observation.human.mano.hand_pose"] = manopose

    if h == "robot":
        pass

    # pprint(spec(batch))
    batch = {k: v for k, v in batch.items() if k in batchspec}
    # design action
    # everything is a state variable if image is not in the name
    isstate = lambda x: "image" not in x
    actions = {}
    for k, v in batch.items():
        if isstate(k):
            a = batch[k]
            # first one is qpos
            q = batch[k][:, 0]  # b,0 not 0,t
            actions[k.replace("observation.", "action.")] = a
            actions[k] = q
    batch = batch | actions

    batch["heads"] = [h, "shared"]
    return batch


def compute_stats(dataset, batchspec, h):
    # Compute statistics for normalization
    # TODO you will need separate stats for actions even though shared
    stats = {}
    data = []
    samples = list(range(len(dataset)))[:10]
    for i in tqdm(samples, total=len(samples), desc="Computing stats", leave=False):
        d = dataset[i]
        d.pop("task")
        d = jax.tree.map(lambda x: x.unsqueeze(0), d)
        d = postprocess(d, batchspec, h="human")
        d.pop("heads")
        data.append(d)
    data = jax.tree.map(lambda *x: torch.concatenate((x)), data[0], *data[1:])

    def make_stat(stack):
        return {
            "mean": stack.mean(dim=0),
            "std": stack.std(dim=0),
            "max": stack.max(dim=0)[0],
            "min": stack.min(dim=0)[0],
            "count": stack.shape[0],
        }

    stats = jax.tree.map(lambda x: make_stat(x), data)

    def take_act(k, v):
        y = k[0].key.startswith("action") and isinstance(v, torch.Tensor)
        return v[0] if y else v

    stats = jax.tree.map_with_path(take_act, stats)
    return stats
