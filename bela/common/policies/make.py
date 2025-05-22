from dataclasses import dataclass, field

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.configs.default import DatasetConfig
from lerobot.configs.types import FeatureType
from rich.pretty import pprint
import torch
import tyro

from bela.common.policies.bela import BELAPolicy
from bela.typ import Head, HeadSpec, Morph, PolicyFeature


@dataclass
class HybridConfig:
    dataset: DatasetConfig = field(default_factory=lambda: DatasetConfig(repo_id="none"))
    human_repos: list[str] = field(default_factory=list)
    robot_repos: list[str] = field(default_factory=list)

    policy: ACTConfig = field(default_factory=ACTConfig)


def make_policy(batchspec: dict[str, PolicyFeature], stats, examples):
    # batchspec = jax.tree.map(lambda x: x.flatten() if x.type == FeatureType.STATE else x, batchspec)
    # pprint(batchspec)
    input_features = {k: v for k, v in batchspec.items() if "observation" in k}
    output_features = {k: v for k, v in batchspec.items() if "action" in k}
    state_features = {k: v for k, v in input_features.items() if v.type == FeatureType.STATE}

    # batchspec = batchspec | unflatten_dict(output_features, sep=".")
    # pprint(batchspec)

    def compute_head(feat, head):
        headfeat = {k: v for k, v in feat if head in k}
        headfeat = {k: Head(None, v.shape) for k, v in headfeat.items()}
        return sum(list(headfeat.values()), Head(None, (0,)))

    headspec = HeadSpec(
        robot=Head(Morph.ROBOT, compute_head(state_features.items(), "robot").shape),
        human=Head(Morph.HUMAN, compute_head(state_features.items(), "human").shape),
        share=Head(Morph.HR, compute_head(state_features.items(), "shared").shape),
    )
    pprint(headspec)

    bs, chunk = 4, 50

    """
    _generate_feat = partial(generate_feat, batch_size=bs, chunk=chunk)
    example_batch = jax.tree.map(_generate_feat, batchspec)
    example_batch = jax.tree.map(torch.Tensor, example_batch)
    example_batch = flatten_dict(example_batch, sep=".")

    example_batch["action_is_pad"] = torch.zeros((bs, chunk)).bool()

    # example_stats = jax.tree.map(generate_stats, batchspec)
    # example_stats = jax.tree.map(torch.Tensor, example_stats)
    # is_leaf = lambda d, k: "count" in k
    # example_stats = flatten_dict(example_stats, sep=".", is_leaf=is_leaf)
    # pprint(spec(example_stats))
    """

    policycfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=chunk,
        n_action_steps=chunk,
    )
    policy = BELAPolicy(
        config=policycfg,
        headspec=headspec,
        dataset_stats=stats,
    )

    policy.eval()
    with torch.no_grad():  # Disable gradient tracking
        for head, ex in examples.items():
            print(f"fwd {head}")
            policy._forward(ex, heads=[head, "shared"])
    policy.train()

    # policy(example_batch, heads=["human", "shared"])
    # policy(example_batch, heads=["robot", "shared"])
    return policy


def test_fwd_bwd(cfg: HybridConfig):
    policy = make_policy()
    hdatasets, rdatasets = [], []
    for h in cfg.human_repos:
        cfg.dataset.repo_id = h
        hdatasets.append(make_dataset(cfg))
    for r in cfg.robot_repos:
        cfg.dataset.repo_id = r
        rdatasets.append(make_dataset(cfg))

    print(hdatasets)
    print(rdatasets)


def main(cfg: HybridConfig):
    test_fwd_bwd(cfg)


if __name__ == "__main__":
    main(tyro.cli(HybridConfig))
