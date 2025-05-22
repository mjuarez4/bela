import einops
from flax.traverse_util import flatten_dict
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import (
    ACT,
    ACTDecoder,
    ACTEncoder,
    ACTPolicy,
    ACTSinusoidalPositionEmbedding2d,
    create_sinusoidal_pos_embedding,
)
from lerobot.common.policies.normalize import Normalize, Unnormalize
import torch
from torch import Tensor, nn
import torch.nn.functional as F  # noqa: N812
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from bela.typ import HeadSpec


def recursive_moduledict(d: dict):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = recursive_moduledict(v)
        elif isinstance(v, nn.Module):
            out[k] = v
        else:
            raise TypeError(f"Unsupported type for key '{k}': {type(v)}")
    return nn.ModuleDict(out)


class BELA(ACT):
    """following egomimic
    also see ACT
    """

    def __init__(self, config: ACTConfig, headspec: HeadSpec):
        # BERT style VAE encoder with input tokens [cls, robot_state, *action_sequence].
        # The cls token forms parameters of the latent's distribution (like this [*means, *log_variances]).
        nn.Module.__init__(self)  # dont use ACT init
        self.config = config
        self.headspec = headspec

        if self.config.use_vae:
            self.build_vae(config)

        # Backbone for image feature extraction.

        if self.config.image_features:
            backbone_model = getattr(torchvision.models, config.vision_backbone)(
                replace_stride_with_dilation=[
                    False,
                    False,
                    config.replace_final_stride_with_dilation,
                ],
                weights=config.pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )
            # Note: The assumption here is that we are using a ResNet model
            # (and hence layer4 is the final feature map).
            # Note: The forward method of this returns a dict: {"feature_map": output}.
            self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

        # Transformer (acts as VAE decoder when training with the variational objective).
        self.encoder = ACTEncoder(config)
        self.decoder = ACTDecoder(config)

        #
        # Transformer encoder input projections. The tokens will be structured like
        # [latent, (robot_state), (env_state), (image_feature_map_pixels)].
        #

        self.proj = {
            "in": {
                "robot": nn.Linear(self.headspec.r[0], config.dim_model),
                "human": nn.Linear(self.headspec.h[0], config.dim_model),
                "shared": nn.Linear(self.headspec.s[0], config.dim_model),
            },
            "out": {
                "robot": nn.Linear(config.dim_model, self.headspec.r[0]),
                "human": nn.Linear(config.dim_model, self.headspec.h[0]),
                "shared": nn.Linear(config.dim_model, self.headspec.s[0]),
            },
        }
        self.proj = recursive_moduledict(self.proj)

        self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)

        if self.config.image_features:
            self.encoder_img_feat_input_proj = nn.Conv2d(backbone_model.fc.in_features, config.dim_model, kernel_size=1)

        # Transformer encoder positional embeddings.
        n_1d_tokens = 1  # for the latent
        n_1d_tokens += 1  # for the robot state
        n_1d_tokens += 1  # for the shared state
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)
        if self.config.image_features:
            self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)

        # Transformer decoder.
        # Learnable positional embedding for the transformer's decoder (in the style of DETR object queries).
        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)

        self._reset_parameters()

    def build_vae(self, config):
        self.vae = {
            "encoder": ACTEncoder(config, is_vae_encoder=True),
            "cls": nn.Embedding(1, config.dim_model),
            # Projection layer from the VAE encoder's output to the latent distribution's parameter space.
            "out": nn.Linear(config.dim_model, config.latent_dim * 2),
            "proj": {
                "robot": {
                    "state": nn.Linear(self.headspec.r[0], config.dim_model),
                    "action": nn.Linear(self.headspec.r[0], config.dim_model),
                },
                "human": {
                    "state": nn.Linear(self.headspec.h[0], config.dim_model),
                    "action": nn.Linear(self.headspec.h[0], config.dim_model),
                },
            },
        }
        self.vae = recursive_moduledict(self.vae)

        # Fixed sinusoidal positional embedding for the input to the VAE encoder.
        # Unsqueeze for batch dimension.
        num_input_token_encoder = 1 + config.chunk_size
        if self.config.robot_state_feature:
            num_input_token_encoder += 1
        self.register_buffer(
            "vae_encoder_pos_enc",
            create_sinusoidal_pos_embedding(num_input_token_encoder, config.dim_model).unsqueeze(0),
        )

    def compute_style(self, batch, use, h):
        batch_size = batch["observation.images"][0].shape[0]
        if not use:
            # When not using the VAE encoder, we set the latent to be all zeros.
            mu = log_sigma_x2 = None
            # TODO(rcadene, alexander-soare): remove call to `.to` to speedup forward ; precompute and use buffer
            latent_sample = torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32).to(
                batch["observation.robot"].device
            )
            return latent_sample, mu, log_sigma_x2

        # Prepare the input to the VAE encoder: [cls, *joint_space_configuration, *action_sequence].
        cls_embed = einops.repeat(self.vae["cls"].weight, "1 d -> b 1 d", b=batch_size)  # (B, 1, D)
        robot_state_embed = self.vae["proj"][h]["state"](batch[f"observation.{h}"])
        robot_state_embed = robot_state_embed.unsqueeze(1)  # (B, 1, D)
        action_embed = self.vae["proj"][h]["action"](batch[f"action.{h}"])  # (B, S, D)

        vae_encoder_input = [
            cls_embed,
            robot_state_embed,
            action_embed,
        ]  # (B, S+2, D)
        vae_encoder_input = torch.cat(vae_encoder_input, axis=1)

        # Prepare fixed positional embedding.
        # Note: detach() shouldn't be necessary but leaving it the same as the original code just in case.
        pos_embed = self.vae_encoder_pos_enc.clone().detach()  # (1, S+2, D)

        # Prepare key padding mask for the transformer encoder. We have 1 or 2 extra tokens at the start of the
        # sequence depending whether we use the input states or not (cls and robot state)
        # False means not a padding token.
        cls_joint_is_pad = torch.full(
            (batch_size, 2),
            False,
            device=batch[f"observation.{h}"].device,
        )
        key_padding_mask = torch.cat([cls_joint_is_pad, batch["action_is_pad"]], axis=1)  # (bs, seq+1 or 2)

        # Forward pass through VAE encoder to get the latent PDF parameters.
        # select the class token, with shape (B, D)
        cls_token_out = self.vae["encoder"](
            vae_encoder_input.permute(1, 0, 2),
            pos_embed=pos_embed.permute(1, 0, 2),
            key_padding_mask=key_padding_mask,
        )[0]
        latent_pdf_params = self.vae["out"](cls_token_out)
        mu = latent_pdf_params[:, : self.config.latent_dim]
        # This is 2log(sigma). Done this way to match the original implementation.
        log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim :]

        # Sample the latent with the reparameterization trick.
        latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        return latent_sample, mu, log_sigma_x2

    def forward(
        self,
        batch: dict[str, Tensor],
        heads: list[str],
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        """A forward pass through the Action Chunking Transformer (with optional VAE encoder).

        `batch` should have the following structure:
        {
            [robot_state_feature] (optional): (B, state_dim) batch of robot states.

            [image_features]: (B, n_cameras, C, H, W) batch of images.
                AND/OR
            [env_state_feature]: (B, env_dim) batch of environment states.

            [action_feature] (optional, only if training with VAE): (B, chunk_size, action dim) batch of actions.
        }

        Returns:
            (B, chunk_size, action_dim) batch of action sequences
            Tuple containing the latent PDF's parameters (mean, log(σ²)) both as (B, L) tensors where L is the
            latent dimension.
        """

        has_act = "action" in batch or any(["action" in x for x in batch])
        if self.config.use_vae and self.training:
            msg = "actions must be provided when using the variational objective in training mode."
            assert has_act, msg

        batch_size = batch["observation.images"][0].shape[0]

        # 1.
        # maybe get style z from VAE
        #

        assert len(heads) == 2
        _h = [x for x in heads if "shared" != x][0]
        use_vae = self.config.use_vae and has_act
        latent_sample, mu, log_sigma_x2 = self.compute_style(batch, use_vae, _h)

        # 2.
        # prepare other tokens for encoder-decoder
        #

        # Prepare transformer encoder inputs.
        encoder_in_tok = [self.encoder_latent_input_proj(latent_sample)]
        encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))

        # Robot state token.
        for h in heads:
            encoder_in_tok.append(self.proj["in"][h](batch[f"observation.{h}"]))

        # Camera observation features and positional embeddings.
        if self.config.image_features:
            all_cam_features = []
            all_cam_pos_embeds = []

            # For a list of images, the H and W may vary but H*W is constant.
            for img in batch["observation.images"]:
                cam_features = self.backbone(img)["feature_map"]
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                cam_features = self.encoder_img_feat_input_proj(cam_features)

                # Rearrange features to (sequence, batch, dim).
                cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")

                all_cam_features.append(cam_features)
                all_cam_pos_embeds.append(cam_pos_embed)

            encoder_in_tok.extend(torch.cat(all_cam_features, axis=0))
            encoder_in_pos_embed.extend(torch.cat(all_cam_pos_embeds, axis=0))

        # Stack all tokens along the sequence dimension.
        encoder_in_tok = torch.stack(encoder_in_tok, axis=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)

        # 3.
        # run encoder-decoder
        #

        # Forward pass through the transformer modules.
        encoder_out = self.encoder(encoder_in_tok, pos_embed=encoder_in_pos_embed)
        # TODO(rcadene, alexander-soare): remove call to `device` ; precompute and use buffer
        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_in_pos_embed.dtype,
            device=encoder_in_pos_embed.device,
        )
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )

        # Move back to (B, S, C).
        decoder_out = decoder_out.transpose(0, 1)

        # 4.
        # project to action space with heads
        #

        act = {
            "robot": self.proj["out"]["robot"](decoder_out),
            "human": self.proj["out"]["human"](decoder_out),
            "shared": self.proj["out"]["shared"](decoder_out),
        }
        out = {
            "action": act,
            "mu": mu,
            "log_sigma_x2": log_sigma_x2,
        }

        return out, (mu, log_sigma_x2)


class BELAPolicy(ACTPolicy):
    """
    Action Chunking Transformer Policy as per Learning Fine-Grained Bimanual Manipulation with Low-Cost
    Hardware (paper: https://arxiv.org/abs/2304.13705, code: https://github.com/tonyzhaozh/act)
    """

    config_class = ACTConfig
    name = "act"

    def __init__(
        self,
        config: ACTConfig,
        headspec: dict,
        dataset_stats: dict[str, dict[str, Tensor]] | None,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.headspec = headspec

        self.norms = {}
        for head, _stat in dataset_stats.items():
            print(head)
            in_feat, out_feat = config.input_features, config.output_features
            norm, stat = config.normalization_mapping, _stat.stats

            # <robot/human> norm can only norm for <robot/human> batch
            in_feat = {k: v for k, v in in_feat.items() if k in _stat.stats}
            out_feat = {k: v for k, v in out_feat.items() if k in _stat.stats}

            self.norms[head] = {
                "in": Normalize(in_feat, norm, stat),
                "out": Normalize(out_feat, norm, stat),
                "unnorm": Unnormalize(out_feat, norm, stat),
            }
        self.norms = recursive_moduledict(self.norms)

        # self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        # self.normalize_targets = Normalize(config.output_features, config.normalization_mapping, dataset_stats)
        # self.unnormalize_outputs = Unnormalize(config.output_features, config.normalization_mapping, dataset_stats)

        self.model = BELA(config, headspec)

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)

        self.reset()

    def normalize_inputs(self, batch: dict[str, Tensor], head) -> dict[str, Tensor]:
        """Normalize the inputs to the model using the dataset statistics."""
        assert head in ["robot", "human"]
        batch = self.norms[head]["in"](batch)
        return batch

    def normalize_targets(self, batch: dict[str, Tensor], head) -> dict[str, Tensor]:
        """Normalize the targets using the dataset statistics."""
        assert head in ["robot", "human"]
        batch = self.norms[head]["out"](batch)
        return batch

    def unnormalize_outputs(self, batch: dict[str, Tensor], head) -> dict[str, Tensor]:
        """Unnormalize the outputs using the dataset statistics."""
        assert head in ["robot", "human"]
        batch = self.norms[head]["unnorm"](batch)
        return batch

    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        self.eval()

        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch["observation.images"] = [batch[key] for key in self.config.image_features]

        # If we are doing temporal ensembling, do online updates where we keep track of the number of actions
        # we are ensembling over.
        if self.config.temporal_ensemble_coeff is not None:
            actions = self.model(batch)[0]  # (batch_size, chunk_size, action_dim)
            actions = self.unnormalize_outputs({"action": actions})["action"]
            action = self.temporal_ensembler.update(actions)
            return action

        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._action_queue) == 0:
            actions = self.model(batch)[0][:, : self.config.n_action_steps]

            # TODO(rcadene): make _forward return output dictionary?
            actions = self.unnormalize_outputs({"action": actions})["action"]

            # `self.model.forward` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    def forward(self, batch: dict[str, Tensor], heads=None) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training or validation."""

        if heads is None:
            assert (heads := batch.get("heads")), "heads must be provided"
        basehead = [h for h in heads if h != "shared"][0]

        batch = self.normalize_inputs(batch, basehead)
        if self.config.image_features:
            # shallow copy so that adding a key doesn't modify the original
            batch = dict(batch)
            imgs = [batch.get(key, None) for key in self.config.image_features]
            batch["observation.images"] = [x for x in imgs if x is not None]
        batch = self.normalize_targets(batch, basehead)

        for h in heads:
            key = f"action.{h}"
            item = [batch[k] for k in batch if key in k]
            batch[key] = torch.cat(item, dim=-1)
        for h in heads:
            key = f"observation.{h}"
            item = [batch[k] for k in batch if (key in k and "image" not in k)]
            batch[key] = torch.cat(item, dim=-1)

        out, (mu_hat, log_sigma_x2_hat) = self.model(batch, heads)

        ah = {k: v for k, v in flatten_dict(out, sep=".").items()}

        losses = {}
        for k, ahv in ah.items():
            if any([h in k for h in heads]):
                losses[k] = (F.l1_loss(batch[k], ahv, reduction="none") * ~batch["action_is_pad"].unsqueeze(-1)).mean()

        losses["l1"] = torch.Tensor(list(losses.values())).mean()
        if self.config.use_vae:
            # Calculate Dₖₗ(latent_pdf || standard_normal). Note: After computing the KL-divergence for
            # each dimension independently, we sum over the latent dimension to get the total
            # KL-divergence per batch element, then take the mean over the batch.
            # (See App. B of https://arxiv.org/abs/1312.6114 for more details).
            mean_kld = (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - (log_sigma_x2_hat).exp())).sum(-1).mean()
            losses["kld"] = mean_kld  # .item()
            loss = losses["l1"] + mean_kld * self.config.kl_weight
        else:
            loss = losses["l1"]

        losses = {k: v.item() for k, v in losses.items()}
        return loss, losses  # , out
