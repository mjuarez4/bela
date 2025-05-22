# uv run -- torchrun --standalone --nnodes=1 --nproc-per-node=4
uv run \
    bela/scripts/train.py \
    --dataset.repo_id=mhyatt000/duck --dataset.revision v2.1 --batch_size=64 \
    --dataset.image_transforms.enable \
    --human_repos mhyatt000/stack_mano --robot_repos mhyatt000/stack405 \
    --human_revisions v2.1 --robot_revisions v2.1 \
    --steps 1000000000 \
    --wandb.project suite --eval_freq -1  --eval.batch_size=50 \
    --policy.chunk_size 50 --policy.n_action_steps 1 --policy.temporal_ensemble_coeff 0.01 \
    $@
    # --wandb.enable \
    # --policy:bela-config \
    # --policy.type=act  # no need since ACT is only option
