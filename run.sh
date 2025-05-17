# uv run -- torchrun --standalone --nnodes=1 --nproc-per-node=4  
uv run \
    bela/scripts/train.py \
    --dataset.repo_id=mhyatt000/duck --dataset.revision v2.1 --batch_size=1024 \
    --steps 1000000000 \
    --wandb.enable true  --wandb.project suite --eval_freq -1  --eval.batch_size=50 \
    --dataset.image_transforms.enable true \
    --policy:bela-config \
    --policy.type=act --policy.chunk_size 50 --policy.n_action_steps 1 --policy.temporal_ensemble_coeff 0.01 \
