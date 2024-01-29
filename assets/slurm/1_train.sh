#!/bin/bash
#SBATCH -J deflow
#SBATCH --gpus 8 -C "thin"
#SBATCH -t 3-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=qingwen@kth.se
#SBATCH --output /proj/berzelius-2023-154/users/x_qinzh/deflow/logs/slurm/%J_deflow.out
#SBATCH --error  /proj/berzelius-2023-154/users/x_qinzh/deflow/logs/slurm/%J_deflow.err

cd /proj/berzelius-2023-154/users/x_qinzh/deflow

SOURCE="/proj/berzelius-2023-154/users/x_qinzh/av2/deflow_preprocess"
DEST="/scratch/local/av2"
SUBDIRS=("sensor/train" "sensor/val")

start_time=$(date +%s)
for dir in "${SUBDIRS[@]}"; do
    mkdir -p "${DEST}/${dir}"
    find "${SOURCE}/${dir}" -type f -print0 | xargs -0 -n1 -P16 cp -t "${DEST}/${dir}" &
done
wait
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Copy ${SOURCE} to ${DEST} Total time: ${elapsed} seconds"
echo "Start training..."

# ====> leaderboard model = [fastflow3d, deflow]
# /proj/berzelius-2023-154/users/x_qinzh/mambaforge/envs/deflow/bin/python 1_train.py \
#     slurm_id=$SLURM_JOB_ID wandb_mode=online dataset_path=/scratch/local/av2/sensor \
#     num_workers=16 model=deflow lr=2e-6 epochs=50 batch_size=10 loss_fn=deflowLoss

# /proj/berzelius-2023-154/users/x_qinzh/mambaforge/envs/deflow/bin/python 1_train.py \
#     slurm_id=$SLURM_JOB_ID wandb_mode=online dataset_path=/scratch/local/av2/sensor \
#     num_workers=16 model=fastflow3d lr=2e-6 epochs=50 batch_size=16 loss_fn=ff3dLoss




# ===> ablation A: iteration num [2, 4 (R), 8, 16]
# /proj/berzelius-2023-154/users/x_qinzh/mambaforge/envs/deflow/bin/python 1_train.py \
#     slurm_id=$SLURM_JOB_ID wandb_mode=online dataset_path=/scratch/local/av2/sensor \
#     num_workers=16 model=deflow lr=2e-6 epochs=50 batch_size=10 loss_fn=deflowLoss "model.target.num_iters=2"

# /proj/berzelius-2023-154/users/x_qinzh/mambaforge/envs/deflow/bin/python 1_train.py \
#     slurm_id=$SLURM_JOB_ID wandb_mode=online dataset_path=/scratch/local/av2/sensor \
#     num_workers=16 model=deflow lr=2e-6 epochs=50 batch_size=8 loss_fn=deflowLoss "model.target.num_iters=8"

# /proj/berzelius-2023-154/users/x_qinzh/mambaforge/envs/deflow/bin/python 1_train.py \
#     slurm_id=$SLURM_JOB_ID wandb_mode=online dataset_path=/scratch/local/av2/sensor \
#     num_workers=16 model=deflow lr=2e-6 epochs=50 batch_size=10 loss_fn=deflowLoss "model.target.num_iters=16"


# ===> ablation B: loss_fn --- loss_fn = [ff3dLoss (R), zeroflowLoss, deflowLoss]
# /proj/berzelius-2023-154/users/x_qinzh/mambaforge/envs/deflow/bin/python 1_train.py \
#     slurm_id=$SLURM_JOB_ID wandb_mode=online dataset_path=/scratch/local/av2/sensor \
#     num_workers=16 model=fastflow3d lr=2e-6 epochs=50 batch_size=16 loss_fn=zeroflowLoss

# /proj/berzelius-2023-154/users/x_qinzh/mambaforge/envs/deflow/bin/python 1_train.py \
#     slurm_id=$SLURM_JOB_ID wandb_mode=online dataset_path=/scratch/local/av2/sensor \
#     num_workers=16 model=fastflow3d lr=2e-6 epochs=50 batch_size=16 loss_fn=deflowLoss


# ===> ablation C: decoder --- model.target.decoder_option = [linear, gru] and fastflow3d resolution [0.1, 0.2 (R), 0.4]
# /proj/berzelius-2023-154/users/x_qinzh/mambaforge/envs/deflow/bin/python 1_train.py \
#     slurm_id=$SLURM_JOB_ID wandb_mode=online dataset_path=/scratch/local/av2/sensor \
#     num_workers=16 model=deflow lr=2e-6 epochs=50 batch_size=10 loss_fn=ff3dLoss "model.target.decoder_option=linear"

# /proj/berzelius-2023-154/users/x_qinzh/mambaforge/envs/deflow/bin/python 1_train.py \
#     slurm_id=$SLURM_JOB_ID wandb_mode=online dataset_path=/scratch/local/av2/sensor \
#     num_workers=16 model=deflow lr=2e-6 epochs=50 batch_size=10 loss_fn=ff3dLoss "model.target.decoder_option=gru"

# /proj/berzelius-2023-154/users/x_qinzh/mambaforge/envs/deflow/bin/python 1_train.py \
#     slurm_id=$SLURM_JOB_ID wandb_mode=online dataset_path=/scratch/local/av2/sensor \
#     num_workers=16 model=fastflow3d lr=2e-6 epochs=50 batch_size=10 loss_fn=ff3dLoss "voxel_size=[0.1, 0.1, 6]"

# /proj/berzelius-2023-154/users/x_qinzh/mambaforge/envs/deflow/bin/python 1_train.py \
#     slurm_id=$SLURM_JOB_ID wandb_mode=online dataset_path=/scratch/local/av2/sensor \
#     num_workers=16 model=fastflow3d lr=2e-6 epochs=50 batch_size=16 loss_fn=ff3dLoss "voxel_size=[0.4, 0.4, 6]"