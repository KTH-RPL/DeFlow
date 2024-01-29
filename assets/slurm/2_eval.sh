#!/bin/bash
#SBATCH -J eval
#SBATCH --gpus 1
#SBATCH -t 3-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=qingwen@kth.se
#SBATCH --output /proj/berzelius-2023-154/users/x_qinzh/deflow/logs/slurm/%J_eval.out
#SBATCH --error  /proj/berzelius-2023-154/users/x_qinzh/deflow/logs/slurm/%J_eval.err

cd /proj/berzelius-2023-154/users/x_qinzh/deflow

SOURCE="/proj/berzelius-2023-154/users/x_qinzh/av2/deflow_preprocess"
DEST="/scratch/local/av2"
SUBDIRS=("sensor/val")

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
# /proj/berzelius-2023-154/users/x_qinzh/mambaforge/envs/deflow/bin/python 2_eval.py \
#     wandb_mode=online dataset_path=/scratch/local/av2/sensor \
#     av2_mode=test save_res=True


/proj/berzelius-2023-154/users/x_qinzh/mambaforge/envs/deflow/bin/python 2_eval.py \
    wandb_mode=online dataset_path=/scratch/local/av2/sensor av2_mode=val \
    checkpoint=/proj/berzelius-2023-154/users/x_qinzh/deflow/logs/wandb/fastflow3d-10086990/checkpoints/epoch_49_fastflow3d.ckpt

/proj/berzelius-2023-154/users/x_qinzh/mambaforge/envs/deflow/bin/python 2_eval.py \
    wandb_mode=online dataset_path=/scratch/local/av2/sensor av2_mode=val \
    checkpoint=/proj/berzelius-2023-154/users/x_qinzh/deflow/logs/wandb/fastflow3d-10088873/checkpoints/epoch_49_fastflow3d.ckpt

/proj/berzelius-2023-154/users/x_qinzh/mambaforge/envs/deflow/bin/python 2_eval.py \
    wandb_mode=online dataset_path=/scratch/local/av2/sensor av2_mode=val \
    checkpoint=/proj/berzelius-2023-154/users/x_qinzh/deflow/logs/wandb/fastflow3d-10088874/checkpoints/epoch_49_fastflow3d.ckpt