#!/bin/bash
#SBATCH -J eval
#SBATCH --gpus 1
#SBATCH -t 01:00:00
#SBATCH --output /proj/berzelius-2023-154/users/x_qinzh/seflow/logs/slurm/%J_eval.out
#SBATCH --error  /proj/berzelius-2023-154/users/x_qinzh/seflow/logs/slurm/%J_eval.err

cd /proj/berzelius-2023-154/users/x_qinzh/seflow

SOURCE="/proj/berzelius-2023-154/users/x_qinzh/av2/preprocess_v2"
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

# ====> leaderboard model
# /proj/berzelius-2023-154/users/x_qinzh/mambaforge/envs/seflow/bin/python eval.py \
#     wandb_mode=online dataset_path=/scratch/local/av2/sensor \
#     checkpoint=/proj/berzelius-2023-154/users/x_qinzh/seflow/logs/wandb/seflow-10086990/checkpoints/epoch_19_seflow.ckpt \
#     av2_mode=test save_res=True

/proj/berzelius-2023-154/users/x_qinzh/mambaforge/envs/seflow/bin/python eval.py \
    wandb_mode=online dataset_path=/scratch/local/av2/sensor av2_mode=val \
    checkpoint=/proj/berzelius-2023-154/users/x_qinzh/seflow/logs/wandb/seflow-10086990/checkpoints/epoch_19_seflow.ckpt