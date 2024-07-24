#!/bin/bash
#SBATCH -J pack_data
#SBATCH --gpus 0
#SBATCH --cpus-per-task 64
#SBATCH --mem 256G
#SBATCH --mincpus=64
#SBATCH -t 1-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=qingwen@kth.se
#SBATCH --output /proj/berzelius-2023-154/users/x_qinzh/workspace/SeFlow/logs/slurm/%J_data.out
#SBATCH --error  /proj/berzelius-2023-154/users/x_qinzh/workspace/SeFlow/logs/slurm/%J_data.err

cd /proj/berzelius-2023-154/users/x_qinzh/workspace/SeFlow
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/proj/berzelius-2023-154/users/x_qinzh/mambaforge/lib
# export HYDRA_FULL_ERROR=1

/proj/berzelius-2023-154/users/x_qinzh/mambaforge/envs/dataprocess/bin/python dataprocess/extract_av2.py --nproc 64 \
    --av2_type sensor \
    --data_mode train \
    --argo_dir /proj/berzelius-2023-154/users/x_qinzh/av2 \
    --output_dir /proj/berzelius-2023-364/users/x_qinzh/data/av2/preprocess_v2

/proj/berzelius-2023-154/users/x_qinzh/mambaforge/envs/dataprocess/bin/python dataprocess/extract_av2.py --nproc 64 \
    --av2_type sensor \
    --data_mode val \
    --argo_dir /proj/berzelius-2023-154/users/x_qinzh/av2 \
    --output_dir /proj/berzelius-2023-364/users/x_qinzh/data/av2/preprocess_v2 \
    --mask_dir /proj/berzelius-2023-154/users/x_qinzh/av2/3d_scene_flow

/proj/berzelius-2023-154/users/x_qinzh/mambaforge/envs/dataprocess/bin/python dataprocess/extract_av2.py --nproc 64 \
    --av2_type sensor \
    --data_mode test \
    --argo_dir /proj/berzelius-2023-154/users/x_qinzh/av2 \
    --output_dir /proj/berzelius-2023-364/users/x_qinzh/data/av2/preprocess_v2 \
    --mask_dir /proj/berzelius-2023-154/users/x_qinzh/av2/3d_scene_flow