#!/bin/bash
#SBATCH -J pack_data
#SBATCH --gpus 0
#SBATCH --cpus-per-task 64
#SBATCH --mem 512G
#SBATCH --mincpus=64
#SBATCH -t 3-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=qingwen@kth.se
#SBATCH --output /proj/berzelius-2023-154/users/x_qinzh/deflow/logs/slurm/%J_data.out
#SBATCH --error  /proj/berzelius-2023-154/users/x_qinzh/deflow/logs/slurm/%J_data.err

cd /proj/berzelius-2023-154/users/x_qinzh/deflow
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/proj/berzelius-2023-154/users/x_qinzh/mambaforge/lib
export HYDRA_FULL_ERROR=1

/proj/berzelius-2023-154/users/x_qinzh/mambaforge/envs/deflow/bin/python 0_preprocess.py \
    --av2_type sensor \
    --data_mode train \
    --argo_dir /proj/berzelius-2023-154/users/x_qinzh/av2 \
    --output_dir /proj/berzelius-2023-154/users/x_qinzh/av2/deflow_preprocess

/proj/berzelius-2023-154/users/x_qinzh/mambaforge/envs/deflow/bin/python 0_preprocess.py \
    --av2_type sensor \
    --data_mode val \
    --argo_dir /proj/berzelius-2023-154/users/x_qinzh/av2 \
    --output_dir /proj/berzelius-2023-154/users/x_qinzh/av2/deflow_preprocess \
    --mask_dir /proj/berzelius-2023-154/users/x_qinzh/av2/3d_scene_flow

/proj/berzelius-2023-154/users/x_qinzh/mambaforge/envs/deflow/bin/python 0_preprocess.py \
    --av2_type sensor \
    --data_mode test \
    --argo_dir /proj/berzelius-2023-154/users/x_qinzh/av2 \
    --output_dir /proj/berzelius-2023-154/users/x_qinzh/av2/deflow_preprocess \
    --mask_dir /proj/berzelius-2023-154/users/x_qinzh/av2/3d_scene_flow