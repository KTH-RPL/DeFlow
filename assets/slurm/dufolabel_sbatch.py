"""
# Created: 2023-11-30 17:02
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)
#
# 
# Description: Write sbatch files for DUFO jobs on cluster (SLURM), no GPU needed
# Reference: 
#   * ZeroFlow data sbatch: https://github.com/kylevedder/zeroflow/blob/master/data_prep_scripts/split_nsfp_jobs_sbatch.py

# Run with following commands (only train need to be processed with dufo label)
- python assets/slurm/dufolabel_sbatch.py --split 50 --total 700 --interval 1 --data_dir /home/kin/data/av2/preprocess/sensor/train --data_mode train
- python assets/slurm/dufolabel_sbatch.py --split 100 --total 800  --interval 2 --data_dir /proj/berzelius-2023-154/users/x_qinzh/dataset/waymo/fix_preprocess/train
"""

import fire, time, os
def main(
    data_dir: str = "/proj/berzelius-2023-154/users/x_qinzh/av2/preprocess/sensor/train",
    split: int = 50,
    total: int = 2001,
    interval: int = 1,
):
    # +1 因为range是左闭右开
    for i in range(0, total+1, split):
        scene_range = [i , min(i + split, total+1)]
        print(scene_range)
        sbatch_file_content = \
        f"""#!/bin/bash
#SBATCH -J pack_{scene_range[0]}_{scene_range[1]}
#SBATCH --gpus 0
#SBATCH --cpus-per-task 32
#SBATCH --mem 64G
#SBATCH --mincpus=32
#SBATCH -t 1-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=qingwen@kth.se
#SBATCH --output /proj/berzelius-2023-154/users/x_qinzh/seflow/logs/slurm/0_lidar/%J_{scene_range[0]}_{scene_range[1]}.out
#SBATCH --error  /proj/berzelius-2023-154/users/x_qinzh/seflow/logs/slurm/0_lidar/%J_{scene_range[0]}_{scene_range[1]}.err

cd /proj/berzelius-2023-154/users/x_qinzh/seflow
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/proj/berzelius-2023-154/users/x_qinzh/mambaforge/lib

/proj/berzelius-2023-154/users/x_qinzh/mambaforge/envs/seflow/bin/python process.py \\
    --data_dir {data_dir} \\
    --interval {interval} \\
    --scene_range {scene_range[0]},{scene_range[1]}

        """
        # run command sh sbatch_file_content
        with open(f"tmp_sbatch.sh", "w") as f:
            f.write(sbatch_file_content)
        print(f"Write sbatch file: tmp_sbatch.sh")
        os.system(f"sbatch tmp_sbatch.sh")

if __name__ == '__main__':
    start_time = time.time()
    fire.Fire(main)
    print(f"\nTime used: {(time.time() - start_time)/60:.2f} mins")