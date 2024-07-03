DeFlow: Decoder of Scene Flow Network in Autonomous Driving
---

[![arXiv](https://img.shields.io/badge/arXiv-2401.16122-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2401.16122) 
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/deflow-decoder-of-scene-flow-network-in/scene-flow-estimation-on-argoverse-2)](https://paperswithcode.com/sota/scene-flow-estimation-on-argoverse-2?p=deflow-decoder-of-scene-flow-network-in) 
[![poster](https://img.shields.io/badge/ICRA24|Poster-6495ed?style=flat&logo=Shotcut&logoColor=wihte)](https://hkustconnect-my.sharepoint.com/:b:/g/personal/qzhangcb_connect_ust_hk/EXP_uXYmm_tItTWc8MafXHoB-1dVrMnvF1-lCzU1PXAvqQ?e=2FPfBS) 
[![video](https://img.shields.io/badge/video-YouTube-FF0000?logo=youtube&logoColor=white)](https://youtu.be/bZ4uUv0nDa0)
[![blog](https://img.shields.io/badge/Blog%7C%E7%9F%A5%E4%B9%8E%E4%B8%AD%E6%96%87-1772f6?style=flat&logo=Shotcut)](https://zhuanlan.zhihu.com/p/706514747) 

Task: Scene Flow Estimation in Autonomous Driving. 

🔥 2024/07/02: Check the self-supervised version in our new ECCV'24 [SeFlow](https://github.com/KTH-RPL/SeFlow). The 1st ranking in new leaderboard among self-supervise methods.

Pre-trained weights for models are available in [Zenodo](https://zenodo.org/records/12632962) or [Onedrive link](https://hkustconnect-my.sharepoint.com/:f:/g/personal/qzhangcb_connect_ust_hk/Et85xv7IGMRKgqrVeJEVkMoB_vxlcXk6OZUyiPjd4AArIg?e=lqRGhx). 
Check usage in [2. Evaluation](#2-evaluation) or [3. Visualization](#3-visualization). 

**Scripts** quick view in our scripts:

- `dataprocess/extract_*.py` : pre-process data before training to speed up the whole training time. 
  [Dataset we included now: Argoverse 2 and Waymo, more on the way: Nuscenes, custom data.]
  
- `1_train.py`: Train the model and get model checkpoints. Pls remember to check the config.

- `2_eval.py` : Evaluate the model on the validation/test set. And also upload to online leaderboard.

- `3_vis.py` : For visualization of the results with a video.

## 0. Setup

**Environment**: Clone the repo and build the environment, check [detail installation](assets/README.md) for more information. [Conda](https://docs.conda.io/projects/miniconda/en/latest/)/[Mamba](https://github.com/mamba-org/mamba) is recommended.
```
git clone --recursive https://github.com/KTH-RPL/DeFlow.git
cd DeFlow
mamba env create -f environment.yaml
```

mmcv:
```bash
mamba activate deflow
cd ~/DeFlow/mmcv && export MMCV_WITH_OPS=1 && export FORCE_CUDA=1 && pip install -e .
```

Or another environment setup choice is [Docker](https://en.wikipedia.org/wiki/Docker_(software)) which isolated environment, you can pull it by. 
If you have different arch, please build it by yourself `cd DeFlow && docker build -t zhangkin/deflow` by going through [build-docker-image](assets/README.md/#build-docker-image) section.
```bash
# option 1: pull from docker hub
docker pull zhangkin/deflow

# run container
docker run -it --gpus all -v /dev/shm:/dev/shm -v /home/kin/data:/home/kin/data --name deflow zhangkin/deflow /bin/zsh
```

## 1. Train

Download tips in [dataprocess/README.md](dataprocess/README.md#argoverse-20)

### Prepare Data

Normally need 10-45 mins finished run following commands totally (my computer 15 mins, our cluster 40 mins).
```bash
python dataprocess/extract_av2.py --av2_type sensor --data_mode train --argo_dir /home/kin/data/av2 --output_dir /home/kin/data/av2/preprocess
python dataprocess/extract_av2.py --av2_type sensor --data_mode val --mask_dir /home/kin/data/av2/3d_scene_flow
python dataprocess/extract_av2.py --av2_type sensor --data_mode test --mask_dir /home/kin/data/av2/3d_scene_flow
```

### Train Model

All local benchmarking methods and ablation studies can be done through command with different config, check [`assets/slurm`](assets/slurm) for all the commands we used in our experiments.

Best fine-tuned model train with following command by other default config in [conf/config.yaml](conf/config.yaml) and [conf/model/deflow.yaml](conf/model/deflow.yaml), if you will set wandb_mode=online, maybe change all `entity="kth-rpl"` to your own account name.
```bash
python 1_train.py model=deflow lr=2e-6 epochs=50 batch_size=16
```

Benchmarking and baseline methods:
```bash
python 1_train.py model=fastflow3d lr=2e-6 epochs=50 batch_size=16
python 1_train.py model=deflow lr=2e-6 epochs=50 batch_size=16
```

To help community benchmarking, we provide our weights including fastflow3d, deflow [Zendo](https://zenodo.org/records/12632962). 
These checkpoints also include parameters and status of that epoch inside it. If you are interested in weights of ablation studies, please contact us.
Note: Please use these weights by following the term of use of the trained dataset (since weights are trained on these datasets) as [Argoverse 2 Term of Use](https://www.argoverse.org/about.html) mentioned: Using it under Non-Commercially (CC BY-NC-SA 4.0).

## 2. Evaluation

You can view Wandb dashboard for the training and evaluation results or [run/submit to av2 leaderboard to get official results](assets/README.md#leaderboard-submission).



Since in training, we save all hyper-parameters and model checkpoints, the only thing you need to do is to specify the checkpoint path. Remember to set the data path correctly also.
```bash
# downloaded pre-trained weight, or train by yourself
wget https://zenodo.org/records/12632962/files/deflow_best.ckpt

python 2_eval.py checkpoint=/home/kin/deflow_best.ckpt av2_mode=val # it will directly prints all metric
# it will output the av2_submit.zip or av2_submit_v2.zip for you to submit to leaderboard
python 2_eval.py checkpoint=/home/kin/deflow_best.ckpt av2_mode=test leaderboard_version=1
python 2_eval.py checkpoint=/home/kin/deflow_best.ckpt av2_mode=test leaderboard_version=2
```

Check all detailed result files (presented in our paper Table 1) in [this discussion](https://github.com/KTH-RPL/DeFlow/discussions/2).

To submit to the Online Leaderboard, if you select `av2_mode=test`, it should be a zip file for you to submit to the leaderboard.
Note: The leaderboard result in DeFlow main paper is [version 1](https://eval.ai/web/challenges/challenge-page/2010/evaluation), as [version 2](https://eval.ai/web/challenges/challenge-page/2210/overview) is updated after DeFlow paper.

```bash
# since the env may conflict we set new on deflow, we directly create new one:
mamba create -n py37 python=3.7
mamba activate py37
pip install "evalai"

# Step 2: login in eval and register your team
evalai set-token <your token>

# Step 3: Submit to leaderboard
evalai challenge 2010 phase 4018 submit --file av2_submit.zip --large --private
evalai challenge 2210 phase 4396 submit --file av2_submit_v2.zip --large --private
```

## 3. Visualization

We provide a script to visualize the results of the model. You can specify the checkpoint path and the data path to visualize the results. The step is quickly similar to evaluation.

```bash
# downloaded pre-trained weight, or train by yourself
wget https://zenodo.org/records/12632962/files/deflow_best.ckpt

python 3_vis.py checkpoint=/home/kin/deflow_best.ckpt dataset_path=/home/kin/data/av2/preprocess/sensor/vis

# Then terminal will tell you the command you need run. For example here is the output of the above:
Model: DeFlow, Checkpoint from: /home/kin/deflow_best.ckpt
We already write the estimate flow: deflow_best into the dataset, please run following commend to visualize the flow. Copy and paste it to your terminal:
python tests/scene_flow.py --flow_mode 'deflow_best' --data_dir /home/kin/data/av2/preprocess/sensor/mini
Enjoy! ^v^ ------ 

# Then run the command in the terminal:
python tests/scene_flow.py --flow_mode 'deflow_best' --data_dir /home/kin/data/av2/preprocess/sensor/mini
```

Note: ego_motion already compensated, so the visualization is more clear.

<!-- ![](assets/docs/vis_res.png) -->


https://github.com/KTH-RPL/DeFlow/assets/35365764/9b265d56-06a9-4300-899c-96047a0da505



## Cite & Acknowledgements

```
@article{zhang2024deflow,
  author={Zhang, Qingwen and Yang, Yi and Fang, Heng and Geng, Ruoyu and Jensfelt, Patric},
  title={DeFlow: Decoder of Scene Flow Network in Autonomous Driving},
  journal={arXiv preprint arXiv:2401.16122},
  year={2024}
}
@article{zhang2024seflow,
  author={Zhang, Qingwen and Yang, Yi and Li, Peizheng and Andersson, Olov and Jensfelt, Patric},
  title={SeFlow: A Self-Supervised Scene Flow Method in Autonomous Driving},
  journal={arXiv preprint arXiv:2407.01702},
  year={2024}
}
```

This implementation is based on codes from several repositories. Thanks to these authors who kindly open-sourcing their work to the community. Please see our paper reference part to get more information. 
Thanks to [Kyle Vedder (ZeroFlow)](https://github.com/kylevedder) who kindly discussed their results with us and HKUST Ramlab's member: Jin Wu who gave constructive comments on this work. 
The computations were enabled by the supercomputing resource Berzelius provided by National Supercomputer Centre at Linköping University and the Knut and Alice Wallenberg Foundation, Sweden.

❤️: [ZeroFlow](https://github.com/kylevedder/zeroflow), [NSFP](https://github.com/Lilac-Lee/Neural_Scene_Flow_Prior), [FastNSF](https://github.com/Lilac-Lee/FastNSF). Others good code style and tools: [forecast-mae](https://github.com/jchengai/forecast-mae), [kiss-icp](https://github.com/PRBonn/kiss-icp)
