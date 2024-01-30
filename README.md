DeFlow 
---

[![arXiv](https://img.shields.io/badge/arXiv-2401.16122-b31b1b.svg)](https://arxiv.org/abs/2401.16122) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/deflow-decoder-of-scene-flow-network-in/scene-flow-estimation-on-argoverse-2)](https://paperswithcode.com/sota/scene-flow-estimation-on-argoverse-2?p=deflow-decoder-of-scene-flow-network-in) [poster coming soon] [video coming soon]

Will present in ICRA'24.

Task: Scene Flow Estimation in Autonomous Driving. Pre-trained weights for models are available in [Onedrive link](https://hkustconnect-my.sharepoint.com/:f:/g/personal/qzhangcb_connect_ust_hk/Et85xv7IGMRKgqrVeJEVkMoB_vxlcXk6OZUyiPjd4AArIg?e=lqRGhx). Check usage in [2. Evaluation](#2-evaluation).

https://github.com/KTH-RPL/DeFlow/assets/35365764/15581af1-3066-4865-bf72-1242a478b938

**Scripts** quick view in our scripts:

- `0_preprocess.py` : pre-process data before training to speed up the whole training time.
- `1_train.py`: Train the model and get model checkpoints. Pls remember to check the config.
- `2_eval.py` : Evaluate the model on the validation/test set. And also upload to online leaderboard.
- `3_vis.py` : For visualization of the results with a video.

## 0. Setup

**Environment**: Clone the repo and build the environment, check [detail installation](assets/README.md) for more information. [Conda](https://docs.conda.io/projects/miniconda/en/latest/)/[Mamba](https://github.com/mamba-org/mamba) is recommended.
```
git clone https://github.com/KTH-RPL/DeFlow
cd DeFlow
mamba env create -f environment.yaml
```

mmcv:
```bash
mamba activate deflow
cd ~/DeFlow/mmcv && export MMCV_WITH_OPS=1 && export FORCE_CUDA=1 && pip install -e .
```

## 1. Train

Download tips in [assets/README.md](assets/README.md#dataset-download)

### Prepare Data

Normally need 10-45 mins finished run following commands totally (my computer 15 mins, our cluster 40 mins).
```bash
python 0_preprocess.py --av2_type sensor --data_mode train --argo_dir /home/kin/data/av2 --output_dir /home/kin/data/av2/preprocess
python 0_preprocess.py --av2_type sensor --data_mode val --mask_dir /home/kin/data/av2/3d_scene_flow
python 0_preprocess.py --av2_type sensor --data_mode test --mask_dir /home/kin/data/av2/3d_scene_flow
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

# for nsfp no need train but optimize iteration running
python 2_eval.py model=nsfp 
python 2_eval.py model=fast_nsfp
```

To help community benchmarking, we provide our weights including fastflow3d, deflow [Onedrive link](https://hkustconnect-my.sharepoint.com/:f:/g/personal/qzhangcb_connect_ust_hk/Et85xv7IGMRKgqrVeJEVkMoB_vxlcXk6OZUyiPjd4AArIg?e=lqRGhx). These checkpoints also include parameters and status of that epoch inside it. If you are interested in weights of ablation studies, please contact us.

## 2. Evaluation

You can view Wandb dashboard for the training and evaluation results or [run/submit to av2 leaderboard to get official results](assets/README.md#leaderboard-submission).

Since in training, we save all hyper-parameters and model checkpoints, so the only things you need to do is to specify the checkpoint path. Remember to set data path correctly also.
```bash
python 2_eval.py checkpoint=/home/kin/model.ckpt
```

Submit to Online Leaderboard, the last step will tell you the result path, copy it here:
```bash
# you will find there is a av2_submit.zip in the folder now. since the env is different and conflict we set new one:
mamba create -n py37 python=3.7
mamba activate py37
pip install "evalai"

# Step 2: login in eval and register your team
evalai set_token <your token>

# Step 3: Submit to leaderboard
evalai challenge 2010 phase 4018 submit --file av2_submit.zip --large --private
```

## 3. Visualization

We provide a script to visualize the results of the model. You can specify the checkpoint path and the data path to visualize the results. The step is quickly similar to evaluation.

```bash
python 3_vis.py checkpoint=/home/kin/model.ckpt dataset_path=/home/kin/data/av2/preprocess/sensor/vis

# Then terminal will tell you the command you need run. For example here is the output of the above:
Model: DeFlow, Checkpoint from: /logs/wandb/deflow-10078447/checkpoints/epoch_35_seflow.ckpt
We already write the flow_est into the dataset, please run following commend to visualize the flow. Copy and paste it to your terminal:
python tests/scene_flow.py --flow_mode='flow_est' --data_dir=/home/kin/data/av2/preprocess/sensor/vis
Enjoy! ^v^ ------ 


# Then run the test with changed flow_mode between estimate and gt [flow_est, flow]
python tests/scene_flow.py --flow_mode='flow_est' --data_dir=/home/kin/data/av2/preprocess/sensor/vis
```

Note: ego_motion already compensated, so the visualization is more clear.

![](assets/docs/vis_res.png)

## Cite & Acknowledgements

```
@article{zhang2024deflow,
  author={Zhang, Qingwen and Yang, Yi and Peizheng, Li and Olov, Andersson and Jensfelt, Patric},
  title={DeFlow: Decoder of Scene Flow Network in Autonomous Driving},
  journal={arXiv preprint arXiv:2401.16122},
  year={2024}
}
```

This implementation is based on codes from several repositories. Thanks for these authors who kindly open-sourcing their work to the community. Please see our paper reference part to get more information.

❤️: [ZeroFlow](https://github.com/kylevedder/zeroflow), [NSFP](https://github.com/Lilac-Lee/Neural_Scene_Flow_Prior), [FastNSF](https://github.com/Lilac-Lee/FastNSF). Others good code style and tools: [forecast-mae](https://github.com/jchengai/forecast-mae), [kiss-icp](https://github.com/PRBonn/kiss-icp)
