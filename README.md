SeFlow: A Self-Supervised Scene Flow Method in Autonomous Driving
---

[![arXiv](https://img.shields.io/badge/arXiv-2407.01702-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2407.01702)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/seflow-a-self-supervised-scene-flow-method-in/self-supervised-scene-flow-estimation-on-1)](https://paperswithcode.com/sota/self-supervised-scene-flow-estimation-on-1?p=seflow-a-self-supervised-scene-flow-method-in)
[poster comming soon]
[video coming soon]

2024/07/07 13:45: I'm working on updating code here now. **Not fully ready yet** until Jul'15.

Pre-trained weights for models are available in [Zenodo](https://zenodo.org/records/12632962) link. Check usage in [2. Evaluation](#2-evaluation) or [3. Visualization](#3-visualization).

Task: __Self-Supervised__ Scene Flow Estimation in Autonomous Driving. 

We directly follow our previous work [code structure](https://github.com/KTH-RPL/DeFlow), so you may want to start from the easier one with supervised learning first: Try our [DeFlow](https://github.com/KTH-RPL/DeFlow). Then you will find this is simple to you (things about how to train under self-supervised). Here are **Scripts** quick view in this repo:

- `dataprocess/extract_*.py` : pre-process data before training to speed up the whole training time. 
  [Dataset we included now: Argoverse 2 and Waymo.  more on the way: Nuscenes, custom data.]
  
- `0_process.py`: process data with save dufomap, cluster labels inside file. Only needed for training.

- `1_train.py`: Train the model and get model checkpoints. Pls remember to check the config.

- `2_eval.py` : Evaluate the model on the validation/test set. And also upload to online leaderboard.

- `3_vis.py` : For visualization of the results with a video.


## 0. Setup

**Environment**: Same to [DeFlow](https://github.com/KTH-RPL/DeFlow). And even lighter here with extracting mmcv module we needed into cuda assets.

```bash
git clone --recursive https://github.com/KTH-RPL/SeFlow.git
cd SeFlow && mamba env create -f environment.yaml
```

CUDA package (need install nvcc compiler), the compile time is around 1-5 minutes:
```bash
mamba activate seflow
# change it if you use different cuda version (I tested 11.3, 11.4, 11.7 all works)
export PATH=/usr/local/cuda-11.7/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH

cd assets/cuda/mmcv && python ./setup.py install && cd ../../..
cd assets/cuda/chamfer3D && python ./setup.py install && cd ../../..
```

Or you always can choose [Docker](https://en.wikipedia.org/wiki/Docker_(software)) which isolated environment and free yourself from installation, you can pull it by. 
If you have different arch, please build it by yourself `cd SeFlow && docker build -t zhangkin/seflow` by going through [build-docker-image](https://github.com/KTH-RPL/DeFlow/blob/main/assets/README.md/#build-docker-image) section.
```bash
# option 1: pull from docker hub, todo test and update
# docker pull zhangkin/seflow

# run container
docker run -it --gpus all -v /dev/shm:/dev/shm -v /home/kin/data:/home/kin/data --name seflow zhangkin/seflow /bin/zsh
```

## 1. Run & Train

Note: Prepare raw data and process train data only needed run once for the task. No need to run till you delete all data.

### Data Preparation

Check [dataprocess/README.md](dataprocess/README.md#argoverse-20) for downloading tips for the raw Argoverse 2 dataset

Maybe you only want to have the mini processed dataset to try the code quickly, We directly provide one scene inside `train` and `val`. It already converted to `.h5` format and processed with the label data. 
<!-- You can download it from [Zenodo](https://zenodo.org/record/12632962) and extract it to the data folder. -->
```bash
# TODO: update the link later when the data is ready
# wget https://zenodo.org/record/12632962/files/demo_data.zip
unzip demo_data.zip -p /home/kin/data/av2
```

#### Prepare raw data 

Extract all data to unified h5 format. [Runtime: Normally need 10 mins finished run following commands totally in my desktop, 45 mins for the cluster I used]
```bash
python dataprocess/extract_av2.py --av2_type sensor --data_mode train --argo_dir /home/kin/data/av2 --output_dir /home/kin/data/av2/preprocess
python dataprocess/extract_av2.py --av2_type sensor --data_mode val --mask_dir /home/kin/data/av2/3d_scene_flow
python dataprocess/extract_av2.py --av2_type sensor --data_mode test --mask_dir /home/kin/data/av2/3d_scene_flow
```

#### Process train data

Process train data for self-supervised learning. Only training data needs this step. [Runtime: Normally need 15 hours for my desktop, 3 hours for the cluster with five available nodes parallel running.]

```bash
python 0_process.py --data_dir /home/kin/data/av2/preprocess/sensor/train --scene_range 0,701
```

### Train the model

Train SeFlow needed to specify the loss function, we set the config of our best model in the leaderboard. 

```bash
python 1_train.py model=deflow lr=2e-4 epochs=20 batch_size=16 loss_fn=seflowLoss "add_seloss={chamfer_dis: 1.0, static_flow_loss: 1.0, dynamic_chamfer_dis: 1.0, cluster_based_pc0pc1: 1.0}" "model.target.num_iters=2" "model.val_monitor=val/Dynamic/Mean"
```

### Other Benchmark Models

```bash
python 1_train.py model=fastflow3d lr=2e-4 epochs=20 batch_size=16 loss_fn=deflowLoss
python 1_train.py model=deflow lr=2e-4 epochs=20 batch_size=16 loss_fn=ff3dLoss
```

## 2. Evaluation

You can view Wandb dashboard for the training and evaluation results or upload result to online leaderboard.

Since in training, we save all hyper-parameters and model checkpoints, the only thing you need to do is to specify the checkpoint path. Remember to set the data path correctly also.

```bash
# downloaded pre-trained weight, or train by yourself
wget https://zenodo.org/records/12632962/files/seflow_official.ckpt

# it will directly prints all metric
python 2_eval.py checkpoint=/home/kin/seflow_official.ckpt av2_mode=val

# it will output the av2_submit.zip or av2_submit_v2.zip for you to submit to leaderboard
python 2_eval.py checkpoint=/home/kin/seflow_official.ckpt av2_mode=test leaderboard_version=1
python 2_eval.py checkpoint=/home/kin/seflow_official.ckpt av2_mode=test leaderboard_version=2
```


## 3. Visualization

We provide a script to visualize the results of the model. You can specify the checkpoint path and the data path to visualize the results. The step is quickly similar to evaluation.

```bash

# Then run the command in the terminal:
python tests/scene_flow.py --flow_mode 'seflow_official' --data_dir /home/kin/data/av2/preprocess/sensor/mini
```


## Cite & Acknowledgements

```
@article{zhang2024seflow,
  author={Zhang, Qingwen and Yang, Yi and Li, Peizheng and Andersson, Olov and Jensfelt, Patric},
  title={SeFlow: A Self-Supervised Scene Flow Method in Autonomous Driving},
  journal={arXiv preprint arXiv:2407.01702},
  year={2024}
}
@article{zhang2024deflow,
  author={Zhang, Qingwen and Yang, Yi and Fang, Heng and Geng, Ruoyu and Jensfelt, Patric},
  title={DeFlow: Decoder of Scene Flow Network in Autonomous Driving},
  journal={arXiv preprint arXiv:2401.16122},
  year={2024}
}
```

💞 Thanks to RPL member: [Li Ling](https://www.kth.se/profile/liling) helps revise our SeFlow manuscript. Thanks to [Kyle Vedder](https://kylevedder.github.io), who kindly opened his code (ZeroFlow) including pre-trained weights, and discussed their result with us which helped this work a lot. 

This work was partially supported by the Wallenberg AI, Autonomous Systems and Software Program (WASP) funded by the Knut and Alice Wallenberg Foundation and Prosense (2020-02963) funded by Vinnova. 
The computations were enabled by the supercomputing resource Berzelius provided by National Supercomputer Centre at Linköping University and the Knut and Alice Wallenberg Foundation, Sweden.

❤️: [DeFlow](https://github.com/KTH-RPL/DeFlow), [BucketedSceneFlowEval](https://github.com/kylevedder/BucketedSceneFlowEval)

