DeFlow: Decoder of Scene Flow Network in Autonomous Driving
---

[![arXiv](https://img.shields.io/badge/arXiv-2401.16122-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2401.16122) 
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/deflow-decoder-of-scene-flow-network-in/scene-flow-estimation-on-argoverse-2)](https://paperswithcode.com/sota/scene-flow-estimation-on-argoverse-2?p=deflow-decoder-of-scene-flow-network-in) 
[![poster](https://img.shields.io/badge/ICRA24|Poster-6495ed?style=flat&logo=Shotcut&logoColor=wihte)](https://hkustconnect-my.sharepoint.com/:b:/g/personal/qzhangcb_connect_ust_hk/EXP_uXYmm_tItTWc8MafXHoB-1dVrMnvF1-lCzU1PXAvqQ?e=2FPfBS) 
[![video](https://img.shields.io/badge/video-YouTube-FF0000?logo=youtube&logoColor=white)](https://youtu.be/bZ4uUv0nDa0)
[![blog](https://img.shields.io/badge/Blog%7C%E7%9F%A5%E4%B9%8E%E4%B8%AD%E6%96%87-1772f6?style=flat&logo=Shotcut)](https://zhuanlan.zhihu.com/p/706514747) 

Task: Scene Flow Estimation in Autonomous Driving. 

🤗 2024/11/18 16:17: Update model and demo data download link through HuggingFace, personally I found that `wget` from the HuggingFace link is much faster than Zenodo.

📜 2024/07/24: Merging SeFlow & DeFlow code together, lighter setup and easier running.

🔥 2024/07/02: Check the self-supervised version in our new ECCV'24 [SeFlow](https://github.com/KTH-RPL/SeFlow). The 1st ranking in new leaderboard among self-supervise methods.

Pre-trained weights for models are available in [Zenodo](https://zenodo.org/records/13744999)/[HuggingFace](https://huggingface.co/kin-zhang/OpenSceneFlow).
Check usage in [2. Evaluation](#2-evaluation) or [3. Visualization](#3-visualization). 

**Scripts** quick view in our scripts:

- `dataprocess/extract_*.py` : pre-process data before training to speed up the whole training time. 
  [Dataset we included now: Argoverse 2 and Waymo, more on the way: Nuscenes, custom data.]
  
- `train.py`: Train the model and get model checkpoints. Pls remember to check the config.

- `eval.py` : Evaluate the model on the validation/test set. And also output the zip file to upload to online leaderboard.

- `save.py` : Will save result into h5py file, using [tool/visualization.py] to show results with interactive window.


<details> <summary>🎁 <b>One repository, All methods!</b> </summary>
<!-- <br> -->
You can try following methods in our code without any effort to make your own benchmark.

- [x] [SeFlow](https://arxiv.org/abs/2407.01702) (Ours 🚀): ECCV 2024
- [x] [DeFlow](https://arxiv.org/abs/2401.16122) (Ours 🚀): ICRA 2024
- [x] [FastFlow3d](https://arxiv.org/abs/2103.01306): RA-L 2021
- [x] [ZeroFlow](https://arxiv.org/abs/2305.10424): ICLR 2024, their pre-trained weight can covert into our format easily through [the script](tools/zerof2ours.py).
- [ ] [NSFP](https://arxiv.org/abs/2111.01253): NeurIPS 2021, faster 3x than original version because of [our CUDA speed up](assets/cuda/README.md), same (slightly better) performance. Done coding, public after review.
- [ ] [FastNSF](https://arxiv.org/abs/2304.09121): ICCV 2023. Done coding, public after review.
<!-- - [ ] [Flow4D](https://arxiv.org/abs/2407.07995): 1st supervise network in the new leaderboard. Done coding, public after review. -->
- [ ] ... more on the way

</details>

💡: Want to learn how to add your own network in this structure? Check [Contribute](assets/README.md#contribute) section and know more about the code. Fee free to pull request!

## 0. Setup

**Environment**: Clone the repo and build the environment, check [detail installation](assets/README.md) for more information. [Conda](https://docs.conda.io/projects/miniconda/en/latest/)/[Mamba](https://github.com/mamba-org/mamba) is recommended.


```bash
git clone --recursive https://github.com/KTH-RPL/DeFlow.git
cd DeFlow
mamba env create -f environment.yaml
```

CUDA package (need install nvcc compiler), the compile time is around 1-5 minutes:
```bash
mamba activate deflow
# CUDA already install in python environment. I also tested others version like 11.3, 11.4, 11.7, 11.8 all works
cd assets/cuda/mmcv && python ./setup.py install && cd ../../..
cd assets/cuda/chamfer3D && python ./setup.py install && cd ../../..
```

Or another environment setup choice is [Docker](https://en.wikipedia.org/wiki/Docker_(software)) which isolated environment, you can pull it by. 
If you have different arch, please build it by yourself `cd DeFlow && docker build -t zhangkin/seflow` by going through [build-docker-image](assets/README.md/#build-docker-image) section.
```bash
# option 1: pull from docker hub
docker pull zhangkin/seflow

# run container
docker run -it --gpus all -v /dev/shm:/dev/shm -v /home/kin/data:/home/kin/data --name deflow zhangkin/seflow /bin/zsh
# then `mamba activate seflow` python environment is ready to use
```

## 1. Run & Train

Note: Prepare raw data and process train data only needed run once for the task. No need repeat the data process steps till you delete all data. We use [wandb](https://wandb.ai/) to log the training process, and you may want to change all `entity="kth-rpl"` to your own entity.

### Data Preparation

Check [dataprocess/README.md](dataprocess/README.md#argoverse-20) for downloading tips for the raw Argoverse 2 dataset. Or maybe you want to have the **mini processed dataset** to try the code quickly, We directly provide one scene inside `train` and `val`. It already converted to `.h5` format and processed with the label data. 
You can download it from [Zenodo](https://zenodo.org/records/13744999/files/demo_data.zip)/[HuggingFace](https://huggingface.co/kin-zhang/OpenSceneFlow/blob/main/demo_data.zip) and extract it to the data folder. And then you can skip following steps and directly run the [training script](#train-the-model).

```bash
wget https://huggingface.co/kin-zhang/OpenSceneFlow/resolve/main/demo_data.zip
unzip demo_data.zip -p /home/kin/data/av2
```

#### Prepare raw data 

Checking more information (step for downloading raw data, storage size, #frame etc) in [dataprocess/README.md](dataprocess/README.md). Extract all data to unified `.h5` format. 
[Runtime: Normally need 45 mins finished run following commands totally in setup mentioned in our paper]
```bash
python dataprocess/extract_av2.py --av2_type sensor --data_mode train --argo_dir /home/kin/data/av2 --output_dir /home/kin/data/av2/preprocess_v2
python dataprocess/extract_av2.py --av2_type sensor --data_mode val --mask_dir /home/kin/data/av2/3d_scene_flow
python dataprocess/extract_av2.py --av2_type sensor --data_mode test --mask_dir /home/kin/data/av2/3d_scene_flow
```

### Train the model

All local benchmarking methods and ablation studies can be done through command with different config, check [`assets/slurm`](assets/slurm) for all the commands we used in DeFlow raw paper. You can check all parameters in [conf/config.yaml](conf/config.yaml) and [conf/model/deflow.yaml](conf/model/deflow.yaml), **if you will set wandb_mode=online**, maybe change all `entity="kth-rpl"` to your own account name.

Train DeFlow with the leaderboard submit config. [Runtime: Around 6-8 hours in 4x A100 GPUs.] Please change `batch_size`&`lr` accoordingly if you don't have enough GPU memory. (e.g. `batch_size=6` for 24GB GPU)
```bash
python train.py model=deflow lr=2e-4 epochs=15 batch_size=16 loss_fn=deflowLoss
# baseline in our paper:
python train.py model=fastflow3d lr=4e-5 epochs=20 batch_size=16 loss_fn=ff3dLoss
```

> [!NOTE]  
> You may found the different settings in the paper that is all methods are enlarge learning rate to 2e-4 and decrease the epochs to 15 for faster converge and better performance (it's also our leaderboard model train config). 
> However, we kept the setting on lr=2e-6 and 50 epochs in (SeFlow & DeFlow) paper experiments for **the fair comparison** with ZeroFlow where we directly use their provided weights. 
> We suggest afterward researchers or users to use the setting here (larger lr and smaller epoch) for faster converge and better performance.

To help community benchmarking, we provide our weights including fastflow3d, deflow in [HuggingFace](https://huggingface.co/kin-zhang/OpenSceneFlow). 
These checkpoints also include parameters and status of that epoch inside it.

## 2. Evaluation

You can view Wandb dashboard for the training and evaluation results or run/submit to av2 leaderboard to get official results follow below steps.

Since in training, we save all hyper-parameters and model checkpoints, the only thing you need to do is to specify the checkpoint path. Remember to set the data path correctly also.
```bash
# downloaded pre-trained weight, or train by yourself
wget https://huggingface.co/kin-zhang/OpenSceneFlow/resolve/main/deflow_best.ckpt

python eval.py checkpoint=/home/kin/deflow_best.ckpt av2_mode=val # it will directly prints all metric
# it will output a command with absolute path of a zip file for you to submit to leaderboard
python eval.py checkpoint=/home/kin/deflow_best.ckpt av2_mode=test leaderboard_version=1
python eval.py checkpoint=/home/kin/deflow_best.ckpt av2_mode=test leaderboard_version=2
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

# Step 3: Copy the command pop above and submit to leaderboard
evalai challenge 2010 phase 4018 submit --file av2_submit.zip --large --private
evalai challenge 2210 phase 4396 submit --file av2_submit_v2.zip --large --private
```

## 3. Visualization

We provide a script to visualize the results of the model. You can specify the checkpoint path and the data path to visualize the results. The step is quickly similar to evaluation.

```bash
# downloaded pre-trained weight, or train by yourself
wget https://huggingface.co/kin-zhang/OpenSceneFlow/resolve/main/deflow_best.ckpt

python save.py checkpoint=/home/kin/deflow_best.ckpt dataset_path=/home/kin/data/av2/preprocess/sensor/vis

# Then terminal will tell you the command you need run. For example here is the output of the above:
Model: DeFlow, Checkpoint from: /home/kin/deflow_best.ckpt
We already write the estimate flow: deflow_best into the dataset, please run following commend to visualize the flow. Copy and paste it to your terminal:
python tools/visualization.py --flow_mode 'deflow_best' --data_dir /home/kin/data/av2/preprocess/sensor/mini
Enjoy! ^v^ ------ 

# Then run the command in the terminal:
python tools/visualization.py --res_name 'deflow_best' --data_dir /home/kin/data/av2/preprocess_v2/sensor/mini
```

https://github.com/KTH-RPL/DeFlow/assets/35365764/9b265d56-06a9-4300-899c-96047a0da505


## Cite & Acknowledgements

```
@inproceedings{zhang2024deflow,
  author={Zhang, Qingwen and Yang, Yi and Fang, Heng and Geng, Ruoyu and Jensfelt, Patric},
  booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)}, 
  title={{DeFlow}: Decoder of Scene Flow Network in Autonomous Driving}, 
  year={2024},
  pages={2105-2111},
  doi={10.1109/ICRA57147.2024.10610278}
}
@inproceedings{zhang2024seflow,
  author={Zhang, Qingwen and Yang, Yi and Li, Peizheng and Andersson, Olov and Jensfelt, Patric},
  title={{SeFlow}: A Self-Supervised Scene Flow Method in Autonomous Driving},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2024},
  pages={353–369},
  organization={Springer},
  doi={10.1007/978-3-031-73232-4_20},
}
```

This implementation is based on codes from several repositories. Thanks to these authors who kindly open-sourcing their work to the community. Please see our paper reference part to get more information. 
Thanks to [Kyle Vedder (ZeroFlow)](https://github.com/kylevedder) who kindly discussed their results with us and HKUST Ramlab's member: Jin Wu who gave constructive comments on this work. 
The computations were enabled by the supercomputing resource Berzelius provided by National Supercomputer Centre at Linköping University and the Knut and Alice Wallenberg Foundation, Sweden.

❤️: [ZeroFlow](https://github.com/kylevedder/zeroflow), [NSFP](https://github.com/Lilac-Lee/Neural_Scene_Flow_Prior), [FastNSF](https://github.com/Lilac-Lee/FastNSF). Others good code style and tools: [forecast-mae](https://github.com/jchengai/forecast-mae), [kiss-icp](https://github.com/PRBonn/kiss-icp)
