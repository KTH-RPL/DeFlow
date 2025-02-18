DeFlow: Decoder of Scene Flow Network in Autonomous Driving
---

[![arXiv](https://img.shields.io/badge/arXiv-2401.16122-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2401.16122) 
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/deflow-decoder-of-scene-flow-network-in/scene-flow-estimation-on-argoverse-2)](https://paperswithcode.com/sota/scene-flow-estimation-on-argoverse-2?p=deflow-decoder-of-scene-flow-network-in) 
[![poster](https://img.shields.io/badge/ICRA24|Poster-6495ed?style=flat&logo=Shotcut&logoColor=wihte)](https://hkustconnect-my.sharepoint.com/:b:/g/personal/qzhangcb_connect_ust_hk/EXP_uXYmm_tItTWc8MafXHoB-1dVrMnvF1-lCzU1PXAvqQ?e=2FPfBS) 
[![video](https://img.shields.io/badge/video-YouTube-FF0000?logo=youtube&logoColor=white)](https://youtu.be/bZ4uUv0nDa0)
[![blog](https://img.shields.io/badge/Blog%7C%E7%9F%A5%E4%B9%8E%E4%B8%AD%E6%96%87-1772f6?style=flat&logo=Shotcut)](https://zhuanlan.zhihu.com/p/706514747) 

Task: Scene Flow Estimation in Autonomous Driving. 

📜 2025/02/18: Merging all scene flow code to [OpenSceneFLow codebase](https://github.com/KTH-RPL/OpenSceneFlow) for afterward code maintenance. This repo saved README, [cluster slurm files](assets/slurm), and [quick core file](decoder.py) in DeFlow. The old source code branch is also [available here](https://github.com/KTH-RPL/DeFlow/tree/source).

🤗 2024/11/18 16:17: Update model and demo data download link through HuggingFace, personally I found that `wget` from the HuggingFace link is much faster than Zenodo.

📜 2024/07/24: Merging SeFlow & DeFlow code together, lighter setup and easier running.

🔥 2024/07/02: Check the self-supervised version in our new ECCV'24 [SeFlow](https://github.com/KTH-RPL/SeFlow). The 1st ranking in new leaderboard among self-supervise methods.

Pre-trained weights for models are available in [Zenodo](https://zenodo.org/records/13744999)/[HuggingFace](https://huggingface.co/kin-zhang/OpenSceneFlow).
Check usage in [2. Evaluation](#2-evaluation) or [3. Visualization](#3-visualization). 

## 0. Setup

**Environment**: Clone the repo and build the environment, check [detail installation](https://github.com/KTH-RPL/OpenSceneFlow/assets/README.md) for more information. [Conda](https://docs.conda.io/projects/miniconda/en/latest/)/[Mamba](https://github.com/mamba-org/mamba) is recommended.


```bash
git clone --recursive https://github.com/KTH-RPL/OpenSceneFlow.git
cd OpenSceneFlow
mamba env create -f environment.yaml
```

CUDA package (need install nvcc compiler), the compile time is around 1-5 minutes:
```bash
mamba activate opensf
# CUDA already install in python environment. I also tested others version like 11.3, 11.4, 11.7, 11.8 all works
cd assets/cuda/mmcv && python ./setup.py install && cd ../../..
cd assets/cuda/chamfer3D && python ./setup.py install && cd ../../..
```

Or another environment setup choice is [Docker](https://en.wikipedia.org/wiki/Docker_(software)) which isolated environment, check more information in [OpenSceneFlow/assets/README.md](https://github.com/KTH-RPL/OpenSceneFlow/assets/README.md#docker-environment).



## 1. Run & Train

Note: Prepare raw data and process train data only needed run once for the task. No need repeat the data process steps till you delete all data. We use [wandb](https://wandb.ai/) to log the training process, and you may want to change all `entity="kth-rpl"` to your own entity.

### Data Preparation

Check [OpenSceneFlow/dataprocess/README.md](https://github.com/KTH-RPL/OpenSceneFlow/dataprocess/README.md#argoverse-20) for downloading tips for the raw Argoverse 2 dataset. Or maybe you want to have the **mini processed dataset** to try the code quickly, We directly provide one scene inside `train` and `val`. It already converted to `.h5` format and processed with the label data. 
You can download it from [Zenodo](https://zenodo.org/records/13744999/files/demo_data.zip)/[HuggingFace](https://huggingface.co/kin-zhang/OpenSceneFlow/blob/main/demo_data.zip) and extract it to the data folder. And then you can skip following steps and directly run the [training script](#train-the-model).

```bash
wget https://huggingface.co/kin-zhang/OpenSceneFlow/resolve/main/demo_data.zip
unzip demo_data.zip -p /home/kin/data/av2
```

### Train the model

All local benchmarking methods and ablation studies can be done through command with different config, check [`assets/slurm`](assets/slurm) for all the commands we used in DeFlow raw paper. You can check all parameters in [OpenSceneFlow/conf/config.yaml](https://github.com/KTH-RPL/OpenSceneFlow/conf/config.yaml) and [OpenSceneFlow/conf/model/deflow.yaml](https://github.com/KTH-RPL/OpenSceneFlow/conf/model/deflow.yaml), **if you will set wandb_mode=online**, maybe change all `entity="kth-rpl"` to your own account name.

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

To submit to the Online Leaderboard, please follow the [updated codebase link](https://github.com/KTH-RPL/OpenSceneFlow/tree/main#3-evaluation).

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
