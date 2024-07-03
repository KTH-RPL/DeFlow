SeFlow: A Self-Supervised Scene Flow Method in Autonomous Driving
---

[![arXiv](https://img.shields.io/badge/arXiv-2407.01702-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2407.01702)
[poster comming soon]
[viceo coming soon]

We got accepted! I will **update the whole code** around the DDL of the camera-ready version (**Jul'15**). Stay Tunned with us.

2024/07/03 10:15: I'm working on updating code here now. Not fully ready yet until Jul'15.

Task: Self-Supervised Scene Flow Estimation in Autonomous Driving. 

**Scripts** quick view in our scripts (We directly follow our previous work [DeFlow code structure](https://github.com/KTH-RPL/DeFlow)):

- `dataprocess/extract_*.py` : pre-process data before training to speed up the whole training time. 
  [Dataset we included now: Argoverse 2 and Waymo.  more on the way: Nuscenes, custom data.]
  
- `0_process.py`: process data with save dufomap, cluster labels inside file.

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
# change it if you use different cuda version (I tested 11.3, 11.4, 11.7 all works)
export PATH=/usr/local/cuda-11.7/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH

mamba activate seflow
cd assets/cuda/mmcv && python ./setup.py install
cd assets/cuda/chamfer3D && python ./setup.py install
```

Or you always can choose [Docker](https://en.wikipedia.org/wiki/Docker_(software)) which isolated environment and free yourself from installation, you can pull it by. 
If you have different arch, please build it by yourself `cd SeFlow && docker build -t zhangkin/seflow` by going through [build-docker-image](https://github.com/KTH-RPL/DeFlow/blob/main/assets/README.md/#build-docker-image) section.
```bash
# option 1: pull from docker hub
docker pull zhangkin/seflow

# run container
docker run -it --gpus all -v /dev/shm:/dev/shm -v /home/kin/data:/home/kin/data --name seflow zhangkin/seflow /bin/zsh
```

## 1. Run & Train

coming soon

## 2. Evaluation & Upload to Leaderboard

You can view Wandb dashboard for the training and evaluation results or upload result to online leaderboard.

Since in training, we save all hyper-parameters and model checkpoints, the only thing you need to do is to specify the checkpoint path. Remember to set the data path correctly also.

```bash
# downloaded pre-trained weight, or train by yourself
wget TODO
python 2_eval.py checkpoint=/home/kin/seflow_best.ckpt av2_mode=val # it will directly prints all metric
python 2_eval.py checkpoint=/home/kin/seflow_best.ckpt av2_mode=test # it will output the av2_submit.zip for you to submit to leaderboard
```


## 3. Visualization

We provide a script to visualize the results of the model. You can specify the checkpoint path and the data path to visualize the results. The step is quickly similar to evaluation.

```bash

# Then run the command in the terminal:
python tests/scene_flow.py --flow_mode 'seflow_best' --data_dir /home/kin/data/av2/preprocess/sensor/mini
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

Thanks to RPL member: Li Ling helps review our SeFlow manuscript. 
Thanks to [Kyle Vedder (ZeroFlow)](https://github.com/kylevedder), who kindly opened his code including pre-trained weights, and discussed their result with us which helped this work a lot. 
This work was partially supported by the Wallenberg AI, Autonomous Systems and Software Program (WASP) funded by the Knut and Alice Wallenberg Foundation and Prosense (2020-02963) funded by Vinnova. 
The computations were enabled by the supercomputing resource Berzelius provided by National Supercomputer Centre at Linköping University and the Knut and Alice Wallenberg Foundation, Sweden.

❤️: [DeFlow](https://github.com/KTH-RPL/DeFlow), [BucketedSceneFlowEval](https://github.com/kylevedder/BucketedSceneFlowEval)

