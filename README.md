DeFlow 
---

[![arXiv](https://img.shields.io/badge/arXiv-TODO.TODO-b31b1b.svg)](https://arxiv.org/abs/TODO.TODO) 
Under review, will public source code after acceptance around Feb.

Task: Scene Flow Estimation in Autonomous Driving

https://github.com/KTH-RPL/DeFlow/assets/35365764/982df3c1-563b-4737-90b2-21f62e045aa9

**Scripts** quick view in `run_steps` folder:

- `0_preprocess.py` : pre-process data before training to speed up the whole training time
- `1_train.py`: Train the model and get model checkpoints. Pls remember to check the config.
- `2_eval.py` : Evaluate the model on the validation/test set. And also upload to online leaderboard.
- `3_vis.py` : For visualization of the results with a video.

## 0. Setup

Install conda for package management and mamba for faster package installation:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash Mambaforge-$(uname)-$(uname -m).sh
```

Clone the repo and build the environment
```
git clone https://github.com/KTH-RPL/DeFlow
cd DeFlow
mamba env create -f environment.yml
```

## 1. Dataset

Since we focus on large point cloud dataset in autonomous driving, we choose Argoverse 2 for our dataset, you can also easily process other driving dataset in this framework. References: [3d_scene_flow user guide](https://argoverse.github.io/user-guide/tasks/3d_scene_flow.html), [Online Leaderboard](https://eval.ai/web/challenges/challenge-page/2010/evaluation).

```bash
# train is really big (750): totally 966 GB
s5cmd --no-sign-request cp "s3://argoverse/datasets/av2/sensor/train/*" sensor/train

# val (150) and test (150): totally 168GB + 168GB
s5cmd --no-sign-request cp "s3://argoverse/datasets/av2/sensor/val/*" sensor/val
s5cmd --no-sign-request cp "s3://argoverse/datasets/av2/sensor/test/*" sensor/test
```

## 2. Train

Best fine-tuned model train with following command:
```bash
python 1_train.py model=deflow lr=2e-6 epochs=50 batch_size=16
```

All local benchmarking methods and ablation studies can be done through command with different config, check [`assets/slurm`](assets/slurm) for all the commands we used in our experiments.

```bash
python 1_train.py model=fastflow3d lr=2e-6 epochs=50 batch_size=16 "model.target.TB=TB"

python 1_train.py model=deflow lr=2e-6 epochs=50 batch_size=16 "model.target.TB=TB"
```

## 3. Evaluation

You can view Wandb dashboard for the training and evaluation results or run the av2 leaderboard scripts to get official results.

### Local Eval
For the av2 leaderboard, we need to follow the official instructions:

1. Download the mask file for 3D scene flow task
    ```bash
    s5cmd --no-sign-request cp "s3://argoverse/tasks/3d_scene_flow/zips/*" .
    ```
2. `make_annotation_files.py`
    ```
    python3 av2-api/src/av2/evaluation/scene_flow/make_annotation_files.py /home/kin/data/av2/3d_scene_flow/eval /home/kin/data /home/kin/data/av2/3d_scene_flow/val-masks.zip --split val
    ```
3. `eval.py` computes all leaderboard metrics.


### Online Eval

1. The directory format should be that in `result_path`:
    ```
    - <test_log_1>/
      - <test_timestamp_ns_1>.feather
      - <test_timestamp_ns_2>.feather
      - ...
    - <test_log_2>/
    - ...
    ```

2. Run `make_submission_archive.py` to make the zip file for submission.
    ```
    python av2-api/src/av2/evaluation/scene_flow/make_submission_archive.py checkpoints/results/test/example /home/kin/data/av2/av2_3d_scene_flow/test-masks.zip --output_filename sub_example.zip
    ```

3. Submit on the website more commands on [EvalAI-CLI](https://cli.eval.ai/). Normally, one file may be around 1GB, so you need to use `--large` flag.
    ```
    evalai set_token <your token>
    evalai challenge 2010 phase 4018 submit --file <submission_file_path> --large --private
    ```
4. Check in online eval leaderboard website: [Argoverse 2 Scene Flow](https://eval.ai/web/challenges/challenge-page/2010/leaderboard/4759).

## Cite & Acknowledgements

```
@article{zhang2024deflow,
  author={Zhang, Qingwen and Yang, Yi and Fang, Heng and Geng, Ruoyu and Jensfelt, Patric},
  title={DeFlow: TB},
  journal={arXiv preprint arXiv:todo.todo},
  year={2024}
}
```

This implementation is based on codes from several repositories. Thanks for these authors who kindly open-sourcing their work to the community. Please see our paper reference part to get more information.

❤️: [ZeroFlow](https://github.com/kylevedder/zeroflow), [NSFP](https://github.com/Lilac-Lee/Neural_Scene_Flow_Prior), [FastNSF](https://github.com/Lilac-Lee/FastNSF)
