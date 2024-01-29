DeFlow Assets
---

## Installation

We will use conda to manage the environment with mamba for faster package installation.

### System
Install conda for package management and mamba for faster package installation:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash Mambaforge-$(uname)-$(uname -m).sh
```

### Environment

Create base env: [~5 mins]

```bash
git clone https://github.com/KTH-RPL/DeFlow
cd DeFlow && git submodule update --init --recursive
mamba env create -f assets/environment.yml
```

Install mmcv-full: [~30 mins] needed CUDA inside the env, echo ${PATH}

```bash
mamba activate deflow
cd mmcv && export MMCV_WITH_OPS=1 && export FORCE_CUDA=1 && pip install -e .
```


Checking the environment:
```bash
mamba activate deflow
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
python -c "import lightning.pytorch as pl"
python -c "from mmcv.ops import Voxelization, DynamicScatter;print('success test on mmcv package')"
```

## Dataset Download

We will note down the dataset download and itself detail here.

### Download

Since we focus on large point cloud dataset in autonomous driving, we choose Argoverse 2 for our dataset, you can also easily process other driving dataset in this framework. References: [3d_scene_flow user guide](https://argoverse.github.io/user-guide/tasks/3d_scene_flow.html), [Online Leaderboard](https://eval.ai/web/challenges/challenge-page/2010/evaluation).

```bash
# train is really big (750): totally 966 GB
s5cmd --no-sign-request cp "s3://argoverse/datasets/av2/sensor/train/*" sensor/train

# val (150) and test (150): totally 168GB + 168GB
s5cmd --no-sign-request cp "s3://argoverse/datasets/av2/sensor/val/*" sensor/val
s5cmd --no-sign-request cp "s3://argoverse/datasets/av2/sensor/test/*" sensor/test

# for local and online eval mask from official repo
s5cmd --no-sign-request cp "s3://argoverse/tasks/3d_scene_flow/zips/*" .
```

Then to quickly pre-process the data, we can run the following command to generate the pre-processed data for training and evaluation. This will take around 2 hour for the whole dataset (train & val) based on how powerful your CPU is.

```bash
python 0_preprocess.py --av2_type sensor --data_mode train --argo_dir /home/kin/data/av2 --output_dir /home/kin/data/av2/preprocess
python 0_preprocess.py --av2_type sensor --data_mode val --argo_dir /home/kin/data/av2 --output_dir /home/kin/data/av2/preprocess
```

## Leaderboard Submission

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




### Other issues

1. looks like open3d and fire package conflict, not sure
   -  need install package like `pip install --ignore-installed`, ref: [pip cannot install distutils installed project](https://stackoverflow.com/questions/53807511/pip-cannot-uninstall-package-it-is-a-distutils-installed-project), my error: `ERROR: Cannot uninstall 'blinker'.`
   -  need specific werkzeug version for open3d 0.16.0, otherwise error: `ImportError: cannot import name 'url_quote' from 'werkzeug.urls'`. But need update to solve the problem: `pip install --upgrade Flask` [ref](https://stackoverflow.com/questions/77213053/why-did-flask-start-failing-with-importerror-cannot-import-name-url-quote-fr)


2. `ImportError: libtorch_cuda.so: undefined symbol: cudaGraphInstantiateWithFlags, version libcudart.so.11.0`
   The cuda version: `pytorch::pytorch-cuda` and `nvidia::cudatoolkit` need be same. [Reference link](https://github.com/pytorch/pytorch/issues/90673#issuecomment-1563799299)


3. In cluster have error: `pandas ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.29' not found`
    Solved by `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/proj/berzelius-2023-154/users/x_qinzh/mambaforge/lib`


<!-- 


COMMANDS FOR Berzelius to copy

python 3_vis.py checkpoint=/proj/berzelius-2023-154/users/x_qinzh/workspace/deflow/logs/wandb/deflow-10078447/checkpoints/epoch_35_seflow.ckpt datasetpath=/proj/berzelius-2023-154/users/x_qinzh/av2/preprocess/sensor/mini

python tests/scene_flow.py --flow_mode='flow_est' --data_dir=/proj/berzelius-2023-154/users/x_qinzh/av2/preprocess/sensor/mini
-->