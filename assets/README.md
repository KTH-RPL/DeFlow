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

Install mmcv-full: [~15/30 mins] needed CUDA inside the env, echo ${PATH}

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