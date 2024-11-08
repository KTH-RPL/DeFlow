DeFlow Assets
---

There are two ways to setup the environment: conda in your desktop and docker container isolate environment.

## Docker Environment

### Build Docker Image
If you want to build docker with compile all things inside, there are some things need setup first in your own desktop environment: 
- [NVIDIA-driver](https://www.nvidia.com/download/index.aspx): which I believe most of people may already have it. Try `nvidia-smi` to check if you have it.
- [Docker](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository):
   ```bash
   # Add Docker's official GPG key:
   sudo apt-get update
   sudo apt-get install ca-certificates curl
   sudo install -m 0755 -d /etc/apt/keyrings
   sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
   sudo chmod a+r /etc/apt/keyrings/docker.asc

   # Add the repository to Apt sources:
   echo \
   "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
   $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
   sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
   sudo apt-get update
   ```
- [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)
   ```bash
   sudo apt update && apt install nvidia-container-toolkit
   ```

Then follow [this stackoverflow answers](https://stackoverflow.com/questions/59691207/docker-build-with-nvidia-runtime):
1. Edit/create the /etc/docker/daemon.json with content:
   ```bash
   {
      "runtimes": {
         "nvidia": {
               "path": "/usr/bin/nvidia-container-runtime",
               "runtimeArgs": []
            } 
      },
      "default-runtime": "nvidia" 
   }
   ```
2. Restart docker daemon:
   ```bash
   sudo systemctl restart docker
   ```

3. Then you can build the docker image:
   ```bash
   cd DeFlow && docker build -t zhangkin/deflow .
   ```
   
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


## Contribute

If you want to contribute to new model, here are tips you can follow:
1. Dataloader: we believe all data could be process to `.h5`, we named as different scene and inside a scene, the key of each data is timestamp.
2. Model: All model files can be found [here: scripts/network/models](../scripts/network/models). You can view deflow and fastflow3d to know how to implement a new model.
3. Loss: All loss files can be found [here: scripts/network/loss_func.py](../scripts/network/loss_func.py). There are three loss functions already inside the file, you can add a new one following the same pattern.
4. Training: Once you have implemented the model, you can add the model to the config file [here: conf/model](../conf/model) and train the model using the command `python 1_train.py model=your_model_name`. One more note here may: if your res_dict from model output is different, you may need add one pattern in `def training_step` and `def validation_step`.

All others like eval and vis will be changed according to the model you implemented as you follow the above steps.