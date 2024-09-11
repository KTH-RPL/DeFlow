SeFlow Assets
---

There are two ways to setup the environment: conda in your desktop and docker container isolate environment.

## Docker

Docker installation check [DeFlow Assets](https://github.com/KTH-RPL/DeFlow/blob/main/assets/README.md#docker-environment). Then you can build and run the container by:

```bash
cd SeFlow
docker build -t zhangkin/seflow .

docker run -it --gpus all -v /dev/shm:/dev/shm -v /home/kin/data:/home/kin/data --name seflow zhangkin/seflow /bin/zsh
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
git clone https://github.com/KTH-RPL/SeFlow.git
mamba env create -f assets/environment.yml
```

CUDA package (nvcc compiler already installed through conda), the compile time is around 1-5 minutes:
```bash
mamba activate seflow
cd assets/cuda/mmcv && python ./setup.py install && cd ../../..
cd assets/cuda/chamfer3D && python ./setup.py install && cd ../../..
```


Checking important packages in our environment now:
```bash
mamba activate seflow
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
python -c "import lightning.pytorch as pl; print(pl.__version__)"
python -c "from assets.cuda.mmcv import Voxelization, DynamicScatter;print('successfully import on our lite mmcv package')"
python -c "from assets.cuda.chamfer3D import nnChamferDis;print('successfully import on our chamfer3D package')"
```


### Other issues

1. looks like open3d and fire package conflict, not sure
   -  need install package like `pip install --ignore-installed`, ref: [pip cannot install distutils installed project](https://stackoverflow.com/questions/53807511/pip-cannot-uninstall-package-it-is-a-distutils-installed-project), my error: `ERROR: Cannot uninstall 'blinker'.`
   -  need specific werkzeug version for open3d 0.16.0, otherwise error: `ImportError: cannot import name 'url_quote' from 'werkzeug.urls'`. But need update to solve the problem: `pip install --upgrade Flask` [ref](https://stackoverflow.com/questions/77213053/why-did-flask-start-failing-with-importerror-cannot-import-name-url-quote-fr)


2. `ImportError: libtorch_cuda.so: undefined symbol: cudaGraphInstantiateWithFlags, version libcudart.so.11.0`
   The cuda version: `pytorch::pytorch-cuda` and `nvidia::cudatoolkit` need be same. [Reference link](https://github.com/pytorch/pytorch/issues/90673#issuecomment-1563799299)


3. In cluster have error: `pandas ImportError: /lib64/libstdc++.so.6: version 'GLIBCXX_3.4.29' not found`
    Solved by `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/proj/berzelius-2023-154/users/x_qinzh/mambaforge/lib`


## Contribute

If you want to contribute to new model, here are tips you can follow:
1. Dataloader: we believe all data could be process to `.h5`, we named as different scene and inside a scene, the key of each data is timestamp. Check [dataprocess/README.md](../dataprocess/README.md#process) for more details.
2. Model: All model files can be found [here: src/models](../src/models). You can view deflow and fastflow3d to know how to implement a new model. Don't forget to add to the `__init__.py` [file to import class](../src/models/__init__.py).
3. Loss: All loss files can be found [here: src/lossfuncs.py](../src/lossfuncs.py). There are three loss functions already inside the file, you can add a new one following the same pattern.
4. Training: Once you have implemented the model, you can add the model to the config file [here: conf/model](../conf/model) and train the model using the command `python train.py model=your_model_name`. One more note here may: if your res_dict from model output is different, you may need add one pattern in `def training_step` and `def validation_step`.

All others like eval and vis will be changed according to the model you implemented as you follow the above steps.