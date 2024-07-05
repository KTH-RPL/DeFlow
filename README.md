DeFlow 
---

[![arXiv](https://img.shields.io/badge/arXiv-2401.16122-b31b1b.svg)](https://arxiv.org/abs/2401.16122) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/deflow-decoder-of-scene-flow-network-in/scene-flow-estimation-on-argoverse-2)](https://paperswithcode.com/sota/scene-flow-estimation-on-argoverse-2?p=deflow-decoder-of-scene-flow-network-in)

Official task check: [https://github.com/KTH-RPL/DeFlow](https://github.com/KTH-RPL/DeFlow), Here is the inference for [DynamicMap Benchmark](https://github.com/KTH-RPL/DynamicMap_Benchmark).

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

Download data: [KTH-RPL/DynamicMap_Benchmark](https://github.com/KTH-RPL/DynamicMap_Benchmark?tab=readme-ov-file#dataset--scripts)

Download pre-trained weights for models are available in [Zenodo](https://zenodo.org/records/12632962). The weight is trained on Argoverse 2 Sensor dataset only, while DynamicMap Benchmark `av2` is from Argoverse 2 Map dataset. So no overfitting issue.

## 1. DynamicMap Inference

```bash
wget https://zenodo.org/records/12632962/files/deflow_official.ckpt

python main.py checkpoint=/home/kin/deflow_official.ckpt dataset_path=/home/kin/data/00
python main.py checkpoint=/home/kin/deflow_official.ckpt dataset_path=/home/kin/data/av2
```


Running time & Result screenshot:
![DynamicMap](assets/dynamic_benchmark.png)

## Cite & Acknowledgements

```
@article{zhang2024deflow,
  author={Zhang, Qingwen and Yang, Yi and Fang, Heng and Geng, Ruoyu and Jensfelt, Patric},
  title={DeFlow: Decoder of Scene Flow Network in Autonomous Driving},
  journal={arXiv preprint arXiv:2401.16122},
  year={2024}
}
```

This implementation is based on codes from several repositories. Thanks to these authors who kindly open-sourcing their work to the community. Please see our paper reference part to get more information. Thanks to Kyle Vedder (ZeroFlow) who kindly discussed their results with us and HKUST Ramlab's member: Jin Wu who gave constructive comments on this work. The computations were enabled by the supercomputing resource Berzelius provided by National Supercomputer Centre at Linköping University and the Knut and Alice Wallenberg Foundation, Sweden.

❤️: [ZeroFlow](https://github.com/kylevedder/zeroflow), [NSFP](https://github.com/Lilac-Lee/Neural_Scene_Flow_Prior), [FastNSF](https://github.com/Lilac-Lee/FastNSF). Others good code style and tools: [forecast-mae](https://github.com/jchengai/forecast-mae), [kiss-icp](https://github.com/PRBonn/kiss-icp)
