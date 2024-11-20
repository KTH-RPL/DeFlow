Mics Tests (Speed,)
---

Some speed comparison testing in algorithms and data structures.


## Cal Chamfer Distance

Points num: pc0: 88132, pc1: 88101

| Function | Time (ms) |
| :---: | :---: |
| CUDA(Old, SCOOP, Batch) | 83.275 |
| Faiss | 368.269 |
| Pytorch3D | 61.474 |
| CUDA(New, NonBatch) Ours| 33.199 |
| CUDA(New, SharedM) Ours | 17.379 |


## Cal HDBSCAN

Points num: pc0: 85396, pc1: 85380

| Function | Time (s) |
| :---: | :---: |
| cuml | 1.604 |
| sklearn | 31.946 |
| scikit-hdbscan | 3.064 |

dependence: [cuml](https://github.com/rapidsai/cuml)
```bash
# cuml
mamba create -n rapids -c rapidsai -c conda-forge -c nvidia rapids=23.12 python=3.9 cuda-version=11.8 pytorch
mamba activate rapids

# sklearn
mamba install -c conda-forge scikit-learn

# scikit-hdbscan (independent version)
mamba install -c conda-forge hdbscan

```