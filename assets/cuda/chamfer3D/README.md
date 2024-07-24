CUDA with Torch 初步尝试
---

主要参考都是mmcv里的库 and [SCOOP](https://github.com/itailang/SCOOP/blob/master/auxiliary/ChamferDistancePytorch/chamfer3D/)，只是想着仅提取这一个功能试一下，然后发现几个有意思的点 
CUDA with Torch C++ Programming, [torch official ref link](https://pytorch.org/tutorials/advanced/cpp_extension.html)

1. `.cu` 不能被`.cpp` include，否则失去cu属性
2. `.cpp`必须命名为`xxx_cuda.cpp` 否则torch CUDAextension不会找
3. `.cu` 的include和平常的cuda编程有所不同：必须先include ATen然后再正常的导入 CUDA等库 [ref link](https://blog.csdn.net/weixin_39849839/article/details/125980694)
	```cpp
	#include <ATen/ATen.h>
	
	#include <cuda.h>
	#include <cuda_runtime.h>
	```
	
注意要自己多写一个lib_xxx.py 内含class以方便调用，但是class内的forward函数必须有 `ctx` 参数，否则会报错


## Install
```bash
# change it if you use different cuda version
export PATH=/usr/local/cuda-11.3/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64:$LD_LIBRARY_PATH

cd assets/cuda/chamfer3Dlib
python ./setup.py install

# then you will see
Installed /home/kin/mambaforge/envs/seflow/lib/python3.8/site-packages/chamfer3D-1.0.0-py3.8-linux-x86_64.egg
Processing dependencies for chamfer3D==1.0.0
Finished processing dependencies for chamfer3D==1.0.0

# then run with lib_voxelize.py to see if it works
python ../chamfer_cuda.py
```

## ChamferDis Speed

The number of points: (pc0: 88132, pc1: 88101)

| Function | Time (ms) |
| :---: | :---: |
| Faiss | 817.698 |
| CUDA([SCOOP](https://github.com/itailang/SCOOP/tree/master/auxiliary/ChamferDistancePytorch), Batch) | 83.275 |
| Pytorch3D | 68.256 |
| CUDA([SeFlow](https://github.com/KTH-RPL/SeFlow), SharedM) | **14.308** |
| ~~mmcv~~(chamfer2D) | 651.510 |

对比命令行：

```bash
cd assets/tests
python chamferdis_speed_test.py
```


Test computer and System:
- Desktop setting: i9-12900KF, GPU 3090, CUDA 11.3
- System setting: Ubuntu 20.04, Python 3.8

Output Example:
```
Output in my desktop with a 3090 GPU:
------ START Faiss Chamfer Distance Cal ------
loss:  tensor(0.1710, device='cuda:0')
Faiss Chamfer Distance Cal time: 809.593 ms

------ START Pytorch3d Chamfer Distance Cal ------
Pytorch3d Chamfer Distance Cal time: 68.906 ms
loss:  tensor(0.1710, device='cuda:0', grad_fn=<AddBackward0>)

------ START CUDA Chamfer Distance Cal ------
Chamfer Distance Cal time: 1.814 ms
loss:  tensor(0.1710, device='cuda:0', grad_fn=<AddBackward0>)
```


## Other issues
In cluster when build cuda things, you may occur problem:
- `gcc: error trying to exec 'cc1plus': execvp: No such file or directory`, 
  	Main reason is gcc and g++ version problem in cluster, you can try to install inside `conda` to solve that, with: `mamba install -c conda-forge gxx==9.5.0`. And the reason why I set 9.5.0 is because of the version for cuda 11.3 need inside specific version.
	```bash
	RuntimeError: The current installed version of g++ (13.2.0) is greater than the maximum required version by CUDA 11.3. Please make sure to use an adequate version of g++ (>=5.0.0, <11.0).
	```
