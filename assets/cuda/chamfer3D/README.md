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

## Mics


### Note for CUDA ChamferDis

主要是 两个月前写的 已经看不懂了；然后问题原因是因为 总是缺0.0003的精度（精度强迫症患者）
然后就以为是自己写错了 后面发现是因为block的这种并行化 线程大小的不同对CUDA的浮点运算会有所不同，所以导致精度差距是有一点的 如果介意的话 可以使用pytorch3d的版本（也就是速度慢4倍左右 从15ms 到 80ms）

这里主要重申一遍 shared memory在这里的用法：
1. 首先我们每个点都会分开走到 `int tid = blockIdx.x * blockDim.x + threadIdx.x;` 也就是全局索引，注意这个每个点都分开了 因为pc0每个点和pc1的临近点 和 其他的pc0点无关
2. 然后走到每个点内部 就是__shared__ 我们首先建立了 pc1的share，但是因为共享内存有限，所以每次只保存THREADS_PER_BLOCK
3. 保存 THREADS_PER_BLOCK 也是每个线程做的 我们在对比距离前 运行了 __syncthreads(); 确保 THREADS_PER_BLOCK 个点的 pc1 已经到了
4. 接着 我们在 `num_elems` 这一部分的数据内进行对比，同步best
5. 最后传给 全局这个点的 `result`

需要注意的是 这种极致的并行化 会对精度产生一定的影响，但是如果你感兴趣 `#define THREADS_PER_BLOCK 256` 可以调整这个，对每个block设置不同的threads 会对精度有影响（当然 影响是 在 gt: 0.1710 但cuda计算会是 0.1711 - 0.1713之间）

以下为chatgpt：
精度差异的原因之一可能是由于在不同的线程块大小下，浮点运算的顺序发生了改变。由于浮点运算是不结合的（即(a + b) + c 可能不等于 a + (b + c)），因此改变运算的顺序可能会导致轻微的结果差异。

这种类型的精度变化在GPU计算中是非常常见的，特别是在使用较大的数据集和进行大量的浮点运算时。要完全消除这种差异是非常困难的，因为即使是非常微小的实现细节变化（例如改变线程块大小、更改循环的结构、甚至是不同的GPU硬件或不同的CUDA版本）都可能导致浮点运算顺序的微小变化。

如果需要确保结果的一致性，可以考虑以下方法：

1. 固定线程块大小：选择一个固定的线程块大小，并始终使用它。

2. 双精度浮点数（Double Precision）：使用double类型代替float，可以提高精度，但代价是更高的内存使用和可能的性能下降。

3. 数值稳定的算法：尽量使用数值稳定的算法，尽管这在GPU上实现起来可能比较复杂且效率较低。

4. 减少并行化程度：通过减少并行化程度来减少由于不同线程执行顺序引起的差异，但这通常会牺牲性能。


复制代码部分如下：
```cpp

for (int i = 0; i < pc1_n; i += THREADS_PER_BLOCK) {
	// Copy a block of pc1 to shared memory
	int pc1_idx = i + threadIdx.x;
	if (pc1_idx < pc1_n) {
		shared_pc1[threadIdx.x * 3 + 0] = pc1_xyz[pc1_idx * 3 + 0];
		shared_pc1[threadIdx.x * 3 + 1] = pc1_xyz[pc1_idx * 3 + 1];
		shared_pc1[threadIdx.x * 3 + 2] = pc1_xyz[pc1_idx * 3 + 2];
	}

	__syncthreads();

	// Compute the distance between pc0[tid] and the points in shared_pc1
	// NOTE(Qingwen):  since after two months I forgot what I did here, I write some notes for future me
	// 0. One reason for the difference in precision may be due to the changing order of floating point operations at different thread block sizes.
	//    But I think it's fine we lose 0.0001 precision for speed up cal time 4x
	// 1. since we use shared to store pc1, here Every BLOCK will have new shared_pc1 start from 0
	// 2. we use THREADS_PER_BLOCK to loop pc1, so we need to check if the last block is not full
	// 3. Based on the CUDA document, the __syncthreads() is not necessary here, but we keep it for safety
	// 4. After running once, we go for next block of pc1, and find the best in that batch
	
	int num_elems = min(THREADS_PER_BLOCK, pc1_n - i);
	for (int j = 0; j < num_elems; j++) {
		float x1 = shared_pc1[j * 3 + 0];
		float y1 = shared_pc1[j * 3 + 1];
		float z1 = shared_pc1[j * 3 + 2];
		float d = (x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0) + (z1 - z0) * (z1 - z0);
		if (d < best) {
			best = d;
			best_i = j + i;
		}
	}
	__syncthreads();
}
```

## Other issues
In cluster when build cuda things, you may occur problem:
- `gcc: error trying to exec 'cc1plus': execvp: No such file or directory`, 
  	Main reason is gcc and g++ version problem in cluster, you can try to install inside `conda` to solve that, with: `mamba install -c conda-forge gxx==9.5.0`. And the reason why I set 9.5.0 is because of the version for cuda 11.3 need inside specific version.
	```bash
	RuntimeError: The current installed version of g++ (13.2.0) is greater than the maximum required version by CUDA 11.3. Please make sure to use an adequate version of g++ (>=5.0.0, <11.0).
	```
