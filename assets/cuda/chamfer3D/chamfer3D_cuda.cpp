/*
 * Copyright (C) 2022-now, RPL, KTH Royal Institute of Technology
 * @Author: Qingwen Zhang (https://kin-zhang.github.io/)
 * @Date: 2023-08-03 16:55
 * @Description: Chamfer distance calculation between two point clouds with CUDA
 * This file is part of SeFlow (https://github.com/KTH-RPL/SeFlow).
 * If you find this repo helpful, please cite the respective publication as 
 * listed on the above website.


 * Reference: Modified from SCOOP chamfer3D [https://github.com/itailang/SCOOP]
 * faster 2x than the original version
*/

#include <torch/torch.h>
#include <vector>

int chamfer_cuda_forward(
    const at::Tensor &pc0, 
    const at::Tensor &pc1, 
    at::Tensor &dist0, 
    at::Tensor &dist1, 
    at::Tensor &idx0, 
    at::Tensor &idx1);
    
int chamfer_cuda_backward(
  const at::Tensor &pc0, const at::Tensor &pc1, 
  const at::Tensor &idx0, const at::Tensor &idx1,
  at::Tensor &grad_dist0, at::Tensor &grad_dist1, 
  at::Tensor &grad_pc0, at::Tensor &grad_pc1);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &chamfer_cuda_forward, "Chamfer Distance (CUDA)");
  m.def("backward", &chamfer_cuda_backward, "Chamfer Distance (CUDA) Backward Grad");
}