#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

typedef enum { SUM = 0, MEAN = 1, MAX = 2 } reduce_t;

void DynamicVoxelizeForwardCUDAKernelLauncher(
    const at::Tensor &points, at::Tensor &coors,
    const std::vector<float> voxel_size, const std::vector<float> coors_range,
    const int NDim = 3);

void dynamic_voxelize_forward_cuda(const at::Tensor &points, at::Tensor &coors,
                                   const std::vector<float> voxel_size,
                                   const std::vector<float> coors_range,
                                   const int NDim) {
  DynamicVoxelizeForwardCUDAKernelLauncher(points, coors, voxel_size,
                                           coors_range, NDim);
};

void dynamic_voxelize_forward_impl(const at::Tensor &points, at::Tensor &coors,
                                   const std::vector<float> voxel_size,
                                   const std::vector<float> coors_range,
                                   const int NDim);


int HardVoxelizeForwardCUDAKernelLauncher(
    const at::Tensor &points, at::Tensor &voxels, at::Tensor &coors,
    at::Tensor &num_points_per_voxel, const std::vector<float> voxel_size,
    const std::vector<float> coors_range, const int max_points,
    const int max_voxels, const int NDim = 3);
    
int hard_voxelize_forward_cuda(const at::Tensor &points, at::Tensor &voxels,
                               at::Tensor &coors,
                               at::Tensor &num_points_per_voxel,
                               const std::vector<float> voxel_size,
                               const std::vector<float> coors_range,
                               const int max_points, const int max_voxels,
                               const int NDim) {
  return HardVoxelizeForwardCUDAKernelLauncher(
      points, voxels, coors, num_points_per_voxel, voxel_size, coors_range,
      max_points, max_voxels, NDim);
};

int hard_voxelize_forward_impl(const at::Tensor &points, at::Tensor &voxels,
                               at::Tensor &coors,
                               at::Tensor &num_points_per_voxel,
                               const std::vector<float> voxel_size,
                               const std::vector<float> coors_range,
                               const int max_points, const int max_voxels,
                               const int NDim);

int nondeterministic_hard_voxelize_forward_impl(
    const at::Tensor &points, at::Tensor &voxels, at::Tensor &coors,
    at::Tensor &num_points_per_voxel, const std::vector<float> voxel_size,
    const std::vector<float> coors_range, const int max_points,
    const int max_voxels, const int NDim);

REGISTER_DEVICE_IMPL(hard_voxelize_forward_impl, CUDA,
                     hard_voxelize_forward_cuda);
REGISTER_DEVICE_IMPL(dynamic_voxelize_forward_impl, CUDA,
                     dynamic_voxelize_forward_cuda);


std::vector<at::Tensor> DynamicPointToVoxelForwardCUDAKernelLauncher(
    const at::Tensor &feats, const at::Tensor &coors,
    const reduce_t reduce_type);


std::vector<torch::Tensor> dynamic_point_to_voxel_forward_cuda(
    const torch::Tensor &feats, const torch::Tensor &coors,
    const reduce_t reduce_type) {
  return DynamicPointToVoxelForwardCUDAKernelLauncher(feats, coors,
                                                      reduce_type);
};

void DynamicPointToVoxelBackwardCUDAKernelLauncher(
    at::Tensor &grad_feats, const at::Tensor &grad_reduced_feats,
    const at::Tensor &feats, const at::Tensor &reduced_feats,
    const at::Tensor &coors_map, const at::Tensor &reduce_count,
    const reduce_t reduce_type);

void dynamic_point_to_voxel_backward_cuda(
    torch::Tensor &grad_feats, const torch::Tensor &grad_reduced_feats,
    const torch::Tensor &feats, const torch::Tensor &reduced_feats,
    const torch::Tensor &coors_idx, const torch::Tensor &reduce_count,
    const reduce_t reduce_type) {
  DynamicPointToVoxelBackwardCUDAKernelLauncher(grad_feats, grad_reduced_feats,
                                                feats, reduced_feats, coors_idx,
                                                reduce_count, reduce_type);
};

std::vector<torch::Tensor> dynamic_point_to_voxel_forward_impl(
    const torch::Tensor &feats, const torch::Tensor &coors,
    const reduce_t reduce_type);

void dynamic_point_to_voxel_backward_impl(
    torch::Tensor &grad_feats, const torch::Tensor &grad_reduced_feats,
    const torch::Tensor &feats, const torch::Tensor &reduced_feats,
    const torch::Tensor &coors_idx, const torch::Tensor &reduce_count,
    const reduce_t reduce_type);

REGISTER_DEVICE_IMPL(dynamic_point_to_voxel_forward_impl, CUDA,
                     dynamic_point_to_voxel_forward_cuda);
REGISTER_DEVICE_IMPL(dynamic_point_to_voxel_backward_impl, CUDA,
                     dynamic_point_to_voxel_backward_cuda);