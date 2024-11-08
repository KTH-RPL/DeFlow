import torch
import torch.nn as nn
from mmcv.ops import Voxelization
from typing import List, Tuple


class HardVoxelizer(nn.Module):

    def __init__(self, voxel_size, point_cloud_range,
                 max_points_per_voxel: int):
        super().__init__()
        assert max_points_per_voxel > 0, f"max_points_per_voxel must be > 0, got {max_points_per_voxel}"

        self.voxelizer = Voxelization(voxel_size,
                                      point_cloud_range,
                                      max_points_per_voxel,
                                      deterministic=False)

    def forward(self, points: torch.Tensor):
        assert isinstance(
            points,
            torch.Tensor), f"points must be a torch.Tensor, got {type(points)}"
        not_nan_mask = ~torch.isnan(points).any(dim=2)
        return {"voxel_coords": self.voxelizer(points[not_nan_mask])}


class DynamicVoxelizer(nn.Module):

    def __init__(self, voxel_size, point_cloud_range):
        super().__init__()
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.voxelizer = Voxelization(voxel_size,
                                      point_cloud_range,
                                      max_num_points=-1)

    def _get_point_offsets(self, points: torch.Tensor,
                           voxel_coords: torch.Tensor):

        point_cloud_range = torch.tensor(self.point_cloud_range,
                                         dtype=points.dtype,
                                         device=points.device)
        min_point = point_cloud_range[:3]
        voxel_size = torch.tensor(self.voxel_size,
                                  dtype=points.dtype,
                                  device=points.device)

        # Voxel coords are in the form Z, Y, X :eyeroll:, convert to X, Y, Z
        voxel_coords = voxel_coords[:, [2, 1, 0]]

        # Offsets are computed relative to min point
        voxel_centers = voxel_coords * voxel_size + min_point + voxel_size / 2

        return points - voxel_centers

    def forward(
            self,
            points: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:

        batch_results = []
        for batch_idx in range(len(points)):
            batch_points = points[batch_idx]
            valid_point_idxes = torch.arange(batch_points.shape[0],
                                             device=batch_points.device)
            not_nan_mask = ~torch.isnan(batch_points).any(dim=1)
            batch_non_nan_points = batch_points[not_nan_mask]
            valid_point_idxes = valid_point_idxes[not_nan_mask]
            batch_voxel_coords = self.voxelizer(batch_non_nan_points)
            # If any of the coords are -1, then the point is not in the voxel grid and should be discarded
            batch_voxel_coords_mask = (batch_voxel_coords != -1).all(dim=1)

            valid_batch_voxel_coords = batch_voxel_coords[
                batch_voxel_coords_mask]
            valid_batch_non_nan_points = batch_non_nan_points[
                batch_voxel_coords_mask]
            valid_point_idxes = valid_point_idxes[batch_voxel_coords_mask]

            point_offsets = self._get_point_offsets(valid_batch_non_nan_points,
                                                    valid_batch_voxel_coords)

            result_dict = {
                "points": valid_batch_non_nan_points,
                "voxel_coords": valid_batch_voxel_coords,
                "point_idxes": valid_point_idxes,
                "point_offsets": point_offsets
            }

            batch_results.append(result_dict)
        return batch_results