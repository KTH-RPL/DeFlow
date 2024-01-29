import torch
import torch.nn as nn

from .make_voxels import HardVoxelizer, DynamicVoxelizer
from .process_voxels import PillarFeatureNet, DynamicPillarFeatureNet
from .scatter import PointPillarsScatter


class HardEmbedder(nn.Module):

    def __init__(self,
                 voxel_size=(0.2, 0.2, 4),
                 pseudo_image_dims=(350, 350),
                 point_cloud_range=(-35, -35, -3, 35, 35, 1),
                 max_points_per_voxel=128,
                 feat_channels=64) -> None:
        super().__init__()
        self.voxelizer = HardVoxelizer(
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            max_points_per_voxel=max_points_per_voxel)
        self.feature_net = PillarFeatureNet(
            in_channels=3,
            feat_channels=(feat_channels, ),
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size)
        self.scatter = PointPillarsScatter(in_channels=feat_channels,
                                           output_shape=pseudo_image_dims)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        assert isinstance(
            points,
            torch.Tensor), f"points must be a torch.Tensor, got {type(points)}"

        output_voxels, output_voxel_coords, points_per_voxel = self.voxelizer(
            points)
        output_features = self.feature_net(output_voxels, points_per_voxel,
                                           output_voxel_coords)
        pseudoimage = self.scatter(output_features, output_voxel_coords)

        return pseudoimage


class DynamicEmbedder(nn.Module):

    def __init__(self, voxel_size, pseudo_image_dims, point_cloud_range,
                 feat_channels: int) -> None:
        super().__init__()
        self.voxelizer = DynamicVoxelizer(voxel_size=voxel_size,
                                          point_cloud_range=point_cloud_range)
        self.feature_net = DynamicPillarFeatureNet(
            in_channels=3,
            feat_channels=(feat_channels, ),
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            mode='avg')
        self.scatter = PointPillarsScatter(in_channels=feat_channels,
                                           output_shape=pseudo_image_dims)

    def forward(self, points: torch.Tensor) -> torch.Tensor:

        # List of points and coordinates for each batch
        voxel_info_list = self.voxelizer(points)

        pseudoimage_lst = []
        for voxel_info_dict in voxel_info_list:
            points = voxel_info_dict['points']
            coordinates = voxel_info_dict['voxel_coords']
            voxel_feats, voxel_coors = self.feature_net(points, coordinates)
            pseudoimage = self.scatter(voxel_feats, voxel_coors)
            pseudoimage_lst.append(pseudoimage)
        # Concatenate the pseudoimages along the batch dimension
        return torch.cat(pseudoimage_lst, dim=0), voxel_info_list
