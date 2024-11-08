import torch
import torch.nn as nn

from .make_voxels import HardVoxelizer, DynamicVoxelizer
from .process_voxels import PillarFeatureNet, DynamicPillarFeatureNet, DynamicPillarFeatureNet_flow4D
from .scatter import PointPillarsScatter

import spconv as spconv_core

spconv_core.constants.SPCONV_ALLOW_TF32 = True
import spconv.pytorch as spconv


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
            feat_channels=(feat_channels,),
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
            feat_channels=(feat_channels,),
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


class DynamicEmbedder_4D(nn.Module):

    def __init__(self, voxel_size, pseudo_image_dims, point_cloud_range,
                 feat_channels: int) -> None:
        super().__init__()
        self.voxelizer = DynamicVoxelizer(voxel_size=voxel_size,
                                          point_cloud_range=point_cloud_range)
        self.feature_net = DynamicPillarFeatureNet_flow4D(
            in_channels=3,
            feat_channels=(feat_channels,),
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            mode='avg')
        self.scatter = PointPillarsScatter(in_channels=feat_channels,
                                           output_shape=pseudo_image_dims)

        self.voxel_spatial_shape = pseudo_image_dims

    def forward(self, input_dict) -> torch.Tensor:
        voxel_feats_list = []
        voxel_coors_list = []
        batch_index = 0

        frame_keys = sorted([key for key in input_dict.keys() if key.startswith('pc_m')], reverse=True)
        frame_keys += ['pc0s', 'pc1s']

        pc0_point_feats_lst = []

        for time_index, frame_key in enumerate(frame_keys):
            pc = input_dict[frame_key]
            voxel_info_list = self.voxelizer(pc)

            voxel_feats_list_batch = []
            voxel_coors_list_batch = []

            for batch_index, voxel_info_dict in enumerate(voxel_info_list):
                points = voxel_info_dict['points']
                coordinates = voxel_info_dict['voxel_coords']
                voxel_feats, voxel_coors, point_feats = self.feature_net(points, coordinates)

                if frame_key == 'pc0s':
                    pc0_point_feats_lst.append(point_feats)

                batch_indices = torch.full((voxel_coors.size(0), 1), batch_index, dtype=torch.long,
                                           device=voxel_coors.device)
                voxel_coors_batch = torch.cat([batch_indices, voxel_coors[:, [2, 1, 0]]], dim=1)

                voxel_feats_list_batch.append(voxel_feats)
                voxel_coors_list_batch.append(voxel_coors_batch)

            voxel_feats_sp = torch.cat(voxel_feats_list_batch, dim=0)
            coors_batch_sp = torch.cat(voxel_coors_list_batch, dim=0).to(dtype=torch.int32)

            time_dimension = torch.full((coors_batch_sp.shape[0], 1), time_index, dtype=torch.int32, device='cuda')
            coors_batch_sp_4d = torch.cat((coors_batch_sp, time_dimension), dim=1)

            voxel_feats_list.append(voxel_feats_sp)
            voxel_coors_list.append(coors_batch_sp_4d)

            if frame_key == 'pc0s':
                pc0s_3dvoxel_infos_lst = voxel_info_list
                pc0s_num_voxels = voxel_feats_sp.shape[0]

        all_voxel_feats_sp = torch.cat(voxel_feats_list, dim=0)
        all_coors_batch_sp_4d = torch.cat(voxel_coors_list, dim=0)

        sparse_tensor_4d = spconv.SparseConvTensor(all_voxel_feats_sp.contiguous(), all_coors_batch_sp_4d.contiguous(),
                                                   self.voxel_spatial_shape, int(batch_index + 1))

        output = {
            '4d_tensor': sparse_tensor_4d,
            'pc0_3dvoxel_infos_lst': pc0s_3dvoxel_infos_lst,
            'pc0_point_feats_lst': pc0_point_feats_lst,
            'pc0_num_voxels': pc0s_num_voxels
        }

        return output
