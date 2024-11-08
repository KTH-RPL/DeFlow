import numpy as np
import pdb, json
import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv as spconv_core
# from easydict import EasyDict
import yaml
from torch_scatter import scatter_mean, scatter_add

spconv_core.constants.SPCONV_ALLOW_TF32 = True

import spconv.pytorch as spconv
import time
from spconv.utils import Point2VoxelCPU3d as VoxelGenerator

tv = None
try:
    import cumm.tensorview as tv
except:
    pass

from torch.autograd import Function
from torch.autograd.function import once_differentiable
import torch.cuda.amp as amp

_TORCH_CUSTOM_FWD = amp.custom_fwd(cast_inputs=torch.float16)
_TORCH_CUSTOM_BWD = amp.custom_bwd

# from typing import List, Tuple, Dict
from .decoder import Scene_Flow_Decoder


def conv1x1x1x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv4d(in_planes, out_planes, kernel_size=(1, 1, 1, 3), stride=stride,
                             padding=(0, 0, 0, 1), bias=False, indice_key=indice_key)


def conv1x1x1x3_dilated(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv4d(in_planes, out_planes, kernel_size=(1, 1, 1, 3),
                             stride=stride, padding=(0, 0, 0, 2), dilation=(1, 1, 1, 2),
                             bias=False, indice_key=indice_key)


def conv3x3x3x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv4d(in_planes, out_planes, kernel_size=(3, 3, 3, 1), stride=stride,
                             padding=(1, 1, 1, 0), bias=False, indice_key=indice_key)


def conv1x1x1x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv4d(in_planes, out_planes, kernel_size=(1, 1, 1, 1), stride=stride,
                             padding=0, bias=False, indice_key=indice_key)


def conv3x3x3x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv4d(in_planes, out_planes, kernel_size=(3, 3, 3, 3), stride=stride,
                             padding=(1, 1, 1, 1), bias=False, indice_key=indice_key)


norm_cfg = {
    # format: layer_type: (abbreviation, module)
    "BN": ("bn", nn.BatchNorm2d),
    "BN1d": ("bn1d", nn.BatchNorm1d),
    "GN": ("gn", nn.GroupNorm),
}


class Seperate_to_3D(nn.Module):
    def __init__(self, num_frames):
        super(Seperate_to_3D, self).__init__()
        self.num_frames = num_frames
        # self.return_pc1 = return_pc1

    def forward(self, sparse_4D_tensor):
        indices_4d = sparse_4D_tensor.indices
        features_4d = sparse_4D_tensor.features

        pc0_time_value = self.num_frames - 2

        mask_pc0 = (indices_4d[:, -1] == pc0_time_value)

        pc0_indices = indices_4d[mask_pc0][:, :-1]
        pc0_features = features_4d[mask_pc0]

        pc0_sparse_3D = sparse_4D_tensor.replace_feature(pc0_features)
        pc0_sparse_3D.spatial_shape = sparse_4D_tensor.spatial_shape[:-1]
        pc0_sparse_3D.indices = pc0_indices

        return pc0_sparse_3D


class SpatioTemporal_Decomposition_Block(nn.Module):
    def __init__(self, in_filters, mid_filters, out_filters, indice_key=None, down_key=None, pooling=False,
                 z_pooling=True, interact=False):
        super(SpatioTemporal_Decomposition_Block, self).__init__()

        self.pooling = pooling

        self.act = nn.LeakyReLU()

        self.spatial_conv_1 = conv3x3x3x1(in_filters, mid_filters, indice_key=indice_key + "bef")
        self.bn_s_1 = nn.BatchNorm1d(mid_filters)

        self.temporal_conv_1 = conv1x1x1x3(in_filters, mid_filters)
        self.bn_t_1 = nn.BatchNorm1d(mid_filters)

        self.fusion_conv_1 = conv1x1x1x1(mid_filters * 2 + in_filters, mid_filters, indice_key=indice_key + "1D")
        self.bn_fusion_1 = nn.BatchNorm1d(mid_filters)

        self.spatial_conv_2 = conv3x3x3x1(mid_filters, mid_filters, indice_key=indice_key + "bef")
        self.bn_s_2 = nn.BatchNorm1d(mid_filters)

        self.temporal_conv_2 = conv1x1x1x3(mid_filters, mid_filters)
        self.bn_t_2 = nn.BatchNorm1d(mid_filters)

        self.fusion_conv_2 = conv1x1x1x1(mid_filters * 3, out_filters, indice_key=indice_key + "1D")
        self.bn_fusion_2 = nn.BatchNorm1d(out_filters)

        if self.pooling:
            if z_pooling == True:
                self.pool = spconv.SparseConv4d(out_filters, out_filters, kernel_size=(2, 2, 2, 1), stride=(2, 2, 2, 1),
                                                indice_key=down_key, bias=False)
            else:
                self.pool = spconv.SparseConv4d(out_filters, out_filters, kernel_size=(2, 2, 1, 1), stride=(2, 2, 1, 1),
                                                indice_key=down_key, bias=False)

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # ST block
        S_feat_1 = self.spatial_conv_1(x)
        S_feat_1 = S_feat_1.replace_feature(self.bn_s_1(S_feat_1.features))
        S_feat_1 = S_feat_1.replace_feature(self.act(S_feat_1.features))

        T_feat_1 = self.temporal_conv_1(x)
        T_feat_1 = T_feat_1.replace_feature(self.bn_t_1(T_feat_1.features))
        T_feat_1 = T_feat_1.replace_feature(self.act(T_feat_1.features))

        ST_feat_1 = x.replace_feature(
            torch.cat([S_feat_1.features, T_feat_1.features, x.features], 1))  # residual까지 concate

        ST_feat_1 = self.fusion_conv_1(ST_feat_1)
        ST_feat_1 = ST_feat_1.replace_feature(self.bn_fusion_1(ST_feat_1.features))
        ST_feat_1 = ST_feat_1.replace_feature(self.act(ST_feat_1.features))

        # TS block
        S_feat_2 = self.spatial_conv_2(ST_feat_1)
        S_feat_2 = S_feat_2.replace_feature(self.bn_s_2(S_feat_2.features))
        S_feat_2 = S_feat_2.replace_feature(self.act(S_feat_2.features))

        T_feat_2 = self.temporal_conv_2(ST_feat_1)
        T_feat_2 = T_feat_2.replace_feature(self.bn_t_2(T_feat_2.features))
        T_feat_2 = T_feat_2.replace_feature(self.act(T_feat_2.features))

        ST_feat_2 = x.replace_feature(
            torch.cat([S_feat_2.features, T_feat_2.features, ST_feat_1.features], 1))  # residual까지 concate

        ST_feat_2 = self.fusion_conv_2(ST_feat_2)
        ST_feat_2 = ST_feat_2.replace_feature(self.bn_fusion_2(ST_feat_2.features))
        ST_feat_2 = ST_feat_2.replace_feature(self.act(ST_feat_2.features))

        if self.pooling:
            pooled = self.pool(ST_feat_2)
            return pooled, ST_feat_2
        else:
            return ST_feat_2



#SoTA Version
class SpatioTemporal_Deep_Coupling_Block(nn.Module):
    def __init__(self, in_filters, mid_filters, out_filters, indice_key=None, down_key=None, pooling=False,
                 z_pooling=True, interact=False):
        super(SpatioTemporal_Deep_Coupling_Block, self).__init__()

        self.pooling = pooling
        self.act = nn.LeakyReLU()

        # self.spatial_conv_1 = conv3x3x3x1(in_filters, mid_filters, indice_key=indice_key + "bef")
        self.spatial_conv_1 = conv3x3x3x1(in_filters, mid_filters)
        self.bn_s_1 = nn.BatchNorm1d(mid_filters)

        self.temporal_conv_1 = conv1x1x1x3(in_filters, mid_filters)
        self.bn_t_1 = nn.BatchNorm1d(mid_filters)

        self.temporal_conv_1_dilated = conv1x1x1x3_dilated(in_filters, mid_filters)
        self.bn_t_1_dilated = nn.BatchNorm1d(mid_filters)

        # SFSM_1
        self.temporal_fusion = conv1x1x1x1(mid_filters * 2, mid_filters)
        self.bn_t_fusion = nn.BatchNorm1d(mid_filters)
        self.temporal_fusion_sigmoid = nn.Sigmoid()

        # SFSM_2
        self.ts_fusion = conv1x1x1x1(mid_filters * 2, mid_filters)
        self.bn_ts_fusion = nn.BatchNorm1d(mid_filters)
        self.ts_fusion_sigmoid = nn.Sigmoid()

        # Temporal Gated
        self.temporal_gate_1_conv_1 = conv1x1x1x1(mid_filters, mid_filters * 2)
        self.temporal_gate_1_relu = nn.ReLU()
        self.temporal_gate_1_conv_2 = conv1x1x1x1(mid_filters * 2, mid_filters)
        self.temporal_gate_1_sigmoid = nn.Sigmoid()

        self.fusion_output = conv1x1x1x1(in_filters+mid_filters, out_filters)
        self.bn_output = nn.BatchNorm1d(out_filters)



        if self.pooling:
            if z_pooling == True:
                self.pool = spconv.SparseConv4d(out_filters, out_filters, kernel_size=(2, 2, 2, 1), stride=(2, 2, 2, 1),
                                                indice_key=down_key, bias=False)
            else:
                self.pool = spconv.SparseConv4d(out_filters, out_filters, kernel_size=(2, 2, 1, 1), stride=(2, 2, 1, 1),
                                                indice_key=down_key, bias=False)

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x_residual = x.features.clone()
        # S Feat
        S_feat_1 = self.spatial_conv_1(x)
        S_feat_1 = S_feat_1.replace_feature(self.bn_s_1(S_feat_1.features))
        S_feat_1 = S_feat_1.replace_feature(self.act(S_feat_1.features))

        # T Feat
        T_feat_1 = self.temporal_conv_1(x)
        T_feat_1 = T_feat_1.replace_feature(self.bn_t_1(T_feat_1.features))
        T_feat_1 = T_feat_1.replace_feature(self.act(T_feat_1.features))

        # T_dilated Feat
        T_feat_1_dilated = self.temporal_conv_1_dilated(x)
        T_feat_1_dilated = T_feat_1_dilated.replace_feature(self.bn_t_1_dilated(T_feat_1_dilated.features))
        T_feat_1_dilated = T_feat_1_dilated.replace_feature(self.act(T_feat_1_dilated.features))

        # SFSM_1
        T_feat = self.temporal_fusion(x.replace_feature(torch.cat([T_feat_1.features, T_feat_1_dilated.features], 1)))
        T_feat = T_feat.replace_feature(self.bn_t_fusion(T_feat.features))
        T_feat = T_feat.replace_feature(self.act(T_feat.features))
        T_Attention = self.temporal_fusion_sigmoid(T_feat.features)
        T_feat = T_feat.replace_feature(T_feat_1.features * T_Attention + T_feat_1_dilated.features * (1 - T_Attention))

        # Temporal Gated
        TG_Sptensor_1 = self.temporal_gate_1_conv_1(T_feat)
        TG_Sptensor_1 = TG_Sptensor_1.replace_feature(self.temporal_gate_1_relu(TG_Sptensor_1.features))
        TG_Sptensor_1 = self.temporal_gate_1_conv_2(TG_Sptensor_1)
        T_gate_1 = self.temporal_gate_1_sigmoid(TG_Sptensor_1.features)
        S_feat_1 = S_feat_1.replace_feature(S_feat_1.features + S_feat_1.features * T_gate_1)

        # SFSM_2
        TS_fused_feat = self.ts_fusion(x.replace_feature(torch.cat([S_feat_1.features, T_feat.features], 1)))
        TS_fused_feat = TS_fused_feat.replace_feature(self.bn_ts_fusion(TS_fused_feat.features))
        TS_fused_feat = TS_fused_feat.replace_feature(self.act(TS_fused_feat.features))
        TS_Attention = self.ts_fusion_sigmoid(TS_fused_feat.features)
        TS_fused_feat = TS_fused_feat.replace_feature(
            T_feat.features * TS_Attention + S_feat_1.features * (1 - TS_Attention))

        output = self.fusion_output(TS_fused_feat.replace_feature(torch.cat([TS_fused_feat.features, x_residual], 1)))
        output = output.replace_feature(self.bn_output(output.features))
        output = output.replace_feature(self.act(output.features))

        if self.pooling:
            pooled = self.pool(output)
            return pooled, output
        else:
            return output


class Network_4D(nn.Module):
    def __init__(self, in_channel=16, out_channel=16, model_size=16):
        super().__init__()

        # SpatioTemporal_Block = SpatioTemporal_Decomposition_Block
        SpatioTemporal_Block = SpatioTemporal_Deep_Coupling_Block

        self.model_size = model_size

        self.STDB_1_1_1 = SpatioTemporal_Block(in_channel, model_size, model_size, indice_key="st1_1",
                                               down_key='floor1')
        self.STDB_1_1_2 = SpatioTemporal_Block(model_size, model_size, model_size * 2, indice_key="st1_1",
                                               down_key='floor1', pooling=True)  # 512 512 32 -> 256 256 16

        self.STDB_2_1_1 = SpatioTemporal_Block(model_size * 2, model_size * 2, model_size * 2, indice_key="st2_1",
                                               down_key='floor2')
        self.STDB_2_1_2 = SpatioTemporal_Block(model_size * 2, model_size * 2, model_size * 4, indice_key="st2_1",
                                               down_key='floor2', pooling=True)  # 256 256 16 -> 128 128 8

        self.STDB_3_1_1 = SpatioTemporal_Block(model_size * 4, model_size * 4, model_size * 4, indice_key="st3_1",
                                               down_key='floor3')
        self.STDB_3_1_2 = SpatioTemporal_Block(model_size * 4, model_size * 4, model_size * 4, indice_key="st3_1",
                                               down_key='floor3', pooling=True)  # 128 128 8 -> 64 64 4

        self.STDB_4_1_1 = SpatioTemporal_Block(model_size * 4, model_size * 4, model_size * 4, indice_key="st4_1",
                                               down_key='floor4')
        self.STDB_4_1_2 = SpatioTemporal_Block(model_size * 4, model_size * 4, model_size * 4, indice_key="st4_1",
                                               down_key='floor4', pooling=True, z_pooling=False)  # 64 64 4 -> 64 64 4
        self.STDB_5_1_1 = SpatioTemporal_Block(model_size * 4, model_size * 4, model_size * 4, indice_key="st5_1")
        self.STDB_5_1_2 = SpatioTemporal_Block(model_size * 4, model_size * 4, model_size * 4, indice_key="st5_1")
        self.up_subm_5 = spconv.SparseInverseConv4d(model_size * 4, model_size * 4, kernel_size=(2, 2, 1, 1),
                                                    indice_key='floor4', bias=False)  # zpooling false

        self.STDB_4_2_1 = SpatioTemporal_Block(model_size * 8, model_size * 8, model_size * 4, indice_key="st4_2")
        self.up_subm_4 = spconv.SparseInverseConv4d(model_size * 4, model_size * 4, kernel_size=(2, 2, 2, 1),
                                                    indice_key='floor3', bias=False)

        self.STDB_3_2_1 = SpatioTemporal_Block(model_size * 8, model_size * 8, model_size * 4, indice_key="st3_2")
        self.up_subm_3 = spconv.SparseInverseConv4d(model_size * 4, model_size * 4, kernel_size=(2, 2, 2, 1),
                                                    indice_key='floor2', bias=False)

        self.STDB_2_2_1 = SpatioTemporal_Block(model_size * 8, model_size * 4, model_size * 4, indice_key="st_2_2")
        self.up_subm_2 = spconv.SparseInverseConv4d(model_size * 4, model_size * 2, kernel_size=(2, 2, 2, 1),
                                                    indice_key='floor1', bias=False)

        self.STDB_1_2_1 = SpatioTemporal_Block(model_size * 4, model_size * 2, out_channel, indice_key="st_1_2")

    def forward(self, sp_tensor):
        sp_tensor = self.STDB_1_1_1(sp_tensor)
        down_2, skip_1 = self.STDB_1_1_2(sp_tensor)

        down_2 = self.STDB_2_1_1(down_2)
        down_3, skip_2 = self.STDB_2_1_2(down_2)

        down_3 = self.STDB_3_1_1(down_3)
        down_4, skip_3 = self.STDB_3_1_2(down_3)

        down_4 = self.STDB_4_1_1(down_4)
        down_5, skip_4 = self.STDB_4_1_2(down_4)

        down_5 = self.STDB_5_1_1(down_5)
        down_5 = self.STDB_5_1_2(down_5)

        up_4 = self.up_subm_5(down_5)
        up_4 = up_4.replace_feature(torch.cat((up_4.features, skip_4.features), 1))
        up_4 = self.STDB_4_2_1(up_4)

        up_3 = self.up_subm_4(up_4)
        up_3 = up_3.replace_feature(torch.cat((up_3.features, skip_3.features), 1))
        up_3 = self.STDB_3_2_1(up_3)

        up_2 = self.up_subm_3(up_3)
        up_2 = up_2.replace_feature(torch.cat((up_2.features, skip_2.features), 1))
        up_2 = self.STDB_2_2_1(up_2)

        up_1 = self.up_subm_2(up_2)
        up_1 = up_1.replace_feature(torch.cat((up_1.features, skip_1.features), 1))
        up_1 = self.STDB_1_2_1(up_1)

        return up_1


class mambaflow_head(nn.Module):
    def __init__(self, voxel_feat_dim: int = 96, point_feat_dim: int = 32, use_decoder=False):
        super().__init__()
        self.input_dim = voxel_feat_dim + point_feat_dim
        self.offset_dim = self.input_dim // 2
        self.decoder_dim = self.input_dim + self.offset_dim
        self.use_decoder = use_decoder

        if self.use_decoder:
            self.offset_encoder = nn.Linear(3, self.offset_dim)
            self.mambaflow_decoder = Scene_Flow_Decoder(d_model=self.input_dim, n_layer=1, d_s=16, d_conv=4, expand=2)
            self.decoder = nn.Sequential(
                nn.Linear(self.decoder_dim, self.decoder_dim // 2), nn.GELU(),
                nn.Linear(self.decoder_dim // 2, 3))
        else:
            self.offset_encoder = nn.Linear(3, self.offset_dim)
            self.decoder = nn.Sequential(
                nn.Linear(self.decoder_dim, self.decoder_dim // 2), nn.GELU(),
                nn.Linear(self.decoder_dim // 2, 3))


    def forward_single(self, voxel_feat, voxel_coords, point_feat, point_offsets):

        voxel_to_point_feat = voxel_feat[:, voxel_coords[:, 2], voxel_coords[:, 1], voxel_coords[:, 0]].T

        if self.use_decoder:
            concated_point_feat = torch.cat([point_feat, voxel_to_point_feat], dim=-1)
            point_offsets_feature = self.offset_encoder(point_offsets)
            
            point_offsets_feature, concated_point_feat, sorted_indices = z_order_sort(voxel_coords, point_offsets_feature,
                                                                               concated_point_feat)

            concated_point_feat = self.mambaflow_decoder(concated_point_feat.unsqueeze(0),
                                                              point_offsets_feature.unsqueeze(0)).squeeze(0)

            concated_point_feat = torch.cat([concated_point_feat, point_offsets_feature], dim=-1)

            concated_point_feat = restore_original_order(torch.cat([concated_point_feat, point_offsets_feature], dim=-1), sorted_indices)

            flow = self.decoder(concated_point_feat)
        else:
            concated_point_feat = torch.cat([point_feat, voxel_to_point_feat], dim=-1)
            point_offsets_feature = self.offset_encoder(point_offsets)
            concated_point_feat = torch.cat([concated_point_feat, point_offsets_feature], dim=-1)
            flow = self.decoder(concated_point_feat)


        return flow

    def forward(self, sparse_tensor, voxelizer_infos, point_feats_lst):
        voxel_feats = sparse_tensor.dense()
        flow_outputs = []
        batch_idx = 0
        for voxelizer_info in voxelizer_infos:
            voxel_coords = voxelizer_info["voxel_coords"]
            voxel_feat = voxel_feats[batch_idx, :]
            point_feat = point_feats_lst[batch_idx]
            point_offsets = voxelizer_info["point_offsets"]
            flow = self.forward_single(voxel_feat, voxel_coords, point_feat, point_offsets)
            batch_idx += 1
            flow_outputs.append(flow)

        return flow_outputs

def get_point_features(pc0_voxel_feats_dict, batch_idx, voxel_coords):
    voxel_features_dict = pc0_voxel_feats_dict[batch_idx]

    N = voxel_coords.shape[0]
    first_feature = next(iter(voxel_features_dict.values()))
    C = len(first_feature)
    dtype = torch.tensor(first_feature).dtype
    point_features = torch.zeros((N, C), dtype=dtype)

    for i, coord in enumerate(voxel_coords):
        # 将坐标转换为元组,以便用作字典的键
        coord_tuple = tuple(coord.tolist())
        # 如果坐标在字典中存在,则提取对应的特征
        if coord_tuple in voxel_features_dict:
            feature = voxel_features_dict[coord_tuple]
            point_features[i] = feature.clone().detach().to(dtype=dtype)
        else:
            print(coord_tuple)

            raise ValueError(1)

    return point_features

def expand_bits(x):
    x = (x | (x << 16)) & 0x030000FF
    x = (x | (x << 8)) & 0x0300F00F
    x = (x | (x << 4)) & 0x030C30C3
    x = (x | (x << 2)) & 0x09249249
    return x


# 排序代码
def morton_code(z, y, x):
    return expand_bits(z) | (expand_bits(y) << 1) | (expand_bits(x) << 2)


def z_order_sort(voxel_coords, point_offsets, voxel_to_point_feat):
    device = voxel_coords.device

    # Calculate Morton codes
    morton_codes = morton_code(voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2])

    # Get sorting indices
    sorted_indices = torch.argsort(morton_codes)

    # Sort all tensors
    sorted_point_offsets = point_offsets[sorted_indices]
    sorted_voxel_to_point_feat = voxel_to_point_feat[sorted_indices]

    return sorted_point_offsets, sorted_voxel_to_point_feat, sorted_indices


def restore_original_order(sorted_voxel_to_point_feat, sorted_indices):
    device = sorted_voxel_to_point_feat.device
    # Create inverse indices
    inverse_indices = torch.zeros_like(sorted_indices, device=device)
    inverse_indices[sorted_indices] = torch.arange(sorted_indices.size(0), device=device)

    # Restore original order
    original_voxel_to_point_feat = sorted_voxel_to_point_feat[inverse_indices]

    return original_voxel_to_point_feat
