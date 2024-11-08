from typing import List, Tuple, Dict
import copy
import time
from torch.utils.checkpoint import checkpoint
from einops import repeat, einsum, rearrange

from torch import einsum
# from torch.utils.checkpoint import checkpoint_sequential
from mamba_ssm import Mamba
from typing import Optional
from torch import Tensor

from mamba_ssm import Mamba
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

SPLIT_BATCH_SIZE = 512

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None

from .FlowSSM import FlowSSM_layer


class MMHeadDecoder(nn.Module):

    def __init__(self, pseudoimage_channels: int = 64):
        super().__init__()

        self.offset_encoder = nn.Linear(3, 128)

        # FIXME: figure out how to set nheads and num_layers properly
        # ref: https://pytorch.org/docs/stable/generated/torch.nn.TransformerDecoder.html
        # https://pytorch.org/docs/stable/generated/torch.nn.TransformerDecoderLayer.html
        transform_decoder_layers = nn.TransformerDecoderLayer(d_model=128, nhead=4)
        self.pts_off_transformer = nn.TransformerDecoder(transform_decoder_layers, num_layers=4)

        self.decoder = nn.Sequential(
            nn.Linear(pseudoimage_channels * 2, 32), nn.GELU(),
            nn.Linear(32, 3))

    def forward_single(self, before_pseudoimage: torch.Tensor,
                       after_pseudoimage: torch.Tensor,
                       point_offsets: torch.Tensor,
                       voxel_coords: torch.Tensor) -> torch.Tensor:
        voxel_coords = voxel_coords.long()
        # assert (voxel_coords[:, 0] == 0).all(), "Z index must be 0"

        # Voxel coords are Z, Y, X, and the pseudoimage is Channel, Y, X
        # I have confirmed via visualization that these coordinates are correct.
        after_voxel_vectors = after_pseudoimage[:, voxel_coords[:, 1],
                              voxel_coords[:, 2]].T
        before_voxel_vectors = before_pseudoimage[:, voxel_coords[:, 1],
                               voxel_coords[:, 2]].T

        # [N, 64] [N, 64] -> [N, 128]
        concatenated_vectors = torch.cat([before_voxel_vectors, after_voxel_vectors], dim=1)

        # [N, 128] [N, 128] -> [N, 1, 128]
        voxel_feature = concatenated_vectors.unsqueeze(1)
        point_offsets_feature = self.offset_encoder(point_offsets).unsqueeze(1)
        concatenated_feature = torch.zeros_like(voxel_feature)

        for spilt_range in range(0, concatenated_feature.shape[0], SPLIT_BATCH_SIZE):
            concatenated_feature[spilt_range:spilt_range + SPLIT_BATCH_SIZE] = self.pts_off_transformer(
                voxel_feature[spilt_range:spilt_range + SPLIT_BATCH_SIZE],
                point_offsets_feature[spilt_range:spilt_range + SPLIT_BATCH_SIZE]
            )

        flow = self.decoder(concatenated_feature.squeeze(1))
        return flow

    def forward(
            self, before_pseudoimages: torch.Tensor,
            after_pseudoimages: torch.Tensor,
            voxelizer_infos: List[Dict[str,
            torch.Tensor]]) -> List[torch.Tensor]:

        flow_results = []
        for before_pseudoimage, after_pseudoimage, voxelizer_info in zip(
                before_pseudoimages, after_pseudoimages, voxelizer_infos):
            point_offsets = voxelizer_info["point_offsets"]
            voxel_coords = voxelizer_info["voxel_coords"]
            flow = self.forward_single(before_pseudoimage, after_pseudoimage,
                                       point_offsets, voxel_coords)
            flow_results.append(flow)
        return flow_results


class LinearDecoder(nn.Module):

    def __init__(self, pseudoimage_channels: int = 64):
        super().__init__()

        self.offset_encoder = nn.Linear(3, 128)

        self.decoder = nn.Sequential(
            nn.Linear(pseudoimage_channels * 4, 32), nn.GELU(),
            nn.Linear(32, 3))

    def forward_single(self, before_pseudoimage: torch.Tensor,
                       after_pseudoimage: torch.Tensor,
                       point_offsets: torch.Tensor,
                       voxel_coords: torch.Tensor) -> torch.Tensor:
        voxel_coords = voxel_coords.long()
        # assert (voxel_coords[:, 0] == 0).all(), "Z index must be 0"

        # Voxel coords are Z, Y, X, and the pseudoimage is Channel, Y, X
        # I have confirmed via visualization that these coordinates are correct.
        after_voxel_vectors = after_pseudoimage[:, voxel_coords[:, 1],
                              voxel_coords[:, 2]].T
        before_voxel_vectors = before_pseudoimage[:, voxel_coords[:, 1],
                               voxel_coords[:, 2]].T

        # [N, 64] [N, 64] -> [N, 128]
        concatenated_vectors = torch.cat([before_voxel_vectors, after_voxel_vectors], dim=1)

        # [N, 3] -> [N, 128]
        point_offsets_feature = self.offset_encoder(point_offsets)

        flow = self.decoder(torch.cat([concatenated_vectors, point_offsets_feature], dim=1))
        return flow

    def forward(
            self, before_pseudoimages: torch.Tensor,
            after_pseudoimages: torch.Tensor,
            voxelizer_infos: List[Dict[str,
            torch.Tensor]]) -> List[torch.Tensor]:
        flow_results = []
        for before_pseudoimage, after_pseudoimage, voxelizer_info in zip(
                before_pseudoimages, after_pseudoimages, voxelizer_infos):
            point_offsets = voxelizer_info["point_offsets"]
            voxel_coords = voxelizer_info["voxel_coords"]
            flow = self.forward_single(before_pseudoimage, after_pseudoimage,
                                       point_offsets, voxel_coords)
            flow_results.append(flow)
        return flow_results


# from https://github.com/weiyithu/PV-RAFT/blob/main/model/update.py
class ConvGRU(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv1d(input_dim + hidden_dim, hidden_dim, 1)
        self.convr = nn.Conv1d(input_dim + hidden_dim, hidden_dim, 1)
        self.convq = nn.Conv1d(input_dim + hidden_dim, hidden_dim, 1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        rh_x = torch.cat([r * h, x], dim=1)
        q = torch.tanh(self.convq(rh_x))

        h = (1 - z) * h + z * q
        return h


class ConvGRUDecoder(nn.Module):

    def __init__(self, pseudoimage_channels: int = 64, num_iters: int = 4):
        super().__init__()

        self.offset_encoder = nn.Linear(3, 64)

        # NOTE: voxel feature is hidden input, point offset is input, check paper's Fig. 3
        self.gru = ConvGRU(input_dim=64, hidden_dim=pseudoimage_channels * 2)

        self.decoder = nn.Sequential(
            nn.Linear(pseudoimage_channels * 3, 32), nn.GELU(),
            nn.Linear(32, 3))
        self.num_iters = num_iters

    def forward_single(self, before_pseudoimage: torch.Tensor,
                       after_pseudoimage: torch.Tensor,
                       point_offsets: torch.Tensor,
                       voxel_coords: torch.Tensor) -> torch.Tensor:
        voxel_coords = voxel_coords.long()
        # assert (voxel_coords[:, 0] == 0).all(), "Z index must be 0"

        # Voxel coords are Z, Y, X, and the pseudoimage is Channel, Y, X
        # I have confirmed via visualization that these coordinates are correct.
        after_voxel_vectors = after_pseudoimage[:, voxel_coords[:, 1],
                              voxel_coords[:, 2]].T
        before_voxel_vectors = before_pseudoimage[:, voxel_coords[:, 1],
                               voxel_coords[:, 2]].T

        # [N, 64] [N, 64] -> [N, 128]
        concatenated_vectors = torch.cat([before_voxel_vectors, after_voxel_vectors], dim=1)

        # [N, 3] -> [N, 64]
        point_offsets_feature = self.offset_encoder(point_offsets)

        # [N, 128] -> [N, 128, 1]
        concatenated_vectors = concatenated_vectors.unsqueeze(2)

        for itr in range(self.num_iters):
            concatenated_vectors = self.gru(concatenated_vectors, point_offsets_feature.unsqueeze(2))

        flow = self.decoder(torch.cat([concatenated_vectors.squeeze(2), point_offsets_feature], dim=1))
        return flow

    def forward(
            self, before_pseudoimages: torch.Tensor,
            after_pseudoimages: torch.Tensor,
            voxelizer_infos: List[Dict[str,
            torch.Tensor]]) -> List[torch.Tensor]:

        flow_results = []
        for before_pseudoimage, after_pseudoimage, voxelizer_info in zip(
                before_pseudoimages, after_pseudoimages, voxelizer_infos):
            point_offsets = voxelizer_info["point_offsets"]
            voxel_coords = voxelizer_info["voxel_coords"]
            flow = self.forward_single(before_pseudoimage, after_pseudoimage,
                                       point_offsets, voxel_coords)
            flow_results.append(flow)
        return flow_results

class Scene_Flow_Decoder(nn.Module):
    def __init__(self, d_model=256, n_layer=6, d_s=128, d_conv=4, expand=4):
        super(Scene_Flow_Decoder, self).__init__()
        self.layers = n_layer
        self.mamba_dec = FlowSSM_layer(d_model=d_model, d_state=d_s, expand=expand, d_conv=d_conv)

    def forward(self, x_0, flow_features, h=None):
        for _ in range(self.layers):
            x_0 = self.mamba_dec(x_0, flow_features, h)

        return x_0
