import torch
import torch.nn as nn
from typing import List, Tuple, Dict


class FastFlowDecoder(nn.Module):

    def __init__(self, pseudoimage_channels: int = 64):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(pseudoimage_channels * 2 + 3, 32), nn.GELU(),
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
        concatenated_vectors = torch.cat(
            [before_voxel_vectors, after_voxel_vectors, point_offsets], dim=1)

        flow = self.decoder(concatenated_vectors)
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


class ConvWithNorms(nn.Module):

    def __init__(self, in_num_channels: int, out_num_channels: int,
                 kernel_size: int, stride: int, padding: int):
        super().__init__()
        self.conv = nn.Conv2d(in_num_channels, out_num_channels, kernel_size,
                              stride, padding)
        self.batchnorm = nn.BatchNorm2d(out_num_channels)
        self.nonlinearity = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_res = self.conv(x)
        if conv_res.shape[2] == 1 and conv_res.shape[3] == 1:
            # This is a hack to get around the fact that batchnorm doesn't support
            # 1x1 convolutions
            batchnorm_res = conv_res
        else:
            batchnorm_res = self.batchnorm(conv_res)
        return self.nonlinearity(batchnorm_res)


class FastFlowDecoderStepDown(FastFlowDecoder):

    def __init__(self, voxel_pillar_size: Tuple[float, float],
                 num_stepdowns: int) -> None:
        super().__init__(pseudoimage_channels=16)
        assert num_stepdowns > 0, "stepdown_factor must be positive"
        self.num_stepdowns = num_stepdowns
        self.voxel_pillar_size = voxel_pillar_size
        assert len(voxel_pillar_size) == 2, "voxel_pillar_size must be 2D"

        self.pseudoimage_stepdown_head = nn.ModuleList()
        # Build convolutional pseudoimage stepdown head
        for i in range(num_stepdowns):
            if i == 0:
                in_channels = 64
            else:
                in_channels = 16
            out_channels = 16
            self.pseudoimage_stepdown_head.append(
                ConvWithNorms(in_channels, out_channels, 3, 2, 1))

    def _plot_pseudoimage(self, pseudoimage: torch.Tensor,
                          point_offsets: torch.Tensor,
                          voxel_coords: torch.Tensor, points: torch.Tensor,
                          pillar_size: Tuple[float, float]):
        assert len(pillar_size) == 2, "pillar_size must be 2D"

        import matplotlib.pyplot as plt
        import numpy as np

        plot_offset_bc_imshow_stupid = -0.5

        full_res_before_pseudoimage_np = pseudoimage.detach().cpu().numpy(
        ).sum(0)
        voxel_coords_np = voxel_coords.detach().cpu().numpy()
        offsets_np = point_offsets.detach().cpu().numpy()
        points_np = points.detach().cpu().numpy()
        plt.title(f"Full res: {full_res_before_pseudoimage_np.shape}")

        def _raw_points_to_scaled_points(raw_points: np.ndarray) -> np.ndarray:
            scaled_points = raw_points.copy()
            scaled_points[:, 0] /= pillar_size[0]
            scaled_points[:, 1] /= pillar_size[1]
            scaled_points[:, :2] += pseudoimage.shape[2] // 2
            return scaled_points

        scaled_points_np = _raw_points_to_scaled_points(points_np)

        # Plot pseudoimage and points from voxel plus offset info

        def _offset_points_to_scaled_points(offsets_np, voxel_coords_np):
            offsets_np = offsets_np.copy()
            # Scale voxel offsets to be in scale with the voxel size
            offsets_np[:, 0] /= pillar_size[0]
            offsets_np[:, 1] /= pillar_size[1]

            voxel_centers_np = voxel_coords_np[:, [2, 1]].astype(
                np.float32) + 0.5

            point_positions_np = voxel_centers_np + offsets_np[:, :2]

            voxel_centers_np = voxel_coords_np[:, [2, 1]].astype(
                np.float32) + 0.5

            point_positions_np = voxel_centers_np + offsets_np[:, :2]

            return voxel_centers_np, point_positions_np

        voxel_centers_np, point_positions_np = _offset_points_to_scaled_points(
            offsets_np, voxel_coords_np)

        plt.scatter(scaled_points_np[:, 0] + plot_offset_bc_imshow_stupid,
                    scaled_points_np[:, 1] + plot_offset_bc_imshow_stupid,
                    s=1,
                    marker="x",
                    color='green')

        plt.scatter(point_positions_np[:, 0] + plot_offset_bc_imshow_stupid,
                    point_positions_np[:, 1] + plot_offset_bc_imshow_stupid,
                    s=1,
                    marker="x",
                    color='red')
        plt.scatter(voxel_centers_np[:, 0] + plot_offset_bc_imshow_stupid,
                    voxel_centers_np[:, 1] + plot_offset_bc_imshow_stupid,
                    s=1,
                    marker="x",
                    color='orange')

        plt.imshow(full_res_before_pseudoimage_np, interpolation='nearest')
        plt.show()

    def _step_down_pseudoimage(self,
                               pseudoimage: torch.Tensor) -> torch.Tensor:
        for stepdown in self.pseudoimage_stepdown_head:
            pseudoimage = stepdown(pseudoimage)
        return pseudoimage

    def _stepdown_voxel_info(
            self, stepped_down_pseudoimage: torch.Tensor, points: torch.Tensor,
            voxel_coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # We need to compute how the voxel coordinates change after the stepdown.
        # We then recompute the point offsets.

        stepdown_scale = 2**self.num_stepdowns
        assert stepdown_scale > 1, "stepdown_scale must be positive"

        scaled_voxel_pillar_size = torch.Tensor([
            self.voxel_pillar_size[0] * stepdown_scale,
            self.voxel_pillar_size[1] * stepdown_scale
        ]).to(points.device)

        # Coordinates are in ZYX order.
        stepped_down_voxel_coords = torch.div(voxel_coords,
                                              stepdown_scale,
                                              rounding_mode='trunc')

        stepped_down_center_translation = torch.Tensor([
            stepped_down_pseudoimage.shape[1] / 2,
            stepped_down_pseudoimage.shape[2] / 2
        ]).float().to(points.device)
        stepped_down_voxel_centers = (
            stepped_down_voxel_coords[:, [2, 1]].float() -
            stepped_down_center_translation[None, :] +
            0.5) * scaled_voxel_pillar_size[None, :]

        stepped_down_point_offsets = points.clone()
        stepped_down_point_offsets[:, :2] -= stepped_down_voxel_centers

        return stepped_down_point_offsets, stepped_down_voxel_coords

    def forward(
            self, full_res_before_pseudoimages: torch.Tensor,
            full_res_after_pseudoimages: torch.Tensor,
            voxelizer_infos: List[Dict[str,
                                       torch.Tensor]]) -> List[torch.Tensor]:

        flow_results = []

        # Step down the pseudoimages
        reduced_res_before_pseudoimages = self._step_down_pseudoimage(
            full_res_before_pseudoimages)
        reduced_res_after_pseudoimages = self._step_down_pseudoimage(
            full_res_after_pseudoimages)

        for before_pseudoimage, after_pseudoimage, voxelizer_info in zip(
                reduced_res_before_pseudoimages,
                reduced_res_after_pseudoimages, voxelizer_infos):
            voxel_coords = voxelizer_info["voxel_coords"]
            points = voxelizer_info["points"]
            point_offsets, voxel_coords = self._stepdown_voxel_info(
                before_pseudoimage, points, voxel_coords)

            flow = self.forward_single(before_pseudoimage, after_pseudoimage,
                                       point_offsets, voxel_coords)
            flow_results.append(flow)
        return flow_results
