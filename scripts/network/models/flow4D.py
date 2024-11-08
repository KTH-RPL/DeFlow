
"""
# Created: 2023-07-18 15:08
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Kin ZHANG  (https://kin-zhang.github.io/)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
"""

import torch.nn as nn
import dztimer, torch

from .basic.embedder_model_flow4D import DynamicEmbedder_4D
from .basic import cal_pose0to1

from .basic.network_4D import Network_4D, Seperate_to_3D, mambaflow_head


def replace_feature(out, new_features):
    if "replace_feature" in out.__dir__():
        # spconv 2.x behaviour
        return out.replace_feature(new_features)
    else:
        out.features = new_features
        return out
    

class Flow4D(nn.Module):
    def __init__(self, voxel_size = [0.2, 0.2, 0.2],
                 point_cloud_range = [-51.2, -51.2, -2.2, 51.2, 51.2, 4.2],
                 grid_feature_size = [512, 512, 32],
                 num_frames = 5 ):
        super().__init__()

        point_output_ch = 16
        voxel_output_ch = 16

        self.num_frames = num_frames
        print('voxel_size = {}, pseudo_dims = {}, input_num_frames = {}'.format(voxel_size, grid_feature_size, self.num_frames))

        self.embedder_4D = DynamicEmbedder_4D(voxel_size=voxel_size,
                                        pseudo_image_dims=[grid_feature_size[0], grid_feature_size[1], grid_feature_size[2], num_frames], 
                                        point_cloud_range=point_cloud_range,
                                        feat_channels=point_output_ch)
        
        self.network_4D = Network_4D(in_channel=point_output_ch, out_channel=voxel_output_ch)

        self.seperate_feat = Seperate_to_3D(num_frames)

        self.mambaflowhead_3D = mambaflow_head(voxel_feat_dim=voxel_output_ch, point_feat_dim=point_output_ch, use_decoder=False)

        
        self.timer = dztimer.Timing()
        self.timer.start("Total")

    def load_from_checkpoint(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        state_dict = {
            k[len("model.") :]: v for k, v in ckpt.items() if k.startswith("model.")
        }
        print("\nLoading... model weight from: ", ckpt_path, "\n")
        return self.load_state_dict(state_dict=state_dict, strict=False)


    def forward(self, batch):
        #t_deflow_start = time.time()
        """
        input: using the batch from dataloader, which is a dict
               Detail: [pc0, pc1, pose0, pose1]
        output: the predicted flow, pose_flow, and the valid point index of pc0
        """
        # self.freeze_backbone()
        self.timer[0].start("Data Preprocess")
        batch_sizes = len(batch["pose0"])

        pose_flows = []
        transform_pc0s = []
        transform_pc_m_frames = [[] for _ in range(self.num_frames - 2)]


        for batch_id in range(batch_sizes):
            selected_pc0 = batch["pc0"][batch_id] 
            self.timer[0][0].start("pose")
            with torch.no_grad():
                if 'ego_motion' in batch:
                    pose_0to1 = batch['ego_motion'][batch_id] 
                else:
                    pose_0to1 = cal_pose0to1(batch["pose0"][batch_id], batch["pose1"][batch_id]) 

                if self.num_frames > 2: 
                    past_poses = []
                    for i in range(1, self.num_frames - 1):
                        # 全部转到pose1的视角下
                        past_pose = cal_pose0to1(batch[f"pose_m{i}"][batch_id], batch["pose1"][batch_id])
                        past_poses.append(past_pose)
            self.timer[0][0].stop()
            
            self.timer[0][1].start("transform")
            transform_pc0 = selected_pc0 @ pose_0to1[:3, :3].T + pose_0to1[:3, 3] #t -> t+1 warping
            self.timer[0][1].stop()
            pose_flows.append(transform_pc0 - selected_pc0)
            transform_pc0s.append(transform_pc0)

            # 固定取两个frame，后续从 dataloder 读取的batch中取出剩下的 num_frame - 2 个frame
            for i in range(1, self.num_frames - 1):
                selected_pc_m = batch[f"pc_m{i}"][batch_id]
                transform_pc_m = selected_pc_m @ past_poses[i-1][:3, :3].T + past_poses[i-1][:3, 3]
                transform_pc_m_frames[i-1].append(transform_pc_m)

        pc_m_frames = [torch.stack(transform_pc_m_frames[i], dim=0) for i in range(self.num_frames - 2)]

        pc0s = torch.stack(transform_pc0s, dim=0) 
        pc1s = batch["pc1"]
        self.timer[0].stop()


        pcs_dict = {
            'pc0s': pc0s,
            'pc1s': pc1s,
        }

        # pcs_dict: [pc0s, pc1s, pc_m1s, pc_m2s, pc_m3s]
        # 存放这个batch的所有连续点云帧
        # pc0s [B, N, 3] pc1s [B, N, 3] ......
        # 按照第一维度的数值一一对应
        for i in range(1, self.num_frames - 1):
            pcs_dict[f'pc_m{i}s'] = pc_m_frames[i-1]


        self.timer[1].start("4D_voxelization")
        dict_4d = self.embedder_4D(pcs_dict)
        pc01_tesnor_4d = dict_4d['4d_tensor']
        pc0_3dvoxel_infos_lst =dict_4d['pc0_3dvoxel_infos_lst']
        pc0_point_feats_lst = dict_4d['pc0_point_feats_lst']
        pc0_num_voxels = dict_4d['pc0_num_voxels']
        self.timer[1].stop()

        self.timer[2].start("4D_backbone")
        pc_all_output_4d = self.network_4D(pc01_tesnor_4d) #all = past, current, next 다 합친것
        self.timer[2].stop()

        self.timer[3].start("4D pc01 to 3D pc0")
        pc0_last = self.seperate_feat(pc_all_output_4d)
        assert pc0_last.features.shape[0] == pc0_num_voxels, 'voxel number mismatch'
        self.timer[3].stop()

        self.timer[4].start("3D_sparsetensor_to_point and head")
        flows = self.mambaflowhead_3D(pc0_last, pc0_3dvoxel_infos_lst, pc0_point_feats_lst)
        self.timer[4].stop()

        pc0_points_lst = [e["points"] for e in pc0_3dvoxel_infos_lst] 
        pc0_valid_point_idxes = [e["point_idxes"] for e in pc0_3dvoxel_infos_lst] 

        model_res = {
            "flow": flows, 
            'pose_flow': pose_flows, 

            "pc0_valid_point_idxes": pc0_valid_point_idxes, 
            "pc0_points_lst": pc0_points_lst, 
            
        }
        return model_res