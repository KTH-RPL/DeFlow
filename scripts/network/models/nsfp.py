
"""
This file is directly copied from: 
https://github.com/kylevedder/zeroflow/blob/master/models/fast_flow_3d.py

with slightly modification to have unified format with all benchmark.
"""

import dztimer, torch, copy
import torch.nn as nn

from .unsfp.model import Neural_Prior, EarlyStopping, my_chamfer_fn
from .basic import cal_pose0to1

class NSFP(nn.Module):
    def __init__(self, filter_size=128, act_fn='relu', layer_size=8, \
                 itr_num=5000, lr=8e-3, min_delta=0.00005, verbose=False,
                 point_cloud_range = [-51.2, -51.2, -3, 51.2, 51.2, 3]):
        super().__init__()
        self.nsfp_model = Neural_Prior(filter_size=filter_size, act_fn=act_fn, layer_size=layer_size)
        self.iteration_num = itr_num
        self.min_delta = min_delta
        self.lr = lr
        self.verbose = verbose
        self.point_cloud_range = point_cloud_range
        self.timer = dztimer.Timing()
        self.timer.start("NSFP Model Inference")

    def optimize(self, pc0, pc1):
        self.nsfp_model.train()
        self.nsfp_model_inv = copy.deepcopy(self.nsfp_model)
        params = [{
            'params': self.nsfp_model.parameters(),
            'lr': self.lr,
            'weight_decay': 0
        }, {
            'params': self.nsfp_model_inv.parameters(),
            'lr': self.lr,
            'weight_decay': 0
        }]
        best_forward = {'loss': torch.inf}

        optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=0)
        early_stopping = EarlyStopping(patience=100, min_delta=self.min_delta)

        self.timer[5].start("One Scan")
        for _ in range(self.iteration_num):
            optimizer.zero_grad()
            self.timer[1].start("Network Time")

            self.timer[1][0].start("Forward")
            forward_flow = self.nsfp_model(pc0)
            self.timer[1][0].stop()
            pc0_to_pc1 = pc0 + forward_flow

            self.timer[1][1].start("Inverse")
            inverse_flow = self.nsfp_model_inv(pc0_to_pc1)
            self.timer[1][1].stop()
            est_pc1_to_pc0 = pc0_to_pc1 - inverse_flow

            self.timer[1].stop()

            self.timer[2].start("Loss")
            self.timer[2][0].start("Forward Loss")
            forward_loss, _ = my_chamfer_fn(pc0_to_pc1.unsqueeze(0), pc1.unsqueeze(0))
            self.timer[2][0].stop()

            self.timer[2][1].start("Inverse Loss")
            inverse_loss, _ = my_chamfer_fn(est_pc1_to_pc0.unsqueeze(0), pc0.unsqueeze(0))
            self.timer[2][1].stop()
            self.timer[2].stop()
            loss = forward_loss + inverse_loss
            if forward_loss <= best_forward['loss']:
                best_forward['loss'] = forward_loss.item()
                best_forward['flow'] = forward_flow

            if early_stopping.step(loss):
                break

            self.timer[3].start("Loss Backward")
            loss.backward()
            self.timer[3].stop()

            self.timer[4].start("Optimizer Step")
            optimizer.step()
            self.timer[4].stop()

        self.timer[5].stop()
        if self.verbose:
            self.timer.print(random_color=True, bold=True)
        return {'flow': best_forward['flow'],
                'loss': best_forward['loss'],}
    def range_limit_(self, pc):
        """
        Limit the point cloud to the given range.
        """
        mask = (pc[:, 0] >= self.point_cloud_range[0]) & (pc[:, 0] <= self.point_cloud_range[3]) & \
               (pc[:, 1] >= self.point_cloud_range[1]) & (pc[:, 1] <= self.point_cloud_range[4]) & \
               (pc[:, 2] >= self.point_cloud_range[2]) & (pc[:, 2] <= self.point_cloud_range[5])
        return pc[mask], mask
    
    def forward(self, batch):
        batch_sizes = len(batch["pose0"])

        
        pose_flows = []
        pose_0to1s = []
        for batch_id in range(batch_sizes):
            self.timer[0].start("Data Processing")
            pc0 = batch["pc0"][batch_id]
            pc1 = batch["pc1"][batch_id]
            selected_pc0, rm0 = self.range_limit_(pc0)
            selected_pc1, _ = self.range_limit_(pc1)
            self.timer[0][0].start("pose")
            pose_0to1 = cal_pose0to1(batch["pose0"][batch_id], batch["pose1"][batch_id])
            pose_0to1s.append(pose_0to1)
            self.timer[0][0].stop()
            
            self.timer[0][1].start("transform")
            # transform selected_pc0 to pc1
            transform_pc0 = selected_pc0 @ pose_0to1[:3, :3].T + pose_0to1[:3, 3]
            self.timer[0][1].stop()
            pose_flows.append(transform_pc0 - selected_pc0)

            self.timer[0].stop()

            with torch.inference_mode(False):
                with torch.enable_grad():
                    pc0_ = transform_pc0.clone().detach().requires_grad_(True)
                    pc1_ = selected_pc1.clone().detach().requires_grad_(True)
                    model_res = self.optimize(pc0_, pc1_)
            
            final_flow = torch.zeros_like(pc0)
            final_flow[rm0] = model_res["flow"]
            
        res_dict = {"flow": final_flow,
                    "pose_flow": pose_flows,
                    "pose_0to1": pose_0to1s
                    }
        
        return res_dict