"""

# Created: 2023-11-05 10:00
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Kin ZHANG  (https://kin-zhang.github.io/)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Description: Model Wrapper for Pytorch Lightning

"""

import numpy as np
import torch
from tabulate import tabulate
import torch.optim as optim
from pathlib import Path

from lightning import LightningModule
from hydra.utils import instantiate
from omegaconf import OmegaConf,open_dict

import os, sys, time, h5py
BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '..' ))
sys.path.append(BASE_DIR)
from scripts.utils.mics import import_func, weights_init, zip_res
from scripts.network.loss_func import evaluate_leaderboard
from scripts.utils.av2_eval import write_output_file
from scripts.network.models.basic import cal_pose0to1

torch.set_float32_matmul_precision('medium')
class ModelWrapper(LightningModule):
    def __init__(self, cfg, eval=False):
        super().__init__()

        # set grid size
        if ('voxel_size' in cfg.model.target) and ('point_cloud_range' in cfg.model.target) and not eval and 'point_cloud_range' in cfg:
            OmegaConf.set_struct(cfg.model.target, True)
            with open_dict(cfg.model.target):
                cfg.model.target['grid_feature_size'] = \
                    [abs(int((cfg.point_cloud_range[0] - cfg.point_cloud_range[3]) / cfg.voxel_size[0])),
                    abs(int((cfg.point_cloud_range[1] - cfg.point_cloud_range[4]) / cfg.voxel_size[1]))]
        else:
            with open_dict(cfg.model.target):
                cfg.model.target['grid_feature_size'] = \
                    [abs(int((cfg.model.target.point_cloud_range[0] - cfg.model.target.point_cloud_range[3]) / cfg.model.target.voxel_size[0])),
                    abs(int((cfg.model.target.point_cloud_range[1] - cfg.model.target.point_cloud_range[4]) / cfg.model.target.voxel_size[1]))]
        
        self.model = instantiate(cfg.model.target)
        self.model.apply(weights_init)
        
        self.loss_fn = import_func("scripts.network.loss_func."+cfg.loss_fn) if 'loss_fn' in cfg else None
        self.batch_size = int(cfg.batch_size) if 'batch_size' in cfg else 1
        self.lr = cfg.lr if 'lr' in cfg else None
        self.epochs = cfg.epochs if 'epochs' in cfg else None
        
        self.load_checkpoint_path = cfg.checkpoint

        if 'dataset_path' in cfg:
            self.dataset_path = cfg.dataset_path
        self.add_seloss = cfg.add_seloss if 'add_seloss' in cfg else None
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        self.model.timer[4].start("One Scan in model")
        res_dict = self.model(batch)
        self.model.timer[4].stop()

        self.model.timer[5].start("Loss")
        # compute loss
        total_loss = 0.0
        batch_sizes = len(batch["pose0"])
        gt_flow = batch['flow']

        pose_flows = res_dict['pose_flow']
        pc0_valid_idx = res_dict['pc0_valid_point_idxes'] # since padding
        est_flow = res_dict['flow']
        
        for batch_id in range(batch_sizes):
            pc0_valid_from_pc2res = pc0_valid_idx[batch_id]
            pose_flow_ = pose_flows[batch_id][pc0_valid_from_pc2res]
            est_flow_ = est_flow[batch_id]
            gt_flow_ = gt_flow[batch_id][pc0_valid_from_pc2res]

            gt_flow_ = gt_flow_ - pose_flow_
            res_dict = {'est_flow': est_flow_, 
                        'gt_flow': gt_flow_, 
                        'gt_classes': None if 'flow_category_indices' not in batch else batch['flow_category_indices'][batch_id][pc0_valid_from_pc2res],
                        }
            loss = self.loss_fn(res_dict)
            total_loss += loss
        
        self.log("trainer/loss", total_loss/batch_sizes, sync_dist=True, batch_size=self.batch_size)
        self.model.timer[5].stop()
        # NOTE (Qingwen): if you want to view the detail breakdown of time cost
        # self.model.timer.print(random_colors=False, bold=False)
        return total_loss

    def train_validation_step_(self, batch, res_dict):
        # means there are ground truth flow so we can evaluate the EPE-3 Way metric
        if batch['flow'][0].shape[0] > 0:
            metric_name = ["EPE_FS", "EPE_BS", "EPE_Dynamic", "EPE_Three", "Dynamic_IoU"]  # Add more names as needed
            metric_eval = {name: 0.0 for name in metric_name}
            pose_flows = res_dict['pose_flow']
            for batch_id, gt_flow in enumerate(batch["flow"]):
                valid_from_pc2res = res_dict['pc0_valid_point_idxes'][batch_id]
                pose_flow = pose_flows[batch_id][valid_from_pc2res]
                final_flow_ = pose_flow.clone()
                

                final_flow_ = final_flow_ + res_dict['flow'][batch_id]
                EPE_BS, EPE_Dynamic, EPE_FS, Dynamic_IoU \
                    = evaluate_leaderboard(final_flow_, pose_flow, batch['pc0'][batch_id][valid_from_pc2res], gt_flow[valid_from_pc2res], \
                                           batch['flow_is_valid'][batch_id][valid_from_pc2res], batch['flow_category_indices'][batch_id][valid_from_pc2res])
                EPE_Three = (EPE_BS + EPE_Dynamic + EPE_FS)/3.0

                # Ensure the order matches the loss_names list
                for idx, loss_value in enumerate([EPE_FS, EPE_BS, EPE_Dynamic, EPE_Three, Dynamic_IoU]):  
                    metric_eval[metric_name[idx]] += loss_value

            for key in metric_eval:
                metric_eval[key] /= (batch_id + 1.0)
                self.log(f"val/{key}", metric_eval[key], sync_dist=True, batch_size=self.batch_size)
        else:
            pass
        
    def on_validation_epoch_end(self):
        self.model.timer.print(random_colors=False, bold=False)
        if self.av2_mode == 'val':
            print(f"\nModel: {self.model.__class__.__name__}, Checkpoint from: {self.load_checkpoint_path}")
            printed_data = []
            for key in self.all_scores:
                printed_data.append([key, np.mean(self.all_scores[key])])
            print(tabulate(printed_data))
            
        if self.av2_mode == 'test':
            print(f"\nModel: {self.model.__class__.__name__}, Checkpoint from: {self.load_checkpoint_path}")
            print(f"Test results saved in: {self.save_res_path}, Please run submit to zip the results and upload to online leaderboard.")
            output_file = zip_res(self.save_res_path)
            # wandb.log_artifact(output_file)
        print(f"Check more details parameters and training status in checkpoints")
        
    def eval_only_step_(self, batch, res_dict):
        batch = {key: batch[key][0] for key in batch if len(batch[key])>0}
        res_dict = {key: res_dict[key][0] for key in res_dict if len(res_dict[key])>0}

        eval_mask = batch['eval_mask'].squeeze()
        pc0 = batch['origin_pc0']
        pose_0to1 = cal_pose0to1(batch["pose0"], batch["pose1"])
        transform_pc0 = pc0 @ pose_0to1[:3, :3].T + pose_0to1[:3, 3]
        pose_flow = transform_pc0 - pc0

        if 'pc0_valid_point_idxes' in res_dict:
            valid_from_pc2res = res_dict['pc0_valid_point_idxes']

            # flow in the original pc0 coordinate
            pred_flow = pose_flow[~batch['gm0']].clone()
            pred_flow[valid_from_pc2res] = pose_flow[~batch['gm0']][valid_from_pc2res] + res_dict['flow']

            final_flow = pose_flow.clone()
            final_flow[~batch['gm0']] = pred_flow
        # else:
        #     # pose_flow = pose_flows
        #     # pred_flow_ = res_dict['flow'].cpu().detach()
        #     # TODO: for other methods.... 
        #     pred_flow = pose_flows.clone()

        if self.av2_mode == 'val': # since only val we have ground truth flow to eval
            metric_name = ["EPE_FS", "EPE_BS", "EPE_Dynamic", "EPE_Three", "Dynamic_IoU"]  # Add more names as needed
            metric_eval = {name: 0.0 for name in metric_name}

            gt_flow = batch["flow"]

            EPE_BS, EPE_Dynamic, EPE_FS, Dynamic_IoU \
                = evaluate_leaderboard(final_flow[eval_mask], pose_flow[eval_mask], pc0[eval_mask], \
                                       gt_flow[eval_mask], batch['flow_is_valid'][eval_mask], \
                                       batch['flow_category_indices'][eval_mask])
            
            EPE_Three = (EPE_BS + EPE_Dynamic + EPE_FS)/3.0

            # Ensure the order matches the loss_names list
            for idx, loss_value in enumerate([EPE_FS, EPE_BS, EPE_Dynamic, EPE_Three, Dynamic_IoU]):  
                metric_eval[metric_name[idx]] += loss_value
                self.all_scores[metric_name[idx]].append(loss_value)

            for key in metric_eval:
                self.log(f"val/{key}", metric_eval[key])
        
        # NOTE (Qingwen): Since val and test, we will force set batch_size = 1 
        if self.save_res or self.av2_mode == 'test': # test must save data to submit in the online leaderboard.    
            save_pred_flow = final_flow[eval_mask, :3].cpu().detach().numpy()
            rigid_flow = pose_flow[eval_mask, :3].cpu().detach().numpy()
            is_dynamic = np.linalg.norm(save_pred_flow - rigid_flow, axis=1, ord=2) >= 0.05
            sweep_uuid = (batch['scene_id'], batch['timestamp'])
            write_output_file(save_pred_flow, is_dynamic, sweep_uuid, self.save_res_path)

    def validation_step(self, batch, batch_idx):
        if self.av2_mode == 'val' or self.av2_mode == 'test':
            batch['origin_pc0'] = batch['pc0'].clone()
            batch['pc0'] = batch['pc0'][~batch['gm0']].unsqueeze(0)
            batch['pc1'] = batch['pc1'][~batch['gm1']].unsqueeze(0)
            self.model.timer[12].start("One Scan")
            res_dict = self.model(batch)
            self.model.timer[12].stop()
            self.eval_only_step_(batch, res_dict)
        else:
            res_dict = self.model(batch)
            self.train_validation_step_(batch, res_dict)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer
    
    def on_train_epoch_start(self):
        self.time_start_train_epoch = time.time()

    def on_train_epoch_end(self):
        self.log("pre_epoch_cost (mins)", (time.time()-self.time_start_train_epoch)/60.0, on_step=False, on_epoch=True, sync_dist=True)
    def on_test_epoch_start(self) -> None:
        # empty N, 3
        self.clean_map = torch.zeros((0, 3), device=self.device)

    def test_step(self, batch, batch_idx):
        # NOTE (Qingwen): again, val and test we only allow batch_size = 1
        batch['origin_pc0'] = batch['pc0'].clone()
        batch['pc0'] = batch['pc0'][~batch['gm0']].unsqueeze(0)
        batch['pc1'] = batch['pc1'][~batch['gm1']].unsqueeze(0)
        res_dict = self.model(batch)
        batch = {key: batch[key][0] for key in batch if len(batch[key])>0}
        res_dict = {key: res_dict[key][0] for key in res_dict if len(res_dict[key])>0}

        pc0 = batch['origin_pc0']
        # pose_0to1 = cal_pose0to1(batch["pose0"], batch["pose1"])
        # transform_pc0 = pc0 @ pose_0to1[:3, :3].T + pose_0to1[:3, 3]
        pose_flow = torch.zeros_like(pc0)

        if 'pc0_valid_point_idxes' in res_dict:
            valid_from_pc2res = res_dict['pc0_valid_point_idxes']

            # flow in the original pc0 coordinate
            pred_flow = pose_flow[~batch['gm0']].clone()
            pred_flow[valid_from_pc2res] = pose_flow[~batch['gm0']][valid_from_pc2res] + res_dict['flow']

            final_flow = pose_flow.clone()
            final_flow[~batch['gm0']] = pred_flow

        # label final_flow >=0.05 as dynamic, 10hz 0.5m/s speed as dynamic
        dynamic_label = torch.norm(final_flow, dim=1, p=2) >= 0.05
        
        # batch['world_pc0'][~dynamic_label]
        self.clean_map = torch.cat([self.clean_map, batch['world_pc0'][~dynamic_label]], dim=0)
        # # write final_flow into the dataset.
        # key = str(batch['timestamp'])
        # scene_id = batch['scene_id']
        # with h5py.File(os.path.join(self.dataset_path, f'{scene_id}.h5'), 'r+') as f:
        #     if 'flow_est' in f[key]:
        #         del f[key]['flow_est']
        #     f[key].create_dataset('flow_est', data=final_flow.cpu().detach().numpy().astype(np.float32))

    def on_test_epoch_end(self):
        from scripts.utils.pcdpy3 import save_pcd
        print(f"\n\nModel: {self.model.__class__.__name__}, Checkpoint from: {self.load_checkpoint_path}")
        self.model.timer.print(random_colors=True, bold=True)
        # to numpy 
        clean_map = self.clean_map.cpu().detach().numpy().astype(np.float32)
        save_pcd(f"{self.dataset_path}/deflow_output.pcd", clean_map)
        print(f"Write clean map to the dataset: {self.dataset_path}/deflow_output.pcd. Enjoy! ^v^ ------ \n")

        # print(f"We already write the flow_est into the dataset, please run following commend to visualize the flow. Copy and paste it to your terminal:")
        # print(f"python tests/scene_flow.py --flow_mode='flow_est' --data_dir={self.dataset_path}")
        # print(f"Enjoy! ^v^ ------ \n")
       