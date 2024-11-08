"""
# 
# Created: 2024-04-14 11:57
# Copyright (C) 2024-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)
# 
#
# Reference to official evaluation scripts:
# - EPE Threeway: https://github.com/argoverse/av2-api/blob/main/src/av2/evaluation/scene_flow/eval.py
# - Bucketed EPE: https://github.com/kylevedder/BucketedSceneFlowEval/blob/master/bucketed_scene_flow_eval/eval/bucketed_epe.py
"""

import torch
import os, sys
import numpy as np
from typing import Dict, Final, List, Tuple
from tabulate import tabulate

BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '../..' ))
sys.path.append(BASE_DIR)
from scripts.utils.av2_eval import compute_metrics, compute_bucketed_epe, CLOSE_DISTANCE_THRESHOLD


# EPE Three-way: Foreground Dynamic, Background Dynamic, Background Static
# leaderboard link: https://eval.ai/web/challenges/challenge-page/2010/evaluation
def evaluate_leaderboard(est_flow, rigid_flow, pc0, gt_flow, is_valid, pts_ids):
    gt_is_dynamic = torch.linalg.vector_norm(gt_flow - rigid_flow, dim=-1) >= 0.05
    mask_ = ~est_flow.isnan().any(dim=1) & ~rigid_flow.isnan().any(dim=1) & ~pc0[:, :3].isnan().any(dim=1) & ~gt_flow.isnan().any(dim=1)
    mask_no_nan = mask_ & ~gt_is_dynamic.isnan() & ~is_valid.isnan() & ~pts_ids.isnan()
    est_flow = est_flow[mask_no_nan, :]
    rigid_flow = rigid_flow[mask_no_nan, :]
    pc0 = pc0[mask_no_nan, :]
    gt_flow = gt_flow[mask_no_nan, :]
    gt_is_dynamic = gt_is_dynamic[mask_no_nan]
    is_valid = is_valid[mask_no_nan]
    pts_ids = pts_ids[mask_no_nan]

    est_is_dynamic = torch.linalg.vector_norm(est_flow - rigid_flow, dim=-1) >= 0.05
    is_close = torch.all(torch.abs(pc0[:, :2]) <= CLOSE_DISTANCE_THRESHOLD, dim=1)
    res_dict = compute_metrics(
        est_flow.detach().cpu().numpy().astype(float),
        est_is_dynamic.detach().cpu().numpy().astype(bool),
        gt_flow.detach().cpu().numpy().astype(float),
        pts_ids.detach().cpu().numpy().astype(np.uint8),
        gt_is_dynamic.detach().cpu().numpy().astype(bool),
        is_close.detach().cpu().numpy().astype(bool),
        is_valid.detach().cpu().numpy().astype(bool)
    )
    return res_dict

# reference to official evaluation: bucketed_scene_flow_eval/eval/bucketed_epe.py
# python >= 3.7
from dataclasses import dataclass
import warnings
@dataclass(frozen=True, eq=True, repr=True)
class OverallError:
    static_epe: float
    dynamic_error: float

    def __repr__(self) -> str:
        static_epe_val_str = (
            f"{self.static_epe:0.6f}" if np.isfinite(self.static_epe) else f"{self.static_epe}"
        )
        dynamic_error_val_str = (
            f"{self.dynamic_error:0.6f}"
            if np.isfinite(self.dynamic_error)
            else f"{self.dynamic_error}"
        )
        return f"({static_epe_val_str}, {dynamic_error_val_str})"

    def to_tuple(self) -> Tuple[float, float]:
        return (self.static_epe, self.dynamic_error)

class OfficialMetrics:
    def __init__(self):
        self.epe_3way = {
            'EPE_FD': [],
            'EPE_BS': [],
            'EPE_FS': [],
            'IoU': [],
            'Three-way': []
        }

        self.norm_flag = False

    def step(self, epe_dict):
        """
        This step function is used to store the results of **each frame**.
        """
        for key in epe_dict:
            self.epe_3way[key].append(epe_dict[key])


    def normalize(self):
        """
        This normalize mean average results between **frame and frame**.
        """
        for key in self.epe_3way:
            self.epe_3way[key] = np.mean(self.epe_3way[key])
        self.epe_3way['Three-way'] = np.mean([self.epe_3way['EPE_FD'], self.epe_3way['EPE_BS'], self.epe_3way['EPE_FS']])

        self.norm_flag = True
    
    def print(self):
        if not self.norm_flag:
            self.normalize()
        printed_data = []
        for key in self.epe_3way:
            printed_data.append([key,self.epe_3way[key]])
        print("Version 1 Metric on EPE Three-way:")
        print(tabulate(printed_data), "\n")