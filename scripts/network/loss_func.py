"""
# Created: 2023-07-17 00:00
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Kin ZHANG  (https://kin-zhang.github.io/)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
import torch

def deflowLoss(res_dict):
    pred = res_dict['est_flow']
    gt = res_dict['gt_flow']

    mask_no_nan = (~gt.isnan() & ~pred.isnan() & ~gt.isinf() & ~pred.isinf())
    
    pred = pred[mask_no_nan].reshape(-1, 3)
    gt = gt[mask_no_nan].reshape(-1, 3)

    speed = gt.norm(dim=1, p=2) / 0.1
    # pts_loss = torch.norm(pred - gt, dim=1, p=2)
    pts_loss = torch.linalg.vector_norm(pred - gt, dim=-1)

    weight_loss = 0.0
    speed_0_4 = pts_loss[speed < 0.4].mean()
    speed_mid = pts_loss[(speed >= 0.4) & (speed <= 1.0)].mean()
    speed_1_0 = pts_loss[speed > 1.0].mean()
    if ~speed_1_0.isnan():
        weight_loss += speed_1_0
    if ~speed_0_4.isnan():
        weight_loss += speed_0_4
    if ~speed_mid.isnan():
        weight_loss += speed_mid
    return weight_loss

# ref from zeroflow loss class FastFlow3DDistillationLoss()
def zeroflowLoss(res_dict):
    pred = res_dict['est_flow']
    gt = res_dict['gt_flow']
    mask_no_nan = (~gt.isnan() & ~pred.isnan() & ~gt.isinf() & ~pred.isinf())
    
    pred = pred[mask_no_nan].reshape(-1, 3)
    gt = gt[mask_no_nan].reshape(-1, 3)

    error = torch.linalg.vector_norm(pred - gt, dim=-1)
    # gt_speed = torch.norm(gt, dim=1, p=2) * 10.0
    gt_speed = torch.linalg.vector_norm(gt, dim=-1) * 10.0
    
    mins = torch.ones_like(gt_speed) * 0.1
    maxs = torch.ones_like(gt_speed)
    importance_scale = torch.max(mins, torch.min(1.8 * gt_speed - 0.8, maxs))
    # error = torch.norm(pred - gt, dim=1, p=2) * importance_scale
    error = error * importance_scale
    return error.mean()

# ref from zeroflow loss class FastFlow3DSupervisedLoss()
def ff3dLoss(res_dict):
    pred = res_dict['est_flow']
    gt = res_dict['gt_flow']
    classes = res_dict['gt_classes']
    # error = torch.norm(pred - gt, dim=1, p=2)
    error = torch.linalg.vector_norm(pred - gt, dim=-1)
    is_foreground_class = (classes > 0) # 0 is background, ref: FOREGROUND_BACKGROUND_BREAKDOWN
    background_scalar = is_foreground_class.float() * 0.9 + 0.1
    error = error * background_scalar
    return error.mean()

# ==========================> From AV2.0 Eval Official Scripts.
from typing import Dict, Final
import os, sys
BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '../..' ))
sys.path.append(BASE_DIR)
from scripts.utils.av2_eval import compute_metrics, FOREGROUND_BACKGROUND_BREAKDOWN
import numpy as np

CLOSE_DISTANCE_THRESHOLD: Final = 35.0
EPS: Final = 1e-6
def compute_epe(res_dict, indices, eps=1e-8):
    epe_sum = 0
    count_sum = 0
    for index in indices:
        count = res_dict['Count'][index]
        if count != 0:
            epe_sum += res_dict['EPE'][index] * count
            count_sum += count
    return epe_sum / (count_sum + eps) if count_sum != 0 else 0.0

# after ground mask already, not origin N, 3 but without ground points
def evaluate_leaderboard(est_flow, rigid_flow, pc0, gt_flow, is_valid, pts_ids):
    
    # gt_is_dynamic = (gt_flow - rigid_flow).norm(dim=1, p=2) > 0.05
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
        is_valid.detach().cpu().numpy().astype(bool),
        FOREGROUND_BACKGROUND_BREAKDOWN,
    )

    # reference: eval.py L503
    # we need Dynamic IoU and EPE 3-Way Average to calculate loss!
    # weighted: (x[metric_type.value] * x.Count).sum() / total
    # 'Class': ['Background', 'Background', 'Background', 'Background', 'Foreground', 'Foreground', 'Foreground', 'Foreground']
    # 'Motion': ['Dynamic', 'Dynamic', 'Static', 'Static', 'Dynamic', 'Dynamic', 'Static', 'Static']
    # 'Distance': ['Close', 'Far', 'Close', 'Far', 'Close', 'Far', 'Close', 'Far']
    
    EPE_Background_Static = compute_epe(res_dict, [2, 3])
    EPE_Dynamic = compute_epe(res_dict, [4, 5])
    EPE_Foreground_Static = compute_epe(res_dict, [6, 7])

    Dynamic_IoU = sum(res_dict['TP']) / (sum(res_dict['TP']) + sum(res_dict['FP']) + sum(res_dict['FN'])+EPS)
    # EPE_Dynamic is nan?
    if np.isnan(EPE_Dynamic) or np.isnan(Dynamic_IoU) or np.isnan(EPE_Background_Static) or np.isnan(EPE_Foreground_Static):
        print(res_dict)

    return EPE_Background_Static, EPE_Dynamic, EPE_Foreground_Static, Dynamic_IoU