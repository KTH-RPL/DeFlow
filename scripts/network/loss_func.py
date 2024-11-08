"""
# Created: 2023-07-17 00:00
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)
#
# Description: Define the loss function for training.
"""
import torch
from scipy.stats import gaussian_kde
import numpy as np
import json



def mambaflowLoss(res_dict):
    pred = res_dict['est_flow']
    gt = res_dict['gt_flow']
    mask_no_nan = (~gt.isnan() & ~pred.isnan() & ~gt.isinf() & ~pred.isinf())

    pred = pred[mask_no_nan].reshape(-1, 3)
    gt = gt[mask_no_nan].reshape(-1, 3)

    speed = gt.norm(dim=1, p=2) / 0.1
    # pts_loss = torch.norm(pred - gt, dim=1, p=2)
    pts_loss = torch.linalg.vector_norm(pred - gt, dim=-1)

    velocities = speed.cpu().numpy()

    # 计算直方图，返回每个区间的计数和区间边界
    counts, bin_edges = np.histogram(velocities, bins=100, density=False)

    # 计算每个区间的点数占总点数的比例
    total_points = len(velocities)
    proportions = counts / total_points

    # 计算每个区间的中心位置，用于绘图
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # 设置占比阈值
    proportion_threshold = 0.01  # 可以根据需要调整这个值

    # 找出第一个占比小于阈值的柱子
    first_below_threshold = next((i for i, prop in enumerate(proportions) if prop < proportion_threshold), None)
    if first_below_threshold is None:
        # with open('/data/jiehao/project/Flow4D/Module_test/velocities_error_v3.csv', 'w') as f:
        #     for v in velocities:
        #         f.write(f"{v}\n")
        for key, value in res_dict.items():
            res_dict[key] = value.tolist()

        # 将字典写入 JSON 文件
        with open('/data/jiehao/project/Flow4D/Module_test/data.json', 'w') as f:
            json.dump(res_dict, f)

        raise ValueError(1)
        #turning_speed = 0
    else:
        turning_speed = bin_centers[first_below_threshold]

    # print(turning_speed)

    weight_loss = 0.0
    speed_mid = 2
    speed_0 = pts_loss[speed < turning_speed].mean()
    speed_1 = pts_loss[(speed >= turning_speed) & (speed <= speed_mid)].mean()
    speed_2 = pts_loss[speed > speed_mid].mean()
    if ~speed_1.isnan():
        weight_loss += speed_1
    if ~speed_0.isnan():
        weight_loss += speed_0
    if ~speed_2.isnan():
        weight_loss += speed_2
    return weight_loss


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
    background_velocities = gt[classes == 0].norm(dim=1, p=2) / 0.1
    background_max_velocity = background_velocities.max()
    print(f"Background max velocity: {background_max_velocity:.2f}")
    background_scalar = is_foreground_class.float() * 0.9 + 0.1
    error = error * background_scalar
    return error.mean()