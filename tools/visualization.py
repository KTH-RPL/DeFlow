"""
# Created: 2023-11-29 21:22
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)
#
# This file is part of DeFlow (https://github.com/KTH-RPL/DeFlow).
# If you find this repo helpful, please cite the respective publication as 
# listed on the above website.
# 
# Description: view scene flow dataset after preprocess.
"""

import numpy as np
import fire, time
from tqdm import tqdm

import open3d as o3d
import os, sys
BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '..' ))
sys.path.append(BASE_DIR)
from src.utils.mics import HDF5Data, flow_to_rgb
from src.utils.o3d_view import MyVisualizer, color_map


VIEW_FILE = f"{BASE_DIR}/assets/view/av2.json"

def check_flow(
    data_dir: str ="/home/kin/data/av2/preprocess/sensor/mini",
    res_name: str = "flow", # "flow", "flow_est"
    start_id: int = 0,
    point_size: float = 3.0,
):
    dataset = HDF5Data(data_dir, vis_name=res_name, flow_view=True)
    o3d_vis = MyVisualizer(view_file=VIEW_FILE, window_title=f"view {'ground truth flow' if res_name == 'flow' else f'{res_name} flow'}, `SPACE` start/stop")

    opt = o3d_vis.vis.get_render_option()
    opt.background_color = np.asarray([80/255, 90/255, 110/255])
    opt.point_size = point_size

    for data_id in (pbar := tqdm(range(start_id, len(dataset)))):
        data = dataset[data_id]
        now_scene_id = data['scene_id']
        pbar.set_description(f"id: {data_id}, scene_id: {now_scene_id}, timestamp: {data['timestamp']}")
        gm0 = data['gm0']
        pc0 = data['pc0'][~gm0]
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc0[:, :3])
        pcd.paint_uniform_color([1.0, 0.0, 0.0]) # red: pc0

        pc1 = data['pc1']
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(pc1[:, :3][~data['gm1']])
        pcd1.paint_uniform_color([0.0, 1.0, 0.0]) # green: pc1

        pcd2 = o3d.geometry.PointCloud()
        # pcd2.points = o3d.utility.Vector3dVector(pc0[:, :3] + pose_flow) # if you want to check pose_flow
        pcd2.points = o3d.utility.Vector3dVector(pc0[:, :3] + data[res_name][~gm0])
        pcd2.paint_uniform_color([0.0, 0.0, 1.0]) # blue: pc0 + flow
        o3d_vis.update([pcd, pcd1, pcd2, o3d.geometry.TriangleMesh.create_coordinate_frame(size=2)])

def vis(
    data_dir: str ="/home/kin/data/av2/preprocess/sensor/mini",
    res_name: str = "flow", # "flow", "flow_est"
    start_id: int = -1,
    point_size: float = 2.0,
):
    dataset = HDF5Data(data_dir, vis_name=res_name, flow_view=True)
    o3d_vis = MyVisualizer(view_file=VIEW_FILE, window_title=f"view {'ground truth flow' if res_name == 'flow' else f'{res_name} flow'}, `SPACE` start/stop")

    opt = o3d_vis.vis.get_render_option()
    # opt.background_color = np.asarray([216, 216, 216]) / 255.0
    opt.background_color = np.asarray([80/255, 90/255, 110/255])
    # opt.background_color = np.asarray([1, 1, 1])
    opt.point_size = point_size

    for data_id in (pbar := tqdm(range(start_id, len(dataset)))):
        data = dataset[data_id]
        now_scene_id = data['scene_id']
        pbar.set_description(f"id: {data_id}, scene_id: {now_scene_id}, timestamp: {data['timestamp']}")

        pc0 = data['pc0']
        gm0 = data['gm0']
        pose0 = data['pose0']
        pose1 = data['pose1']
        ego_pose = np.linalg.inv(pose1) @ pose0

        pose_flow = pc0[:, :3] @ ego_pose[:3, :3].T + ego_pose[:3, 3] - pc0[:, :3]
        
        pcd = o3d.geometry.PointCloud()
        if res_name in ['dufo_label', 'label']:
            labels = data[res_name]
            pcd_i = o3d.geometry.PointCloud()
            for label_i in np.unique(labels):
                pcd_i.points = o3d.utility.Vector3dVector(pc0[labels == label_i][:, :3])
                if label_i <= 0:
                    pcd_i.paint_uniform_color([1.0, 1.0, 1.0])
                else:
                    pcd_i.paint_uniform_color(color_map[label_i % len(color_map)])
                pcd += pcd_i
        elif res_name in data:
            pcd.points = o3d.utility.Vector3dVector(pc0[:, :3])
            flow = data[res_name] - pose_flow # ego motion compensation here.
            flow_color = flow_to_rgb(flow) / 255.0
            is_dynamic = np.linalg.norm(flow, axis=1) > 0.1
            flow_color[~is_dynamic] = [1, 1, 1]
            flow_color[gm0] = [1, 1, 1]
            pcd.colors = o3d.utility.Vector3dVector(flow_color)
        o3d_vis.update([pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(size=2)])

if __name__ == '__main__':
    start_time = time.time()
    # fire.Fire(check_flow)
    fire.Fire(vis)
    print(f"Time used: {time.time() - start_time:.2f} s")